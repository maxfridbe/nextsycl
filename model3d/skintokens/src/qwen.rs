//! The decoder: Qwen3's shape (28 pre-norm layers, width 896, 16 query heads over 8 key / value heads of 128, per-head q / k
//! norms, rotary embedding with theta 1e6, SwiGLU of 3072, an untied output head over 33,036 tokens), its prefix the shape
//! encoder's 512 latents through output_proj (a linear layer and an RMS norm) and the start tokens, as input embeddings. Keys
//! and values go to a pool of slots shared by the beams: a beam's history is a list of slots, so following another beam
//! copies that list, not the cache.

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::ops::{dev_i32, Lin, Ops, Weights};
use crate::tokens::TOTAL;

pub const WIDTH: usize = 896;
const LAYERS: usize = 28;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const FFN: usize = 3072;
const EPS: f32 = 1e-6;
const THETA: f64 = 1e6;
/// max_position_embeddings
const POSITIONS: usize = 3192;
/// A slot's row in the pool (keys or values)
const KV: usize = KV_HEADS * HD;
const QKV: usize = (HEADS + 2 * KV_HEADS) * HD;

struct Layer {
    in_norm: DevBuf,
    qkv: Lin,
    q_norm: DevBuf,
    k_norm: DevBuf,
    o: Lin,
    post_norm: DevBuf,
    gate_up: Lin,
    down: Lin,
}

pub struct Qwen {
    /// output_proj.0 (512 -> 896) and the RMS norm after it (output_proj.1)
    proj: Lin,
    proj_norm: DevBuf,
    layers: Vec<Layer>,
    norm: DevBuf,
    /// the output head [33036, 896], which is the token embedding too (the checkpoint's two are equal)
    lm_head: Lin,
    inv_freq: DevBuf,
}

/// The keys and values of every layer, by slot
pub struct Pool {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub cap: usize,
    pub used: usize,
}

impl Qwen {
    pub fn load(w: &Weights) -> Result<Qwen> {
        let m = "transformer.model";
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("{m}.layers.{l}");
            layers.push(Layer {
                in_norm: w.vec(&format!("{p}.input_layernorm.weight"), WIDTH)?,
                qkv: w.lin_cat_bf16(&[&format!("{p}.self_attn.q_proj"), &format!("{p}.self_attn.k_proj"), &format!("{p}.self_attn.v_proj")],
                               &[HEADS * HD, KV, KV], WIDTH)?,
                q_norm: w.vec(&format!("{p}.self_attn.q_norm.weight"), HD)?,
                k_norm: w.vec(&format!("{p}.self_attn.k_norm.weight"), HD)?,
                o: w.lin_bf16(&format!("{p}.self_attn.o_proj"), WIDTH, HEADS * HD)?,
                post_norm: w.vec(&format!("{p}.post_attention_layernorm.weight"), WIDTH)?,
                gate_up: w.lin_cat_bf16(&[&format!("{p}.mlp.gate_proj"), &format!("{p}.mlp.up_proj")], &[FFN, FFN], WIDTH)?,
                down: w.lin_bf16(&format!("{p}.mlp.down_proj"), WIDTH, FFN)?,
            });
        }
        let inv: Vec<f32> = (0..HD / 2).map(|i| THETA.powf(-(2.0 * i as f64) / HD as f64) as f32).collect();
        Ok(Qwen {
            proj: w.lin("output_proj.0", WIDTH, crate::encoder::WIDTH, true)?,
            proj_norm: w.vec("output_proj.1.weight", WIDTH)?,
            layers,
            norm: w.vec(&format!("{m}.norm.weight"), WIDTH)?,
            lm_head: w.lin_bf16("transformer.lm_head", TOTAL, WIDTH)?,
            inv_freq: DevBuf::from_f32(w.gpu, &inv)?,
        })
    }

    /// The most positions it has: prefix and generated tokens together
    pub fn max_positions(&self) -> usize {
        POSITIONS
    }

    pub fn pool(&self, o: &Ops, cap: usize) -> Result<Pool> {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for _ in 0..LAYERS {
            k.push(o.buf(cap * KV)?);
            v.push(o.buf(cap * KV)?);
        }
        Ok(Pool { k, v, cap, used: 0 })
    }

    /// The pool with room for `cap` slots, its used ones copied over (the old one freed)
    pub fn grow(&self, o: &Ops, pool: &mut Pool, cap: usize) -> Result<()> {
        if cap <= pool.cap {
            return Err(Error("the key / value pool cannot grow further (the card's memory)".into()));
        }
        let mut new = self.pool(o, cap)?;
        for l in 0..LAYERS {
            new.k[l].copy_within(0, &pool.k[l], 0, pool.used * KV * 4)?;
            new.v[l].copy_within(0, &pool.v[l], 0, pool.used * KV * 4)?;
        }
        new.used = pool.used;
        *pool = new;
        Ok(())
    }

    /// Bytes of a pool of `cap` slots
    pub fn pool_bytes(cap: usize) -> usize {
        LAYERS * 2 * cap * KV * 4
    }

    /// The decoder's input for the shape latents [n, 512]: output_proj, linear then RMS norm (eps = bfloat16's machine epsilon, as
    /// the checkpoint was trained with) -> [n, 896]
    pub fn project(&self, o: &Ops, lat: &DevBuf, n: usize) -> Result<DevBuf> {
        let x = o.buf(n * WIDTH)?;
        o.linear(lat, n, &self.proj, &x)?;
        o.k.rms(&x, WIDTH, n, WIDTH, 0.0078125, &self.proj_norm, &x)?;
        Ok(x)
    }

    /// The token embeddings of `ids` [n, 896]
    pub fn embed(&self, o: &Ops, ids: &[u32]) -> Result<DevBuf> {
        let idx: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let idx = dev_i32(&o.gpu, &idx)?;
        let out = o.buf(ids.len().max(1) * WIDTH)?;
        o.k.gather(&self.lm_head.w, WIDTH, &idx, ids.len(), WIDTH, &out)?;
        Ok(out)
    }

    /// One layer on x [rows, 896]: `pos` the rows' positions, `slot` the pool rows their keys and values go to (int32); `attend`
    /// does the attention from the normed-and-rotated q (in qkv, rows of 4096) into a [rows, 2048]
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, o: &Ops, li: usize, x: &DevBuf, rows: usize, pool: &Pool, pos: &DevBuf, slot: &DevBuf, bufs: &Bufs,
             attend: &dyn Fn(&DevBuf, &DevBuf, &DevBuf) -> Result<()>) -> Result<()> {
        let ly = &self.layers[li];
        let Bufs { t, qkv, a, h, g } = bufs;
        o.k.rms(x, WIDTH, rows, WIDTH, EPS, &ly.in_norm, t)?;
        o.linear(t, rows, &ly.qkv, qkv)?;
        o.k.qkv_post(qkv, rows, HEADS, KV_HEADS, HD, EPS, &ly.q_norm, &ly.k_norm, pos, &self.inv_freq, slot, &pool.k[li], &pool.v[li])?;
        attend(qkv, &pool.k[li], &pool.v[li])?;
        // `attend` wrote a
        o.linear(a, rows, &ly.o, t)?;
        o.k.add(x, t, rows * WIDTH)?;
        o.k.rms(x, WIDTH, rows, WIDTH, EPS, &ly.post_norm, t)?;
        o.linear(t, rows, &ly.gate_up, h)?;
        o.k.swiglu(h, rows, FFN, g)?;
        o.linear(g, rows, &ly.down, t)?;
        o.k.add(x, t, rows * WIDTH)
    }

    /// Reads a whole sequence of input embeddings x [L, 896] (the prefix, and in a check the tokens after it), writing its keys
    /// and values to slots 0..L of the pool; the logits [last, 33036] of its last `last` rows
    pub fn prefill(&self, o: &Ops, x: &DevBuf, l: usize, pool: &mut Pool, last: usize) -> Result<Vec<f32>> {
        if l > pool.cap || l > self.max_positions() {
            return Err(Error(format!("a prefix of {l} positions: the pool holds {}, the model {}", pool.cap, self.max_positions())));
        }
        let ids: Vec<i32> = (0..l as i32).collect();
        let pos = dev_i32(&o.gpu, &ids)?;
        let bufs = Bufs::new(o, l)?;
        let scratch = o.buf((HEADS * l * l).min(64 << 20).max(HEADS * l))?;
        let scale = 1.0 / (HD as f32).sqrt();
        for li in 0..LAYERS {
            let a = &bufs.a;
            self.layer(o, li, x, l, pool, &pos, &pos, &bufs, &|qkv, kp, vp| {
                o.k.attention((qkv, QKV, HD), (kp, KV, HD), (vp, KV, HD), 1, l, l, HEADS, KV_HEADS, HD, HD, scale, true, &scratch, (a, HEADS * HD))
            })?;
        }
        pool.used = l;
        self.logits(o, &x.view((l - last) * WIDTH * 4, last * WIDTH * 4)?, last)
    }

    fn logits(&self, o: &Ops, x: &DevBuf, rows: usize) -> Result<Vec<f32>> {
        let t = o.buf(rows * WIDTH)?;
        o.k.rms(x, WIDTH, rows, WIDTH, EPS, &self.norm, &t)?;
        let out = o.buf(rows * TOTAL)?;
        o.linear(&t, rows, &self.lm_head, &out)?;
        out.to_f32()
    }

    /// One decode step of `tokens.len()` rows (beams) at position `pos`: each row's token's keys and values go to a new
    /// slot (`pool.used + b`), appended to its history `slots` (int32 [rows, stride], `pos` long before, `pos + 1`
    /// after: the caller has written the new slot there); its logits [rows, 33036]
    pub fn step(&self, o: &Ops, tokens: &[u32], pos: usize, pool: &mut Pool, slots: &DevBuf, stride: usize) -> Result<Vec<f32>> {
        let b = tokens.len();
        if pool.used + b > pool.cap {
            return Err(Error("the key / value pool is full".into()));
        }
        if pos + 1 > self.max_positions() {
            return Err(Error("past the model's positions".into()));
        }
        let x = self.embed(o, tokens)?;
        let bufs = Bufs::new(o, b)?;
        let p = dev_i32(&o.gpu, &vec![pos as i32; b])?;
        let fresh: Vec<i32> = (0..b).map(|i| (pool.used + i) as i32).collect();
        let fresh = dev_i32(&o.gpu, &fresh)?;
        let scale = 1.0 / (HD as f32).sqrt();
        for li in 0..LAYERS {
            let a = &bufs.a;
            self.layer(o, li, &x, b, pool, &p, &fresh, &bufs, &|qkv, kp, vp| {
                o.k.attn_slots(qkv, QKV, kp, vp, slots, stride, pos + 1, b, HEADS, KV_HEADS, HD, scale, a)
            })?;
        }
        pool.used += b;
        self.logits(o, &x, b)
    }
}

/// The scratch of a layer on `rows` rows
struct Bufs {
    t: DevBuf,
    qkv: DevBuf,
    a: DevBuf,
    h: DevBuf,
    g: DevBuf,
}

impl Bufs {
    fn new(o: &Ops, rows: usize) -> Result<Bufs> {
        Ok(Bufs { t: o.buf(rows * WIDTH)?, qkv: o.buf(rows * QKV)?, a: o.buf(rows * HEADS * HD)?, h: o.buf(rows * 2 * FFN)?, g: o.buf(rows * FFN)? })
    }
}
