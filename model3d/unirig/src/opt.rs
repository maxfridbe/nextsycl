//! The skeleton decoder: OPT-350m's shape (24 pre-norm layers, width 1024, 16 heads, ReLU FFN of 4096, learned
//! positions offset by 2) over a 267-token vocabulary, its prefix the shape encoder's 1,024 latents through output_proj
//! and the start tokens. Keys and values go to a pool of slots shared by the beams: a beam's history is a list of
//! slots, so following another beam copies that list, not the cache.

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::ffi::RELU;
use crate::ops::{Lin, Ln, Ops, Weights};
use crate::tokens::VOCAB;

pub const WIDTH: usize = 1024;
const LAYERS: usize = 24;
const HEADS: usize = 16;
const FFN: usize = 4096;
const POSITIONS: usize = 3078;

struct Layer {
    ln1: Ln,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2: Ln,
    fc1: Lin,
    fc2: Lin,
}

pub struct Opt {
    pub output_proj: Lin,
    embed: DevBuf,
    pos: DevBuf,
    layers: Vec<Layer>,
    final_ln: Ln,
    lm_head: Lin,
}

/// The keys and values of every layer, by slot
pub struct Pool {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub cap: usize,
    pub used: usize,
}

impl Opt {
    pub fn load(w: &Weights) -> Result<Opt> {
        let d = "transformer.model.decoder";
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("{d}.layers.{l}");
            layers.push(Layer {
                ln1: w.ln(&format!("{p}.self_attn_layer_norm"), WIDTH)?,
                q: w.lin(&format!("{p}.self_attn.q_proj"), WIDTH, WIDTH, true)?,
                k: w.lin(&format!("{p}.self_attn.k_proj"), WIDTH, WIDTH, true)?,
                v: w.lin(&format!("{p}.self_attn.v_proj"), WIDTH, WIDTH, true)?,
                o: w.lin(&format!("{p}.self_attn.out_proj"), WIDTH, WIDTH, true)?,
                ln2: w.ln(&format!("{p}.final_layer_norm"), WIDTH)?,
                fc1: w.lin(&format!("{p}.fc1"), FFN, WIDTH, true)?,
                fc2: w.lin(&format!("{p}.fc2"), WIDTH, FFN, true)?,
            });
        }
        Ok(Opt {
            output_proj: w.lin("output_proj", WIDTH, crate::encoder::WIDTH, true)?,
            embed: w.dev(&format!("{d}.embed_tokens.weight"), &[VOCAB, WIDTH])?,
            pos: w.dev(&format!("{d}.embed_positions.weight"), &[POSITIONS, WIDTH])?,
            layers,
            final_ln: w.ln(&format!("{d}.final_layer_norm"), WIDTH)?,
            lm_head: w.lin("transformer.lm_head", VOCAB, WIDTH, false)?,
        })
    }

    /// The most positions it has: prefix and generated tokens together
    pub fn max_positions(&self) -> usize {
        POSITIONS - 2
    }

    pub fn pool(&self, o: &Ops, cap: usize) -> Result<Pool> {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for _ in 0..LAYERS {
            k.push(o.buf(cap * WIDTH)?);
            v.push(o.buf(cap * WIDTH)?);
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
            new.k[l].copy_within(0, &pool.k[l], 0, pool.used * WIDTH * 4)?;
            new.v[l].copy_within(0, &pool.v[l], 0, pool.used * WIDTH * 4)?;
        }
        new.used = pool.used;
        *pool = new;
        Ok(())
    }

    /// Bytes of a pool of `cap` slots
    pub fn pool_bytes(cap: usize) -> usize {
        LAYERS * 2 * cap * WIDTH * 4
    }

    /// x [rows, 1024] += the positions' embeddings (positions p0, p0 + 1, ...: offset by 2)
    fn add_positions(&self, o: &Ops, x: &DevBuf, rows: usize, p0: usize, same: bool) -> Result<()> {
        let idx: Vec<i32> = (0..rows).map(|r| (if same { p0 } else { p0 + r } + 2) as i32).collect();
        let idx = crate::ops::dev_i32(&o.gpu, &idx)?;
        let t = o.buf(rows * WIDTH)?;
        o.k.gather(&self.pos, WIDTH, &idx, rows, WIDTH, &t)?;
        o.k.add(x, &t, rows * WIDTH)
    }

    /// The token embeddings of `ids` [n, 1024]
    pub fn embed(&self, o: &Ops, ids: &[u32]) -> Result<DevBuf> {
        let idx: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let idx = crate::ops::dev_i32(&o.gpu, &idx)?;
        let out = o.buf(ids.len().max(1) * WIDTH)?;
        o.k.gather(&self.embed, WIDTH, &idx, ids.len(), WIDTH, &out)?;
        Ok(out)
    }

    /// Reads a whole sequence of input embeddings x [L, 1024] (the prefix), writing its keys and values to slots 0..L
    /// of the pool; the logits [last, 267] of its last `last` rows
    pub fn prefill(&self, o: &Ops, x: &DevBuf, l: usize, pool: &mut Pool, last: usize) -> Result<Vec<f32>> {
        if l > pool.cap || l > self.max_positions() {
            return Err(Error(format!("a prefix of {l} positions: the pool holds {}, the model {}", pool.cap, self.max_positions())));
        }
        self.add_positions(o, x, l, 0, false)?;
        let hd = WIDTH / HEADS;
        let (t, q, a, h) = (o.buf(l * WIDTH)?, o.buf(l * WIDTH)?, o.buf(l * WIDTH)?, o.buf(l * FFN)?);
        let scratch = o.buf((HEADS * l * l).min(64 << 20).max(HEADS * l))?;
        for (li, ly) in self.layers.iter().enumerate() {
            let kc = pool.k[li].view(0, l * WIDTH * 4)?;
            let vc = pool.v[li].view(0, l * WIDTH * 4)?;
            o.ln(x, l, WIDTH, Some(&ly.ln1), &t)?;
            o.linear(&t, l, &ly.q, &q)?;
            o.linear(&t, l, &ly.k, &kc)?;
            o.linear(&t, l, &ly.v, &vc)?;
            o.k.attention((&q, WIDTH, hd), (&kc, WIDTH, hd), (&vc, WIDTH, hd), 1, l, l, HEADS, hd, hd, 1.0 / (hd as f32).sqrt(), true, &scratch,
                          (&a, WIDTH))?;
            o.linear(&a, l, &ly.o, &t)?;
            o.k.add(x, &t, l * WIDTH)?;
            self.ffn(o, ly, x, l, &t, &h)?;
        }
        pool.used = l;
        self.logits(o, &x.view((l - last) * WIDTH * 4, last * WIDTH * 4)?, last)
    }

    fn ffn(&self, o: &Ops, ly: &Layer, x: &DevBuf, rows: usize, t: &DevBuf, h: &DevBuf) -> Result<()> {
        o.ln(x, rows, WIDTH, Some(&ly.ln2), t)?;
        o.linear(t, rows, &ly.fc1, h)?;
        o.k.act(h, rows * FFN, RELU)?;
        o.linear(h, rows, &ly.fc2, t)?;
        o.k.add(x, t, rows * WIDTH)
    }

    fn logits(&self, o: &Ops, x: &DevBuf, rows: usize) -> Result<Vec<f32>> {
        let t = o.buf(rows * WIDTH)?;
        o.ln(x, rows, WIDTH, Some(&self.final_ln), &t)?;
        let out = o.buf(rows * VOCAB)?;
        o.linear(&t, rows, &self.lm_head, &out)?;
        out.to_f32()
    }

    /// One decode step of `tokens.len()` rows (beams) at position `pos`: each row's token's keys and values go to a new
    /// slot (`pool.used + b`), appended to its history `slots` (int32 [rows, stride], `pos` long before, `pos + 1`
    /// after: the caller has written the new slot there); its logits [rows, 267]
    pub fn step(&self, o: &Ops, tokens: &[u32], pos: usize, pool: &mut Pool, slots: &DevBuf, stride: usize) -> Result<Vec<f32>> {
        let b = tokens.len();
        if pool.used + b > pool.cap {
            return Err(Error("the key / value pool is full".into()));
        }
        if pos + 1 > self.max_positions() {
            return Err(Error("past the model's positions".into()));
        }
        let x = self.embed(o, tokens)?;
        self.add_positions(o, &x, b, pos, true)?;
        let hd = WIDTH / HEADS;
        let (t, q, kk, vv, a, h) = (o.buf(b * WIDTH)?, o.buf(b * WIDTH)?, o.buf(b * WIDTH)?, o.buf(b * WIDTH)?, o.buf(b * WIDTH)?, o.buf(b * FFN)?);
        let fresh: Vec<i32> = (0..b).map(|i| (pool.used + i) as i32).collect();
        let fresh = crate::ops::dev_i32(&o.gpu, &fresh)?;
        for (li, ly) in self.layers.iter().enumerate() {
            o.ln(&x, b, WIDTH, Some(&ly.ln1), &t)?;
            o.linear(&t, b, &ly.q, &q)?;
            o.linear(&t, b, &ly.k, &kk)?;
            o.linear(&t, b, &ly.v, &vv)?;
            o.k.scatter(&kk, &fresh, b, WIDTH, &pool.k[li])?;
            o.k.scatter(&vv, &fresh, b, WIDTH, &pool.v[li])?;
            o.k.attn_slots(&q, &pool.k[li], &pool.v[li], slots, stride, pos + 1, b, HEADS, hd, 1.0 / (hd as f32).sqrt(), &a)?;
            o.linear(&a, b, &ly.o, &t)?;
            o.k.add(&x, &t, b * WIDTH)?;
            self.ffn(o, ly, &x, b, &t, &h)?;
        }
        pool.used += b;
        self.logits(o, &x, b)
    }
}
