//! The paint model's denoiser: `UNet2p5DConditionModel` - Stable Diffusion 2.1's UNet (320 / 640 / 1280 / 1280,
//! 64-feature heads, GEGLU feed-forwards, v-prediction) with 12 input channels (the noisy latent, the view's normal
//! and position latents) whose transformer blocks gain, beside their self-attention and their cross-attention to a
//! learned text embedding per material:
//!
//! - **material-aware self-attention**: the albedo and the metallic-roughness (mr) pictures each have their own
//!   q / k / v / out;
//! - **reference attention**: the albedo stream's queries (all six views together) over the reference UNet's states
//!   of the same layer (`unet_dual`, run once on the reference picture's latent at t = 0), one value projection
//!   (and output) per material - Tencent concatenates the two values and splits them per head, so each head's value
//!   is columns [2 h d, 2 h d + 2 d) of [v_albedo | v_mr]; reordered at load into the two values each output reads;
//! - **multiview attention**: each material's six views attend jointly, q and k rotated by their pixels' 3D
//!   positions (voxel indices of the position maps, at the layer's resolution);
//! - **DINO attention**: cross-attention to DINOv2-giant's tokens of the reference picture (projected to 1,028
//!   tokens of 1,024).
//!
//! Classifier-free guidance runs three branches in one batch: (no reference, no DINO), (reference, no DINO) and
//! (reference, DINO); the batch is laid out (material, branch, view) - 36 pictures - so each material's rows, each
//! (material, branch)'s six views and the reference queries are contiguous.
#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;

use nextsycl_core::{Arena, DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt};

use super::nn::{floats, halves, timestep_features, Conv, Gn, Map, Mem, Resnet};
use crate::ops::{fp, hp, Lin, Ops, Weights, P};

const HEAD: usize = 64;
pub const VIEWS: usize = 6;
pub const MATS: usize = 2;
pub const BRANCHES: usize = 3;
const EPS_GN: f32 = 1e-5;
const EPS_TF_GN: f32 = 1e-6;
const EPS_LN: f32 = 1e-5;

/// The extra attentions of the main UNet's blocks
struct Extra {
    /// per material: q | k | v (no bias), out
    mda: Vec<(Lin, Lin)>,
    /// multiview: q | k | v (q and k rows reordered for the rotation's pairs), out
    mv: (Lin, Lin),
    /// reference: q, k, the two values, the two outputs
    rq: Lin,
    rk: Lin,
    rv: [Lin; 2],
    rout: [Lin; 2],
    /// DINO: q, k | v, out
    dq: Lin,
    dkv: Lin,
    dout: Lin,
}

struct Block {
    n1: (DevBuf, DevBuf),
    n2: (DevBuf, DevBuf),
    n3: (DevBuf, DevBuf),
    /// the reference UNet's plain self-attention: q | k | v, out
    sa: Option<(Lin, Lin)>,
    q2: Lin,
    kv2: Lin,
    out2: Lin,
    ff1: Lin,
    ff2: Lin,
    extra: Option<Extra>,
}

struct Tf {
    name: String,
    gn: Gn,
    proj_in: Lin,
    proj_out: Lin,
    blk: Block,
    c: usize,
}

#[allow(clippy::large_enum_variant)] // a few dozen, once
enum Layer {
    Res(Resnet),
    Tf(Tf),
    Down(Conv),
    Up(Conv),
}

pub struct Unet {
    conv_in: Conv,
    t1: Lin,
    t2: Lin,
    down: Vec<Vec<Layer>>,
    mid: Vec<Layer>,
    up: Vec<Vec<Layer>>,
    norm_out: Gn,
    conv_out: Conv,
    pub main: bool,
    pub bytes: usize,
}

/// Rows of a per-head rotation's pairs (2i, 2i + 1) regrouped as the shared kernel pairs them (i, 32 + i)
fn rope_rows(c: usize) -> Vec<usize> {
    (0..c).map(|r| {
        let (h, j) = (r / HEAD, r % HEAD);
        h * HEAD + if j < HEAD / 2 { 2 * j } else { 2 * (j - HEAD / 2) + 1 }
    }).collect()
}

fn norm(w: &Weights, p: &str) -> Result<(DevBuf, DevBuf)> {
    Ok((w.vec(&format!("{p}.weight"))?, w.vec(&format!("{p}.bias"))?))
}

fn qkv(w: &Weights, p: &str, sfx: &str, q_rows: Option<&[usize]>) -> Result<Lin> {
    let q = w.host_mat(&format!("{p}.to_q{sfx}.weight"))?;
    let k = w.host_mat(&format!("{p}.to_k{sfx}.weight"))?;
    let v = w.host_mat(&format!("{p}.to_v{sfx}.weight"))?;
    let (q, k) = match q_rows {
        Some(r) => (q.rows(r.iter().copied()), k.rows(r.iter().copied())),
        None => (q, k),
    };
    w.upload(&q.stack(&k).stack(&v), None)
}

impl Block {
    fn load(w: &Weights, p: &str, main: bool) -> Result<Block> {
        let t = format!("{p}.transformer");
        let c = w.shape(&format!("{t}.attn1.to_q.weight"))?[0];
        let kv2 = w.host_mat(&format!("{t}.attn2.to_k.weight"))?.stack(&w.host_mat(&format!("{t}.attn2.to_v.weight"))?);
        let extra = if main {
            let rv = w.host_mat(&format!("{p}.attn_refview.to_v.weight"))?.stack(&w.host_mat(&format!("{p}.attn_refview.processor.to_v_mr.weight"))?);
            // each head's value: [2 h d, 2 h d + 2 d) of [v_albedo | v_mr], its first d for albedo, its second for mr
            let heads = c / HEAD;
            let part = |s: usize| -> Vec<usize> { (0..heads).flat_map(|h| (0..HEAD).map(move |j| 2 * h * HEAD + s * HEAD + j)).collect() };
            let dkv = w.host_mat(&format!("{p}.attn_dino.to_k.weight"))?.stack(&w.host_mat(&format!("{p}.attn_dino.to_v.weight"))?);
            Some(Extra {
                mda: vec![
                    (qkv(w, &format!("{t}.attn1"), "", None)?, w.lin(&format!("{t}.attn1.to_out.0"))?),
                    (qkv(w, &format!("{t}.attn1.processor"), "_mr", None)?, w.lin(&format!("{t}.attn1.processor.to_out_mr.0"))?),
                ],
                mv: (qkv(w, &format!("{p}.attn_multiview"), "", Some(&rope_rows(c)))?, w.lin(&format!("{p}.attn_multiview.to_out.0"))?),
                rq: w.lin(&format!("{p}.attn_refview.to_q"))?,
                rk: w.lin(&format!("{p}.attn_refview.to_k"))?,
                rv: [w.upload(&rv.rows(part(0)), None)?, w.upload(&rv.rows(part(1)), None)?],
                rout: [w.lin(&format!("{p}.attn_refview.to_out.0"))?, w.lin(&format!("{p}.attn_refview.processor.to_out_mr.0"))?],
                dq: w.lin(&format!("{p}.attn_dino.to_q"))?,
                dkv: w.upload(&dkv, None)?,
                dout: w.lin(&format!("{p}.attn_dino.to_out.0"))?,
            })
        } else {
            None
        };
        Ok(Block {
            n1: norm(w, &format!("{t}.norm1"))?,
            n2: norm(w, &format!("{t}.norm2"))?,
            n3: norm(w, &format!("{t}.norm3"))?,
            sa: if main { None } else { Some((qkv(w, &format!("{t}.attn1"), "", None)?, w.lin(&format!("{t}.attn1.to_out.0"))?)) },
            q2: w.lin(&format!("{t}.attn2.to_q"))?,
            kv2: w.upload(&kv2, None)?,
            out2: w.lin(&format!("{t}.attn2.to_out.0"))?,
            ff1: w.lin(&format!("{t}.ff.net.0.proj"))?,
            ff2: w.lin(&format!("{t}.ff.net.2"))?,
            extra,
        })
    }
}

/// What a generation's main passes read beside the latents, per transformer block (by its name)
pub struct Cond {
    /// the text's keys | values per material (main) or the reference's (dual): [77, 2c] half
    pub text: BTreeMap<String, Vec<DevBuf>>,
    pub text_rows: usize,
    /// the reference attention's k, v_albedo, v_mr: [L, c] half each
    pub refkv: BTreeMap<String, [DevBuf; 3]>,
    /// DINO's keys | values, the unconditional (zero) and the picture's: [1028, 2c] half
    pub dino: BTreeMap<String, [DevBuf; 2]>,
    pub dino_rows: usize,
    /// the multiview rotation's (cos, sin) per row and pair, by token count of a view group: [36 L, 32, 2]
    pub rope: BTreeMap<usize, DevBuf>,
    /// the reference UNet's stored states by block name: [L, c] half
    pub states: BTreeMap<String, DevBuf>,
}

impl Unet {
    /// The main UNet (`unet.`) or the reference one (`unet_dual.`)
    pub fn load(w: &Weights, main: bool) -> Result<Unet> {
        let mut down = Vec::new();
        for i in 0..4 {
            let p = format!("down_blocks.{i}");
            let mut v = Vec::new();
            for j in 0..2 {
                v.push(Layer::Res(Resnet::load(w, &format!("{p}.resnets.{j}"), EPS_GN)?));
                if w.has(&format!("{p}.attentions.{j}.proj_in.weight")) {
                    v.push(Layer::Tf(Tf::load(w, &format!("{p}.attentions.{j}"), &format!("down_{i}_{j}_0"), main)?));
                }
            }
            if w.has(&format!("{p}.downsamplers.0.conv.weight")) {
                v.push(Layer::Down(Conv::load(w, &format!("{p}.downsamplers.0.conv"), Some((1, 1, 1, 1)))?));
            }
            down.push(v);
        }
        let mid = vec![
            Layer::Res(Resnet::load(w, "mid_block.resnets.0", EPS_GN)?),
            Layer::Tf(Tf::load(w, "mid_block.attentions.0", "mid_0_0", main)?),
            Layer::Res(Resnet::load(w, "mid_block.resnets.1", EPS_GN)?),
        ];
        let mut up = Vec::new();
        for i in 0..4 {
            let p = format!("up_blocks.{i}");
            let mut v = Vec::new();
            for j in 0..3 {
                v.push(Layer::Res(Resnet::load(w, &format!("{p}.resnets.{j}"), EPS_GN)?));
                if w.has(&format!("{p}.attentions.{j}.proj_in.weight")) {
                    v.push(Layer::Tf(Tf::load(w, &format!("{p}.attentions.{j}"), &format!("up_{i}_{j}_0"), main)?));
                }
            }
            if w.has(&format!("{p}.upsamplers.0.conv.weight")) {
                v.push(Layer::Up(Conv::load(w, &format!("{p}.upsamplers.0.conv"), None)?));
            }
            up.push(v);
        }
        Ok(Unet {
            conv_in: Conv::load(w, "conv_in", None)?,
            t1: w.lin("time_embedding.linear_1")?,
            t2: w.lin("time_embedding.linear_2")?,
            down,
            mid,
            up,
            norm_out: Gn::load(w, "conv_norm_out", EPS_GN)?,
            conv_out: Conv::load(w, "conv_out", None)?,
            main,
            bytes: w.bytes.get(),
        })
    }

    fn blocks(&self) -> impl Iterator<Item = &Tf> {
        self.down.iter().flatten().chain(self.mid.iter()).chain(self.up.iter().flatten()).filter_map(|l| if let Layer::Tf(t) = l { Some(t) } else { None })
    }

    /// silu(the timestep embedding) as half [1280], for the residual blocks
    fn temb(&self, ops: &Ops, a: &Arena, t: f32) -> Result<DevBuf> {
        let f = super::nn::upload_half(a, &timestep_features(t, self.t1.k))?;
        let h = floats(a, self.t1.n)?;
        ops.lin(&self.t1, f.ptr(), 1, h.ptr(), Dt::F32)?;
        let hs = halves(a, self.t1.n)?;
        ops.silu(h.ptr(), hs.ptr(), self.t1.n)?;
        let e = floats(a, self.t2.n)?;
        ops.lin(&self.t2, hs.ptr(), 1, e.ptr(), Dt::F32)?;
        let es = halves(a, self.t2.n)?;
        ops.silu(e.ptr(), es.ptr(), self.t2.n)?;
        Ok(es)
    }

    /// The text's keys and values for every block: `text[m]` [rows, 1024] half (the main UNet: one a material;
    /// the reference UNet: one)
    pub fn text_kv(&self, ops: &Ops, text: &[DevBuf], rows: usize, out: &mut Cond) -> Result<()> {
        for b in self.blocks() {
            let mut v = Vec::new();
            for t in text {
                let o = DevBuf::new(&ops.gpu, rows * b.blk.kv2.n * 2)?;
                ops.lin(&b.blk.kv2, t.ptr(), rows, o.ptr(), Dt::F16)?;
                v.push(o);
            }
            out.text.insert(b.name.clone(), v);
        }
        out.text_rows = rows;
        Ok(())
    }

    /// DINO's keys and values for every block: `dino` = [zero, picture] projected tokens [rows, 1024] half
    pub fn dino_kv(&self, ops: &Ops, dino: &[DevBuf; 2], rows: usize, out: &mut Cond) -> Result<()> {
        for b in self.blocks() {
            let e = b.blk.extra.as_ref().ok_or_else(|| Error("the reference UNet has no DINO attention".into()))?;
            let mk = |d: &DevBuf| -> Result<DevBuf> {
                let o = DevBuf::new(&ops.gpu, rows * e.dkv.n * 2)?;
                ops.lin(&e.dkv, d.ptr(), rows, o.ptr(), Dt::F16)?;
                Ok(o)
            };
            out.dino.insert(b.name.clone(), [mk(&dino[0])?, mk(&dino[1])?]);
        }
        out.dino_rows = rows;
        Ok(())
    }

    /// The reference attention's keys and values for every block, from the reference UNet's states
    pub fn ref_kv(&self, ops: &Ops, cond: &mut Cond) -> Result<()> {
        for b in self.blocks() {
            let e = b.blk.extra.as_ref().ok_or_else(|| Error("the reference UNet has no reference attention".into()))?;
            let s = cond.states.get(&b.name).ok_or_else(|| Error(format!("no reference state for {}", b.name)))?;
            let rows = s.len / 2 / b.c;
            let mk = |l: &Lin| -> Result<DevBuf> {
                let o = DevBuf::new(&ops.gpu, rows * l.n * 2)?;
                ops.lin(l, s.ptr(), rows, o.ptr(), Dt::F16)?;
                Ok(o)
            };
            let kv = [mk(&e.rk)?, mk(&e.rv[0])?, mk(&e.rv[1])?];
            cond.refkv.insert(b.name.clone(), kv);
        }
        Ok(())
    }

    /// The UNet on x (n pictures, `channels` in) at timestep t; the main UNet's output (v) into `out` (float32
    /// [n, h, w, 4]); the reference UNet's into `cond.states` (its blocks' normalized inputs)
    pub fn forward(&self, ops: &Ops, mem: &Mem, x: &Map, t: f32, cond: &mut Cond, out: Option<&DevBuf>) -> Result<()> {
        let mark = mem.keep.mark();
        let temb = self.temb(ops, &mem.keep, t)?;
        let te = Some(temb.ptr() as P);
        let mut h = self.conv_in.run(ops, &mem.tmp, &mem.keep, x, None)?;
        mem.end_block();
        let mut skips: Vec<Map> = Vec::new();
        let keep_skip = |m: &Map| Map { buf: m.buf.view(0, m.buf.len).expect("a view"), n: m.n, h: m.h, w: m.w, c: m.c };
        skips.push(keep_skip(&h));
        for blk in &self.down {
            for l in blk {
                h = match l {
                    Layer::Res(r) => r.run(ops, mem, &h, te)?,
                    Layer::Tf(tf) => tf.run(ops, mem, &h, cond, self.main)?,
                    Layer::Down(c) => c.run(ops, &mem.tmp, &mem.keep, &h, None)?,
                    Layer::Up(_) => return Err(Error("an upsampler in a down block".into())),
                };
                crate::prof::part(&ops.nsd, if matches!(l, Layer::Res(_)) { "unet resnet" } else if matches!(l, Layer::Tf(_)) { "unet tf out" } else { "unet resample" })?;
                mem.end_block();
                // the skips: every resnet / attention pair's output (the attention's when there is one) and the
                // downsampler's
                let next_is_tf = false;
                let _ = next_is_tf;
                if !matches!(l, Layer::Res(_)) || !blk.iter().any(|x| matches!(x, Layer::Tf(_))) {
                    skips.push(keep_skip(&h));
                }
            }
        }
        for l in &self.mid {
            h = match l {
                Layer::Res(r) => r.run(ops, mem, &h, te)?,
                Layer::Tf(tf) => tf.run(ops, mem, &h, cond, self.main)?,
                _ => return Err(Error("a resampler in the middle block".into())),
            };
            crate::prof::part(&ops.nsd, if matches!(l, Layer::Res(_)) { "unet resnet" } else { "unet tf out" })?;
            mem.end_block();
        }
        for blk in &self.up {
            for l in blk {
                h = match l {
                    Layer::Res(r) => {
                        let s = skips.pop().ok_or_else(|| Error("the skips ran out".into()))?;
                        let cat = Map::new(&mem.tmp, h.n, h.h, h.w, h.c + s.c)?;
                        ops.concat(h.p(), h.c, s.p(), s.c, h.rows(), cat.buf.ptr())?;
                        r.run(ops, mem, &cat, te)?
                    }
                    Layer::Tf(tf) => tf.run(ops, mem, &h, cond, self.main)?,
                    Layer::Up(c) => {
                        let up = Map::new(&mem.tmp, h.n, 2 * h.h, 2 * h.w, h.c)?;
                        ops.up2(h.p(), h.n, h.h, h.w, h.c, up.buf.ptr())?;
                        c.run(ops, &mem.tmp, &mem.keep, &up, None)?
                    }
                    Layer::Down(_) => return Err(Error("a downsampler in an up block".into())),
                };
                crate::prof::part(&ops.nsd, if matches!(l, Layer::Res(_)) { "unet resnet" } else if matches!(l, Layer::Tf(_)) { "unet tf out" } else { "unet resample" })?;
                mem.end_block();
            }
        }
        if let Some(o) = out {
            let hn = self.norm_out.run(ops, &mem.tmp, &mem.tmp, &h, true)?;
            let v = self.conv_out.run(ops, &mem.tmp, &mem.tmp, &hn, None)?;
            ops.to_float(v.p(), o.ptr(), v.len())?;
        }
        ops.gpu.sync()?;
        mem.end_block();
        mem.keep.rewind(mark);
        Ok(())
    }
}

impl Tf {
    fn load(w: &Weights, p: &str, name: &str, main: bool) -> Result<Tf> {
        let c = w.shape(&format!("{p}.proj_in.weight"))?[0];
        Ok(Tf {
            name: name.to_string(),
            gn: Gn::load(w, &format!("{p}.norm"), EPS_TF_GN)?,
            proj_in: w.lin(&format!("{p}.proj_in"))?,
            proj_out: w.lin(&format!("{p}.proj_out"))?,
            blk: Block::load(w, &format!("{p}.transformer_blocks.0"), main)?,
            c,
        })
    }

    /// The attention block (Transformer2DModel) on x: x + proj_out(block(proj_in(gn(x))))
    fn run(&self, ops: &Ops, mem: &Mem, x: &Map, cond: &mut Cond, main: bool) -> Result<Map> {
        let (tmp, c) = (&mem.tmp, self.c);
        let rows = x.rows();
        let l = x.h * x.w;
        let g = self.gn.run(ops, tmp, tmp, x, false)?;
        // the residual stream in float32
        let hs = floats(tmp, rows * c)?;
        ops.lin(&self.proj_in, g.p(), rows, hs.ptr(), Dt::F32)?;
        let b = &self.blk;
        let heads = c / HEAD;
        let nh = halves(tmp, rows * c)?;
        let att = halves(tmp, rows * c)?;
        let o = halves(tmp, rows * c)?;
        ops.ln(hs.ptr(), false, rows, c, EPS_LN, b.n1.0.ptr(), b.n1.1.ptr(), none(), none(), nh.ptr(), true)?;
        crate::prof::part(&ops.nsd, "unet tf in")?;
        match (&b.extra, main) {
            (None, false) => {
                // the reference UNet: its states kept (the main UNet's reference attention reads them), then its
                // plain self-attention
                let st = DevBuf::new(&ops.gpu, rows * c * 2)?;
                st.copy_within(0, &nh, 0, rows * c * 2)?;
                cond.states.insert(self.name.clone(), st);
                let (wqkv, wo) = b.sa.as_ref().expect("the reference UNet's self-attention");
                let qkv = halves(tmp, rows * 3 * c)?;
                ops.lin(wqkv, nh.ptr(), rows, qkv.ptr(), Dt::F16)?;
                ops.attention_batch(qkv.ptr(), hp(&qkv, c), hp(&qkv, 2 * c), x.n, l, heads, HEAD, 3 * c, att.ptr())?;
                ops.lin(wo, att.ptr(), rows, o.ptr(), Dt::F16)?;
                ops.add(hs.ptr(), o.ptr(), true, rows, c, none())?;
            }
            (Some(e), true) => self.extras(ops, tmp, e, x, &hs, &nh, &att, &o, cond)?,
            _ => return Err(Error("a block's attentions and its UNet disagree".into())),
        }
        crate::prof::part(&ops.nsd, "unet tf multiview")?;
        // the text cross-attention (and the main UNet's DINO attention) on norm2
        ops.ln(hs.ptr(), false, rows, c, EPS_LN, b.n2.0.ptr(), b.n2.1.ptr(), none(), none(), nh.ptr(), true)?;
        let q = halves(tmp, rows * c)?;
        ops.lin(&b.q2, nh.ptr(), rows, q.ptr(), Dt::F16)?;
        let text = cond.text.get(&self.name).ok_or_else(|| Error(format!("no text keys for {}", self.name)))?;
        let per = rows / text.len();
        for (m, kv) in text.iter().enumerate() {
            ops.cross_attention(hp(&q, m * per * c), per, c, kv.ptr(), hp(kv, c), cond.text_rows, 2 * c, heads, HEAD, hp(&att, m * per * c))?;
        }
        ops.lin(&b.out2, att.ptr(), rows, o.ptr(), Dt::F16)?;
        ops.add(hs.ptr(), o.ptr(), true, rows, c, none())?;
        crate::prof::part(&ops.nsd, "unet tf text")?;
        if let Some(e) = &b.extra {
            let dino = cond.dino.get(&self.name).ok_or_else(|| Error(format!("no DINO keys for {}", self.name)))?;
            ops.lin(&e.dq, nh.ptr(), rows, q.ptr(), Dt::F16)?;
            // per material: branches 0 and 1 (no DINO) together, then branch 2
            let img = l;
            for m in 0..MATS {
                let r0 = m * BRANCHES * VIEWS * img;
                for (kv, first, count) in [(&dino[0], r0, 2 * VIEWS * img), (&dino[1], r0 + 2 * VIEWS * img, VIEWS * img)] {
                    ops.cross_attention(hp(&q, first * c), count, c, kv.ptr(), hp(kv, c), cond.dino_rows, 2 * c, heads, HEAD, hp(&att, first * c))?;
                }
            }
            ops.lin(&e.dout, att.ptr(), rows, o.ptr(), Dt::F16)?;
            ops.add(hs.ptr(), o.ptr(), true, rows, c, none())?;
        }
        crate::prof::part(&ops.nsd, "unet tf dino")?;
        // the feed-forward (GEGLU)
        ops.ln(hs.ptr(), false, rows, c, EPS_LN, b.n3.0.ptr(), b.n3.1.ptr(), none(), none(), nh.ptr(), true)?;
        let f2 = b.ff1.n;
        let mid = halves(tmp, rows * f2)?;
        ops.lin(&b.ff1, nh.ptr(), rows, mid.ptr(), Dt::F16)?;
        let act = halves(tmp, rows * f2 / 2)?;
        ops.geglu(mid.ptr(), rows, f2 / 2, act.ptr())?;
        ops.lin(&b.ff2, act.ptr(), rows, o.ptr(), Dt::F16)?;
        ops.add(hs.ptr(), o.ptr(), true, rows, c, none())?;
        crate::prof::part(&ops.nsd, "unet tf ff")?;
        // out
        let hh = halves(tmp, rows * c)?;
        ops.to_half(hs.ptr(), hh.ptr(), rows * c)?;
        let out = Map::new(&mem.keep, x.n, x.h, x.w, c)?;
        ops.lin(&self.proj_out, hh.ptr(), rows, out.buf.ptr(), Dt::F16)?;
        ops.add_half(out.buf.ptr(), x.p(), out.len(), 1.0)?;
        Ok(out)
    }

    /// The main UNet's self-attentions: per material, the reference, the multiview
    #[allow(clippy::too_many_arguments)]
    fn extras(&self, ops: &Ops, tmp: &Arena, e: &Extra, x: &Map, hs: &DevBuf, nh: &DevBuf, att: &DevBuf, o: &DevBuf, cond: &Cond) -> Result<()> {
        let c = self.c;
        let rows = x.rows();
        let l = x.h * x.w;
        let heads = c / HEAD;
        if x.n != MATS * BRANCHES * VIEWS {
            return Err(Error(format!("the main UNet takes {} pictures, not {}", MATS * BRANCHES * VIEWS, x.n)));
        }
        let per_mat = rows / MATS;
        // material-aware self-attention, each picture on its own
        let qkv = halves(tmp, rows * 3 * c)?;
        for (m, (w, _)) in e.mda.iter().enumerate() {
            ops.lin(w, hp(nh, m * per_mat * c), per_mat, hp(&qkv, m * per_mat * 3 * c), Dt::F16)?;
        }
        ops.attention_batch(qkv.ptr(), hp(&qkv, c), hp(&qkv, 2 * c), x.n, l, heads, HEAD, 3 * c, att.ptr())?;
        for (m, (_, wo)) in e.mda.iter().enumerate() {
            ops.lin(wo, hp(att, m * per_mat * c), per_mat, hp(o, m * per_mat * c), Dt::F16)?;
        }
        ops.add(hs.ptr(), o.ptr(), true, rows, c, none())?;
        crate::prof::part(&ops.nsd, "unet tf self (mda)")?;
        // the reference attention: the albedo rows of branches 1 and 2 (branch 0 has none: its scale is 0) ask the
        // reference UNet's states; one value and output a material, added into that material's rows
        let kv = cond.refkv.get(&self.name).ok_or_else(|| Error(format!("no reference keys for {}", self.name)))?;
        let lr = kv[0].len / 2 / c;
        let q0 = VIEWS * l;
        let nq = 2 * VIEWS * l;
        let q = halves(tmp, nq * c)?;
        ops.lin(&e.rq, hp(nh, q0 * c), nq, q.ptr(), Dt::F16)?;
        let ra = halves(tmp, nq * c)?;
        let ro = halves(tmp, nq * c)?;
        for m in 0..MATS {
            ops.cross_attention(q.ptr(), nq, c, kv[0].ptr(), kv[1 + m].ptr(), lr, c, heads, HEAD, ra.ptr())?;
            ops.lin(&e.rout[m], ra.ptr(), nq, ro.ptr(), Dt::F16)?;
            let dst = (m * BRANCHES * VIEWS * l + q0) * c;
            ops.add(fp(hs, dst).cast(), ro.ptr(), true, nq, c, none())?;
        }
        crate::prof::part(&ops.nsd, "unet tf reference")?;
        // the multiview attention: each (material, branch)'s six views together, q and k rotated by position
        let (w, wo) = &e.mv;
        ops.lin(w, nh.ptr(), rows, qkv.ptr(), Dt::F16)?;
        let group = VIEWS * l;
        let cs = cond.rope.get(&group).ok_or_else(|| Error(format!("no rotation for {group} tokens")))?;
        ops.nsd.rope(qkv.ptr(), Dt::F16, rows, heads, HEAD, 3 * c, cs.ptr(), HEAD)?;
        ops.nsd.rope(hp(&qkv, c), Dt::F16, rows, heads, HEAD, 3 * c, cs.ptr(), HEAD)?;
        ops.attention_batch(qkv.ptr(), hp(&qkv, c), hp(&qkv, 2 * c), MATS * BRANCHES, group, heads, HEAD, 3 * c, att.ptr())?;
        ops.lin(wo, att.ptr(), rows, o.ptr(), Dt::F16)?;
        ops.add(hs.ptr(), o.ptr(), true, rows, c, none())?;
        Ok(())
    }
}
