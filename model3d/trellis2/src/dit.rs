//! The flow transformer every TRELLIS.2 stage runs (SparseStructureFlowModel on the dense 16^3 grid, SLatFlowModel on
//! the sparse voxels: the same block): 30 blocks of 1536 (12 heads of 128), a shared adaLN modulation plus each
//! block's own, self-attention with TRELLIS's per-head RMS norm and interleaved 3-D rotary positions over every
//! token, cross-attention to the picture's tokens (RMS-normed too), a GELU-tanh MLP of 8192; float32 residual stream,
//! half activations, the Q8_0 matrices expanded to half before each product. Pixal3D's blocks add their projected
//! features after the cross-attention (`proj_linear`).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::w::{dev_half, File, Keep, Lin};
use crate::Ctx;

pub const C: usize = 1536;
pub const HEADS: usize = 12;
pub const HD: usize = 128;
const MLP: usize = 8192;

struct Block {
    modulation: Vec<f32>,
    qkv: Lin,
    q_gamma: DevBuf,
    k_gamma: DevBuf,
    out: Lin,
    norm2: (DevBuf, DevBuf),
    cq: Lin,
    ckv: Lin,
    cq_gamma: DevBuf,
    ck_gamma: DevBuf,
    cout: Lin,
    proj: Option<Lin>,
    mlp0: Lin,
    mlp2: Lin,
}

/// A host matrix [n, k] and bias for the timestep embedding and the shared modulation
struct HostLin {
    w: Vec<f32>,
    b: Vec<f32>,
    n: usize,
    k: usize,
}

impl HostLin {
    fn load(f: &File, p: &str) -> Result<HostLin> {
        let (sh, w) = f.host(&format!("{p}.weight"))?;
        Ok(HostLin { w, b: f.vec(&format!("{p}.bias"))?, n: sh[0], k: sh[1] })
    }
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n).map(|o| {
            let row = &self.w[o * self.k..(o + 1) * self.k];
            self.b[o] + row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    }
}

fn silu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x / (1.0 + (-x).exp())).collect()
}

pub struct Dit {
    pub in_ch: usize,
    pub out_ch: usize,
    pub cond_ch: usize,
    /// Pixal3D: the projected features' width (2048 or 1024)
    pub proj_ch: Option<usize>,
    input: Lin,
    t0: HostLin,
    t2: HostLin,
    ada: HostLin,
    blocks: Vec<Block>,
    out: Lin,
    scratch: DevBuf,
    pub bytes: usize,
}

/// A call's buffers, sized for n tokens and l condition tokens
pub struct Work {
    n: usize,
    h: DevBuf,
    hn: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    o: DevBuf,
    o2: DevBuf,
    mlp: DevBuf,
    kv: DevBuf,
    x: DevBuf,
    out: DevBuf,
}

/// The first call's insides, for the check: the input layer's output, block 0's and the last block's
#[derive(Default)]
pub struct Probe {
    pub input: Vec<f32>,
    pub block0: Vec<f32>,
    pub last: Vec<f32>,
}

/// TRELLIS's 3-D rotary phases for token coordinates: 21 frequencies (10000^(-j/21)) per axis x, y, z, then one
/// identity pair - (cos, sin) per token and pair [n, 64, 2]
pub fn rope(coords: &[[i32; 3]]) -> Vec<f32> {
    let fd = HD / 2 / 3;
    let freqs: Vec<f32> = (0..fd).map(|j| 1.0 / 10000f32.powf(j as f32 / fd as f32)).collect();
    let mut cs = vec![0f32; coords.len() * HD];
    for (i, c) in coords.iter().enumerate() {
        let row = &mut cs[i * HD..(i + 1) * HD];
        for a in 0..3 {
            for (j, f) in freqs.iter().enumerate() {
                let ang = c[a] as f32 * f;
                row[(a * fd + j) * 2] = ang.cos();
                row[(a * fd + j) * 2 + 1] = ang.sin();
            }
        }
        for p in 3 * fd..HD / 2 {
            row[p * 2] = 1.0;
            row[p * 2 + 1] = 0.0;
        }
    }
    cs
}

impl Dit {
    pub fn load(f: &File, gpu: &Arc<Gpu>) -> Result<Dit> {
        let nb = (0..).take_while(|i| f.has(&format!("blocks.{i}.modulation"))).count();
        if nb == 0 {
            return Err(Error(format!("{}: no TRELLIS flow blocks (blocks.N.modulation)", f.path())));
        }
        let cross = if f.has("blocks.0.cross_attn.cross_attn_block.to_q.weight") { "cross_attn.cross_attn_block" } else { "cross_attn" };
        let gamma = |p: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &f.vec(p)?) };
        let mut blocks = Vec::new();
        let mut bytes = 0;
        for i in 0..nb {
            let b = format!("blocks.{i}");
            let c = format!("{b}.{cross}");
            let blk = Block {
                modulation: f.vec(&format!("{b}.modulation"))?,
                qkv: Lin::load(f, gpu, &format!("{b}.self_attn.to_qkv"), Keep::Stored)?,
                q_gamma: gamma(&format!("{b}.self_attn.q_rms_norm.gamma"))?,
                k_gamma: gamma(&format!("{b}.self_attn.k_rms_norm.gamma"))?,
                out: Lin::load(f, gpu, &format!("{b}.self_attn.to_out"), Keep::Stored)?,
                norm2: (gamma(&format!("{b}.norm2.weight"))?, gamma(&format!("{b}.norm2.bias"))?),
                cq: Lin::load(f, gpu, &format!("{c}.to_q"), Keep::Stored)?,
                ckv: Lin::load(f, gpu, &format!("{c}.to_kv"), Keep::Stored)?,
                cq_gamma: gamma(&format!("{c}.q_rms_norm.gamma"))?,
                ck_gamma: gamma(&format!("{c}.k_rms_norm.gamma"))?,
                cout: Lin::load(f, gpu, &format!("{c}.to_out"), Keep::Stored)?,
                proj: if f.has(&format!("{b}.cross_attn.proj_linear.weight")) {
                    Some(Lin::load(f, gpu, &format!("{b}.cross_attn.proj_linear"), Keep::Stored)?)
                } else {
                    None
                },
                mlp0: Lin::load(f, gpu, &format!("{b}.mlp.mlp.0"), Keep::Stored)?,
                mlp2: Lin::load(f, gpu, &format!("{b}.mlp.mlp.2"), Keep::Stored)?,
            };
            if blk.modulation.len() != 6 * C || blk.qkv.n != 3 * C || blk.mlp0.n != MLP {
                return Err(Error(format!("{}: block {i} is not TRELLIS's 1536-wide block", f.path())));
            }
            bytes += [&blk.qkv, &blk.out, &blk.cq, &blk.ckv, &blk.cout, &blk.mlp0, &blk.mlp2].iter().map(|l| l.bytes()).sum::<usize>()
                + blk.proj.as_ref().map_or(0, |p| p.bytes());
            blocks.push(blk);
        }
        let input = Lin::load(f, gpu, "input_layer", Keep::Stored)?;
        let out = Lin::load(f, gpu, "out_layer", Keep::Stored)?;
        let cond_ch = blocks[0].ckv.k;
        let proj_ch = blocks[0].proj.as_ref().map(|p| p.k);
        let scratch = DevBuf::new(gpu, MLP * C * 2 + proj_ch.unwrap_or(0) * C * 2)?;
        Ok(Dit {
            in_ch: input.k,
            out_ch: out.n,
            cond_ch,
            proj_ch,
            t0: HostLin::load(f, "t_embedder.mlp.0")?,
            t2: HostLin::load(f, "t_embedder.mlp.2")?,
            ada: HostLin::load(f, "adaLN_modulation.1")?,
            input,
            blocks,
            out,
            scratch,
            bytes,
        })
    }

    /// Every block's modulation for timestep t (1000 t'): the sinusoidal embedding (cos | sin), the timestep MLP, the
    /// shared adaLN, plus each block's own [blocks, 9216]
    pub fn modulation(&self, t: f32) -> Vec<f32> {
        let half = 128;
        let ln = (10000f64).ln();
        let freqs: Vec<f32> = (0..half).map(|i| (-(ln as f32) * i as f32 / half as f32).exp()).collect();
        let args: Vec<f32> = freqs.iter().map(|f| t * f).collect();
        let emb: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        let h = self.t2.apply(&silu(&self.t0.apply(&emb)));
        let m = self.ada.apply(&silu(&h));
        self.blocks.iter().flat_map(|b| b.modulation.iter().zip(&m).map(|(a, b)| a + b).collect::<Vec<_>>()).collect()
    }

    pub fn work(&self, gpu: &Arc<Gpu>, n: usize, l: usize) -> Result<Work> {
        Ok(Work {
            n,
            h: DevBuf::f32(gpu, n * C)?,
            hn: DevBuf::new(gpu, n * C * 2)?,
            qkv: DevBuf::new(gpu, n * 3 * C * 2)?,
            att: DevBuf::new(gpu, n * C * 2)?,
            o: DevBuf::f32(gpu, n * C)?,
            o2: DevBuf::f32(gpu, if self.proj_ch.is_some() { n * C } else { 1 })?,
            mlp: DevBuf::new(gpu, n * MLP * 2)?,
            kv: DevBuf::new(gpu, l * 2 * C * 2)?,
            x: DevBuf::new(gpu, n * self.in_ch.max(1) * 2)?,
            out: DevBuf::f32(gpu, n * self.out_ch)?,
        })
    }

    /// The velocity [n, out] for x [n, in] at timestep t (1000 t'), cross-attending to `cond` (half [l, cond_ch]),
    /// rotary phases `cs` ([n, 64, 2], `rope`), Pixal3D's projected features `proj` (half [n, proj_ch])
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, cx: &Ctx, w: &Work, x: &[f32], t: f32, cond: &DevBuf, l: usize, cs: &DevBuf, proj: Option<&DevBuf>,
                   mut probe: Option<&mut Probe>) -> Result<Vec<f32>> {
        let (nsd, k, g) = (&cx.nsd, &cx.k, &cx.gpu);
        let n = w.n;
        if x.len() != n * self.in_ch {
            return Err(Error(format!("the flow's input: {} values for {n} tokens of {}", x.len(), self.in_ch)));
        }
        if proj.is_some() != self.proj_ch.is_some() {
            return Err(Error("projected features given to a model without proj_linear (or missing for one with it)".into()));
        }
        let mods = DevBuf::from_f32(g, &self.modulation(t))?;
        w.x.write(0, &crate::w::half_bytes(x))?;
        self.input.apply(nsd, k, &self.scratch, w.x.ptr(), n, w.h.ptr(), Dt::F32)?;
        if let Some(p) = probe.as_deref_mut() {
            p.input = w.h.to_f32()?;
        }
        let s = &self.scratch;
        let nb = self.blocks.len();
        let hp = |b: &DevBuf, off: usize| -> *mut std::ffi::c_void {
            // SAFETY: offsets inside the half buffers sized above.
            unsafe { b.ptr().cast::<u16>().add(off).cast() }
        };
        for (bi, b) in self.blocks.iter().enumerate() {
            // SAFETY: rows of the [blocks, 6, 1536] modulation table.
            let md = |j: usize| unsafe { mods.fp().add(bi * 6 * C + j * C) as *const f32 };
            k.ln_mod(w.h.ptr(), Dt::F32, n, C, 1e-6, std::ptr::null(), std::ptr::null(), md(1), md(0), w.hn.ptr(), Dt::F16)?;
            b.qkv.apply(nsd, k, s, w.hn.ptr(), n, w.qkv.ptr(), Dt::F16)?;
            k.qk_norm_rope(w.qkv.ptr(), n, HEADS, HD, 3 * C, b.q_gamma.fp(), cs.fp())?;
            k.qk_norm_rope(hp(&w.qkv, C), n, HEADS, HD, 3 * C, b.k_gamma.fp(), cs.fp())?;
            nsd.attention_qk(w.qkv.ptr(), n, 3 * C, hp(&w.qkv, C), hp(&w.qkv, 2 * C), n, 3 * C, Dt::F16, HEADS, HD, w.att.ptr(), Dt::F16)?;
            b.out.apply(nsd, k, s, w.att.ptr(), n, w.o.ptr(), Dt::F32)?;
            k.gate_add(w.h.fp(), n, C, w.o.ptr(), Dt::F32, md(2))?;
            // cross-attention to the picture
            k.ln_mod(w.h.ptr(), Dt::F32, n, C, 1e-6, b.norm2.0.fp(), b.norm2.1.fp(), std::ptr::null(), std::ptr::null(), w.hn.ptr(), Dt::F16)?;
            b.cq.apply(nsd, k, s, w.hn.ptr(), n, w.att.ptr(), Dt::F16)?;
            k.qk_norm_rope(w.att.ptr(), n, HEADS, HD, C, b.cq_gamma.fp(), std::ptr::null())?;
            b.ckv.apply(nsd, k, s, cond.ptr(), l, w.kv.ptr(), Dt::F16)?;
            k.qk_norm_rope(w.kv.ptr(), l, HEADS, HD, 2 * C, b.ck_gamma.fp(), std::ptr::null())?;
            nsd.attention_qk(w.att.ptr(), n, C, w.kv.ptr(), hp(&w.kv, C), l, 2 * C, Dt::F16, HEADS, HD, w.hn.ptr(), Dt::F16)?;
            b.cout.apply(nsd, k, s, w.hn.ptr(), n, w.o.ptr(), Dt::F32)?;
            if let (Some(pl), Some(pf)) = (&b.proj, proj) {
                pl.apply(nsd, k, s, pf.ptr(), n, w.o2.ptr(), Dt::F32)?;
                k.add(w.o.ptr(), Dt::F32, w.o2.ptr(), Dt::F32, n * C)?;
            }
            k.gate_add(w.h.fp(), n, C, w.o.ptr(), Dt::F32, std::ptr::null())?;
            // the MLP
            k.ln_mod(w.h.ptr(), Dt::F32, n, C, 1e-6, std::ptr::null(), std::ptr::null(), md(4), md(3), w.hn.ptr(), Dt::F16)?;
            b.mlp0.apply(nsd, k, s, w.hn.ptr(), n, w.mlp.ptr(), Dt::F16)?;
            nsd.gelu(w.mlp.ptr(), Dt::F16, n * MLP, true)?;
            b.mlp2.apply(nsd, k, s, w.mlp.ptr(), n, w.o.ptr(), Dt::F32)?;
            k.gate_add(w.h.fp(), n, C, w.o.ptr(), Dt::F32, md(5))?;
            if let Some(p) = probe.as_deref_mut() {
                if bi == 0 {
                    p.block0 = w.h.to_f32()?;
                } else if bi == nb - 1 {
                    p.last = w.h.to_f32()?;
                }
            }
        }
        k.ln_mod(w.h.ptr(), Dt::F32, n, C, 1e-5, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), w.hn.ptr(), Dt::F16)?;
        self.out.apply(nsd, k, s, w.hn.ptr(), n, w.out.ptr(), Dt::F32)?;
        w.out.to_f32()
    }

    /// The picture's tokens as the cross-attention reads them: half [l, cond_ch]
    pub fn cond(&self, gpu: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
        if !v.len().is_multiple_of(self.cond_ch) {
            return Err(Error(format!("a condition of {} values for {}-wide tokens", v.len(), self.cond_ch)));
        }
        dev_half(gpu, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_phases_are_per_axis_then_identity() {
        let cs = rope(&[[0, 0, 0], [1, 2, 3]]);
        assert_eq!(cs.len(), 2 * 128);
        // the origin: no rotation anywhere
        assert!(cs[..128].chunks(2).all(|p| p == [1.0, 0.0]));
        // token 1: pair 0 is x's first frequency (1): angle 1; pair 21 y's (angle 2); pair 42 z's (3); pair 63 identity
        let r = &cs[128..];
        assert!((r[0] - 1f32.cos()).abs() < 1e-6 && (r[1] - 1f32.sin()).abs() < 1e-6);
        assert!((r[42] - 2f32.cos()).abs() < 1e-6);
        assert!((r[84] - 3f32.cos()).abs() < 1e-6);
        assert_eq!(&r[126..], &[1.0, 0.0]);
    }
}
