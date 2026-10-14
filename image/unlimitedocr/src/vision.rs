//! The DeepEncoder (deepencoder.py): SAM ViT-B, then CLIP-L reading SAM's features, both into the decoder's width
//! through a linear projector - from the projector GGUF (`v.sam.*`, `v.*`, `mm.*`, F16 weights), float32 on the GPU.
//!
//! ```text
//!   view [3][S][S] (S = 1024: a 64 x 64 grid of 16-pixel patches; 640: 40 x 40)
//!   SAM: patch linear 768 + the 64 x 64 position grid (bicubic, antialiased, for 40) ->
//!     12 blocks: x += out(attn(qkv(ln(x)))) - over 14 x 14 windows, but blocks 2, 5, 8, 11 over the whole grid, with
//!       decomposed relative positions (the tables linearly resampled to the grid);  x += lin2(gelu(lin1(ln(x))))
//!     neck: 1x1 conv 256, LayerNorm2d, 3x3 conv 256, LayerNorm2d; 3x3 stride 2 -> 512; 3x3 stride 2 -> 1024
//!     -> [S / 64]^2 tokens of 1024 (16 x 16, or 10 x 10)
//!   CLIP-L: [CLS] + SAM's tokens + the 16 x 16 + 1 positions (resampled for 10) -> pre_ln ->
//!     24 blocks: x += out(attn(qkv(ln1(x))));  x += down(quick_gelu(up(ln2(x))))   (16 heads of 64)
//!   projector: [CLIP's tokens without CLS | SAM's tokens] (2048) -> 1280
//! ```

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;

const SAM_EPS: f32 = 1e-6;
/// deepencoder.py's vit_model_cfg: layernorm_epsilon, pre_layernorm_epsilon
const CLIP_EPS: f32 = 1e-5;
const SAM_W: usize = 768;
const SAM_HEADS: usize = 12;
const WINDOW: usize = 14;
const CLIP_W: usize = 1024;
const CLIP_HEADS: usize = 16;
pub const OUT: usize = 1280;
/// SAM's patch and the native position grid (1024 / 16)
const PATCH: usize = 16;
const NATIVE: usize = 64;

/// A tensor as float32 on the host (F32 or F16)
pub fn f32s(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| Error(format!("the projector has no {name}")))?;
    let raw = g.read(t).map_err(|e| Error(e.0))?;
    match t.ty {
        GType::F32 => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        GType::F16 => Ok(raw.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
        other => Err(Error(format!("{name}: {} (F32 or F16 here)", other.name()))),
    }
}

pub fn half_to_f32(b: u16) -> f32 {
    let (s, e, m) = ((b >> 15) as u32, ((b >> 10) & 31) as u32, (b & 1023) as u32);
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => {
            // subnormal: normalize
            let mut m = m;
            let mut e = 113u32;
            while m & 1024 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 1023) << 13)
        }
        31 => (s << 31) | (255 << 23) | (m << 13),
        _ => (s << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// PyTorch's antialiased resampling weights for one axis (compute_indices_weights_aa, align_corners=False) with the
/// bicubic filter (a = -0.5, support 2): for each output its first input and normalized weights
fn aa_bicubic(inp: usize, out: usize) -> Vec<(usize, Vec<f64>)> {
    let cubic = |x: f64| {
        let (a, x) = (-0.5, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    let scale = inp as f64 / out as f64;
    let (support, inv) = if scale >= 1.0 { (2.0 * scale, 1.0 / scale) } else { (2.0, 1.0) };
    (0..out).map(|i| {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xsize = ((center + support + 0.5) as i64).min(inp as i64) as usize - xmin;
        let w: Vec<f64> = (0..xsize).map(|j| cubic((j as f64 + xmin as f64 - center + 0.5) * inv)).collect();
        let total: f64 = w.iter().sum();
        (xmin, w.iter().map(|v| if total != 0.0 { v / total } else { *v }).collect())
    }).collect()
}

/// A grid [s][s][c] resampled to [t][t][c] (F.interpolate bicubic, antialias=True, align_corners=False)
pub fn resample_grid(src: &[f32], s: usize, c: usize, t: usize) -> Vec<f32> {
    if s == t {
        return src.to_vec();
    }
    let k = aa_bicubic(s, t);
    // along x, then along y (separable, in float as PyTorch)
    let mut tmp = vec![0f64; s * t * c];
    for y in 0..s {
        for (x, (x0, w)) in k.iter().enumerate() {
            for ch in 0..c {
                tmp[(y * t + x) * c + ch] = w.iter().enumerate().map(|(i, wv)| wv * src[(y * s + x0 + i) * c + ch] as f64).sum();
            }
        }
    }
    let mut out = vec![0f32; t * t * c];
    for (y, (y0, w)) in k.iter().enumerate() {
        for x in 0..t {
            for ch in 0..c {
                out[(y * t + x) * c + ch] = w.iter().enumerate().map(|(i, wv)| wv * tmp[((y0 + i) * t + x) * c + ch]).sum::<f64>() as f32;
            }
        }
    }
    out
}

/// A relative-position table [l][c] resampled to [n][c] (F.interpolate linear, align_corners=False)
pub fn resample_rows(src: &[f32], l: usize, c: usize, n: usize) -> Vec<f32> {
    if l == n {
        return src.to_vec();
    }
    let scale = l as f64 / n as f64;
    let mut out = vec![0f32; n * c];
    for i in 0..n {
        let x = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let lo = (x.floor() as usize).min(l - 1);
        let hi = (lo + 1).min(l - 1);
        let f = x - lo as f64;
        for ch in 0..c {
            out[i * c + ch] = ((1.0 - f) * src[lo * c + ch] as f64 + f * src[hi * c + ch] as f64) as f32;
        }
    }
    out
}

struct SamBlock {
    global: bool,
    ln1w: DevBuf,
    ln1b: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    /// the relative-position tables as stored ([2 x side - 1][64]) on the host, and per grid on the GPU
    rel_h: Vec<f32>,
    rel_w: Vec<f32>,
    rel: std::sync::Mutex<std::collections::BTreeMap<usize, (DevBuf, DevBuf)>>,
    out: DevBuf,
    out_b: DevBuf,
    ln2w: DevBuf,
    ln2b: DevBuf,
    lin1: DevBuf,
    lin1_b: DevBuf,
    lin2: DevBuf,
    lin2_b: DevBuf,
}

struct ClipBlock {
    ln1w: DevBuf,
    ln1b: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    out: DevBuf,
    out_b: DevBuf,
    ln2w: DevBuf,
    ln2b: DevBuf,
    up: DevBuf,
    up_b: DevBuf,
    down: DevBuf,
    down_b: DevBuf,
}

pub struct Vision {
    gpu: Arc<Gpu>,
    patch: DevBuf,
    patch_b: DevBuf,
    /// SAM's position grid on the host [64][64][768], and per grid side on the GPU
    sam_pos: Vec<f32>,
    clip_pos: Vec<f32>,
    pos: std::sync::Mutex<std::collections::BTreeMap<usize, (DevBuf, DevBuf)>>,
    sam: Vec<SamBlock>,
    neck0: DevBuf,
    neck1w: DevBuf,
    neck1b: DevBuf,
    neck2: DevBuf,
    neck3w: DevBuf,
    neck3b: DevBuf,
    net2: DevBuf,
    net3: DevBuf,
    cls: Vec<f32>,
    pre_w: DevBuf,
    pre_b: DevBuf,
    clip: Vec<ClipBlock>,
    proj: DevBuf,
    proj_b: DevBuf,
    pub newline: Vec<f32>,
    pub separator: Vec<f32>,
}

/// Copies counted in floats (DevBuf::copy_within counts bytes)
pub trait CopyF {
    fn copy_within_f(&self, at: usize, src: &DevBuf, src_at: usize, n: usize) -> Result<()>;
}

impl CopyF for DevBuf {
    fn copy_within_f(&self, at: usize, src: &DevBuf, src_at: usize, n: usize) -> Result<()> {
        self.copy_within(at * 4, src, src_at * 4, n * 4)
    }
}

/// A buffer's float `i` as a pointer (a row of a batch)
fn at(b: &DevBuf, i: usize) -> *mut std::ffi::c_void {
    b.fp().wrapping_add(i).cast()
}

impl Vision {
    pub fn load(g: &Gguf, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vision> {
        let t0 = std::time::Instant::now();
        if g.meta("clip.projector_type").and_then(|v| v.as_str()) != Some("deepseekocr") {
            return Err(Error("the projector file is not DeepSeek-OCR's (clip.projector_type deepseekocr)".into()));
        }
        let up = |n: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &f32s(g, n)?) };
        let n_sam = (0..).take_while(|i| g.tensor(&format!("v.sam.blk.{i}.attn.qkv.weight")).is_some()).count();
        let mut sam = Vec::with_capacity(n_sam);
        for i in 0..n_sam {
            let p = |s: &str| format!("v.sam.blk.{i}.{s}");
            let rel_h = f32s(g, &p("attn.pos_h.weight"))?;
            let global = rel_h.len() / 64 != 2 * WINDOW - 1;
            sam.push(SamBlock {
                global,
                ln1w: up(&p("pre_ln.weight"))?, ln1b: up(&p("pre_ln.bias"))?,
                qkv: up(&p("attn.qkv.weight"))?, qkv_b: up(&p("attn.qkv.bias"))?,
                rel_w: f32s(g, &p("attn.pos_w.weight"))?, rel_h,
                rel: Default::default(),
                out: up(&p("attn.out.weight"))?, out_b: up(&p("attn.out.bias"))?,
                ln2w: up(&p("post_ln.weight"))?, ln2b: up(&p("post_ln.bias"))?,
                lin1: up(&p("mlp.lin1.weight"))?, lin1_b: up(&p("mlp.lin1.bias"))?,
                lin2: up(&p("mlp.lin2.weight"))?, lin2_b: up(&p("mlp.lin2.bias"))?,
            });
        }
        let n_clip = (0..).take_while(|i| g.tensor(&format!("v.blk.{i}.attn_qkv.weight")).is_some()).count();
        let mut clip = Vec::with_capacity(n_clip);
        for i in 0..n_clip {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            clip.push(ClipBlock {
                ln1w: up(&p("ln1.weight"))?, ln1b: up(&p("ln1.bias"))?,
                qkv: up(&p("attn_qkv.weight"))?, qkv_b: up(&p("attn_qkv.bias"))?,
                out: up(&p("attn_out.weight"))?, out_b: up(&p("attn_out.bias"))?,
                ln2w: up(&p("ln2.weight"))?, ln2b: up(&p("ln2.bias"))?,
                up: up(&p("ffn_up.weight"))?, up_b: up(&p("ffn_up.bias"))?,
                down: up(&p("ffn_down.weight"))?, down_b: up(&p("ffn_down.bias"))?,
            });
        }
        let v = Vision {
            gpu: gpu.clone(),
            patch: up("v.sam.patch_embd.weight")?,
            patch_b: up("v.sam.patch_embd.bias")?,
            sam_pos: f32s(g, "v.sam.pos_embd.weight")?,
            clip_pos: f32s(g, "v.position_embd.weight")?,
            pos: Default::default(),
            sam,
            neck0: up("v.sam.neck.0.weight")?,
            neck1w: up("v.sam.neck.1.weight")?,
            neck1b: up("v.sam.neck.1.bias")?,
            neck2: up("v.sam.neck.2.weight")?,
            neck3w: up("v.sam.neck.3.weight")?,
            neck3b: up("v.sam.neck.3.bias")?,
            net2: up("v.sam.net_2.weight")?,
            net3: up("v.sam.net_3.weight")?,
            cls: f32s(g, "v.class_embd")?,
            pre_w: up("v.pre_ln.weight")?,
            pre_b: up("v.pre_ln.bias")?,
            clip,
            proj: up("mm.model.fc.weight")?,
            proj_b: up("mm.model.fc.bias")?,
            newline: f32s(g, "v.image_newline")?,
            separator: f32s(g, "v.view_seperator")?,
        };
        if v.sam_pos.len() != NATIVE * NATIVE * SAM_W || v.clip_pos.len() != (NATIVE / 4 * NATIVE / 4 + 1) * CLIP_W {
            return Err(Error("the projector's position grids are not SAM-B's 64 x 64 and CLIP-L's 16 x 16 + 1".into()));
        }
        log(format!("unlimitedocr vision: SAM {n_sam} blocks, CLIP {n_clip} blocks, on {} in {:.1} s", gpu.name, t0.elapsed().as_secs_f64()));
        Ok(v)
    }

    /// The position tables for a view side: SAM's [G][G][768], CLIP's [1 + (G / 4)^2][1024]
    fn positions(&self, gside: usize) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
        let mut m = self.pos.lock().unwrap();
        if let std::collections::btree_map::Entry::Vacant(v) = m.entry(gside) {
            let sam = resample_grid(&self.sam_pos, NATIVE, SAM_W, gside);
            let (cs, ct) = (NATIVE / 4, gside / 4);
            // the CLS position first, then the grid's
            let mut clip = self.clip_pos[..CLIP_W].to_vec();
            clip.extend(resample_grid(&self.clip_pos[CLIP_W..], cs, CLIP_W, ct));
            v.insert((DevBuf::from_f32(&self.gpu, &sam)?, DevBuf::from_f32(&self.gpu, &clip)?));
        }
        let (a, b) = &m[&gside];
        Ok((a.ptr(), b.ptr()))
    }

    /// A block's relative-position tables for an attention grid side (windows: 14)
    fn rel(&self, b: &SamBlock, side: usize) -> Result<(*const f32, *const f32)> {
        let mut m = b.rel.lock().unwrap();
        if let std::collections::btree_map::Entry::Vacant(v) = m.entry(side) {
            let l = b.rel_h.len() / 64;
            let h = resample_rows(&b.rel_h, l, 64, 2 * side - 1);
            let w = resample_rows(&b.rel_w, l, 64, 2 * side - 1);
            v.insert((DevBuf::from_f32(&self.gpu, &h)?, DevBuf::from_f32(&self.gpu, &w)?));
        }
        let (h, w) = &m[&side];
        Ok((h.fp(), w.fp()))
    }

    /// Views of one side - `planes` [B][3][S][S] - to their tokens [B][(S / 64)^2][1280]; `tap` sees the stages
    /// (sam_layer_out-L [B][G][G][768], sam_output [B][g][g][1024], layer_out-L [B][1 + g^2][1024], dsocr_output)
    pub fn see(&self, nsd: &Nsd, planes: &[f32], b: usize, side: usize, tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<DevBuf> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let gs = side / PATCH;
        let n = gs * gs;
        if !side.is_multiple_of(64) || planes.len() != b * 3 * side * side {
            return Err(Error(format!("a view of {side}: sides are multiples of 64")));
        }
        let (sam_pos, clip_pos) = self.positions(gs)?;
        let img = DevBuf::from_f32(gpu, planes)?;
        let px = DevBuf::f32(gpu, b * n * SAM_W)?;
        // SAFETY: img [b][3][side][side], px [b][gs][gs][768].
        ffi::check(unsafe { (k.patchify)(gpu.raw(), img.fp(), b as i64, side as i64, side as i64, PATCH as i64, px.fp()) }, "patchify")?;
        // the queue first: sycl::free does not wait for work queued on the buffers
        nsd.wait()?;
        drop(img);
        let x = DevBuf::f32(gpu, b * n * SAM_W)?;
        nsd.linear(px.ptr(), Dt::F32, b * n, SAM_W, self.patch.ptr(), SAM_W, self.patch_b.ptr(), x.ptr(), Dt::F32)?;
        // the queue first: sycl::free does not wait for work queued on the buffers
        nsd.wait()?;
        drop(px);
        for i in 0..b {
            nsd.gate_add(at(&x, i * n * SAM_W), Dt::F32, n, SAM_W, sam_pos, Dt::F32, none(), none())?;
        }
        let h = DevBuf::f32(gpu, b * n * SAM_W)?;
        let nwin = gs.div_ceil(WINDOW) * gs.div_ceil(WINDOW);
        let wrows = b * nwin * WINDOW * WINDOW;
        let rows_max = wrows.max(b * n);
        let hw = DevBuf::f32(gpu, rows_max * SAM_W)?;
        let qkv = DevBuf::f32(gpu, rows_max * 3 * SAM_W)?;
        let att = DevBuf::f32(gpu, rows_max * SAM_W)?;
        let o = DevBuf::f32(gpu, rows_max * SAM_W)?;
        let mlp = DevBuf::f32(gpu, b * n * 4 * SAM_W)?;
        for (li, blk) in self.sam.iter().enumerate() {
            nsd.layer_norm(x.ptr(), Dt::F32, b * n, SAM_W, blk.ln1w.ptr(), blk.ln1b.ptr(), SAM_EPS, h.ptr(), Dt::F32)?;
            let (src, rows, gb, ag) = if blk.global {
                (&h, b * n, b, gs)
            } else {
                // SAFETY: h [b][gs][gs][768] -> hw, its windows [b][nw][14][14][768].
                ffi::check(unsafe { (k.window)(gpu.raw(), h.fp(), b as i64, gs as i64, gs as i64, SAM_W as i64, WINDOW as i64, 0, hw.fp()) }, "windows")?;
                (&hw, wrows, b * nwin, WINDOW)
            };
            nsd.linear(src.ptr(), Dt::F32, rows, SAM_W, blk.qkv.ptr(), 3 * SAM_W, blk.qkv_b.ptr(), qkv.ptr(), Dt::F32)?;
            let (rh, rw) = self.rel(blk, ag)?;
            // SAFETY: qkv [gb][ag^2][3][12][64], the tables [2 ag - 1][64], att [gb][ag^2][768].
            ffi::check(unsafe { (k.sam_attn)(gpu.raw(), qkv.fp(), gb as i64, ag as i64, ag as i64, SAM_HEADS as i64, 64, rh, rw, att.fp()) }, "sam attention")?;
            nsd.linear(att.ptr(), Dt::F32, rows, SAM_W, blk.out.ptr(), SAM_W, blk.out_b.ptr(), o.ptr(), Dt::F32)?;
            if blk.global {
                nsd.gate_add(x.ptr(), Dt::F32, b * n, SAM_W, o.ptr(), Dt::F32, none(), none())?;
            } else {
                // SAFETY: the windows back to h [b][gs][gs][768] (the padding dropped).
                ffi::check(unsafe { (k.window)(gpu.raw(), o.fp(), b as i64, gs as i64, gs as i64, SAM_W as i64, WINDOW as i64, 1, h.fp()) }, "windows back")?;
                nsd.gate_add(x.ptr(), Dt::F32, b * n, SAM_W, h.ptr(), Dt::F32, none(), none())?;
            }
            nsd.layer_norm(x.ptr(), Dt::F32, b * n, SAM_W, blk.ln2w.ptr(), blk.ln2b.ptr(), SAM_EPS, h.ptr(), Dt::F32)?;
            nsd.linear(h.ptr(), Dt::F32, b * n, SAM_W, blk.lin1.ptr(), 4 * SAM_W, blk.lin1_b.ptr(), mlp.ptr(), Dt::F32)?;
            nsd.gelu(mlp.ptr(), Dt::F32, b * n * 4 * SAM_W, false)?;
            nsd.linear(mlp.ptr(), Dt::F32, b * n, 4 * SAM_W, blk.lin2.ptr(), SAM_W, blk.lin2_b.ptr(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, b * n, SAM_W, o.ptr(), Dt::F32, none(), none())?;
            tap(&format!("sam_layer_out-{li}"), &x)?;
        }
        // the queue first: sycl::free does not wait for work queued on the buffers
        nsd.wait()?;
        drop((hw, qkv, att, mlp));
        // the neck
        let c1 = DevBuf::f32(gpu, b * n * 256)?;
        let c1n = DevBuf::f32(gpu, b * n * 256)?;
        nsd.linear(x.ptr(), Dt::F32, b * n, SAM_W, self.neck0.ptr(), 256, none(), c1.ptr(), Dt::F32)?;
        nsd.layer_norm(c1.ptr(), Dt::F32, b * n, 256, self.neck1w.ptr(), self.neck1b.ptr(), SAM_EPS, c1n.ptr(), Dt::F32)?;
        let col = DevBuf::f32(gpu, b * n * 256 * 9)?;
        let conv = |inp: &DevBuf, g: usize, c: usize, s: usize, w: &DevBuf, co: usize, out: &DevBuf| -> Result<()> {
            // SAFETY: inp [b][g][g][c] -> col [b][go][go][c * 9]; out [b][go][go][co].
            ffi::check(unsafe { (k.im2col3)(gpu.raw(), inp.fp(), b as i64, g as i64, g as i64, c as i64, s as i64, col.fp()) }, "im2col")?;
            let go = (g - 1) / s + 1;
            nsd.linear(col.ptr(), Dt::F32, b * go * go, c * 9, w.ptr(), co, none(), out.ptr(), Dt::F32)
        };
        conv(&c1n, gs, 256, 1, &self.neck2, 256, &c1)?;
        nsd.layer_norm(c1.ptr(), Dt::F32, b * n, 256, self.neck3w.ptr(), self.neck3b.ptr(), SAM_EPS, c1n.ptr(), Dt::F32)?;
        let (g2, g4) = (gs / 2, gs / 4);
        let x2 = DevBuf::f32(gpu, b * g2 * g2 * 512)?;
        conv(&c1n, gs, 256, 2, &self.net2, 512, &x2)?;
        let m = g4 * g4;
        let sam = DevBuf::f32(gpu, b * m * CLIP_W)?;
        conv(&x2, g2, 512, 2, &self.net3, CLIP_W, &sam)?;
        // the queue first: sycl::free does not wait for work queued on the buffers
        nsd.wait()?;
        drop((c1, c1n, col, x2, x, h, o));
        tap("sam_output", &sam)?;
        // CLIP over [CLS] + SAM's tokens
        let s = m + 1;
        let xc = DevBuf::f32(gpu, b * s * CLIP_W)?;
        let cls = DevBuf::from_f32(gpu, &self.cls)?;
        for i in 0..b {
            xc.copy_within_f(i * s * CLIP_W, &cls, 0, CLIP_W)?;
            xc.copy_within_f((i * s + 1) * CLIP_W, &sam, i * m * CLIP_W, m * CLIP_W)?;
            nsd.gate_add(at(&xc, i * s * CLIP_W), Dt::F32, s, CLIP_W, clip_pos, Dt::F32, none(), none())?;
        }
        let xn = DevBuf::f32(gpu, b * s * CLIP_W)?;
        nsd.layer_norm(xc.ptr(), Dt::F32, b * s, CLIP_W, self.pre_w.ptr(), self.pre_b.ptr(), CLIP_EPS, xn.ptr(), Dt::F32)?;
        // the pre-norm's output is the residual stream (VitModel.forward)
        // the queue first: sycl::free does not wait for work queued on the buffers
        nsd.wait()?;
        drop(xc);
        let xc = xn;
        tap("pre_ln", &xc)?;
        let h = DevBuf::f32(gpu, b * s * CLIP_W)?;
        let qkv = DevBuf::f32(gpu, b * s * 3 * CLIP_W)?;
        let att = DevBuf::f32(gpu, b * s * CLIP_W)?;
        let o = DevBuf::f32(gpu, b * s * CLIP_W)?;
        let mlp = DevBuf::f32(gpu, b * s * 4 * CLIP_W)?;
        for (li, blk) in self.clip.iter().enumerate() {
            nsd.layer_norm(xc.ptr(), Dt::F32, b * s, CLIP_W, blk.ln1w.ptr(), blk.ln1b.ptr(), CLIP_EPS, h.ptr(), Dt::F32)?;
            tap(&format!("layer_inp_normed-{li}"), &h)?;
            nsd.linear(h.ptr(), Dt::F32, b * s, CLIP_W, blk.qkv.ptr(), 3 * CLIP_W, blk.qkv_b.ptr(), qkv.ptr(), Dt::F32)?;
            for i in 0..b {
                let q = at(&qkv, i * s * 3 * CLIP_W);
                nsd.attention(q, q.wrapping_byte_add(CLIP_W * 4), q.wrapping_byte_add(2 * CLIP_W * 4), Dt::F32, s, CLIP_HEADS, 64, 3 * CLIP_W,
                              at(&att, i * s * CLIP_W), Dt::F32)?;
            }
            nsd.linear(att.ptr(), Dt::F32, b * s, CLIP_W, blk.out.ptr(), CLIP_W, blk.out_b.ptr(), o.ptr(), Dt::F32)?;
            tap(&format!("clip_attn_out-{li}"), &o)?;
            nsd.gate_add(xc.ptr(), Dt::F32, b * s, CLIP_W, o.ptr(), Dt::F32, none(), none())?;
            nsd.layer_norm(xc.ptr(), Dt::F32, b * s, CLIP_W, blk.ln2w.ptr(), blk.ln2b.ptr(), CLIP_EPS, h.ptr(), Dt::F32)?;
            nsd.linear(h.ptr(), Dt::F32, b * s, CLIP_W, blk.up.ptr(), 4 * CLIP_W, blk.up_b.ptr(), mlp.ptr(), Dt::F32)?;
            // SAFETY: b s 4096 floats.
            ffi::check(unsafe { (k.quick_gelu)(gpu.raw(), mlp.fp(), (b * s * 4 * CLIP_W) as i64) }, "quick gelu")?;
            nsd.linear(mlp.ptr(), Dt::F32, b * s, 4 * CLIP_W, blk.down.ptr(), CLIP_W, blk.down_b.ptr(), o.ptr(), Dt::F32)?;
            nsd.gate_add(xc.ptr(), Dt::F32, b * s, CLIP_W, o.ptr(), Dt::F32, none(), none())?;
            tap(&format!("layer_out-{li}"), &xc)?;
        }
        // the queue first: sycl::free does not wait for work queued on the buffers
        nsd.wait()?;
        drop((h, qkv, att, mlp));
        // the projector over [CLIP's tokens (no CLS) | SAM's]
        let feat = DevBuf::f32(gpu, b * m * 2 * CLIP_W)?;
        for i in 0..b {
            // SAFETY: rows of CLIP's (after its CLS) and of SAM's side by side.
            unsafe {
                ffi::check((k.copy_rows)(gpu.raw(), xc.fp().add((i * s + 1) * CLIP_W), CLIP_W as i64, feat.fp().add(i * m * 2 * CLIP_W), 2 * CLIP_W as i64,
                                         m as i64, CLIP_W as i64), "concat")?;
                ffi::check((k.copy_rows)(gpu.raw(), sam.fp().add(i * m * CLIP_W), CLIP_W as i64, feat.fp().add(i * m * 2 * CLIP_W + CLIP_W),
                                         2 * CLIP_W as i64, m as i64, CLIP_W as i64), "concat")?;
            }
        }
        let out = DevBuf::f32(gpu, b * m * OUT)?;
        nsd.linear(feat.ptr(), Dt::F32, b * m, 2 * CLIP_W, self.proj.ptr(), OUT, self.proj_b.ptr(), out.ptr(), Dt::F32)?;
        // the queue first: sycl::free does not wait for work queued on the buffers
        nsd.wait()?;
        drop(o);
        tap("projected", &out)?;
        // the temporaries go when this returns: their work first
        nsd.wait()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn resampling_keeps_constants() {
        let g = vec![3.0f32; 16 * 16 * 2];
        assert!(resample_grid(&g, 16, 2, 10).iter().all(|v| (v - 3.0).abs() < 1e-6));
        let r: Vec<f32> = (0..127).map(|i| i as f32).collect();
        let s = resample_rows(&r, 127, 1, 79);
        assert!(s[0] >= 0.0 && s[78] <= 126.0 && s.windows(2).all(|w| w[1] > w[0]));
    }
}
