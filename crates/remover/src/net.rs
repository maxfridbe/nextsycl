//! BiRefNet (ZhengPeng7/BiRefNet: its config - Swin-L backbone, multi-scale input concatenated, three context
//! skips, ASPPDeformable decoder blocks with the input's tiles, gradient-attention gates) in float32 on the GPU.
//! Pictures are channels-last [H * W, C] buffers; every convolution is an im2col (a row block at a time) and a GEMM,
//! BatchNorm folded into its convolution at load.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi::Vision;

const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;
/// floats a row block of im2col may take (256 MiB)
const IM2COL_FLOATS: usize = 64 << 20;

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A picture on the GPU: [h * w, c] float32
pub struct Map {
    pub buf: DevBuf,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Map {
    pub fn new(gpu: &Arc<Gpu>, h: usize, w: usize, c: usize) -> Result<Map> {
        Ok(Map { buf: DevBuf::f32(gpu, h * w * c)?, h, w, c })
    }
    fn p(&self) -> *const f32 {
        self.buf.fp()
    }
    fn m(&self) -> *mut f32 {
        self.buf.fp()
    }
    fn px(&self) -> usize {
        self.h * self.w
    }
}

struct W<'a> {
    st: &'a SafeTensors,
    gpu: &'a Arc<Gpu>,
}

impl W<'_> {
    fn host(&self, name: &str) -> Result<Vec<f32>> {
        self.st.f32(name).map_err(ge)
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.st.need(name).map_err(ge)?.shape.iter().map(|d| *d as usize).collect())
    }
    fn dev(&self, v: &[f32]) -> Result<DevBuf> {
        DevBuf::from_f32(self.gpu, v)
    }
    fn vec(&self, name: &str) -> Result<DevBuf> {
        self.dev(&self.host(name)?)
    }
    /// A BatchNorm's (scale, shift) in evaluation
    fn bn(&self, p: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        let (g, b, m, v) = (self.host(&format!("{p}.weight"))?, self.host(&format!("{p}.bias"))?, self.host(&format!("{p}.running_mean"))?,
                            self.host(&format!("{p}.running_var"))?);
        let s: Vec<f32> = g.iter().zip(&v).map(|(g, v)| g / (v + BN_EPS).sqrt()).collect();
        let t = b.iter().zip(&m).zip(&s).map(|((b, m), s)| b - m * s).collect();
        Ok((s, t))
    }
}

/// A convolution as a GEMM: w [co, kh kw ci] ((ky, kx, c) order), bias; stride, padding, dilation
pub struct Conv {
    w: DevBuf,
    b: Option<DevBuf>,
    co: usize,
    ci: usize,
    kh: usize,
    kw: usize,
    s: usize,
    p: usize,
    d: usize,
}

/// [co, ci, kh, kw] -> [co, kh, kw, ci]
fn reorder(w: &[f32], co: usize, ci: usize, kh: usize, kw: usize) -> Vec<f32> {
    let mut o = vec![0f32; w.len()];
    for a in 0..co {
        for c in 0..ci {
            for y in 0..kh {
                for x in 0..kw {
                    o[((a * kh + y) * kw + x) * ci + c] = w[((a * ci + c) * kh + y) * kw + x];
                }
            }
        }
    }
    o
}

impl Conv {
    /// `p.weight` (and `p.bias` when the file has it), a BatchNorm `bn` folded in
    fn load(w: &W, p: &str, s: usize, pad: usize, d: usize, bn: Option<&str>) -> Result<Conv> {
        let sh = w.shape(&format!("{p}.weight"))?;
        if sh.len() != 4 {
            return Err(Error(format!("{p}.weight: {sh:?}, not a 2-D convolution")));
        }
        let (co, ci, kh, kw) = (sh[0], sh[1], sh[2], sh[3]);
        let mut wt = reorder(&w.host(&format!("{p}.weight"))?, co, ci, kh, kw);
        let mut b = if w.st.tensor(&format!("{p}.bias")).is_some() { Some(w.host(&format!("{p}.bias"))?) } else { None };
        if let Some(bn) = bn {
            let (sc, sh) = w.bn(bn)?;
            let k = kh * kw * ci;
            for (o, s) in sc.iter().enumerate() {
                wt[o * k..(o + 1) * k].iter_mut().for_each(|v| *v *= s);
            }
            let bb = b.take().unwrap_or_else(|| vec![0.0; co]);
            b = Some(bb.iter().zip(&sc).zip(&sh).map(|((b, s), t)| b * s + t).collect());
        }
        Ok(Conv { w: w.dev(&wt)?, b: b.map(|b| w.dev(&b)).transpose()?, co, ci, kh, kw, s, p: pad, d })
    }

    /// Two convolutions of the same input and shape side by side (their outputs concatenated)
    fn pair(w: &W, a: &str, b: &str, pad: usize) -> Result<Conv> {
        let (sa, sb) = (w.shape(&format!("{a}.weight"))?, w.shape(&format!("{b}.weight"))?);
        let (ci, kh, kw) = (sa[1], sa[2], sa[3]);
        let mut wt = reorder(&w.host(&format!("{a}.weight"))?, sa[0], ci, kh, kw);
        wt.extend(reorder(&w.host(&format!("{b}.weight"))?, sb[0], ci, kh, kw));
        let mut bias = w.host(&format!("{a}.bias"))?;
        bias.extend(w.host(&format!("{b}.bias"))?);
        Ok(Conv { w: w.dev(&wt)?, b: Some(w.dev(&bias)?), co: sa[0] + sb[0], ci, kh, kw, s: 1, p: pad, d: 1 })
    }

    fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        ((h + 2 * self.p - self.d * (self.kh - 1) - 1) / self.s + 1, (w + 2 * self.p - self.d * (self.kw - 1) - 1) / self.s + 1)
    }
}

/// A modulated deformable convolution (BiRefNet's DeformableConv2d) with its BatchNorm folded: the offset and mask
/// convolutions as one, then the sampled im2col and the regular convolution's GEMM
struct Deform {
    om: Conv,
    w: DevBuf,
    b: DevBuf,
    k: usize,
    co: usize,
}

impl Deform {
    fn load(w: &W, p: &str, bn: &str) -> Result<Deform> {
        let sh = w.shape(&format!("{p}.regular_conv.weight"))?;
        let (co, ci, k) = (sh[0], sh[1], sh[2]);
        let om = Conv::pair(w, &format!("{p}.offset_conv"), &format!("{p}.modulator_conv"), k / 2)?;
        let mut wt = reorder(&w.host(&format!("{p}.regular_conv.weight"))?, co, ci, k, k);
        let (sc, sf) = w.bn(bn)?;
        let kk = k * k * ci;
        for (o, s) in sc.iter().enumerate() {
            wt[o * kk..(o + 1) * kk].iter_mut().for_each(|v| *v *= s);
        }
        Ok(Deform { om, w: w.dev(&wt)?, b: w.dev(&sf)?, k, co })
    }
}

/// ASPPDeformable: 1x1 and deformable 1 / 3 / 7 branches, the global mean, 1x1 to its output
struct Aspp {
    branches: Vec<Deform>,
    pool: Conv,
    conv1: Conv,
}

/// BasicDecBlk: conv, BN, ReLU, ASPPDeformable, conv, BN
struct DecBlk {
    conv_in: Conv,
    att: Aspp,
    conv_out: Conv,
}

impl DecBlk {
    fn load(w: &W, p: &str) -> Result<DecBlk> {
        let a = format!("{p}.dec_att");
        let mut branches = vec![Deform::load(w, &format!("{a}.aspp1.atrous_conv"), &format!("{a}.aspp1.bn"))?];
        for i in 0..3 {
            branches.push(Deform::load(w, &format!("{a}.aspp_deforms.{i}.atrous_conv"), &format!("{a}.aspp_deforms.{i}.bn"))?);
        }
        Ok(DecBlk {
            conv_in: Conv::load(w, &format!("{p}.conv_in"), 1, 1, 1, Some(&format!("{p}.bn_in")))?,
            att: Aspp {
                branches,
                pool: Conv::load(w, &format!("{a}.global_avg_pool.1"), 1, 0, 1, Some(&format!("{a}.global_avg_pool.2")))?,
                conv1: Conv::load(w, &format!("{a}.conv1"), 1, 0, 1, Some(&format!("{a}.bn1")))?,
            },
            conv_out: Conv::load(w, &format!("{p}.conv_out"), 1, 1, 1, Some(&format!("{p}.bn_out")))?,
        })
    }
}

struct SwinBlock {
    n1w: DevBuf,
    n1b: DevBuf,
    qkv_w: DevBuf,
    qkv_b: DevBuf,
    proj_w: DevBuf,
    proj_b: DevBuf,
    n2w: DevBuf,
    n2b: DevBuf,
    fc1_w: DevBuf,
    fc1_b: DevBuf,
    fc2_w: DevBuf,
    fc2_b: DevBuf,
    /// the relative position bias expanded: [heads, N, N]
    bias: DevBuf,
}

struct SwinLayer {
    dim: usize,
    heads: usize,
    blocks: Vec<SwinBlock>,
    /// patch merging: its norm and reduction (none after the last layer)
    down: Option<(DevBuf, DevBuf, DevBuf)>,
    norm: (DevBuf, DevBuf),
}

const WS: usize = 12;

pub struct BiRefNet {
    pub gpu: Arc<Gpu>,
    nsd: Nsd,
    v: Vision,
    patch: Conv,
    patch_norm: (DevBuf, DevBuf),
    layers: Vec<SwinLayer>,
    squeeze: DecBlk,
    ipt: Vec<(Conv, Conv)>,
    dec: Vec<DecBlk>,
    lateral: Vec<Conv>,
    gdt: Vec<(Conv, Conv)>,
    out: Conv,
}

/// What the check compares besides the logits: the encoder's finest and coarsest features (after their
/// concatenations), channels-last
pub struct Probe {
    pub x1: Vec<f32>,
    pub x4: Vec<f32>,
}

impl BiRefNet {
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>) -> Result<BiRefNet> {
        let w = W { st, gpu };
        let ln = |p: &str| -> Result<(DevBuf, DevBuf)> { Ok((w.vec(&format!("{p}.weight"))?, w.vec(&format!("{p}.bias"))?)) };
        let depths = [2, 2, 18, 2];
        let heads = [6, 12, 24, 48];
        let n = WS * WS;
        let mut layers = Vec::new();
        for (li, (&depth, &hh)) in depths.iter().zip(&heads).enumerate() {
            let dim = 192 << li;
            let mut blocks = Vec::new();
            for bi in 0..depth {
                let p = format!("bb.layers.{li}.blocks.{bi}");
                let table = w.host(&format!("{p}.attn.relative_position_bias_table"))?;
                if table.len() != (2 * WS - 1) * (2 * WS - 1) * hh {
                    return Err(Error(format!("{p}: a relative position table of {} values", table.len())));
                }
                let mut bias = vec![0f32; hh * n * n];
                for i in 0..n {
                    for j in 0..n {
                        let (yi, xi, yj, xj) = (i / WS, i % WS, j / WS, j % WS);
                        let idx = (yi + WS - 1 - yj) * (2 * WS - 1) + (xi + WS - 1 - xj);
                        for h in 0..hh {
                            bias[(h * n + i) * n + j] = table[idx * hh + h];
                        }
                    }
                }
                let (n1w, n1b) = ln(&format!("{p}.norm1"))?;
                let (n2w, n2b) = ln(&format!("{p}.norm2"))?;
                blocks.push(SwinBlock {
                    n1w,
                    n1b,
                    qkv_w: w.vec(&format!("{p}.attn.qkv.weight"))?,
                    qkv_b: w.vec(&format!("{p}.attn.qkv.bias"))?,
                    proj_w: w.vec(&format!("{p}.attn.proj.weight"))?,
                    proj_b: w.vec(&format!("{p}.attn.proj.bias"))?,
                    n2w,
                    n2b,
                    fc1_w: w.vec(&format!("{p}.mlp.fc1.weight"))?,
                    fc1_b: w.vec(&format!("{p}.mlp.fc1.bias"))?,
                    fc2_w: w.vec(&format!("{p}.mlp.fc2.weight"))?,
                    fc2_b: w.vec(&format!("{p}.mlp.fc2.bias"))?,
                    bias: w.dev(&bias)?,
                });
            }
            let down = if li < 3 {
                let (a, b) = ln(&format!("bb.layers.{li}.downsample.norm"))?;
                Some((a, b, w.vec(&format!("bb.layers.{li}.downsample.reduction.weight"))?))
            } else {
                None
            };
            layers.push(SwinLayer { dim, heads: hh, blocks, down, norm: ln(&format!("bb.norm{li}"))? });
        }
        let d = "decoder";
        let mut ipt = Vec::new();
        for i in 1..=5 {
            ipt.push((Conv::load(&w, &format!("{d}.ipt_blk{i}.conv1"), 1, 1, 1, None)?, Conv::load(&w, &format!("{d}.ipt_blk{i}.conv_out"), 1, 1, 1, None)?));
        }
        let mut dec = Vec::new();
        for i in 1..=4 {
            dec.push(DecBlk::load(&w, &format!("{d}.decoder_block{i}"))?);
        }
        let mut lateral = Vec::new();
        for i in 2..=4 {
            lateral.push(Conv::load(&w, &format!("{d}.lateral_block{i}.conv"), 1, 0, 1, None)?);
        }
        let mut gdt = Vec::new();
        for i in 2..=4 {
            gdt.push((Conv::load(&w, &format!("{d}.gdt_convs_{i}.0"), 1, 1, 1, Some(&format!("{d}.gdt_convs_{i}.1")))?,
                      Conv::load(&w, &format!("{d}.gdt_convs_attn_{i}.0"), 1, 0, 1, None)?));
        }
        Ok(BiRefNet {
            gpu: gpu.clone(),
            nsd: Nsd::new(gpu)?,
            v: Vision::new(gpu)?,
            patch: Conv::load(&w, "bb.patch_embed.proj", 4, 0, 1, None)?,
            patch_norm: ln("bb.patch_embed.norm")?,
            layers,
            squeeze: DecBlk::load(&w, "squeeze_module.0")?,
            ipt,
            dec,
            lateral,
            gdt,
            out: Conv::load(&w, "decoder.conv_out1.0", 1, 0, 1, None)?,
        })
    }

    fn map(&self, h: usize, w: usize, c: usize) -> Result<Map> {
        Map::new(&self.gpu, h, w, c)
    }

    /// out [m, n] = x [m, k] W^T + b, float32
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, x: *const f32, m: usize, k: usize, w: &DevBuf, n: usize, b: Option<&DevBuf>, out: *mut f32) -> Result<()> {
        self.nsd.linear(x.cast(), Dt::F32, m, k, w.ptr(), n, b.map_or(none(), |b| b.ptr()), out.cast(), Dt::F32)
    }

    fn layer_norm(&self, x: *const f32, m: usize, c: usize, wb: (&DevBuf, &DevBuf), out: *mut f32) -> Result<()> {
        self.nsd.layer_norm(x.cast(), Dt::F32, m, c, wb.0.ptr(), wb.1.ptr(), LN_EPS, out.cast(), Dt::F32)
    }

    /// A convolution of x into a new map
    fn conv(&self, x: &Map, cv: &Conv) -> Result<Map> {
        if x.c != cv.ci {
            return Err(Error(format!("a convolution of {} channels given {}", cv.ci, x.c)));
        }
        let (ho, wo) = cv.out_hw(x.h, x.w);
        let out = self.map(ho, wo, cv.co)?;
        let m = ho * wo;
        if cv.kh == 1 && cv.kw == 1 && cv.s == 1 && cv.p == 0 {
            self.linear(x.p(), m, cv.ci, &cv.w, cv.co, cv.b.as_ref(), out.m())?;
            return Ok(out);
        }
        let k = cv.kh * cv.kw * cv.ci;
        let rows = (IM2COL_FLOATS / k).clamp(1, m);
        let col = DevBuf::f32(&self.gpu, rows * k)?;
        for r0 in (0..m).step_by(rows) {
            let n = rows.min(m - r0);
            self.v.im2col(x.p(), (x.h, x.w, x.c), (cv.kh, cv.kw), cv.s, cv.p, cv.d, (ho, wo), r0, n, col.fp())?;
            // SAFETY: rows r0 .. r0 + n of out's ho * wo rows of co floats.
            self.linear(col.fp(), n, k, &cv.w, cv.co, cv.b.as_ref(), unsafe { out.m().add(r0 * cv.co) })?;
        }
        Ok(out)
    }

    fn deform(&self, x: &Map, d: &Deform, out: *mut f32, stride: usize) -> Result<()> {
        let om = self.conv(x, &d.om)?;
        let m = x.px();
        let k = d.k * d.k * x.c;
        let rows = (IM2COL_FLOATS / k).clamp(1, m);
        let col = DevBuf::f32(&self.gpu, rows * k)?;
        let tmp = DevBuf::f32(&self.gpu, rows * d.co)?;
        for r0 in (0..m).step_by(rows) {
            let n = rows.min(m - r0);
            self.v.deform_im2col(x.p(), (x.h, x.w, x.c), d.k, d.k / 2, om.p(), r0, n, col.fp())?;
            self.linear(col.fp(), n, k, &d.w, d.co, Some(&d.b), tmp.fp())?;
            // SAFETY: rows r0 .. r0 + n of a buffer of m rows `stride` apart.
            self.v.copy_cols(tmp.fp(), n, d.co, d.co, unsafe { out.add(r0 * stride) }, stride)?;
        }
        Ok(())
    }

    fn dec_blk(&self, x: &Map, b: &DecBlk) -> Result<Map> {
        let h = self.conv(x, &b.conv_in)?;
        self.v.relu(h.m(), h.px() * h.c)?;
        // ASPPDeformable: the four deformable branches and the pooled one side by side, each ReLU'd
        let m = h.px();
        let bc = b.att.pool.co;
        let cat = self.map(h.h, h.w, bc * 5)?;
        for (i, d) in b.att.branches.iter().enumerate() {
            // SAFETY: column block i of cat's rows.
            self.deform(&h, d, unsafe { cat.m().add(i * bc) }, bc * 5)?;
        }
        let mean = DevBuf::f32(&self.gpu, h.c)?;
        self.v.col_mean(h.p(), m, h.c, mean.fp())?;
        let pooled = DevBuf::f32(&self.gpu, bc)?;
        self.linear(mean.fp(), 1, h.c, &b.att.pool.w, bc, b.att.pool.b.as_ref(), pooled.fp())?;
        // SAFETY: column block 4 of cat's rows.
        self.v.broadcast_row(pooled.fp(), m, bc, unsafe { cat.m().add(4 * bc) }, bc * 5)?;
        self.v.relu(cat.m(), m * bc * 5)?;
        let a = self.conv(&cat, &b.att.conv1)?;
        self.v.relu(a.m(), m * a.c)?;
        self.conv(&a, &b.conv_out)
    }

    /// The Swin-L backbone: the four stages' normed outputs
    fn swin(&self, x: &Map) -> Result<Vec<Map>> {
        let e = self.conv(x, &self.patch)?;
        let mut h = self.map(e.h, e.w, e.c)?;
        self.layer_norm(e.p(), e.px(), e.c, (&self.patch_norm.0, &self.patch_norm.1), h.m())?;
        drop(e);
        let mut outs = Vec::new();
        for l in &self.layers {
            let (hh, ww, c) = (h.h, h.w, l.dim);
            let m = hh * ww;
            let (hp, wp) = (hh.div_ceil(WS) * WS, ww.div_ceil(WS) * WS);
            let mw = hp * wp;
            let t = DevBuf::f32(&self.gpu, m * c)?;
            let win = DevBuf::f32(&self.gpu, mw * c)?;
            let qkv = DevBuf::f32(&self.gpu, mw * 3 * c)?;
            let att = DevBuf::f32(&self.gpu, mw * c)?;
            let mlp = DevBuf::f32(&self.gpu, m * 4 * c)?;
            for (bi, b) in l.blocks.iter().enumerate() {
                let shift = if bi % 2 == 1 { WS / 2 } else { 0 };
                self.layer_norm(h.p(), m, c, (&b.n1w, &b.n1b), t.fp())?;
                self.v.window_gather(t.fp(), hh, ww, c, WS, shift, win.fp())?;
                self.linear(win.fp(), mw, c, &b.qkv_w, 3 * c, Some(&b.qkv_b), qkv.fp())?;
                self.v.window_attention(qkv.fp(), hp / WS, wp / WS, WS, l.heads, b.bias.fp(), shift, (32f32).powf(-0.5), att.fp())?;
                self.linear(att.fp(), mw, c, &b.proj_w, c, Some(&b.proj_b), win.fp())?;
                self.v.window_scatter_add(win.fp(), hh, ww, c, WS, shift, h.m())?;
                self.layer_norm(h.p(), m, c, (&b.n2w, &b.n2b), t.fp())?;
                self.linear(t.fp(), m, c, &b.fc1_w, 4 * c, Some(&b.fc1_b), mlp.fp())?;
                self.nsd.gelu(mlp.ptr(), Dt::F32, m * 4 * c, false)?;
                self.linear(mlp.fp(), m, 4 * c, &b.fc2_w, c, Some(&b.fc2_b), t.fp())?;
                self.v.add(h.m(), t.fp(), m * c)?;
            }
            let o = self.map(hh, ww, c)?;
            self.layer_norm(h.p(), m, c, (&l.norm.0, &l.norm.1), o.m())?;
            outs.push(o);
            if let Some((nw, nb, red)) = &l.down {
                let (h2, w2) = (hh.div_ceil(2), ww.div_ceil(2));
                let g = DevBuf::f32(&self.gpu, h2 * w2 * 4 * c)?;
                let gn = DevBuf::f32(&self.gpu, h2 * w2 * 4 * c)?;
                self.v.patch_merge(h.p(), hh, ww, c, g.fp())?;
                self.layer_norm(g.fp(), h2 * w2, 4 * c, (nw, nb), gn.fp())?;
                let n = self.map(h2, w2, 2 * c)?;
                self.linear(gn.fp(), h2 * w2, 4 * c, red, 2 * c, None, n.m())?;
                h = n;
            }
        }
        Ok(outs)
    }

    /// Maps side by side: each resized (bilinear, align_corners) to (h, w) and concatenated in order
    fn cat_resized(&self, parts: &[&Map], h: usize, w: usize) -> Result<Map> {
        let c: usize = parts.iter().map(|p| p.c).sum();
        let out = self.map(h, w, c)?;
        let mut at = 0;
        for p in parts {
            // SAFETY: column block [at, at + p.c) of out's rows.
            let dst = unsafe { out.m().add(at) };
            if (p.h, p.w) == (h, w) {
                self.v.copy_cols(p.p(), h * w, p.c, p.c, dst, c)?;
            } else {
                self.v.resize_ac(p.p(), p.h, p.w, p.c, h, w, dst, c)?;
            }
            at += p.c;
        }
        Ok(out)
    }

    /// The input block i (ipt_blk): the picture as g x g tiles of channels, two convolutions
    fn ipt(&self, x: &Map, i: usize, g: usize) -> Result<Map> {
        let t = if g == 1 {
            None
        } else {
            let t = self.map(x.h / g, x.w / g, x.c * g * g)?;
            self.v.tiles(x.p(), x.h, x.w, x.c, g, t.m())?;
            Some(t)
        };
        let a = self.conv(t.as_ref().unwrap_or(x), &self.ipt[i].0)?;
        self.conv(&a, &self.ipt[i].1)
    }

    /// The gradient-attention gate: p *= sigmoid(attn(relu(bn(conv(p)))))
    fn gdt(&self, p: &Map, i: usize) -> Result<()> {
        let g = self.conv(p, &self.gdt[i].0)?;
        self.v.relu(g.m(), g.px() * g.c)?;
        let a = self.conv(&g, &self.gdt[i].1)?;
        self.v.mul_sigmoid(p.m(), p.px(), p.c, a.p())
    }

    /// The prediction's logits [1024 * 1024] for the normalized picture x [1024, 1024, 3], and the encoder's probes
    pub fn forward(&self, x: &Map, probe: bool) -> Result<(Vec<f32>, Option<Probe>)> {
        let f = self.swin(x)?;
        let half = self.map(x.h / 2, x.w / 2, 3)?;
        self.v.resize_ac(x.p(), x.h, x.w, 3, half.h, half.w, half.m(), 3)?;
        let g = self.swin(&half)?;
        drop(half);
        let mut xs = Vec::new();
        for (a, b) in f.iter().zip(&g) {
            xs.push(self.cat_resized(&[a, b], a.h, a.w)?);
        }
        drop((f, g));
        let x4 = self.cat_resized(&[&xs[0], &xs[1], &xs[2], &xs[3]], xs[3].h, xs[3].w)?;
        let pr = if probe { Some(Probe { x1: xs[0].buf.to_f32()?, x4: x4.buf.to_f32()? }) } else { None };
        let x4 = self.dec_blk(&x4, &self.squeeze)?;
        let (x1, x2, x3) = (&xs[0], &xs[1], &xs[2]);
        // the decoder: each level its input's tiles, a block, the gate, up to the next level plus its lateral
        let x4 = self.cat_resized(&[&x4, &self.ipt(x, 4, x.h / x4.h)?], x4.h, x4.w)?;
        let p4 = self.dec_blk(&x4, &self.dec[3])?;
        self.gdt(&p4, 2)?;
        let lat = self.conv(x3, &self.lateral[2])?;
        let p3in = self.map(x3.h, x3.w, p4.c)?;
        self.v.resize_ac(p4.p(), p4.h, p4.w, p4.c, x3.h, x3.w, p3in.m(), p4.c)?;
        self.v.add(p3in.m(), lat.p(), p3in.px() * p3in.c)?;
        let p3in = self.cat_resized(&[&p3in, &self.ipt(x, 3, x.h / x3.h)?], x3.h, x3.w)?;
        let p3 = self.dec_blk(&p3in, &self.dec[2])?;
        self.gdt(&p3, 1)?;
        let lat = self.conv(x2, &self.lateral[1])?;
        let p2in = self.map(x2.h, x2.w, p3.c)?;
        self.v.resize_ac(p3.p(), p3.h, p3.w, p3.c, x2.h, x2.w, p2in.m(), p3.c)?;
        self.v.add(p2in.m(), lat.p(), p2in.px() * p2in.c)?;
        let p2in = self.cat_resized(&[&p2in, &self.ipt(x, 2, x.h / x2.h)?], x2.h, x2.w)?;
        let p2 = self.dec_blk(&p2in, &self.dec[1])?;
        self.gdt(&p2, 0)?;
        let lat = self.conv(x1, &self.lateral[0])?;
        let p1in = self.map(x1.h, x1.w, p2.c)?;
        self.v.resize_ac(p2.p(), p2.h, p2.w, p2.c, x1.h, x1.w, p1in.m(), p2.c)?;
        self.v.add(p1in.m(), lat.p(), p1in.px() * p1in.c)?;
        let p1in = self.cat_resized(&[&p1in, &self.ipt(x, 1, x.h / x1.h)?], x1.h, x1.w)?;
        let p1 = self.dec_blk(&p1in, &self.dec[0])?;
        let top = self.cat_resized(&[&p1, &self.ipt(x, 0, 1)?], x.h, x.w)?;
        let out = self.conv(&top, &self.out)?;
        Ok((out.buf.to_f32()?, pr))
    }
}
