//! The ShapeVAE's decoder (3DShape2VecSet style): the latents lifted to 1,024 features and through 16 self-attention
//! blocks, then the geometry decoder - any point in the box asks the tokens, by one cross-attention block, whether it
//! is inside (a logit; the surface at 0).
//!
//! ```text
//!   tokens = transformer(post_kl(latents / scale_factor))        ln -> attention (q, k: layer norm per head) -> mlp
//!   data   = ln2(latents_proj(tokens))                            (the turbo decoder: 1,024 -> 512)
//!   x      = query_proj(fourier(point))                           (x, sin, cos of x * 2^0..7: 51 features)
//!   x     += c_proj(attention(c_q(ln1(x)), c_kv(data)))
//!   x     += mlp(ln3(x));   logit = output_proj(ln_post(x))       (the turbo decoder: no ln_post, no q / k norms)
//! ```
//!
//! Tencent's attention takes q, k, v (and the cross-attention's k, v) interleaved per head out of one product; the
//! rows of those matrices are reordered at load so each comes out whole ([q | k | v]), the attention reading them
//! through strides.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt};

use crate::ops::{hp, read_f32, HostMat, Lin, Ops, Weights, P};

const EPS: f32 = 1e-6;
const HEAD: usize = 64;
pub const FREQS: usize = 8;
pub const QUERY: usize = 3 * (2 * FREQS + 1);
/// The query embedding as the product reads it: zero features after the 51 (oneDNN takes 51 columns slowly - a
/// quarter of a second to build each product's kernel - and 64 at once)
const QUERY_PAD: usize = 64;

struct Res {
    ln1w: DevBuf,
    ln1b: DevBuf,
    qkv: Lin,
    qn: Option<(DevBuf, DevBuf)>,
    kn: Option<(DevBuf, DevBuf)>,
    proj: Lin,
    ln2w: DevBuf,
    ln2b: DevBuf,
    fc: Lin,
    fc2: Lin,
}

struct Geo {
    width: usize,
    query: Lin,
    latents: Option<Lin>,
    ln1: (DevBuf, DevBuf),
    ln2: (DevBuf, DevBuf),
    ln3: (DevBuf, DevBuf),
    q: Lin,
    kv: Lin,
    qn: Option<(DevBuf, DevBuf)>,
    kn: Option<(DevBuf, DevBuf)>,
    proj: Lin,
    fc: Lin,
    fc2: Lin,
    post: Option<(DevBuf, DevBuf)>,
    out: Lin,
}

pub struct Vae {
    pub width: usize,
    pub channels: usize,
    pub scale_factor: f32,
    post_kl: Lin,
    blocks: Vec<Res>,
    geo: Geo,
    pub bytes: usize,
}

/// The decoder's view of one set of latents: the tokens' keys and values, [n, 2 w] half (k | v)
pub struct Data {
    pub n: usize,
    kv: DevBuf,
}

/// Rows of an interleaved per-head product ([head][part][64]) regrouped by part: [part][head][64]
fn by_part(parts: usize, heads: usize) -> Vec<usize> {
    (0..parts).flat_map(|p| (0..heads).flat_map(move |h| (0..HEAD).map(move |j| h * parts * HEAD + p * HEAD + j))).collect()
}

fn norm(w: &Weights, p: &str) -> Result<(DevBuf, DevBuf)> {
    Ok((w.vec(&format!("{p}.weight"))?, w.vec(&format!("{p}.bias"))?))
}

fn norm_opt(w: &Weights, p: &str) -> Result<Option<(DevBuf, DevBuf)>> {
    if w.has(&format!("{p}.weight")) { norm(w, p).map(Some) } else { Ok(None) }
}

impl Vae {
    /// The decoder's tensors under `w`'s prefix (`vae.` in a whole model file, none in a decoder file)
    pub fn load(w: &Weights, scale_factor: f32) -> Result<Vae> {
        let width = w.shape("post_kl.weight")?[0];
        let channels = w.shape("post_kl.weight")?[1];
        let heads = width / HEAD;
        let mut blocks = Vec::new();
        while w.has(&format!("transformer.resblocks.{}.ln_1.weight", blocks.len())) {
            let p = format!("transformer.resblocks.{}.", blocks.len());
            let qkv = w.host_mat(&format!("{p}attn.c_qkv.weight"))?.rows(by_part(3, heads));
            let qkv_b = if w.has(&format!("{p}attn.c_qkv.bias")) {
                let b = w.host_f32(&format!("{p}attn.c_qkv.bias"))?;
                Some(by_part(3, heads).into_iter().map(|r| b[r]).collect())
            } else {
                None
            };
            blocks.push(Res {
                ln1w: w.vec(&format!("{p}ln_1.weight"))?,
                ln1b: w.vec(&format!("{p}ln_1.bias"))?,
                qkv: w.upload(&qkv, qkv_b)?,
                qn: norm_opt(w, &format!("{p}attn.attention.q_norm"))?,
                kn: norm_opt(w, &format!("{p}attn.attention.k_norm"))?,
                proj: w.lin(&format!("{p}attn.c_proj"))?,
                ln2w: w.vec(&format!("{p}ln_2.weight"))?,
                ln2b: w.vec(&format!("{p}ln_2.bias"))?,
                fc: w.lin(&format!("{p}mlp.c_fc"))?,
                fc2: w.lin(&format!("{p}mlp.c_proj"))?,
            });
        }
        if blocks.is_empty() {
            return Err(Error(format!("{}: no ShapeVAE decoder blocks under {:?}", w.st.path.display(), w.prefix)));
        }
        let g = "geo_decoder.";
        let a = format!("{g}cross_attn_decoder.");
        let gw = w.shape(&format!("{g}query_proj.weight"))?[0];
        let gh = gw / HEAD;
        let kv = w.host_mat(&format!("{a}attn.c_kv.weight"))?.rows(by_part(2, gh));
        let kv_b = if w.has(&format!("{a}attn.c_kv.bias")) {
            let b = w.host_f32(&format!("{a}attn.c_kv.bias"))?;
            Some(by_part(2, gh).into_iter().map(|r| b[r]).collect())
        } else {
            None
        };
        let qm = w.host_mat(&format!("{g}query_proj.weight"))?;
        if qm.k != QUERY {
            return Err(Error(format!("the geometry decoder takes {} query features, not {QUERY}", qm.k)));
        }
        let qm = HostMat { v: (0..qm.n).flat_map(|r| qm.v[r * QUERY..(r + 1) * QUERY].iter().copied().chain(std::iter::repeat_n(0, QUERY_PAD - QUERY))).collect(),
                           n: qm.n, k: QUERY_PAD };
        let geo = Geo {
            width: gw,
            query: w.upload(&qm, if w.has(&format!("{g}query_proj.bias")) { Some(w.host_f32(&format!("{g}query_proj.bias"))?) } else { None })?,
            latents: if w.has(&format!("{g}latents_proj.weight")) { Some(w.lin(&format!("{g}latents_proj"))?) } else { None },
            ln1: norm(w, &format!("{a}ln_1"))?,
            ln2: norm(w, &format!("{a}ln_2"))?,
            ln3: norm(w, &format!("{a}ln_3"))?,
            q: w.lin(&format!("{a}attn.c_q"))?,
            kv: w.upload(&kv, kv_b)?,
            qn: norm_opt(w, &format!("{a}attn.attention.q_norm"))?,
            kn: norm_opt(w, &format!("{a}attn.attention.k_norm"))?,
            proj: w.lin(&format!("{a}attn.c_proj"))?,
            fc: w.lin(&format!("{a}mlp.c_fc"))?,
            fc2: w.lin(&format!("{a}mlp.c_proj"))?,
            post: norm_opt(w, &format!("{g}ln_post"))?,
            out: w.lin(&format!("{g}output_proj"))?,
        };
        Ok(Vae { width, channels, scale_factor, post_kl: w.lin("post_kl")?, blocks, geo, bytes: w.bytes.get() })
    }

    /// The transformer's tokens (float32 [n, width]) of latents (float32 host [n, channels], as sampled)
    pub fn tokens(&self, ops: &Ops, latents: &[f32]) -> Result<DevBuf> {
        let gpu = &ops.gpu;
        let (wd, c) = (self.width, self.channels);
        let n = latents.len() / c;
        let heads = wd / HEAD;
        let lh: Vec<u16> = latents.iter().map(|&v| crate::ops::f32_to_f16(v / self.scale_factor)).collect();
        let lin = DevBuf::new(gpu, lh.len() * 2)?;
        // SAFETY: u16s viewed as bytes.
        lin.write(0, unsafe { std::slice::from_raw_parts(lh.as_ptr().cast::<u8>(), lh.len() * 2) })?;
        let x = DevBuf::new(gpu, n * wd * 4)?;
        ops.lin(&self.post_kl, lin.ptr(), n, x.ptr(), Dt::F32)?;
        let h = DevBuf::new(gpu, n * wd * 2)?;
        let qkv = DevBuf::new(gpu, n * 3 * wd * 2)?;
        let att = DevBuf::new(gpu, n * wd * 2)?;
        let ffn = self.blocks[0].fc.n;
        let mid = DevBuf::new(gpu, n * ffn * 2)?;
        for b in &self.blocks {
            ops.ln(x.ptr(), false, n, wd, EPS, b.ln1w.ptr(), b.ln1b.ptr(), none(), none(), h.ptr(), true)?;
            ops.lin(&b.qkv, h.ptr(), n, qkv.ptr(), Dt::F16)?;
            if let Some((nw, nb)) = &b.qn {
                ops.head_norm(qkv.ptr(), n, heads, HEAD, 3 * wd, nw.ptr(), nb.ptr(), EPS, false)?;
            }
            if let Some((nw, nb)) = &b.kn {
                ops.head_norm(hp(&qkv, wd), n, heads, HEAD, 3 * wd, nw.ptr(), nb.ptr(), EPS, false)?;
            }
            ops.attention(qkv.ptr(), hp(&qkv, wd), hp(&qkv, 2 * wd), n, heads, HEAD, 3 * wd, att.ptr())?;
            ops.lin(&b.proj, att.ptr(), n, h.ptr(), Dt::F16)?;
            ops.add(x.ptr(), h.ptr(), true, n, wd, none())?;
            ops.ln(x.ptr(), false, n, wd, EPS, b.ln2w.ptr(), b.ln2b.ptr(), none(), none(), h.ptr(), true)?;
            ops.lin(&b.fc, h.ptr(), n, mid.ptr(), Dt::F16)?;
            ops.gelu(mid.ptr(), n * ffn, false)?;
            ops.lin(&b.fc2, mid.ptr(), n, h.ptr(), Dt::F16)?;
            ops.add(x.ptr(), h.ptr(), true, n, wd, none())?;
        }
        // the temporaries are freed on return: their kernels first
        ops.gpu.sync()?;
        Ok(x)
    }

    /// The geometry decoder's keys and values of the tokens (float32 [n, width])
    pub fn data(&self, ops: &Ops, tokens: &DevBuf, n: usize) -> Result<Data> {
        let gpu = &ops.gpu;
        let g = &self.geo;
        let gw = g.width;
        let th = DevBuf::new(gpu, n * self.width * 2)?;
        let d = match &g.latents {
            Some(l) => {
                ops.to_half(tokens.ptr(), th.ptr(), n * self.width)?;
                let d = DevBuf::new(gpu, n * gw * 4)?;
                ops.lin(l, th.ptr(), n, d.ptr(), Dt::F32)?;
                d
            }
            None => tokens.view(0, n * gw * 4)?,
        };
        let h = DevBuf::new(gpu, n * gw * 2)?;
        ops.ln(d.ptr(), false, n, gw, EPS, g.ln2.0.ptr(), g.ln2.1.ptr(), none(), none(), h.ptr(), true)?;
        let kv = DevBuf::new(gpu, n * 2 * gw * 2)?;
        ops.lin(&g.kv, h.ptr(), n, kv.ptr(), Dt::F16)?;
        if let Some((nw, nb)) = &g.kn {
            ops.head_norm(kv.ptr(), n, gw / HEAD, HEAD, 2 * gw, nw.ptr(), nb.ptr(), EPS, false)?;
        }
        ops.gpu.sync()?;
        Ok(Data { n, kv })
    }

    /// Points a chunk holds at most (the working set: about 2 + 2 r halves and 1 float a feature a point)
    pub fn chunk(&self) -> usize {
        let per = self.geo.width * (4 + 2 * 2 + 2 * self.geo.fc.n / self.geo.width.max(1));
        ((1usize << 30) / per.max(1)).clamp(4096, 1 << 18) / 1024 * 1024
    }

    /// Working buffers for chunks of `q` points
    pub fn query_scratch(&self, ops: &Ops, q: usize) -> Result<QueryScratch> {
        let gpu = &ops.gpu;
        let gw = self.geo.width;
        Ok(QueryScratch {
            q,
            emb: DevBuf::new(gpu, q * QUERY * 2)?,
            emb_pad: DevBuf::new(gpu, q * QUERY_PAD * 2)?,
            zeros: {
                let z = DevBuf::new(gpu, q * (QUERY_PAD - QUERY) * 2)?;
                z.fill(0)?;
                z
            },
            x: DevBuf::new(gpu, q * gw * 4)?,
            h: DevBuf::new(gpu, q * gw * 2)?,
            qq: DevBuf::new(gpu, q * gw * 2)?,
            att: DevBuf::new(gpu, q * gw * 2)?,
            mid: DevBuf::new(gpu, q * self.geo.fc.n * 2)?,
            logits: DevBuf::new(gpu, q * 4)?,
            pts: DevBuf::new(gpu, q * 3 * 4)?,
            idx: DevBuf::new(gpu, q * 4)?,
        })
    }

    /// Logits of `n` <= chunk points, their embedding already in `sc.emb`; into `sc.logits`
    fn run(&self, ops: &Ops, data: &Data, sc: &QueryScratch, n: usize) -> Result<()> {
        let g = &self.geo;
        let gw = g.width;
        let heads = gw / HEAD;
        ops.concat(sc.emb.ptr(), QUERY, sc.zeros.ptr(), QUERY_PAD - QUERY, n, sc.emb_pad.ptr())?;
        ops.lin(&g.query, sc.emb_pad.ptr(), n, sc.x.ptr(), Dt::F32)?;
        ops.ln(sc.x.ptr(), false, n, gw, EPS, g.ln1.0.ptr(), g.ln1.1.ptr(), none(), none(), sc.h.ptr(), true)?;
        ops.lin(&g.q, sc.h.ptr(), n, sc.qq.ptr(), Dt::F16)?;
        if let Some((nw, nb)) = &g.qn {
            ops.head_norm(sc.qq.ptr(), n, heads, HEAD, gw, nw.ptr(), nb.ptr(), EPS, false)?;
        }
        crate::prof::part(&ops.nsd, "geo query")?;
        ops.cross_attention(sc.qq.ptr(), n, gw, data.kv.ptr(), hp(&data.kv, gw), data.n, 2 * gw, heads, HEAD, sc.att.ptr())?;
        crate::prof::part(&ops.nsd, "geo attention")?;
        ops.lin(&g.proj, sc.att.ptr(), n, sc.h.ptr(), Dt::F16)?;
        ops.add(sc.x.ptr(), sc.h.ptr(), true, n, gw, none())?;
        ops.ln(sc.x.ptr(), false, n, gw, EPS, g.ln3.0.ptr(), g.ln3.1.ptr(), none(), none(), sc.h.ptr(), true)?;
        ops.lin(&g.fc, sc.h.ptr(), n, sc.mid.ptr(), Dt::F16)?;
        ops.gelu(sc.mid.ptr(), n * g.fc.n, false)?;
        ops.lin(&g.fc2, sc.mid.ptr(), n, sc.h.ptr(), Dt::F16)?;
        ops.add(sc.x.ptr(), sc.h.ptr(), true, n, gw, none())?;
        match &g.post {
            // nn.LayerNorm's default eps
            Some((w, b)) => ops.ln(sc.x.ptr(), false, n, gw, 1e-5, w.ptr(), b.ptr(), none(), none(), sc.h.ptr(), true)?,
            None => ops.to_half(sc.x.ptr(), sc.h.ptr(), n * gw)?,
        }
        ops.lin(&g.out, sc.h.ptr(), n, sc.logits.ptr(), Dt::F32)?;
        crate::prof::part(&ops.nsd, "geo mlp")
    }

    /// Logits at points (host float32 [n, 3])
    pub fn query_points(&self, ops: &Ops, data: &Data, sc: &QueryScratch, pts: &[f32]) -> Result<Vec<f32>> {
        let n = pts.len() / 3;
        let mut out = Vec::with_capacity(n);
        for c0 in (0..n).step_by(sc.q) {
            let m = (n - c0).min(sc.q);
            crate::ops::write_f32(&sc.pts, 0, &pts[c0 * 3..(c0 + m) * 3])?;
            ops.fourier(sc.pts.ptr(), none(), 0, m, 0, 0.0, 0.0, FREQS, false, sc.emb.ptr())?;
            self.run(ops, data, sc, m)?;
            out.extend(read_f32(&sc.logits, m)?);
        }
        Ok(out)
    }

    /// Logits at lattice points of an r1^3 lattice at lo + i * step: the ones `idx` names (host), or all of them in
    /// order (`idx` None: x slowest, z fastest); `progress(done, total)` after each chunk
    #[allow(clippy::too_many_arguments)]
    pub fn query_lattice(&self, ops: &Ops, data: &Data, sc: &QueryScratch, r1: usize, lo: f32, step: f32, idx: Option<&[i32]>,
                         progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let n = idx.map_or(r1 * r1 * r1, |i| i.len());
        let mut out = Vec::with_capacity(n);
        for c0 in (0..n).step_by(sc.q) {
            let m = (n - c0).min(sc.q);
            let ip: P = match idx {
                Some(i) => {
                    // SAFETY: i32s viewed as bytes.
                    sc.idx.write(0, unsafe { std::slice::from_raw_parts(i[c0..c0 + m].as_ptr().cast::<u8>(), m * 4) })?;
                    sc.idx.ptr()
                }
                None => none(),
            };
            ops.fourier(none(), ip, c0, m, r1, lo, step, FREQS, false, sc.emb.ptr())?;
            self.run(ops, data, sc, m)?;
            out.extend(read_f32(&sc.logits, m)?);
            progress(c0 + m, n)?;
        }
        Ok(out)
    }
}

pub struct QueryScratch {
    pub q: usize,
    emb: DevBuf,
    /// [q, 64]: emb and zeros
    emb_pad: DevBuf,
    zeros: DevBuf,
    x: DevBuf,
    h: DevBuf,
    qq: DevBuf,
    att: DevBuf,
    mid: DevBuf,
    logits: DevBuf,
    pts: DevBuf,
    idx: DevBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_heads_regroup_by_part() {
        // two heads of q | k | v: rows [h0 q, h0 k, h0 v, h1 q, h1 k, h1 v] (64 each) -> [h0 q, h1 q, h0 k, ...]
        let r = by_part(3, 2);
        assert_eq!(r.len(), 6 * HEAD);
        assert_eq!(r[0], 0);
        assert_eq!(r[HEAD], 3 * HEAD);
        assert_eq!(r[2 * HEAD], HEAD);
        assert_eq!(r[5 * HEAD + 7], 5 * HEAD + 7);
    }
}
