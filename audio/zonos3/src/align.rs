//! FuseAlign (Zyphra): word-level forced alignment - a recording and its words to each word's start and end on a 1 ms
//! grid and its presence - the timings speech editing reads. One forward a clip of at most 30 s and 1022 UTF-8 bytes of
//! words (the long-form routing over voice-activity chunks is not ported: speech editing aligns a clip whole, as
//! Zyphra's in-browser server does).
//!
//! ```text
//!   words -> UTF-8 bytes ([S1] prefix, a space between words) -> Dia's 12-layer text encoder (bf16 weights) -> [T, 1024]
//!   16 kHz audio, standardized -> patches of 512 / hop 256 -> [A, 1024]
//!   [audio + mod 0 | text proj + mod 1] -> 12 joint blocks (RMS norm, RoPE at audio frame / scaled byte positions,
//!     full attention, SwiGLU) -> final norm
//!   audio -> start / end refinement heads (1x1 conv, 4 x (transposed conv x2 + GELU), conv 5 + GELU, 1x1) -> [16 A, 256]
//!   a word's bytes, mean-pooled -> start / end / presence MLPs; the start (end) frame = argmax over the frames of
//!     start_mlp(word) . start_features[frame]; present when sigmoid(presence) >= 0.1
//! ```
//!
//! Float32 throughout (the reference's CPU recipe), the Dia weights as stored (bf16 values), Dia's RoPE tables and its
//! output rounded to bf16 as torch has them. Origin: Zyphra's fusealign-multilingual (fusealign_hub/model.py,
//! text_encoder.py, ggml/src/model.cpp, fusealign.cpp, longform.cpp fill_unplaced; Apache-2.0) and the GGUF
//! fusealign-multilingual-ns4-step60000-f32.gguf (FuseAlign Apache-2.0; Dia's text encoder from nari-labs/Dia-1.6B-0626,
//! Apache-2.0).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ops::{fp, nulf, Mat, Ops, Weights};
use crate::spk::resample_any;

pub const RATE: u32 = 16_000;

fn bf16(x: f32) -> f32 {
    let u = x.to_bits();
    if u & 0x7f80_0000 == 0x7f80_0000 {
        return x;
    }
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

fn mat_from(ops: &Ops, v: &[f32], n: usize, k: usize) -> Result<Mat> {
    Ok(Mat { w: ops.from_f32(v)?, n, k })
}

struct Block {
    attn_norm: DevBuf,
    qkv: Mat,
    out: Mat,
    ffn_norm: DevBuf,
    w1: Mat,
    w2: Mat,
    w3: Mat,
}

struct DiaLayer {
    pre: DevBuf,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    post: DevBuf,
    gate_up: Mat,
    down: Mat,
}

struct Head {
    inp: Mat,
    inp_b: DevBuf,
    /// ConvTranspose1d as a product [(k, out), in] and its bias
    up: Vec<(Mat, DevBuf)>,
    /// Conv1d kernel 5: weight [out, in, 5] (conv1d's layout), bias
    smooth: DevBuf,
    smooth_b: DevBuf,
    out: Mat,
    out_b: DevBuf,
}

struct Mlp {
    w0: Mat,
    b0: DevBuf,
    w2: Mat,
    b2: DevBuf,
}

/// The aligner: FuseAlign and Dia's text encoder on the card
pub struct FuseAlign {
    dim: usize,
    heads: usize,
    ffn: usize,
    feat: usize,
    window: usize,
    hop: usize,
    upsample: usize,
    theta: f32,
    pub threshold: f32,
    pub max_bytes: usize,
    pub max_seconds: f32,
    dia_d: usize,
    dia_heads: usize,
    dia_hd: usize,
    dia_inter: usize,
    dia_eps: f32,
    dia_theta: f32,
    patch: Mat,
    patch_b: DevBuf,
    text_proj: Mat,
    text_proj_b: DevBuf,
    mod_emb: Vec<f32>,
    blocks: Vec<Block>,
    final_norm: DevBuf,
    start: Head,
    end: Head,
    start_mlp: Mlp,
    end_mlp: Mlp,
    presence_mlp: Mlp,
    dia_emb: Vec<f32>,
    dia: Vec<DiaLayer>,
    dia_norm: DevBuf,
}

/// One word's place: (start, end) seconds when it was placed, and its presence
#[derive(Clone, Debug)]
pub struct Placed {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub presence: f32,
    pub placed: bool,
}

impl FuseAlign {
    pub fn load(ops: &Ops, w: &Weights) -> Result<FuseAlign> {
        if w.g.architecture() != "fusealign" {
            return Err(Error(format!("{}: not a FuseAlign GGUF", w.name())));
        }
        let u = |k: &str| w.u(k);
        let f = |k: &str| w.f(k).map(|x| x as f32);
        let (dim, heads, ffn, feat) = (u("fusealign.dim")?, u("fusealign.n_heads")?, u("fusealign.ffn_dim")?, u("fusealign.feat_dim")?);
        let (window, hop, upsample) = (u("fusealign.patch_window")?, u("fusealign.patch_hop")?, u("fusealign.n_upsample")?);
        let mat = |n: &str| w.mat(ops, n);
        let dev = |n: &str| w.dev(ops, n);
        let raw = |n: &str, rows: usize, cols: usize| -> Result<Mat> { mat_from(ops, &w.f32(n)?, rows, cols) };
        let mut blocks = Vec::new();
        for i in 0..u("fusealign.n_layers")? {
            let p = |s: &str| format!("model.blocks.{i}.{s}");
            blocks.push(Block { attn_norm: dev(&p("attn_norm.weight"))?, qkv: mat(&p("attn.qkv.weight"))?, out: mat(&p("attn.out.weight"))?,
                                ffn_norm: dev(&p("ffn_norm.weight"))?, w1: mat(&p("ffn.w1.weight"))?, w2: mat(&p("ffn.w2.weight"))?, w3: mat(&p("ffn.w3.weight"))? });
        }
        let head = |name: &str| -> Result<Head> {
            let p = |s: &str| format!("model.{name}_refinement.{s}");
            let mut up = Vec::new();
            for i in 0..upsample {
                // torch [in, out, k] -> rows (k, out), columns in
                let v = w.f32(&p(&format!("{}.weight", 1 + 2 * i)))?;
                let k = 4;
                let mut m = vec![0f32; k * feat * feat];
                for ic in 0..feat {
                    for oc in 0..feat {
                        for kk in 0..k {
                            m[(kk * feat + oc) * feat + ic] = v[(ic * feat + oc) * k + kk];
                        }
                    }
                }
                up.push((mat_from(ops, &m, k * feat, feat)?, dev(&p(&format!("{}.bias", 1 + 2 * i)))?));
            }
            Ok(Head {
                inp: raw(&p("0.weight"), feat, dim)?,
                inp_b: dev(&p("0.bias"))?,
                up,
                smooth: dev(&p(&format!("{}.weight", 1 + 2 * upsample)))?,
                smooth_b: dev(&p(&format!("{}.bias", 1 + 2 * upsample)))?,
                out: raw(&p(&format!("{}.weight", 3 + 2 * upsample)), feat, feat)?,
                out_b: dev(&p(&format!("{}.bias", 3 + 2 * upsample)))?,
            })
        };
        let mlp = |name: &str| -> Result<Mlp> {
            let p = |s: &str| format!("model.{name}_mlp.{s}");
            Ok(Mlp { w0: mat(&p("0.weight"))?, b0: dev(&p("0.bias"))?, w2: mat(&p("2.weight"))?, b2: dev(&p("2.bias"))? })
        };
        let mut dia = Vec::new();
        for i in 0..u("dia.num_layers")? {
            let p = |s: &str| format!("text_encoder.layers.{i}.{s}");
            dia.push(DiaLayer { pre: dev(&p("pre_sa_norm.weight"))?, q: mat(&p("self_attention.q_proj.weight"))?, k: mat(&p("self_attention.k_proj.weight"))?,
                                v: mat(&p("self_attention.v_proj.weight"))?, o: mat(&p("self_attention.o_proj.weight"))?, post: dev(&p("post_sa_norm.weight"))?,
                                gate_up: mat(&p("mlp.gate_up_proj.weight"))?, down: mat(&p("mlp.down_proj.weight"))? });
        }
        Ok(FuseAlign {
            dim, heads, ffn, feat, window, hop, upsample,
            theta: f("fusealign.rope_theta")?,
            threshold: f("fusealign.presence_threshold")?,
            max_bytes: u("fusealign.max_text_bytes")?,
            max_seconds: f("fusealign.max_seconds")?,
            dia_d: u("dia.hidden_size")?,
            dia_heads: u("dia.num_heads")?,
            dia_hd: u("dia.head_dim")?,
            dia_inter: u("dia.intermediate_size")?,
            dia_eps: f("dia.norm_eps")?,
            dia_theta: f("dia.rope_theta")?,
            patch: raw("model.audio_patch.weight", dim, window)?,
            patch_b: dev("model.audio_patch.bias")?,
            text_proj: mat("model.text_proj.weight")?,
            text_proj_b: dev("model.text_proj.bias")?,
            mod_emb: w.f32("model.mod_emb")?,
            blocks,
            final_norm: dev("model.final_norm.weight")?,
            start: head("start")?,
            end: head("end")?,
            start_mlp: mlp("start")?,
            end_mlp: mlp("end")?,
            presence_mlp: mlp("presence")?,
            dia_emb: w.f32("text_encoder.embedding.weight")?,
            dia,
            dia_norm: dev("text_encoder.norm.weight")?,
        })
    }

    /// cos and sin tables [n, hd] of rotate-half RoPE at `pos` (inv_freq[i] for i < hd / 2, repeated)
    fn tables(pos: &[f32], inv: &[f32], round: bool) -> (Vec<f32>, Vec<f32>) {
        let half = inv.len();
        let hd = 2 * half;
        let (mut c, mut s) = (vec![0f32; pos.len() * hd], vec![0f32; pos.len() * hd]);
        for (p, x) in pos.iter().enumerate() {
            for i in 0..hd {
                let a = x * inv[i % half];
                let (mut cc, mut ss) = (a.cos(), a.sin());
                if round {
                    cc = bf16(cc);
                    ss = bf16(ss);
                }
                c[p * hd + i] = cc;
                s[p * hd + i] = ss;
            }
        }
        (c, s)
    }

    /// Dia's text features of `ids` (with the two-token prefix): [len - 2, D], rounded to bf16
    fn text(&self, ops: &Ops, ids: &[u32]) -> Result<Vec<f32>> {
        let (l, d, nh, hd) = (ids.len(), self.dia_d, self.dia_heads, self.dia_hd);
        let hdw = nh * hd;
        let mut x0 = vec![0f32; l * d];
        for (i, id) in ids.iter().enumerate() {
            x0[i * d..(i + 1) * d].copy_from_slice(&self.dia_emb[*id as usize * d..(*id as usize + 1) * d]);
        }
        let inv: Vec<f32> = (0..hd / 2).map(|i| 1.0 / (self.dia_theta as f64).powf((2 * i) as f64 / hd as f64) as f32).collect();
        let pos: Vec<f32> = (0..l).map(|i| i as f32).collect();
        let (c, s) = Self::tables(&pos, &inv, true);
        let (cs, sn) = (ops.from_f32(&c)?, ops.from_f32(&s)?);
        let x = ops.from_f32(&x0)?;
        let n = |c: usize| ops.f32(l * c);
        let (h, q, k, v, a, b) = (n(d)?, n(hdw)?, n(hdw)?, n(hdw)?, n(hdw.max(d))?, n(d)?);
        let big = n(2 * self.dia_inter)?;
        let gate = n(self.dia_inter)?;
        let upb = n(self.dia_inter)?;
        let ones = ops.from_f32(&vec![1.0f32; d])?;
        for ly in &self.dia {
            ops.rms(x.fp(), d, l, d, ly.pre.fp(), self.dia_eps, h.fp())?;
            ops.linear(h.fp(), l, &ly.q, nulf(), q.fp())?;
            ops.linear(h.fp(), l, &ly.k, nulf(), k.fp())?;
            ops.linear(h.fp(), l, &ly.v, nulf(), v.fp())?;
            ops.rope_tab(q.fp(), hdw, l, nh, hd, cs.fp(), sn.fp())?;
            ops.rope_tab(k.fp(), hdw, l, nh, hd, cs.fp(), sn.fp())?;
            // DiaSelfAttention: scaling 1
            ops.attn_full(q.fp(), k.fp(), v.fp(), hdw, l, nh, hd, 1.0, a.fp())?;
            ops.linear(a.fp(), l, &ly.o, nulf(), b.fp())?;
            ops.scale_add(x.fp(), b.fp(), ones.fp(), l, d)?;
            ops.rms(x.fp(), d, l, d, ly.post.fp(), self.dia_eps, h.fp())?;
            ops.linear(h.fp(), l, &ly.gate_up, nulf(), big.fp())?;
            // gate first, then up: silu(gate) * up
            gate.copy_2d(0, self.dia_inter * 4, &big, 0, 2 * self.dia_inter * 4, self.dia_inter * 4, l).map_err(crate::ops::e)?;
            upb.copy_2d(0, self.dia_inter * 4, &big, self.dia_inter * 4, 2 * self.dia_inter * 4, self.dia_inter * 4, l).map_err(crate::ops::e)?;
            ops.silu_mul(gate.fp(), upb.fp(), upb.fp(), l * self.dia_inter)?;
            ops.linear(upb.fp(), l, &ly.down, nulf(), b.fp())?;
            ops.scale_add(x.fp(), b.fp(), ones.fp(), l, d)?;
        }
        ops.rms(x.fp(), d, l, d, self.dia_norm.fp(), self.dia_eps, h.fp())?;
        let out = h.to_f32().map_err(crate::ops::e)?;
        Ok(out[2 * d..l * d].iter().map(|v| bf16(*v)).collect())
    }

    /// A refinement head over the joint audio rows [a, dim] -> features [16 a, feat]
    fn refine(&self, ops: &Ops, head: &Head, audio: &DevBuf, a: usize) -> Result<(DevBuf, usize)> {
        let feat = self.feat;
        let fr = a << self.upsample;
        let n = |c: usize| ops.f32(c);
        let (x, y, cols) = (n(fr * feat)?, n(fr * feat)?, n(fr * 4 * feat)?);
        ops.linear(audio.fp(), a, &head.inp, head.inp_b.fp(), x.fp())?;
        let mut t = a;
        for (m, b) in &head.up {
            ops.linear(x.fp(), t, m, nulf(), cols.fp())?;
            ops.col2im(cols.fp(), t, 4, feat, 2, 1, b.fp(), 2 * t, y.fp())?;
            t *= 2;
            ops.gelu(y.fp(), t * feat, false)?;
            x.copy_within(0, &y, 0, t * feat * 4).map_err(crate::ops::e)?;
        }
        // conv 5 (padding 2) channels-first, then GELU and the 1x1
        let pad = n(feat * (t + 4))?;
        pad.fill(0).map_err(crate::ops::e)?;
        ops.swap01(x.fp(), t, feat, 1, y.fp())?;
        pad.copy_2d(2 * 4, (t + 4) * 4, &y, 0, t * 4, t * 4, feat).map_err(crate::ops::e)?;
        let lo = ops.conv1d(pad.fp(), feat, t + 4, head.smooth.fp(), feat, 5, head.smooth_b.fp(), 1, 1, y.fp())?;
        debug_assert_eq!(lo, t);
        ops.swap01(y.fp(), feat, t, 1, x.fp())?;
        ops.gelu(x.fp(), t * feat, false)?;
        ops.linear(x.fp(), t, &head.out, head.out_b.fp(), y.fp())?;
        Ok((y, t))
    }

    fn mlp(&self, ops: &Ops, m: &Mlp, x: &DevBuf, w: usize) -> Result<DevBuf> {
        let h = ops.f32(w * self.dim)?;
        ops.linear(x.fp(), w, &m.w0, m.b0.fp(), h.fp())?;
        ops.gelu(h.fp(), w * self.dim, false)?;
        let out = ops.f32(w * m.w2.n)?;
        ops.linear(h.fp(), w, &m.w2, m.b2.fp(), out.fp())?;
        Ok(out)
    }

    /// One forward: 16 kHz audio and its words -> each word's (start, end) seconds when present, and presence
    pub fn forward(&self, ops: &Ops, wav: &[f32], words: &[String]) -> Result<Vec<(Option<(f64, f64)>, f32)>> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        if wav.len() < self.window {
            return Err(Error("the recording is shorter than one FuseAlign patch".into()));
        }
        let mut ids: Vec<u32> = vec![1, 32];
        let mut spans = Vec::new();
        for (i, wd) in words.iter().enumerate() {
            if i > 0 {
                ids.push(32);
            }
            let s = ids.len() - 2;
            ids.extend(wd.bytes().map(|b| b as u32));
            spans.push((s, ids.len() - 2));
        }
        let t = ids.len() - 2;
        if t > self.max_bytes {
            return Err(Error(format!("the words are {t} UTF-8 bytes; FuseAlign aligns at most {} at once", self.max_bytes)));
        }
        let (dim, nh) = (self.dim, self.heads);
        let hd = dim / nh;
        let a = (wav.len() - self.window) / self.hop + 1;
        let nn = a + t;
        // standardized (mean and deviation over the clip, accumulated in double)
        let mean = (wav.iter().map(|x| *x as f64).sum::<f64>() / wav.len() as f64) as f32;
        let centered: Vec<f32> = wav.iter().map(|x| x - mean).collect();
        let sd = ((centered.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / wav.len() as f64) as f32).sqrt() + 1e-5;
        let mut frames = vec![0f32; a * self.window];
        for i in 0..a {
            for j in 0..self.window {
                frames[i * self.window + j] = centered[i * self.hop + j] / sd;
            }
        }
        let text = self.text(ops, &ids)?;
        // the joint sequence: audio patches + mod 0, projected text + mod 1
        let x = ops.f32(nn * dim)?;
        let fr = ops.from_f32(&frames)?;
        ops.linear(fr.fp(), a, &self.patch, self.patch_b.fp(), x.fp())?;
        let tf = ops.from_f32(&text)?;
        ops.linear(tf.fp(), t, &self.text_proj, self.text_proj_b.fp(), fp(&x, a * dim))?;
        let m0 = ops.from_f32(&self.mod_emb[..dim])?;
        let m1 = ops.from_f32(&self.mod_emb[dim..2 * dim])?;
        ops.add_vec(x.fp(), a, dim, m0.fp())?;
        ops.add_vec(fp(&x, a * dim), t, dim, m1.fp())?;
        // RoPE positions: audio frames, then the text bytes spread over the audio
        let scale = a as f32 / (t.saturating_sub(1).max(1)) as f32;
        let pos: Vec<f32> = (0..a).map(|i| i as f32).chain((0..t).map(|j| j as f32 * scale)).collect();
        let inv: Vec<f32> = (0..hd / 2).map(|i| (self.theta as f64).powf(-((2 * i) as f64) / hd as f64) as f32).collect();
        let (c, s) = Self::tables(&pos, &inv, false);
        let (cs, sn) = (ops.from_f32(&c)?, ops.from_f32(&s)?);
        let n = |c: usize| ops.f32(nn * c);
        let (h, qkv, att, b, g) = (n(dim)?, n(3 * dim)?, n(dim)?, n(dim)?, n(self.ffn)?);
        let u = n(self.ffn)?;
        let ones = ops.from_f32(&vec![1.0f32; dim])?;
        let sc = 1.0 / (hd as f32).sqrt();
        for bl in &self.blocks {
            ops.rms(x.fp(), dim, nn, dim, bl.attn_norm.fp(), 1e-6, h.fp())?;
            ops.linear(h.fp(), nn, &bl.qkv, nulf(), qkv.fp())?;
            ops.rope_tab(qkv.fp(), 3 * dim, nn, nh, hd, cs.fp(), sn.fp())?;
            ops.rope_tab(fp(&qkv, dim), 3 * dim, nn, nh, hd, cs.fp(), sn.fp())?;
            ops.attn_full(qkv.fp(), fp(&qkv, dim), fp(&qkv, 2 * dim), 3 * dim, nn, nh, hd, sc, att.fp())?;
            ops.linear(att.fp(), nn, &bl.out, nulf(), b.fp())?;
            ops.scale_add(x.fp(), b.fp(), ones.fp(), nn, dim)?;
            ops.rms(x.fp(), dim, nn, dim, bl.ffn_norm.fp(), 1e-6, h.fp())?;
            ops.linear(h.fp(), nn, &bl.w1, nulf(), g.fp())?;
            ops.linear(h.fp(), nn, &bl.w3, nulf(), u.fp())?;
            ops.silu_mul(g.fp(), u.fp(), u.fp(), nn * self.ffn)?;
            ops.linear(u.fp(), nn, &bl.w2, nulf(), b.fp())?;
            ops.scale_add(x.fp(), b.fp(), ones.fp(), nn, dim)?;
        }
        ops.rms(x.fp(), dim, nn, dim, self.final_norm.fp(), 1e-6, h.fp())?;
        // the words: their bytes mean-pooled (host: W x T x dim)
        let all = h.to_f32().map_err(crate::ops::e)?;
        let wn = words.len();
        let mut wv = vec![0f32; wn * dim];
        for (wi, (s0, s1)) in spans.iter().enumerate() {
            let cnt = (s1 - s0).max(1) as f32;
            for j in *s0..*s1 {
                let row = &all[(a + j) * dim..(a + j + 1) * dim];
                for (o, v) in wv[wi * dim..(wi + 1) * dim].iter_mut().zip(row) {
                    *o += v / cnt;
                }
            }
        }
        let wvd = ops.from_f32(&wv)?;
        let audio = ops.f32(a * dim)?;
        audio.copy_within(0, &h, 0, a * dim * 4).map_err(crate::ops::e)?;
        let mut best: Vec<[usize; 2]> = vec![[0, 0]; wn];
        let mut frames_out = 0;
        for (side, (head, mlp)) in [(&self.start, &self.start_mlp), (&self.end, &self.end_mlp)].into_iter().enumerate() {
            let (feats, f) = self.refine(ops, head, &audio, a)?;
            frames_out = f;
            let q = self.mlp(ops, mlp, &wvd, wn)?;
            // logits [f, wn] = features [f, feat] . word [wn, feat]
            let m = Mat { w: q, n: wn, k: self.feat };
            let lg = ops.f32(f * wn)?;
            ops.linear(feats.fp(), f, &m, nulf(), lg.fp())?;
            let v = lg.to_f32().map_err(crate::ops::e)?;
            for (wi, b) in best.iter_mut().enumerate() {
                let mut arg = 0;
                for fi in 0..f {
                    if v[fi * wn + wi] > v[arg * wn + wi] {
                        arg = fi;
                    }
                }
                b[side] = arg;
            }
        }
        let _ = frames_out;
        let pres = self.mlp(ops, &self.presence_mlp, &wvd, wn)?.to_f32().map_err(crate::ops::e)?;
        Ok((0..wn).map(|i| {
            let p = 1.0 / (1.0 + (-pres[i]).exp());
            let p = if p.is_nan() { 0.0 } else { p };
            let iv = (p >= self.threshold).then(|| (best[i][0] as f64 / 1000.0, best[i][0].max(best[i][1]) as f64 / 1000.0));
            (iv, p)
        }).collect())
    }

    /// The words of a recording (any rate) placed: the words with a letter or digit aligned, an unplaced one put in
    /// the gap around it, the others at the end of the word before (fusealign.cpp Aligner::align)
    pub fn align(&self, ops: &Ops, samples: &[f32], rate: u32, words: &[String]) -> Result<Vec<Placed>> {
        let wav = if rate == RATE { samples.to_vec() } else { resample_any(samples, rate, RATE) };
        if wav.len() as f32 > self.max_seconds * RATE as f32 {
            return Err(Error(format!("the recording is {:.1} s; FuseAlign aligns at most {} s at once", wav.len() as f32 / RATE as f32, self.max_seconds)));
        }
        let wordy: Vec<usize> = (0..words.len()).filter(|i| words[*i].chars().any(char::is_alphanumeric)).collect();
        let toks: Vec<String> = wordy.iter().map(|i| words[*i].clone()).collect();
        let placed = if wav.len() < self.window { vec![(None, 0.0); toks.len()] } else { self.forward(ops, &wav, &toks)? };
        let duration = wav.len() as f64 / RATE as f64;
        let filled = fill_unplaced(&placed.iter().map(|p| p.0).collect::<Vec<_>>(), duration);
        let mut out = Vec::new();
        let first = filled.first().map_or(0.0, |f| f.0);
        let mut prev: Option<f64> = None;
        let mut j = 0;
        for (k, wd) in words.iter().enumerate() {
            if j < wordy.len() && wordy[j] == k {
                out.push(Placed { word: wd.clone(), start: filled[j].0, end: filled[j].1, presence: placed[j].1, placed: placed[j].0.is_some() });
                prev = Some(filled[j].1);
                j += 1;
            } else {
                let t = prev.unwrap_or(first);
                out.push(Placed { word: wd.clone(), start: t, end: t, presence: 0.0, placed: false });
                prev = Some(t);
            }
        }
        Ok(out)
    }
}

/// Unplaced words spread evenly over the gap between their placed neighbours (the clip's ends at the edges)
pub fn fill_unplaced(iv: &[Option<(f64, f64)>], duration: f64) -> Vec<(f64, f64)> {
    let mut f: Vec<Option<(f64, f64)>> = iv.to_vec();
    let mut i = 0;
    while i < f.len() {
        if f[i].is_some() {
            i += 1;
            continue;
        }
        let lo = i;
        while i < f.len() && f[i].is_none() {
            i += 1;
        }
        let left = if lo > 0 { f[lo - 1].unwrap().1 } else { 0.0 };
        let right = if i < f.len() { f[i].unwrap().0 } else { duration };
        let width = (right - left) / (i - lo) as f64;
        for o in 0..i - lo {
            f[lo + o] = Some((left + o as f64 * width, if lo + o + 1 == i { right } else { left + (o + 1) as f64 * width }));
        }
    }
    f.into_iter().map(|x| x.unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unplaced_words_share_their_gap() {
        let f = fill_unplaced(&[Some((0.1, 0.3)), None, None, Some((0.9, 1.0)), None], 2.0);
        assert!((f[1].0 - 0.3).abs() < 1e-12 && (f[1].1 - 0.6).abs() < 1e-12, "{:?}", f[1]);
        assert!((f[2].0 - 0.6).abs() < 1e-12 && f[2].1 == 0.9);
        assert_eq!(f[4], (1.0, 2.0));
    }
}
