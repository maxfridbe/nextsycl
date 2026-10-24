//! Whisper's transformer on one GPU, from the checkpoint's safetensors (half): the encoder (two convolutions and GELU
//! in float32, the sinusoidal-free learned positions, pre-norm layers of attention and a GELU MLP) and the decoder (its
//! tokens' embedding, tied to the output, causal self-attention over a cache, cross-attention over the encoder's keys
//! and values, kept a window at a time). The residual stream is float32; the norms write half for the products, which
//! accumulate in float32 (oneMKL). Everything is the shared kernels' (nsd), and one transpose.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::{f16_to_f32, SafeTensors};
use serde_json::Value;

use crate::ffi;
use crate::mel::CHUNK_FRAMES;

type P = *const c_void;
type M = *mut c_void;

const EPS: f32 = 1e-5;

/// config.json's sizes
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub d: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub enc_heads: usize,
    pub dec_heads: usize,
    pub enc_ffn: usize,
    pub dec_ffn: usize,
    pub n_mels: usize,
    /// encoder positions (1500: 30 s)
    pub audio_ctx: usize,
    /// decoder positions (448)
    pub text_ctx: usize,
    pub vocab: usize,
}

impl Config {
    pub fn read(path: &Path) -> Result<Config> {
        let v: Value = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or_else(|| Error(format!("{}: not JSON", path.display())))?;
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{}: no {k}", path.display())));
        if v["model_type"].as_str().is_some_and(|t| t != "whisper") {
            return Err(Error(format!("{}: model_type {}, not whisper", path.display(), v["model_type"])));
        }
        Ok(Config { d: n("d_model")?, enc_layers: n("encoder_layers")?, dec_layers: n("decoder_layers")?, enc_heads: n("encoder_attention_heads")?,
                    dec_heads: n("decoder_attention_heads")?, enc_ffn: n("encoder_ffn_dim")?, dec_ffn: n("decoder_ffn_dim")?, n_mels: n("num_mel_bins")?,
                    audio_ctx: n("max_source_positions")?, text_ctx: n("max_target_positions")?, vocab: n("vocab_size")? })
    }

    /// bytes of weights on the card (half), roughly
    pub fn weight_bytes(&self) -> u64 {
        let d = self.d as u64;
        let enc = self.enc_layers as u64 * (4 * d * d + 2 * d * self.enc_ffn as u64);
        let dec = self.dec_layers as u64 * (8 * d * d + 2 * d * self.dec_ffn as u64);
        2 * (enc + dec + self.vocab as u64 * d) + 4 * (3 * d * d + d * self.n_mels as u64 * 3 + self.audio_ctx as u64 * d)
    }
}

/// A pointer `bytes` into a buffer
fn at(b: &DevBuf, bytes: usize) -> M {
    b.ptr().cast::<u8>().wrapping_add(bytes).cast()
}

struct Lin {
    w: DevBuf,
    b: Option<DevBuf>,
    n: usize,
    k: usize,
}

struct Ln {
    w: DevBuf,
    b: DevBuf,
}

struct EncLayer {
    ln1: Ln,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2: Ln,
    fc1: Lin,
    fc2: Lin,
}

struct DecLayer {
    ln1: Ln,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    lnx: Ln,
    xq: Lin,
    xk: Lin,
    xv: Lin,
    xo: Lin,
    ln2: Ln,
    fc1: Lin,
    fc2: Lin,
}

struct Loader<'a> {
    st: &'a SafeTensors,
    gpu: &'a Arc<Gpu>,
}

impl Loader<'_> {
    fn half(&self, name: &str) -> Result<(DevBuf, Vec<u64>)> {
        let t = self.st.need(name).map_err(|e| Error(e.0))?;
        let bytes = match t.dtype.as_str() {
            "F16" => self.st.read(t).map_err(|e| Error(e.0))?,
            // float32 or bfloat16 checkpoints: to half on the host
            _ => self.st.f32(name).map_err(|e| Error(e.0))?.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect(),
        };
        let b = DevBuf::new(self.gpu, bytes.len())?;
        b.write(0, &bytes)?;
        Ok((b, t.shape.clone()))
    }

    fn float(&self, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(self.gpu, &self.st.f32(name).map_err(|e| Error(e.0))?)
    }

    fn lin(&self, p: &str, bias: bool) -> Result<Lin> {
        let (w, shape) = self.half(&format!("{p}.weight"))?;
        let b = if bias { Some(self.float(&format!("{p}.bias"))?) } else { None };
        Ok(Lin { w, b, n: shape[0] as usize, k: shape[1] as usize })
    }

    fn ln(&self, p: &str) -> Result<Ln> {
        Ok(Ln { w: self.float(&format!("{p}.weight"))?, b: self.float(&format!("{p}.bias"))? })
    }
}

/// float32 to IEEE half (round to nearest even)
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let r = if rem > half || (rem == half && r & 1 == 1) { r + 1 } else { r };
        return s | r as u16;
    }
    let r = m >> 13;
    let rem = m & 0x1fff;
    let mut v = ((e2 as u32) << 10) | r;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        v += 1;
    }
    s | v as u16
}

pub struct Model {
    pub cfg: Config,
    pub nsd: Nsd,
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
    conv1_w: DevBuf,
    conv1_b: DevBuf,
    conv2_w: DevBuf,
    conv2_b: DevBuf,
    enc_pos: DevBuf,
    enc: Vec<EncLayer>,
    enc_ln: Ln,
    /// the token embedding: the output's matrix too (half, on the card) and the rows read on the host
    emb: DevBuf,
    emb_host: Vec<u16>,
    dec_pos_host: Vec<f32>,
    dec: Vec<DecLayer>,
    dec_ln: Ln,
    // the encoder's work
    mel: DevBuf,
    c1: DevBuf,
    c2: DevBuf,
    x: DevBuf,
    h: DevBuf,
    q: DevBuf,
    kk: DevBuf,
    v: DevBuf,
    att: DevBuf,
    y: DevBuf,
    ff: DevBuf,
    enc_out: DevBuf,
    // the decoder's: the window's cross keys and values, the self-attention caches, its rows
    xk: Vec<DevBuf>,
    xv: Vec<DevBuf>,
    kc: Vec<DevBuf>,
    vc: Vec<DevBuf>,
    qbuf: DevBuf,
    dx: DevBuf,
    dh: DevBuf,
    dq: DevBuf,
    datt: DevBuf,
    dy: DevBuf,
    dff: DevBuf,
    logits: DevBuf,
}

/// What a decoder pass also brings back: the alignment layers' cross-attention queries ([rows, d] floats, by layer)
#[derive(Default)]
pub struct Taps {
    pub layers: Vec<usize>,
    pub q: Vec<Vec<f32>>,
}

/// How many rows of logits a pass reads back at most (the decoder's context)
const LOGIT_ROWS: usize = 448;

impl Model {
    pub fn load(weights: &Path, cfg: Config, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Model> {
        let st = SafeTensors::open(weights).map_err(|e| Error(e.0))?;
        let l = Loader { st: &st, gpu };
        let (d, ac, tc) = (cfg.d, cfg.audio_ctx, cfg.text_ctx);
        log(format!("whisper: {} encoder and {} decoder layers, d {d}, {} mels, {:.2} GiB of weights on {}", cfg.enc_layers, cfg.dec_layers, cfg.n_mels,
                    cfg.weight_bytes() as f64 / (1u64 << 30) as f64, gpu.name));
        let e = "model.encoder";
        let conv1_w = l.float(&format!("{e}.conv1.weight"))?;
        let conv1_b = l.float(&format!("{e}.conv1.bias"))?;
        let conv2_w = l.float(&format!("{e}.conv2.weight"))?;
        let conv2_b = l.float(&format!("{e}.conv2.bias"))?;
        let enc_pos = l.float(&format!("{e}.embed_positions.weight"))?;
        let mut enc = Vec::new();
        for i in 0..cfg.enc_layers {
            let p = format!("{e}.layers.{i}");
            enc.push(EncLayer {
                ln1: l.ln(&format!("{p}.self_attn_layer_norm"))?,
                q: l.lin(&format!("{p}.self_attn.q_proj"), true)?,
                k: l.lin(&format!("{p}.self_attn.k_proj"), false)?,
                v: l.lin(&format!("{p}.self_attn.v_proj"), true)?,
                o: l.lin(&format!("{p}.self_attn.out_proj"), true)?,
                ln2: l.ln(&format!("{p}.final_layer_norm"))?,
                fc1: l.lin(&format!("{p}.fc1"), true)?,
                fc2: l.lin(&format!("{p}.fc2"), true)?,
            });
        }
        let enc_ln = l.ln(&format!("{e}.layer_norm"))?;
        let dd = "model.decoder";
        let (emb, _) = l.half(&format!("{dd}.embed_tokens.weight"))?;
        let emb_host: Vec<u16> = {
            let t = st.need(&format!("{dd}.embed_tokens.weight")).map_err(|e| Error(e.0))?;
            match t.dtype.as_str() {
                "F16" => st.read(t).map_err(|e| Error(e.0))?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
                _ => st.f32(&t.name).map_err(|e| Error(e.0))?.iter().map(|x| f32_to_f16(*x)).collect(),
            }
        };
        let dec_pos_host = st.f32(&format!("{dd}.embed_positions.weight")).map_err(|e| Error(e.0))?;
        let mut dec = Vec::new();
        for i in 0..cfg.dec_layers {
            let p = format!("{dd}.layers.{i}");
            dec.push(DecLayer {
                ln1: l.ln(&format!("{p}.self_attn_layer_norm"))?,
                q: l.lin(&format!("{p}.self_attn.q_proj"), true)?,
                k: l.lin(&format!("{p}.self_attn.k_proj"), false)?,
                v: l.lin(&format!("{p}.self_attn.v_proj"), true)?,
                o: l.lin(&format!("{p}.self_attn.out_proj"), true)?,
                lnx: l.ln(&format!("{p}.encoder_attn_layer_norm"))?,
                xq: l.lin(&format!("{p}.encoder_attn.q_proj"), true)?,
                xk: l.lin(&format!("{p}.encoder_attn.k_proj"), false)?,
                xv: l.lin(&format!("{p}.encoder_attn.v_proj"), true)?,
                xo: l.lin(&format!("{p}.encoder_attn.out_proj"), true)?,
                ln2: l.ln(&format!("{p}.final_layer_norm"))?,
                fc1: l.lin(&format!("{p}.fc1"), true)?,
                fc2: l.lin(&format!("{p}.fc2"), true)?,
            });
        }
        let dec_ln = l.ln(&format!("{dd}.layer_norm"))?;
        let h16 = |n: usize| DevBuf::new(gpu, n * 2);
        let f32b = |n: usize| DevBuf::f32(gpu, n);
        let ffn = cfg.enc_ffn.max(cfg.dec_ffn);
        let m = Model {
            nsd: Nsd::new(gpu)?,
            gpu: gpu.clone(),
            k: ffi::api()?,
            conv1_w, conv1_b, conv2_w, conv2_b, enc_pos, enc, enc_ln, emb, emb_host, dec_pos_host, dec_ln,
            mel: f32b(cfg.n_mels * CHUNK_FRAMES)?,
            c1: f32b(d * CHUNK_FRAMES)?,
            c2: f32b(d * ac)?,
            x: f32b(ac * d)?,
            h: h16(ac * d)?,
            q: h16(ac * d)?,
            kk: h16(ac * d)?,
            v: h16(ac * d)?,
            att: h16(ac * d)?,
            y: h16(ac * d)?,
            ff: h16(ac * ffn)?,
            enc_out: h16(ac * d)?,
            xk: (0..cfg.dec_layers).map(|_| h16(ac * d)).collect::<Result<_>>()?,
            xv: (0..cfg.dec_layers).map(|_| h16(ac * d)).collect::<Result<_>>()?,
            kc: (0..cfg.dec_layers).map(|_| h16(tc * d)).collect::<Result<_>>()?,
            vc: (0..cfg.dec_layers).map(|_| h16(tc * d)).collect::<Result<_>>()?,
            qbuf: h16(tc * d)?,
            dx: f32b(tc * d)?,
            dh: h16(tc * d)?,
            dq: h16(tc * d)?,
            datt: h16(tc * d)?,
            dy: h16(tc * d)?,
            dff: h16(tc * ffn)?,
            logits: f32b(LOGIT_ROWS * cfg.vocab)?,
            dec,
            cfg,
        };
        Ok(m)
    }

    fn linear(&self, x: P, m: usize, w: &Lin, out: M) -> Result<()> {
        self.nsd.linear(x, Dt::F16, m, w.k, w.w.ptr(), w.n, w.b.as_ref().map_or(none(), |b| b.ptr() as P), out, Dt::F16)
    }

    fn ln(&self, x: P, m: usize, n: &Ln, out: M) -> Result<()> {
        self.nsd.layer_norm(x, Dt::F32, m, self.cfg.d, n.w.ptr(), n.b.ptr(), EPS, out, Dt::F16)
    }

    /// x (float32 [m, d]) += y (half [m, d])
    fn add(&self, x: M, y: P, m: usize) -> Result<()> {
        self.nsd.gate_add(x, Dt::F32, m, self.cfg.d, y, Dt::F16, none(), none())
    }

    /// The encoder over one window's spectrogram [n_mels, 3000]: its output, and each decoder layer's cross keys and
    /// values from it (kept for the window's decoding)
    pub fn encode(&self, mel: &[f32]) -> Result<()> {
        let c = &self.cfg;
        let (d, s) = (c.d, c.audio_ctx);
        if mel.len() != c.n_mels * CHUNK_FRAMES {
            return Err(Error(format!("a window of {} mel values, not {}", mel.len(), c.n_mels * CHUNK_FRAMES)));
        }
        // SAFETY: plain floats as bytes.
        self.mel.write(0, unsafe { std::slice::from_raw_parts(mel.as_ptr().cast::<u8>(), mel.len() * 4) })?;
        let n = &self.nsd;
        n.conv1d(self.mel.ptr(), 1, c.n_mels, CHUNK_FRAMES, self.conv1_w.ptr(), d, 3, self.conv1_b.ptr(), 1, 1, 1, self.c1.ptr(), CHUNK_FRAMES)?;
        n.gelu(self.c1.ptr(), Dt::F32, d * CHUNK_FRAMES, false)?;
        n.conv1d(self.c1.ptr(), 1, d, CHUNK_FRAMES, self.conv2_w.ptr(), d, 3, self.conv2_b.ptr(), 2, 1, 1, self.c2.ptr(), s)?;
        n.gelu(self.c2.ptr(), Dt::F32, d * s, false)?;
        // [d, s] -> [s, d]
        // SAFETY: c2 holds d x s floats, x s x d.
        ffi::check(unsafe { (self.k.transpose)(self.gpu.raw(), self.c2.fp(), d as i64, s as i64, self.x.fp()) }, "transpose")?;
        n.gate_add(self.x.ptr(), Dt::F32, s, d, self.enc_pos.ptr(), Dt::F32, none(), none())?;
        let hd = d / c.enc_heads;
        for l in &self.enc {
            self.ln(self.x.ptr(), s, &l.ln1, self.h.ptr())?;
            self.linear(self.h.ptr(), s, &l.q, self.q.ptr())?;
            self.linear(self.h.ptr(), s, &l.k, self.kk.ptr())?;
            self.linear(self.h.ptr(), s, &l.v, self.v.ptr())?;
            n.attention(self.q.ptr(), self.kk.ptr(), self.v.ptr(), Dt::F16, s, c.enc_heads, hd, d, self.att.ptr(), Dt::F16)?;
            self.linear(self.att.ptr(), s, &l.o, self.y.ptr())?;
            self.add(self.x.ptr(), self.y.ptr(), s)?;
            self.ln(self.x.ptr(), s, &l.ln2, self.h.ptr())?;
            self.linear(self.h.ptr(), s, &l.fc1, self.ff.ptr())?;
            n.gelu(self.ff.ptr(), Dt::F16, s * l.fc1.n, false)?;
            self.linear(self.ff.ptr(), s, &l.fc2, self.y.ptr())?;
            self.add(self.x.ptr(), self.y.ptr(), s)?;
        }
        self.ln(self.x.ptr(), s, &self.enc_ln, self.enc_out.ptr())?;
        for (i, l) in self.dec.iter().enumerate() {
            self.linear(self.enc_out.ptr(), s, &l.xk, self.xk[i].ptr())?;
            self.linear(self.enc_out.ptr(), s, &l.xv, self.xv[i].ptr())?;
        }
        Ok(())
    }

    /// The encoder's output of the last window, as floats [audio_ctx, d] (the check)
    pub fn encoder_output(&self) -> Result<Vec<f32>> {
        let mut b = vec![0u8; self.cfg.audio_ctx * self.cfg.d * 2];
        self.enc_out.read(0, &mut b)?;
        Ok(b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
    }

    /// A decoder layer's cross keys of the window, as floats [audio_ctx, d] (the word alignment)
    pub fn cross_keys(&self, layer: usize) -> Result<Vec<f32>> {
        let mut b = vec![0u8; self.cfg.audio_ctx * self.cfg.d * 2];
        self.xk[layer].read(0, &mut b)?;
        Ok(b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
    }

    /// The decoder over `tokens` at positions [pos, pos + tokens.len()) (the cache holds the ones before): the logits
    /// of rows [from, tokens.len()) of these, [rows, vocab]; `taps` collects the cross queries of its layers
    pub fn decode(&self, tokens: &[u32], pos: usize, from: usize, taps: Option<&mut Taps>) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (d, m) = (c.d, tokens.len());
        if m == 0 || pos + m > c.text_ctx {
            return Err(Error(format!("decoder positions {pos}..{} past its {}", pos + m, c.text_ctx)));
        }
        if m - from > LOGIT_ROWS {
            return Err(Error(format!("{} rows of logits", m - from)));
        }
        let mut x = vec![0f32; m * d];
        for (i, t) in tokens.iter().enumerate() {
            let t = *t as usize;
            if t >= c.vocab {
                return Err(Error(format!("token {t} past the vocabulary")));
            }
            for j in 0..d {
                x[i * d + j] = f16_to_f32(self.emb_host[t * d + j]) + self.dec_pos_host[(pos + i) * d + j];
            }
        }
        // SAFETY: plain floats as bytes.
        self.dx.write(0, unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len() * 4) })?;
        let n = &self.nsd;
        let hd = d / c.dec_heads;
        let row = d * 2;
        let mut taps = taps;
        for (i, l) in self.dec.iter().enumerate() {
            self.ln(self.dx.ptr(), m, &l.ln1, self.dh.ptr())?;
            self.linear(self.dh.ptr(), m, &l.q, at(&self.qbuf, pos * row))?;
            self.linear(self.dh.ptr(), m, &l.k, at(&self.kc[i], pos * row))?;
            self.linear(self.dh.ptr(), m, &l.v, at(&self.vc[i], pos * row))?;
            n.attention_causal_rows(self.qbuf.ptr(), self.kc[i].ptr(), self.vc[i].ptr(), Dt::F16, pos, m, c.dec_heads, c.dec_heads, hd, d, d, self.datt.ptr())?;
            self.linear(self.datt.ptr(), m, &l.o, self.dy.ptr())?;
            self.add(self.dx.ptr(), self.dy.ptr(), m)?;
            self.ln(self.dx.ptr(), m, &l.lnx, self.dh.ptr())?;
            self.linear(self.dh.ptr(), m, &l.xq, self.dq.ptr())?;
            if let Some(t) = taps.as_deref_mut() {
                if t.layers.contains(&i) {
                    let mut b = vec![0u8; m * row];
                    self.dq.read(0, &mut b)?;
                    t.q.push(b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect());
                }
            }
            n.attention_qk(self.dq.ptr(), m, d, self.xk[i].ptr(), self.xv[i].ptr(), c.audio_ctx, d, Dt::F16, c.dec_heads, hd, self.datt.ptr(), Dt::F16)?;
            self.linear(self.datt.ptr(), m, &l.xo, self.dy.ptr())?;
            self.add(self.dx.ptr(), self.dy.ptr(), m)?;
            self.ln(self.dx.ptr(), m, &l.ln2, self.dh.ptr())?;
            self.linear(self.dh.ptr(), m, &l.fc1, self.dff.ptr())?;
            n.gelu(self.dff.ptr(), Dt::F16, m * l.fc1.n, false)?;
            self.linear(self.dff.ptr(), m, &l.fc2, self.dy.ptr())?;
            self.add(self.dx.ptr(), self.dy.ptr(), m)?;
        }
        let r = m - from;
        let x0 = at(&self.dx, from * d * 4);
        n.layer_norm(x0, Dt::F32, r, d, self.dec_ln.w.ptr(), self.dec_ln.b.ptr(), EPS, self.dh.ptr(), Dt::F16)?;
        n.linear(self.dh.ptr(), Dt::F16, r, d, self.emb.ptr(), c.vocab, none(), self.logits.ptr(), Dt::F32)?;
        let mut out = vec![0f32; r * c.vocab];
        // SAFETY: plain floats as bytes.
        self.logits.read(0, unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) })?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_to_nearest_even() {
        for x in [0.0f32, 1.0, -2.5, 65504.0, 1e-5, 6.1e-5, 3.14159, -0.000_123, 1e-8] {
            let h = f32_to_f16(x);
            let back = f16_to_f32(h);
            let tol = (x.abs() * 1e-3).max(6e-8);
            assert!((back - x).abs() <= tol, "{x} -> {h:#06x} -> {back}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65520.0), 0x7c00, "past half's range: infinity");
        assert_eq!(f32_to_f16(f32::NAN) & 0x7c00, 0x7c00);
    }

    #[test]
    fn large_v3_turbos_config() {
        let Ok(p) = std::env::var("NS_TEST_WHISPER_CONFIG") else { return };
        let c = Config::read(Path::new(&p)).unwrap();
        assert_eq!(c, Config { d: 1280, enc_layers: 32, dec_layers: 4, enc_heads: 20, dec_heads: 20, enc_ffn: 5120, dec_ffn: 5120, n_mels: 128,
                               audio_ctx: 1500, text_ctx: 448, vocab: 51866 });
        let gib = c.weight_bytes() as f64 / (1u64 << 30) as f64;
        assert!((1.4..1.7).contains(&gib), "{gib}");
    }
}
