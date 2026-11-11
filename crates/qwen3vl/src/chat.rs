//! Qwen3-VL's language model writing text: the same file and layers as the text encoder (`TextEncoder`), plus the
//! final norm, the output head and a key / value cache, so a picture can be described by the image model's own text
//! encoder (image.describe's `describer: native`). Nothing here changes the encoder: `Writer` reads its weights.
//!
//! The head is the file's `lm_head` (Qwen-Image 2.1's 8B: int8 ConvRot like every matrix) or, when the file has none,
//! the token embedding itself (Qwen3-VL-4B ties them: P-H3-Image's encoder file). Both are int8 rows with a scale a
//! row and ConvRot's rotation of the input, so the shared int8 linear computes the logits either way:
//!
//! ```text
//!   prompt:  the encoder's pass over every token (pictures, M-RoPE, deepstack as encode_taps), each layer's q | k | v
//!            rows written into that layer's cache; the last row through the final norm and the head
//!   a token: its embedding row, each layer on one row - q | k | v into the cache's next row, RMS + RoPE at its text
//!            position (the three M-RoPE axes equal), causal attention over the cache - the norm, the head, a pick
//! ```
//!
//! Activations and the cache are float32 (as the encoder's): a layer's cache row is q | k | v (6,144 floats for both
//! sizes), ~0.9 MB a token over 36 layers. The products are the shared kernels' (int8 linear, RMS + RoPE, causal
//! attention by rows); nothing is tuned for writing - this is a describer of a few hundred tokens, not a chat engine.

use std::time::Instant;

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::{ge, upload_f32, vision, TextEncoder, EPS, GROUP, HEAD};

/// The output head on the card: int8 rows [vocab, hidden] and a scale a row
pub struct Writer {
    w: DevBuf,
    s: DevBuf,
    norm: DevBuf,
    pub vocab: usize,
    /// whether the head is the token embedding (the file has no lm_head)
    pub tied: bool,
}

/// How a writing ended
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// an end-of-turn token
    Stop,
    /// `max_tokens` written
    Length,
    /// the caller's `each` said stop (a cancel)
    Stopped,
}

impl Finish {
    pub fn name(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::Stopped => "cancelled",
        }
    }
}

/// What a writing came to
#[derive(Clone, Debug)]
pub struct Written {
    pub ids: Vec<u32>,
    pub text: String,
    pub prompt_tokens: usize,
    /// the prompt's pass (the pictures are seen before: the caller's)
    pub prompt_seconds: f64,
    /// the tokens written after it
    pub write_seconds: f64,
    pub finish: Finish,
}

/// How to write: at most `max_tokens`; `temperature` 0 greedy, else sampled over the tokens within `top_p` with `seed`;
/// `stop` the end-of-turn ids
#[derive(Clone, Debug)]
pub struct WriteOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub stop: Vec<u32>,
}

/// One written token's attention, as `Writer::write` asks an engine's own kernel for it: q [hq][d] at `q`; the cache's
/// keys and values `n` rows of stride `kvs` floats (head g at g * d) from `k` / `v`; out [hq][d]. All float32 on the card
pub struct OneRow {
    pub q: *const std::ffi::c_void,
    pub k: *const std::ffi::c_void,
    pub v: *const std::ffi::c_void,
    pub n: usize,
    pub kvs: usize,
    pub hq: usize,
    pub hkv: usize,
    pub d: usize,
    pub out: *mut std::ffi::c_void,
}

/// An engine's one-row attention (`OneRow`); none: the shared causal attention by rows
pub type Attend<'a> = Option<&'a dyn Fn(&OneRow) -> Result<()>>;

/// splitmix64: a sampled token's randomness, the same for the same seed
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The token to write from the logits: the largest at temperature 0, else one drawn from the softmax at `temperature`
/// over the most likely tokens whose probabilities sum to `top_p`
fn pick(logits: &[f32], temperature: f32, top_p: f32, rng: &mut Rng) -> u32 {
    let best = logits.iter().enumerate().fold(0usize, |b, (i, v)| if *v > logits[b] { i } else { b });
    if temperature <= 0.0 {
        return best as u32;
    }
    let mx = logits[best];
    let mut p: Vec<(u32, f64)> = logits.iter().enumerate().map(|(i, v)| (i as u32, (((v - mx) / temperature) as f64).exp())).collect();
    p.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = p.iter().map(|x| x.1).sum();
    let mut keep = 0;
    let mut acc = 0.0;
    for (i, x) in p.iter().enumerate() {
        acc += x.1 / total;
        keep = i + 1;
        if acc >= top_p as f64 {
            break;
        }
    }
    let p = &p[..keep];
    let mut r = rng.next() * p.iter().map(|x| x.1).sum::<f64>();
    for x in p {
        r -= x.1;
        if r <= 0.0 {
            return x.0;
        }
    }
    p[p.len() - 1].0
}

/// (cos, sin) per pair (i, 64 + i) of each of `pos`'s tokens: Qwen3-VL's interleaved M-RoPE (24 / 20 / 20 pairs: t,
/// then h and w on every third pair from the second) - the encoder's
fn rotations(pos: &[[i64; 3]], theta: f64) -> Vec<f32> {
    let mut cs = vec![0f32; pos.len() * HEAD];
    for (t, p) in pos.iter().enumerate() {
        for i in 0..HEAD / 2 {
            let axis = if i < 60 && i % 3 == 1 { 1 } else if i < 60 && i % 3 == 2 { 2 } else { 0 };
            let a = p[axis] as f64 / theta.powf(2.0 * i as f64 / HEAD as f64);
            cs[t * HEAD + 2 * i] = a.cos() as f32;
            cs[t * HEAD + 2 * i + 1] = a.sin() as f32;
        }
    }
    cs
}

/// The bytes of `ids` decoded so far that make whole UTF-8: what may be shown (a character split over two tokens waits)
fn whole_utf8(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        Err(_) => bytes.len(),
    }
}

impl Writer {
    /// The head and the final norm of `enc`'s file (lm_head, else the tied token embedding); refused for a file
    /// without a final norm (a text encoder cut short: it cannot write)
    pub fn new(enc: &TextEncoder) -> Result<Writer> {
        let st = &enc.file;
        let norm_name = format!("{}norm.weight", enc.prefix);
        if st.tensor(&norm_name).is_none() {
            return Err(Error("the file has no final norm (a text encoder cut short): it cannot write text".into()));
        }
        let norm = upload_f32(&enc.gpu, &st.f32(&norm_name).map_err(ge)?)?;
        let (name, tied) = if st.tensor("lm_head.weight").is_some() { ("lm_head".to_string(), false) } else { (format!("{}embed_tokens", enc.prefix), true) };
        let t = st.need(&format!("{name}.weight")).map_err(ge)?;
        if t.dtype != "I8" || t.shape.get(1).copied() != Some(enc.hidden as u64) {
            return Err(Error(format!("{name}: {} {:?}, not int8 rows of {}", t.dtype, t.shape, enc.hidden)));
        }
        let raw = st.tensor(&format!("{name}.comfy_quant")).and_then(|q| st.read(q).ok());
        let meta: serde_json::Value = raw.and_then(|r| serde_json::from_slice(&r).ok()).unwrap_or_default();
        if meta["convrot"] != true || meta["convrot_groupsize"] != GROUP {
            return Err(Error(format!("{name}: {meta}, not int8 ConvRot in groups of {GROUP}")));
        }
        let vocab = t.shape[0] as usize;
        let bytes = st.read(t).map_err(ge)?;
        let w = DevBuf::new(&enc.gpu, bytes.len())?;
        w.write(0, &bytes)?;
        let s = upload_f32(&enc.gpu, &st.f32(&format!("{name}.weight_scale")).map_err(ge)?)?;
        Ok(Writer { w, s, norm, vocab, tied })
    }

    /// The prompt `ids` (pictures `seen` at their `<|image_pad|>` runs) read, then tokens written until a stop id,
    /// `max_tokens`, or `each` (given the text so far as it grows) answering false
    #[allow(clippy::too_many_arguments)]
    pub fn write(&self, enc: &TextEncoder, nsd: &Nsd, ids: &[u32], seen: &[&vision::Seen], o: &WriteOptions, attend: Attend,
                 each: &mut dyn FnMut(&str) -> bool) -> Result<Written> {
        let t0 = Instant::now();
        let s = ids.len();
        if s == 0 {
            return Err(Error("an empty prompt".into()));
        }
        let (d, h, kv, f) = (enc.hidden, enc.heads, enc.kv_heads, enc.ffn);
        let qkv_w = (h + 2 * kv) * HEAD;
        let gpu = &enc.gpu;
        let cap = s + o.max_tokens.max(1);
        let caches: Vec<DevBuf> = (0..enc.layers.len()).map(|_| DevBuf::f32(gpu, cap * qkv_w)).collect::<Result<_>>()?;
        let f32p = |b: &DevBuf, at: usize| -> *mut std::ffi::c_void { b.fp().wrapping_add(at).cast() };
        // the prompt: the encoder's pass, q | k | v into the caches
        let pad = enc.tok.id("<|image_pad|>").unwrap_or(u32::MAX);
        let mut runs = Vec::new();
        let mut i = 0;
        while i < s {
            if ids[i] == pad {
                let st = i;
                while i < s && ids[i] == pad {
                    i += 1;
                }
                runs.push((st, i - st));
            } else {
                i += 1;
            }
        }
        if runs.len() != seen.len() || runs.iter().zip(seen).any(|((_, len), v)| *len != v.n) {
            return Err(Error(format!("the prompt holds {} picture runs; {} pictures were seen", runs.len(), seen.len())));
        }
        let x = upload_f32(gpu, &enc.embed(ids)?)?;
        for ((st, len), v) in runs.iter().zip(seen) {
            x.copy_within(st * d * 4, &v.tokens, 0, len * d * 4)?;
        }
        let pos = if seen.is_empty() {
            (0..s as i64).map(|p| [p, p, p]).collect()
        } else {
            TextEncoder::positions(ids, pad, &seen.iter().map(|v| v.grid).collect::<Vec<_>>())?
        };
        // the next text position: past the largest so far
        let mut next = pos.iter().map(|p| p[0].max(p[1]).max(p[2])).max().unwrap_or(-1) + 1;
        {
            let hb = DevBuf::f32(gpu, s * d)?;
            let att = DevBuf::f32(gpu, s * h * HEAD)?;
            let ob = DevBuf::f32(gpu, s * d)?;
            let gu = DevBuf::f32(gpu, s * 2 * f)?;
            let act = DevBuf::f32(gpu, s * f)?;
            let cs = upload_f32(gpu, &rotations(&pos, enc.theta))?;
            for (li, (l, c)) in enc.layers.iter().zip(&caches).enumerate() {
                nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, l.in_norm.ptr(), EPS, none(), none(), none(), hb.ptr(), Dt::F32)?;
                nsd.int8_linear(hb.ptr(), Dt::F32, s, d, l.qkv.ptr(), qkv_w, l.qkv_s.ptr(), qkv_w, none(), c.ptr(), Dt::F32, GROUP)?;
                nsd.rms_rope(c.ptr(), Dt::F32, s, h, HEAD, qkv_w, l.q_norm.ptr(), EPS, cs.ptr(), HEAD)?;
                nsd.rms_rope(f32p(c, h * HEAD), Dt::F32, s, kv, HEAD, qkv_w, l.k_norm.ptr(), EPS, cs.ptr(), HEAD)?;
                nsd.attention_causal(c.ptr(), f32p(c, h * HEAD), f32p(c, (h + kv) * HEAD), Dt::F32, s, h, kv, HEAD, qkv_w, qkv_w, att.ptr())?;
                nsd.int8_linear(att.ptr(), Dt::F32, s, h * HEAD, l.o.ptr(), d, l.o_s.ptr(), d, none(), ob.ptr(), Dt::F32, GROUP)?;
                nsd.gate_add(x.ptr(), Dt::F32, s, d, ob.ptr(), Dt::F32, none(), none())?;
                nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, l.post_norm.ptr(), EPS, none(), none(), none(), hb.ptr(), Dt::F32)?;
                nsd.int8_linear(hb.ptr(), Dt::F32, s, d, l.gu.ptr(), 2 * f, l.gu_s.ptr(), 2 * f, none(), gu.ptr(), Dt::F32, GROUP)?;
                nsd.swiglu(gu.ptr(), Dt::F32, s, f, act.ptr(), Dt::F32)?;
                nsd.int8_linear(act.ptr(), Dt::F32, s, f, l.down.ptr(), d, l.down_s.ptr(), d, none(), ob.ptr(), Dt::F32, GROUP)?;
                nsd.gate_add(x.ptr(), Dt::F32, s, d, ob.ptr(), Dt::F32, none(), none())?;
                for ((st, len), v) in runs.iter().zip(seen) {
                    if let Some(ds) = v.deep.get(li) {
                        nsd.gate_add(f32p(&x, st * d), Dt::F32, *len, d, ds.ptr(), Dt::F32, none(), none())?;
                    }
                }
            }
        }
        // the work of one row
        let x1 = DevBuf::f32(gpu, d)?;
        let hb1 = DevBuf::f32(gpu, d)?;
        let att1 = DevBuf::f32(gpu, h * HEAD)?;
        let o1 = DevBuf::f32(gpu, d)?;
        let gu1 = DevBuf::f32(gpu, 2 * f)?;
        let act1 = DevBuf::f32(gpu, f)?;
        let logits = DevBuf::f32(gpu, self.vocab)?;
        x1.copy_within(0, &x, (s - 1) * d * 4, d * 4)?;
        drop(x);
        let head = |x1: &DevBuf| -> Result<Vec<f32>> {
            nsd.rms_norm_mod(x1.ptr(), Dt::F32, 1, d, self.norm.ptr(), EPS, none(), none(), none(), hb1.ptr(), Dt::F32)?;
            nsd.int8_linear(hb1.ptr(), Dt::F32, 1, d, self.w.ptr(), self.vocab, self.s.ptr(), self.vocab, none(), logits.ptr(), Dt::F32, GROUP)?;
            nsd.wait()?;
            logits.to_f32()
        };
        let mut lg = head(&x1)?;
        let prompt_seconds = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut rng = Rng(o.seed);
        let mut out: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        let mut finish = Finish::Length;
        let mut row = s;
        while out.len() < o.max_tokens {
            let t = pick(&lg, o.temperature, o.top_p, &mut rng);
            if o.stop.contains(&t) {
                finish = Finish::Stop;
                break;
            }
            out.push(t);
            let bytes = enc.tok.decode_bytes(&out);
            let ok = whole_utf8(&bytes);
            if ok > shown {
                shown = ok;
                if !each(&String::from_utf8_lossy(&bytes[..ok])) {
                    finish = Finish::Stopped;
                    break;
                }
            }
            if out.len() >= o.max_tokens {
                break;
            }
            // the token through every layer at `row`, its text position `next`
            x1.write(0, &enc.embed(&[t])?.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            let cs1 = upload_f32(gpu, &rotations(&[[next; 3]], enc.theta))?;
            for (l, c) in enc.layers.iter().zip(&caches) {
                let r = f32p(c, row * qkv_w);
                nsd.rms_norm_mod(x1.ptr(), Dt::F32, 1, d, l.in_norm.ptr(), EPS, none(), none(), none(), hb1.ptr(), Dt::F32)?;
                nsd.int8_linear(hb1.ptr(), Dt::F32, 1, d, l.qkv.ptr(), qkv_w, l.qkv_s.ptr(), qkv_w, none(), r, Dt::F32, GROUP)?;
                nsd.rms_rope(r, Dt::F32, 1, h, HEAD, qkv_w, l.q_norm.ptr(), EPS, cs1.ptr(), HEAD)?;
                nsd.rms_rope(f32p(c, row * qkv_w + h * HEAD), Dt::F32, 1, kv, HEAD, qkv_w, l.k_norm.ptr(), EPS, cs1.ptr(), HEAD)?;
                match attend {
                    Some(a) => a(&OneRow { q: r, k: f32p(c, h * HEAD), v: f32p(c, (h + kv) * HEAD), n: row + 1, kvs: qkv_w, hq: h, hkv: kv, d: HEAD, out: att1.ptr() })?,
                    None => nsd.attention_causal_rows(c.ptr(), f32p(c, h * HEAD), f32p(c, (h + kv) * HEAD), Dt::F32, row, 1, h, kv, HEAD, qkv_w, qkv_w, att1.ptr())?,
                }
                nsd.int8_linear(att1.ptr(), Dt::F32, 1, h * HEAD, l.o.ptr(), d, l.o_s.ptr(), d, none(), o1.ptr(), Dt::F32, GROUP)?;
                nsd.gate_add(x1.ptr(), Dt::F32, 1, d, o1.ptr(), Dt::F32, none(), none())?;
                nsd.rms_norm_mod(x1.ptr(), Dt::F32, 1, d, l.post_norm.ptr(), EPS, none(), none(), none(), hb1.ptr(), Dt::F32)?;
                nsd.int8_linear(hb1.ptr(), Dt::F32, 1, d, l.gu.ptr(), 2 * f, l.gu_s.ptr(), 2 * f, none(), gu1.ptr(), Dt::F32, GROUP)?;
                nsd.swiglu(gu1.ptr(), Dt::F32, 1, f, act1.ptr(), Dt::F32)?;
                nsd.int8_linear(act1.ptr(), Dt::F32, 1, f, l.down.ptr(), d, l.down_s.ptr(), d, none(), o1.ptr(), Dt::F32, GROUP)?;
                nsd.gate_add(x1.ptr(), Dt::F32, 1, d, o1.ptr(), Dt::F32, none(), none())?;
            }
            lg = head(&x1)?;
            row += 1;
            next += 1;
        }
        let text = enc.tok.decode(&out);
        Ok(Written { ids: out, text, prompt_tokens: s, prompt_seconds, write_seconds: t1.elapsed().as_secs_f64(), finish })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pick_is_greedy_at_zero_and_seeded_above() {
        let lg = [0.1f32, 2.0, 1.9, -3.0];
        assert_eq!(pick(&lg, 0.0, 1.0, &mut Rng(1)), 1);
        // top_p small: only the most likely is left
        assert_eq!(pick(&lg, 1.0, 0.01, &mut Rng(7)), 1);
        // the same seed, the same draws
        let a: Vec<u32> = { let mut r = Rng(42); (0..20).map(|_| pick(&lg, 1.0, 0.95, &mut r)).collect() };
        let b: Vec<u32> = { let mut r = Rng(42); (0..20).map(|_| pick(&lg, 1.0, 0.95, &mut r)).collect() };
        assert_eq!(a, b);
        assert!(a.iter().all(|t| *t != 3), "the unlikely tail is cut by top_p");
        assert!(a.contains(&1) && a.contains(&2));
    }

    #[test]
    fn text_positions_rotate_like_the_encoders() {
        // the three axes equal: every pair turned by the position (plain RoPE)
        let cs = rotations(&[[5, 5, 5]], 5_000_000.0);
        for i in 0..HEAD / 2 {
            let a = 5.0f64 / 5_000_000f64.powf(2.0 * i as f64 / HEAD as f64);
            assert!((cs[2 * i] as f64 - a.cos()).abs() < 1e-6 && (cs[2 * i + 1] as f64 - a.sin()).abs() < 1e-6);
        }
        // a picture's token: pair 1 turns with its row, pair 2 with its column, pair 0 with time
        let cs = rotations(&[[3, 7, 11]], 1.0);
        assert!((cs[0] - 3f32.cos()).abs() < 1e-6 && (cs[2] - 7f32.cos()).abs() < 1e-6 && (cs[4] - 11f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn a_split_character_waits_for_its_next_token() {
        let e = "é".as_bytes();
        assert_eq!(whole_utf8(&[b'a', e[0]]), 1);
        assert_eq!(whole_utf8(&[b'a', e[0], e[1]]), 3);
        assert_eq!(whole_utf8(b"abc"), 3);
        assert_eq!(Finish::Length.name(), "length");
    }
}
