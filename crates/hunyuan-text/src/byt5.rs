//! The glyph encoder: Glyph-SDXL-v2's byT5-small (transformers' T5EncoderModel: 12 layers of 1472, 6 heads of 64,
//! relative position buckets, gated-GELU 3584) for the text a prompt quotes: the quoted pieces (`Quotes`), each
//! formatted `Text "{piece}". ` (MultilingualPromptFormat with no style), byte-tokenized (each byte + 3, then </s> = 1),
//! encoded with the padding masked. Only the valid tokens' rows are kept: the DiT masks the rest.
//!
//! The 1126 color and font tokens the glyph tokenizer adds (`<color-N>`, `<xx-font-N>`) are not read here: such a tag
//! in quoted text is encoded as its bytes, as transformers' plain ByT5Tokenizer would.
//!
//! float32 throughout (T5 overflows half); the weights are F16 in the file.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::Gguf;

use crate::{floats, ge, host, stacked, tensor, Kern};

pub const DIM: usize = 1472;
const HEADS: usize = 6;
const HEAD: usize = 64;
const FF: usize = 3584;
const EPS: f32 = 1e-6;
const BUCKETS: usize = 32;
const MAX_DISTANCE: f64 = 128.0;

/// How a pipeline finds the quoted text
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quotes {
    /// HunyuanImage 2.1's _encode_glyph: "...", then ‘...’, then “...”, each kind in turn
    ByKind,
    /// HunyuanVideo-1.5's _extract_glyph_texts: `\"(.*?)\"|“(.*?)”` left to right, deduplicated when several
    InOrder,
}

/// One `re.findall(r'<open>(.*?)<close>')` match from byte `from` of `prompt`: (its text, where the match ends), or
/// None. Non-greedy, `.` stops at a newline (the match then starts again after that opening mark).
fn next_quote(prompt: &str, from: usize, open: char, close: char) -> Option<(usize, String, usize)> {
    let mut at = from;
    loop {
        let a = at + prompt[at..].find(open)?;
        let after = a + open.len_utf8();
        let b = after + prompt[after..].find(close)?;
        let inner = &prompt[after..b];
        if inner.contains('\n') {
            at = after;
            continue;
        }
        return Some((a, inner.to_string(), b + close.len_utf8()));
    }
}

/// The quoted pieces of a prompt as the pipeline formats them, or None when it quotes nothing
pub fn quoted(prompt: &str, how: Quotes) -> Option<String> {
    let mut texts = Vec::new();
    match how {
        Quotes::ByKind => {
            for (open, close) in [('"', '"'), ('\u{2018}', '\u{2019}'), ('\u{201c}', '\u{201d}')] {
                let mut at = 0;
                while let Some((_, t, end)) = next_quote(prompt, at, open, close) {
                    texts.push(t);
                    at = end;
                }
            }
        }
        Quotes::InOrder => {
            // the alternation tries both at each position: the earliest match wins (at one position the first
            // alternative)
            let mut at = 0;
            loop {
                let a = next_quote(prompt, at, '"', '"');
                let b = next_quote(prompt, at, '\u{201c}', '\u{201d}');
                let m = match (a, b) {
                    (Some(a), Some(b)) => if b.0 < a.0 { b } else { a },
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => break,
                };
                texts.push(m.1);
                at = m.2;
            }
            if texts.len() > 1 {
                let mut seen = std::collections::HashSet::new();
                texts.retain(|t| seen.insert(t.clone()));
            }
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.iter().map(|t| format!("Text \"{t}\". ")).collect())
    }
}

/// ByT5's ids: each UTF-8 byte + 3, then </s> (1), truncated to `max` with the </s> kept (HunyuanImage 128, HunyuanVideo
/// 256)
pub fn ids(text: &str, max: usize) -> Vec<u32> {
    let mut v: Vec<u32> = text.bytes().map(|b| b as u32 + 3).collect();
    v.truncate(max.max(1) - 1);
    v.push(1);
    v
}

/// T5's bidirectional relative position bucket for key position minus query position
pub fn bucket(rel: i64) -> usize {
    let nb = (BUCKETS / 2) as i64;
    let mut ret = if rel > 0 { nb } else { 0 };
    let n = rel.abs();
    let max_exact = nb / 2;
    if n < max_exact {
        ret += n;
    } else {
        // in float32, as transformers computes it on a tensor
        let large = max_exact + ((n as f32 / max_exact as f32).ln() / (MAX_DISTANCE as f32 / max_exact as f32).ln() * (nb - max_exact) as f32) as i64;
        ret += large.min(nb - 1);
    }
    ret as usize
}

/// A float32 matrix [n, k] on the card
struct Lin {
    w: DevBuf,
    n: usize,
    k: usize,
}

impl Lin {
    fn load(f: &Gguf, nsd: &Nsd, names: &[String]) -> Result<Lin> {
        let (raw, n, k, ty) = stacked(f, names)?;
        Ok(Lin { w: DevBuf::from_f32(&nsd.gpu, &floats(&names[0], ty, &raw)?)?, n, k })
    }
}

struct Layer {
    ln1: DevBuf,
    qkv: Lin,
    o: Lin,
    ln2: DevBuf,
    /// wi_0 | wi_1
    wi: Lin,
    wo: Lin,
}

pub struct Byt5 {
    k: Arc<dyn Kern>,
    layers: Vec<Layer>,
    final_ln: DevBuf,
    /// shared.weight [vocab, 1472], on the host
    embd: Vec<f32>,
    vocab: usize,
    /// relative_attention_bias [32 buckets, 6 heads]
    rel: Vec<f32>,
    pub bytes: usize,
}

impl Byt5 {
    pub fn load(file: &Path, nsd: &Nsd, k: Arc<dyn Kern>, log: &mut dyn FnMut(String)) -> Result<Byt5> {
        let t0 = std::time::Instant::now();
        let gpu = &nsd.gpu;
        let f = Gguf::open(file).map_err(ge)?;
        let et = tensor(&f, "shared.weight")?;
        if et.shape.get(1) != Some(&(DIM as u64)) {
            return Err(Error(format!("{}: shared.weight {:?}; byT5-small's [vocab, 1472] here", file.display(), et.shape)));
        }
        let vocab = et.shape[0] as usize;
        let embd = host(&f, "shared.weight")?;
        let rel = host(&f, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;
        let mut layers = Vec::new();
        for l in 0.. {
            let p = format!("encoder.block.{l}.layer");
            if f.tensor(&format!("{p}.0.layer_norm.weight")).is_none() {
                break;
            }
            layers.push(Layer {
                ln1: DevBuf::from_f32(gpu, &host(&f, &format!("{p}.0.layer_norm.weight"))?)?,
                qkv: Lin::load(&f, nsd, &[format!("{p}.0.SelfAttention.q.weight"), format!("{p}.0.SelfAttention.k.weight"),
                                          format!("{p}.0.SelfAttention.v.weight")])?,
                o: Lin::load(&f, nsd, &[format!("{p}.0.SelfAttention.o.weight")])?,
                ln2: DevBuf::from_f32(gpu, &host(&f, &format!("{p}.1.layer_norm.weight"))?)?,
                wi: Lin::load(&f, nsd, &[format!("{p}.1.DenseReluDense.wi_0.weight"), format!("{p}.1.DenseReluDense.wi_1.weight")])?,
                wo: Lin::load(&f, nsd, &[format!("{p}.1.DenseReluDense.wo.weight")])?,
            });
        }
        if layers.is_empty() {
            return Err(Error(format!("{}: no encoder blocks (byT5's GGUF?)", file.display())));
        }
        let bytes = layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.wi, &l.wo]).map(|m| m.w.len).sum::<usize>();
        log(format!("glyph encoder: byT5-small, {} layers (float32, {:.2} GiB) in {:.1} s", layers.len(), bytes as f64 / (1u64 << 30) as f64,
                    t0.elapsed().as_secs_f64()));
        Ok(Byt5 { k, layers, final_ln: DevBuf::from_f32(gpu, &host(&f, "encoder.final_layer_norm.weight")?)?, embd, vocab, rel, bytes })
    }

    /// The encoder's output for `ids`: float32 [L, 1472]
    pub fn encode(&self, nsd: &Nsd, ids: &[u32]) -> Result<Vec<f32>> {
        let gpu = &nsd.gpu;
        let l = ids.len();
        let k = &*self.k;
        let mut e = Vec::with_capacity(l * DIM);
        for id in ids {
            let i = *id as usize;
            if i >= self.vocab {
                return Err(Error(format!("byT5 token {i} is past the vocabulary")));
            }
            e.extend_from_slice(&self.embd[i * DIM..(i + 1) * DIM]);
        }
        // the position bias [heads, L, L] (block 0's, shared by every block)
        let mut bias = vec![0f32; HEADS * l * l];
        for h in 0..HEADS {
            for i in 0..l {
                for j in 0..l {
                    bias[(h * l + i) * l + j] = self.rel[bucket(j as i64 - i as i64) * HEADS + h];
                }
            }
        }
        let bias = DevBuf::from_f32(gpu, &bias)?;
        let x = DevBuf::from_f32(gpu, &e)?;
        let h = DevBuf::f32(gpu, l * DIM)?;
        let inner = HEADS * HEAD;
        let qkv = DevBuf::f32(gpu, l * 3 * inner)?;
        let att = DevBuf::f32(gpu, l * inner)?;
        let o = DevBuf::f32(gpu, l * DIM)?;
        let wide = DevBuf::f32(gpu, l * 2 * FF)?;
        let act = DevBuf::f32(gpu, l * FF)?;
        let lin = |m: &Lin, x: &DevBuf, out: &DevBuf| -> Result<()> { nsd.linear(x.ptr(), Dt::F32, l, m.k, m.w.ptr(), m.n, none(), out.ptr(), Dt::F32) };
        for ly in &self.layers {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, DIM, ly.ln1.ptr(), EPS, none(), none(), none(), h.ptr(), Dt::F32)?;
            lin(&ly.qkv, &h, &qkv)?;
            // T5: no 1 / sqrt(d) scaling; the position bias added to the scores
            k.attn_small(qkv.fp(), 3 * inner, qkv.fp().wrapping_add(inner), qkv.fp().wrapping_add(2 * inner), 3 * inner, l, HEADS, HEADS, HEAD, 1.0,
                         bias.fp(), false, att.fp())?;
            lin(&ly.o, &att, &o)?;
            nsd.gate_add(x.ptr(), Dt::F32, l, DIM, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, DIM, ly.ln2.ptr(), EPS, none(), none(), none(), h.ptr(), Dt::F32)?;
            lin(&ly.wi, &h, &wide)?;
            k.geglu_f32(wide.ptr(), l, FF, act.ptr())?;
            lin(&ly.wo, &act, &o)?;
            nsd.gate_add(x.ptr(), Dt::F32, l, DIM, o.ptr(), Dt::F32, none(), none())?;
        }
        nsd.rms_norm_mod(x.ptr(), Dt::F32, l, DIM, self.final_ln.ptr(), EPS, none(), none(), none(), h.ptr(), Dt::F32)?;
        nsd.wait()?;
        h.to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::Quotes::{ByKind, InOrder};

    #[test]
    fn quoted_text_is_found_and_formatted_as_each_pipeline_does() {
        assert_eq!(super::quoted("a sign that says \"HELLO\" and ‘hi’ or “你好”", ByKind).as_deref(), Some("Text \"HELLO\". Text \"hi\". Text \"你好\". "));
        assert_eq!(super::quoted("no quotes here", ByKind), None);
        assert_eq!(super::quoted("an \"unclosed quote", ByKind), None);
        assert_eq!(super::quoted("\"a\" then \"b\"", ByKind).as_deref(), Some("Text \"a\". Text \"b\". "));
        // HunyuanVideo: in order of appearance, ‘’ not read, duplicates dropped
        assert_eq!(super::quoted("“x” then \"a\" and ‘hi’ and \"a\"", InOrder).as_deref(), Some("Text \"x\". Text \"a\". "));
        assert_eq!(super::quoted("\"a\" \"a\"", InOrder).as_deref(), Some("Text \"a\". "));
        assert_eq!(super::quoted("\"only\"", InOrder).as_deref(), Some("Text \"only\". "));
    }

    #[test]
    fn byt5_ids_are_bytes_plus_three_and_end_with_eos() {
        // transformers' ByT5Tokenizer()("Text \"Hi\". ").input_ids
        assert_eq!(super::ids("Text \"Hi\". ", 128), vec![87, 104, 123, 119, 35, 37, 75, 108, 37, 49, 35, 1]);
        assert_eq!(super::ids(&"x".repeat(300), 128).len(), 128);
        assert_eq!(super::ids(&"x".repeat(300), 256).len(), 256);
    }

    #[test]
    fn t5_buckets_match_transformers() {
        let want = [(0, 0), (1, 17), (-1, 1), (7, 23), (-8, 8), (8, 24), (20, 26), (-100, 15), (127, 31), (500, 31), (-9, 8), (13, 25), (-50, 13),
                    (64, 30)];
        for (r, b) in want {
            assert_eq!(super::bucket(r), b, "relative position {r}");
        }
    }
}
