//! The text side: flan-T5-base's SentencePiece (Unigram) tokenizer and its encoder (12 blocks of d 768, 12 heads of 64,
//! gated GELU, relative position buckets), float32 on the GPU. UniMate pools the encoder's hidden states of a text by
//! their mean (a caption's, and each joint name's).

use std::collections::HashMap;
use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_model3d::{Error, Result};

use crate::ffi::{Attn, Rows, GATED_GELU};
use crate::ops::{Lin, Ops, Store};

pub const DIM: usize = 768;
const HEADS: usize = 12;
const HD: usize = 64;
const FF: usize = 2048;
const BLOCKS: usize = 12;
const EPS: f32 = 1e-6;
const BUCKETS: usize = 32;
const MAX_DISTANCE: f32 = 128.0;

/// T5's Unigram tokenizer (tokenizer.json): the text normalized (spaces collapsed; the file's NFKC table is not
/// applied: non-ASCII text may tokenize differently from the reference), cut by the best-scoring path through the
/// vocabulary's pieces (Viterbi), and ended by `</s>`.
pub struct T5Tok {
    ids: HashMap<String, (u32, f32)>,
    max_piece: usize,
    unk: u32,
    unk_score: f32,
}

pub const EOS: u32 = 1;

impl T5Tok {
    pub fn open(path: &Path) -> Result<T5Tok> {
        let text = std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let j: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let m = &j["model"];
        if m["type"] != "Unigram" {
            return Err(Error(format!("{}: not a Unigram tokenizer", path.display())));
        }
        let unk = m["unk_id"].as_u64().unwrap_or(2) as u32;
        let mut ids = HashMap::new();
        let mut min = 0f32;
        let mut max_piece = 1;
        for (i, e) in m["vocab"].as_array().ok_or_else(|| Error("no vocab".into()))?.iter().enumerate() {
            let piece = e[0].as_str().ok_or_else(|| Error("a piece without text".into()))?;
            let score = e[1].as_f64().unwrap_or(0.0) as f32;
            if i as u32 != unk && i > 2 {
                min = min.min(score);
            }
            max_piece = max_piece.max(piece.chars().count());
            ids.insert(piece.to_string(), (i as u32, score));
        }
        Ok(T5Tok { ids, max_piece, unk, unk_score: min - 10.0 })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut out = Vec::new();
        if !words.is_empty() {
            let norm: Vec<char> = format!("▁{}", words.join("▁")).chars().collect();
            out = self.viterbi(&norm);
        }
        out.push(EOS);
        out
    }

    fn viterbi(&self, s: &[char]) -> Vec<u32> {
        let n = s.len();
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut from: Vec<(usize, u32)> = vec![(0, self.unk); n + 1];
        best[0] = 0.0;
        for i in 0..n {
            if best[i] == f32::NEG_INFINITY {
                continue;
            }
            let mut any = false;
            for l in 1..=self.max_piece.min(n - i) {
                let piece: String = s[i..i + l].iter().collect();
                if let Some(&(id, score)) = self.ids.get(&piece) {
                    if id == self.unk {
                        continue;
                    }
                    any = true;
                    if best[i] + score > best[i + l] {
                        best[i + l] = best[i] + score;
                        from[i + l] = (i, id);
                    }
                }
            }
            if !any && best[i] + self.unk_score > best[i + 1] {
                best[i + 1] = best[i] + self.unk_score;
                from[i + 1] = (i, self.unk);
            }
        }
        let mut ids = Vec::new();
        let mut at = n;
        while at > 0 {
            let (p, id) = from[at];
            ids.push(id);
            at = p;
        }
        ids.reverse();
        // consecutive unknown characters are one unknown piece
        ids.dedup_by(|a, b| *a == self.unk && *b == self.unk);
        ids
    }
}

struct Block {
    ln1: DevBuf,
    qkv: Lin,
    o: Lin,
    ln2: DevBuf,
    wi: Lin,
    wo: Lin,
}

/// flan-T5-base's encoder
pub struct T5 {
    st: SafeTensors,
    blocks: Vec<Block>,
    rel: Vec<f32>,
    final_ln: DevBuf,
}

impl T5 {
    pub fn load(ops: &Ops, path: &Path) -> Result<T5> {
        let st = SafeTensors::open(path).map_err(|e| Error(e.0))?;
        let mut s = Store::new(&ops.gpu);
        let want = |s: &mut Store, name: String, shape: &[usize]| -> Result<()> {
            let v = st.f32(&name).map_err(|e| Error(e.0))?;
            let t = st.need(&name).map_err(|e| Error(e.0))?;
            let got: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
            if got != shape {
                return Err(Error(format!("{}: {name} is {got:?}, not {shape:?}", path.display())));
            }
            s.map.insert(name, (got, v));
            Ok(())
        };
        let mut blocks = Vec::new();
        for b in 0..BLOCKS {
            let p = format!("encoder.block.{b}.layer");
            for n in ["q", "k", "v"] {
                want(&mut s, format!("{p}.0.SelfAttention.{n}.weight"), &[DIM, DIM])?;
            }
            want(&mut s, format!("{p}.0.SelfAttention.o.weight"), &[DIM, DIM])?;
            want(&mut s, format!("{p}.0.layer_norm.weight"), &[DIM])?;
            want(&mut s, format!("{p}.1.layer_norm.weight"), &[DIM])?;
            want(&mut s, format!("{p}.1.DenseReluDense.wi_0.weight"), &[FF, DIM])?;
            want(&mut s, format!("{p}.1.DenseReluDense.wi_1.weight"), &[FF, DIM])?;
            want(&mut s, format!("{p}.1.DenseReluDense.wo.weight"), &[DIM, FF])?;
            let cat = |names: &[&str], s: &mut Store| -> Result<Vec<f32>> {
                let mut v = Vec::new();
                for n in names {
                    v.extend(s.take(&format!("{p}.{n}.weight"), if n.contains("wi_") { &[FF, DIM] } else { &[DIM, DIM] })?);
                }
                Ok(v)
            };
            let qkv = cat(&["0.SelfAttention.q", "0.SelfAttention.k", "0.SelfAttention.v"], &mut s)?;
            let wi = cat(&["1.DenseReluDense.wi_0", "1.DenseReluDense.wi_1"], &mut s)?;
            let lin = |w: &[f32], n: usize, k: usize| -> Result<Lin> { Ok(Lin { w: ops.up(w)?, b: None, n, k }) };
            blocks.push(Block {
                ln1: s.dev(&format!("{p}.0.layer_norm.weight"), &[DIM])?,
                qkv: lin(&qkv, 3 * DIM, DIM)?,
                o: s.lin(&format!("{p}.0.SelfAttention.o"), DIM, DIM, false)?,
                ln2: s.dev(&format!("{p}.1.layer_norm.weight"), &[DIM])?,
                wi: lin(&wi, 2 * FF, DIM)?,
                wo: s.lin(&format!("{p}.1.DenseReluDense.wo"), DIM, FF, false)?,
            });
        }
        want(&mut s, "encoder.final_layer_norm.weight".into(), &[DIM])?;
        want(&mut s, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(), &[BUCKETS, HEADS])?;
        let rel = s.take("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", &[BUCKETS, HEADS])?;
        let final_ln = s.dev("encoder.final_layer_norm.weight", &[DIM])?;
        if st.tensor("shared.weight").is_none() {
            return Err(Error(format!("{}: no shared.weight (not flan-T5)", path.display())));
        }
        Ok(T5 { st, blocks, rel, final_ln })
    }

    /// The rows of the embedding table for `ids`
    fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.st.need("shared.weight").map_err(|e| Error(e.0))?;
        if t.dtype != "F32" || t.shape.len() != 2 || t.shape[1] as usize != DIM {
            return Err(Error("shared.weight is not float32 [vocab, 768]".into()));
        }
        let mut out = Vec::with_capacity(ids.len() * DIM);
        let mut row = vec![0u8; DIM * 4];
        for &i in ids {
            if i as u64 >= t.shape[0] {
                return Err(Error(format!("token {i} past the vocabulary")));
            }
            self.st.read_into(t, i as u64 * DIM as u64 * 4, &mut row).map_err(|e| Error(e.0))?;
            out.extend(row.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }
        Ok(out)
    }

    /// T5's bidirectional relative position bucket of key - query
    fn bucket(rel: i64) -> usize {
        let nb = BUCKETS / 2;
        let mut b = if rel > 0 { nb } else { 0 };
        let n = rel.unsigned_abs() as usize;
        let exact = nb / 2;
        b += if n < exact {
            n
        } else {
            let large = exact as f32 + ((n as f32 / exact as f32).ln() / (MAX_DISTANCE / exact as f32).ln() * (nb - exact) as f32).trunc();
            (large as usize).min(nb - 1)
        };
        b
    }

    /// The hidden states [S, 768] of the token ids (the final norm applied), on the host
    pub fn encode(&self, ops: &Ops, ids: &[u32]) -> Result<Vec<f32>> {
        let s = ids.len();
        if s == 0 || s > 128 {
            return Err(Error(format!("T5 takes 1 to 128 tokens, not {s}")));
        }
        let x = ops.up(&self.embed(ids)?)?;
        let mut bias = vec![0f32; HEADS * s * s];
        for h in 0..HEADS {
            for i in 0..s {
                for j in 0..s {
                    bias[(h * s + i) * s + j] = self.rel[Self::bucket(j as i64 - i as i64) * HEADS + h];
                }
            }
        }
        let bias = ops.up(&bias)?;
        let (h, qkv, att, o, wi, g, wo) = (ops.buf(s * DIM)?, ops.buf(s * 3 * DIM)?, ops.buf(s * DIM)?, ops.buf(s * DIM)?, ops.buf(s * 2 * FF)?, ops.buf(s * FF)?, ops.buf(s * DIM)?);
        for b in &self.blocks {
            ops.k.rms(Rows::new(&x, DIM), s, 1, DIM, EPS, &b.ln1, None, Rows::new(&h, DIM))?;
            ops.linear(&h, s, &b.qkv, &qkv)?;
            let a = Attn { groups: 1, sq: s, skv: s, heads: HEADS, d: HD, q_gs: 0, q_ss: 1, kv_gs: 0, kv_ss: 1, nvalid: s, scale: 1.0 };
            ops.k.attention(Rows::new(&qkv, 3 * DIM), Rows { buf: &qkv, at: DIM, row: 3 * DIM }, Rows { buf: &qkv, at: 2 * DIM, row: 3 * DIM }, a, Some(&bias), Rows::new(&att, DIM))?;
            ops.linear(&att, s, &b.o, &o)?;
            ops.k.gated_add(&x, &o, s, DIM, None)?;
            ops.k.rms(Rows::new(&x, DIM), s, 1, DIM, EPS, &b.ln2, None, Rows::new(&h, DIM))?;
            ops.linear(&h, s, &b.wi, &wi)?;
            ops.k.glu(&wi, s, FF, GATED_GELU, &g)?;
            ops.linear(&g, s, &b.wo, &wo)?;
            ops.k.gated_add(&x, &wo, s, DIM, None)?;
        }
        ops.k.rms(Rows::new(&x, DIM), s, 1, DIM, EPS, &self.final_ln, None, Rows::new(&h, DIM))?;
        h.to_f32()
    }
}

/// The mean of a hidden state's rows (UniMate's pooled text embedding); zeros for no rows
pub fn pooled(hidden: &[f32]) -> Vec<f32> {
    let n = hidden.len() / DIM;
    let mut out = vec![0f32; DIM];
    for r in hidden.chunks_exact(DIM) {
        for (o, x) in out.iter_mut().zip(r) {
            *o += x;
        }
    }
    for o in &mut out {
        *o /= n.max(1) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_t5s() {
        // exact for small distances, log-spaced beyond, the sign picking the half
        assert_eq!(T5::bucket(0), 0);
        assert_eq!(T5::bucket(-3), 3);
        assert_eq!(T5::bucket(3), 16 + 3);
        assert_eq!(T5::bucket(-8), 8);
        assert_eq!(T5::bucket(-20), 10);
        assert_eq!(T5::bucket(-200), 15);
        assert_eq!(T5::bucket(200), 31);
    }

    #[test]
    fn a_unigram_cuts_by_score() {
        let mut ids = HashMap::new();
        for (i, (p, s)) in [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -2.0), ("▁walk", -5.0), ("▁wal", -4.0), ("k", -3.0), ("s", -3.0)].iter().enumerate() {
            ids.insert(p.to_string(), (i as u32, *s));
        }
        let t = T5Tok { ids, max_piece: 5, unk: 2, unk_score: -20.0 };
        assert_eq!(t.encode("walk"), vec![4, 1]);
        assert_eq!(t.encode("walks"), vec![4, 7, 1]);
        assert_eq!(t.encode(""), vec![1]);
        assert_eq!(t.encode("q"), vec![3, 2, 1]);
    }
}
