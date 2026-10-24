//! The 5 Hz planner LM (acestep-5Hz-lm-1.7B: Qwen3, 28 layers, 16 query heads over 8 key heads of 128, tied
//! embeddings, a vocabulary of 217204 - Qwen3's and 65535 audio codes): a decoder with each sequence's key / value
//! cache on the card, bfloat16. A prompt is read at once (causal attention over its rows), then a token a step - the
//! classifier-free guidance's two sequences in one step, their projections batched, each its own cache.

use std::path::Path;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Dt;
use nextsycl_gguf::safetensors::SafeTensors;

use crate::nn::{bf, read_f32, rope_table, Layer, Lin, Loader, Mask, Ops, Scratch, HD};
use crate::qwen3::Config;

pub struct Lm {
    pub cfg: Config,
    layers: Vec<Layer>,
    norm: DevBuf,
    /// the embedding table: bfloat16 on the host (a token's row looked up there) and on the card (the tied head)
    embed: Vec<u16>,
    head: Lin,
    pub vocab: usize,
    pub bytes: u64,
}

/// One sequence's cache: each layer's keys and values (bfloat16 [cap, 8 x 128])
pub struct Seq {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub len: usize,
    cap: usize,
}

impl Lm {
    pub fn load(weights: &Path, cfg: Config, gpu: &Arc<Gpu>) -> Result<Lm> {
        let st = SafeTensors::open(weights).map_err(|e| Error(e.0))?;
        let l = Loader::new(&st, gpu);
        let layers = (0..cfg.layers).map(|i| Layer::load(&l, &format!("layers.{i}"), cfg.heads, cfg.kv_heads, Mask::Causal)).collect::<Result<Vec<_>>>()?;
        let (raw, shape) = l.bf_raw("embed_tokens.weight")?;
        let (vocab, h) = (shape[0] as usize, shape[1] as usize);
        if h != cfg.hidden {
            return Err(Error(format!("embed_tokens [{vocab}, {h}]: hidden {} expected", cfg.hidden)));
        }
        let head = Lin { w: l.upload(&raw)?, b: None, n: vocab, k: h };
        let embed = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let norm = l.float("norm.weight")?;
        Ok(Lm { layers, norm, embed, head, vocab, bytes: l.bytes.get(), cfg })
    }

    /// An empty cache for up to `cap` tokens
    pub fn seq(&self, gpu: &Arc<Gpu>, cap: usize) -> Result<Seq> {
        let kw = self.cfg.kv_heads * HD;
        let mut k = Vec::new();
        let mut v = Vec::new();
        for _ in 0..self.cfg.layers {
            k.push(DevBuf::new(gpu, cap * kw * 2)?);
            v.push(DevBuf::new(gpu, cap * kw * 2)?);
        }
        Ok(Seq { k, v, len: 0, cap })
    }

    fn embed_rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let mut x = Vec::with_capacity(ids.len() * h);
        for id in ids {
            let i = *id as usize;
            if i >= self.vocab {
                return Err(Error(format!("token {i}: past the vocabulary ({})", self.vocab)));
            }
            x.extend(self.embed[i * h..(i + 1) * h].iter().map(|b| bf(*b)));
        }
        Ok(x)
    }

    /// The logits of the residual rows named ("lm_x" after the layers) over the vocabulary's rows [r0, r0 + n) alone
    /// (the tied head is the embedding table: its rows a slice of it), float32 [rows, n] on the host
    fn head_range(&self, ops: &Ops, s: &mut Scratch, rows: &[usize], r0: usize, n: usize) -> Result<Vec<f32>> {
        let (h, g) = (self.cfg.hidden, &ops.gpu);
        let k = rows.len();
        let last = s.get(g, "lm_last", k * h * 4)?;
        for (i, r) in rows.iter().enumerate() {
            s.buf("lm_last").copy_within(i * h * 4, s.buf("lm_x"), r * h * 4, h * 4)?;
        }
        let hn = s.get(g, "lm_hn", k * h * 2)?;
        ops.rms(last, k, h, &self.norm, hn)?;
        let lo = s.get(g, "lm_logits", k * n * 4)?;
        let w = (self.head.w.ptr() as usize + r0 * h * 2) as *const std::ffi::c_void;
        ops.nsd.linear(hn, Dt::BF16, k, h, w, n, nextsycl_diffusion::kernels::none(), lo, Dt::F32)?;
        read_f32(s.buf("lm_logits"), k * n)
    }

    /// The logits (float32 [rows, vocab] on the host) of the residual rows named ("lm_x" after the layers)
    fn head(&self, ops: &Ops, s: &mut Scratch, rows: &[usize]) -> Result<Vec<f32>> {
        let (h, g) = (self.cfg.hidden, &ops.gpu);
        let n = rows.len();
        let last = s.get(g, "lm_last", n * h * 4)?;
        for (i, r) in rows.iter().enumerate() {
            s.buf("lm_last").copy_within(i * h * 4, s.buf("lm_x"), r * h * 4, h * 4)?;
        }
        let hn = s.get(g, "lm_hn", n * h * 2)?;
        ops.rms(last, n, h, &self.norm, hn)?;
        let lo = s.get(g, "lm_logits", n * self.vocab * 4)?;
        ops.linear(hn, n, &self.head, lo, Dt::F32)?;
        read_f32(s.buf("lm_logits"), n * self.vocab)
    }

    /// The prompt `ids` into `seq` (its cache filled from its start): the last token's logits [vocab]
    pub fn prefill(&self, ops: &Ops, s: &mut Scratch, seq: &mut Seq, ids: &[u32]) -> Result<Vec<f32>> {
        let (c, g) = (self.cfg.hidden, &ops.gpu);
        let m = ids.len();
        if m == 0 || m > seq.cap {
            return Err(Error(format!("a prompt of {m} tokens for a cache of {}", seq.cap)));
        }
        let x = ops.put_f32(s, "lm_x", &self.embed_rows(ids)?)?;
        let cs = ops.put_f32(s, "lm_cs", &rope_table(m, self.cfg.theta, 0))?;
        for (li, l) in self.layers.iter().enumerate() {
            let a = &l.attn;
            let (qw, kw, i2) = (a.hq * HD, a.hkv * HD, l.mlp.gu.n);
            let hb = s.get(g, "lm_h", m * c * 2)?;
            let q = s.get(g, "lm_q", m * qw * 2)?;
            let o = s.get(g, "lm_o", m * qw * 2)?;
            let y = s.get(g, "lm_y", m * c * 2)?;
            let gu = s.get(g, "lm_gu", m * i2 * 2)?;
            let ff = s.get(g, "lm_ff", m * (i2 / 2) * 2)?;
            let (kc, vc) = (seq.k[li].ptr(), seq.v[li].ptr());
            ops.rms(x, m, c, &l.ln1, hb)?;
            ops.linear(hb, m, &a.q, q, Dt::BF16)?;
            ops.linear(hb, m, &a.k, kc, Dt::BF16)?;
            ops.linear(hb, m, &a.v, vc, Dt::BF16)?;
            ops.qk_norm_rope(q, m, a.hq, qw, &a.qn, cs)?;
            ops.qk_norm_rope(kc, m, a.hkv, kw, &a.kn, cs)?;
            ops.nsd.attention_causal(q, kc, vc, Dt::BF16, m, a.hq, a.hkv, HD, qw, kw, o)?;
            ops.linear(o, m, &a.o, y, Dt::BF16)?;
            ops.add(x, y, m, c, nextsycl_diffusion::kernels::none())?;
            ops.rms(x, m, c, &l.ln2, hb)?;
            ops.linear(hb, m, &l.mlp.gu, gu, Dt::BF16)?;
            ops.nsd.swiglu(gu, Dt::BF16, m, i2 / 2, ff, Dt::BF16)?;
            ops.linear(ff, m, &l.mlp.down, y, Dt::BF16)?;
            ops.add(x, y, m, c, nextsycl_diffusion::kernels::none())?;
        }
        seq.len = m;
        self.head(ops, s, &[m - 1])
    }

    /// One token for each sequence (`tokens[i]` appended to `seqs[i]`): their logits, a row each [n, vocab]
    pub fn step(&self, ops: &Ops, s: &mut Scratch, seqs: &mut [&mut Seq], tokens: &[u32]) -> Result<Vec<f32>> {
        self.layers_step(ops, s, seqs, tokens)?;
        let rows: Vec<usize> = (0..seqs.len()).collect();
        self.head(ops, s, &rows)
    }

    /// The same, the logits over the vocabulary's rows [r0, r0 + n) and the row `extra` alone: [seqs, n + 1] (the codes
    /// phase: the audio codes, then the end)
    #[allow(clippy::too_many_arguments)]
    pub fn step_rows(&self, ops: &Ops, s: &mut Scratch, seqs: &mut [&mut Seq], tokens: &[u32], r0: usize, n: usize, extra: usize) -> Result<Vec<f32>> {
        self.layers_step(ops, s, seqs, tokens)?;
        let rows: Vec<usize> = (0..seqs.len()).collect();
        let a = self.head_range(ops, s, &rows, r0, n)?;
        let b = self.head_range(ops, s, &rows, extra, 1)?;
        let mut out = Vec::with_capacity(rows.len() * (n + 1));
        for i in 0..rows.len() {
            out.extend_from_slice(&a[i * n..(i + 1) * n]);
            out.push(b[i]);
        }
        Ok(out)
    }

    fn layers_step(&self, ops: &Ops, s: &mut Scratch, seqs: &mut [&mut Seq], tokens: &[u32]) -> Result<()> {
        let (c, g) = (self.cfg.hidden, &ops.gpu);
        let n = seqs.len();
        if tokens.len() != n {
            return Err(Error("a token a sequence".into()));
        }
        if let Some(q) = seqs.iter().find(|q| q.len >= q.cap) {
            return Err(Error(format!("the planner's cache is full ({} tokens)", q.cap)));
        }
        let x = ops.put_f32(s, "lm_x", &self.embed_rows(tokens)?)?;
        let mut cs = Vec::with_capacity(n * HD);
        for q in seqs.iter() {
            cs.extend(rope_table(1, self.cfg.theta, q.len));
        }
        let cs = ops.put_f32(s, "lm_cs", &cs)?;
        let cs_row = |i: usize| (cs as usize + i * HD * 4) as *mut std::ffi::c_void;
        for (li, l) in self.layers.iter().enumerate() {
            let a = &l.attn;
            let (qw, kw, i2) = (a.hq * HD, a.hkv * HD, l.mlp.gu.n);
            let hb = s.get(g, "lm_h", n * c * 2)?;
            let q = s.get(g, "lm_q", n * qw * 2)?;
            let o = s.get(g, "lm_o", n * qw * 2)?;
            let y = s.get(g, "lm_y", n * c * 2)?;
            let gu = s.get(g, "lm_gu", n * i2 * 2)?;
            let ff = s.get(g, "lm_ff", n * (i2 / 2) * 2)?;
            ops.rms(x, n, c, &l.ln1, hb)?;
            ops.linear(hb, n, &a.q, q, Dt::BF16)?;
            ops.qk_norm_rope(q, n, a.hq, qw, &a.qn, cs)?;
            for (i, sq) in seqs.iter().enumerate() {
                let row = sq.len;
                let hi = (hb as usize + i * c * 2) as *mut std::ffi::c_void;
                let kc = (sq.k[li].ptr() as usize + row * kw * 2) as *mut std::ffi::c_void;
                let vc = (sq.v[li].ptr() as usize + row * kw * 2) as *mut std::ffi::c_void;
                ops.linear(hi, 1, &a.k, kc, Dt::BF16)?;
                ops.linear(hi, 1, &a.v, vc, Dt::BF16)?;
                ops.qk_norm_rope(kc, 1, a.hkv, kw, &a.kn, cs_row(i))?;
                // the row's query as row `row` of a table that starts `row` rows before it (nsd reads that row alone)
                let qi = (q as usize + i * qw * 2).wrapping_sub(row * qw * 2) as *const std::ffi::c_void;
                let oi = (o as usize + i * qw * 2) as *mut std::ffi::c_void;
                ops.nsd.attention_causal_rows(qi, sq.k[li].ptr(), sq.v[li].ptr(), Dt::BF16, row, 1, a.hq, a.hkv, HD, qw, kw, oi)?;
            }
            ops.linear(o, n, &a.o, y, Dt::BF16)?;
            ops.add(x, y, n, c, nextsycl_diffusion::kernels::none())?;
            ops.rms(x, n, c, &l.ln2, hb)?;
            ops.linear(hb, n, &l.mlp.gu, gu, Dt::BF16)?;
            ops.nsd.swiglu(gu, Dt::BF16, n, i2 / 2, ff, Dt::BF16)?;
            ops.linear(ff, n, &l.mlp.down, y, Dt::BF16)?;
            ops.add(x, y, n, c, nextsycl_diffusion::kernels::none())?;
        }
        for sq in seqs.iter_mut() {
            sq.len += 1;
        }
        Ok(())
    }
}
