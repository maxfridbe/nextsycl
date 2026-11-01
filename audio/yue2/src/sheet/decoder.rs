//! SheetSage2's decoder on the card: transformers' `BartDecoder` as modeling_sheetsage2.py configures it - 6 post-norm
//! layers of 512 (8 heads of 64, a 2048 GELU feed-forward), learned positions (offset 2, 5120 of them), a layer norm on
//! the embeddings, no embedding scale, the output tied to the token embedding (no bias).
//!
//! ```text
//!   x = LN_embd(tok[id] + pos[2 + position])
//!   a layer: x = LN(x + self_attn(x))   causal, its keys and values kept in the window's cache
//!            x = LN(x + cross_attn(x, memory))   the encoder's 7500 frames, their keys and values made once a window
//!            x = LN(x + fc2(gelu(fc1(x))))
//!   logits = x . tok^T
//! ```
//!
//! Rows go through in a block (a prompt or an overlap prefix: causal attention over the block and the cache before
//! it) or one a step (the greedy decode). The residual stream and the norms float32, the products' inputs half, their
//! sums float32; the caches half. The embedding tables stay on the host too (float32): a step's row is looked up
//! there and goes up with its position.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::Gguf;

use crate::ffi::{self, Api};
use crate::lm::{at, host_vec, Scratch, M, P};
use crate::sheet::mert::{Lin, Norm, Up, FRAMES};
use crate::sheet::tokens::VOCAB;

pub const WIDTH: usize = 512;
pub const LAYERS: usize = 6;
pub const HEADS: usize = 8;
pub const HEAD: usize = 64;
pub const FFN: usize = 2048;
/// the decoder's context (max_output_seq_len)
pub const CONTEXT: usize = 5120;
const POS_OFFSET: usize = 2;
const EPS: f32 = 1e-5;

struct Layer {
    sq: Lin,
    sk: Lin,
    sv: Lin,
    so: Lin,
    s_norm: Norm,
    cq: Lin,
    ck: Lin,
    cv: Lin,
    co: Lin,
    c_norm: Norm,
    up: Lin,
    down: Lin,
    f_norm: Norm,
}

/// One window's decoder state: the self-attention caches (half [cap, 512] a layer, `len` rows held) and the
/// cross-attention's keys and values of its memory (half [7500, 512] a layer)
pub struct State {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    xk: Vec<DevBuf>,
    xv: Vec<DevBuf>,
    pub len: usize,
    pub cap: usize,
}

pub struct Decoder {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    k: &'static Api,
    tok: Lin,
    tok_host: Vec<f32>,
    pos_host: Vec<f32>,
    norm_embd: Norm,
    layers: Vec<Layer>,
    pub bytes: u64,
}

impl Decoder {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>) -> Result<Decoder> {
        let mut u = Up { f, gpu, bytes: 0 };
        // the token table (no ".weight" in its name): the embedding and, tied, the output
        let t = f.tensor("dec.tok_embd").ok_or_else(|| Error("no dec.tok_embd".into()))?;
        if t.elements() != (VOCAB * WIDTH) as u64 || t.ty != nextsycl_gguf::GType::F16 {
            return Err(Error(format!("dec.tok_embd: {} {:?}, half [{VOCAB}, {WIDTH}] expected", t.ty.name(), t.shape)));
        }
        let tok = Lin { w: u.raw(&f.read(t).map_err(|e| Error(e.0))?)?, b: None, n: VOCAB, k: WIDTH };
        let tok_host = host_vec(f, "dec.tok_embd")?;
        let pos_host = host_vec(f, "dec.pos_embd")?;
        if pos_host.len() != (CONTEXT + POS_OFFSET) * WIDTH {
            return Err(Error(format!("dec.pos_embd: {} values, [{}, {WIDTH}] expected", pos_host.len(), CONTEXT + POS_OFFSET)));
        }
        let norm_embd = u.norm("dec.norm_embd", WIDTH)?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let p = format!("dec.blk.{i}");
            let l = |u: &mut Up, n: &str, rows: usize, k: usize| u.lin(&format!("{p}.{n}"), rows, k, true);
            layers.push(Layer { sq: l(&mut u, "self_attn_q", WIDTH, WIDTH)?, sk: l(&mut u, "self_attn_k", WIDTH, WIDTH)?, sv: l(&mut u, "self_attn_v", WIDTH, WIDTH)?,
                                so: l(&mut u, "self_attn_output", WIDTH, WIDTH)?, s_norm: u.norm(&format!("{p}.self_attn_norm"), WIDTH)?,
                                cq: l(&mut u, "cross_attn_q", WIDTH, WIDTH)?, ck: l(&mut u, "cross_attn_k", WIDTH, WIDTH)?, cv: l(&mut u, "cross_attn_v", WIDTH, WIDTH)?,
                                co: l(&mut u, "cross_attn_output", WIDTH, WIDTH)?, c_norm: u.norm(&format!("{p}.cross_attn_norm"), WIDTH)?,
                                up: l(&mut u, "ffn_up", FFN, WIDTH)?, down: l(&mut u, "ffn_down", WIDTH, FFN)?, f_norm: u.norm(&format!("{p}.final_norm"), WIDTH)? });
        }
        let bytes = u.bytes;
        Ok(Decoder { nsd: Nsd::new(gpu)?, k: ffi::api()?, gpu: gpu.clone(), tok, tok_host, pos_host, norm_embd, layers, bytes })
    }

    /// float32 -> half (n values)
    fn half(&self, x: P, n: usize, out: M) -> Result<()> {
        // SAFETY: x holds n floats, out n halves.
        ffi::check(unsafe { (self.k.to_half)(self.gpu.raw(), x.cast(), n as i64, out) }, "half")
    }

    fn lin(&self, x: P, m: usize, l: &Lin, out: M, dt: Dt) -> Result<()> {
        self.nsd.linear(x, Dt::F16, m, l.k, l.w.ptr(), l.n, l.b.as_ref().map_or(none(), |b| b.ptr().cast_const()), out, dt)
    }

    /// A window's state: its memory (float32 [7500, 512] on the card) as every layer's cross keys and values, and
    /// caches for `cap` rows
    pub fn start(&self, s: &mut Scratch, memory: &DevBuf, cap: usize) -> Result<State> {
        let g = &self.gpu;
        let n = FRAMES * WIDTH;
        let mh = s.get(g, "d_mem", n * 2)?;
        self.half(memory.ptr(), n, mh)?;
        let (mut xk, mut xv, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for l in &self.layers {
            let a = DevBuf::new(g, n * 2)?;
            let b = DevBuf::new(g, n * 2)?;
            self.lin(mh, FRAMES, &l.ck, a.ptr(), Dt::F16)?;
            self.lin(mh, FRAMES, &l.cv, b.ptr(), Dt::F16)?;
            xk.push(a);
            xv.push(b);
            k.push(DevBuf::new(g, cap * WIDTH * 2)?);
            v.push(DevBuf::new(g, cap * WIDTH * 2)?);
        }
        Ok(State { k, v, xk, xv, len: 0, cap })
    }

    /// `ids` at the next positions of `st` through the decoder; the logits (float32 [rows, 31678] on the host) of the
    /// block's rows `rows` (indices into ids)
    pub fn forward(&self, s: &mut Scratch, st: &mut State, ids: &[u32], rows: &[usize]) -> Result<Vec<f32>> {
        let g = &self.gpu;
        let m = ids.len();
        let p0 = st.len;
        if p0 + m > st.cap.min(CONTEXT) {
            return Err(Error(format!("{m} decoder rows at {p0}: past the cache's {} (the context is {CONTEXT})", st.cap.min(CONTEXT))));
        }
        let mut x0 = vec![0f32; m * WIDTH];
        for (i, t) in ids.iter().enumerate() {
            let t = *t as usize;
            if t >= VOCAB {
                return Err(Error(format!("token {t} past the vocabulary")));
            }
            let (e, p) = (&self.tok_host[t * WIDTH..(t + 1) * WIDTH], &self.pos_host[(POS_OFFSET + p0 + i) * WIDTH..(POS_OFFSET + p0 + i + 1) * WIDTH]);
            for j in 0..WIDTH {
                x0[i * WIDTH + j] = e[j] + p[j];
            }
        }
        let x = s.get(g, "d_x", m * WIDTH * 4)?;
        s.buf("d_x").write(0, crate::f32_bytes(&x0))?;
        let n = m * WIDTH;
        let ln = |x: M, nm: &Norm| self.nsd.layer_norm(x, Dt::F32, m, WIDTH, nm.w.ptr(), nm.b.ptr(), EPS, x, Dt::F32);
        ln(x, &self.norm_embd)?;
        let xh = s.get(g, "d_xh", n * 2)?;
        let q = s.get(g, "d_q", n * 2)?;
        let a = s.get(g, "d_a", n * 2)?;
        let o = s.get(g, "d_o", n * 4)?;
        let h = s.get(g, "d_h", m * FFN * 4)?;
        let hh = s.get(g, "d_hh", m * FFN * 2)?;
        let add = |o: P| self.nsd.gate_add(x, Dt::F32, m, WIDTH, o, Dt::F32, none(), none());
        for (li, l) in self.layers.iter().enumerate() {
            // self-attention: this block's keys and values into the cache at its rows, causal over the cache
            self.half(x, n, xh)?;
            self.lin(xh, m, &l.sq, q, Dt::F16)?;
            let (kc, vc) = (st.k[li].ptr(), st.v[li].ptr());
            self.lin(xh, m, &l.sk, at(kc, p0 * WIDTH * 2), Dt::F16)?;
            self.lin(xh, m, &l.sv, at(vc, p0 * WIDTH * 2), Dt::F16)?;
            if m == 1 {
                self.nsd.attention_qk(q, 1, WIDTH, kc, vc, p0 + 1, WIDTH, Dt::F16, HEADS, HEAD, a, Dt::F16)?;
            } else {
                // the rows' queries as if at their positions: q points at position 0 (the kernel reads rows p0 on)
                let q0 = q.cast::<u8>().wrapping_sub(p0 * WIDTH * 2).cast_const().cast();
                self.nsd.attention_causal_rows(q0, kc, vc, Dt::F16, p0, m, HEADS, HEADS, HEAD, WIDTH, WIDTH, a)?;
            }
            self.lin(a, m, &l.so, o, Dt::F32)?;
            add(o)?;
            ln(x, &l.s_norm)?;
            // cross-attention over the window's memory
            self.half(x, n, xh)?;
            self.lin(xh, m, &l.cq, q, Dt::F16)?;
            self.nsd.attention_qk(q, m, WIDTH, st.xk[li].ptr(), st.xv[li].ptr(), FRAMES, WIDTH, Dt::F16, HEADS, HEAD, a, Dt::F16)?;
            self.lin(a, m, &l.co, o, Dt::F32)?;
            add(o)?;
            ln(x, &l.c_norm)?;
            // the feed-forward
            self.half(x, n, xh)?;
            self.lin(xh, m, &l.up, h, Dt::F32)?;
            self.nsd.gelu(h, Dt::F32, m * FFN, false)?;
            self.half(h, m * FFN, hh)?;
            self.lin(hh, m, &l.down, o, Dt::F32)?;
            add(o)?;
            ln(x, &l.f_norm)?;
        }
        st.len += m;
        // the rows asked for, side by side, through the tied output
        let r = rows.len();
        if r == 0 {
            return Ok(Vec::new());
        }
        let sel = s.get(g, "d_sel", r * WIDTH * 2)?;
        self.half(x, n, xh)?;
        for (i, row) in rows.iter().enumerate() {
            if *row >= m {
                return Err(Error(format!("row {row} of a block of {m}")));
            }
            s.buf("d_sel").copy_within(i * WIDTH * 2, s.buf("d_xh"), row * WIDTH * 2, WIDTH * 2)?;
        }
        let lg = s.get(g, "d_logits", r * VOCAB * 4)?;
        self.lin(sel, r, &self.tok, lg, Dt::F32)?;
        let mut out = vec![0f32; r * VOCAB];
        // SAFETY: out is r x VOCAB floats, viewed as their bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        s.buf("d_logits").read(0, bytes)?;
        Ok(out)
    }
}
