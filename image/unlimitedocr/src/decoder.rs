//! The decoder (modeling_deepseekv2.py, `mha_eager`: SlidingWindowLlamaAttention), from the language GGUF
//! (`deepseek2-ocr`): 12 layers of 1280 - attention of 10 heads of 128 with rotary positions over the whole head (NEOX
//! pairs, base 10000), then layer 0 a SwiGLU of 6848, layers 1..11 a mixture of 64 experts of 896 (softmax gate, top 6,
//! weights not renormalized) beside 2 shared experts (one SwiGLU of 1792). RMS norms (eps 1e-6). Weights are expanded
//! to half once at load (oneDNN's half linears); the residual stream, the caches and the attention are float32.
//!
//! The cache keeps the whole prompt (pictures and text) and, of what is written, only the last 128 tokens: a ring the
//! reference keeps the same way (`_ring_window`: generated token g at slot prompt + g % 128), so a page of any length
//! is read in fixed memory. Rotary positions keep counting.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;

pub const RING: usize = 128;
const THETA: f64 = 10000.0;

pub struct Geometry {
    pub layers: usize,
    pub n_embd: usize,
    pub heads: usize,
    pub head: usize,
    pub n_ff: usize,
    pub experts: usize,
    pub used: usize,
    pub n_ff_exp: usize,
    pub shared: usize,
    pub dense: usize,
    pub vocab: usize,
    pub eps: f32,
}

enum Ffn {
    /// gate | up stacked [2 n_ff][n_embd], down [n_embd][n_ff] (half)
    Dense { gu: DevBuf, down: DevBuf },
    Moe {
        /// the gate's scores [experts][n_embd] (float32)
        router: DevBuf,
        /// each expert's gate | up [experts][2 n_ff_exp][n_embd], down [experts][n_embd][n_ff_exp] (half)
        gu: DevBuf,
        down: DevBuf,
        sgu: DevBuf,
        sdown: DevBuf,
    },
}

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    /// q | k | v stacked [3 n_embd][n_embd], o [n_embd][n_embd] (half)
    qkv: DevBuf,
    o: DevBuf,
    ffn: Ffn,
}

pub struct Decoder {
    gpu: Arc<Gpu>,
    pub g: Geometry,
    /// the token embedding's stored blocks (rows expanded as they are needed)
    embd: DevBuf,
    embd_ty: GType,
    layers: Vec<Layer>,
    out_norm: DevBuf,
    head: DevBuf,
    pub loaded_bytes: u64,
}

/// A reading's caches: per layer K and V [prompt + RING][n_embd] float32
pub struct Cache {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    /// the prompt's length (fixed after the prompt pass) and tokens written since
    pub prompt: usize,
    pub written: usize,
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// The GGUF type code nsd_dequant takes
fn qcode(t: GType) -> Result<u32> {
    match t {
        GType::Q8_0 | GType::Q4K | GType::Q6K | GType::BF16 => Ok(t.code()),
        other => Err(Error(format!("{} weights are not read here (Q8_0, Q5_0, Q4_K, Q6_K, BF16)", other.name()))),
    }
}

impl Decoder {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Decoder> {
        let t0 = std::time::Instant::now();
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("the model has no {k}")));
        let n_embd = u("embedding_length")?;
        let heads = u("attention.head_count")?;
        let g = Geometry {
            layers: u("block_count")?,
            n_embd,
            heads,
            head: n_embd / heads,
            n_ff: u("feed_forward_length")?,
            experts: u("expert_count")?,
            used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            shared: u("expert_shared_count")?,
            dense: u("leading_dense_block_count").unwrap_or(1),
            vocab: u("vocab_size")?,
            eps: f.arch_meta("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
        };
        let mut loaded = 0u64;
        // a tensor's rows [r0, r0 + rows) of `cols` values expanded to half into `dst` at row `at`
        let mut expand = |name: &str, r0: usize, rows: usize, cols: usize, dst: &DevBuf, at: usize| -> Result<()> {
            let t = f.tensor(name).ok_or_else(|| Error(format!("the model has no {name}")))?;
            let rb = t.ty.bytes(cols as u64).ok_or_else(|| Error(format!("{name}: rows of {cols} do not fit its blocks")))? as usize;
            let mut raw = vec![0u8; rows * rb];
            f.read_into(t, (r0 * rb) as u64, &mut raw).map_err(ge)?;
            loaded += raw.len() as u64;
            let src = DevBuf::new(gpu, raw.len())?;
            src.write(0, &raw)?;
            let out = dst.view(at * cols * 2, rows * cols * 2)?;
            if t.ty == GType::Q5_0 {
                let k = ffi::api()?;
                // SAFETY: rows x cols values of Q5_0 blocks into as many halfs.
                ffi::check(unsafe { (k.q5_0_half)(gpu.raw(), src.ptr(), (rows * cols) as i64, out.ptr()) }, name)?;
            } else {
                nsd.dequant(src.ptr(), qcode(t.ty)?, rows * cols, out.ptr(), Dt::F16)?;
            }
            nsd.wait()
        };
        let f32buf = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &crate::vision::f32s(f, name)?) };
        let half = |rows: usize, cols: usize| DevBuf::new(gpu, rows * cols * 2);
        let (e, ffe) = (n_embd, g.n_ff_exp);
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let qkv = half(3 * e, e)?;
            expand(&n("attn_q"), 0, e, e, &qkv, 0)?;
            expand(&n("attn_k"), 0, e, e, &qkv, e)?;
            expand(&n("attn_v"), 0, e, e, &qkv, 2 * e)?;
            let o = half(e, e)?;
            expand(&n("attn_output"), 0, e, e, &o, 0)?;
            let ffn = if l < g.dense {
                let gu = half(2 * g.n_ff, e)?;
                expand(&n("ffn_gate"), 0, g.n_ff, e, &gu, 0)?;
                expand(&n("ffn_up"), 0, g.n_ff, e, &gu, g.n_ff)?;
                let down = half(e, g.n_ff)?;
                expand(&n("ffn_down"), 0, e, g.n_ff, &down, 0)?;
                Ffn::Dense { gu, down }
            } else {
                let gu = half(g.experts * 2 * ffe, e)?;
                let down = half(g.experts * e, ffe)?;
                for x in 0..g.experts {
                    expand(&n("ffn_gate_exps"), x * ffe, ffe, e, &gu, x * 2 * ffe)?;
                    expand(&n("ffn_up_exps"), x * ffe, ffe, e, &gu, x * 2 * ffe + ffe)?;
                    expand(&n("ffn_down_exps"), x * e, e, ffe, &down, x * e)?;
                }
                let sf = ffe * g.shared;
                let sgu = half(2 * sf, e)?;
                expand(&n("ffn_gate_shexp"), 0, sf, e, &sgu, 0)?;
                expand(&n("ffn_up_shexp"), 0, sf, e, &sgu, sf)?;
                let sdown = half(e, sf)?;
                expand(&n("ffn_down_shexp"), 0, e, sf, &sdown, 0)?;
                Ffn::Moe { router: f32buf(&n("ffn_gate_inp"))?, gu, down, sgu, sdown }
            };
            layers.push(Layer { attn_norm: f32buf(&n("attn_norm"))?, ffn_norm: f32buf(&n("ffn_norm"))?, qkv, o, ffn });
        }
        let head = half(g.vocab, e)?;
        let hn = if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        expand(hn, 0, g.vocab, e, &head, 0)?;
        let et = f.tensor("token_embd.weight").ok_or_else(|| Error("the model has no token_embd.weight".into()))?;
        let raw = f.read(et).map_err(ge)?;
        let embd = DevBuf::new(gpu, raw.len())?;
        embd.write(0, &raw)?;
        let d = Decoder { gpu: gpu.clone(), embd_ty: et.ty, embd, layers, out_norm: f32buf("output_norm.weight")?, head, g, loaded_bytes: loaded };
        log(format!("unlimitedocr decoder: {} layers of {} ({} experts, {} used, {} shared), expanded to half on {} in {:.1} s", d.g.layers,
                    d.g.n_embd, d.g.experts, d.g.used, d.g.shared, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(d)
    }

    pub fn cache(&self, prompt: usize) -> Result<Cache> {
        let cap = (prompt + RING) * self.g.n_embd;
        let mut k = Vec::with_capacity(self.g.layers);
        let mut v = Vec::with_capacity(self.g.layers);
        for _ in 0..self.g.layers {
            k.push(DevBuf::f32(&self.gpu, cap)?);
            v.push(DevBuf::f32(&self.gpu, cap)?);
        }
        Ok(Cache { k, v, prompt, written: 0 })
    }

    /// The tokens' embeddings [n][n_embd]
    pub fn embed(&self, nsd: &Nsd, tokens: &[u32]) -> Result<DevBuf> {
        let e = self.g.n_embd;
        let rb = self.embd_ty.bytes(e as u64).ok_or_else(|| Error("the embedding's rows".into()))? as usize;
        let x = DevBuf::f32(&self.gpu, tokens.len() * e)?;
        for (i, t) in tokens.iter().enumerate() {
            if *t as usize >= self.g.vocab {
                return Err(Error(format!("token {t} is past the vocabulary")));
            }
            nsd.dequant(self.embd.view(*t as usize * rb, rb)?.ptr(), qcode(self.embd_ty)?, e, x.view(i * e * 4, e * 4)?.ptr(), Dt::F32)?;
        }
        Ok(x)
    }

    fn to_half(&self, x: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        // SAFETY: n floats in, n halfs out.
        ffi::check(unsafe { (k.to_half)(self.gpu.raw(), x.fp(), out.ptr(), n as i64) }, "to half")
    }

    /// One pass over `t` rows of embeddings `x` [t][n_embd]: the prompt (cache empty: the rows at slots 0..t, causal)
    /// or one written token (its slot in the ring, attending to everything cached). The last row's logits.
    pub fn pass(&self, nsd: &Nsd, c: &mut Cache, x: &DevBuf, t: usize, tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<Vec<f32>> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let g = &self.g;
        let (e, hd) = (g.n_embd, g.head);
        let prompt = c.written == 0 && c.prompt == t;
        if !prompt && t != 1 {
            return Err(Error("a pass is the prompt or one token".into()));
        }
        // positions and slots
        let (p0, slot) = if prompt { (0, 0) } else { (c.prompt + c.written - 1, c.prompt + (c.written - 1) % RING) };
        let keys = if prompt { t } else { c.prompt + c.written.min(RING) };
        let mut cs = vec![0f32; t * hd];
        for r in 0..t {
            for i in 0..hd / 2 {
                let a = (p0 + r) as f64 / THETA.powf(2.0 * i as f64 / hd as f64);
                cs[r * hd + 2 * i] = a.cos() as f32;
                cs[r * hd + 2 * i + 1] = a.sin() as f32;
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let h = DevBuf::f32(gpu, t * e)?;
        let hh = DevBuf::new(gpu, t * g.n_ff.max(2 * g.n_ff_exp * g.shared).max(e) * 2)?;
        let qkv = DevBuf::f32(gpu, t * 3 * e)?;
        let att = DevBuf::f32(gpu, t * e)?;
        let o = DevBuf::f32(gpu, t * e)?;
        let wide = DevBuf::f32(gpu, t * 2 * g.n_ff.max(g.n_ff_exp * g.shared))?;
        let act = DevBuf::f32(gpu, t * g.n_ff.max(g.n_ff_exp * g.shared))?;
        let none_rows = none();
        for (l, ly) in self.layers.iter().enumerate() {
            if l == 0 {
                tap("inp_embd", x)?;
            }
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, e, ly.attn_norm.ptr(), g.eps, none_rows, none(), none(), h.ptr(), Dt::F32)?;
            tap(&format!("attn_norm-{l}"), &h)?;
            self.to_half(&h, t * e, &hh)?;
            nsd.linear(hh.ptr(), Dt::F16, t, e, ly.qkv.ptr(), 3 * e, none(), qkv.ptr(), Dt::F32)?;
            tap(&format!("qkv-{l}"), &qkv)?;
            nsd.rope(qkv.ptr(), Dt::F32, t, g.heads, hd, 3 * e, cs.ptr(), hd)?;
            nsd.rope(qkv.fp().wrapping_add(e).cast(), Dt::F32, t, g.heads, hd, 3 * e, cs.ptr(), hd)?;
            // SAFETY: the rows' k and v into their cache slots (row stride n_embd).
            unsafe {
                ffi::check((k.copy_rows)(gpu.raw(), qkv.fp().add(e), 3 * e as i64, c.k[l].fp().add(slot * e), e as i64, t as i64, e as i64), "k cache")?;
                ffi::check((k.copy_rows)(gpu.raw(), qkv.fp().add(2 * e), 3 * e as i64, c.v[l].fp().add(slot * e), e as i64, t as i64, e as i64), "v cache")?;
            }
            if prompt {
                nsd.attention_causal(qkv.ptr(), c.k[l].ptr(), c.v[l].ptr(), Dt::F32, t, g.heads, g.heads, hd, 3 * e, e, att.ptr())?;
            } else {
                nsd.attention_qk(qkv.ptr(), 1, 3 * e, c.k[l].ptr(), c.v[l].ptr(), keys, e, Dt::F32, g.heads, hd, att.ptr(), Dt::F32)?;
            }
            self.to_half(&att, t * e, &hh)?;
            nsd.linear(hh.ptr(), Dt::F16, t, e, ly.o.ptr(), e, none(), o.ptr(), Dt::F32)?;
            tap(&format!("attn_out-{l}"), &o)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, e, o.ptr(), Dt::F32, none(), none())?;
            tap(&format!("ffn_inp-{l}"), x)?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, e, ly.ffn_norm.ptr(), g.eps, none_rows, none(), none(), h.ptr(), Dt::F32)?;
            self.to_half(&h, t * e, &hh)?;
            match &ly.ffn {
                Ffn::Dense { gu, down } => {
                    self.swiglu(nsd, &hh, t, gu, down, g.n_ff, &wide, &act, &o)?;
                    nsd.gate_add(x.ptr(), Dt::F32, t, e, o.ptr(), Dt::F32, none(), none())?;
                }
                Ffn::Moe { router, gu, down, sgu, sdown } => {
                    // the shared experts, then each routed expert over its rows, weighted into the residual
                    self.swiglu(nsd, &hh, t, sgu, sdown, g.n_ff_exp * g.shared, &wide, &act, &o)?;
                    nsd.gate_add(x.ptr(), Dt::F32, t, e, o.ptr(), Dt::F32, none(), none())?;
                    let scores = DevBuf::f32(gpu, t * g.experts)?;
                    nsd.linear(h.ptr(), Dt::F32, t, e, router.ptr(), g.experts, none(), scores.ptr(), Dt::F32)?;
                    nsd.wait()?;
                    let s = scores.to_f32()?;
                    let mut by: Vec<(Vec<i32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); g.experts];
                    for r in 0..t {
                        let row = &s[r * g.experts..(r + 1) * g.experts];
                        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let ex: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
                        let sum: f32 = ex.iter().sum();
                        let mut idx: Vec<usize> = (0..g.experts).collect();
                        // torch.topk: the largest first (ties: the lower index)
                        idx.sort_by(|a, b| ex[*b].partial_cmp(&ex[*a]).unwrap().then(a.cmp(b)));
                        for &xi in &idx[..g.used] {
                            by[xi].0.push(r as i32);
                            by[xi].1.push(ex[xi] / sum);
                        }
                    }
                    let gx = DevBuf::f32(gpu, t * e)?;
                    for (xi, (rows, w)) in by.iter().enumerate() {
                        if rows.is_empty() {
                            continue;
                        }
                        let m = rows.len();
                        let idx = DevBuf::new(gpu, m * 4)?;
                        idx.write(0, &rows.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                        let wb = DevBuf::from_f32(gpu, w)?;
                        // SAFETY: m of the normed rows gathered [m][n_embd].
                        ffi::check(unsafe { (k.gather)(gpu.raw(), h.fp(), e as i64, idx.ptr().cast(), m as i64, gx.fp()) }, "gather")?;
                        self.to_half(&gx, m * e, &hh)?;
                        let (ffe, w2) = (g.n_ff_exp, 2 * g.n_ff_exp * e);
                        let guv = gu.view(xi * w2 * 2, w2 * 2)?;
                        let dv = down.view(xi * e * ffe * 2, e * ffe * 2)?;
                        self.swiglu(nsd, &hh, m, &guv, &dv, ffe, &wide, &act, &o)?;
                        // SAFETY: o [m][n_embd] weighted into x's rows.
                        ffi::check(unsafe { (k.scatter_add)(gpu.raw(), x.fp(), e as i64, o.fp(), idx.ptr().cast(), wb.fp(), m as i64) }, "scatter")?;
                        nsd.wait()?;
                    }
                }
            }
            tap(&format!("l_out-{l}"), x)?;
        }
        // the last row's logits
        let last = x.view((t - 1) * e * 4, e * 4)?;
        nsd.rms_norm_mod(last.ptr(), Dt::F32, 1, e, self.out_norm.ptr(), g.eps, none_rows, none(), none(), h.ptr(), Dt::F32)?;
        tap("result_norm", &h.view(0, e * 4)?)?;
        self.to_half(&h, e, &hh)?;
        let logits = DevBuf::f32(gpu, g.vocab)?;
        nsd.linear(hh.ptr(), Dt::F16, 1, e, self.head.ptr(), g.vocab, none(), logits.ptr(), Dt::F32)?;
        nsd.wait()?;
        if prompt {
            c.written = 0;
        }
        logits.to_f32()
    }

    /// out [m][n_embd] = down(silu(gate(x)) * up(x)) for x [m][n_embd] in half, gate | up stacked
    #[allow(clippy::too_many_arguments)]
    fn swiglu(&self, nsd: &Nsd, xh: &DevBuf, m: usize, gu: &DevBuf, down: &DevBuf, ff: usize, wide: &DevBuf, act: &DevBuf, out: &DevBuf) -> Result<()> {
        let e = self.g.n_embd;
        nsd.linear(xh.ptr(), Dt::F16, m, e, gu.ptr(), 2 * ff, none(), wide.ptr(), Dt::F32)?;
        nsd.swiglu(wide.ptr(), Dt::F32, m, ff, act.ptr(), Dt::F32)?;
        // the activations to half in the input's buffer (it is read no more by this call's caller after the gate)
        let ah = DevBuf::new(&self.gpu, m * ff * 2)?;
        self.to_half(act, m * ff, &ah)?;
        nsd.linear(ah.ptr(), Dt::F16, m, ff, down.ptr(), e, none(), out.ptr(), Dt::F32)?;
        // ah goes when this returns: its work first (sycl::free does not wait for queued work)
        nsd.wait()
    }
}
