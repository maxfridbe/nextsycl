//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices: half on the GPU (with an
//! optional float32 bias), expanded at load from the GGUF's blocks (weights.rs).
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ffi;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// The values a product's split input needs (Ops::prod): its m rows ldx apart of k, and the rows past m its last
/// tile (of up to 64 rows) reads
pub fn split_len(m: usize, ldx: usize, k: usize) -> usize {
    (m + 64) * ldx + k
}

/// A product's input split for the matrix engines (two bfloat16 buffers of at least split_len values) and the
/// scratch its K pieces meet in (Ops::gemm_scratch floats)
pub struct Split {
    pub hi: DevBuf,
    pub lo: DevBuf,
    pub part: DevBuf,
}

/// The engine's kernels on one GPU
/// The last mark and each label's (seconds, calls)
type OpTimes = (Instant, BTreeMap<String, (f64, u32)>);

pub struct Ops {
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
    /// NS_VV_PROFILE=2: the tokenizers' operations timed one by one (the GPU waited for after each), by label
    prof: Option<Mutex<OpTimes>>,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        let prof = std::env::var("NS_VV_PROFILE").is_ok_and(|v| v == "2").then(|| Mutex::new((Instant::now(), BTreeMap::new())));
        Ok(Ops { gpu: gpu.clone(), k: ffi::api()?, prof })
    }

    /// NS_VV_PROFILE=2: the time since the last mark (the GPU waited for) under `label()`
    pub fn mark(&self, label: impl FnOnce() -> String) -> Result<()> {
        if let Some(p) = &self.prof {
            self.gpu.sync()?;
            let mut p = p.lock().map_err(|_| Error("profile lock".into()))?;
            let now = Instant::now();
            let dt = (now - p.0).as_secs_f64();
            p.0 = now;
            let e = p.1.entry(label()).or_insert((0.0, 0));
            e.0 += dt;
            e.1 += 1;
        }
        Ok(())
    }

    /// NS_VV_PROFILE=2: the marks' table on stderr, a frame's share (`frames`)
    pub fn report(&self, frames: usize) {
        if let Some(Ok(p)) = self.prof.as_ref().map(|p| p.lock()) {
            let mut rows: Vec<_> = p.1.iter().collect();
            rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            for (k, (t, n)) in rows {
                eprintln!("  {k:<44} {:>8.3} ms a frame {n:>7} calls {:>8.1} us each", 1e3 * t / frames.max(1) as f64, 1e6 * t / (*n).max(1) as f64);
            }
        }
    }

    fn g(&self) -> *mut c_void {
        self.gpu.raw()
    }

    /// out[m] (+)= x[m] . W^T (+ bias), m < M <= 20, out rows `ldo` apart
    pub fn gemv(&self, x: *const f32, m: usize, w: &Mat, out: *mut f32, ldo: usize, acc: bool) -> Result<()> {
        self.gemv_epi(x, m, w.k, w, out, ldo, if acc { Epi::Acc } else { Epi::Store }, None)
    }

    /// out[m] (epi) x[m] . W^T + bias for m < M <= 20 rows of x `ldx` floats apart (a window of a longer signal, a
    /// convolution's), out rows `ldo` apart; `gamma` the layer scale of Epi::Res and its rows' distance (0: one for
    /// every row)
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_epi(&self, x: *const f32, m: usize, ldx: usize, w: &Mat, out: *mut f32, ldo: usize, epi: Epi, gamma: Option<(*const f32, usize)>) -> Result<()> {
        let bias = w.bias.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        let scales = w.scales.as_ref().map_or(std::ptr::null(), |s| s.ptr() as *const c_void);
        let (gamma, ldg) = gamma.unwrap_or((std::ptr::null(), 0));
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gemv)(self.g(), x, m as i64, ldx as i64, w.k as i64, w.w.ptr(), w.wt as i32, scales, w.n as i64, bias, epi as i32, gamma,
                                           ldg as i64, out, ldo as i64) }, "gemv")
    }

    /// RMS norm of the rows of x [m, c] into out: x / rms * w (w None: 1); with `modulate` (scale, shift, their rows
    /// `lds` apart): then v * (1 + scale) + shift
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(&self, x: *const f32, m: usize, c: usize, w: Option<&DevBuf>, eps: f32, modulate: Option<(*const f32, *const f32, usize)>, out: *mut f32)
                    -> Result<()> {
        let (sc, sh, lds) = modulate.unwrap_or((std::ptr::null(), std::ptr::null(), 0));
        // SAFETY: x and out m * c floats; w c; scale / shift m rows of c, lds apart.
        ffi::check(unsafe { (self.k.rms_norm)(self.g(), x, m as i64, c as i64, w.map_or(std::ptr::null(), |w| w.fp() as *const f32), eps, sc, sh, lds as i64, out) },
                   "rms norm")
    }

    /// A decode step's RoPE and key / value store: rows of `qkv` (`ws` apart), row r of sequence `seqs[r]` (its
    /// cache's keys, values, positions) at the position in `pos` (device int32 [rows])
    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv(&self, qkv: *mut f32, ws: usize, pos: &DevBuf, hq: usize, hkv: usize, d: usize, inv: &DevBuf, seqs: &[(&DevBuf, &DevBuf, usize)]) -> Result<()> {
        let (a, b) = (seqs[0], *seqs.get(1).unwrap_or(&seqs[0]));
        // SAFETY: qkv rows of (hq + 2 hkv) d floats; the caches [hkv, t, d] half; pos rows int32.
        ffi::check(unsafe { (self.k.rope_kv)(self.g(), qkv, ws as i64, seqs.len() as i64, pos.ptr() as *const i32, hq as i64, hkv as i64, d as i64, inv.fp(),
                                             a.0.ptr(), a.1.ptr(), a.2 as i64, b.0.ptr(), b.1.ptr(), b.2 as i64) }, "rope + kv store")
    }

    /// A decode step's attention: row r of sequence `seqs[r]` up to its position in `pos`; the keys in `pieces`
    /// pieces (`part`: rows * hq * pieces * 130 floats); out [rows, hq * 128]
    #[allow(clippy::too_many_arguments)]
    pub fn attn_dec(&self, q: *const f32, qs: usize, pos: &DevBuf, seqs: &[(&DevBuf, &DevBuf, usize)], hq: usize, hkv: usize, pieces: usize, out: *mut f32,
                    part: &DevBuf) -> Result<()> {
        let (a, b) = (seqs[0], *seqs.get(1).unwrap_or(&seqs[0]));
        if part.len < seqs.len() * hq * pieces * 130 * 4 {
            return Err(Error(format!("attention scratch of {} bytes for {} rows in {pieces} pieces", part.len, seqs.len())));
        }
        // SAFETY: as rope_kv; part checked above.
        ffi::check(unsafe { (self.k.attn_dec)(self.g(), q, qs as i64, seqs.len() as i64, pos.ptr() as *const i32, a.0.ptr(), a.1.ptr(), a.2 as i64, b.0.ptr(),
                                              b.1.ptr(), b.2 as i64, hq as i64, hkv as i64, 128, pieces as i64, out, part.fp()) }, "attention")
    }

    /// The product over m rows of x `ldx` apart (windows of a signal may overlap): the small-batch one for up to 8
    /// rows, the tiled one for more - bfloat16 weights on the matrix engines when `split` (two bfloat16 buffers of at
    /// least split_len(m, ldx, k) values) is given: x is split into them first
    #[allow(clippy::too_many_arguments)]
    pub fn prod(&self, x: *const f32, m: usize, ldx: usize, w: &Mat, out: *mut f32, ldo: usize, epi: Epi, gamma: Option<(*const f32, usize)>,
                split: Option<&Split>) -> Result<()> {
        if m <= 8 && w.k.is_multiple_of(8) && ldx.is_multiple_of(4) {
            return self.gemv_epi(x, m, ldx, w, out, ldo, epi, gamma);
        }
        let bias = w.bias.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        let (gamma, ldg) = gamma.unwrap_or((std::ptr::null(), 0));
        let (mut xh, mut xl) = (std::ptr::null(), std::ptr::null());
        let (mut part, mut part_len) = (std::ptr::null_mut(), 0);
        if let Some(Split { hi, lo, part: pb }) = split.filter(|_| w.wt <= 1 && w.k.is_multiple_of(16) && ldx.is_multiple_of(8)) {
            (part, part_len) = (pb.fp(), pb.floats());
            if hi.len.min(lo.len) < split_len(m, ldx, w.k) * 2 {
                return Err(Error(format!("split buffers of {} bytes for {m} rows of {} ({ldx} apart)", hi.len.min(lo.len), w.k)));
            }
            // SAFETY: x holds (m - 1) * ldx + k floats (the caller's), hi / lo at least that many halves (checked).
            ffi::check(unsafe { (self.k.split)(self.g(), x, ((m - 1) * ldx + w.k) as i64, w.wt as i32, hi.ptr(), lo.ptr()) }, "split")?;
            (xh, xl) = (hi.ptr() as *const c_void, lo.ptr() as *const c_void);
        }
        // SAFETY: the caller's buffers hold the sizes named; the split ones have room for the tiles' rows past m.
        ffi::check(unsafe { (self.k.gemm)(self.g(), x, xh, xl, m as i64, ldx as i64, w.k as i64, w.w.ptr(), w.wt as i32, w.n as i64, bias, epi as i32, gamma,
                                          ldg as i64, out, ldo as i64, part, part_len as i64) }, "gemm")
    }

    /// Floats of scratch the matrix engines' form of an [m, k] x [k, n] product splits K with
    pub fn gemm_scratch(&self, m: usize, n: usize, k: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.gemm_scratch)(m as i64, n as i64, k as i64) as usize }
    }

    /// cur [p + l, c] = prev's rows [prev_off, prev_off + p), then the l rows of x normed (times w)
    #[allow(clippy::too_many_arguments)]
    pub fn norm_hist(&self, prev: &DevBuf, prev_off: usize, x: *const f32, l: usize, c: usize, p: usize, w: &DevBuf, eps: f32, cur: &DevBuf) -> Result<()> {
        if prev.len < (prev_off + p) * c * 4 || cur.len < (p + l) * c * 4 {
            return Err(Error(format!("norm_hist: buffers of {} / {} bytes for {p} + {l} rows of {c}", prev.len, cur.len)));
        }
        // SAFETY: sizes checked above; x l * c floats, w c.
        ffi::check(unsafe { (self.k.norm_hist)(self.g(), prev.fp(), prev_off as i64, x, l as i64, c as i64, p as i64, w.fp(), eps, cur.fp()) }, "norm")
    }

    /// x[t] += gamma * depthwise conv (k taps over cur [k - 1 + l, c]); h[t] = norm(x[t]) * w2
    #[allow(clippy::too_many_arguments)]
    pub fn block_mix(&self, cur: &DevBuf, x: *mut f32, l: usize, c: usize, k: usize, dw_w: &DevBuf, dw_b: &DevBuf, gamma: &DevBuf, w2: &DevBuf, eps: f32,
                     h: *mut f32) -> Result<()> {
        // SAFETY: cur (k - 1 + l) * c floats (the caller's), x and h l * c, the weights c * k and c.
        ffi::check(unsafe { (self.k.block_mix)(self.g(), cur.fp(), x, l as i64, c as i64, k as i64, dw_w.fp(), dw_b.fp(), gamma.fp(), w2.fp(), eps, h) },
                   "block mix")
    }

    /// buf [p + l, c]: rows [l, l + p) moved to the front
    pub fn shift(&self, buf: *mut f32, p: usize, l: usize, c: usize) -> Result<()> {
        // SAFETY: buf (p + l) * c floats (the caller's).
        ffi::check(unsafe { (self.k.shift)(self.g(), buf, p as i64, l as i64, c as i64) }, "shift")
    }

    /// out [s * b, h]: row i * b + j = silu(c0[j] + temb[i]) (every step's conditions)
    pub fn silu_rows(&self, c0: *const f32, temb: *const f32, s: usize, b: usize, h: usize, out: *mut f32) -> Result<()> {
        // SAFETY: c0 b * h floats, temb s * h, out s * b * h.
        ffi::check(unsafe { (self.k.silu_rows)(self.g(), c0, temb, s as i64, b as i64, h as i64, out) }, "silu rows")
    }

    /// One DPM-Solver++ step on the device (vv.h's ns_audio_vv_dpm): `c` the step's coefficients (sched.rs)
    #[allow(clippy::too_many_arguments)]
    pub fn dpm(&self, v: *const f32, l: usize, cfg: f32, x: *mut f32, prev: *mut f32, c: &crate::sched::Coef, x_in: Option<(*mut f32, usize)>, lat: Option<*mut f32>,
               un: Option<(*mut f32, f32, f32)>) -> Result<()> {
        let (xi, ldin) = x_in.unwrap_or((std::ptr::null_mut(), 0));
        let (u, scale, bias) = un.unwrap_or((std::ptr::null_mut(), 1.0, 0.0));
        // SAFETY: v 2l floats, x / prev / lat / un l, x_in two rows of l, ldin apart.
        ffi::check(unsafe { (self.k.dpm)(self.g(), v, l as i64, cfg, x, prev, c.a_s, c.s_s, c.ratio, c.ae, c.inv_r0, c.second as i32, xi, ldin as i64,
                                         lat.unwrap_or(std::ptr::null_mut()), u, scale, bias) }, "dpm step")
    }

    /// out [m, c] = silu(gate) * up of x [m, 2c] holding interleaved (gate, up) pairs
    pub fn swiglu_pairs(&self, x: *const f32, m: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: x m * 2c floats, out m * c.
        ffi::check(unsafe { (self.k.swiglu_pairs)(self.g(), x, m as i64, c as i64, out) }, "swiglu")
    }

    /// Floats of scratch `attn` needs
    pub fn attn_scratch(&self, b: usize, s: usize, hq: usize, d: usize, p0: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.attn_scratch)(b as i64, s as i64, hq as i64, d as i64, p0 as i64) as usize }
    }

    /// Causal attention against a half cache [b, hkv, t, d] holding positions [0, p0 + s) (six query heads a key head)
    pub fn attn(&self, q: *const f32, qs: usize, kc: &DevBuf, vc: &DevBuf, b: usize, s: usize, hq: usize, hkv: usize, d: usize, t: usize, p0: usize,
                out: *mut f32, part: *mut f32) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.attn)(self.g(), q, qs as i64, kc.ptr(), vc.ptr(), b as i64, s as i64, hq as i64, hkv as i64, d as i64, t as i64, p0 as i64,
                                          out, part) }, "attention")
    }

    pub fn kv_store(&self, k: *const f32, v: *const f32, ks: usize, b: usize, s: usize, hkv: usize, d: usize, t: usize, p0: usize, kc: &DevBuf,
                    vc: &DevBuf) -> Result<()> {
        // SAFETY: as attn.
        ffi::check(unsafe { (self.k.kv_store)(self.g(), k, v, ks as i64, b as i64, s as i64, hkv as i64, d as i64, t as i64, p0 as i64, kc.ptr(), vc.ptr()) },
                   "kv store")
    }

    /// RoPE (rotate-half) from a frequency table on rows of h heads x d, row r at position p0 + r % s
    pub fn rope(&self, x: *mut f32, xs: usize, rows: usize, s: usize, h: usize, d: usize, inv: &DevBuf, p0: usize) -> Result<()> {
        // SAFETY: rows of h heads x d floats, xs apart; inv d / 2 floats.
        ffi::check(unsafe { (self.k.head_norm_rope)(self.g(), x, xs as i64, rows as i64, s as i64, h as i64, d as i64, std::ptr::null(), 0.0, inv.fp(),
                                                    p0 as i64) }, "rope")
    }

    pub fn to_half(&self, x: *const f32, out: *mut c_void, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64) }, "to half")
    }

    /// x += y (n floats)
    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    /// x = silu(x) over n
    pub fn silu(&self, x: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.silu)(self.g(), x, n as i64) }, "silu")
    }
}

/// What a product does with its sums (vv.h's NS_VV_*)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Epi {
    Store = 0,
    Acc = 1,
    /// rows 2i, 2i + 1 are gate i and up i: out[i] = silu(gate) * up
    Swiglu = 2,
    Gelu = 3,
    /// out += gamma * v
    Res = 4,
}

/// A matrix [n, k] on the GPU: half (wt 0), bfloat16 (1) or Q8_0 (2: int8 codes [n, k], `scales` half [n, k / 32]),
/// its bias float32 [n] (or none). `k` may be padded with zero columns to a multiple of 256: the input rows are then
/// `k` wide, zero past `k_in`.
pub struct Mat {
    pub w: DevBuf,
    pub wt: u8,
    pub scales: Option<DevBuf>,
    pub bias: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
    pub k_in: usize,
}

impl Mat {
    /// Bytes its weights take on the card
    pub fn bytes(&self) -> usize {
        self.w.len + self.scales.as_ref().map_or(0, |s| s.len)
    }

    /// out [m, n] f32 = x [m, k] f32 . W^T + bias: a few rows by the engine's own product (bound by reading W once), more
    /// by oneDNN's half GEMM (x converted into `xh`, a half buffer of m * k)
    pub fn apply(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, xh: &DevBuf, out: *mut f32) -> Result<()> {
        self.apply_with(ops, nsd, x, m, xh, out, None)
    }

    /// As apply; a Q8_0 matrix's wide product expands it into `deq` (half [n, k]) first
    #[allow(clippy::too_many_arguments)]
    pub fn apply_with(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, xh: &DevBuf, out: *mut f32, deq: Option<&DevBuf>) -> Result<()> {
        if m <= 8 && self.k.is_multiple_of(8) {
            return ops.gemv(x, m, self, out, self.n, false);
        }
        if xh.len < m * self.k * 2 {
            return Err(Error(format!("a half staging buffer of {} bytes for a [{m}, {}] product", xh.len, self.k)));
        }
        let w = match self.wt {
            0 => self.w.ptr(),
            2 => {
                let d = deq.filter(|d| d.len >= self.n * self.k * 2)
                    .ok_or_else(|| Error(format!("a Q8_0 [{}, {}] product of {m} rows needs room to expand it", self.n, self.k)))?;
                let sc = self.scales.as_ref().ok_or_else(|| Error("a Q8_0 matrix without its scales".into()))?;
                // SAFETY: codes n * k bytes, scales n * k / 32 halves, d n * k halves (checked).
                ffi::check(unsafe { (ops.k.dequant_q8)(ops.g(), self.w.ptr(), sc.ptr(), self.n as i64, self.k as i64, d.ptr()) }, "Q8_0 expand")?;
                d.ptr()
            }
            t => return Err(Error(format!("a wide product of weights of type {t}"))),
        };
        ops.to_half(x, xh.ptr(), m * self.k)?;
        let bias = self.bias.as_ref().map_or(none(), |b| b.ptr() as *const c_void);
        nsd.linear(xh.ptr(), Dt::F16, m, self.k, w, self.n, bias, out.cast(), Dt::F32)
    }
}
