//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices (bfloat16 on the GPU: the
//! checkpoint's own values, or float32).
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};

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

/// A matrix [n, k] on the GPU: bfloat16 (wt 1) or float32 (3), its bias float32 [n] (or none)
pub struct Mat {
    pub w: DevBuf,
    pub wt: u8,
    pub bias: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Mat {
    pub fn bytes(&self) -> usize {
        self.w.len + self.bias.as_ref().map_or(0, |b| b.len)
    }
}

/// The last mark and each label's (seconds, calls)
type OpTimes = (Instant, BTreeMap<String, (f64, u32)>);

/// The engine's kernels on one GPU
pub struct Ops {
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
    /// NS_HARU_PROFILE=1: the frame's operations timed one by one (the GPU waited for after each), by label
    prof: Option<Mutex<OpTimes>>,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>, profile: bool) -> Result<Ops> {
        Ok(Ops { gpu: gpu.clone(), k: ffi::api()?, prof: profile.then(|| Mutex::new((Instant::now(), BTreeMap::new()))) })
    }

    pub fn profiling(&self) -> bool {
        self.prof.is_some()
    }

    /// NS_HARU_PROFILE=1: the time since the last mark (the GPU waited for) under `label()`
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

    /// NS_HARU_PROFILE=1: the marks' table on stderr, a frame's share (`frames`), and cleared
    pub fn report(&self, frames: usize) {
        if let Some(Ok(mut p)) = self.prof.as_ref().map(|p| p.lock()) {
            let mut rows: Vec<_> = p.1.iter().collect();
            rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            for (k, (t, n)) in rows {
                eprintln!("  {k:<44} {:>8.3} ms a frame {n:>7} calls {:>8.1} us each", 1e3 * t / frames.max(1) as f64, 1e6 * t / (*n).max(1) as f64);
            }
            p.1.clear();
        }
    }

    fn g(&self) -> *mut c_void {
        self.gpu.raw()
    }

    /// out[m] (epi) x[m] . W^T + bias for m < M <= 20 rows of x `ldx` floats apart, out rows `ldo` apart; `gamma` the
    /// layer scale of Epi::Res and its rows' distance (0: one for every row)
    pub fn gemv(&self, x: *const f32, m: usize, ldx: usize, w: &Mat, out: *mut f32, ldo: usize, epi: Epi, gamma: Option<(*const f32, usize)>) -> Result<()> {
        let bias = w.bias.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        let (gamma, ldg) = gamma.unwrap_or((std::ptr::null(), 0));
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gemv)(self.g(), x, m as i64, ldx as i64, w.k as i64, w.w.ptr(), w.wt as i32, std::ptr::null(), w.n as i64, bias, epi as i32,
                                           gamma, ldg as i64, out, ldo as i64) }, "gemv")
    }

    /// The product over m rows of x `ldx` apart (windows of a signal may overlap): the small-batch one for up to 8
    /// rows, the tiled one for more - bfloat16 weights on the matrix engines when `split` is given (x split into it)
    pub fn prod(&self, x: *const f32, m: usize, ldx: usize, w: &Mat, out: *mut f32, ldo: usize, epi: Epi, gamma: Option<(*const f32, usize)>,
                split: Option<&Split>) -> Result<()> {
        if m <= 8 && w.k.is_multiple_of(8) && ldx.is_multiple_of(4) {
            return self.gemv(x, m, ldx, w, out, ldo, epi, gamma);
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

    /// RMS norm of the rows of x [m, c] into out: x / rms * w (w None: 1); with `modulate` (scale, shift, their rows
    /// `lds` apart): then v * (1 + scale) + shift
    pub fn rms_norm(&self, x: *const f32, m: usize, c: usize, w: Option<&DevBuf>, eps: f32, modulate: Option<(*const f32, *const f32, usize)>, out: *mut f32)
                    -> Result<()> {
        let (sc, sh, lds) = modulate.unwrap_or((std::ptr::null(), std::ptr::null(), 0));
        // SAFETY: x and out m * c floats; w c; scale / shift m rows of c, lds apart.
        ffi::check(unsafe { (self.k.rms_norm)(self.g(), x, m as i64, c as i64, w.map_or(std::ptr::null(), |w| w.fp() as *const f32), eps, sc, sh, lds as i64, out) },
                   "rms norm")
    }

    /// cur [p + l, c] = prev's rows [prev_off, prev_off + p), then the l rows of x normed (times w)
    pub fn norm_hist(&self, prev: &DevBuf, prev_off: usize, x: *const f32, l: usize, c: usize, p: usize, w: &DevBuf, eps: f32, cur: &DevBuf) -> Result<()> {
        if prev.len < (prev_off + p) * c * 4 || cur.len < (p + l) * c * 4 {
            return Err(Error(format!("norm_hist: buffers of {} / {} bytes for {p} + {l} rows of {c}", prev.len, cur.len)));
        }
        // SAFETY: sizes checked above; x l * c floats, w c.
        ffi::check(unsafe { (self.k.norm_hist)(self.g(), prev.fp(), prev_off as i64, x, l as i64, c as i64, p as i64, w.fp(), eps, cur.fp()) }, "norm")
    }

    /// x[t] += gamma * depthwise conv (k taps over cur [k - 1 + l, c]); h[t] = norm(x[t]) * w2
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

    /// out rows (ldo apart) = the bfloat16 table's rows idx (device int32 [n])
    pub fn embed(&self, table: &DevBuf, d: usize, idx: &DevBuf, n: usize, out: *mut f32, ldo: usize) -> Result<()> {
        // SAFETY: table [V, d] bfloat16 with every id below V (the caller's), idx n int32, out n rows of d.
        ffi::check(unsafe { (self.k.embed)(self.g(), table.ptr(), d as i64, idx.ptr() as *const i32, n as i64, out, ldo as i64) }, "embed")
    }

    /// The DeltaNet's convolution over r rows of x (ldx apart, c channels, k taps), its carried rows in `state`
    pub fn gdn_conv(&self, x: *const f32, ldx: usize, r: usize, c: usize, k: usize, w: &DevBuf, state: &DevBuf, out: *mut f32) -> Result<()> {
        if state.len < (k - 1) * c * 4 || w.len < c * k * 4 {
            return Err(Error(format!("gdn conv: a state of {} bytes, weights of {} for {c} channels of {k}", state.len, w.len)));
        }
        // SAFETY: x r rows of c (ldx apart), out r * c floats (the caller's); state and w checked above.
        ffi::check(unsafe { (self.k.gdn_conv)(self.g(), x, ldx as i64, r as i64, c as i64, k as i64, w.fp(), state.fp(), out) }, "gdn conv")
    }

    /// The gated delta rule over r rows (haru.h)
    pub fn gdn_rule(&self, qkv: *const f32, ldq: usize, ba: *const f32, ldb: usize, a_log: &DevBuf, dt_bias: &DevBuf, state: &DevBuf, r: usize, hk: usize,
                    hv: usize, dk: usize, dv: usize, out: *mut f32) -> Result<()> {
        if state.len < hv * dk * dv * 4 {
            return Err(Error(format!("gdn rule: a state of {} bytes for {hv} heads of {dk} x {dv}", state.len)));
        }
        // SAFETY: the rows as the caller names them; the state checked above.
        ffi::check(unsafe { (self.k.gdn_rule)(self.g(), qkv, ldq as i64, ba, ldb as i64, a_log.fp(), dt_bias.fp(), state.fp(), r as i64, hk as i64, hv as i64,
                                              dk as i64, dv as i64, out) }, "gdn rule")
    }

    /// out = w * norm(y per head) * silu(z)
    pub fn gated_norm(&self, y: *const f32, z: *const f32, ldz: usize, w: &DevBuf, eps: f32, r: usize, h: usize, d: usize, out: *mut f32) -> Result<()> {
        // SAFETY: y and out r * h * d floats, z r rows of h * d (ldz apart), w d.
        ffi::check(unsafe { (self.k.gated_norm)(self.g(), y, z, ldz as i64, w.fp(), eps, r as i64, h as i64, d as i64, out) }, "gated norm")
    }

    /// Attention's queries, keys and values for r rows at p0 (haru.h's attn_prep)
    pub fn attn_prep(&self, qkv: *const f32, ldq: usize, r: usize, hq: usize, hkv: usize, d: usize, rot: usize, qw: &DevBuf, kw: &DevBuf, eps: f32,
                     inv: &DevBuf, p0: usize, q: *mut f32, kc: &DevBuf, vc: &DevBuf, t: usize) -> Result<()> {
        if kc.len.min(vc.len) < hkv * t * d * 2 {
            return Err(Error(format!("attention: a cache of {} bytes for {t} positions", kc.len.min(vc.len))));
        }
        // SAFETY: the rows as the caller names them; the cache checked above (and p0 + r <= t in the kernel).
        ffi::check(unsafe { (self.k.attn_prep)(self.g(), qkv, ldq as i64, r as i64, hq as i64, hkv as i64, d as i64, rot as i64, qw.fp(), kw.fp(), eps, inv.fp(),
                                               p0 as i64, q, kc.ptr(), vc.ptr(), t as i64) }, "attention prep")
    }

    /// The split attention and its gated merge: out [r, hq * d]; `part` holds r * hq * pieces * (d + 2) floats
    pub fn attn(&self, q: *const f32, kc: &DevBuf, vc: &DevBuf, r: usize, hq: usize, hkv: usize, d: usize, t: usize, p0: usize, chunk: usize, part: &DevBuf,
                gate: *const f32, ldg: usize, hs: usize, out: *mut f32) -> Result<()> {
        let pieces = (p0 + r).div_ceil(chunk);
        if part.len < r * hq * pieces * (d + 2) * 4 {
            return Err(Error(format!("attention: scratch of {} bytes for {r} rows in {pieces} pieces", part.len)));
        }
        // SAFETY: the sizes checked above and in the kernel (p0 + r <= t).
        ffi::check(unsafe { (self.k.attn)(self.g(), q, kc.ptr(), vc.ptr(), r as i64, hq as i64, hkv as i64, d as i64, t as i64, p0 as i64, chunk as i64,
                                          part.fp()) }, "attention")?;
        // SAFETY: as above; the gate rows the caller's.
        ffi::check(unsafe { (self.k.attn_merge)(self.g(), part.fp(), r as i64, hq as i64, d as i64, pieces as i64, gate, ldg as i64, hs as i64, out) },
                   "attention merge")
    }
}
