//! This engine's kernels on one GPU (`kernels/llm/deepseek4/silo/ds4.h`, and glm5next's generic `glm.h`): the checked
//! wrappers the engine calls.

use std::ffi::c_void;
use std::sync::Arc;

use nextsycl_core::{Api, DevBuf, Error, Gpu, Result};

use crate::ffi;

/// The kernels on one GPU (`kernels/ns`, and the engines' own: `kernels/<kind>/<arch>`). Every buffer is float32 unless named otherwise; the
/// sizes each kernel reads are checked against the buffers before the call.
pub struct Ops {
    pub gpu: Arc<Gpu>,
    /// the kernels (`ds4.h`, `glm.h`)
    k: &'static ffi::Api,
}

macro_rules! need {
    ($b:expr, $n:expr, $what:expr) => {
        if $b.floats() < ($n) as usize {
            return Err(Error(format!("{}: a buffer of {} floats where {} are needed", $what, $b.floats(), $n)));
        }
    };
}

#[allow(clippy::too_many_arguments)] // the kernels' own arity
impl Ops {
    pub fn new(gpu: Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { gpu, k: ffi::api()? })
    }
    fn ok(&self, rc: i32, what: &str) -> Result<()> {
        ffi::check(rc, what)
    }
    fn raw(&self) -> *mut c_void {
        self.gpu.raw()
    }
    /// the shared library's ABI (copies, tickets, timers)
    fn a(&self) -> &'static Api {
        self.k.shared
    }

    /// `n` values of ggml type `ty` stored at bytes `[at, at + bytes)` of `src` (`bytes` = what n values of the type
    /// take) into the front of `dst` as float32.
    pub fn dequant(&self, ty: u32, src: &DevBuf, at: usize, bytes: usize, n: usize, dst: &DevBuf) -> Result<()> {
        need!(dst, n, "dequant");
        src.bounds(at, bytes)?;
        // SAFETY: the byte range is inside src (checked) and holds n values of `ty` (the caller's `bytes`); dst
        // holds n floats (checked).
        let rc = unsafe { (self.k.dequant)(self.raw(), ty as i32, src.ptr().cast::<u8>().add(at).cast::<c_void>(), n, dst.fp()) };
        self.ok(rc, "dequant")
    }
    /// n values of a stored matrix (bytes [at, at + bytes) of src) expanded to fp16 into dst (2 bytes a value)
    pub fn dequant_f16(&self, ty: u32, src: &DevBuf, at: usize, bytes: usize, n: usize, dst: &DevBuf) -> Result<()> {
        src.bounds(at, bytes)?;
        dst.bounds(0, n * 2)?;
        // SAFETY: both ranges checked.
        let rc = unsafe { (self.k.dequant_f16)(self.raw(), ty as i32, src.ptr().cast::<u8>().add(at).cast::<c_void>(), n as i64, dst.ptr().cast()) };
        self.ok(rc, "dequant_f16")
    }
    /// A ticket for everything submitted to this GPU's queue so far
    pub fn mark(&self) -> Result<i64> {
        let mut t = 0i64;
        // SAFETY: t is a valid out pointer.
        let rc = unsafe { (self.a().mark)(self.raw(), &mut t) };
        self.ok(rc, "mark")?;
        Ok(t)
    }
    /// n bytes of src (from src_at) into dst (from at) on the GPU's copy queue, beside its work, after ticket
    /// `after`: the copy's ticket
    pub fn stream_copy(&self, dst: &DevBuf, at: usize, src: &DevBuf, src_at: usize, n: usize, after: Option<i64>) -> Result<i64> {
        self.stream_copy_on(0, dst, at, src, src_at, n, &[after])
    }
    /// `stream_copy` on copy lane `lane` (0 or 1: the two run side by side - one PCIe direction each), after every
    /// ticket of `after`
    #[allow(clippy::too_many_arguments)]
    pub fn stream_copy_on(&self, lane: i32, dst: &DevBuf, at: usize, src: &DevBuf, src_at: usize, n: usize, after: &[Option<i64>]) -> Result<i64> {
        let deps: Vec<i64> = after.iter().flatten().copied().collect();
        dst.bounds(at, n)?;
        src.bounds(src_at, n)?;
        let mut t = 0i64;
        // SAFETY: both ranges in bounds (checked); t is a valid out pointer.
        let rc = unsafe {
            (self.a().stream_copy)(self.raw(), dst.ptr().cast::<u8>().add(at).cast(), src.ptr().cast::<u8>().add(src_at).cast(), n, deps.as_ptr(),
                                   deps.len() as i32, lane, &mut t)
        };
        self.ok(rc, "stream_copy")?;
        Ok(t)
    }
    /// A device timestamp after everything queued so far (NS_PROFILE=gpu queues keep them): its ticket
    pub fn stamp(&self) -> Result<i64> {
        let mut t = 0i64;
        // SAFETY: t is a valid out pointer.
        let rc = unsafe { (self.a().stamp)(self.raw(), &mut t) };
        self.ok(rc, "stamp")?;
        Ok(t)
    }
    /// Seconds between two stamps (waits for them)
    pub fn elapsed(&self, t0: i64, t1: i64) -> Result<f64> {
        let mut ns = 0f64;
        // SAFETY: ns is a valid out pointer.
        let rc = unsafe { (self.a().elapsed)(self.raw(), t0, t1, &mut ns) };
        self.ok(rc, "elapsed")?;
        Ok(ns * 1e-9)
    }
    /// The GPU's queue waits, on the device, for ticket `t` (a copy, or a mark)
    pub fn await_ticket(&self, t: i64) -> Result<()> {
        // SAFETY: plain call.
        let rc = unsafe { (self.a().await_)(self.raw(), t) };
        self.ok(rc, "await")
    }
    /// x [n] float32 -> y [n] fp16
    pub fn to_f16(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<()> {
        need!(x, n, "to_f16 x");
        y.bounds(0, n * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.to_f16)(self.raw(), x.fp(), y.ptr().cast(), n as i64) };
        self.ok(rc, "to_f16")
    }
    /// y [t, n] (+)= x [t, k] . w [n, k]^T, fp16 x (from value `x.1`, rows `x.2` apart) and w (from value `w.1`),
    /// float32 y (from float `y.1`, rows `y.2` apart)
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(&self, t: usize, n: usize, k: usize, x: (&DevBuf, usize, usize), w: (&DevBuf, usize), y: (&DevBuf, usize, usize), acc: bool) -> Result<()> {
        if t == 0 || n == 0 {
            return Ok(());
        }
        x.0.bounds(x.1 * 2, ((t - 1) * x.2 + k) * 2)?;
        w.0.bounds(w.1 * 2, n * k * 2)?;
        need!(y.0, y.1 + (t - 1) * y.2 + n, "gemm_f16 y");
        // SAFETY: extents checked.
        let rc = unsafe {
            (self.k.gemm_f16)(self.raw(), t as i64, n as i64, k as i64, x.0.ptr().cast::<u16>().add(x.1), x.2 as i64, w.0.ptr().cast::<u16>().add(w.1),
                                y.0.fp().add(y.1), y.2 as i64, acc as i32)
        };
        self.ok(rc, "gemm_f16")
    }
    /// y [t, n] (+)= x [t, k] . w [n, k]^T, all contiguous
    pub fn gemm(&self, t: usize, n: usize, k: usize, x: &DevBuf, w: &DevBuf, y: &DevBuf, acc: bool) -> Result<()> {
        self.gemm_at(t, n, k, (x, 0, k), (w, 0), (y, 0, n), acc)
    }
    /// On slices: x from float `x.1` with rows `x.2` apart; w from float `w.1` ([n, k] contiguous); y from float
    /// `y.1` with rows `y.2` apart. Every slice's extent is checked against its buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_at(&self, t: usize, n: usize, k: usize, x: (&DevBuf, usize, usize), w: (&DevBuf, usize), y: (&DevBuf, usize, usize), acc: bool) -> Result<()> {
        if t == 0 || n == 0 {
            return Ok(());
        }
        need!(x.0, x.1 + (t - 1) * x.2 + k, "gemm x");
        need!(w.0, w.1 + n * k, "gemm w");
        need!(y.0, y.1 + (t - 1) * y.2 + n, "gemm y");
        if x.2 < k || y.2 < n {
            return Err(Error(format!("gemm: row strides {} / {} below the widths {k} / {n}", x.2, y.2)));
        }
        // SAFETY: device pointers of this GPU at offsets inside their buffers (extents checked above).
        let rc = unsafe {
            (self.k.gemm)(self.raw(), t as i64, n as i64, k as i64, x.0.fp().add(x.1), x.2 as i64, w.0.fp().add(w.1), y.0.fp().add(y.1),
                            y.2 as i64, acc as i32)
        };
        self.ok(rc, "gemm")
    }
    /// `batch` products on strided slices (in floats): x from x.1, rows x.2 apart, batch b at + b * x.3; w from w.1,
    /// [n, k] contiguous, batch b at + b * w.2; y as x.
    pub fn gemm_batch(&self, batch: usize, t: usize, n: usize, k: usize, x: (&DevBuf, usize, usize, usize), w: (&DevBuf, usize, usize),
                      y: (&DevBuf, usize, usize, usize), acc: bool) -> Result<()> {
        if batch == 0 || t == 0 || n == 0 {
            return Ok(());
        }
        let last = batch - 1;
        need!(x.0, x.1 + last * x.3 + (t - 1) * x.2 + k, "gemm_batch x");
        need!(w.0, w.1 + last * w.2 + n * k, "gemm_batch w");
        need!(y.0, y.1 + last * y.3 + (t - 1) * y.2 + n, "gemm_batch y");
        // SAFETY: every batch's extent checked above.
        let rc = unsafe {
            (self.k.gemm_batch)(self.raw(), batch as i64, t as i64, n as i64, k as i64, x.0.fp().add(x.1), x.2 as i64, x.3 as i64, w.0.fp().add(w.1),
                                  w.2 as i64, y.0.fp().add(y.1), y.2 as i64, y.3 as i64, acc as i32)
        };
        self.ok(rc, "gemm_batch")
    }
    /// m [t, 24] = fn [24, n] . rms(x [t, n]); `part` scratch of t * 32 * 25 floats
    pub fn hc_mix(&self, x: &DevBuf, fn_: &DevBuf, m: &DevBuf, part: &DevBuf, t: usize, n: usize, eps: f32) -> Result<()> {
        need!(x, t * n, "hc_mix x");
        need!(fn_, 24 * n, "hc_mix fn");
        need!(m, t * 24, "hc_mix m");
        need!(part, t * 32 * 25, "hc_mix scratch");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.hc_mix)(self.raw(), x.fp(), fn_.fp(), m.fp(), part.fp(), t as i64, n as i64, eps) };
        self.ok(rc, "hc_mix")
    }
    pub fn rms_norm(&self, x: &DevBuf, w: Option<&DevBuf>, y: &DevBuf, rows: usize, c: usize, eps: f32) -> Result<()> {
        need!(x, rows * c, "rms_norm x");
        need!(y, rows * c, "rms_norm y");
        if let Some(w) = w {
            need!(w, c, "rms_norm w");
        }
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.map_or(std::ptr::null(), |w| w.fp()), y.fp(), rows as i64, c as i64, eps) };
        self.ok(rc, "rms_norm")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre(&self, m: &DevBuf, scale: &DevBuf, base: &DevBuf, x: &DevBuf, h: &DevBuf, post: &DevBuf, comb: &DevBuf, pre: &DevBuf, t: usize, c: usize,
                  eps: f32, iters: u32) -> Result<()> {
        need!(pre, t * 4, "hc_pre pre");
        need!(m, t * 24, "hc_pre m");
        need!(scale, 3, "hc_pre scale");
        need!(base, 24, "hc_pre base");
        need!(x, t * 4 * c, "hc_pre X");
        need!(h, t * c, "hc_pre h");
        need!(post, t * 4, "hc_pre post");
        need!(comb, t * 16, "hc_pre comb");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.hc_pre)(self.raw(), m.fp(), scale.fp(), base.fp(), x.fp(), h.fp(), post.fp(), comb.fp(), pre.fp(), t as i64, c as i64, eps, iters as i32) };
        self.ok(rc, "hc_pre")
    }
    /// Decode widths: hc_mix + hc_pre + rms_norm(h) * nw in one launch (fn [24, 4c] float32)
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_fused(&self, x: &DevBuf, fn_: &DevBuf, scale: &DevBuf, base: &DevBuf, nw: &DevBuf, h: &DevBuf, post: &DevBuf, comb: &DevBuf,
                        pre: &DevBuf, normed: &DevBuf, part: &DevBuf, t: usize, c: usize, eps: f32, hc_eps: f32, iters: u32) -> Result<()> {
        need!(part, t * 32 * 25, "hc_pre_fused scratch");
        need!(x, t * 4 * c, "hc_pre_fused X");
        need!(fn_, 24 * 4 * c, "hc_pre_fused fn");
        need!(scale, 3, "hc_pre_fused scale");
        need!(base, 24, "hc_pre_fused base");
        need!(nw, c, "hc_pre_fused norm");
        need!(h, t * c, "hc_pre_fused h");
        need!(normed, t * c, "hc_pre_fused normed");
        need!(post, t * 4, "hc_pre_fused post");
        need!(comb, t * 16, "hc_pre_fused comb");
        need!(pre, t * 4, "hc_pre_fused pre");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.hc_pre_fused)(self.raw(), x.fp(), fn_.fp(), scale.fp(), base.fp(), nw.fp(), h.fp(), post.fp(), comb.fp(), pre.fp(), normed.fp(),
                                    part.fp(), t as i64, c as i64, eps, hc_eps, iters as i32)
        };
        self.ok(rc, "hc_pre_fused")
    }
    pub fn hc_post(&self, y: &DevBuf, x: &DevBuf, post: &DevBuf, comb: &DevBuf, xo: &DevBuf, t: usize, c: usize) -> Result<()> {
        need!(y, t * c, "hc_post y");
        need!(x, t * 4 * c, "hc_post X");
        need!(xo, t * 4 * c, "hc_post Xo");
        need!(post, t * 4, "hc_post post");
        need!(comb, t * 16, "hc_post comb");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.hc_post)(self.raw(), y.fp(), x.fp(), post.fp(), comb.fp(), xo.fp(), t as i64, c as i64) };
        self.ok(rc, "hc_post")
    }
    pub fn hc_mean(&self, x: &DevBuf, y: &DevBuf, t: usize, c: usize) -> Result<()> {
        need!(x, t * 4 * c, "hc_mean X");
        need!(y, t * c, "hc_mean y");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.hc_mean)(self.raw(), x.fp(), y.fp(), t as i64, c as i64) };
        self.ok(rc, "hc_mean")
    }
    /// out [t, f] (fp16) = swiglu of gu [t, 2f] (each row's gate, then its up), clamped like swiglu_clamp
    pub fn swiglu_gu_f16(&self, gu: &DevBuf, out: &DevBuf, t: usize, f: usize, limit: f32) -> Result<()> {
        need!(gu, t * 2 * f, "swiglu_gu gu");
        out.bounds(0, t * f * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.swiglu_gu_f16)(self.raw(), gu.fp(), out.ptr().cast(), t as i64, f as i64, limit) };
        self.ok(rc, "swiglu_gu_f16")
    }
    pub fn swiglu_clamp(&self, gate: &DevBuf, up: &DevBuf, out: &DevBuf, n: usize, limit: f32) -> Result<()> {
        need!(gate, n, "swiglu gate");
        need!(up, n, "swiglu up");
        need!(out, n, "swiglu out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.swiglu_clamp)(self.raw(), gate.fp(), up.fp(), out.fp(), n as i64, limit) };
        self.ok(rc, "swiglu_clamp")
    }
    /// score[r][j0 + jj] for rows [t] and pools [j0, j0 + n) from S [t * h, n] and w [t, h]
    #[allow(clippy::too_many_arguments)]
    pub fn idx_score(&self, s: &DevBuf, w: &DevBuf, score: &DevBuf, t: usize, h: usize, j0: usize, n: usize, ld: usize, pos0: usize) -> Result<()> {
        need!(s, t * h * n, "idx_score S");
        need!(w, t * h, "idx_score w");
        need!(score, (t - 1) * ld + j0 + n, "idx_score score");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.idx_score)(self.raw(), s.fp(), w.fp(), score.fp(), t as i64, h as i64, j0 as i64, n as i64, ld as i64, pos0 as i64) };
        self.ok(rc, "idx_score")
    }
    /// batch products in fp16, float32 out: y_b [t, n] = x_b [t, k] . (w_b^T with `trans_w`: w_b [n, k]; else w_b
    /// [k, n]). x and w (buffer, offset, row stride, batch stride) in halves; y in floats.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batch_h(&self, batch: usize, trans_w: bool, t: usize, n: usize, k: usize, x: (&DevBuf, usize, usize, usize),
                        w: (&DevBuf, usize, usize, usize), y: (&DevBuf, usize, usize, usize)) -> Result<()> {
        if batch == 0 || t == 0 || n == 0 {
            return Ok(());
        }
        let last = batch - 1;
        let (wr, wc) = if trans_w { (n, k) } else { (k, n) };
        x.0.bounds(0, (x.1 + last * x.3 + (t - 1) * x.2 + k) * 2)?;
        w.0.bounds(0, (w.1 + last * w.3 + (wr - 1) * w.2 + wc) * 2)?;
        need!(y.0, y.1 + last * y.3 + (t - 1) * y.2 + n, "gemm_batch_h y");
        // SAFETY: every batch's extent checked above.
        let rc = unsafe {
            (self.k.gemm_batch_h)(self.raw(), batch as i64, trans_w as i32, t as i64, n as i64, k as i64, x.0.ptr().cast::<u16>().add(x.1), x.2 as i64,
                                    x.3 as i64, w.0.ptr().cast::<u16>().add(w.1), w.2 as i64, w.3 as i64, y.0.fp().add(y.1), y.2 as i64, y.3 as i64, 0)
        };
        self.ok(rc, "gemm_batch_h")
    }
    /// An expert's gate | up for m tokens (x [m, k] fp16, m a multiple of 32): out [m, n] = x . w^T with w IQ2_XXS
    /// (n rows of k), decoded inside the matrix-engine GEMM
    pub fn moe_fused_gu(&self, x: &DevBuf, w: &DevBuf, w_at: usize, out: &DevBuf, m: usize, n: usize, k: usize) -> Result<()> {
        x.bounds(0, m * k * 2)?;
        w.bounds(w_at, n * k / 256 * 66)?;
        need!(out, m * n, "moe_fused_gu out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.moe_fused_gu)(self.raw(), x.ptr().cast(), w.ptr().cast::<u8>().add(w_at).cast(), out.fp(), m as i64, n as i64, k as i64) };
        self.ok(rc, "moe_fused_gu")
    }
    /// The most rows `moe_fused_gu_esimd` takes on this GPU (0: none)
    pub fn moe_esimd_gu_max(&self) -> usize {
        // SAFETY: a live handle.
        let r = unsafe { (self.k.moe_esimd_gu_max)(self.raw()) };
        r.max(0) as usize
    }
    /// `moe_fused_gu` in ESIMD: m one of 32, 64, 128, 256
    pub fn moe_fused_gu_esimd(&self, x: &DevBuf, w: &DevBuf, w_at: usize, out: &DevBuf, m: usize, n: usize, k: usize) -> Result<()> {
        x.bounds(0, m * k * 2)?;
        w.bounds(w_at, n * k / 256 * 66)?;
        need!(out, m * n, "moe_fused_gu_esimd out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.moe_fused_gu_esimd)(self.raw(), x.ptr().cast(), w.ptr().cast::<u8>().add(w_at).cast(), out.fp(), m as i64, n as i64, k as i64) };
        self.ok(rc, "moe_fused_gu_esimd")
    }
    /// An expert's down for m tokens (x [m, k] fp16, the activations): out [m, n] = x . w^T with w Q2_K (n rows of
    /// k), decoded inside the matrix-engine GEMM
    pub fn moe_fused_down(&self, x: &DevBuf, w: &DevBuf, w_at: usize, out: &DevBuf, m: usize, n: usize, k: usize) -> Result<()> {
        x.bounds(0, m * k * 2)?;
        w.bounds(w_at, n * k / 256 * 84)?;
        need!(out, m * n, "moe_fused_down out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.moe_fused_down)(self.raw(), x.ptr().cast(), w.ptr().cast::<u8>().add(w_at).cast(), out.fp(), m as i64, n as i64, k as i64) };
        self.ok(rc, "moe_fused_down")
    }
    /// sel [t, k] (int32): the k highest of each score row [0, n), rows ld apart
    pub fn topk(&self, score: &DevBuf, sel: &DevBuf, t: usize, n: usize, ld: usize, k: usize) -> Result<()> {
        need!(score, (t - 1) * ld + n, "topk score");
        need!(sel, t * k, "topk sel");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.topk)(self.raw(), score.fp(), sel.ptr().cast(), t as i64, n as i64, ld as i64, k as i64) };
        self.ok(rc, "topk")
    }
    /// y[idx[i]] += w[i] * src[i] (rows of c); idx and w on the device (`n` int32 / float32)
    pub fn scatter_add(&self, y: &DevBuf, src: &DevBuf, idx: &DevBuf, w: &DevBuf, n: usize, c: usize) -> Result<()> {
        need!(src, n * c, "scatter src");
        need!(idx, n, "scatter idx");
        need!(w, n, "scatter w");
        // SAFETY: sizes checked; the indices are the caller's (tokens of this batch, below y's rows).
        let rc = unsafe { (self.k.scatter_add)(self.raw(), y.fp(), src.fp(), idx.ptr().cast(), w.fp(), n as i64, c as i64) };
        self.ok(rc, "scatter_add")
    }
    pub fn gather_f16(&self, src: &DevBuf, idx: &DevBuf, out: &DevBuf, n: usize, c: usize) -> Result<()> {
        idx.bounds(0, n * 4)?;
        out.bounds(0, n * c * 2)?;
        // SAFETY: sizes checked; indices are rows of src (the caller's).
        let rc = unsafe { (self.k.gather_f16)(self.raw(), src.fp(), idx.ptr().cast(), out.ptr().cast(), n as i64, c as i64) };
        self.ok(rc, "gather_f16")
    }
    pub fn gather(&self, src: &DevBuf, idx: &DevBuf, out: &DevBuf, n: usize, c: usize) -> Result<()> {
        need!(idx, n, "gather idx");
        need!(out, n * c, "gather out");
        // SAFETY: sizes checked; indices are rows of src (the caller's).
        let rc = unsafe { (self.k.gather)(self.raw(), src.fp(), idx.ptr().cast(), out.fp(), n as i64, c as i64) };
        self.ok(rc, "gather")
    }
    /// Bytes of Q8_1 for `ncols` columns of `n_in`.
    pub fn q8_1_bytes(&self, n_in: usize, ncols: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.q8_1_bytes)(n_in as i64, ncols as i64) }
    }
    pub fn mmvq_supported(&self, ty: u32) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.k.mmvq_supported)(ty as i32) != 0 }
    }
    /// x [ncols, n_in] (floats from `x.1`) -> Q8_1 into `q`.
    pub fn quantize_q8_1(&self, x: (&DevBuf, usize), q: &DevBuf, n_in: usize, ncols: usize) -> Result<()> {
        need!(x.0, x.1 + n_in * ncols, "quantize x");
        if q.len < self.q8_1_bytes(n_in, ncols) {
            return Err(Error(format!("quantize: {} bytes for {} of Q8_1", q.len, self.q8_1_bytes(n_in, ncols))));
        }
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.quantize_q8_1)(self.raw(), x.0.fp().add(x.1), q.ptr(), n_in as i64, ncols as i64) };
        self.ok(rc, "quantize_q8_1")
    }
    /// y [ncols, n_out] (floats from `y.1`) = W . x: W the stored blocks at bytes `w.1..w.1 + w_bytes` of `w.0`.
    pub fn mmvq(&self, ty: u32, w: (&DevBuf, usize), w_bytes: usize, q: &DevBuf, y: (&DevBuf, usize), n_in: usize, n_out: usize, ncols: usize) -> Result<()> {
        w.0.bounds(w.1, w_bytes)?;
        if q.len < self.q8_1_bytes(n_in, ncols) {
            return Err(Error("mmvq: the Q8_1 input is short".into()));
        }
        need!(y.0, y.1 + n_out * ncols, "mmvq y");
        // SAFETY: sizes checked (w_bytes is the caller's count of the matrix's stored bytes).
        let rc = unsafe {
            (self.k.mmvq)(self.raw(), ty as i32, w.0.ptr().cast::<u8>().add(w.1).cast(), q.ptr(), y.0.fp().add(y.1), n_in as i64, n_out as i64, ncols as i64)
        };
        self.ok(rc, "mmvq")
    }
    /// y [t, c] += per token the weighted sum of its entries' rows: `ints` holds t_ptr [t + 1] then ent [entries]
    /// (int32), `w` [entries]
    pub fn moe_combine(&self, y: &DevBuf, rows: &DevBuf, ints: &DevBuf, w: &DevBuf, t: usize, entries: usize, c: usize) -> Result<()> {
        need!(y, t * c, "combine y");
        need!(rows, entries * c, "combine rows");
        need!(ints, t + 1 + entries, "combine index");
        need!(w, entries, "combine w");
        // SAFETY: sizes checked; the index values are the caller's (rows of `rows`, prefix sums within entries).
        let rc = unsafe {
            (self.k.moe_combine)(self.raw(), y.fp(), rows.fp(), ints.ptr().cast(), ints.ptr().cast::<i32>().add(t + 1), w.fp(), t as i64, c as i64)
        };
        self.ok(rc, "moe_combine")
    }
    pub fn moe_grouped_supported(&self, gu: u32, down: u32, n_embd: usize, n_ff: usize) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.k.moe_grouped_supported)(gu as i32, down as i32, n_embd as i64, n_ff as i64) != 0 }
    }
    pub fn moe_scratch_bytes(&self, entries: usize, n_ff: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.moe_scratch_bytes)(entries as i64, n_ff as i64) }
    }
    /// A layer's experts in two launches. `table` holds, as u64: the groups' blob pointers [groups]; then as i32:
    /// grp_start [groups + 1], n_groups [1], ent_dst [entries], ent_tok [entries] (at the offsets returned by
    /// `moe_table`). `xq` the tokens' Q8_1, `out` [entries, n_embd] (unweighted rows).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped(&self, gu: u32, down: u32, n_embd: usize, n_ff: usize, table: &DevBuf, groups: usize, entries: usize, xq: &DevBuf, scratch: &DevBuf,
                       out: &DevBuf, limit: f32, lanes: usize, phase: i32) -> Result<()> {
        let ints = groups * 2 + (groups + 1) + 1 + 2 * entries; // the u64 pointers as two i32 each, then the i32 tables
        if table.len < ints * 4 || scratch.len < self.moe_scratch_bytes(entries, n_ff) {
            return Err(Error("moe_grouped: the table or the scratch is short".into()));
        }
        need!(out, entries * n_embd, "moe_grouped out");
        let base = table.ptr().cast::<u8>();
        // SAFETY: the table's layout as written by the caller (above), sizes checked; the blob pointers are the
        // caller's (VRAM slots of this GPU holding the experts).
        let rc = unsafe {
            let ptrs = base.cast::<u64>();
            let i32s = base.add(groups * 8).cast::<i32>();
            let (gs, ng) = (i32s, i32s.add(groups + 1));
            let (dst, tok) = (ng.add(1), ng.add(1 + entries));
            (self.k.moe_grouped)(self.raw(), gu as i32, down as i32, n_embd as i64, n_ff as i64, ptrs, gs, ng, dst, tok, groups as i64, entries as i64,
                                   xq.ptr(), scratch.ptr(), out.fp(), limit, lanes as i32, phase)
        };
        self.ok(rc, "moe_grouped")
    }
    pub fn add(&self, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        need!(y, n, "add y");
        need!(x, n, "add x");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.add)(self.raw(), y.fp(), x.fp(), n as i64) };
        self.ok(rc, "add")
    }

    // ---- ds4.h

    /// YaRN RoPE in place on x: t rows `row` floats apart, h heads `head` apart, `n_dims` values from float `off` of each
    /// head; row i at position pos0 + i * pstep; `inverse`: rotated back
    pub fn rope(&self, x: &DevBuf, t: usize, h: usize, row: usize, head: usize, off: usize, n_dims: usize, pos0: usize, pstep: usize,
                r: crate::model::Rope, inverse: bool) -> Result<()> {
        if t == 0 {
            return Ok(());
        }
        need!(x, (t - 1) * row + (h - 1) * head + off + n_dims, "rope x");
        // SAFETY: the last value rotated is inside x (checked).
        let rc = unsafe {
            (self.k.ds4_rope)(self.raw(), x.fp(), t as i64, h as i64, row as i64, head as i64, off as i64, n_dims as i64, pos0 as i64, pstep as i64,
                              r.base, r.freq_scale, r.ext_factor, r.mscale, r.corr0, r.corr1, inverse as i32)
        };
        self.ok(rc, "ds4_rope")
    }
    /// The compressor's blocks completed by tokens [pos0, pos0 + t): kv, sc [t, w] (w = d, or 2d with `overlap`), ring
    /// [ring_n][2w], ape [ratio][w] -> out [blocks][d]
    pub fn pool(&self, ring: &DevBuf, kv: &DevBuf, sc: &DevBuf, ape: &DevBuf, out: &DevBuf, pos0: usize, t: usize, d: usize, ratio: usize, overlap: bool,
                ring_n: usize) -> Result<()> {
        let w = if overlap { 2 * d } else { d };
        need!(ring, ring_n * 2 * w, "pool ring");
        need!(kv, t * w, "pool kv");
        need!(sc, t * w, "pool sc");
        need!(ape, ratio * w, "pool ape");
        let blocks = (pos0 + t) / ratio - pos0 / ratio;
        need!(out, blocks * d, "pool out");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.ds4_pool)(self.raw(), ring.fp(), kv.fp(), sc.fp(), ape.fp(), out.fp(), pos0 as i64, t as i64, d as i64, ratio as i64, overlap as i32,
                              ring_n as i64)
        };
        self.ok(rc, "ds4_pool")
    }
    /// kv [t, d] into the raw keys' ring (fp16 [r][d]) at positions pos0..
    pub fn ring_put(&self, kv: &DevBuf, ring: &DevBuf, t: usize, pos0: usize, r: usize, d: usize) -> Result<()> {
        need!(kv, t * d, "ring_put kv");
        ring.bounds(0, r * d * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds4_ring_put)(self.raw(), kv.fp(), ring.ptr().cast(), t as i64, pos0 as i64, r as i64, d as i64) };
        self.ok(rc, "ds4_ring_put")
    }
    /// x [n, d] -> fp16 rows j0.. of `cache`
    pub fn rows_f16(&self, x: &DevBuf, cache: &DevBuf, n: usize, j0: usize, d: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        need!(x, n * d, "rows_f16 x");
        cache.bounds(0, (j0 + n) * d * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds4_rows_f16)(self.raw(), x.fp(), cache.ptr().cast(), n as i64, j0 as i64, d as i64) };
        self.ok(rc, "ds4_rows_f16")
    }
    /// Attention of q [t, h, 512] over each row's window of raw keys (ring [r][512] fp16) and its compressed blocks
    /// (comp: (cache, ratio, rows it holds); sel: [t][k] the indexer's picks) -> out [t, h, 512]
    pub fn attend(&self, q: &DevBuf, raw: &DevBuf, r: usize, comp: Option<(&DevBuf, usize, usize)>, sel: Option<(&DevBuf, usize)>, sinks: &DevBuf, out: &DevBuf,
                  t: usize, h: usize, pos0: usize, window: usize, scale: f32) -> Result<()> {
        const D: usize = 512;
        need!(q, t * h * D, "attend q");
        need!(out, t * h * D, "attend out");
        need!(sinks, h, "attend sinks");
        raw.bounds(0, r * D * 2)?;
        if let Some((c, ratio, rows)) = comp {
            if (pos0 + t) / ratio > rows {
                return Err(Error(format!("attend: {} blocks seen, the cache holds {rows}", (pos0 + t) / ratio)));
            }
            c.bounds(0, rows * D * 2)?;
        }
        if let Some((s, k)) = sel {
            s.bounds(0, t * k * 4)?;
        }
        let (cp, ratio) = comp.map_or((std::ptr::null(), 1), |(c, ratio, _)| (c.ptr().cast_const().cast(), ratio));
        let (sp, k) = sel.map_or((std::ptr::null(), 0), |(s, k)| (s.ptr().cast_const().cast(), k));
        // SAFETY: sizes checked; the picks are the indexer's (blocks below what each row sees).
        let rc = unsafe {
            (self.k.ds4_attend)(self.raw(), q.fp(), raw.ptr().cast(), r as i64, cp, ratio as i64, sp, k as i64, sinks.fp(), out.fp(), t as i64, h as i64,
                                pos0 as i64, window as i64, scale)
        };
        self.ok(rc, "ds4_attend")
    }
    /// The head's mHC: m [t, 4], X [t, 4, c] -> h [t, c]
    pub fn hc_head(&self, m: &DevBuf, scale: &DevBuf, base: &DevBuf, x: &DevBuf, h: &DevBuf, t: usize, c: usize, eps: f32) -> Result<()> {
        need!(m, t * 4, "hc_head m");
        need!(scale, 1, "hc_head scale");
        need!(base, 4, "hc_head base");
        need!(x, t * 4 * c, "hc_head X");
        need!(h, t * c, "hc_head h");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds4_hc_head)(self.raw(), m.fp(), scale.fp(), base.fp(), x.fp(), h.fp(), t as i64, c as i64, eps) };
        self.ok(rc, "ds4_hc_head")
    }
    /// dst [b][a][c] = src [a][b][c]
    pub fn permute(&self, src: &DevBuf, dst: &DevBuf, a: usize, b: usize, c: usize) -> Result<()> {
        need!(src, a * b * c, "permute src");
        need!(dst, a * b * c, "permute dst");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds4_permute)(self.raw(), src.fp(), dst.fp(), a as i64, b as i64, c as i64) };
        self.ok(rc, "ds4_permute")
    }
    /// n MXFP4 values at bytes [at, at + n / 32 * 17) of src -> float32 / fp16 dst
    pub fn mxfp4_dequant(&self, src: &DevBuf, at: usize, n: usize, dst: &DevBuf, f16: bool) -> Result<()> {
        src.bounds(at, n / 32 * 17)?;
        // SAFETY: the source range checked; dst below.
        let rc = if f16 {
            dst.bounds(0, n * 2)?;
            unsafe { (self.k.ds4_mxfp4_dequant_f16)(self.raw(), src.ptr().cast::<u8>().add(at).cast(), n as i64, dst.ptr().cast()) }
        } else {
            need!(dst, n, "mxfp4 dst");
            unsafe { (self.k.ds4_mxfp4_dequant)(self.raw(), src.ptr().cast::<u8>().add(at).cast(), n as i64, dst.fp()) }
        };
        self.ok(rc, "ds4_mxfp4_dequant")
    }
    /// y [ncols, n_out] (floats from `y.1`) = W . x for MXFP4 W at bytes w.1.. of w.0, x the Q8_1 of ncols rows
    pub fn mxfp4_mmvq(&self, w: (&DevBuf, usize), q: &DevBuf, y: (&DevBuf, usize), n_in: usize, n_out: usize, ncols: usize) -> Result<()> {
        w.0.bounds(w.1, n_out * n_in / 32 * 17)?;
        if q.len < self.q8_1_bytes(n_in, ncols) {
            return Err(Error("mxfp4_mmvq: the Q8_1 input is short".into()));
        }
        need!(y.0, y.1 + n_out * ncols, "mxfp4_mmvq y");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.ds4_mxfp4_mmvq)(self.raw(), w.0.ptr().cast::<u8>().add(w.1).cast(), q.ptr(), y.0.fp().add(y.1), n_in as i64, n_out as i64, ncols as i64)
        };
        self.ok(rc, "ds4_mxfp4_mmvq")
    }
    /// Bytes of the split Q8 layout for `ncols` columns of `n_in`
    pub fn q8_bytes(&self, n_in: usize, ncols: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.ds4_q8_bytes)(n_in as i64, ncols as i64) }
    }
    /// x [ncols, n_in] (floats from `x.1`) -> the split Q8 layout
    pub fn quantize_q8(&self, x: (&DevBuf, usize), xs: &DevBuf, n_in: usize, ncols: usize) -> Result<()> {
        need!(x.0, x.1 + n_in * ncols, "quantize_q8 x");
        xs.bounds(0, self.q8_bytes(n_in, ncols))?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds4_quantize_q8)(self.raw(), x.0.fp().add(x.1), xs.ptr(), n_in as i64, ncols as i64) };
        self.ok(rc, "ds4_quantize_q8")
    }
    pub fn multi_supported(&self, ty: u32, n_in: usize) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.k.ds4_multi_supported)(ty as i32, n_in as i64) != 0 }
    }
    /// y [entries, n_out] (floats from `y.1`): entry e the matrix at the e-th pointer of `table` (u64 [entries], then
    /// i32 cols [entries]) against that column of xs (`ncols` columns of n_in)
    pub fn mmvq_multi(&self, ty: u32, table: &DevBuf, entries: usize, xs: &DevBuf, ncols: usize, y: (&DevBuf, usize), n_in: usize, n_out: usize) -> Result<()> {
        if entries == 0 {
            return Ok(());
        }
        table.bounds(0, entries * 12)?;
        xs.bounds(0, self.q8_bytes(n_in, ncols))?;
        need!(y.0, y.1 + entries * n_out, "mmvq_multi y");
        // SAFETY: sizes checked; the pointers are the caller's (VRAM slots of this GPU holding the matrices).
        let rc = unsafe {
            let base = table.ptr().cast::<u8>();
            (self.k.ds4_mmvq_multi)(self.raw(), ty as i32, base.cast(), base.add(entries * 8).cast(), entries as i64, xs.ptr(), ncols as i64, y.0.fp().add(y.1),
                                    n_in as i64, n_out as i64)
        };
        self.ok(rc, "ds4_mmvq_multi")
    }
    /// n bytes of src (from src_at) into dst (from at) by a kernel in queue order (not a copy engine): its ticket
    pub fn kcopy(&self, dst: &DevBuf, at: usize, src: &DevBuf, src_at: usize, n: usize) -> Result<i64> {
        dst.bounds(at, n)?;
        src.bounds(src_at, n)?;
        let mut t = 0i64;
        // SAFETY: both ranges in bounds (checked); t is a valid out pointer.
        let rc = unsafe { (self.k.ds4_kcopy)(self.raw(), dst.ptr().cast::<u8>().add(at).cast(), src.ptr().cast::<u8>().add(src_at).cast(), n, &mut t) };
        self.ok(rc, "ds4_kcopy")?;
        Ok(t)
    }
    /// The host waits for ticket t (the queue goes on)
    pub fn ticket_wait(&self, t: i64) -> Result<()> {
        // SAFETY: plain call.
        let rc = unsafe { (self.k.ds4_ticket_wait)(self.raw(), t) };
        self.ok(rc, "ds4_ticket_wait")
    }
}
