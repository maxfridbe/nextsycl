//! This engine's kernels on one GPU (`kernels/llm/deepseek41/silo/ds41.h`, and glm5next's generic `glm.h`): the checked
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
            (self.k.ds41_rope)(self.raw(), x.fp(), t as i64, h as i64, row as i64, head as i64, off as i64, n_dims as i64, pos0 as i64, pstep as i64,
                              r.base, r.freq_scale, r.ext_factor, r.mscale, r.corr0, r.corr1, inverse as i32)
        };
        self.ok(rc, "ds41_rope")
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
            (self.k.ds41_pool)(self.raw(), ring.fp(), kv.fp(), sc.fp(), ape.fp(), out.fp(), pos0 as i64, t as i64, d as i64, ratio as i64, overlap as i32,
                              ring_n as i64)
        };
        self.ok(rc, "ds41_pool")
    }
    /// kv [t, d] into the raw keys' ring (fp16 [r][d]) at positions pos0..
    pub fn ring_put(&self, kv: &DevBuf, ring: &DevBuf, t: usize, pos0: usize, r: usize, d: usize) -> Result<()> {
        need!(kv, t * d, "ring_put kv");
        ring.bounds(0, r * d * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_ring_put)(self.raw(), kv.fp(), ring.ptr().cast(), t as i64, pos0 as i64, r as i64, d as i64) };
        self.ok(rc, "ds41_ring_put")
    }
    /// x [n, d] -> fp16 rows j0.. of `cache`
    pub fn rows_f16(&self, x: &DevBuf, cache: &DevBuf, n: usize, j0: usize, d: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        need!(x, n * d, "rows_f16 x");
        cache.bounds(0, (j0 + n) * d * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_rows_f16)(self.raw(), x.fp(), cache.ptr().cast(), n as i64, j0 as i64, d as i64) };
        self.ok(rc, "ds41_rows_f16")
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
            (self.k.ds41_attend)(self.raw(), q.fp(), raw.ptr().cast(), r as i64, cp, ratio as i64, sp, k as i64, sinks.fp(), out.fp(), t as i64, h as i64,
                                pos0 as i64, window as i64, scale)
        };
        self.ok(rc, "ds41_attend")
    }
    /// dst [b][a][c] = src [a][b][c]
    pub fn permute(&self, src: &DevBuf, dst: &DevBuf, a: usize, b: usize, c: usize) -> Result<()> {
        need!(src, a * b * c, "permute src");
        need!(dst, a * b * c, "permute dst");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_permute)(self.raw(), src.fp(), dst.fp(), a as i64, b as i64, c as i64) };
        self.ok(rc, "ds41_permute")
    }
    /// n bytes of src (from src_at) into dst (from at) by a kernel in queue order (not a copy engine): its ticket
    pub fn kcopy(&self, dst: &DevBuf, at: usize, src: &DevBuf, src_at: usize, n: usize) -> Result<i64> {
        dst.bounds(at, n)?;
        src.bounds(src_at, n)?;
        let mut t = 0i64;
        // SAFETY: both ranges in bounds (checked); t is a valid out pointer.
        let rc = unsafe { (self.k.ds41_kcopy)(self.raw(), dst.ptr().cast::<u8>().add(at).cast(), src.ptr().cast::<u8>().add(src_at).cast(), n, &mut t) };
        self.ok(rc, "ds41_kcopy")?;
        Ok(t)
    }
    /// The host waits for ticket t (the queue goes on)
    pub fn ticket_wait(&self, t: i64) -> Result<()> {
        // SAFETY: plain call.
        let rc = unsafe { (self.k.ds41_ticket_wait)(self.raw(), t) };
        self.ok(rc, "ds41_ticket_wait")
    }

    // ---- k41.cpp / vq.cpp (this model's own)

    /// The model's low-precision simulation in place on x [rows][cols], a scale a `block` (kind 0 E4M3 / power-of-two
    /// scale, 1 E2M1 / power of two, 2 E2M1 / E4M3 scale)
    pub fn quant_sim(&self, x: &DevBuf, rows: usize, cols: usize, block: usize, kind: i32) -> Result<()> {
        // NS_DS41_QSIM=0: none (a check against a reference without them: their rounding amplifies float noise)
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if rows == 0 || !*ON.get_or_init(|| std::env::var("NS_DS41_QSIM").map_or(true, |v| v != "0")) {
            return Ok(());
        }
        need!(x, rows * cols, "quant_sim x");
        // SAFETY: size checked.
        let rc = unsafe { (self.k.ds41_quant_sim)(self.raw(), x.fp(), rows as i64, cols as i64, block as i64, kind) };
        self.ok(rc, "ds41_quant_sim")
    }
    /// h [t, c] = sum_s pre[t][s] X[t][s]
    pub fn hc_collapse(&self, x: &DevBuf, pre: &DevBuf, h: &DevBuf, t: usize, c: usize) -> Result<()> {
        need!(x, t * 4 * c, "hc_collapse X");
        need!(pre, t * 4, "hc_collapse pre");
        need!(h, t * c, "hc_collapse h");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_hc_collapse)(self.raw(), x.fp(), pre.fp(), h.fp(), t as i64, c as i64) };
        self.ok(rc, "ds41_hc_collapse")
    }
    /// The Engram's raw rows [n][hd + hd / 32] -> float32 [n][hd]
    pub fn engram_rows(&self, raw: &DevBuf, out: &DevBuf, n: usize, hd: usize) -> Result<()> {
        raw.bounds(0, n * (hd + hd / 32))?;
        need!(out, n * hd, "engram_rows out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_engram_rows)(self.raw(), raw.ptr().cast(), out.fp(), n as i64, hd as i64) };
        self.ok(rc, "ds41_engram_rows")
    }
    /// The Engram's gate on the streams x [t, 4, c] in place (kv [t][5c], qk [4][c])
    pub fn engram_gate(&self, x: &DevBuf, kv: &DevBuf, qk: &DevBuf, t: usize, c: usize, eps: f32) -> Result<()> {
        need!(x, t * 4 * c, "engram_gate X");
        need!(kv, t * 5 * c, "engram_gate kv");
        need!(qk, 4 * c, "engram_gate qk");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_engram_gate)(self.raw(), x.fp(), kv.fp(), qk.fp(), t as i64, c as i64, eps) };
        self.ok(rc, "ds41_engram_gate")
    }
    /// The indexer's scores score [t][ld] (q [t][h][d], w [t][h], k fp16 [n][d]; -inf past what a row sees)
    #[allow(clippy::too_many_arguments)]
    pub fn idx_score(&self, q: &DevBuf, w: &DevBuf, k: &DevBuf, score: &DevBuf, t: usize, h: usize, d: usize, n: usize, ld: usize, pos0: usize, ratio: usize)
                     -> Result<()> {
        need!(q, t * h * d, "idx_score q");
        need!(w, t * h, "idx_score w");
        k.bounds(0, n * d * 2)?;
        need!(score, t * ld, "idx_score score");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.ds41_idx_score)(self.raw(), q.fp(), w.fp(), k.ptr().cast(), score.fp(), t as i64, h as i64, d as i64, n as i64, ld as i64, pos0 as i64,
                                    ratio as i64)
        };
        self.ok(rc, "ds41_idx_score")
    }
    /// The candidate source's block scores bmax [t][ceil(n / bs)]
    #[allow(clippy::too_many_arguments)]
    pub fn block_max(&self, score: &DevBuf, bmax: &DevBuf, t: usize, n: usize, ld: usize, bs: usize, pos0: usize, ratio: usize) -> Result<()> {
        need!(score, t * ld, "block_max score");
        need!(bmax, t * n.div_ceil(bs), "block_max out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_block_max)(self.raw(), score.fp(), bmax.fp(), t as i64, n as i64, ld as i64, bs as i64, pos0 as i64, ratio as i64) };
        self.ok(rc, "ds41_block_max")
    }
    /// score [t][ld] to -inf outside the blocks cand [t][k] lists
    #[allow(clippy::too_many_arguments)]
    pub fn cand_mask(&self, score: &DevBuf, cand: &DevBuf, mark: &DevBuf, t: usize, n: usize, ld: usize, k: usize, bs: usize) -> Result<()> {
        need!(score, t * ld, "cand_mask score");
        cand.bounds(0, t * k * 4)?;
        mark.bounds(0, t * n.div_ceil(bs))?;
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.ds41_cand_mask)(self.raw(), score.fp(), cand.ptr().cast(), mark.ptr().cast(), t as i64, n as i64, ld as i64, k as i64, bs as i64)
        };
        self.ok(rc, "ds41_cand_mask")
    }
    pub fn vq_supported(&self, rows: usize, cols: usize, nbit: usize) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.k.ds41_vq_supported)(rows as i64, cols as i64, nbit as i64) != 0 }
    }
    /// An expert matrix (its payload at bytes `at` of `src`, `bytes` long) expanded into dst (fp16 or float32)
    #[allow(clippy::too_many_arguments)]
    pub fn vq_dequant(&self, src: &DevBuf, at: usize, bytes: usize, cb: &DevBuf, rows: usize, cols: usize, nbit: usize, dst: &DevBuf, f16: bool) -> Result<()> {
        src.bounds(at, bytes)?;
        cb.bounds(0, (1usize << nbit) * 8)?;
        dst.bounds(0, rows * cols * if f16 { 2 } else { 4 })?;
        // SAFETY: sizes checked; the payload is the file's (its stream and plane inside `bytes`, checked at load).
        let rc = unsafe {
            (self.k.ds41_vq_dequant)(self.raw(), src.ptr().cast::<u8>().add(at).cast(), cb.ptr().cast(), rows as i64, cols as i64, nbit as i64, dst.ptr(),
                                     f16 as i32)
        };
        self.ok(rc, "ds41_vq_dequant")
    }
    /// Decode products of expert matrices: `table` = u64 payload pointers [entries], then i32 meta [entries][10]
    /// ({ncol, y0, cols}); y [(y0 + c) * rows + r] (floats from `y.1`, `y_rows` rows of `rows` in all); `variant` 0 a
    /// sub-group a row, 1 a lane a row; 5 by lookup table, which takes `lut` = (nx, maxn): the table's third part, after
    /// the pointers and the meta, lists for each of nx columns of x the (entry, column slot) pairs that read it (lut_lists)
    #[allow(clippy::too_many_arguments)]
    pub fn vq_mmv(&self, table: &DevBuf, entries: usize, cb: (&DevBuf, Option<&DevBuf>), x: &DevBuf, y: (&DevBuf, usize), y_rows: usize, rows: usize, cols: usize,
                  nbit: usize, variant: i32, lut: Option<(usize, usize)>) -> Result<()> {
        if entries == 0 {
            return Ok(());
        }
        let (cb, cbf) = cb;
        table.bounds(0, entries * 48)?;
        cb.bounds(0, (1usize << nbit) * 8)?;
        if let Some(f) = cbf.filter(|_| matches!(variant, 0 | 2 | 3)) {
            f.bounds(0, (1usize << nbit) * if variant == 0 { 32 } else { 16 })?;
        }
        need!(y.0, y.1 + y_rows * rows, "vq_mmv y");
        if variant == VQ_LUT {
            let (nx, maxn) = lut.ok_or_else(|| Error("vq_mmv: the lookup-table variant needs the columns' lists".into()))?;
            table.bounds(0, entries * 48 + nx * LUT_STRIDE * 4)?;
            need!(x, nx * cols, "vq_lut x");
            // SAFETY: as below; the lists after the meta, each slot's entry < entries (lut_lists).
            let rc = unsafe {
                let base = table.ptr().cast::<u8>();
                (self.k.ds41_vq_lut)(self.raw(), base.cast(), base.add(entries * 8).cast(), base.add(entries * 48).cast(), nx as i64, maxn as i64, cb.ptr().cast(),
                                     x.fp(), y.0.fp().add(y.1), y_rows as i64, rows as i64, cols as i64, nbit as i64, lut_units(cols) as i64)
            };
            return self.ok(rc, "ds41_vq_lut");
        }
        // SAFETY: sizes checked; the pointers are VRAM / pinned slots of this GPU holding the payloads, the columns
        // the caller's rows of x.
        let rc = unsafe {
            let base = table.ptr().cast::<u8>();
            (self.k.ds41_vq_mmv)(self.raw(), base.cast(), base.add(entries * 8).cast(), entries as i64, cb.ptr().cast(), cbf.map_or(std::ptr::null(), |f| f.fp()),
                                 x.fp(), y.0.fp().add(y.1), rows as i64, cols as i64, nbit as i64, variant)
        };
        self.ok(rc, "ds41_vq_mmv")
    }
    /// A prompt pass's expert product on XMX from the indices: y [n][ldy] (from float y.1) = x [n][cols] (fp16, rows ldx
    /// apart, its buffer readable to a multiple of 32 rows) . W^T; W's rows < split from payload p0, the rest from p1
    #[allow(clippy::too_many_arguments)]
    pub fn vq_gemm(&self, p0: &DevBuf, p1: Option<&DevBuf>, split: usize, rows: usize, cols: usize, nbit: usize, cbh: &DevBuf, x: &DevBuf, ldx: usize, n: usize,
                   y: (&DevBuf, usize), ldy: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        cbh.bounds(0, (1usize << nbit) * 16)?;
        x.bounds(0, (n.next_multiple_of(32) - 1) * ldx * 2 + cols * 2)?;
        need!(y.0, y.1 + (n - 1) * ldy + rows, "vq_gemm y");
        // SAFETY: x and y checked; the payloads are expert slots of this GPU (or pinned, device-visible) of a layer's
        // matrices with these shapes; cbh holds 2^nbit entries.
        let rc = unsafe {
            (self.k.ds41_vq_gemm)(self.raw(), p0.ptr(), p1.map_or(std::ptr::null(), |b| b.ptr()), split as i64, rows as i64, cols as i64, nbit as i64, cbh.ptr(),
                                  x.ptr(), ldx as i64, n as i64, y.0.fp().add(y.1), ldy as i64)
        };
        self.ok(rc, "ds41_vq_gemm")
    }
    /// *tok (an i32 at byte `at` of `toks`) = argmax(logits + bias) over n
    pub fn markov_step(&self, logits: (&DevBuf, usize), bias: &DevBuf, n: usize, toks: &DevBuf, at: usize) -> Result<()> {
        need!(logits.0, logits.1 + n, "markov_step logits");
        need!(bias, n, "markov_step bias");
        toks.bounds(at, 4)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_markov_step)(self.raw(), logits.0.fp().add(logits.1), bias.fp(), n as i64, toks.ptr().cast::<u8>().add(at).cast()) };
        self.ok(rc, "ds41_markov_step")
    }
    /// out [count][cols] = rows of table [rows][cols] named by the i32s at byte `at` of idx (on the GPU; each below `rows`)
    #[allow(clippy::too_many_arguments)]
    pub fn gather_rows(&self, table: &DevBuf, rows: usize, idx: &DevBuf, at: usize, count: usize, cols: usize, out: &DevBuf) -> Result<()> {
        need!(table, rows * cols, "gather_rows table");
        idx.bounds(at, count * 4)?;
        need!(out, count * cols, "gather_rows out");
        // SAFETY: sizes checked; the indices are argmaxes below `rows` (the vocabulary).
        let rc = unsafe { (self.k.ds41_gather_rows)(self.raw(), table.fp(), idx.ptr().cast::<u8>().add(at).cast(), count as i64, cols as i64, out.fp()) };
        self.ok(rc, "ds41_gather_rows")
    }
    /// A codebook's E4M3 bytes as fp16 (x 2^-8, exact) for the decode products' variants that gather 16 bytes an index
    pub fn vq_codebook_f16(&self, cb: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        cb.bounds(0, n)?;
        out.bounds(0, n * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_vq_codebook_f16)(self.raw(), cb.ptr().cast(), n as i64, out.ptr()) };
        self.ok(rc, "ds41_vq_codebook_f16")
    }
    /// A codebook's E4M3 bytes widened to float32 (x 2^-8) for the decode products
    pub fn vq_codebook_f32(&self, cb: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        cb.bounds(0, n)?;
        need!(out, n, "vq_codebook_f32 out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.ds41_vq_codebook_f32)(self.raw(), cb.ptr().cast(), n as i64, out.fp()) };
        self.ok(rc, "ds41_vq_codebook_f32")
    }
    /// Rows [r0, r0 + n) of an fp8_32x32 matrix [rows][cols] (stored at bytes `at` of w) expanded into dst
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_dequant(&self, w: &DevBuf, at: usize, rows: usize, cols: usize, r0: usize, n: usize, dst: &DevBuf, f16: bool) -> Result<()> {
        w.bounds(at, rows * cols + rows.div_ceil(32) * cols.div_ceil(32))?;
        dst.bounds(0, n * cols * if f16 { 2 } else { 4 })?;
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.ds41_fp8_dequant)(self.raw(), w.ptr().cast::<u8>().add(at).cast(), rows as i64, cols as i64, r0 as i64, n as i64, dst.ptr(), f16 as i32)
        };
        self.ok(rc, "ds41_fp8_dequant")
    }
    /// y = W . x for a q4_K W [rows][cols] and t (1..8) float32 columns: x (floats from x.1, columns x.2 apart; with
    /// `group` = (rows a group, floats apart) a row block's input moves by its group), y (from y.1, columns y.2 apart)
    #[allow(clippy::too_many_arguments)]
    pub fn q4k_mv(&self, w: &DevBuf, rows: usize, cols: usize, x: (&DevBuf, usize, usize), group: Option<(usize, usize)>, y: (&DevBuf, usize, usize), t: usize)
                  -> Result<()> {
        w.bounds(0, rows * cols / 256 * 144)?;
        let (rpg, xgs) = group.unwrap_or((0, 0));
        let groups = if rpg > 0 { rows.div_ceil(rpg) } else { 1 };
        need!(x.0, x.1 + (t - 1) * x.2 + (groups - 1) * xgs + cols, "q4k_mv x");
        need!(y.0, y.1 + (t - 1) * y.2 + rows, "q4k_mv y");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.ds41_q4k_mv)(self.raw(), w.ptr(), rows as i64, cols as i64, x.0.fp().add(x.1), x.2 as i64, rpg as i64, xgs as i64, y.0.fp().add(y.1),
                                 y.2 as i64, t as i64)
        };
        self.ok(rc, "ds41_q4k_mv")
    }
    /// y [t][rows] (floats from y.1) = W . x [t][cols] (floats from x.1), t 1..8
    pub fn fp8_mmv(&self, w: &DevBuf, rows: usize, cols: usize, x: (&DevBuf, usize, usize), groups: Option<(usize, usize)>, y: (&DevBuf, usize, usize),
                   t: usize) -> Result<()> {
        w.bounds(0, rows * cols + rows.div_ceil(32) * cols.div_ceil(32))?;
        let (rpg, xgs) = groups.unwrap_or((0, 0));
        let xspan = match groups {
            Some((rpg, xgs)) => (rows.div_ceil(rpg) - 1) * xgs + cols,
            None => cols,
        };
        need!(x.0, x.1 + (t - 1) * x.2 + xspan, "fp8_mmv x");
        need!(y.0, y.1 + (t - 1) * y.2 + rows, "fp8_mmv y");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.ds41_fp8_mmv)(self.raw(), w.ptr().cast(), rows as i64, cols as i64, x.0.fp().add(x.1), x.2 as i64, rpg as i64, xgs as i64, y.0.fp().add(y.1),
                                  y.2 as i64, t as i64)
        };
        self.ok(rc, "ds41_fp8_mmv")
    }
}

/// The VQ decode products' lookup-table variant (ns_ds41_vq_lut)
pub const VQ_LUT: i32 = 5;
/// A column's list in the lookup-table variant's table: its count, then 16 (entry, column slot) pairs
pub const LUT_STRIDE: usize = 33;

/// 64-column units a work-group of the lookup-table variant takes (NS_DS41_LUT_RUNS, default 16 runs a row)
fn lut_units(cols: usize) -> usize {
    static RUNS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let runs = *RUNS.get_or_init(|| std::env::var("NS_DS41_LUT_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(16).max(1));
    (cols / 64).div_ceil(runs).max(1)
}

/// The lookup-table variant's lists (the bytes after a table's meta) from (column of x, entry, column slot) triples: for
/// each of nx columns its count and pairs; (bytes, the largest count). None when a column has more than 16 readers.
pub fn lut_lists(nx: usize, pairs: impl IntoIterator<Item = (usize, usize, usize)>) -> Option<(Vec<u8>, usize)> {
    let mut v = vec![0i32; nx * LUT_STRIDE];
    for (c, e, q) in pairs {
        let at = c * LUT_STRIDE;
        let n = v[at] as usize;
        if c >= nx || n >= (LUT_STRIDE - 1) / 2 {
            return None;
        }
        v[at + 1 + 2 * n] = e as i32;
        v[at + 2 + 2 * n] = q as i32;
        v[at] += 1;
    }
    let maxn = (0..nx).map(|c| v[c * LUT_STRIDE] as usize).max().unwrap_or(0).max(1);
    Some((v.iter().flat_map(|x| x.to_le_bytes()).collect(), maxn))
}
