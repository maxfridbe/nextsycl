//! This engine's kernels on one GPU: the checked wrappers the engine calls (`ffi`: its own, ds2.h, and glm5next's
//! generic ones, glm.h). Every buffer is float32 unless named otherwise; the sizes each kernel reads are checked
//! against the buffers before the call.

use std::ffi::c_void;
use std::sync::Arc;

use nextsycl_core::{Api, DevBuf, Error, Gpu, Result};

use crate::ffi;

pub struct Ops {
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
}

macro_rules! need {
    ($b:expr, $n:expr, $what:expr) => {
        if $b.floats() < ($n) as usize {
            return Err(Error(format!("{}: a buffer of {} floats where {} are needed", $what, $b.floats(), $n)));
        }
    };
}

/// YaRN's parameters for the rotations (Model::yarn)
#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub mscale: f32,
    pub corr: (f32, f32),
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
    fn a(&self) -> &'static Api {
        self.k.shared
    }

    // ---- tickets and copies (ns.h)

    /// A ticket for everything submitted to this GPU's queue so far
    pub fn mark(&self) -> Result<i64> {
        let mut t = 0i64;
        // SAFETY: t is a valid out pointer.
        let rc = unsafe { (self.a().mark)(self.raw(), &mut t) };
        self.ok(rc, "mark")?;
        Ok(t)
    }
    /// n bytes of src (from src_at) into dst (from at) on copy lane `lane` (0 or 1), after every ticket of `after`:
    /// the copy's ticket
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
    /// n bytes of src (from src_at) into dst (from at) by a kernel on the compute queue, after ticket `after`: its
    /// ticket (a copy to pinned memory this way runs beside the copy engine's copies)
    pub fn kcopy(&self, dst: &DevBuf, at: usize, src: &DevBuf, src_at: usize, n: usize, after: Option<i64>) -> Result<i64> {
        dst.bounds(at, n)?;
        src.bounds(src_at, n)?;
        let mut t = 0i64;
        // SAFETY: both ranges in bounds (checked).
        let rc = unsafe {
            (self.k.kcopy)(self.raw(), dst.ptr().cast::<u8>().add(at).cast(), src.ptr().cast::<u8>().add(src_at).cast(), n, after.unwrap_or(-1), &mut t)
        };
        self.ok(rc, "kcopy")?;
        Ok(t)
    }
    /// n floats of x = v (a kernel: memset goes to a copy engine, behind the store's copies)
    pub fn fill(&self, x: &DevBuf, v: f32, n: usize) -> Result<()> {
        need!(x, n, "fill");
        // SAFETY: size checked.
        let rc = unsafe { (self.k.fill)(self.raw(), x.fp(), v, n as i64) };
        self.ok(rc, "fill")
    }
    /// The GPU's queue waits, on the device, for ticket `t`
    pub fn await_ticket(&self, t: i64) -> Result<()> {
        // SAFETY: plain call.
        let rc = unsafe { (self.a().await_)(self.raw(), t) };
        self.ok(rc, "await")
    }
    /// The host waits for ticket `t`
    pub fn wait_ticket(&self, t: i64) -> Result<()> {
        // SAFETY: plain call.
        let rc = unsafe { (self.k.ticket_wait)(self.raw(), t) };
        self.ok(rc, "ticket_wait")
    }
    /// Whether ticket `t` has run (no wait)
    pub fn ticket_done(&self, t: i64) -> bool {
        // SAFETY: plain call.
        unsafe { (self.k.ticket_done)(self.raw(), t) == 1 }
    }
    /// A device timestamp after everything queued so far (NS_PROFILE=gpu): its ticket
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

    // ---- expansion

    /// Whether the generic expanders (glm.h) take `ty`, or this engine's own (IQ1_S)
    fn own_type(&self, ty: u32) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.k.own_dequant_supported)(ty as i32) != 0 }
    }
    /// `n` values of type `ty` stored at bytes `[at, at + bytes)` of `src` into the front of `dst` as float32
    pub fn dequant(&self, ty: u32, src: &DevBuf, at: usize, bytes: usize, n: usize, dst: &DevBuf) -> Result<()> {
        need!(dst, n, "dequant");
        src.bounds(at, bytes)?;
        // SAFETY: the byte range is inside src (checked) and holds n values of `ty`; dst holds n floats (checked).
        let rc = unsafe {
            let p = src.ptr().cast::<u8>().add(at).cast::<c_void>();
            if self.own_type(ty) { (self.k.own_dequant)(self.raw(), ty as i32, p, n as i64, dst.fp()) } else { (self.k.dequant)(self.raw(), ty as i32, p, n, dst.fp()) }
        };
        self.ok(rc, "dequant")
    }
    /// n values of a stored matrix expanded to fp16 into dst (2 bytes a value)
    pub fn dequant_f16(&self, ty: u32, src: &DevBuf, at: usize, bytes: usize, n: usize, dst: &DevBuf) -> Result<()> {
        src.bounds(at, bytes)?;
        dst.bounds(0, n * 2)?;
        // SAFETY: both ranges checked.
        let rc = unsafe {
            let p = src.ptr().cast::<u8>().add(at).cast::<c_void>();
            if self.own_type(ty) {
                (self.k.own_dequant_f16)(self.raw(), ty as i32, p, n as i64, dst.ptr().cast())
            } else {
                (self.k.dequant_f16)(self.raw(), ty as i32, p, n as i64, dst.ptr().cast())
            }
        };
        self.ok(rc, "dequant_f16")
    }
    /// The types the fp16 expanders take (glm.h's list, and IQ1_S)
    pub fn f16_type(&self, ty: u32) -> bool {
        const GENERIC: [u32; 16] = [6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 20, 21, 22, 23, 29, 42];
        GENERIC.contains(&ty) || self.own_type(ty)
    }
    pub fn to_f16(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<()> {
        need!(x, n, "to_f16 x");
        y.bounds(0, n * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.to_f16)(self.raw(), x.fp(), y.ptr().cast(), n as i64) };
        self.ok(rc, "to_f16")
    }
    /// x [n] floats from float `x_at` -> y fp16 from half `y_at`
    pub fn to_f16_at(&self, x: &DevBuf, x_at: usize, y: &DevBuf, y_at: usize, n: usize) -> Result<()> {
        x.bounds(x_at * 4, n * 4)?;
        y.bounds(y_at * 2, n * 2)?;
        // SAFETY: both ranges checked.
        let rc = unsafe { (self.k.to_f16)(self.raw(), x.fp().add(x_at), y.ptr().cast::<u16>().add(y_at), n as i64) };
        self.ok(rc, "to_f16")
    }

    // ---- products

    /// y [t, n] (+)= x [t, k] . w [n, k]^T: x from float x.1, rows x.2 apart; w from float w.1; y from float y.1,
    /// rows y.2 apart
    pub fn gemm_at(&self, t: usize, n: usize, k: usize, x: (&DevBuf, usize, usize), w: (&DevBuf, usize), y: (&DevBuf, usize, usize), acc: bool) -> Result<()> {
        if t == 0 || n == 0 {
            return Ok(());
        }
        need!(x.0, x.1 + (t - 1) * x.2 + k, "gemm x");
        need!(w.0, w.1 + n * k, "gemm w");
        need!(y.0, y.1 + (t - 1) * y.2 + n, "gemm y");
        // SAFETY: extents checked.
        let rc = unsafe {
            (self.k.gemm)(self.raw(), t as i64, n as i64, k as i64, x.0.fp().add(x.1), x.2 as i64, w.0.fp().add(w.1), y.0.fp().add(y.1), y.2 as i64, acc as i32)
        };
        self.ok(rc, "gemm")
    }
    /// The same in fp16 (x from half x.1, rows x.2 apart; w from half w.1), float32 y
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
    /// batch products in fp16, float32 out: y_b [t, n] = x_b [t, k] . (w_b^T with `trans_w`: w_b [n, k]; else
    /// w_b [k, n]). x and w (buffer, offset, row stride, batch stride) in halves; y in floats.
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
    /// Bytes of Q8_1 for `ncols` columns of `n_in` (the generic decode products' input)
    pub fn q8_1_bytes(&self, n_in: usize, ncols: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.q8_1_bytes)(n_in as i64, ncols as i64) }
    }
    pub fn mmvq_supported(&self, ty: u32) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.k.mmvq_supported)(ty as i32) != 0 }
    }
    /// x [ncols, n_in] (floats from `x.1`) -> Q8_1 into `q`
    pub fn quantize_q8_1(&self, x: (&DevBuf, usize), q: &DevBuf, n_in: usize, ncols: usize) -> Result<()> {
        need!(x.0, x.1 + n_in * ncols, "quantize x");
        if q.len < self.q8_1_bytes(n_in, ncols) {
            return Err(Error(format!("quantize: {} bytes for {} of Q8_1", q.len, self.q8_1_bytes(n_in, ncols))));
        }
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.quantize_q8_1)(self.raw(), x.0.fp().add(x.1), q.ptr(), n_in as i64, ncols as i64) };
        self.ok(rc, "quantize_q8_1")
    }
    /// y [ncols, n_out] (floats from `y.1`) = W . x: W the stored blocks at bytes `w.1..w.1 + w_bytes` of `w.0`
    pub fn mmvq(&self, ty: u32, w: (&DevBuf, usize), w_bytes: usize, q: &DevBuf, y: (&DevBuf, usize), n_in: usize, n_out: usize, ncols: usize) -> Result<()> {
        w.0.bounds(w.1, w_bytes)?;
        if q.len < self.q8_1_bytes(n_in, ncols) {
            return Err(Error("mmvq: the Q8_1 input is short".into()));
        }
        need!(y.0, y.1 + n_out * ncols, "mmvq y");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.mmvq)(self.raw(), ty as i32, w.0.ptr().cast::<u8>().add(w.1).cast(), q.ptr(), y.0.fp().add(y.1), n_in as i64, n_out as i64, ncols as i64)
        };
        self.ok(rc, "mmvq")
    }
    /// Bytes of this engine's split Q8 layout for `ncols` columns of `n_in`
    pub fn q8_bytes(&self, n_in: usize, ncols: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.q8_bytes)(n_in as i64, ncols as i64) }
    }
    /// x [ncols, n_in] (floats from x.1) -> the split layout
    pub fn quantize_q8(&self, x: (&DevBuf, usize), xs: &DevBuf, n_in: usize, ncols: usize) -> Result<()> {
        need!(x.0, x.1 + n_in * ncols, "quantize_q8 x");
        xs.bounds(0, self.q8_bytes(n_in, ncols))?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.quantize_q8)(self.raw(), x.0.fp().add(x.1), xs.ptr(), n_in as i64, ncols as i64) };
        self.ok(rc, "quantize_q8")
    }
    pub fn own_mmvq_supported(&self, ty: u32, n_in: usize) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.k.own_mmvq_supported)(ty as i32, n_in as i64) != 0 }
    }
    /// y [ncols, n_out] = W . x, this engine's types (IQ1_S, IQ2_XXS), x in the split layout
    pub fn own_mmvq(&self, ty: u32, w: (&DevBuf, usize), w_bytes: usize, xs: &DevBuf, y: (&DevBuf, usize), n_in: usize, n_out: usize, ncols: usize) -> Result<()> {
        w.0.bounds(w.1, w_bytes)?;
        xs.bounds(0, self.q8_bytes(n_in, ncols))?;
        need!(y.0, y.1 + n_out * ncols, "own_mmvq y");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.own_mmvq)(self.raw(), ty as i32, w.0.ptr().cast::<u8>().add(w.1).cast(), xs.ptr(), y.0.fp().add(y.1), n_in as i64, n_out as i64, ncols as i64)
        };
        self.ok(rc, "own_mmvq")
    }
    /// A layer's experts in one launch: `table` holds the entries' matrix addresses [entries] (u64), then their
    /// columns [entries] (i32); y [entries, n_out]. The addresses are the caller's (VRAM slots of this GPU holding
    /// matrices of `ty`, [n_out, n_in] each).
    pub fn mmvq_multi(&self, ty: u32, table: &DevBuf, entries: usize, xs: &DevBuf, ncols: usize, y: &DevBuf, n_in: usize, n_out: usize) -> Result<()> {
        table.bounds(0, entries * 12)?;
        xs.bounds(0, self.q8_bytes(n_in, ncols))?;
        need!(y, entries * n_out, "mmvq_multi y");
        // SAFETY: sizes checked; the addresses are the caller's.
        let rc = unsafe {
            let p = table.ptr().cast::<u8>();
            (self.k.mmvq_multi)(self.raw(), ty as i32, p.cast(), p.add(entries * 8).cast(), entries as i64, xs.ptr(), ncols as i64, y.fp(), n_in as i64, n_out as i64)
        };
        self.ok(rc, "mmvq_multi")
    }

    // ---- element-wise and reductions

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
    /// out = silu(gate) * up (n values)
    pub fn swiglu(&self, gate: &DevBuf, up: &DevBuf, out: &DevBuf, n: usize) -> Result<()> {
        need!(gate, n, "swiglu gate");
        need!(up, n, "swiglu up");
        need!(out, n, "swiglu out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.swiglu_clamp)(self.raw(), gate.fp(), up.fp(), out.fp(), n as i64, f32::INFINITY) };
        self.ok(rc, "swiglu")
    }
    /// out [t, f] fp16 = swiglu of gu [t, 2f] (each row's gate, then its up)
    pub fn swiglu_gu_f16(&self, gu: &DevBuf, out: &DevBuf, t: usize, f: usize) -> Result<()> {
        need!(gu, t * 2 * f, "swiglu_gu gu");
        out.bounds(0, t * f * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.swiglu_gu_f16)(self.raw(), gu.fp(), out.ptr().cast(), t as i64, f as i64, f32::INFINITY) };
        self.ok(rc, "swiglu_gu_f16")
    }
    /// out [rows, f] = swiglu of gu [rows, 2f]
    pub fn swiglu_rows(&self, gu: &DevBuf, out: &DevBuf, rows: usize, f: usize) -> Result<()> {
        need!(gu, rows * 2 * f, "swiglu_rows gu");
        need!(out, rows * f, "swiglu_rows out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.swiglu_rows)(self.raw(), gu.fp(), out.fp(), rows as i64, f as i64) };
        self.ok(rc, "swiglu_rows")
    }
    /// softmax over each of r * h rows of nc scores, row r's heads over its first n[r] (scaled), the rest 0
    pub fn softmax_masked(&self, s: &DevBuf, r: usize, h: usize, nc: usize, n: &DevBuf, scale: f32) -> Result<()> {
        need!(s, r * h * nc, "softmax_masked S");
        n.bounds(0, r * 4)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.softmax_masked)(self.raw(), s.fp(), r as i64, h as i64, nc as i64, n.ptr().cast(), scale) };
        self.ok(rc, "softmax_masked")
    }
    /// out[i] = fp16 of rows idx[i] of src (rows of c)
    pub fn gather_f16(&self, src: &DevBuf, idx: &DevBuf, out: &DevBuf, n: usize, c: usize) -> Result<()> {
        idx.bounds(0, n * 4)?;
        out.bounds(0, n * c * 2)?;
        // SAFETY: sizes checked; indices are rows of src (the caller's).
        let rc = unsafe { (self.k.gather_f16)(self.raw(), src.fp(), idx.ptr().cast(), out.ptr().cast(), n as i64, c as i64) };
        self.ok(rc, "gather_f16")
    }
    /// y[idx[i]] += w[i] * src[i] (rows of c)
    pub fn scatter_add(&self, y: &DevBuf, src: &DevBuf, idx: &DevBuf, w: &DevBuf, n: usize, c: usize) -> Result<()> {
        need!(src, n * c, "scatter src");
        need!(idx, n, "scatter idx");
        need!(w, n, "scatter w");
        // SAFETY: sizes checked; the indices are the caller's (tokens of this pass, below y's rows).
        let rc = unsafe { (self.k.scatter_add)(self.raw(), y.fp(), src.fp(), idx.ptr().cast(), w.fp(), n as i64, c as i64) };
        self.ok(rc, "scatter_add")
    }
    /// y [t, c] += per token the weighted sum of its entries' rows: `ints` holds t_ptr [t + 1] then ent [entries]
    pub fn moe_combine(&self, y: &DevBuf, rows: &DevBuf, ints: &DevBuf, w: &DevBuf, t: usize, entries: usize, c: usize) -> Result<()> {
        need!(y, t * c, "combine y");
        need!(rows, entries * c, "combine rows");
        need!(ints, t + 1 + entries, "combine index");
        need!(w, entries, "combine w");
        // SAFETY: sizes checked; the index values are the caller's.
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

    // ---- MLA's own steps

    /// RoPE in place on x: t rows `row` floats apart, h heads `head` apart, `n_dims` values from float `off` of
    /// each head, row i at position pos0 + i; `out` (buffer, row, head strides in floats) the rotated values too
    pub fn rope(&self, x: &DevBuf, t: usize, h: usize, row: usize, head: usize, off: usize, n_dims: usize, pos0: usize, r: &Rope,
                out: Option<(&DevBuf, usize, usize)>) -> Result<()> {
        need!(x, (t - 1) * row + (h - 1) * head + off + n_dims, "rope x");
        if let Some((o, orow, ohead)) = out {
            need!(o, (t - 1) * orow + (h - 1) * ohead + n_dims, "rope out");
        }
        let (op, orow, ohead) = out.map_or((std::ptr::null_mut(), 0, 0), |(o, a, b)| (o.fp(), a, b));
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.rope)(self.raw(), x.fp(), t as i64, h as i64, row as i64, head as i64, off as i64, n_dims as i64, pos0 as i64, r.base, r.freq_scale,
                          r.ext_factor, r.mscale, r.corr.0, r.corr.1, op, orow as i64, ohead as i64)
        };
        self.ok(rc, "rope")
    }
    /// MLA at decode widths in one launch (ns_ds2_mla_decode): q [t, h, 192] (rope part rotated), k_b / v_b fp16, the
    /// cache fp16 [.., 576] -> out [t, h, 128]
    pub fn mla_decode(&self, q: &DevBuf, kb: &DevBuf, vb: &DevBuf, cache: &DevBuf, out: &DevBuf, t: usize, h: usize, pos0: usize, scale: f32) -> Result<()> {
        need!(q, t * h * 192, "mla_decode q");
        kb.bounds(0, h * 512 * 128 * 2)?;
        vb.bounds(0, h * 128 * 512 * 2)?;
        cache.bounds(0, (pos0 + t) * 576 * 2)?;
        need!(out, t * h * 128, "mla_decode out");
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.mla_decode)(self.raw(), q.fp(), kb.ptr().cast(), vb.ptr().cast(), cache.ptr().cast(), out.fp(), t as i64, h as i64, pos0 as i64, scale)
        };
        self.ok(rc, "mla_decode")
    }
    /// The same with kv_b as stored (Q6_K, [h x 256] rows of 512)
    pub fn mla_decode_q6k(&self, q: &DevBuf, kvb: &DevBuf, cache: &DevBuf, out: &DevBuf, t: usize, h: usize, pos0: usize, scale: f32) -> Result<()> {
        need!(q, t * h * 192, "mla_decode q");
        kvb.bounds(0, h * 256 * 2 * 210)?;
        cache.bounds(0, (pos0 + t) * 576 * 2)?;
        need!(out, t * h * 128, "mla_decode out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.k.mla_decode_q6k)(self.raw(), q.fp(), kvb.ptr(), cache.ptr().cast(), out.fp(), t as i64, h as i64, pos0 as i64, scale) };
        self.ok(rc, "mla_decode_q6k")
    }
    /// The cache's rows pos0.. (fp16, [l + rd] each) from kv_a's output kv [t, l + rd]: the latent normed, the key part
    /// rotated
    pub fn kv_store(&self, kv: &DevBuf, w: &DevBuf, eps: f32, cache: &DevBuf, t: usize, l: usize, rd: usize, pos0: usize, r: &Rope) -> Result<()> {
        need!(kv, t * (l + rd), "kv_store kv");
        need!(w, l, "kv_store w");
        cache.bounds(0, (pos0 + t) * (l + rd) * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe {
            (self.k.kv_store)(self.raw(), kv.fp(), w.fp(), eps, cache.ptr().cast(), t as i64, l as i64, rd as i64, pos0 as i64, r.base, r.freq_scale,
                              r.ext_factor, r.mscale, r.corr.0, r.corr.1)
        };
        self.ok(rc, "kv_store")
    }
}
