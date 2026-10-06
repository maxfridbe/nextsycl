//! The safe layer over `libnextsycl`: GPUs (each opened with its own context), device and pinned host buffers that
//! free themselves, copies with their sizes checked. Everything the kernels run on comes from here.

use std::ffi::{c_char, c_void};
use std::sync::{Arc, OnceLock};

pub use ns_sys::Api;

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

/// The kernel library, loaded once (`NEXTSYCL_LIB`, or beside the executable).
pub fn api() -> Result<&'static Api> {
    API.get_or_init(Api::load).as_ref().map_err(|e| Error(e.clone()))
}

fn check(api: &Api, rc: i32, what: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(format!("{what}: {}", api.error())))
    }
}

/// The GPUs the library sees: (index, name).
pub fn gpus() -> Result<Vec<(usize, String)>> {
    let a = api()?;
    // SAFETY: no arguments.
    let n = unsafe { (a.gpu_count)() };
    if n < 0 {
        return Err(Error(format!("listing the GPUs: {}", a.error())));
    }
    (0..n)
        .map(|i| {
            let mut buf = [0 as c_char; 256];
            // SAFETY: buf is 256 bytes, its length passed.
            check(a, unsafe { (a.gpu_name)(i, buf.as_mut_ptr(), buf.len()) }, "the GPU's name")?;
            // SAFETY: the library wrote a NUL-terminated string into buf.
            Ok((i as usize, unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()))
        })
        .collect()
}

/// One GPU, with its own context and in-order queue.
pub struct Gpu {
    api: &'static Api,
    raw: ns_sys::Gpu,
    pub index: usize,
    pub name: String,
    /// its PCI address ("0000:03:00.0"), when the driver says - where its sensors are (sysfs hwmon)
    pub pci: Option<String>,
    /// pinned host memory the small uploads go through without waiting (`DevBuf::write_async`)
    up: std::sync::Mutex<Upload>,
}

/// A ring of pinned host memory: bytes written at `off`, copied to the GPU in queue order
struct Upload {
    ptr: *mut u8,
    len: usize,
    off: usize,
}

/// The upload ring's size: a pass's small uploads (routing tables, counts) fit many times over
const UPLOAD_BYTES: usize = 16 << 20;

// SAFETY: the library's GPU handle is a queue and a context; SYCL queues are thread-safe.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the handle came from gpu_open; buffers keep an Arc<Gpu>, so none outlives it. The upload ring
        // was pinned by this context; the close waits for the queue first, so freeing it after is safe.
        unsafe {
            let up = self.up.get_mut().map_or(std::ptr::null_mut(), |u| u.ptr);
            let _ = (self.api.sync)(self.raw);
            if !up.is_null() {
                (self.api.free_host)(self.raw, up.cast());
            }
            (self.api.gpu_close)(self.raw)
        }
    }
}

impl Gpu {
    pub fn open(index: usize) -> Result<Arc<Gpu>> {
        let api = api()?;
        let name = gpus()?.into_iter().find(|g| g.0 == index).map(|g| g.1).ok_or_else(|| Error(format!("no GPU {index}")))?;
        let mut raw: ns_sys::Gpu = std::ptr::null_mut();
        // SAFETY: an out-pointer to a local.
        check(api, unsafe { (api.gpu_open)(index as i32, &mut raw) }, &format!("opening GPU {index}"))?;
        let mut up = std::ptr::null_mut();
        // SAFETY: an out-pointer to a local; the handle is live.
        check(api, unsafe { (api.alloc_host)(raw, UPLOAD_BYTES, &mut up) }, "pinning the upload ring")?;
        let mut pb = [0 as c_char; 64];
        // SAFETY: pb is 64 bytes, its length passed; the library NUL-terminates.
        let pci = (unsafe { (api.gpu_pci)(index as i32, pb.as_mut_ptr(), pb.len()) } == 0)
            .then(|| unsafe { std::ffi::CStr::from_ptr(pb.as_ptr()) }.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty());
        Ok(Arc::new(Gpu { api, raw, index, name, pci, up: std::sync::Mutex::new(Upload { ptr: up.cast(), len: UPLOAD_BYTES, off: 0 }) }))
    }

    /// The queue, for the imported kernels' `void* stream`.
    pub fn queue(&self) -> *mut c_void {
        // SAFETY: a live handle.
        unsafe { (self.api.gpu_queue)(self.raw) }
    }

    /// (total bytes, free bytes when the driver says)
    pub fn memory(&self) -> Result<(u64, Option<u64>)> {
        let (mut t, mut f) = (0u64, 0i64);
        // SAFETY: out-pointers to locals.
        check(self.api, unsafe { (self.api.gpu_memory)(self.raw, &mut t, &mut f) }, "the GPU's memory")?;
        Ok((t, (f >= 0).then_some(f as u64)))
    }

    pub fn sync(&self) -> Result<()> {
        // SAFETY: a live handle.
        check(self.api, unsafe { (self.api.sync)(self.raw) }, "waiting for the GPU")
    }
}

/// Device memory of one GPU.
pub struct DevBuf {
    gpu: Arc<Gpu>,
    ptr: *mut c_void,
    pub len: usize,
    /// false for a view (`view`, an `Arena`'s): the owner frees the memory
    owned: bool,
}

// SAFETY: a device pointer; access goes through the GPU's queue.
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if self.owned {
            // SAFETY: allocated by this GPU; the queue is in order, so work queued on the buffer runs first.
            unsafe { (self.gpu.api.free)(self.gpu.raw, self.ptr) };
        }
    }
}

impl DevBuf {
    pub fn new(gpu: &Arc<Gpu>, len: usize) -> Result<DevBuf> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: an out-pointer to a local.
        check(gpu.api, unsafe { (gpu.api.alloc)(gpu.raw, len.max(1), &mut ptr) }, &format!("allocating {len} bytes on {}", gpu.name))?;
        Ok(DevBuf { gpu: gpu.clone(), ptr, len, owned: true })
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }
    /// Host bytes into `[at, at + src.len())`, waiting for the copy (so `src` may be a temporary).
    pub fn write(&self, at: usize, src: &[u8]) -> Result<()> {
        self.bounds(at, src.len())?;
        let g = &self.gpu;
        // SAFETY: in bounds (checked); synced before `src` can go.
        check(g.api, unsafe { (g.api.copy_to)(g.raw, self.ptr.cast::<u8>().add(at).cast(), src.as_ptr().cast(), src.len()) }, "copying to the GPU")?;
        g.sync()
    }
    /// Host bytes into `[at, at + src.len())` without waiting: staged in the GPU's pinned upload ring, copied in
    /// queue order (the kernels after it see them). A wrap of the ring waits for the copies out of it so far; bytes
    /// past a quarter of it take `write`, and so does a buffer that owns its memory (it could be freed, the copy
    /// still queued: views of an arena are what this is for).
    pub fn write_async(&self, at: usize, src: &[u8]) -> Result<()> {
        self.bounds(at, src.len())?;
        let g = &self.gpu;
        let mut u = g.up.lock().unwrap();
        if src.len() > u.len / 4 || self.owned {
            drop(u);
            return self.write(at, src);
        }
        if u.off + src.len() > u.len {
            g.sync()?;
            u.off = 0;
        }
        // SAFETY: [off, off + len) inside the ring (checked above); no queued copy reads it (the ring wrapped past
        // it only after a sync); the destination is in bounds (checked).
        unsafe {
            let p = u.ptr.add(u.off);
            std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
            check(g.api, (g.api.copy_to)(g.raw, self.ptr.cast::<u8>().add(at).cast(), p.cast(), src.len()), "copying to the GPU")?;
        }
        u.off = (u.off + src.len()).next_multiple_of(64);
        Ok(())
    }
    /// `[at, at + dst.len())` into host memory, waiting.
    pub fn read(&self, at: usize, dst: &mut [u8]) -> Result<()> {
        self.bounds(at, dst.len())?;
        let g = &self.gpu;
        // SAFETY: in bounds (checked); synced before returning.
        check(g.api, unsafe { (g.api.copy_from)(g.raw, dst.as_mut_ptr().cast(), self.ptr.cast::<u8>().add(at).cast(), dst.len()) }, "copying from the GPU")?;
        g.sync()
    }
    pub fn fill(&self, v: u8) -> Result<()> {
        let g = &self.gpu;
        // SAFETY: the whole buffer.
        check(g.api, unsafe { (g.api.fill)(g.raw, self.ptr, v, self.len) }, "filling")
    }
    fn bounds(&self, at: usize, n: usize) -> Result<()> {
        if at.checked_add(n).is_none_or(|e| e > self.len) {
            return Err(Error(format!("bytes {at}..{} of a {}-byte buffer", at + n, self.len)));
        }
        Ok(())
    }

    /// All of `src` (another GPU's buffer) into the front of this one, through `staging` (host memory). Waits.
    pub fn copy_from_peer(&self, src: &DevBuf, staging: &mut [u8]) -> Result<()> {
        let n = src.len;
        self.bounds(0, n)?;
        if staging.len() < n {
            return Err(Error(format!("a staging area of {} bytes for {n}", staging.len())));
        }
        let (a, b) = (&self.gpu, &src.gpu);
        // SAFETY: both buffers live, sizes checked; the call waits for both copies.
        check(a.api, unsafe { (a.api.copy_peer)(a.raw, self.ptr, b.raw, src.ptr, n, staging.as_mut_ptr().cast()) }, "copying between GPUs")
    }
}

/// Pinned host memory of one GPU's context: the fast side of its copies.
pub struct HostBuf {
    gpu: Arc<Gpu>,
    ptr: *mut u8,
    pub len: usize,
}

// SAFETY: plain host memory, accessed through &/&mut self.
unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

impl Drop for HostBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by this GPU's context.
        unsafe { (self.gpu.api.free_host)(self.gpu.raw, self.ptr.cast()) };
    }
}

impl HostBuf {
    pub fn new(gpu: &Arc<Gpu>, len: usize) -> Result<HostBuf> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: an out-pointer to a local.
        check(gpu.api, unsafe { (gpu.api.alloc_host)(gpu.raw, len.max(1), &mut ptr) }, &format!("pinning {len} bytes"))?;
        Ok(HostBuf { gpu: gpu.clone(), ptr: ptr.cast(), len })
    }
    /// Bytes `[at, at + len)` as a buffer this GPU's kernels read in place, over PCIe: pinned memory of a GPU's own
    /// context is device-accessible. A view: it does not free them, and only this GPU may use it (pinned memory of
    /// another GPU's context is not its to read).
    pub fn device_view(&self, at: usize, len: usize) -> Result<DevBuf> {
        if at.checked_add(len).is_none_or(|e| e > self.len) {
            return Err(Error(format!("device_view: [{at}, {}) outside {} bytes", at + len, self.len)));
        }
        // SAFETY: in bounds (checked); the memory outlives the view by the caller's contract.
        Ok(DevBuf { gpu: self.gpu.clone(), ptr: unsafe { self.ptr.add(at).cast() }, len, owned: false })
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: len bytes, owned.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: len bytes, owned, borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl DevBuf {
    /// A buffer of `n` float32 values.
    pub fn f32(gpu: &Arc<Gpu>, n: usize) -> Result<DevBuf> {
        DevBuf::new(gpu, n * 4)
    }
    /// A buffer holding `v`.
    pub fn from_f32(gpu: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
        let b = DevBuf::f32(gpu, v.len())?;
        // SAFETY: plain floats as bytes.
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
        b.write(0, bytes)?;
        gpu.sync()?;
        Ok(b)
    }
    pub fn floats(&self) -> usize {
        self.len / 4
    }
    pub fn fp(&self) -> *mut f32 {
        self.ptr.cast()
    }
    /// The whole buffer as float32, waiting.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let mut v = vec![0f32; self.floats()];
        // SAFETY: v is floats() floats, viewed as their bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4) };
        self.read(0, bytes)?;
        Ok(v)
    }
}

/// The bring-up kernels of `kernels/ns/glm.cpp` on one GPU. Every buffer is float32 unless named otherwise; the
/// sizes each kernel reads are checked against the buffers before the call.
pub struct Ops {
    pub gpu: Arc<Gpu>,
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
    fn ok(&self, rc: i32, what: &str) -> Result<()> {
        check(self.gpu.api, rc, what)
    }
    fn raw(&self) -> ns_sys::Gpu {
        self.gpu.raw
    }
    fn a(&self) -> &'static Api {
        self.gpu.api
    }

    /// `n` values of ggml type `ty` stored at bytes `[at, at + bytes)` of `src` (`bytes` = what n values of the type
    /// take) into the front of `dst` as float32.
    pub fn dequant(&self, ty: u32, src: &DevBuf, at: usize, bytes: usize, n: usize, dst: &DevBuf) -> Result<()> {
        need!(dst, n, "dequant");
        src.bounds(at, bytes)?;
        // SAFETY: the byte range is inside src (checked) and holds n values of `ty` (the caller's `bytes`); dst
        // holds n floats (checked).
        let rc = unsafe { (self.a().dequant)(self.raw(), ty as i32, src.ptr.cast::<u8>().add(at).cast::<c_void>(), n, dst.fp()) };
        self.ok(rc, "dequant")
    }
    /// n values of a stored matrix (bytes [at, at + bytes) of src) expanded to fp16 into dst (2 bytes a value)
    pub fn dequant_f16(&self, ty: u32, src: &DevBuf, at: usize, bytes: usize, n: usize, dst: &DevBuf) -> Result<()> {
        src.bounds(at, bytes)?;
        dst.bounds(0, n * 2)?;
        // SAFETY: both ranges checked.
        let rc = unsafe { (self.a().dequant_f16)(self.raw(), ty as i32, src.ptr.cast::<u8>().add(at).cast::<c_void>(), n as i64, dst.ptr.cast()) };
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
            (self.a().stream_copy)(self.raw(), dst.ptr.cast::<u8>().add(at).cast(), src.ptr.cast::<u8>().add(src_at).cast(), n, deps.as_ptr(),
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
        let rc = unsafe { (self.a().to_f16)(self.raw(), x.fp(), y.ptr.cast(), n as i64) };
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
            (self.a().gemm_f16)(self.raw(), t as i64, n as i64, k as i64, x.0.ptr.cast::<u16>().add(x.1), x.2 as i64, w.0.ptr.cast::<u16>().add(w.1),
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
            (self.a().gemm)(self.raw(), t as i64, n as i64, k as i64, x.0.fp().add(x.1), x.2 as i64, w.0.fp().add(w.1), y.0.fp().add(y.1),
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
            (self.a().gemm_batch)(self.raw(), batch as i64, t as i64, n as i64, k as i64, x.0.fp().add(x.1), x.2 as i64, x.3 as i64, w.0.fp().add(w.1),
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
        let rc = unsafe { (self.a().hc_mix)(self.raw(), x.fp(), fn_.fp(), m.fp(), part.fp(), t as i64, n as i64, eps) };
        self.ok(rc, "hc_mix")
    }
    pub fn rms_norm(&self, x: &DevBuf, w: Option<&DevBuf>, y: &DevBuf, rows: usize, c: usize, eps: f32) -> Result<()> {
        need!(x, rows * c, "rms_norm x");
        need!(y, rows * c, "rms_norm y");
        if let Some(w) = w {
            need!(w, c, "rms_norm w");
        }
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().rms_norm)(self.raw(), x.fp(), w.map_or(std::ptr::null(), |w| w.fp()), y.fp(), rows as i64, c as i64, eps) };
        self.ok(rc, "rms_norm")
    }
    pub fn layer_norm(&self, x: &DevBuf, w: &DevBuf, b: &DevBuf, y: &DevBuf, rows: usize, c: usize, eps: f32) -> Result<()> {
        need!(x, rows * c, "layer_norm x");
        need!(y, rows * c, "layer_norm y");
        need!(w, c, "layer_norm w");
        need!(b, c, "layer_norm b");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().layer_norm)(self.raw(), x.fp(), w.fp(), b.fp(), y.fp(), rows as i64, c as i64, eps) };
        self.ok(rc, "layer_norm")
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
        let rc = unsafe { (self.a().hc_pre)(self.raw(), m.fp(), scale.fp(), base.fp(), x.fp(), h.fp(), post.fp(), comb.fp(), pre.fp(), t as i64, c as i64, eps, iters as i32) };
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
            (self.a().hc_pre_fused)(self.raw(), x.fp(), fn_.fp(), scale.fp(), base.fp(), nw.fp(), h.fp(), post.fp(), comb.fp(), pre.fp(), normed.fp(),
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
        let rc = unsafe { (self.a().hc_post)(self.raw(), y.fp(), x.fp(), post.fp(), comb.fp(), xo.fp(), t as i64, c as i64) };
        self.ok(rc, "hc_post")
    }
    pub fn hc_mean(&self, x: &DevBuf, y: &DevBuf, t: usize, c: usize) -> Result<()> {
        need!(x, t * 4 * c, "hc_mean X");
        need!(y, t * c, "hc_mean y");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().hc_mean)(self.raw(), x.fp(), y.fp(), t as i64, c as i64) };
        self.ok(rc, "hc_mean")
    }
    #[allow(clippy::too_many_arguments)]
    /// `snap`: the state after each row but the last, [t-1][k-1][d] (rolling back a rejected draft)
    pub fn conv_silu(&self, x: &DevBuf, state: &DevBuf, w: &DevBuf, out: &DevBuf, t: usize, d: usize, k: usize, snap: Option<&DevBuf>) -> Result<()> {
        need!(x, t * d, "conv x");
        need!(state, (k - 1) * d, "conv state");
        need!(w, d * k, "conv w");
        need!(out, t * d, "conv out");
        if let Some(s) = snap {
            need!(s, t.saturating_sub(1) * (k - 1) * d, "conv snapshots");
        }
        let sp = snap.map_or(std::ptr::null_mut(), |s| s.fp());
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().conv_silu)(self.raw(), x.fp(), state.fp(), w.fp(), out.fp(), t as i64, d as i64, k as i32, sp) };
        self.ok(rc, "conv_silu")
    }
    pub fn l2_norm(&self, x: &DevBuf, rows: usize, n: usize, eps: f32) -> Result<()> {
        need!(x, rows * n, "l2_norm");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().l2_norm)(self.raw(), x.fp(), rows as i64, n as i64, eps) };
        self.ok(rc, "l2_norm")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn kda_gate(&self, g: &DevBuf, dt_bias: &DevBuf, a: &DevBuf, t: usize, h: usize, dh: usize, low: f32) -> Result<()> {
        need!(g, t * h * dh, "kda_gate g");
        need!(dt_bias, h * dh, "kda_gate dt_bias");
        need!(a, h, "kda_gate A");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().kda_gate)(self.raw(), g.fp(), dt_bias.fp(), a.fp(), t as i64, h as i64, dh as i64, low) };
        self.ok(rc, "kda_gate")
    }
    pub fn sigmoid(&self, x: &DevBuf, n: usize) -> Result<()> {
        need!(x, n, "sigmoid");
        // SAFETY: size checked.
        let rc = unsafe { (self.a().sigmoid)(self.raw(), x.fp(), n as i64) };
        self.ok(rc, "sigmoid")
    }
    /// in place: x = exp(x)
    pub fn exp(&self, x: &DevBuf, n: usize) -> Result<()> {
        need!(x, n, "exp");
        // SAFETY: size checked.
        let rc = unsafe { (self.a().exp)(self.raw(), x.fp(), n as i64) };
        self.ok(rc, "exp")
    }
    #[allow(clippy::too_many_arguments)]
    /// `snap`: the state after each row but the last, [t-1][h][d][d] (rolling back a rejected draft)
    pub fn kda_scan(&self, q: &DevBuf, k: &DevBuf, v: &DevBuf, g: &DevBuf, beta: &DevBuf, s: &DevBuf, o: &DevBuf, t: usize, h: usize, d: usize,
                    snap: Option<&DevBuf>) -> Result<()> {
        for (b, n, what) in [(q, t * h * d, "q"), (k, t * h * d, "k"), (v, t * h * d, "v"), (g, t * h * d, "g"), (beta, t * h, "beta"), (s, h * d * d, "S"), (o, t * h * d, "o")] {
            need!(b, n, format!("kda_scan {what}"));
        }
        if let Some(sn) = snap {
            need!(sn, t.saturating_sub(1) * h * d * d, "kda_scan snapshots");
        }
        let sp = snap.map_or(std::ptr::null_mut(), |s| s.fp());
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().kda_scan)(self.raw(), q.fp(), k.fp(), v.fp(), g.fp(), beta.fp(), s.fp(), o.fp(), t as i64, h as i64, d as i64, sp) };
        self.ok(rc, "kda_scan")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn kda_out(&self, o: &DevBuf, g: &DevBuf, w: &DevBuf, y: &DevBuf, t: usize, h: usize, d: usize, eps: f32) -> Result<()> {
        need!(o, t * h * d, "kda_out o");
        need!(g, t * h * d, "kda_out g");
        need!(w, d, "kda_out w");
        need!(y, t * h * d, "kda_out y");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().kda_out)(self.raw(), o.fp(), g.fp(), w.fp(), y.fp(), t as i64, h as i64, d as i64, eps) };
        self.ok(rc, "kda_out")
    }
    /// out [t, f] (fp16) = swiglu of gu [t, 2f] (each row's gate, then its up), clamped like swiglu_clamp
    pub fn swiglu_gu_f16(&self, gu: &DevBuf, out: &DevBuf, t: usize, f: usize, limit: f32) -> Result<()> {
        need!(gu, t * 2 * f, "swiglu_gu gu");
        out.bounds(0, t * f * 2)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().swiglu_gu_f16)(self.raw(), gu.fp(), out.ptr.cast(), t as i64, f as i64, limit) };
        self.ok(rc, "swiglu_gu_f16")
    }
    pub fn swiglu_clamp(&self, gate: &DevBuf, up: &DevBuf, out: &DevBuf, n: usize, limit: f32) -> Result<()> {
        need!(gate, n, "swiglu gate");
        need!(up, n, "swiglu up");
        need!(out, n, "swiglu out");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().swiglu_clamp)(self.raw(), gate.fp(), up.fp(), out.fp(), n as i64, limit) };
        self.ok(rc, "swiglu_clamp")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn mla_attend(&self, qa: &DevBuf, c: &DevBuf, u: &DevBuf, t: usize, h: usize, l: usize, pos0: usize, scale: f32) -> Result<()> {
        need!(qa, t * h * l, "mla qa");
        need!(c, (pos0 + t) * l, "mla latents");
        need!(u, t * h * l, "mla u");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().mla_attend)(self.raw(), qa.fp(), c.fp(), u.fp(), t as i64, h as i64, l as i64, pos0 as i64, scale) };
        self.ok(rc, "mla_attend")
    }
    /// The pools tokens [pos0, pos0 + t) complete: pooled[j] from their members' ik / ig (`ring` [4][2d] holds the
    /// previous pass's last tokens), then the ring takes this pass's last tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn idx_pool(&self, ring: &DevBuf, ik: &DevBuf, ig: &DevBuf, ape: &DevBuf, pooled: &DevBuf, pos0: usize, t: usize, d: usize) -> Result<()> {
        need!(ring, 8 * d, "idx_pool ring");
        need!(ik, t * d, "idx_pool ik");
        need!(ig, t * d, "idx_pool ig");
        need!(ape, 4 * d, "idx_pool ape");
        need!(pooled, (pos0 + t) / 4 * d, "idx_pool pooled");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().idx_pool)(self.raw(), ring.fp(), ik.fp(), ig.fp(), ape.fp(), pooled.fp(), pos0 as i64, t as i64, d as i64) };
        self.ok(rc, "idx_pool")
    }
    /// score[r][j0 + jj] for rows [t] and pools [j0, j0 + n) from S [t * h, n] and w [t, h]
    #[allow(clippy::too_many_arguments)]
    pub fn idx_score(&self, s: &DevBuf, w: &DevBuf, score: &DevBuf, t: usize, h: usize, j0: usize, n: usize, ld: usize, pos0: usize) -> Result<()> {
        need!(s, t * h * n, "idx_score S");
        need!(w, t * h, "idx_score w");
        need!(score, (t - 1) * ld + j0 + n, "idx_score score");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().idx_score)(self.raw(), s.fp(), w.fp(), score.fp(), t as i64, h as i64, j0 as i64, n as i64, ld as i64, pos0 as i64) };
        self.ok(rc, "idx_score")
    }
    /// Each row's MLA cells (idx [t][nc] int32, count n [t] int32) from a selection (None: every earlier cell)
    #[allow(clippy::too_many_arguments)]
    pub fn mla_cells(&self, sel: Option<(&DevBuf, &DevBuf)>, t: usize, k: usize, pos0: usize, idx: &DevBuf, n: &DevBuf, nc: usize) -> Result<()> {
        idx.bounds(0, t * nc * 4)?;
        n.bounds(0, t * 4)?;
        let (sp, cp) = match sel {
            Some((s, c)) => {
                s.bounds(0, t * k * 4)?;
                c.bounds(0, t * 4)?;
                (s.ptr as *const i32, c.ptr as *const i32)
            }
            None => (std::ptr::null(), std::ptr::null()),
        };
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().mla_cells)(self.raw(), sp, cp, t as i64, k as i64, pos0 as i64, idx.ptr.cast(), n.ptr.cast(), nc as i64) };
        self.ok(rc, "mla_cells")
    }
    /// softmax over each of r * h rows of nc scores, row r's heads over its first n[r] (scaled), the rest 0
    pub fn softmax_masked(&self, s: &DevBuf, r: usize, h: usize, nc: usize, n: &DevBuf, scale: f32) -> Result<()> {
        need!(s, r * h * nc, "softmax_masked S");
        n.bounds(0, r * 4)?;
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().softmax_masked)(self.raw(), s.fp(), r as i64, h as i64, nc as i64, n.ptr.cast(), scale) };
        self.ok(rc, "softmax_masked")
    }
    /// batch products y_b [t, n] = x_b [t, k] . w_b [k, n] (offsets and strides in floats)
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batch_nn(&self, batch: usize, t: usize, n: usize, k: usize, x: (&DevBuf, usize, usize, usize), w: (&DevBuf, usize, usize, usize),
                         y: (&DevBuf, usize, usize, usize)) -> Result<()> {
        if batch == 0 || t == 0 || n == 0 {
            return Ok(());
        }
        let last = batch - 1;
        need!(x.0, x.1 + last * x.3 + (t - 1) * x.2 + k, "gemm_batch_nn x");
        need!(w.0, w.1 + last * w.3 + (k - 1) * w.2 + n, "gemm_batch_nn w");
        need!(y.0, y.1 + last * y.3 + (t - 1) * y.2 + n, "gemm_batch_nn y");
        // SAFETY: every batch's extent checked above.
        let rc = unsafe {
            (self.a().gemm_batch_nn)(self.raw(), batch as i64, t as i64, n as i64, k as i64, x.0.fp().add(x.1), x.2 as i64, x.3 as i64, w.0.fp().add(w.1),
                                     w.2 as i64, w.3 as i64, y.0.fp().add(y.1), y.2 as i64, y.3 as i64, 0)
        };
        self.ok(rc, "gemm_batch_nn")
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
            (self.a().gemm_batch_h)(self.raw(), batch as i64, trans_w as i32, t as i64, n as i64, k as i64, x.0.ptr.cast::<u16>().add(x.1), x.2 as i64,
                                    x.3 as i64, w.0.ptr.cast::<u16>().add(w.1), w.2 as i64, w.3 as i64, y.0.fp().add(y.1), y.2 as i64, y.3 as i64, 0)
        };
        self.ok(rc, "gemm_batch_h")
    }
    /// sel [t, k] (int32): the k highest of each score row [0, n), rows ld apart
    pub fn topk(&self, score: &DevBuf, sel: &DevBuf, t: usize, n: usize, ld: usize, k: usize) -> Result<()> {
        need!(score, (t - 1) * ld + n, "topk score");
        need!(sel, t * k, "topk sel");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().topk)(self.raw(), score.fp(), sel.ptr.cast(), t as i64, n as i64, ld as i64, k as i64) };
        self.ok(rc, "topk")
    }
    /// MLA over each row's selection (`sel` [t, k] pools, `cnt` [t]: < 0 = every earlier token); None: dense
    #[allow(clippy::too_many_arguments)]
    pub fn mla_attend_sel(&self, qa: &DevBuf, c: &DevBuf, u: &DevBuf, t: usize, h: usize, l: usize, pos0: usize, scale: f32,
                          sel: Option<(&DevBuf, &DevBuf)>, k: usize) -> Result<()> {
        need!(qa, t * h * l, "mla qa");
        c.bounds(0, (pos0 + t) * l * 2)?; // fp16 latents
        need!(u, t * h * l, "mla u");
        let (sp, cp) = match sel {
            Some((s, n)) => {
                need!(s, t * k, "mla sel");
                need!(n, t, "mla sel count");
                (s.ptr as *const i32, n.ptr as *const i32)
            }
            None => (std::ptr::null(), std::ptr::null()),
        };
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().mla_attend_sel)(self.raw(), qa.fp(), c.ptr.cast(), u.fp(), t as i64, h as i64, l as i64, pos0 as i64, scale, sp, cp, k as i64) };
        self.ok(rc, "mla_attend_sel")
    }
    /// y[idx[i]] += w[i] * src[i] (rows of c); idx and w on the device (`n` int32 / float32)
    pub fn scatter_add(&self, y: &DevBuf, src: &DevBuf, idx: &DevBuf, w: &DevBuf, n: usize, c: usize) -> Result<()> {
        need!(src, n * c, "scatter src");
        need!(idx, n, "scatter idx");
        need!(w, n, "scatter w");
        // SAFETY: sizes checked; the indices are the caller's (tokens of this batch, below y's rows).
        let rc = unsafe { (self.a().scatter_add)(self.raw(), y.fp(), src.fp(), idx.ptr.cast(), w.fp(), n as i64, c as i64) };
        self.ok(rc, "scatter_add")
    }
    /// out [n, c] fp16 = rows idx of src (float32)
    /// out[i] = src[idx[i]], rows of c fp16 values (the MLA cache's cells)
    pub fn gather_h(&self, src: &DevBuf, idx: &DevBuf, out: &DevBuf, n: usize, c: usize) -> Result<()> {
        idx.bounds(0, n * 4)?;
        out.bounds(0, n * c * 2)?;
        // SAFETY: idx and out sized (checked); the indices are cells below the position, inside src (mla_cells).
        let rc = unsafe { (self.a().gather_h)(self.raw(), src.ptr.cast(), idx.ptr.cast(), out.ptr.cast(), n as i64, c as i64) };
        self.ok(rc, "gather_h")
    }
    pub fn gather_f16(&self, src: &DevBuf, idx: &DevBuf, out: &DevBuf, n: usize, c: usize) -> Result<()> {
        idx.bounds(0, n * 4)?;
        out.bounds(0, n * c * 2)?;
        // SAFETY: sizes checked; indices are rows of src (the caller's).
        let rc = unsafe { (self.a().gather_f16)(self.raw(), src.fp(), idx.ptr.cast(), out.ptr.cast(), n as i64, c as i64) };
        self.ok(rc, "gather_f16")
    }
    pub fn gather(&self, src: &DevBuf, idx: &DevBuf, out: &DevBuf, n: usize, c: usize) -> Result<()> {
        need!(idx, n, "gather idx");
        need!(out, n * c, "gather out");
        // SAFETY: sizes checked; indices are rows of src (the caller's).
        let rc = unsafe { (self.a().gather)(self.raw(), src.fp(), idx.ptr.cast(), out.fp(), n as i64, c as i64) };
        self.ok(rc, "gather")
    }
    /// Bytes of Q8_1 for `ncols` columns of `n_in`.
    pub fn q8_1_bytes(&self, n_in: usize, ncols: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.a().q8_1_bytes)(n_in as i64, ncols as i64) }
    }
    pub fn mmvq_supported(&self, ty: u32) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.a().mmvq_supported)(ty as i32) != 0 }
    }
    /// x [ncols, n_in] (floats from `x.1`) -> Q8_1 into `q`.
    pub fn quantize_q8_1(&self, x: (&DevBuf, usize), q: &DevBuf, n_in: usize, ncols: usize) -> Result<()> {
        need!(x.0, x.1 + n_in * ncols, "quantize x");
        if q.len < self.q8_1_bytes(n_in, ncols) {
            return Err(Error(format!("quantize: {} bytes for {} of Q8_1", q.len, self.q8_1_bytes(n_in, ncols))));
        }
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().quantize_q8_1)(self.raw(), x.0.fp().add(x.1), q.ptr, n_in as i64, ncols as i64) };
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
            (self.a().mmvq)(self.raw(), ty as i32, w.0.ptr.cast::<u8>().add(w.1).cast(), q.ptr, y.0.fp().add(y.1), n_in as i64, n_out as i64, ncols as i64)
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
            (self.a().moe_combine)(self.raw(), y.fp(), rows.fp(), ints.ptr.cast(), ints.ptr.cast::<i32>().add(t + 1), w.fp(), t as i64, c as i64)
        };
        self.ok(rc, "moe_combine")
    }
    pub fn moe_grouped_supported(&self, gu: u32, down: u32, n_embd: usize, n_ff: usize) -> bool {
        // SAFETY: a capability query.
        unsafe { (self.a().moe_grouped_supported)(gu as i32, down as i32, n_embd as i64, n_ff as i64) != 0 }
    }
    pub fn moe_scratch_bytes(&self, entries: usize, n_ff: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.a().moe_scratch_bytes)(entries as i64, n_ff as i64) }
    }
    /// A layer's experts in two launches. `table` holds, as u64: the groups' blob pointers [groups]; then as i32:
    /// grp_start [groups + 1], n_groups [1], ent_dst [entries], ent_tok [entries] (at the offsets returned by
    /// `moe_table`). `xq` the tokens' Q8_1, `out` [entries, n_embd] (unweighted rows).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped(&self, gu: u32, down: u32, n_embd: usize, n_ff: usize, table: &DevBuf, groups: usize, entries: usize, xq: &DevBuf, scratch: &DevBuf,
                       out: &DevBuf, limit: f32, lanes: usize) -> Result<()> {
        let ints = groups * 2 + (groups + 1) + 1 + 2 * entries; // the u64 pointers as two i32 each, then the i32 tables
        if table.len < ints * 4 || scratch.len < self.moe_scratch_bytes(entries, n_ff) {
            return Err(Error("moe_grouped: the table or the scratch is short".into()));
        }
        need!(out, entries * n_embd, "moe_grouped out");
        let base = table.ptr.cast::<u8>();
        // SAFETY: the table's layout as written by the caller (above), sizes checked; the blob pointers are the
        // caller's (VRAM slots of this GPU holding the experts).
        let rc = unsafe {
            let ptrs = base.cast::<u64>();
            let i32s = base.add(groups * 8).cast::<i32>();
            let (gs, ng) = (i32s, i32s.add(groups + 1));
            let (dst, tok) = (ng.add(1), ng.add(1 + entries));
            (self.a().moe_grouped)(self.raw(), gu as i32, down as i32, n_embd as i64, n_ff as i64, ptrs, gs, ng, dst, tok, groups as i64, entries as i64,
                                   xq.ptr, scratch.ptr, out.fp(), limit, lanes as i32)
        };
        self.ok(rc, "moe_grouped")
    }
    pub fn add(&self, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        need!(y, n, "add y");
        need!(x, n, "add x");
        // SAFETY: sizes checked.
        let rc = unsafe { (self.a().add)(self.raw(), y.fp(), x.fp(), n as i64) };
        self.ok(rc, "add")
    }
}

impl DevBuf {
    /// Bytes `[src_at, src_at + n)` of `src` (same GPU) into `[at, at + n)`, queued in order.
    pub fn copy_within(&self, at: usize, src: &DevBuf, src_at: usize, n: usize) -> Result<()> {
        self.bounds(at, n)?;
        src.bounds(src_at, n)?;
        if !Arc::ptr_eq(&self.gpu, &src.gpu) {
            return Err(Error("copy_within: the buffers are on different GPUs (copy_from_peer)".into()));
        }
        let g = &self.gpu;
        // SAFETY: both ranges in bounds (checked), same GPU and context.
        check(g.api, unsafe { (g.api.copy_dev)(g.raw, self.ptr.cast::<u8>().add(at).cast(), src.ptr.cast::<u8>().add(src_at).cast(), n) }, "copying on the GPU")
    }
}

impl DevBuf {
    /// Bytes `[at, at + len)` of this buffer as a buffer of its own that does not free them. The caller keeps the
    /// owner alive while the view is used.
    pub fn view(&self, at: usize, len: usize) -> Result<DevBuf> {
        self.bounds(at, len)?;
        // SAFETY: in bounds (checked).
        Ok(DevBuf { gpu: self.gpu.clone(), ptr: unsafe { self.ptr.cast::<u8>().add(at).cast() }, len, owned: false })
    }
}

/// A bump allocator over one device allocation: temporaries without a `malloc_device` each. `reset` makes all of it
/// free again - the caller resets only when no view of it is in use (the queue is in order, so work already queued
/// on the old views runs before work on the new ones). Requests that do not fit get an allocation of their own.
pub struct Arena {
    buf: DevBuf,
    off: std::sync::Mutex<usize>,
    /// the most used between resets, and the requests that did not fit
    pub peak: std::sync::atomic::AtomicUsize,
    pub spills: std::sync::atomic::AtomicUsize,
}

impl Arena {
    pub fn new(gpu: &Arc<Gpu>, bytes: usize) -> Result<Arena> {
        Ok(Arena { buf: DevBuf::new(gpu, bytes)?, off: std::sync::Mutex::new(0), peak: 0.into(), spills: 0.into() })
    }
    pub fn bytes(&self, n: usize) -> Result<DevBuf> {
        let mut off = self.off.lock().unwrap();
        let at = off.next_multiple_of(256);
        if at + n > self.buf.len {
            self.spills.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return DevBuf::new(&self.buf.gpu, n);
        }
        *off = at + n;
        self.peak.fetch_max(*off, std::sync::atomic::Ordering::Relaxed);
        self.buf.view(at, n.max(1).min(self.buf.len - at))
    }
    pub fn f32(&self, n: usize) -> Result<DevBuf> {
        self.bytes(n * 4)
    }
    pub fn reset(&self) {
        *self.off.lock().unwrap() = 0;
    }
}
