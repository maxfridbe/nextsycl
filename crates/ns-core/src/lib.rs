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
}

// SAFETY: the library's GPU handle is a queue and a context; SYCL queues are thread-safe.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the handle came from gpu_open; buffers keep an Arc<Gpu>, so none outlives it.
        unsafe { (self.api.gpu_close)(self.raw) }
    }
}

impl Gpu {
    pub fn open(index: usize) -> Result<Arc<Gpu>> {
        let api = api()?;
        let name = gpus()?.into_iter().find(|g| g.0 == index).map(|g| g.1).ok_or_else(|| Error(format!("no GPU {index}")))?;
        let mut raw: ns_sys::Gpu = std::ptr::null_mut();
        // SAFETY: an out-pointer to a local.
        check(api, unsafe { (api.gpu_open)(index as i32, &mut raw) }, &format!("opening GPU {index}"))?;
        Ok(Arc::new(Gpu { api, raw, index, name }))
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
}

// SAFETY: a device pointer; access goes through the GPU's queue.
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by this GPU; the queue is in order, so work queued on the buffer runs first.
        unsafe { (self.gpu.api.free)(self.gpu.raw, self.ptr) };
    }
}

impl DevBuf {
    pub fn new(gpu: &Arc<Gpu>, len: usize) -> Result<DevBuf> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: an out-pointer to a local.
        check(gpu.api, unsafe { (gpu.api.alloc)(gpu.raw, len.max(1), &mut ptr) }, &format!("allocating {len} bytes on {}", gpu.name))?;
        Ok(DevBuf { gpu: gpu.clone(), ptr, len })
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }
    /// Host bytes into `[at, at + src.len())`, queued (the source must stay valid until a sync).
    pub fn write(&self, at: usize, src: &[u8]) -> Result<()> {
        self.bounds(at, src.len())?;
        let g = &self.gpu;
        // SAFETY: in bounds (checked); the caller keeps `src` alive until the queue has run the copy.
        check(g.api, unsafe { (g.api.copy_to)(g.raw, self.ptr.cast::<u8>().add(at).cast(), src.as_ptr().cast(), src.len()) }, "copying to the GPU")
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
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: len bytes, owned.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: len bytes, owned, borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
