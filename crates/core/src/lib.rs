//! The safe layer over `libnextsycl`: GPUs (each opened with its own context), device and pinned host buffers that
//! free themselves, copies with their sizes checked. Everything the kernels run on comes from here.

use std::ffi::{c_char, c_void};
use std::sync::{Arc, OnceLock};

pub mod options;

pub use nextsycl_sys::Api;
pub use options::{At, EngineOption};

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
static KIND: OnceLock<&'static str> = OnceLock::new();

/// The kind whose kernel library this process uses ("llm", "image", "video", "audio"; `libnextsycl-<kind>.so`), set before the
/// first GPU call - a process runs one kind. The default is "llm". False when the library was chosen already.
pub fn use_kind(kind: &'static str) -> bool {
    KIND.set(kind).is_ok() && API.get().is_none()
}

/// The kernel library, loaded once: the kind's (`use_kind`), or `NEXTSYCL_LIB`.
pub fn api() -> Result<&'static Api> {
    API.get_or_init(|| Api::load(KIND.get().copied().unwrap_or("llm"))).as_ref().map_err(|e| Error(e.clone()))
}

fn check(api: &Api, rc: i32, what: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(format!("{what}: {}", api.error())))
    }
}

/// The GPUs the library sees: (index, name).
/// GPU `index`'s compute units and their clock (MHz)
pub fn gpu_units(index: usize) -> Result<(u32, u32)> {
    let a = api()?;
    let (mut u, mut m) = (0, 0);
    // SAFETY: out-pointers to locals.
    check(a, unsafe { (a.gpu_units)(index as i32, &mut u, &mut m) }, "the GPU's compute units")?;
    Ok((u as u32, m as u32))
}

/// Every GPU's index, the one that computes most last (`--gpu all`: the last GPU takes the head and the draft
/// block); ties keep the driver's order
pub fn gpus_weakest_first() -> Result<Vec<usize>> {
    let mut v: Vec<(usize, u64)> = gpus()?.into_iter().map(|(i, _)| (i, gpu_units(i).map_or(0, |(u, m)| u as u64 * m as u64))).collect();
    v.sort_by_key(|x| x.1);
    Ok(v.into_iter().map(|x| x.0).collect())
}

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
    raw: nextsycl_sys::Gpu,
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
        let mut raw: nextsycl_sys::Gpu = std::ptr::null_mut();
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

    /// The library's handle of this GPU (`ns_gpu*`), for an engine's own ABI
    pub fn raw(&self) -> *mut c_void {
        self.raw
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
    /// `[at, at + n)` inside this buffer, or an error
    pub fn bounds(&self, at: usize, n: usize) -> Result<()> {
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
    /// the memory it hands out (a view; `set` swaps it)
    buf: std::sync::Mutex<DevBuf>,
    off: std::sync::Mutex<usize>,
    /// the most used between resets, and the requests that did not fit
    pub peak: std::sync::atomic::AtomicUsize,
    pub spills: std::sync::atomic::AtomicUsize,
}

impl Arena {
    pub fn new(gpu: &Arc<Gpu>, bytes: usize) -> Result<Arena> {
        Ok(Arena::on(DevBuf::new(gpu, bytes)?))
    }
    /// An arena over `buf` (which must outlive it: a view of memory its owner keeps)
    pub fn on(buf: DevBuf) -> Arena {
        Arena { buf: std::sync::Mutex::new(buf), off: std::sync::Mutex::new(0), peak: 0.into(), spills: 0.into() }
    }
    /// Hands out `buf` from now on (from its start); nothing handed out before may be used after
    pub fn set(&self, buf: DevBuf) {
        *self.buf.lock().unwrap() = buf;
        *self.off.lock().unwrap() = 0;
    }
    pub fn len(&self) -> usize {
        self.buf.lock().unwrap().len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn bytes(&self, n: usize) -> Result<DevBuf> {
        let mut off = self.off.lock().unwrap();
        let buf = self.buf.lock().unwrap();
        let at = off.next_multiple_of(256);
        if at + n > buf.len {
            self.spills.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return DevBuf::new(&buf.gpu, n);
        }
        *off = at + n;
        self.peak.fetch_max(*off, std::sync::atomic::Ordering::Relaxed);
        buf.view(at, n.max(1).min(buf.len - at))
    }
    pub fn f32(&self, n: usize) -> Result<DevBuf> {
        self.bytes(n * 4)
    }
    pub fn reset(&self) {
        *self.off.lock().unwrap() = 0;
    }
    /// The offset now, to hand back what is taken after it (`rewind`)
    pub fn mark(&self) -> usize {
        *self.off.lock().unwrap()
    }
    /// Everything taken since `mark` free again (the caller uses none of it after: the queue is in order)
    pub fn rewind(&self, mark: usize) {
        let mut off = self.off.lock().unwrap();
        if mark < *off {
            *off = mark;
        }
    }
}
