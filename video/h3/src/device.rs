//! The GPU and its memory: a context of `libh3sycl`, buffers that free themselves, tensors with a type and a shape.

use std::ffi::{c_void, CStr};
use std::sync::Arc;

use crate::sys::Api;

use crate::dtype::DType;
use crate::{Error, Result};

/// A GPU as the runtime lists it.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub mem_bytes: u64,
    /// "0000:0b:00.0"; empty when the runtime cannot tell
    pub pci: String,
}

/// One GPU: the kernel library and its context (an in-order SYCL queue). Shared by reference count; the context is
/// destroyed, with everything still allocated on it, when the last holder goes.
pub struct Device {
    pub(crate) api: Api,
    pub(crate) ctx: *mut c_void,
}

// SAFETY: the context is a SYCL queue plus state guarded inside the library (h3sycl.h: the copy and allocation
// functions are thread-safe, errors are per thread). Kernels are queued from one thread at a time by the engine.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// The GPUs there are: (index, name, memory bytes, PCI address).
    pub fn list() -> Result<Vec<GpuInfo>> {
        let api = Api::load()?;
        // SAFETY: plain calls; the buffers are sized as passed.
        let n = unsafe { (api.gpu_count)() };
        if n < 0 {
            return Err(Error(api.error()));
        }
        (0..n)
            .map(|i| {
                let (mut name, mut pci, mut mem) = (vec![0u8; 256], vec![0u8; 64], 0u64);
                let rc = unsafe { (api.gpu_info)(i, name.as_mut_ptr().cast(), 256, &mut mem, pci.as_mut_ptr().cast(), 64) };
                if rc != 0 {
                    return Err(Error(api.error()));
                }
                let s = |b: &[u8]| String::from_utf8_lossy(&b[..b.iter().position(|c| *c == 0).unwrap_or(b.len())]).into_owned();
                Ok(GpuInfo { index: i as usize, name: s(&name), mem_bytes: mem, pci: s(&pci) })
            })
            .collect()
    }

    /// Opens GPU `index` of `list()`.
    pub fn open_index(index: usize) -> Result<Arc<Device>> {
        let api = Api::load()?;
        // SAFETY: NULL is the failure.
        let ctx = unsafe { (api.open_gpu)(index as i32) };
        if ctx.is_null() {
            return Err(Error(format!("GPU {index}: {}", api.error())));
        }
        Ok(Arc::new(Device { api, ctx }))
    }

    /// Opens the first GPU.
    pub fn open() -> Result<Arc<Device>> {
        let api = Api::load()?;
        // SAFETY: h3s_open takes nothing and returns NULL on failure.
        let ctx = unsafe { (api.open)() };
        if ctx.is_null() {
            return Err(Error(format!("no usable GPU: {}", api.error())));
        }
        Ok(Arc::new(Device { api, ctx }))
    }

    pub fn name(&self) -> String {
        // SAFETY: the name lives as long as the context.
        unsafe { CStr::from_ptr((self.api.device_name)(self.ctx)).to_string_lossy().into_owned() }
    }

    /// Bytes allocated through this device, and the most it will allocate.
    pub fn mem_used(&self) -> u64 {
        unsafe { (self.api.mem_used)(self.ctx) }
    }
    pub fn mem_cap(&self) -> u64 {
        unsafe { (self.api.mem_cap)(self.ctx) }
    }
    /// What the whole card has free now, every process counted; `None` when the driver cannot tell.
    pub fn mem_free(&self) -> Option<u64> {
        match unsafe { (self.api.mem_free)(self.ctx) } {
            0 => None,
            n => Some(n),
        }
    }

    /// Device memory. Refused (an error, not a stall) when it would pass the cap.
    pub fn alloc(self: &Arc<Self>, bytes: usize) -> Result<Buf> {
        // SAFETY: plain call; NULL is the failure.
        let ptr = unsafe { (self.api.alloc)(self.ctx, bytes as u64) };
        if ptr.is_null() {
            return Err(Error(self.api.error()));
        }
        Ok(Buf { dev: self.clone(), ptr, bytes })
    }

    /// Waits until everything queued has run.
    pub fn wait(&self) -> Result<()> {
        self.check(unsafe { (self.api.wait)(self.ctx) })
    }

    /// Pinned host memory, for `Buf::upload`.
    pub fn alloc_pinned(self: &Arc<Self>, bytes: usize) -> Result<Pinned> {
        // SAFETY: plain call; NULL is the failure.
        let ptr = unsafe { (self.api.alloc_host)(self.ctx, bytes as u64) };
        if ptr.is_null() {
            return Err(Error(self.api.error()));
        }
        Ok(Pinned { dev: self.clone(), ptr: ptr.cast(), bytes })
    }

    /// Waits until every `Buf::upload` queued has landed.
    pub fn upload_wait(&self) -> Result<()> {
        self.check(unsafe { (self.api.upload_wait)(self.ctx) })
    }

    /// Whether `other` is this same context (a cache tied to one context checks it).
    pub fn same(&self, other: &Device) -> bool {
        std::ptr::eq(self.ctx, other.ctx)
    }

    pub(crate) fn check(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(Error(self.api.error()))
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: last holder; every Buf holds an Arc, so none is alive.
        unsafe { (self.api.destroy)(self.ctx) }
    }
}

/// A block of device memory. Freed on drop (the library waits for queued work first).
pub struct Buf {
    dev: Arc<Device>,
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: a device pointer is only dereferenced by the library, on the queue.
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn device(&self) -> &Arc<Device> {
        &self.dev
    }

    /// Copies host bytes to `offset`. Returns when the copy is done.
    pub fn write(&self, offset: usize, src: &[u8]) -> Result<()> {
        if offset + src.len() > self.bytes {
            return Err(Error(format!("write of {} bytes at {offset} passes a {}-byte buffer", src.len(), self.bytes)));
        }
        // SAFETY: bounds checked above; the library waits for the copy before returning.
        let rc = unsafe {
            (self.dev.api.write)(self.dev.ctx, self.ptr.cast::<u8>().add(offset).cast(), src.as_ptr().cast(), src.len() as u64)
        };
        self.dev.check(rc)
    }

    /// Copies device bytes from `offset` to the host, after everything queued before it.
    pub fn read(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        if offset + dst.len() > self.bytes {
            return Err(Error(format!("read of {} bytes at {offset} passes a {}-byte buffer", dst.len(), self.bytes)));
        }
        // SAFETY: bounds checked above.
        let rc = unsafe {
            (self.dev.api.read)(self.dev.ctx, dst.as_mut_ptr().cast(), self.ptr.cast::<u8>().add(offset).cast_const().cast(), dst.len() as u64)
        };
        self.dev.check(rc)
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        // SAFETY: allocated by this device; freed once.
        unsafe { (self.dev.api.free)(self.dev.ctx, self.ptr) }
    }
}

impl Buf {
    /// Queues a copy of `src[src_off .. src_off + n]` to `offset` on the copy queue, beside the kernels; it lands only
    /// after `Device::upload_wait`, and `src` must not be written before that.
    pub fn upload(&self, offset: usize, src: &Pinned, src_off: usize, n: usize) -> Result<()> {
        if offset + n > self.bytes || src_off + n > src.bytes {
            return Err(Error(format!("upload of {n} bytes ({src_off} in a {}-byte pinned buffer, {offset} in a {}-byte one) out of bounds", src.bytes, self.bytes)));
        }
        // SAFETY: bounds checked above; both belong to this device's context (the library checks the kinds).
        let rc = unsafe {
            (self.dev.api.upload)(self.dev.ctx, self.ptr.cast::<u8>().add(offset).cast(), src.ptr.add(src_off).cast_const().cast(), n as u64)
        };
        self.dev.check(rc)
    }
}

/// Pinned host memory (`Device::alloc_pinned`): the GPU reads it by DMA at the link's full rate. Freed on drop (after
/// the uploads queued from it).
pub struct Pinned {
    dev: Arc<Device>,
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: plain host memory; the engine hands each region to one writer at a time and reads it only after
// `Device::upload_wait`.
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Pinned {
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    pub fn device(&self) -> &Arc<Device> {
        &self.dev
    }
    /// A mutable view of `[off, off + n)`.
    ///
    /// # Safety
    /// No other view of these bytes may be alive, and no upload from them may be in flight.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn region(&self, off: usize, n: usize) -> &mut [u8] {
        assert!(off + n <= self.bytes, "pinned region out of bounds");
        // SAFETY: in bounds; exclusivity is the caller's promise.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), n) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by this device; freed once (the library waits for the uploads first).
        unsafe { (self.dev.api.free_host)(self.dev.ctx, self.ptr.cast()) }
    }
}

/// Device memory with a type and a shape (row-major).
pub struct Tensor {
    pub buf: Buf,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(dev: &Arc<Device>, dtype: DType, shape: &[usize]) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        Ok(Tensor { buf: dev.alloc(n * dtype.size())?, dtype, shape: shape.to_vec() })
    }

    pub fn from_bytes(dev: &Arc<Device>, dtype: DType, shape: &[usize], bytes: &[u8]) -> Result<Tensor> {
        let t = Tensor::new(dev, dtype, shape)?;
        if bytes.len() != t.buf.len() {
            return Err(Error(format!("{} bytes for a {:?} tensor of shape {:?}", bytes.len(), dtype, shape)));
        }
        t.buf.write(0, bytes)?;
        Ok(t)
    }

    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes per row (the product of every dimension but the first).
    pub fn row_bytes(&self) -> usize {
        self.shape.iter().skip(1).product::<usize>() * self.dtype.size()
    }

    /// Copies `rows` rows of `src` from row `src_row` into this tensor at row `dst_row`, on the device (queued).
    pub fn copy_rows(&self, dst_row: usize, src: &Tensor, src_row: usize, rows: usize) -> Result<()> {
        let rb = self.row_bytes();
        if src.dtype != self.dtype || src.row_bytes() != rb {
            return Err(Error(format!("copy_rows: {:?} {:?} into {:?} {:?}", src.dtype, src.shape, self.dtype, self.shape)));
        }
        if (dst_row + rows) * rb > self.buf.len() || (src_row + rows) * rb > src.buf.len() {
            return Err(Error(format!("copy_rows: rows {src_row}+{rows} of {:?} into {dst_row}.. of {:?}", src.shape, self.shape)));
        }
        let dev = self.buf.device();
        // SAFETY: both ranges checked to lie in their buffers.
        let rc = unsafe {
            (dev.api.copy)(dev.ctx, self.buf.ptr().cast::<u8>().add(dst_row * rb).cast(), src.buf.ptr().cast::<u8>().add(src_row * rb).cast_const().cast(), (rows * rb) as u64)
        };
        dev.check(rc)
    }

    /// The tensor's bytes, read back from the device.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut v = vec![0u8; self.buf.len()];
        self.buf.read(0, &mut v)?;
        Ok(v)
    }

    /// The tensor's values as float32, read back from the device.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        crate::dtype::bytes_to_f32(&self.to_bytes()?, self.dtype)
    }
}
