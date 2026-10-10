//! Raw bindings to `libnextsycl.so`: one function pointer per shared declaration of `kernels/ns/ns.h` (an engine
//! binds its own kernels by name: `Api::symbol`).
//!
//! The library is opened at run time (`dlopen`), not linked: the Rust side builds anywhere without the SYCL
//! toolchain, and the kernels rebuild without relinking the runtime. Nothing here is safe to call directly;
//! `ns-core` wraps it.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

/// An opened GPU (`ns_gpu*`).
pub type Gpu = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    /// the library's handle: an engine binds its own ABI from it (`symbol`)
    handle: *mut c_void,
    /// where it was opened from: the silos are in `silo/` beside it
    path: PathBuf,
    pub last_error: unsafe extern "C" fn() -> *const c_char,
    pub version: unsafe extern "C" fn() -> *const c_char,
    pub gpu_count: unsafe extern "C" fn() -> c_int,
    pub gpu_name: unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int,
    pub gpu_pci: unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int,
    pub gpu_units: unsafe extern "C" fn(c_int, *mut c_int, *mut c_int) -> c_int,
    pub gpu_open: unsafe extern "C" fn(c_int, *mut Gpu) -> c_int,
    pub gpu_close: unsafe extern "C" fn(Gpu),
    pub gpu_queue: unsafe extern "C" fn(Gpu) -> *mut c_void,
    pub gpu_memory: unsafe extern "C" fn(Gpu, *mut u64, *mut i64) -> c_int,
    pub alloc: unsafe extern "C" fn(Gpu, usize, *mut *mut c_void) -> c_int,
    pub free: unsafe extern "C" fn(Gpu, *mut c_void) -> c_int,
    pub alloc_host: unsafe extern "C" fn(Gpu, usize, *mut *mut c_void) -> c_int,
    pub free_host: unsafe extern "C" fn(Gpu, *mut c_void) -> c_int,
    pub copy_to: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub copy_from: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub copy_dev: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub mark: unsafe extern "C" fn(Gpu, *mut i64) -> c_int,
    pub stream_copy: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize, *const i64, c_int, c_int, *mut i64) -> c_int,
    pub await_: unsafe extern "C" fn(Gpu, i64) -> c_int,
    pub stamp: unsafe extern "C" fn(Gpu, *mut i64) -> c_int,
    pub elapsed: unsafe extern "C" fn(Gpu, i64, i64, *mut f64) -> c_int,
    pub fill: unsafe extern "C" fn(Gpu, *mut c_void, u8, usize) -> c_int,
    pub sync: unsafe extern "C" fn(Gpu) -> c_int,
    pub copy_peer: unsafe extern "C" fn(Gpu, *mut c_void, Gpu, *const c_void, usize, *mut c_void) -> c_int,
}

// SAFETY: plain function pointers into a library that stays loaded for the process's life.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn dl_error() -> String {
    // SAFETY: dlerror returns null or a C string valid until the next dl* call on this thread.
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown dlopen error".into()
    } else {
        // SAFETY: non-null, NUL-terminated (above).
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

impl Api {
    /// A kind's kernel library (`libnextsycl-<kind>.so`: "llm", "image", "video"): `NEXTSYCL_LIB` when set, else the
    /// library beside the executable, else in `dist/` of the working directory.
    pub fn load(kind: &str) -> Result<Api, String> {
        let name = format!("libnextsycl-{kind}.so");
        let p = std::env::var_os("NEXTSYCL_LIB").map(PathBuf::from).or_else(|| {
            let exe = std::env::current_exe().ok()?;
            let beside = exe.parent()?.join(&name);
            beside.exists().then_some(beside)
        });
        Api::load_from(&p.unwrap_or_else(|| PathBuf::from("dist").join(&name)))
    }

    #[allow(clippy::missing_transmute_annotations)] // the target type is each field's, from ns.h
    pub fn load_from(path: &Path) -> Result<Api, String> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: dlopen with a valid C string; the handle is checked.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if h.is_null() {
            return Err(format!("{}: {}", path.display(), dl_error()));
        }
        macro_rules! sym {
            ($n:literal) => {{
                let n = CString::new($n).unwrap();
                // SAFETY: a symbol lookup in the handle above; the pointer is checked and has ns.h's signature.
                let p = unsafe { dlsym(h, n.as_ptr()) };
                if p.is_null() {
                    return Err(format!("{}: no symbol {}", path.display(), $n));
                }
                // SAFETY: the declaration in ns.h is the field's type.
                unsafe { std::mem::transmute::<*mut c_void, _>(p) }
            }};
        }
        Ok(Api {
            handle: h,
            path: path.to_path_buf(),
            last_error: sym!("ns_last_error"),
            version: sym!("ns_version"),
            gpu_count: sym!("ns_gpu_count"),
            gpu_name: sym!("ns_gpu_name"),
            gpu_pci: sym!("ns_gpu_pci"),
            gpu_units: sym!("ns_gpu_units"),
            gpu_open: sym!("ns_gpu_open"),
            gpu_close: sym!("ns_gpu_close"),
            gpu_queue: sym!("ns_gpu_queue"),
            gpu_memory: sym!("ns_gpu_memory"),
            alloc: sym!("ns_alloc"),
            free: sym!("ns_free"),
            alloc_host: sym!("ns_alloc_host"),
            free_host: sym!("ns_free_host"),
            copy_to: sym!("ns_copy_to"),
            copy_from: sym!("ns_copy_from"),
            copy_dev: sym!("ns_copy_dev"),
            mark: sym!("ns_mark"),
            stream_copy: sym!("ns_stream_copy"),
            await_: sym!("ns_await"),
            stamp: sym!("ns_stamp"),
            elapsed: sym!("ns_elapsed"),
            fill: sym!("ns_fill"),
            sync: sym!("ns_sync"),
            copy_peer: sym!("ns_copy_peer"),
        })
    }

    /// A symbol of the library by name (null when it has none): an engine's own ABI (`kernels/<kind>/<arch>`) is
    /// bound by its crate, not here.
    pub fn symbol(&self, name: &str) -> *mut c_void {
        let Ok(n) = CString::new(name) else { return std::ptr::null_mut() };
        // SAFETY: a lookup in the handle the library was opened with; it stays open for the process.
        unsafe { dlsym(self.handle, n.as_ptr()) }
    }

    /// A model's silo (`<the library's directory>/silo/libnextsycl-<arch>.so`, or `NS_SILO_DIR`): kernels tuned for
    /// that one model, built from `kernels/<kind>/<arch>/silo/`, opened on demand. None when there is none or
    /// `NS_SILO=0` (the shared kernels then); an error when the file is there but does not open.
    pub fn silo(&self, arch: &str) -> Result<Option<Silo>, String> {
        if std::env::var("NS_SILO").is_ok_and(|v| v == "0") {
            return Ok(None);
        }
        let dir = std::env::var_os("NS_SILO_DIR").map(PathBuf::from)
                                                  .unwrap_or_else(|| self.path.parent().unwrap_or(Path::new(".")).join("silo"));
        let path = dir.join(format!("libnextsycl-{arch}.so"));
        if !path.exists() {
            return Ok(None);
        }
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: dlopen with a valid C string; the handle is checked. Its undefined symbols (the shared part's
        // ns_fail ...) resolve against the kind's library, opened RTLD_GLOBAL before it.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if h.is_null() {
            return Err(format!("{}: {}", path.display(), dl_error()));
        }
        Ok(Some(Silo { handle: h, path }))
    }

    /// The reason of the last failed call on this thread.
    pub fn error(&self) -> String {
        // SAFETY: ns_last_error returns a C string owned by the library, valid until the next call on this thread.
        unsafe { CStr::from_ptr((self.last_error)()) }.to_string_lossy().into_owned()
    }
}

/// A model's silo library (`Api::silo`): its tuned kernels, bound by the engine by name
pub struct Silo {
    handle: *mut c_void,
    pub path: PathBuf,
}

// SAFETY: a handle to a library that stays loaded for the process's life.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

impl Silo {
    /// A symbol of the silo by name (null when it has none)
    pub fn symbol(&self, name: &str) -> *mut c_void {
        let Ok(n) = CString::new(name) else { return std::ptr::null_mut() };
        // SAFETY: a lookup in the silo's handle; it stays open for the process.
        unsafe { dlsym(self.handle, n.as_ptr()) }
    }
}
