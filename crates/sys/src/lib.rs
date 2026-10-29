//! Raw bindings to `libnextsycl.so`: one function pointer per shared declaration of `kernels/ns/ns.h` (an engine
//! binds its own kernels by name: `Api::symbol`).
//!
//! The library is opened at run time (`dlopen`), not linked: the Rust side builds anywhere without the SYCL
//! toolchain, and the kernels rebuild without relinking the runtime. Nothing here is safe to call directly;
//! `ns-core` wraps it.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

include!(concat!(env!("OUT_DIR"), "/libs_embedded.rs"));

extern "C" {
    fn memfd_create(name: *const c_char, flags: u32) -> c_int;
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
    /// where it was opened from (a name, for a library out of the binary)
    path: PathBuf,
    /// the directory the silos are in (`silo/`): beside the library, or beside the executable for one out of the binary
    dir: PathBuf,
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
    /// (gpu, dst, dst_pitch, src, src_pitch, width, height): None in a library built before it
    pub copy_2d: Option<unsafe extern "C" fn(Gpu, *mut c_void, usize, *const c_void, usize, usize, usize) -> c_int>,
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

/// NS_LIB_FROM_DISK=1: the kernel libraries from files, not from the binary (rebuild a kernel without relinking)
fn from_disk() -> bool {
    std::env::var("NS_LIB_FROM_DISK").is_ok_and(|v| v == "1")
}

/// A library of the binary by file name
fn embedded(name: &str) -> Option<&'static [u8]> {
    LIBS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

/// An embedded library as a file the loader can open: an anonymous memory file (the descriptor stays open, a library's
/// path may be handed to kernels later), its /proc/self/fd path. Once a name.
fn materialize(name: &str) -> Result<PathBuf, String> {
    use std::io::Write;
    use std::os::fd::FromRawFd;
    static MADE: std::sync::Mutex<Vec<(String, PathBuf)>> = std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, p)) = made.iter().find(|(n, _)| n == name) {
        return Ok(p.clone());
    }
    let bytes = embedded(name).ok_or_else(|| format!("{name}: not in this binary"))?;
    let c = CString::new(name).map_err(|e| e.to_string())?;
    // SAFETY: memfd_create with a valid C string; the descriptor is checked.
    let fd = unsafe { memfd_create(c.as_ptr(), 0) };
    if fd < 0 {
        return Err(format!("{name}: memfd_create: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: a descriptor just made and owned here.
    let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
    f.write_all(bytes).map_err(|e| format!("{name}: {e}"))?;
    std::mem::forget(f);
    let p = PathBuf::from(format!("/proc/self/fd/{fd}"));
    made.push((name.to_string(), p.clone()));
    Ok(p)
}

/// dlopen a path globally (its symbols and its soname are then found by the libraries opened after it)
fn open(path: &Path) -> Result<(), String> {
    let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
    // SAFETY: dlopen with a valid C string; the handle is checked.
    let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if h.is_null() {
        return Err(format!("{}: {}", path.display(), dl_error()));
    }
    Ok(())
}

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
    /// copy inside the binary (unless NS_LIB_FROM_DISK=1), else the library beside the executable, else in `dist/` of
    /// the working directory.
    pub fn load(kind: &str) -> Result<Api, String> {
        let name = format!("libnextsycl-{kind}.so");
        if std::env::var_os("NEXTSYCL_LIB").is_none() && !from_disk() {
            if let Some(api) = Api::load_embedded(&name)? {
                return Ok(api);
            }
        }
        let p = std::env::var_os("NEXTSYCL_LIB").map(PathBuf::from).or_else(|| {
            let exe = std::env::current_exe().ok()?;
            let beside = exe.parent()?.join(&name);
            beside.exists().then_some(beside)
        });
        Api::load_from(&p.unwrap_or_else(|| PathBuf::from("dist").join(&name)))
    }

    /// The kind's library out of the binary (None when this binary has none): oneDNN and the flash-attention library
    /// first (the kind's library needs the one by its name; the flash path is handed to the kernels in NS_FLASH_LIB),
    /// each as an anonymous memory file opened by its /proc/self/fd path
    fn load_embedded(name: &str) -> Result<Option<Api>, String> {
        if embedded(name).is_none() {
            return Ok(None);
        }
        if embedded("libdnnl.so.3").is_some() {
            let p = materialize("libdnnl.so.3")?;
            open(&p)?;
        }
        if embedded("libnextsycl-flash.so").is_some() && std::env::var_os("NS_FLASH_LIB").is_none() {
            let p = materialize("libnextsycl-flash.so")?;
            // before the kernels (which read it) are loaded
            std::env::set_var("NS_FLASH_LIB", &p);
        }
        let p = materialize(name)?;
        let dir = std::env::var_os("NS_SILO_DIR").map(PathBuf::from).and_then(|d| d.parent().map(Path::to_path_buf))
            .or_else(|| std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf))).unwrap_or_else(|| PathBuf::from("dist"));
        Api::load_at(&p, PathBuf::from(format!("{name} (in the binary)")), dir).map(Some)
    }

    pub fn load_from(path: &Path) -> Result<Api, String> {
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Api::load_at(path, path.to_path_buf(), dir)
    }

    #[allow(clippy::missing_transmute_annotations)] // the target type is each field's, from ns.h
    fn load_at(path: &Path, shown: PathBuf, dir: PathBuf) -> Result<Api, String> {
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
            path: shown,
            dir,
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
            copy_2d: {
                let n = std::ffi::CString::new("ns_copy_2d").unwrap_or_default();
                // SAFETY: the declaration in ns.h is the field's type; a missing symbol is None.
                let p = unsafe { dlsym(h, n.as_ptr()) };
                // SAFETY: the declaration in ns.h is the field's type.
                (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(Gpu, *mut c_void, usize, *const c_void, usize, usize, usize) -> c_int>(p) })
            },
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

    /// Where the library was opened from (a name and "in the binary" for an embedded one)
    pub fn path(&self) -> &Path {
        &self.path
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
        // a model card's silo: NS_SILO_NAME another silo of this engine's ABI (default: the architecture's own),
        // NS_SILO_REQUIRED=1 no load without it (not the shared kernels)
        let name = std::env::var("NS_SILO_NAME").ok().filter(|n| !n.is_empty()).unwrap_or_else(|| arch.to_string());
        let required = std::env::var("NS_SILO_REQUIRED").is_ok_and(|v| v == "1");
        let dir = std::env::var_os("NS_SILO_DIR").map(PathBuf::from)
                                                  .unwrap_or_else(|| self.dir.join("silo"));
        let path = dir.join(format!("libnextsycl-{name}.so"));
        if !path.exists() {
            if required {
                return Err(format!("{}: the model's silo is not built (NS_SILO_REQUIRED=1; kernels/build.sh builds the silos)", path.display()));
            }
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
