//! The drives the models sit on, and moving a model's files from one to another - shared by the front door (its
//! Drives panel and `system.drives` / `system.drive.measure` / `system.move`) and `nextsycl drives` / `nextsycl models
//! move`, which works without it.
//!
//! - **A drive** is a filesystem holding a registered model's files or a models folder (NS_MODELS, NS_MODEL_DIRS): each
//!   one once, however many mounts show it (btrfs mounts one partition at several places). What it is (the disk's
//!   model, an NVMe's PCIe link from sysfs), its mount, size and free space, the models' bytes on it, and its read speed
//!   when measured (`measure`: up to 4 GiB of its largest model files read with O_DIRECT - past the page cache, no
//!   writes).
//! - **Moving** a model (`plan`, then `Move::run`): every file it names - its file and the shards beside it, its files
//!   by role, settings that are paths (a folder with what is in it), the sidecars its engine finds beside its file
//!   (`sidecars`: its model card says which) - is copied to the other drive and read back past
//!   the cache (size and SHA-256) before the next; then the registry entry is repointed in one write; then the old
//!   files go, except those another registered model still names (they stay, and that model with them). A cancel or a
//!   failure removes what the move wrote and leaves the registry as it was.
//! - **Where** the files go: a file below a models folder (NS_MODELS, NS_MODEL_DIRS, or a drive's `nextsycl-models/`)
//!   keeps its path below it, in the other drive's models folder - so a move back lands where it was; a file anywhere
//!   else keeps its layout below its model's folder (the nearest folder its files share, under its own name). A drive's
//!   models folder is the one set on it, else `<mount>/nextsycl-models`.
//! - **One move at a time** on the machine: `moving.json` beside the registry names it (the front door's or the
//!   command's), and nothing starts that model meanwhile (`moving`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::home::registry::Store;

const GIB: u64 = 1 << 30;
/// What a measurement reads at most
pub const MEASURE_BYTES: u64 = 4 * GIB;
/// A drive's models folder when none is set on it: `<mount>/nextsycl-models`
pub const FOLDER: &str = "nextsycl-models";
/// A copy's name until it has been read back
const PART: &str = ".nsmove";
/// One read (O_DIRECT: the offset, size and buffer aligned to 4 KiB)
const CHUNK: usize = 16 << 20;
const ALIGN: usize = 4096;
/// A copy's written pages flushed and dropped from the page cache every this many bytes (a model's worth of cache
/// would push out what the loaded models use)
const FLUSH: u64 = 256 << 20;
/// Readers a measurement runs at once (an NVMe reaches its rate only with several reads in flight)
const READERS: usize = 4;

#[cfg(target_arch = "x86_64")]
const O_DIRECT: i32 = 0o40000;
#[cfg(not(target_arch = "x86_64"))]
const O_DIRECT: i32 = 0o200000;

extern "C" {
    fn statvfs(path: *const std::ffi::c_char, buf: *mut u64) -> i32;
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// A mount (/proc/self/mountinfo): where, its source (a device), its filesystem
#[derive(Clone, Debug, PartialEq)]
pub struct Mount {
    pub point: PathBuf,
    pub source: String,
    pub fstype: String,
}

/// mountinfo's octal escapes (`\040` a space)
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && i + 3 < b.len() && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c)) {
            out.push((b[i + 1] - b'0') * 64 + (b[i + 2] - b'0') * 8 + (b[i + 3] - b'0'));
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The mounts of a mountinfo text, in its order
pub fn parse_mounts(text: &str) -> Vec<Mount> {
    text.lines().filter_map(|l| {
        let (a, b) = l.split_once(" - ")?;
        let point = a.split(' ').nth(4)?;
        let mut b = b.split(' ');
        let fstype = b.next()?;
        let source = b.next()?;
        Some(Mount { point: PathBuf::from(unescape(point)), source: unescape(source), fstype: fstype.to_string() })
    }).collect()
}

pub fn mounts() -> Vec<Mount> {
    parse_mounts(&std::fs::read_to_string("/proc/self/mountinfo").unwrap_or_default())
}

/// The mount a path is on: the deepest mount point it is below (of two at one point, the later)
pub fn mount_of<'a>(p: &Path, ms: &'a [Mount]) -> Option<&'a Mount> {
    ms.iter().enumerate().filter(|(_, m)| p.starts_with(&m.point)).max_by_key(|(i, m)| (m.point.components().count(), *i)).map(|(_, m)| m)
}

/// A filesystem's size and the bytes free to a user (statvfs)
pub fn space(p: &Path) -> Option<(u64, u64)> {
    let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
    // struct statvfs on 64-bit Linux: f_bsize, f_frsize, f_blocks, f_bfree, f_bavail, ... 112 bytes
    let mut b = [0u64; 16];
    // SAFETY: c is a NUL-terminated path; b (128 bytes) is larger than the struct statvfs fills.
    if unsafe { statvfs(c.as_ptr(), b.as_mut_ptr()) } != 0 {
        return None;
    }
    Some((b[2] * b[1], b[4] * b[1]))
}

/// The page cache's copy of a file dropped (its dirty pages written first: sync_data)
fn drop_cache(f: &File) {
    // SAFETY: the descriptor is the open file's; POSIX_FADV_DONTNEED (4) only advises the kernel.
    unsafe { posix_fadvise(std::os::fd::AsRawFd::as_raw_fd(f), 0, 0, 4) };
}

/// Space kept free on a drive beyond a move's files: 1% of it, at least 4 GiB
pub fn margin(size: u64) -> u64 {
    (size / 100).max(4 * GIB)
}

/// Whether `need` bytes fit on a drive with `free` of `size` and the margin left
pub fn room(need: u64, free: u64, size: u64) -> Result<(), String> {
    let m = margin(size);
    if need + m > free {
        return Err(format!("needs {:.1} GiB + {:.1} GiB kept free; {:.1} GiB free", need as f64 / GIB as f64, m as f64 / GIB as f64, free as f64 / GIB as f64));
    }
    Ok(())
}

fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// The disk a device is (`/dev/nvme0n1p3` -> `nvme0n1`): its name in /sys/block
fn disk_of(source: &str) -> Option<String> {
    let name = source.strip_prefix("/dev/")?;
    let sys = Path::new("/sys/class/block").join(name);
    let real = std::fs::canonicalize(&sys).ok()?;
    let disk = if sys.join("partition").exists() { real.parent()?.to_path_buf() } else { real };
    disk.file_name().map(|n| n.to_string_lossy().into_owned())
}

/// A PCIe link from sysfs ("16.0 GT/s PCIe", "4"): its words ("PCIe 4.0 x4") and its rate in GB/s
pub fn link(speed: &str, width: &str) -> Option<(String, f64)> {
    let gts: f64 = speed.split_whitespace().next()?.parse().ok()?;
    let w: f64 = width.trim().parse().ok()?;
    let gen = match gts as u32 {
        0..=2 => 1,
        3..=5 => 2,
        6..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        _ => 6,
    };
    // 8b/10b to PCIe 2, 128b/130b after
    let enc = if gen <= 2 { 0.8 } else { 128.0 / 130.0 };
    Some((format!("PCIe {gen}.0 x{w}"), gts * w * enc / 8.0))
}

/// A drive: a filesystem that holds models
#[derive(Clone, Debug)]
pub struct Drive {
    /// its name: the device's (`nvme0n1p3`), else the mount point
    pub id: String,
    /// the disk (`nvme0n1`) and its model, as sysfs has them
    pub disk: String,
    pub model: String,
    /// its PCIe link (an NVMe's) in words and GB/s
    pub link: Option<(String, f64)>,
    pub source: String,
    pub fstype: String,
    /// the mount its models folder is on
    pub mount: PathBuf,
    /// where models moved here go
    pub folder: PathBuf,
    pub size: u64,
    pub free: u64,
    /// the registered models' bytes on it (a file once), and those models
    pub bytes: u64,
    pub models: Vec<String>,
}

impl Drive {
    /// Whether a path is on this drive
    pub fn holds(&self, p: &Path, ms: &[Mount]) -> bool {
        mount_of(p, ms).is_some_and(|m| m.source == self.source)
    }

    pub fn json(&self) -> Value {
        json!({"id": self.id, "disk": self.disk, "model": self.model, "link": self.link.as_ref().map(|l| &l.0),
               "link_gb_s": self.link.as_ref().map(|l| (l.1 * 100.0).round() / 100.0), "device": self.source, "fstype": self.fstype,
               "mount": self.mount, "folder": self.folder, "size": self.size, "free": self.free, "margin": margin(self.size),
               "models_bytes": self.bytes, "models": self.models})
    }
}

/// A path made absolute through its links where it exists (as it is where it does not)
fn real(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Every drive holding a registered model's file or one of the models folders `dirs` (each filesystem once, in the
/// order first met: the folders first)
pub fn drives(models: &[Value], dirs: &[PathBuf]) -> Vec<Drive> {
    let ms = mounts();
    let mut out: Vec<Drive> = Vec::new();
    let mut seen: BTreeMap<String, BTreeSet<PathBuf>> = BTreeMap::new();
    let mut add = |p: &Path, folder: bool, model: Option<(&str, u64)>, out: &mut Vec<Drive>| {
        let Some(mt) = mount_of(p, &ms) else { return };
        let i = match out.iter().position(|d| d.source == mt.source) {
            Some(i) => i,
            None => {
                let disk = disk_of(&mt.source).unwrap_or_default();
                let sys = Path::new("/sys/block").join(&disk).join("device");
                let lk = read_trim(&sys.join("device/current_link_speed")).zip(read_trim(&sys.join("device/current_link_width")))
                    .and_then(|(s, w)| link(&s, &w));
                let (size, free) = space(&mt.point).unwrap_or((0, 0));
                let id = mt.source.strip_prefix("/dev/").map(|s| s.rsplit('/').next().unwrap_or(s).to_string()).unwrap_or_else(|| mt.point.display().to_string());
                out.push(Drive { id, disk: disk.clone(), model: read_trim(&sys.join("model")).unwrap_or_default(), link: lk, source: mt.source.clone(),
                                 fstype: mt.fstype.clone(), mount: mt.point.clone(), folder: mt.point.join(FOLDER), size, free, bytes: 0, models: Vec::new() });
                // a folder set on it is where its models go (the first)
                if folder {
                    out.last_mut().unwrap().folder = p.to_path_buf();
                }
                out.len() - 1
            }
        };
        if let Some((id, len)) = model {
            let d = &mut out[i];
            if seen.entry(d.source.clone()).or_default().insert(p.to_path_buf()) {
                d.bytes += len;
            }
            if !d.models.iter().any(|m| m == id) {
                d.models.push(id.to_string());
            }
        }
    };
    for d in dirs {
        add(&real(d), true, None, &mut out);
    }
    for m in models {
        let id = m["id"].as_str().unwrap_or("");
        for f in files_of(m) {
            if let Ok(md) = std::fs::metadata(&f) {
                add(&f, false, Some((id, md.len())), &mut out);
            }
        }
    }
    // a drive's mount: the one its folder is on (btrfs shows a partition at several)
    for d in &mut out {
        if let Some(mt) = mount_of(&d.folder, &ms) {
            d.mount = mt.point.clone();
        }
    }
    out
}

/// The models folders set: NS_MODELS, then NS_MODEL_DIRS (commas or spaces) - each the folder models moved to its
/// drive go to
pub fn folders(models: Option<String>, more: Option<String>) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    let more: Vec<String> = more.iter().flat_map(|m| m.split([',', ' '])).map(str::to_string).collect();
    for d in models.into_iter().chain(more) {
        let d = d.trim();
        if d.starts_with('/') {
            let p = real(Path::new(d.trim_end_matches('/')));
            if !v.contains(&p) {
                v.push(p);
            }
        }
    }
    v
}

/// Every models folder: those set, and each drive's
pub fn roots(set: &[PathBuf], all: &[Drive]) -> Vec<PathBuf> {
    let mut v = set.to_vec();
    for d in all {
        if !v.contains(&d.folder) {
            v.push(d.folder.clone());
        }
    }
    v
}

/// Whether a model reads its weights from the disk while it serves (its card's `streams`: experts or rows read from
/// the file as decode needs them) - its drive's speed is its speed
pub fn streams(m: &Value, cards: &[Value]) -> bool {
    if let Some(b) = m["streams"].as_bool() {
        return b;
    }
    let id = m["catalog"].as_str().or(m["id"].as_str()).unwrap_or("");
    cards.iter().any(|c| c["id"] == id && c["streams"] == true)
}

/// A drive's read rate in GB/s: as measured (MB/s), else its link's
pub fn rate(d: &Drive, measured_mb_s: Option<f64>) -> Option<f64> {
    measured_mb_s.map(|m| m / 1000.0).or(d.link.as_ref().map(|l| l.1))
}

/// A drive by its id or its mount point
pub fn find<'a>(all: &'a [Drive], name: &str) -> Option<&'a Drive> {
    let name = name.trim_end_matches('/');
    all.iter().find(|d| d.id == name || d.mount == Path::new(name) || d.disk == name)
}

/// A split GGUF's shards from its first (`-00001-of-0000N`), the first first
pub fn shards(first: &Path) -> Vec<PathBuf> {
    let name = first.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some(i) = name.find("-00001-of-") {
        if let Some(of) = name.get(i + 10..i + 15) {
            if let Ok(n) = of.parse::<usize>() {
                return (1..=n).map(|k| first.with_file_name(format!("{}-{k:05}-of-{of}{}", &name[..i], &name[i + 15..]))).collect();
            }
        }
    }
    vec![first.to_path_buf()]
}

/// A path setting's parts: `path`, `path:scale`, comma lists - each part as (path, what follows it)
fn parts(val: &str) -> Vec<(&str, &str)> {
    val.split(',').map(|part| match part.rsplit_once(':') {
        Some((p, scale)) if scale.parse::<f64>().is_ok() => (p, &part[p.len()..]),
        _ => (part, ""),
    }).collect()
}

/// The paths an entry names, as it writes them: its file, its files by role, settings that are absolute paths that
/// exist (`path`, `path:scale`, comma lists)
pub fn named(m: &Value) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let mut push = |s: &str| {
        if !s.is_empty() && !v.iter().any(|x| x == s) {
            v.push(s.to_string());
        }
    };
    push(m["file"].as_str().unwrap_or(""));
    for p in m["files"].as_object().into_iter().flatten().filter_map(|(_, p)| p.as_str()) {
        push(p);
    }
    for val in m["env"].as_object().into_iter().flatten().filter_map(|(_, v)| v.as_str()) {
        for (p, _) in parts(val) {
            if p.starts_with('/') && Path::new(p).exists() {
                push(p);
            }
        }
    }
    // the sidecars found beside its file (with_sidecars)
    for p in m[SIDECAR_PATHS].as_array().into_iter().flatten().filter_map(Value::as_str) {
        push(p);
    }
    v
}

/// Where with_sidecars puts an entry's sidecars (in memory only: never written to the registry)
const SIDECAR_PATHS: &str = "_sidecar_paths";

/// Whether a file name matches a pattern with `*` (any run of characters)
fn glob(pat: &str, name: &str) -> bool {
    match pat.split_once('*') {
        None => pat == name,
        Some((head, rest)) => name.strip_prefix(head).is_some_and(|tail| (0..=tail.len()).filter(|i| tail.is_char_boundary(*i)).any(|i| glob(rest, &tail[i..]))),
    }
}

/// An entry's sidecars: what its engine finds beside its model file by convention and no registry path names - a
/// draft block (deepseek4's `deepseek-v4-flash-mtp.gguf`), a vision tower (`*.vision.gguf`), tables in a folder
/// (`deepseek-engram/`). The entry's `sidecars`, else its model card's: names or `*` patterns relative to the model
/// file's folder (a folder: what is in it); those that exist
pub fn sidecars(m: &Value, cards: &[Value]) -> Vec<PathBuf> {
    let id = m["catalog"].as_str().or(m["id"].as_str()).unwrap_or("");
    let pats = m["sidecars"].as_array().or_else(|| cards.iter().find(|c| c["id"] == id).and_then(|c| c["sidecars"].as_array()));
    let main = m["file"].as_str().filter(|f| !f.is_empty()).or_else(|| m["files"]["model"].as_str());
    let (Some(pats), Some(dir)) = (pats, main.and_then(|f| Path::new(f).parent())) else { return Vec::new() };
    let mut out: Vec<PathBuf> = Vec::new();
    for pat in pats.iter().filter_map(Value::as_str).map(|p| p.trim_end_matches('/')).filter(|p| !p.is_empty() && !p.contains("..")) {
        let found: Vec<PathBuf> = if pat.contains('*') {
            let (sub, leaf) = pat.rsplit_once('/').map_or((dir.to_path_buf(), pat), |(s, l)| (dir.join(s), l));
            let mut v: Vec<PathBuf> = std::fs::read_dir(&sub).into_iter().flatten().flatten()
                .filter(|e| glob(leaf, &e.file_name().to_string_lossy())).map(|e| e.path()).collect();
            v.sort();
            v
        } else {
            vec![dir.join(pat)]
        };
        for p in found.into_iter().filter(|p| p.exists()) {
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// The entries with their sidecars (sidecars) among the paths they name, for a move and the drives' view: a sidecar
/// moves with its model and stays where another entry still has it
pub fn with_sidecars(models: &[Value], cards: &[Value]) -> Vec<Value> {
    models.iter().map(|m| {
        let mut m = m.clone();
        let s = sidecars(&m, cards);
        if !s.is_empty() {
            m[SIDECAR_PATHS] = json!(s.iter().map(|p| p.display().to_string()).collect::<Vec<_>>());
        }
        m
    }).collect()
}

/// The files below a folder (links to folders not followed)
fn walk(d: &Path, out: &mut Vec<PathBuf>) {
    let mut es: Vec<_> = std::fs::read_dir(d).into_iter().flatten().flatten().collect();
    es.sort_by_key(|e| e.file_name());
    for e in es {
        let p = e.path();
        match std::fs::symlink_metadata(&p) {
            Ok(md) if md.is_dir() => walk(&p, out),
            Ok(_) => out.push(p),
            Err(_) => {}
        }
    }
}

/// Every file of an entry, each by its real path, once: what it names, a split file's other shards, what is in a
/// folder it names
pub fn files_of(m: &Value) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for n in named(m) {
        let p = PathBuf::from(&n);
        let mut fs = Vec::new();
        if p.is_dir() {
            walk(&p, &mut fs);
        } else {
            fs = shards(&p).into_iter().filter(|s| *s == p || s.exists()).collect();
        }
        for f in fs {
            let f = real(&f);
            if !out.contains(&f) {
                out.push(f);
            }
        }
    }
    out
}

/// The folder a model's files outside every models folder share, and the name it keeps on the other drive: the nearest
/// folder holding them all (one file: its folder), under its own name - or the model's id where that folder is a mount
/// point or `/` (a name that says nothing)
pub fn base_of(paths: &[PathBuf], mount_points: &[PathBuf], id: &str) -> (PathBuf, String) {
    let mut base: Option<PathBuf> = None;
    for p in paths {
        let parent = p.parent().unwrap_or(Path::new("/")).to_path_buf();
        base = Some(match base {
            None => parent,
            Some(b) => {
                let mut c = PathBuf::new();
                for (x, y) in b.components().zip(parent.components()) {
                    if x != y {
                        break;
                    }
                    c.push(x);
                }
                c
            }
        });
    }
    let base = base.unwrap_or_else(|| PathBuf::from("/"));
    let plain = base.parent().is_none() || mount_points.contains(&base);
    let name = if plain { id.to_string() } else { base.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| id.to_string()) };
    (base, name)
}

/// Where `p` goes in the models folder `to`: its path below the models folder it is in (the deepest of `roots`), else
/// its path below `base` under `name` (base_of)
pub fn target_of(p: &Path, roots: &[PathBuf], base: &Path, name: &str, to: &Path) -> PathBuf {
    if let Some(r) = roots.iter().filter(|r| p.starts_with(r) && p != r.as_path()).max_by_key(|r| r.components().count()) {
        return to.join(p.strip_prefix(r).unwrap_or(p));
    }
    match p.strip_prefix(base) {
        Ok(rel) => to.join(name).join(rel),
        Err(_) => to.join(name).join(p.file_name().unwrap_or_default()),
    }
}

/// The entry with every path it names renamed (`file`, `files`, path settings - `path:scale` and lists kept)
pub fn rewrite(m: &mut Value, rename: &BTreeMap<String, String>) {
    let to = |s: &str| rename.get(s).cloned();
    if let Some(n) = m["file"].as_str().and_then(to) {
        m["file"] = json!(n);
    }
    if let Some(fs) = m.get_mut("files").and_then(Value::as_object_mut) {
        for v in fs.values_mut() {
            if let Some(n) = v.as_str().and_then(to) {
                *v = json!(n);
            }
        }
    }
    if let Some(env) = m.get_mut("env").and_then(Value::as_object_mut) {
        for v in env.values_mut() {
            let Some(val) = v.as_str() else { continue };
            let new: Vec<String> = parts(val).into_iter().map(|(p, rest)| format!("{}{rest}", to(p).unwrap_or_else(|| p.to_string()))).collect();
            let new = new.join(",");
            if new != val {
                *v = json!(new);
            }
        }
    }
}

/// Which entries name a file (each file of every entry, by its real path)
pub fn users(models: &[Value]) -> BTreeMap<PathBuf, Vec<String>> {
    let mut out: BTreeMap<PathBuf, Vec<String>> = BTreeMap::new();
    for m in models {
        let id = m["id"].as_str().unwrap_or("").to_string();
        for f in files_of(m) {
            let v = out.entry(f).or_default();
            if !v.contains(&id) {
                v.push(id.clone());
            }
        }
    }
    out
}

/// How a file gets to the other drive
#[derive(Clone, Debug, PartialEq)]
pub enum How {
    /// copied and read back
    Copy,
    /// another name of a file copied before it (a hard link): linked to that copy
    Link(PathBuf),
    /// there already (a copy left by a move that did not finish, or another model's): checked to be the same bytes
    There,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub from: PathBuf,
    pub to: PathBuf,
    pub bytes: u64,
    pub how: How,
}

/// What moving a model to a drive does
#[derive(Clone, Debug)]
pub struct Plan {
    pub model: String,
    /// the drive's id and models folder
    pub to: String,
    pub folder: PathBuf,
    /// the drives its files leave
    pub from: Vec<String>,
    pub steps: Vec<Step>,
    /// each path the entry names that moves, to its new path
    pub rename: BTreeMap<String, String>,
    /// bytes to write there
    pub need: u64,
    /// the moved files' bytes
    pub bytes: u64,
    /// the folders the old files' empty folders are removed up to (models folders, mounts, the model folder's parent)
    pub stops: Vec<PathBuf>,
    /// the model cards (each entry's sidecars: the old ones another entry still has stay)
    pub cards: Vec<Value>,
}

/// The plan for moving entry `id` of `models` to drive `to` (`all`: every drive; `roots`: the models folders): its
/// files not on `to` already, each to its place there (target_of)
pub fn plan(models: &[Value], id: &str, to: &Drive, all: &[Drive], roots: &[PathBuf], ms: &[Mount], cards: &[Value]) -> Result<Plan, String> {
    let models = &with_sidecars(models, cards);
    let m = models.iter().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id} (nextsycl models list)"))?;
    let files = files_of(m);
    if files.is_empty() {
        return Err(format!("{id}: no files"));
    }
    let mut leave: Vec<(PathBuf, std::fs::Metadata)> = Vec::new();
    for f in files {
        let md = std::fs::metadata(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        if !to.holds(&f, ms) {
            leave.push((f, md));
        }
    }
    if leave.is_empty() {
        return Err(format!("{id}: its files are on {} already", to.id));
    }
    // the named paths that move (a folder named with its files)
    let names: Vec<(String, PathBuf)> = named(m).into_iter().map(|n| (n.clone(), real(Path::new(&n)))).filter(|(_, p)| !to.holds(p, ms)).collect();
    let points: Vec<PathBuf> = ms.iter().map(|x| x.point.clone()).collect();
    let in_root = |p: &Path| roots.iter().any(|r| p.starts_with(r) && p != r.as_path());
    let loose: Vec<PathBuf> = names.iter().map(|(_, p)| p.clone()).chain(leave.iter().map(|(p, _)| p.clone())).filter(|p| !in_root(p)).collect();
    let (base, bname) = base_of(&loose, &points, id);
    let place = |p: &Path| target_of(p, roots, &base, &bname, &to.folder);
    let mut steps: Vec<Step> = Vec::new();
    let mut inodes: BTreeMap<(u64, u64), PathBuf> = BTreeMap::new();
    let mut from: Vec<String> = Vec::new();
    for (f, md) in &leave {
        if let Some(d) = all.iter().find(|d| d.holds(f, ms)) {
            if !from.contains(&d.id) {
                from.push(d.id.clone());
            }
        }
        let t = place(f);
        if steps.iter().any(|s| s.to == t) {
            return Err(format!("{} and another of its files would both be {}", f.display(), t.display()));
        }
        let how = if let Some(first) = inodes.get(&(md.dev(), md.ino())) {
            How::Link(first.clone())
        } else if let Ok(there) = std::fs::metadata(&t) {
            if there.len() != md.len() || !there.is_file() {
                return Err(format!("{}: another file is there already ({} bytes, this one {})", t.display(), there.len(), md.len()));
            }
            How::There
        } else {
            How::Copy
        };
        if how == How::Copy || how == How::There {
            inodes.insert((md.dev(), md.ino()), t.clone());
        }
        steps.push(Step { from: f.clone(), to: t, bytes: md.len(), how });
    }
    let rename = names.iter().map(|(n, p)| (n.clone(), place(p).display().to_string())).collect();
    let need = steps.iter().filter(|s| s.how == How::Copy).map(|s| s.bytes).sum();
    let bytes = steps.iter().map(|s| s.bytes).sum();
    let mut stops: Vec<PathBuf> = roots.to_vec();
    stops.extend(points);
    stops.extend(base.parent().map(Path::to_path_buf));
    Ok(Plan { model: id.to_string(), to: to.id.clone(), folder: to.folder.clone(), from, steps, rename, need, bytes, stops, cards: cards.to_vec() })
}

/// A file opened to read past the page cache (O_DIRECT; plainly where the filesystem has none), and whether it is
fn open_direct(p: &Path) -> Result<(File, bool), String> {
    match OpenOptions::new().read(true).custom_flags(O_DIRECT).open(p) {
        Ok(f) => Ok((f, true)),
        Err(_) => File::open(p).map(|f| (f, false)).map_err(|e| format!("{}: {e}", p.display())),
    }
}

/// A 4 KiB-aligned window of `buf` (CHUNK bytes): where O_DIRECT reads land
fn aligned(buf: &mut [u8]) -> (usize, &mut [u8]) {
    let a = buf.as_ptr().align_offset(ALIGN);
    (a, &mut buf[a..a + CHUNK])
}

/// One chunk at `off` (short at the end of the file); a filesystem that refuses O_DIRECT reads gets plain ones
fn read_chunk(f: &mut File, direct: &mut bool, p: &Path, buf: &mut [u8], off: u64) -> Result<usize, String> {
    let mut n = 0;
    while n < buf.len() {
        match f.read_at(&mut buf[n..], off + n as u64) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if *direct && e.kind() == std::io::ErrorKind::InvalidInput => {
                *f = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
                *direct = false;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(format!("{}: {e}", p.display())),
        }
    }
    Ok(n)
}

/// A whole file read past the page cache, chunk by chunk into `sink` (a reader thread keeps the next chunks coming
/// while it works): the bytes read
fn stream(p: &Path, cancel: &AtomicBool, mut sink: impl FnMut(&[u8]) -> Result<(), String>) -> Result<u64, String> {
    let (mut f, mut direct) = open_direct(p)?;
    let path = p.to_path_buf();
    std::thread::scope(|sc| {
        let (tx, rx) = mpsc::sync_channel::<Result<(Vec<u8>, usize, usize), String>>(2);
        let (back, spare) = mpsc::channel::<Vec<u8>>();
        sc.spawn(move || {
            let mut off = 0u64;
            loop {
                let mut buf = spare.try_recv().unwrap_or_else(|_| vec![0u8; CHUNK + ALIGN]);
                let (a, win) = aligned(&mut buf);
                let r = read_chunk(&mut f, &mut direct, &path, win, off);
                let n = match r {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                };
                off += n as u64;
                if tx.send(Ok((buf, a, n))).is_err() || n < CHUNK {
                    break;
                }
            }
        });
        let mut total = 0u64;
        for msg in rx {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            let (buf, a, n) = msg?;
            sink(&buf[a..a + n])?;
            total += n as u64;
            let _ = back.send(buf);
        }
        Ok(total)
    })
}

/// `from` copied to `to` (a new file) and read back past the cache: the same size and SHA-256, or the copy removed and
/// why. `done` counts the bytes copied and read back.
pub fn copy_checked(from: &Path, to: &Path, done: &AtomicU64, cancel: &AtomicBool) -> Result<(), String> {
    let r = (|| {
        let len = std::fs::metadata(from).map_err(|e| format!("{}: {e}", from.display()))?.len();
        let mut out = OpenOptions::new().write(true).create_new(true).open(to).map_err(|e| format!("{}: {e}", to.display()))?;
        let mut h = Sha256::new();
        let mut unflushed = 0u64;
        let n = stream(from, cancel, |b| {
            h.update(b);
            out.write_all(b).map_err(|e| format!("{}: {e}", to.display()))?;
            done.fetch_add(b.len() as u64, Ordering::Relaxed);
            unflushed += b.len() as u64;
            if unflushed >= FLUSH {
                out.sync_data().map_err(|e| format!("{}: {e}", to.display()))?;
                drop_cache(&out);
                unflushed = 0;
            }
            Ok(())
        })?;
        out.sync_all().map_err(|e| format!("{}: {e}", to.display()))?;
        drop_cache(&out);
        drop(out);
        if n != len {
            return Err(format!("{}: read {n} bytes of {len}", from.display()));
        }
        let want = h.finalize();
        let mut h = Sha256::new();
        let back = stream(to, cancel, |b| {
            h.update(b);
            done.fetch_add(b.len() as u64, Ordering::Relaxed);
            Ok(())
        })?;
        if back != len || h.finalize() != want {
            return Err(format!("{}: the copy reads back otherwise ({back} bytes of {len}, or another SHA-256)", to.display()));
        }
        // the old file's mode and time
        if let Ok(md) = std::fs::metadata(from) {
            let _ = std::fs::set_permissions(to, md.permissions());
            if let (Ok(t), Ok(f)) = (md.modified(), File::options().write(true).open(to)) {
                let _ = f.set_modified(t);
            }
        }
        Ok(())
    })();
    if r.is_err() {
        let _ = std::fs::remove_file(to);
    }
    r
}

/// Whether two files hold the same bytes (size, then SHA-256 past the cache); `done` counts both reads
fn same(a: &Path, b: &Path, done: &AtomicU64, cancel: &AtomicBool) -> Result<bool, String> {
    let la = std::fs::metadata(a).map_err(|e| format!("{}: {e}", a.display()))?.len();
    let lb = std::fs::metadata(b).map_err(|e| format!("{}: {e}", b.display()))?.len();
    if la != lb {
        return Ok(false);
    }
    let mut ha = Sha256::new();
    stream(a, cancel, |x| {
        ha.update(x);
        done.fetch_add(x.len() as u64, Ordering::Relaxed);
        Ok(())
    })?;
    let mut hb = Sha256::new();
    stream(b, cancel, |x| {
        hb.update(x);
        done.fetch_add(x.len() as u64, Ordering::Relaxed);
        Ok(())
    })?;
    Ok(ha.finalize() == hb.finalize())
}

/// A device number's major and minor (glibc's encoding), as /proc/<pid>/maps writes them
fn dev_parts(dev: u64) -> (u64, u64) {
    (((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff), (dev & 0xff) | ((dev >> 12) & !0xff))
}

/// The processes that have one of these files open or mapped, as "pid (program)" (a loaded model, a container's
/// server: they show here as on the host)
pub fn open_by(files: &[PathBuf]) -> Vec<String> {
    let ids: BTreeSet<(u64, u64)> = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| (m.dev(), m.ino())).collect();
    let maps: BTreeSet<(u64, u64, u64)> = ids.iter().map(|(d, i)| {
        let (a, b) = dev_parts(*d);
        (a, b, *i)
    }).collect();
    let me = std::process::id().to_string();
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let pid = e.file_name().to_string_lossy().into_owned();
        if !pid.bytes().all(|c| c.is_ascii_digit()) || pid == me {
            continue;
        }
        let fds = std::fs::read_dir(e.path().join("fd")).into_iter().flatten().flatten()
            .any(|fd| std::fs::metadata(fd.path()).is_ok_and(|m| ids.contains(&(m.dev(), m.ino()))));
        let mapped = !fds && std::fs::read_to_string(e.path().join("maps")).unwrap_or_default().lines().any(|l| {
            let mut w = l.split_whitespace().skip(3);
            let (Some(dev), Some(ino)) = (w.next(), w.next()) else { return false };
            let Some((a, b)) = dev.split_once(':') else { return false };
            match (u64::from_str_radix(a, 16), u64::from_str_radix(b, 16), ino.parse::<u64>()) {
                (Ok(a), Ok(b), Ok(i)) => i != 0 && maps.contains(&(a, b, i)),
                _ => false,
            }
        });
        if fds || mapped {
            let comm = read_trim(&e.path().join("comm")).unwrap_or_default();
            out.push(format!("{pid} ({comm})"));
        }
    }
    out
}

/// What a measurement of drive `d` reads: the registered models' largest files there (8 at most), the largest first
pub fn largest_on(models: &[Value], d: &Drive, ms: &[Mount]) -> Vec<PathBuf> {
    let mut files: Vec<(u64, PathBuf)> = models.iter().flat_map(files_of).filter(|f| d.holds(f, ms))
        .filter_map(|f| std::fs::metadata(&f).ok().map(|m| (m.len(), f))).collect();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    let mut out: Vec<PathBuf> = Vec::new();
    for (_, f) in files {
        if !out.contains(&f) && out.len() < 8 {
            out.push(f);
        }
    }
    out
}

/// A drive's read speed: up to `limit` bytes of these files (the caller lists the largest first) read past the page
/// cache, READERS reads of 16 MiB in flight in file order - (bytes, seconds). Nothing is written.
pub fn measure(files: &[PathBuf], limit: u64, cancel: &AtomicBool) -> Result<(u64, f64), String> {
    let mut chunks: Vec<(usize, u64, usize)> = Vec::new();
    let mut total = 0u64;
    'files: for (i, f) in files.iter().enumerate() {
        let len = std::fs::metadata(f).map_err(|e| format!("{}: {e}", f.display()))?.len();
        let mut off = 0;
        while off < len {
            let n = (len - off).min(CHUNK as u64);
            chunks.push((i, off, n as usize));
            total += n;
            off += n;
            if total >= limit {
                break 'files;
            }
        }
    }
    if chunks.is_empty() {
        return Err("no model file to read there".into());
    }
    // O_DIRECT or nothing: a plain read would measure the page cache
    for f in files {
        if !open_direct(f)?.1 {
            return Err(format!("{}: the filesystem does not read past the page cache (O_DIRECT)", f.display()));
        }
    }
    let next = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    let read: Result<Vec<u64>, String> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..READERS).map(|_| sc.spawn(|| {
            let mut buf = vec![0u8; CHUNK + ALIGN];
            let mut open: BTreeMap<usize, File> = BTreeMap::new();
            let mut got = 0u64;
            loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(i, off, n)) = chunks.get(k) else { break };
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".to_string());
                }
                let f = match open.entry(i) {
                    std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::btree_map::Entry::Vacant(e) => e.insert(open_direct(&files[i])?.0),
                };
                let (_, win) = aligned(&mut buf);
                // the last chunk of a file rounded up to the alignment: the read stops at the end
                let want = n.div_ceil(ALIGN) * ALIGN;
                let mut r = 0;
                while r < n {
                    match f.read_at(&mut win[r..want], off + r as u64) {
                        Ok(0) => break,
                        Ok(k) => r += k,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(e) => return Err(format!("{}: {e}", files[i].display())),
                    }
                }
                got += r.min(n) as u64;
            }
            Ok(got)
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a reader panicked".into()))).collect()
    });
    let bytes: u64 = read?.iter().sum();
    Ok((bytes, t0.elapsed().as_secs_f64()))
}

/// `moving.json` beside the registry: the move under way on this machine
pub fn lock_file(registry: &Path) -> PathBuf {
    registry.with_file_name("moving.json")
}

/// The move under way on this machine ({model, to, pid, started}); one whose process has ended is none
pub fn moving(registry: &Path) -> Option<Value> {
    let v: Value = serde_json::from_slice(&std::fs::read(lock_file(registry)).ok()?).ok()?;
    let pid = v["pid"].as_u64()?;
    Path::new(&format!("/proc/{pid}")).exists().then_some(v)
}

/// Why `id` cannot start now: it is moving (a start refuses it)
pub fn moving_why(registry: &Path, id: &str) -> Option<String> {
    let v = moving(registry).filter(|v| v["model"] == id)?;
    Some(format!("{id} is moving to {} (its files are being copied): it starts once the move ends", v["to"].as_str().unwrap_or("another drive")))
}

/// Held while a move runs: `moving.json`, removed when dropped
pub struct Lock(PathBuf);

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The machine's move lock for moving `model` to drive `to` (refused while another move runs)
pub fn lock(registry: &Path, model: &str, to: &str) -> Result<Lock, String> {
    if let Some(v) = moving(registry) {
        return Err(format!("{} is moving to {} (pid {}): one move at a time", v["model"].as_str().unwrap_or("?"), v["to"].as_str().unwrap_or("?"), v["pid"]));
    }
    let f = lock_file(registry);
    // a file left by a move whose process ended
    let _ = std::fs::remove_file(&f);
    let mut h = OpenOptions::new().write(true).create_new(true).open(&f).map_err(|e| format!("{}: {e}", f.display()))?;
    h.write_all(json!({"model": model, "to": to, "pid": std::process::id(), "started": now()}).to_string().as_bytes()).map_err(|e| format!("{}: {e}", f.display()))?;
    Ok(Lock(f))
}

/// A move: its plan, how far it is, a cancel
pub struct Move {
    pub plan: Plan,
    pub started: f64,
    /// bytes copied and read back (each copied byte counts twice: written, then checked)
    pub done: AtomicU64,
    /// the phase (copying, registry, removing, done, failed, cancelled) and the step
    at: Mutex<(String, usize)>,
    pub cancel: AtomicBool,
    /// the outcome, once it ended (and when)
    end: Mutex<Option<(f64, Result<Value, String>)>>,
    /// a word for the page (a streaming model to a slower drive)
    pub warning: Option<String>,
}

impl Move {
    pub fn new(plan: Plan, warning: Option<String>) -> Move {
        Move { plan, started: now(), done: AtomicU64::new(0), at: Mutex::new(("copying".into(), 0)), cancel: AtomicBool::new(false), end: Mutex::new(None), warning }
    }

    /// Whether it ended
    pub fn ended(&self) -> bool {
        self.end.lock().unwrap().is_some()
    }

    fn set(&self, phase: &str, step: usize) {
        *self.at.lock().unwrap() = (phase.to_string(), step);
    }

    /// The work it reads and writes: each copied byte twice (copied, read back), a file there already twice (both read)
    fn work(&self) -> u64 {
        self.plan.steps.iter().filter(|s| !matches!(s.how, How::Link(_))).map(|s| 2 * s.bytes).sum()
    }

    /// How far it is, for the page and the API
    pub fn view(&self) -> Value {
        let (phase, i) = self.at.lock().unwrap().clone();
        let end = self.end.lock().unwrap();
        let done = self.done.load(Ordering::Relaxed);
        let work = self.work().max(1);
        let secs = end.as_ref().map_or(now(), |e| e.0) - self.started;
        let rate = if secs > 0.5 { done as f64 / secs } else { 0.0 };
        let step = self.plan.steps.get(i);
        let (state, outcome, error) = match end.as_ref() {
            None => (phase.clone(), Value::Null, Value::Null),
            Some((_, Ok(v))) => ("done".to_string(), v.clone(), Value::Null),
            Some((_, Err(e))) if e == "cancelled" => ("cancelled".to_string(), Value::Null, json!(e)),
            Some((_, Err(e))) => ("failed".to_string(), Value::Null, json!(e)),
        };
        json!({"model": self.plan.model, "to": self.plan.to, "from": self.plan.from, "folder": self.plan.folder, "state": state,
               "running": end.is_none(), "files": self.plan.steps.len(), "file_index": i + 1,
               "file": step.map(|s| s.from.display().to_string()), "file_to": step.map(|s| s.to.display().to_string()),
               "bytes": self.plan.bytes, "need": self.plan.need, "done": done, "work": work,
               "pct": ((done as f64 / work as f64) * 1000.0).round() / 10.0, "rate_mb_s": (rate / 1e6).round(),
               "eta_s": if rate > 0.0 && end.is_none() { json!(((work.saturating_sub(done)) as f64 / rate).round()) } else { Value::Null },
               "started": self.started, "seconds": secs.round(), "result": outcome, "error": error, "warning": self.warning})
    }

    /// The move, to its end (the outcome kept for `view` too): copied and checked file by file, the registry entry
    /// repointed, the old files removed but those another entry names. A cancel or a failure before the registry is
    /// changed removes what it wrote.
    pub fn run(&self, store: &dyn Store) -> Result<Value, String> {
        let r = self.go(store);
        *self.end.lock().unwrap() = Some((now(), r.clone()));
        r
    }

    fn go(&self, store: &dyn Store) -> Result<Value, String> {
        let p = &self.plan;
        let mut made: Vec<PathBuf> = Vec::new();
        let mut dirs: Vec<PathBuf> = Vec::new();
        let copied = (|| -> Result<(), String> {
            for (i, s) in p.steps.iter().enumerate() {
                self.set("copying", i);
                if self.cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                let parent = s.to.parent().ok_or_else(|| format!("{}: no folder", s.to.display()))?;
                let mut fresh = Vec::new();
                let mut d = parent;
                while !d.exists() {
                    fresh.push(d.to_path_buf());
                    d = d.parent().unwrap_or(Path::new("/"));
                }
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
                dirs.extend(fresh);
                match &s.how {
                    How::Copy => {
                        let part = s.to.with_file_name(format!("{}{PART}", s.to.file_name().unwrap_or_default().to_string_lossy()));
                        // one left by a move that did not end
                        let _ = std::fs::remove_file(&part);
                        copy_checked(&s.from, &part, &self.done, &self.cancel)?;
                        std::fs::rename(&part, &s.to).map_err(|e| {
                            let _ = std::fs::remove_file(&part);
                            format!("{}: {e}", s.to.display())
                        })?;
                        made.push(s.to.clone());
                    }
                    How::Link(first) => {
                        std::fs::hard_link(first, &s.to).map_err(|e| format!("{} -> {}: {e}", first.display(), s.to.display()))?;
                        made.push(s.to.clone());
                    }
                    How::There => {
                        if !same(&s.from, &s.to, &self.done, &self.cancel)? {
                            return Err(format!("{}: another file is there already (other bytes)", s.to.display()));
                        }
                    }
                }
            }
            self.set("registry", p.steps.len().saturating_sub(1));
            store.update(&p.model, &mut |e| rewrite(e, &p.rename)).map(|_| ())
        })();
        if let Err(e) = copied {
            for f in made.iter().rev() {
                let _ = std::fs::remove_file(f);
            }
            // the deepest first
            dirs.sort_by_key(|d| std::cmp::Reverse(d.components().count()));
            for d in &dirs {
                let _ = std::fs::remove_dir(d);
            }
            return Err(e);
        }
        // the old files: those no entry names now (the registry read again: another may have been added meanwhile)
        self.set("removing", p.steps.len().saturating_sub(1));
        let names = users(&with_sidecars(&store.models().unwrap_or_default(), &p.cards));
        let mut removed = Vec::new();
        let mut kept = Vec::new();
        for s in &p.steps {
            match names.get(&s.from) {
                Some(who) => kept.push(json!({"file": s.from, "models": who})),
                None => match std::fs::remove_file(&s.from) {
                    Ok(()) => {
                        removed.push(json!(s.from));
                        prune(s.from.parent(), &p.stops);
                    }
                    Err(e) => kept.push(json!({"file": s.from, "error": e.to_string()})),
                },
            }
        }
        self.set("done", p.steps.len().saturating_sub(1));
        Ok(json!({"model": p.model, "to": p.to, "folder": p.folder, "files": p.steps.len(), "bytes": p.bytes, "written": p.need,
                  "seconds": (now() - self.started).round(), "renamed": p.rename, "removed": removed, "kept": kept}))
    }
}

/// Empty folders removed upward from `d`, stopping at the first of `stops` (or one that is not empty)
fn prune(mut d: Option<&Path>, stops: &[PathBuf]) {
    while let Some(x) = d {
        if stops.iter().any(|s| s == x) || x.parent().is_none() || std::fs::remove_dir(x).is_err() {
            break;
        }
        d = x.parent();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn mounts_parse_and_the_deepest_wins() {
        let text = "70 48 0:35 /var /var rw,relatime shared:169 - btrfs /dev/nvme0n1p3 rw\n\
                    112 70 0:35 /home /var/home rw,relatime shared:217 - btrfs /dev/nvme0n1p3 rw\n\
                    218 70 259:4 / /var/mnt/2TB\\040SSD rw,relatime shared:223 - ext4 /dev/nvme1n1 rw\n";
        let ms = parse_mounts(text);
        assert_eq!(ms.len(), 3);
        assert_eq!(ms[2].point, p("/var/mnt/2TB SSD"));
        assert_eq!(mount_of(&p("/var/home/u/models/a.gguf"), &ms).unwrap().point, p("/var/home"));
        assert_eq!(mount_of(&p("/var/mnt/2TB SSD/x"), &ms).unwrap().source, "/dev/nvme1n1");
        assert_eq!(mount_of(&p("/var/lib/x"), &ms).unwrap().point, p("/var"));
        // a sibling with a common prefix is not below it
        assert_eq!(mount_of(&p("/var/mnt/2TB SSDx/y"), &ms).unwrap().point, p("/var"));
    }

    #[test]
    fn links_read_as_their_generation() {
        let (w, gbs) = link("16.0 GT/s PCIe", "4").unwrap();
        assert_eq!(w, "PCIe 4.0 x4");
        assert!((gbs - 7.88).abs() < 0.01, "{gbs}");
        assert_eq!(link("8.0 GT/s PCIe", "4").unwrap().0, "PCIe 3.0 x4");
        assert!(link("Unknown", "4").is_none());
    }

    #[test]
    fn paths_keep_their_place_below_a_models_folder() {
        let roots = [p("/home/u/models"), p("/mnt/big/nextsycl-models")];
        let (base, name) = base_of(&[], &[], "x");
        // below a models folder: the same path below the other drive's
        assert_eq!(target_of(&p("/home/u/models/olm/a.gguf"), &roots, &base, &name, &p("/mnt/big/nextsycl-models")), p("/mnt/big/nextsycl-models/olm/a.gguf"));
        // and back
        assert_eq!(target_of(&p("/mnt/big/nextsycl-models/olm/a.gguf"), &roots, &base, &name, &p("/home/u/models")), p("/home/u/models/olm/a.gguf"));
    }

    #[test]
    fn loose_files_keep_their_layout_below_their_folder() {
        let points = [p("/"), p("/mnt/big")];
        // a model's folder with subfolders: named as it is
        let files = [p("/mnt/big/music/models/mm3/vocoder/a.st"), p("/mnt/big/music/models/mm3/transformer/b.st")];
        let (base, name) = base_of(&files, &points, "mm3-int8");
        assert_eq!((base.clone(), name.as_str()), (p("/mnt/big/music/models/mm3"), "mm3"));
        assert_eq!(target_of(&files[0], &[], &base, &name, &p("/home/u/models")), p("/home/u/models/mm3/vocoder/a.st"));
        // one file: its folder's name
        let (base, name) = base_of(&[p("/mnt/big/q27/Q8.gguf")], &points, "q27");
        assert_eq!(target_of(&p("/mnt/big/q27/Q8.gguf"), &[], &base, &name, &p("/m")), p("/m/q27/Q8.gguf"));
        // files right on a mount: the model's id names their folder
        let (base, name) = base_of(&[p("/mnt/big/a.gguf"), p("/mnt/big/b.gguf")], &points, "ab");
        assert_eq!(target_of(&p("/mnt/big/b.gguf"), &[], &base, &name, &p("/m")), p("/m/ab/b.gguf"));
        // a folder a setting names keeps its name
        let (base, name) = base_of(&[p("/mnt/big/strata/mtp/rt")], &points, "q");
        assert_eq!(target_of(&p("/mnt/big/strata/mtp/rt/w.bin"), &[], &base, &name, &p("/m")), p("/m/mtp/rt/w.bin"));
    }

    #[test]
    fn the_registry_entry_is_repointed_and_nothing_else() {
        let mut m = json!({"id": "a", "file": "/old/a-00001-of-00002.gguf", "files": {"model": "/old/a-00001-of-00002.gguf", "mmproj": "/old/mm.gguf", "other": "/keep/x"},
                           "env": {"NS_QW_CVEC": "/old/cv.gguf:1.0", "NS_LIST": "/old/p.bin,/keep/q.bin", "NS_N": "4", "NS_QW_MTP": "/old/rt"}, "title": "/old/a"});
        let rename: BTreeMap<String, String> = [("/old/a-00001-of-00002.gguf", "/new/a-00001-of-00002.gguf"), ("/old/mm.gguf", "/new/mm.gguf"),
            ("/old/cv.gguf", "/new/cv.gguf"), ("/old/p.bin", "/new/p.bin"), ("/old/rt", "/new/rt")].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        rewrite(&mut m, &rename);
        assert_eq!(m["file"], "/new/a-00001-of-00002.gguf");
        assert_eq!(m["files"]["mmproj"], "/new/mm.gguf");
        assert_eq!(m["files"]["other"], "/keep/x");
        assert_eq!(m["env"]["NS_QW_CVEC"], "/new/cv.gguf:1.0");
        assert_eq!(m["env"]["NS_LIST"], "/new/p.bin,/keep/q.bin");
        assert_eq!(m["env"]["NS_N"], "4");
        assert_eq!(m["env"]["NS_QW_MTP"], "/new/rt");
        assert_eq!(m["title"], "/old/a");
    }

    #[test]
    fn room_keeps_a_margin() {
        let size = 2000 * GIB;
        assert_eq!(margin(size), 20 * GIB);
        assert_eq!(margin(100 * GIB), 4 * GIB);
        assert!(room(100 * GIB, 121 * GIB, size).is_ok());
        assert!(room(100 * GIB, 119 * GIB, size).unwrap_err().contains("needs 100.0 GiB"));
    }

    #[test]
    fn shards_follow_the_first() {
        assert_eq!(shards(&p("/m/x-00001-of-00003.gguf")), vec![p("/m/x-00001-of-00003.gguf"), p("/m/x-00002-of-00003.gguf"), p("/m/x-00003-of-00003.gguf")]);
        assert_eq!(shards(&p("/m/x.gguf")), vec![p("/m/x.gguf")]);
    }

    /// A registry in memory
    struct Mem(Mutex<Vec<Value>>);

    impl Store for Mem {
        fn models(&self) -> Result<Vec<Value>, String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String> {
            let mut v = self.0.lock().unwrap();
            let m = v.iter_mut().find(|m| m["id"] == id).ok_or("no model")?;
            f(m);
            Ok(m.clone())
        }
        fn registry_file(&self) -> PathBuf {
            PathBuf::new()
        }
        fn get(&self, _: &str) -> Option<String> {
            None
        }
        fn in_env(&self, _: &str) -> bool {
            false
        }
        fn file(&self) -> PathBuf {
            PathBuf::new()
        }
        fn guide(&self, _: &Value) -> Option<String> {
            None
        }
    }

    /// A scratch folder of its own, gone at the end
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let d = std::env::temp_dir().join(format!("nextsycl-drives-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Tmp(real(&d))
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(p: &Path, n: usize, seed: u8) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect::<Vec<u8>>()).unwrap();
    }

    /// Two "drives" in one filesystem: a mount table that makes `a` and `b` two devices
    fn setup(t: &Tmp) -> (Vec<Mount>, Drive, Drive) {
        let (a, b) = (t.0.join("a"), t.0.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let ms = vec![Mount { point: p("/"), source: "root".into(), fstype: "x".into() },
                      Mount { point: a.clone(), source: "/dev/da".into(), fstype: "x".into() },
                      Mount { point: b.clone(), source: "/dev/db".into(), fstype: "x".into() }];
        let drive = |id: &str, m: &Path| Drive { id: id.into(), disk: id.into(), model: String::new(), link: None, source: format!("/dev/{id}"), fstype: "x".into(),
                                               mount: m.to_path_buf(), folder: m.join("models"), size: 0, free: 0, bytes: 0, models: Vec::new() };
        (ms.clone(), drive("da", &a), drive("db", &b))
    }

    #[test]
    fn a_move_copies_checks_repoints_and_keeps_what_others_name() {
        let t = Tmp::new("move");
        let (ms, da, db) = setup(&t);
        let roots = vec![da.folder.clone(), db.folder.clone()];
        let (fa, fs, fd) = (da.folder.join("m1/w-00001-of-00002.gguf"), da.folder.join("shared/tok.json"), da.folder.join("m1/rt"));
        write(&fa, 100_000, 1);
        write(&da.folder.join("m1/w-00002-of-00002.gguf"), 70_000, 2);
        write(&fs, 5_000, 3);
        write(&fd.join("x.bin"), 3_000, 4);
        write(&fd.join("sub/y.bin"), 3_000, 5);
        let models = vec![
            json!({"id": "m1", "file": fa, "files": {"model": fa, "tok": fs}, "env": {"NS_X_DIR": fd, "NS_N": "3"}}),
            json!({"id": "m2", "file": fs}),
        ];
        let store = Mem(Mutex::new(models.clone()));
        let pl = plan(&models, "m1", &db, &[da.clone(), db.clone()], &roots, &ms, &[]).unwrap();
        assert_eq!(pl.steps.len(), 5);
        assert_eq!(pl.from, vec!["da".to_string()]);
        assert_eq!(pl.need, 100_000 + 70_000 + 5_000 + 6_000);
        let mv = Move::new(pl, None);
        let out = mv.run(&store).unwrap();
        let m1 = store.models().unwrap()[0].clone();
        let nb = db.folder.join("m1/w-00001-of-00002.gguf");
        assert_eq!(m1["file"], json!(nb));
        assert_eq!(m1["files"]["tok"], json!(db.folder.join("shared/tok.json")));
        assert_eq!(m1["env"]["NS_X_DIR"], json!(db.folder.join("m1/rt")));
        assert_eq!(m1["env"]["NS_N"], "3");
        assert_eq!(std::fs::read(&nb).unwrap().len(), 100_000);
        assert!(db.folder.join("m1/w-00002-of-00002.gguf").exists() && db.folder.join("m1/rt/sub/y.bin").exists());
        // the old files are gone, but the one m2 names stays where it is
        assert!(!fa.exists() && !fd.exists() && !da.folder.join("m1").exists());
        assert!(fs.exists());
        assert_eq!(out["kept"][0]["models"], json!(["m2"]));
        assert_eq!(mv.view()["state"], "done");
        assert!(!mv.view()["running"].as_bool().unwrap());
    }

    #[test]
    fn sidecars_move_with_their_model_and_stay_for_another() {
        let t = Tmp::new("sidecars");
        let (ms, da, db) = setup(&t);
        let roots = vec![da.folder.clone(), db.folder.clone()];
        // deepseek41's layout: the model, its vision sidecar and Engram folder beside it (two shards the entry names,
        // a README the engine does not read); deepseek4's: an MTP sidecar beside the first shard
        let d41 = da.folder.join("ds41");
        let f41 = d41.join("V4.1-vq.gguf");
        write(&f41, 40_000, 1);
        write(&d41.join("V4.1.vision.gguf"), 8_000, 2);
        write(&d41.join("deepseek-engram/model-00047-of-00048.safetensors"), 9_000, 3);
        write(&d41.join("deepseek-engram/model-00048-of-00048.safetensors"), 9_000, 4);
        write(&d41.join("deepseek-engram/config.json"), 100, 5);
        write(&d41.join("README.md"), 100, 6);
        let d4 = da.folder.join("ds4");
        let f4 = d4.join("V4-00001-of-00002.gguf");
        write(&f4, 30_000, 7);
        write(&d4.join("V4-00002-of-00002.gguf"), 30_000, 8);
        write(&d4.join("deepseek-v4-flash-mtp.gguf"), 7_000, 9);
        let cards = vec![json!({"id": "ds41", "sidecars": ["*.vision.gguf", "deepseek-engram/", "none-*.gguf"]}),
                         json!({"id": "ds4", "sidecars": ["deepseek-v4-flash-mtp.gguf"]})];
        let models = vec![
            json!({"id": "ds41", "file": f41, "files": {"model": f41, "engram-1": d41.join("deepseek-engram/model-00047-of-00048.safetensors"),
                                                       "engram-2": d41.join("deepseek-engram/model-00048-of-00048.safetensors")}}),
            json!({"id": "ds4", "file": f4}),
            // a variant of ds4 on the same files (its card by `catalog`)
            json!({"id": "ds4-256k", "catalog": "ds4", "file": f4}),
        ];
        assert_eq!(sidecars(&models[0], &cards), vec![d41.join("V4.1.vision.gguf"), d41.join("deepseek-engram")]);
        assert!(sidecars(&models[0], &[]).is_empty(), "no card: none");
        assert_eq!(sidecars(&models[2], &cards), vec![d4.join("deepseek-v4-flash-mtp.gguf")]);
        // an entry's own list wins over its card's
        assert!(sidecars(&json!({"id": "ds4", "file": f4, "sidecars": []}), &cards).is_empty());
        let store = Mem(Mutex::new(models.clone()));
        let all = [da.clone(), db.clone()];
        let pl = plan(&models, "ds41", &db, &all, &roots, &ms, &cards).unwrap();
        // the model, the vision sidecar and the Engram folder's three files (the README is not its)
        assert_eq!(pl.steps.len(), 5);
        Move::new(pl, None).run(&store).unwrap();
        let n41 = db.folder.join("ds41");
        assert_eq!(store.models().unwrap()[0]["file"], json!(n41.join("V4.1-vq.gguf")));
        assert_eq!(store.models().unwrap()[0]["files"]["engram-2"], json!(n41.join("deepseek-engram/model-00048-of-00048.safetensors")));
        assert!(store.models().unwrap()[0].get(SIDECAR_PATHS).is_none(), "the registry gets no sidecar paths");
        assert!(n41.join("V4.1.vision.gguf").exists() && n41.join("deepseek-engram/config.json").exists());
        assert!(!d41.join("V4.1.vision.gguf").exists() && !d41.join("deepseek-engram").exists() && d41.join("README.md").exists());
        // ds4 moves; its variant still has the files: they are copied and stay - the sidecar with them
        let pl = plan(&models, "ds4", &db, &all, &roots, &ms, &cards).unwrap();
        assert_eq!(pl.steps.len(), 3);
        let out = Move::new(pl, None).run(&store).unwrap();
        assert!(db.folder.join("ds4/deepseek-v4-flash-mtp.gguf").exists());
        assert!(d4.join("deepseek-v4-flash-mtp.gguf").exists(), "ds4-256k still finds its sidecar");
        assert_eq!(out["kept"].as_array().unwrap().len(), 3);
        // then the variant: now nothing names the old ones
        let pl = plan(&store.models().unwrap(), "ds4-256k", &db, &all, &roots, &ms, &cards).unwrap();
        assert!(pl.steps.iter().all(|s| s.how == How::There));
        Move::new(pl, None).run(&store).unwrap();
        assert!(!d4.exists());
    }

    #[test]
    fn globs() {
        assert!(glob("*.vision.gguf", "DeepSeek-V4.1-Flash.vision.gguf"));
        assert!(!glob("*.vision.gguf", "x.gguf"));
        assert!(glob("a*b*c", "aXbYc") && glob("a*", "a") && !glob("a*b", "ab-"));
    }

    #[test]
    fn a_cancelled_move_leaves_nothing_behind() {
        let t = Tmp::new("cancel");
        let (ms, da, db) = setup(&t);
        let f = da.folder.join("m/w.gguf");
        write(&f, 50_000, 7);
        let models = vec![json!({"id": "m", "file": f})];
        let store = Mem(Mutex::new(models.clone()));
        let roots = vec![da.folder.clone(), db.folder.clone()];
        let mv = Move::new(plan(&models, "m", &db, &[da.clone(), db.clone()], &roots, &ms, &[]).unwrap(), None);
        mv.cancel.store(true, Ordering::Relaxed);
        assert_eq!(mv.run(&store).unwrap_err(), "cancelled");
        assert_eq!(store.models().unwrap()[0]["file"], json!(f));
        assert!(f.exists());
        assert!(!db.folder.exists(), "the folders it made are removed");
        assert_eq!(mv.view()["state"], "cancelled");
    }

    #[test]
    fn a_copy_there_already_is_used_only_when_it_is_the_same() {
        let t = Tmp::new("there");
        let (ms, da, db) = setup(&t);
        let f = da.folder.join("m/w.gguf");
        write(&f, 20_000, 1);
        let models = vec![json!({"id": "m", "file": f})];
        let roots = vec![da.folder.clone(), db.folder.clone()];
        // other bytes of the same size there: refused, and that file is left alone
        write(&db.folder.join("m/w.gguf"), 20_000, 2);
        let store = Mem(Mutex::new(models.clone()));
        let pl = plan(&models, "m", &db, &[da.clone(), db.clone()], &roots, &ms, &[]).unwrap();
        assert_eq!(pl.steps[0].how, How::There);
        assert_eq!(pl.need, 0);
        assert!(Move::new(pl, None).run(&store).unwrap_err().contains("other bytes"));
        assert!(db.folder.join("m/w.gguf").exists() && f.exists());
        // the same bytes: used, the old one removed
        write(&db.folder.join("m/w.gguf"), 20_000, 1);
        let pl = plan(&models, "m", &db, &[da.clone(), db.clone()], &roots, &ms, &[]).unwrap();
        Move::new(pl, None).run(&store).unwrap();
        assert!(!f.exists());
        // a size that differs is refused when planned
        write(&f, 10, 1);
        assert!(plan(&models, "m", &db, &[da, db.clone()], &roots, &ms, &[]).unwrap_err().contains("another file is there"));
    }

    #[test]
    fn users_name_every_entry_of_a_file() {
        let t = Tmp::new("users");
        let f = t.0.join("x.gguf");
        write(&f, 10, 0);
        let u = users(&[json!({"id": "a", "file": f}), json!({"id": "b", "files": {"m": f}}), json!({"id": "c", "file": t.0.join("none.gguf")})]);
        assert_eq!(u[&f], vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn the_lock_is_one_move_at_a_time() {
        let t = Tmp::new("lock");
        let reg = t.0.join("models.json");
        assert!(moving(&reg).is_none());
        let l = lock(&reg, "m", "db").unwrap();
        assert_eq!(moving(&reg).unwrap()["model"], "m");
        assert!(moving_why(&reg, "m").unwrap().contains("moving to db"));
        assert!(moving_why(&reg, "other").is_none());
        assert!(lock(&reg, "n", "da").is_err());
        drop(l);
        assert!(moving(&reg).is_none());
        // a lock whose process ended is none
        std::fs::write(lock_file(&reg), r#"{"model": "m", "to": "db", "pid": 4294967}"#).unwrap();
        assert!(moving(&reg).is_none());
        assert!(lock(&reg, "n", "da").is_ok());
    }
}
