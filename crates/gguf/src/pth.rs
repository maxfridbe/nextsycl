//! (The safetensors-style table of contents of a checkpoint, `open`, is `crate::pthst`'s, re-exported here.)
//! PyTorch checkpoints (`torch.save`), read without Python: the format of the models that ship as `.pth`, `.pt`,
//! `.ckpt` (PyTorch Lightning) or `.pth.tar`.
//!
//! - **ZIP** (PyTorch 1.6 and later): `<root>/data.pkl`, the pickled object, and each storage's bytes as
//!   `<root>/data/<key>`, stored uncompressed (ZIP64 for files past 4 GiB).
//! - **Legacy** (before 1.6): pickles back to back - a magic number, the protocol, the system's sizes, the object -
//!   then the sorted storage keys, then each storage as its element count (int64) and its bytes.
//!
//! The pickle is run by a small machine that knows what checkpoints hold: dicts, lists, tuples, numbers, strings,
//! the storages' persistent ids and `torch._utils._rebuild_tensor_v2` / `_rebuild_parameter`; any other object
//! (an optimizer's settings, a Lightning run's hyper-parameters) is read and kept opaque. Every tensor found under
//! the root object gets a dotted name by its path (`state_dict.model.encoder.weight`, `optimizer_states.0...`); its
//! bytes are read only when asked for (`f32`, `i64`), by `pread`, so a large checkpoint with optimizer state costs
//! only the tensors used.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::{Error, Result};

/// A storage's element type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtType {
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl PtType {
    pub fn size(self) -> usize {
        match self {
            PtType::F64 | PtType::I64 => 8,
            PtType::F32 | PtType::I32 => 4,
            PtType::F16 | PtType::BF16 | PtType::I16 => 2,
            PtType::I8 | PtType::U8 | PtType::Bool => 1,
        }
    }

    /// By the storage class's name (`FloatStorage`, `HalfStorage` ...)
    fn of_storage(name: &str) -> Option<PtType> {
        Some(match name {
            "FloatStorage" => PtType::F32,
            "DoubleStorage" => PtType::F64,
            "HalfStorage" => PtType::F16,
            "BFloat16Storage" => PtType::BF16,
            "LongStorage" => PtType::I64,
            "IntStorage" => PtType::I32,
            "ShortStorage" => PtType::I16,
            "CharStorage" => PtType::I8,
            "ByteStorage" => PtType::U8,
            "BoolStorage" => PtType::Bool,
            _ => return None,
        })
    }

    fn value(self, b: &[u8]) -> f64 {
        match self {
            PtType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            PtType::F64 => f64::from_le_bytes(b[..8].try_into().expect("8 bytes")),
            PtType::F16 => crate::safetensors::f16_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64,
            PtType::BF16 => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64,
            PtType::I64 => i64::from_le_bytes(b[..8].try_into().expect("8 bytes")) as f64,
            PtType::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            PtType::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            PtType::I8 => b[0] as i8 as f64,
            PtType::U8 | PtType::Bool => b[0] as f64,
        }
    }
}

/// A tensor of the checkpoint: a view of one storage
#[derive(Clone, Debug)]
pub struct PthTensor {
    pub name: String,
    pub dtype: PtType,
    pub shape: Vec<usize>,
    /// in elements
    pub stride: Vec<usize>,
    storage: String,
    /// in elements
    offset: usize,
}

impl PthTensor {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Where a storage's bytes are
enum Layout {
    /// the archive's root folder, and each entry's (local header offset, size)
    Zip { root: String, entries: HashMap<String, (u64, u64)> },
    /// each storage's (data offset, element count)
    Legacy { storages: HashMap<String, (u64, u64)> },
}

/// An open checkpoint: its tensors by dotted name
pub struct Pth {
    pub path: PathBuf,
    file: File,
    layout: Layout,
    pub tensors: BTreeMap<String, PthTensor>,
}

// ---- the pickle machine ----------------------------------------------------------------------------------------

/// A pickled value (numbers and module names are kept for what reads them, and for debugging)
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Tuple(Rc<Vec<V>>),
    List(Rc<RefCell<Vec<V>>>),
    Dict(Rc<RefCell<Vec<(V, V)>>>),
    Global(Rc<str>, Rc<str>),
    /// a persistent id: (storage key, element type)
    Storage(Rc<str>, Option<PtType>),
    Tensor(Rc<PthTensor>),
    /// anything else, kept opaque
    Obj,
    Mark,
}

fn ints(v: &V) -> Vec<usize> {
    match v {
        V::Tuple(x) => x.iter().map(|e| if let V::Int(i) = e { *i as usize } else { 0 }).collect(),
        V::List(x) => x.borrow().iter().map(|e| if let V::Int(i) = e { *i as usize } else { 0 }).collect(),
        _ => Vec::new(),
    }
}

/// Runs the pickle at `p[at..]` to its STOP: (the value, the offset after it)
fn unpickle(p: &[u8], at: usize) -> Result<(V, usize)> {
    let bad = |m: String| Error(format!("a checkpoint's pickle: {m}"));
    let mut st: Vec<V> = Vec::new();
    let mut memo: HashMap<usize, V> = HashMap::new();
    let mut i = at;
    let need = |i: usize, n: usize| -> Result<()> { if i + n <= p.len() { Ok(()) } else { Err(bad("cut short".into())) } };
    let u32at = |i: usize| u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]) as usize;
    let line = |i: usize| -> Result<(String, usize)> {
        let e = p[i..].iter().position(|c| *c == b'\n').ok_or_else(|| bad("an unterminated line".into()))?;
        Ok((String::from_utf8_lossy(&p[i..i + e]).into_owned(), i + e + 1))
    };
    fn pop_mark(st: &mut Vec<V>) -> Vec<V> {
        let m = st.iter().rposition(|v| matches!(v, V::Mark)).unwrap_or(0);
        let items = st.split_off(m + 1);
        st.pop();
        items
    }
    fn top(st: &mut [V]) -> Option<&mut V> {
        st.last_mut()
    }
    loop {
        need(i, 1)?;
        let op = p[i];
        i += 1;
        match op {
            0x80 => i += 1,                                           // PROTO
            0x95 => i += 8,                                           // FRAME
            b'.' => return Ok((st.pop().ok_or_else(|| bad("an empty stack".into()))?, i)),
            b'(' => st.push(V::Mark),
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'}' => st.push(V::Dict(Rc::new(RefCell::new(Vec::new())))),
            b']' => st.push(V::List(Rc::new(RefCell::new(Vec::new())))),
            b')' => st.push(V::Tuple(Rc::new(Vec::new()))),
            0x8f => st.push(V::Obj),                                  // EMPTY_SET
            b'K' => {
                need(i, 1)?;
                st.push(V::Int(p[i] as i64));
                i += 1;
            }
            b'M' => {
                need(i, 2)?;
                st.push(V::Int(u16::from_le_bytes([p[i], p[i + 1]]) as i64));
                i += 2;
            }
            b'J' => {
                need(i, 4)?;
                st.push(V::Int(i32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]) as i64));
                i += 4;
            }
            0x8a | 0x8b => {
                // LONG1 / LONG4: little-endian two's complement of n bytes (wider than 8: the low 8 kept)
                let (n, h) = if op == 0x8a { need(i, 1)?; (p[i] as usize, 1) } else { need(i, 4)?; (u32at(i), 4) };
                need(i + h, n)?;
                let b = &p[i + h..i + h + n];
                let mut v: i64 = 0;
                for (k, x) in b.iter().take(8).enumerate() {
                    v |= (*x as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1 << (8 * n);
                }
                st.push(V::Int(v));
                i += h + n;
            }
            b'I' | b'L' => {
                let (s, e) = line(i)?;
                i = e;
                let s = s.trim_end_matches('L');
                st.push(match s {
                    "01" => V::Bool(true),
                    "00" => V::Bool(false),
                    _ => V::Int(s.parse().unwrap_or(0)),
                });
            }
            b'F' => {
                let (s, e) = line(i)?;
                i = e;
                st.push(V::Float(s.trim().parse().unwrap_or(0.0)));
            }
            b'G' => {
                need(i, 8)?;
                st.push(V::Float(f64::from_be_bytes(p[i..i + 8].try_into().expect("8 bytes"))));
                i += 8;
            }
            b'X' | b'T' | b'B' | 0x8c | b'U' | b'C' | 0x8d | 0x8e => {
                // strings and bytes: a length of 1, 4 or 8 bytes, then the bytes
                let h = match op { 0x8c | b'U' | b'C' => 1, 0x8d | 0x8e => 8, _ => 4 };
                need(i, h)?;
                let n = match h { 1 => p[i] as usize, 4 => u32at(i), _ => u64::from_le_bytes(p[i..i + 8].try_into().expect("8 bytes")) as usize };
                need(i + h, n)?;
                let s = &p[i + h..i + h + n];
                st.push(if matches!(op, b'B' | b'C' | 0x8e) { V::Obj } else { V::Str(String::from_utf8_lossy(s).into()) });
                i += h + n;
            }
            b'S' | b'V' => {
                let (s, e) = line(i)?;
                i = e;
                st.push(V::Str(s.trim_matches(|c| c == '\'' || c == '"').into()));
            }
            b'c' | b'i' => {
                // GLOBAL / INST: module\nname\n
                let (m, e) = line(i)?;
                let (n, e) = line(e)?;
                i = e;
                if op == b'i' {
                    pop_mark(&mut st);
                    st.push(V::Obj);
                } else {
                    st.push(V::Global(m.into(), n.into()));
                }
            }
            0x93 => {
                // STACK_GLOBAL
                let n = st.pop();
                let m = st.pop();
                st.push(match (m, n) {
                    (Some(V::Str(m)), Some(V::Str(n))) => V::Global(m, n),
                    _ => V::Obj,
                });
            }
            b'p' => {
                let (s, e) = line(i)?;
                i = e;
                memo.insert(s.parse().unwrap_or(0), st.last().cloned().unwrap_or(V::None));
            }
            b'q' => {
                need(i, 1)?;
                memo.insert(p[i] as usize, st.last().cloned().unwrap_or(V::None));
                i += 1;
            }
            b'r' => {
                need(i, 4)?;
                memo.insert(u32at(i), st.last().cloned().unwrap_or(V::None));
                i += 4;
            }
            0x94 => {
                let n = memo.len();
                memo.insert(n, st.last().cloned().unwrap_or(V::None));
            }
            b'g' => {
                let (s, e) = line(i)?;
                i = e;
                st.push(memo.get(&s.parse().unwrap_or(usize::MAX)).cloned().unwrap_or(V::None));
            }
            b'h' => {
                need(i, 1)?;
                st.push(memo.get(&(p[i] as usize)).cloned().unwrap_or(V::None));
                i += 1;
            }
            b'j' => {
                need(i, 4)?;
                st.push(memo.get(&u32at(i)).cloned().unwrap_or(V::None));
                i += 4;
            }
            b'0' => {
                st.pop();
            }
            b'1' => {
                pop_mark(&mut st);
            }
            b'2' => {
                let v = st.last().cloned().unwrap_or(V::None);
                st.push(v);
            }
            b't' => {
                let items = pop_mark(&mut st);
                st.push(V::Tuple(Rc::new(items)));
            }
            b'l' => {
                let items = pop_mark(&mut st);
                st.push(V::List(Rc::new(RefCell::new(items))));
            }
            b'd' => {
                let items = pop_mark(&mut st);
                let kv = items.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).collect();
                st.push(V::Dict(Rc::new(RefCell::new(kv))));
            }
            0x91 => {
                // FROZENSET
                pop_mark(&mut st);
                st.push(V::Obj);
            }
            b'o' => {
                // OBJ: (mark, cls, args...)
                pop_mark(&mut st);
                st.push(V::Obj);
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                if st.len() < n {
                    return Err(bad("TUPLE past the stack".into()));
                }
                let items = st.split_off(st.len() - n);
                st.push(V::Tuple(Rc::new(items)));
            }
            b'Q' => {
                // BINPERSID: ('storage', storage_type, key, location, numel[, view_metadata])
                let pid = st.pop().unwrap_or(V::None);
                let v = match &pid {
                    V::Tuple(t) => match (t.get(1), t.get(2)) {
                        (Some(V::Global(_, ty)), Some(V::Str(k))) => V::Storage(k.clone(), PtType::of_storage(ty)),
                        (_, Some(V::Str(k))) => V::Storage(k.clone(), None),
                        _ => V::Obj,
                    },
                    _ => V::Obj,
                };
                st.push(v);
            }
            b'P' => {
                let (_, e) = line(i)?;
                i = e;
                st.push(V::Obj);
            }
            b'R' | 0x81 | 0x92 => {
                // REDUCE: callable(args); NEWOBJ: cls.__new__(cls, *args); NEWOBJ_EX: with kwargs too
                if op == 0x92 {
                    st.pop();
                }
                let args = st.pop().unwrap_or(V::None);
                let f = st.pop().unwrap_or(V::None);
                let a: Vec<V> = match &args { V::Tuple(a) => a.as_ref().clone(), _ => Vec::new() };
                let name = if let V::Global(_, n) = &f { n.as_ref() } else { "" };
                st.push(rebuild(name, &a));
            }
            b'b' => {
                // BUILD: an object's state (dropped: opaque objects stay opaque)
                st.pop();
            }
            b's' => {
                let v = st.pop().unwrap_or(V::None);
                let k = st.pop().unwrap_or(V::None);
                if let Some(V::Dict(d)) = top(&mut st) {
                    d.borrow_mut().push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut st);
                if let Some(V::Dict(d)) = top(&mut st) {
                    let mut d = d.borrow_mut();
                    for kv in items.chunks(2) {
                        if let [k, v] = kv {
                            d.push((k.clone(), v.clone()));
                        }
                    }
                }
            }
            b'a' => {
                let v = st.pop().unwrap_or(V::None);
                if let Some(V::List(l)) = top(&mut st) {
                    l.borrow_mut().push(v);
                }
            }
            b'e' | 0x90 => {
                // APPENDS / ADDITEMS (a set's: dropped)
                let items = pop_mark(&mut st);
                if let Some(V::List(l)) = top(&mut st) {
                    l.borrow_mut().extend(items);
                }
            }
            o => return Err(bad(format!("opcode 0x{o:02x} at byte {}", i - 1))),
        }
    }
}

/// What a REDUCE of the callable `name` with `a` makes: a tensor (torch's rebuild functions, also wrapped in
/// `_rebuild_from_type_v2`, as a Parameter or a tensor subclass saves), a dict, or an opaque object
fn rebuild(name: &str, a: &[V]) -> V {
    match name {
        "_rebuild_tensor_v2" | "_rebuild_tensor" | "_rebuild_tensor_v3" => match a.first() {
            Some(V::Storage(key, ty)) => {
                let off = if let Some(V::Int(o)) = a.get(1) { *o as usize } else { 0 };
                // v3 names the dtype itself (a global torch.float32 ...); v2 takes the storage's
                let dtype = ty.or_else(|| match a.get(5) {
                    Some(V::Global(_, d)) => dtype_named(d),
                    _ => None,
                });
                match dtype {
                    Some(dtype) => V::Tensor(Rc::new(PthTensor { name: String::new(), dtype, shape: a.get(2).map(ints).unwrap_or_default(),
                                                                stride: a.get(3).map(ints).unwrap_or_default(), storage: key.to_string(), offset: off })),
                    None => V::Obj,
                }
            }
            _ => V::Obj,
        },
        "_rebuild_parameter" | "_rebuild_parameter_with_state" => a.first().cloned().unwrap_or(V::Obj),
        // (func, type, func's args, state): the tensor func makes
        "_rebuild_from_type_v2" | "_rebuild_from_type" => match (a.first(), a.get(2)) {
            (Some(V::Global(_, f)), Some(V::Tuple(args))) => rebuild(f, args),
            _ => V::Obj,
        },
        "OrderedDict" | "dict" | "defaultdict" => V::Dict(Rc::new(RefCell::new(Vec::new()))),
        _ => V::Obj,
    }
}

fn dtype_named(d: &str) -> Option<PtType> {
    Some(match d {
        "float32" | "float" => PtType::F32,
        "float64" | "double" => PtType::F64,
        "float16" | "half" => PtType::F16,
        "bfloat16" => PtType::BF16,
        "int64" | "long" => PtType::I64,
        "int32" | "int" => PtType::I32,
        "int16" | "short" => PtType::I16,
        "int8" => PtType::I8,
        "uint8" => PtType::U8,
        "bool" => PtType::Bool,
        _ => return None,
    })
}

/// Every tensor under `v`, named by its path
fn collect(v: &V, name: &str, out: &mut BTreeMap<String, PthTensor>, depth: usize) {
    if depth > 64 {
        return;
    }
    let join = |k: &str| if name.is_empty() { k.to_string() } else { format!("{name}.{k}") };
    match v {
        V::Tensor(t) => {
            let mut t = t.as_ref().clone();
            t.name = name.to_string();
            out.insert(name.to_string(), t);
        }
        V::Dict(d) => {
            for (k, x) in d.borrow().iter() {
                let key = match k {
                    V::Str(s) => s.to_string(),
                    V::Int(i) => i.to_string(),
                    _ => continue,
                };
                collect(x, &join(&key), out, depth + 1);
            }
        }
        V::List(l) => {
            for (i, x) in l.borrow().iter().enumerate() {
                collect(x, &join(&i.to_string()), out, depth + 1);
            }
        }
        V::Tuple(l) => {
            for (i, x) in l.iter().enumerate() {
                collect(x, &join(&i.to_string()), out, depth + 1);
            }
        }
        _ => {}
    }
}

// ---- the containers --------------------------------------------------------------------------------------------

fn u16le(b: &[u8], i: usize) -> u64 {
    u16::from_le_bytes([b[i], b[i + 1]]) as u64
}
fn u32le(b: &[u8], i: usize) -> u64 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as u64
}
fn u64le(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"))
}

fn read_at(file: &File, at: u64, n: usize, what: &str) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    file.read_exact_at(&mut v, at).map_err(|e| Error(format!("{what}: {e}")))?;
    Ok(v)
}

/// A ZIP's central directory: every entry's (name, local header offset, size, method)
fn zip_entries(file: &File, len: u64, what: &str) -> Result<Vec<(String, u64, u64, u64)>> {
    let bad = |m: &str| Error(format!("{what}: not a PyTorch ZIP checkpoint: {m}"));
    let tail_len = len.min(65_557 + 20) as usize;
    let tail_at = len - tail_len as u64;
    let tail = read_at(file, tail_at, tail_len, what)?;
    let eocd = (0..tail.len().saturating_sub(21)).rev().find(|&i| u32le(&tail, i) == 0x0605_4b50).ok_or_else(|| bad("no end record"))?;
    let (mut n, mut cd_at, mut cd_len) = (u16le(&tail, eocd + 10), u32le(&tail, eocd + 16), u32le(&tail, eocd + 12));
    if eocd >= 20 && u32le(&tail, eocd - 20) == 0x0706_4b50 {
        let z = u64le(&tail, eocd - 12);
        let rec = read_at(file, z, 56, what)?;
        if u32le(&rec, 0) == 0x0606_4b50 {
            n = u64le(&rec, 32);
            cd_len = u64le(&rec, 40);
            cd_at = u64le(&rec, 48);
        }
    }
    let cd = read_at(file, cd_at, cd_len as usize, what)?;
    let mut out = Vec::with_capacity(n as usize);
    let mut i = 0usize;
    for _ in 0..n {
        if i + 46 > cd.len() || u32le(&cd, i) != 0x0201_4b50 {
            return Err(bad("a broken central directory"));
        }
        let method = u16le(&cd, i + 10);
        let mut size = u32le(&cd, i + 24);
        let (nl, el, cl) = (u16le(&cd, i + 28) as usize, u16le(&cd, i + 30) as usize, u16le(&cd, i + 32) as usize);
        let mut off = u32le(&cd, i + 42);
        let name = String::from_utf8_lossy(&cd[i + 46..i + 46 + nl]).into_owned();
        // ZIP64: the uncompressed size, the compressed size and the offset that did not fit, in that order
        let mut e = i + 46 + nl;
        while e + 4 <= i + 46 + nl + el {
            let (id, l) = (u16le(&cd, e), u16le(&cd, e + 2) as usize);
            if id == 1 {
                let mut q = e + 4;
                if size == 0xFFFF_FFFF {
                    size = u64le(&cd, q);
                    q += 8;
                }
                if u32le(&cd, i + 20) == 0xFFFF_FFFF {
                    q += 8;
                }
                if off == 0xFFFF_FFFF {
                    off = u64le(&cd, q);
                }
            }
            e += 4 + l;
        }
        out.push((name, off, size, method));
        i += 46 + nl + el + cl;
    }
    Ok(out)
}

impl Pth {
    pub fn open(path: &Path) -> Result<Pth> {
        let what = path.display().to_string();
        let file = File::open(path).map_err(|e| Error(format!("{what}: {e}")))?;
        let len = file.metadata().map_err(|e| Error(format!("{what}: {e}")))?.len();
        let head = read_at(&file, 0, len.min(4) as usize, &what)?;
        let (root, layout) = if head.starts_with(b"PK\x03\x04") {
            let ents = zip_entries(&file, len, &what)?;
            let (pkl, at, size, method) = ents.iter().find(|e| e.0.ends_with("/data.pkl") || e.0 == "data.pkl").cloned()
                .ok_or_else(|| Error(format!("{what}: no data.pkl in the archive")))?;
            if method != 0 {
                return Err(Error(format!("{what}: {pkl} is compressed")));
            }
            let entries: HashMap<String, (u64, u64)> = ents.into_iter().map(|(n, o, s, _)| (n, (o, s))).collect();
            let prefix = pkl.strip_suffix("data.pkl").unwrap_or("").to_string();
            let lh = read_at(&file, at, 30, &what)?;
            let data = at + 30 + u16le(&lh, 26) + u16le(&lh, 28);
            let p = read_at(&file, data, size as usize, &what)?;
            let (v, _) = unpickle(&p, 0)?;
            (v, Layout::Zip { root: prefix, entries })
        } else {
            // legacy: the whole file's pickles are read; the storages after them are only measured
            let b = std::fs::read(path).map_err(|e| Error(format!("{what}: {e}")))?;
            let (_magic, i) = unpickle(&b, 0)?;
            let (_proto, i) = unpickle(&b, i)?;
            let (_sys, i) = unpickle(&b, i)?;
            let (v, i) = unpickle(&b, i)?;
            let (keys, mut i) = unpickle(&b, i)?;
            let keys: Vec<String> = match &keys {
                V::List(l) => l.borrow().iter().filter_map(|k| if let V::Str(s) = k { Some(s.to_string()) } else { None }).collect(),
                _ => return Err(Error(format!("{what}: not a PyTorch checkpoint (no storage keys after the object)"))),
            };
            // each storage's element type, from the tensors that view it
            let mut all = BTreeMap::new();
            collect(&v, "", &mut all, 0);
            let mut size_of: HashMap<String, usize> = HashMap::new();
            for t in all.values() {
                size_of.insert(t.storage.clone(), t.dtype.size());
            }
            let mut storages = HashMap::new();
            for k in keys {
                if i + 8 > b.len() {
                    return Err(Error(format!("{what}: cut short in storage {k}")));
                }
                let count = u64le(&b, i);
                let es = *size_of.get(&k).ok_or_else(|| Error(format!("{what}: storage {k} has no tensor to give its type")))?;
                storages.insert(k, (i as u64 + 8, count));
                i += 8 + count as usize * es;
            }
            (v, Layout::Legacy { storages })
        };
        let mut tensors = BTreeMap::new();
        collect(&root, "", &mut tensors, 0);
        Ok(Pth { path: path.to_path_buf(), file, layout, tensors })
    }

    pub fn tensor(&self, name: &str) -> Option<&PthTensor> {
        self.tensors.get(name)
    }

    pub fn need(&self, name: &str) -> Result<&PthTensor> {
        self.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.path.display())))
    }

    /// The prefix the model's weights sit under: "state_dict." (Lightning), "model_state_dict.", "model." or "" -
    /// whichever holds `probe` (a name the model must have, without a prefix)
    pub fn prefix_of(&self, probe: &str) -> Option<String> {
        ["", "state_dict.", "model_state_dict.", "model.", "state_dict.model.", "net."].iter()
            .find(|p| self.tensors.contains_key(&format!("{p}{probe}"))).map(|p| p.to_string())
    }

    /// A tensor's elements in row-major order, each as f64 (any storage type; any strides)
    fn values(&self, t: &PthTensor) -> Result<Vec<f64>> {
        let n = t.elements();
        if n == 0 {
            return Ok(Vec::new());
        }
        let es = t.dtype.size();
        // the span of the storage the view reaches
        let last: usize = t.shape.iter().zip(&t.stride).map(|(d, s)| (d - 1) * s).sum();
        let span = last + 1;
        let (at, have) = match &self.layout {
            Layout::Zip { root, entries } => {
                let key = format!("{root}data/{}", t.storage);
                let &(lh_at, size) = entries.get(&key).ok_or_else(|| Error(format!("{}: {}'s storage {key} is missing", self.path.display(), t.name)))?;
                let lh = read_at(&self.file, lh_at, 30, &t.name)?;
                (lh_at + 30 + u16le(&lh, 26) + u16le(&lh, 28), size / es as u64)
            }
            Layout::Legacy { storages } => {
                let &(at, count) = storages.get(&t.storage).ok_or_else(|| Error(format!("{}: {}'s storage {} is missing", self.path.display(), t.name, t.storage)))?;
                (at, count)
            }
        };
        if (t.offset + span) as u64 > have {
            return Err(Error(format!("{}: {} reaches past its storage", self.path.display(), t.name)));
        }
        let raw = read_at(&self.file, at + (t.offset * es) as u64, span * es, &t.name)?;
        let contiguous = {
            let mut want = 1;
            let mut ok = true;
            for (s, d) in t.stride.iter().zip(&t.shape).rev() {
                if *d > 1 && *s != want {
                    ok = false;
                }
                want *= d;
            }
            ok
        };
        if contiguous {
            return Ok(raw.chunks_exact(es).take(n).map(|c| t.dtype.value(c)).collect());
        }
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..n {
            let o: usize = idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum();
            out.push(t.dtype.value(&raw[o * es..o * es + es]));
            for k in (0..idx.len()).rev() {
                idx[k] += 1;
                if idx[k] < t.shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        Ok(out)
    }

    /// A tensor as float32 (converted from its type), row-major
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.values(self.need(name)?)?.into_iter().map(|x| x as f32).collect())
    }

    /// A tensor as int64 (an index buffer, a step count), row-major
    pub fn i64(&self, name: &str) -> Result<Vec<i64>> {
        Ok(self.values(self.need(name)?)?.into_iter().map(|x| x as i64).collect())
    }
}

pub use crate::pthst::open;

#[cfg(test)]
mod tests {
    use super::*;

    /// A pickle (protocol 2) of {"a.weight": tensor(storage "0" float, offset 1, [2, 2], strides [1, 2]), "n": 3},
    /// as torch writes it
    fn pickle() -> Vec<u8> {
        let mut p = vec![0x80, 2, b'}', b'q', 0, b'('];
        let s = |p: &mut Vec<u8>, x: &str| {
            p.push(b'X');
            p.extend((x.len() as u32).to_le_bytes());
            p.extend(x.as_bytes());
        };
        s(&mut p, "a.weight");
        p.extend(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.push(b'(');
        p.push(b'(');
        s(&mut p, "storage");
        p.extend(b"ctorch\nFloatStorage\n");
        s(&mut p, "0");
        s(&mut p, "cpu");
        p.extend([b'K', 5, b't', b'Q']);
        p.extend([b'K', 1, b'K', 2, b'K', 2, 0x86, b'K', 1, b'K', 2, 0x86, 0x89]);
        p.extend(b"ccollections\nOrderedDict\n");
        p.extend(*b")RtR");
        s(&mut p, "n");
        p.extend([b'K', 3, b'u', b'.']);
        p
    }

    #[test]
    fn a_state_dict_unpickles() {
        let (v, end) = unpickle(&pickle(), 0).unwrap();
        assert_eq!(end, pickle().len());
        let mut ts = BTreeMap::new();
        collect(&v, "", &mut ts, 0);
        let t = &ts["a.weight"];
        assert_eq!((t.dtype, t.shape.clone(), t.stride.clone(), t.offset, t.storage.as_str()), (PtType::F32, vec![2, 2], vec![1, 2], 1, "0"));
    }

    #[test]
    fn a_legacy_file_reads_with_strides() {
        // magic, protocol, sys info, the object, the keys, then storage "0": 5 floats
        let mut b = Vec::new();
        b.extend([0x80, 2, b'K', 7, b'.']);
        b.extend([0x80, 2, b'M', 0xe9, 0x03, b'.']);
        b.extend([0x80, 2, b'}', b'.']);
        b.extend(pickle());
        b.extend([0x80, 2, b']', b'(', b'X', 1, 0, 0, 0, b'0', b'e', b'.']);
        b.extend(5u64.to_le_bytes());
        for x in [9.0f32, 1.0, 2.0, 3.0, 4.0] {
            b.extend(x.to_le_bytes());
        }
        let dir = std::env::temp_dir().join(format!("ns-pth-{}", std::process::id()));
        std::fs::write(&dir, &b).unwrap();
        let p = Pth::open(&dir).unwrap();
        // storage [9, 1, 2, 3, 4] from offset 1, [2, 2] with strides [1, 2]: rows (1, 3), (2, 4)
        assert_eq!(p.f32("a.weight").unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
        std::fs::remove_file(&dir).ok();
    }
}

