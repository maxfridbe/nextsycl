//! A PyTorch checkpoint (`torch.save` of a state dict: a ZIP of `<name>/data.pkl` and the tensors' storages as
//! `<name>/data/<key>`) read without Python: the archive's stored entries, and the subset of the pickle machine a
//! state dict needs (dicts, tuples, the storages' persistent ids, `_rebuild_tensor_v2`). Float32 tensors only (the
//! AudioVAE's are), contiguous.

use std::collections::HashMap;
use std::path::Path;

use nextsycl_audio::{Error, Result};

/// A tensor: its shape and values
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn u16le(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}

fn u32le(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

fn u64le(b: &[u8], i: usize) -> usize {
    u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes")) as usize
}

/// The archive's stored (uncompressed) entries: name -> byte range
fn entries(b: &[u8]) -> Result<HashMap<String, (usize, usize)>> {
    let bad = |m: &str| Error(format!("not a PyTorch checkpoint (ZIP): {m}"));
    let eocd = (0..b.len().saturating_sub(21)).rev().take(1 << 16).find(|&i| u32le(b, i) == 0x0605_4b50).ok_or_else(|| bad("no end record"))?;
    let (mut n, mut cd) = (u16le(b, eocd + 10), u32le(b, eocd + 16));
    // ZIP64: the end record's locator just before it
    if eocd >= 20 && u32le(b, eocd - 20) == 0x0706_4b50 {
        let z = u64le(b, eocd - 12);
        if u32le(b, z) == 0x0606_4b50 {
            n = u64le(b, z + 32);
            cd = u64le(b, z + 48);
        }
    }
    let mut out = HashMap::new();
    let mut i = cd;
    for _ in 0..n {
        if i + 46 > b.len() || u32le(b, i) != 0x0201_4b50 {
            return Err(bad("a broken central directory"));
        }
        let method = u16le(b, i + 10);
        let (mut size, nl, el, cl) = (u32le(b, i + 20), u16le(b, i + 28), u16le(b, i + 30), u16le(b, i + 32));
        let mut off = u32le(b, i + 42);
        let name = String::from_utf8_lossy(&b[i + 46..i + 46 + nl]).into_owned();
        // ZIP64 extra field: the sizes and offset that did not fit
        let mut e = i + 46 + nl;
        while e + 4 <= i + 46 + nl + el {
            let (id, len) = (u16le(b, e), u16le(b, e + 2));
            if id == 1 {
                let mut p = e + 4;
                if u32le(b, i + 24) == 0xFFFF_FFFF {
                    p += 8;
                }
                if size == 0xFFFF_FFFF {
                    size = u64le(b, p);
                    p += 8;
                }
                if off == 0xFFFF_FFFF {
                    off = u64le(b, p);
                }
            }
            e += 4 + len;
        }
        if method != 0 {
            return Err(bad(&format!("{name} is compressed")));
        }
        let data = off + 30 + u16le(b, off + 26) + u16le(b, off + 28);
        out.insert(name, (data, data + size));
        i += 46 + nl + el + cl;
    }
    Ok(out)
}

/// A pickled value (only what a state dict holds is kept; the rest are read and dropped)
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String, String),
    Persistent(Box<V>),
    /// (storage key, offset, shape, stride)
    Tensor(String, usize, Vec<usize>, Vec<usize>),
    Mark,
}

fn ints(v: &V) -> Vec<usize> {
    match v {
        V::Tuple(x) | V::List(x) => x.iter().map(|e| if let V::Int(i) = e { *i as usize } else { 0 }).collect(),
        _ => Vec::new(),
    }
}

/// The pickle machine, enough for a state dict
fn unpickle(p: &[u8]) -> Result<V> {
    let bad = |m: String| Error(format!("the checkpoint's pickle: {m}"));
    let mut st: Vec<V> = Vec::new();
    let mut memo: HashMap<usize, V> = HashMap::new();
    let mut i = 0;
    let pop_mark = |st: &mut Vec<V>| -> Vec<V> {
        let m = st.iter().rposition(|v| matches!(v, V::Mark)).unwrap_or(0);
        let items = st.split_off(m + 1);
        st.pop();
        items
    };
    while i < p.len() {
        let op = p[i];
        i += 1;
        match op {
            0x80 => i += 1,                                           // PROTO
            0x95 => i += 8,                                           // FRAME
            b'}' => st.push(V::Dict(Vec::new())),                    // EMPTY_DICT
            b']' => st.push(V::List(Vec::new())),                    // EMPTY_LIST
            b')' => st.push(V::Tuple(Vec::new())),                   // EMPTY_TUPLE
            b'(' => st.push(V::Mark),                                 // MARK
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'K' => {
                st.push(V::Int(p[i] as i64));
                i += 1;
            }
            b'M' => {
                st.push(V::Int(u16le(p, i) as i64));
                i += 2;
            }
            b'J' => {
                st.push(V::Int(i32::from_le_bytes(p[i..i + 4].try_into().expect("4")) as i64));
                i += 4;
            }
            0x8a => {
                // LONG1
                let n = p[i] as usize;
                let mut v: i64 = 0;
                for k in 0..n.min(8) {
                    v |= (p[i + 1 + k] as i64) << (8 * k);
                }
                if n > 0 && n < 8 && p[i + n] & 0x80 != 0 {
                    v -= 1 << (8 * n);
                }
                st.push(V::Int(v));
                i += 1 + n;
            }
            b'G' => {
                st.push(V::Float(f64::from_be_bytes(p[i..i + 8].try_into().expect("8"))));
                i += 8;
            }
            b'X' => {
                let n = u32le(p, i);
                st.push(V::Str(String::from_utf8_lossy(&p[i + 4..i + 4 + n]).into_owned()));
                i += 4 + n;
            }
            0x8c => {
                let n = p[i] as usize;
                st.push(V::Str(String::from_utf8_lossy(&p[i + 1..i + 1 + n]).into_owned()));
                i += 1 + n;
            }
            b'c' => {
                // GLOBAL module\nname\n
                let e1 = i + p[i..].iter().position(|c| *c == b'\n').ok_or_else(|| bad("GLOBAL".into()))?;
                let e2 = e1 + 1 + p[e1 + 1..].iter().position(|c| *c == b'\n').ok_or_else(|| bad("GLOBAL".into()))?;
                st.push(V::Global(String::from_utf8_lossy(&p[i..e1]).into_owned(), String::from_utf8_lossy(&p[e1 + 1..e2]).into_owned()));
                i = e2 + 1;
            }
            0x93 => {
                // STACK_GLOBAL
                let n = st.pop();
                let m = st.pop();
                if let (Some(V::Str(m)), Some(V::Str(n))) = (m, n) {
                    st.push(V::Global(m, n));
                }
            }
            b'q' => {
                memo.insert(p[i] as usize, st.last().cloned().unwrap_or(V::None));
                i += 1;
            }
            b'r' => {
                memo.insert(u32le(p, i), st.last().cloned().unwrap_or(V::None));
                i += 4;
            }
            0x94 => {
                let n = memo.len();
                memo.insert(n, st.last().cloned().unwrap_or(V::None));
            }
            b'h' => {
                st.push(memo.get(&(p[i] as usize)).cloned().unwrap_or(V::None));
                i += 1;
            }
            b'j' => {
                st.push(memo.get(&u32le(p, i)).cloned().unwrap_or(V::None));
                i += 4;
            }
            b't' => {
                let items = pop_mark(&mut st);
                st.push(V::Tuple(items));
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                let items = st.split_off(st.len() - n);
                st.push(V::Tuple(items));
            }
            b'Q' => {
                let pid = st.pop().unwrap_or(V::None);
                st.push(V::Persistent(Box::new(pid)));
            }
            b'R' | 0x81 => {
                // REDUCE / NEWOBJ: callable(args)
                let args = st.pop().unwrap_or(V::None);
                let f = st.pop().unwrap_or(V::None);
                let a = if let V::Tuple(a) = args { a } else { Vec::new() };
                st.push(match &f {
                    V::Global(_, n) if n == "_rebuild_tensor_v2" || n == "_rebuild_tensor" => {
                        let key = match a.first() {
                            Some(V::Persistent(pid)) => match pid.as_ref() {
                                V::Tuple(t) => match t.get(2) {
                                    Some(V::Str(k)) => k.clone(),
                                    _ => String::new(),
                                },
                                _ => String::new(),
                            },
                            _ => String::new(),
                        };
                        let off = if let Some(V::Int(o)) = a.get(1) { *o as usize } else { 0 };
                        V::Tensor(key, off, a.get(2).map(ints).unwrap_or_default(), a.get(3).map(ints).unwrap_or_default())
                    }
                    V::Global(_, n) if n == "OrderedDict" => V::Dict(Vec::new()),
                    _ => V::None,
                });
            }
            b'b' => {
                // BUILD: the state is dropped (a dict's metadata)
                st.pop();
            }
            b's' => {
                let v = st.pop().unwrap_or(V::None);
                let k = st.pop().unwrap_or(V::None);
                if let Some(V::Dict(d)) = st.last_mut() {
                    d.push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut st);
                if let Some(V::Dict(d)) = st.last_mut() {
                    for kv in items.chunks(2) {
                        if let [k, v] = kv {
                            d.push((k.clone(), v.clone()));
                        }
                    }
                }
            }
            b'a' => {
                let v = st.pop().unwrap_or(V::None);
                if let Some(V::List(l)) = st.last_mut() {
                    l.push(v);
                }
            }
            b'e' => {
                let items = pop_mark(&mut st);
                if let Some(V::List(l)) = st.last_mut() {
                    l.extend(items);
                }
            }
            b'.' => return st.pop().ok_or_else(|| bad("an empty stack".into())),
            o => return Err(bad(format!("opcode 0x{o:02x} at {}", i - 1))),
        }
    }
    Err(bad("no STOP".into()))
}

/// Every float32 tensor of a checkpoint (a state dict, or one under "state_dict"), by name
pub fn load(path: &Path) -> Result<HashMap<String, Tensor>> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let ents = entries(&b)?;
    let (pkl, &(a, z)) = ents.iter().find(|(n, _)| n.ends_with("/data.pkl") || *n == "data.pkl")
        .ok_or_else(|| Error(format!("{}: no data.pkl", path.display())))?;
    let root = pkl.strip_suffix("data.pkl").unwrap_or("").to_string();
    let mut v = unpickle(&b[a..z])?;
    if let V::Dict(d) = &v {
        if let Some((_, inner)) = d.iter().find(|(k, _)| matches!(k, V::Str(s) if s == "state_dict")) {
            v = inner.clone();
        }
    }
    let V::Dict(d) = v else { return Err(Error(format!("{}: not a state dict", path.display()))) };
    let mut out = HashMap::new();
    for (k, t) in d {
        let (V::Str(name), V::Tensor(key, off, shape, stride)) = (k, t) else { continue };
        let n: usize = shape.iter().product();
        // contiguous (row-major strides)
        let mut want = 1;
        for (s, d) in stride.iter().zip(&shape).rev() {
            if *d > 1 && *s != want {
                return Err(Error(format!("{}: {name} is not contiguous", path.display())));
            }
            want *= d;
        }
        let &(a, z) = ents.get(&format!("{root}data/{key}")).ok_or_else(|| Error(format!("{}: {name}'s storage {key} is missing", path.display())))?;
        let raw = &b[a..z];
        if (off + n) * 4 > raw.len() {
            // a storage of another type than float32
            continue;
        }
        let data = raw[off * 4..(off + n) * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.insert(name, Tensor { shape, data });
    }
    Ok(out)
}
