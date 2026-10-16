//! PyTorch checkpoints (`torch.save` of a state dict: a ZIP of `<name>/data.pkl` and each storage, stored
//! uncompressed, as `<name>/data/<key>`) read as a table of contents: the archive's directory and the pickle are read,
//! the tensors are not - each becomes a `SafeTensors` entry (its dtype, shape and byte range in the file), read by
//! `pread` when it is used, like a safetensors file's. Contiguous float tensors (half, bfloat16, float32) only;
//! anything else in the dict is skipped. The pickle machine covers what a state dict holds (dicts, tuples, the
//! storages' persistent ids, `_rebuild_tensor_v2`).

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::safetensors::{SafeTensors, StTensor};
use crate::{Error, Result};

fn u16le(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}

fn u32le(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

fn u64le(b: &[u8], i: usize) -> usize {
    u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes")) as usize
}

fn read_at(f: &File, at: usize, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    f.read_exact_at(&mut v, at as u64).map_err(|e| Error(format!("reading {n} bytes at {at}: {e}")))?;
    Ok(v)
}

/// The archive's stored entries: name -> (data offset, size), from its central directory
fn entries(f: &File, len: usize) -> Result<HashMap<String, (usize, usize)>> {
    let bad = |m: &str| Error(format!("not a PyTorch checkpoint (ZIP): {m}"));
    let tail_len = len.min(1 << 16);
    let tail = read_at(f, len - tail_len, tail_len)?;
    let at = (0..tail.len().saturating_sub(21)).rev().find(|&i| u32le(&tail, i) == 0x0605_4b50).ok_or_else(|| bad("no end record"))?;
    let (mut n, mut cd, mut cd_len) = (u16le(&tail, at + 10), u32le(&tail, at + 16), u32le(&tail, at + 12));
    // ZIP64: the locator just before the end record
    if at >= 20 && u32le(&tail, at - 20) == 0x0706_4b50 {
        let z = u64le(&tail, at - 12);
        let rec = read_at(f, z, 56)?;
        if u32le(&rec, 0) == 0x0606_4b50 {
            n = u64le(&rec, 32);
            cd_len = u64le(&rec, 40);
            cd = u64le(&rec, 48);
        }
    }
    let b = read_at(f, cd, cd_len)?;
    let mut out = HashMap::new();
    let mut i = 0;
    for _ in 0..n {
        if i + 46 > b.len() || u32le(&b, i) != 0x0201_4b50 {
            return Err(bad("a broken central directory"));
        }
        let method = u16le(&b, i + 10);
        let (mut size, nl, el, cl) = (u32le(&b, i + 20), u16le(&b, i + 28), u16le(&b, i + 30), u16le(&b, i + 32));
        let mut off = u32le(&b, i + 42);
        let name = String::from_utf8_lossy(&b[i + 46..i + 46 + nl]).into_owned();
        // ZIP64 extra field: the sizes and offset that did not fit
        let mut e = i + 46 + nl;
        while e + 4 <= i + 46 + nl + el {
            let (id, l) = (u16le(&b, e), u16le(&b, e + 2));
            if id == 1 {
                let mut p = e + 4;
                if u32le(&b, i + 24) == 0xFFFF_FFFF {
                    p += 8;
                }
                if size == 0xFFFF_FFFF {
                    size = u64le(&b, p);
                    p += 8;
                }
                if off == 0xFFFF_FFFF {
                    off = u64le(&b, p);
                }
            }
            e += 4 + l;
        }
        if method != 0 {
            return Err(bad(&format!("{name} is compressed")));
        }
        out.insert(name, (off, size));
        i += 46 + nl + el + cl;
    }
    // the data begins after each entry's local header (its name and extra field have their own lengths there)
    let mut fixed = HashMap::new();
    for (name, (off, size)) in out {
        let h = read_at(f, off, 30)?;
        if u32le(&h, 0) != 0x0403_4b50 {
            return Err(bad(&format!("{name}: no local header")));
        }
        fixed.insert(name, (off + 30 + u16le(&h, 26) + u16le(&h, 28), size));
    }
    Ok(fixed)
}

/// A pickled value (only what a state dict holds is kept; the rest are read and dropped)
#[derive(Clone, Debug)]
enum V {
    None,
    Bool,
    Int(i64),
    Float,
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String),
    Persistent(Box<V>),
    /// (storage type, storage key, offset, shape, stride)
    Tensor(String, String, usize, Vec<usize>, Vec<usize>),
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
    let need = |i: usize, n: usize| -> Result<()> { if i + n <= p.len() { Ok(()) } else { Err(Error("the checkpoint's pickle ends early".into())) } };
    while i < p.len() {
        let op = p[i];
        i += 1;
        match op {
            0x80 => i += 1,
            0x95 => i += 8,
            b'}' => st.push(V::Dict(Vec::new())),
            b']' => st.push(V::List(Vec::new())),
            b')' => st.push(V::Tuple(Vec::new())),
            b'(' => st.push(V::Mark),
            b'N' => st.push(V::None),
            0x88 | 0x89 => st.push(V::Bool),
            b'K' => {
                need(i, 1)?;
                st.push(V::Int(p[i] as i64));
                i += 1;
            }
            b'M' => {
                need(i, 2)?;
                st.push(V::Int(u16le(p, i) as i64));
                i += 2;
            }
            b'J' => {
                need(i, 4)?;
                st.push(V::Int(i32::from_le_bytes(p[i..i + 4].try_into().expect("4")) as i64));
                i += 4;
            }
            0x8a => {
                need(i, 1)?;
                let n = p[i] as usize;
                need(i + 1, n)?;
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
                need(i, 8)?;
                st.push(V::Float);
                i += 8;
            }
            b'X' => {
                need(i, 4)?;
                let n = u32le(p, i);
                need(i + 4, n)?;
                st.push(V::Str(String::from_utf8_lossy(&p[i + 4..i + 4 + n]).into_owned()));
                i += 4 + n;
            }
            0x8c => {
                need(i, 1)?;
                let n = p[i] as usize;
                need(i + 1, n)?;
                st.push(V::Str(String::from_utf8_lossy(&p[i + 1..i + 1 + n]).into_owned()));
                i += 1 + n;
            }
            b'c' => {
                let e1 = i + p[i..].iter().position(|c| *c == b'\n').ok_or_else(|| bad("GLOBAL".into()))?;
                let e2 = e1 + 1 + p[e1 + 1..].iter().position(|c| *c == b'\n').ok_or_else(|| bad("GLOBAL".into()))?;
                st.push(V::Global(String::from_utf8_lossy(&p[e1 + 1..e2]).into_owned()));
                i = e2 + 1;
            }
            0x93 => {
                let n = st.pop();
                st.pop();
                st.push(V::Global(if let Some(V::Str(n)) = n { n } else { String::new() }));
            }
            b'q' => {
                need(i, 1)?;
                memo.insert(p[i] as usize, st.last().cloned().unwrap_or(V::None));
                i += 1;
            }
            b'r' => {
                need(i, 4)?;
                memo.insert(u32le(p, i), st.last().cloned().unwrap_or(V::None));
                i += 4;
            }
            0x94 => {
                let n = memo.len();
                memo.insert(n, st.last().cloned().unwrap_or(V::None));
            }
            b'h' => {
                need(i, 1)?;
                st.push(memo.get(&(p[i] as usize)).cloned().unwrap_or(V::None));
                i += 1;
            }
            b'j' => {
                need(i, 4)?;
                st.push(memo.get(&u32le(p, i)).cloned().unwrap_or(V::None));
                i += 4;
            }
            b't' => {
                let items = pop_mark(&mut st);
                st.push(V::Tuple(items));
            }
            0x85..=0x87 => {
                let n = ((op - 0x84) as usize).min(st.len());
                let items = st.split_off(st.len() - n);
                st.push(V::Tuple(items));
            }
            b'Q' => {
                let pid = st.pop().unwrap_or(V::None);
                st.push(V::Persistent(Box::new(pid)));
            }
            b'R' | 0x81 => {
                let args = st.pop().unwrap_or(V::None);
                let f = st.pop().unwrap_or(V::None);
                let a = if let V::Tuple(a) = args { a } else { Vec::new() };
                st.push(match &f {
                    V::Global(n) if n == "_rebuild_tensor_v2" || n == "_rebuild_tensor" => {
                        // the storage: ('storage', <type>, key, location, numel)
                        let (ty, key) = match a.first() {
                            Some(V::Persistent(pid)) => match pid.as_ref() {
                                V::Tuple(t) => (
                                    match t.get(1) {
                                        Some(V::Global(g)) => g.clone(),
                                        _ => String::new(),
                                    },
                                    match t.get(2) {
                                        Some(V::Str(k)) => k.clone(),
                                        _ => String::new(),
                                    },
                                ),
                                _ => (String::new(), String::new()),
                            },
                            _ => (String::new(), String::new()),
                        };
                        let off = if let Some(V::Int(o)) = a.get(1) { *o as usize } else { 0 };
                        V::Tensor(ty, key, off, a.get(2).map(ints).unwrap_or_default(), a.get(3).map(ints).unwrap_or_default())
                    }
                    V::Global(n) if n == "OrderedDict" => V::Dict(Vec::new()),
                    _ => V::None,
                });
            }
            b'b' => {
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

/// A storage type's dtype (safetensors' name) and element size
fn dtype(storage: &str) -> Option<(&'static str, usize)> {
    match storage {
        "HalfStorage" => Some(("F16", 2)),
        "BFloat16Storage" => Some(("BF16", 2)),
        "FloatStorage" => Some(("F32", 4)),
        _ => None,
    }
}

/// The float tensors of a checkpoint (a state dict, or one under "state_dict"), as a table read by offset
pub fn open(path: &Path) -> Result<SafeTensors> {
    let f = File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let len = f.metadata().map_err(|e| Error(format!("{}: {e}", path.display())))?.len() as usize;
    let ents = entries(&f, len).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
    let (pkl, &(a, n)) = ents.iter().find(|(k, _)| k.ends_with("/data.pkl") || *k == "data.pkl")
        .ok_or_else(|| Error(format!("{}: no data.pkl", path.display())))?;
    let root = pkl.strip_suffix("data.pkl").unwrap_or("").to_string();
    let mut v = unpickle(&read_at(&f, a, n)?).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
    if let V::Dict(d) = &v {
        if let Some((_, inner)) = d.iter().find(|(k, _)| matches!(k, V::Str(s) if s == "state_dict")) {
            v = inner.clone();
        }
    }
    let V::Dict(d) = v else { return Err(Error(format!("{}: not a state dict", path.display()))) };
    let mut tensors = Vec::new();
    for (k, t) in d {
        let (V::Str(name), V::Tensor(ty, key, off, shape, stride)) = (k, t) else { continue };
        let Some((dt, es)) = dtype(&ty) else { continue };
        let numel: usize = shape.iter().product();
        let mut want = 1;
        for (s, d) in stride.iter().zip(&shape).rev() {
            if *d > 1 && *s != want {
                return Err(Error(format!("{}: {name} is not contiguous", path.display())));
            }
            want *= d;
        }
        let &(at, size) = ents.get(&format!("{root}data/{key}")).ok_or_else(|| Error(format!("{}: {name}'s storage {key} is missing", path.display())))?;
        if (off + numel) * es > size {
            return Err(Error(format!("{}: {name} runs past its storage", path.display())));
        }
        tensors.push(StTensor { name, dtype: dt.to_string(), shape: shape.iter().map(|&x| x as u64).collect(), offset: (at + off * es) as u64, bytes: (numel * es) as u64 });
    }
    SafeTensors::from_tensors(path, tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal torch.save layout: an uncompressed ZIP of data.pkl (protocol 2) and one half storage
    fn checkpoint() -> Vec<u8> {
        // {'w': _rebuild_tensor_v2(('storage', torch.HalfStorage, '0', 'cpu', 4), 1, (3,), (1,), False, None)}
        let mut pk = vec![0x80, 2, b'}', b'q', 0, b'(', b'X'];
        pk.extend(1u32.to_le_bytes());
        pk.push(b'w');
        pk.extend(b"ctorch._utils\n_rebuild_tensor_v2\n");
        pk.extend(b"((X");
        pk.extend(7u32.to_le_bytes());
        pk.extend(b"storage");
        pk.extend(b"ctorch\nHalfStorage\nX");
        pk.extend(1u32.to_le_bytes());
        pk.push(b'0');
        pk.push(b'X');
        pk.extend(3u32.to_le_bytes());
        pk.extend(b"cpuK\x04tQK\x01K\x03\x85K\x01\x85\x89Ntq\x01Rq\x02u.");
        let storage: Vec<u8> = [0x3c00u16, 0x4000, 0x4200, 0x4400].iter().flat_map(|h| h.to_le_bytes()).collect();
        let files: Vec<(&str, Vec<u8>)> = vec![("ck/data.pkl", pk), ("ck/data/0", storage)];
        let mut out = Vec::new();
        let mut cd = Vec::new();
        for (name, data) in &files {
            let off = out.len() as u32;
            out.extend(0x0403_4b50u32.to_le_bytes());
            out.extend([0u8; 14]);
            out.extend((data.len() as u32).to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes());
            out.extend((name.len() as u16).to_le_bytes());
            out.extend(0u16.to_le_bytes());
            out.extend(name.as_bytes());
            out.extend(data);
            cd.extend(0x0201_4b50u32.to_le_bytes());
            cd.extend([0u8; 16]);
            cd.extend((data.len() as u32).to_le_bytes());
            cd.extend((data.len() as u32).to_le_bytes());
            cd.extend((name.len() as u16).to_le_bytes());
            cd.extend([0u8; 12]);
            cd.extend(off.to_le_bytes());
            cd.extend(name.as_bytes());
        }
        let cd_at = out.len() as u32;
        out.extend(&cd);
        out.extend(0x0605_4b50u32.to_le_bytes());
        out.extend([0u8; 4]);
        out.extend((files.len() as u16).to_le_bytes());
        out.extend((files.len() as u16).to_le_bytes());
        out.extend((cd.len() as u32).to_le_bytes());
        out.extend(cd_at.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out
    }

    #[test]
    fn a_half_tensor_is_found_at_its_offset() {
        let dir = std::env::temp_dir().join(format!("ns-pthst-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.bin");
        std::fs::write(&p, checkpoint()).unwrap();
        let st = open(&p).unwrap();
        let t = st.need("w").unwrap();
        assert_eq!((t.dtype.as_str(), t.shape.clone(), t.bytes), ("F16", vec![3], 6));
        assert_eq!(st.f32("w").unwrap(), vec![2.0, 3.0, 4.0]);
        std::fs::remove_file(&p).unwrap();
    }
}
