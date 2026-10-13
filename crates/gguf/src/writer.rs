//! A GGUF v3 file written as a stream: the metadata and the tensor directory first (every tensor's type and shape
//! known up front), then each tensor's bytes in the directory's order, each aligned. What a converter needs to
//! write a file larger than memory (`Writer::create`, then `data` for every tensor in turn, then `finish`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::{Error, GType, Result, Value};

const ALIGN: u64 = 32;

pub struct Writer {
    f: BufWriter<File>,
    /// each tensor's bytes, in order
    sizes: Vec<u64>,
    /// the tensor being written, and its bytes so far
    at: usize,
    done: u64,
}

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T> {
    r.map_err(|e| Error(format!("{what}: {e}")))
}

fn put_str(w: &mut Vec<u8>, s: &str) {
    w.extend((s.len() as u64).to_le_bytes());
    w.extend(s.as_bytes());
}

/// A value's GGUF type and its bytes (arrays: the element type of the first element)
fn put_value(w: &mut Vec<u8>, v: &Value, with_type: bool) -> Result<()> {
    let ty: u32 = match v {
        Value::U(_) => 10,
        Value::I(_) => 11,
        Value::F(_) => 12,
        Value::Bool(_) => 7,
        Value::Str(_) => 8,
        Value::Array(_) => 9,
    };
    if with_type {
        w.extend(ty.to_le_bytes());
    }
    match v {
        Value::U(x) => w.extend(x.to_le_bytes()),
        Value::I(x) => w.extend(x.to_le_bytes()),
        Value::F(x) => w.extend(x.to_le_bytes()),
        Value::Bool(x) => w.push(*x as u8),
        Value::Str(s) => put_str(w, s),
        Value::Array(a) => {
            let et = match a.first() {
                None | Some(Value::U(_)) => 10u32,
                Some(Value::I(_)) => 11,
                Some(Value::F(_)) => 12,
                Some(Value::Bool(_)) => 7,
                Some(Value::Str(_)) => 8,
                Some(Value::Array(_)) => return Err(Error("arrays of arrays are not written".into())),
            };
            w.extend(et.to_le_bytes());
            w.extend((a.len() as u64).to_le_bytes());
            for x in a {
                put_value(w, x, false)?;
            }
        }
    }
    Ok(())
}

impl Writer {
    /// The file's head: `metadata`, then the directory of `tensors` - (name, type, shape outermost first)
    pub fn create(path: &Path, metadata: &[(&str, Value)], tensors: &[(String, GType, Vec<u64>)]) -> Result<Writer> {
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes());
        h.extend((tensors.len() as u64).to_le_bytes());
        h.extend((metadata.len() as u64).to_le_bytes());
        for (k, v) in metadata {
            put_str(&mut h, k);
            put_value(&mut h, v, true)?;
        }
        let mut off = 0u64;
        let mut sizes = Vec::with_capacity(tensors.len());
        for (name, ty, shape) in tensors {
            let n: u64 = shape.iter().product();
            let bytes = ty.bytes(n).ok_or_else(|| Error(format!("{name}: {n} values are not whole {} blocks", ty.name())))?;
            put_str(&mut h, name);
            h.extend((shape.len() as u32).to_le_bytes());
            for d in shape.iter().rev() {
                h.extend(d.to_le_bytes());
            }
            h.extend(ty.code().to_le_bytes());
            h.extend(off.to_le_bytes());
            sizes.push(bytes);
            off = (off + bytes).div_ceil(ALIGN) * ALIGN;
        }
        h.resize((h.len() as u64).div_ceil(ALIGN) as usize * ALIGN as usize, 0);
        let mut f = BufWriter::with_capacity(8 << 20, io(File::create(path), &format!("creating {}", path.display()))?);
        io(f.write_all(&h), "writing the head")?;
        Ok(Writer { f, sizes, at: 0, done: 0 })
    }

    /// The next bytes of the tensors in order (a tensor may come in several calls; the padding after each is added)
    pub fn data(&mut self, mut b: &[u8]) -> Result<()> {
        while !b.is_empty() {
            let size = *self.sizes.get(self.at).ok_or("more bytes than the tensors hold")?;
            let k = (size - self.done).min(b.len() as u64) as usize;
            io(self.f.write_all(&b[..k]), "writing a tensor")?;
            self.done += k as u64;
            b = &b[k..];
            if self.done == size {
                let pad = (size.div_ceil(ALIGN) * ALIGN - size) as usize;
                io(self.f.write_all(&vec![0u8; pad]), "padding")?;
                self.at += 1;
                self.done = 0;
            }
        }
        Ok(())
    }

    /// Every tensor written: the file closed
    pub fn finish(mut self) -> Result<()> {
        if self.at != self.sizes.len() {
            return Err(Error(format!("{} of {} tensors written", self.at, self.sizes.len())));
        }
        io(self.f.flush(), "flushing")
    }
}
