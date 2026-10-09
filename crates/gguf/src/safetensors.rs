//! safetensors: a JSON header (each tensor's dtype, shape and byte range) and the raw bytes after it - the format of
//! image and video models' text encoders, VAEs and LoRAs. Read by `pread`, like a GGUF.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// One tensor: where its bytes are
#[derive(Clone, Debug)]
pub struct StTensor {
    pub name: String,
    /// as the file names it: "F32", "F16", "BF16", "I8", "U8", "F8_E4M3", ...
    pub dtype: String,
    /// row-major, outermost first
    pub shape: Vec<u64>,
    /// absolute byte offset in the file
    pub offset: u64,
    pub bytes: u64,
}

impl StTensor {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    pub path: PathBuf,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<StTensor>,
    index: BTreeMap<String, usize>,
    file: File,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors> {
        let file = File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let mut n = [0u8; 8];
        file.read_exact_at(&mut n, 0).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let n = u64::from_le_bytes(n);
        if n > 512 << 20 {
            return Err(Error(format!("{}: not a safetensors file (a {n}-byte header)", path.display())));
        }
        let mut h = vec![0u8; n as usize];
        file.read_exact_at(&mut h, 8).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&h).map_err(|e| Error(format!("{}: its header: {e}", path.display())))?;
        let base = 8 + n;
        let mut tensors = Vec::new();
        let mut metadata = BTreeMap::new();
        for (name, e) in v.as_object().ok_or_else(|| Error(format!("{}: its header is not an object", path.display())))? {
            if name == "__metadata__" {
                for (k, x) in e.as_object().into_iter().flatten() {
                    metadata.insert(k.clone(), x.as_str().map_or_else(|| x.to_string(), str::to_string));
                }
                continue;
            }
            let off = e["data_offsets"].as_array().filter(|a| a.len() == 2).ok_or_else(|| Error(format!("{name}: no data_offsets")))?;
            let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
            tensors.push(StTensor {
                name: name.clone(),
                dtype: e["dtype"].as_str().unwrap_or("").to_string(),
                shape: e["shape"].as_array().map_or_else(Vec::new, |s| s.iter().filter_map(|d| d.as_u64()).collect()),
                offset: base + a,
                bytes: b.saturating_sub(a),
            });
        }
        tensors.sort_by_key(|t| t.offset);
        let index = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(SafeTensors { path: path.to_path_buf(), metadata, tensors, index, file })
    }

    pub fn tensor(&self, name: &str) -> Option<&StTensor> {
        self.index.get(name).map(|i| &self.tensors[*i])
    }

    /// The tensor `name`, or an error naming the file
    pub fn need(&self, name: &str) -> Result<&StTensor> {
        self.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.path.display())))
    }

    pub fn read(&self, t: &StTensor) -> Result<Vec<u8>> {
        let mut v = vec![0u8; t.bytes as usize];
        self.read_into(t, 0, &mut v)?;
        Ok(v)
    }

    /// Bytes `[at, at + dst.len())` of a tensor
    pub fn read_into(&self, t: &StTensor, at: u64, dst: &mut [u8]) -> Result<()> {
        if at + dst.len() as u64 > t.bytes {
            return Err(Error(format!("{}: reading past its {} bytes", t.name, t.bytes)));
        }
        self.file.read_exact_at(dst, t.offset + at).map_err(|e| Error(format!("reading {}: {e}", t.name)))
    }

    /// A tensor as float32 (F32, F16, BF16), on the host
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.need(name)?;
        let b = self.read(t)?;
        Ok(match t.dtype.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F16" => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            d => return Err(Error(format!("{name}: {d}, not a float type"))),
        })
    }
}

/// IEEE half to float32
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
