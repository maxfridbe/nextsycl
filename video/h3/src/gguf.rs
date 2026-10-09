//! Reading a `.gguf` file (llama.cpp's format, version 3): a header of key/value pairs and tensor records, then the
//! tensors' bytes, aligned. Only the table of contents is read here; the tensors are read where they are used.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::{Ctx, Error, Result};

/// A tensor's element type, by llama.cpp's numbering (the ones the MiniMax H3 files use).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GType {
    F32,
    F16,
    BF16,
    Q4K,
    Q6K,
}

impl GType {
    fn from_code(c: u32) -> Result<GType> {
        Ok(match c {
            0 => GType::F32,
            1 => GType::F16,
            30 => GType::BF16,
            12 => GType::Q4K,
            14 => GType::Q6K,
            other => return Err(Error(format!("GGUF tensor type {other} is not supported"))),
        })
    }

    /// The library's code for `h3s_dequant`.
    pub fn quant_code(self) -> Option<i32> {
        match self {
            GType::Q4K => Some(12),
            GType::Q6K => Some(14),
            _ => None,
        }
    }

    /// Bytes for `n` values.
    pub fn bytes(self, n: usize) -> usize {
        match self {
            GType::F32 => n * 4,
            GType::F16 | GType::BF16 => n * 2,
            GType::Q4K => n / 256 * 144,
            GType::Q6K => n / 256 * 210,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GEntry {
    pub name: String,
    pub ty: GType,
    /// PyTorch order (llama.cpp stores the innermost dimension first; this is reversed)
    pub shape: Vec<usize>,
    pub offset: u64,
    pub bytes: usize,
}

impl GEntry {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct Gguf {
    pub path: PathBuf,
    pub entries: BTreeMap<String, GEntry>,
}

fn rd<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b).ctx("reading the GGUF header")?;
    Ok(b)
}
fn u32_(r: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(rd(r)?))
}
fn u64_(r: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(rd(r)?))
}
fn string(r: &mut impl Read) -> Result<String> {
    let n = u64_(r)? as usize;
    if n > 1 << 24 {
        return Err(Error("GGUF: a string of more than 16 MiB in the header".into()));
    }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b).ctx("reading the GGUF header")?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Skips one value of type `t`; returns it when it is an unsigned integer (for general.alignment).
fn value(r: &mut impl Read, t: u32) -> Result<Option<u64>> {
    Ok(match t {
        0 | 1 | 7 => Some(rd::<1>(r)?[0] as u64),
        2 | 3 => Some(u16::from_le_bytes(rd(r)?) as u64),
        4..=6 => Some(u32_(r)? as u64),
        10..=12 => Some(u64_(r)?),
        8 => {
            string(r)?;
            None
        }
        9 => {
            let et = u32_(r)?;
            let n = u64_(r)?;
            for _ in 0..n {
                value(r, et)?;
            }
            None
        }
        other => return Err(Error(format!("GGUF: unknown value type {other}"))),
    })
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let f = File::open(path).ctx(format!("opening {}", path.display()))?;
        let mut r = BufReader::new(f);
        if &rd::<4>(&mut r)? != b"GGUF" {
            return Err(Error(format!("{}: not a GGUF file", path.display())));
        }
        let version = u32_(&mut r)?;
        if version != 3 {
            return Err(Error(format!("{}: GGUF version {version}, only 3 is read", path.display())));
        }
        let (nt, nkv) = (u64_(&mut r)?, u64_(&mut r)?);
        let mut align = 32u64;
        for _ in 0..nkv {
            let k = string(&mut r)?;
            let t = u32_(&mut r)?;
            let v = value(&mut r, t)?;
            if k == "general.alignment" {
                align = v.unwrap_or(32).max(1);
            }
        }
        let mut raw = Vec::with_capacity(nt as usize);
        for _ in 0..nt {
            let name = string(&mut r)?;
            let nd = u32_(&mut r)? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(u64_(&mut r)? as usize);
            }
            let ty = GType::from_code(u32_(&mut r)?).ctx(&name)?;
            let off = u64_(&mut r)?;
            dims.reverse();
            raw.push((name, ty, dims, off));
        }
        let pos = r.stream_position()?;
        let base = pos.div_ceil(align) * align;
        let len = r.seek(SeekFrom::End(0))?;
        let mut entries = BTreeMap::new();
        for (name, ty, shape, off) in raw {
            let bytes = ty.bytes(shape.iter().product());
            if base + off + bytes as u64 > len {
                return Err(Error(format!("{}: {name} runs past the end of the file", path.display())));
            }
            entries.insert(name.clone(), GEntry { name, ty, shape, offset: base + off, bytes });
        }
        Ok(Gguf { path: path.to_path_buf(), entries })
    }

    pub fn get(&self, name: &str) -> Result<&GEntry> {
        self.entries.get(name).ok_or_else(|| Error(format!("{}: no tensor named {name}", self.path.display())))
    }
}
