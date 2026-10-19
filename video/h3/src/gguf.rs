//! Reading a `.gguf` file (llama.cpp's format, version 3): a header of key/value pairs and tensor records, then the
//! tensors' bytes, aligned. Only the table of contents is read here; the tensors are read where they are used.

use std::collections::BTreeMap;
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

impl Gguf {
    /// The table of contents, read by the foundation's reader (nextsycl_gguf): the tensors of the types H3's files use
    pub fn open(path: &Path) -> Result<Gguf> {
        let g = nextsycl_gguf::Gguf::open(path).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
        if g.version != 3 {
            return Err(Error(format!("{}: GGUF version {}, only 3 is read", path.display(), g.version)));
        }
        let mut entries = BTreeMap::new();
        for t in g.tensors {
            if t.shard != 0 {
                return Err(Error(format!("{}: a split file (H3's are one file each)", path.display())));
            }
            let ty = GType::from_code(t.ty.code()).ctx(&t.name)?;
            let shape: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
            let bytes = ty.bytes(shape.iter().product());
            if bytes as u64 != t.bytes {
                return Err(Error(format!("{}: {} holds {} bytes, its shape {bytes}", path.display(), t.name, t.bytes)));
            }
            entries.insert(t.name.clone(), GEntry { name: t.name, ty, shape, offset: t.offset, bytes });
        }
        Ok(Gguf { path: path.to_path_buf(), entries })
    }
    pub fn get(&self, name: &str) -> Result<&GEntry> {
        self.entries.get(name).ok_or_else(|| Error(format!("{}: no tensor named {name}", self.path.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextsycl_gguf::writer::Writer;
    use nextsycl_gguf::GType as G;

    /// (Checked when it moved onto the foundation's reader: every H3 GGUF - the text encoder, the Q6_K and Q4_K_M
    /// denoisers - read to the same table of contents as H3's own reader did, 2026-10-19.)
    #[test]
    fn the_toc_in_pytorch_order_with_absolute_offsets() {
        let p = std::env::temp_dir().join(format!("h3-gguf-{}.gguf", std::process::id()));
        let ts = vec![("a.weight".to_string(), G::F16, vec![3u64, 5]), ("b".to_string(), G::BF16, vec![7]),
                      ("c.weight".to_string(), G::Q4K, vec![2, 256]), ("d".to_string(), G::Q6K, vec![256]), ("e".to_string(), G::F32, vec![2, 2, 3])];
        let mut w = Writer::create(&p, &[("general.architecture", nextsycl_gguf::Value::Str("h3".into()))], &ts).unwrap();
        for (i, (_, ty, shape)) in ts.iter().enumerate() {
            w.data(&vec![i as u8 + 1; ty.bytes(shape.iter().product()).unwrap() as usize]).unwrap();
        }
        w.finish().unwrap();
        let g = Gguf::open(&p).unwrap();
        let c = g.get("c.weight").unwrap();
        assert_eq!((c.ty, c.shape.clone(), c.bytes), (GType::Q4K, vec![2, 256], 2 * 144));
        assert_eq!((g.get("a.weight").unwrap().shape.clone(), g.get("e").unwrap().shape.clone()), (vec![3, 5], vec![2, 2, 3]));
        // the offsets are the file's: the bytes there are the tensor's
        let bytes = std::fs::read(&p).unwrap();
        for (i, n) in ["a.weight", "b", "c.weight", "d", "e"].iter().enumerate() {
            let t = g.get(n).unwrap();
            assert!(bytes[t.offset as usize..t.offset as usize + t.bytes].iter().all(|b| *b == i as u8 + 1), "{n}");
        }
        assert!(g.get("f").is_err());
        let _ = std::fs::remove_file(p);
    }
}
