//! Reading a `.safetensors` checkpoint: an 8-byte length, a JSON header (name -> type, shape, byte range), then the
//! tensors' bytes back to back. A `.gguf` file opens as a checkpoint too (the denoiser's llama.cpp-quantized forms,
//! Q4_K / Q6_K; Q8_0 / Q5_K; Q3_K, read as Q6_K): its float tensors as such, its k-quant matrices as their raw blocks
//! with `kquant` set.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::dtype::DType;
use crate::gguf::{GType, Gguf};
use crate::{Ctx, Error, Result};

/// Where one tensor lives in the file.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub bytes: usize,
    /// A llama.cpp k-quant (a GGUF file's Q4_K / Q6_K matrix): `bytes` of its blocks, `dtype` U8, `shape` the
    /// matrix's own.
    pub kquant: Option<GType>,
    /// What the file holds when it is not what `kquant` says: a Q3_K matrix, transcoded to Q6_K as it is read (`bytes`
    /// are then the Q6_K blocks'; the file's are `file_bytes()`)
    pub stored: Option<GType>,
}

impl Entry {
    /// The tensor's bytes in the file
    pub fn file_bytes(&self) -> usize {
        match self.stored {
            Some(t) => t.bytes(self.shape.iter().product()),
            None => self.bytes,
        }
    }

    /// The shape of the bytes as they go to the device: a k-quant's blocks are a flat byte buffer.
    pub fn stored_shape(&self) -> Vec<usize> {
        match self.kquant {
            Some(_) => vec![self.bytes],
            None => self.shape.clone(),
        }
    }
}

/// A checkpoint's table of contents. The file is opened per read; nothing is mapped.
pub struct Checkpoint {
    pub path: PathBuf,
    pub entries: BTreeMap<String, Entry>,
    pub metadata: BTreeMap<String, String>,
    pub file_bytes: u64,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Checkpoint> {
        let f = File::open(path).ctx(format!("opening {}", path.display()))?;
        let file_bytes = f.metadata()?.len();
        let mut len8 = [0u8; 8];
        f.read_exact_at(&mut len8, 0).ctx("reading the header length")?;
        if &len8[..4] == b"GGUF" {
            return Checkpoint::open_gguf(path, file_bytes);
        }
        // the header by the foundation's reader (nextsycl_gguf::safetensors), each entry checked against its shape
        let st = nextsycl_gguf::safetensors::SafeTensors::open(path).map_err(|e| Error(e.0))?;
        let mut entries = BTreeMap::new();
        for t in st.tensors {
            let dtype = DType::parse(&t.dtype)?;
            let shape: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
            let bytes = t.bytes as usize;
            if bytes != shape.iter().product::<usize>() * dtype.size() || t.offset + t.bytes > file_bytes {
                return Err(Error(format!("header entry {}: byte range does not match its shape", t.name)));
            }
            entries.insert(t.name.clone(), Entry { name: t.name, dtype, shape, offset: t.offset, bytes, kquant: None, stored: None });
        }
        Ok(Checkpoint { path: path.to_path_buf(), entries, metadata: st.metadata, file_bytes })
    }

    fn open_gguf(path: &Path, file_bytes: u64) -> Result<Checkpoint> {
        let g = Gguf::open(path)?;
        let entries = g
            .entries
            .into_values()
            .map(|e| {
                let (dtype, kquant, stored) = match e.ty {
                    GType::F32 => (DType::F32, None, None),
                    GType::F16 => (DType::F16, None, None),
                    GType::BF16 => (DType::BF16, None, None),
                    GType::Q8_0 | GType::Q4K | GType::Q5K | GType::Q6K => (DType::U8, Some(e.ty), None),
                    // the same values as Q6_K blocks (crate::kquant), which the kernels expand
                    GType::Q3K => (DType::U8, Some(GType::Q6K), Some(GType::Q3K)),
                };
                let bytes = match stored {
                    Some(_) => GType::Q6K.bytes(e.elements()),
                    None => e.bytes,
                };
                // ComfyUI-GGUF files made by stable-diffusion.cpp name the tensors under the ComfyUI model's prefix
                let name = e.name.strip_prefix("model.diffusion_model.").unwrap_or(&e.name).to_string();
                (name.clone(), Entry { name, dtype, shape: e.shape, offset: e.offset, bytes, kquant, stored })
            })
            .collect();
        Ok(Checkpoint { path: path.to_path_buf(), entries, metadata: BTreeMap::new(), file_bytes })
    }

    /// Whether the block matrices are llama.cpp k-quants (a GGUF denoiser) - and which.
    pub fn kquant_of(&self, name: &str) -> Option<GType> {
        self.entries.get(name).and_then(|e| e.kquant)
    }

    pub fn get(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or_else(|| Error(format!("{}: no tensor named {name}", self.path.display())))
    }

    /// One tensor's bytes, read from the file (a Q3_K matrix's as Q6_K blocks).
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.get(name)?;
        let f = File::open(&self.path)?;
        let mut v = vec![0u8; e.file_bytes()];
        f.read_exact_at(&mut v, e.offset).ctx(format!("reading {name}"))?;
        match e.stored {
            Some(GType::Q3K) => crate::kquant::q3k_to_q6k(&v).ctx(name),
            Some(other) => Err(Error(format!("{name}: {other:?} is not transcoded"))),
            None => Ok(v),
        }
    }

    /// The per-layer quantization record ComfyUI stores beside a quantized weight (`<layer>.comfy_quant`: JSON as
    /// bytes), if the layer has one.
    pub fn quant(&self, layer: &str) -> Result<Option<Quant>> {
        let key = format!("{layer}.comfy_quant");
        if !self.entries.contains_key(&key) {
            return Ok(None);
        }
        let v: serde_json::Value = serde_json::from_slice(&self.read(&key)?).ctx(format!("parsing {key}"))?;
        Ok(Some(Quant {
            format: v.get("format").and_then(|f| f.as_str()).unwrap_or_default().to_string(),
            convrot: v.get("convrot").and_then(|c| c.as_bool()).unwrap_or(false),
            group: v.get("convrot_groupsize").and_then(|g| g.as_u64()).unwrap_or(256) as usize,
        }))
    }

    /// Total bytes of tensor data.
    pub fn data_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.bytes as u64).sum()
    }
}

/// How a layer's weight was quantized.
#[derive(Clone, Debug)]
pub struct Quant {
    /// "int8_tensorwise" for the int8 checkpoints.
    pub format: String,
    /// The activations must be rotated (grouped Hadamard) before quantizing: the weights were stored rotated.
    pub convrot: bool,
    pub group: usize,
}

/// Writes float32 tensors (name -> shape, values) and string metadata as a `.safetensors` file.
pub fn write_f32(path: &Path, tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>, metadata: &BTreeMap<String, String>) -> Result<()> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), serde_json::json!(metadata));
    }
    let mut offset = 0usize;
    for (name, (shape, v)) in tensors {
        if shape.iter().product::<usize>() != v.len() {
            return Err(Error(format!("{name}: {} values for shape {shape:?}", v.len())));
        }
        header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + v.len() * 4]}));
        offset += v.len() * 4;
    }
    let mut h = serde_json::to_vec(&header).ctx("writing the header")?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + offset);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for (_, v) in tensors.values() {
        out.extend(v.iter().flat_map(|f| f.to_le_bytes()));
    }
    std::fs::write(path, out).ctx(format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (Checked when it moved onto the foundation's reader: every H3 safetensors file - the int8 denoisers, the VAEs, the
    /// vision tower, the ControlNet, the upscaler, an embedding - read to the same header as H3's own reader did,
    /// 2026-10-19.)
    #[test]
    fn the_header_with_absolute_offsets_and_its_metadata() {
        let p = std::env::temp_dir().join(format!("h3-st-{}.safetensors", std::process::id()));
        let t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = [("x.weight".to_string(), (vec![2, 3], vec![1.0; 6])), ("y".to_string(), (vec![5], vec![2.0; 5]))].into();
        write_f32(&p, &t, &[("seed".to_string(), "7".to_string())].into()).unwrap();
        let c = Checkpoint::open(&p).unwrap();
        let y = c.get("y").unwrap();
        assert_eq!((c.get("x.weight").unwrap().shape.clone(), y.dtype, y.bytes, c.metadata["seed"].as_str()), (vec![2, 3], DType::F32, 20, "7"));
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[y.offset as usize..y.offset as usize + 4], &2f32.to_le_bytes());
        let _ = std::fs::remove_file(p);
    }
}
