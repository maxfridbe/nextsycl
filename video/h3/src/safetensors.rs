//! Reading a `.safetensors` checkpoint: an 8-byte length, a JSON header (name -> type, shape, byte range), then the
//! tensors' bytes back to back. A `.gguf` file opens as a checkpoint too (the denoiser's llama.cpp-quantized forms,
//! Q4_K / Q6_K): its float tensors as such, its k-quant matrices as their raw blocks with `kquant` set.

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
}

impl Entry {
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
        let hlen = u64::from_le_bytes(len8);
        if hlen == 0 || hlen > 256 << 20 || 8 + hlen > file_bytes {
            return Err(Error(format!("{}: not a safetensors file (header length {hlen})", path.display())));
        }
        let mut hdr = vec![0u8; hlen as usize];
        f.read_exact_at(&mut hdr, 8).ctx("reading the header")?;
        let json: serde_json::Value = serde_json::from_slice(&hdr).ctx("parsing the header")?;
        let obj = json.as_object().ok_or("the header is not a JSON object")?;
        let base = 8 + hlen;
        let mut entries = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, val) in m {
                        metadata.insert(k.clone(), val.as_str().unwrap_or_default().to_string());
                    }
                }
                continue;
            }
            let bad = || Error(format!("header entry {name} is malformed"));
            let dtype = DType::parse(v.get("dtype").and_then(|d| d.as_str()).ok_or_else(bad)?)?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(bad)?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(bad))
                .collect::<Result<_>>()?;
            let offs = v.get("data_offsets").and_then(|o| o.as_array()).ok_or_else(bad)?;
            let (a, b) = (offs.first().and_then(|x| x.as_u64()).ok_or_else(bad)?, offs.get(1).and_then(|x| x.as_u64()).ok_or_else(bad)?);
            let bytes = (b - a) as usize;
            if bytes != shape.iter().product::<usize>() * dtype.size() || base + b > file_bytes {
                return Err(Error(format!("header entry {name}: byte range does not match its shape")));
            }
            entries.insert(name.clone(), Entry { name: name.clone(), dtype, shape, offset: base + a, bytes, kquant: None });
        }
        Ok(Checkpoint { path: path.to_path_buf(), entries, metadata, file_bytes })
    }

    fn open_gguf(path: &Path, file_bytes: u64) -> Result<Checkpoint> {
        let g = Gguf::open(path)?;
        let entries = g
            .entries
            .into_values()
            .map(|e| {
                let (dtype, kquant) = match e.ty {
                    GType::F32 => (DType::F32, None),
                    GType::F16 => (DType::F16, None),
                    GType::BF16 => (DType::BF16, None),
                    GType::Q4K | GType::Q6K => (DType::U8, Some(e.ty)),
                };
                (e.name.clone(), Entry { name: e.name, dtype, shape: e.shape, offset: e.offset, bytes: e.bytes, kquant })
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

    /// One tensor's bytes, read from the file.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.get(name)?;
        let f = File::open(&self.path)?;
        let mut v = vec![0u8; e.bytes];
        f.read_exact_at(&mut v, e.offset).ctx(format!("reading {name}"))?;
        Ok(v)
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
