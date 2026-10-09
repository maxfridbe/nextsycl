//! GGUF v3 (llama.cpp's format): an 8-byte magic and version, metadata key/values of any type (arrays included -
//! GLM5-Next keeps per-layer settings in them), the tensor directory, then the tensors' bytes, aligned. Only the
//! table of contents is read here; tensor bytes are read where they are used (`Gguf::read`, or by offset).
//!
//! A model split over several files (`split.count` > 1, `<stem>-00001-of-00003.gguf`) opens as one: the
//! metadata comes from the first shard, the tensors from all of them.

pub mod safetensors;

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T> {
    r.map_err(|e| Error(format!("{what}: {e}")))
}

/// A tensor's element type, by ggml's numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ1S,
    IQ1M,
    IQ4NL,
    IQ4XS,
    TQ1_0,
    TQ2_0,
    MXFP4,
    /// GSQ-RCO's 2-bit type (Qwen3.8-Flash-Next's files): 64 values, an f16 scale and 2-bit codes
    Q2_0,
}

impl GType {
    pub fn from_code(c: u32) -> Option<GType> {
        use GType::*;
        Some(match c {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => IQ2XXS,
            17 => IQ2XS,
            18 => IQ3XXS,
            19 => IQ1S,
            20 => IQ4NL,
            21 => IQ3S,
            22 => IQ2S,
            23 => IQ4XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            39 => MXFP4,
            42 => Q2_0,
            _ => return None,
        })
    }

    /// ggml's number for the type.
    pub fn code(self) -> u32 {
        use GType::*;
        match self {
            F32 => 0,
            F16 => 1,
            Q4_0 => 2,
            Q4_1 => 3,
            Q5_0 => 6,
            Q5_1 => 7,
            Q8_0 => 8,
            Q8_1 => 9,
            Q2K => 10,
            Q3K => 11,
            Q4K => 12,
            Q5K => 13,
            Q6K => 14,
            Q8K => 15,
            IQ2XXS => 16,
            IQ2XS => 17,
            IQ3XXS => 18,
            IQ1S => 19,
            IQ4NL => 20,
            IQ3S => 21,
            IQ2S => 22,
            IQ4XS => 23,
            I8 => 24,
            I16 => 25,
            I32 => 26,
            I64 => 27,
            F64 => 28,
            IQ1M => 29,
            BF16 => 30,
            TQ1_0 => 34,
            TQ2_0 => 35,
            MXFP4 => 39,
            Q2_0 => 42,
        }
    }

    /// (values per block, bytes per block).
    pub fn block(self) -> (usize, usize) {
        use GType::*;
        match self {
            F32 | I32 => (1, 4),
            F16 | BF16 | I16 => (1, 2),
            F64 | I64 => (1, 8),
            I8 => (1, 1),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            IQ4NL => (32, 18),
            MXFP4 => (32, 17),
            Q2_0 => (64, 18),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8K => (256, 292),
            IQ2XXS => (256, 66),
            IQ2XS => (256, 74),
            IQ2S => (256, 82),
            IQ3XXS => (256, 98),
            IQ3S => (256, 110),
            IQ1S => (256, 50),
            IQ1M => (256, 56),
            IQ4XS => (256, 136),
            TQ1_0 => (256, 54),
            TQ2_0 => (256, 66),
        }
    }

    /// Bytes of `n` values (a whole number of blocks).
    pub fn bytes(self, n: u64) -> Option<u64> {
        let (vpb, bpb) = self.block();
        n.is_multiple_of(vpb as u64).then(|| n / vpb as u64 * bpb as u64)
    }

    /// Bits per weight.
    pub fn bpw(self) -> f64 {
        let (v, b) = self.block();
        b as f64 * 8.0 / v as f64
    }

    pub fn name(self) -> &'static str {
        use GType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            BF16 => "BF16",
            F64 => "F64",
            I8 => "I8",
            I16 => "I16",
            I32 => "I32",
            I64 => "I64",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q8_1 => "Q8_1",
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            Q8K => "Q8_K",
            IQ2XXS => "IQ2_XXS",
            IQ2XS => "IQ2_XS",
            IQ2S => "IQ2_S",
            IQ3XXS => "IQ3_XXS",
            IQ3S => "IQ3_S",
            IQ1S => "IQ1_S",
            IQ1M => "IQ1_M",
            IQ4NL => "IQ4_NL",
            IQ4XS => "IQ4_XS",
            TQ1_0 => "TQ1_0",
            TQ2_0 => "TQ2_0",
            MXFP4 => "MXFP4",
            Q2_0 => "Q2_0",
        }
    }
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) => u64::try_from(*v).ok(),
            Value::Bool(b) => Some(*b as u64),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I(v) => Some(*v),
            Value::U(v) => i64::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    /// A short text form (long arrays and strings cut).
    pub fn brief(&self) -> String {
        match self {
            Value::Str(s) if s.chars().count() > 60 => format!("{:?}... ({} chars)", s.chars().take(60).collect::<String>(), s.chars().count()),
            Value::Str(s) => format!("{s:?}"),
            Value::Array(a) if a.len() > 12 => format!("[{} ... ] ({} values)", a.iter().take(8).map(Value::brief).collect::<Vec<_>>().join(", "), a.len()),
            Value::Array(a) => format!("[{}]", a.iter().map(Value::brief).collect::<Vec<_>>().join(", ")),
            Value::U(v) => v.to_string(),
            Value::I(v) => v.to_string(),
            Value::F(v) => format!("{v}"),
            Value::Bool(b) => b.to_string(),
        }
    }
}

/// One tensor: where its bytes are.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub ty: GType,
    /// row-major, outermost first (ggml stores the innermost dimension first; reversed here)
    pub shape: Vec<u64>,
    /// which file of a split model, and the absolute byte offset in it
    pub shard: usize,
    pub offset: u64,
    pub bytes: u64,
}

impl Tensor {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

pub struct Gguf {
    pub paths: Vec<PathBuf>,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<Tensor>,
    index: BTreeMap<String, usize>,
    files: Vec<File>,
}

struct Rd<R: Read> {
    r: R,
}

impl<R: Read> Rd<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        io(self.r.read_exact(&mut b), "reading the GGUF header")?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        if n > 64 << 20 {
            return Err(Error(format!("GGUF: a string of {n} bytes")));
        }
        let mut b = vec![0u8; n];
        io(self.r.read_exact(&mut b), "reading the GGUF header")?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn value(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U(self.bytes::<1>()?[0] as u64),
            1 => Value::I(self.bytes::<1>()?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.bytes()?) as u64),
            3 => Value::I(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(i32::from_le_bytes(self.bytes()?) as i64),
            6 => Value::F(f32::from_le_bytes(self.bytes()?) as f64),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if n > 1 << 26 {
                    return Err(Error(format!("GGUF: an array of {n} values")));
                }
                Value::Array((0..n).map(|_| self.value(et)).collect::<Result<_>>()?)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F(f64::from_le_bytes(self.bytes()?)),
            other => return Err(Error(format!("GGUF: unknown metadata type {other}"))),
        })
    }
}

/// One file: (version, metadata, tensors with shard 0, file length).
fn read_one(path: &Path) -> Result<(u32, BTreeMap<String, Value>, Vec<Tensor>)> {
    let f = io(File::open(path), &format!("opening {}", path.display()))?;
    let len = io(f.metadata(), "stat")?.len();
    let mut r = Rd { r: BufReader::new(f) };
    if &r.bytes::<4>()? != b"GGUF" {
        return Err(Error(format!("{}: not a GGUF file", path.display())));
    }
    let version = r.u32()?;
    if version != 3 {
        return Err(Error(format!("{}: GGUF version {version}; only 3 is read", path.display())));
    }
    let (nt, nkv) = (r.u64()?, r.u64()?);
    let mut metadata = BTreeMap::new();
    for _ in 0..nkv {
        let k = r.string()?;
        let t = r.u32()?;
        let v = r.value(t).map_err(|e| Error(format!("{}: metadata {k}: {}", path.display(), e.0)))?;
        metadata.insert(k, v);
    }
    let align = metadata.get("general.alignment").and_then(Value::as_u64).unwrap_or(32).max(1);
    let mut raw = Vec::with_capacity(nt as usize);
    for _ in 0..nt {
        let name = r.string()?;
        let nd = r.u32()? as usize;
        if nd > 8 {
            return Err(Error(format!("{}: tensor {name} has {nd} dimensions", path.display())));
        }
        let mut dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
        dims.reverse();
        let code = r.u32()?;
        let ty = GType::from_code(code).ok_or_else(|| Error(format!("{}: tensor {name}: ggml type {code} is not known here", path.display())))?;
        let off = r.u64()?;
        raw.push((name, ty, dims, off));
    }
    let pos = io(r.r.stream_position(), "seek")?;
    let base = pos.div_ceil(align) * align;
    let mut tensors = Vec::with_capacity(raw.len());
    for (name, ty, shape, off) in raw {
        let n: u64 = shape.iter().product();
        let bytes = ty.bytes(n).ok_or_else(|| Error(format!("{}: tensor {name}: {n} values are not whole {} blocks", path.display(), ty.name())))?;
        if base + off + bytes > len {
            return Err(Error(format!("{}: tensor {name} runs past the end of the file", path.display())));
        }
        tensors.push(Tensor { name, ty, shape, shard: 0, offset: base + off, bytes });
    }
    let _ = r.r.seek(SeekFrom::Start(0));
    Ok((version, metadata, tensors))
}

/// The other shards' paths of a split model, from the first one's name.
fn shard_paths(first: &Path, count: usize) -> Result<Vec<PathBuf>> {
    let name = first.file_name().and_then(|n| n.to_str()).ok_or("a shard name")?.to_string();
    let tag = format!("-00001-of-{count:05}.gguf");
    let stem = name.strip_suffix(&tag).ok_or_else(|| Error(format!("{name}: a split model's first file is named <stem>{tag}")))?;
    Ok((1..=count).map(|i| first.with_file_name(format!("{stem}-{i:05}-of-{count:05}.gguf"))).collect())
}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let (version, metadata, mut tensors) = read_one(path)?;
        let count = metadata.get("split.count").and_then(Value::as_u64).unwrap_or(1) as usize;
        let mut paths = vec![path.to_path_buf()];
        if count > 1 {
            paths = shard_paths(path, count)?;
            for (i, p) in paths.iter().enumerate().skip(1) {
                let (_, _, ts) = read_one(p)?;
                tensors.extend(ts.into_iter().map(|t| Tensor { shard: i, ..t }));
            }
        }
        let files = paths.iter().map(|p| io(File::open(p), &format!("opening {}", p.display()))).collect::<Result<Vec<_>>>()?;
        let mut index = BTreeMap::new();
        for (i, t) in tensors.iter().enumerate() {
            if index.insert(t.name.clone(), i).is_some() {
                return Err(Error(format!("{}: tensor {} appears twice", path.display(), t.name)));
            }
        }
        Ok(Gguf { paths, version, metadata, tensors, index, files })
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// `<arch>.<key>`, the architecture's own settings.
    pub fn arch_meta(&self, key: &str) -> Option<&Value> {
        let arch = self.meta("general.architecture")?.as_str()?;
        self.meta(&format!("{arch}.{key}"))
    }

    pub fn architecture(&self) -> &str {
        self.meta("general.architecture").and_then(Value::as_str).unwrap_or("?")
    }

    /// A tensor's bytes, read from its file.
    pub fn read(&self, t: &Tensor) -> Result<Vec<u8>> {
        let mut v = vec![0u8; t.bytes as usize];
        self.read_into(t, 0, &mut v)?;
        Ok(v)
    }

    /// Bytes `[at, at + dst.len())` of a tensor.
    pub fn read_into(&self, t: &Tensor, at: u64, dst: &mut [u8]) -> Result<()> {
        if at + dst.len() as u64 > t.bytes {
            return Err(Error(format!("{}: reading past its {} bytes", t.name, t.bytes)));
        }
        io(self.files[t.shard].read_exact_at(dst, t.offset + at), &format!("reading {}", t.name))
    }

    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_give_the_known_bit_rates() {
        assert_eq!(GType::Q4K.bpw(), 4.5);
        assert_eq!(GType::Q6K.bpw(), 6.5625);
        assert_eq!(GType::IQ2XXS.bpw(), 2.0625);
        assert_eq!(GType::Q8_0.bpw(), 8.5);
        for c in 0..64 {
            if let Some(t) = GType::from_code(c) {
                assert_eq!(t.code(), c);
            }
        }
    }

    #[test]
    fn shard_names() {
        let p = shard_paths(Path::new("/m/GLM-00001-of-00003.gguf"), 3).unwrap();
        assert_eq!(p[2], PathBuf::from("/m/GLM-00003-of-00003.gguf"));
    }

    /// A tiny file written here and read back: metadata of every kind, two tensors, alignment.
    #[test]
    fn round_trip() {
        let mut b: Vec<u8> = Vec::new();
        let s = |b: &mut Vec<u8>, x: &str| {
            b.extend((x.len() as u64).to_le_bytes());
            b.extend(x.as_bytes());
        };
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes()); // tensors
        b.extend(3u64.to_le_bytes()); // kv
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "glm5-next");
        s(&mut b, "glm5-next.attention.head_count_kv");
        b.extend(9u32.to_le_bytes());
        b.extend(4u32.to_le_bytes());
        b.extend(4u64.to_le_bytes());
        for v in [0u32, 0, 0, 1] {
            b.extend(v.to_le_bytes());
        }
        s(&mut b, "glm5-next.rope.freq_base");
        b.extend(6u32.to_le_bytes());
        b.extend(5e6f32.to_le_bytes());
        // a: f32 [2, 3]; b: Q8_0 [1, 32]
        s(&mut b, "a");
        b.extend(2u32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        s(&mut b, "b");
        b.extend(2u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b.extend((0..6).flat_map(|i| (i as f32).to_le_bytes()));
        b.resize(b.len() + 8, 0);
        b.extend([7u8; 34]);
        let dir = std::env::temp_dir().join(format!("ns-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.gguf");
        std::fs::write(&p, &b).unwrap();
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.architecture(), "glm5-next");
        assert_eq!(g.arch_meta("attention.head_count_kv").unwrap().as_array().unwrap().len(), 4);
        assert!((g.arch_meta("rope.freq_base").unwrap().as_f64().unwrap() - 5e6).abs() < 1.0);
        let a = g.tensor("a").unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        let v: Vec<f32> = g.read(a).unwrap().chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = g.tensor("b").unwrap();
        assert_eq!((t.ty, t.bytes), (GType::Q8_0, 34));
        assert_eq!(g.read(t).unwrap(), vec![7u8; 34]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
