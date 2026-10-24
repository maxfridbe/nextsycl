//! ncnn's model files, read on the host: the `.param` graph (layers in order, their blobs by name, their parameters
//! `id=value`, arrays `-233xx=n,v1,...`) and the `.bin` weights each layer takes in that order - a weight blob behind
//! a 32-bit flag (0: float32; 0x01306B47: half, padded to 4 bytes), a bias or slope as plain float32. Only what the
//! upscalers' graphs use is read: Input, Convolution, Deconvolution, PReLU, PixelShuffle, Interp, BinaryOp, Eltwise,
//! Concat, Split, Crop, Pooling, InnerProduct, Scale; another layer is refused at load, by name.

use std::collections::HashMap;

/// A parameter value: one number or an array
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Int(i32),
    Float(f32),
    Array(Vec<f32>),
}

/// A layer's parameters by id
#[derive(Clone, Debug, Default)]
pub struct Params(pub HashMap<i32, Val>);

impl Params {
    pub fn int(&self, id: i32, default: i32) -> i32 {
        match self.0.get(&id) {
            Some(Val::Int(v)) => *v,
            Some(Val::Float(v)) => *v as i32,
            _ => default,
        }
    }
    pub fn float(&self, id: i32, default: f32) -> f32 {
        match self.0.get(&id) {
            Some(Val::Float(v)) => *v,
            Some(Val::Int(v)) => *v as f32,
            _ => default,
        }
    }
    /// An array parameter (`-23300 - id` in the file)
    pub fn array(&self, id: i32) -> Vec<f32> {
        match self.0.get(&id) {
            Some(Val::Array(v)) => v.clone(),
            _ => Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub kind: String,
    pub name: String,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub params: Params,
}

/// The graph: layers in run order, blobs numbered as they appear
#[derive(Clone, Debug)]
pub struct Graph {
    pub layers: Vec<Layer>,
    pub blobs: Vec<String>,
}

const KNOWN: &[&str] = &["Input", "Convolution", "Deconvolution", "PReLU", "PixelShuffle", "Interp", "BinaryOp", "Eltwise", "Concat", "Split",
    "Crop", "Pooling", "InnerProduct", "Scale"];

fn value(v: &str) -> Result<Val, String> {
    if v.contains('.') || v.contains('e') || v.contains('E') {
        v.parse::<f32>().map(Val::Float).map_err(|e| format!("{v}: {e}"))
    } else {
        v.parse::<i32>().map(Val::Int).map_err(|e| format!("{v}: {e}"))
    }
}

/// A `.param` file's text
pub fn parse_param(text: &str) -> Result<Graph, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    if lines.next() != Some("7767517") {
        return Err("not an ncnn param file (its magic is not 7767517)".into());
    }
    let counts: Vec<usize> = lines.next().ok_or("no layer count")?.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    if counts.len() != 2 {
        return Err("the layer and blob counts are not two numbers".into());
    }
    let mut blobs: Vec<String> = Vec::new();
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let mut blob = |name: &str| -> usize {
        if let Some(i) = by_name.get(name) {
            return *i;
        }
        blobs.push(name.to_string());
        by_name.insert(name.to_string(), blobs.len() - 1);
        blobs.len() - 1
    };
    let mut layers = Vec::new();
    for line in lines {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 4 {
            return Err(format!("a short layer line: {line}"));
        }
        let (kind, name) = (t[0].to_string(), t[1].to_string());
        if !KNOWN.contains(&kind.as_str()) {
            return Err(format!("layer {name}: {kind} is not run here (the upscalers' graphs use {})", KNOWN.join(", ")));
        }
        let ni: usize = t[2].parse().map_err(|_| format!("{name}: its input count"))?;
        let no: usize = t[3].parse().map_err(|_| format!("{name}: its output count"))?;
        if t.len() < 4 + ni + no {
            return Err(format!("{name}: fewer blobs than it says"));
        }
        let inputs = t[4..4 + ni].iter().map(|b| blob(b)).collect();
        let outputs = t[4 + ni..4 + ni + no].iter().map(|b| blob(b)).collect();
        let mut params = Params::default();
        for kv in &t[4 + ni + no..] {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("{name}: a parameter without '=': {kv}"))?;
            let id: i32 = k.parse().map_err(|_| format!("{name}: parameter id {k}"))?;
            if id <= -23300 {
                let mut parts = v.split(',');
                let n: usize = parts.next().unwrap_or("0").parse().map_err(|_| format!("{name}: array {k}'s length"))?;
                let vals: Vec<f32> = parts.map(|x| x.parse::<f32>().map_err(|e| format!("{name}: {x}: {e}"))).collect::<Result<_, _>>()?;
                if vals.len() != n {
                    return Err(format!("{name}: array {k} says {n} values, has {}", vals.len()));
                }
                params.0.insert(-id - 23300, Val::Array(vals));
            } else {
                params.0.insert(id, value(v).map_err(|e| format!("{name}: parameter {k}: {e}"))?);
            }
        }
        layers.push(Layer { kind, name, inputs, outputs, params });
    }
    if layers.len() != counts[0] {
        return Err(format!("{} layers, the file says {}", layers.len(), counts[0]));
    }
    Ok(Graph { layers, blobs })
}

/// A weight as the engine keeps it: half bits (as ncnn stored them, or made from its float32)
#[derive(Clone, Debug)]
pub enum Data {
    F32(Vec<f32>),
    F16(Vec<u16>),
}

impl Data {
    pub fn len(&self) -> usize {
        match self {
            Data::F32(v) => v.len(),
            Data::F16(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Data::F32(v) => v.clone(),
            Data::F16(v) => v.iter().map(|h| f16_to_f32(*h)).collect(),
        }
    }
    pub fn to_f16(&self) -> Vec<u16> {
        match self {
            Data::F32(v) => v.iter().map(|f| f32_to_f16(*f)).collect(),
            Data::F16(v) => v.clone(),
        }
    }
}

/// The `.bin` file read in order
pub struct Bin<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Bin<'a> {
    pub fn new(b: &'a [u8]) -> Bin<'a> {
        Bin { b, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.at + n > self.b.len() {
            return Err(format!("the weights end early ({} bytes, {} more asked at {})", self.b.len(), n, self.at));
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    /// `mb.load(n, 0)`: a flag, then the data in the type it names
    pub fn weight(&mut self, n: usize) -> Result<Data, String> {
        let flag = u32::from_le_bytes(self.take(4)?.try_into().unwrap());
        match flag {
            0x01306B47 => {
                let raw = self.take(n * 2)?;
                let v = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                // half data is padded to a multiple of 4 bytes
                let pad = (4 - (n * 2) % 4) % 4;
                self.take(pad)?;
                Ok(Data::F16(v))
            }
            0 | 0x0002C056 => Ok(Data::F32(self.f32s(n)?)),
            f => Err(format!("a weight blob of type {f:#010x} (int8 or quantized tables are not read)")),
        }
    }
    /// `mb.load(n, 1)`: plain float32
    pub fn floats(&mut self, n: usize) -> Result<Vec<f32>, String> {
        self.f32s(n)
    }
    pub fn rest(&self) -> usize {
        self.b.len() - self.at
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalise
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Round to nearest even, as the GPU's __float2half_rn
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let s = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return s | r as u16;
    }
    let mut r = ((e2 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    s | r as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_param_file_is_read_with_its_arrays_and_blobs() {
        let p = "7767517\n3 4\nInput in 0 1 data 0=200 1=200 2=3\n\
                 Convolution c1 1 1 data a 0=32 1=3 5=1 6=864 9=2 -23310=1,0.100000\n\
                 Eltwise e 2 1 a data b 0=1 -23301=2,2.000000e-01,1.000000e+00\n";
        let g = parse_param(p).unwrap();
        assert_eq!(g.layers.len(), 3);
        assert_eq!(g.blobs, ["data", "a", "b"]);
        let c = &g.layers[1];
        assert_eq!((c.params.int(0, 0), c.params.int(1, 0), c.params.int(4, 0), c.params.int(9, 0)), (32, 3, 0, 2));
        assert_eq!(c.params.array(10), [0.1]);
        assert_eq!(g.layers[2].params.array(1), [0.2, 1.0]);
        assert_eq!(g.layers[2].inputs, [1, 0]);
        assert!(parse_param("7767517\n1 1\nSoftmax s 1 1 a b\n").unwrap_err().contains("not run here"));
    }

    #[test]
    fn weights_come_flagged_and_biases_plain() {
        let mut b = Vec::new();
        b.extend(0x01306B47u32.to_le_bytes());
        for h in [f32_to_f16(1.5), f32_to_f16(-2.0), f32_to_f16(0.25)] {
            b.extend(h.to_le_bytes());
        }
        b.extend([0, 0]); // padding to 4 bytes
        b.extend(0u32.to_le_bytes());
        b.extend(3.0f32.to_le_bytes());
        b.extend(7.0f32.to_le_bytes());
        let mut r = Bin::new(&b);
        assert_eq!(r.weight(3).unwrap().to_f32(), [1.5, -2.0, 0.25]);
        assert_eq!(r.weight(1).unwrap().to_f32(), [3.0]);
        assert_eq!(r.floats(1).unwrap(), [7.0]);
        assert_eq!(r.rest(), 0);
    }

    #[test]
    fn halves_round_trip() {
        for f in [0.0f32, 1.0, -1.0, 0.1, 65504.0, 1e-5, -3.14159, 6e-8] {
            let h = f32_to_f16(f);
            let back = f16_to_f32(h);
            assert!((back - f).abs() <= f.abs() * 1e-3 + 6e-8, "{f} -> {back}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(70000.0), 0x7c00);
    }
}
