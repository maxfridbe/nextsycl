//! The denoiser's file as a table of contents: ComfyUI's int8 ConvRot safetensors, or a stable-diffusion.cpp GGUF whose
//! block matrices are llama.cpp quants (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K). The foundation's H3 checkpoint reads the first as
//! it is; a GGUF is laid over it here - every float tensor as such, every quantized matrix as its raw bytes with its
//! type kept beside (`quants`) - so Q3_K, which H3's own reader does not take, loads like the others.

use std::collections::BTreeMap;
use std::path::Path;

use nextsycl_gguf::GType;
use nextsycl_h3::dtype::DType;
use nextsycl_h3::safetensors::{Checkpoint, Entry};

use crate::{h3e, Error, Result};

/// A block matrix's llama.cpp quantization
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    Q8_0,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Quant {
    pub fn of(t: GType) -> Option<Quant> {
        Some(match t {
            GType::Q8_0 => Quant::Q8_0,
            GType::Q3K => Quant::Q3K,
            GType::Q4K => Quant::Q4K,
            GType::Q5K => Quant::Q5K,
            GType::Q6K => Quant::Q6K,
            _ => return None,
        })
    }

    /// ggml's number for the type (what the kernels are told)
    pub fn code(self) -> i32 {
        match self {
            Quant::Q8_0 => 8,
            Quant::Q3K => 11,
            Quant::Q4K => 12,
            Quant::Q5K => 13,
            Quant::Q6K => 14,
        }
    }

    /// (values, bytes) of one block
    pub fn block(self) -> (usize, usize) {
        match self {
            Quant::Q8_0 => (32, 34),
            Quant::Q3K => (256, 110),
            Quant::Q4K => (256, 144),
            Quant::Q5K => (256, 176),
            Quant::Q6K => (256, 210),
        }
    }

    /// Bytes of `n` values (whole blocks)
    pub fn bytes(self, n: usize) -> usize {
        let (v, b) = self.block();
        n / v * b
    }

    /// The type as H3's shared expansion knows it (None: only the silo expands it)
    pub fn shared(self) -> Option<nextsycl_h3::gguf::GType> {
        use nextsycl_h3::gguf::GType as H;
        Some(match self {
            Quant::Q8_0 => H::Q8_0,
            Quant::Q4K => H::Q4K,
            Quant::Q5K => H::Q5K,
            Quant::Q6K => H::Q6K,
            Quant::Q3K => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Quant::Q8_0 => "Q8_0",
            Quant::Q3K => "Q3_K",
            Quant::Q4K => "Q4_K",
            Quant::Q5K => "Q5_K",
            Quant::Q6K => "Q6_K",
        }
    }
}

/// The file: its tensors by name (`ck`), and the quantized matrices' types and shapes
pub struct Weights {
    pub ck: Checkpoint,
    pub quants: BTreeMap<String, (Quant, Vec<usize>)>,
}

/// One GGUF tensor as a checkpoint entry: a float tensor in its type, a quantized one as bytes
pub fn entry(name: &str, ty: GType, shape: &[usize], offset: u64, bytes: usize) -> Result<(Entry, Option<Quant>)> {
    let n: usize = shape.iter().product();
    let float = |dtype: DType| -> Result<(Entry, Option<Quant>)> {
        if bytes != n * dtype.size() {
            return Err(Error(format!("{name}: {bytes} bytes for {shape:?} {dtype:?}")));
        }
        Ok((Entry { name: name.to_string(), dtype, shape: shape.to_vec(), offset, bytes, kquant: None }, None))
    };
    match ty {
        GType::F32 => float(DType::F32),
        GType::F16 => float(DType::F16),
        GType::BF16 => float(DType::BF16),
        other => {
            let q = Quant::of(other).ok_or_else(|| Error(format!("{name}: GGUF type {} is not read (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K and floats are)", other.name())))?;
            let (v, _) = q.block();
            let k = shape.last().copied().unwrap_or(0);
            if shape.len() != 2 || !k.is_multiple_of(v) || bytes != q.bytes(n) {
                return Err(Error(format!("{name}: a {} tensor of shape {shape:?} in {bytes} bytes (a matrix of whole blocks a row is expected)", q.name())));
            }
            // its bytes as they go to the card: a flat buffer
            Ok((Entry { name: name.to_string(), dtype: DType::U8, shape: vec![bytes], offset, bytes, kquant: None }, Some(q)))
        }
    }
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        let mut magic = [0u8; 4];
        {
            use std::os::unix::fs::FileExt;
            let f = std::fs::File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
            f.read_exact_at(&mut magic, 0).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        }
        if &magic != b"GGUF" {
            return Ok(Weights { ck: Checkpoint::open(path).map_err(h3e)?, quants: BTreeMap::new() });
        }
        let g = nextsycl_gguf::Gguf::open(path).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
        let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let mut entries = BTreeMap::new();
        let mut quants = BTreeMap::new();
        for t in &g.tensors {
            if t.shard != 0 {
                return Err(Error(format!("{}: a split file", path.display())));
            }
            // stable-diffusion.cpp's converter keeps ComfyUI's prefix on some files
            let name = t.name.strip_prefix("model.diffusion_model.").unwrap_or(&t.name);
            let shape: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
            if t.offset + t.bytes > file_bytes {
                return Err(Error(format!("{}: {name} ends past the file", path.display())));
            }
            let (e, q) = entry(name, t.ty, &shape, t.offset, t.bytes as usize)?;
            if let Some(q) = q {
                quants.insert(name.to_string(), (q, shape));
            }
            entries.insert(name.to_string(), e);
        }
        Ok(Weights { ck: Checkpoint { path: path.to_path_buf(), entries, metadata: BTreeMap::new(), file_bytes }, quants })
    }

    /// The type the blocks' matrices are in (None: the int8 file), from the first block's MLP
    pub fn block_quant(&self) -> Option<Quant> {
        self.quants.get("blocks.0.mlp.fc1.weight").map(|q| q.0)
    }

    /// A tensor's values as float32 on the host (a float tensor)
    pub fn host(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.ck.get(name).map_err(h3e)?;
        if self.quants.contains_key(name) {
            return Err(Error(format!("{name}: quantized; expanded on the card")));
        }
        nextsycl_h3::dtype::bytes_to_f32(&self.ck.read(name).map_err(h3e)?, e.dtype).map_err(h3e)
    }

    /// A matrix's shape [N, K]
    pub fn shape2(&self, name: &str) -> Result<(usize, usize)> {
        let s = match self.quants.get(name) {
            Some((_, s)) => s.clone(),
            None => self.ck.get(name).map_err(h3e)?.shape.clone(),
        };
        if s.len() != 2 {
            return Err(Error(format!("{name}: a matrix is expected, got {s:?}")));
        }
        Ok((s[0], s[1]))
    }
}

/// One Q8_0 / Q3_K / Q4_K / Q5_K / Q6_K block's values, as ggml's `dequantize_row_*` computes them (the host's copy of
/// what the kernels do: the unit tests' reference, and the check that a silo's expansion is ggml's)
pub fn dequant_block(q: Quant, b: &[u8], out: &mut [f32]) {
    let f16 = |lo: u8, hi: u8| nextsycl_h3::dtype::f16_to_f32(u16::from_le_bytes([lo, hi]));
    // ggml's get_scale_min_k4: the 6-bit scale and minimum of sub-block j from the 12 packed bytes
    let k4 = |s: &[u8], j: usize| -> (u8, u8) {
        if j < 4 {
            (s[j] & 63, s[j + 4] & 63)
        } else {
            ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))
        }
    };
    match q {
        Quant::Q8_0 => {
            let d = f16(b[0], b[1]);
            for i in 0..32 {
                out[i] = d * (b[2 + i] as i8) as f32;
            }
        }
        Quant::Q4K => {
            // d, dmin (half), scales[12], qs[128]: 8 sub-blocks of 32, low nibbles then high nibbles of 32 bytes
            let (d, dmin) = (f16(b[0], b[1]), f16(b[2], b[3]));
            let (sc, qs) = (&b[4..16], &b[16..144]);
            for j in 0..8 {
                let (s, m) = k4(sc, j);
                let (dl, ml) = (d * s as f32, dmin * m as f32);
                let q = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
                for l in 0..32 {
                    let v = if j % 2 == 0 { q[l] & 0xF } else { q[l] >> 4 };
                    out[j * 32 + l] = dl * v as f32 - ml;
                }
            }
        }
        Quant::Q5K => {
            // d, dmin, scales[12], qh[32], qs[128]: as Q4_K with a fifth bit per value in qh
            let (d, dmin) = (f16(b[0], b[1]), f16(b[2], b[3]));
            let (sc, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
            for j in 0..8 {
                let (s, m) = k4(sc, j);
                let (dl, ml) = (d * s as f32, dmin * m as f32);
                let q = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
                for l in 0..32 {
                    let lo = if j % 2 == 0 { q[l] & 0xF } else { q[l] >> 4 };
                    let hi = if qh[l] & (1 << j) != 0 { 16 } else { 0 };
                    out[j * 32 + l] = dl * (lo + hi) as f32 - ml;
                }
            }
        }
        Quant::Q6K => {
            // ql[128], qh[64], scales[16] (int8), d (half): 16 sub-blocks of 16, 6-bit values less 32
            let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
            let d = f16(b[208], b[209]);
            for half in 0..2 {
                let (ql, qh, sc) = (&ql[half * 64..], &qh[half * 32..], &sc[half * 8..]);
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    let o = half * 128;
                    out[o + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    out[o + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    out[o + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    out[o + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }
        Quant::Q3K => {
            // hmask[32], qs[64], scales[12], d (half): 16 sub-blocks of 16, 3-bit values (2 bits in qs, the third - inverted -
            // in hmask) less 4, 6-bit scales less 32
            let (hm, qs, sc) = (&b[0..32], &b[32..96], &b[96..108]);
            let d = f16(b[108], b[109]);
            let w = |i: usize| u32::from_le_bytes([sc[i], sc[i + 1], sc[i + 2], sc[i + 3]]);
            let (a0, a1, tmp) = (w(0), w(4), w(8));
            const K1: u32 = 0x0303_0303;
            const K2: u32 = 0x0f0f_0f0f;
            let aux = [(a0 & K2) | ((tmp & K1) << 4), (a1 & K2) | (((tmp >> 2) & K1) << 4), ((a0 >> 4) & K2) | (((tmp >> 4) & K1) << 4),
                       ((a1 >> 4) & K2) | (((tmp >> 6) & K1) << 4)];
            let scale = |i: usize| (aux[i / 4].to_le_bytes()[i % 4] as i8) as i32 - 32;
            let mut is = 0;
            for half in 0..2 {
                let m0 = 1u8 << (half * 4);
                let q = &qs[half * 32..half * 32 + 32];
                for (j, shift) in [0u8, 2, 4, 6].into_iter().enumerate() {
                    let m = m0 << j;
                    for part in 0..2 {
                        let dl = d * scale(is) as f32;
                        is += 1;
                        for l in 0..16 {
                            let i = part * 16 + l;
                            let v = ((q[i] >> shift) & 3) as i32 - if hm[i] & m != 0 { 0 } else { 4 };
                            out[half * 128 + j * 32 + i] = dl * v as f32;
                        }
                    }
                }
            }
        }
    }
}

/// A quantized matrix's values on the host (tests, and the small tensors a GGUF quantizes outside the blocks)
pub fn dequant(q: Quant, bytes: &[u8]) -> Vec<f32> {
    let (v, b) = q.block();
    let mut out = vec![0f32; bytes.len() / b * v];
    for (blk, o) in bytes.chunks_exact(b).zip(out.chunks_exact_mut(v)) {
        dequant_block(q, blk, o);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(v: f32) -> [u8; 2] {
        nextsycl_h3::dtype::f32_to_f16(v).to_le_bytes()
    }

    #[test]
    fn block_sizes_are_ggmls() {
        for (q, bpw) in [(Quant::Q8_0, 8.5), (Quant::Q3K, 3.4375), (Quant::Q4K, 4.5), (Quant::Q5K, 5.5), (Quant::Q6K, 6.5625)] {
            let (v, b) = q.block();
            assert_eq!(b as f64 * 8.0 / v as f64, bpw, "{q:?}");
            assert_eq!(q.bytes(5376 * 21504), 5376 * 21504 / v * b);
            assert_eq!(Quant::of(nextsycl_gguf::GType::from_code(q.code() as u32).unwrap()), Some(q));
        }
        assert!(Quant::Q3K.shared().is_none() && Quant::Q4K.shared().is_some());
    }

    #[test]
    fn q8_0_is_scale_times_int8() {
        let mut b = half(0.5).to_vec();
        b.extend((0..32).map(|i| (i as i8 - 16) as u8));
        let v = dequant(Quant::Q8_0, &b);
        assert_eq!(v[0], -8.0);
        assert_eq!(v[31], 7.5);
    }

    #[test]
    fn q4_k_reads_nibbles_scales_and_minimums() {
        // d = 1, dmin = 0.5; sub-block 0: scale 3, min 2; sub-block 5 (the packed upper form): scale 37, min 41
        let mut b = Vec::new();
        b.extend(half(1.0));
        b.extend(half(0.5));
        let mut sc = [0u8; 12];
        sc[0] = 3;
        sc[4] = 2;
        // j = 5: scale = (sc[9] & 0xF) | ((sc[1] >> 6) << 4), min = (sc[9] >> 4) | ((sc[5] >> 6) << 4)
        sc[9] = (37 & 0xF) | ((41 & 0xF) << 4);
        sc[1] = (37 >> 4) << 6;
        sc[5] = (41 >> 4) << 6;
        b.extend(sc);
        let mut qs = [0u8; 128];
        qs[0] = 0x0A; // sub-block 0, value 0: 10
        qs[64 + 7] = 0xC0; // sub-block 5 (high nibbles of bytes 64..96), value 7: 12
        b.extend(qs);
        let v = dequant(Quant::Q4K, &b);
        assert_eq!(v[0], 3.0 * 10.0 - 0.5 * 2.0);
        assert_eq!(v[1], -1.0);
        assert_eq!(v[5 * 32 + 7], 37.0 * 12.0 - 0.5 * 41.0);
    }

    #[test]
    fn q5_k_adds_the_fifth_bit() {
        let mut b = Vec::new();
        b.extend(half(2.0));
        b.extend(half(0.0));
        let mut sc = [0u8; 12];
        sc[1] = 1; // sub-block 1: scale 1
        b.extend(sc);
        let mut qh = [0u8; 32];
        qh[3] = 1 << 1; // value 3 of sub-block 1 has its fifth bit
        b.extend(qh);
        let mut qs = [0u8; 128];
        qs[3] = 0x50; // sub-block 1 = high nibbles of bytes 0..32
        b.extend(qs);
        let v = dequant(Quant::Q5K, &b);
        assert_eq!(v[32 + 3], 2.0 * (5 + 16) as f32);
        assert_eq!(v[32 + 4], 0.0);
    }

    #[test]
    fn q6_k_is_six_bits_less_32_by_int8_scales() {
        let mut b = vec![0u8; 210];
        b[0] = 0x07; // ql[0] low nibble: value 0 of sub-block 0
        b[128] = 0x02; // qh[0] bits 0..2: its upper two bits
        b[192] = (-3i8) as u8; // scale of sub-block 0
        b[208..210].copy_from_slice(&half(0.25));
        let v = dequant(Quant::Q6K, &b);
        assert_eq!(v[0], 0.25 * -3.0 * ((7 | (2 << 4)) - 32) as f32);
        // an all-zero value is -32
        assert_eq!(v[1], 0.25 * -3.0 * -32.0);
    }

    #[test]
    fn q3_k_is_three_bits_less_4_by_scales_less_32() {
        let mut b = vec![0u8; 110];
        // sub-block 0's scale: the low 4 bits of scales[0], its upper 2 bits in scales[8] bits 0..2 -> 0b10_0101 = 37 -> +5
        b[96] = 0x05;
        b[96 + 8] = 0x02;
        b[32] = 0x03; // qs[0] bits 0..2: value 0 = 3
        b[0] = 0x01; // hmask[0] bit 0 set: no -4
        b[108..110].copy_from_slice(&half(0.5));
        let v = dequant(Quant::Q3K, &b);
        assert_eq!(v[0], 0.5 * 5.0 * 3.0);
        // value 1: bits 0, mask clear -> -4
        assert_eq!(v[1], 0.5 * 5.0 * -4.0);
        // sub-block 2 (values 32..48) uses scale index 2 = 0 - 32
        assert_eq!(v[32], 0.5 * -32.0 * -4.0);
    }

    #[test]
    fn a_gguf_tensor_becomes_an_entry() {
        let (e, q) = entry("blocks.0.mlp.fc2.weight", GType::Q3K, &[5376, 14336], 100, 5376 * 14336 / 256 * 110).unwrap();
        assert_eq!((e.dtype, e.shape.clone(), q), (DType::U8, vec![5376 * 14336 / 256 * 110], Some(Quant::Q3K)));
        let (e, q) = entry("rope.inv_freq", GType::F32, &[16], 0, 64).unwrap();
        assert_eq!((e.dtype, e.shape.clone(), q), (DType::F32, vec![16], None));
        assert!(entry("x", GType::Q4K, &[3, 100], 0, 1).is_err());
        assert!(entry("x", GType::IQ4XS, &[1, 256], 0, 136).is_err());
    }
}
