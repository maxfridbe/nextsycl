//! The model's GGUF (CrispASR's conversion, `general.architecture` breeze-tts-2): its tensors found by name, expanded
//! from their blocks on the host - F32, F16, Q8_0, Q4_0, Q4_K (the q8_0 and q4_k files' types) - and put on the GPU in
//! half (the matrices) or float32 (the norms). The text encoder's embedding stays in the file: a row read a token.

use std::path::Path;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_gguf::safetensors::f16_to_f32;
use nextsycl_gguf::{GType, Gguf, Tensor, Value};

use crate::ops::{Mat, Ops};

pub struct Weights {
    pub g: Gguf,
}

/// One block type's values (host): `b` holds whole blocks of `t`
pub fn dequant(t: GType, b: &[u8], out: &mut [f32]) -> Result<()> {
    let h = |i: usize| f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]));
    match t {
        GType::F32 => out.iter_mut().zip(b.chunks_exact(4)).for_each(|(o, c)| *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        GType::F16 => out.iter_mut().zip(b.chunks_exact(2)).for_each(|(o, c)| *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        GType::Q8_0 => {
            for (blk, o) in out.chunks_exact_mut(32).enumerate() {
                let at = blk * 34;
                let d = h(at);
                for (j, v) in o.iter_mut().enumerate() {
                    *v = d * b[at + 2 + j] as i8 as f32;
                }
            }
        }
        GType::Q4_0 => {
            for (blk, o) in out.chunks_exact_mut(32).enumerate() {
                let at = blk * 18;
                let d = h(at);
                for j in 0..16 {
                    let q = b[at + 2 + j];
                    o[j] = d * ((q & 0xF) as i32 - 8) as f32;
                    o[j + 16] = d * ((q >> 4) as i32 - 8) as f32;
                }
            }
        }
        GType::Q4K => {
            // llama.cpp's block_q4_K: d, dmin (half), 12 bytes of 6-bit scales and mins, 128 bytes of nibbles
            for (blk, o) in out.chunks_exact_mut(256).enumerate() {
                let at = blk * 144;
                let (d, dmin) = (h(at), h(at + 2));
                let sc = &b[at + 4..at + 16];
                let qs = &b[at + 16..at + 144];
                let scale_min = |j: usize| -> (f32, f32) {
                    if j < 4 {
                        ((sc[j] & 63) as f32, (sc[j + 4] & 63) as f32)
                    } else {
                        (((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as f32, ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f32)
                    }
                };
                for (g, y) in o.chunks_exact_mut(64).enumerate() {
                    let (s1, m1) = scale_min(2 * g);
                    let (s2, m2) = scale_min(2 * g + 1);
                    let q = &qs[32 * g..32 * g + 32];
                    for l in 0..32 {
                        y[l] = d * s1 * (q[l] & 0xF) as f32 - dmin * m1;
                        y[32 + l] = d * s2 * (q[l] >> 4) as f32 - dmin * m2;
                    }
                }
            }
        }
        t => return Err(Error(format!("tensor type {} is not one this engine reads (F32, F16, Q8_0, Q4_0, Q4_K)", t.name()))),
    }
    Ok(())
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        let g = Gguf::open(path).map_err(|e| Error(e.0))?;
        if g.architecture() != crate::ARCH {
            return Err(Error(format!("{}: architecture {}, not {}", path.display(), g.architecture(), crate::ARCH)));
        }
        Ok(Weights { g })
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.g.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.g.paths[0].display())))
    }

    pub fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }

    /// A metadata number (`breeze.<key>`)
    pub fn num(&self, key: &str) -> Result<f64> {
        self.g.meta(&format!("breeze.{key}")).and_then(Value::as_f64).ok_or_else(|| Error(format!("{}: no breeze.{key}", self.g.paths[0].display())))
    }

    pub fn flag(&self, key: &str) -> bool {
        matches!(self.g.meta(&format!("breeze.{key}")), Some(Value::Bool(true)))
    }

    /// Rows [r0, r1) of a 2-D tensor (or a whole 1-D one: rows of one value), as float32 on the host
    pub fn rows(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let k = if t.shape.len() > 1 { t.shape[1..].iter().product::<u64>() as usize } else { 1 };
        let (per, bytes) = t.ty.block();
        if k % per != 0 {
            return Err(Error(format!("{name}: rows of {k} values, not whole {} blocks", t.ty.name())));
        }
        let row_bytes = k / per * bytes;
        let mut b = vec![0u8; (r1 - r0) * row_bytes];
        self.g.read_into(t, (r0 * row_bytes) as u64, &mut b).map_err(|e| Error(e.0))?;
        let mut out = vec![0f32; (r1 - r0) * k];
        // the rows split over threads (a matrix of tens of millions of values)
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
        let per_t = (r1 - r0).div_ceil(threads).max(1);
        std::thread::scope(|s| -> Result<()> {
            let jobs: Vec<_> = out.chunks_mut(per_t * k).zip(b.chunks(per_t * row_bytes)).map(|(o, bb)| s.spawn(move || dequant(t.ty, bb, o))).collect();
            for j in jobs {
                j.join().map_err(|_| Error("a dequantizing thread failed".into()))??;
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// A 1-D tensor (a norm's weight) as float32 on the GPU
    pub fn vec(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        let n = self.tensor(name)?.elements() as usize;
        DevBuf::from_f32(&ops.gpu, &self.rows(name, 0, n)?)
    }

    /// Matrices [n_i, k] stacked by rows (a fused q / k / v, gate / up) in half on the GPU
    pub fn mat(&self, ops: &Ops, names: &[&str]) -> Result<Mat> {
        let mut k = 0;
        let mut n = 0;
        for name in names {
            let s = &self.tensor(name)?.shape;
            if s.len() != 2 || (k != 0 && s[1] as usize != k) {
                return Err(Error(format!("{name}: shape {s:?}, a matrix {k} wide expected")));
            }
            k = s[1] as usize;
            n += s[0] as usize;
        }
        let gpu = &ops.gpu;
        let w = DevBuf::new(gpu, n * k * 2)?;
        let mut row = 0;
        for name in names {
            let t = self.tensor(name)?;
            let rows = t.shape[0] as usize;
            if t.ty == GType::F16 {
                // already half: the bytes as they are
                let b = self.g.read(t).map_err(|e| Error(e.0))?;
                w.write(row * k * 2, &b)?;
            } else {
                // in pieces of up to 16M values through a float32 staging buffer
                let step = ((16usize << 20) / k).max(1);
                let stage = DevBuf::f32(gpu, step.min(rows) * k)?;
                let mut r0 = 0;
                while r0 < rows {
                    let r1 = (r0 + step).min(rows);
                    let v = self.rows(name, r0, r1)?;
                    stage.write(0, &crate::bytes(&v))?;
                    ops.to_half(stage.fp(), w.ptr().wrapping_byte_add((row + r0) * k * 2), v.len())?;
                    gpu.sync()?;
                    r0 = r1;
                }
            }
            row += rows;
        }
        Ok(Mat { w, n, k })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_expand_as_llama_cpp_does() {
        // Q8_0: d = 0.5 (half 0x3800), codes -2, 3, then zeros
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        b[2] = (-2i8) as u8;
        b[3] = 3;
        let mut o = vec![0f32; 32];
        dequant(GType::Q8_0, &b, &mut o).unwrap();
        assert_eq!(&o[..3], &[-1.0, 1.5, 0.0]);
        // Q4_0: d = 1, a byte 0x9F: low nibble 15 - 8 = 7 (value 0), high 9 - 8 = 1 (value 16)
        let mut b = vec![0x88u8; 18];
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[2] = 0x9F;
        dequant(GType::Q4_0, &b, &mut o).unwrap();
        assert_eq!((o[0], o[16], o[1]), (7.0, 1.0, 0.0));
        // Q4_K: d = 1, dmin = 1; sub-block 0's scale 2 and min 1, the nibble 5: 1 * 2 * 5 - 1 * 1 = 9
        let mut b = vec![0u8; 144];
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[4] = 2;
        b[8] = 1;
        b[16] = 0x05;
        let mut o = vec![0f32; 256];
        dequant(GType::Q4K, &b, &mut o).unwrap();
        assert_eq!(o[0], 9.0);
        // sub-block 1 (scale 0, min 0 here): the high nibble 0 -> 0
        assert_eq!(o[32], 0.0);
    }
}
