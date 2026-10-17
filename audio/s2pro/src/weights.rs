//! The model's GGUF (s2.cpp's conversion, `general.architecture` fish-speech: the Dual-AR transformer and the codec,
//! `c.`-prefixed, in one file): its tensors found by name and put on the GPU - the transformers' matrices as the file
//! holds them (Q8_0 blocks split into codes and scales, or half), the codec's and the norms in float32. The token and
//! codebook embeddings stay in the file: the rows a prompt or a frame needs are read from it.

use std::path::Path;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_gguf::safetensors::f16_to_f32;
use nextsycl_gguf::{GType, Gguf, Tensor, Value};

use crate::ops::{Lin, Mat, Ops, Store};

pub struct Weights {
    pub g: Gguf,
}

/// One block type's values (host): `b` holds whole blocks of `t`
pub fn dequant(t: GType, b: &[u8], out: &mut [f32]) -> Result<()> {
    match t {
        GType::F32 => out.iter_mut().zip(b.chunks_exact(4)).for_each(|(o, c)| *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        GType::F16 => out.iter_mut().zip(b.chunks_exact(2)).for_each(|(o, c)| *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        GType::Q8_0 => {
            for (o, blk) in out.chunks_exact_mut(32).zip(b.chunks_exact(34)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for (v, q) in o.iter_mut().zip(&blk[2..]) {
                    *v = d * *q as i8 as f32;
                }
            }
        }
        t => return Err(Error(format!("tensor type {} is not one this engine reads (F32, F16, Q8_0)", t.name()))),
    }
    Ok(())
}

/// Q8_0 blocks split into their codes and their scales' bits (the GPU's layout: 8 aligned codes a lane)
pub fn split_q8(b: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let n = b.len() / 34;
    let (mut q, mut s) = (Vec::with_capacity(n * 32), Vec::with_capacity(n * 2));
    for blk in b.chunks_exact(34) {
        s.extend_from_slice(&blk[..2]);
        q.extend_from_slice(&blk[2..]);
    }
    (q, s)
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        let g = Gguf::open(path).map_err(|e| Error(e.0))?;
        if g.architecture() != crate::ARCH_GGUF {
            return Err(Error(format!("{}: architecture {}, not {} (s2.cpp's S2 Pro GGUF)", path.display(), g.architecture(), crate::ARCH_GGUF)));
        }
        Ok(Weights { g })
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.g.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.g.paths[0].display())))
    }

    pub fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }

    /// A metadata number (`fish_speech.<key>`, else `fish-speech.<key>`: the converter writes both namespaces)
    pub fn num(&self, key: &str) -> Result<f64> {
        self.g.meta(&format!("fish_speech.{key}")).or_else(|| self.g.meta(&format!("fish-speech.{key}"))).and_then(Value::as_f64)
            .ok_or_else(|| Error(format!("{}: no fish_speech.{key}", self.g.paths[0].display())))
    }

    pub fn flag(&self, key: &str) -> bool {
        matches!(self.g.meta(&format!("fish_speech.{key}")), Some(Value::Bool(true)))
    }

    /// Bytes a row of a 2-D tensor takes, and its values
    fn row_layout(&self, t: &Tensor) -> Result<(usize, usize)> {
        let k = if t.shape.len() > 1 { t.shape[1..].iter().product::<u64>() as usize } else { 1 };
        let (per, bytes) = t.ty.block();
        if k % per != 0 {
            return Err(Error(format!("{}: rows of {k} values, not whole {} blocks", t.name, t.ty.name())));
        }
        Ok((k / per * bytes, k))
    }

    /// Rows [r0, r1) of a tensor (a 1-D one: rows of one value), their bytes as stored
    pub fn raw_rows(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<u8>> {
        let t = self.tensor(name)?;
        let (rb, _) = self.row_layout(t)?;
        if r1 as u64 > t.shape[0] || r0 > r1 {
            return Err(Error(format!("{name}: rows {r0}..{r1} of {}", t.shape[0])));
        }
        let mut b = vec![0u8; (r1 - r0) * rb];
        self.g.read_into(t, (r0 * rb) as u64, &mut b).map_err(|e| Error(e.0))?;
        Ok(b)
    }

    /// Rows [r0, r1) as float32 on the host (a large read split over threads)
    pub fn rows(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let (rb, k) = self.row_layout(t)?;
        let b = self.raw_rows(name, r0, r1)?;
        let mut out = vec![0f32; (r1 - r0) * k];
        if r1 - r0 < 64 {
            dequant(t.ty, &b, &mut out)?;
            return Ok(out);
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
        let per_t = (r1 - r0).div_ceil(threads).max(1);
        std::thread::scope(|s| -> Result<()> {
            let jobs: Vec<_> = out.chunks_mut(per_t * k).zip(b.chunks(per_t * rb)).map(|(o, bb)| s.spawn(move || dequant(t.ty, bb, o))).collect();
            for j in jobs {
                j.join().map_err(|_| Error("a dequantizing thread failed".into()))??;
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// A whole tensor as float32 on the host
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let n = self.tensor(name)?.shape[0] as usize;
        self.rows(name, 0, n)
    }

    /// A tensor as float32 on the GPU
    pub fn dev(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(&ops.gpu, &self.f32(name)?)
    }

    /// The shape of a tensor, outermost first (torch's order)
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensor(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    /// Rows of matrices [n_i, k] stacked (a fused gate / up; a head of chosen rows: `rows` ranges a matrix, None:
    /// all) on the GPU as the file stores them: Q8_0 as codes and scales, F16 as half, anything else expanded to half
    pub fn mat_rows(&self, ops: &Ops, parts: &[(&str, Option<std::ops::Range<usize>>)]) -> Result<Mat> {
        let mut k = 0;
        let mut ty = None;
        let mut spans = Vec::new();
        for (name, r) in parts {
            let t = self.tensor(name)?;
            if t.shape.len() != 2 || (k != 0 && t.shape[1] as usize != k) {
                return Err(Error(format!("{name}: shape {:?}, a matrix {k} wide expected", t.shape)));
            }
            k = t.shape[1] as usize;
            let r = r.clone().unwrap_or(0..t.shape[0] as usize);
            if ty.is_some_and(|x| x != t.ty) {
                ty = Some(GType::F32);
            } else if ty.is_none() {
                ty = Some(t.ty);
            }
            spans.push((*name, r));
        }
        let n: usize = spans.iter().map(|(_, r)| r.len()).sum();
        let gpu = &ops.gpu;
        let w = match ty {
            Some(GType::Q8_0) => {
                let q = DevBuf::new(gpu, n * k)?;
                let s = DevBuf::new(gpu, n * k / 32 * 2)?;
                let mut row = 0;
                for (name, r) in &spans {
                    // in pieces of up to 64 MB of codes
                    let step = ((64usize << 20) / k).max(1);
                    let mut r0 = r.start;
                    while r0 < r.end {
                        let r1 = (r0 + step).min(r.end);
                        let (qb, sb) = split_q8(&self.raw_rows(name, r0, r1)?);
                        q.write(row * k, &qb)?;
                        s.write(row * k / 32 * 2, &sb)?;
                        row += r1 - r0;
                        r0 = r1;
                    }
                }
                Store::Q8 { q, s }
            }
            Some(GType::F16) => {
                let w = DevBuf::new(gpu, n * k * 2)?;
                let mut row = 0;
                for (name, r) in &spans {
                    w.write(row * k * 2, &self.raw_rows(name, r.start, r.end)?)?;
                    row += r.len();
                }
                Store::Half(w)
            }
            _ => {
                let w = DevBuf::new(gpu, n * k * 2)?;
                let mut row = 0;
                for (name, r) in &spans {
                    let step = ((16usize << 20) / k).max(1);
                    let stage = DevBuf::f32(gpu, step.min(r.len()).max(1) * k)?;
                    let mut r0 = r.start;
                    while r0 < r.end {
                        let r1 = (r0 + step).min(r.end);
                        let v = self.rows(name, r0, r1)?;
                        stage.write(0, &crate::bytes(&v))?;
                        ops.to_half(stage.fp(), w.ptr().wrapping_byte_add(row * k * 2), v.len())?;
                        gpu.sync()?;
                        row += r1 - r0;
                        r0 = r1;
                    }
                }
                Store::Half(w)
            }
        };
        gpu.sync()?;
        Ok(Mat { w, n, k })
    }

    /// Whole matrices stacked by rows
    pub fn mat(&self, ops: &Ops, names: &[&str]) -> Result<Mat> {
        self.mat_rows(ops, &names.iter().map(|n| (*n, None)).collect::<Vec<_>>())
    }

    /// A float32 linear layer of the codec (`name.weight` [n, k], `name.bias` when `bias`)
    pub fn lin(&self, ops: &Ops, name: &str, bias: bool) -> Result<Lin> {
        let s = self.shape(&format!("{name}.weight"))?;
        Ok(Lin {
            w: self.dev(ops, &format!("{name}.weight"))?,
            b: if bias { Some(self.dev(ops, &format!("{name}.bias"))?) } else { None },
            n: s[0],
            k: s[1..].iter().product(),
        })
    }

    /// Linear layers stacked by rows, float32 (the codec's w1 / w3)
    pub fn lin_stack(&self, ops: &Ops, names: &[String]) -> Result<Lin> {
        let mut v = Vec::new();
        let mut k = 0;
        for n in names {
            k = self.shape(n)?[1];
            v.extend(self.f32(n)?);
        }
        Ok(Lin { n: v.len() / k, k, w: DevBuf::from_f32(&ops.gpu, &v)?, b: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_blocks_expand_and_split_as_stored() {
        // d = 0.5 (half 0x3800), codes -2, 3, then zeros
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        b[2] = (-2i8) as u8;
        b[3] = 3;
        let mut o = vec![0f32; 32];
        dequant(GType::Q8_0, &b, &mut o).unwrap();
        assert_eq!(&o[..3], &[-1.0, 1.5, 0.0]);
        let two = [b.clone(), b].concat();
        let (q, s) = split_q8(&two);
        assert_eq!((q.len(), s.len()), (64, 4));
        assert_eq!((q[0] as i8, q[32 + 1], &s[2..4]), (-2, 3, &0x3800u16.to_le_bytes()[..]));
    }
}
