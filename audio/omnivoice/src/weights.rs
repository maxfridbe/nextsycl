//! The two GGUFs (omnivoice.cpp's conversion: `omnivoice-base-*.gguf`, general.architecture omnivoice-lm - the Qwen3
//! backbone, the audio embeddings and heads, the text tokenizer; `omnivoice-tokenizer-*.gguf`, omnivoice-tokenizer -
//! the Higgs Audio v2 codec: HuBERT, the semantic encoder, DAC's encoder and decoder, the quantizer) read by our own
//! reader, their tensors found by name and put on the GPU in the type the engine runs them in: the backbone's matrices
//! half (any of the file's types expanded once at load: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K), the rest float32.

use std::path::Path;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::Dt;
use nextsycl_gguf::safetensors::f16_to_f32;
use nextsycl_gguf::{GType, Gguf, Tensor, Value};

use crate::ops::{bytes, read_f32, Ops};

pub struct Weights {
    pub g: Gguf,
}

/// The GGUF types the shared kernels expand (nsd_dequant)
fn expanded(t: GType) -> bool {
    matches!(t, GType::Q8_0 | GType::Q4K | GType::Q5K | GType::Q6K | GType::BF16)
}

impl Weights {
    pub fn open(path: &Path, arch: &str) -> Result<Weights> {
        let g = Gguf::open(path).map_err(|e| Error(e.0))?;
        if g.architecture() != arch {
            return Err(Error(format!("{}: architecture {}, not {arch} (omnivoice.cpp's OmniVoice GGUFs)", path.display(), g.architecture())));
        }
        Ok(Weights { g })
    }

    fn name(&self) -> String {
        self.g.paths[0].display().to_string()
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.g.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.name())))
    }

    pub fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensor(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    /// A metadata value by its full key
    pub fn meta(&self, key: &str) -> Result<&Value> {
        self.g.meta(key).ok_or_else(|| Error(format!("{}: no {key}", self.name())))
    }

    pub fn num(&self, key: &str) -> Result<f64> {
        self.meta(key)?.as_f64().ok_or_else(|| Error(format!("{}: {key} is not a number", self.name())))
    }

    pub fn nums(&self, key: &str) -> Result<Vec<usize>> {
        self.meta(key)?.as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
            .ok_or_else(|| Error(format!("{}: {key} is not a list", self.name())))
    }

    /// Bytes a row of a tensor takes (rows: its outermost dimension), and its values
    fn row_layout(&self, t: &Tensor) -> Result<(usize, usize)> {
        let k = if t.shape.len() > 1 { t.shape[1..].iter().product::<u64>() as usize } else { 1 };
        let (per, b) = t.ty.block();
        if k % per != 0 {
            return Err(Error(format!("{}: rows of {k} values, not whole {} blocks", t.name, t.ty.name())));
        }
        Ok((k / per * b, k))
    }

    /// Rows [r0, r1) of a tensor, their bytes as stored
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

    /// Rows [r0, r1) as float32 on the host (a K-quant expanded by the shared kernels on the GPU)
    pub fn rows(&self, ops: &Ops, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let (_, k) = self.row_layout(t)?;
        let raw = self.raw_rows(name, r0, r1)?;
        let n = (r1 - r0) * k;
        match t.ty {
            GType::F32 => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            GType::F16 => Ok(raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            GType::BF16 => Ok(raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()),
            GType::Q8_0 => Ok(raw.chunks_exact(34).flat_map(|b| {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                b[2..].iter().map(move |q| d * *q as i8 as f32)
            }).collect()),
            t if expanded(t) => {
                let src = DevBuf::new(&ops.gpu, raw.len())?;
                src.write(0, &raw)?;
                let out = DevBuf::f32(&ops.gpu, n)?;
                ops.nsd.dequant(src.ptr(), t.code(), n, out.ptr(), Dt::F32)?;
                ops.gpu.sync()?;
                read_f32(&out, 0, n)
            }
            t => Err(Error(format!("{name}: type {} is not one this engine reads", t.name()))),
        }
    }

    /// A whole tensor as float32 on the host
    pub fn f32(&self, ops: &Ops, name: &str) -> Result<Vec<f32>> {
        let n = self.tensor(name)?.shape[0] as usize;
        self.rows(ops, name, 0, n)
    }

    /// A tensor as float32 on the GPU
    pub fn dev(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        let t = self.tensor(name)?;
        if expanded(t.ty) {
            let raw = self.raw_rows(name, 0, t.shape[0] as usize)?;
            let src = DevBuf::new(&ops.gpu, raw.len())?;
            src.write(0, &raw)?;
            let n = t.elements() as usize;
            let out = DevBuf::f32(&ops.gpu, n)?;
            ops.nsd.dequant(src.ptr(), t.ty.code(), n, out.ptr(), Dt::F32)?;
            ops.gpu.sync()?;
            return Ok(out);
        }
        DevBuf::from_f32(&ops.gpu, &self.f32(ops, name)?)
    }

    /// Matrices [n_i, k] stacked by rows as one half matrix on the GPU
    pub fn half(&self, ops: &Ops, names: &[String]) -> Result<(DevBuf, usize, usize)> {
        let mut k = 0;
        let mut n = 0;
        for name in names {
            let s = self.shape(name)?;
            if s.len() != 2 || (k != 0 && s[1] != k) {
                return Err(Error(format!("{name}: shape {s:?}, a matrix {k} wide expected")));
            }
            k = s[1];
            n += s[0];
        }
        let out = DevBuf::new(&ops.gpu, n * k * 2)?;
        let mut row = 0;
        for name in names {
            let t = self.tensor(name)?;
            let rows = t.shape[0] as usize;
            let raw = self.raw_rows(name, 0, rows)?;
            let dst = out.ptr().wrapping_byte_add(row * k * 2);
            match t.ty {
                GType::F16 => out.write(row * k * 2, &raw)?,
                GType::F32 => {
                    let src = DevBuf::new(&ops.gpu, raw.len())?;
                    src.write(0, &raw)?;
                    ops.to_half(src.fp(), dst, rows * k)?;
                    ops.gpu.sync()?;
                }
                ty if expanded(ty) => {
                    let src = DevBuf::new(&ops.gpu, raw.len())?;
                    src.write(0, &raw)?;
                    ops.nsd.dequant(src.ptr(), ty.code(), rows * k, dst, Dt::F16)?;
                    ops.gpu.sync()?;
                }
                ty => return Err(Error(format!("{name}: type {} is not one this engine reads", ty.name()))),
            }
            row += rows;
        }
        Ok((out, n, k))
    }

    /// A float32 matrix of the host's values on the GPU (a product made at load)
    pub fn upload(ops: &Ops, v: &[f32]) -> Result<DevBuf> {
        let b = DevBuf::new(&ops.gpu, v.len() * 4)?;
        b.write(0, &bytes(v))?;
        Ok(b)
    }
}
