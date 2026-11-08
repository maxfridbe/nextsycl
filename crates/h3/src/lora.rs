//! LoRAs: low-rank additions to the denoiser's linear layers, from a `.safetensors` in ComfyUI's layout for this model
//! (`diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight` [r, K] / `lora_B.weight` [N, r], optionally `.alpha`).
//!
//! Applied beside the int8 weights at run time - `out += strength * alpha / r * B (A x)` - rather than merged into
//! them: merging would mean expanding every touched matrix, adding, and quantizing again, which costs time and
//! precision; the side path costs two thin matrix products per touched layer.

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::host;
use crate::dtype::DType;
use crate::ops::Lora;
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

/// The four linears of a block, in the order the block runs them.
pub const LINEARS: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];

/// A block's LoRAs, per linear of `LINEARS` (several LoRAs stack: each adds its own side path).
pub type BlockLora = [Vec<Lora>; 4];

pub struct LoraSet {
    pub blocks: Vec<BlockLora>,
    pub refiner: Vec<BlockLora>,
    pub max_rank: usize,
    pub layers: usize,
}

fn bf16(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::BF16, shape, &crate::denoiser::bf16_bytes(v))
}

impl LoraSet {
    /// `blocks` / `refiner`: how many of each the model has.
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, strength: f32, blocks: usize, refiner: usize) -> Result<LoraSet> {
        let mut max_rank = 0;
        let mut layers = 0;
        let mut side = |prefix: &str, n: usize| -> Result<Vec<BlockLora>> {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut bl: BlockLora = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for (j, l) in LINEARS.iter().enumerate() {
                    let p = format!("diffusion_model.{prefix}.{i}.{l}");
                    let (ka, kb) = (format!("{p}.lora_A.weight"), format!("{p}.lora_B.weight"));
                    if !ck.entries.contains_key(&ka) {
                        continue;
                    }
                    let (a, mut b) = (host(ck, &ka)?, host(ck, &kb)?);
                    let (sa, sb) = (ck.get(&ka)?.shape.clone(), ck.get(&kb)?.shape.clone());
                    let r = sa[0];
                    if sb[1] != r {
                        return Err(Error(format!("{p}: lora_A rank {r}, lora_B {:?}", sb)));
                    }
                    let alpha = match ck.entries.contains_key(&format!("{p}.alpha")) {
                        true => host(ck, &format!("{p}.alpha"))?[0] / r as f32,
                        false => 1.0,
                    };
                    let scale = strength * alpha;
                    b.iter_mut().for_each(|v| *v *= scale);
                    bl[j].push(Lora { a: bf16(dev, &sa, &a)?, b: bf16(dev, &sb, &b)? });
                    max_rank = max_rank.max(r);
                    layers += 1;
                }
                out.push(bl);
            }
            Ok(out)
        };
        let blocks = side("blocks", blocks)?;
        let refiner = side("token_refiner.blocks", refiner)?;
        if layers == 0 {
            return Err(Error(format!("{}: no LoRA layers this model has", ck.path.display())));
        }
        Ok(LoraSet { blocks, refiner, max_rank, layers })
    }

    /// Another LoRA's layers on top of these.
    pub fn stack(&mut self, other: LoraSet) {
        for (mine, theirs) in self.blocks.iter_mut().zip(other.blocks).chain(self.refiner.iter_mut().zip(other.refiner)) {
            for (m, t) in mine.iter_mut().zip(theirs) {
                m.extend(t);
            }
        }
        self.max_rank = self.max_rank.max(other.max_rank);
        self.layers += other.layers;
    }
}
