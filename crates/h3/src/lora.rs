//! LoRAs: low-rank additions to the denoiser's linear layers, from a `.safetensors` in either layout for this model:
//!
//! - ComfyUI's (`diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight` [r, K] / `lora_B.weight` [N, r], optionally
//!   `.alpha`); a LoRA on the AdaLN projections (`blocks.N.adaln_proj.linear`, LMS Sharp's 50 curve branches: A
//!   [r, 8] on the timestep curve) is kept as each block's table delta `adaln` [rows, 8];
//! - Diffusers' (`transformer.transformer_blocks.N.attn.to_q.lora_A.weight` ...: Open-H3-Regenerate-2K-IC's IC LoRA,
//!   HyperFlow's adapter): q, k and v apart (each a LoRA of its own whose B is zero outside its third of qkv's
//!   outputs), the MLP's first matrix with its halves in the other order ([value; gate], ComfyUI's [gate; value]), the
//!   scale `lora_alpha / lora_rank` from the file's metadata (1 without). Time-embedder keys are HyperFlow's
//!   (hyperflow.rs) and left out here.
//!
//! Either layout may also come without its root (`blocks.N...`, `transformer_blocks.N...`: pi-H3's adapter files,
//! piflow.rs).
//!
//! Applied beside the int8 weights at run time - `out += strength * alpha / r * B (A x)` - rather than merged into
//! them: merging would mean expanding every touched matrix, adding, and quantizing again, which costs time and
//! precision; the side path costs two thin matrix products per touched layer. A set too large to sit on the card
//! (the IC LoRA is rank 384 on every linear) stays in host memory (`on_host`) and goes to the card a block at a time
//! (`LoraSet::device_block`).

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

/// A LoRA kept on the host: A [r, K] and B [N, r] as bfloat16 bytes (the scale folded into B)
pub struct HostLora {
    a: Vec<u8>,
    sa: [usize; 2],
    b: Vec<u8>,
    sb: [usize; 2],
}

impl HostLora {
    fn bytes(&self) -> usize {
        self.a.len() + self.b.len()
    }
}

/// A block's host LoRAs, per linear
pub type HostBlock = [Vec<HostLora>; 4];

pub struct LoraSet {
    pub blocks: Vec<BlockLora>,
    pub refiner: Vec<BlockLora>,
    pub max_rank: usize,
    pub layers: usize,
    /// per block: what the LoRAs add to its AdaLN projection on the timestep curve, [rows, t_dim] (LMS Sharp)
    pub adaln: Vec<Option<Vec<f32>>>,
    /// the blocks' LoRAs in host memory instead of `blocks` (which is then empty per block): see `device_block`
    pub host: Option<Vec<HostBlock>>,
}

fn bf16(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::BF16, shape, &crate::denoiser::bf16_bytes(v))
}

/// A LoRA's `.alpha`: a float tensor, or an integer one (the pDMD 2-step LoRA's ComfyUI conversion stores I64 scalars)
fn alpha_value(ck: &Checkpoint, name: &str) -> Result<f32> {
    let e = ck.get(name)?;
    match e.dtype {
        DType::I64 | DType::I32 => int_scalar(&ck.read(name)?, e.dtype),
        _ => Ok(host(ck, name)?[0]),
    }
}

/// The first value of little-endian I64 / I32 bytes, as a float
fn int_scalar(b: &[u8], dt: DType) -> Result<f32> {
    match (dt, b.len()) {
        (DType::I64, n) if n >= 8 => Ok(i64::from_le_bytes(b[..8].try_into().map_err(|_| Error("alpha".into()))?) as f32),
        (DType::I32, n) if n >= 4 => Ok(i32::from_le_bytes(b[..4].try_into().map_err(|_| Error("alpha".into()))?) as f32),
        _ => Err(Error(format!("a LoRA alpha of {} bytes as {dt:?}", b.len()))),
    }
}

/// One A / B pair before it goes anywhere: A [r, K], B [N, r] as float32, the scale already in B
struct Pair {
    a: Vec<f32>,
    sa: [usize; 2],
    b: Vec<f32>,
    sb: [usize; 2],
}

/// The pairs of one block (or refiner block) by linear, in either layout; `root`: what the file's keys begin with
/// (`diffusion_model.`, `transformer.`, or nothing)
fn block_pairs(ck: &Checkpoint, diffusers: bool, root: &str, prefix: &str, i: usize, scale: f32) -> Result<[Vec<Pair>; 4]> {
    let mut out: [Vec<Pair>; 4] = std::array::from_fn(|_| Vec::new());
    let pair = |p: &str| -> Result<Option<Pair>> {
        let (ka, kb) = (format!("{p}.lora_A.weight"), format!("{p}.lora_B.weight"));
        if !ck.entries.contains_key(&ka) {
            return Ok(None);
        }
        let (a, mut b) = (host(ck, &ka)?, host(ck, &kb)?);
        let (sa, sb) = (ck.get(&ka)?.shape.clone(), ck.get(&kb)?.shape.clone());
        if sa.len() != 2 || sb.len() != 2 || sb[1] != sa[0] {
            return Err(Error(format!("{p}: lora_A {sa:?}, lora_B {sb:?}")));
        }
        let alpha = match ck.entries.contains_key(&format!("{p}.alpha")) {
            true => alpha_value(ck, &format!("{p}.alpha"))? / sa[0] as f32,
            false => 1.0,
        };
        b.iter_mut().for_each(|v| *v *= scale * alpha);
        Ok(Some(Pair { a, sa: [sa[0], sa[1]], b, sb: [sb[0], sb[1]] }))
    };
    if !diffusers {
        for (j, l) in LINEARS.iter().enumerate() {
            if let Some(p) = pair(&format!("{root}{prefix}.{i}.{l}"))? {
                out[j].push(p);
            }
        }
        return Ok(out);
    }
    let p = format!("{root}{prefix}.{i}");
    // q, k, v: each its own LoRA on qkv's outputs, B zero outside its third
    for (part, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        if let Some(mut x) = pair(&format!("{p}.attn.{name}"))? {
            let (n, r) = (x.sb[0], x.sb[1]);
            let mut b = vec![0f32; 3 * n * r];
            b[part * n * r..(part + 1) * n * r].copy_from_slice(&x.b);
            x.b = b;
            x.sb = [3 * n, r];
            out[0].push(x);
        }
    }
    if let Some(x) = pair(&format!("{p}.attn.to_out.0"))? {
        out[1].push(x);
    }
    if let Some(mut x) = pair(&format!("{p}.ff.net.0.proj"))? {
        // Diffusers' [value; gate] rows to ComfyUI's [gate; value]
        let (n, r) = (x.sb[0], x.sb[1]);
        if !n.is_multiple_of(2) {
            return Err(Error(format!("{p}.ff.net.0.proj: {n} outputs do not halve")));
        }
        let half = n / 2 * r;
        let mut b = x.b[half..].to_vec();
        b.extend_from_slice(&x.b[..half]);
        x.b = b;
        out[2].push(x);
    }
    if let Some(x) = pair(&format!("{p}.ff.net.2"))? {
        out[3].push(x);
    }
    Ok(out)
}

impl LoraSet {
    /// A set with nothing in it, for `blocks` blocks and `refiner` refiner blocks
    pub fn empty(blocks: usize, refiner: usize) -> LoraSet {
        let none = || -> BlockLora { std::array::from_fn(|_| Vec::new()) };
        LoraSet { blocks: (0..blocks).map(|_| none()).collect(), refiner: (0..refiner).map(|_| none()).collect(), max_rank: 0, layers: 0,
                  adaln: vec![None; blocks], host: None }
    }

    /// `blocks` / `refiner`: how many of each the model has; the LoRAs on the card.
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, strength: f32, blocks: usize, refiner: usize) -> Result<LoraSet> {
        Self::load_to(dev, ck, strength, blocks, refiner, false)
    }

    /// As `load`; `on_host`: the blocks' LoRAs kept in host memory (`device_block` sends a block's when it runs).
    pub fn load_to(dev: &Arc<Device>, ck: &Checkpoint, strength: f32, blocks: usize, refiner: usize, on_host: bool) -> Result<LoraSet> {
        let diffusers = ck.entries.keys().any(|k| k.starts_with("transformer.") || k.starts_with("transformer_blocks."));
        let rooted = |r: &str| ck.entries.keys().any(|k| k.starts_with(r));
        let root = match diffusers {
            true if rooted("transformer.") => "transformer.",
            false if rooted("diffusion_model.") => "diffusion_model.",
            _ => "",
        };
        let meta_scale = match (ck.metadata.get("lora_alpha"), ck.metadata.get("lora_rank")) {
            (Some(a), Some(r)) => match (a.parse::<f32>(), r.parse::<f32>()) {
                (Ok(a), Ok(r)) if r > 0.0 => a / r,
                _ => return Err(Error(format!("{}: lora_alpha {a} / lora_rank {r}?", ck.path.display()))),
            },
            _ => 1.0,
        };
        let scale = strength * meta_scale;
        let (block_prefix, refiner_prefix) = if diffusers { ("transformer_blocks", "token_refiner.refiner_blocks") } else { ("blocks", "token_refiner.blocks") };
        let mut set = LoraSet::empty(blocks, refiner);
        let mut host_blocks: Vec<HostBlock> = Vec::new();
        for i in 0..blocks {
            let pairs = block_pairs(ck, diffusers, root, block_prefix, i, scale)?;
            let mut hb: HostBlock = std::array::from_fn(|_| Vec::new());
            for (j, ps) in pairs.into_iter().enumerate() {
                for p in ps {
                    set.max_rank = set.max_rank.max(p.sa[0]);
                    set.layers += 1;
                    if on_host {
                        hb[j].push(HostLora { a: crate::denoiser::bf16_bytes(&p.a), sa: p.sa, b: crate::denoiser::bf16_bytes(&p.b), sb: p.sb });
                    } else {
                        set.blocks[i][j].push(Lora { a: bf16(dev, &p.sa, &p.a)?, b: bf16(dev, &p.sb, &p.b)? });
                    }
                }
            }
            host_blocks.push(hb);
            // an AdaLN branch on the timestep curve (ComfyUI layout only: the curve is that form's)
            let p = format!("{root}blocks.{i}.adaln_proj.linear");
            let (ka, kb) = (format!("{p}.lora_A.weight"), format!("{p}.lora_B.weight"));
            if !diffusers && ck.entries.contains_key(&ka) {
                let (a, b) = (host(ck, &ka)?, host(ck, &kb)?);
                let (sa, sb) = (ck.get(&ka)?.shape.clone(), ck.get(&kb)?.shape.clone());
                let (r, td, rows) = (sa[0], sa[1], sb[0]);
                if sb[1] != r {
                    return Err(Error(format!("{p}: lora_A {sa:?}, lora_B {sb:?}")));
                }
                let alpha = match ck.entries.contains_key(&format!("{p}.alpha")) {
                    true => alpha_value(ck, &format!("{p}.alpha"))? / r as f32,
                    false => 1.0,
                };
                let mut d = vec![0f32; rows * td];
                for o in 0..rows {
                    for k in 0..r {
                        let bv = b[o * r + k] * scale * alpha;
                        if bv != 0.0 {
                            for c in 0..td {
                                d[o * td + c] += bv * a[k * td + c];
                            }
                        }
                    }
                }
                set.adaln[i] = Some(d);
                set.layers += 1;
            }
        }
        for i in 0..refiner {
            for (j, ps) in block_pairs(ck, diffusers, root, refiner_prefix, i, scale)?.into_iter().enumerate() {
                for p in ps {
                    set.max_rank = set.max_rank.max(p.sa[0]);
                    set.layers += 1;
                    set.refiner[i][j].push(Lora { a: bf16(dev, &p.sa, &p.a)?, b: bf16(dev, &p.sb, &p.b)? });
                }
            }
        }
        if on_host {
            set.host = Some(host_blocks);
        }
        if set.layers == 0 {
            return Err(Error(format!("{}: no LoRA layers this model has", ck.path.display())));
        }
        Ok(set)
    }

    /// Another LoRA's layers on top of these (both on the card, or both in host memory).
    pub fn stack(&mut self, other: LoraSet) {
        for (mine, theirs) in self.blocks.iter_mut().zip(other.blocks).chain(self.refiner.iter_mut().zip(other.refiner)) {
            for (m, t) in mine.iter_mut().zip(theirs) {
                m.extend(t);
            }
        }
        for (mine, theirs) in self.adaln.iter_mut().zip(other.adaln) {
            *mine = match (mine.take(), theirs) {
                (Some(mut a), Some(b)) if a.len() == b.len() => {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    Some(a)
                }
                (Some(a), None) => Some(a),
                (_, b) => b,
            };
        }
        match (self.host.as_mut(), other.host) {
            (Some(mine), Some(theirs)) => {
                for (m, t) in mine.iter_mut().zip(theirs) {
                    for (a, b) in m.iter_mut().zip(t) {
                        a.extend(b);
                    }
                }
            }
            (None, Some(theirs)) => self.host = Some(theirs),
            _ => {}
        }
        self.max_rank = self.max_rank.max(other.max_rank);
        self.layers += other.layers;
    }

    /// Whether the blocks' LoRAs are in host memory
    pub fn on_host(&self) -> bool {
        self.host.is_some()
    }

    /// Bytes of LoRA weights kept in host memory
    pub fn host_bytes(&self) -> usize {
        self.host.iter().flatten().flatten().flatten().map(HostLora::bytes).sum()
    }

    /// Block `i`'s LoRAs on the card: copied there now from host memory (dropped after the block has run, the copy
    /// waits for what the block queued), or None when they are on the card already (`blocks[i]`)
    pub fn device_block(&self, dev: &Arc<Device>, i: usize) -> Result<Option<BlockLora>> {
        let Some(h) = self.host.as_ref().and_then(|h| h.get(i)) else { return Ok(None) };
        let mut out: BlockLora = std::array::from_fn(|_| Vec::new());
        for (j, ls) in h.iter().enumerate() {
            for l in ls {
                out[j].push(Lora { a: Tensor::from_bytes(dev, DType::BF16, &l.sa, &l.a)?, b: Tensor::from_bytes(dev, DType::BF16, &l.sb, &l.b)? });
            }
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_alphas_read_as_their_value() {
        assert_eq!(int_scalar(&384i64.to_le_bytes(), DType::I64).unwrap(), 384.0);
        assert_eq!(int_scalar(&128i32.to_le_bytes(), DType::I32).unwrap(), 128.0);
        assert!(int_scalar(&[1, 2], DType::I64).is_err());
        assert!(int_scalar(&[0; 8], DType::F32).is_err());
    }

    /// A Diffusers-layout LoRA (q, k, v apart; the MLP's halves swapped) lands where ComfyUI's layout puts it
    #[test]
    fn diffusers_keys_map_to_the_fused_linears() {
        let dir = std::env::temp_dir().join(format!("ns-h3-lora-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("l.safetensors");
        let (k, n, r) = (4, 2, 1);
        let mut t = std::collections::BTreeMap::new();
        for (name, b) in [("to_q", 1.0), ("to_k", 2.0), ("to_v", 3.0)] {
            t.insert(format!("transformer.transformer_blocks.0.attn.{name}.lora_A.weight"), (vec![r, k], vec![1.0; r * k]));
            t.insert(format!("transformer.transformer_blocks.0.attn.{name}.lora_B.weight"), (vec![n, r], vec![b; n * r]));
        }
        t.insert("transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight".into(), (vec![r, k], vec![1.0; r * k]));
        t.insert("transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight".into(), (vec![4, r], vec![10.0, 11.0, 20.0, 21.0]));
        let meta = std::collections::BTreeMap::from([("lora_alpha".to_string(), "2".to_string()), ("lora_rank".to_string(), "1".to_string())]);
        crate::safetensors::write_f32(&p, &t, &meta).unwrap();
        let ck = Checkpoint::open(&p).unwrap();
        let pairs = block_pairs(&ck, true, "transformer.", "transformer_blocks", 0, 2.0).unwrap();
        assert_eq!(pairs[0].len(), 3);
        assert_eq!(pairs[0][1].sb, [6, 1]);
        assert_eq!(pairs[0][1].b, vec![0.0, 0.0, 4.0, 4.0, 0.0, 0.0]);
        // [value; gate] = [10, 11; 20, 21] -> [gate; value]
        assert_eq!(pairs[2][0].b, vec![40.0, 42.0, 20.0, 22.0]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
