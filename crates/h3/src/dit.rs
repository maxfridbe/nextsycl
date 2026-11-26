//! The denoiser's block stack: 50 identical blocks over one sequence of tokens (text, conditioning, audio and video
//! tokens together), each block = attention over all tokens, then a two-layer MLP, both added back into the stream.
//!
//! One block, on a stream `x` [S, hidden]:
//!
//! ```text
//!   h   = norm(x) * (1 + scale_a[row]) + shift_a[row]           rows: which timestep/modality table row a token uses
//!   qkv = linear(h)                                             int8
//!   q,k = per-head norm + position rotation (in place in qkv)
//!   a   = attention(q, k, v)
//!   x  += linear(a) * gate_a[row]                               int8
//!   h   = norm(x) * (1 + scale_m[row]) + shift_m[row]
//!   x  += linear(silu(u) * w) * gate_m[row]   with [u | w] = linear(h)      int8, int8
//! ```
//!
//! The six tables come from a small linear on the timestep embedding, computed on the host once per step.
//!
//! The four block matrices come in two forms: int8 (ComfyUI's `int8_convrot` safetensors, int8 GEMMs) or a
//! llama.cpp k-quant (the GGUF denoisers, Q4_K / Q6_K: expanded to bfloat16 one matrix at a time, then a 16-bit
//! GEMM). The k-quants hold less of the card (Q4_K 10.6 GiB, Q6_K 15.4, int8 19.5), which leaves the room for
//! longer sequences; the int8 form computes faster.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dtype::{bytes_to_f32, DType};
use crate::gguf::GType;
use crate::ops::{self, Int8Linear, KLinear, Mod, Rows};
use crate::safetensors::Checkpoint;
use crate::{load, Ctx, Error, Result};

/// The sizes of the denoiser, read from a checkpoint's shapes.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// The MLP's inner width (its first linear writes twice this).
    pub ffn: usize,
    /// Features of q and k that carry position (rotated); the rest are only normalized.
    pub rot_dim: usize,
    /// Table rows per distinct timestep: one per modality (video, text, audio).
    pub modalities: usize,
    /// Width of the timestep embedding the tables are made from.
    pub t_dim: usize,
    pub norm_eps: f32,
    pub qk_eps: f32,
    pub blocks: usize,
}

impl Config {
    pub fn from_checkpoint(ck: &Checkpoint) -> Result<Config> {
        let hidden = ck.get("blocks.0.norm1.weight")?.shape[0];
        let head_dim = ck.get("blocks.0.attn.q_norm.weight")?.shape[0];
        let heads = ck.get("blocks.0.attn.out_proj.weight")?.shape[1] / head_dim;
        let ffn = ck.get("blocks.0.mlp.fc2.weight")?.shape[1];
        let adaln = &ck.get("blocks.0.adaln_proj.linear.weight")?.shape;
        let inv_freq = ck.get("rope.inv_freq")?.shape[0];
        let blocks = ck
            .entries
            .keys()
            .filter_map(|k| k.strip_prefix("blocks.")?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |m| m + 1);
        Ok(Config {
            hidden,
            heads,
            head_dim,
            ffn,
            rot_dim: inv_freq * 3 * 2, // three axes (time, height, width), a pair of features per frequency
            modalities: adaln[0] / (6 * hidden),
            t_dim: adaln[1],
            norm_eps: 1e-5,
            qk_eps: 1e-5,
            blocks,
        })
    }
}

/// A block matrix in either of the checkpoint forms, or an int8 one left in the file (`Blocks::load_streamed`)
pub enum Weight {
    I8(Int8Linear),
    K(KLinear),
    Streamed(StreamedLinear),
}

impl Weight {
    fn size(&self) -> (usize, usize) {
        match self {
            Weight::I8(l) => (l.outputs(), l.inputs()),
            Weight::K(l) => (l.n, l.k),
            Weight::Streamed(l) => (l.n, l.k),
        }
    }
}

/// An int8 block matrix read from the checkpoint each time its block runs: where its weights and scales are in the file
#[derive(Clone, Copy, Debug)]
pub struct StreamedLinear {
    w_offset: u64,
    w_bytes: usize,
    s_offset: u64,
    s_bytes: usize,
    pub n: usize,
    pub k: usize,
    group: Option<usize>,
}

/// The card's copy of one streamed block at a time: its four int8 matrices and their scales, and which block it holds
struct Staging {
    lins: [Int8Linear; 4],
    file: std::path::PathBuf,
    host: std::sync::Mutex<Vec<u8>>,
    current: std::sync::atomic::AtomicUsize,
}

/// One block's weights, on the device (the big ones) and on the host (the small table projection).
pub struct Block {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    qkv: Weight,
    out_proj: Weight,
    fc1: Weight,
    fc2: Weight,
    /// [6 * hidden * modalities, t_dim] and its bias
    adaln_w: Vec<f32>,
    adaln_b: Vec<f32>,
}

const LINEARS: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];

pub fn small(dev: &Arc<Device>, ck: &Checkpoint, name: &str) -> Result<Tensor> {
    let e = ck.get(name)?;
    let v = bytes_to_f32(&ck.read(name)?, e.dtype)?;
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    Tensor::from_bytes(dev, DType::F32, &[v.len()], &bytes)
}

pub fn host(ck: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    bytes_to_f32(&ck.read(name)?, ck.get(name)?.dtype)
}

impl Block {
    /// `big`: the int8 weights and their scales, already on the device (see `Blocks::load`); `streamed`: none of them
    /// are, the block's matrices are read from the file when it runs.
    fn assemble(dev: &Arc<Device>, ck: &Checkpoint, prefix: &str, index: usize, big: &mut BTreeMap<String, Tensor>, streamed: bool) -> Result<Block> {
        let p = format!("{prefix}.{index}");
        let mut linear = |l: &str| -> Result<Weight> {
            let name = format!("{p}.{l}");
            if streamed {
                let q = ck.quant(&name)?.ok_or_else(|| Error(format!("{name}: only int8 checkpoints stream their blocks")))?;
                let (w, sc) = (ck.get(&format!("{name}.weight"))?, ck.get(&format!("{name}.weight_scale"))?);
                if w.dtype != DType::I8 || sc.dtype != DType::F32 || w.kquant.is_some() {
                    return Err(Error(format!("{name}: int8 weights with float32 scales expected to stream")));
                }
                return Ok(Weight::Streamed(StreamedLinear { w_offset: w.offset, w_bytes: w.bytes, s_offset: sc.offset, s_bytes: sc.bytes, n: w.shape[0],
                                                            k: w.shape[1], group: q.convrot.then_some(q.group) }));
            }
            let mut take = |k: String| big.remove(&k).ok_or_else(|| Error(format!("{k} was not loaded")));
            let wn = format!("{name}.weight");
            if let Some(kind) = ck.kquant_of(&wn) {
                let shape = ck.get(&wn)?.shape.clone();
                return Ok(Weight::K(KLinear { blocks: take(wn)?, kind, n: shape[0], k: shape[1] }));
            }
            let q = ck.quant(&name)?.ok_or_else(|| Error(format!("{name} has no quantization record: neither an int8 nor a k-quant checkpoint")))?;
            Ok(Weight::I8(Int8Linear { weight: take(wn)?, scale: take(format!("{name}.weight_scale"))?, bias: None, group: q.convrot.then_some(q.group) }))
        };
        Ok(Block {
            qkv: linear(LINEARS[0])?,
            out_proj: linear(LINEARS[1])?,
            fc1: linear(LINEARS[2])?,
            fc2: linear(LINEARS[3])?,
            norm1: small(dev, ck, &format!("{p}.norm1.weight"))?,
            norm2: small(dev, ck, &format!("{p}.norm2.weight"))?,
            q_norm: small(dev, ck, &format!("{p}.attn.q_norm.weight"))?,
            k_norm: small(dev, ck, &format!("{p}.attn.k_norm.weight"))?,
            adaln_w: host(ck, &format!("{p}.adaln_proj.linear.weight"))?,
            adaln_b: host(ck, &format!("{p}.adaln_proj.linear.bias"))?,
        })
    }

    /// The per-head key norm's weight (float32 [head_dim])
    pub fn k_norm(&self) -> &Tensor {
        &self.k_norm
    }

    /// An adapter's AdaLN projection on the timestep curve in place of the block's ([rows, t_dim], the same shape), and its
    /// bias where it has one (pi-H3's adapters tune every block's projection, the bias of their four fully tuned blocks)
    pub fn set_adaln(&mut self, w: Vec<f32>, b: Option<Vec<f32>>) -> Result<()> {
        if w.len() != self.adaln_w.len() || b.as_ref().is_some_and(|b| b.len() != self.adaln_b.len()) {
            return Err(Error(format!("an AdaLN projection of {} values (bias {:?}) for the block's {} ({})", w.len(), b.as_ref().map(Vec::len),
                                     self.adaln_w.len(), self.adaln_b.len())));
        }
        self.adaln_w = w;
        if let Some(b) = b {
            self.adaln_b = b;
        }
        Ok(())
    }

    /// An adapter's matrix in place of one of the block's four (`linears` order: qkv, out, fc1, fc2), the same size;
    /// a block left in the file (`Blocks::load_streamed`) takes none
    pub fn set_linear(&mut self, j: usize, w: Weight) -> Result<()> {
        let slot = match j {
            0 => &mut self.qkv,
            1 => &mut self.out_proj,
            2 => &mut self.fc1,
            3 => &mut self.fc2,
            _ => return Err(Error(format!("a block has four matrices, not {}", j + 1))),
        };
        if matches!(slot, Weight::Streamed(_)) {
            return Err(Error("a block streamed from the file takes no replacement matrix".into()));
        }
        if slot.size() != w.size() {
            return Err(Error(format!("a [{}, {}] matrix in place of a [{}, {}] one", w.size().0, w.size().1, slot.size().0, slot.size().1)));
        }
        *slot = w;
        Ok(())
    }

    /// The four block matrices, in product order: qkv, out, fc1, fc2
    pub fn linears(&self) -> [&Weight; 4] {
        [&self.qkv, &self.out_proj, &self.fc1, &self.fc2]
    }

    /// The block's six tables for the timestep embeddings `t_emb` [R, t_dim]: shift, scale, gate for the attention
    /// half, then for the MLP half; each [R * modalities, hidden], row = timestep index * modalities + modality.
    pub fn tables(&self, cfg: &Config, t_emb: &[f32]) -> [Vec<f32>; 6] {
        self.tables_with(cfg, t_emb, None)
    }

    /// As `tables`, the projection plus `extra` [rows, t_dim] (a LoRA's AdaLN branch on the curve: LMS Sharp's)
    pub fn tables_with(&self, cfg: &Config, t_emb: &[f32], extra: Option<&[f32]>) -> [Vec<f32>; 6] {
        let (c, m, td) = (cfg.hidden, cfg.modalities, cfg.t_dim);
        let merged: Option<Vec<f32>> = extra.filter(|e| e.len() == self.adaln_w.len()).map(|e| self.adaln_w.iter().zip(e).map(|(a, b)| a + b).collect());
        let adaln_w = merged.as_deref().unwrap_or(&self.adaln_w);
        let r = t_emb.len() / td;
        let mut out: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0f32; r * m * c]);
        for ri in 0..r {
            let t = &t_emb[ri * td..(ri + 1) * td];
            for mi in 0..m {
                for (ci, table) in out.iter_mut().enumerate() {
                    let o0 = mi * 6 * c + ci * c;
                    let row = &mut table[(ri * m + mi) * c..(ri * m + mi + 1) * c];
                    for (i, v) in row.iter_mut().enumerate() {
                        let o = o0 + i;
                        let w = &adaln_w[o * td..(o + 1) * td];
                        *v = self.adaln_b[o] + w.iter().zip(t).map(|(a, b)| a * b).sum::<f32>();
                    }
                }
            }
        }
        out
    }
}

/// What a step fixes for every block: each token's table row, the position rotations, and every block's tables.
pub struct Step {
    pub tokens: usize,
    /// int32 [S]
    rows: Tensor,
    /// float32 [S, rot_dim / 2, 2]: (cos, sin)
    cs: Tensor,
    /// per block: shift_a, scale_a, gate_a, shift_m, scale_m, gate_m
    tables: Vec<[Tensor; 6]>,
}

/// (cos, sin) of every token's rotation angles: positions [S, 3] (time, height, width) times the model's frequencies,
/// the three axes one after the other.
pub fn rotations(positions: &[f64], inv_freq: &[f32]) -> Vec<f32> {
    let s = positions.len() / 3;
    let mut cs = Vec::with_capacity(s * 3 * inv_freq.len() * 2);
    for p in positions.chunks_exact(3) {
        for axis in p {
            for f in inv_freq {
                let a = *axis as f32 * f;
                cs.push(a.cos());
                cs.push(a.sin());
            }
        }
    }
    cs
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

impl Step {
    pub fn new(dev: &Arc<Device>, model: &Blocks, rows: &[i32], positions: &[f64], inv_freq: &[f32], t_emb: &[f32]) -> Result<Step> {
        Self::new_with(dev, model, rows, positions, inv_freq, t_emb, None)
    }

    /// As `new`; `adaln`: per block what LoRAs add to its AdaLN projection on the curve (`LoraSet::adaln`)
    pub fn new_with(dev: &Arc<Device>, model: &Blocks, rows: &[i32], positions: &[f64], inv_freq: &[f32], t_emb: &[f32], adaln: Option<&[Option<Vec<f32>>]>)
                    -> Result<Step> {
        let cfg = &model.cfg;
        if !t_emb.len().is_multiple_of(cfg.t_dim) {
            return Err(Error("step: the timestep embedding does not fit the model".into()));
        }
        let tables: Vec<[Vec<f32>; 6]> = model.blocks.iter().enumerate()
            .map(|(i, b)| b.tables_with(cfg, t_emb, adaln.and_then(|a| a.get(i)).and_then(|x| x.as_deref()))).collect();
        Self::from_tables(dev, model, rows, positions, inv_freq, t_emb.len() / cfg.t_dim, tables)
    }

    /// A step from tables made elsewhere (HyperFlow's two-time embedding, hyperflow.rs): per block the six tables, each
    /// [values * modalities, hidden]
    pub fn from_tables(dev: &Arc<Device>, model: &Blocks, rows: &[i32], positions: &[f64], inv_freq: &[f32], values: usize, host_tables: Vec<[Vec<f32>; 6]>)
                       -> Result<Step> {
        let cfg = &model.cfg;
        let tokens = rows.len();
        if positions.len() != tokens * 3 || inv_freq.len() * 6 != cfg.rot_dim || host_tables.len() != model.blocks.len() {
            return Err(Error("step: positions, frequencies or tables do not fit the model".into()));
        }
        let r = values * cfg.modalities;
        if let Some(bad) = rows.iter().find(|x| **x < 0 || **x as usize >= r) {
            return Err(Error(format!("step: table row {bad} of {r}")));
        }
        let cs = rotations(positions, inv_freq);
        let row_bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut tables = Vec::with_capacity(model.blocks.len());
        for t in &host_tables {
            let mut dev_t = Vec::with_capacity(6);
            for v in t {
                if v.len() != r * cfg.hidden {
                    return Err(Error(format!("step: a table of {} values for {r} rows", v.len())));
                }
                dev_t.push(Tensor::from_bytes(dev, DType::F32, &[r, cfg.hidden], &f32_bytes(v))?);
            }
            tables.push(dev_t.try_into().map_err(|_| Error("six tables per block".into()))?);
        }
        Ok(Step {
            tokens,
            rows: Tensor::from_bytes(dev, DType::I32, &[tokens], &row_bytes)?,
            cs: Tensor::from_bytes(dev, DType::F32, &[tokens, cfg.rot_dim / 2, 2], &f32_bytes(&cs))?,
            tables,
        })
    }
}

/// The buffers a block writes between its kernels; one set serves every block.
pub struct Scratch {
    pub(crate) h: Tensor,
    pub(crate) qkv: Tensor,
    pub(crate) att: Tensor,
    pub(crate) proj: Tensor,
    pub(crate) fc1: Tensor,
    pub(crate) act: Tensor,
    /// a LoRA's middle product [tokens, rank]
    pub(crate) lora: Tensor,
    /// the MLP in row chunks (`Scratch::chunked`): the chunk's input and output rows [rows, hidden]; fc1 and act are
    /// then [rows, ...] too
    mlp_chunk: Option<(Tensor, Tensor)>,
}

/// The widest LoRA the scratch has room for.
pub const MAX_LORA_RANK: usize = 384;

impl Scratch {
    pub fn new(dev: &Arc<Device>, cfg: &Config, tokens: usize, dtype: DType) -> Result<Scratch> {
        Self::chunked(dev, cfg, tokens, dtype, tokens)
    }

    /// The MLP half `mlp_rows` rows at a time (its two widest buffers that many rows instead of every token's: a long
    /// sequence's memory); `tokens` or more: all at once, as `new`
    pub fn chunked(dev: &Arc<Device>, cfg: &Config, tokens: usize, dtype: DType, mlp_rows: usize) -> Result<Scratch> {
        let w = cfg.heads * cfg.head_dim;
        let mr = mlp_rows.clamp(1, tokens.max(1));
        let chunk = match mr < tokens {
            true => Some((Tensor::new(dev, dtype, &[mr, cfg.hidden])?, Tensor::new(dev, dtype, &[mr, cfg.hidden])?)),
            false => None,
        };
        Ok(Scratch {
            h: Tensor::new(dev, dtype, &[tokens, cfg.hidden])?,
            qkv: Tensor::new(dev, dtype, &[tokens, 3 * w])?,
            att: Tensor::new(dev, dtype, &[tokens, w])?,
            proj: Tensor::new(dev, dtype, &[tokens, cfg.hidden])?,
            fc1: Tensor::new(dev, dtype, &[mr, 2 * cfg.ffn])?,
            act: Tensor::new(dev, dtype, &[mr, cfg.ffn])?,
            lora: Tensor::new(dev, dtype, &[tokens, MAX_LORA_RANK])?,
            mlp_chunk: chunk,
        })
    }

    /// Bytes a 16-bit scratch of `tokens` with the MLP `mlp_rows` at a time holds
    pub fn bytes(cfg: &Config, tokens: usize, mlp_rows: usize) -> usize {
        let w = cfg.heads * cfg.head_dim;
        let mr = mlp_rows.min(tokens);
        let chunk = if mr < tokens { 2 * mr * cfg.hidden } else { 0 };
        2 * (tokens * (2 * cfg.hidden + 4 * w + MAX_LORA_RANK) + mr * 3 * cfg.ffn + chunk)
    }
}

/// A LoRA's addition to linear `j`'s output, when the block has one there.
pub(crate) fn side(lora: Option<&crate::lora::BlockLora>, j: usize, x: &Tensor, out: &Tensor, tmp: &Tensor) -> Result<()> {
    for l in lora.map(|l| l[j].as_slice()).unwrap_or(&[]) {
        l.apply(x, tmp, out)?;
    }
    Ok(())
}

/// Called after each stage of a block with the stage's name and its result: how a run is checked against a reference.
pub type Tap<'a> = &'a mut dyn FnMut(&str, &Tensor) -> Result<()>;

/// Another engine's attention for the blocks (a silo's: q, k, v rows of the qkv buffer -> out), used where it returns
/// true; false (or none set) leaves a call to the shared kernels
pub type AttentionFn = Box<dyn Fn(Rows, Rows, Rows, &Tensor) -> Result<bool> + Send + Sync>;

/// An attention that is not one over every row of the sequence (MPEG-Diffusion's two tracks, mpeg.rs): told the qkv
/// buffer before q and k are normalized and rotated, then asked for the attention in place of the one over all rows
pub trait SplitAttention {
    /// qkv [S, 3 H D] as the qkv linear wrote it (k_norm: this block's per-head key norm)
    fn before_rope(&self, index: usize, qkv: &Tensor, k_norm: &Tensor) -> Result<()>;
    /// q and k rotated in qkv; write every row's attention output into att [S, H D]
    fn attend(&self, index: usize, qkv: &Tensor, att: &Tensor) -> Result<()>;
}

/// Another engine's whole product with a k-quant block matrix (the matrix, x, out, the stack's 16-bit expansion buffer
/// for its own use): true when it did it; false leaves the expansion and the product to the rest
pub type ProductFn = Box<dyn Fn(&KLinear, &Tensor, &Tensor, &Tensor) -> Result<bool> + Send + Sync>;

/// The block stack.
pub struct Blocks {
    pub cfg: Config,
    pub blocks: Vec<Block>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// the k-quant form's (`None` for int8): which, and the bfloat16 buffer each matrix is expanded into
    pub kquant: Option<GType>,
    wbuf: Option<Tensor>,
    /// the attention the blocks call, when an engine brings its own (P-H3-Image's silo); else the shared kernels'
    pub attention: Option<AttentionFn>,
    /// the k-quant products, when an engine brings its own (MiniMax-H3-Image's silo); else the shared expansion and
    /// product
    pub product: Option<ProductFn>,
    /// the streamed blocks' card copy (`load_streamed`): one block's matrices at a time
    staging: Option<Staging>,
    /// rows gathered by index from a strided buffer, when an engine brings the kernel (the H3 video silo's, for
    /// MPEG-Diffusion's two-track attention); else one copy a row
    pub gather: Option<crate::mpeg::GatherFn>,
}

impl Blocks {
    /// Loads the first `count` blocks (all of them when `None`): the int8 weights with `threads` parallel readers.
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, count: Option<usize>, threads: usize) -> Result<Blocks> {
        let mut cfg = Config::from_checkpoint(ck)?;
        cfg.blocks = count.unwrap_or(cfg.blocks).min(cfg.blocks);
        Self::load_from(dev, ck, "blocks", cfg, threads)
    }

    /// As `load`, the first `resident` blocks on the card and the rest left in the int8 checkpoint: each streamed block's
    /// matrices are read and sent to one card copy when it runs (about 0.4 GB a block; the page cache keeps the file
    /// hot when the host has the room). For sequences whose activations need the card: a 2560x1440 refine.
    pub fn load_streamed(dev: &Arc<Device>, ck: &Checkpoint, threads: usize, resident: usize) -> Result<Blocks> {
        let cfg = Config::from_checkpoint(ck)?;
        Self::load_part(dev, ck, "blocks", cfg, threads, resident.min(cfg.blocks))
    }

    /// `cfg.blocks` blocks under `prefix` (`blocks`; a ControlNet's `control_blocks`), shaped as `cfg` says
    pub fn load_from(dev: &Arc<Device>, ck: &Checkpoint, prefix: &str, cfg: Config, threads: usize) -> Result<Blocks> {
        Self::load_part(dev, ck, prefix, cfg, threads, cfg.blocks)
    }

    fn load_part(dev: &Arc<Device>, ck: &Checkpoint, prefix: &str, cfg: Config, threads: usize, resident: usize) -> Result<Blocks> {
        let n = cfg.blocks;
        let lead = format!("{prefix}.");
        let wanted = |name: &str| -> bool {
            let Some(rest) = name.strip_prefix(lead.as_str()) else { return false };
            let Some((idx, tail)) = rest.split_once('.') else { return false };
            idx.parse::<usize>().is_ok_and(|i| i < resident) && LINEARS.iter().any(|l| tail == format!("{l}.weight") || tail == format!("{l}.weight_scale"))
        };
        let mut loaded = load::load(dev, ck, threads, wanted).ctx("loading the denoiser's weights")?;
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            blocks.push(Block::assemble(dev, ck, prefix, i, &mut loaded.tensors, i >= resident).ctx(format!("block {i}"))?);
        }
        let staging = match blocks.iter().find(|b| matches!(b.qkv, Weight::Streamed(_))) {
            Some(b) => {
                let lin = |w: &Weight| -> Result<Int8Linear> {
                    let Weight::Streamed(l) = w else { return Err(Error("a streamed block with a resident matrix".into())) };
                    Ok(Int8Linear { weight: Tensor::new(dev, DType::I8, &[l.n, l.k])?, scale: Tensor::new(dev, DType::F32, &[l.s_bytes / 4])?, bias: None,
                                    group: l.group })
                };
                let lins = [lin(&b.qkv)?, lin(&b.out_proj)?, lin(&b.fc1)?, lin(&b.fc2)?];
                Some(Staging { lins, file: ck.path.clone(), host: std::sync::Mutex::new(Vec::new()), current: std::sync::atomic::AtomicUsize::new(usize::MAX) })
            }
            None => None,
        };
        let kquant = ck.kquant_of(&format!("{prefix}.0.mlp.fc1.weight"));
        let wbuf = match kquant {
            Some(_) => {
                let most = blocks.iter().flat_map(|b| [&b.qkv, &b.out_proj, &b.fc1, &b.fc2]).map(|w| { let (n, k) = w.size(); n * k }).max().unwrap_or(0);
                Some(Tensor::new(dev, DType::BF16, &[most]).ctx("the buffer the k-quant matrices expand into")?)
            }
            None => None,
        };
        Ok(Blocks { cfg, blocks, load_seconds: loaded.seconds, load_bytes: loaded.bytes, kquant, wbuf, attention: None, product: None, staging, gather: None })
    }

    /// The card the blocks are on
    pub fn device(&self) -> Option<Arc<Device>> {
        self.blocks.first().map(|b| b.norm1.buf.device().clone())
    }

    /// How many blocks stay on the card (the rest stream from the file)
    pub fn resident(&self) -> usize {
        self.blocks.iter().filter(|b| !matches!(b.qkv, Weight::Streamed(_))).count()
    }

    /// Block `index`'s matrices on the card copy, when it is a streamed block not already there: read from the file
    /// and sent (the write waits for the work queued before it, which may still read the copy)
    fn stage(&self, index: usize) -> Result<()> {
        let Some(st) = &self.staging else { return Ok(()) };
        let b = &self.blocks[index];
        if !matches!(b.qkv, Weight::Streamed(_)) || st.current.load(std::sync::atomic::Ordering::Relaxed) == index {
            return Ok(());
        }
        use std::os::unix::fs::FileExt;
        let f = std::fs::File::open(&st.file).ctx(format!("opening {}", st.file.display()))?;
        let mut host = st.host.lock().map_err(|_| Error("the staging buffer's lock".into()))?;
        for (w, lin) in b.linears().iter().zip(&st.lins) {
            let Weight::Streamed(l) = w else { return Err(Error(format!("block {index}: a streamed block with a resident matrix"))) };
            for (off, n, dst) in [(l.w_offset, l.w_bytes, &lin.weight), (l.s_offset, l.s_bytes, &lin.scale)] {
                if n != dst.buf.len() {
                    return Err(Error(format!("block {index}: {n} bytes for a {}-byte card copy", dst.buf.len())));
                }
                host.resize(n, 0);
                f.read_exact_at(&mut host[..n], off).ctx("reading a streamed block")?;
                dst.buf.write(0, &host[..n])?;
            }
        }
        st.current.store(index, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// `out = linear(x)` with one of the blocks' matrices.
    fn mm(&self, w: &Weight, x: &Tensor, out: &Tensor) -> Result<()> {
        match w {
            Weight::I8(l) => l.forward(x, out),
            Weight::Streamed(l) => {
                let st = self.staging.as_ref().ok_or("a streamed matrix without its card copy")?;
                let c = st.lins.iter().find(|c| c.outputs() == l.n && c.inputs() == l.k).ok_or("no card copy of that shape")?;
                c.forward(x, out)
            }
            Weight::K(l) => {
                let wbuf = self.wbuf.as_ref().ok_or("a k-quant matrix without its expansion buffer")?;
                if let Some(f) = &self.product {
                    if f(l, x, out, wbuf)? {
                        return Ok(());
                    }
                }
                l.forward(x, out, wbuf)
            }
        }
    }

    /// The expansion buffer of a k-quant stack (None for int8): what an engine's checks expand into
    pub fn expansion_buffer(&self) -> Option<&Tensor> {
        self.wbuf.as_ref()
    }

    /// Runs block `index` on the stream `x` [S, hidden], in place.
    /// `lora`: this block's LoRAs, if any.
    pub fn block(&self, index: usize, x: &Tensor, step: &Step, s: &Scratch, lora: Option<&crate::lora::BlockLora>, tap: Option<Tap>) -> Result<()> {
        self.block_split(index, x, step, s, lora, tap, None)
    }

    /// `block`, with an attention of its own (`SplitAttention`) in place of the one over every row
    #[allow(clippy::too_many_arguments)]
    pub fn block_split(&self, index: usize, x: &Tensor, step: &Step, s: &Scratch, lora: Option<&crate::lora::BlockLora>, mut tap: Option<Tap>,
                       split: Option<&dyn SplitAttention>) -> Result<()> {
        let cfg = &self.cfg;
        let b = &self.blocks[index];
        self.stage(index)?;
        let [shift_a, scale_a, gate_a, shift_m, scale_m, gate_m] = &step.tables[index];
        let mut tap = |name: &str, t: &Tensor| -> Result<()> {
            match tap.as_mut() {
                Some(f) => f(name, t),
                None => Ok(()),
            }
        };
        // attention half
        ops::rms_norm_mod(x, &b.norm1, cfg.norm_eps, Some(&Mod { rows: &step.rows, scale: scale_a, shift: shift_a }), &s.h)?;
        tap("h1", &s.h)?;
        self.mm(&b.qkv, &s.h, &s.qkv)?;
        side(lora, 0, &s.h, &s.qkv, &s.lora)?;
        tap("qkv", &s.qkv)?;
        let w = cfg.heads * cfg.head_dim;
        let part = |i: usize| Rows { t: &s.qkv, offset: i * w, stride: 3 * w, tokens: step.tokens, heads: cfg.heads, dim: cfg.head_dim };
        if let Some(sp) = split {
            sp.before_rope(index, &s.qkv, &b.k_norm)?;
        }
        ops::rms_rope(part(0), &b.q_norm, cfg.qk_eps, &step.cs, cfg.rot_dim)?;
        ops::rms_rope(part(1), &b.k_norm, cfg.qk_eps, &step.cs, cfg.rot_dim)?;
        tap("qkv_rotated", &s.qkv)?;
        if let Some(sp) = split {
            sp.attend(index, &s.qkv, &s.att)?;
        } else if !self.attention.as_ref().map_or(Ok(false), |f| f(part(0), part(1), part(2), &s.att))? {
            ops::attention(part(0), part(1), part(2), &s.att)?;
        }
        tap("att", &s.att)?;
        self.mm(&b.out_proj, &s.att, &s.proj)?;
        side(lora, 1, &s.att, &s.proj, &s.lora)?;
        tap("attn_out", &s.proj)?;
        ops::gate_add(x, &s.proj, &step.rows, gate_a)?;
        tap("x1", x)?;
        // MLP half
        ops::rms_norm_mod(x, &b.norm2, cfg.norm_eps, Some(&Mod { rows: &step.rows, scale: scale_m, shift: shift_m }), &s.h)?;
        tap("h2", &s.h)?;
        match &s.mlp_chunk {
            None => {
                self.mm(&b.fc1, &s.h, &s.fc1)?;
                side(lora, 2, &s.h, &s.fc1, &s.lora)?;
                tap("fc1", &s.fc1)?;
                ops::swiglu(&s.fc1, &s.act)?;
                self.mm(&b.fc2, &s.act, &s.proj)?;
                side(lora, 3, &s.act, &s.proj, &s.lora)?;
            }
            Some((hc, pc)) => {
                // a chunk of rows through the MLP (the last chunk's rows past the sequence are stale; never copied back)
                let mr = hc.shape[0];
                let mut r0 = 0;
                while r0 < step.tokens {
                    let m = mr.min(step.tokens - r0);
                    hc.copy_rows(0, &s.h, r0, m)?;
                    self.mm(&b.fc1, hc, &s.fc1)?;
                    side(lora, 2, hc, &s.fc1, &s.lora)?;
                    ops::swiglu(&s.fc1, &s.act)?;
                    self.mm(&b.fc2, &s.act, pc)?;
                    side(lora, 3, &s.act, pc, &s.lora)?;
                    s.proj.copy_rows(r0, pc, 0, m)?;
                    r0 += m;
                }
            }
        }
        tap("mlp", &s.proj)?;
        ops::gate_add(x, &s.proj, &step.rows, gate_m)?;
        tap("x2", x)
    }
}
