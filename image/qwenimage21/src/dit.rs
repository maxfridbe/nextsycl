//! The denoiser: Qwen-Image 2.1's single-stream transformer (diffusers' QwenImage21Transformer2DModel), 32 blocks of
//! 4,096 features (32 heads x 128), its weights from the GGUF (Q8_0 and BF16) dequantized to half on the GPU.
//!
//! The text and the image share one sequence, but attention is block-causal: the text is causal and never sees the
//! image, and text tokens are modulated from t = 0 (`causal_condition`) - so the text's keys and values in every
//! layer depend on the prompt alone. They are computed once (`prefix`) and every step runs only the image's tokens,
//! each attending to [the text's keys | the image's own]. One block, for the rows it runs:
//!
//! ```text
//!   h = layernorm(x) * (1 + scale1)          q k v = h [Wq|Wk|Wv]; q, k: per-head rmsnorm + 3-axis RoPE
//!   x += tanh(gate1) * attention(q, k, v) Wo
//!   h = layernorm(x) * (1 + scale2)
//!   x += tanh(gate2) * (silu(h Wgate) * (h Wproj)) Wout
//! ```
//!
//! scale / gate: one modulation shared by every block, from the timestep (the image's rows) or t = 0 (the text's).
//! The RoPE pairs features (2i, 2i + 1); the shared kernel rotates (i, 64 + i), so q's and k's output rows (and their
//! norms) are permuted at load - attention scores do not change, v is untouched.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_diffusion::lora::{self, Delta};
use nextsycl_gguf::{GType, Gguf};

use crate::{ffi, prof};

pub const DIM: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD: usize = 128;
pub const FFN: usize = 12288;
pub const LATENT: usize = 64;
const EPS: f32 = 1e-6;
/// RoPE: the frame, height and width axes' features (pairs: 8, 28, 28)
const AXES: [usize; 3] = [16, 56, 56];

/// The ConvRot group: int8 matrices are rotated by the normalized 256 x 256 Hadamard matrix along their inputs
const GROUP: usize = 256;

/// A block's matrix [n, k]: half, or int8 (ConvRot, a scale per row) - the activations then quantized per row on
/// the fly (`NS_QI_INT8=1`: the card's int8 rate, twice its half one, at some cost in accuracy)
struct Mat {
    w: DevBuf,
    scale: Option<DevBuf>,
    n: usize,
    /// what a LoRA names in it: its parts' module keys (stacked by rows: q | k | v, gate | up), their rows, whether
    /// their rows are RoPE-permuted (q, k), and its inputs
    keys: Vec<String>,
    rows: Vec<usize>,
    perm: Vec<bool>,
    k: usize,
}

/// A request's LoRAs beside one matrix: out += B (A x) - A [R, k] every update's A stacked, B [n, R] each update's B
/// in its part's rows (zero elsewhere), the strength folded in; half for the block matrices, float32 for the small ones
pub struct SideMat {
    a: DevBuf,
    b: DevBuf,
    r: usize,
}

/// A request's LoRAs on a side path (Dit::side): applied after each matrix they name, dropped when the request ends -
/// the weights stay as loaded (no reload, no merge)
pub struct Side {
    /// per block: qkv, out, gate|up, down
    blocks: Vec<[Option<SideMat>; 4]>,
    /// the small linears by module key
    small: std::collections::BTreeMap<String, SideMat>,
    /// scratch for A x: [rows, R], grown as needed
    tmp: std::sync::Mutex<Option<DevBuf>>,
    /// its device memory, bytes
    pub bytes: usize,
}

impl Mat {
    /// out [m, n] = x [m, k] (half) . W^T
    fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, k: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        match &self.scale {
            None => nsd.linear(x, Dt::F16, m, k, self.w.ptr(), self.n, none(), out, out_dt),
            Some(s) => nsd.int8_linear(x, Dt::F16, m, k, self.w.ptr(), self.n, s.ptr(), self.n, none(), out, out_dt, GROUP),
        }
    }
}

/// One block's weights
struct Block {
    qkv: Mat,
    out: Mat,
    /// gate | proj rows (SwiGLU reads them as one product)
    gp: Mat,
    down: Mat,
    norm_q: DevBuf,
    norm_k: DevBuf,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    /// the block matrices are int8
    pub int8: bool,
    blocks: Vec<Block>,
    // the small linears, float32
    img_in: DevBuf,
    txt_norm: DevBuf,
    txt_in1: DevBuf,
    txt_in2: DevBuf,
    t_lin1: DevBuf,
    t_lin2: DevBuf,
    modulation: DevBuf,
    norm_out: DevBuf,
    proj_out: DevBuf,
    /// the small linears a LoRA may name: module key, rows, inputs
    smalls: Vec<(String, usize, usize)>,
    /// a request's LoRAs on the side path, while it runs
    side: std::sync::Mutex<Option<Side>>,
}

/// The text's state every step reads: each layer's q | k | v rows of the text (k with its RoPE), half
pub struct Prefix {
    /// rows: the text's, and the condition pictures' latents
    pub tokens: usize,
    kv: Vec<DevBuf>,
    /// the frame position the target picture takes (after the text and the pictures)
    pub frame: i64,
}

/// q / k row order: pair (2i, 2i + 1) -> (i, 64 + i)
fn rope_perm(j: usize) -> usize {
    if j < HEAD / 2 {
        2 * j
    } else {
        2 * (j - HEAD / 2) + 1
    }
}

/// The updates beside one matrix [n, k] as one pair: A [R, k] (each update's A stacked) and B [n, R] (each update's
/// B, times its strength, in its part's rows - reordered as RoPE-permuted rows are - and its own R columns, zero
/// elsewhere), so B (A x) = the sum of their updates - what a merge adds to W. `p`: (first row, permuted, update,
/// strength) each
fn stacked(p: &[(usize, bool, &Delta, f32)], n: usize, k: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let r: usize = p.iter().map(|x| x.2.r).sum();
    let mut a = vec![0f32; r * k];
    let mut bm = vec![0f32; n * r];
    let mut ro = 0;
    for (off, perm, d, s) in p {
        a[ro * k..(ro + d.r) * k].copy_from_slice(&d.a);
        for row in 0..d.n {
            let src = if *perm { (row / HEAD) * HEAD + rope_perm(row % HEAD) } else { row };
            for (o, v) in bm[(off + row) * r + ro..(off + row) * r + ro + d.r].iter_mut().zip(&d.b[src * d.r..(src + 1) * d.r]) {
                *o = v * s;
            }
        }
        ro += d.r;
    }
    (a, bm, r)
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

impl Dit {
    /// The weights from `f`; `int8`: the block matrices as int8 ConvRot (else half)
    /// `loras`: updates merged into the block matrices at load (one that matches no block matrix is an error)
    pub fn load(f: &Gguf, nsd: &Nsd, int8: bool, loras: &[Delta], log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        let tensor = |name: &str| f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())));
        // a matrix's rows as stored, optionally reordered (`perm`: new row r reads old row perm(r) within each
        // 128-row head), dequantized to `dt` on the GPU
        let load_rows = |names: &[&str], perm: &[bool], dt: Dt| -> Result<DevBuf> {
            let mut raw = Vec::new();
            let mut n = 0usize;
            let mut ty = None;
            for (name, p) in names.iter().zip(perm) {
                let t = tensor(name)?;
                if ty.is_some_and(|x| x != t.ty) {
                    return Err(Error(format!("{name}: {} beside {:?}", t.ty.name(), ty)));
                }
                ty = Some(t.ty);
                let bytes = f.read(t).map_err(ge)?;
                if *p {
                    let rows = t.shape[0] as usize;
                    let rb = bytes.len() / rows;
                    for r in 0..rows {
                        let (h, j) = (r / HEAD, r % HEAD);
                        let src = h * HEAD + rope_perm(j);
                        raw.extend_from_slice(&bytes[src * rb..(src + 1) * rb]);
                    }
                } else {
                    raw.extend_from_slice(&bytes);
                }
                n += t.elements() as usize;
            }
            let ty = ty.ok_or_else(|| Error("no tensors".into()))?;
            // Q8_0, Q4_K, Q6_K, BF16: what the dequant kernel reads
            let code = ty.code();
            if ![8, 12, 14, 30].contains(&code) {
                return Err(Error(format!("{}: {} weights are not read yet", names[0], ty.name())));
            }
            let src = DevBuf::new(&gpu, raw.len())?;
            src.write(0, &raw)?;
            let out = DevBuf::new(&gpu, n * if dt == Dt::F32 { 4 } else { 2 })?;
            nsd.dequant(src.ptr(), code, n, out.ptr(), dt)?;
            nsd.wait()?;
            Ok(out)
        };
        let k8 = ffi::api()?;
        let had = DevBuf::from_f32(&gpu, &nextsycl_qwen3vl::hadamard(GROUP))?;
        // a block matrix: half, or rotated along its inputs (W . H, a GEMM on the matrix engine) and quantized per row
        // the LoRA updates by module key, and which modules took one
        let by_key: std::collections::BTreeMap<String, Vec<&Delta>> = loras.iter().fold(Default::default(), |mut m, d| {
            m.entry(lora::key(&d.target)).or_default().push(d);
            m
        });
        let merged = std::cell::RefCell::new(std::collections::BTreeSet::new());
        // a block matrix (one or more stored matrices stacked by rows): its LoRA updates merged in float32 on the GPU
        // (W += B . A; B's rows reordered as the matrix's are), then half, or rotated along its inputs (W . H, a GEMM
        // on the matrix engine) and quantized per row
        // the LoRA updates of a float32 matrix [rows stacked, k] in place: W += B . A per part (B's rows reordered as
        // the part's are)
        let merge = |w: &DevBuf, keys: &[String], rows: &[usize], perm: &[bool], k: usize| -> Result<()> {
            let mut off = 0;
            for (i, key) in keys.iter().enumerate() {
                for d in by_key.get(key).into_iter().flatten() {
                    if d.n != rows[i] || d.k != k {
                        return Err(Error(format!("LoRA {}: {}x{}, the matrix is {}x{k}", d.target, d.n, d.k, rows[i])));
                    }
                    let b: Vec<f32> = if perm[i] {
                        (0..d.n).flat_map(|row| {
                            let src = (row / HEAD) * HEAD + rope_perm(row % HEAD);
                            d.b[src * d.r..(src + 1) * d.r].iter().copied()
                        }).collect()
                    } else {
                        d.b.clone()
                    };
                    let at: Vec<f32> = (0..d.k).flat_map(|c| (0..d.r).map(move |j| (c, j))).map(|(c, j)| d.a[j * d.k + c]).collect();
                    let (bd, ad) = (DevBuf::from_f32(&gpu, &b)?, DevBuf::from_f32(&gpu, &at)?);
                    nsd.linear_acc(bd.ptr(), Dt::F32, d.n, d.r, ad.ptr(), k, w.fp().wrapping_add(off * k).cast())?;
                    nsd.wait()?;
                    merged.borrow_mut().insert(key.clone());
                }
                off += rows[i];
            }
            Ok(())
        };
        // `parts`: the matrices a LoRA names, with their rows, when they differ from the stored ones (a fused gate_up
        // holds gate_layer's rows, then proj's)
        let big = |names: &[&str], perm: &[bool], parts: Option<&[(&str, usize)]>| -> Result<Mat> {
            let (keys, rows, perm): (Vec<String>, Vec<usize>, Vec<bool>) = match parts {
                Some(p) => (p.iter().map(|(nm, _)| lora::key(nm)).collect(), p.iter().map(|(_, r)| *r).collect(), vec![false; p.len()]),
                None => (names.iter().map(|nm| lora::key(nm.strip_suffix(".weight").unwrap_or(nm))).collect(),
                         names.iter().map(|nm| tensor(nm).map(|t| t.shape[0] as usize)).collect::<Result<_>>()?, perm.to_vec()),
            };
            let n: usize = rows.iter().sum();
            let has_lora = keys.iter().any(|k| by_key.contains_key(k));
            let stored_perm: Vec<bool> = if parts.is_some() { vec![false; names.len()] } else { perm.clone() };
            let inputs = |w: &DevBuf, bytes: usize| w.len / bytes / n;
            if !int8 && !has_lora {
                let w = load_rows(names, &stored_perm, Dt::F16)?;
                let k = inputs(&w, 2);
                return Ok(Mat { w, scale: None, n, keys, rows, perm, k });
            }
            let w = load_rows(names, &stored_perm, Dt::F32)?;
            let k = w.floats() / n;
            merge(&w, &keys, &rows, &perm, k)?;
            if !int8 {
                let h = DevBuf::new(&gpu, n * k * 2)?;
                // SAFETY: n x k values each way.
                ffi::check(unsafe { (k8.to_half)(gpu.raw(), w.fp(), h.ptr(), (n * k) as i64) }, "to half")?;
                nsd.wait()?;
                return Ok(Mat { w: h, scale: None, n, keys, rows, perm, k });
            }
            let rot = DevBuf::f32(&gpu, n * k)?;
            nsd.linear(w.ptr(), Dt::F32, n * k / GROUP, GROUP, had.ptr(), GROUP, none(), rot.ptr(), Dt::F32)?;
            let q = DevBuf::new(&gpu, n * k)?;
            let scale = DevBuf::f32(&gpu, n)?;
            // SAFETY: rot holds n x k floats, q n x k bytes, scale n floats.
            ffi::check(unsafe { (k8.quant_rows)(gpu.raw(), rot.fp(), n as i64, k as i64, q.ptr().cast(), scale.fp()) }, "int8 weights")?;
            nsd.wait()?;
            Ok(Mat { w: q, scale: Some(scale), n, keys, rows, perm, k })
        };
        // a small linear, float32, its LoRA updates merged (a few-step LoRA adapts the timestep and modulation ones)
        let smalls = std::cell::RefCell::new(Vec::new());
        let single = |name: &str| -> Result<DevBuf> {
            let w = load_rows(&[name], &[false], Dt::F32)?;
            let rows = tensor(name)?.shape[0] as usize;
            let key = lora::key(name.strip_suffix(".weight").unwrap_or(name));
            merge(&w, std::slice::from_ref(&key), &[rows], &[false], w.floats() / rows)?;
            smalls.borrow_mut().push((key, rows, w.floats() / rows));
            Ok(w)
        };
        // a vector (norm weights) as float32 on the host, whatever its stored type (BF16 in the base file, F32 in
        // Viggle's turbo files)
        let host_vec = |name: &str| -> Result<Vec<f32>> {
            let t = tensor(name)?;
            let b = f.read(t).map_err(ge)?;
            Ok(match t.ty {
                GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
                GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                GType::F16 => b.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                other => return Err(Error(format!("{name}: a {} vector", other.name()))),
            })
        };
        // ... on the GPU, permuted like q / k when asked
        let vector = |name: &str, permute: bool| -> Result<DevBuf> {
            let v = host_vec(name)?;
            let v = if permute { (0..v.len()).map(|j| v[rope_perm(j)]).collect() } else { v };
            DevBuf::from_f32(&gpu, &v)
        };
        let n_blocks = (0..).take_while(|i| f.tensor(&format!("transformer_blocks.{i}.attn.to_q.weight")).is_some()).count();
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let p = format!("transformer_blocks.{i}.");
            blocks.push(Block {
                qkv: big(&[&format!("{p}attn.to_q.weight"), &format!("{p}attn.to_k.weight"), &format!("{p}attn.to_v.weight")], &[true, true, false], None)?,
                out: big(&[&format!("{p}attn.to_out.0.weight")], &[false], None)?,
                // the MLP's gate and up rows: two matrices, or one fused (`gate_up`, as Viggle's turbo files store them)
                gp: if f.tensor(&format!("{p}img_mlp.gate_up.weight")).is_some() {
                    let (g, u) = (format!("{p}img_mlp.gate_layer"), format!("{p}img_mlp.proj"));
                    big(&[&format!("{p}img_mlp.gate_up.weight")], &[false], Some(&[(g.as_str(), FFN), (u.as_str(), FFN)]))?
                } else {
                    big(&[&format!("{p}img_mlp.gate_layer.weight"), &format!("{p}img_mlp.proj.weight")], &[false, false], None)?
                },
                down: big(&[&format!("{p}img_mlp.out.weight")], &[false], None)?,
                norm_q: vector(&format!("{p}attn.norm_q.weight"), true)?,
                norm_k: vector(&format!("{p}attn.norm_k.weight"), true)?,
            });
        }
        // the text norm stores (scale - 1): the kernel multiplies by its weight
        let tn = {
            let v: Vec<f32> = host_vec("txt_in.text_norm.weight")?.iter().map(|x| x + 1.0).collect();
            DevBuf::from_f32(&gpu, &v)?
        };
        let dit = Dit {
            gpu: gpu.clone(),
            int8,
            blocks,
            img_in: single("img_in.weight")?,
            txt_norm: tn,
            txt_in1: single("txt_in.in_layer.weight")?,
            txt_in2: single("txt_in.out_layer.weight")?,
            t_lin1: single("time_text_embed.timestep_embedder.linear_1.weight")?,
            t_lin2: single("time_text_embed.timestep_embedder.linear_2.weight")?,
            modulation: single("modulation.1.weight")?,
            norm_out: single("norm_out.linear.weight")?,
            proj_out: single("proj_out.weight")?,
            smalls: smalls.borrow().clone(),
            side: std::sync::Mutex::new(None),
        };
        let merged = merged.borrow().clone();
        let missed: std::collections::BTreeSet<&str> = loras.iter().filter(|d| !merged.contains(&lora::key(&d.target))).map(|d| d.target.as_str()).collect();
        if !missed.is_empty() {
            let few: Vec<&str> = missed.iter().take(4).copied().collect();
            return Err(Error(format!("LoRA modules this engine does not merge ({} of them): {}", missed.len(), few.join(", "))));
        }
        if !loras.is_empty() {
            log(format!("qwen-image 2.1: {} LoRA updates merged into {} matrices", loras.len(), merged.len()));
        }
        log(format!("qwen-image 2.1: {n_blocks} blocks on {} ({}) in {:.1} s", gpu.name, if int8 { "int8 ConvRot" } else { "half" }, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    /// A request's LoRA updates as a side path (`deltas`: each with its strength, by which its B is multiplied): every
    /// update beside the matrix it names (one that names none is an error, as at load); its memory checked first (1.5 GiB
    /// kept free)
    pub fn side(&self, nsd: &Nsd, deltas: &[(&Delta, f32)]) -> Result<Side> {
        let gpu = &self.gpu;
        let by_key: std::collections::BTreeMap<String, Vec<(&Delta, f32)>> = deltas.iter().fold(Default::default(), |mut m, d| {
            m.entry(lora::key(&d.0.target)).or_default().push(*d);
            m
        });
        // per matrix: the updates in it, (first row, RoPE-permuted, update, strength)
        type Plan<'a> = Vec<(usize, bool, &'a Delta, f32)>;
        let plan = |keys: &[String], rows: &[usize], perm: &[bool], k: usize| -> Result<Plan<'_>> {
            let mut v = Vec::new();
            let mut off = 0;
            for (i, key) in keys.iter().enumerate() {
                for (d, s) in by_key.get(key).into_iter().flatten() {
                    if d.n != rows[i] || d.k != k {
                        return Err(Error(format!("LoRA {}: {}x{}, the matrix is {}x{k}", d.target, d.n, d.k, rows[i])));
                    }
                    v.push((off, perm[i], *d, *s));
                }
                off += rows[i];
            }
            Ok(v)
        };
        let size = |p: &Plan, n: usize, k: usize, eb: usize| { let r: usize = p.iter().map(|x| x.2.r).sum(); (r * k + n * r) * eb };
        let mut plans = Vec::with_capacity(self.blocks.len());
        let mut bytes = 0usize;
        // the largest float32 staging of one matrix (its half copy is made from it)
        let mut stage = 0usize;
        let mut used: std::collections::BTreeSet<String> = Default::default();
        for b in &self.blocks {
            let mut four: Vec<Plan> = Vec::with_capacity(4);
            for m in [&b.qkv, &b.out, &b.gp, &b.down] {
                let p = plan(&m.keys, &m.rows, &m.perm, m.k)?;
                bytes += size(&p, m.n, m.k, 2);
                stage = stage.max(size(&p, m.n, m.k, 4));
                used.extend(p.iter().map(|x| lora::key(&x.2.target)));
                four.push(p);
            }
            plans.push(four);
        }
        let mut small_plans = Vec::new();
        for (key, n, k) in &self.smalls {
            let p = plan(std::slice::from_ref(key), &[*n], &[false], *k)?;
            bytes += size(&p, *n, *k, 4);
            used.extend(p.iter().map(|x| lora::key(&x.2.target)));
            small_plans.push((key.clone(), *n, *k, p));
        }
        let missed: Vec<&str> = deltas.iter().filter(|d| !used.contains(&lora::key(&d.0.target))).map(|d| d.0.target.as_str()).collect();
        if !missed.is_empty() {
            return Err(Error(format!("LoRA modules this engine does not take ({} of them): {}", missed.len(), missed[..missed.len().min(4)].join(", "))));
        }
        // the xe driver has no out-of-memory error (an allocation past the card spills into host memory): the side
        // path's memory, its float32 staging and 1.5 GiB more must be free
        let (total, free) = gpu.memory()?;

        if (bytes + stage) as u64 + (3u64 << 29) > free.unwrap_or(total) {
            return Err(Error(format!("the request's LoRAs need {:.1} GiB on {}; {:.1} GiB are free", bytes as f64 / (1u64 << 30) as f64, gpu.name,
                                     free.unwrap_or(total) as f64 / (1u64 << 30) as f64)));
        }
        let k8 = ffi::api()?;
        // A stacked, B in its part's rows (permuted as the matrix's), on the GPU: half or float32
        let build = |p: &Plan, n: usize, k: usize, half: bool| -> Result<Option<SideMat>> {
            if p.is_empty() {
                return Ok(None);
            }
            let (a, bm, r) = stacked(p, n, k);
            let dev = |v: &[f32]| -> Result<DevBuf> {
                let f = DevBuf::from_f32(gpu, v)?;
                if !half {
                    return Ok(f);
                }
                let h = DevBuf::new(gpu, v.len() * 2)?;
                // SAFETY: v.len() values each way.
                ffi::check(unsafe { (k8.to_half)(gpu.raw(), f.fp(), h.ptr(), v.len() as i64) }, "to half")?;
                nsd.wait()?;
                Ok(h)
            };
            Ok(Some(SideMat { a: dev(&a)?, b: dev(&bm)?, r }))
        };
        let mut blocks = Vec::with_capacity(plans.len());
        for (b, four) in self.blocks.iter().zip(&plans) {
            let ms = [&b.qkv, &b.out, &b.gp, &b.down];
            blocks.push([build(&four[0], ms[0].n, ms[0].k, true)?, build(&four[1], ms[1].n, ms[1].k, true)?,
                         build(&four[2], ms[2].n, ms[2].k, true)?, build(&four[3], ms[3].n, ms[3].k, true)?]);
        }
        let mut small = std::collections::BTreeMap::new();
        for (key, n, k, p) in &small_plans {
            if let Some(s) = build(p, *n, *k, false)? {
                small.insert(key.clone(), s);
            }
        }
        Ok(Side { blocks, small, tmp: std::sync::Mutex::new(None), bytes })
    }

    /// Run with a request's LoRAs on the side path (none: the weights alone); the old one dropped
    pub fn set_side(&self, s: Option<Side>) {
        *self.side.lock().unwrap_or_else(|p| p.into_inner()) = s;
    }

    /// out [m, n] += B (A x) for side matrix `s` (x [m, k] and out in `dt`)
    #[allow(clippy::too_many_arguments)]
    fn side_apply(&self, nsd: &Nsd, side: &Side, s: &SideMat, x: *const std::ffi::c_void, dt: Dt, m: usize, k: usize, n: usize,
                  out: *mut std::ffi::c_void) -> Result<()> {
        let need = m * s.r * if dt == Dt::F32 { 4 } else { 2 };
        let mut t = side.tmp.lock().unwrap_or_else(|p| p.into_inner());
        if t.as_ref().is_none_or(|b| b.len < need) {
            // (the queue is in order: a smaller scratch still read by a queued product is freed after it)
            nsd.wait()?;
            *t = Some(DevBuf::new(&self.gpu, need)?);
        }
        let tmp = t.as_ref().expect("made above");
        nsd.linear(x, dt, m, k, s.a.ptr(), s.r, none(), tmp.ptr(), dt)?;
        nsd.linear_acc(tmp.ptr(), dt, m, s.r, s.b.ptr(), n, out)
    }

    /// Block `bi`'s matrix `j` (0 qkv, 1 out, 2 gate|up, 3 down): out [m, n] (half) = x [m, k] (half) . W^T, and the
    /// request's LoRAs beside it
    #[allow(clippy::too_many_arguments)]
    fn mat(&self, nsd: &Nsd, bi: usize, j: usize, w: &Mat, x: *const std::ffi::c_void, m: usize, k: usize, out: *mut std::ffi::c_void) -> Result<()> {
        w.apply(nsd, x, m, k, out, Dt::F16)?;
        let g = self.side.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(side) = g.as_ref() {
            if let Some(s) = side.blocks.get(bi).and_then(|b| b[j].as_ref()) {
                self.side_apply(nsd, side, s, x, Dt::F16, m, k, w.n, out)?;
            }
        }
        Ok(())
    }

    /// A small linear `name` (float32 w [n, k]): out [m, n] = x [m, k] . W^T, and the request's LoRAs beside it
    #[allow(clippy::too_many_arguments)]
    fn small(&self, nsd: &Nsd, name: &str, w: &DevBuf, x: *const std::ffi::c_void, m: usize, k: usize, n: usize, out: *mut std::ffi::c_void) -> Result<()> {
        nsd.linear(x, Dt::F32, m, k, w.ptr(), n, none(), out, Dt::F32)?;
        let g = self.side.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(side) = g.as_ref() {
            if let Some(s) = side.small.get(&lora::key(name)) {
                self.side_apply(nsd, side, s, x, Dt::F32, m, k, n, out)?;
            }
        }
        Ok(())
    }

    /// The modulation for timestep `t` (a sigma in [0, 1]) and for t = 0: per row (0: t, 1: zero) scale1, tanh(gate1),
    /// scale2, tanh(gate2), the output norm's scale - each [2, 4096] float32 on the GPU
    fn modulation(&self, nsd: &Nsd, t: f32) -> Result<[DevBuf; 5]> {
        let gpu = &self.gpu;
        // the sinusoid: cos then sin of 1000 t * 10000^(-i / 128)
        let mut emb = vec![0f32; 2 * 256];
        for (row, tt) in [t, 0.0].iter().enumerate() {
            for i in 0..128 {
                let a = 1000.0 * *tt as f64 * (-(10000f64).ln() * i as f64 / 128.0).exp();
                emb[row * 256 + i] = a.cos() as f32;
                emb[row * 256 + 128 + i] = a.sin() as f32;
            }
        }
        let silu = |v: &mut [f32]| v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
        let e = DevBuf::from_f32(gpu, &emb)?;
        let h = DevBuf::f32(gpu, 2 * DIM)?;
        self.small(nsd, "time_text_embed.timestep_embedder.linear_1", &self.t_lin1, e.ptr(), 2, 256, DIM, h.ptr())?;
        let mut hv = h.to_f32()?;
        silu(&mut hv);
        let h = DevBuf::from_f32(gpu, &hv)?;
        let temb = DevBuf::f32(gpu, 2 * DIM)?;
        self.small(nsd, "time_text_embed.timestep_embedder.linear_2", &self.t_lin2, h.ptr(), 2, DIM, DIM, temb.ptr())?;
        let mut tv = temb.to_f32()?;
        silu(&mut tv);
        let st = DevBuf::from_f32(gpu, &tv)?;
        let m = DevBuf::f32(gpu, 2 * 4 * DIM)?;
        self.small(nsd, "modulation.1", &self.modulation, st.ptr(), 2, DIM, 4 * DIM, m.ptr())?;
        let o = DevBuf::f32(gpu, 2 * DIM)?;
        self.small(nsd, "norm_out.linear", &self.norm_out, st.ptr(), 2, DIM, DIM, o.ptr())?;
        let mv = m.to_f32()?;
        // [scale1 | gate1 | scale2 | gate2] per row
        let part = |k: usize, tanh: bool| -> Result<DevBuf> {
            let mut v = vec![0f32; 2 * DIM];
            for row in 0..2 {
                for c in 0..DIM {
                    let x = mv[row * 4 * DIM + k * DIM + c];
                    v[row * DIM + c] = if tanh { x.tanh() } else { x };
                }
            }
            DevBuf::from_f32(gpu, &v)
        };
        Ok([part(0, false)?, part(1, true)?, part(2, false)?, part(3, true)?, DevBuf::from_f32(gpu, &o.to_f32()?)?])
    }

    /// (cos, sin) per token and pair for positions `idx` (frame, height, width per token), in the permuted order
    fn rope_table(&self, idx: &[[i64; 3]]) -> Result<DevBuf> {
        let mut cs = vec![0f32; idx.len() * HEAD];
        for (t, p) in idx.iter().enumerate() {
            let mut pair = 0;
            for (axis, dim) in AXES.iter().enumerate() {
                for i in 0..dim / 2 {
                    let inv = 1.0 / 10000f64.powf(2.0 * i as f64 / *dim as f64);
                    let a = p[axis] as f64 * inv;
                    cs[t * HEAD + 2 * pair] = a.cos() as f32;
                    cs[t * HEAD + 2 * pair + 1] = a.sin() as f32;
                    pair += 1;
                }
            }
        }
        DevBuf::from_f32(&self.gpu, &cs)
    }

    /// One block's rows `x` (float32 [m, 4096], in place): modulation row `row` of `md`; attention through `att`
    #[allow(clippy::too_many_arguments)]
    fn block_ffn(&self, nsd: &Nsd, bi: usize, b: &Block, x: &DevBuf, m: usize, md: &[DevBuf; 5], row: usize, zeros: &DevBuf, h: &DevBuf, gp: &DevBuf,
                 act: &DevBuf, o: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        let off = |buf: &DevBuf| -> *const f32 { buf.fp().wrapping_add(row * DIM) };
        // SAFETY: x holds m x 4096 floats, h as many halves; the scale row 4096 floats.
        ffi::check(unsafe { (k.ln_mod)(self.gpu.raw(), x.fp(), m as i64, DIM as i64, EPS, off(&md[2]), h.ptr()) }, "norm + modulate")?;
        prof::mark(nsd, "norm + modulate")?;
        self.mat(nsd, bi, 2, &b.gp, h.ptr(), m, DIM, gp.ptr())?;
        prof::mark(nsd, "gate|up linear")?;
        nsd.swiglu(gp.ptr(), Dt::F16, m, FFN, act.ptr(), Dt::F16)?;
        prof::mark(nsd, "swiglu")?;
        self.mat(nsd, bi, 3, &b.down, act.ptr(), m, FFN, o.ptr())?;
        prof::mark(nsd, "down linear")?;
        nsd.gate_add(x.ptr(), Dt::F32, m, DIM, o.ptr(), Dt::F16, zeros.ptr(), off(&md[3]).cast())?;
        prof::mark(nsd, "gate add")
    }

    /// The text's pass through every block (once a prompt): `embeds` float32 [T, 3584 or 4096] -> each layer's q|k|v
    pub fn prefix(&self, nsd: &Nsd, embeds: &DevBuf, tokens: usize, context_dim: usize) -> Result<Prefix> {
        self.prefix_with(nsd, embeds, tokens, context_dim, &[], &[])
    }

    /// The prefix of an edit: the text (`pads`: which of its tokens are `<|image_pad|>` slots - each stands for 2 x 2
    /// latents) with the condition pictures' latents (`conds`: float32 [h * w, 64] and (h, w), in the order of their
    /// slot runs) put in their slots. Block-causal: the text is causal, each picture's latents attend to everything
    /// before and to each other. Positions: text advances one on all three axes; a picture sits at the next frame
    /// with its rows and columns centred on zero, and the next text starts past its larger side. All at t = 0.
    pub fn prefix_with(&self, nsd: &Nsd, embeds: &DevBuf, tokens: usize, context_dim: usize, pads: &[bool], conds: &[(&DevBuf, (usize, usize))])
                       -> Result<Prefix> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let t = tokens;
        let md = self.modulation(nsd, 0.0)?;
        // txt_in: zero-centred RMS norm, linear, gelu (tanh), linear
        let hn = DevBuf::f32(gpu, t * context_dim)?;
        nsd.rms_norm_mod(embeds.ptr(), Dt::F32, t, context_dim, self.txt_norm.ptr(), EPS, none(), none(), none(), hn.ptr(), Dt::F32)?;
        let h1 = DevBuf::f32(gpu, t * DIM)?;
        self.small(nsd, "txt_in.in_layer", &self.txt_in1, hn.ptr(), t, context_dim, DIM, h1.ptr())?;
        let h1h = DevBuf::new(gpu, t * DIM * 2)?;
        // SAFETY: t x 4096 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), h1.fp(), h1h.ptr(), (t * DIM) as i64) }, "to half")?;
        ffi::check(unsafe { (k.gelu_tanh)(gpu.raw(), h1h.ptr(), (t * DIM) as i64) }, "gelu")?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), h1h.ptr(), h1.fp(), (t * DIM) as i64) }, "to float")?;
        let xt = DevBuf::f32(gpu, t * DIM)?;
        self.small(nsd, "txt_in.out_layer", &self.txt_in2, h1.ptr(), t, DIM, DIM, xt.ptr())?;
        // the joint rows: (start in x, rows, text?) segments, their positions; the text's rows and the pictures' latents
        let mut segs: Vec<(usize, usize, bool)> = Vec::new();
        let mut idx: Vec<[i64; 3]> = Vec::new();
        let mut pos = 0i64;
        let mut copies: Vec<(usize, usize, Option<usize>, usize)> = Vec::new(); // (dst row, src row, picture, rows)
        let (mut i, mut c) = (0usize, 0usize);
        while i < t {
            let row = idx.len();
            if pads.get(i).copied().unwrap_or(false) {
                let (_, (ch, cw)) = *conds.get(c).ok_or_else(|| Error("more picture slots in the prompt than pictures".into()))?;
                let slots = (ch * cw) / 4;
                if pads.len() < i + slots || !pads[i..i + slots].iter().all(|p| *p) {
                    return Err(Error(format!("picture {c}: {slots} slots expected in the prompt")));
                }
                for n in 0..ch * cw {
                    let (r, q) = ((n / cw) as i64, (n % cw) as i64);
                    idx.push([pos, r - (ch as i64 - ch as i64 / 2), q - (cw as i64 - cw as i64 / 2)]);
                }
                copies.push((row, 0, Some(c), ch * cw));
                segs.push((row, ch * cw, false));
                pos += ch.max(cw) as i64;
                i += slots;
                c += 1;
            } else {
                let st = i;
                while i < t && !pads.get(i).copied().unwrap_or(false) {
                    idx.push([pos, pos, pos]);
                    pos += 1;
                    i += 1;
                }
                copies.push((row, st, None, i - st));
                segs.push((row, i - st, true));
            }
        }
        if c != conds.len() {
            return Err(Error(format!("{} pictures for {c} slot runs in the prompt", conds.len())));
        }
        let p = idx.len();
        let x = DevBuf::f32(gpu, p * DIM)?;
        for (dst, src, pic, rows) in &copies {
            match pic {
                None => x.copy_within(dst * DIM * 4, &xt, src * DIM * 4, rows * DIM * 4)?,
                Some(ci) => self.small(nsd, "img_in", &self.img_in, conds[*ci].0.ptr(), *rows, LATENT, DIM, x.fp().wrapping_add(dst * DIM).cast())?,
            }
        }
        let zeros = DevBuf::new(gpu, p.max(1) * 4)?;
        zeros.fill(0)?;
        let cs = self.rope_table(&idx)?;
        let (h, qkv, att, o) = (DevBuf::new(gpu, p * DIM * 2)?, DevBuf::new(gpu, p * 3 * DIM * 2)?, DevBuf::new(gpu, p * DIM * 2)?, DevBuf::new(gpu, p * DIM * 2)?);
        let (gp, act) = (DevBuf::new(gpu, p * 2 * FFN * 2)?, DevBuf::new(gpu, p * FFN * 2)?);
        let mut kv = Vec::with_capacity(self.blocks.len());
        let off = |buf: &DevBuf| -> *const f32 { buf.fp().wrapping_add(DIM) }; // the t = 0 row
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        let text_only = segs.len() == 1 && segs[0].2;
        for (bi, b) in self.blocks.iter().enumerate() {
            // SAFETY: x holds p x 4096 floats, h as many halves; the scale row 4096 floats.
            ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), p as i64, DIM as i64, EPS, off(&md[0]), h.ptr()) }, "norm + modulate")?;
            self.mat(nsd, bi, 0, &b.qkv, h.ptr(), p, DIM, qkv.ptr())?;
            nsd.rms_rope(qkv.ptr(), Dt::F16, p, HEADS, HEAD, 3 * DIM, b.norm_q.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&qkv, DIM), Dt::F16, p, HEADS, HEAD, 3 * DIM, b.norm_k.ptr(), EPS, cs.ptr(), HEAD)?;
            if text_only {
                nsd.attention_causal(qkv.ptr(), half_at(&qkv, DIM), half_at(&qkv, 2 * DIM), Dt::F16, p, HEADS, HEADS, HEAD, 3 * DIM, 3 * DIM, att.ptr())?;
            } else {
                for (st, n, text) in &segs {
                    let out = half_at(&att, st * DIM);
                    if *text {
                        nsd.attention_causal_rows(qkv.ptr(), half_at(&qkv, DIM), half_at(&qkv, 2 * DIM), Dt::F16, *st, *n, HEADS, HEADS, HEAD, 3 * DIM, 3 * DIM, out)?;
                    } else {
                        nsd.attention_qk(half_at(&qkv, st * 3 * DIM), *n, 3 * DIM, half_at(&qkv, DIM), half_at(&qkv, 2 * DIM), st + n, 3 * DIM, Dt::F16, HEADS, HEAD, out,
                                         Dt::F16)?;
                    }
                }
            }
            // the layer's prefix q|k|v, kept for every step
            let keep = DevBuf::new(gpu, p * 3 * DIM * 2)?;
            keep.copy_within(0, &qkv, 0, p * 3 * DIM * 2)?;
            kv.push(keep);
            self.mat(nsd, bi, 1, &b.out, att.ptr(), p, DIM, o.ptr())?;
            nsd.gate_add(x.ptr(), Dt::F32, p, DIM, o.ptr(), Dt::F16, zeros.ptr(), off(&md[1]).cast())?;
            self.block_ffn(nsd, bi, b, &x, p, &md, 1, &zeros, &h, &gp, &act, &o)?;
        }
        nsd.wait()?;
        Ok(Prefix { tokens: p, kv, frame: pos })
    }

    /// The velocity for latents `lat` (float32 [hw.0 * hw.1, 64]) at sigma `t`, after the text `pre`: float32 [N, 64].
    /// `block0`: block 0's output rows kept there (a check), when given.
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), t: f32, block0: Option<&DevBuf>) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let n = hw.0 * hw.1;
        let tt = pre.tokens;
        let md = self.modulation(nsd, t)?;
        let zeros = DevBuf::new(gpu, n * 4)?;
        zeros.fill(0)?;
        let x = DevBuf::f32(gpu, n * DIM)?;
        self.small(nsd, "img_in", &self.img_in, lat.ptr(), n, LATENT, DIM, x.ptr())?;
        // positions: the frame after the text; height and width centred on zero, row-major
        let (hh, ww) = (hw.0 as i64, hw.1 as i64);
        let idx: Vec<[i64; 3]> = (0..n as i64).map(|i| [pre.frame, i / ww - (hh - hh / 2), i % ww - (ww - ww / 2)]).collect();
        let cs = self.rope_table(&idx)?;
        let h = DevBuf::new(gpu, n * DIM * 2)?;
        // [the text's q|k|v | the image's], one layer at a time
        let all = DevBuf::new(gpu, (tt + n) * 3 * DIM * 2)?;
        let att = DevBuf::new(gpu, n * DIM * 2)?;
        // the linears' outputs before their gated add: half
        let o = DevBuf::new(gpu, n * DIM * 2)?;
        let (gp, act) = (DevBuf::new(gpu, n * 2 * FFN * 2)?, DevBuf::new(gpu, n * FFN * 2)?);
        let half_at = |buf: &DevBuf, at: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(at).cast() };
        let img = tt * 3 * DIM; // the image's first row in `all`, in halves
        prof::mark(nsd, "step setup")?;
        for (bi, b) in self.blocks.iter().enumerate() {
            all.copy_within(0, &pre.kv[bi], 0, tt * 3 * DIM * 2)?;
            prof::mark(nsd, "prefix copy")?;
            // SAFETY: x holds n x 4096 floats, h as many halves; the scale row 4096 floats.
            ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), n as i64, DIM as i64, EPS, md[0].fp(), h.ptr()) }, "norm + modulate")?;
            prof::mark(nsd, "norm + modulate")?;
            self.mat(nsd, bi, 0, &b.qkv, h.ptr(), n, DIM, half_at(&all, img))?;
            prof::mark(nsd, "qkv linear")?;
            nsd.rms_rope(half_at(&all, img), Dt::F16, n, HEADS, HEAD, 3 * DIM, b.norm_q.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&all, img + DIM), Dt::F16, n, HEADS, HEAD, 3 * DIM, b.norm_k.ptr(), EPS, cs.ptr(), HEAD)?;
            prof::mark(nsd, "rms + rope")?;
            nsd.attention_qk(half_at(&all, img), n, 3 * DIM, half_at(&all, DIM), half_at(&all, 2 * DIM), tt + n, 3 * DIM, Dt::F16, HEADS, HEAD,
                             att.ptr(), Dt::F16)?;
            prof::mark(nsd, "attention")?;
            self.mat(nsd, bi, 1, &b.out, att.ptr(), n, DIM, o.ptr())?;
            prof::mark(nsd, "out linear")?;
            nsd.gate_add(x.ptr(), Dt::F32, n, DIM, o.ptr(), Dt::F16, zeros.ptr(), md[1].ptr())?;
            prof::mark(nsd, "gate add")?;
            self.block_ffn(nsd, bi, b, &x, n, &md, 0, &zeros, &h, &gp, &act, &o)?;
            if bi == 0 {
                if let Some(keep) = block0 {
                    keep.copy_within(0, &x, 0, n * DIM * 4)?;
                }
            }
        }
        // the output norm: layernorm * (1 + scale), then the projection to 64 latent channels
        let xh = DevBuf::new(gpu, n * DIM * 2)?;
        let xn = DevBuf::f32(gpu, n * DIM)?;
        // SAFETY: n x 4096 values each way; the scale row 4096 floats.
        ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), n as i64, DIM as i64, EPS, md[4].fp(), xh.ptr()) }, "norm + modulate")?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), xh.ptr(), xn.fp(), (n * DIM) as i64) }, "to float")?;
        let v = DevBuf::f32(gpu, n * LATENT)?;
        self.small(nsd, "proj_out", &self.proj_out, xn.ptr(), n, DIM, LATENT, v.ptr())?;
        prof::mark(nsd, "output")?;
        nsd.wait()?;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a merge adds to W (dit::load's merge: each part's B reordered as its rows, W += s B . A), on the host
    fn merged(p: &[(usize, bool, &Delta, f32)], n: usize, k: usize) -> Vec<f32> {
        let mut w = vec![0f32; n * k];
        for (off, perm, d, s) in p {
            for row in 0..d.n {
                let src = if *perm { (row / HEAD) * HEAD + rope_perm(row % HEAD) } else { row };
                for c in 0..k {
                    w[(off + row) * k + c] += s * (0..d.r).map(|j| d.b[src * d.r + j] * d.a[j * k + c]).sum::<f32>();
                }
            }
        }
        w
    }

    #[test]
    fn the_side_path_adds_what_a_merge_adds() {
        // q | k | v of two heads each (RoPE-permuted q and k), k = 8 inputs; updates on q (two LoRAs) and v
        let (k, rows) = (8, 2 * HEAD);
        let delta = |r: usize, seed: f32| Delta { target: String::new(), r, n: rows, k,
            a: (0..r * k).map(|i| (i as f32 * 0.37 + seed).sin()).collect(), b: (0..rows * r).map(|i| (i as f32 * 0.11 - seed).cos()).collect() };
        let (q1, q2, v) = (delta(2, 0.3), delta(3, 1.7), delta(4, -0.9));
        let p = [(0, true, &q1, 0.8f32), (0, true, &q2, 1.0), (2 * rows, false, &v, 0.5)];
        let n = 3 * rows;
        let (a, b, r) = stacked(&p, n, k);
        assert_eq!(r, 9);
        // B . A, densely
        let side: Vec<f32> = (0..n * k).map(|i| (0..r).map(|j| b[(i / k) * r + j] * a[j * k + i % k]).sum()).collect();
        let want = merged(&p, n, k);
        let worst = side.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "{worst}");
        // the k part takes nothing
        assert!(side[rows * k..2 * rows * k].iter().all(|x| *x == 0.0));
    }
}
