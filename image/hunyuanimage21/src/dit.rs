//! The denoiser: HunyuanImage 2.1's DiT (the official HYImageDiffusionTransformer), 3,584 features (28 heads of 128),
//! from the GGUF with ComfyUI's names (fused qkv). Its large matrices stay Q8_0 on the card (17B weights would not fit in
//! half) and are expanded to half just before their product - once a step for every branch (guidance's two passes
//! share each expansion).
//!
//! ```text
//!   vec = time_in(t) [+ time_r_in(t_next), / 2 (MeanFlow)] [+ guidance_in(g * 1000)]       one a step
//!   text = [byt5_in(glyph tokens) | token_refiner(Qwen tokens, t)]  only the valid tokens (the official model masks
//!          the rest: attention with the padding's keys masked is attention over the valid ones)
//!   img = img_in(latents)                                         [n, 3584], RoPE 2-D on these rows only
//!   20 double blocks: each stream modulated (6 x 3584 from vec), its own q k v (rms-normed per head), joint attention
//!     over [img | text], its own projection and GELU-tanh MLP, gated residuals
//!   40 single blocks over x = [img | text]: modulated (3 x 3584), linear1 -> q k v | mlp, attention, linear2 over
//!     [attention | gelu(mlp)], gated residual
//!   final: layernorm * (1 + scale) + shift (2 x 3584 from vec), linear -> 64 channels: the velocity
//! ```
//!
//! The RoPE pairs features (2i, 2i + 1) of a head (the first 64 by row, the next 64 by column, theta 256); the shared
//! kernel rotates (i, 64 + i), so the q and k rows of every qkv matrix (and their norms) are permuted at load -
//! attention scores do not change, v is untouched. The text's rows are normed but not rotated.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::Gguf;

use crate::ffi::{Act, K, T};
use crate::prof;
use crate::w::{host, rope_perm, tensor, Keep, Lin, Perm, Post};

pub const DIM: usize = 3584;
pub const HEADS: usize = 28;
pub const HEAD: usize = 128;
pub const FFN: usize = 14336;
pub const LATENT: usize = 64;
pub const BYT5: usize = 1472;
const EPS: f32 = 1e-6;
/// the RoPE's base
const THETA: f64 = 256.0;

/// One stream of a double block (the image's or the text's)
struct Stream {
    modulation: Lin,
    qkv: Lin,
    q_norm: DevBuf,
    k_norm: DevBuf,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

struct Double {
    img: Stream,
    txt: Stream,
}

struct Single {
    modulation: Lin,
    /// q | k | v | mlp (with the silo: q | k | v, the mlp's rows in linear1_mlp)
    linear1: Lin,
    linear1_mlp: Option<Lin>,
    /// [attention | mlp] -> 3584
    linear2: Lin,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

/// A TimestepEmbedder (sinusoid of 256 -> linear -> SiLU -> linear) or the refiner's TextProjection (linear, SiLU, linear)
struct Mlp2 {
    l1: Lin,
    l2: Lin,
}

struct RefinerBlock {
    norm1: (DevBuf, DevBuf),
    qkv: Lin,
    proj: Lin,
    norm2: (DevBuf, DevBuf),
    fc1: Lin,
    fc2: Lin,
    ada: Lin,
}

pub struct Dit {
    k: K,
    doubles: Vec<Double>,
    singles: Vec<Single>,
    img_in: Lin,
    time_in: Mlp2,
    time_r_in: Option<Mlp2>,
    guidance_in: Option<Mlp2>,
    ref_in: Lin,
    ref_t: Mlp2,
    ref_c: Mlp2,
    ref_blocks: Vec<RefinerBlock>,
    byt5_norm: (DevBuf, DevBuf),
    byt5_fc: [Lin; 3],
    final_ada: Lin,
    final_lin: Lin,
    /// the largest Q8_0 matrix in half: where each is expanded before its product
    scratch: DevBuf,
    /// zeros: the gate rows kernels' row index
    zeros: DevBuf,
    /// the products on the silo (its matrices split: `Lin::split_q8`), with GELU and the gated residuals fused
    silo: bool,
    /// the silo's products in int8 x int8 (approximate, NS_HI_INT8: each row of the inputs quantized as it goes)
    int8: bool,
    pub meanflow: bool,
    pub guided: bool,
    pub bytes: usize,
}

/// The prompt's part that does not change with the step: the refiner's input rows and pooled context, the glyph rows
pub struct Text {
    /// input_embedder(qwen tokens) [tq, 3584]
    qx: DevBuf,
    /// c_embedder(mean of the qwen tokens) [3584]
    c: DevBuf,
    pub tq: usize,
    /// byt5_in(glyph tokens) [tb, 3584]
    byt5: Option<DevBuf>,
    pub tb: usize,
}

impl Text {
    pub fn tokens(&self) -> usize {
        self.tq + self.tb
    }
}

/// What every step's pass reads from the timestep: vec's modulations for each block, the refiner's timestep embedding
/// - for a list of steps at once (one product over every step's rows)
pub struct Steps {
    n: usize,
    /// [n, 6 x 3584] for each double block's image and text streams
    dbl: Vec<(DevBuf, DevBuf)>,
    /// [n, 3 x 3584]
    sgl: Vec<DevBuf>,
    /// [n, 2 x 3584]: shift, scale
    fin: DevBuf,
    /// the refiner's t_embedder(t) [n, 3584]
    tref: DevBuf,
    /// vec [n, 3584] (a check)
    pub vec: DevBuf,
}

/// The timestep's sinusoid (256: cos then sin of t * 10000^(-i / 128)), as the official timestep_embedding in float32
pub fn sinusoid(t: f32) -> [f32; 256] {
    let mut e = [0f32; 256];
    for i in 0..128 {
        let freq = (-(10000f32).ln() * i as f32 / 128.0).exp();
        let a = t * freq;
        e[i] = a.cos();
        e[128 + i] = a.sin();
    }
    e
}

/// The RoPE's (cos, sin) for each of `rows` rows (an h x w grid, row-major, then `text` unrotated rows): pair i of a
/// head (features i and 64 + i after the permutation; 2i and 2i + 1 before) turns with the row (i < 32) or the column
/// (i >= 32) at theta^(-2 (i % 32) / 64)
pub fn rope_table(h: usize, w: usize, text: usize) -> Vec<f32> {
    let mut cs = vec![0f32; (h * w + text) * HEAD];
    for r in 0..h * w {
        let (y, x) = ((r / w) as f64, (r % w) as f64);
        for i in 0..HEAD / 2 {
            let (p, j) = if i < 32 { (y, i) } else { (x, i - 32) };
            let freq = 1.0 / THETA.powf(2.0 * j as f64 / 64.0);
            // the official tables are float32: positions times float32 frequencies
            let a = (p as f32 * freq as f32) as f64;
            cs[r * HEAD + 2 * i] = a.cos() as f32;
            cs[r * HEAD + 2 * i + 1] = a.sin() as f32;
        }
    }
    for r in h * w..h * w + text {
        for i in 0..HEAD / 2 {
            cs[r * HEAD + 2 * i] = 1.0;
        }
    }
    cs
}

/// A branch of guidance (the prompt's, the negative's): its rows in the batch's buffers
pub struct Branch<'a> {
    pub text: &'a Text,
    /// rows: n image, t text, from row r0 of the batch
    n: usize,
    t: usize,
    r0: usize,
    /// the RoPE's table for its n + t rows
    cs: DevBuf,
}

/// One pass's inputs and working buffers for every branch at once: each branch's rows [image | text] follow the
/// previous branch's, so a product over every row (the single blocks', the image stream's as a batch) is one call
pub struct Batch<'a> {
    pub branches: Vec<Branch<'a>>,
    /// every branch's rows
    rows: usize,
    x: DevBuf,
    h: DevBuf,
    big: DevBuf,
    att: DevBuf,
    o: DevBuf,
    mlp: DevBuf,
    /// the refiner's rows and buffers (one branch at a time)
    rx: DevBuf,
    /// the int8 products' quantized inputs: int8 [rows, FFN], float32 [rows]
    q: Option<(DevBuf, DevBuf)>,
}

impl Batch<'_> {
    /// the branches' image rows as a batch of the silo's products: (count, the rows between them), when they are evenly
    /// spaced (one or two branches always are)
    fn q(&self) -> (*mut i8, *mut f32) {
        self.q.as_ref().map_or((std::ptr::null_mut(), std::ptr::null_mut()), |(x, s)| (x.ptr().cast(), s.fp()))
    }

    fn img_batch(&self) -> Option<(usize, usize)> {
        let b = &self.branches;
        let step = b.get(1).map_or(0, |x| x.r0 - b[0].r0);
        b.windows(2).all(|w| w[1].r0 - w[0].r0 == step).then_some((b.len(), step))
    }
}

fn hptr(b: &DevBuf, at: usize) -> *mut std::ffi::c_void {
    b.ptr().cast::<u16>().wrapping_add(at).cast()
}

fn fptr(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

impl Dit {
    /// The DiT of `f`; `int8`: its products in int8 x int8 on the silo (approximate)
    pub fn load(f: &Gguf, nsd: &Nsd, k: &K, int8: bool, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let t0 = std::time::Instant::now();
        let gpu = &nsd.gpu;
        let lin = |names: &[&str], bias: &[&str], perms: &[Perm], keep: Keep| Lin::load(f, nsd, k, names, bias, perms, keep);
        let one = |name: &str, keep: Keep| lin(&[&format!("{name}.weight")], &[&format!("{name}.bias")], &[], keep);
        let vec_dev = |name: &str, perm: bool| -> Result<DevBuf> {
            let v = host(f, name)?;
            let v: Vec<f32> = if perm { (0..v.len()).map(|j| v[rope_perm(j)]).collect() } else { v };
            DevBuf::from_f32(gpu, &v)
        };
        let qk = [Perm { from: 0, to: DIM }, Perm { from: DIM, to: 2 * DIM }];
        // with the silo the products' matrices are split as loaded (the same bytes; the modulations stay Q8_0, for
        // their float32 products)
        let silo = k.silo().is_some();
        if int8 && !k.int8() {
            return Err(Error("NS_HI_INT8=1: the silo's int8 products are not available here (built? NS_SILO=0?)".into()));
        }
        let prod = |mut l: Lin| -> Result<Lin> {
            if int8 {
                l.rows_q8(nsd, k)?;
            } else if silo {
                l.split_q8(nsd, k)?;
            }
            Ok(l)
        };
        let mlp2 = |p: &str, a: &str, b: &str| -> Result<Mlp2> { Ok(Mlp2 { l1: one(&format!("{p}.{a}"), Keep::F32)?, l2: one(&format!("{p}.{b}"), Keep::F32)? }) };
        let doubles_n = (0..).take_while(|i| f.tensor(&format!("double_blocks.{i}.img_attn_qkv.weight")).is_some()).count();
        let singles_n = (0..).take_while(|i| f.tensor(&format!("single_blocks.{i}.linear1.weight")).is_some()).count();
        if doubles_n == 0 || singles_n == 0 {
            return Err(Error(format!("{}: not a HunyuanImage 2.1 DiT (no double_blocks / single_blocks)", f.paths[0].display())));
        }
        let qkv_w = tensor(f, "double_blocks.0.img_attn_qkv.weight")?;
        if qkv_w.shape != [3 * DIM as u64, DIM as u64] {
            return Err(Error(format!("double_blocks.0.img_attn_qkv.weight is {:?}: not HunyuanImage 2.1's 3584-wide DiT (the refiner?)", qkv_w.shape)));
        }
        let stream = |p: &str| -> Result<Stream> {
            Ok(Stream {
                modulation: one(&format!("{p}_mod.linear"), Keep::Stored)?,
                qkv: prod(lin(&[&format!("{p}_attn_qkv.weight")], &[&format!("{p}_attn_qkv.bias")], &qk, Keep::Stored)?)?,
                q_norm: vec_dev(&format!("{p}_attn_q_norm.weight"), true)?,
                k_norm: vec_dev(&format!("{p}_attn_k_norm.weight"), true)?,
                proj: prod(one(&format!("{p}_attn_proj"), Keep::Stored)?)?,
                fc1: prod(one(&format!("{p}_mlp.fc1"), Keep::Stored)?)?,
                fc2: prod(one(&format!("{p}_mlp.fc2"), Keep::Stored)?)?,
            })
        };
        let mut doubles = Vec::with_capacity(doubles_n);
        for i in 0..doubles_n {
            doubles.push(Double { img: stream(&format!("double_blocks.{i}.img"))?, txt: stream(&format!("double_blocks.{i}.txt"))? });
        }
        let mut singles = Vec::with_capacity(singles_n);
        for i in 0..singles_n {
            let p = format!("single_blocks.{i}");
            let l1 = lin(&[&format!("{p}.linear1.weight")], &[&format!("{p}.linear1.bias")], &qk, Keep::Stored)?;
            // the silo's linear1 is two products: q k v, and the mlp's rows with GELU on their outputs
            let (linear1, linear1_mlp) = if silo { (prod(l1.rows(nsd, 0, 3 * DIM)?)?, Some(prod(l1.rows(nsd, 3 * DIM, FFN)?)?)) } else { (l1, None) };
            singles.push(Single {
                modulation: one(&format!("{p}.modulation.linear"), Keep::Stored)?,
                linear1,
                linear1_mlp,
                linear2: prod(one(&format!("{p}.linear2"), Keep::Stored)?)?,
                q_norm: vec_dev(&format!("{p}.q_norm.weight"), true)?,
                k_norm: vec_dev(&format!("{p}.k_norm.weight"), true)?,
            });
        }
        let mut ref_blocks = Vec::new();
        for i in 0.. {
            let p = format!("txt_in.individual_token_refiner.blocks.{i}");
            if f.tensor(&format!("{p}.norm1.weight")).is_none() {
                break;
            }
            ref_blocks.push(RefinerBlock {
                norm1: (vec_dev(&format!("{p}.norm1.weight"), false)?, vec_dev(&format!("{p}.norm1.bias"), false)?),
                qkv: one(&format!("{p}.self_attn_qkv"), Keep::F32)?,
                proj: one(&format!("{p}.self_attn_proj"), Keep::F32)?,
                norm2: (vec_dev(&format!("{p}.norm2.weight"), false)?, vec_dev(&format!("{p}.norm2.bias"), false)?),
                fc1: one(&format!("{p}.mlp.fc1"), Keep::F32)?,
                fc2: one(&format!("{p}.mlp.fc2"), Keep::F32)?,
                ada: one(&format!("{p}.adaLN_modulation.1"), Keep::F32)?,
            });
        }
        // img_in: a 1 x 1 convolution [3584, 64, 1, 1] - a linear
        let img_in = {
            let w = host(f, "img_in.proj.weight")?;
            let b = host(f, "img_in.proj.bias")?;
            Lin { w: crate::w::Wt::F32(DevBuf::from_f32(gpu, &w)?), bias: Some(DevBuf::from_f32(gpu, &b)?), n: DIM, k: LATENT }
        };
        let meanflow = f.tensor("time_r_in.mlp.0.weight").is_some();
        let guided = f.tensor("guidance_in.mlp.0.weight").is_some();
        let biggest = doubles.iter().flat_map(|d| [&d.img, &d.txt]).flat_map(|s| [&s.qkv, &s.proj, &s.fc1, &s.fc2])
            .chain(singles.iter().flat_map(|s| [&s.linear1, &s.linear2])).map(|l| l.n * l.k).max().unwrap_or(0);
        // the expansion's scratch: none on the silo, which reads the matrices as they are
        let biggest = if silo { 1 } else { biggest };
        let dit = Dit {
            k: k.clone(),
            img_in,
            time_in: mlp2("time_in", "mlp.0", "mlp.2")?,
            time_r_in: if meanflow { Some(mlp2("time_r_in", "mlp.0", "mlp.2")?) } else { None },
            guidance_in: if guided { Some(mlp2("guidance_in", "mlp.0", "mlp.2")?) } else { None },
            ref_in: one("txt_in.input_embedder", Keep::F32)?,
            ref_t: mlp2("txt_in.t_embedder", "mlp.0", "mlp.2")?,
            ref_c: mlp2("txt_in.c_embedder", "linear_1", "linear_2")?,
            ref_blocks,
            byt5_norm: (vec_dev("byt5_in.layernorm.weight", false)?, vec_dev("byt5_in.layernorm.bias", false)?),
            byt5_fc: [one("byt5_in.fc1", Keep::F32)?, one("byt5_in.fc2", Keep::F32)?, one("byt5_in.fc3", Keep::F32)?],
            final_ada: one("final_layer.adaLN_modulation.1", Keep::F32)?,
            final_lin: one("final_layer.linear", Keep::F32)?,
            scratch: DevBuf::new(gpu, biggest * 2)?,
            zeros: {
                let z = DevBuf::new(gpu, 16384 * 4)?;
                z.fill(0)?;
                z
            },
            silo,
            int8,
            meanflow,
            guided,
            bytes: 0,
            doubles,
            singles,
        };
        let bytes = dit.doubles.iter().flat_map(|d| [&d.img, &d.txt]).flat_map(|s| [&s.modulation, &s.qkv, &s.proj, &s.fc1, &s.fc2]).map(|l| l.bytes()).sum::<usize>()
            + dit.singles.iter().flat_map(|s| [Some(&s.modulation), Some(&s.linear1), s.linear1_mlp.as_ref(), Some(&s.linear2)]).flatten().map(|l| l.bytes())
                .sum::<usize>()
            + dit.ref_blocks.iter().flat_map(|b| [&b.qkv, &b.proj, &b.fc1, &b.fc2, &b.ada]).map(|l| l.bytes()).sum::<usize>()
            + dit.scratch.len;
        log(format!("hunyuanimage 2.1 dit: {doubles_n} double + {singles_n} single blocks{}{}, {:.1} GiB on {} in {:.1} s{}",
                    if meanflow { ", MeanFlow" } else { "" }, if guided { ", guidance embedded" } else { "" }, bytes as f64 / (1u64 << 30) as f64, gpu.name,
                    t0.elapsed().as_secs_f64(),
                    k.silo().map_or_else(String::new, |p| format!(" (products on the silo{}, {})", if int8 { " in int8" } else { "" }, p.display()))));
        Ok(Dit { bytes, ..dit })
    }

    /// The modulation input for each step: vec = time_in(t) [+ time_r_in(t_next), / 2] [+ guidance_in(g)], as
    /// timesteps (x 1000)
    fn vecs(&self, nsd: &Nsd, ts: &[(f32, Option<f32>)], guidance: Option<f32>) -> Result<DevBuf> {
        let gpu = &nsd.gpu;
        let n = ts.len();
        let embed = |m: &Mlp2, vals: &[f32]| -> Result<DevBuf> {
            let e: Vec<f32> = vals.iter().flat_map(|t| sinusoid(*t)).collect();
            let e = DevBuf::from_f32(gpu, &e)?;
            let h = DevBuf::f32(gpu, vals.len() * DIM)?;
            nsd.linear(e.ptr(), Dt::F32, vals.len(), 256, w32(&m.l1)?, DIM, bias(&m.l1), h.ptr(), Dt::F32)?;
            self.k.act(h.ptr(), T::F32, vals.len() * DIM, Act::Silu)?;
            let o = DevBuf::f32(gpu, vals.len() * DIM)?;
            nsd.linear(h.ptr(), Dt::F32, vals.len(), DIM, w32(&m.l2)?, DIM, bias(&m.l2), o.ptr(), Dt::F32)?;
            Ok(o)
        };
        let t: Vec<f32> = ts.iter().map(|x| x.0).collect();
        let vec = embed(&self.time_in, &t)?;
        if let Some(r) = &self.time_r_in {
            let tr: Vec<f32> = ts.iter().map(|x| x.1.unwrap_or(x.0)).collect();
            let vr = embed(r, &tr)?;
            self.k.axpy(vec.fp(), vr.fp(), n * DIM, 1.0)?;
            nsd.scale(vec.ptr(), n * DIM, 0.5)?;
        }
        if let (Some(g), Some(gi)) = (guidance, &self.guidance_in) {
            let vg = embed(gi, &vec![g; n])?;
            self.k.axpy(vec.fp(), vg.fp(), n * DIM, 1.0)?;
        }
        nsd.wait()?;
        Ok(vec)
    }

    /// Every step's modulations and timestep embeddings: `ts` (t, t_next) as timesteps (sigma x 1000); `guidance`
    /// the embedded guidance's value (x 1000), for a model that embeds it
    pub fn steps(&self, nsd: &Nsd, ts: &[(f32, Option<f32>)], guidance: Option<f32>) -> Result<Steps> {
        let gpu = &nsd.gpu;
        let n = ts.len();
        let vec = self.vecs(nsd, ts, guidance)?;
        let sv = DevBuf::f32(gpu, n * DIM)?;
        sv.copy_within(0, &vec, 0, n * DIM * 4)?;
        self.k.act(sv.ptr(), T::F32, n * DIM, Act::Silu)?;
        // the modulation matrices in float32 (their inputs and outputs stay float32)
        let biggest = self.doubles.first().map_or(0, |d| d.img.modulation.n * d.img.modulation.k);
        let s32 = DevBuf::f32(gpu, biggest)?;
        let m = |l: &Lin| -> Result<DevBuf> {
            let o = DevBuf::f32(gpu, n * l.n)?;
            l.apply_f32(nsd, &self.k, &s32, sv.ptr(), n, o.ptr())?;
            Ok(o)
        };
        let mut dbl = Vec::with_capacity(self.doubles.len());
        for d in &self.doubles {
            dbl.push((m(&d.img.modulation)?, m(&d.txt.modulation)?));
        }
        let mut sgl = Vec::with_capacity(self.singles.len());
        for s in &self.singles {
            sgl.push(m(&s.modulation)?);
        }
        let fin = m(&self.final_ada)?;
        // the refiner's own timestep embedding (of t alone)
        let t: Vec<f32> = ts.iter().map(|x| x.0).collect();
        let e = DevBuf::from_f32(gpu, &t.iter().flat_map(|t| sinusoid(*t)).collect::<Vec<f32>>())?;
        let h = DevBuf::f32(gpu, n * DIM)?;
        nsd.linear(e.ptr(), Dt::F32, n, 256, w32(&self.ref_t.l1)?, DIM, bias(&self.ref_t.l1), h.ptr(), Dt::F32)?;
        self.k.act(h.ptr(), T::F32, n * DIM, Act::Silu)?;
        let tref = DevBuf::f32(gpu, n * DIM)?;
        nsd.linear(h.ptr(), Dt::F32, n, DIM, w32(&self.ref_t.l2)?, DIM, bias(&self.ref_t.l2), tref.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(Steps { n, dbl, sgl, fin, tref, vec })
    }

    /// The prompt's constant part: `qwen` float32 [tq, 3584] (the text encoder's tokens after the template's), `byt5`
    /// float32 [tb, 1472] (the glyph encoder's valid tokens) when the prompt quotes text
    pub fn text(&self, nsd: &Nsd, qwen: &[f32], byt5: Option<&[f32]>) -> Result<Text> {
        let gpu = &nsd.gpu;
        let tq = qwen.len() / DIM;
        if tq == 0 {
            return Err(Error("no text tokens".into()));
        }
        let raw = DevBuf::from_f32(gpu, qwen)?;
        let qx = DevBuf::f32(gpu, tq * DIM)?;
        nsd.linear(raw.ptr(), Dt::F32, tq, DIM, w32(&self.ref_in)?, DIM, bias(&self.ref_in), qx.ptr(), Dt::F32)?;
        // c_embedder of the tokens' mean (the mask's valid tokens: all of them here)
        let mut mean = vec![0f64; DIM];
        for r in 0..tq {
            for (m, v) in mean.iter_mut().zip(&qwen[r * DIM..(r + 1) * DIM]) {
                *m += *v as f64;
            }
        }
        let mean: Vec<f32> = mean.iter().map(|m| (*m / tq as f64) as f32).collect();
        let md = DevBuf::from_f32(gpu, &mean)?;
        let h = DevBuf::f32(gpu, DIM)?;
        nsd.linear(md.ptr(), Dt::F32, 1, DIM, w32(&self.ref_c.l1)?, DIM, bias(&self.ref_c.l1), h.ptr(), Dt::F32)?;
        self.k.act(h.ptr(), T::F32, DIM, Act::Silu)?;
        let c = DevBuf::f32(gpu, DIM)?;
        nsd.linear(h.ptr(), Dt::F32, 1, DIM, w32(&self.ref_c.l2)?, DIM, bias(&self.ref_c.l2), c.ptr(), Dt::F32)?;
        // byt5_in: LayerNorm (eps 1e-5, affine), fc1, GELU, fc2, GELU, fc3
        let (byt5, tb) = match byt5.filter(|b| !b.is_empty()) {
            None => (None, 0),
            Some(b) => {
                let tb = b.len() / BYT5;
                let x = DevBuf::from_f32(gpu, b)?;
                let n = DevBuf::f32(gpu, tb * BYT5)?;
                self.k.ln_mod(x.fp(), tb, BYT5, 1e-5, self.byt5_norm.0.fp(), self.byt5_norm.1.fp(), std::ptr::null(), std::ptr::null(), n.ptr(), T::F32)?;
                let [f1, f2, f3] = &self.byt5_fc;
                let a = DevBuf::f32(gpu, tb * f1.n)?;
                nsd.linear(n.ptr(), Dt::F32, tb, BYT5, w32(f1)?, f1.n, bias(f1), a.ptr(), Dt::F32)?;
                self.k.act(a.ptr(), T::F32, tb * f1.n, Act::GeluErf)?;
                let b2 = DevBuf::f32(gpu, tb * f2.n)?;
                nsd.linear(a.ptr(), Dt::F32, tb, f1.n, w32(f2)?, f2.n, bias(f2), b2.ptr(), Dt::F32)?;
                self.k.act(b2.ptr(), T::F32, tb * f2.n, Act::GeluErf)?;
                let o = DevBuf::f32(gpu, tb * DIM)?;
                nsd.linear(b2.ptr(), Dt::F32, tb, f2.n, w32(f3)?, DIM, bias(f3), o.ptr(), Dt::F32)?;
                (Some(o), tb)
            }
        };
        nsd.wait()?;
        Ok(Text { qx, c, tq, byt5, tb })
    }

    /// The buffers of one pass over every branch (`texts`: the prompt's, then the negative's) for an h x w latent grid
    pub fn batch<'a>(&self, nsd: &Nsd, texts: &[&'a Text], hw: (usize, usize)) -> Result<Batch<'a>> {
        let gpu = &nsd.gpu;
        let n = hw.0 * hw.1;
        let mut branches = Vec::with_capacity(texts.len());
        let mut rows = 0;
        for text in texts {
            let t = text.tokens();
            branches.push(Branch { text, n, t, r0: rows, cs: DevBuf::from_f32(gpu, &rope_table(hw.0, hw.1, t))? });
            rows += n + t;
        }
        if rows > self.zeros.len / 4 {
            return Err(Error(format!("{rows} rows: at most {} here", self.zeros.len / 4)));
        }
        let wide = 3 * DIM + FFN;
        let tq = texts.iter().map(|t| t.tq).max().unwrap_or(0);
        // the mlp's buffer: the double blocks' fc1 outputs; without the silo also the single blocks' [att | gelu(mlp)]
        let mlp = if self.silo { FFN } else { DIM + FFN };
        Ok(Batch {
            branches,
            rows,
            x: DevBuf::f32(gpu, rows * DIM)?,
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            big: DevBuf::new(gpu, rows * wide * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            o: DevBuf::f32(gpu, rows * DIM)?,
            mlp: DevBuf::new(gpu, rows * mlp * 2)?,
            rx: DevBuf::f32(gpu, tq * (3 * DIM + FFN + 3 * DIM))?,
            q: if self.int8 { Some((DevBuf::new(gpu, rows * FFN)?, DevBuf::f32(gpu, rows)?)) } else { None },
        })
    }

    /// The token refiner for step row `si` into branch b's text rows (after the glyph rows): x = input_embedder rows;
    /// c = t_embedder(t) + c_embedder(mean); per block: gates from SiLU(c), x += gate_msa attn(LN1(x)), x += gate_mlp
    /// mlp(LN2(x)) (SiLU)
    fn refine(&self, nsd: &Nsd, st: &Steps, si: usize, bt: &Batch, b: &Branch) -> Result<()> {
        let gpu = &nsd.gpu;
        let tq = b.text.tq;
        let x = fptr(&bt.x, (b.r0 + b.n + b.text.tb) * DIM);
        // the glyph rows first (reorder_txt_token: [valid byT5 | valid Qwen])
        if let Some(by) = &b.text.byt5 {
            bt.x.copy_within((b.r0 + b.n) * DIM * 4, by, 0, b.text.tb * DIM * 4)?;
        }
        bt.x.copy_within((b.r0 + b.n + b.text.tb) * DIM * 4, &b.text.qx, 0, tq * DIM * 4)?;
        let c = DevBuf::f32(gpu, DIM)?;
        c.copy_within(0, &st.tref, si * DIM * 4, DIM * 4)?;
        self.k.axpy(c.fp(), b.text.c.fp(), DIM, 1.0)?;
        self.k.act(c.ptr(), T::F32, DIM, Act::Silu)?;
        let gates = DevBuf::f32(gpu, 2 * DIM * self.ref_blocks.len())?;
        for (i, rb) in self.ref_blocks.iter().enumerate() {
            nsd.linear(c.ptr(), Dt::F32, 1, DIM, w32(&rb.ada)?, 2 * DIM, bias(&rb.ada), fptr(&gates, i * 2 * DIM).cast(), Dt::F32)?;
        }
        // the refiner's buffers in rx: normed rows | qkv | mlp; attention output reuses the normed rows' space
        let nrm = bt.rx.fp();
        let qkv = fptr(&bt.rx, tq * DIM);
        let mlp = fptr(&bt.rx, tq * 4 * DIM);
        let att = fptr(&bt.rx, tq * (4 * DIM + FFN));
        for (i, rb) in self.ref_blocks.iter().enumerate() {
            let gate = fptr(&gates, i * 2 * DIM);
            self.k.ln_mod(x, tq, DIM, EPS, rb.norm1.0.fp(), rb.norm1.1.fp(), std::ptr::null(), std::ptr::null(), nrm.cast(), T::F32)?;
            nsd.linear(nrm.cast(), Dt::F32, tq, DIM, w32(&rb.qkv)?, 3 * DIM, bias(&rb.qkv), qkv.cast(), Dt::F32)?;
            self.k.attn_small(qkv, 3 * DIM, qkv.wrapping_add(DIM), qkv.wrapping_add(2 * DIM), 3 * DIM, tq, HEADS, HEADS, HEAD, 1.0 / (HEAD as f32).sqrt(),
                              std::ptr::null(), false, att)?;
            nsd.linear(att.cast(), Dt::F32, tq, DIM, w32(&rb.proj)?, DIM, bias(&rb.proj), nrm.cast(), Dt::F32)?;
            nsd.gate_add(x.cast(), Dt::F32, tq, DIM, nrm.cast(), Dt::F32, self.zeros.ptr(), gate.cast())?;
            self.k.ln_mod(x, tq, DIM, EPS, rb.norm2.0.fp(), rb.norm2.1.fp(), std::ptr::null(), std::ptr::null(), nrm.cast(), T::F32)?;
            nsd.linear(nrm.cast(), Dt::F32, tq, DIM, w32(&rb.fc1)?, FFN, bias(&rb.fc1), mlp.cast(), Dt::F32)?;
            self.k.act(mlp.cast(), T::F32, tq * FFN, Act::Silu)?;
            nsd.linear(mlp.cast(), Dt::F32, tq, FFN, w32(&rb.fc2)?, DIM, bias(&rb.fc2), nrm.cast(), Dt::F32)?;
            nsd.gate_add(x.cast(), Dt::F32, tq, DIM, nrm.cast(), Dt::F32, self.zeros.ptr(), gate.wrapping_add(DIM).cast())?;
        }
        Ok(())
    }

    /// A double block's product for one stream on the silo: the image stream's rows of every branch as one batched
    /// call (when evenly spaced), the text stream's a branch at a time. `x` / `out`: (buffer, its row width in
    /// elements, its element size).
    #[allow(clippy::too_many_arguments)]
    fn stream_mm(&self, l: &Lin, bt: &Batch, img: bool, x: (&DevBuf, usize, usize), out: (&DevBuf, usize, usize), post: Post) -> Result<()> {
        let at = |b: &DevBuf, row: usize, w: usize, size: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u8>().wrapping_add(row * w * size).cast() };
        if let (true, Some((count, step))) = (img, bt.img_batch()) {
            let b0 = &bt.branches[0];
            return l.mm(&self.k, at(x.0, b0.r0, x.1, x.2), x.1, b0.n, at(out.0, b0.r0, out.1, out.2), out.1, post, None, true,
                        (count, step * x.1, step * out.1), bt.q());
        }
        for b in &bt.branches {
            let (r0, rows) = if img { (b.r0, b.n) } else { (b.r0 + b.n, b.t) };
            l.mm(&self.k, at(x.0, r0, x.1, x.2), x.1, rows, at(out.0, r0, out.1, out.2), out.1, post, None, true, (1, 0, 0), bt.q())?;
        }
        Ok(())
    }

    /// The velocity of each branch's latents (`lats[i]`: float32 [n, 64]) at step row `si` of `st`: float32 [n, 64]
    /// each. `tap(name, buffer, first row, rows)` sees branch 0's stages a check compares (refiner output, double block
    /// 0, the double blocks' image rows).
    pub fn forward(&self, nsd: &Nsd, st: &Steps, si: usize, bt: &Batch, lats: &[&DevBuf],
                   tap: &mut dyn FnMut(&str, &DevBuf, usize, usize) -> Result<()>) -> Result<Vec<DevBuf>> {
        let k = &self.k;
        let sc = &self.scratch;
        let branches = &bt.branches;
        if si >= st.n {
            return Err(Error(format!("step {si} of {}", st.n)));
        }
        for (b, lat) in branches.iter().zip(lats) {
            nsd.linear(lat.ptr(), Dt::F32, b.n, LATENT, w32(&self.img_in)?, DIM, bias(&self.img_in), fptr(&bt.x, b.r0 * DIM).cast(), Dt::F32)?;
            self.refine(nsd, st, si, bt, b)?;
        }
        let b0 = &branches[0];
        tap("txt0", &bt.x, b0.r0 + b0.n + b0.text.tb, b0.text.tq)?;
        prof::mark(nsd, "embed + refiner")?;
        let z = self.zeros.ptr() as *const std::ffi::c_void;
        for (bi, d) in self.doubles.iter().enumerate() {
            let (mi, mt) = (&st.dbl[bi].0, &st.dbl[bi].1);
            // a stream's modulation chunk j (shift1, scale1, gate1, shift2, scale2, gate2) at step si
            let md = |m: &DevBuf, j: usize| -> *const f32 { fptr(m, si * 6 * DIM + j * DIM) };
            // the streams' rows: image [r0, r0 + n), text [r0 + n, r0 + n + t)
            let parts = |b: &Branch| [(b.r0, b.n), (b.r0 + b.n, b.t)];
            for b in branches {
                for (s, (r0, rows)) in [(&d.img, mi), (&d.txt, mt)].iter().zip(parts(b)) {
                    let m = s.1;
                    k.ln_mod(fptr(&bt.x, r0 * DIM), rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(m, 1), md(m, 0), hptr(&bt.h, r0 * DIM), T::F16)?;
                }
            }
            prof::mark(nsd, "norm + modulate")?;
            for (si_, s) in [&d.img, &d.txt].iter().enumerate() {
                if self.silo {
                    self.stream_mm(&s.qkv, bt, si_ == 0, (&bt.h, DIM, 2), (&bt.big, 3 * DIM, 2), Post::Half)?;
                    continue;
                }
                let w = s.qkv.ready(nsd, k, sc)?;
                for b in branches {
                    let (r0, rows) = parts(b)[si_];
                    s.qkv.gemm(nsd, k, w, hptr(&bt.h, r0 * DIM), rows, hptr(&bt.big, r0 * 3 * DIM), Dt::F16)?;
                }
            }
            prof::mark(nsd, "qkv")?;
            for b in branches {
                let (q, s) = (b.r0 * 3 * DIM, b.n + b.t);
                nsd.rms_rope(hptr(&bt.big, q), Dt::F16, b.n, HEADS, HEAD, 3 * DIM, d.img.q_norm.ptr(), EPS, b.cs.ptr(), HEAD)?;
                nsd.rms_rope(hptr(&bt.big, q + DIM), Dt::F16, b.n, HEADS, HEAD, 3 * DIM, d.img.k_norm.ptr(), EPS, b.cs.ptr(), HEAD)?;
                nsd.rms_rope(hptr(&bt.big, q + b.n * 3 * DIM), Dt::F16, b.t, HEADS, HEAD, 3 * DIM, d.txt.q_norm.ptr(), EPS, b.cs.ptr(), 0)?;
                nsd.rms_rope(hptr(&bt.big, q + b.n * 3 * DIM + DIM), Dt::F16, b.t, HEADS, HEAD, 3 * DIM, d.txt.k_norm.ptr(), EPS, b.cs.ptr(), 0)?;
                prof::mark(nsd, "rms + rope")?;
                nsd.attention_qk(hptr(&bt.big, q), s, 3 * DIM, hptr(&bt.big, q + DIM), hptr(&bt.big, q + 2 * DIM), s, 3 * DIM, Dt::F16, HEADS, HEAD,
                                 hptr(&bt.att, b.r0 * DIM), Dt::F16)?;
                prof::mark(nsd, "attention")?;
            }
            for (si_, (s, m)) in [(&d.img, mi), (&d.txt, mt)].iter().enumerate() {
                if self.silo {
                    self.stream_mm(&s.proj, bt, si_ == 0, (&bt.att, DIM, 2), (&bt.x, DIM, 4), Post::Gate(md(m, 2)))?;
                    for b in branches {
                        let (r0, rows) = parts(b)[si_];
                        k.ln_mod(fptr(&bt.x, r0 * DIM), rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(m, 4), md(m, 3), hptr(&bt.h, r0 * DIM), T::F16)?;
                    }
                    continue;
                }
                let w = s.proj.ready(nsd, k, sc)?;
                for b in branches {
                    let (r0, rows) = parts(b)[si_];
                    s.proj.gemm(nsd, k, w, hptr(&bt.att, r0 * DIM), rows, fptr(&bt.o, r0 * DIM).cast(), Dt::F32)?;
                    nsd.gate_add(fptr(&bt.x, r0 * DIM).cast(), Dt::F32, rows, DIM, fptr(&bt.o, r0 * DIM).cast(), Dt::F32, z, md(m, 2).cast())?;
                    k.ln_mod(fptr(&bt.x, r0 * DIM), rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(m, 4), md(m, 3), hptr(&bt.h, r0 * DIM), T::F16)?;
                }
            }
            prof::mark(nsd, "proj")?;
            for (si_, s) in [&d.img, &d.txt].iter().enumerate() {
                if self.silo {
                    self.stream_mm(&s.fc1, bt, si_ == 0, (&bt.h, DIM, 2), (&bt.mlp, FFN, 2), Post::Gelu)?;
                    continue;
                }
                let w = s.fc1.ready(nsd, k, sc)?;
                for b in branches {
                    let (r0, rows) = parts(b)[si_];
                    s.fc1.gemm(nsd, k, w, hptr(&bt.h, r0 * DIM), rows, hptr(&bt.mlp, r0 * FFN), Dt::F16)?;
                    k.act(hptr(&bt.mlp, r0 * FFN), T::F16, rows * FFN, Act::GeluTanh)?;
                }
            }
            prof::mark(nsd, "fc1")?;
            for (si_, (s, m)) in [(&d.img, mi), (&d.txt, mt)].iter().enumerate() {
                if self.silo {
                    self.stream_mm(&s.fc2, bt, si_ == 0, (&bt.mlp, FFN, 2), (&bt.x, DIM, 4), Post::Gate(md(m, 5)))?;
                    continue;
                }
                let w = s.fc2.ready(nsd, k, sc)?;
                for b in branches {
                    let (r0, rows) = parts(b)[si_];
                    s.fc2.gemm(nsd, k, w, hptr(&bt.mlp, r0 * FFN), rows, fptr(&bt.o, r0 * DIM).cast(), Dt::F32)?;
                    nsd.gate_add(fptr(&bt.x, r0 * DIM).cast(), Dt::F32, rows, DIM, fptr(&bt.o, r0 * DIM).cast(), Dt::F32, z, md(m, 5).cast())?;
                }
            }
            prof::mark(nsd, "fc2")?;
            if bi == 0 {
                tap("dbl0_img", &bt.x, b0.r0, b0.n)?;
                tap("dbl0_txt", &bt.x, b0.r0 + b0.n, b0.t)?;
            }
        }
        tap("dbl_img", &bt.x, b0.r0, b0.n)?;
        // the single blocks: every branch's rows at once, but for attention
        let wide = 3 * DIM + FFN;
        let rows = bt.rows;
        for (i, s) in self.singles.iter().enumerate() {
            let m = &st.sgl[i];
            let md = |j: usize| -> *const f32 { fptr(m, si * 3 * DIM + j * DIM) };
            k.ln_mod(bt.x.fp(), rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(1), md(0), bt.h.ptr(), T::F16)?;
            prof::mark(nsd, "norm + modulate")?;
            if let (true, Some(l1m)) = (self.silo, &s.linear1_mlp) {
                // q k v, then the mlp's columns with their GELU: [att | gelu(mlp)] needs no pass of its own
                s.linear1.mm(k, bt.h.ptr(), DIM, rows, bt.big.ptr(), wide, Post::Half, None, true, (1, 0, 0), bt.q())?;
                prof::mark(nsd, "linear1")?;
                l1m.mm(k, bt.h.ptr(), DIM, rows, hptr(&bt.big, 3 * DIM), wide, Post::Gelu, None, true, (1, 0, 0), bt.q())?;
                prof::mark(nsd, "linear1 mlp + gelu")?;
            } else {
                let w = s.linear1.ready(nsd, k, sc)?;
                s.linear1.gemm(nsd, k, w, bt.h.ptr(), rows, bt.big.ptr(), Dt::F16)?;
                prof::mark(nsd, "linear1")?;
            }
            for b in branches {
                let (q, s_) = (b.r0 * wide, b.n + b.t);
                nsd.rms_rope(hptr(&bt.big, q), Dt::F16, s_, HEADS, HEAD, wide, s.q_norm.ptr(), EPS, b.cs.ptr(), HEAD)?;
                nsd.rms_rope(hptr(&bt.big, q + DIM), Dt::F16, s_, HEADS, HEAD, wide, s.k_norm.ptr(), EPS, b.cs.ptr(), HEAD)?;
                prof::mark(nsd, "rms + rope")?;
                nsd.attention_qk(hptr(&bt.big, q), s_, wide, hptr(&bt.big, q + DIM), hptr(&bt.big, q + 2 * DIM), s_, wide, Dt::F16, HEADS, HEAD,
                                 hptr(&bt.att, b.r0 * DIM), Dt::F16)?;
                prof::mark(nsd, "attention")?;
            }
            if self.silo {
                // linear2 over [att | gelu(mlp)] as two products into the residual: its first DIM columns on the
                // attention's rows, the rest on linear1's GELU columns where they are
                s.linear2.mm(k, bt.att.ptr(), DIM, rows, bt.x.ptr(), DIM, Post::Gate(md(2)), Some((0, DIM)), true, (1, 0, 0), bt.q())?;
                s.linear2.mm(k, hptr(&bt.big, 3 * DIM), wide, rows, bt.x.ptr(), DIM, Post::Gate(md(2)), Some((DIM, FFN)), false, (1, 0, 0), bt.q())?;
                prof::mark(nsd, "linear2")?;
                continue;
            }
            k.cat_gelu(bt.att.ptr(), rows, DIM, hptr(&bt.big, 3 * DIM), wide, FFN, bt.mlp.ptr())?;
            prof::mark(nsd, "concat + gelu")?;
            let w = s.linear2.ready(nsd, k, sc)?;
            s.linear2.gemm(nsd, k, w, bt.mlp.ptr(), rows, bt.o.ptr(), Dt::F32)?;
            nsd.gate_add(bt.x.ptr(), Dt::F32, rows, DIM, bt.o.ptr(), Dt::F32, z, md(2).cast())?;
            prof::mark(nsd, "linear2")?;
        }
        let mut out = Vec::with_capacity(branches.len());
        for b in branches {
            // the final layer: shift, scale; then the linear to 64 channels (float32)
            k.ln_mod(fptr(&bt.x, b.r0 * DIM), b.n, DIM, EPS, std::ptr::null(), std::ptr::null(), fptr(&st.fin, si * 2 * DIM + DIM), fptr(&st.fin, si * 2 * DIM),
                     fptr(&bt.o, b.r0 * DIM).cast(), T::F32)?;
            let v = DevBuf::f32(&nsd.gpu, b.n * LATENT)?;
            nsd.linear(fptr(&bt.o, b.r0 * DIM).cast(), Dt::F32, b.n, DIM, w32(&self.final_lin)?, LATENT, bias(&self.final_lin), v.ptr(), Dt::F32)?;
            out.push(v);
        }
        nsd.wait()?;
        prof::mark(nsd, "final")?;
        Ok(out)
    }
}

/// A float32 matrix's pointer
fn w32(l: &Lin) -> Result<*const std::ffi::c_void> {
    match &l.w {
        crate::w::Wt::F32(b) => Ok(b.ptr()),
        _ => Err(Error("a float32 matrix was expected".into())),
    }
}

fn bias(l: &Lin) -> *const std::ffi::c_void {
    l.bias.as_ref().map_or(none(), |b| b.ptr() as *const _)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_rope_table_turns_image_rows_and_leaves_text_alone() {
        let cs = super::rope_table(2, 3, 2);
        let head = super::HEAD;
        assert_eq!(cs.len(), 8 * head);
        // row 0, column 0: no rotation
        assert!(cs[..head].chunks(2).all(|p| p[0] == 1.0 && p[1] == 0.0));
        // row 1 (grid row 0, column 1): the row pairs still, the column pairs turned - pair 32 by 1 radian
        assert_eq!(cs[head], 1.0);
        assert!((cs[head + 64] - 1f32.cos()).abs() < 1e-6 && (cs[head + 65] - 1f32.sin()).abs() < 1e-6);
        // row 3 (grid row 1, column 0): pair 0 by 1 radian, pair 1 by 256^(-1/32)
        let a1 = 256f64.powf(-2.0 / 64.0) as f32;
        assert!((cs[3 * head + 1] - 1f32.sin()).abs() < 1e-6 && (cs[3 * head + 3] - a1.sin()).abs() < 1e-6);
        // the text rows: identity
        assert!(cs[6 * head..].chunks(2).all(|p| p[0] == 1.0 && p[1] == 0.0));
    }

    #[test]
    fn the_sinusoid_is_cos_then_sin() {
        let e = super::sinusoid(500.0);
        assert!((e[0] - 500f32.cos()).abs() < 1e-6 && (e[128] - 500f32.sin()).abs() < 1e-6);
        assert!((e[127] - (500.0 * (-(10000f32).ln() * 127.0 / 128.0).exp()).cos()).abs() < 1e-6);
    }
}
