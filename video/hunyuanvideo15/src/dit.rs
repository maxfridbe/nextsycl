//! The denoiser: HunyuanVideo-1.5's DiT (HunyuanVideo_1_5_DiffusionTransformer), 2,048 features (16 heads of 128), 54
//! double-stream blocks and no single-stream ones, from the official float32 safetensors. Its block matrices are
//! quantized to Q8_0 at load and multiplied on the silo (int8 x half scales, oneDNN's weight decompression); the rest
//! stays float32.
//!
//! ```text
//!   vec = time_in(t) [+ time_r_in(t_next) (MeanFlow: the step-distilled card)]                       one a step
//!   text = [vision_in(SigLIP tokens) + e2 (I2V)] [byt5_in(glyph tokens) + e1] [token_refiner(Qwen tokens, t) + e0]
//!          only the valid tokens (upstream's flash attention runs the unpadded sequence: the same thing); e the
//!          condition type embeddings
//!   img = img_in([latents | condition latents | mask])           a 1x1x1 convolution of 65 channels: [n, 2048]
//!   54 double blocks: each stream modulated (6 x 2048 from SiLU(vec)), its own q k v (rms-normed per head; the
//!     image's q, k turned by the 3-D RoPE: t 16, h 56, w 56 features, theta 256), joint attention over [img | text],
//!     its own projection and GELU-tanh MLP, gated residuals
//!   final: layernorm * (1 + scale) + shift (2 x 2048 from SiLU(vec)), linear -> 32 channels: the velocity
//! ```
//!
//! The RoPE pairs features (2i, 2i + 1) of a head; the shared kernel rotates (i, 64 + i), so the q and k rows of each
//! stream (and their norms) are permuted at load - attention scores do not change, v is untouched.
//!
//! The attention is the shared `attention_qk` (queries over keys and values given apart): a later rolling window
//! (MovieRoll) runs the same call with its cached keys and values before the window's own.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ffi::{Act, K, T};
use crate::w::{Keep, Lin, Post, St};

pub const DIM: usize = 2048;
pub const HEADS: usize = 16;
pub const HEAD: usize = 128;
pub const FFN: usize = 8192;
/// the double blocks of the released DiTs
pub const DOUBLES: usize = 54;
pub const LATENT: usize = 32;
/// img_in's channels: the latents, the condition latents, the mask
pub const IN: usize = 2 * LATENT + 1;
pub const TEXT: usize = 3584;
pub const BYT5: usize = 1472;
pub const VISION: usize = 1152;
pub(crate) const EPS: f32 = 1e-6;
const THETA: f64 = 256.0;
/// the RoPE's features by axis (t, h, w)
const ROPE: [usize; 3] = [16, 56, 56];

/// The row order of a q / k projection whose RoPE pairs features (2i, 2i + 1) of each 128-wide head, for kernels that
/// rotate (i, 64 + i): new row j of a head reads old row perm(j)
pub fn rope_perm(j: usize) -> usize {
    if j < 64 {
        2 * j
    } else {
        2 * (j - 64) + 1
    }
}

/// qkv's rows: q and k permuted per head, v as it is
fn qkv_perm(r: usize) -> usize {
    if r < 2 * DIM {
        (r / HEAD) * HEAD + rope_perm(r % HEAD)
    } else {
        r
    }
}

pub(crate) struct Stream {
    pub(crate) modulation: Lin,
    pub(crate) qkv: Lin,
    pub(crate) q_norm: DevBuf,
    pub(crate) k_norm: DevBuf,
    pub(crate) proj: Lin,
    pub(crate) fc1: Lin,
    pub(crate) fc2: Lin,
}

pub(crate) struct Double {
    pub(crate) img: Stream,
    pub(crate) txt: Stream,
}

/// A TimestepEmbedder or a TextProjection: linear, SiLU, linear
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
    pub(crate) k: K,
    pub(crate) doubles: Vec<Double>,
    pub(crate) img_in: Lin,
    time_in: Mlp2,
    time_r_in: Option<Mlp2>,
    ref_in: Lin,
    ref_t: Mlp2,
    ref_c: Mlp2,
    ref_blocks: Vec<RefinerBlock>,
    byt5_norm: (DevBuf, DevBuf),
    byt5_fc: [Lin; 3],
    vision: Option<(DevBuf, DevBuf, Lin, Lin, DevBuf, DevBuf)>,
    /// cond_type_embedding [3, 2048] on the host
    cond: Vec<f32>,
    pub(crate) final_ada: Lin,
    pub(crate) final_lin: Lin,
    /// the shared products' half matrix (none on the silo), and their float32 output
    pub(crate) scratch: DevBuf,
    pub meanflow: bool,
    pub bytes: usize,
    /// where a pass's time goes (NS_HV15_PROFILE=1)
    pub prof: crate::prof::Prof,
    /// the joint attention's kernel by default (the silo's half when it has one: the engine's NS_HV15_ATTN over it)
    pub attn: Attn,
    /// the silo's half tiles (0: no sycl-tla part)
    pub tiles: usize,
}

/// The joint attention's kernel: the shared kernels' (ARK's half flash), the silo's half on a tile, or the silo's
/// SageAttention (int8 q and k: approximate, an opt-in)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attn {
    Shared,
    Half(usize),
    Sage,
}

impl Attn {
    /// NS_HV15_ATTN: shared | half | half:N | sage; the silo's half (its best tile) when it has one, else shared
    pub fn parse(v: Option<&str>, tiles: usize) -> std::result::Result<Attn, String> {
        let best = if tiles > 0 { Attn::Half(0) } else { Attn::Shared };
        match v.map(str::trim) {
            None | Some("") => Ok(best),
            Some("shared") => Ok(Attn::Shared),
            Some("half") => Ok(best),
            Some("sage") if tiles > 0 => Ok(Attn::Sage),
            Some(x) => match x.strip_prefix("half:").and_then(|n| n.parse::<usize>().ok()) {
                Some(n) if n < tiles => Ok(Attn::Half(n)),
                _ => Err(format!("NS_HV15_ATTN={x}: shared, half, half:0..{}, or sage (the silo's need its sycl-tla part)", tiles.max(1) - 1)),
            },
        }
    }
}

/// DeepCache's rows (Dit::deep_cache): every branch's rows as block `from` got them on the last full pass
pub struct DeepCache {
    x: DevBuf,
    pub from: usize,
    filled: std::cell::Cell<bool>,
}

/// Which steps DeepCache skips (upstream's no_cache_steps): every step before `start`, from `end` on, and every
/// `interval`-th from `start` runs in full; the rest reuse the last full one's rows. Upstream's defaults for 50 steps:
/// 11, 45, 4 - 25 cached.
pub fn deep_cache_skips(steps: usize, start: usize, end: usize, interval: usize) -> Vec<bool> {
    let interval = interval.max(1);
    (0..steps).map(|i| i >= start && i < end && !(i - start).is_multiple_of(interval)).collect()
}

/// The prompt's part that does not change with the step
pub struct Text {
    /// input_embedder(qwen tokens) [tq, 2048]
    qx: DevBuf,
    /// c_embedder(mean of the qwen tokens) [2048]
    c: DevBuf,
    pub tq: usize,
    /// [vision_in(...) + e2 | byt5_in(...) + e1]: the rows before the refined ones [tb, 2048]
    pre: Option<DevBuf>,
    pub tb: usize,
}

impl Text {
    pub fn tokens(&self) -> usize {
        self.tq + self.tb
    }
}

/// What every step's pass reads from the timestep
pub struct Steps {
    pub n: usize,
    /// [n, 6 x 2048] for each double block's image and text streams
    pub(crate) dbl: Vec<(DevBuf, DevBuf)>,
    /// [n, 2 x 2048]: shift, scale
    pub(crate) fin: DevBuf,
    /// the refiner's t_embedder(t) [n, 2048]
    tref: DevBuf,
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

/// The RoPE's (cos, sin) for an f x h x w grid (row-major), then `text` unrotated rows: pair i of a head (features i,
/// 64 + i after the permutation; 2i, 2i + 1 before) turns with the frame (i < 8), the row (i < 36) or the column, at
/// theta^(-2j / d) for its axis' d features and its index j there - as get_nd_rotary_pos_embed's float32 tables
pub fn rope_table(f: usize, h: usize, w: usize, text: usize) -> Vec<f32> {
    let n = f * h * w;
    let mut cs = vec![0f32; (n + text) * HEAD];
    let freq = |d: usize, j: usize| -> f32 { 1.0 / (THETA as f32).powf((2 * j) as f32 / d as f32) };
    for r in 0..n {
        let pos = [(r / (h * w)) as f32, ((r / w) % h) as f32, (r % w) as f32];
        for i in 0..HEAD / 2 {
            let (axis, j) = if i < ROPE[0] / 2 { (0, i) } else if i < (ROPE[0] + ROPE[1]) / 2 { (1, i - ROPE[0] / 2) } else { (2, i - (ROPE[0] + ROPE[1]) / 2) };
            let a = pos[axis] * freq(ROPE[axis], j);
            cs[r * HEAD + 2 * i] = a.cos();
            cs[r * HEAD + 2 * i + 1] = a.sin();
        }
    }
    for r in n..n + text {
        for i in 0..HEAD / 2 {
            cs[r * HEAD + 2 * i] = 1.0;
        }
    }
    cs
}

/// A branch of guidance (the prompt's, the negative's): its rows in the batch's buffers
pub struct Branch<'a> {
    pub text: &'a Text,
    pub n: usize,
    pub t: usize,
    pub r0: usize,
    pub(crate) cs: DevBuf,
}

/// One pass's inputs and working buffers for every branch: each branch's rows [image | text] after the previous one's
pub struct Batch<'a> {
    pub branches: Vec<Branch<'a>>,
    pub rows: usize,
    /// the latent grid (frames, rows, columns)
    pub grid: (usize, usize, usize),
    pub(crate) x: DevBuf,
    pub(crate) h: DevBuf,
    pub(crate) big: DevBuf,
    pub(crate) att: DevBuf,
    pub(crate) mlp: DevBuf,
    /// img_in's input [n, 65]
    pub(crate) inp: DevBuf,
    /// the refiner's buffers
    pub(crate) rx: DevBuf,
    /// the shared products' float32 output (none on the silo)
    pub(crate) tmp: DevBuf,
}

pub(crate) fn hptr(b: &DevBuf, at: usize) -> *mut std::ffi::c_void {
    b.ptr().cast::<u16>().wrapping_add(at).cast()
}

pub(crate) fn fptr(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

pub(crate) fn w32(l: &Lin) -> Result<*const std::ffi::c_void> {
    match &l.w {
        crate::w::Wt::F32(b) => Ok(b.ptr()),
        _ => Err(Error("a float32 matrix was expected".into())),
    }
}

pub(crate) fn bias(l: &Lin) -> *const std::ffi::c_void {
    l.bias.as_ref().map_or(none(), |b| b.ptr() as *const _)
}

impl Dit {
    pub fn load(st: &St, k: &K, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let t0 = std::time::Instant::now();
        let gpu = &k.gpu;
        let one = |name: &str, keep: Keep| Lin::load(st, k, &[format!("{name}.weight")], &[format!("{name}.bias")], None, keep);
        let vdev = |name: &str, perm: bool| -> Result<DevBuf> {
            let v = st.vec(name)?;
            let v: Vec<f32> = if perm { (0..v.len()).map(|j| v[rope_perm(j)]).collect() } else { v };
            DevBuf::from_f32(gpu, &v)
        };
        let mlp2 = |p: &str, a: &str, b: &str| -> Result<Mlp2> { Ok(Mlp2 { l1: one(&format!("{p}.{a}"), Keep::F32)?, l2: one(&format!("{p}.{b}"), Keep::F32)? }) };
        // HunyuanVideo-1.5's files hold q, k and v apart; MovieRoll's (its fine-tune) one fused qkv [3 x 2048, 2048]
        let fused = st.has("double_blocks.0.img_attn_qkv.weight");
        let first = if fused { "img_attn_qkv" } else { "img_attn_q" };
        let doubles_n = (0..).take_while(|i| st.has(&format!("double_blocks.{i}.{first}.weight"))).count();
        if doubles_n == 0 || st.has("single_blocks.0.linear1_q.weight") {
            return Err(Error(format!("{}: not a HunyuanVideo-1.5 DiT (54 double blocks, no single ones)", st.0.path.display())));
        }
        let q = st.0.need(&format!("double_blocks.0.{first}.weight")).map_err(crate::w::se)?;
        if q.shape != [if fused { 3 * DIM } else { DIM } as u64, DIM as u64] {
            return Err(Error(format!("double_blocks.0.{first}.weight is {:?}: not the 2048-wide DiT", q.shape)));
        }
        let stream = |p: &str| -> Result<Stream> {
            let parts: &[&str] = if fused { &["qkv"] } else { &["q", "k", "v"] };
            let names: Vec<String> = parts.iter().map(|x| format!("{p}_attn_{x}.weight")).collect();
            let biases: Vec<String> = parts.iter().map(|x| format!("{p}_attn_{x}.bias")).collect();
            Ok(Stream {
                modulation: one(&format!("{p}_mod.linear"), Keep::Q8Blocks)?,
                qkv: Lin::load(st, k, &names, &biases, Some(&qkv_perm), Keep::Q8)?,
                q_norm: vdev(&format!("{p}_attn_q_norm.weight"), true)?,
                k_norm: vdev(&format!("{p}_attn_k_norm.weight"), true)?,
                proj: one(&format!("{p}_attn_proj"), Keep::Q8)?,
                fc1: one(&format!("{p}_mlp.fc1"), Keep::Q8)?,
                fc2: one(&format!("{p}_mlp.fc2"), Keep::Q8)?,
            })
        };
        let mut doubles = Vec::with_capacity(doubles_n);
        for i in 0..doubles_n {
            doubles.push(Double { img: stream(&format!("double_blocks.{i}.img"))?, txt: stream(&format!("double_blocks.{i}.txt"))? });
            if i % 9 == 8 {
                log(format!("dit: {} of {doubles_n} blocks in {:.0} s", i + 1, t0.elapsed().as_secs_f64()));
            }
        }
        let mut ref_blocks = Vec::new();
        for i in 0.. {
            let p = format!("txt_in.individual_token_refiner.blocks.{i}");
            if !st.has(&format!("{p}.norm1.weight")) {
                break;
            }
            ref_blocks.push(RefinerBlock {
                norm1: (vdev(&format!("{p}.norm1.weight"), false)?, vdev(&format!("{p}.norm1.bias"), false)?),
                qkv: one(&format!("{p}.self_attn_qkv"), Keep::F32)?,
                proj: one(&format!("{p}.self_attn_proj"), Keep::F32)?,
                norm2: (vdev(&format!("{p}.norm2.weight"), false)?, vdev(&format!("{p}.norm2.bias"), false)?),
                fc1: one(&format!("{p}.mlp.fc1"), Keep::F32)?,
                fc2: one(&format!("{p}.mlp.fc2"), Keep::F32)?,
                ada: one(&format!("{p}.adaLN_modulation.1"), Keep::F32)?,
            });
        }
        let img_in = Lin::load(st, k, &["img_in.proj.weight".into()], &["img_in.proj.bias".into()], None, Keep::F32)?;
        if img_in.k != IN {
            return Err(Error(format!("img_in takes {} channels, not {IN}", img_in.k)));
        }
        let meanflow = st.has("time_r_in.mlp.0.weight");
        let vision = if st.has("vision_in.proj.1.weight") {
            Some((vdev("vision_in.proj.0.weight", false)?, vdev("vision_in.proj.0.bias", false)?, one("vision_in.proj.1", Keep::F32)?, one("vision_in.proj.3", Keep::F32)?,
                  vdev("vision_in.proj.4.weight", false)?, vdev("vision_in.proj.4.bias", false)?))
        } else {
            None
        };
        let silo = k.silo().is_some();
        let biggest = doubles.iter().flat_map(|d| [&d.img, &d.txt]).flat_map(|s| [&s.qkv, &s.proj, &s.fc1, &s.fc2]).map(|l| l.n * l.k).max().unwrap_or(0);
        let mods = doubles.first().map_or(0, |d| d.img.modulation.n * d.img.modulation.k);
        let dit = Dit {
            k: k.clone(),
            img_in,
            time_in: mlp2("time_in", "mlp.0", "mlp.2")?,
            time_r_in: if meanflow { Some(mlp2("time_r_in", "mlp.0", "mlp.2")?) } else { None },
            ref_in: one("txt_in.input_embedder", Keep::F32)?,
            ref_t: mlp2("txt_in.t_embedder", "mlp.0", "mlp.2")?,
            ref_c: mlp2("txt_in.c_embedder", "linear_1", "linear_2")?,
            ref_blocks,
            byt5_norm: (vdev("byt5_in.layernorm.weight", false)?, vdev("byt5_in.layernorm.bias", false)?),
            byt5_fc: [one("byt5_in.fc1", Keep::F32)?, one("byt5_in.fc2", Keep::F32)?, one("byt5_in.fc3", Keep::F32)?],
            vision,
            cond: st.vec("cond_type_embedding.weight")?,
            final_ada: one("final_layer.adaLN_modulation.1", Keep::F32)?,
            final_lin: one("final_layer.linear", Keep::F32)?,
            // the modulations' float32 expansion; without the silo also the products' half matrix
            scratch: DevBuf::new(gpu, (mods * 4).max(if silo { 0 } else { biggest * 2 }))?,
            meanflow,
            bytes: 0,
            prof: crate::prof::Prof::new(std::env::var("NS_HV15_PROFILE").is_ok_and(|v| v == "1")),
            attn: Attn::parse(None, k.attention_tiles()).map_err(Error)?,
            tiles: k.attention_tiles(),
            doubles,
        };
        let bytes = dit.doubles.iter().flat_map(|d| [&d.img, &d.txt]).flat_map(|s| [&s.modulation, &s.qkv, &s.proj, &s.fc1, &s.fc2]).map(|l| l.bytes()).sum::<usize>()
            + dit.scratch.len;
        log(format!("hunyuanvideo-1.5 dit: {doubles_n} double blocks{}{}, {:.1} GiB on {} in {:.1} s{}", if meanflow { ", MeanFlow" } else { "" },
                    if dit.vision.is_some() { ", vision" } else { "" }, bytes as f64 / (1u64 << 30) as f64, gpu.name, t0.elapsed().as_secs_f64(),
                    k.silo().map_or_else(|| " (no silo: the matrices expanded before each product)".to_string(), |p| format!(" (products on the silo, {})", p.display()))));
        Ok(Dit { bytes, ..dit })
    }

    /// Whether it reads SigLIP tokens (an I2V DiT; the T2V ones carry the projection too, unused)
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// An MLP embedder (sinusoid -> linear -> SiLU -> linear) of `vals`
    fn embed(&self, nsd: &Nsd, m: &Mlp2, vals: &[f32]) -> Result<DevBuf> {
        let gpu = &nsd.gpu;
        let e: Vec<f32> = vals.iter().flat_map(|t| sinusoid(*t)).collect();
        let e = DevBuf::from_f32(gpu, &e)?;
        let h = DevBuf::f32(gpu, vals.len() * DIM)?;
        self.k.linear32(e.ptr(), vals.len(), 256, w32(&m.l1)?, DIM, bias(&m.l1), h.ptr())?;
        self.k.act(h.ptr(), T::F32, vals.len() * DIM, Act::Silu)?;
        let o = DevBuf::f32(gpu, vals.len() * DIM)?;
        self.k.linear32(h.ptr(), vals.len(), DIM, w32(&m.l2)?, DIM, bias(&m.l2), o.ptr())?;
        Ok(o)
    }

    /// Every step's modulations and timestep embeddings: `ts` (t, t_next) as timesteps (sigma x 1000)
    pub fn steps(&self, nsd: &Nsd, ts: &[(f32, Option<f32>)]) -> Result<Steps> {
        let gpu = &nsd.gpu;
        let n = ts.len();
        let t: Vec<f32> = ts.iter().map(|x| x.0).collect();
        let vec = self.embed(nsd, &self.time_in, &t)?;
        if let Some(r) = &self.time_r_in {
            let tr: Vec<f32> = ts.iter().map(|x| x.1.unwrap_or(0.0)).collect();
            let vr = self.embed(nsd, r, &tr)?;
            self.k.axpy(vec.fp(), vr.fp(), n * DIM, 1.0)?;
        }
        let sv = DevBuf::f32(gpu, n * DIM)?;
        sv.copy_within(0, &vec, 0, n * DIM * 4)?;
        self.k.act(sv.ptr(), T::F32, n * DIM, Act::Silu)?;
        let m = |l: &Lin| -> Result<DevBuf> {
            let o = DevBuf::f32(gpu, n * l.n)?;
            l.apply_f32(&self.k, &self.scratch, sv.ptr(), n, o.ptr())?;
            Ok(o)
        };
        let mut dbl = Vec::with_capacity(self.doubles.len());
        for d in &self.doubles {
            dbl.push((m(&d.img.modulation)?, m(&d.txt.modulation)?));
        }
        let fin = m(&self.final_ada)?;
        let tref = self.embed(nsd, &self.ref_t, &t)?;
        nsd.wait()?;
        Ok(Steps { n, dbl, fin, tref, vec })
    }

    /// The prompt's constant part: `qwen` float32 [tq, 3584], `byt5` float32 [tb, 1472] (the glyph encoder's valid
    /// tokens) when the prompt quotes text, `vision` float32 [729, 1152] (I2V's SigLIP tokens)
    pub fn text(&self, nsd: &Nsd, qwen: &[f32], byt5: Option<&[f32]>, vision: Option<&[f32]>) -> Result<Text> {
        let gpu = &nsd.gpu;
        let k = &self.k;
        let tq = qwen.len() / TEXT;
        if tq == 0 {
            return Err(Error("no text tokens".into()));
        }
        let raw = DevBuf::from_f32(gpu, qwen)?;
        let qx = DevBuf::f32(gpu, tq * DIM)?;
        self.k.linear32(raw.ptr(), tq, TEXT, w32(&self.ref_in)?, DIM, bias(&self.ref_in), qx.ptr())?;
        // c_embedder of the tokens' mean (every token valid)
        let mut mean = vec![0f64; TEXT];
        for r in 0..tq {
            for (m, v) in mean.iter_mut().zip(&qwen[r * TEXT..(r + 1) * TEXT]) {
                *m += *v as f64;
            }
        }
        let mean: Vec<f32> = mean.iter().map(|m| (*m / tq as f64) as f32).collect();
        let md = DevBuf::from_f32(gpu, &mean)?;
        let h = DevBuf::f32(gpu, DIM)?;
        self.k.linear32(md.ptr(), 1, TEXT, w32(&self.ref_c.l1)?, DIM, bias(&self.ref_c.l1), h.ptr())?;
        k.act(h.ptr(), T::F32, DIM, Act::Silu)?;
        let c = DevBuf::f32(gpu, DIM)?;
        self.k.linear32(h.ptr(), 1, DIM, w32(&self.ref_c.l2)?, DIM, bias(&self.ref_c.l2), c.ptr())?;
        let mut pre: Vec<f32> = Vec::new();
        let add_type = |rows: &mut Vec<f32>, which: usize| {
            for (i, v) in rows.iter_mut().enumerate() {
                *v += self.cond[which * DIM + i % DIM];
            }
        };
        if let Some(v) = vision {
            let (n1w, n1b, l1, l2, n2w, n2b) = self.vision.as_ref().ok_or_else(|| Error("this DiT has no vision projection".into()))?;
            let tv = v.len() / VISION;
            let x = DevBuf::from_f32(gpu, v)?;
            let n = DevBuf::f32(gpu, tv * VISION)?;
            k.ln_mod(x.fp(), tv, VISION, 1e-5, n1w.fp(), n1b.fp(), std::ptr::null(), std::ptr::null(), n.ptr(), T::F32)?;
            let a = DevBuf::f32(gpu, tv * VISION)?;
            self.k.linear32(n.ptr(), tv, VISION, w32(l1)?, VISION, bias(l1), a.ptr())?;
            k.act(a.ptr(), T::F32, tv * VISION, Act::GeluErf)?;
            let b = DevBuf::f32(gpu, tv * DIM)?;
            self.k.linear32(a.ptr(), tv, VISION, w32(l2)?, DIM, bias(l2), b.ptr())?;
            let o = DevBuf::f32(gpu, tv * DIM)?;
            k.ln_mod(b.fp(), tv, DIM, 1e-5, n2w.fp(), n2b.fp(), std::ptr::null(), std::ptr::null(), o.ptr(), T::F32)?;
            let mut rows = o.to_f32()?;
            add_type(&mut rows, 2);
            pre.extend(rows);
        }
        if let Some(b) = byt5.filter(|b| !b.is_empty()) {
            // byt5_in: LayerNorm (eps 1e-5, affine), fc1, GELU, fc2, GELU, fc3
            let tb = b.len() / BYT5;
            let x = DevBuf::from_f32(gpu, b)?;
            let n = DevBuf::f32(gpu, tb * BYT5)?;
            k.ln_mod(x.fp(), tb, BYT5, 1e-5, self.byt5_norm.0.fp(), self.byt5_norm.1.fp(), std::ptr::null(), std::ptr::null(), n.ptr(), T::F32)?;
            let [f1, f2, f3] = &self.byt5_fc;
            let a = DevBuf::f32(gpu, tb * f1.n)?;
            self.k.linear32(n.ptr(), tb, BYT5, w32(f1)?, f1.n, bias(f1), a.ptr())?;
            k.act(a.ptr(), T::F32, tb * f1.n, Act::GeluErf)?;
            let b2 = DevBuf::f32(gpu, tb * f2.n)?;
            self.k.linear32(a.ptr(), tb, f1.n, w32(f2)?, f2.n, bias(f2), b2.ptr())?;
            k.act(b2.ptr(), T::F32, tb * f2.n, Act::GeluErf)?;
            let o = DevBuf::f32(gpu, tb * DIM)?;
            self.k.linear32(b2.ptr(), tb, f2.n, w32(f3)?, DIM, bias(f3), o.ptr())?;
            let mut rows = o.to_f32()?;
            add_type(&mut rows, 1);
            pre.extend(rows);
        }
        nsd.wait()?;
        let tb = pre.len() / DIM;
        let pre = if pre.is_empty() { None } else { Some(DevBuf::from_f32(gpu, &pre)?) };
        Ok(Text { qx, c, tq, pre, tb })
    }

    /// The buffers of one pass over every branch (`texts`: the prompt's, then the negative's) for an f x h x w grid
    pub fn batch<'a>(&self, nsd: &Nsd, texts: &[&'a Text], grid: (usize, usize, usize)) -> Result<Batch<'a>> {
        let gpu = &nsd.gpu;
        let n = grid.0 * grid.1 * grid.2;
        let mut branches = Vec::with_capacity(texts.len());
        let mut rows = 0;
        for text in texts {
            let t = text.tokens();
            branches.push(Branch { text, n, t, r0: rows, cs: DevBuf::from_f32(gpu, &rope_table(grid.0, grid.1, grid.2, t))? });
            rows += n + t;
        }
        let tq = texts.iter().map(|t| t.tq).max().unwrap_or(0);
        let silo = self.k.silo().is_some();
        let longest = branches.iter().map(|b| b.n.max(b.t)).max().unwrap_or(0);
        Ok(Batch {
            branches,
            rows,
            grid,
            x: DevBuf::f32(gpu, rows * DIM)?,
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            big: DevBuf::new(gpu, rows * 3 * DIM * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            mlp: DevBuf::new(gpu, rows * FFN * 2)?,
            inp: DevBuf::f32(gpu, n * IN)?,
            rx: DevBuf::f32(gpu, tq * (4 * DIM + FFN + DIM))?,
            tmp: DevBuf::f32(gpu, if silo { 1 } else { longest * FFN })?,
        })
    }

    /// The token refiner for step row `si` into branch b's text rows (after the vision / glyph rows), plus the
    /// condition type embedding 0: x = input_embedder rows; c = t_embedder(t) + c_embedder(mean); per block: gates from
    /// SiLU(c), x += gate_msa attn(LN1(x)), x += gate_mlp mlp(LN2(x)) (SiLU)
    pub(crate) fn refine(&self, nsd: &Nsd, st: &Steps, si: usize, bt: &Batch, b: &Branch, tap: &mut dyn FnMut(&str, &DevBuf, usize, usize) -> Result<()>) -> Result<()> {
        let gpu = &nsd.gpu;
        let tq = b.text.tq;
        let x = fptr(&bt.x, (b.r0 + b.n + b.text.tb) * DIM);
        if let Some(pre) = &b.text.pre {
            bt.x.copy_within((b.r0 + b.n) * DIM * 4, pre, 0, b.text.tb * DIM * 4)?;
        }
        bt.x.copy_within((b.r0 + b.n + b.text.tb) * DIM * 4, &b.text.qx, 0, tq * DIM * 4)?;
        let c = DevBuf::f32(gpu, DIM)?;
        c.copy_within(0, &st.tref, si * DIM * 4, DIM * 4)?;
        self.k.axpy(c.fp(), b.text.c.fp(), DIM, 1.0)?;
        self.k.act(c.ptr(), T::F32, DIM, Act::Silu)?;
        let gates = DevBuf::f32(gpu, 2 * DIM * self.ref_blocks.len())?;
        for (i, rb) in self.ref_blocks.iter().enumerate() {
            self.k.linear32(c.ptr(), 1, DIM, w32(&rb.ada)?, 2 * DIM, bias(&rb.ada), fptr(&gates, i * 2 * DIM).cast())?;
        }
        let zeros = DevBuf::new(gpu, tq * 4)?;
        zeros.fill(0)?;
        let nrm = bt.rx.fp();
        let qkv = fptr(&bt.rx, tq * DIM);
        let mlp = fptr(&bt.rx, tq * 4 * DIM);
        let att = fptr(&bt.rx, tq * (4 * DIM + FFN));
        for (i, rb) in self.ref_blocks.iter().enumerate() {
            let gate = fptr(&gates, i * 2 * DIM);
            self.k.ln_mod(x, tq, DIM, EPS, rb.norm1.0.fp(), rb.norm1.1.fp(), std::ptr::null(), std::ptr::null(), nrm.cast(), T::F32)?;
            self.k.linear32(nrm.cast(), tq, DIM, w32(&rb.qkv)?, 3 * DIM, bias(&rb.qkv), qkv.cast())?;
            self.k.attn_small(qkv, 3 * DIM, qkv.wrapping_add(DIM), qkv.wrapping_add(2 * DIM), 3 * DIM, tq, HEADS, HEADS, HEAD, 1.0 / (HEAD as f32).sqrt(),
                              std::ptr::null(), false, att)?;
            self.k.linear32(att.cast(), tq, DIM, w32(&rb.proj)?, DIM, bias(&rb.proj), nrm.cast())?;
            nsd.gate_add(x.cast(), Dt::F32, tq, DIM, nrm.cast(), Dt::F32, zeros.ptr(), gate.cast())?;
            self.k.ln_mod(x, tq, DIM, EPS, rb.norm2.0.fp(), rb.norm2.1.fp(), std::ptr::null(), std::ptr::null(), nrm.cast(), T::F32)?;
            self.k.linear32(nrm.cast(), tq, DIM, w32(&rb.fc1)?, FFN, bias(&rb.fc1), mlp.cast())?;
            self.k.act(mlp.cast(), T::F32, tq * FFN, Act::Silu)?;
            self.k.linear32(mlp.cast(), tq, FFN, w32(&rb.fc2)?, DIM, bias(&rb.fc2), nrm.cast())?;
            nsd.gate_add(x.cast(), Dt::F32, tq, DIM, nrm.cast(), Dt::F32, zeros.ptr(), gate.wrapping_add(DIM).cast())?;
        }
        if b.r0 == 0 {
            tap("txt0", &bt.x, b.n + b.text.tb, tq)?;
        }
        // + the condition type embedding 0 on the refined rows
        let e0 = DevBuf::from_f32(gpu, &self.cond[..DIM])?;
        for r in 0..tq {
            self.k.axpy(x.wrapping_add(r * DIM), e0.fp(), DIM, 1.0)?;
        }
        nsd.wait()?;
        Ok(())
    }

    /// The velocity of each branch's latents at step row `si` of `st`: `lats[i]` float32 [n, 32] (frames, rows, columns
    /// in order, channels last), `cond` float32 [n, 33] (the condition latents and the mask). Returns float32 [n, 32]
    /// each. `tap(name, buffer, first row, rows)` sees branch 0's stages a check compares.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, nsd: &Nsd, st: &Steps, si: usize, bt: &Batch, lats: &[&DevBuf], cond: &DevBuf,
                   tap: &mut dyn FnMut(&str, &DevBuf, usize, usize) -> Result<()>) -> Result<Vec<DevBuf>> {
        let n = bt.branches.first().map_or(0, |b| b.n);
        self.forward_cached(nsd, st, si, &[(0, n, si)], bt, lats, cond, None, self.attn, tap)
    }

    /// `forward_spans`
    #[allow(clippy::too_many_arguments)]
    pub fn forward_spans(&self, nsd: &Nsd, st: &Steps, si: usize, spans: &[(usize, usize, usize)], bt: &Batch, lats: &[&DevBuf], cond: &DevBuf,
                         tap: &mut dyn FnMut(&str, &DevBuf, usize, usize) -> Result<()>) -> Result<Vec<DevBuf>> {
        self.forward_cached(nsd, st, si, spans, bt, lats, cond, None, self.attn, tap)
    }

    /// A buffer for DeepCache (upstream's --cache_type deepcache): every row of a pass as the last `blocks` double
    /// blocks get them
    pub fn deep_cache(&self, nsd: &Nsd, bt: &Batch, blocks: usize) -> Result<DeepCache> {
        if blocks == 0 || blocks > self.doubles.len() {
            return Err(Error(format!("deepcache: 1 to {} blocks recomputed, not {blocks}", self.doubles.len())));
        }
        Ok(DeepCache { x: DevBuf::f32(&nsd.gpu, bt.rows * DIM)?, from: self.doubles.len() - blocks, filled: std::cell::Cell::new(false) })
    }

    /// `forward` with the image rows in spans at their own step rows of `st`: (first image row, rows, step row) - a
    /// rolling window's chunks at their own noise levels (MovieRoll). The text stream and the token refiner take step
    /// row `si`. One span over every image row at `si` is `forward`.
    ///
    /// With `cache` (DeepCache): Some((c, false)) a full pass that keeps every row as block `c.from` gets them;
    /// Some((c, true)) a cached pass - those rows put back and only blocks `c.from`.. run (with this step's modulation),
    /// the embedding, the token refiner and the blocks before skipped. `attn`: the joint attention's kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(&self, nsd: &Nsd, st: &Steps, si: usize, spans: &[(usize, usize, usize)], bt: &Batch, lats: &[&DevBuf], cond: &DevBuf,
                          cache: Option<(&DeepCache, bool)>, attn: Attn, tap: &mut dyn FnMut(&str, &DevBuf, usize, usize) -> Result<()>)
                          -> Result<Vec<DevBuf>> {
        let k = &self.k;
        let branches = &bt.branches;
        if si >= st.n || spans.iter().any(|x| x.2 >= st.n) {
            return Err(Error(format!("a step row past the {} given", st.n)));
        }
        // the spans tile the image rows, in order
        let n = branches.first().map_or(0, |b| b.n);
        let mut at = 0;
        for (r0, rows, _) in spans {
            if *r0 != at || *rows == 0 {
                return Err(Error(format!("spans must tile the {n} image rows in order ({spans:?})")));
            }
            at += rows;
        }
        if at != n {
            return Err(Error(format!("spans cover {at} of the {n} image rows")));
        }
        if lats.len() != branches.len() {
            return Err(Error(format!("{} latents for {} branches", lats.len(), branches.len())));
        }
        let pf = &self.prof;
        pf.start(nsd)?;
        let reuse = match cache {
            Some((c, true)) if c.filled.get() => Some(c),
            Some((_, true)) => return Err(Error("deepcache: a cached pass before any full one".into())),
            _ => None,
        };
        if let Some(c) = reuse {
            bt.x.copy_within(0, &c.x, 0, bt.rows * DIM * 4)?;
        }
        for (b, lat) in branches.iter().zip(lats).filter(|_| reuse.is_none()) {
            // img_in over [latents | condition | mask]
            bt.inp.copy_2d(0, IN * 4, lat, 0, LATENT * 4, LATENT * 4, b.n)?;
            bt.inp.copy_2d(LATENT * 4, IN * 4, cond, 0, (IN - LATENT) * 4, (IN - LATENT) * 4, b.n)?;
            self.k.linear32(bt.inp.ptr(), b.n, IN, w32(&self.img_in)?, DIM, bias(&self.img_in), fptr(&bt.x, b.r0 * DIM).cast())?;
            self.refine(nsd, st, si, bt, b, tap)?;
        }
        let b0 = &branches[0];
        tap("txt_all0", &bt.x, b0.n, b0.t)?;
        pf.mark(nsd, "embed + refiner")?;
        let sc = &self.scratch;
        for (bi, d) in self.doubles.iter().enumerate() {
            match cache {
                Some((c, _)) if bi < c.from && reuse.is_some() => continue,
                Some((c, false)) if bi == c.from => {
                    c.x.copy_within(0, &bt.x, 0, bt.rows * DIM * 4)?;
                    c.filled.set(true);
                }
                _ => {}
            }
            let (mi, mt) = (&st.dbl[bi].0, &st.dbl[bi].1);
            let md = |m: &DevBuf, s: usize, j: usize| -> *const f32 { fptr(m, s * 6 * DIM + j * DIM) };
            let parts = |b: &Branch| [(b.r0, b.n), (b.r0 + b.n, b.t)];
            let streams = [(&d.img, mi), (&d.txt, mt)];
            // each stream's modulated row ranges: the image's spans, the text's at si
            let mod_parts = |b: &Branch| -> Vec<(usize, &DevBuf, usize, usize, usize)> {
                let mut v: Vec<(usize, &DevBuf, usize, usize, usize)> = spans.iter().map(|(r0, rows, s)| (0, mi, b.r0 + r0, *rows, *s)).collect();
                v.push((1, mt, b.r0 + b.n, b.t, si));
                v
            };
            for b in branches {
                for (_, m, r0, rows, s) in mod_parts(b) {
                    k.ln_mod(fptr(&bt.x, r0 * DIM), rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(m, s, 1), md(m, s, 0), hptr(&bt.h, r0 * DIM), T::F16)?;
                }
                pf.mark(nsd, "norm + modulate")?;
                for (si_, (s, _)) in streams.iter().enumerate() {
                    let (r0, rows) = parts(b)[si_];
                    s.qkv.mm(nsd, k, sc, &bt.tmp, hptr(&bt.h, r0 * DIM), DIM, rows, hptr(&bt.big, r0 * 3 * DIM), 3 * DIM, Post::Half)?;
                    pf.mark(nsd, if si_ == 0 { "qkv img" } else { "qkv txt" })?;
                }
                let (q, s) = (b.r0 * 3 * DIM, b.n + b.t);
                nsd.rms_rope(hptr(&bt.big, q), Dt::F16, b.n, HEADS, HEAD, 3 * DIM, d.img.q_norm.ptr(), EPS, b.cs.ptr(), HEAD)?;
                nsd.rms_rope(hptr(&bt.big, q + DIM), Dt::F16, b.n, HEADS, HEAD, 3 * DIM, d.img.k_norm.ptr(), EPS, b.cs.ptr(), HEAD)?;
                nsd.rms_rope(hptr(&bt.big, q + b.n * 3 * DIM), Dt::F16, b.t, HEADS, HEAD, 3 * DIM, d.txt.q_norm.ptr(), EPS, b.cs.ptr(), 0)?;
                nsd.rms_rope(hptr(&bt.big, q + b.n * 3 * DIM + DIM), Dt::F16, b.t, HEADS, HEAD, 3 * DIM, d.txt.k_norm.ptr(), EPS, b.cs.ptr(), 0)?;
                pf.mark(nsd, "rms + rope")?;
                match attn {
                    Attn::Shared => nsd.attention_qk(hptr(&bt.big, q), s, 3 * DIM, hptr(&bt.big, q + DIM), hptr(&bt.big, q + 2 * DIM), s, 3 * DIM, Dt::F16,
                                                     HEADS, HEAD, hptr(&bt.att, b.r0 * DIM), Dt::F16)?,
                    Attn::Half(t) => k.attention(hptr(&bt.big, q), s, HEADS, HEAD, 3 * DIM, hptr(&bt.att, b.r0 * DIM), DIM, false, t)?,
                    Attn::Sage => k.attention(hptr(&bt.big, q), s, HEADS, HEAD, 3 * DIM, hptr(&bt.att, b.r0 * DIM), DIM, true, 0)?,
                }
                pf.mark(nsd, "attention")?;
                for (which, m, r0, rows, s) in mod_parts(b) {
                    let st_ = streams[which].0;
                    let img = which == 0;
                    st_.proj.mm(nsd, k, sc, &bt.tmp, hptr(&bt.att, r0 * DIM), DIM, rows, fptr(&bt.x, r0 * DIM).cast(), DIM, Post::Gate(md(m, s, 2)))?;
                    pf.mark(nsd, if img { "proj img" } else { "proj txt" })?;
                    k.ln_mod(fptr(&bt.x, r0 * DIM), rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(m, s, 4), md(m, s, 3), hptr(&bt.h, r0 * DIM), T::F16)?;
                    pf.mark(nsd, "norm + modulate")?;
                    st_.fc1.mm(nsd, k, sc, &bt.tmp, hptr(&bt.h, r0 * DIM), DIM, rows, hptr(&bt.mlp, r0 * FFN), FFN, Post::Gelu)?;
                    pf.mark(nsd, if img { "fc1 img" } else { "fc1 txt" })?;
                    st_.fc2.mm(nsd, k, sc, &bt.tmp, hptr(&bt.mlp, r0 * FFN), FFN, rows, fptr(&bt.x, r0 * DIM).cast(), DIM, Post::Gate(md(m, s, 5)))?;
                    pf.mark(nsd, if img { "fc2 img" } else { "fc2 txt" })?;
                }
            }
            if bi == 0 {
                tap("dbl0_img", &bt.x, b0.r0, b0.n)?;
                tap("dbl0_txt", &bt.x, b0.r0 + b0.n, b0.t)?;
            }
        }
        tap("dbl_img", &bt.x, b0.r0, b0.n)?;
        let mut out = Vec::with_capacity(branches.len());
        let o = DevBuf::f32(&nsd.gpu, b0.n * DIM)?;
        for b in branches {
            for (r0, rows, s) in spans {
                k.ln_mod(fptr(&bt.x, (b.r0 + r0) * DIM), *rows, DIM, EPS, std::ptr::null(), std::ptr::null(), fptr(&st.fin, s * 2 * DIM + DIM),
                         fptr(&st.fin, s * 2 * DIM), fptr(&o, r0 * DIM).cast(), T::F32)?;
            }
            let v = DevBuf::f32(&nsd.gpu, b.n * LATENT)?;
            self.k.linear32(o.ptr(), b.n, DIM, w32(&self.final_lin)?, LATENT, bias(&self.final_lin), v.ptr())?;
            out.push(v);
        }
        nsd.wait()?;
        pf.mark(nsd, "final")?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_rope_table_turns_frames_rows_and_columns() {
        let head = super::HEAD;
        // 2 frames of 2 x 3, then 2 text rows
        let cs = super::rope_table(2, 2, 3, 2);
        assert_eq!(cs.len(), 14 * head);
        assert!(cs[..head].chunks(2).all(|p| p[0] == 1.0 && p[1] == 0.0));
        // grid row 1 (frame 0, row 0, column 1): the column pairs (36..) turned, pair 36 by 1 radian
        assert!(cs[head..head + 72].chunks(2).all(|p| p[0] == 1.0));
        assert!((cs[head + 72] - 1f32.cos()).abs() < 1e-6 && (cs[head + 73] - 1f32.sin()).abs() < 1e-6);
        // row 6 (frame 1, row 0, column 0): pair 0 by 1 radian, pair 1 by 256^(-2/16)
        let a1 = 256f32.powf(-2.0 / 16.0);
        assert!((cs[6 * head + 1] - 1f32.sin()).abs() < 1e-6 && (cs[6 * head + 3] - a1.sin()).abs() < 1e-6);
        // row 3 (frame 0, row 1, column 0): pair 8 by 1 radian (the h axis' first)
        assert!((cs[3 * head + 17] - 1f32.sin()).abs() < 1e-6);
        assert!(cs[12 * head..].chunks(2).all(|p| p[0] == 1.0 && p[1] == 0.0));
    }

    #[test]
    fn deepcache_skips_the_steps_upstream_does() {
        let s = super::deep_cache_skips(50, 11, 45, 4);
        let full: Vec<usize> = (0..50).filter(|i| !s[*i]).collect();
        let want: Vec<usize> = (0..11).chain((11..45).step_by(4)).chain(45..50).collect();
        assert_eq!(full, want);
        assert_eq!(s.iter().filter(|x| **x).count(), 25);
        assert!(super::deep_cache_skips(20, 0, 0, 4).iter().all(|x| !x));
    }

    #[test]
    fn the_attention_option_names_a_kernel_the_silo_has() {
        use super::Attn;
        assert_eq!(Attn::parse(None, 5), Ok(Attn::Half(0)));
        assert_eq!(Attn::parse(None, 0), Ok(Attn::Shared));
        assert_eq!(Attn::parse(Some("shared"), 5), Ok(Attn::Shared));
        assert_eq!(Attn::parse(Some("half:3"), 5), Ok(Attn::Half(3)));
        assert_eq!(Attn::parse(Some("sage"), 5), Ok(Attn::Sage));
        assert!(Attn::parse(Some("half:5"), 5).is_err());
        assert!(Attn::parse(Some("sage"), 0).is_err());
        assert!(Attn::parse(Some("fast"), 5).is_err());
    }

    #[test]
    fn qkv_rows_are_permuted_per_head_for_q_and_k_only() {
        assert_eq!(super::qkv_perm(0), 0);
        assert_eq!(super::qkv_perm(1), 2);
        assert_eq!(super::qkv_perm(64), 1);
        assert_eq!(super::qkv_perm(128 + 65), 128 + 3);
        assert_eq!(super::qkv_perm(2048 + 1), 2048 + 2);
        assert_eq!(super::qkv_perm(4096 + 1), 4096 + 1);
    }
}
