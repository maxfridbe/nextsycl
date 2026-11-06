//! The denoiser: FlashVSR's Wan 2.1 T2V-1.3B (diffsynth's `WanModel` as FlashVSR changes it), one step from noise at
//! timestep 1000, 30 blocks of 1,536 features (12 heads x 128), its weights in half on the card.
//!
//! ```text
//!   tokens: the noise's 1 x 2 x 2 patches [L, 64] -> patch embedding -> + the LQ projection (before block 0)
//!   block:  x += g1 * o(sparse_attention(rope(rms(q h)), rope(rms(k h)), v h)),  h = LN(x) * (1 + s1) + b1
//!           x += o(attention(rms(q LN3(x)), the prompt's cached k, v))
//!           x += g2 * W2 gelu_tanh(W1 h),  h = LN(x) * (1 + s2) + b2
//!   head:   LN(x) * (1 + scale) + shift -> linear -> [L, 64] -> latents = noise - output
//! ```
//!
//! The modulations are constant (one timestep, 1000) and the prompt is FlashVSR's fixed one (posi_prompt.pth), so both
//! are made once at load. The self-attention is FlashVSR's locality-constrained block-sparse attention over windows of
//! 2 x 8 x 8 tokens: per head, the means of each block's queries and keys score every key block (outside an
//! lr x lr window of blocks: none), the scores' softmax keeps its top k across each query group, and each query block
//! attends to the 128-key blocks it kept. The keys and values of the last three latent pairs are kept (the KV cache, the
//! stream's memory) in a ring of four pair slots a layer: a chunk's pair goes into the slot of the pair four before it,
//! which no query sees any more, so the attention reads the ring in place (its lists name slots) and nothing is moved.
//! A ring row holds a token's key and its value side by side (3,072 halves), so one product of 3,072 outputs makes
//! both, straight into the ring.
//!
//! The tokens are in window order throughout (see fv.h), so blocks are runs of 128 rows. Every product is a oneDNN
//! matmul with what follows it fused (bias, GELU, the gate and the residual sum into x, float32). The layers can be split
//! over cards (`Stage`): each holds its blocks, their ring and its working buffers.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi::{self, check, g, half_bytes};
use crate::prof;

pub const DIM: usize = 1536;
pub const HEADS: usize = 12;
pub const HEAD: usize = 128;
pub const FFN: usize = 8960;
pub const LAYERS: usize = 30;
pub const TEXT: usize = 4096;
const EPS: f32 = 1e-6;
/// rows of the MLP made at a time (its 8,960-wide hidden layer): a pair's at 1792 x 1024 (one product), half that on
/// larger grids (2560 x 1408's card is full)
const FFN_ROWS: usize = 16384;
fn ffn_rows(nb: usize) -> usize {
    if nb * 128 > FFN_ROWS { FFN_ROWS / 2 } else { FFN_ROWS }
}
/// the KV ring's slots: three cached pairs and the chunk's own
pub const SLOTS: usize = 4;

/// What follows a product (fused into it)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None = 0,
    Relu = 1,
    Gelu = 2,
}

/// A linear layer on the card: weight half [n, k], bias float32 [n]
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    pub fn from(gpu: &Arc<Gpu>, w: &[f32], b: Option<&[f32]>, n: usize, k: usize) -> Result<Lin> {
        if w.len() != n * k {
            return Err(Error(format!("a linear layer of {} values, {n} x {k} expected", w.len())));
        }
        let wd = DevBuf::new(gpu, n * k * 2)?;
        wd.write(0, &half_bytes(w))?;
        let b = b.map(|b| DevBuf::from_f32(gpu, b)).transpose()?;
        Ok(Lin { w: wd, b, n, k })
    }

    /// out [m, n] = act(x [m, k] (half, rows `lda` apart) . w^T + b), half
    #[allow(clippy::too_many_arguments)]
    pub fn half(&self, gpu: &Arc<Gpu>, x: *const std::ffi::c_void, lda: usize, m: usize, out: *mut std::ffi::c_void, ldo: usize, act: Act) -> Result<()> {
        self.run(gpu, x, lda, m, out, ldo, act, None, false, false)
    }

    /// x [m, n] float32 += gate * (h . w^T + b) (gate None: 1) - a gated residual
    pub fn residual(&self, gpu: &Arc<Gpu>, h: *const std::ffi::c_void, m: usize, x: *mut f32, gate: Option<&DevBuf>) -> Result<()> {
        self.run(gpu, h, self.k, m, x.cast(), self.n, Act::None, gate, true, true)
    }

    /// out [m, n] float32 = x . w^T + b
    pub fn float(&self, gpu: &Arc<Gpu>, x: *const std::ffi::c_void, m: usize, out: *mut f32) -> Result<()> {
        self.run(gpu, x, self.k, m, out.cast(), self.n, Act::None, None, false, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&self, gpu: &Arc<Gpu>, x: *const std::ffi::c_void, lda: usize, m: usize, out: *mut std::ffi::c_void, ldo: usize, act: Act, gate: Option<&DevBuf>,
           accum: bool, f32_out: bool) -> Result<()> {
        let k = ffi::api()?;
        let b = self.b.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        let gt = gate.map_or(std::ptr::null(), |g| g.fp() as *const f32);
        let flops = 2.0 * (m * self.k * self.n) as f64;
        prof::timed(gpu, "dit: products", flops, ((m * self.k + self.n * self.k) * 2 + m * self.n * if f32_out { 4 } else { 2 }) as f64, || {
            // SAFETY: x holds m rows of lda halves, out m rows of ldo values, the weights n x k halves.
            check(unsafe { (k.gemm)(g(gpu), x, m as i64, self.k as i64, lda as i64, self.w.ptr(), self.n as i64, b, act as i32, gt, accum as i32, out, ldo as i64,
                                    f32_out as i32) }, "a linear layer")
        })
    }

    pub fn bytes(&self) -> usize {
        self.n * self.k * 2 + self.b.as_ref().map_or(0, |_| self.n * 4)
    }
}

/// One block: its matrices, norms and constant modulation, the prompt's keys and values, its KV ring
pub struct Layer {
    q: Lin,
    /// the key and the value products as one: weights [k; v] [3072, 1536]
    kvp: Lin,
    o: Lin,
    cq: Lin,
    co: Lin,
    w1: Lin,
    w2: Lin,
    norm_q: DevBuf,
    norm_k: DevBuf,
    cnorm_q: DevBuf,
    n3_w: DevBuf,
    n3_b: DevBuf,
    /// shift and 1 + scale before the attention and the MLP, and their gates: float32 [1536] each
    sh1: DevBuf,
    sc1: DevBuf,
    g1: DevBuf,
    sh2: DevBuf,
    sc2: DevBuf,
    g2: DevBuf,
    /// the prompt's keys (normed) and values for the cross-attention: half [512, 1536]
    ck: DevBuf,
    cv: DevBuf,
    ctx_rows: usize,
    /// the KV ring: SLOTS pairs' keys and values side by side, half [SLOTS nb 128, 2 x 1536], and the block means of
    /// the keys (float32 [SLOTS nb, 1536])
    kv: Option<DevBuf>,
    km: Option<DevBuf>,
}

/// The shape of a chunk: its query groups (latent pairs), the block grid, its place in the stream
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    /// latent pairs in the chunk (3 for the first, then 1)
    pub groups: usize,
    pub nbh: usize,
    pub nbw: usize,
    /// the ring slot of the chunk's first pair (its pairs go to consecutive slots)
    pub slot: usize,
    /// which of the ring's slots hold a pair the chunk sees (its own included)
    pub valid: [bool; SLOTS],
    /// top-k blocks a query group keeps (FlashVSR's topk: int(nb^2 * ratio) - 1)
    pub topk: i64,
    /// the local window, in blocks
    pub local: usize,
}

impl Chunk {
    pub fn nb(&self) -> usize {
        self.nbh * self.nbw
    }
    pub fn rows(&self) -> usize {
        self.groups * self.nb() * 128
    }
    /// The chunk of the stream's `n`th step: the first holds three pairs (slots 0-2), each later one pair, in slot
    /// (n + 2) % 4 - the slot of the pair four before it, which no query sees any more
    pub fn of(n: usize, nbh: usize, nbw: usize, topk: i64, local: usize) -> Chunk {
        if n == 0 {
            Chunk { groups: 3, nbh, nbw, slot: 0, valid: [true, true, true, false], topk, local }
        } else {
            Chunk { groups: 1, nbh, nbw, slot: (n + 2) % SLOTS, valid: [true; SLOTS], topk, local }
        }
    }
    /// The key blocks the reference's draft sees (its threshold's rank is among them): the cached pairs and the chunk's own
    pub fn seen(&self) -> usize {
        self.valid.iter().filter(|v| **v).count() * self.nb()
    }
    /// The rank of the draft's threshold in a query group's nb x seen() scores: FlashVSR's min(n - 1, topk) + 1
    pub fn rank(&self) -> i64 {
        let n = (self.nb() * self.seen()) as i64;
        self.topk.min(n - 1).max(0) + 1
    }
}

/// FlashVSR's top k for a frame of th x tw (the padded size) and nb blocks a pair: int(nb^2 * sparse_ratio * 768 *
/// 1280 / (th tw)) - 1, the reference's arithmetic in its order
pub fn topk(nb: usize, th: usize, tw: usize, sparse_ratio: f64) -> i64 {
    let ratio = sparse_ratio * 768.0 * 1280.0 / (th * tw) as f64;
    ((nb * nb) as f64 * ratio) as i64 - 1
}

/// The working buffers of a stage, sized for the largest chunk
struct Work {
    rows: usize,
    nb: usize,
    h: DevBuf,
    q: DevBuf,
    attn: DevBuf,
    hid: DevBuf,
    qm: DevBuf,
    probs: DevBuf,
    thr: DevBuf,
    lists: DevBuf,
    counts: DevBuf,
    valid: DevBuf,
}

/// Layers [from, to) of the DiT on one card
pub struct Stage {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    pub from: usize,
    pub to: usize,
    layers: Vec<Layer>,
    work: Option<Work>,
    pub weight_bytes: usize,
    silo: Option<&'static ffi::Silo>,
}

/// The parts before and after the blocks: the patch embedding (on the first stage's card), the head (on the last's)
pub struct Ends {
    pub patch: Lin,
    pub head: Lin,
    pub head_sc: DevBuf,
    pub head_sh: DevBuf,
}

fn need(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    st.f32(name).map_err(|e| Error(e.0))
}

fn silu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
}

fn host_linear(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let k = x.len();
    w.chunks_exact(k).zip(b).map(|(row, bi)| row.iter().zip(x).map(|(a, c)| (*a as f64) * (*c as f64)).sum::<f64>() as f32 + bi).collect()
}

/// The timestep's embedding (1000, the one step) and the blocks' shared modulation: (t [1536], t_mod [6 x 1536])
pub fn time_embedding(st: &SafeTensors) -> Result<(Vec<f32>, Vec<f32>)> {
    let half = 128;
    let mut x = vec![0f32; 256];
    for i in 0..half {
        let f = 10000f64.powf(-(i as f64) / half as f64);
        let a = 1000.0 * f;
        x[i] = a.cos() as f32;
        x[half + i] = a.sin() as f32;
    }
    let mut h = host_linear(&need(st, "time_embedding.0.weight")?, &need(st, "time_embedding.0.bias")?, &x);
    silu(&mut h);
    let t = host_linear(&need(st, "time_embedding.2.weight")?, &need(st, "time_embedding.2.bias")?, &h);
    let mut ts = t.clone();
    silu(&mut ts);
    let tm = host_linear(&need(st, "time_projection.1.weight")?, &need(st, "time_projection.1.bias")?, &ts);
    Ok((t, tm))
}

impl Ends {
    pub fn load(st: &SafeTensors, first: &Arc<Gpu>, last: &Arc<Gpu>, t: &[f32]) -> Result<Ends> {
        let patch = Lin::from(first, &need(st, "patch_embedding.weight")?, Some(&need(st, "patch_embedding.bias")?), DIM, 64)?;
        let head = Lin::from(last, &need(st, "head.head.weight")?, Some(&need(st, "head.head.bias")?), 64, DIM)?;
        let m = need(st, "head.modulation")?;
        let sh: Vec<f32> = (0..DIM).map(|i| m[i] + t[i]).collect();
        let sc: Vec<f32> = (0..DIM).map(|i| 1.0 + m[DIM + i] + t[i]).collect();
        Ok(Ends { patch, head, head_sc: DevBuf::from_f32(last, &sc)?, head_sh: DevBuf::from_f32(last, &sh)? })
    }
}

/// The checks of a stage's tensors, by layer: the residual stream after it (float32 [rows, 1536], window order)
pub type Grab<'a> = dyn FnMut(usize, &DevBuf) -> Result<()> + 'a;

impl Stage {
    /// Layers [from, to) onto `gpu`; `ctx` the prompt after the text embedding (float32 [rows, 1536], on the host)
    #[allow(clippy::too_many_arguments)]
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>, from: usize, to: usize, tmod: &[f32], ctx: &[f32], log: &mut dyn FnMut(String)) -> Result<Stage> {
        let nsd = Nsd::new(gpu)?;
        let rows = ctx.len() / DIM;
        let ctx_d = DevBuf::new(gpu, rows * DIM * 2)?;
        ctx_d.write(0, &half_bytes(ctx))?;
        let k = ffi::api()?;
        let mut layers = Vec::new();
        let mut bytes = 0;
        for l in from..to {
            let p = |s: &str| format!("blocks.{l}.{s}");
            let lin = |name: &str| -> Result<Lin> {
                let w = need(st, &p(&format!("{name}.weight")))?;
                let b = need(st, &p(&format!("{name}.bias")))?;
                let (n, kk) = (b.len(), w.len() / b.len().max(1));
                Lin::from(gpu, &w, Some(&b), n, kk)
            };
            let vecd = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &need(st, &p(name))?) };
            let m = need(st, &p("modulation"))?;
            if m.len() != 6 * DIM {
                return Err(Error(format!("blocks.{l}.modulation: {} values, 6 x {DIM} expected", m.len())));
            }
            let md = |i: usize, one: f32| DevBuf::from_f32(gpu, &(0..DIM).map(|c| one + m[i * DIM + c] + tmod[i * DIM + c]).collect::<Vec<f32>>());
            // the prompt's keys and values, once: k normed, v as is
            let ck_lin = lin("cross_attn.k")?;
            let cv_lin = lin("cross_attn.v")?;
            let ck = DevBuf::new(gpu, rows * DIM * 2)?;
            let cv = DevBuf::new(gpu, rows * DIM * 2)?;
            ck_lin.half(gpu, ctx_d.ptr(), DIM, rows, ck.ptr(), DIM, Act::None)?;
            cv_lin.half(gpu, ctx_d.ptr(), DIM, rows, cv.ptr(), DIM, Act::None)?;
            let nk = vecd("cross_attn.norm_k.weight")?;
            // SAFETY: ck holds rows x 1536 halves, nk 1536 floats.
            check(unsafe { (k.rms_rope)(g(gpu), ck.ptr(), rows as i64, DIM as i64, DIM as i64, nk.fp(), EPS, std::ptr::null(), std::ptr::null(), 0, 0) },
                  "prompt keys' norm")?;
            nsd.wait()?;
            let layer = Layer {
                q: lin("self_attn.q")?,
                kvp: {
                    let cat = |what: &str| -> Result<Vec<f32>> {
                        let mut v = need(st, &p(&format!("self_attn.k.{what}")))?;
                        v.extend(need(st, &p(&format!("self_attn.v.{what}")))?);
                        Ok(v)
                    };
                    Lin::from(gpu, &cat("weight")?, Some(&cat("bias")?), 2 * DIM, DIM)?
                },
                o: lin("self_attn.o")?,
                cq: lin("cross_attn.q")?,
                co: lin("cross_attn.o")?,
                w1: lin("ffn.0")?,
                w2: lin("ffn.2")?,
                norm_q: vecd("self_attn.norm_q.weight")?,
                norm_k: vecd("self_attn.norm_k.weight")?,
                cnorm_q: vecd("cross_attn.norm_q.weight")?,
                n3_w: vecd("norm3.weight")?,
                n3_b: vecd("norm3.bias")?,
                sh1: md(0, 0.0)?,
                sc1: md(1, 1.0)?,
                g1: md(2, 0.0)?,
                sh2: md(3, 0.0)?,
                sc2: md(4, 1.0)?,
                g2: md(5, 0.0)?,
                ck,
                cv,
                ctx_rows: rows,
                kv: None,
                km: None,
            };
            if layer.w1.n != FFN || layer.q.k != DIM {
                return Err(Error(format!("blocks.{l}: a {} x {} attention, a {}-wide MLP: not Wan 2.1 1.3B's", layer.q.n, layer.q.k, layer.w1.n)));
            }
            bytes += [&layer.q, &layer.kvp, &layer.o, &layer.cq, &layer.co, &layer.w1, &layer.w2].iter().map(|l| l.bytes()).sum::<usize>()
                + rows * DIM * 4;
            layers.push(layer);
        }
        let silo = ffi::silo_on(gpu)?;
        log(format!("flashvsr: blocks {from}-{} on {} ({:.2} GiB; attention on {})", to - 1, gpu.name, bytes as f64 / (1u64 << 30) as f64,
                    if silo.is_some() { "the silo (sycl-tla)" } else { "the engine's plain kernel" }));
        Ok(Stage { gpu: gpu.clone(), nsd, from, to, layers, work: None, weight_bytes: bytes, silo })
    }

    /// What the KV ring and the working buffers take for a grid of nb blocks a pair (bytes)
    pub fn need_bytes(&self, nb: usize) -> usize {
        Self::need_bytes_for(self.layers.len(), nb)
    }

    /// The same for `layers` layers
    pub fn need_bytes_for(layers: usize, nb: usize) -> usize {
        let pair = nb * 128;
        let per = SLOTS * pair * 2 * DIM * 2 + SLOTS * nb * DIM * 4;
        let rows = 3 * pair;
        let (nq, nk) = (3 * nb, SLOTS * nb);
        let work = rows * DIM * 2 * 3 + ffn_rows(nb) * FFN * 2 + nq * DIM * 4 + HEADS * nq * nk * 4 * 2 + (64 << 20);
        per * layers + work
    }

    /// Whether the ring and the buffers are made for grids of nb blocks
    pub fn is_prepared(&self, nb: usize) -> bool {
        self.work.as_ref().is_some_and(|w| w.nb == nb)
    }

    /// The ring and the buffers for grids of nb blocks (made again when the grid changes)
    /// The per-size buffers let go (the KV ring and the work rows)
    pub fn release(&mut self) {
        self.work = None;
        for l in &mut self.layers {
            l.kv = None;
            l.km = None;
        }
    }

    pub fn prepare(&mut self, nb: usize) -> Result<()> {
        if self.work.as_ref().is_some_and(|w| w.nb == nb) {
            return Ok(());
        }
        self.work = None;
        for l in &mut self.layers {
            l.kv = None;
            l.km = None;
        }
        let gpu = &self.gpu;
        let pair = nb * 128;
        for l in &mut self.layers {
            l.kv = Some(DevBuf::new(gpu, SLOTS * pair * 2 * DIM * 2)?);
            l.km = Some(DevBuf::new(gpu, SLOTS * nb * DIM * 4)?);
        }
        let rows = 3 * pair;
        let (nq, nk) = (3 * nb, SLOTS * nb);
        self.work = Some(Work {
            rows,
            nb,
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            q: DevBuf::new(gpu, rows * DIM * 2)?,
            attn: DevBuf::new(gpu, rows * DIM * 2)?,
            hid: DevBuf::new(gpu, ffn_rows(nb).min(rows) * FFN * 2)?,
            qm: DevBuf::new(gpu, nq * DIM * 4)?,
            probs: DevBuf::new(gpu, HEADS * nq * nk * 4)?,
            thr: DevBuf::new(gpu, HEADS * 3 * 4)?,
            // the lists for LUT rows of 64 queries (twice as many rows as blocks of 128)
            lists: DevBuf::new(gpu, HEADS * 2 * nq * nk * 4)?,
            counts: DevBuf::new(gpu, HEADS * 2 * nq * 4)?,
            valid: DevBuf::new(gpu, SLOTS * 4)?,
        });
        Ok(())
    }

    /// The stage's blocks over x (float32 [rows, 1536] on this card); `rope` the RoPE tables on this card (`Rope`: the
    /// patches' for the grid, then the chunk's frames')
    pub fn run(&mut self, x: &DevBuf, rope: &DevBuf, c: &Chunk, grab: &mut Grab) -> Result<()> {
        let k = ffi::api()?;
        let gp = g(&self.gpu);
        let gpu = self.gpu.clone();
        let nb = c.nb();
        let rows = c.rows();
        let w = self.work.as_ref().ok_or_else(|| Error("flashvsr: the stage was not prepared".into()))?;
        if w.nb != nb || rows > w.rows {
            return Err(Error("flashvsr: the stage's buffers are for another grid".into()));
        }
        let xp = x.fp();
        let (hp, wp) = (8 * c.nbh, 8 * c.nbw);
        let (hw, ts) = (rope.fp() as *const f32, unsafe { rope.fp().add(Rope::hw_len(hp, wp)) } as *const f32);
        let (nq, nk) = (c.groups * nb, SLOTS * nb);
        let pair = nb * 128;
        let valid: Vec<u8> = c.valid.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect();
        w.valid.write(0, &valid)?;
        // the plain kernel takes plain lists of 128-row query blocks; the silo's LUT rows of 64 (its 64-row query tile)
        // (measured on the B70, 1728 x 960: 64 - ARK's - 7.0 s of 65 frames, 256 4.7, 128 with 64 keys a step 5.9, 128
        // with 32: 4.6)
        let (qrow, qtile) = match std::env::var("NS_FVSR_QTILE").ok().and_then(|v| v.trim().parse::<i64>().ok()) {
            Some(64) => (64usize, 64i64),
            Some(t @ (128..=133 | 256)) => (128, t),
            _ => (128, 129),
        };
        let rep = 128 / qrow;
        let at = |p: *mut std::ffi::c_void, r0: usize, el: usize| -> *mut std::ffi::c_void {
            // SAFETY: r0 is a row of the buffer (the callers stay inside it)
            unsafe { (p as *mut u8).add(r0 * DIM * el) as *mut std::ffi::c_void }
        };
        let fl = |r: usize| (r * DIM) as f64;
        for (li, l) in self.layers.iter().enumerate() {
            let (kv, km) = (l.kv.as_ref().unwrap(), l.km.as_ref().unwrap());
            // the ring's keys at column 0 of its rows, the values at 1536
            let (kc, vc) = (kv.ptr(), unsafe { (kv.ptr() as *mut u16).add(DIM) } as *mut std::ffi::c_void);
            let ring = |r0: usize| -> *mut std::ffi::c_void { unsafe { (kv.ptr() as *mut u8).add(r0 * 2 * DIM * 2) as *mut std::ffi::c_void } };
            // SAFETY (every call below): the buffers hold what the call reads and writes, sized by prepare.
            // ---- self-attention: q, and the chunk's keys and values into its ring slots
            let k0 = c.slot * pair;
            prof::timed(&gpu, "dit: norms, rope, gates", 0.0, fl(rows) * 6.0, || {
                check(unsafe { (k.ln)(gp, xp, rows as i64, DIM as i64, EPS, l.sc1.fp(), l.sh1.fp(), w.h.ptr()) }, "ln1")
            })?;
            l.q.half(&gpu, w.h.ptr(), DIM, rows, w.q.ptr(), DIM, Act::None)?;
            l.kvp.half(&gpu, w.h.ptr(), DIM, rows, ring(k0), 2 * DIM, Act::None)?;
            prof::timed(&gpu, "dit: norms, rope, gates", 0.0, fl(rows) * 8.0, || {
                check(unsafe { (k.rms_rope)(gp, w.q.ptr(), rows as i64, DIM as i64, DIM as i64, l.norm_q.fp(), EPS, ts, hw, hp as i64, wp as i64) },
                      "q norm + rope")?;
                check(unsafe { (k.rms_rope)(gp, ring(k0), rows as i64, DIM as i64, (2 * DIM) as i64, l.norm_k.fp(), EPS, ts, hw, hp as i64, wp as i64) },
                      "k norm + rope")
            })?;
            // ---- the draft: block means, their scores, the top k, the lists
            prof::timed(&gpu, "dit: draft (means, scores, top k, lists)", 0.0, fl(rows) * 4.0, || {
                check(unsafe { (k.block_mean)(gp, w.q.ptr(), nq as i64, DIM as i64, DIM as i64, w.qm.fp()) }, "query means")?;
                check(unsafe { (k.block_mean)(gp, ring(k0), nq as i64, (2 * DIM) as i64, DIM as i64, (km.ptr() as *mut u8).add(c.slot * nb * DIM * 4) as *mut f32) },
                      "key means")?;
                check(unsafe { (k.block_probs)(gp, w.qm.fp(), km.fp(), nq as i64, nk as i64, HEADS as i64, c.nbh as i64, c.nbw as i64, c.local as i64,
                                               w.valid.ptr() as *const i32, w.probs.fp()) }, "block scores")?;
                check(unsafe { (k.topk_thr)(gp, w.probs.fp(), (HEADS * c.groups) as i64, (nb * nk) as i64, c.rank(), w.thr.fp()) }, "top k")?;
                let (delta, r) = if self.silo.is_some() { (1, rep) } else { (0, 1) };
                check(unsafe { (k.block_lists)(gp, w.probs.fp(), w.thr.fp(), HEADS as i64, nq as i64, nk as i64, nb as i64, delta, r as i64,
                                               w.lists.ptr() as *mut i32, w.counts.ptr() as *mut i32) }, "block lists")
            })?;
            // ---- the attention over the ring
            // (profiling: its FLOPs from the lists - 4 x 128^3 a selected block of a 128-row query block - and the K / V
            // blocks read once a query tile)
            let (attn_flops, attn_bytes) = if prof::on() {
                let r = if self.silo.is_some() { rep } else { 1 };
                let mut raw = vec![0u8; HEADS * nq * r * 4];
                w.counts.read(0, &mut raw)?;
                let sel: u64 = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]).max(0) as u64).sum();
                let blocks = sel as f64 / r as f64;
                (blocks * 4.0 * 128f64.powi(3), sel as f64 * 2.0 * 128.0 * 128.0 * 2.0 + (rows * DIM * 2 * 2) as f64)
            } else {
                (0.0, 0.0)
            };
            prof::timed(&gpu, "dit: sparse attention", attn_flops, attn_bytes, || match self.silo {
                Some(s) => {
                    // SAFETY: the GPU's queue; q rows x 1536, the ring SLOTS pairs, the lists H x nq rep x nk.
                    s.check(unsafe { (s.sparse)(gpu.queue(), w.q.ptr(), kc, vc, w.attn.ptr(), rows as i64, (SLOTS * pair) as i64, HEADS as i64, (2 * DIM) as i64,
                                                w.lists.ptr() as *const i32, w.counts.ptr() as *const i32, qrow as i64, qtile) }, "sparse attention")?;
                    // a query row with no block: 0 (the reference's kernel; the silo's divides by its empty sum)
                    check(unsafe { (k.zero_empty)(gp, w.attn.ptr(), HEADS as i64, (nq * rep) as i64, qrow as i64, w.counts.ptr() as *const i32) }, "empty rows")
                }
                None => check(unsafe { (k.sparse_attn)(gp, w.q.ptr(), rows as i64, HEADS as i64, kc, vc, (2 * DIM) as i64, w.lists.ptr() as *const i32,
                                                       w.counts.ptr() as *const i32, nk as i64, w.attn.ptr()) }, "sparse attention"),
            })?;
            l.o.residual(&gpu, w.attn.ptr(), rows, xp, Some(&l.g1))?;
            // ---- cross-attention to the prompt
            prof::timed(&gpu, "dit: norms, rope, gates", 0.0, fl(rows) * 6.0, || {
                check(unsafe { (k.ln)(gp, xp, rows as i64, DIM as i64, EPS, l.n3_w.fp(), l.n3_b.fp(), w.h.ptr()) }, "ln3")
            })?;
            l.cq.half(&gpu, w.h.ptr(), DIM, rows, w.q.ptr(), DIM, Act::None)?;
            prof::timed(&gpu, "dit: cross attention", 4.0 * (rows * l.ctx_rows * DIM) as f64, fl(rows) * 4.0, || {
                check(unsafe { (k.rms_rope)(gp, w.q.ptr(), rows as i64, DIM as i64, DIM as i64, l.cnorm_q.fp(), EPS, std::ptr::null(), std::ptr::null(), 0, 0) },
                      "cross q norm")?;
                match self.silo {
                    // SAFETY: the GPU's queue; q rows x 1536, the prompt's 512 x 1536.
                    Some(s) => s.check(unsafe { (s.attention)(gpu.queue(), w.q.ptr(), l.ck.ptr(), l.cv.ptr(), w.attn.ptr(), rows as i64, l.ctx_rows as i64,
                                                               HEADS as i64, DIM as i64, DIM as i64, DIM as i64) }, "cross attention"),
                    None => self.nsd.attention_qk(w.q.ptr(), rows, DIM, l.ck.ptr(), l.cv.ptr(), l.ctx_rows, DIM, Dt::F16, HEADS, HEAD, w.attn.ptr(), Dt::F16),
                }
            })?;
            l.co.residual(&gpu, w.attn.ptr(), rows, xp, None)?;
            // ---- the MLP, its hidden layer FFN_ROWS rows at a time
            prof::timed(&gpu, "dit: norms, rope, gates", 0.0, fl(rows) * 6.0, || {
                check(unsafe { (k.ln)(gp, xp, rows as i64, DIM as i64, EPS, l.sc2.fp(), l.sh2.fp(), w.h.ptr()) }, "ln2")
            })?;
            let mut r0 = 0;
            while r0 < rows {
                let r = ffn_rows(nb).min(rows - r0);
                l.w1.half(&gpu, at(w.h.ptr(), r0, 2), DIM, r, w.hid.ptr(), FFN, Act::Gelu)?;
                // SAFETY: x's rows r0 .. r0 + r
                l.w2.residual(&gpu, w.hid.ptr(), r, unsafe { xp.add(r0 * DIM) }, Some(&l.g2))?;
                r0 += r;
            }
            grab(self.from + li, x)?;
        }
        Ok(())
    }
}

/// The prompt through the text embedding (on `gpu`), float32 [rows, 1536] on the host
pub fn text_embedding(st: &SafeTensors, gpu: &Arc<Gpu>, ctx: &[f32]) -> Result<Vec<f32>> {
    let rows = ctx.len() / TEXT;
    let l0 = Lin::from(gpu, &need(st, "text_embedding.0.weight")?, Some(&need(st, "text_embedding.0.bias")?), DIM, TEXT)?;
    let l2 = Lin::from(gpu, &need(st, "text_embedding.2.weight")?, Some(&need(st, "text_embedding.2.bias")?), DIM, DIM)?;
    let x = DevBuf::new(gpu, rows * TEXT * 2)?;
    x.write(0, &half_bytes(ctx))?;
    let h = DevBuf::new(gpu, rows * DIM * 2)?;
    let o = DevBuf::new(gpu, rows * DIM * 4)?;
    l0.half(gpu, x.ptr(), TEXT, rows, h.ptr(), DIM, Act::Gelu)?;
    l2.float(gpu, h.ptr(), rows, o.fp())?;
    gpu.sync()?;
    o.to_f32()
}

/// Wan's 3-D RoPE as two small tables (fv.h, rms_rope): each patch's (cos, sin) for the 42 row and column pairs, made once
/// a grid, and each frame's for the 22 frame pairs, made each chunk - rather than one a token (a chunk's 22 MB at
/// 1792 x 1024). The angles in float64, reduced to [0, 2 pi) before the cosine (positions grow without end in a stream).
pub struct Rope;

impl Rope {
    /// Wan's split of a head's 64 pairs: 22 for the frame, 21 for the row, 21 for the column, each its own frequencies
    fn axis(dim: usize) -> Vec<f64> {
        (0..dim / 2).map(|i| 1.0 / 10000f64.powf((2 * i) as f64 / dim as f64)).collect()
    }

    fn put(out: &mut Vec<f32>, a: f64) {
        let a = a.rem_euclid(std::f64::consts::TAU);
        out.push(a.cos() as f32);
        out.push(a.sin() as f32);
    }

    /// floats of the patches' table: [hp wp, 42, 2]
    pub fn hw_len(hp: usize, wp: usize) -> usize {
        hp * wp * 42 * 2
    }

    /// floats of both, for a chunk of at most 6 frames
    pub fn len(hp: usize, wp: usize) -> usize {
        Self::hw_len(hp, wp) + 6 * 22 * 2
    }

    /// The patches' table [hp wp, 42, 2] (patch y * wp + x)
    pub fn hw(hp: usize, wp: usize) -> Vec<f32> {
        let (fh, fw) = (Self::axis(42), Self::axis(42));
        let mut v = Vec::with_capacity(Self::hw_len(hp, wp));
        for y in 0..hp {
            for x in 0..wp {
                fh.iter().for_each(|f| Self::put(&mut v, y as f64 * f));
                fw.iter().for_each(|f| Self::put(&mut v, x as f64 * f));
            }
        }
        v
    }

    /// The frames' table [f, 22, 2], frames t0 .. t0 + f
    pub fn time(f: usize, t0: usize) -> Vec<f32> {
        let ft = Self::axis(44);
        let mut v = Vec::with_capacity(f * 44);
        for t in 0..f {
            ft.iter().for_each(|fr| Self::put(&mut v, (t0 + t) as f64 * fr));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_reuses_the_slot_no_query_sees() {
        // chunk n's pair is pair n + 2 of the stream (the first holds pairs 0-2); it sees pairs n - 1 .. n + 2
        for n in 1..20 {
            let c = Chunk::of(n, 8, 14, 10, 11);
            assert_eq!(c.slot, (n + 2) % SLOTS);
            // the slot it overwrites held pair n - 2, which chunk n no longer sees
            assert_eq!((n + SLOTS - 2) % SLOTS, c.slot);
            assert_eq!(c.seen(), 4 * 112);
        }
        let c0 = Chunk::of(0, 8, 14, 10, 11);
        assert_eq!((c0.groups, c0.slot, c0.seen(), c0.rows()), (3, 0, 3 * 112, 3 * 112 * 128));
    }

    #[test]
    fn top_k_is_the_references() {
        // 1792 x 1024 (1728 x 960 padded): nb = 8 x 14 = 112, sparse_ratio 2.0 and 1.5
        // (python: int(12544 * (r * 768 * 1280 / (1024 * 1792))) - 1)
        assert_eq!(topk(112, 1024, 1792, 2.0), 13439);
        assert_eq!(topk(112, 1024, 1792, 1.5), 10079);
        // the rank is capped by the scores there are (a small grid: everything but the least)
        let c = Chunk::of(0, 1, 1, 100, 11);
        assert_eq!(c.rank(), 3);
    }

    #[test]
    fn the_rope_tables_are_wans() {
        // frame 5's first pair rotates by 5 (frequency 1), its last (21) by 5 / 10000^(42 / 44)
        let t = Rope::time(2, 4);
        assert!((t[22 * 2] - 5f64.cos() as f32).abs() < 1e-6 && (t[22 * 2 + 1] - 5f64.sin() as f32).abs() < 1e-6);
        let last = 5.0 / 10000f64.powf(42.0 / 44.0);
        assert!((t[43 * 2] - last.cos() as f32).abs() < 1e-6);
        // patch (y 1, x 2) of an 8 x 8 grid: its row pairs (0-20) rotate by 1, its column pairs (21-41) by 2
        let hw = Rope::hw(8, 8);
        let p = (8 + 2) * 42 * 2;
        assert!((hw[p] - 1f64.cos() as f32).abs() < 1e-6 && (hw[p + 21 * 2] - 2f64.cos() as f32).abs() < 1e-6);
        assert_eq!(hw.len(), Rope::hw_len(8, 8));
        // a long stream's positions stay exact (reduced before the cosine)
        let far = Rope::time(1, 100_000);
        assert!((far[0] - 100_000f64.cos() as f32).abs() < 1e-6);
    }
}
