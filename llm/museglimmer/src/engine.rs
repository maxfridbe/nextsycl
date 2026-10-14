//! Muse Glimmer 30B (`muse-glimmer`, llama.cpp's src/models/muse-glimmer.cpp): 52 dense layers, each gated attention
//! (32 query / 2 key heads of 128, q / k norms, a sigmoid gate from its own projection between the attention and the
//! output projection) and a SwiGLU, both followed by their own RMS norm (eps 1e-8) before the residual. Three of every
//! four layers attend over a sliding window (2,048) with rotary positions (llama.cpp's NORM: adjacent pairs); the
//! fourth over everything, without positions. The token embedding is RMS-normed (no weight) before the first layer;
//! the logits are scaled, then soft-capped (c tanh(x / c)). The weights stay in their GGUF blocks on the GPU: decode
//! reads them through the Q8_1 products (up to 8 rows), a prompt chunk expands each matrix to half and multiplies.
//!
//! ```text
//!   x = rms(embed(tokens))
//!   per layer:  x += post_attn_norm(attn(attn_norm(x)));  x += post_ffw_norm(ffn(ffn_norm(x)))
//!   logits = softcap(scale * output(output_norm(x)))
//! ```

use std::ffi::c_int;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Gpu};
use nextsycl_gguf::{GType, Gguf, Tensor};
use nextsycl_llm::{Error, GpuInfo, Result, Sampler, Tap};

use crate::ffi;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

/// The model's shape, from the file's metadata
#[derive(Clone, Debug)]
pub struct Geometry {
    pub layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub heads: usize,
    pub heads_kv: usize,
    pub head: usize,
    pub theta: f32,
    pub eps: f32,
    /// the sliding window (positions a windowed layer's row sees, itself included)
    pub window: usize,
    /// each layer: windowed (rotary positions) or full (none)
    pub swa: Vec<bool>,
    pub logit_scale: f32,
    pub softcap: f32,
    pub vocab: usize,
    pub ctx: usize,
}

/// The most rows the silo's decode path takes in a pass (a verify pass of the drafter's block); prompt rows beyond
const MAXV: usize = 16;

/// The post-attention and post-FFN norms' epsilon (llama.cpp's graph: not the file's)
const POST_EPS: f32 = 1e-8;

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let layers = u("block_count")?;
        // the pattern as an array of flags, else a period (the last of each period full), else 4
        let swa = match f.arch_meta("attention.sliding_window_pattern") {
            Some(v) if v.as_array().is_some() => {
                let a = v.as_array().unwrap_or(&[]);
                let s: Vec<bool> = a.iter().map(|x| x.as_u64().unwrap_or(0) != 0).collect();
                if s.len() != layers {
                    return Err(format!("attention.sliding_window_pattern has {} flags for {layers} layers", s.len()));
                }
                s
            }
            other => {
                let n = other.and_then(|v| v.as_u64()).unwrap_or(4) as usize;
                (0..layers).map(|l| n == 0 || l % n < n - 1).collect()
            }
        };
        let g = Geometry {
            layers,
            n_embd: u("embedding_length")?,
            n_ff: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            heads_kv: u("attention.head_count_kv")?,
            head: u("attention.key_length")?,
            theta: fl("rope.freq_base")?,
            eps: fl("attention.layer_norm_rms_epsilon")?,
            window: u("attention.sliding_window")?,
            swa,
            logit_scale: fl("logit_scale").unwrap_or(1.0),
            softcap: fl("final_logit_softcapping").unwrap_or(0.0),
            // shapes are outermost first: [vocab, n_embd]
            vocab: f.tensor("token_embd.weight").map(|t| t.shape[0] as usize).ok_or("no token_embd.weight")?,
            ctx: u("context_length").unwrap_or(131_072),
        };
        if g.head != 128 || !g.heads.is_multiple_of(g.heads_kv) || !(g.heads / g.heads_kv).is_multiple_of(4) {
            return Err(format!("attention heads of {} ({} / {}): head 128 with a multiple of 4 query heads a key head", g.head, g.heads, g.heads_kv));
        }
        if f.arch_meta("rope.dimension_count").and_then(|v| v.as_u64()).is_some_and(|n| n as usize != g.head) {
            return Err("rotary positions on part of a head: not this engine's".into());
        }
        Ok(g)
    }

    /// A layer's cache slots for a context of `max_ctx` and passes of `chunk` rows: a windowed layer's ring holds its
    /// window and a pass
    pub fn slots(&self, l: usize, max_ctx: usize, chunk: usize) -> usize {
        if self.swa[l] { max_ctx.min(self.window + chunk - 1) } else { max_ctx }
    }

    /// The window a layer's rows see (0: all)
    pub fn win(&self, l: usize) -> usize {
        if self.swa[l] { self.window } else { 0 }
    }
}

/// A matrix [rows (outputs), cols (inputs)]: its stored blocks, or float32 (small or unquantized tensors)
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
    /// Q8_0 repacked by the silo: `buf` its quants [rows][cols], these its scales [rows][cols / 32] (a view of rows
    /// of a matrix several share)
    pub scales: Option<DevBuf>,
}

impl Mat {
    fn packed(&self) -> bool {
        self.scales.is_some()
    }
}

struct Layer {
    /// q | k | v | gate and the FFN's gate | up repacked as one matrix each (the silo; a decode row's one product);
    /// the matrices below are views of their rows
    qkvg: Option<Mat>,
    gu: Option<Mat>,
    /// the memory the views are of
    _own: Vec<DevBuf>,
    attn_norm: DevBuf,
    q: Mat,
    k: Mat,
    v: Mat,
    gate: Mat,
    o: Mat,
    q_norm: DevBuf,
    k_norm: DevBuf,
    post_attn: DevBuf,
    ffn_norm: DevBuf,
    ffn_gate: Mat,
    ffn_up: Mat,
    ffn_down: Mat,
    post_ffn: DevBuf,
}

/// A conversation: per layer its key / value caches ([kv heads][slots][128] half, slot = position % slots); with the
/// drafter its layers' caches too (the target's features injected at each position, rings of the window and a pass)
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<(DevBuf, DevBuf, usize)>,
    draft: Vec<(DevBuf, DevBuf, usize)>,
    /// rows of the last pass (a rollback keeps some of them)
    last_rows: usize,
}

/// The DFlash drafter (llama.cpp's src/models/dflash.cpp, common/speculative.cpp; its own GGUF beside the model):
/// five layers that draft a block at once. Each position the target passes over leaves its hidden states going into
/// the target layers `targets` - fc, a norm, then each layer's k (normed, NEOX rotation) and v into its cache. A draft
/// is one pass of [the last token, MASK x (block - 1)] attending to that context and to the whole block (not causal),
/// through the target's embedding and head: the block's argmaxes. Its K-quant matrices are requantized to the silo's
/// repacked Q8 at load, so a block's 16 rows take the silo's products.
struct Dflash {
    layers: Vec<DLayer>,
    fc: Mat,
    enc_norm: DevBuf,
    norm: DevBuf,
    targets: Vec<usize>,
    heads: usize,
    heads_kv: usize,
    window: usize,
    theta: f32,
    eps: f32,
    block: usize,
    mask: u32,
    /// the GPU it runs on (the target's last: its head)
    dev: usize,
}

struct DLayer {
    attn_norm: DevBuf,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: DevBuf,
    k_norm: DevBuf,
    ffn_norm: DevBuf,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// NS_MG_DRAFTS: tokens drafted a step (at most the block's less one; 0: none)
fn draft_count() -> Option<usize> {
    std::env::var("NS_MG_DRAFTS").ok().and_then(|v| v.parse().ok())
}

/// NS_MG_DRAFT: the drafter's GGUF; else the dflash-*.gguf beside the model (its smallest)
fn draft_path(f: &Gguf) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NS_MG_DRAFT") {
        return Some(p.into());
    }
    let dir = f.paths.first()?.parent()?;
    let mut v: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("dflash-") && n.ends_with(".gguf"))).collect();
    v.sort_by_key(|p| std::fs::metadata(p).map_or(u64::MAX, |m| m.len()));
    v.into_iter().next()
}

/// A conversation's state in host memory: per layer its caches' positions still seen (the window's, or all)
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(b"NSMG1")?;
        u(w, self.pos)?;
        u(w, self.bufs.len())?;
        for b in &self.bufs {
            u(w, b.len())?;
            w.write_all(b)?;
        }
        Ok(())
    }

    pub fn read_from(r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        let u = |r: &mut dyn std::io::Read| -> std::io::Result<usize> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b) as usize)
        };
        let mut magic = [0u8; 5];
        r.read_exact(&mut magic)?;
        if &magic != b"NSMG1" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a museglimmer checkpoint"));
        }
        let pos = u(r)?;
        let n = u(r)?;
        let mut bufs = Vec::with_capacity(n);
        let mut bytes = 0;
        for _ in 0..n {
            let len = u(r)?;
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            bytes += len;
            bufs.push(v);
        }
        Ok(Checkpoint { pos, bufs, bytes })
    }
}

/// Wall time of a step's parts (ns): the verify pass, the injection in it, the draft - reported after a run
static T_VERIFY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static T_INJECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static T_DRAFT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn tick(c: &std::sync::atomic::AtomicU64, t0: Instant) {
    c.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// A pass's result: its rows' logits, or their argmaxes
enum Out {
    Logits(Vec<Vec<f32>>),
    Ids(Vec<u32>),
}

/// Generation from a fed prompt: the logits to draw the next token from, or the committed token not fed yet
pub struct Decoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
    /// draft with the DFlash drafter
    draft: bool,
    /// the drafts for the positions after `next`
    drafts: Vec<u32>,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn new(logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder { logits: Some(logits), next: None, draft, drafts: Vec::new(), drafted: 0, accepted: 0 }
    }
    pub fn after(next: u32, draft: bool) -> Decoder {
        Decoder { logits: None, next: Some(next), draft, drafts: Vec::new(), drafted: 0, accepted: 0 }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.drafts.clear();
        self.next.take()
    }
}

/// One GPU's share: its layers [lb, le), a prompt chunk's work memory and the half buffer its products expand into
struct Dev {
    gpu: Arc<Gpu>,
    arena: Arena,
    /// the largest matrix expanded to half (a prompt chunk's products)
    w16: DevBuf,
    lb: usize,
    le: usize,
}

pub struct MuseGlimmer<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    /// the GPUs in layer order (the first embeds, the last holds the head)
    devs: Vec<Dev>,
    /// each layer's GPU (an index into devs)
    layer_dev: Vec<usize>,
    /// the GPU the kernels go to now (set under `work`: a pass walks the GPUs in order)
    cur: std::sync::atomic::AtomicUsize,
    /// host memory a pass's rows cross between GPUs through
    staging: Mutex<Vec<u8>>,
    k: &'static ffi::Api,
    /// the silo's tuned kernels (kernels/llm/museglimmer/silo), None: the shared ones throughout
    silo: Option<&'static ffi::Silo>,
    layers: Vec<Layer>,
    /// ones for the token embedding's RMS norm (no weight)
    ones: DevBuf,
    output_norm: DevBuf,
    output: Mat,
    /// the token embedding's rows stay in the file (read a row a token)
    embd: Tensor,
    /// the rows a forward pass takes at most
    chunk: usize,
    work: Mutex<()>,
    /// NS_MG_PROFILE=1: each kernel's time and calls
    prof: Option<Mutex<std::collections::BTreeMap<&'static str, (f64, u64)>>>,
    /// the DFlash drafter (drafts on, its file found, the silo there) and the tokens it drafts a step
    dflash: Option<Dflash>,
    drafts: usize,
    load_s: f64,
    load_bytes: u64,
}

/// NS_MG_CHUNK when set; otherwise chosen at load (2,048 rows when they fit)
fn chunk_setting() -> Option<usize> {
    std::env::var("NS_MG_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= 16)
}

/// A prompt chunk's work memory (the arena) for `chunk` rows
fn arena_bytes(g: &Geometry, chunk: usize) -> usize {
    let qd = g.heads * g.head;
    chunk * (5 * g.n_embd + 4 * qd + 2 * g.heads_kv * g.head * 2 + 2 * g.n_ff) * 4 + chunk * g.n_ff.max(g.n_embd).max(qd) * 2 + 8 * g.vocab * 4
        + (64 << 20)
}

/// The bytes of a layer's caches for `max_ctx` and passes of `chunk` rows
fn cache_bytes(g: &Geometry, l: usize, max_ctx: usize, chunk: usize) -> usize {
    2 * g.heads_kv * g.slots(l, max_ctx, chunk) * g.head * 2
}

impl<'g> MuseGlimmer<'g> {
    /// Loads the model over `gpus` in order: with one GPU all of it there; with more, the layers split so each GPU keeps
    /// room for its layers' caches (NS_SPLIT: the first layer of each GPU after the first), the head on the last. `kv`:
    /// the context the sessions reserve (tokens in all; 0: 32K).
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), draft: bool, log: &mut dyn FnMut(String)) -> Result<MuseGlimmer<'g>> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(e("no GPU"));
        }
        let g = Geometry::read(f).map_err(e)?;
        let k = ffi::api()?;
        let silo = ffi::silo()?;
        log(match silo {
            Some(s) => format!("museglimmer: tuned kernels from its silo {} (Q8_0 repacked)", s.path.display()),
            None => "museglimmer: no silo - the shared kernels".into(),
        });
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        let weights: u64 = f.tensors.iter().filter(|t| t.name != "token_embd.weight").map(|t| t.bytes).sum();
        let layer_w: Vec<u64> = (0..g.layers).map(|l| {
            let p = format!("blk.{l}.");
            f.tensors.iter().filter(|t| t.name.starts_with(&p)).map(|t| t.bytes).sum()
        }).collect();
        let head_w = weights - layer_w.iter().sum::<u64>();
        // the drafter: drafts on, the silo there (its products), its file found
        let dfile: Option<Box<Gguf>> = match (draft && draft_count() != Some(0), silo, draft_path(f)) {
            (true, Some(_), Some(p)) => match Gguf::open(&p) {
                Ok(d) if d.architecture() == "dflash" => {
                    log(format!("museglimmer: the DFlash drafter {}", p.display()));
                    Some(Box::new(d))
                }
                Ok(_) => {
                    log(format!("museglimmer: {} is not a DFlash drafter; no drafts", p.display()));
                    None
                }
                Err(x) => {
                    log(format!("museglimmer: the drafter {} does not open ({}); no drafts", p.display(), x.0));
                    None
                }
            },
            (true, None, _) => {
                log("museglimmer: drafts need the silo; none".into());
                None
            }
            _ => None,
        };
        // its matrices requantized to Q8 (about 1.07 bytes a weight), its caches, the features of a chunk
        let d_elems: u64 = dfile.as_ref().map_or(0, |d| d.tensors.iter().filter(|t| t.shape.len() == 2).map(|t| t.shape.iter().product::<u64>()).sum());
        let d_targets = dfile.as_ref().and_then(|d| d.arch_meta("target_layers")).and_then(|v| v.as_array()).map_or(0, |a| a.len());
        let draft_w = d_elems * 17 / 16 + if dfile.is_some() { 128 << 20 } else { 0 };
        let w16_bytes = [g.n_ff * g.n_embd, g.heads * g.head * g.n_embd, d_targets * g.n_embd * g.n_embd].into_iter().max().unwrap_or(0) * 2;
        // a pass's target features and the injection's work beside the arena
        let feat = |c: usize| if d_targets > 0 { c * (d_targets + 4) * g.n_embd * 4 } else { 0 };
        let mut free = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let (total, fr) = gpu.memory()?;
            free.push(fr.unwrap_or(total));
        }
        let margin: u64 = 1536 << 20;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let ctx = if kv.0 > 0 { kv.0 } else { 32768 };
        let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        // the drafter's GPU: with two or more the first (the server orders them weakest first: the target's layers
        // stay on the faster card, a slower one's XMX ran a 16-row verify ~3x slower), else the only one
        let d_on = if gpus.len() > 1 { 0 } else { gpus.len() - 1 };
        // the plan for a chunk: the last GPU first (the server orders them weakest first), each taking the last layers
        // left while they fit; the first the rest
        let plan = |c: usize| -> std::result::Result<Vec<(usize, usize)>, String> {
            let n = gpus.len();
            let cost: Vec<u64> = (0..g.layers).map(|l| layer_w[l] + cache_bytes(&g, l, ctx, c) as u64).collect();
            let room: Vec<i64> = (0..n).map(|i| {
                let mut over = margin + (512 << 20) + w16_bytes as u64 + (arena_bytes(&g, c) + feat(c)) as u64;
                if i + 1 == n {
                    over += head_w;
                }
                if i == d_on {
                    over += draft_w;
                }
                free[i] as i64 - over as i64
            }).collect();
            let bounds = if let Some(sp) = &split_env {
                let mut b = Vec::with_capacity(n);
                let mut lb = 0;
                for i in 0..n {
                    let le = if i + 1 == n { g.layers } else { *sp.get(i).ok_or("NS_SPLIT: a first layer for each GPU after the first")? };
                    b.push((lb, le.min(g.layers)));
                    lb = le.min(g.layers);
                }
                b
            } else {
                let mut le = g.layers;
                let mut rev = Vec::with_capacity(n);
                for i in (0..n).rev() {
                    let mut lb = le;
                    if i == 0 {
                        lb = 0;
                    } else {
                        let mut have = room[i];
                        while lb > 0 && cost[lb - 1] as i64 <= have {
                            have -= cost[lb - 1] as i64;
                            lb -= 1;
                        }
                    }
                    rev.push((lb, le));
                    le = lb;
                }
                rev.reverse();
                rev
            };
            for (i, &(lb, le)) in bounds.iter().enumerate() {
                if le <= lb {
                    if i + 1 < n && split_env.is_none() {
                        continue;
                    }
                    return Err(format!("{} gets no layers", gpus[i].name));
                }
                let need: u64 = cost[lb..le].iter().sum();
                if need as i64 > room[i] {
                    return Err(format!("layers {lb}-{} need {:.1} GiB on {}; {:.1} GiB are left for them", le - 1, gib(need), gpus[i].name, gib(room[i].max(0) as u64)));
                }
            }
            Ok(bounds)
        };
        let tries: Vec<usize> = chunk_setting().map_or_else(|| vec![2048, 1024, 512], |c| vec![c]);
        let mut last_err = String::new();
        let mut found = None;
        for c in tries {
            match plan(c) {
                Ok(b) => {
                    found = Some((c, b));
                    break;
                }
                Err(x) => last_err = x,
            }
        }
        let (chunk, bounds) = found.ok_or_else(|| e(format!("{} over {} GPUs at a {ctx}-token context: {last_err}", f.architecture(), gpus.len())))?;
        let given = gpus.len();
        // a GPU the layers do not reach is not used, unless the drafter is on it
        let keep: Vec<usize> = (0..gpus.len()).filter(|&i| bounds[i].1 > bounds[i].0 || (dfile.is_some() && i == d_on)).collect();
        let d_dev = keep.iter().position(|&i| i == d_on).unwrap_or(keep.len() - 1);
        let (gpus, bounds): (Vec<Arc<Gpu>>, Vec<(usize, usize)>) = keep.iter().map(|&i| (gpus[i].clone(), bounds[i])).unzip();
        if gpus.len() < given {
            log(format!("museglimmer: all of it fits {}: {} of the {given} GPUs not used", gpus.iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(" + "),
                        given - gpus.len()));
        }
        let mut layer_dev = vec![0usize; g.layers];
        for (i, &(lb, le)) in bounds.iter().enumerate() {
            for d in &mut layer_dev[lb..le] {
                *d = i;
            }
        }
        let lastg = gpus[gpus.len() - 1].clone();
        let mut loaded = 0u64;
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let gpu = &gpus[layer_dev[l]];
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            // with the silo: q | k | v | gate and gate | up stacked, repacked once each
            let (mut own, mut qkvg, mut gu) = (Vec::new(), None, None);
            let mut attn_parts = None;
            let mut ffn_parts = None;
            if let Some(sl) = silo {
                if let Some((o, whole, parts)) = pack_stack(f, gpu, sl, &[n("attn_q"), n("attn_k"), n("attn_v"), n("attn_gate")], &mut loaded)? {
                    own.push(o);
                    qkvg = Some(whole);
                    attn_parts = Some(parts);
                }
                if let Some((o, whole, parts)) = pack_stack(f, gpu, sl, &[n("ffn_gate"), n("ffn_up")], &mut loaded)? {
                    own.push(o);
                    gu = Some(whole);
                    ffn_parts = Some(parts);
                }
            }
            let mut read = |name: &str| -> Result<Mat> { vec_mat(f, gpu, k, silo, name, &mut loaded) };
            let [q, kk, v, gate] = match attn_parts {
                Some(p) => <[Mat; 4]>::try_from(p).map_err(|_| e("q | k | v | gate"))?,
                None => [read(&n("attn_q"))?, read(&n("attn_k"))?, read(&n("attn_v"))?, read(&n("attn_gate"))?],
            };
            let [ffn_gate, ffn_up] = match ffn_parts {
                Some(p) => <[Mat; 2]>::try_from(p).map_err(|_| e("gate | up"))?,
                None => [read(&n("ffn_gate"))?, read(&n("ffn_up"))?],
            };
            layers.push(Layer {
                qkvg,
                gu,
                _own: own,
                attn_norm: read(&n("attn_norm"))?.buf,
                q,
                k: kk,
                v,
                gate,
                o: read(&n("attn_output"))?,
                q_norm: read(&n("attn_q_norm"))?.buf,
                k_norm: read(&n("attn_k_norm"))?.buf,
                post_attn: read(&n("post_attention_norm"))?.buf,
                ffn_norm: read(&n("ffn_norm"))?.buf,
                ffn_gate,
                ffn_up,
                ffn_down: read(&n("ffn_down"))?,
                post_ffn: read(&n("post_ffw_norm"))?.buf,
            });
            if l % 13 == 12 {
                log(format!("museglimmer: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, gib(loaded), t0.elapsed().as_secs_f64()));
            }
        }
        let output_norm = vec_mat(f, &lastg, k, silo, "output_norm.weight", &mut loaded)?.buf;
        let output = vec_mat(f, &lastg, k, silo, if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }, &mut loaded)?;
        let dflash = match &dfile {
            Some(d) => Some(load_dflash(d, &g, &gpus[d_dev], d_dev, k, silo.ok_or_else(|| e("drafts need the silo"))?, &mut loaded)?),
            None => None,
        };
        let drafts = dflash.as_ref().map_or(0, |d| draft_count().unwrap_or(d.block - 1).min(d.block - 1));
        if let Some(d) = &dflash {
            log(format!("museglimmer: DFlash drafts ({drafts} a step; {} layers, the target's layers {:?})", d.layers.len(), d.targets));
        }
        let ones = DevBuf::f32(&gpus[0], g.n_embd)?;
        ones.write(0, &vec![1.0f32; g.n_embd].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let mut devs = Vec::with_capacity(gpus.len());
        for (gpu, &(lb, le)) in gpus.iter().zip(&bounds) {
            gpu.sync()?;
            devs.push(Dev { gpu: gpu.clone(), w16: DevBuf::new(gpu, w16_bytes)?, arena: Arena::new(gpu, arena_bytes(&g, chunk) + feat(chunk))?, lb, le });
        }
        let load_s = t0.elapsed().as_secs_f64();
        let wh = devs.iter().map(|d| if d.lb == 0 && d.le == g.layers { d.gpu.name.clone() } else { format!("{} (layers {}-{})", d.gpu.name, d.lb, d.le - 1) })
            .collect::<Vec<_>>().join(", ");
        log(format!("museglimmer: {} layers of {} ({} windowed), {:.1} GiB on {wh} in {load_s:.0} s; prompt chunks of {chunk}", g.layers, g.n_embd,
                    g.swa.iter().filter(|s| **s).count(), gib(loaded)));
        Ok(MuseGlimmer { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(Vec::new()), k, silo, layers, ones, output_norm, output, embd, chunk,
                         work: Mutex::new(()), prof: std::env::var("NS_MG_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(Default::default())), dflash,
                         drafts, load_s,
                         load_bytes: loaded })
    }

    /// A kernel call, its return code checked (NS_MG_PROFILE=1: timed, the queue drained before and after)
    fn op(&self, name: &'static str, f: impl FnOnce() -> c_int) -> Result<()> {
        let Some(p) = &self.prof else { return ffi::check(f(), name) };
        self.dev().gpu.sync()?;
        let t0 = Instant::now();
        ffi::check(f(), name)?;
        self.dev().gpu.sync()?;
        let mut m = p.lock().unwrap();
        let ent = m.entry(name).or_insert((0.0, 0));
        ent.0 += t0.elapsed().as_secs_f64();
        ent.1 += 1;
        Ok(())
    }

    /// The profile's lines (`report`): each kernel's time in all and a token
    pub fn profile_lines(&self, tokens: usize) -> Vec<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let steps = STEPS.load(Relaxed).max(1) as f64;
        let ms = |c: &std::sync::atomic::AtomicU64| c.load(Relaxed) as f64 / 1e6 / steps;
        let mut out = vec![format!("[steps {}: verify pass {:.1} ms (its injection {:.1}), draft {:.1} ms a step]", STEPS.load(Relaxed), ms(&T_VERIFY), ms(&T_INJECT),
                                   ms(&T_DRAFT))];
        let Some(p) = &self.prof else { return out };
        let m = p.lock().unwrap();
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        out.extend(v.iter().map(|(n, (s, c))| format!("[profile {n:<16} {s:>8.3} s {c:>8} calls {:>8.3} ms/token]", s * 1000.0 / (tokens + 1) as f64)));
        out
    }

    pub fn load_seconds(&self) -> f64 {
        self.load_s
    }
    pub fn load_bytes(&self) -> u64 {
        self.load_bytes
    }
    pub fn prefill_chunk(&self) -> usize {
        self.chunk
    }

    /// The GPU the kernels go to now
    fn dev(&self) -> &Dev {
        &self.devs[self.cur.load(std::sync::atomic::Ordering::Relaxed)]
    }

    fn on(&self, i: usize) {
        self.cur.store(i, std::sync::atomic::Ordering::Relaxed);
    }

    fn arena(&self) -> &Arena {
        &self.dev().arena
    }

    fn raw(&self) -> *mut std::ffi::c_void {
        self.dev().gpu.raw()
    }

    /// x (rows of the GPU it is on) onto GPU `to`: a buffer of that GPU's arena
    fn hop(&self, x: &DevBuf, to: usize) -> Result<DevBuf> {
        self.on(to);
        let y = self.arena().bytes(x.len)?;
        let mut st = self.staging.lock().unwrap();
        if st.len() < x.len {
            st.resize(x.len, 0);
        }
        y.copy_from_peer(x, &mut st)?;
        Ok(y)
    }

    /// y [t, w.rows] = x [t, w.cols] . W^T; `xq`: x already in the silo's split Q8 layout (decode rows)
    fn matmul(&self, w: &Mat, x: &DevBuf, xq: Option<&DevBuf>, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        let (rows, cols) = (w.rows, w.cols);
        if w.ty == GType::F32 {
            // SAFETY: x [t, cols], the matrix [rows, cols] float32, y [t, rows].
            return self.op("gemm", || unsafe { (k.gemm)(self.raw(), t as i64, n, kk, x.fp(), kk, w.buf.fp(), y.fp(), n, 0) });
        }
        if let (Some(sc), Some(s)) = (&w.scales, self.silo) {
            if t <= MAXV {
                let own;
                let xqb = match xq {
                    Some(q) => q,
                    None => {
                        // SAFETY: arithmetic only.
                        own = self.arena().bytes(unsafe { (s.xq_bytes)(kk, t as i64) })?;
                        // SAFETY: x [t, cols] -> its split Q8 form.
                        self.op("xq silo", || unsafe { (s.xq)(self.raw(), x.fp(), std::ptr::null(), 0, own.ptr(), kk, t as i64) })?;
                        &own
                    }
                };
                // SAFETY: the repacked matrix (quants, scales), x's split form, y [t, rows].
                return self.op("mmvq silo", || unsafe { (s.mmvq)(self.raw(), w.buf.ptr(), sc.ptr(), xqb.ptr(), y.fp(), kk, n, t as i64) });
            }
            if rows * cols * 2 > self.dev().w16.len {
                return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
            }
            // SAFETY: the repacked matrix -> rows x cols halfs in w16.
            self.op("q8r f16 silo", || unsafe { (s.q8r_f16)(self.raw(), w.buf.ptr(), sc.ptr(), n, kk, self.dev().w16.ptr().cast()) })?;
            let xh = self.arena().bytes(t * cols * 2)?;
            // SAFETY: x [t, cols] -> half; then y [t, rows] = xh . w16^T.
            self.op("x to half", || unsafe { (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (t * cols) as i64) })?;
            return self.op("gemm f16", || unsafe { (k.gemm_f16)(self.raw(), t as i64, n, kk, xh.ptr().cast(), kk, self.dev().w16.ptr().cast(), y.fp(), n, 0) });
        }
        // SAFETY: arithmetic only.
        let mmvq = unsafe { (k.mmvq_supported)(w.ty.code() as i32) } != 0;
        if t <= 8 && mmvq {
            // SAFETY: arithmetic only.
            let q8 = unsafe { (k.q8_1_bytes)(kk, t as i64) };
            let qb = self.arena().bytes(q8)?;
            // SAFETY: x [t, cols] float32 -> its Q8_1 blocks; then the product from the stored blocks.
            self.op("quantize", || unsafe { (k.quantize_q8_1)(self.raw(), x.fp(), qb.ptr(), kk, t as i64) })?;
            return self.op("mmvq", || unsafe { (k.mmvq)(self.raw(), w.ty.code() as i32, w.buf.ptr(), qb.ptr(), y.fp(), kk, n, t as i64) });
        }
        // the matrix expanded to half (directly, or through float32), x to half, then the half product
        let elems = rows * cols;
        if elems * 2 > self.dev().w16.len {
            return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
        }
        // SAFETY: the matrix's blocks -> rows x cols halfs in w16 (non-zero: a type it does not expand directly).
        let rc = unsafe { (k.dequant_f16)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems as i64, self.dev().w16.ptr().cast()) };
        if rc != 0 {
            let f = self.arena().f32(elems)?;
            // SAFETY: the blocks -> float32, then -> half.
            self.op("dequant", || unsafe { (k.dequant)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems, f.fp()) })?;
            self.op("to half", || unsafe { (k.to_f16)(self.raw(), f.fp(), self.dev().w16.ptr().cast(), elems as i64) })?;
        }
        let xh = self.arena().bytes(t * cols * 2)?;
        // SAFETY: x [t, cols] -> half; then y [t, rows] = xh . w16^T.
        self.op("x to half", || unsafe { (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (t * cols) as i64) })?;
        self.op("gemm f16", || unsafe { (k.gemm_f16)(self.raw(), t as i64, n, kk, xh.ptr().cast(), kk, self.dev().w16.ptr().cast(), y.fp(), n, 0) })
    }

    fn rms(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize, eps: f32) -> Result<()> {
        // SAFETY: x and y [rows, c], w c floats.
        self.op("rms norm", || unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, eps) })
    }

    /// The tokens' embeddings [t, n_embd] (their rows read from the file, expanded on the GPU)
    fn embed(&self, tokens: &[u32]) -> Result<DevBuf> {
        let g = &self.g;
        let row = self.embd.ty.bytes(g.n_embd as u64).ok_or_else(|| e("the embedding's row size"))? as usize;
        let mut raw = vec![0u8; tokens.len() * row];
        for (i, t) in tokens.iter().enumerate() {
            if *t as usize >= g.vocab {
                return Err(e(format!("token {t} past the vocabulary ({})", g.vocab)));
            }
            self.file.read_into(&self.embd, *t as u64 * row as u64, &mut raw[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
        }
        let src = self.arena().bytes(raw.len())?;
        src.write(0, &raw)?;
        let x = self.arena().f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32.
        self.op("dequant", || unsafe { (self.k.dequant)(self.raw(), self.embd.ty.code() as i32, src.ptr(), tokens.len() * g.n_embd, x.fp()) })?;
        Ok(x)
    }

    /// Layer `l`'s attention on h [t, n_embd] (normed; or `xq`, its split Q8 form, for decode rows with the silo): its
    /// output [t, n_embd], before the post norm
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, l: usize, a: &Layer, kc: &DevBuf, vc: &DevBuf, cap: usize, h: &DevBuf, xq: Option<&DevBuf>, t: usize, p0: usize, tap: Tap)
                 -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        let (qd, kd) = (hq * d, hk * d);
        let (q, kr, vr, gate) = match (&a.qkvg, xq) {
            (Some(all), Some(x)) => {
                // one product for the four (a row's results the same as theirs apart); more rows: each one's columns out
                let y = self.arena().f32(t * all.rows)?;
                self.matmul(all, h, Some(x), t, &y)?;
                if t == 1 {
                    (y.view(0, qd * 4)?, y.view(qd * 4, kd * 4)?, y.view((qd + kd) * 4, kd * 4)?, y.view((qd + 2 * kd) * 4, qd * 4)?)
                } else {
                    let parts = [(qd, 0), (kd, qd), (kd, qd + kd), (qd, qd + 2 * kd)].map(|(n, c0)| -> Result<DevBuf> {
                        let b = self.arena().f32(t * n)?;
                        // SAFETY: y [t, all.rows] -> b [t, n].
                        self.op("cols", || unsafe { (k.get_cols)(self.raw(), y.fp(), b.fp(), t as i64, n as i64, all.rows as i64, c0 as i64) })?;
                        Ok(b)
                    });
                    let [q, kr, vr, gate] = parts;
                    (q?, kr?, vr?, gate?)
                }
            }
            _ => {
                let (q, kr, vr, gate) = (self.arena().f32(t * qd)?, self.arena().f32(t * kd)?, self.arena().f32(t * kd)?, self.arena().f32(t * qd)?);
                self.matmul(&a.q, h, xq, t, &q)?;
                self.matmul(&a.k, h, xq, t, &kr)?;
                self.matmul(&a.v, h, xq, t, &vr)?;
                self.matmul(&a.gate, h, xq, t, &gate)?;
                (q, kr, vr, gate)
            }
        };
        let rope = g.swa[l] as c_int;
        let win = g.win(l) as i64;
        let att = self.arena().f32(t * qd)?;
        // SAFETY: the rows and caches as named in mg.h.
        unsafe {
            self.op("q norm rope", || (k.qk_norm_rope)(self.raw(), q.fp(), qd as i64, q.fp(), qd as i64, t as i64, hq as i64, d as i64, a.q_norm.fp(), g.eps,
                                                       rope, g.theta, p0 as i64))?;
            self.op("k norm rope", || (k.qk_norm_rope)(self.raw(), kr.fp(), kd as i64, kr.fp(), kd as i64, t as i64, hk as i64, d as i64, a.k_norm.fp(), g.eps,
                                                       rope, g.theta, p0 as i64))?;
            // llama.cpp's names: after the rotation on the windowed layers, after the norm on the full ones
            let stage = if g.swa[l] { "rope" } else { "normed" };
            tap(&format!("Qcur_{stage}-{l}"), &q)?;
            tap(&format!("Kcur_{stage}-{l}"), &kr)?;
            self.op("kv store", || (k.kv_store)(self.raw(), kr.fp(), vr.fp(), t as i64, hk as i64, d as i64, cap as i64, p0 as i64, kc.ptr(), vc.ptr()))?;
            match self.silo.filter(|s| (s.attn_prompt_supported)(t as i64, hq as i64, hk as i64, d as i64) != 0) {
                // the silo's flash attention on XMX (prompt rows)
                Some(s) => self.op("attention silo", || (s.attn_prompt)(self.raw(), q.fp(), kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64,
                                                                       cap as i64, p0 as i64, win, att.fp()))?,
                // and its split form for decode rows (a verify pass's: each row as its one-row pass)
                None if self.silo.is_some_and(|s| (s.attn_decode_supported)(t as i64, hq as i64, hk as i64, d as i64) != 0) => {
                    let s = self.silo.ok_or_else(|| e("the silo"))?;
                    let part = self.arena().f32((s.attn_decode_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64, win) as usize)?;
                    self.op("attention dec silo", || (s.attn_decode)(self.raw(), q.fp(), kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64,
                                                                     cap as i64, p0 as i64, win, att.fp(), part.fp()))?;
                }
                None => {
                    let sc = (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64, win, 1) as usize;
                    let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
                    self.op("attention", || (k.attn)(self.raw(), q.fp(), kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64, p0 as i64,
                                                     win, 1, att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
                }
            }
        }
        tap(&format!("attn_out-{l}"), &att)?;
        let out = self.arena().f32(t * g.n_embd)?;
        match self.silo.filter(|_| t <= MAXV && a.o.packed()) {
            Some(s) => {
                // the gate applied as the output projection's input is quantized
                // SAFETY: arithmetic only.
                let xo = self.arena().bytes(unsafe { (s.xq_bytes)(qd as i64, t as i64) })?;
                // SAFETY: att and gate [t, qd] -> att * sigmoid(gate) in the split Q8 form.
                self.op("gate xq silo", || unsafe { (s.xq)(self.raw(), att.fp(), gate.fp(), 1, xo.ptr(), qd as i64, t as i64) })?;
                self.matmul(&a.o, &att, Some(&xo), t, &out)?;
            }
            None => {
                // SAFETY: att and gate [t, qd].
                self.op("output gate", || unsafe { (k.gate_mul)(self.raw(), att.fp(), gate.fp(), (t * qd) as i64) })?;
                self.matmul(&a.o, &att, None, t, &out)?;
            }
        }
        Ok(out)
    }

    /// The post norm into the residual, then the next norm (`next`: its weight and eps) as `norm_in` makes it - one launch
    /// for decode rows with the silo, the split Q8 form into `into` when given (else the arena)
    #[allow(clippy::too_many_arguments)]
    fn post_norm(&self, out: &DevBuf, w: &DevBuf, x: &DevBuf, next: &DevBuf, eps: f32, h: &DevBuf, t: usize, into: Option<&DevBuf>)
                 -> Result<Option<DevBuf>> {
        let ne = self.g.n_embd;
        match self.silo.filter(|_| t <= MAXV) {
            Some(s) => {
                // SAFETY: arithmetic only.
                let bytes = unsafe { (s.xq_bytes)(ne as i64, t as i64) };
                let xq = match into {
                    Some(b) => b.view(0, bytes)?,
                    None => self.arena().bytes(bytes)?,
                };
                // SAFETY: out and x [t, ne], the weights ne floats -> x updated, the next norm's split Q8 form.
                self.op("post rms xq silo", || unsafe {
                    (s.post_rms_xq)(self.raw(), out.fp(), w.fp(), POST_EPS, x.fp(), next.fp(), eps, xq.ptr(), t as i64, ne as i64)
                })?;
                Ok(Some(xq))
            }
            None => {
                self.post(out, w, x, t)?;
                self.rms(x, next, h, t, ne, eps)?;
                Ok(None)
            }
        }
    }

    /// The post norm into the residual: x [t, n_embd] += rms(out) * w
    fn post(&self, out: &DevBuf, w: &DevBuf, x: &DevBuf, t: usize) -> Result<()> {
        // SAFETY: out and x [t, n_embd], w n_embd floats.
        self.op("post norm add", || unsafe { (self.k.rms_add)(self.raw(), out.fp(), w.fp(), x.fp(), t as i64, self.g.n_embd as i64, POST_EPS) })
    }

    /// x [t, n_embd] normed by w: as floats into h, or (decode rows with the silo) straight to its split Q8 form
    fn norm_in(&self, x: &DevBuf, w: &DevBuf, h: &DevBuf, t: usize) -> Result<Option<DevBuf>> {
        let ne = self.g.n_embd;
        match self.silo.filter(|_| t <= MAXV) {
            Some(s) => {
                // SAFETY: arithmetic only.
                let xq = self.arena().bytes(unsafe { (s.xq_bytes)(ne as i64, t as i64) })?;
                // SAFETY: x [t, ne], w ne floats -> the normed rows' split Q8 form.
                self.op("rms xq silo", || unsafe { (s.rms_xq)(self.raw(), x.fp(), w.fp(), std::ptr::null_mut(), xq.ptr(), t as i64, ne as i64, self.g.eps) })?;
                Ok(Some(xq))
            }
            None => {
                self.rms(x, w, h, t, ne, self.g.eps)?;
                Ok(None)
            }
        }
    }

    /// Layer `l`'s SwiGLU on its normed input (h, or its split Q8 form `xq`): its output [t, n_embd], before the post norm
    #[allow(clippy::too_many_arguments)]
    fn ffn(&self, ly: &Layer, xq: Option<DevBuf>, h: &DevBuf, fg: &DevBuf, fu: &DevBuf, out: &DevBuf, t: usize) -> Result<()> {
        let g = &self.g;
        let (gv, uv) = match (&ly.gu, &xq) {
            (Some(all), Some(q)) => {
                let y = self.arena().f32(t * all.rows)?;
                self.matmul(all, h, Some(q), t, &y)?;
                if t == 1 {
                    (y.view(0, g.n_ff * 4)?, y.view(g.n_ff * 4, g.n_ff * 4)?)
                } else {
                    for (b, c0) in [(fg, 0), (fu, g.n_ff)] {
                        // SAFETY: y [t, 2 n_ff] -> b [t, n_ff].
                        self.op("cols", || unsafe { (self.k.get_cols)(self.raw(), y.fp(), b.fp(), t as i64, g.n_ff as i64, all.rows as i64, c0 as i64) })?;
                    }
                    (fg.view(0, t * g.n_ff * 4)?, fu.view(0, t * g.n_ff * 4)?)
                }
            }
            _ => {
                self.matmul(&ly.ffn_gate, h, xq.as_ref(), t, fg)?;
                self.matmul(&ly.ffn_up, h, xq.as_ref(), t, fu)?;
                (fg.view(0, t * g.n_ff * 4)?, fu.view(0, t * g.n_ff * 4)?)
            }
        };
        let nf = t * g.n_ff;
        match self.silo.filter(|_| t <= MAXV && ly.ffn_down.packed()) {
            Some(s) => {
                // SAFETY: arithmetic only.
                let xd = self.arena().bytes(unsafe { (s.xq_bytes)(g.n_ff as i64, t as i64) })?;
                // SAFETY: gate and up [t, n_ff] -> silu(gate) * up in the split Q8 form.
                self.op("swiglu xq silo", || unsafe { (s.xq)(self.raw(), gv.fp(), uv.fp(), 2, xd.ptr(), g.n_ff as i64, t as i64) })?;
                self.matmul(&ly.ffn_down, fg, Some(&xd), t, out)
            }
            None => {
                // SAFETY: n floats each.
                self.op("swiglu", || unsafe { (self.k.swiglu_clamp)(self.raw(), gv.fp(), uv.fp(), gv.fp(), nf as i64, f32::INFINITY) })?;
                self.matmul(&ly.ffn_down, &gv, None, t, out)
            }
        }
    }

    /// One pass of at most a chunk: the last `n_out` rows' logits
    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        self.pass_in(s, tokens, n_out, tap)
    }

    /// `pass` with the work lock held; with the drafter the target's features of every row go into its caches
    fn pass_in(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        match self.pass_out(s, tokens, n_out, None, tap)? {
            Out::Logits(l) => Ok(l),
            Out::Ids(_) => Err(e("a pass's logits")),
        }
    }

    /// A pass's last `n_out` rows: their logits, or (`pick`: a decoder whose logits nothing reads) only each one's token,
    /// picked on the GPU - the argmax (greedy), or a draw (`Some(sampling)`)
    fn pass_out(&self, s: &mut Session, tokens: &[u32], n_out: usize, pick: Option<Option<nextsycl_llm::DeviceSampling>>, tap: Tap) -> Result<Out> {
        let g = &self.g;
        let t = tokens.len();
        if t == 0 {
            return Err(e("an empty pass"));
        }
        if t > self.chunk {
            return Err(e(format!("a pass of {t} rows; it takes {}", self.chunk)));
        }
        if s.pos + t > s.max_ctx {
            return Err(e(format!("the context is full ({} of {})", s.pos + t, s.max_ctx)));
        }
        for d in &self.devs {
            d.arena.reset();
        }
        self.on(self.layer_dev[0]);
        let p0 = s.pos;
        let ne = g.n_embd;
        let emb = self.embed(tokens)?;
        let mut x = self.arena().f32(t * ne)?;
        self.rms(&emb, &self.ones, &x, t, ne, g.eps)?;
        tap("embd_norm", &x)?;
        let mut h = self.arena().f32(t * ne)?;
        let mut ffn_g = self.arena().f32(t * g.n_ff)?;
        let mut ffn_u = self.arena().f32(t * g.n_ff)?;
        let mut pending: Option<DevBuf> = None;
        // the drafter's input: the rows' hidden states going into its target layers, side by side
        let feats = match &self.dflash {
            Some(d) => Some(self.devs[d.dev].arena.f32(t * d.targets.len() * ne)?),
            None => None,
        };
        let mut held: Vec<(usize, DevBuf)> = Vec::new();
        for (l, (ly, (kc, vc, cap))) in self.layers.iter().zip(s.layers.iter()).enumerate() {
            // the next GPU's layers: the rows cross over, its own buffers for them
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                x = self.hop(&x, self.layer_dev[l])?;
                h = self.arena().f32(t * ne)?;
                ffn_g = self.arena().f32(t * g.n_ff)?;
                ffn_u = self.arena().f32(t * g.n_ff)?;
            }
            if let (Some(d), Some(f)) = (&self.dflash, &feats) {
                // arrived on the drafter's GPU: the features taken on the GPUs before come over (the hop has waited)
                if self.layer_dev[l] == d.dev && !held.is_empty() {
                    self.bring(d, f, &mut held, t)?;
                }
                if let Some(k) = d.targets.iter().position(|x| *x == l) {
                    if self.layer_dev[l] == d.dev {
                        self.capture(d, &x, f, k, t)?;
                    } else {
                        // a copy kept here until then (no wait in the middle of this GPU's layers)
                        let c = self.arena().bytes(t * ne * 4)?;
                        c.copy_within(0, &x, 0, t * ne * 4)?;
                        held.push((k, c));
                    }
                }
            }
            // where this layer's last step leaves the next layer's normed input (outside this layer's share of the arena)
            let dec = self.silo.filter(|_| t <= MAXV);
            let carry = match dec {
                // SAFETY: arithmetic only.
                Some(sl) => Some(self.arena().bytes(unsafe { (sl.xq_bytes)(ne as i64, t as i64) })?),
                None => None,
            };
            let mark = self.arena().mark();
            // the attention's normed input: made by the layer before (its last step) when it could
            let xq = match pending.take() {
                Some(q) => Some(q),
                None => self.norm_in(&x, &ly.attn_norm, &h, t)?,
            };
            let out = self.attention(l, ly, kc, vc, *cap, &h, xq.as_ref(), t, p0, &mut *tap)?;
            let xf = self.post_norm(&out, &ly.post_attn, &x, &ly.ffn_norm, g.eps, &h, t, None)?;
            tap(&format!("ffn_inp-{l}"), &x)?;
            self.ffn(ly, xf, &h, &ffn_g, &ffn_u, &out, t)?;
            // the next layer's norm with this one's last step, on the same GPU and outside this layer's arena share
            let next = self.layers.get(l + 1).filter(|_| self.layer_dev.get(l + 1) == Some(&self.layer_dev[l]));
            match (next, &carry) {
                (Some(nl), Some(c)) => pending = self.post_norm(&out, &ly.post_ffn, &x, &nl.attn_norm, g.eps, &h, t, Some(c))?,
                _ => self.post(&out, &ly.post_ffn, &x, t)?,
            }
            tap(&format!("l_out-{l}"), &x)?;
            self.arena().rewind(mark);
        }
        if let (Some(d), Some(f)) = (&self.dflash, &feats) {
            let ti = Instant::now();
            if !held.is_empty() {
                self.bring(d, f, &mut held, t)?;
            }
            self.inject(d, f, &s.draft, t, p0)?;
            if t <= MAXV {
                self.dev().gpu.sync()?;
                tick(&T_INJECT, ti);
            }
            self.on(self.layer_dev[g.layers - 1]);
        }
        s.pos += t;
        s.last_rows = t;
        let n = n_out.clamp(1, t);
        let last = x.view((t - n) * ne * 4, n * ne * 4)?;
        let normed = self.arena().f32(n * ne)?;
        self.rms(&last, &self.output_norm, &normed, n, ne, g.eps)?;
        tap("result_norm", &normed)?;
        let v = self.output.rows;
        let logits = self.arena().f32(n * v)?;
        self.matmul(&self.output, &normed, None, n, &logits)?;
        if let Some(how) = pick {
            let best = self.arena().bytes(n * 4)?;
            match how {
                // the multiplier and the cap change no argmax
                // SAFETY: logits [n, vocab] -> n int32.
                None => self.op("argmax", || unsafe { (self.k.argmax)(self.raw(), logits.fp(), best.ptr().cast(), n as i64, v as i64) })?,
                Some(ds) => {
                    // SAFETY: n x vocab floats; then each row's draw (row i at its position) into n int32.
                    self.op("softcap", || unsafe { (self.k.softcap)(self.raw(), logits.fp(), (n * v) as i64, g.logit_scale, g.softcap) })?;
                    let pos0 = (s.pos - n) as i64;
                    self.op("sample", || unsafe {
                        (self.k.sample)(self.raw(), logits.fp(), best.ptr().cast(), n as i64, v as i64, ds.temperature, ds.top_p, ds.seed, pos0)
                    })?;
                }
            }
            let mut b = vec![0u8; n * 4];
            best.read(0, &mut b)?;
            return Ok(Out::Ids(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        }
        // the output multiplier, then the soft cap
        // SAFETY: n x vocab floats.
        self.op("softcap", || unsafe { (self.k.softcap)(self.raw(), logits.fp(), (n * v) as i64, g.logit_scale, g.softcap) })?;
        let all = logits.to_f32()?;
        Ok(Out::Logits(all.chunks(v).map(|c| c.to_vec()).collect()))
    }

    /// Rows x [t, n_embd] (of the GPU the pass is on) into columns k of the drafter's features (on its GPU)
    /// Rows x [t, n_embd] (on the drafter's GPU) into columns k of its features
    fn capture(&self, d: &Dflash, x: &DevBuf, f: &DevBuf, k: usize, t: usize) -> Result<()> {
        let (ne, nt) = (self.g.n_embd, d.targets.len());
        // SAFETY: x [t, ne], f [t, nt ne].
        self.op("features", || unsafe { (self.k.put_cols)(self.raw(), x.fp(), f.fp(), t as i64, ne as i64, (nt * ne) as i64, (k * ne) as i64) })
    }

    /// The features taken on the GPUs before the drafter's (each kept there) onto it, into their columns
    fn bring(&self, d: &Dflash, f: &DevBuf, held: &mut Vec<(usize, DevBuf)>, t: usize) -> Result<()> {
        let ne = self.g.n_embd;
        let mut st = self.staging.lock().unwrap();
        for (k, c) in held.drain(..) {
            let y = self.devs[d.dev].arena.bytes(t * ne * 4)?;
            if st.len() < y.len {
                st.resize(y.len, 0);
            }
            y.copy_from_peer(&c, &mut st)?;
            self.on(d.dev);
            self.capture(d, &y, f, k, t)?;
        }
        Ok(())
    }

    /// The drafter's caches at positions p0.. from the target's features f [t, targets x n_embd]: fc, its norm, then
    /// each layer's k (normed, rotated) and v
    fn inject(&self, d: &Dflash, f: &DevBuf, caches: &[(DevBuf, DevBuf, usize)], t: usize, p0: usize) -> Result<()> {
        let (g, k) = (&self.g, self.k);
        let ne = g.n_embd;
        self.on(d.dev);
        let gx = self.arena().f32(t * ne)?;
        self.matmul(&d.fc, f, None, t, &gx)?;
        let gn = self.arena().f32(t * ne)?;
        let gq = match self.silo.filter(|_| t <= MAXV) {
            Some(sl) => {
                // SAFETY: arithmetic only.
                let xq = self.arena().bytes(unsafe { (sl.xq_bytes)(ne as i64, t as i64) })?;
                // SAFETY: gx [t, ne], the norm ne floats -> its split Q8 form.
                self.op("rms xq silo", || unsafe { (sl.rms_xq)(self.raw(), gx.fp(), d.enc_norm.fp(), std::ptr::null_mut(), xq.ptr(), t as i64, ne as i64, d.eps) })?;
                Some(xq)
            }
            None => {
                self.rms(&gx, &d.enc_norm, &gn, t, ne, d.eps)?;
                None
            }
        };
        let kd = d.heads_kv * g.head;
        for (dl, (kc, vc, cap)) in d.layers.iter().zip(caches) {
            let (kx, vx) = (self.arena().f32(t * kd)?, self.arena().f32(t * kd)?);
            self.matmul(&dl.k, &gn, gq.as_ref(), t, &kx)?;
            self.matmul(&dl.v, &gn, gq.as_ref(), t, &vx)?;
            // SAFETY: the rows and caches as named in mg.h.
            unsafe {
                self.op("draft k norm rope", || (k.qk_norm_rope)(self.raw(), kx.fp(), kd as i64, kx.fp(), kd as i64, t as i64, d.heads_kv as i64, g.head as i64,
                                                                 dl.k_norm.fp(), d.eps, 2, d.theta, p0 as i64))?;
                self.op("draft kv store", || (k.kv_store)(self.raw(), kx.fp(), vx.fp(), t as i64, d.heads_kv as i64, g.head as i64, *cap as i64, p0 as i64,
                                                          kc.ptr(), vc.ptr()))?;
            }
        }
        Ok(())
    }

    /// Drafts for the positions after `tail` (committed at s.pos, not fed): one pass of the drafter over [tail, MASK..]
    /// at s.pos.., the block's argmaxes after its first
    fn draft_in(&self, s: &Session, tail: u32) -> Result<Vec<u32>> {
        let Some(d) = &self.dflash else { return Ok(Vec::new()) };
        let nd = self.drafts.min(s.max_ctx.saturating_sub(s.pos + 1));
        if nd == 0 {
            return Ok(Vec::new());
        }
        let (g, k) = (&self.g, self.k);
        let sl = self.silo.ok_or_else(|| e("drafts need the silo"))?;
        let (ne, n, rows) = (g.n_embd, s.pos, nd + 1);
        self.on(d.dev);
        self.arena().reset();
        let mut toks = vec![d.mask; rows];
        toks[0] = tail;
        let x = self.embed(&toks)?;
        let (qd, kd) = (d.heads * g.head, d.heads_kv * g.head);
        let xq_of = |src: &DevBuf, w: &DevBuf| -> Result<DevBuf> {
            // SAFETY: arithmetic only.
            let xq = self.arena().bytes(unsafe { (sl.xq_bytes)(ne as i64, rows as i64) })?;
            // SAFETY: src [rows, ne], w ne floats -> the normed rows' split Q8 form.
            self.op("draft rms xq", || unsafe { (sl.rms_xq)(self.raw(), src.fp(), w.fp(), std::ptr::null_mut(), xq.ptr(), rows as i64, ne as i64, d.eps) })?;
            Ok(xq)
        };
        for (dl, (kc, vc, cap)) in d.layers.iter().zip(&s.draft) {
            let mark = self.arena().mark();
            let hq = xq_of(&x, &dl.attn_norm)?;
            let (q, kx, vx) = (self.arena().f32(rows * qd)?, self.arena().f32(rows * kd)?, self.arena().f32(rows * kd)?);
            self.matmul(&dl.q, &x, Some(&hq), rows, &q)?;
            self.matmul(&dl.k, &x, Some(&hq), rows, &kx)?;
            self.matmul(&dl.v, &x, Some(&hq), rows, &vx)?;
            let att = self.arena().f32(rows * qd)?;
            // SAFETY: the rows and caches as named in mg.h.
            unsafe {
                self.op("draft q norm rope", || (k.qk_norm_rope)(self.raw(), q.fp(), qd as i64, q.fp(), qd as i64, rows as i64, d.heads as i64, g.head as i64,
                                                                 dl.q_norm.fp(), d.eps, 2, d.theta, n as i64))?;
                self.op("draft k norm rope", || (k.qk_norm_rope)(self.raw(), kx.fp(), kd as i64, kx.fp(), kd as i64, rows as i64, d.heads_kv as i64,
                                                                 g.head as i64, dl.k_norm.fp(), d.eps, 2, d.theta, n as i64))?;
                self.op("draft kv store", || (k.kv_store)(self.raw(), kx.fp(), vx.fp(), rows as i64, d.heads_kv as i64, g.head as i64, *cap as i64, n as i64,
                                                          kc.ptr(), vc.ptr()))?;
                let win = d.window as i64;
                let sc = (k.attn_scratch)(rows as i64, d.heads as i64, d.heads_kv as i64, g.head as i64, n as i64, win, 0) as usize;
                let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
                self.op("draft attention", || (k.attn)(self.raw(), q.fp(), kc.ptr(), vc.ptr(), rows as i64, d.heads as i64, d.heads_kv as i64, g.head as i64,
                                                       *cap as i64, n as i64, win, 0, att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
            }
            let o = self.arena().f32(rows * ne)?;
            self.matmul(&dl.o, &att, None, rows, &o)?;
            // SAFETY: rows x ne floats each.
            self.op("draft add", || unsafe { (k.add)(self.raw(), x.fp(), o.fp(), (rows * ne) as i64) })?;
            let fq = xq_of(&x, &dl.ffn_norm)?;
            let (fg, fu) = (self.arena().f32(rows * g.n_ff)?, self.arena().f32(rows * g.n_ff)?);
            self.matmul(&dl.gate, &x, Some(&fq), rows, &fg)?;
            self.matmul(&dl.up, &x, Some(&fq), rows, &fu)?;
            // SAFETY: arithmetic only.
            let xd = self.arena().bytes(unsafe { (sl.xq_bytes)(g.n_ff as i64, rows as i64) })?;
            // SAFETY: gate and up [rows, n_ff] -> silu(gate) * up in the split Q8 form.
            self.op("draft swiglu xq", || unsafe { (sl.xq)(self.raw(), fg.fp(), fu.fp(), 2, xd.ptr(), g.n_ff as i64, rows as i64) })?;
            self.matmul(&dl.down, &fg, Some(&xd), rows, &o)?;
            // SAFETY: rows x ne floats each.
            self.op("draft add", || unsafe { (k.add)(self.raw(), x.fp(), o.fp(), (rows * ne) as i64) })?;
            self.arena().rewind(mark);
        }
        // the block's rows after the first through the drafter's norm and the target's head, each one's argmax
        let after = x.view(ne * 4, nd * ne * 4)?;
        let xn = self.arena().f32(nd * ne)?;
        self.rms(&after, &d.norm, &xn, nd, ne, d.eps)?;
        // the head is on the target's last GPU
        let last = self.layer_dev[g.layers - 1];
        let xn = if last != d.dev { self.hop(&xn, last)? } else { xn };
        let logits = self.arena().f32(nd * self.output.rows)?;
        self.matmul(&self.output, &xn, None, nd, &logits)?;
        let best = self.arena().bytes(nd * 4)?;
        // SAFETY: logits [nd, vocab] -> nd int32.
        self.op("argmax", || unsafe { (k.argmax)(self.raw(), logits.fp(), best.ptr().cast(), nd as i64, self.output.rows as i64) })?;
        let mut b = vec![0u8; nd * 4];
        best.read(0, &mut b)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn has_draft(&self) -> bool {
        self.drafts > 0
    }

    pub fn max_verify(&self) -> usize {
        (self.drafts + 1).max(1)
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        // the caches, checked against what is free first: past the card the xe driver spills into host memory (no
        // out-of-memory error) and can take the machine down
        for d in &self.devs {
            let need: usize = (d.lb..d.le).map(|l| cache_bytes(g, l, max_ctx, self.chunk)).sum();
            let (total, free) = d.gpu.memory()?;
            let free = free.unwrap_or(total);
            if need as u64 + (1536 << 20) > free {
                return Err(e(format!("a session of {max_ctx} tokens needs {:.2} GiB on {}; {:.2} GiB are free (1.5 GiB kept spare)", need as f64 / (1u64 << 30) as f64,
                                     d.gpu.name, free as f64 / (1u64 << 30) as f64)));
            }
        }
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let gpu = &self.devs[self.layer_dev[l]].gpu;
            let cap = g.slots(l, max_ctx, self.chunk);
            let n = g.heads_kv * cap * g.head * 2;
            layers.push((DevBuf::new(gpu, n)?, DevBuf::new(gpu, n)?, cap));
        }
        let mut draft = Vec::new();
        if let Some(d) = &self.dflash {
            // a ring of the window and a pass (a prompt chunk's injection, or a block)
            let cap = if d.window > 0 { max_ctx.min(d.window + self.chunk.max(d.block)) } else { max_ctx + d.block };
            let gpu = &self.devs[d.dev].gpu;
            for _ in 0..d.layers.len() {
                let n = d.heads_kv * cap * g.head * 2;
                draft.push((DevBuf::new(gpu, n)?, DevBuf::new(gpu, n)?, cap));
            }
        }
        Ok(Session { pos: 0, max_ctx, layers, draft, last_rows: 0 })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        s.pos = 0;
        s.last_rows = 0;
        Ok(())
    }

    /// Positions [from, to) of one cache (`heads` of cap slots) into another (dcap slots), head by head, in runs that
    /// wrap neither
    #[allow(clippy::too_many_arguments)]
    fn copy_ring(&self, dst: &DevBuf, dcap: usize, src: &DevBuf, scap: usize, heads: usize, from: usize, to: usize) -> Result<()> {
        let row = self.g.head * 2;
        for h in 0..heads {
            let mut p = from;
            while p < to {
                let (si, di) = (p % scap, p % dcap);
                let run = (to - p).min(scap - si).min(dcap - di);
                dst.copy_within((h * dcap + di) * row, src, (h * scap + si) * row, run * row)?;
                p += run;
            }
        }
        Ok(())
    }

    /// A session's caches - the target's layers, then the drafter's - each with its key heads and window (0: all)
    fn rings<'a>(&self, s: &'a Session) -> Vec<(&'a (DevBuf, DevBuf, usize), usize, usize)> {
        let g = &self.g;
        let mut v: Vec<_> = s.layers.iter().enumerate().map(|(l, c)| (c, g.heads_kv, g.win(l))).collect();
        if let Some(d) = &self.dflash {
            v.extend(s.draft.iter().map(|c| (c, d.heads_kv, d.window)));
        }
        v
    }

    /// The first position a cache with window `w` still reads at `pos`
    fn kept(w: usize, pos: usize) -> usize {
        if w > 0 { pos.saturating_sub(w - 1) } else { 0 }
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        for ((d, heads, w), (sc, _, _)) in self.rings(dst).into_iter().zip(self.rings(src)) {
            let from = Self::kept(w, src.pos);
            self.copy_ring(&d.0, d.2, &sc.0, sc.2, heads, from, src.pos)?;
            self.copy_ring(&d.1, d.2, &sc.1, sc.2, heads, from, src.pos)?;
        }
        dst.pos = src.pos;
        dst.last_rows = 0;
        Ok(())
    }

    pub fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        let mut logits = Vec::new();
        let mut done = 0;
        for c in tokens.chunks(self.chunk) {
            if done > 0 && stop(done) {
                break;
            }
            logits = self.pass(s, c, 1, &mut *tap)?.pop().unwrap_or_default();
            done += c.len();
        }
        Ok((done, logits))
    }

    pub fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        self.pass(s, tokens, n_out, tap)
    }

    /// Back to the first `keep` rows of the last pass: the caches' later slots are simply written again (a windowed
    /// layer's ring keeps a pass beyond its window, so what a kept row overwrote is still not needed)
    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep > s.last_rows {
            return Err(e(format!("a rollback to {keep} rows of a pass of {}", s.last_rows)));
        }
        s.pos -= s.last_rows - keep;
        s.last_rows = keep;
        Ok(())
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    /// The next committed token(s). With drafts: [token, drafts] in one pass, each row's token drawn as plain
    /// sampling would draw it, the drafts kept while they match (what is committed is distributed as without drafts)
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let _w = self.work.lock().unwrap();
        let drafting = d.draft && self.drafts > 0;
        let out = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => vec![smp.sample(&l)],
            (None, Some(x)) => {
                let mut rows = vec![x];
                if drafting {
                    rows.extend(d.drafts.iter().take(s.max_ctx.saturating_sub(s.pos + 1)));
                }
                d.drafts.clear();
                let p0 = s.pos;
                // the tokens from the GPU when the decoder allows (64 bytes back, not 16 rows of logits)
                let picks: Vec<Option<u32>>;
                let tv = Instant::now();
                // greedy: the argmaxes; a sampled request nothing reads the logits of: its draws (same distribution)
                let pick = if smp.greedy() { Some(None) } else { smp.device().map(Some) };
                let res = self.pass_out(s, &rows, rows.len(), pick, tap)?;
                tick(&T_VERIFY, tv);
                STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let logits = match res {
                    Out::Logits(l) => {
                        picks = vec![None; l.len()];
                        l
                    }
                    Out::Ids(v) => {
                        picks = v.into_iter().map(Some).collect();
                        vec![Vec::new(); picks.len()]
                    }
                };
                let mut out = Vec::with_capacity(rows.len());
                for (i, l) in logits.iter().enumerate() {
                    let y = picks[i].unwrap_or_else(|| smp.sample(l));
                    out.push(y);
                    if i + 1 < rows.len() {
                        d.drafted += 1;
                        if y == rows[i + 1] {
                            d.accepted += 1;
                            continue;
                        }
                    }
                    break;
                }
                // the rejected rows' cache slots are written again later
                s.pos = p0 + out.len();
                s.last_rows = out.len();
                out
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        let tail = *out.last().unwrap_or(&0);
        d.next = Some(tail);
        if drafting {
            let td = Instant::now();
            d.drafts = self.draft_in(s, tail)?;
            tick(&T_DRAFT, td);
        }
        Ok(out)
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let row = self.g.head * 2;
        let mut bufs = Vec::new();
        for ((kc, vc, cap), heads, w) in self.rings(s) {
            let from = Self::kept(w, s.pos);
            let n = s.pos - from;
            for c in [kc, vc] {
                let mut b = vec![0u8; heads * n * row];
                for h in 0..heads {
                    let mut p = from;
                    while p < s.pos {
                        let si = p % cap;
                        let run = (s.pos - p).min(cap - si);
                        let at = (h * n + p - from) * row;
                        c.read((h * cap + si) * row, &mut b[at..at + run * row])?;
                        p += run;
                    }
                }
                bufs.push(b);
            }
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let row = self.g.head * 2;
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        let mut it = ck.bufs.iter();
        for ((kc, vc, cap), heads, w) in self.rings(s) {
            let from = Self::kept(w, ck.pos);
            let n = ck.pos - from;
            for c in [kc, vc] {
                let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                if b.len() != heads * n * row {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..heads {
                    let mut p = from;
                    while p < ck.pos {
                        let si = p % cap;
                        let run = (ck.pos - p).min(cap - si);
                        let at = (h * n + p - from) * row;
                        c.write((h * cap + si) * row, &b[at..at + run * row])?;
                        p += run;
                    }
                }
            }
        }
        s.pos = ck.pos;
        s.last_rows = 0;
        Ok(())
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.devs
            .iter()
            .map(|d| {
                let (total, free) = d.gpu.memory().unwrap_or((0, None));
                GpuInfo { index: d.gpu.index, name: d.gpu.name.clone(), pci: d.gpu.pci.clone(), total, free, layers: (d.lb as u64, d.le as u64), expert_slots: 0,
                          host_slots: 0 }
            })
            .collect()
    }
}

/// A tensor read onto the GPU: its blocks (a Q8_0 matrix repacked by the silo), or float32 for F32 / F16 / BF16
fn vec_mat(f: &Gguf, gpu: &Arc<Gpu>, k: &ffi::Api, silo: Option<&ffi::Silo>, name: &str, loaded: &mut u64) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    // outermost first: [rows (outputs), cols (inputs)]; a vector is one row
    let (rows, cols) = match t.shape.len() {
        1 => (1, t.shape[0] as usize),
        _ => (t.shape[0] as usize, t.shape[1..].iter().product::<u64>() as usize),
    };
    let raw = f.read(t).map_err(|x| e(x.0))?;
    *loaded += raw.len() as u64;
    if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
        let src = DevBuf::new(gpu, raw.len())?;
        src.write(0, &raw)?;
        let out = DevBuf::f32(gpu, rows * cols)?;
        // SAFETY: src holds the tensor, out its elements as float32.
        ffi::check(unsafe { (k.dequant)(gpu.raw(), t.ty.code() as i32, src.ptr(), rows * cols, out.fp()) }, name)?;
        gpu.sync()?;
        return Ok(Mat { buf: out, ty: GType::F32, rows, cols, scales: None });
    }
    if let (GType::Q8_0, Some(s), true) = (t.ty, silo, cols.is_multiple_of(32)) {
        // the matrix owns its repacked memory (its quants first), its scales a view of it
        let (own, m) = pack(gpu, s, &raw, rows, cols, name)?;
        return Ok(Mat { buf: own, ty: t.ty, rows, cols, scales: m.scales });
    }
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    Ok(Mat { buf, ty: t.ty, rows, cols, scales: None })
}

/// Q8_0 blocks of `rows` x `cols` repacked by the silo: the memory, and the matrix as a view of it
fn pack(gpu: &Arc<Gpu>, s: &ffi::Silo, raw: &[u8], rows: usize, cols: usize, name: &str) -> Result<(DevBuf, Mat)> {
    let src = DevBuf::new(gpu, raw.len())?;
    src.write(0, raw)?;
    // SAFETY: arithmetic only.
    let (bytes, at) = unsafe { ((s.q8r_bytes)(rows as i64, cols as i64), (s.q8r_scales_at)(rows as i64, cols as i64)) };
    let own = DevBuf::new(gpu, bytes)?;
    // SAFETY: src holds the matrix's Q8_0 blocks, own its repacked size.
    ffi::check(unsafe { (s.q8r_pack)(gpu.raw(), src.ptr(), rows as i64, cols as i64, own.ptr()) }, name)?;
    gpu.sync()?;
    let m = Mat { buf: own.view(0, rows * cols)?, ty: GType::Q8_0, rows, cols, scales: Some(own.view(at, rows * (cols / 32) * 2)?) };
    Ok((own, m))
}

/// Matrices of the same width stacked (their rows one after another) and repacked as one: the memory, the whole, and
/// each as a view of its rows
fn pack_stack(f: &Gguf, gpu: &Arc<Gpu>, s: &ffi::Silo, names: &[String], loaded: &mut u64) -> Result<Option<(DevBuf, Mat, Vec<Mat>)>> {
    let mut raw = Vec::new();
    let mut shapes = Vec::new();
    for n in names {
        let t = f.tensor(n).ok_or_else(|| e(format!("no {n}")))?;
        if t.ty != GType::Q8_0 || t.shape.len() != 2 {
            return Ok(None);
        }
        shapes.push((t.shape[0] as usize, t.shape[1] as usize));
        let b = f.read(t).map_err(|x| e(x.0))?;
        *loaded += b.len() as u64;
        raw.extend_from_slice(&b);
    }
    let cols = shapes[0].1;
    if shapes.iter().any(|x| x.1 != cols) || !cols.is_multiple_of(32) {
        return Err(e(format!("{}: matrices of different widths", names.join(", "))));
    }
    let rows: usize = shapes.iter().map(|x| x.0).sum();
    let (own, whole) = pack(gpu, s, &raw, rows, cols, &names[0])?;
    let (q, sc) = (whole.buf.ptr() as usize - own.ptr() as usize, whole.scales.as_ref().map_or(0, |x| x.ptr() as usize - own.ptr() as usize));
    let mut parts = Vec::with_capacity(shapes.len());
    let mut r0 = 0;
    for (r, _) in &shapes {
        parts.push(Mat { buf: own.view(q + r0 * cols, r * cols)?, ty: GType::Q8_0, rows: *r, cols, scales: Some(own.view(sc + r0 * (cols / 32) * 2, r * (cols / 32) * 2)?) });
        r0 += r;
    }
    Ok(Some((own, whole, parts)))
}

/// A drafter matrix onto the GPU as the silo's repacked Q8: its blocks expanded to float32 there, requantized
fn dmat(f: &Gguf, gpu: &Arc<Gpu>, k: &ffi::Api, s: &ffi::Silo, name: &str, loaded: &mut u64) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("the drafter has no {name}")))?;
    if t.shape.len() != 2 {
        return Err(e(format!("{name}: not a matrix")));
    }
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
    let raw = f.read(t).map_err(|x| e(x.0))?;
    *loaded += raw.len() as u64;
    let src = DevBuf::new(gpu, raw.len())?;
    src.write(0, &raw)?;
    let fl = DevBuf::f32(gpu, rows * cols)?;
    // SAFETY: src holds the tensor, fl its elements as float32.
    ffi::check(unsafe { (k.dequant)(gpu.raw(), t.ty.code() as i32, src.ptr(), rows * cols, fl.fp()) }, name)?;
    // SAFETY: arithmetic only.
    let (bytes, at) = unsafe { ((s.q8r_bytes)(rows as i64, cols as i64), (s.q8r_scales_at)(rows as i64, cols as i64)) };
    let own = DevBuf::new(gpu, bytes)?;
    // SAFETY: fl [rows, cols] float32 -> own in the repacked layout.
    ffi::check(unsafe { (s.q8r_from_f32)(gpu.raw(), fl.fp(), rows as i64, cols as i64, own.ptr()) }, name)?;
    gpu.sync()?;
    let scales = own.view(at, rows * (cols / 32) * 2)?;
    Ok(Mat { buf: own, ty: GType::Q8_0, rows, cols, scales: Some(scales) })
}

/// The DFlash drafter from its file, on GPU `dev` (the target's last)
fn load_dflash(f: &Gguf, g: &Geometry, gpu: &Arc<Gpu>, dev: usize, k: &ffi::Api, s: &ffi::Silo, loaded: &mut u64) -> Result<Dflash> {
    let u = |key: &str| f.arch_meta(key).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| e(format!("the drafter has no dflash.{key}")));
    let fl = |key: &str| f.arch_meta(key).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| e(format!("the drafter has no dflash.{key}")));
    let targets: Vec<usize> = f.arch_meta("target_layers").and_then(|v| v.as_array()).ok_or_else(|| e("the drafter has no dflash.target_layers"))?
        .iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
    if targets.is_empty() || targets.iter().any(|l| *l >= g.layers) {
        return Err(e(format!("the drafter's target layers {targets:?} for {} layers", g.layers)));
    }
    let (n_embd, heads, heads_kv, head) = (u("embedding_length")?, u("attention.head_count")?, u("attention.head_count_kv")?, u("attention.key_length")?);
    if n_embd != g.n_embd || head != 128 || !heads.is_multiple_of(heads_kv) || !(heads / heads_kv).is_multiple_of(4) {
        return Err(e(format!("a drafter of {n_embd} with {heads} / {heads_kv} heads of {head}: not this model's")));
    }
    if f.tensors.iter().any(|t| t.name.contains("selector") || t.name.contains("markov") || t.name.contains("conv")) {
        return Err(e("a DFlash2 / DSpark drafter: only the plain DFlash one is run"));
    }
    let n_layers = u("block_count")?;
    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let n = |x: &str| format!("blk.{l}.{x}.weight");
        layers.push(DLayer {
            attn_norm: vec_mat(f, gpu, k, None, &n("attn_norm"), loaded)?.buf,
            q_norm: vec_mat(f, gpu, k, None, &n("attn_q_norm"), loaded)?.buf,
            k_norm: vec_mat(f, gpu, k, None, &n("attn_k_norm"), loaded)?.buf,
            ffn_norm: vec_mat(f, gpu, k, None, &n("ffn_norm"), loaded)?.buf,
            q: dmat(f, gpu, k, s, &n("attn_q"), loaded)?,
            k: dmat(f, gpu, k, s, &n("attn_k"), loaded)?,
            v: dmat(f, gpu, k, s, &n("attn_v"), loaded)?,
            o: dmat(f, gpu, k, s, &n("attn_output"), loaded)?,
            gate: dmat(f, gpu, k, s, &n("ffn_gate"), loaded)?,
            up: dmat(f, gpu, k, s, &n("ffn_up"), loaded)?,
            down: dmat(f, gpu, k, s, &n("ffn_down"), loaded)?,
        });
    }
    let fc = dmat(f, gpu, k, s, "fc.weight", loaded)?;
    if fc.cols != targets.len() * g.n_embd || fc.rows != g.n_embd {
        return Err(e(format!("the drafter's fc is {} x {}", fc.rows, fc.cols)));
    }
    let mask = f.meta("tokenizer.ggml.mask_token_id").and_then(|v| v.as_u64()).ok_or_else(|| e("the drafter has no tokenizer.ggml.mask_token_id"))? as u32;
    Ok(Dflash {
        layers,
        fc,
        enc_norm: vec_mat(f, gpu, k, None, "enc.output_norm.weight", loaded)?.buf,
        norm: vec_mat(f, gpu, k, None, "output_norm.weight", loaded)?.buf,
        targets,
        heads,
        heads_kv,
        window: u("attention.sliding_window").unwrap_or(0),
        theta: fl("rope.freq_base")?,
        eps: fl("attention.layer_norm_rms_epsilon")?,
        block: u("block_size").unwrap_or(16),
        mask,
        dev,
    })
}
