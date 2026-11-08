//! Ricursive Ripple-65B (`ricursive`, llama.cpp's src/models/ricursive.cpp; the reference is ricursiveAI's
//! modeling_ricursive.py): 80 dense blocks of 5,120 features, 80 query heads over 8 key heads of 128, a SwiGLU of
//! 40,960 clamped at 20. Four RMS norms a block (before and after the attention, before and after the SwiGLU), each
//! head's q and k normed, an elementwise sigmoid gate (`g_proj`, from the block's normed input) on the attention's
//! output, and a value residual: the value projection reads `x0 + sigmoid(lambda) (h - x0)`, x0 the scaled token
//! embeddings. Four blocks of five attend a sliding window of 1,024 keys with rotary positions (NEOX, theta 1e4, all
//! 128 features); the fifth attends everything with no positions at all.
//!
//! ```text
//!   x = x0 = embed(tokens) sqrt(5120)
//!   per block:  h = attn_norm(x);  x += post_attn_norm(o(gate(h) . attn(q(h), k(h), v(x0 + s (h - x0)))))
//!               x += post_ffn_norm(down(swiglu_clamp(gate(ffn_norm(x)), up(ffn_norm(x)))))
//!   logits = output(output_norm(x))
//! ```
//!
//! The weights stay in their GGUF blocks (Q4_K / Q5_K / Q6_K / Q8_0): decode reads them through the Q8_1 products (up
//! to 8 rows), a prompt chunk expands each matrix to half and multiplies. The layers go over the GPUs by what fits
//! (the last GPU first); layers that fit no GPU stay in pinned host memory, which the products read in place over PCIe
//! (or copy to a staging buffer first: `NS_RC_STREAM=copy`). A sliding-window layer's cache is a ring of the window
//! and a chunk; a global layer's holds the whole context.

use std::ffi::c_int;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Gpu, HostBuf};
use nextsycl_gguf::{GType, Gguf, Tensor};
use nextsycl_llm::{Error, GpuInfo, Result, Sampler, Tap};

use crate::ffi;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

const GIB: f64 = (1u64 << 30) as f64;

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
    pub vocab: usize,
    pub ctx: usize,
    pub window: usize,
    /// per layer: a sliding window with rotary positions (else global, no positions)
    pub swa: Vec<bool>,
    pub embed_scale: f32,
    pub clamp: f32,
}

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        let a = f.architecture().to_string();
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {a}.{k}"));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {a}.{k}"));
        let layers = u("block_count")?;
        let window = u("attention.sliding_window").unwrap_or(0);
        let swa: Vec<bool> = match f.arch_meta("attention.sliding_window_pattern").and_then(|v| v.as_array()) {
            Some(p) if p.len() == layers => p.iter().map(|v| v.as_u64().unwrap_or(0) != 0).collect(),
            Some(p) => return Err(format!("{a}.attention.sliding_window_pattern has {} entries for {layers} layers", p.len())),
            // the reference's default: every fifth layer global
            None => (0..layers).map(|l| l % 5 != 4).collect(),
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
            // shapes are outermost first: [vocab, n_embd]
            vocab: f.tensor("token_embd.weight").map(|t| t.shape[0] as usize).ok_or("no token_embd.weight")?,
            ctx: u("context_length").unwrap_or(1 << 20),
            window,
            swa,
            embed_scale: fl("embedding_scale").unwrap_or(1.0),
            clamp: fl("swiglu_clamp").unwrap_or(0.0),
        };
        if g.head != 128 || g.heads_kv == 0 || !g.heads.is_multiple_of(g.heads_kv) || !matches!(g.heads / g.heads_kv, 1 | 2 | 4 | 6 | 8 | 10) {
            return Err(format!("attention heads of {} ({} / {}): head 128 with 1, 2, 4, 6, 8 or 10 query heads a key head", g.head, g.heads, g.heads_kv));
        }
        if !g.n_embd.is_multiple_of(256) || !g.n_ff.is_multiple_of(256) || g.window == 0 && g.swa.iter().any(|s| *s) {
            return Err(format!("widths {} / {} and a window of {}: multiples of 256, and a window for the sliding layers", g.n_embd, g.n_ff, g.window));
        }
        Ok(g)
    }

    /// bytes of one position's keys and values in a layer (half)
    pub fn kv_pos(&self) -> usize {
        2 * self.heads_kv * self.head * 2
    }
}

/// The tensors of block `l` by role: (name, rows, cols) - checked against the file at load and by `info`
fn block_tensors(g: &Geometry, l: usize) -> Vec<(String, usize, usize)> {
    let (ne, nf, hd) = (g.n_embd, g.n_ff, g.head);
    let n = |s: &str| format!("blk.{l}.{s}.weight");
    vec![(n("attn_norm"), 1, ne), (n("attn_q"), g.heads * hd, ne), (n("attn_k"), g.heads_kv * hd, ne), (n("attn_v"), g.heads_kv * hd, ne),
         (n("attn_output"), ne, g.heads * hd), (n("attn_q_norm"), 1, hd), (n("attn_k_norm"), 1, hd), (n("attn_gate"), g.heads * hd, ne),
         (n("attn_vres_lambda"), 1, 1), (n("post_attention_norm"), 1, ne), (n("ffn_norm"), 1, ne), (n("ffn_gate"), nf, ne), (n("ffn_up"), nf, ne),
         (n("ffn_down"), ne, nf), (n("post_ffw_norm"), 1, ne)]
}

/// Every tensor of the file by role with its shape checked (`info`, `load`): bytes of the weights, and the types met
pub fn check_tensors(f: &Gguf, g: &Geometry) -> std::result::Result<(u64, Vec<(String, usize)>), String> {
    let mut types: std::collections::BTreeMap<String, usize> = Default::default();
    let mut bytes = 0u64;
    let mut want: Vec<(String, usize, usize)> = vec![("token_embd.weight".into(), g.vocab, g.n_embd), ("output_norm.weight".into(), 1, g.n_embd)];
    if f.tensor("output.weight").is_some() {
        want.push(("output.weight".into(), g.vocab, g.n_embd));
    }
    for l in 0..g.layers {
        want.extend(block_tensors(g, l));
    }
    for (name, rows, cols) in want {
        let t = f.tensor(&name).ok_or_else(|| format!("no {name}"))?;
        let n: u64 = t.shape.iter().product();
        if n != (rows * cols) as u64 || (rows > 1 && cols > 1 && t.shape.len() != 2) {
            return Err(format!("{name}: shape {:?}, not {rows} x {cols}", t.shape));
        }
        if rows > 1 && cols % 256 != 0 && !matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
            return Err(format!("{name}: {} with rows of {cols}", t.ty.name()));
        }
        bytes += t.bytes;
        *types.entry(format!("{} {}", t.ty.name(), if name.contains("blk.") { name.split('.').nth(2).unwrap_or("") } else { name.as_str() })).or_default() += 1;
    }
    Ok((bytes, types.into_iter().collect()))
}

/// A decode row set's Q8_1 forms: the blocks (the shared products'), and the silo's split layout when the silo made it
pub struct Q8 {
    std: DevBuf,
    split: Option<DevBuf>,
}

/// A matrix [rows (outputs), cols (inputs)] at `off` in its layer's blob: its stored blocks
#[derive(Clone, Copy)]
struct Mat {
    off: usize,
    len: usize,
    ty: GType,
    rows: usize,
    cols: usize,
}

/// One block's weights: the matrices in one blob (device memory, or pinned host memory a view of which the products
/// read), the small vectors on the GPU
struct Layer {
    swa: bool,
    host: bool,
    base: DevBuf,
    bytes: usize,
    q: Mat,
    k: Mat,
    v: Mat,
    g: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
    attn_norm: DevBuf,
    post_attn: DevBuf,
    ffn_norm: DevBuf,
    post_ffn: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    /// sigmoid(the value residual's logit)
    vres: f32,
}

/// A conversation: per layer its key and value caches ([kv heads][ring rows][128] half; the ring a window and a chunk
/// on a sliding layer, the whole context on a global one)
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<(DevBuf, DevBuf)>,
    /// each layer's ring rows
    ring: Vec<usize>,
    last_rows: usize,
}

/// A conversation's state in host memory: per layer the keys and values of the positions a layer still needs (the last
/// window of a sliding layer, all of a global one), in position order
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(b"NSRC1")?;
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
        if &magic != b"NSRC1" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a ricursive checkpoint"));
        }
        let pos = u(r)?;
        let n = u(r)?;
        let mut bufs = Vec::with_capacity(n.min(1024));
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

/// Generation from a fed prompt: the logits to draw the next token from, or the committed token not fed yet
pub struct Decoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
}

impl Decoder {
    pub fn new(logits: Vec<f32>) -> Decoder {
        Decoder { logits: Some(logits), next: None }
    }
    pub fn after(next: u32) -> Decoder {
        Decoder { logits: None, next: Some(next) }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
}

/// One GPU's share: its layers [lb, le) (the first `res` of them in VRAM, the rest streamed from pinned host memory),
/// a prompt chunk's work memory and the half buffer its products expand into
struct Dev {
    gpu: Arc<Gpu>,
    arena: Arena,
    /// the largest matrix expanded to half (a prompt chunk's products)
    w16: DevBuf,
    /// a host layer's copy in VRAM when the weights are copied before use (`NS_RC_STREAM=copy`)
    stage: Option<DevBuf>,
    lb: usize,
    le: usize,
}

pub struct Ricursive<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    devs: Vec<Dev>,
    layer_dev: Vec<usize>,
    cur: std::sync::atomic::AtomicUsize,
    staging: Mutex<Vec<u8>>,
    k: &'static ffi::Api,
    silo: Option<&'static ffi::Silo>,
    layers: Vec<Layer>,
    /// the layers' pinned host memory (the blobs of the host layers are views of these)
    #[allow(dead_code)]
    hosts: Vec<HostBuf>,
    output_norm: DevBuf,
    output_buf: DevBuf,
    output: Mat,
    embd: Tensor,
    chunk: usize,
    work: Mutex<()>,
    /// NS_RC_DUMP=DIR: every tapped tensor of a pass written there as float32
    dump: Option<std::path::PathBuf>,
    copy_stream: bool,
    /// the silo's XMX attention (NS_RC_FLASH=0: the shared kernel, an A/B)
    flash: bool,
    /// NS_RC_PROFILE=1: GPU time by operation over the decode passes (a sync after each; slows the run)
    prof: Option<Mutex<Prof>>,
    host_layers: usize,
    load_s: f64,
    load_bytes: u64,
}

fn chunk_setting() -> Option<usize> {
    std::env::var("NS_RC_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= 16)
}

/// A prompt chunk's work memory (the arena) for `chunk` rows
/// NS_RC_PROFILE's accumulators
struct Prof {
    last: Instant,
    by: std::collections::BTreeMap<String, (f64, u64)>,
    passes: u64,
}

impl Default for Prof {
    fn default() -> Self {
        Prof { last: Instant::now(), by: Default::default(), passes: 0 }
    }
}

fn arena_bytes(g: &Geometry, chunk: usize) -> usize {
    let (qd, kd) = (g.heads * g.head, g.heads_kv * g.head);
    chunk * (6 * g.n_embd + 4 * qd + 3 * kd + 2 * g.n_ff) * 4 + chunk * g.n_ff.max(qd) * 2 + 16 * g.vocab * 4 + (128 << 20)
}

fn mem_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
        s.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok())
    }).map_or(0, |kb| kb * 1024)
}

/// Host memory kept free whatever is pinned (NS_RC_HOST_SPARE_GIB, default 8)
fn host_spare() -> u64 {
    (std::env::var("NS_RC_HOST_SPARE_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(8.0) * GIB) as u64
}

/// Rows of a ring (positions `from..to`) as the slot runs they occupy: (first slot, rows)
fn runs(from: usize, to: usize, ring: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut p = from;
    while p < to {
        let slot = p % ring;
        let n = (ring - slot).min(to - p);
        out.push((slot, n));
        p += n;
    }
    out
}

const ROW: usize = 256;

impl<'g> Ricursive<'g> {
    /// Loads the model over `gpus` in order. The layers split by what fits: the last GPU takes as many as fit beside
    /// the head and the planned caches, the one before it the rest, and what fits no GPU stays in pinned host memory
    /// on the first (its products read it over PCIe). `NS_RC_HOST=N` puts N more layers there (a measurement of
    /// streaming; the last GPU's layers first), `NS_RC_LAYERS=N` loads only the first N layers (a reference check on
    /// a truncated model), NS_SPLIT as the other engines.
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), log: &mut dyn FnMut(String)) -> Result<Ricursive<'g>> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(e("no GPU"));
        }
        // what each card already holds (other programs'), to tell this engine's share
        let others: Vec<u64> = gpus.iter().map(|d| d.memory().map_or(0, |(t, fr)| t - fr.unwrap_or(t).min(t))).collect();
        let g = Geometry::read(f).map_err(e)?;
        check_tensors(f, &g).map_err(e)?;
        let nl = std::env::var("NS_RC_LAYERS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n >= 1 && *n <= g.layers).unwrap_or(g.layers);
        let k = ffi::api()?;
        let silo = ffi::silo()?;
        log(match silo {
            Some(s) => format!("ricursive: tuned kernels from its silo {}", s.path.display()),
            None => "ricursive: no silo - the shared kernels".into(),
        });
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        let layer_w: Vec<u64> = (0..nl).map(|l| {
            let p = format!("blk.{l}.");
            f.tensors.iter().filter(|t| t.name.starts_with(&p)).map(|t| t.bytes).sum()
        }).collect();
        let head_name = if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let head_w: u64 = f.tensor(head_name).map_or(0, |t| t.bytes) + f.tensor("output_norm.weight").map_or(0, |t| t.bytes);
        let w16_bytes = g.n_ff * g.n_embd * 2;
        let mut free = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let (total, fr) = gpu.memory()?;
            free.push(fr.unwrap_or(total));
        }
        let margin: u64 = 1536 << 20;
        // the planned caches: tokens in all, over the sessions (each its own rings on the sliding layers)
        let (ctx_all, sessions) = (if kv.0 > 0 { kv.0 } else { 32768 }, kv.1.max(1));
        let ctx_one = ctx_all / sessions;
        let kvp = g.kv_pos() as u64;
        let copy_stream = std::env::var("NS_RC_STREAM").is_ok_and(|v| v == "copy");
        let forced_host: usize = std::env::var("NS_RC_HOST").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        let n = gpus.len();
        // per GPU: (first layer, end, layers in VRAM)
        let plan = |c: usize| -> std::result::Result<Vec<(usize, usize, usize)>, String> {
            let cache_of = |l: usize| -> u64 {
                if g.swa[l] { sessions as u64 * ctx_one.min(g.window + c) as u64 * kvp } else { ctx_all as u64 * kvp }
            };
            let fixed = |i: usize, hosted: bool| -> i64 {
                let mut over = margin + (512 << 20) + w16_bytes as u64 + arena_bytes(&g, c) as u64;
                if hosted && copy_stream {
                    over += layer_w.iter().copied().max().unwrap_or(0);
                }
                if i + 1 == n {
                    over += head_w;
                }
                free[i] as i64 - over as i64
            };
            let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(n);
            if let Some(sp) = &split_env {
                let mut lb = 0;
                for i in 0..n {
                    let le = if i + 1 == n { nl } else { *sp.get(i).ok_or("NS_SPLIT: a first layer for each GPU after the first")? };
                    bounds.push((lb, le.min(nl)));
                    lb = le.min(nl);
                }
            } else {
                // the last GPU first (the server orders them weakest first): each takes the last layers left while
                // they fit; the first the rest
                let mut le = nl;
                let mut rev = Vec::with_capacity(n);
                for i in (0..n).rev() {
                    let mut lb = le;
                    if i == 0 {
                        lb = 0;
                    } else {
                        let mut have = fixed(i, false);
                        while lb > 0 && layer_w[lb - 1] as i64 + cache_of(lb - 1) as i64 <= have {
                            have -= layer_w[lb - 1] as i64 + cache_of(lb - 1) as i64;
                            lb -= 1;
                        }
                    }
                    rev.push((lb, le));
                    le = lb;
                }
                rev.reverse();
                bounds = rev;
            }
            let mut out = Vec::with_capacity(n);
            let mut force = forced_host;
            for i in (0..n).rev() {
                let (lb, le) = bounds[i];
                // how many of its layers fit VRAM: the layers in order while the weights and caches do
                let mut have = fixed(i, true);
                let mut res = 0;
                for (l, w) in layer_w.iter().enumerate().take(le).skip(lb) {
                    let need = *w as i64 + cache_of(l) as i64;
                    if need > have {
                        break;
                    }
                    have -= need;
                    res += 1;
                }
                // a layer past the last that fits keeps only its cache in VRAM: the caches of the host layers
                let host_cache: i64 = (lb + res..le).map(|l| cache_of(l) as i64).sum();
                if host_cache > have && le > lb + res {
                    return Err(format!("{}: the caches of its {} streamed layers need {:.1} GiB more than it has", gpus[i].name, le - lb - res, (host_cache - have) as f64 / GIB));
                }
                let take = force.min(res);
                force -= take;
                out.push((lb, le, res - take));
            }
            out.reverse();
            Ok(out)
        };
        let tries: Vec<usize> = chunk_setting().map_or_else(|| vec![2048, 1024, 512], |c| vec![c]);
        let mut last_err = String::new();
        let mut found = None;
        for &c in &tries {
            match plan(c) {
                Ok(b) => {
                    found = Some((c, b));
                    break;
                }
                Err(x) => last_err = x,
            }
        }
        let (chunk, bounds3) = found.ok_or_else(|| e(format!("{} over {} GPUs at a {ctx_all}-token context: {last_err}", f.architecture(), n)))?;
        let host_bytes: u64 = bounds3.iter().map(|&(lb, le, res)| layer_w[lb + res..le].iter().sum::<u64>()).sum();
        let host_layers: usize = bounds3.iter().map(|&(lb, le, res)| le - lb - res).sum();
        if host_bytes > 0 {
            let avail = mem_available();
            if host_bytes + host_spare() > avail {
                return Err(e(format!("{host_layers} layers ({:.1} GiB) stay in pinned host memory; {:.1} GiB are available and {:.1} GiB must stay free",
                                     host_bytes as f64 / GIB, avail as f64 / GIB, host_spare() as f64 / GIB)));
            }
            log(format!("ricursive: {host_layers} layers ({:.1} GiB) in pinned host memory, read {}", host_bytes as f64 / GIB,
                        if copy_stream { "through a staging copy in VRAM" } else { "in place over PCIe" }));
        }
        let given = gpus.len();
        let keep: Vec<usize> = (0..given).filter(|i| bounds3[*i].1 > bounds3[*i].0).collect();
        let gpus_used: Vec<Arc<Gpu>> = keep.iter().map(|i| gpus[*i].clone()).collect();
        let bounds3: Vec<(usize, usize, usize)> = keep.iter().map(|i| bounds3[*i]).collect();
        if gpus_used.len() < given {
            log(format!("ricursive: all of it fits {}: {} of the {given} GPUs not used", gpus_used.iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(" + "),
                        given - gpus_used.len()));
        }
        let gpus_all: Vec<Arc<Gpu>> = gpus.to_vec();
        let gpus = gpus_used;
        let mut layer_dev = vec![0usize; nl];
        for (i, &(lb, le, _)) in bounds3.iter().enumerate() {
            for d in &mut layer_dev[lb..le] {
                *d = i;
            }
        }
        let lastg = gpus[gpus.len() - 1].clone();
        let mut loaded = 0u64;
        let mut hosts: Vec<HostBuf> = Vec::new();
        let mut layers = Vec::with_capacity(nl);
        for (l, &di) in layer_dev.iter().enumerate() {
            let gpu = &gpus[di];
            let (lb, _, res) = bounds3[di];
            let host = l >= lb + res;
            let ly = load_layer(f, &g, l, gpu, host, &mut hosts, &mut loaded)?;
            layers.push(ly);
            if l % 16 == 15 || l + 1 == nl {
                log(format!("ricursive: layer {} of {nl} ({:.1} GiB, {:.0} s)", l + 1, loaded as f64 / GIB, t0.elapsed().as_secs_f64()));
            }
        }
        let output_norm = vec_f32(f, &lastg, k, "output_norm.weight")?;
        let (output_buf, output) = {
            let t = f.tensor(head_name).ok_or_else(|| e("no output"))?;
            let raw = f.read(t).map_err(|x| e(x.0))?;
            loaded += raw.len() as u64;
            let b = DevBuf::new(&lastg, raw.len())?;
            b.write(0, &raw)?;
            (b, Mat { off: 0, len: raw.len(), ty: t.ty, rows: t.shape[0] as usize, cols: t.shape[1] as usize })
        };
        let max_host = layers.iter().filter(|l| l.host).map(|l| l.bytes).max().unwrap_or(0);
        let mut devs = Vec::with_capacity(gpus.len());
        for (i, gpu) in gpus.iter().enumerate() {
            gpu.sync()?;
            let (lb, le, res) = bounds3[i];
            let stage = if copy_stream && lb + res < le { Some(DevBuf::new(gpu, max_host)?) } else { None };
            devs.push(Dev { gpu: gpu.clone(), w16: DevBuf::new(gpu, w16_bytes)?, arena: Arena::new(gpu, arena_bytes(&g, chunk))?, stage, lb, le });
        }
        let load_s = t0.elapsed().as_secs_f64();
        let wh = devs.iter().zip(&bounds3).map(|(d, &(lb, le, res))| {
            let host = if lb + res < le { format!(", {} streamed", le - lb - res) } else { String::new() };
            format!("{} (layers {}-{}{host})", d.gpu.name, lb, le - 1)
        }).collect::<Vec<_>>().join(", ");
        log(format!("ricursive: {nl} layers of {} ({} sliding, {} global), {:.1} GiB on {wh} in {load_s:.0} s; prompt chunks of {chunk}", g.n_embd,
                    (0..nl).filter(|l| g.swa[*l]).count(), (0..nl).filter(|l| !g.swa[*l]).count(), loaded as f64 / GIB));
        let used = devs.iter().enumerate().map(|(i, d)| match d.gpu.memory() {
            Ok((t, Some(fr))) => {
                let now = t - fr.min(t);
                let gi = gpus_all.iter().position(|x| x.index == d.gpu.index).unwrap_or(i);
                format!("{} {:.2} GiB (+ {:.2} GiB of other programs) of {:.1}", d.gpu.name, now.saturating_sub(others[gi]) as f64 / GIB, others[gi] as f64 / GIB, t as f64 / GIB)
            }
            _ => "?".into(),
        }).collect::<Vec<_>>().join(", ");
        log(format!("ricursive: card memory in use before any session (weights, work memory, half buffer): {used}"));
        Ok(Ricursive { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(Vec::new()), k, silo, layers, hosts, output_norm, output_buf, output, embd, chunk,
                       work: Mutex::new(()), dump: std::env::var("NS_RC_DUMP").ok().filter(|d| !d.is_empty()).map(Into::into), copy_stream, flash: std::env::var("NS_RC_FLASH").map_or(true, |v| v != "0"), prof: std::env::var("NS_RC_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(Prof::default())), host_layers, load_s,
                       load_bytes: loaded })
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
    pub fn streams(&self) -> usize {
        self.host_layers
    }

    /// Charge the GPU time since the last tick to `label` (NS_RC_PROFILE only)
    fn tick(&self, label: &str) {
        let Some(p) = &self.prof else { return };
        for d in &self.devs {
            let _ = d.gpu.sync();
        }
        let mut p = p.lock().unwrap();
        let now = Instant::now();
        let dt = now.duration_since(p.last).as_secs_f64();
        p.last = now;
        let e = p.by.entry(label.to_string()).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
    }

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

    /// y [t, w.rows] = x [t, w.cols] . W^T, W in `wb` at `w.off`; `xq`: x's Q8_1 forms when the norm made them
    fn matmul(&self, w: &Mat, wb: &DevBuf, x: &DevBuf, xq: Option<&Q8>, t: usize, y: &DevBuf) -> Result<()> {
        self.tick("other");
        let label = format!("mm {:?} t{}", w.ty, if t <= 8 { "<=8" } else { ">8" });
        let r = self.matmul_inner(w, wb, x, xq, t, y);
        self.tick(&label);
        r
    }

    fn matmul_inner(&self, w: &Mat, wb: &DevBuf, x: &DevBuf, xq: Option<&Q8>, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        let (rows, cols) = (w.rows, w.cols);
        let wv = wb.view(w.off, w.len)?;
        let code = w.ty.code() as i32;
        // SAFETY: arithmetic only.
        let mmvq = unsafe { (k.mmvq_supported)(code) } != 0;
        if t <= 8 && mmvq {
            // SAFETY: arithmetic only.
            let silo = self.silo.filter(|s| unsafe { (s.mmvq_supported)(code, kk) } != 0);
            let own;
            let qb = match xq {
                Some(q) => q,
                None => {
                    own = self.quantize(x, cols, t, silo.is_some())?;
                    &own
                }
            };
            return match (silo, &qb.split) {
                // SAFETY: the matrix in its blocks, x in both forms, y [t, rows].
                (Some(s), Some(sp)) => ffi::check(unsafe { (s.mmvq)(self.raw(), code, wv.ptr(), qb.std.ptr(), sp.ptr(), y.fp(), kk, n, t as i64) }, "mmvq silo"),
                // SAFETY: the matrix in its blocks, x's Q8_1 blocks, y [t, rows].
                _ => ffi::check(unsafe { (k.mmvq)(self.raw(), code, wv.ptr(), qb.std.ptr(), y.fp(), kk, n, t as i64) }, "mmvq"),
            };
        }
        // the matrix expanded to half, x to half, then the half product
        let elems = rows * cols;
        if elems * 2 > self.dev().w16.len {
            return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
        }
        // SAFETY: the matrix's blocks -> rows x cols halfs in w16 (non-zero: a type it does not expand directly).
        let rc = match self.silo {
            Some(s) if unsafe { (s.dequant_f16_supported)(code) } != 0 => unsafe { (s.dequant_f16)(self.raw(), code, wv.ptr(), elems as i64, self.dev().w16.ptr().cast()) },
            _ => unsafe { (k.dequant_f16)(self.raw(), code, wv.ptr(), elems as i64, self.dev().w16.ptr().cast()) },
        };
        if rc != 0 {
            let f = self.arena().f32(elems)?;
            // SAFETY: the blocks -> float32, then -> half.
            ffi::check(unsafe { (k.dequant)(self.raw(), code, wv.ptr(), elems, f.fp()) }, "dequant")?;
            ffi::check(unsafe { (k.to_f16)(self.raw(), f.fp(), self.dev().w16.ptr().cast(), elems as i64) }, "to half")?;
        }
        let xh = self.arena().bytes(t * cols * 2)?;
        // SAFETY: x [t, cols] -> half; then y [t, rows] = xh . w16^T.
        ffi::check(unsafe { (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (t * cols) as i64) }, "x to half")?;
        ffi::check(unsafe { (k.gemm_f16)(self.raw(), t as i64, n, kk, xh.ptr().cast(), kk, self.dev().w16.ptr().cast(), y.fp(), n, 0) }, "gemm f16")
    }

    /// The RMS norm into y; for decode rows with the silo also y's Q8_1 forms (the products after it read them)
    fn norm(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<Option<Q8>> {
        let Some(s) = self.silo.filter(|_| rows <= 8 && c.is_multiple_of(256)) else {
            // SAFETY: x and y [rows, c], w c floats.
            ffi::check(unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, self.g.eps) }, "rms norm")?;
            return Ok(None);
        };
        // SAFETY: arithmetic only.
        let (q8, sb) = unsafe { ((self.k.q8_1_bytes)(c as i64, rows as i64), (s.q8_split_bytes)(c as i64, rows as i64)) };
        let (qb, sp) = (self.arena().bytes(q8)?, self.arena().bytes(sb)?);
        // SAFETY: x and y [rows, c], w c floats, qb rows x c / 32 Q8_1 blocks, sp the split layout.
        ffi::check(unsafe { (s.rms_q8)(self.raw(), x.fp(), w.fp(), y.fp(), qb.ptr(), sp.ptr(), rows as i64, c as i64, self.g.eps) }, "rms norm q8")?;
        Ok(Some(Q8 { std: qb, split: Some(sp) }))
    }

    fn quantize(&self, x: &DevBuf, cols: usize, t: usize, split: bool) -> Result<Q8> {
        let k = self.k;
        // SAFETY: arithmetic only.
        let q8 = unsafe { (k.q8_1_bytes)(cols as i64, t as i64) };
        let std = self.arena().bytes(q8)?;
        if let Some(s) = self.silo.filter(|_| split) {
            // SAFETY: arithmetic only.
            let sb = unsafe { (s.q8_split_bytes)(cols as i64, t as i64) };
            let sp = self.arena().bytes(sb)?;
            // SAFETY: x [t, cols] float32 -> its Q8_1 blocks and the split layout.
            ffi::check(unsafe { (s.quantize_q8)(self.raw(), x.fp(), std.ptr(), sp.ptr(), cols as i64, t as i64) }, "quantize")?;
            return Ok(Q8 { std, split: Some(sp) });
        }
        // SAFETY: x [t, cols] float32 -> its Q8_1 blocks.
        ffi::check(unsafe { (k.quantize_q8_1)(self.raw(), x.fp(), std.ptr(), cols as i64, t as i64) }, "quantize")?;
        Ok(Q8 { std, split: None })
    }

    /// The tokens' scaled embeddings [t, n_embd] (their rows read from the file, expanded on the GPU)
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
        if self.embd.ty == GType::IQ4XS {
            // the file's embedding is IQ4_XS (no float32 expansion in the kernels): the few rows expand on the host
            let mut f = Vec::with_capacity(tokens.len() * g.n_embd * 4);
            for r in raw.chunks(row) {
                for v in iq4xs_row(r) {
                    f.extend_from_slice(&(v * g.embed_scale).to_le_bytes());
                }
            }
            let x = self.arena().f32(tokens.len() * g.n_embd)?;
            x.write(0, &f)?;
            return Ok(x);
        }
        let src = self.arena().bytes(raw.len())?;
        src.write(0, &raw)?;
        let x = self.arena().f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32, then scaled.
        ffi::check(unsafe { (self.k.dequant)(self.raw(), self.embd.ty.code() as i32, src.ptr(), tokens.len() * g.n_embd, x.fp()) }, "embedding")?;
        ffi::check(unsafe { (self.k.scale)(self.raw(), x.fp(), (tokens.len() * g.n_embd) as i64, g.embed_scale) }, "embedding scale")?;
        Ok(x)
    }

    /// One block on x [t, n_embd] (updated in place), x0 the scaled embeddings, `s` the session's caches of this layer
    #[allow(clippy::too_many_arguments)]
    fn block(&self, l: usize, x: &DevBuf, x0: &DevBuf, s: &Session, p0: usize, t: usize, tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<()> {
        let (g, k) = (&self.g, self.k);
        let ly = &self.layers[l];
        let (ne, hq, hk, d) = (g.n_embd, g.heads, g.heads_kv, g.head);
        let mark = self.arena().mark();
        // the layer's weights: in VRAM, in pinned host memory read in place, or copied to the staging buffer first
        let staged;
        let wb: &DevBuf = if ly.host && self.copy_stream {
            let st = self.dev().stage.as_ref().ok_or_else(|| e("no staging buffer"))?;
            st.copy_within(0, &ly.base, 0, ly.bytes)?;
            staged = st.view(0, ly.bytes)?;
            &staged
        } else {
            &ly.base
        };
        let h = self.arena().f32(t * ne)?;
        let hq8 = self.norm(x, &ly.attn_norm, &h, t, ne)?;
        let qr = self.arena().f32(t * hq * d)?;
        let gt = self.arena().f32(t * hq * d)?;
        let kr = self.arena().f32(t * hk * d)?;
        let vr = self.arena().f32(t * hk * d)?;
        self.matmul(&ly.q, wb, &h, hq8.as_ref(), t, &qr)?;
        self.matmul(&ly.g, wb, &h, hq8.as_ref(), t, &gt)?;
        self.matmul(&ly.k, wb, &h, hq8.as_ref(), t, &kr)?;
        // the value residual: v reads x0 + s (h - x0)
        let vin = self.arena().f32(t * ne)?;
        // SAFETY: x0, h, vin [t, ne].
        ffi::check(unsafe { (k.vres)(self.raw(), x0.fp(), h.fp(), vin.fp(), (t * ne) as i64, ly.vres) }, "value residual")?;
        self.matmul(&ly.v, wb, &vin, None, t, &vr)?;
        let qn = self.arena().f32(t * hq * d)?;
        let kn = self.arena().f32(t * hk * d)?;
        let rope = ly.swa as c_int;
        let (kc, vc) = (&s.layers[l].0, &s.layers[l].1);
        let ring = s.ring[l];
        let window = if ly.swa { g.window } else { 0 };
        let att = self.arena().f32(t * hq * d)?;
        // SAFETY: the rows and caches as named in rc.h.
        unsafe {
            ffi::check((k.qk_norm_rope)(self.raw(), qr.fp(), (hq * d) as i64, d as i64, qn.fp(), (hq * d) as i64, t as i64, hq as i64, d as i64, ly.q_norm.fp(), g.eps, rope,
                                        g.theta, p0 as i64), "q norm rope")?;
            ffi::check((k.qk_norm_rope)(self.raw(), kr.fp(), (hk * d) as i64, d as i64, kn.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64, ly.k_norm.fp(), g.eps, rope,
                                        g.theta, p0 as i64), "k norm rope")?;
            ffi::check((k.kv_store)(self.raw(), kn.fp(), (hk * d) as i64, vr.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64, ring as i64, p0 as i64, kc.ptr(), vc.ptr()),
                       "kv store")?;
            let (ti, hqi, hki, di) = (t as i64, hq as i64, hk as i64, d as i64);
            let flash = self.silo.filter(|_| self.flash).filter(|s| (s.flash_prompt_supported)(ti, hqi, hki, di) != 0 || (s.flash_decode_supported)(ti, hqi, hki, di) != 0);
            if let Some(s) = flash {
                let (q, kk, vv) = (qn.fp(), kc.ptr(), vc.ptr());
                if t > 16 {
                    ffi::check((s.flash_prompt)(self.raw(), q, kk, vv, ti, hqi, hki, di, ring as i64, p0 as i64, window as i64, att.fp()), "attention")?;
                } else {
                    let sc = (s.flash_decode_scratch)(ti, hqi, hki, di, p0 as i64, window as i64) as usize;
                    let part = self.arena().f32(sc.max(1))?;
                    ffi::check((s.flash_decode)(self.raw(), q, kk, vv, ti, hqi, hki, di, ring as i64, p0 as i64, window as i64, att.fp(), part.fp()), "attention")?;
                }
            } else {
            let sc = (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, window as i64, p0 as i64) as usize;
            let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
            ffi::check((k.attn)(self.raw(), qn.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, ring as i64, window as i64, p0 as i64, att.fp(),
                                part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())), "attention")?;
            }
            self.tick("qk norm, rope, kv store, attention");
            ffi::check((k.gate_mul)(self.raw(), att.fp(), gt.fp(), (t * hq * d) as i64), "output gate")?;
        }
        let o = self.arena().f32(t * ne)?;
        self.matmul(&ly.o, wb, &att, None, t, &o)?;
        // SAFETY: x, o [t, ne]: x += rms_norm(o) post_attn.
        ffi::check(unsafe { (k.norm_add)(self.raw(), x.fp(), o.fp(), ly.post_attn.fp(), t as i64, ne as i64, g.eps) }, "post attention norm")?;
        let hq8 = self.norm(x, &ly.ffn_norm, &h, t, ne)?;
        let (fg, fu) = (self.arena().f32(t * g.n_ff)?, self.arena().f32(t * g.n_ff)?);
        self.matmul(&ly.gate, wb, &h, hq8.as_ref(), t, &fg)?;
        self.matmul(&ly.up, wb, &h, hq8.as_ref(), t, &fu)?;
        let nf = t * g.n_ff;
        // SAFETY: nf floats each.
        ffi::check(unsafe { (k.swiglu_clamp)(self.raw(), fg.fp(), fu.fp(), fg.fp(), nf as i64, if g.clamp > 0.0 { g.clamp } else { f32::INFINITY }) }, "swiglu")?;
        let ff = self.arena().f32(t * ne)?;
        self.matmul(&ly.down, wb, &fg, None, t, &ff)?;
        // SAFETY: x, ff [t, ne].
        ffi::check(unsafe { (k.norm_add)(self.raw(), x.fp(), ff.fp(), ly.post_ffn.fp(), t as i64, ne as i64, g.eps) }, "post feed-forward norm")?;
        self.emit(&format!("l_out-{l}"), x, tap)?;
        self.arena().rewind(mark);
        Ok(())
    }

    /// A tapped tensor, and (NS_RC_DUMP) written to a file as float32
    fn emit(&self, name: &str, x: &DevBuf, tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<()> {
        tap(name, x)?;
        if let Some(dir) = &self.dump {
            let v = x.to_f32()?;
            let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{name}.f32")), b).map_err(|x| e(format!("{}: {x}", dir.display())))?;
        }
        Ok(())
    }

    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
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
        self.on(0);
        let p0 = s.pos;
        let ne = g.n_embd;
        let mut x = self.embed(tokens)?;
        let mut x0 = self.arena().f32(t * ne)?;
        x0.copy_within(0, &x, 0, t * ne * 4)?;
        self.emit("embd", &x, &mut *tap)?;
        for l in 0..self.layers.len() {
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                x = self.hop(&x, self.layer_dev[l])?;
                x0 = self.hop(&x0, self.layer_dev[l])?;
            }
            self.block(l, &x, &x0, s, p0, t, &mut *tap)?;
        }
        s.pos += t;
        s.last_rows = t;
        let n = n_out.clamp(1, t);
        let last = x.view((t - n) * ne * 4, n * ne * 4)?;
        let normed = self.arena().f32(n * ne)?;
        let q = self.norm(&last, &self.output_norm, &normed, n, ne)?;
        self.emit("result_norm", &normed, &mut *tap)?;
        let logits = self.arena().f32(n * self.output.rows)?;
        // the head's rows eight at a time (decode-width products: a wider pass would expand the 100K-row matrix to half)
        let mut at = 0;
        while at < n {
            let m = (n - at).min(8);
            let xs = normed.view(at * ne * 4, m * ne * 4)?;
            let ys = logits.view(at * self.output.rows * 4, m * self.output.rows * 4)?;
            self.matmul(&self.output, &self.output_buf, &xs, if m == n { q.as_ref() } else { None }, m, &ys)?;
            at += m;
        }
        let all = logits.to_f32()?;
        self.tick("logits readback");
        if let Some(p) = &self.prof {
            let mut p = p.lock().unwrap();
            if t == 1 {
                p.passes += 1;
                if p.passes % 32 == 0 {
                    let tot: f64 = p.by.values().map(|v| v.0).sum();
                    let mut v: Vec<_> = p.by.iter().collect();
                    v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
                    eprintln!("[ricursive profile: {} one-row passes; {:.1} ms a pass (prompt passes included in the totals)]", p.passes, tot * 1e3 / p.passes as f64);
                    for (k, (secs, n)) in v {
                        eprintln!("  {:<40} {:>9.1} ms total {:>7} calls {:>8.1} us each", k, secs * 1e3, n, secs * 1e6 / *n as f64);
                    }
                }
            }
        }
        Ok(all.chunks(self.output.rows).map(|c| c.to_vec()).collect())
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        let n = self.layers.len();
        let ring: Vec<usize> = (0..n).map(|l| if g.swa[l] { max_ctx.min(g.window + self.chunk) } else { max_ctx }).collect();
        // per GPU: its layers' caches against what is free first (past the card the xe driver spills into host memory)
        for (i, d) in self.devs.iter().enumerate() {
            let need: u64 = (d.lb..d.le).map(|l| (ring[l] * g.kv_pos()) as u64).sum();
            let (total, free) = d.gpu.memory()?;
            let free = free.unwrap_or(total);
            if need + (1536 << 20) > free {
                return Err(e(format!("a session of {max_ctx} tokens needs {:.2} GiB of cache on {} (device {i}); {:.2} GiB are free (1.5 GiB kept spare)", need as f64 / GIB,
                                     d.gpu.name, free as f64 / GIB)));
            }
        }
        let mut layers = Vec::with_capacity(n);
        for (l, &r) in ring.iter().enumerate() {
            let gpu = &self.devs[self.layer_dev[l]].gpu;
            let b = g.heads_kv * r * g.head * 2;
            layers.push((DevBuf::new(gpu, b)?, DevBuf::new(gpu, b)?));
        }
        Ok(Session { pos: 0, max_ctx, layers, ring, last_rows: 0 })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        s.pos = 0;
        s.last_rows = 0;
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        let g = &self.g;
        for l in 0..self.layers.len() {
            let from = if g.swa[l] { src.pos.saturating_sub(g.window) } else { 0 };
            for (d, s) in [(&dst.layers[l].0, &src.layers[l].0), (&dst.layers[l].1, &src.layers[l].1)] {
                let b = kv_read(s, g.heads_kv, src.ring[l], from, src.pos)?;
                kv_write(d, g.heads_kv, dst.ring[l], from, src.pos, &b)?;
            }
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

    /// Keeps the first `keep` rows of the last pass: the caches hold their positions already, the rest are overwritten
    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep > s.last_rows {
            return Err(e(format!("rollback to {keep} rows of a pass of {}", s.last_rows)));
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

    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let tok = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => smp.sample(&l),
            (None, Some(x)) => {
                let l = self.pass(s, &[x], 1, tap)?.pop().unwrap_or_default();
                smp.sample(&l)
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        d.next = Some(tok);
        Ok(vec![tok])
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let g = &self.g;
        let mut bufs = Vec::with_capacity(2 * self.layers.len());
        for l in 0..self.layers.len() {
            let from = if g.swa[l] { s.pos.saturating_sub(g.window) } else { 0 };
            for c in [&s.layers[l].0, &s.layers[l].1] {
                bufs.push(kv_read(c, g.heads_kv, s.ring[l], from, s.pos)?);
            }
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let g = &self.g;
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        if ck.bufs.len() != 2 * self.layers.len() {
            return Err(e("a checkpoint of another shape"));
        }
        for l in 0..self.layers.len() {
            for (j, c) in [&s.layers[l].0, &s.layers[l].1].into_iter().enumerate() {
                let b = &ck.bufs[2 * l + j];
                let per = g.heads_kv * ROW;
                if per == 0 || !b.len().is_multiple_of(per) || b.len() / per > ck.pos {
                    return Err(e("a checkpoint of another shape"));
                }
                let n = b.len() / per;
                let want = if g.swa[l] { ck.pos.min(g.window) } else { ck.pos };
                if n != want {
                    return Err(e("a checkpoint of another shape"));
                }
                kv_write(c, g.heads_kv, s.ring[l], ck.pos - n, ck.pos, b)?;
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

/// A cache's positions `from..to` of every head, in position order: [head][position][ROW bytes]
fn kv_read(c: &DevBuf, heads: usize, ring: usize, from: usize, to: usize) -> Result<Vec<u8>> {
    let n = to - from;
    let mut out = vec![0u8; heads * n * ROW];
    for h in 0..heads {
        let mut at = 0;
        for (slot, rows) in runs(from, to, ring) {
            let dst = &mut out[(h * n + at) * ROW..(h * n + at + rows) * ROW];
            c.read((h * ring + slot) * ROW, dst)?;
            at += rows;
        }
    }
    Ok(out)
}

/// `kv_read`'s bytes back into a cache of `ring` rows a head
fn kv_write(c: &DevBuf, heads: usize, ring: usize, from: usize, to: usize, b: &[u8]) -> Result<()> {
    let n = to - from;
    if b.len() != heads * n * ROW {
        return Err(e("a cache's bytes of another size"));
    }
    for h in 0..heads {
        let mut at = 0;
        for (slot, rows) in runs(from, to, ring) {
            c.write((h * ring + slot) * ROW, &b[(h * n + at) * ROW..(h * n + at + rows) * ROW])?;
            at += rows;
        }
    }
    Ok(())
}

/// A float32 / half / bf16 tensor onto the GPU as float32
fn vec_f32(f: &Gguf, gpu: &Arc<Gpu>, k: &ffi::Api, name: &str) -> Result<DevBuf> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    let n: usize = t.shape.iter().product::<u64>() as usize;
    let raw = f.read(t).map_err(|x| e(x.0))?;
    let src = DevBuf::new(gpu, raw.len())?;
    src.write(0, &raw)?;
    let out = DevBuf::f32(gpu, n)?;
    // SAFETY: src holds the tensor, out its elements as float32.
    ffi::check(unsafe { (k.dequant)(gpu.raw(), t.ty.code() as i32, src.ptr(), n, out.fp()) }, name)?;
    gpu.sync()?;
    Ok(out)
}

/// Block `l` onto `gpu`: its matrices in one blob (device memory, or pinned host memory when `host`), its vectors as
/// float32 on the GPU
fn load_layer(f: &Gguf, g: &Geometry, l: usize, gpu: &Arc<Gpu>, host: bool, hosts: &mut Vec<HostBuf>, loaded: &mut u64) -> Result<Layer> {
    let k = ffi::api()?;
    let n = |s: &str| format!("blk.{l}.{s}.weight");
    let names = ["attn_q", "attn_k", "attn_v", "attn_gate", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
    let mut mats = Vec::with_capacity(names.len());
    let mut total = 0usize;
    for nm in names {
        let t = f.tensor(&n(nm)).ok_or_else(|| e(format!("no {}", n(nm))))?;
        if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
            return Err(e(format!("{}: {} - a matrix in blocks is expected", t.name, t.ty.name())));
        }
        let off = total.next_multiple_of(256);
        total = off + t.bytes as usize;
        mats.push((t, Mat { off, len: t.bytes as usize, ty: t.ty, rows: t.shape[0] as usize, cols: t.shape[1] as usize }));
    }
    let base = if host {
        let mut hb = HostBuf::new(gpu, total)?;
        for (t, m) in &mats {
            f.read_into(t, 0, &mut hb.as_mut_slice()[m.off..m.off + m.len]).map_err(|x| e(x.0))?;
        }
        let view = hb.device_view(0, total)?;
        hosts.push(hb);
        view
    } else {
        let b = DevBuf::new(gpu, total)?;
        for (t, m) in &mats {
            let raw = f.read(t).map_err(|x| e(x.0))?;
            b.write(m.off, &raw)?;
        }
        b
    };
    *loaded += mats.iter().map(|(t, _)| t.bytes).sum::<u64>();
    let m = |i: usize| mats[i].1;
    let lambda = vec_f32(f, gpu, k, &n("attn_vres_lambda"))?.to_f32()?;
    let vres = 1.0 / (1.0 + (-lambda.first().copied().unwrap_or(0.0)).exp());
    let _ = g;
    Ok(Layer {
        swa: g.swa[l],
        host,
        base,
        bytes: total,
        q: m(0),
        k: m(1),
        v: m(2),
        g: m(3),
        o: m(4),
        gate: m(5),
        up: m(6),
        down: m(7),
        attn_norm: vec_f32(f, gpu, k, &n("attn_norm"))?,
        post_attn: vec_f32(f, gpu, k, &n("post_attention_norm"))?,
        ffn_norm: vec_f32(f, gpu, k, &n("ffn_norm"))?,
        post_ffn: vec_f32(f, gpu, k, &n("post_ffw_norm"))?,
        q_norm: vec_f32(f, gpu, k, &n("attn_q_norm"))?,
        k_norm: vec_f32(f, gpu, k, &n("attn_k_norm"))?,
        vres,
    })
}

/// ggml's IQ4_NL codebook
const IQ4NL: [f32; 16] = [-127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.];

/// One row of IQ4_XS blocks (136 bytes a 256: half d, scales_h u16, scales_l[4], qs[128]) as float32
fn iq4xs_row(row: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(row.len() / 136 * 256);
    for b in row.chunks_exact(136) {
        let d = half_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let sh = u16::from_le_bytes([b[2], b[3]]);
        for ib in 0..8 {
            let lo = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            let hi = ((sh >> (2 * ib)) & 3) as u8;
            let dl = d * (f32::from(lo | (hi << 4)) - 32.0);
            let qs = &b[8 + 16 * ib..8 + 16 * ib + 16];
            out.extend(qs.iter().map(|q| dl * IQ4NL[(q & 0xf) as usize]));
            out.extend(qs.iter().map(|q| dl * IQ4NL[(q >> 4) as usize]));
        }
    }
    out
}

fn half_to_f32(h: u16) -> f32 {
    let (s, e, m) = (u32::from(h >> 15), u32::from((h >> 10) & 31), u32::from(h & 1023));
    let v = match e {
        0 => (m as f32) * 2f32.powi(-24),
        31 => if m == 0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    };
    if s == 1 { -v } else { v }
}

#[cfg(test)]
mod tests {
    use super::{half_to_f32, iq4xs_row, runs};

    #[test]
    fn ring_runs_wrap_once() {
        assert_eq!(runs(0, 5, 8), vec![(0, 5)]);
        assert_eq!(runs(6, 11, 8), vec![(6, 2), (0, 3)]);
        assert_eq!(runs(8, 16, 8), vec![(0, 8)]);
        assert!(runs(3, 3, 8).is_empty());
    }

    #[test]
    fn iq4xs_block_expands_by_ggml_formula() {
        // d = 1.0 (0x3c00), every sub-block scale 33 (low nibble 1, high bits 2 -> 33), all bytes 0x80: low nibble 0 -> -127
        let mut b = vec![0u8; 136];
        b[0] = 0x00;
        b[1] = 0x3c;
        b[2..4].copy_from_slice(&0xaaaau16.to_le_bytes());
        b[4..8].fill(0x11);
        b[8..].fill(0x80);
        let v = iq4xs_row(&b);
        assert_eq!(v.len(), 256);
        assert_eq!((v[0], v[16]), (-127.0, 1.0));
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
    }
}

