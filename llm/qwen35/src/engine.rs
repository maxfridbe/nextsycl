//! Dense Qwen3.5 / Qwen3.8 (`qwen35`, llama.cpp's src/models/qwen35.cpp): 64 layers, three of every four gated
//! DeltaNet (linear attention: a conv + SiLU, the delta-rule recurrence, a SiLU-gated norm), the fourth gated full
//! attention (a sigmoid gate a head in the query projection, q / k norms, rotary positions on the first 64 of 256
//! features), each followed by a dense SwiGLU. The weights stay in their GGUF blocks on the GPU: decode reads them
//! through the Q8_1 products (up to 8 rows), a prompt chunk expands each matrix to half and multiplies.
//!
//! ```text
//!   x = embed(tokens)
//!   per layer:  h = attn_norm(x);  x += linear(h) or full(h);  x += ffn(post_attention_norm(x))
//!   logits = output(output_norm(x))
//! ```
//!
//! MTP (the GGUF's `MTP/` file: its block `layers`): a draft block that reads a position's normed hidden state h with
//! the token that follows it and predicts the one after - eh_proj(enorm(embed(x_p)) | hnorm(h_{p-1})), a gated
//! full-attention layer with its own cache, a SwiGLU, its head norm and head. It runs behind every pass over that
//! pass's rows (so its cache follows the conversation); `step` drafts `NS_Q35_DRAFTS` tokens (the block chained on its
//! own normed output), verifies [token, drafts] in one pass and keeps the longest prefix the sampled tokens match - a
//! rejected row's DeltaNet states come back from the pass's snapshots.

use std::ffi::c_int;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Gpu};
use nextsycl_gguf::{GType, Gguf, Tensor};
use nextsycl_llm::{Error, GpuInfo, Result, Sampler, Tap};

use crate::ffi;
use crate::probe::Probe;

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
    pub n_rot: usize,
    pub theta: f32,
    pub eps: f32,
    pub conv: usize,
    pub state: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub inner: usize,
    pub interval: usize,
    pub vocab: usize,
    pub ctx: usize,
}

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let nextn = u("nextn_predict_layers").unwrap_or(0);
        let g = Geometry {
            layers: u("block_count")? - nextn,
            n_embd: u("embedding_length")?,
            n_ff: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            heads_kv: u("attention.head_count_kv")?,
            head: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            theta: fl("rope.freq_base")?,
            eps: fl("attention.layer_norm_rms_epsilon")?,
            conv: u("ssm.conv_kernel")?,
            state: u("ssm.state_size")?,
            k_heads: u("ssm.group_count")?,
            v_heads: u("ssm.time_step_rank")?,
            inner: u("ssm.inner_size")?,
            interval: u("full_attention_interval").unwrap_or(4),
            // shapes are outermost first: [vocab, n_embd]
            vocab: f.tensor("token_embd.weight").map(|t| t.shape[0] as usize).ok_or("no token_embd.weight")?,
            ctx: u("context_length").unwrap_or(262_144),
        };
        if g.state != 128 || g.inner != g.v_heads * g.state || !g.v_heads.is_multiple_of(g.k_heads) {
            return Err(format!("DeltaNet heads of {} ({} key / {} value heads, inner {}): head 128 and value heads a multiple of the key heads",
                               g.state, g.k_heads, g.v_heads, g.inner));
        }
        if g.head != 256 || !matches!(g.heads / g.heads_kv.max(1), 2 | 4 | 6 | 8) || !g.heads.is_multiple_of(g.heads_kv) {
            return Err(format!("attention heads of {} ({} / {}): head 256 with 2, 4, 6 or 8 query heads a key head", g.head, g.heads, g.heads_kv));
        }
        Ok(g)
    }

    pub fn full(&self, l: usize) -> bool {
        (l + 1).is_multiple_of(self.interval)
    }

    /// conv channels: q, k (key heads) and v (value heads)
    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.state + self.inner
    }
}

/// A decode row set's Q8_1 forms: the blocks (the shared products'), and the silo's split layout when the silo made it
pub struct Q8 {
    std: DevBuf,
    split: Option<DevBuf>,
}

/// A matrix [rows (outputs), cols (inputs)]: its stored blocks, or float32 (small or unquantized tensors)
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

struct Full {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

struct Linear {
    qkv: Mat,
    z: Mat,
    beta: Mat,
    alpha: Mat,
    /// alpha's rows then beta's in one matrix (both the same block type): one decode product for a row, not two
    ab: Option<Mat>,
    conv: DevBuf,
    dt: DevBuf,
    a: DevBuf,
    norm: DevBuf,
    out: Mat,
}

struct Layer {
    attn_norm: DevBuf,
    post_norm: DevBuf,
    full: Option<Full>,
    linear: Option<Linear>,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// A conversation: per full-attention layer its key / value cache ([kv heads][max_ctx][256] half), per DeltaNet layer
/// its recurrent state ([48][128][128] float32) and the conv's last inputs ([3][channels])
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<LState>,
    /// rows of the last forward pass
    last_rows: usize,
    /// which session (the engine's snapshots name the one they belong to)
    id: u64,
    /// the draft block's cache and inputs (MTP loaded)
    mtp: Option<MtpState>,
}

/// A session's draft block state: its cache ([kv heads][max_ctx][256] half), the normed hidden state of the last
/// position fed (the next token pairs with it), and the block's own normed output (a chained draft pairs with it)
struct MtpState {
    k: DevBuf,
    v: DevBuf,
    pending: DevBuf,
    chain: DevBuf,
}

enum LState {
    Full { k: DevBuf, v: DevBuf },
    Linear { s: DevBuf, conv: DevBuf },
}

/// A conversation's state in host memory
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(b"NSQ35")?;
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
        if &magic != b"NSQ35" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a qwen35 checkpoint"));
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

/// Generation from a fed prompt: the logits to draw the next token from, or the committed token not fed yet
pub struct Decoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
    /// draft with the MTP block
    mtp: bool,
    /// the drafts for the positions after `next`
    drafts: Vec<u32>,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn new(logits: Vec<f32>, mtp: bool) -> Decoder {
        Decoder { logits: Some(logits), next: None, mtp, drafts: Vec::new(), drafted: 0, accepted: 0 }
    }
    pub fn after(next: u32, mtp: bool) -> Decoder {
        Decoder { logits: None, next: Some(next), mtp, drafts: Vec::new(), drafted: 0, accepted: 0 }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.drafts.clear();
        self.next.take()
    }
}

/// The MTP draft block (its own file: block `layers`, its token embedding and head)
struct Mtp {
    file: Box<Gguf>,
    embd: Tensor,
    vocab: usize,
    /// eh_proj's columns for the embedding and for the hidden state (split at load: two products, no concatenation)
    eh_e: Mat,
    eh_h: Mat,
    enorm: DevBuf,
    hnorm: DevBuf,
    head_norm: DevBuf,
    attn_norm: DevBuf,
    post_norm: DevBuf,
    attn: Full,
    gate: Mat,
    up: Mat,
    down: Mat,
    output: Mat,
}

/// Each DeltaNet layer's state after each row of a verify pass but the last (its recurrence [rows-1][48][128][128],
/// its conv inputs [rows-1][3][channels]), and whose pass they are (session, first position, rows)
struct Snaps {
    layers: Vec<(DevBuf, DevBuf)>,
    of: Option<(u64, usize, usize)>,
}

/// NS_Q35_DRAFTS: tokens drafted a step (0: none; at most 7 - a verify pass is at most 8 rows)
fn draft_count() -> usize {
    std::env::var("NS_Q35_DRAFTS").ok().and_then(|v| v.parse().ok()).unwrap_or(3).min(7)
}

/// NS_Q35_MTP: the draft block's file; else an `MTP/` folder beside the model's file holding one
fn mtp_path(f: &Gguf) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NS_Q35_MTP") {
        return Some(p.into());
    }
    let dir = f.paths.first()?.parent()?.join("MTP");
    let mut v: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "gguf")).collect();
    v.sort();
    v.into_iter().next()
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

pub struct Qwen35<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    /// the GPUs in layer order (the first embeds, the last holds the head and the draft block)
    devs: Vec<Dev>,
    /// each layer's GPU (an index into devs)
    layer_dev: Vec<usize>,
    /// the GPU the kernels go to now (set under `work`: a pass walks the GPUs in order)
    cur: std::sync::atomic::AtomicUsize,
    /// host memory a pass's rows cross between GPUs through
    staging: Mutex<Vec<u8>>,
    k: &'static ffi::Api,
    /// the silo's tuned kernels (kernels/llm/qwen35/silo), None: the shared ones throughout
    silo: Option<&'static ffi::Silo>,
    layers: Vec<Layer>,
    output_norm: DevBuf,
    output: Mat,
    /// the token embedding's rows stay in the file (read a row a token)
    embd: Tensor,
    /// the rows a forward pass takes at most
    chunk: usize,
    work: Mutex<()>,
    /// the draft block (loaded with drafts on and its file found)
    mtp: Option<Mtp>,
    /// drafts a step
    drafts: usize,
    /// the verify passes' DeltaNet snapshots (under `work`)
    snaps: Mutex<Snaps>,
    /// the normed hidden states of the last pass's rows (up to 8: a rollback's pending state)
    vh: DevBuf,
    next_id: std::sync::atomic::AtomicU64,
    /// NS_Q35_PROFILE=1: every kernel timed on the device, with its bytes and arithmetic (probe.rs)
    probe: Option<Probe>,
    load_s: f64,
    load_bytes: u64,
}

/// NS_Q35_CHUNK when set; otherwise chosen at load (`Qwen35::load`: 2,048 rows when they fit)
fn chunk_setting() -> Option<usize> {
    std::env::var("NS_Q35_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= 16)
}

/// A prompt chunk's work memory (the arena) for `chunk` rows
fn arena_bytes(g: &Geometry, chunk: usize) -> usize {
    chunk * (4 * g.n_embd + 2 * g.n_ff + g.conv_dim() + g.inner + 4 * g.v_heads * g.state + g.heads * 2 * g.head + 3 * g.heads * g.head) * 4
        + chunk * g.n_ff.max(g.conv_dim()) * 2 + 8 * g.vocab * 4 + (64 << 20)
}

/// Room kept for a session when the chunk is chosen (a 32K context's cache on the 27B is 2 GiB)
const SESSION_ROOM: u64 = 5 << 29;

impl<'g> Qwen35<'g> {
    /// Loads the model over `gpus` in order: with one GPU all of it there; with more, the layers split so each GPU keeps
    /// the same share of its memory free for the sessions (NS_SPLIT: the first layer of each GPU after the first), the
    /// head and the draft block on the last. `kv`: the context the sessions reserve (tokens in all; 0: 32K) - a split
    /// plans their caches on the GPUs of their layers.
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], draft: bool, kv: (usize, usize), log: &mut dyn FnMut(String)) -> Result<Qwen35<'g>> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(e("no GPU"));
        }
        let g = Geometry::read(f).map_err(e)?;
        let k = ffi::api()?;
        let silo = ffi::silo()?;
        log(match silo {
            Some(s) => format!("qwen35: tuned kernels from its silo {}", s.path.display()),
            None => "qwen35: no silo - the shared kernels".into(),
        });
        // the plan: every tensor but the token embedding, a prompt chunk's work, 1.5 GiB to spare (the xe driver has
        // no out-of-memory error: past the card it spills to host memory)
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        let weights: u64 = f.tensors.iter().filter(|t| t.name != "token_embd.weight" && !t.name.starts_with(&format!("blk.{}.", g.layers))).map(|t| t.bytes).sum();
        let layer_w: Vec<u64> = (0..g.layers).map(|l| {
            let p = format!("blk.{l}.");
            f.tensors.iter().filter(|t| t.name.starts_with(&p)).map(|t| t.bytes).sum()
        }).collect();
        let head_w = weights - layer_w.iter().sum::<u64>();
        let w16_bytes = [g.n_ff * g.n_embd, g.heads * 2 * g.head * g.n_embd, g.conv_dim() * g.n_embd].into_iter().max().unwrap_or(0) * 2;
        let mut free = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let (total, fr) = gpu.memory()?;
            free.push(fr.unwrap_or(total));
        }
        // the prompt chunk: 2,048 rows (each matrix expanded to half once for them: 8K prompts 760 -> 1,100 tok/s against
        // 512) when they leave room for a session, else 1,024, else 512
        // the draft block: its file's tensors but its token embedding (rows read from the file), its snapshots
        let drafts = if draft { draft_count() } else { 0 };
        let mfile = match (drafts > 0).then(|| mtp_path(f)).flatten() {
            Some(p) => match Gguf::open(&p) {
                Ok(m) => Some(Box::new(m)),
                Err(x) => {
                    log(format!("qwen35: the MTP file {} does not open ({}); no drafts", p.display(), x.0));
                    None
                }
            },
            None => None,
        };
        let lin_state = (g.v_heads * g.state * g.state + (g.conv - 1) * g.conv_dim()) * 4;
        let snap_bytes = (g.layers - (0..g.layers).filter(|l| g.full(*l)).count()) * drafts * lin_state;
        let mtp_w = mfile.as_ref().map_or(0, |m| m.tensors.iter().filter(|t| t.name != "token_embd.weight").map(|t| t.bytes).sum::<u64>());
        let mtp_bytes = mtp_w + if mfile.is_some() { snap_bytes as u64 } else { 0 };
        let margin: u64 = 1536 << 20;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        // the draft block's pass behind a chunk's: its inputs and outputs beside the main pass's
        let has_mtp = mfile.is_some();
        let arena_of = |c: usize| arena_bytes(&g, c) + if has_mtp { c * 8 * g.n_embd * 4 } else { 0 };
        let (chunk, bounds) = if gpus.len() == 1 {
            let fixed = weights + mtp_bytes + w16_bytes as u64 + margin;
            let chunk = chunk_setting().unwrap_or_else(|| {
                [2048, 1024].into_iter().find(|c| fixed + arena_bytes(&g, *c) as u64 + SESSION_ROOM <= free[0]).unwrap_or(512)
            });
            let need = fixed + arena_of(chunk) as u64;
            if need > free[0] {
                return Err(e(format!("{} needs about {:.1} GiB on {} ({:.1} GiB of weights); {:.1} GiB are free", f.architecture(), gib(need),
                                     gpus[0].name, gib(weights), gib(free[0]))));
            }
            (chunk, vec![(0, g.layers)])
        } else {
            // a layer's cost where it lands: its weights, its cache for the planned context (full attention) or its
            // state and verify snapshots (DeltaNet)
            let ctx = if kv.0 > 0 { kv.0 } else { 32768 };
            let mtp_on = has_mtp;
            let cost: Vec<u64> = (0..g.layers).map(|l| layer_w[l] + if g.full(l) { (2 * g.heads_kv * ctx * g.head * 2) as u64 } else {
                (lin_state * (1 + if mtp_on { drafts } else { 0 })) as u64
            }).collect();
            let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            let plan = |c: usize| -> std::result::Result<Vec<(usize, usize)>, String> {
                let n = gpus.len();
                // what each GPU has for layers: its free memory less the spare, its half buffer and work memory; the
                // last's less the head and the draft block (its weights, its cache)
                let room: Vec<i64> = (0..n).map(|i| {
                    // (and 512 MiB the plan does not see: the allocator's rounding, small buffers)
                    let mut over = margin + (512 << 20) + w16_bytes as u64 + arena_of(c) as u64;
                    if i + 1 == n {
                        over += head_w + if mtp_on { mtp_w + (2 * g.heads_kv * ctx * g.head * 2) as u64 } else { 0 };
                    }
                    free[i] as i64 - over as i64
                }).collect();
                let mut bounds = Vec::with_capacity(n);
                if let Some(sp) = &split_env {
                    let mut lb = 0;
                    for i in 0..n {
                        let le = if i + 1 == n { g.layers } else { *sp.get(i).ok_or("NS_SPLIT: a first layer for each GPU after the first")? };
                        bounds.push((lb, le.min(g.layers)));
                        lb = le.min(g.layers);
                    }
                } else {
                    // the last GPU first (the one that computes most: the server orders them weakest first), then each
                    // before it: each takes the last layers left while they fit; the first the rest. A GPU the layers
                    // do not reach is not used (a B70 runs a layer ~1.5x faster than a B65: no layer goes to the slower
                    // card while the faster one has room)
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
                    bounds = rev;
                }
                for (i, &(lb, le)) in bounds.iter().enumerate() {
                    let need: u64 = cost[lb..le].iter().sum();
                    if le <= lb {
                        // a GPU before the last with no layers left for it is dropped (`load`)
                        if i + 1 < n && split_env.is_none() {
                            continue;
                        }
                        return Err(format!("{} gets no layers", gpus[i].name));
                    }
                    if need as i64 > room[i] {
                        return Err(format!("layers {lb}-{} need {:.1} GiB on {}; {:.1} GiB are left for them", le - 1, gib(need), gpus[i].name,
                                           gib(room[i].max(0) as u64)));
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
            found.ok_or_else(|| e(format!("{} over {} GPUs at a {ctx}-token context: {last_err}", f.architecture(), gpus.len())))?
        };
        let given = gpus.len();
        let (gpus, bounds): (Vec<Arc<Gpu>>, Vec<(usize, usize)>) = gpus.iter().cloned().zip(bounds).filter(|(_, b)| b.1 > b.0).unzip();
        if gpus.len() < given {
            log(format!("qwen35: all of it fits {}: {} of the {given} GPUs not used", gpus.iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(" + "),
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
            let mut read = |name: &str| -> Result<Mat> { vec_mat(f, gpu, k, name, &mut loaded) };
            let full = if g.full(l) {
                Some(Full { q: read(&n("attn_q"))?, k: read(&n("attn_k"))?, v: read(&n("attn_v"))?, o: read(&n("attn_output"))?,
                            q_norm: read(&n("attn_q_norm"))?.buf, k_norm: read(&n("attn_k_norm"))?.buf })
            } else {
                None
            };
            let linear = if g.full(l) {
                None
            } else {
                Some(Linear {
                    qkv: read(&n("attn_qkv"))?,
                    z: read(&n("attn_gate"))?,
                    beta: read(&n("ssm_beta"))?,
                    alpha: read(&n("ssm_alpha"))?,
                    ab: None,
                    conv: read(&n("ssm_conv1d"))?.buf,
                    dt: read(&format!("blk.{l}.ssm_dt.bias"))?.buf,
                    a: read(&format!("blk.{l}.ssm_a"))?.buf,
                    norm: read(&n("ssm_norm"))?.buf,
                    out: read(&n("ssm_out"))?,
                })
            };
            layers.push(Layer {
                attn_norm: read(&n("attn_norm"))?.buf,
                post_norm: read(&n("post_attention_norm"))?.buf,
                full,
                linear,
                gate: read(&n("ffn_gate"))?,
                up: read(&n("ffn_up"))?,
                down: read(&n("ffn_down"))?,
            });
            if l % 16 == 15 {
                log(format!("qwen35: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, loaded as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64()));
            }
        }
        // alpha and beta stacked (their bytes one after the other) where they share a block type
        for ly in layers.iter_mut() {
            if let Some(a) = ly.linear.as_mut() {
                if a.alpha.ty == a.beta.ty && a.alpha.cols == a.beta.cols && a.alpha.ty != GType::F32 {
                    let (na, nb) = (a.alpha.buf.len, a.beta.buf.len);
                    let buf = DevBuf::new(&a.alpha.buf.gpu().clone(), na + nb)?;
                    buf.copy_within(0, &a.alpha.buf, 0, na)?;
                    buf.copy_within(na, &a.beta.buf, 0, nb)?;
                    a.ab = Some(Mat { buf, ty: a.alpha.ty, rows: a.alpha.rows + a.beta.rows, cols: a.alpha.cols });
                }
            }
        }
        let output_norm = vec_mat(f, &lastg, k, "output_norm.weight", &mut loaded)?.buf;
        let output = vec_mat(f, &lastg, k, if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }, &mut loaded)?;
        let mtp = match mfile {
            Some(m) => Some(load_mtp(m, &g, &lastg, k, &mut loaded)?),
            None => None,
        };
        let drafts = if mtp.is_some() { drafts } else { 0 };
        let mut snap_layers = Vec::new();
        if drafts > 0 {
            for l in (0..g.layers).filter(|l| !g.full(*l)) {
                let gpu = &gpus[layer_dev[l]];
                snap_layers.push((DevBuf::f32(gpu, drafts * g.v_heads * g.state * g.state)?, DevBuf::f32(gpu, drafts * (g.conv - 1) * g.conv_dim())?));
            }
        }
        log(match &mtp {
            Some(_) => format!("qwen35: MTP drafts ({drafts} a step)"),
            None if draft => "qwen35: no MTP file (NS_Q35_MTP or an MTP/ folder beside the model): no drafts".into(),
            None => "qwen35: drafts off".into(),
        });
        let vh = DevBuf::f32(&lastg, 8 * g.n_embd)?;
        let mut devs = Vec::with_capacity(gpus.len());
        for (gpu, &(lb, le)) in gpus.iter().zip(&bounds) {
            gpu.sync()?;
            devs.push(Dev { gpu: gpu.clone(), w16: DevBuf::new(gpu, w16_bytes)?, arena: Arena::new(gpu, arena_of(chunk))?, lb, le });
        }
        let load_s = t0.elapsed().as_secs_f64();
        let wh = devs.iter().map(|d| if d.lb == 0 && d.le == g.layers { d.gpu.name.clone() } else { format!("{} (layers {}-{})", d.gpu.name, d.lb, d.le - 1) })
            .collect::<Vec<_>>().join(", ");
        log(format!("qwen35: {} layers of {} ({} full attention), {:.1} GiB on {wh} in {load_s:.0} s; prompt chunks of {chunk}", g.layers, g.n_embd,
                    (0..g.layers).filter(|l| g.full(*l)).count(), gib(loaded)));
        // the probe stamps one GPU's queue: a split runs without it
        let probe = if devs.len() == 1 {
            Probe::new(devs[0].gpu.raw())?
        } else {
            if std::env::var("NS_Q35_PROFILE").is_ok_and(|v| v == "1") {
                log("qwen35: NS_Q35_PROFILE times one GPU: off for a split".into());
            }
            None
        };
        Ok(Qwen35 { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(Vec::new()), k, silo, layers, output_norm, output, embd, chunk,
                    work: Mutex::new(()), mtp, drafts, snaps: Mutex::new(Snaps { layers: snap_layers, of: None }), vh, next_id: 0.into(), probe, load_s,
                    load_bytes: loaded })
    }

    /// A kernel call (its return code checked): timed on the device with NS_Q35_PROFILE=1, with the bytes it moves
    /// and the arithmetic it does
    fn op(&self, name: &'static str, key: impl FnOnce() -> String, bytes: usize, flops: usize, f: impl FnOnce() -> c_int) -> Result<()> {
        let rc = self.op_rc(name, key, bytes, flops, f)?;
        ffi::check(rc, name)
    }

    /// `op`, its return code left to the caller
    fn op_rc(&self, name: &'static str, key: impl FnOnce() -> String, bytes: usize, flops: usize, f: impl FnOnce() -> c_int) -> Result<c_int> {
        match &self.probe {
            Some(p) => p.op(name, key(), bytes as u64, flops as u64, f),
            None => Ok(f()),
        }
    }

    /// What the next kernels are part of (the probe's grouping)
    fn phase(&self, p: &'static str) {
        if let Some(pr) = &self.probe {
            pr.phase(p);
        }
    }

    /// The probe's tables (`report`)
    pub fn profile_lines(&self) -> Vec<String> {
        self.probe.as_ref().map_or_else(Vec::new, |p| p.lines())
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

    fn last(&self) -> usize {
        self.devs.len() - 1
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
        // the copy waits for the GPU before (its layers) and the copy into this one
        y.copy_from_peer(x, &mut st)?;
        Ok(y)
    }

    /// y [t, w.rows] = x [t, w.cols] . W^T; `role` names it in the probe
    /// `xq`: x's Q8_1 forms when the norm made them (decode rows), so the product does not quantize x again
    fn matmul(&self, role: &'static str, w: &Mat, x: &DevBuf, xq: Option<&Q8>, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        let (rows, cols) = (w.rows, w.cols);
        let key = || format!("{role} {} {rows}x{cols}", w.ty.name());
        let wb = w.ty.bytes((rows * cols) as u64).unwrap_or(0) as usize;
        let flops = 2 * t * rows * cols;
        if w.ty == GType::F32 {
            // SAFETY: x [t, cols], the matrix [rows, cols] float32, y [t, rows].
            return self.op("gemm", key, 4 * (t * cols + rows * cols + t * rows), flops,
                           || unsafe { (k.gemm)(self.raw(), t as i64, n, kk, x.fp(), kk, w.buf.fp(), y.fp(), n, 0) });
        }
        // SAFETY: arithmetic only.
        let mmvq = unsafe { (k.mmvq_supported)(w.ty.code() as i32) } != 0;
        if t <= 8 && mmvq {
            // SAFETY: arithmetic only.
            let q8 = unsafe { (k.q8_1_bytes)(kk, t as i64) };
            // the silo's product for the types it covers (kernels/llm/qwen35/silo; it reads the split layout too), the
            // shared one otherwise
            // SAFETY: arithmetic only.
            let silo = self.silo.filter(|s| unsafe { (s.mmvq_supported)(w.ty.code() as i32, kk) } != 0);
            let own;
            let qb = match xq {
                Some(q) => q,
                None => {
                    own = self.quantize(x, cols, t, silo.is_some(), &key)?;
                    &own
                }
            };
            return match (silo, &qb.split) {
                (Some(s), Some(sp)) => self.op("mmvq silo", key, wb + q8 + 4 * t * rows, flops, || unsafe {
                    // SAFETY: the matrix in its blocks, x in both forms, y [t, rows].
                    (s.mmvq)(self.raw(), w.ty.code() as i32, w.buf.ptr(), qb.std.ptr(), sp.ptr(), y.fp(), kk, n, t as i64)
                }),
                // SAFETY: the matrix in its blocks, x's Q8_1 blocks, y [t, rows].
                _ => self.op("mmvq", key, wb + q8 + 4 * t * rows, flops,
                             || unsafe { (k.mmvq)(self.raw(), w.ty.code() as i32, w.buf.ptr(), qb.std.ptr(), y.fp(), kk, n, t as i64) }),
            };
        }
        // the matrix expanded to half (directly, or through float32), x to half, then the half product
        let elems = rows * cols;
        if elems * 2 > self.dev().w16.len {
            return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
        }
        // SAFETY: the matrix's blocks -> rows x cols halfs in w16 (non-zero: a type it does not expand directly) - the
        // silo's expansion for the types it covers
        let rc = match self.silo {
            Some(s) if unsafe { (s.dequant_f16_supported)(w.ty.code() as i32) } != 0 => self.op_rc("dequant f16 silo", key, wb + 2 * elems, 0, || unsafe {
                (s.dequant_f16)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems as i64, self.dev().w16.ptr().cast())
            })?,
            _ => self.op_rc("dequant f16", key, wb + 2 * elems, 0,
                            || unsafe { (k.dequant_f16)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems as i64, self.dev().w16.ptr().cast()) })?,
        };
        if rc != 0 {
            let f = self.arena().f32(elems)?;
            // SAFETY: the blocks -> float32, then -> half.
            self.op("dequant", key, wb + 4 * elems, 0, || unsafe { (k.dequant)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems, f.fp()) })?;
            self.op("to half", key, 6 * elems, 0, || unsafe { (k.to_f16)(self.raw(), f.fp(), self.dev().w16.ptr().cast(), elems as i64) })?;
        }
        let xh = self.arena().bytes(t * cols * 2)?;
        // SAFETY: x [t, cols] -> half; then y [t, rows] = xh . w16^T.
        self.op("x to half", key, 6 * t * cols, 0, || unsafe { (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (t * cols) as i64) })?;
        self.op("gemm f16", key, 2 * t * cols + 2 * elems + 4 * t * rows, flops,
                || unsafe { (k.gemm_f16)(self.raw(), t as i64, n, kk, xh.ptr().cast(), kk, self.dev().w16.ptr().cast(), y.fp(), n, 0) })
    }

    /// The RMS norm into y; for decode rows with the silo also y's Q8_1 forms (the products after it read them)
    fn norm(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<Option<Q8>> {
        let Some(s) = self.silo.filter(|_| rows <= 8 && c.is_multiple_of(256)) else {
            self.rms(x, w, y, rows, c)?;
            return Ok(None);
        };
        // SAFETY: arithmetic only.
        let (q8, sb) = unsafe { ((self.k.q8_1_bytes)(c as i64, rows as i64), (s.q8_split_bytes)(c as i64, rows as i64)) };
        let (qb, sp) = (self.arena().bytes(q8)?, self.arena().bytes(sb)?);
        // SAFETY: x and y [rows, c], w c floats, qb rows x c / 32 Q8_1 blocks, sp the split layout.
        self.op("rms norm q8", String::new, 12 * rows * c + 4 * c + q8 + sb, 6 * rows * c,
                || unsafe { (s.rms_q8)(self.raw(), x.fp(), w.fp(), y.fp(), qb.ptr(), sp.ptr(), rows as i64, c as i64, self.g.eps) })?;
        Ok(Some(Q8 { std: qb, split: Some(sp) }))
    }

    /// x [t, cols] -> its Q8_1 forms: with `split` (a silo product) the silo's quantization, both layouts in one launch
    fn quantize(&self, x: &DevBuf, cols: usize, t: usize, split: bool, key: &dyn Fn() -> String) -> Result<Q8> {
        let k = self.k;
        // SAFETY: arithmetic only.
        let q8 = unsafe { (k.q8_1_bytes)(cols as i64, t as i64) };
        let std = self.arena().bytes(q8)?;
        if let Some(s) = self.silo.filter(|_| split) {
            // SAFETY: arithmetic only.
            let sb = unsafe { (s.q8_split_bytes)(cols as i64, t as i64) };
            let sp = self.arena().bytes(sb)?;
            // SAFETY: x [t, cols] float32 -> its Q8_1 blocks and the split layout.
            self.op("quantize", key, 4 * t * cols + q8 + sb, 0,
                    || unsafe { (s.quantize_q8)(self.raw(), x.fp(), std.ptr(), sp.ptr(), cols as i64, t as i64) })?;
            return Ok(Q8 { std, split: Some(sp) });
        }
        // SAFETY: x [t, cols] float32 -> its Q8_1 blocks.
        self.op("quantize", key, 4 * t * cols + q8, 0, || unsafe { (k.quantize_q8_1)(self.raw(), x.fp(), std.ptr(), cols as i64, t as i64) })?;
        Ok(Q8 { std, split: None })
    }

    fn rms(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<()> {
        // SAFETY: x and y [rows, c], w c floats.
        self.op("rms norm", String::new, 8 * rows * c + 4 * c, 4 * rows * c,
                || unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, self.g.eps) })
    }

    fn add(&self, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        self.op("add", String::new, 12 * n, n, || unsafe { (self.k.add)(self.raw(), y.fp(), x.fp(), n as i64) })
    }

    /// The tokens' embeddings [t, n_embd] (their rows read from the file, expanded on the GPU)
    fn embed(&self, tokens: &[u32]) -> Result<DevBuf> {
        self.embed_from(self.file, &self.embd, self.g.vocab, tokens)
    }

    /// Rows of a token embedding kept in its file
    fn embed_from(&self, file: &Gguf, embd: &Tensor, vocab: usize, tokens: &[u32]) -> Result<DevBuf> {
        let g = &self.g;
        let row = embd.ty.bytes(g.n_embd as u64).ok_or_else(|| e("the embedding's row size"))? as usize;
        let mut raw = vec![0u8; tokens.len() * row];
        for (i, t) in tokens.iter().enumerate() {
            if *t as usize >= vocab {
                return Err(e(format!("token {t} past the vocabulary ({vocab})")));
            }
            file.read_into(embd, *t as u64 * row as u64, &mut raw[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
        }
        let src = self.arena().bytes(raw.len())?;
        src.write(0, &raw)?;
        let x = self.arena().f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32.
        self.op("dequant", String::new, raw.len() + 4 * tokens.len() * g.n_embd, 0,
                || unsafe { (self.k.dequant)(self.raw(), embd.ty.code() as i32, src.ptr(), tokens.len() * g.n_embd, x.fp()) })?;
        Ok(x)
    }

    /// A full-attention layer on h [t, n_embd]: its output [t, n_embd]
    #[allow(clippy::too_many_arguments)]
    fn full(&self, a: &Full, kc: &DevBuf, vc: &DevBuf, cap: usize, h: &DevBuf, hq8: Option<&Q8>, t: usize, p0: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        let qf = self.arena().f32(t * hq * 2 * d)?;
        let kr = self.arena().f32(t * hk * d)?;
        let vr = self.arena().f32(t * hk * d)?;
        self.matmul("q", &a.q, h, hq8, t, &qf)?;
        self.matmul("k", &a.k, h, hq8, t, &kr)?;
        self.matmul("v", &a.v, h, hq8, t, &vr)?;
        let q = self.arena().f32(t * hq * d)?;
        let kn = self.arena().f32(t * hk * d)?;
        let att = self.arena().f32(t * hq * d)?;
        let none = String::new;
        // the keys the rows see in all (causal): what attention reads and multiplies
        let keys = t * p0 + t * (t + 1) / 2;
        // SAFETY: the rows and caches as named in q35.h.
        unsafe {
            self.op("q norm rope", none, 8 * t * hq * d, 10 * t * hq * d,
                    || (k.qk_norm_rope)(self.raw(), qf.fp(), (hq * 2 * d) as i64, (2 * d) as i64, q.fp(), (hq * d) as i64, t as i64, hq as i64, d as i64,
                                        a.q_norm.fp(), g.eps, g.n_rot as i64, g.theta, p0 as i64))?;
            self.op("k norm rope", none, 8 * t * hk * d, 10 * t * hk * d,
                    || (k.qk_norm_rope)(self.raw(), kr.fp(), (hk * d) as i64, d as i64, kn.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64,
                                        a.k_norm.fp(), g.eps, g.n_rot as i64, g.theta, p0 as i64))?;
            self.op("kv store", none, 12 * t * hk * d, 0,
                    || (k.kv_store)(self.raw(), kn.fp(), (hk * d) as i64, vr.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64, cap as i64, p0 as i64,
                                    kc.ptr(), vc.ptr()))?;
            let silo = self.silo.filter(|s| (s.attn_prompt_supported)(t as i64, hq as i64, hk as i64, d as i64) != 0);
            if let Some(s) = silo {
                // the silo's flash attention (prompt rows): each key read once a work-group of 8 rows x 6 heads
                // each block of 8 rows reads K and V (half) up to its last row's key, once for the 6 heads
                let nb = t.div_ceil(8);
                let block_keys = nb * p0 + 8 * nb * (nb + 1) / 2;
                self.op("attention silo", none, block_keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                        || (s.attn_prompt)(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64,
                                           p0 as i64, att.fp()))?;
                self.op("output gate", none, 12 * t * hq * d, 4 * t * hq * d,
                        || (k.gate_mul)(self.raw(), att.fp(), qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64))?;
                let out = self.arena().f32(t * g.n_embd)?;
                self.matmul("o", &a.o, &att, None, t, &out)?;
                return Ok(out);
            }
            if let Some(s) = self.silo.filter(|s| (s.attn_decode_supported)(t as i64, hq as i64, hk as i64, d as i64) != 0) {
                // the silo's decode attention: a lane a key for the scores, the keys over ~512 work-groups
                let sc = (s.attn_decode_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64) as usize;
                let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
                self.op("attention silo", none, keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                        || (s.attn_decode)(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64,
                                           p0 as i64, att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
                self.op("output gate", none, 12 * t * hq * d, 4 * t * hq * d,
                        || (k.gate_mul)(self.raw(), att.fp(), qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64))?;
                let out = self.arena().f32(t * g.n_embd)?;
                self.matmul("o", &a.o, &att, None, t, &out)?;
                return Ok(out);
            }
            let sc = (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64) as usize;
            let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
            self.op("attention", none, keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                    || (k.attn)(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64, p0 as i64,
                                att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
            self.op("output gate", none, 12 * t * hq * d, 4 * t * hq * d,
                    || (k.gate_mul)(self.raw(), att.fp(), qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64))?;
        }
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul("o", &a.o, &att, None, t, &out)?;
        Ok(out)
    }

    /// A DeltaNet layer on h [t, n_embd]: its output [t, n_embd]
    /// `snap`: where the state after each row but the last goes (a verify pass's: its recurrence's, its conv's)
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, a: &Linear, s: &DevBuf, conv: &DevBuf, h: &DevBuf, hq8: Option<&Q8>, t: usize, snap: Option<(&DevBuf, &DevBuf)>) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hv, hk, d, cd) = (g.v_heads, g.k_heads, g.state, g.conv_dim());
        let qkv = self.arena().f32(t * cd)?;
        let z = self.arena().f32(t * g.inner)?;
        self.matmul("qkv", &a.qkv, h, hq8, t, &qkv)?;
        self.matmul("z", &a.z, h, hq8, t, &z)?;
        // one row: alpha and beta from their stacked matrix in one product
        let (alpha, beta) = match &a.ab {
            Some(ab) if t == 1 => {
                let both = self.arena().f32(2 * hv)?;
                self.matmul("alpha beta", ab, h, hq8, t, &both)?;
                (both.view(0, hv * 4)?, both.view(hv * 4, hv * 4)?)
            }
            _ => {
                let beta = self.arena().f32(t * hv)?;
                let alpha = self.arena().f32(t * hv)?;
                self.matmul("beta", &a.beta, h, hq8, t, &beta)?;
                self.matmul("alpha", &a.alpha, h, hq8, t, &alpha)?;
                (alpha, beta)
            }
        };
        let eg = self.arena().f32(t * hv * d)?;
        let co = self.arena().f32(t * cd)?;
        let q = self.arena().f32(t * hv * d)?;
        let kk = self.arena().f32(t * hv * d)?;
        let v = self.arena().f32(t * hv * d)?;
        let o = self.arena().f32(t * hv * d)?;
        let y = self.arena().f32(t * g.inner)?;
        let qd = (hk * d) as i64;
        let none = String::new;
        let thd = t * hv * d;
        // SAFETY: the buffers as named in glm.h / q35.h.
        unsafe {
            self.op("gates", none, 4 * (3 * t * hv + 2 * hv) + 4 * thd, 20 * t * hv,
                    || (k.gdn_gates)(self.raw(), alpha.fp(), beta.fp(), a.dt.fp(), a.a.fp(), eg.fp(), t as i64, hv as i64, d as i64))?;
            self.op("conv silu", none, 4 * (2 * t * cd + 2 * (g.conv - 1) * cd + g.conv * cd), 2 * g.conv * t * cd + 4 * t * cd,
                    || (k.conv_silu)(self.raw(), qkv.fp(), conv.fp(), a.conv.fp(), co.fp(), t as i64, cd as i64, g.conv as i32, snap.map_or(std::ptr::null_mut(), |x| x.1.fp())))?;
            self.op("expand q", none, 4 * t * hk * d + 4 * thd, 0, || (k.expand)(self.raw(), co.fp(), cd as i64, q.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand k", none, 4 * t * hk * d + 4 * thd, 0,
                    || (k.expand)(self.raw(), co.fp().add(qd as usize), cd as i64, kk.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand v", none, 8 * thd, 0, || (k.expand)(self.raw(), co.fp().add(2 * qd as usize), cd as i64, v.fp(), t as i64, hv as i64, hv as i64, d as i64))?;
            self.op("l2 q", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), q.fp(), (t * hv) as i64, d as i64, g.eps))?;
            self.op("l2 k", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), kk.fp(), (t * hv) as i64, d as i64, g.eps))?;
            // the state [hv][d][d] read and written once, each row's q, k, v, decay in and output out; per row and head
            // the state decayed, read against k, updated, read against q: ~8 d^2
            self.op("delta rule", none, 8 * hv * d * d + 4 * (5 * thd + t * hv), 8 * t * hv * d * d,
                    || (k.kda_scan)(self.raw(), q.fp(), kk.fp(), v.fp(), eg.fp(), beta.fp(), s.fp(), o.fp(), t as i64, hv as i64, d as i64, snap.map_or(std::ptr::null_mut(), |x| x.0.fp())))?;
            self.op("gated norm", none, 4 * (3 * t * g.inner + d), 10 * t * g.inner,
                    || (k.gdn_out)(self.raw(), o.fp(), z.fp(), g.inner as i64, a.norm.fp(), y.fp(), t as i64, hv as i64, d as i64, g.eps))?;
        }
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul("out", &a.out, &y, None, t, &out)?;
        Ok(out)
    }

    /// One pass of at most a chunk: the last `n_out` rows' logits
    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        self.pass_in(s, tokens, n_out, tap)
    }

    /// `pass` with the work lock held: a verify pass (2 ..= drafts + 1 rows, drafts on) keeps the DeltaNet
    /// snapshots; with MTP the draft block runs over the rows behind it
    fn pass_in(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
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
        let mut snaps = self.snaps.lock().unwrap();
        let snapping = (2..=self.drafts + 1).contains(&t) && !snaps.layers.is_empty();
        snaps.of = None;
        self.phase("embed");
        let mut x = self.embed(tokens)?;
        let mut h = self.arena().f32(t * g.n_embd)?;
        let mut ffn_g = self.arena().f32(t * g.n_ff)?;
        let mut ffn_u = self.arena().f32(t * g.n_ff)?;
        let mut ffn = self.arena().f32(t * g.n_embd)?;
        let mut li = 0;
        // the arena past this point is the layer's own, made again each layer
        for (l, (ly, st)) in self.layers.iter().zip(s.layers.iter()).enumerate() {
            // the next GPU's layers: the rows cross over, its own buffers for them
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                self.phase("GPU to GPU");
                x = self.hop(&x, self.layer_dev[l])?;
                h = self.arena().f32(t * g.n_embd)?;
                ffn_g = self.arena().f32(t * g.n_ff)?;
                ffn_u = self.arena().f32(t * g.n_ff)?;
                ffn = self.arena().f32(t * g.n_embd)?;
            }
            let mark = self.arena().mark();
            self.phase(if ly.full.is_some() { "attention" } else { "deltanet" });
            let hq8 = self.norm(&x, &ly.attn_norm, &h, t, g.n_embd)?;
            let out = match (st, &ly.full, &ly.linear) {
                (LState::Full { k, v }, Some(a), _) => self.full(a, k, v, s.max_ctx, &h, hq8.as_ref(), t, p0)?,
                (LState::Linear { s: rs, conv }, _, Some(a)) => {
                    let sn = snapping.then(|| (&snaps.layers[li].0, &snaps.layers[li].1));
                    li += 1;
                    self.linear(a, rs, conv, &h, hq8.as_ref(), t, sn)?
                }
                _ => return Err(e(format!("layer {l}: its state and its weights disagree"))),
            };
            self.add(&x, &out, t * g.n_embd)?;
            self.phase("ffn");
            let hq8 = self.norm(&x, &ly.post_norm, &h, t, g.n_embd)?;
            self.matmul("gate", &ly.gate, &h, hq8.as_ref(), t, &ffn_g)?;
            self.matmul("up", &ly.up, &h, hq8.as_ref(), t, &ffn_u)?;
            let nf = t * g.n_ff;
            // SAFETY: n floats each.
            self.op("swiglu", String::new, 12 * nf, 6 * nf,
                    || unsafe { (self.k.swiglu_clamp)(self.raw(), ffn_g.fp(), ffn_u.fp(), ffn_g.fp(), nf as i64, f32::INFINITY) })?;
            self.matmul("down", &ly.down, &ffn_g, None, t, &ffn)?;
            self.add(&x, &ffn, t * g.n_embd)?;
            tap(&format!("l_out-{l}"), &x)?;
            self.arena().rewind(mark);
        }
        if snapping {
            snaps.of = Some((s.id, p0, t));
        }
        drop(snaps);
        s.pos += t;
        s.last_rows = t;
        // the output norm: the last rows' for the head; every row's with MTP (the draft block reads them)
        let n = n_out.clamp(1, t);
        let ne = g.n_embd;
        self.phase("head");
        let (normed, nq8) = if let Some(ms) = s.mtp.as_ref().filter(|_| self.mtp.is_some()) {
            let all = self.arena().f32(t * ne)?;
            let q = self.norm(&x, &self.output_norm, &all, t, ne)?;
            tap("result_norm", &all.view((t - n) * ne * 4, n * ne * 4)?)?;
            // the rows kept for a rollback; the draft block over this pass's rows: row i pairs token i with the hidden
            // state before it (the pending one, then this pass's)
            self.vh.copy_within(0, &all, 0, t.min(8) * ne * 4)?;
            self.phase("mtp");
            let hs = self.arena().f32(t * ne)?;
            hs.copy_within(0, &ms.pending, 0, ne * 4)?;
            if t > 1 {
                hs.copy_within(ne * 4, &all, 0, (t - 1) * ne * 4)?;
            }
            self.mtp_rows(ms, tokens, &hs, p0, s.max_ctx, false)?;
            ms.pending.copy_within(0, &all, (t - 1) * ne * 4, ne * 4)?;
            self.phase("head");
            // the head reads the rows' Q8_1 forms when it takes them all; fewer rows (a short prompt's last) it quantizes
            // itself (the split layout does not slice by rows)
            (all.view((t - n) * ne * 4, n * ne * 4)?, if n == t { q } else { None })
        } else {
            let last = x.view((t - n) * ne * 4, n * ne * 4)?;
            let normed = self.arena().f32(n * ne)?;
            let q = self.norm(&last, &self.output_norm, &normed, n, ne)?;
            tap("result_norm", &normed)?;
            (normed, q)
        };
        let logits = self.arena().f32(n * self.output.rows)?;
        self.matmul("output", &self.output, &normed, nq8.as_ref(), n, &logits)?;
        let all = logits.to_f32()?;
        if let Some(p) = &self.probe {
            p.settle(t)?;
        }
        Ok(all.chunks(self.output.rows).map(|c| c.to_vec()).collect())
    }

    /// The draft block over `tokens` at positions p0.. (row i: token i with hidden state hs[i]): its cache written
    /// there; with `head`, the last row's logits (and its normed output kept for a chained draft)
    /// the last row's likeliest token (`head`: its logits' argmax, taken on the GPU - 4 bytes read back, not the logits)
    fn mtp_rows(&self, ms: &MtpState, tokens: &[u32], hs: &DevBuf, p0: usize, cap: usize, head: bool) -> Result<Option<u32>> {
        let Some(m) = &self.mtp else { return Ok(None) };
        let g = &self.g;
        let (t, ne) = (tokens.len(), g.n_embd);
        let emb = self.embed_from(&m.file, &m.embd, m.vocab, tokens)?;
        let en = self.arena().f32(t * ne)?;
        self.rms(&emb, &m.enorm, &en, t, ne)?;
        let hn = self.arena().f32(t * ne)?;
        self.rms(hs, &m.hnorm, &hn, t, ne)?;
        // eh_proj . [en | hn] = W_e . en + W_h . hn
        let cur = self.arena().f32(t * ne)?;
        self.matmul("mtp eh e", &m.eh_e, &en, None, t, &cur)?;
        let tmp = self.arena().f32(t * ne)?;
        self.matmul("mtp eh h", &m.eh_h, &hn, None, t, &tmp)?;
        self.add(&cur, &tmp, t * ne)?;
        let h = self.arena().f32(t * ne)?;
        let hq8 = self.norm(&cur, &m.attn_norm, &h, t, ne)?;
        let att = self.full(&m.attn, &ms.k, &ms.v, cap, &h, hq8.as_ref(), t, p0)?;
        self.add(&cur, &att, t * ne)?;
        let hq8 = self.norm(&cur, &m.post_norm, &h, t, ne)?;
        let fg = self.arena().f32(t * g.n_ff)?;
        let fu = self.arena().f32(t * g.n_ff)?;
        self.matmul("mtp gate", &m.gate, &h, hq8.as_ref(), t, &fg)?;
        self.matmul("mtp up", &m.up, &h, hq8.as_ref(), t, &fu)?;
        let nf = t * g.n_ff;
        // SAFETY: n floats each.
        self.op("swiglu", String::new, 12 * nf, 6 * nf, || unsafe { (self.k.swiglu_clamp)(self.raw(), fg.fp(), fu.fp(), fg.fp(), nf as i64, f32::INFINITY) })?;
        self.matmul("mtp down", &m.down, &fg, None, t, &tmp)?;
        self.add(&cur, &tmp, t * ne)?;
        if !head {
            return Ok(None);
        }
        let last = cur.view((t - 1) * ne * 4, ne * 4)?;
        let hn2 = self.arena().f32(ne)?;
        let q = self.norm(&last, &m.head_norm, &hn2, 1, ne)?;
        ms.chain.copy_within(0, &hn2, 0, ne * 4)?;
        let logits = self.arena().f32(m.output.rows)?;
        self.matmul("mtp head", &m.output, &hn2, q.as_ref(), 1, &logits)?;
        let best = self.arena().bytes(4)?;
        // SAFETY: logits [1, vocab] float32 -> its argmax (one int32).
        self.op("argmax", String::new, m.output.rows * 4, 0,
                || unsafe { (self.k.argmax)(self.raw(), logits.fp(), 1, m.output.rows as i64, best.ptr().cast()) })?;
        let mut b = [0u8; 4];
        best.read(0, &mut b)?;
        Ok(Some(u32::from_le_bytes(b)))
    }

    /// Drafts for the positions after `tail` (committed at s.pos, not fed): the draft block on (the last fed
    /// position's hidden state, tail), then chained on its own normed output and each draft
    fn draft_in(&self, s: &Session, tail: u32) -> Result<Vec<u32>> {
        let Some(ms) = &s.mtp else { return Ok(Vec::new()) };
        let k = self.drafts.min(s.max_ctx.saturating_sub(s.pos + 1));
        let mut out = Vec::with_capacity(k);
        let mut tok = tail;
        // the draft block is on the last GPU
        self.on(self.last());
        for i in 0..k {
            self.arena().reset();
            self.phase("mtp draft");
            let hs = if i == 0 { &ms.pending } else { &ms.chain };
            let Some(t) = self.mtp_rows(ms, &[tok], hs, s.pos + i, s.max_ctx, true)? else { break };
            tok = t;
            out.push(tok);
        }
        Ok(out)
    }

    /// Back to the first `keep` rows of the session's last pass: its DeltaNet states from the pass's snapshots, the
    /// attention caches and the draft block's cache simply rewritten later
    fn rollback_in(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep == s.last_rows {
            return Ok(());
        }
        let mut snaps = self.snaps.lock().unwrap();
        let Some((id, p0, rows)) = snaps.of else {
            return Err(e("qwen35: a rollback needs the last pass's snapshots (a verify pass of 2 ..= drafts + 1 rows, drafts on)"));
        };
        if id != s.id || p0 + rows != s.pos || keep == 0 || keep >= rows {
            return Err(e(format!("qwen35: a rollback to {keep} of a pass of {rows} rows at {p0} (session {id}); this session is at {}", s.pos)));
        }
        let g = &self.g;
        let (sb, cb) = (g.v_heads * g.state * g.state * 4, (g.conv - 1) * g.conv_dim() * 4);
        let mut li = 0;
        for st in &s.layers {
            if let LState::Linear { s: rs, conv } = st {
                let (ss, cs) = &snaps.layers[li];
                rs.copy_within(0, ss, (keep - 1) * sb, sb)?;
                conv.copy_within(0, cs, (keep - 1) * cb, cb)?;
                li += 1;
            }
        }
        if let Some(ms) = &s.mtp {
            ms.pending.copy_within(0, &self.vh, (keep - 1) * g.n_embd * 4, g.n_embd * 4)?;
        }
        snaps.of = None;
        s.pos = p0 + keep;
        s.last_rows = keep;
        Ok(())
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        // the caches and states, checked against what is free first: past the card the xe driver spills into host
        // memory (no out-of-memory error) and can take the machine down
        let mtp_kv = if self.mtp.is_some() { 2 * g.heads_kv * max_ctx * g.head * 2 } else { 0 };
        // per GPU: its layers' share (the draft block's cache on the last)
        for (i, d) in self.devs.iter().enumerate() {
            let full = (d.lb..d.le).filter(|l| g.full(*l)).count();
            let kv = if i == self.last() { mtp_kv } else { 0 };
            let need = (full * 2 * g.heads_kv * max_ctx * g.head * 2 + kv + (d.le - d.lb - full) * (g.v_heads * g.state * g.state + (g.conv - 1) * g.conv_dim()) * 4) as u64;
            let (total, free) = d.gpu.memory()?;
            let free = free.unwrap_or(total);
            if need + (1536 << 20) > free {
                return Err(e(format!("a session of {max_ctx} tokens needs {:.2} GiB on {}; {:.2} GiB are free (1.5 GiB kept spare)", need as f64 / (1u64 << 30) as f64,
                                     d.gpu.name, free as f64 / (1u64 << 30) as f64)));
            }
        }
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let gpu = &self.devs[self.layer_dev[l]].gpu;
            layers.push(if g.full(l) {
                let n = g.heads_kv * max_ctx * g.head * 2;
                LState::Full { k: DevBuf::new(gpu, n)?, v: DevBuf::new(gpu, n)? }
            } else {
                let s = DevBuf::f32(gpu, g.v_heads * g.state * g.state)?;
                let conv = DevBuf::f32(gpu, (g.conv - 1) * g.conv_dim())?;
                s.fill(0)?;
                conv.fill(0)?;
                LState::Linear { s, conv }
            });
        }
        let mtp = match &self.mtp {
            Some(_) => {
                let n = g.heads_kv * max_ctx * g.head * 2;
                let gpu = &self.devs[self.last()].gpu;
                let ms = MtpState { k: DevBuf::new(gpu, n)?, v: DevBuf::new(gpu, n)?, pending: DevBuf::f32(gpu, g.n_embd)?, chain: DevBuf::f32(gpu, g.n_embd)? };
                ms.pending.fill(0)?;
                Some(ms)
            }
            None => None,
        };
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Session { pos: 0, max_ctx, layers, last_rows: 0, id, mtp })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        for st in &s.layers {
            if let LState::Linear { s, conv } = st {
                s.fill(0)?;
                conv.fill(0)?;
            }
        }
        if let Some(ms) = &s.mtp {
            ms.pending.fill(0)?;
        }
        s.pos = 0;
        s.last_rows = 0;
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        let g = &self.g;
        for (d, s) in dst.layers.iter().zip(&src.layers) {
            match (d, s) {
                (LState::Full { k: dk, v: dv }, LState::Full { k: sk, v: sv }) => {
                    for h in 0..g.heads_kv {
                        let n = src.pos * g.head * 2;
                        dk.copy_within(h * dst.max_ctx * g.head * 2, sk, h * src.max_ctx * g.head * 2, n)?;
                        dv.copy_within(h * dst.max_ctx * g.head * 2, sv, h * src.max_ctx * g.head * 2, n)?;
                    }
                }
                (LState::Linear { s: ds, conv: dc }, LState::Linear { s: ss, conv: sc }) => {
                    ds.copy_within(0, ss, 0, ss.len)?;
                    dc.copy_within(0, sc, 0, sc.len)?;
                }
                _ => return Err(e("sessions of other shapes")),
            }
        }
        if let (Some(d), Some(sm)) = (&dst.mtp, &src.mtp) {
            for h in 0..g.heads_kv {
                let n = src.pos * g.head * 2;
                d.k.copy_within(h * dst.max_ctx * g.head * 2, &sm.k, h * src.max_ctx * g.head * 2, n)?;
                d.v.copy_within(h * dst.max_ctx * g.head * 2, &sm.v, h * src.max_ctx * g.head * 2, n)?;
            }
            d.pending.copy_within(0, &sm.pending, 0, g.n_embd * 4)?;
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

    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        let _w = self.work.lock().unwrap();
        self.rollback_in(s, keep)
    }

    pub fn has_draft(&self) -> bool {
        self.drafts > 0
    }

    pub fn max_verify(&self) -> usize {
        self.drafts + 1
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
        let drafting = d.mtp && self.drafts > 0 && s.mtp.is_some();
        let out = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => vec![smp.sample(&l)],
            (None, Some(x)) => {
                let mut rows = vec![x];
                if drafting {
                    rows.extend(d.drafts.iter().take(s.max_ctx.saturating_sub(s.pos + 1)));
                }
                d.drafts.clear();
                let logits = self.pass_in(s, &rows, rows.len(), tap)?;
                let mut out = Vec::with_capacity(rows.len());
                for (i, l) in logits.iter().enumerate() {
                    let y = smp.sample(l);
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
                if out.len() < rows.len() {
                    self.rollback_in(s, out.len())?;
                }
                out
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        let tail = *out.last().unwrap();
        d.next = Some(tail);
        if drafting {
            d.drafts = self.draft_in(s, tail)?;
        }
        Ok(out)
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let g = &self.g;
        let mut bufs = Vec::new();
        for st in &s.layers {
            match st {
                LState::Full { k, v } => {
                    for c in [k, v] {
                        let mut b = vec![0u8; g.heads_kv * s.pos * g.head * 2];
                        for h in 0..g.heads_kv {
                            let n = s.pos * g.head * 2;
                            c.read(h * s.max_ctx * g.head * 2, &mut b[h * n..(h + 1) * n])?;
                        }
                        bufs.push(b);
                    }
                }
                LState::Linear { s: rs, conv } => {
                    for c in [rs, conv] {
                        let mut b = vec![0u8; c.len];
                        c.read(0, &mut b)?;
                        bufs.push(b);
                    }
                }
            }
        }
        if let Some(ms) = &s.mtp {
            for c in [&ms.k, &ms.v] {
                let mut b = vec![0u8; g.heads_kv * s.pos * g.head * 2];
                for h in 0..g.heads_kv {
                    let n = s.pos * g.head * 2;
                    c.read(h * s.max_ctx * g.head * 2, &mut b[h * n..(h + 1) * n])?;
                }
                bufs.push(b);
            }
            let mut b = vec![0u8; g.n_embd * 4];
            ms.pending.read(0, &mut b)?;
            bufs.push(b);
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let g = &self.g;
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        let mut it = ck.bufs.iter();
        for st in &s.layers {
            match st {
                LState::Full { k, v } => {
                    for c in [k, v] {
                        let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                        let n = ck.pos * g.head * 2;
                        if b.len() != g.heads_kv * n {
                            return Err(e("a checkpoint of another shape"));
                        }
                        for h in 0..g.heads_kv {
                            c.write(h * s.max_ctx * g.head * 2, &b[h * n..(h + 1) * n])?;
                        }
                    }
                }
                LState::Linear { s: rs, conv } => {
                    for c in [rs, conv] {
                        let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                        if b.len() != c.len {
                            return Err(e("a checkpoint of another shape"));
                        }
                        c.write(0, b)?;
                    }
                }
            }
        }
        if let Some(ms) = &s.mtp {
            for c in [&ms.k, &ms.v] {
                let b = it.next().ok_or_else(|| e("a checkpoint without the draft block's cache"))?;
                let n = ck.pos * g.head * 2;
                if b.len() != g.heads_kv * n {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..g.heads_kv {
                    c.write(h * s.max_ctx * g.head * 2, &b[h * n..(h + 1) * n])?;
                }
            }
            let b = it.next().ok_or_else(|| e("a checkpoint without the draft block's state"))?;
            ms.pending.write(0, b)?;
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

/// A tensor read onto the GPU: its blocks, or float32 for F32 / F16 / BF16
fn vec_mat(f: &Gguf, gpu: &Arc<Gpu>, k: &ffi::Api, name: &str, loaded: &mut u64) -> Result<Mat> {
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
        return Ok(Mat { buf: out, ty: GType::F32, rows, cols });
    }
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    Ok(Mat { buf, ty: t.ty, rows, cols })
}

/// The draft block from its file: block `layers` (eh_proj split into its embedding and hidden-state columns on the
/// host), its head norm and head; its token embedding stays in the file
fn load_mtp(m: Box<Gguf>, g: &Geometry, gpu: &Arc<Gpu>, k: &ffi::Api, loaded: &mut u64) -> Result<Mtp> {
    let l = g.layers;
    let f: &Gguf = &m;
    let n = |s: &str| format!("blk.{l}.{s}.weight");
    let mut read = |name: &str| -> Result<Mat> { vec_mat(f, gpu, k, name, loaded) };
    let attn = Full { q: read(&n("attn_q"))?, k: read(&n("attn_k"))?, v: read(&n("attn_v"))?, o: read(&n("attn_output"))?,
                      q_norm: read(&n("attn_q_norm"))?.buf, k_norm: read(&n("attn_k_norm"))?.buf };
    let (enorm, hnorm, head_norm) = (read(&n("nextn.enorm"))?.buf, read(&n("nextn.hnorm"))?.buf, read(&n("nextn.shared_head_norm"))?.buf);
    let (attn_norm, post_norm) = (read(&n("attn_norm"))?.buf, read(&n("post_attention_norm"))?.buf);
    let (gate, up, down) = (read(&n("ffn_gate"))?, read(&n("ffn_up"))?, read(&n("ffn_down"))?);
    let output = read(if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" })?;
    // eh_proj [n_embd, 2 n_embd]: each row's first half of blocks multiplies the embedding, the second the hidden state
    let t = f.tensor(&n("nextn.eh_proj")).ok_or_else(|| e("the MTP file has no nextn.eh_proj"))?;
    let (rows, cols) = (t.shape[0] as usize, t.shape[1..].iter().product::<u64>() as usize);
    if cols != 2 * g.n_embd || matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
        return Err(e("nextn.eh_proj: a quantized [n_embd, 2 n_embd] matrix"));
    }
    let raw = f.read(t).map_err(|x| e(x.0))?;
    *loaded += raw.len() as u64;
    let (rb, hb) = (t.ty.bytes(cols as u64).unwrap_or(0) as usize, t.ty.bytes(g.n_embd as u64).unwrap_or(0) as usize);
    let (mut ea, mut ha) = (Vec::with_capacity(rows * hb), Vec::with_capacity(rows * hb));
    for r in 0..rows {
        ea.extend_from_slice(&raw[r * rb..r * rb + hb]);
        ha.extend_from_slice(&raw[r * rb + hb..(r + 1) * rb]);
    }
    let (be, bh) = (DevBuf::new(gpu, ea.len())?, DevBuf::new(gpu, ha.len())?);
    be.write(0, &ea)?;
    bh.write(0, &ha)?;
    let embd = f.tensor("token_embd.weight").ok_or_else(|| e("the MTP file has no token_embd.weight"))?.clone();
    let vocab = embd.shape[0] as usize;
    Ok(Mtp { eh_e: Mat { buf: be, ty: t.ty, rows, cols: g.n_embd }, eh_h: Mat { buf: bh, ty: t.ty, rows, cols: g.n_embd }, enorm, hnorm, head_norm,
             attn_norm, post_norm, attn, gate, up, down, output, embd, vocab, file: m })
}
