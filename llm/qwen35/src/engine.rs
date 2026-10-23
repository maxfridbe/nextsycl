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

/// The attention cache's form a GPU (NS_Q35_KV): half, q8_0 or q4_0 (llama.cpp's blocks of 32, the scales behind a
/// row's values: kernels/llm/qwen35/silo/attn.cpp)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KvTy {
    F16,
    Q8,
    Q4,
}

impl KvTy {
    /// the kernels' code
    fn code(self) -> c_int {
        match self {
            KvTy::F16 => 0,
            KvTy::Q8 => 1,
            KvTy::Q4 => 2,
        }
    }
    /// bytes a head's row of `head` values
    fn row(self, head: usize) -> usize {
        match self {
            KvTy::F16 => head * 2,
            KvTy::Q8 => head + head / 32 * 2,
            KvTy::Q4 => head / 2 + head / 32 * 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KvTy::F16 => "f16",
            KvTy::Q8 => "q8_0",
            KvTy::Q4 => "q4_0",
        }
    }
}

/// NS_Q35_KV: the forms (the GPU the plan fills first - the last, which holds the head; the others). f16 (the default:
/// the fastest - B70, Q4, 43K prompt: read 742 / 673 / 718 tok/s and decode 43.0 / 38.9 / 40.9 with half / q8_0 /
/// q4_0); auto: q8_0 there, half on the others; auto-q4: q4_0 there; q8 (q8_0), q4 (q4_0): that on every GPU. One B70
/// session's most context with MTP: half 196K, q8_0 368K, q4_0 704K
fn kv_mode() -> std::result::Result<(KvTy, KvTy), String> {
    match std::env::var("NS_Q35_KV").as_deref() {
        Ok("auto") => Ok((KvTy::Q8, KvTy::F16)),
        Err(_) | Ok("") => Ok((KvTy::F16, KvTy::F16)),
        Ok("auto-q4") => Ok((KvTy::Q4, KvTy::F16)),
        Ok("f16") => Ok((KvTy::F16, KvTy::F16)),
        Ok("q8") | Ok("q8_0") => Ok((KvTy::Q8, KvTy::Q8)),
        Ok("q4") | Ok("q4_0") => Ok((KvTy::Q4, KvTy::Q4)),
        Ok(v) => Err(format!("NS_Q35_KV={v}: auto, auto-q4, f16, q8_0 or q4_0")),
    }
}

/// NS_Q35_KV_ROT=1: a quantized cache's keys, values and queries rotated (a 32-value Hadamard transform a block, the
/// attention's output turned back): the quantization error spread over the block
fn kv_rot() -> bool {
    std::env::var("NS_Q35_KV_ROT").is_ok_and(|v| v == "1")
}

/// A cache buffer's bytes a head's row ([kv heads][cap] rows)
fn kv_rb(b: &DevBuf, heads_kv: usize, cap: usize) -> usize {
    b.len / (heads_kv * cap).max(1)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
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
    /// the rotary pairs' sections by position (t, row, column: Qwen3-VL's interleaved M-RoPE; text turns all by one)
    pub sections: (i64, i64, i64),
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
            sections: {
                let v: Vec<i64> = f.arch_meta("rope.dimension_sections").and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_i64().or_else(|| x.as_u64().map(|u| u as i64))).collect()).unwrap_or_default();
                if v.len() >= 3 && v[0] + v[1] + v[2] > 0 { (v[0], v[1], v[2]) } else { (11, 11, 10) }
            },
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
    /// how far the text's rotary positions are behind `pos`: what the pictures read so far took in positions beyond
    /// their advance (Qwen3-VL's positions; 0 without pictures)
    pub shift: usize,
    layers: Vec<LState>,
    /// rows of the last forward pass
    last_rows: usize,
    /// which session (the engine's snapshots name the one they belong to)
    id: u64,
    /// the draft block's cache and inputs (MTP loaded)
    mtp: Option<MtpState>,
}

/// A session's draft block state: its cache ([heads][cap][head] half: MTP's the target's kv heads and max_ctx, EAGLE's
/// its 40 heads of 128 and its own positions), the normed hidden state of the last position fed (EAGLE: its feature;
/// the next token pairs with it), and the block's own normed output (a chained draft pairs with it)
struct MtpState {
    k: DevBuf,
    v: DevBuf,
    pending: DevBuf,
    chain: DevBuf,
    heads: usize,
    cap: usize,
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
    /// per DeltaNet layer: the recurrence's and the conv's states after a verify pass's rows but the last
    layers: Vec<(DevBuf, DevBuf)>,
    /// a second set for a batch step's second conversation (made with the first such step, when the cards have room)
    second: Option<Vec<(DevBuf, DevBuf)>>,
    /// the last pass's verify rows: (session, position, rows, set)
    of: Vec<(u64, usize, usize, usize)>,
}

impl Snaps {
    fn set(&self, i: usize) -> Option<&Vec<(DevBuf, DevBuf)>> {
        if i == 0 { Some(&self.layers) } else { self.second.as_ref() }
    }
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
    /// the EAGLE-4 draft (eagle.rs), in MTP's place when its folder is found (one GPU)
    eagle: Option<crate::eagle::Eagle>,
    /// drafts a step
    drafts: usize,
    /// the verify passes' DeltaNet snapshots (under `work`)
    snaps: Mutex<Snaps>,
    /// the normed hidden states of the last pass's rows (up to 8: a rollback's pending state)
    vh: DevBuf,
    next_id: std::sync::atomic::AtomicU64,
    /// the vision tower (the model's mmproj file: NS_Q35_VISION), when there is one
    pub vision: Option<crate::vision::Vision>,
    /// LogProbChain's attention capture (None: off)
    capture: Mutex<Option<Capture>>,
    /// the attention cache's form on each GPU (devs' order), and whether a quantized one is rotated
    pub kv_tys: Vec<KvTy>,
    kv_rot: bool,
    /// decode passes as SYCL graphs (under `work`)
    graphs: Mutex<Graphs>,
    /// inside a replayed stretch: kernel calls are not made (the graph makes them)
    skip: std::sync::atomic::AtomicBool,
    /// graphs failed to record here: off for the process
    graphs_off: std::sync::atomic::AtomicBool,
    /// NS_Q35_PROFILE=1: every kernel timed on the device, with its bytes and arithmetic (probe.rs)
    probe: Option<Probe>,
    load_s: f64,
    load_bytes: u64,
}

/// Where a pass's rows turn (the full-attention layers' rotary positions): one after another from a position (text),
/// or each at its own (t, row, column) - a [rows][3] int32 buffer on the layer's GPU (a pass with a picture's rows)
#[derive(Clone, Copy)]
enum Rope<'a> {
    From(usize),
    Rows(&'a DevBuf),
}

/// A conversation's rows in a pass, as a full-attention layer reads them: from row r0, t rows at position p0, their
/// rotary positions, its cache ([kv heads][cap][head] half)
struct FullSeg<'a> {
    r0: usize,
    t: usize,
    p0: usize,
    rope: Rope<'a>,
    cap: usize,
    k: &'a DevBuf,
    v: &'a DevBuf,
    /// LogProbChain: this one-row pass's attention captured (`Qwen35::capture_attention`)
    probe: bool,
}

impl FullSeg<'_> {
    /// the cache's form, by its buffer's size
    fn ty(&self, g: &Geometry) -> KvTy {
        let rb = kv_rb(self.k, g.heads_kv, self.cap);
        [KvTy::F16, KvTy::Q8, KvTy::Q4].into_iter().find(|t| t.row(g.head) == rb).unwrap_or(KvTy::F16)
    }
}

/// A recorded SYCL graph (ns_q35_graph_*), freed with it
struct GraphH(*mut std::ffi::c_void);

// SAFETY: a handle the kernel library made; used under the engine's work lock, freed once.
unsafe impl Send for GraphH {}

impl Drop for GraphH {
    fn drop(&mut self) {
        if let Ok(k) = ffi::api() {
            // SAFETY: made by ns_q35_graph_begin, freed once here.
            unsafe { (k.graph_free)(self.0) };
        }
    }
}

/// A decode pass as graphs (NS_Q35_GRAPHS): its position-free stretches - everything but the full-attention layers'
/// rotary positions, cache writes and attention, which take the position as an argument - recorded the first time a
/// session runs a pass of that many rows, replayed after (the same buffers: the arena hands them out in the same order).
/// `map`: the stretches' graphs by (session, rows, snapshots kept), each with its GPU
#[derive(Default)]
struct Graphs {
    map: std::collections::HashMap<(u64, usize, bool), Vec<(usize, GraphH)>>,
    mode: GMode,
}

#[derive(Default)]
enum GMode {
    #[default]
    Off,
    /// recording: the stretches so far, the open one
    Record { key: (u64, usize, bool), segs: Vec<(usize, GraphH)>, open: Option<(usize, GraphH)> },
    /// replaying the stretches of `key`, the next one `idx`
    Replay { key: (u64, usize, bool), idx: usize },
}

/// NS_Q35_GRAPHS=1: decode passes as graphs. Off by default: the same tokens, no faster (2026-10-19, Q4, 256 tokens:
/// B70 47.47 tok/s with them and without (MTP), 27.76 / 27.43 one row a pass; B65 + B70 42.8 / 44.4, 23.6 / 23.9) -
/// the host stays ahead of the GPU, whose kernels run back to back already
fn graphs_on() -> bool {
    std::env::var("NS_Q35_GRAPHS").is_ok_and(|v| v == "1")
}

/// LogProbChain's capture: per GPU the attention weights added per position (the full-attention layers of one-token
/// passes), how many layers added, the heads' mean (0) or largest (1)
struct Capture {
    mode: i32,
    bufs: Vec<Option<DevBuf>>,
    layers: usize,
    n: usize,
}

/// A conversation's rows in a pass, as a DeltaNet layer reads them: from row r0, t rows, its state and conv inputs, and
/// where the states after its rows but the last go (the recurrence's, the conv's: a verify pass's snapshots)
struct LinSeg<'a> {
    r0: usize,
    t: usize,
    s: &'a DevBuf,
    conv: &'a DevBuf,
    snap: Option<(*mut f32, *mut f32)>,
}

/// The vision tower's work memory: what a layer's weights and the blocks take, and what each patch adds (its rows of
/// 1152 through the block: x, the normed rows, q / k / v, the turned keys, the attention's output, their half copies)
const VIT_FIXED: usize = 192 << 20;
const VIT_PER_PATCH: usize = 1152 * 4 * 8;

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

/// The EAGLE-4 draft's rows a step behind a prompt pass (its work memory bounded by them), and the largest SwiGLU the
/// work memory is planned for
const EAGLE_ROWS: usize = 256;
const EAGLE_FF: usize = 16384;

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
        // the attention cache's forms (the GPU filled first, the others): a quantized cache is read by the silo's attention
        let (mut kv_fill, mut kv_other) = kv_mode().map_err(e)?;
        if silo.is_none() && (kv_fill, kv_other) != (KvTy::F16, KvTy::F16) {
            log("qwen35: a quantized attention cache needs the silo: half".into());
            (kv_fill, kv_other) = (KvTy::F16, KvTy::F16);
        }
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
        // the EAGLE-4 draft (one GPU): in MTP's place
        let eagle_dir = match (drafts > 0).then(|| f.paths.first().and_then(|p| crate::eagle::path(p))).flatten() {
            Some(d) if gpus.len() > 1 => {
                log(format!("qwen35: the EAGLE-4 draft {} runs on one GPU: MTP instead", d.display()));
                None
            }
            Some(d) => match crate::eagle::weight_bytes(&d) {
                Ok(b) => Some((d, b)),
                Err(x) => {
                    log(format!("qwen35: the EAGLE-4 draft {} is not read ({}): MTP instead", d.display(), x.0));
                    None
                }
            },
            None => None,
        };
        let mfile = match (drafts > 0 && eagle_dir.is_none()).then(|| mtp_path(f)).flatten() {
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
        let mtp_w = mfile.as_ref().map_or(0, |m| m.tensors.iter().filter(|t| t.name != "token_embd.weight").map(|t| t.bytes).sum::<u64>())
            + eagle_dir.as_ref().map_or(0, |d| d.1);
        let mtp_bytes = mtp_w + if mfile.is_some() || eagle_dir.is_some() { snap_bytes as u64 } else { 0 };
        let margin: u64 = 1536 << 20;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        // the draft block's pass behind a chunk's: its inputs and outputs beside the main pass's
        let has_mtp = mfile.is_some();
        // EAGLE: the pass's three aux states, its rows' work (EAGLE_ROWS at a time)
        let has_eagle = eagle_dir.is_some();
        let arena_of = |c: usize| arena_bytes(&g, c) + if has_mtp { c * 8 * g.n_embd * 4 } else { 0 }
            + if has_eagle { (3 * c * g.n_embd + EAGLE_ROWS.min(c) * (18 * g.n_embd + 2 * EAGLE_FF)) * 4 + (64 << 20) } else { 0 };
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
            // a layer's cost with the cache's form of the GPU it lands on: the last GPU's (filled first), the others'
            let cost_of = |ty: KvTy| -> Vec<u64> {
                (0..g.layers).map(|l| layer_w[l] + if g.full(l) { (2 * g.heads_kv * ctx * ty.row(g.head)) as u64 } else {
                    (lin_state * (1 + if mtp_on { drafts } else { 0 })) as u64
                }).collect()
            };
            let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            let plan = |c: usize, other: KvTy| -> std::result::Result<Vec<(usize, usize)>, String> {
                let n = gpus.len();
                let (cost_last, cost_other) = (cost_of(kv_fill), cost_of(other));
                let cost_at = |i: usize| if i + 1 == n { &cost_last } else { &cost_other };
                // what each GPU has for layers: its free memory less the spare, its half buffer and work memory; the
                // last's less the head and the draft block (its weights, its cache)
                let room: Vec<i64> = (0..n).map(|i| {
                    // (and 512 MiB the plan does not see: the allocator's rounding, small buffers)
                    let mut over = margin + (512 << 20) + w16_bytes as u64 + arena_of(c) as u64;
                    if i + 1 == n {
                        over += head_w + if mtp_on { mtp_w + (2 * g.heads_kv * ctx * kv_fill.row(g.head)) as u64 } else { 0 };
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
                            while lb > 0 && cost_at(i)[lb - 1] as i64 <= have {
                                have -= cost_at(i)[lb - 1] as i64;
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
                    let need: u64 = cost_at(i)[lb..le].iter().sum();
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
            // the other GPUs' form, else the first's where theirs does not fit
            'plans: for other in [kv_other, kv_fill] {
                for &c in &tries {
                    match plan(c, other) {
                        Ok(b) => {
                            found = Some((c, b));
                            kv_other = other;
                            break 'plans;
                        }
                        Err(x) => last_err = x,
                    }
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
        let kv_tys: Vec<KvTy> = (0..gpus.len()).map(|i| if i + 1 == gpus.len() { kv_fill } else { kv_other }).collect();
        log(format!("qwen35: the attention cache {}{}", kv_tys.iter().map(|t| t.name()).collect::<Vec<_>>().join(" + "),
                    if kv_rot() && kv_tys.iter().any(|t| *t != KvTy::F16) { ", rotated" } else { "" }));
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
        let eagle = match eagle_dir {
            Some((d, _)) => match crate::eagle::Eagle::load(&d, &lastg, g.n_embd, g.layers, g.vocab) {
                Ok(x) if (x.eps - g.eps).abs() <= 1e-9 && x.n_ff <= EAGLE_FF => {
                    loaded += x.bytes;
                    Some(x)
                }
                Ok(x) => return Err(e(format!("the EAGLE-4 draft's norm eps {} (the model's {}) or its SwiGLU of {}", x.eps, g.eps, x.n_ff))),
                Err(x) => return Err(e(format!("the EAGLE-4 draft {}: {}", d.display(), x.0))),
            },
            None => None,
        };
        let drafts = if mtp.is_some() || eagle.is_some() { drafts } else { 0 };
        let mut snap_layers = Vec::new();
        if drafts > 0 {
            for l in (0..g.layers).filter(|l| !g.full(*l)) {
                let gpu = &gpus[layer_dev[l]];
                snap_layers.push((DevBuf::f32(gpu, drafts * g.v_heads * g.state * g.state)?, DevBuf::f32(gpu, drafts * (g.conv - 1) * g.conv_dim())?));
            }
        }
        log(match (&mtp, &eagle) {
            (_, Some(x)) => format!("qwen35: EAGLE-4 drafts ({drafts} a step; aux layers {:?}, window {}, its cache {} positions, {:.2} GiB)", x.ids,
                                    x.window, x.ctx, gib(x.bytes)),
            (Some(_), _) => format!("qwen35: MTP drafts ({drafts} a step)"),
            (None, _) if draft => "qwen35: no MTP file (NS_Q35_MTP or an MTP/ folder beside the model): no drafts".into(),
            (None, _) => "qwen35: drafts off".into(),
        });
        // two sets of 8 rows: a batch step's two conversations
        let vh = DevBuf::f32(&lastg, 16 * g.n_embd)?;
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
        // the vision tower: its file's settings and learned positions here, its weights read a layer at a time when a
        // picture comes (into the first GPU's prompt arena, which bounds a picture's patches)
        let vision = match crate::vision::sidecar(&f.paths[0]) {
            Some(path) => {
                let room = devs[0].arena.len().saturating_sub(VIT_FIXED) / VIT_PER_PATCH / 4;
                let most = crate::vision::max_tokens().min(room);
                match crate::vision::Vision::open(&path, f, g.n_embd, most) {
                    Ok(v) => {
                        log(format!("qwen35: the vision tower {} ({} blocks of {}; pictures of up to {most} positions)", path.display(), v.cfg.layers, v.cfg.dim));
                        Some(v)
                    }
                    Err(x) => {
                        log(format!("qwen35: the vision file {} is not read ({}): no pictures", path.display(), x.0));
                        None
                    }
                }
            }
            None => None,
        };
        Ok(Qwen35 { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(Vec::new()), k, silo, layers, output_norm, output, embd, chunk,
                    work: Mutex::new(()), mtp, eagle, drafts, snaps: Mutex::new(Snaps { layers: snap_layers, second: None, of: Vec::new() }), vh, next_id: 0.into(), vision, capture: Mutex::new(None), kv_rot: kv_rot() && kv_tys.iter().any(|t| *t != KvTy::F16), kv_tys, graphs: Mutex::new(Graphs::default()), skip: false.into(), graphs_off: false.into(),
                    probe,
                    load_s, load_bytes: loaded })
    }

    /// A kernel call (its return code checked): timed on the device with NS_Q35_PROFILE=1, with the bytes it moves
    /// and the arithmetic it does
    fn op(&self, name: &'static str, key: impl FnOnce() -> String, bytes: usize, flops: usize, f: impl FnOnce() -> c_int) -> Result<()> {
        let rc = self.op_rc(name, key, bytes, flops, f)?;
        ffi::check(rc, name)
    }

    /// `op`, its return code left to the caller
    fn op_rc(&self, name: &'static str, key: impl FnOnce() -> String, bytes: usize, flops: usize, f: impl FnOnce() -> c_int) -> Result<c_int> {
        if self.skip.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(0);
        }
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
            // a picture's position: the span's start token's row (the picture's own row is written over it: put_images)
            let t = if crate::vision::is_image(*t) { self.vision.as_ref().map_or(*t, |v| v.start) } else { *t };
            if t as usize >= vocab {
                return Err(e(format!("token {t} past the vocabulary ({vocab})")));
            }
            file.read_into(embd, t as u64 * row as u64, &mut raw[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
        }
        let src = self.arena().bytes(raw.len())?;
        src.write(0, &raw)?;
        let x = self.arena().f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32.
        self.op("dequant", String::new, raw.len() + 4 * tokens.len() * g.n_embd, 0,
                || unsafe { (self.k.dequant)(self.raw(), embd.ty.code() as i32, src.ptr(), tokens.len() * g.n_embd, x.fp()) })?;
        Ok(x)
    }

    /// The rows of every picture among `tokens` made (`image_rows`), on the first GPU
    fn image_rows_for(&self, tokens: &[u32]) -> Result<()> {
        let vis = self.vision.as_ref().ok_or_else(|| e("a picture's span, and no vision tower (NS_Q35_VISION)"))?;
        let mut seen = Vec::new();
        for &t in tokens.iter().filter(|t| crate::vision::is_image(**t)) {
            let key = crate::vision::key_of(t);
            if !seen.contains(&key) {
                seen.push(key);
                self.image_rows(&*vis.image(t)?)?;
            }
        }
        Ok(())
    }

    /// The pictures' rows written over x's ([t][n_embd]) at their positions among `tokens` (made already)
    fn put_images(&self, tokens: &[u32], x: &DevBuf) -> Result<()> {
        use crate::vision::{is_image, key_of, pos_of};
        let vis = self.vision.as_ref().ok_or_else(|| e("a picture's span, and no vision tower (NS_Q35_VISION)"))?;
        let ne = self.g.n_embd;
        let mut i = 0;
        while i < tokens.len() {
            if !is_image(tokens[i]) {
                i += 1;
                continue;
            }
            // a run of one picture's positions, one after another in its span
            let (key, k0) = (key_of(tokens[i]), pos_of(tokens[i]));
            let mut j = i + 1;
            while j < tokens.len() && is_image(tokens[j]) && key_of(tokens[j]) == key && pos_of(tokens[j]) == k0 + (j - i) {
                j += 1;
            }
            let img = vis.image(tokens[i])?;
            let rows = img.rows.lock().unwrap().clone().ok_or_else(|| e("a picture's rows not made"))?;
            let (a, b) = (k0 * ne, (k0 + j - i) * ne);
            if b > rows.len() {
                return Err(e(format!("an image position {} past its span of {}", k0 + j - i - 1, rows.len() / ne)));
            }
            x.view(i * ne * 4, (j - i) * ne * 4)?.write(0, &f32_bytes(&rows[a..b]))?;
            i = j;
        }
        Ok(())
    }

    /// A picture's input rows [span][n_embd]: the vision tower over its patches (the mmproj file's weights a layer at a
    /// time, half, into the first GPU's prompt arena; float32 rows), kept with the picture once made
    pub fn image_rows(&self, img: &crate::vision::Img) -> Result<Arc<Vec<f32>>> {
        if let Some(r) = img.rows.lock().unwrap().as_ref() {
            return Ok(r.clone());
        }
        let vis = self.vision.as_ref().ok_or_else(|| e("a picture's span, and no vision tower (NS_Q35_VISION)"))?;
        let px = img.patches.lock().unwrap().clone().ok_or_else(|| e("a picture without its patches"))?;
        let t0 = Instant::now();
        let (c, pl, k) = (&vis.cfg, img.plan, self.k);
        let (n, dim, nh, ff) = (pl.n(), c.dim, c.heads, c.ff);
        let hd = dim / nh;
        let kp = 3 * c.patch * c.patch;
        self.on(0);
        let a = self.arena();
        let mk = a.mark();
        if a.len() < mk + VIT_FIXED + n * VIT_PER_PATCH {
            return Err(e(format!("a picture of {n} patches past the prompt arena ({} MiB): NS_Q35_IMAGE_TOKENS", a.len() >> 20)));
        }
        let none = String::new;
        let half = |name: &str, len: usize| -> Result<DevBuf> {
            let b = crate::vision::halves(&vis.file, name)?;
            if b.len() != 2 * len {
                return Err(e(format!("{name}: {} values, {len} expected", b.len() / 2)));
            }
            let d = a.bytes(b.len())?;
            d.write(0, &b)?;
            Ok(d)
        };
        let float = |name: &str, len: usize| -> Result<DevBuf> {
            let v = crate::vision::floats(&vis.file, name)?;
            if v.len() != len {
                return Err(e(format!("{name}: {} values, {len} expected", v.len())));
            }
            let d = a.f32(len)?;
            d.write(0, &f32_bytes(&v))?;
            Ok(d)
        };
        // y [rows, nn] = x [rows, kk] . w^T (+ bias, then act): x to half, the half product
        let lin = |x: &DevBuf, rows: usize, kk: usize, w: &DevBuf, nn: usize, b: Option<&DevBuf>, act: i32, y: &DevBuf| -> Result<()> {
            let m = a.mark();
            let xh = a.bytes(rows * kk * 2)?;
            // SAFETY: x [rows, kk] float32 -> half; y [rows, nn] = xh . w^T (w [nn, kk] half); then y's bias and act.
            unsafe {
                self.op("vit to half", none, 6 * rows * kk, 0, || (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (rows * kk) as i64))?;
                self.op("vit gemm", none, 2 * rows * kk + 2 * nn * kk + 4 * rows * nn, 2 * rows * nn * kk,
                        || (k.gemm_f16)(self.raw(), rows as i64, nn as i64, kk as i64, xh.ptr().cast(), kk as i64, w.ptr().cast(), y.fp(), nn as i64, 0))?;
                self.op("vit bias", none, 8 * rows * nn, 8 * rows * nn,
                        || (k.vit_bias_act)(self.raw(), y.fp(), b.map_or(std::ptr::null(), |b| b.fp()), rows as i64, nn as i64, act))?;
            }
            a.rewind(m);
            Ok(())
        };
        let ln = |x: &DevBuf, w: &DevBuf, b: &DevBuf, y: &DevBuf, rows: usize, cc: usize| -> Result<()> {
            // SAFETY: rows of cc floats each.
            self.op("vit norm", none, 8 * rows * cc, 8 * rows * cc, || unsafe { (k.vit_ln)(self.raw(), x.fp(), w.fp(), b.fp(), y.fp(), rows as i64, cc as i64, c.eps) })
        };
        let add = |y: &DevBuf, x: &DevBuf, len: usize| -> Result<()> { self.add(y, x, len) };
        // the patches: one product with both frames' kernels summed (the same picture in both), the bias, the learned
        // positions
        let x = a.f32(n * dim)?;
        {
            let m = a.mark();
            let mut w = crate::vision::floats(&vis.file, "v.patch_embd.weight")?;
            if vis.file.tensor("v.patch_embd.weight.1").is_some() {
                for (p, q) in w.iter_mut().zip(crate::vision::floats(&vis.file, "v.patch_embd.weight.1")?) {
                    *p += q;
                }
            }
            if w.len() != dim * kp {
                return Err(e(format!("v.patch_embd.weight: {} values, {} expected", w.len(), dim * kp)));
            }
            let wb: Vec<u8> = w.iter().flat_map(|v| crate::vision::f32_to_f16(*v).to_le_bytes()).collect();
            let pw = a.bytes(wb.len())?;
            pw.write(0, &wb)?;
            let xin = a.f32(n * kp)?;
            xin.write(0, &f32_bytes(&px))?;
            let pb = float("v.patch_embd.bias", dim)?;
            lin(&xin, n, kp, &pw, dim, Some(&pb), 0, &x)?;
            let pe = a.f32(n * dim)?;
            pe.write(0, &f32_bytes(&crate::vision::positions(&vis.table, &pl, c)))?;
            add(&x, &pe, n * dim)?;
            a.rewind(m);
        }
        let cs = a.f32(n * hd)?;
        cs.write(0, &f32_bytes(&crate::vision::rotary(&pl, c)))?;
        let h = a.f32(n * dim)?;
        let qkv = a.f32(n * 3 * dim)?;
        let kh = a.f32(n * dim)?;
        let att = a.f32(n * dim)?;
        let lm = a.mark();
        let wbytes = 2 * (3 * dim * dim + dim * dim + 2 * dim * ff) + 4 * (3 * dim + 2 * dim + ff + 4 * dim);
        // what is left for the blocks of rows: the attention's scores (heads x rows x n) and the MLP's (rows x ff, and
        // its half copy)
        let left = a.len().saturating_sub(lm + wbytes + (8 << 20));
        let rb = (left / (4 * nh * n + 4 * dim)).clamp(1, n);
        let mb = (left / (6 * ff + 2 * dim)).clamp(1, n);
        let scale = 1.0 / (hd as f32).sqrt();
        for l in 0..c.layers {
            a.rewind(lm);
            let p = |s: &str| format!("v.blk.{l}.{s}");
            let (n1w, n1b) = (float(&p("ln1.weight"), dim)?, float(&p("ln1.bias"), dim)?);
            let (n2w, n2b) = (float(&p("ln2.weight"), dim)?, float(&p("ln2.bias"), dim)?);
            let (wqkv, bqkv) = (half(&p("attn_qkv.weight"), 3 * dim * dim)?, float(&p("attn_qkv.bias"), 3 * dim)?);
            let (wo, bo) = (half(&p("attn_out.weight"), dim * dim)?, float(&p("attn_out.bias"), dim)?);
            let (wu, bu) = (half(&p("ffn_up.weight"), ff * dim)?, float(&p("ffn_up.bias"), ff)?);
            let (wd, bd) = (half(&p("ffn_down.weight"), dim * ff)?, float(&p("ffn_down.bias"), dim)?);
            ln(&x, &n1w, &n1b, &h, n, dim)?;
            lin(&h, n, dim, &wqkv, 3 * dim, Some(&bqkv), 0, &qkv)?;
            // SAFETY: qkv [n][3 dim], cs [n][hd / 2][2], kh [nh][n][hd].
            self.op("vit rope", none, 4 * n * 4 * dim, 6 * n * dim, || unsafe { (k.vit_rope)(self.raw(), qkv.fp(), n as i64, nh as i64, hd as i64, cs.fp(), kh.fp()) })?;
            let bm = a.mark();
            let sc = a.f32(nh * rb * n)?;
            let ob = a.f32(rb * dim)?;
            for r0 in (0..n).step_by(rb) {
                let r = rb.min(n - r0);
                // per head: s = q k^T (q a head's columns of qkv's rows, k its [n][hd]), softmax, then s v (v in qkv's
                // rows) into the head's columns of att
                // SAFETY: the batch products over the views named; the scores [nh][r][n].
                unsafe {
                    self.op("vit scores", none, 4 * (nh * r * hd + nh * n * hd + nh * r * n), 2 * nh * r * n * hd,
                            || (k.gemm_batch)(self.raw(), nh as i64, r as i64, n as i64, hd as i64, qkv.fp().add(r0 * 3 * dim), (3 * dim) as i64, hd as i64,
                                              kh.fp(), (n * hd) as i64, sc.fp(), n as i64, (r * n) as i64, 0))?;
                    self.op("vit softmax", none, 8 * nh * r * n, 4 * nh * r * n, || (k.softmax_rows)(self.raw(), sc.fp(), (nh * r) as i64, n as i64, scale))?;
                    self.op("vit values", none, 4 * (nh * r * n + nh * n * hd + nh * r * hd), 2 * nh * r * n * hd,
                            || (k.gemm_batch_nn)(self.raw(), nh as i64, r as i64, hd as i64, n as i64, sc.fp(), n as i64, (r * n) as i64,
                                                 qkv.fp().add(2 * dim), (3 * dim) as i64, hd as i64, ob.fp(), hd as i64, (r * hd) as i64, 0))?;
                    // (oneMKL takes no interleaved outputs: each head's [r][hd] apart, then into att's rows)
                    self.op("vit heads", none, 8 * r * dim, 0, || (k.vit_heads)(self.raw(), ob.fp(), nh as i64, r as i64, hd as i64, att.fp().add(r0 * dim)))?;
                }
            }
            a.rewind(bm);
            lin(&att, n, dim, &wo, dim, Some(&bo), 0, &h)?;
            add(&x, &h, n * dim)?;
            ln(&x, &n2w, &n2b, &h, n, dim)?;
            let mid = a.f32(mb * ff)?;
            for r0 in (0..n).step_by(mb) {
                let r = mb.min(n - r0);
                lin(&h.view(r0 * dim * 4, r * dim * 4)?, r, dim, &wu, ff, Some(&bu), 1, &mid)?;
                lin(&mid, r, ff, &wd, dim, Some(&bd), 0, &att.view(r0 * dim * 4, r * dim * 4)?)?;
            }
            add(&x, &att, n * dim)?;
        }
        a.rewind(lm);
        // the merger: the last norm, a unit's 4 patches (4 rows in a row) as one row, two products with GELU between
        let (pw, pb) = (float("v.post_ln.weight", dim)?, float("v.post_ln.bias", dim)?);
        ln(&x, &pw, &pb, &h, n, dim)?;
        let (u, ku, out) = (n / 4, 4 * dim, c.out);
        let (w0, b0) = (half("mm.0.weight", ku * ku)?, float("mm.0.bias", ku)?);
        // in q / k / v's memory (n x 3 dim: the unit rows n x dim, then the rows n / 4 x out)
        let mid = qkv.view(0, u * ku * 4)?;
        lin(&h, u, ku, &w0, ku, Some(&b0), 2, &mid)?;
        let (w2, b2) = (half("mm.2.weight", out * ku)?, float("mm.2.bias", out)?);
        let rows = qkv.view(u * ku * 4, u * out * 4)?;
        lin(&mid, u, ku, &w2, out, Some(&b2), 0, &rows)?;
        let rows = Arc::new(rows.to_f32()?);
        a.rewind(mk);
        // NS_Q35_VISION_DUMP=PREFIX: the patches and the rows as raw float32 (a check against the reference's)
        if let Ok(pre) = std::env::var("NS_Q35_VISION_DUMP") {
            for (name, v) in [("patches", &px[..]), ("rows", &rows[..])] {
                std::fs::write(format!("{pre}.{name}.f32"), f32_bytes(v)).map_err(|x| e(format!("NS_Q35_VISION_DUMP: {x}")))?;
            }
            eprintln!("[the picture's patches [{n}][{kp}] and rows [{u}][{out}] in {pre}.*.f32 (patch grid {} x {}, {} x {} pixels)]", pl.gh, pl.gw, pl.w, pl.h);
        }
        eprintln!("[a picture of {} x {} patches -> {} x {} positions in {:.2} s]", pl.gh, pl.gw, pl.lh, pl.lw, t0.elapsed().as_secs_f64());
        *img.rows.lock().unwrap() = Some(rows.clone());
        *img.patches.lock().unwrap() = None;
        Ok(rows)
    }

    /// A decode pass's graphs begin (`key`: its session, rows, snapshots): its stretches replayed when recorded before,
    /// else recorded now (the first stretch opened on the current GPU)
    fn graph_start(&self, key: (u64, usize, bool)) -> Result<()> {
        let mut gr = self.graphs.lock().unwrap();
        if gr.map.contains_key(&key) {
            gr.mode = GMode::Replay { key, idx: 0 };
            self.skip.store(true, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if gr.map.len() >= 64 {
            gr.map.clear();
        }
        match self.graph_open() {
            Ok(open) => gr.mode = GMode::Record { key, segs: Vec::new(), open: Some(open) },
            Err(x) => {
                // no graphs on this runtime: the passes run as they are
                eprintln!("[qwen35: SYCL graphs unavailable ({}): decode runs kernel by kernel]", x.0);
                self.graphs_off.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    fn graph_open(&self) -> Result<(usize, GraphH)> {
        let mut h = std::ptr::null_mut();
        // SAFETY: the current GPU's queue records from now; h receives the graph.
        ffi::check(unsafe { (self.k.graph_begin)(self.raw(), &mut h) }, "graph begin")?;
        Ok((self.cur.load(std::sync::atomic::Ordering::Relaxed), GraphH(h)))
    }

    fn graphing(&self) -> bool {
        !matches!(self.graphs.lock().unwrap().mode, GMode::Off)
    }

    /// A stretch ends (a position-taking kernel, a GPU change or the pass's end next): recorded and run, or replayed
    fn seg_end(&self) -> Result<()> {
        let mut gr = self.graphs.lock().unwrap();
        let Graphs { map, mode } = &mut *gr;
        match mode {
            GMode::Off => Ok(()),
            GMode::Record { segs, open, .. } => {
                let (dev, h) = open.take().ok_or_else(|| e("qwen35: a graph stretch ended twice"))?;
                let g = self.devs[dev].gpu.raw();
                // SAFETY: the stretch recorded on this GPU's queue: finished, then run once.
                ffi::check(unsafe { (self.k.graph_end)(g, h.0) }, "graph end")?;
                ffi::check(unsafe { (self.k.graph_launch)(g, h.0) }, "graph launch")?;
                segs.push((dev, h));
                Ok(())
            }
            GMode::Replay { key, idx } => {
                self.skip.store(false, std::sync::atomic::Ordering::Relaxed);
                let segs = map.get(key).ok_or_else(|| e("qwen35: a replayed pass without its graphs"))?;
                let (dev, h) = segs.get(*idx).ok_or_else(|| e("qwen35: a replayed pass with more stretches than recorded"))?;
                // SAFETY: a finished graph of this GPU.
                ffi::check(unsafe { (self.k.graph_launch)(self.devs[*dev].gpu.raw(), h.0) }, "graph launch")?;
                *idx += 1;
                Ok(())
            }
        }
    }

    /// A stretch begins (after a position-taking kernel, or on the next GPU)
    fn seg_begin(&self) -> Result<()> {
        let recording = matches!(self.graphs.lock().unwrap().mode, GMode::Record { .. });
        if recording {
            let open = self.graph_open()?;
            if let GMode::Record { open: o, .. } = &mut self.graphs.lock().unwrap().mode {
                *o = Some(open);
            }
        } else if matches!(self.graphs.lock().unwrap().mode, GMode::Replay { .. }) {
            self.skip.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// The pass's graphs end: the last stretch closed, a recording kept
    fn graph_finish(&self) -> Result<()> {
        self.seg_end()?;
        let mut gr = self.graphs.lock().unwrap();
        match std::mem::take(&mut gr.mode) {
            GMode::Record { key, segs, .. } => {
                if gr.map.is_empty() {
                    eprintln!("[qwen35: decode passes as SYCL graphs: {} stretches a pass of {} rows]", segs.len(), key.1);
                }
                gr.map.insert(key, segs);
            }
            GMode::Replay { key, idx } => {
                if gr.map.get(&key).map(|v| v.len()) != Some(idx) {
                    gr.map.remove(&key);
                    return Err(e("qwen35: a replayed pass with fewer stretches than recorded"));
                }
            }
            GMode::Off => {}
        }
        Ok(())
    }

    /// After a failed pass: a recording closed and dropped, replaying stopped
    fn graph_abort(&self) {
        self.skip.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut gr = self.graphs.lock().unwrap();
        if let GMode::Record { open: Some((dev, h)), .. } = std::mem::take(&mut gr.mode) {
            // SAFETY: the open recording ended (the queue back to running work); the graph dropped unrun.
            unsafe { (self.k.graph_end)(self.devs[dev].gpu.raw(), h.0) };
        }
    }

    /// LogProbChain: a one-row pass's attention over its n keys (its query q, normed and turned, against the cache)
    /// added to this GPU's capture
    fn capture_row(&self, q: &DevBuf, kc: &DevBuf, cap: usize, n: usize, ty: KvTy) -> Result<()> {
        let mut c = self.capture.lock().unwrap();
        let Some(c) = c.as_mut() else { return Ok(()) };
        let dev = self.cur.load(std::sync::atomic::Ordering::Relaxed);
        if c.bufs[dev].as_ref().is_none_or(|b| b.len < cap * 4) {
            let b = DevBuf::f32(&self.devs[dev].gpu, cap)?;
            b.fill(0)?;
            c.bufs[dev] = Some(b);
        }
        let out = c.bufs[dev].as_ref().ok_or_else(|| e("the capture's buffer"))?;
        let g = &self.g;
        let scratch = self.arena().f32(2 * g.heads)?;
        // SAFETY: q [heads x head], the cache [kv heads][cap][head] half, n <= cap keys, out [cap].
        self.op("chain attention", String::new, n * g.heads_kv * ty.row(g.head), 4 * n * g.heads * g.head,
                || unsafe { (self.k.attn_probs)(self.raw(), ty.code(), q.fp(), kc.ptr(), g.heads as i64, g.heads_kv as i64, g.head as i64, cap as i64, n as i64, c.mode,
                                                scratch.fp(), out.fp()) })?;
        c.layers += 1;
        c.n = n;
        Ok(())
    }

    /// Starts (or stops) capturing attention for LogProbChain: each one-token pass from now adds, per full-attention
    /// layer, its heads' attention over the positions it reads (NS_Q35_CHAIN_HEADS: mean, the default, or max)
    pub fn capture_attention(&self, on: bool) {
        let mode = if std::env::var("NS_Q35_CHAIN_HEADS").is_ok_and(|v| v == "max") { 1 } else { 0 };
        *self.capture.lock().unwrap() = on.then(|| Capture { mode, bufs: (0..self.devs.len()).map(|_| None).collect(), layers: 0, n: 0 });
    }

    /// The attention captured since the last take: per position, the mean over the layers of the heads' weights the last
    /// one-token pass's query gave it; None when no layer captured
    pub fn take_attention(&self) -> Option<Vec<f32>> {
        let mut c = self.capture.lock().unwrap();
        let c = c.as_mut()?;
        if c.layers == 0 {
            return None;
        }
        let mut out = vec![0f32; c.n];
        for b in c.bufs.iter().flatten() {
            let v = b.view(0, c.n * 4).and_then(|v| v.to_f32()).ok()?;
            for (o, x) in out.iter_mut().zip(v) {
                *o += x;
            }
            b.fill(0).ok()?;
        }
        let n = c.layers as f32;
        c.layers = 0;
        Some(out.into_iter().map(|x| x / n).collect())
    }

    /// A full-attention layer on h [t, n_embd]: its output [t, n_embd]. The rows are one conversation's or several's
    /// (`segs`: each its rows, position, rotary positions and cache): the products take all rows at once, the positions,
    /// caches and attention each conversation's own rows - each as its own pass would
    fn full(&self, a: &Full, segs: &[FullSeg], h: &DevBuf, hq8: Option<&Q8>, t: usize) -> Result<DevBuf> {
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
        for sg in segs {
            let (r0, t, p0, cap, kc, vc) = (sg.r0, sg.t, sg.p0, sg.cap, sg.k, sg.v);
            let rows = |b: &DevBuf, w: usize| b.view(r0 * w * 4, t * w * 4);
            let (qf, kr, vr) = (rows(&qf, hq * 2 * d)?, rows(&kr, hk * d)?, rows(&vr, hk * d)?);
            let (q, kn, att) = (rows(&q, hq * d)?, rows(&kn, hk * d)?, rows(&att, hq * d)?);
            // the keys the rows see in all (causal): what attention reads and multiplies
            let keys = t * p0 + t * (t + 1) / 2;
            // a graphed pass: the position-taking kernels run on their own, their scratch the most the cache needs (the
            // arena's offsets after it the same each pass)
            let graphing = self.graphing();
            let p_sz = if graphing { cap.saturating_sub(t) } else { p0 };
            self.seg_end()?;
            let ty = sg.ty(g);
            let rot = self.kv_rot && ty != KvTy::F16;
            // SAFETY: the rows and caches as named in q35.h.
            unsafe {
                match sg.rope {
                    Rope::From(r0) => {
                        self.op("q norm rope", none, 8 * t * hq * d, 10 * t * hq * d,
                                || (k.qk_norm_rope)(self.raw(), qf.fp(), (hq * 2 * d) as i64, (2 * d) as i64, q.fp(), (hq * d) as i64, t as i64, hq as i64,
                                                    d as i64, a.q_norm.fp(), g.eps, g.n_rot as i64, g.theta, r0 as i64))?;
                        self.op("k norm rope", none, 8 * t * hk * d, 10 * t * hk * d,
                                || (k.qk_norm_rope)(self.raw(), kr.fp(), (hk * d) as i64, d as i64, kn.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64,
                                                    a.k_norm.fp(), g.eps, g.n_rot as i64, g.theta, r0 as i64))?;
                    }
                    Rope::Rows(pos) => {
                        let (s0, s1, s2) = g.sections;
                        self.op("q norm rope", none, 8 * t * hq * d, 10 * t * hq * d,
                                || (k.qk_norm_rope_pos)(self.raw(), qf.fp(), (hq * 2 * d) as i64, (2 * d) as i64, q.fp(), (hq * d) as i64, t as i64, hq as i64,
                                                        d as i64, a.q_norm.fp(), g.eps, g.n_rot as i64, g.theta, pos.ptr().cast(), s0, s1, s2))?;
                        self.op("k norm rope", none, 8 * t * hk * d, 10 * t * hk * d,
                                || (k.qk_norm_rope_pos)(self.raw(), kr.fp(), (hk * d) as i64, d as i64, kn.fp(), (hk * d) as i64, t as i64, hk as i64,
                                                        d as i64, a.k_norm.fp(), g.eps, g.n_rot as i64, g.theta, pos.ptr().cast(), s0, s1, s2))?;
                    }
                }
                if rot {
                    // the keys, values and queries turned by the same orthogonal blocks: q . k unchanged, the values'
                    // turn undone on the output below
                    for (x, rows) in [(&kn, t * hk), (&vr, t * hk), (&q, t * hq)] {
                        self.op("kv rotate", none, 8 * rows * d, 5 * rows * d, || (k.fwht32)(self.raw(), x.fp(), rows as i64, d as i64, d as i64))?;
                    }
                }
                if ty == KvTy::F16 {
                    self.op("kv store", none, 12 * t * hk * d, 0,
                            || (k.kv_store)(self.raw(), kn.fp(), (hk * d) as i64, vr.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64, cap as i64,
                                            p0 as i64, kc.ptr(), vc.ptr()))?;
                } else {
                    self.op("kv store q", none, 8 * t * hk * d + 2 * t * hk * ty.row(d), 0,
                            || (k.kv_store_q)(self.raw(), ty.code(), kn.fp(), (hk * d) as i64, vr.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64,
                                              cap as i64, p0 as i64, kc.ptr(), vc.ptr()))?;
                }
                if let Some(s) = self.silo.filter(|s| (s.attn_prompt_supported)(t as i64, hq as i64, hk as i64, d as i64) != 0) {
                    // the silo's flash attention (prompt rows): each key read once a work-group of 8 rows x 6 heads
                    // each block of 8 rows reads K and V (half) up to its last row's key, once for the 6 heads
                    let nb = t.div_ceil(8);
                    let block_keys = nb * p0 + 8 * nb * (nb + 1) / 2;
                    self.op("attention silo", none, block_keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                            || (s.attn_prompt_kv)(self.raw(), ty.code(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64,
                                                  cap as i64, p0 as i64, att.fp()))?;
                } else if let Some(s) = self.silo.filter(|s| (s.attn_decode_supported)(t as i64, hq as i64, hk as i64, d as i64) != 0) {
                    // the silo's decode attention: a lane a key for the scores, the keys over ~512 work-groups
                    let sc = (s.attn_decode_scratch)(t as i64, hq as i64, hk as i64, d as i64, p_sz as i64) as usize;
                    let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
                    self.op("attention silo", none, keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                            || (s.attn_decode_kv)(self.raw(), ty.code(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64,
                                                  cap as i64, p0 as i64, att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
                } else if ty != KvTy::F16 {
                    return Err(e(format!("qwen35: a {} cache and no silo attention for these rows", ty.name())));
                } else {
                    let sc = (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, p_sz as i64) as usize;
                    let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
                    self.op("attention", none, keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                            || (k.attn)(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64,
                                        p0 as i64, att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
                }
            }
            if sg.probe && t == 1 {
                self.capture_row(&q, kc, cap, p0 + 1, ty)?;
            }
            if rot {
                // SAFETY: the attention's rows [t][hq x d].
                self.op("kv rotate", none, 8 * t * hq * d, 5 * t * hq * d, || unsafe { (k.fwht32)(self.raw(), att.fp(), (t * hq) as i64, d as i64, d as i64) })?;
            }
            self.seg_begin()?;
        }
        // SAFETY: att [t, hq x d] gated by qf's gate columns.
        self.op("output gate", none, 12 * t * hq * d, 4 * t * hq * d,
                || unsafe { (k.gate_mul)(self.raw(), att.fp(), qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64) })?;
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul("o", &a.o, &att, None, t, &out)?;
        Ok(out)
    }

    /// A DeltaNet layer on h [t, n_embd]: its output [t, n_embd]. `segs`: each conversation's rows with its state, conv
    /// inputs and where the state after each row but the last goes (a verify pass's snapshots: the recurrence's, the
    /// conv's) - the recurrence and the conv each conversation's own, the rest over all rows
    fn linear(&self, a: &Linear, segs: &[LinSeg], h: &DevBuf, hq8: Option<&Q8>, t: usize) -> Result<DevBuf> {
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
            for sg in segs {
                let t = sg.t;
                self.op("conv silu", none, 4 * (2 * t * cd + 2 * (g.conv - 1) * cd + g.conv * cd), 2 * g.conv * t * cd + 4 * t * cd,
                        || (k.conv_silu)(self.raw(), qkv.fp().add(sg.r0 * cd), sg.conv.fp(), a.conv.fp(), co.fp().add(sg.r0 * cd), t as i64, cd as i64, g.conv as i32,
                                         sg.snap.map_or(std::ptr::null_mut(), |x| x.1)))?;
            }
            self.op("expand q", none, 4 * t * hk * d + 4 * thd, 0, || (k.expand)(self.raw(), co.fp(), cd as i64, q.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand k", none, 4 * t * hk * d + 4 * thd, 0,
                    || (k.expand)(self.raw(), co.fp().add(qd as usize), cd as i64, kk.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand v", none, 8 * thd, 0, || (k.expand)(self.raw(), co.fp().add(2 * qd as usize), cd as i64, v.fp(), t as i64, hv as i64, hv as i64, d as i64))?;
            self.op("l2 q", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), q.fp(), (t * hv) as i64, d as i64, g.eps))?;
            self.op("l2 k", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), kk.fp(), (t * hv) as i64, d as i64, g.eps))?;
            // the state [hv][d][d] read and written once, each row's q, k, v, decay in and output out; per row and head
            // the state decayed, read against k, updated, read against q: ~8 d^2
            for sg in segs {
                let (t, r) = (sg.t, sg.r0 * hv * d);
                let thd = t * hv * d;
                self.op("delta rule", none, 8 * hv * d * d + 4 * (5 * thd + t * hv), 8 * t * hv * d * d,
                        || (k.kda_scan)(self.raw(), q.fp().add(r), kk.fp().add(r), v.fp().add(r), eg.fp().add(r), beta.fp().add(sg.r0 * hv), sg.s.fp(), o.fp().add(r),
                                        t as i64, hv as i64, d as i64, sg.snap.map_or(std::ptr::null_mut(), |x| x.0)))?;
            }
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

    /// `pass_in`, a failed pass's graph recording closed
    fn pass_in(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let r = self.pass_eager_or_graphed(s, tokens, n_out, tap);
        if r.is_err() {
            self.graph_abort();
        }
        r
    }

    /// `pass` with the work lock held: a verify pass (2 ..= drafts + 1 rows, drafts on) keeps the DeltaNet
    /// snapshots; with MTP the draft block runs over the rows behind it
    fn pass_eager_or_graphed(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
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
        // a picture's rows first (the tower in the first GPU's arena, given back after), then each row's rotary
        // positions on each GPU that runs layers of this pass
        let pictures = tokens.iter().any(|t| crate::vision::is_image(*t));
        let mut shift = s.shift;
        let mut pos_rows: Vec<Option<DevBuf>> = Vec::new();
        if pictures {
            self.phase("vision");
            self.image_rows_for(tokens)?;
            let vis = self.vision.as_ref().ok_or_else(|| e("a picture's span, and no vision tower (NS_Q35_VISION)"))?;
            let (pos, after) = crate::vision::rope_rows(tokens, p0, s.shift, &|id| Ok(vis.image(id)?.plan))?;
            let bytes: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
            for d in &self.devs {
                let b = d.arena.bytes(bytes.len())?;
                b.write(0, &bytes)?;
                pos_rows.push(Some(b));
            }
            shift = after;
        }
        let r0 = p0 - s.shift;
        let rope_at = |dev: usize| -> Rope { match pos_rows.get(dev).and_then(|b| b.as_ref()) { Some(b) => Rope::Rows(b), None => Rope::From(r0) } };
        let probing = t == 1 && self.capture.lock().unwrap().is_some();
        let mut snaps = self.snaps.lock().unwrap();
        let snapping = (2..=self.drafts + 1).contains(&t) && !snaps.layers.is_empty();
        snaps.of.clear();
        self.phase("embed");
        let mut x = self.embed(tokens)?;
        if pictures {
            self.put_images(tokens, &x)?;
        }
        let mut h = self.arena().f32(t * g.n_embd)?;
        let mut ffn_g = self.arena().f32(t * g.n_ff)?;
        let mut ffn_u = self.arena().f32(t * g.n_ff)?;
        let mut ffn = self.arena().f32(t * g.n_embd)?;
        let mut li = 0;
        // EAGLE: the residual stream after its three layers (its cache holding these positions)
        let eagle = self.eagle.as_ref().zip(s.mtp.as_ref()).filter(|(_, ms)| p0 + t <= ms.cap);
        let aux = match eagle {
            Some(_) => Some(self.arena().f32(3 * t * g.n_embd)?),
            None => None,
        };
        // decode rows (no pictures, no capture, no profile, no EAGLE): the layers as graphs
        let graphed = t <= 8 && !pictures && !probing && self.probe.is_none() && self.eagle.is_none() && graphs_on()
            && !self.graphs_off.load(std::sync::atomic::Ordering::Relaxed);
        if graphed {
            self.graph_start((s.id, t, snapping))?;
        }
        // the arena past this point is the layer's own, made again each layer
        for (l, (ly, st)) in self.layers.iter().zip(s.layers.iter()).enumerate() {
            // the next GPU's layers: the rows cross over, its own buffers for them
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                self.phase("GPU to GPU");
                self.seg_end()?;
                x = self.hop(&x, self.layer_dev[l])?;
                self.seg_begin()?;
                h = self.arena().f32(t * g.n_embd)?;
                ffn_g = self.arena().f32(t * g.n_ff)?;
                ffn_u = self.arena().f32(t * g.n_ff)?;
                ffn = self.arena().f32(t * g.n_embd)?;
            }
            let mark = self.arena().mark();
            self.phase(if ly.full.is_some() { "attention" } else { "deltanet" });
            let hq8 = self.norm(&x, &ly.attn_norm, &h, t, g.n_embd)?;
            let out = match (st, &ly.full, &ly.linear) {
                (LState::Full { k, v }, Some(a), _) => {
                    let seg = FullSeg { r0: 0, t, p0, rope: rope_at(self.layer_dev[l]), cap: s.max_ctx, k, v, probe: probing };
                    self.full(a, &[seg], &h, hq8.as_ref(), t)?
                }
                (LState::Linear { s: rs, conv }, _, Some(a)) => {
                    let sn = snapping.then(|| (snaps.layers[li].0.fp(), snaps.layers[li].1.fp()));
                    li += 1;
                    self.linear(a, &[LinSeg { r0: 0, t, s: rs, conv, snap: sn }], &h, hq8.as_ref(), t)?
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
            if let (Some((ea, _)), Some(a)) = (eagle, &aux) {
                if let Some(j) = ea.ids.iter().position(|i| *i == l) {
                    a.copy_within(j * t * g.n_embd * 4, &x, 0, t * g.n_embd * 4)?;
                }
            }
            self.arena().rewind(mark);
        }
        if graphed {
            self.graph_finish()?;
        }
        if snapping {
            snaps.of.push((s.id, p0, t, 0));
        }
        drop(snaps);
        s.pos += t;
        s.last_rows = t;
        s.shift = shift;
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
            self.mtp_rows(ms, tokens, &hs, p0, s.max_ctx, false, rope_at(self.last()))?;
            ms.pending.copy_within(0, &all, (t - 1) * ne * 4, ne * 4)?;
            self.phase("head");
            // the head reads the rows' Q8_1 forms when it takes them all; fewer rows (a short prompt's last) it quantizes
            // itself (the split layout does not slice by rows)
            (all.view((t - n) * ne * 4, n * ne * 4)?, if n == t { q } else { None })
        } else if let (Some((ea, ms)), Some(a)) = (eagle, &aux) {
            self.phase("eagle");
            self.eagle_rows(ea, ms, tokens, a, p0, r0)?;
            self.phase("head");
            let last = x.view((t - n) * ne * 4, n * ne * 4)?;
            let normed = self.arena().f32(n * ne)?;
            let q = self.norm(&last, &self.output_norm, &normed, n, ne)?;
            tap("result_norm", &normed)?;
            (normed, q)
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
    #[allow(clippy::too_many_arguments)]
    fn mtp_rows(&self, ms: &MtpState, tokens: &[u32], hs: &DevBuf, p0: usize, cap: usize, head: bool, rope: Rope) -> Result<Option<u32>> {
        let Some(m) = &self.mtp else { return Ok(None) };
        let (t, ne) = (tokens.len(), self.g.n_embd);
        let cur = self.mtp_block(m, &[FullSeg { r0: 0, t, p0, rope, cap, k: &ms.k, v: &ms.v, probe: false }], tokens, hs)?;
        if !head {
            return Ok(None);
        }
        let last = cur.view((t - 1) * ne * 4, ne * 4)?;
        Ok(self.mtp_heads(m, &[ms], &last)?.pop())
    }

    /// The draft block's head on rows ([n][n_embd], the block's outputs, one a conversation): each one's normed row kept
    /// for a chained draft, each one's likeliest token (the argmax on the GPU: 4 bytes a row read back)
    fn mtp_heads(&self, m: &Mtp, mss: &[&MtpState], rows: &DevBuf) -> Result<Vec<u32>> {
        let (n, ne) = (mss.len(), self.g.n_embd);
        let hn2 = self.arena().f32(n * ne)?;
        let q = self.norm(rows, &m.head_norm, &hn2, n, ne)?;
        for (j, ms) in mss.iter().enumerate() {
            ms.chain.copy_within(0, &hn2, j * ne * 4, ne * 4)?;
        }
        let logits = self.arena().f32(n * m.output.rows)?;
        self.matmul("mtp head", &m.output, &hn2, q.as_ref(), n, &logits)?;
        let best = self.arena().bytes(4 * n)?;
        // SAFETY: logits [n, vocab] float32 -> each row's argmax (an int32 each).
        self.op("argmax", String::new, n * m.output.rows * 4, 0,
                || unsafe { (self.k.argmax)(self.raw(), logits.fp(), n as i64, m.output.rows as i64, best.ptr().cast()) })?;
        let mut b = vec![0u8; 4 * n];
        best.read(0, &mut b)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// The draft block over `tokens` (row i: token i with hidden state hs[i]), each conversation's rows (`segs`) at its
    /// positions in its own cache: its output rows [t][n_embd]
    fn mtp_block(&self, m: &Mtp, segs: &[FullSeg], tokens: &[u32], hs: &DevBuf) -> Result<DevBuf> {
        let g = &self.g;
        let (t, ne) = (tokens.len(), g.n_embd);
        let emb = self.embed_from(&m.file, &m.embd, m.vocab, tokens)?;
        // a picture's positions read its rows here too
        if tokens.iter().any(|t| crate::vision::is_image(*t)) {
            self.put_images(tokens, &emb)?;
        }
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
        let att = self.full(&m.attn, segs, &h, hq8.as_ref(), t)?;
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
        Ok(cur)
    }

    /// EAGLE-4 behind a pass of t rows at p0 (rotary positions from r0): each row's feature from the aux states, the
    /// draft over the rows EAGLE_ROWS at a time (row i: token i with the feature of the row before - the pending one,
    /// then this pass's), the first rows' features kept for a rollback, the last one pending
    fn eagle_rows(&self, ea: &crate::eagle::Eagle, ms: &MtpState, tokens: &[u32], aux: &DevBuf, p0: usize, r0: usize) -> Result<()> {
        let (t, ne) = (tokens.len(), self.g.n_embd);
        let prev = self.arena().f32(ne)?;
        prev.copy_within(0, &ms.pending, 0, ne * 4)?;
        for a in (0..t).step_by(EAGLE_ROWS) {
            let n = EAGLE_ROWS.min(t - a);
            let mark = self.arena().mark();
            // feature = sum_j fc_j . rms_j(aux_j)
            let feat = self.arena().f32(n * ne)?;
            let (xn, tmp) = (self.arena().f32(n * ne)?, self.arena().f32(n * ne)?);
            for j in 0..3 {
                let xa = aux.view((j * t + a) * ne * 4, n * ne * 4)?;
                let q = self.norm(&xa, &ea.fc_norm[j], &xn, n, ne)?;
                self.matmul("eagle fc", &ea.fc[j], &xn, q.as_ref(), n, if j == 0 { &feat } else { &tmp })?;
                if j > 0 {
                    self.add(&feat, &tmp, n * ne)?;
                }
            }
            if a == 0 {
                self.vh.copy_within(0, &feat, 0, n.min(8) * ne * 4)?;
            }
            let hs = self.arena().f32(n * ne)?;
            hs.copy_within(0, &prev, 0, ne * 4)?;
            if n > 1 {
                hs.copy_within(ne * 4, &feat, 0, (n - 1) * ne * 4)?;
            }
            prev.copy_within(0, &feat, (n - 1) * ne * 4, ne * 4)?;
            self.eagle_block(ea, ms, &tokens[a..a + n], &hs, p0 + a, r0 + a)?;
            self.arena().rewind(mark);
        }
        ms.pending.copy_within(0, &prev, 0, ne * 4)
    }

    /// The EAGLE-4 layer over `tokens` at positions p0.. (rotary r0..; row i: token i with hidden hs[i]), its cache
    /// written there: its normed output rows [t][n_embd]
    fn eagle_block(&self, ea: &crate::eagle::Eagle, ms: &MtpState, tokens: &[u32], hs: &DevBuf, p0: usize, r0: usize) -> Result<DevBuf> {
        let (k, t, ne) = (self.k, tokens.len(), self.g.n_embd);
        let (hq, d) = (ea.heads, ea.head_dim);
        let emb = self.embed(tokens)?;
        if tokens.iter().any(|t| crate::vision::is_image(*t)) {
            self.put_images(tokens, &emb)?;
        }
        let (en, hn) = (self.arena().f32(t * ne)?, self.arena().f32(t * ne)?);
        let eq = self.norm(&emb, &ea.in_norm, &en, t, ne)?;
        let hq8 = self.norm(hs, &ea.hidden_norm, &hn, t, ne)?;
        // [en | hn] . W^T = W_e . en + W_h . hn
        let proj = |w: &[Mat; 2], role: &'static str| -> Result<DevBuf> {
            let y = self.arena().f32(t * hq * d)?;
            let y2 = self.arena().f32(t * hq * d)?;
            self.matmul(role, &w[0], &en, eq.as_ref(), t, &y)?;
            self.matmul(role, &w[1], &hn, hq8.as_ref(), t, &y2)?;
            self.add(&y, &y2, t * hq * d)?;
            Ok(y)
        };
        let (q, kr, v) = (proj(&ea.q, "eagle q")?, proj(&ea.k, "eagle k")?, proj(&ea.v, "eagle v")?);
        let (qr, kn) = (self.arena().f32(t * hq * d)?, self.arena().f32(t * hq * d)?);
        let none = String::new;
        let w = (hq * d) as i64;
        // the window: W keys ending at the query (decode rows exactly; a prompt's rows from its first row's window)
        let start = match ea.window {
            0 => 0,
            wn if t <= 8 => (p0 + t).saturating_sub(wn),
            wn => (p0 + 1).saturating_sub(wn),
        };
        let rb = d * 2;
        let att = self.arena().f32(t * hq * d)?;
        // SAFETY: the rows and the cache ([heads][cap][d] half) as named in q35.h; the attention reads the cache from
        // key `start` (its pointers moved there, its positions counted from there).
        unsafe {
            self.op("eagle rope", none, 8 * t * hq * d, 10 * t * hq * d,
                    || (k.rope)(self.raw(), q.fp(), w, d as i64, qr.fp(), w, t as i64, hq as i64, d as i64, d as i64, ea.theta, r0 as i64))?;
            self.op("eagle rope", none, 8 * t * hq * d, 10 * t * hq * d,
                    || (k.rope)(self.raw(), kr.fp(), w, d as i64, kn.fp(), w, t as i64, hq as i64, d as i64, d as i64, ea.theta, r0 as i64))?;
            self.op("eagle kv store", none, 12 * t * hq * d, 0,
                    || (k.kv_store)(self.raw(), kn.fp(), w, v.fp(), w, t as i64, hq as i64, d as i64, ms.cap as i64, p0 as i64, ms.k.ptr(), ms.v.ptr()))?;
            let (kc, vc) = (ms.k.view(start * rb, ms.k.len - start * rb)?, ms.v.view(start * rb, ms.v.len - start * rb)?);
            let pa = p0 - start;
            let keys = t * pa + t * (t + 1) / 2;
            let sc = (k.attn_scratch)(t as i64, hq as i64, hq as i64, d as i64, pa as i64) as usize;
            let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
            self.op("eagle attention", none, keys * hq * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                    || (k.attn)(self.raw(), qr.fp(), w, kc.ptr(), vc.ptr(), t as i64, hq as i64, hq as i64, d as i64, ms.cap as i64, pa as i64, att.fp(),
                                part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
        }
        // r = hs + o(att); s = r + mlp(post_norm(r)); out = norm(s)
        let r = self.arena().f32(t * ne)?;
        self.matmul("eagle o", &ea.o, &att, None, t, &r)?;
        self.add(&r, hs, t * ne)?;
        let h = self.arena().f32(t * ne)?;
        let q8 = self.norm(&r, &ea.post_norm, &h, t, ne)?;
        let (fg, fu) = (self.arena().f32(t * ea.n_ff)?, self.arena().f32(t * ea.n_ff)?);
        self.matmul("eagle gate", &ea.gate, &h, q8.as_ref(), t, &fg)?;
        self.matmul("eagle up", &ea.up, &h, q8.as_ref(), t, &fu)?;
        let nf = t * ea.n_ff;
        // SAFETY: n floats each.
        self.op("swiglu", String::new, 12 * nf, 6 * nf, || unsafe { (k.swiglu_clamp)(self.raw(), fg.fp(), fu.fp(), fg.fp(), nf as i64, f32::INFINITY) })?;
        self.matmul("eagle down", &ea.down, &fg, None, t, &h)?;
        self.add(&r, &h, t * ne)?;
        let out = self.arena().f32(t * ne)?;
        self.rms(&r, &ea.norm, &out, t, ne)?;
        Ok(out)
    }

    /// EAGLE-4 drafts after `tail` (committed at s.pos, not fed): its layer on (the pending feature, tail), then
    /// chained on its own normed output and each draft; none past its cache
    fn eagle_draft(&self, ea: &crate::eagle::Eagle, s: &Session, ms: &MtpState, tail: u32) -> Result<Vec<u32>> {
        let ne = self.g.n_embd;
        let k = self.drafts.min(s.max_ctx.saturating_sub(s.pos + 1)).min(ms.cap.saturating_sub(s.pos + 1));
        let mut out = Vec::with_capacity(k);
        let mut tok = tail;
        self.on(self.last());
        for i in 0..k {
            self.arena().reset();
            self.phase("eagle draft");
            let hs = if i == 0 { &ms.pending } else { &ms.chain };
            let y = self.eagle_block(ea, ms, &[tok], hs, s.pos + i, s.pos + i - s.shift)?;
            ms.chain.copy_within(0, &y, 0, ne * 4)?;
            tok = if ea.full_head { self.eagle_head_full(ea, &y)? } else { self.eagle_head(ea, &y)? };
            out.push(tok);
        }
        Ok(out)
    }

    /// The draft's token from its output row y (eagle.rs): the low-rank scores, their top candidates, those scored by
    /// the full head's rows - the best of them (ids and scores read back together, the choice made here)
    fn eagle_head(&self, ea: &crate::eagle::Eagle, y: &DevBuf) -> Result<u32> {
        let (k, ne, vocab, rank, c) = (self.k, self.g.n_embd, ea.head.rows, ea.lr_down.rows, ea.cands);
        let low = self.arena().f32(rank)?;
        let scores = self.arena().f32(vocab)?;
        let pick = self.arena().bytes(8 * c)?;
        let (ids, cs) = (pick.view(0, 4 * c)?, pick.view(4 * c, 4 * c)?);
        let none = String::new;
        // SAFETY: the Q8_0 matrices with their rows and columns, y ne floats, low rank floats, scores vocab floats, ids
        // c int32s, cs c floats.
        unsafe {
            self.matmul("eagle low rank", &ea.lr_down, y, None, 1, &low)?;
            self.op("eagle low rank", none, ea.lr_q.len + ea.lr_d.len + 4 * vocab, 2 * vocab * rank,
                    || (k.scores_i8)(self.raw(), ea.lr_q.ptr(), ea.lr_d.ptr(), vocab as i64, low.fp(), scores.fp()))?;
            let sc = (k.topk_scratch)(vocab as i64, c as i64) as usize;
            let scratch = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
            self.op("eagle top k", none, 20 * vocab, 0, || (k.topk)(self.raw(), scores.fp(), vocab as i64, c as i64, ids.ptr().cast(),
                                                                    scratch.as_ref().map_or(std::ptr::null_mut(), |s| s.ptr().cast())))?;
            self.op("eagle candidates", none, c * ne * 34 / 32, 2 * c * ne,
                    || (k.rows_q8_0)(self.raw(), ea.head.buf.ptr(), ne as i64, ids.ptr().cast(), c as i64, y.fp(), cs.fp()))?;
        }
        let mut b = vec![0u8; 8 * c];
        pick.read(0, &mut b)?;
        let word = |i: usize| [b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]];
        let best = (0..c).max_by(|&i, &j| f32::from_le_bytes(word(c + i)).total_cmp(&f32::from_le_bytes(word(c + j)))).unwrap_or(0);
        Ok(u32::from_le_bytes(word(best)))
    }

    /// The draft's token from the full head (NS_Q35_EAGLE_HEAD=full): its logits' argmax
    fn eagle_head_full(&self, ea: &crate::eagle::Eagle, y: &DevBuf) -> Result<u32> {
        let logits = self.arena().f32(ea.head.rows)?;
        self.matmul("eagle head", &ea.head, y, None, 1, &logits)?;
        let best = self.arena().bytes(4)?;
        // SAFETY: logits [1, vocab] float32 -> its argmax (an int32).
        self.op("argmax", String::new, ea.head.rows * 4, 0, || unsafe { (self.k.argmax)(self.raw(), logits.fp(), 1, ea.head.rows as i64, best.ptr().cast()) })?;
        let mut b = [0u8; 4];
        best.read(0, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Drafts for several conversations at once (`tails[i]` committed at ss[i].pos, not fed): each draft step one row
    /// a conversation through the draft block, as `draft_in` makes them one at a time
    fn draft_many(&self, ss: &[&Session], tails: &[u32]) -> Result<Vec<Vec<u32>>> {
        if self.eagle.is_some() {
            return ss.iter().zip(tails).map(|(s, t)| self.draft_in(s, *t)).collect();
        }
        let Some(m) = &self.mtp else { return Ok(vec![Vec::new(); ss.len()]) };
        let ne = self.g.n_embd;
        let ks: Vec<usize> = ss.iter().map(|s| if s.mtp.is_some() { self.drafts.min(s.max_ctx.saturating_sub(s.pos + 1)) } else { 0 }).collect();
        let mut out: Vec<Vec<u32>> = ks.iter().map(|k| Vec::with_capacity(*k)).collect();
        let mut toks: Vec<u32> = tails.to_vec();
        self.on(self.last());
        for i in 0..ks.iter().copied().max().unwrap_or(0) {
            self.arena().reset();
            self.phase("mtp draft");
            let live: Vec<usize> = (0..ss.len()).filter(|j| i < ks[*j]).collect();
            let n = live.len();
            let hs = self.arena().f32(n * ne)?;
            let mut segs = Vec::with_capacity(n);
            let mut mss = Vec::with_capacity(n);
            for (r, &j) in live.iter().enumerate() {
                let (s, ms) = (ss[j], ss[j].mtp.as_ref().ok_or_else(|| e("a draft without the draft block's state"))?);
                hs.copy_within(r * ne * 4, if i == 0 { &ms.pending } else { &ms.chain }, 0, ne * 4)?;
                segs.push(FullSeg { r0: r, t: 1, p0: s.pos + i, rope: Rope::From(s.pos + i - s.shift), cap: s.max_ctx, k: &ms.k, v: &ms.v, probe: false });
                mss.push(ms);
            }
            let rows: Vec<u32> = live.iter().map(|j| toks[*j]).collect();
            let cur = self.mtp_block(m, &segs, &rows, &hs)?;
            for (r, y) in self.mtp_heads(m, &mss, &cur)?.into_iter().enumerate() {
                toks[live[r]] = y;
                out[live[r]].push(y);
            }
        }
        Ok(out)
    }

    /// Drafts for the positions after `tail` (committed at s.pos, not fed): the draft block on (the last fed
    /// position's hidden state, tail), then chained on its own normed output and each draft
    fn draft_in(&self, s: &Session, tail: u32) -> Result<Vec<u32>> {
        let Some(ms) = &s.mtp else { return Ok(Vec::new()) };
        if let Some(ea) = &self.eagle {
            return self.eagle_draft(ea, s, ms, tail);
        }
        let k = self.drafts.min(s.max_ctx.saturating_sub(s.pos + 1));
        let mut out = Vec::with_capacity(k);
        let mut tok = tail;
        // the draft block is on the last GPU
        self.on(self.last());
        for i in 0..k {
            self.arena().reset();
            self.phase("mtp draft");
            let hs = if i == 0 { &ms.pending } else { &ms.chain };
            let Some(t) = self.mtp_rows(ms, &[tok], hs, s.pos + i, s.max_ctx, true, Rope::From(s.pos + i - s.shift))? else { break };
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
        let Some(&(id, p0, rows, set)) = snaps.of.iter().find(|o| o.0 == s.id) else {
            return Err(e("qwen35: a rollback needs the last pass's snapshots (a verify pass of 2 ..= drafts + 1 rows, drafts on)"));
        };
        if p0 + rows != s.pos || keep == 0 || keep >= rows {
            return Err(e(format!("qwen35: a rollback to {keep} of a pass of {rows} rows at {p0} (session {id}); this session is at {}", s.pos)));
        }
        let layers = snaps.set(set).ok_or_else(|| e("qwen35: a rollback to a snapshot set not made"))?;
        let g = &self.g;
        let (sb, cb) = (g.v_heads * g.state * g.state * 4, (g.conv - 1) * g.conv_dim() * 4);
        let mut li = 0;
        for st in &s.layers {
            if let LState::Linear { s: rs, conv } = st {
                let (ss, cs) = &layers[li];
                rs.copy_within(0, ss, (keep - 1) * sb, sb)?;
                conv.copy_within(0, cs, (keep - 1) * cb, cb)?;
                li += 1;
            }
        }
        if let Some(ms) = &s.mtp {
            ms.pending.copy_within(0, &self.vh, (set * 8 + keep - 1) * g.n_embd * 4, g.n_embd * 4)?;
        }
        snaps.of.retain(|o| o.0 != id);
        s.pos = p0 + keep;
        s.last_rows = keep;
        Ok(())
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        // the caches and states, checked against what is free first: past the card the xe driver spills into host
        // memory (no out-of-memory error) and can take the machine down
        let last_ty = self.kv_tys[self.last()];
        let mtp_kv = match (&self.mtp, &self.eagle) {
            (_, Some(ea)) => 2 * ea.heads * max_ctx.min(ea.ctx) * ea.head_dim * 2,
            (Some(_), _) => 2 * g.heads_kv * max_ctx * last_ty.row(g.head),
            _ => 0,
        };
        // per GPU: its layers' share (the draft block's cache on the last)
        for (i, d) in self.devs.iter().enumerate() {
            let full = (d.lb..d.le).filter(|l| g.full(*l)).count();
            let kv = if i == self.last() { mtp_kv } else { 0 };
            let need = (full * 2 * g.heads_kv * max_ctx * self.kv_tys[i].row(g.head) + kv + (d.le - d.lb - full) * (g.v_heads * g.state * g.state + (g.conv - 1) * g.conv_dim()) * 4) as u64;
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
                let n = g.heads_kv * max_ctx * self.kv_tys[self.layer_dev[l]].row(g.head);
                LState::Full { k: DevBuf::new(gpu, n)?, v: DevBuf::new(gpu, n)? }
            } else {
                let s = DevBuf::f32(gpu, g.v_heads * g.state * g.state)?;
                let conv = DevBuf::f32(gpu, (g.conv - 1) * g.conv_dim())?;
                s.fill(0)?;
                conv.fill(0)?;
                LState::Linear { s, conv }
            });
        }
        // the draft's cache: MTP's in the target's form, EAGLE's [40][its positions][128] half
        let shape = match (&self.mtp, &self.eagle) {
            (_, Some(ea)) => Some((ea.heads, max_ctx.min(ea.ctx), ea.head_dim * 2)),
            (Some(_), _) => Some((g.heads_kv, max_ctx, last_ty.row(g.head))),
            _ => None,
        };
        let mtp = match shape {
            Some((heads, cap, rb)) => {
                let n = heads * cap * rb;
                let gpu = &self.devs[self.last()].gpu;
                let ms = MtpState { k: DevBuf::new(gpu, n)?, v: DevBuf::new(gpu, n)?, pending: DevBuf::f32(gpu, g.n_embd)?, chain: DevBuf::f32(gpu, g.n_embd)?,
                                    heads, cap };
                ms.pending.fill(0)?;
                Some(ms)
            }
            None => None,
        };
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Session { pos: 0, max_ctx, shift: 0, layers, last_rows: 0, id, mtp })
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
        s.shift = 0;
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
                        let rb = kv_rb(sk, g.heads_kv, src.max_ctx);
                        let n = src.pos * rb;
                        dk.copy_within(h * dst.max_ctx * rb, sk, h * src.max_ctx * rb, n)?;
                        dv.copy_within(h * dst.max_ctx * rb, sv, h * src.max_ctx * rb, n)?;
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
            let rb = kv_rb(&sm.k, sm.heads, sm.cap);
            let n = src.pos.min(sm.cap).min(d.cap) * rb;
            for h in 0..sm.heads {
                d.k.copy_within(h * d.cap * rb, &sm.k, h * sm.cap * rb, n)?;
                d.v.copy_within(h * d.cap * rb, &sm.v, h * sm.cap * rb, n)?;
            }
            d.pending.copy_within(0, &sm.pending, 0, g.n_embd * 4)?;
        }
        dst.pos = src.pos;
        dst.shift = src.shift;
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

    /// whether a quantized attention cache is rotated (NS_Q35_KV_ROT)
    pub fn kv_rotated(&self) -> bool {
        self.kv_rot
    }

    /// whether the drafts are EAGLE-4's (else MTP's)
    pub fn drafts_with_eagle(&self) -> bool {
        self.eagle.is_some()
    }

    pub fn has_draft(&self) -> bool {
        self.drafts > 0
    }

    pub fn max_verify(&self) -> usize {
        self.drafts + 1
    }

    /// A token each for several conversations: one pass over all their rows (8 at a time)
    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        let mut out = Vec::with_capacity(tokens.len());
        for (ss, ts) in sessions.chunks_mut(8).zip(tokens.chunks(8)) {
            let rows: Vec<Vec<u32>> = ts.iter().map(|t| vec![*t]).collect();
            for mut l in self.pass_batch(ss, &rows, &mut *tap)? {
                out.push(l.pop().unwrap_or_default());
            }
        }
        Ok(out)
    }

    /// Whether a batch step can keep a second conversation's verify snapshots: the second set made now (the cards'
    /// free memory checked first, 1.5 GiB kept), or false (the step then runs the conversations one after another)
    fn second_set(&self) -> Result<bool> {
        let mut snaps = self.snaps.lock().unwrap();
        if snaps.second.is_some() || snaps.layers.is_empty() {
            return Ok(true);
        }
        let g = &self.g;
        let (sb, cb) = (self.drafts * g.v_heads * g.state * g.state * 4, self.drafts * (g.conv - 1) * g.conv_dim() * 4);
        for (i, d) in self.devs.iter().enumerate() {
            let n = (0..g.layers).filter(|l| !g.full(*l) && self.layer_dev[*l] == i).count();
            let (total, free) = d.gpu.memory()?;
            if (n * (sb + cb)) as u64 + (1536 << 20) > free.unwrap_or(total) {
                return Ok(false);
            }
        }
        let mut v = Vec::with_capacity(snaps.layers.len());
        for l in (0..g.layers).filter(|l| !g.full(*l)) {
            let gpu = &self.devs[self.layer_dev[l]].gpu;
            v.push((DevBuf::f32(gpu, sb / 4)?, DevBuf::f32(gpu, cb / 4)?));
        }
        snaps.second = Some(v);
        Ok(true)
    }

    /// Whether `step_many` steps conversations together (else one after another)
    pub fn steps_together(&self) -> bool {
        self.eagle.is_none()
    }

    /// The next committed token(s) of several conversations, as `step` gives each alone: their verify rows ([token,
    /// drafts]) in one pass - the products read the weights once for all of them, the attention, recurrence and caches
    /// each its own - then each one's drafts. Two at a time with drafts (the snapshot sets), any number without
    pub fn step_many(&self, ss: &mut [&mut Session], ds: &mut [&mut Decoder], smps: &mut [&mut dyn Sampler], tap: Tap) -> Result<Vec<Vec<u32>>> {
        let _w = self.work.lock().unwrap();
        let n = ss.len();
        let drafting: Vec<bool> = (0..n).map(|i| ds[i].mtp && self.drafts > 0 && ss[i].mtp.is_some()).collect();
        // EAGLE: its features come from a single conversation's pass (`pass_batch` does not make them)
        let together = (n <= 2 || !drafting.iter().any(|d| *d)) && self.eagle.is_none();
        if !together || (drafting.iter().any(|d| *d) && n == 2 && !self.second_set()?) {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(self.step_in(ss[i], ds[i], smps[i], &mut *tap)?);
            }
            return Ok(out);
        }
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut fed: Vec<usize> = Vec::new();
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for i in 0..n {
            match (ds[i].logits.take(), ds[i].next.take()) {
                (Some(l), _) => out[i] = vec![smps[i].sample(&l)],
                (None, Some(x)) => {
                    let mut r = vec![x];
                    if drafting[i] {
                        r.extend(ds[i].drafts.iter().take(ss[i].max_ctx.saturating_sub(ss[i].pos + 1)));
                    }
                    ds[i].drafts.clear();
                    fed.push(i);
                    rows.push(r);
                }
                (None, None) => return Err(e("a decoder with nothing to draw from")),
            }
        }
        if !fed.is_empty() {
            let logits = {
                let mut sel: Vec<&mut Session> = ss.iter_mut().enumerate().filter(|(i, _)| fed.contains(i)).map(|(_, s)| &mut **s).collect();
                self.pass_batch(&mut sel, &rows, &mut *tap)?
            };
            for ((&i, r), ls) in fed.iter().zip(&rows).zip(logits) {
                let mut o = Vec::with_capacity(r.len());
                for (j, l) in ls.iter().enumerate() {
                    let y = smps[i].sample(l);
                    o.push(y);
                    if j + 1 < r.len() {
                        ds[i].drafted += 1;
                        if y == r[j + 1] {
                            ds[i].accepted += 1;
                            continue;
                        }
                    }
                    break;
                }
                if o.len() < r.len() {
                    self.rollback_in(ss[i], o.len())?;
                }
                out[i] = o;
            }
        }
        let mut tails = Vec::with_capacity(n);
        for i in 0..n {
            let tail = *out[i].last().ok_or_else(|| e("a step without a token"))?;
            ds[i].next = Some(tail);
            tails.push(tail);
        }
        // the drafts: one pass a draft step for all that draft
        let who: Vec<usize> = (0..n).filter(|i| drafting[*i]).collect();
        if !who.is_empty() {
            let sel: Vec<&Session> = who.iter().map(|i| &*ss[*i]).collect();
            let tl: Vec<u32> = who.iter().map(|i| tails[*i]).collect();
            for (i, d) in who.iter().zip(self.draft_many(&sel, &tl)?) {
                ds[*i].drafts = d;
            }
        }
        Ok(out)
    }

    /// One decode pass over several conversations' rows (`rows[i]` the next of `ss[i]`: a token, or a verify pass's
    /// [token, drafts]; 8 rows in all at most): each one's logits for each of its rows. The products run once over all
    /// rows; each conversation's positions, attention, recurrence and caches are its own, as its own pass would make
    /// them (the same tokens as alone). Verify rows keep snapshots for a rollback (two conversations at most: a set
    /// each); the draft block runs behind each.
    fn pass_batch(&self, ss: &mut [&mut Session], rows: &[Vec<u32>], tap: Tap) -> Result<Vec<Vec<Vec<f32>>>> {
        let g = &self.g;
        let ne = g.n_embd;
        let t: usize = rows.iter().map(|r| r.len()).sum();
        if t == 0 || t > 8 || rows.len() != ss.len() || rows.iter().any(|r| r.is_empty()) {
            return Err(e(format!("qwen35: a batch pass of {t} rows over {} conversations (1 to 8 rows, each at least one)", ss.len())));
        }
        for (s, r) in ss.iter().zip(rows) {
            if s.pos + r.len() > s.max_ctx {
                return Err(e(format!("the context is full ({} of {})", s.pos + r.len(), s.max_ctx)));
            }
            if r.iter().any(|x| crate::vision::is_image(*x)) {
                return Err(e("qwen35: a picture's span in a batch pass"));
            }
        }
        let r0s: Vec<usize> = rows.iter().scan(0, |a, r| {
            let v = *a;
            *a += r.len();
            Some(v)
        }).collect();
        for d in &self.devs {
            d.arena.reset();
        }
        self.on(0);
        let mut snaps = self.snaps.lock().unwrap();
        snaps.of.clear();
        // which conversations keep snapshots, in which set
        let mut sets: Vec<Option<usize>> = Vec::with_capacity(rows.len());
        let mut next_set = 0;
        for r in rows {
            if (2..=self.drafts + 1).contains(&r.len()) && !snaps.layers.is_empty() {
                if snaps.set(next_set).is_none() {
                    return Err(e("qwen35: verify rows of more conversations than snapshot sets"));
                }
                sets.push(Some(next_set));
                next_set += 1;
            } else {
                sets.push(None);
            }
        }
        let tokens: Vec<u32> = rows.iter().flatten().copied().collect();
        self.phase("embed");
        let mut x = self.embed(&tokens)?;
        let mut h = self.arena().f32(t * ne)?;
        let mut ffn_g = self.arena().f32(t * g.n_ff)?;
        let mut ffn_u = self.arena().f32(t * g.n_ff)?;
        let mut ffn = self.arena().f32(t * ne)?;
        let mut li = 0;
        for (l, ly) in self.layers.iter().enumerate() {
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                self.phase("GPU to GPU");
                x = self.hop(&x, self.layer_dev[l])?;
                h = self.arena().f32(t * ne)?;
                ffn_g = self.arena().f32(t * g.n_ff)?;
                ffn_u = self.arena().f32(t * g.n_ff)?;
                ffn = self.arena().f32(t * ne)?;
            }
            let mark = self.arena().mark();
            self.phase(if ly.full.is_some() { "attention" } else { "deltanet" });
            let hq8 = self.norm(&x, &ly.attn_norm, &h, t, ne)?;
            let out = match (&ly.full, &ly.linear) {
                (Some(a), _) => {
                    let mut segs = Vec::with_capacity(ss.len());
                    for (i, s) in ss.iter().enumerate() {
                        let LState::Full { k, v } = &s.layers[l] else { return Err(e(format!("layer {l}: its state and its weights disagree"))) };
                        segs.push(FullSeg { r0: r0s[i], t: rows[i].len(), p0: s.pos, rope: Rope::From(s.pos - s.shift), cap: s.max_ctx, k, v, probe: false });
                    }
                    self.full(a, &segs, &h, hq8.as_ref(), t)?
                }
                (_, Some(a)) => {
                    let mut segs = Vec::with_capacity(ss.len());
                    for (i, s) in ss.iter().enumerate() {
                        let LState::Linear { s: rs, conv } = &s.layers[l] else { return Err(e(format!("layer {l}: its state and its weights disagree"))) };
                        let snap = match sets[i] {
                            Some(k) => snaps.set(k).map(|v| (v[li].0.fp(), v[li].1.fp())),
                            None => None,
                        };
                        segs.push(LinSeg { r0: r0s[i], t: rows[i].len(), s: rs, conv, snap });
                    }
                    li += 1;
                    self.linear(a, &segs, &h, hq8.as_ref(), t)?
                }
                _ => return Err(e(format!("layer {l}: no weights"))),
            };
            self.add(&x, &out, t * ne)?;
            self.phase("ffn");
            let hq8 = self.norm(&x, &ly.post_norm, &h, t, ne)?;
            self.matmul("gate", &ly.gate, &h, hq8.as_ref(), t, &ffn_g)?;
            self.matmul("up", &ly.up, &h, hq8.as_ref(), t, &ffn_u)?;
            let nf = t * g.n_ff;
            // SAFETY: n floats each.
            self.op("swiglu", String::new, 12 * nf, 6 * nf,
                    || unsafe { (self.k.swiglu_clamp)(self.raw(), ffn_g.fp(), ffn_u.fp(), ffn_g.fp(), nf as i64, f32::INFINITY) })?;
            self.matmul("down", &ly.down, &ffn_g, None, t, &ffn)?;
            self.add(&x, &ffn, t * ne)?;
            tap(&format!("l_out-{l}"), &x)?;
            self.arena().rewind(mark);
        }
        let p0s: Vec<usize> = ss.iter().map(|s| s.pos).collect();
        let rope0: Vec<usize> = ss.iter().map(|s| s.pos - s.shift).collect();
        for (i, s) in ss.iter_mut().enumerate() {
            if let Some(k) = sets[i] {
                snaps.of.push((s.id, s.pos, rows[i].len(), k));
            }
            s.pos += rows[i].len();
            s.last_rows = rows[i].len();
        }
        drop(snaps);
        self.phase("head");
        let all = self.arena().f32(t * ne)?;
        let q = self.norm(&x, &self.output_norm, &all, t, ne)?;
        tap("result_norm", &all)?;
        if let (Some(m), true) = (&self.mtp, ss.iter().all(|s| s.mtp.is_some())) {
            self.phase("mtp");
            // the rows kept for a rollback (each set's); the draft block over all the rows in one pass: a conversation's
            // row j pairs its token j with the hidden state before it (its pending one, then this pass's)
            let hs = self.arena().f32(t * ne)?;
            let mut segs = Vec::with_capacity(ss.len());
            for (i, s) in ss.iter().enumerate() {
                let ms = s.mtp.as_ref().ok_or_else(|| e("a session without the draft block's state"))?;
                let (r0, n) = (r0s[i], rows[i].len());
                if let Some(k) = sets[i] {
                    self.vh.copy_within(k * 8 * ne * 4, &all, r0 * ne * 4, n.min(8) * ne * 4)?;
                }
                hs.copy_within(r0 * ne * 4, &ms.pending, 0, ne * 4)?;
                if n > 1 {
                    hs.copy_within((r0 + 1) * ne * 4, &all, r0 * ne * 4, (n - 1) * ne * 4)?;
                }
                segs.push(FullSeg { r0, t: n, p0: p0s[i], rope: Rope::From(rope0[i]), cap: s.max_ctx, k: &ms.k, v: &ms.v, probe: false });
            }
            self.mtp_block(m, &segs, &tokens, &hs)?;
            for (i, s) in ss.iter().enumerate() {
                if let Some(ms) = s.mtp.as_ref() {
                    ms.pending.copy_within(0, &all, (r0s[i] + rows[i].len() - 1) * ne * 4, ne * 4)?;
                }
            }
            self.phase("head");
        }
        let logits = self.arena().f32(t * self.output.rows)?;
        self.matmul("output", &self.output, &all, q.as_ref(), t, &logits)?;
        let lv = logits.to_f32()?;
        if let Some(p) = &self.probe {
            p.settle(t)?;
        }
        let per: Vec<Vec<f32>> = lv.chunks(self.output.rows).map(|c| c.to_vec()).collect();
        Ok(rows.iter().zip(&r0s).map(|(r, &r0)| per[r0..r0 + r.len()].to_vec()).collect())
    }

    /// The next committed token(s). With drafts: [token, drafts] in one pass, each row's token drawn as plain
    /// sampling would draw it, the drafts kept while they match (what is committed is distributed as without drafts)
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let _w = self.work.lock().unwrap();
        self.step_in(s, d, smp, tap)
    }

    /// `step` with the work lock held
    fn step_in(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
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
                        let rb = kv_rb(c, g.heads_kv, s.max_ctx);
                        let mut b = vec![0u8; g.heads_kv * s.pos * rb];
                        for h in 0..g.heads_kv {
                            let n = s.pos * rb;
                            c.read(h * s.max_ctx * rb, &mut b[h * n..(h + 1) * n])?;
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
            let rb = kv_rb(&ms.k, ms.heads, ms.cap);
            let n = s.pos.min(ms.cap) * rb;
            for c in [&ms.k, &ms.v] {
                let mut b = vec![0u8; ms.heads * n];
                for h in 0..ms.heads {
                    c.read(h * ms.cap * rb, &mut b[h * n..(h + 1) * n])?;
                }
                bufs.push(b);
            }
            let mut b = vec![0u8; g.n_embd * 4];
            ms.pending.read(0, &mut b)?;
            bufs.push(b);
        }
        // the rotary offset the pictures left (a last buffer of 8 bytes; a checkpoint without it has none)
        if s.shift > 0 {
            bufs.push((s.shift as u64).to_le_bytes().to_vec());
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
                        let rb = kv_rb(c, g.heads_kv, s.max_ctx);
                        let n = ck.pos * rb;
                        if b.len() != g.heads_kv * n {
                            return Err(e("a checkpoint of another shape"));
                        }
                        for h in 0..g.heads_kv {
                            c.write(h * s.max_ctx * rb, &b[h * n..(h + 1) * n])?;
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
            let rb = kv_rb(&ms.k, ms.heads, ms.cap);
            let n = ck.pos.min(ms.cap) * rb;
            for c in [&ms.k, &ms.v] {
                let b = it.next().ok_or_else(|| e("a checkpoint without the draft block's cache"))?;
                if b.len() != ms.heads * n {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..ms.heads {
                    c.write(h * ms.cap * rb, &b[h * n..(h + 1) * n])?;
                }
            }
            let b = it.next().ok_or_else(|| e("a checkpoint without the draft block's state"))?;
            ms.pending.write(0, b)?;
        }
        s.shift = match it.next() {
            Some(b) if b.len() == 8 => u64::from_le_bytes(b[..8].try_into().unwrap_or_default()) as usize,
            Some(_) => return Err(e("a checkpoint of another shape")),
            None => 0,
        };
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
