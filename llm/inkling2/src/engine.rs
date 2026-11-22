//! Inkling-2-Mini (`inkling2_loop`, llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp; Thinking Machines'
//! Inkling2LoopForConditionalGeneration): 24 decoder layers of 2,304, the middle twelve (6..18) run three times with
//! the same weights - 48 layer applications a token, each with its own caches. A layer:
//!
//! ```text
//!   inputs   = mix(blocks, x)               attention over the depth (block residuals, normed values); 3 queries
//!                                           (q / k / v) on the full layers, one on the windowed ones; layer 0 none
//!   passes 2-3: inputs[j] = (1 - tanh F u) inputs[j] + tanh(G u) msg[j]   u = D [rms(input) | rms(msg)]
//!   q, k, v  = W (rms(inputs) . norm)        k, v through causal short convolutions (4 taps; the full layers add
//!                                           input-dependent taps per head), q / k L2-clipped, v RMS-normed
//!   attn     = softmax(q k / sqrt 128 + r . relbias[distance] ; sink) v      r [16] a head, sink a logit of no value;
//!                                           windowed layers 1,024 back; full layers scale q, r by 1 + 0.1 ln(pos / 8192)
//!   out      = conv(Wo (attn * sigmoid(Wg x)))
//!   x        = out (a block's first layer: the old x joins the blocks) or x + out
//!   x       += conv(MoE(rms(mix(blocks, x)) . norm))   128 experts, 8 by sigmoid + bias, SiTU, NVFP4 with scales
//! ```
//!
//! Passes 2 and 3 of the loop start from the prelude's output again, their inputs mixed with messages from the
//! previous pass (its blocks after the first and its output). The head: mix, norm, the output matrix, soft cap 30, the
//! padded vocabulary dropped. The weights stay in their GGUF blocks (Q8_0, BF16, NVFP4, F32): decode reads them
//! through this engine's own products, a prompt chunk expands each matrix (an expert at a time) to half and multiplies.

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
    /// physical layers
    pub layers: usize,
    pub n_embd: usize,
    pub heads: usize,
    pub heads_kv: usize,
    pub head: usize,
    pub eps: f32,
    /// the sliding window (positions a windowed layer's row sees, itself included)
    pub window: usize,
    /// each physical layer: windowed or full
    pub swa: Vec<bool>,
    pub rel_len: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub conv: usize,
    pub block: usize,
    pub experts: usize,
    pub used: usize,
    pub n_ff: usize,
    pub w_scale: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub softcap: f32,
    /// the loop: [core_start, core_end) run `passes` times
    pub core: (usize, usize),
    pub passes: usize,
    pub rank: usize,
    /// the output matrix's rows, and the tokenizer's (the logits a row)
    pub vocab_pad: usize,
    pub vocab: usize,
    pub ctx: usize,
}

/// The most rows the decode path takes in a pass (a verify pass); prompt rows beyond
pub const MAXV: usize = 16;

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        // a per-layer array's first value, or the value itself
        let first = |k: &str| match f.arch_meta(k) {
            Some(v) if v.as_array().is_some() => v.as_array().and_then(|a| a.first()).and_then(|x| x.as_u64()).map(|x| x as usize),
            Some(v) => v.as_u64().map(|x| x as usize),
            None => None,
        }.ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let layers = u("block_count")?;
        let swa: Vec<bool> = f.arch_meta("attention.sliding_window_pattern").and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) != 0).collect()).ok_or("no attention.sliding_window_pattern")?;
        if swa.len() != layers {
            return Err(format!("attention.sliding_window_pattern has {} flags for {layers} layers", swa.len()));
        }
        let g = Geometry {
            layers,
            n_embd: u("embedding_length")?,
            heads: first("attention.head_count")?,
            heads_kv: first("attention.head_count_kv")?,
            head: u("attention.key_length")?,
            eps: fl("attention.layer_norm_rms_epsilon")?,
            window: u("attention.sliding_window")?,
            swa,
            rel_len: u("attention.relative_length")?,
            rel_extent: u("attention.relative_extent")?,
            qk_clip: fl("attention.qk_clip")?,
            rel_clip: fl("attention.relative_clip")?,
            log_floor: u("attention.log_scale_floor")? as f32,
            log_alpha: fl("attention.log_scale_alpha")?,
            conv: u("shortconv.l_cache")?,
            block: u("attn_res.block_size")?,
            experts: u("expert_count")?,
            used: u("expert_used_count")?,
            n_ff: first("expert_feed_forward_length")?,
            w_scale: fl("expert_weights_scale").unwrap_or(1.0),
            situ_beta: fl("activation.situ_beta")?,
            situ_lb: fl("activation.situ_linear_beta")?,
            softcap: fl("final_logit_softcapping").unwrap_or(0.0),
            core: (u("loop.start")?, u("loop.end")?),
            passes: u("loop.count")?,
            rank: u("loop.gate_rank")?,
            vocab_pad: u("vocab_size")?,
            vocab: u("vocab_size_unpadded")?,
            ctx: u("context_length").unwrap_or(1 << 20),
        };
        if g.head != 128 || g.heads != 3 * g.heads_kv || g.rel_len > 16 || g.conv != 4 {
            return Err(format!("heads of {} ({} / {}), a bias of {}, {} taps: this engine takes heads of 128, 3 query heads a key head, a bias of at most 16 \
                                and 4 taps", g.head, g.heads, g.heads_kv, g.rel_len, g.conv));
        }
        if f.arch_meta("loop.varying").and_then(|v| v.as_u64()).is_some_and(|v| v != 0) {
            return Err("a varying loop: not this engine's".into());
        }
        if g.core.0 == 0 || g.core.0 >= g.core.1 || g.core.1 > g.layers || g.passes < 2 || !g.core.0.is_multiple_of(g.block) || !g.core.1.is_multiple_of(g.block) {
            return Err(format!("a loop of [{}, {}) x {} over {} layers (blocks of {})", g.core.0, g.core.1, g.passes, g.layers, g.block));
        }
        if g.used > 16 || g.experts > 256 {
            return Err(format!("{} of {} experts: at most 16 of 256", g.used, g.experts));
        }
        Ok(g)
    }

    /// Cache slots (layer applications): the physical layers, then each later pass's core
    pub fn slots(&self) -> usize {
        self.layers + (self.passes - 1) * (self.core.1 - self.core.0)
    }

    /// The physical layer of slot `s`
    pub fn phys(&self, s: usize) -> usize {
        if s < self.layers { s } else { self.core.0 + (s - self.layers) % (self.core.1 - self.core.0) }
    }

    /// The order a pass runs: (slot, physical layer, pass)
    pub fn order(&self) -> Vec<(usize, usize, usize)> {
        let (a, b) = self.core;
        let mut v: Vec<(usize, usize, usize)> = (0..a).map(|l| (l, l, 0)).collect();
        for p in 0..self.passes {
            for l in a..b {
                let s = if p == 0 { l } else { self.layers + (p - 1) * (b - a) + (l - a) };
                v.push((s, l, p));
            }
        }
        v.extend((b..self.layers).map(|l| (l, l, 0)));
        v
    }

    /// A slot's ring of keys for a context and passes of `chunk` rows: a windowed layer's holds its window and a pass
    pub fn cap(&self, slot: usize, max_ctx: usize, chunk: usize) -> usize {
        if self.swa[self.phys(slot)] { max_ctx.min(self.window + chunk - 1) } else { max_ctx }
    }

    /// The window a slot's rows see (0: all)
    pub fn win(&self, slot: usize) -> usize {
        if self.swa[self.phys(slot)] { self.window } else { 0 }
    }

    /// A slot's convolution state's channel groups: k, v, the attention's output, the FFN's
    pub fn conv_groups(&self) -> [usize; 4] {
        let kd = self.heads_kv * self.head;
        [kd, kd, self.n_embd, self.n_embd]
    }
}

/// A matrix [rows (outputs), cols (inputs)] in its stored type (F32, BF16, Q8_0, NVFP4)
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

/// A layer's routed experts: [experts][rows][cols] stored blocks, an expert `stride` bytes; scale: a float an expert
pub struct Experts {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub scale: Option<DevBuf>,
    /// the scales on the host (the prompt path's per-row factors)
    pub scale_h: Vec<f32>,
}

pub struct Layer {
    swa: bool,
    /// windowed: one norm for q, k and v; full: one each
    norm: [DevBuf; 3],
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    rel: Mat,
    sink: Mat,
    /// full layers: the k / v convolutions' input-dependent taps
    k_dyn: Option<Mat>,
    v_dyn: Option<Mat>,
    q_clip: DevBuf,
    rel_clip: DevBuf,
    rel_bias: DevBuf,
    /// [C, 4] each: k, v, the attention's output, the FFN's
    convs: [DevBuf; 4],
    /// the depth attention's queries before the attention (1 or 3; none on layer 0) and before the FFN
    attn_res: Option<(DevBuf, usize)>,
    ffn_res: DevBuf,
    ffn_norm: DevBuf,
    router: Mat,
    exp_bias: Option<DevBuf>,
    gu: Experts,
    down: Experts,
}

/// A looped layer's controllers: the message queries (q, k, v, FFN) and, a controller each, its down / gain / forget
pub struct Loop {
    msg_q: DevBuf,
    down: [Mat; 4],
    gain: [Mat; 4],
    forget: [Mat; 4],
}

/// A slot's state: its key / value rings ([kv heads][cap][128] half, slot = position % cap), its convolutions' last
/// three inputs (the four groups' [3][C] one after another) and the last decode pass's inputs (a rollback's)
pub struct SlotState {
    kc: DevBuf,
    vc: DevBuf,
    cap: usize,
    conv: DevBuf,
    snap: DevBuf,
}

/// A conversation
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    slots: Vec<SlotState>,
    /// rows of the last pass (a rollback keeps some of them; past MAXV it cannot)
    last_rows: usize,
}

/// A conversation's state in host memory: per slot its keys / values still seen, then its convolution state
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

const CK_MAGIC: &[u8; 6] = b"NSINK1";

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(CK_MAGIC)?;
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
        let mut magic = [0u8; 6];
        r.read_exact(&mut magic)?;
        if &magic != CK_MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an inkling2 checkpoint"));
        }
        let pos = u(r)?;
        let n = u(r)?;
        if n > 1 << 16 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "an inkling2 checkpoint of too many parts"));
        }
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

/// A pass's result: its rows' logits, or their tokens picked on the GPU
enum Out {
    Logits(Vec<Vec<f32>>),
    Ids(Vec<u32>),
}

/// Generation from a fed prompt: the logits to draw the next token from, or the committed token not fed yet
pub struct Decoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn new(logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder { logits: Some(logits), next: None, drafted: 0, accepted: 0 }
    }
    pub fn after(next: u32, _draft: bool) -> Decoder {
        Decoder { logits: None, next: Some(next), drafted: 0, accepted: 0 }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
}

pub struct Inkling2<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    gpu: Arc<Gpu>,
    arena: Arena,
    /// the largest matrix (or expert) expanded to half (a prompt chunk's products)
    w16: DevBuf,
    k: &'static ffi::Api,
    /// the silo's decode kernels (kernels/llm/inkling2/silo), None: this engine's own throughout
    silo: Option<&'static ffi::Silo>,
    layers: Vec<Layer>,
    /// the looped layers' controllers (core_start..core_end)
    loops: Vec<Loop>,
    /// ones: the RMS norms without a weight
    tok_norm: DevBuf,
    output_norm: DevBuf,
    output_res: DevBuf,
    output: Mat,
    /// the token embedding's rows stay in the file (read a row a token)
    embd: Tensor,
    chunk: usize,
    work: Mutex<()>,
    /// NS_INK2_PROFILE=1: each kernel's time and calls
    prof: Option<Mutex<std::collections::BTreeMap<&'static str, (f64, u64)>>>,
    load_s: f64,
    load_bytes: u64,
}

/// NS_INK2_CHUNK when set; otherwise chosen at load
fn chunk_setting() -> Option<usize> {
    std::env::var("NS_INK2_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v > MAXV)
}

/// A prompt chunk's work memory for `chunk` rows (`ab` refuses to go past it): the pass's persistent rows (the
/// embeddings, the stream's two buffers, the blocks the loop keeps, the previous pass's bank and output, the messages:
/// 16 rows of C), then the larger of a layer's two stages - the attention's (its mixed and controlled inputs, the norms,
/// the projections, a controller's or a product's transient work) and the FFN's (the mix, the controller, the norm, the
/// experts' grouped rows) - and the head's
fn arena_bytes(g: &Geometry, chunk: usize) -> usize {
    let (c, qd, kd) = (g.n_embd, g.heads * g.head, g.heads_kv * g.head);
    let row = |floats: usize| floats * 4;
    let persistent = row(16 * c) + c * 2;
    let ctrl = row(2 * c + 2 * g.rank + 2 * c) + c * 4;
    let attn = row(9 * c + 3 * qd + 4 * kd + 2 * g.heads * g.rel_len + g.heads + 2 * 4 * g.heads_kv + 2 * c) + ctrl.max(qd * 2);
    let rows = g.used;
    let ffn = row(3 * c) + ctrl.max(row(g.experts + 2 * g.used) + rows * (c * 2 + row(2 * g.n_ff) + g.n_ff * 2 + row(c)) + rows * 16) + row(c);
    let per_row = persistent + attn.max(ffn) + row(c);
    // a quarter more for the arena's 256-byte alignment and what the count above leaves out
    chunk * per_row * 5 / 4 + MAXV * g.vocab_pad * 4 * 3 + (64 << 20)
}

/// The bytes of a session's caches for `max_ctx` and passes of `chunk` rows
fn session_bytes(g: &Geometry, max_ctx: usize, chunk: usize) -> usize {
    let conv: usize = g.conv_groups().iter().sum::<usize>() * 4;
    (0..g.slots()).map(|s| 2 * g.heads_kv * g.cap(s, max_ctx, chunk) * g.head * 2 + conv * (3 + 3 + MAXV)).sum()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

impl<'g> Inkling2<'g> {
    /// Loads the model onto one GPU (the last of `gpus`: the server orders them weakest first). `kv`: the context the
    /// sessions reserve (tokens in all; 0: 32K).
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), _draft: bool, log: &mut dyn FnMut(String)) -> Result<Inkling2<'g>> {
        let t0 = Instant::now();
        let gpu = gpus.last().ok_or_else(|| e("no GPU"))?.clone();
        if gpus.len() > 1 {
            log(format!("inkling2: one GPU - {} ({} more given, not used)", gpu.name, gpus.len() - 1));
        }
        let g = Geometry::read(f).map_err(e)?;
        let k = ffi::api()?;
        let silo = ffi::silo()?;
        log(match silo {
            Some(s) => format!("inkling2: decode kernels from its silo {}", s.path.display()),
            None => "inkling2: no silo - its own kernels throughout".into(),
        });
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        let weights: u64 = f.tensors.iter().filter(|t| t.name != "token_embd.weight").map(|t| t.bytes).sum();
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        let ctx = if kv.0 > 0 { kv.0 } else { 32768 };
        let margin: u64 = 1536 << 20;
        let w16_bytes = [g.heads * g.head * g.n_embd, 2 * g.n_ff * g.n_embd].into_iter().max().unwrap_or(0) * 2;
        let tries: Vec<usize> = chunk_setting().map_or_else(|| vec![2048, 1024, 512], |c| vec![c]);
        let mut chunk = 0;
        let mut need = 0u64;
        for &c in &tries {
            need = weights + w16_bytes as u64 + arena_bytes(&g, c) as u64 + session_bytes(&g, ctx, c) as u64 + margin;
            if need <= free {
                chunk = c;
                break;
            }
        }
        if chunk == 0 {
            return Err(Error::no_room(format!("inkling2: {:.1} GiB of weights, work memory and a {ctx}-token session need {:.1} GiB on {}; {:.1} GiB are free",
                                              gib(weights), gib(need), gpu.name, gib(free))));
        }
        let loaded = std::cell::Cell::new(0u64);
        let read = |name: &str| -> Result<Mat> { mat(f, &gpu, name, &loaded) };
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let swa = g.swa[l];
            let norm = if swa {
                let a = read(&n("attn_norm"))?.buf;
                [a.view(0, a.len)?, a.view(0, a.len)?, a]
            } else {
                [read(&n("attn_q_inp_norm"))?.buf, read(&n("attn_k_inp_norm"))?.buf, read(&n("attn_v_inp_norm"))?.buf]
            };
            let attn_res = match f.tensor(&n("attn_res_query")) {
                Some(t) => {
                    let nq = if t.shape.len() > 1 { t.shape[0] as usize } else { 1 };
                    Some((read(&n("attn_res_query"))?.buf, nq))
                }
                None if l == 0 => None,
                None => return Err(e(format!("no {}", n("attn_res_query")))),
            };
            let exp = |which: &str, loaded: &std::cell::Cell<u64>| -> Result<Experts> {
                let name = format!("blk.{l}.ffn_{which}_exps.weight");
                let m = mat(f, &gpu, &name, loaded)?;
                let t = f.tensor(&name).ok_or_else(|| e(format!("no {name}")))?;
                if t.shape.len() != 3 || t.shape[0] as usize != g.experts {
                    return Err(e(format!("{name}: {:?}, not [{} experts, rows, cols]", t.shape, g.experts)));
                }
                let (rows, cols) = (t.shape[1] as usize, t.shape[2] as usize);
                let stride = m.buf.len / g.experts;
                let (scale, scale_h) = match f.tensor(&format!("blk.{l}.ffn_{which}_exps.scale")) {
                    Some(st) => {
                        let raw = f.read(st).map_err(|x| e(x.0))?;
                        let h: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        (Some(DevBuf::from_f32(&gpu, &h)?), h)
                    }
                    None => (None, vec![1.0; g.experts]),
                };
                Ok(Experts { buf: m.buf, ty: m.ty, rows, cols, stride, scale, scale_h })
            };
            let gu = exp("gate_up", &loaded)?;
            let down = exp("down", &loaded)?;
            let read = |name: &str| -> Result<Mat> { mat(f, &gpu, name, &loaded) };
            let ly = Layer {
                swa,
                norm,
                q: read(&n("attn_q"))?,
                k: read(&n("attn_k"))?,
                v: read(&n("attn_v"))?,
                o: read(&n("attn_output"))?,
                gate: read(&n("attn_gate"))?,
                rel: read(&n("attn_rel"))?,
                sink: read(&n("attn_sink_proj"))?,
                k_dyn: if swa { None } else { Some(read(&n("attn_k_conv_proj"))?) },
                v_dyn: if swa { None } else { Some(read(&n("attn_v_conv_proj"))?) },
                q_clip: read(&n("attn_q_clip"))?.buf,
                rel_clip: read(&n("attn_rel_clip"))?.buf,
                rel_bias: read(&n("attn_rel_bias"))?.buf,
                convs: [read(&n("attn_k_conv"))?.buf, read(&n("attn_v_conv"))?.buf, read(&n("attn_post_conv"))?.buf, read(&n("ffn_post_conv"))?.buf],
                attn_res,
                ffn_res: read(&n("ffn_res_query"))?.buf,
                ffn_norm: read(&n("ffn_norm"))?.buf,
                router: read(&n("ffn_gate_inp"))?,
                exp_bias: match f.tensor(&format!("blk.{l}.exp_probs_b.bias")) {
                    Some(_) => Some(read(&format!("blk.{l}.exp_probs_b.bias"))?.buf),
                    None => None,
                },
                gu,
                down,
            };
            if ly.gu.rows != 2 * g.n_ff || ly.gu.cols != g.n_embd || ly.down.rows != g.n_embd || ly.down.cols != g.n_ff {
                return Err(e(format!("layer {l}'s experts: gate|up {}x{}, down {}x{}", ly.gu.rows, ly.gu.cols, ly.down.rows, ly.down.cols)));
            }
            layers.push(ly);
            if l % 6 == 5 {
                log(format!("inkling2: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, gib(loaded.get()), t0.elapsed().as_secs_f64()));
            }
        }
        let read = |name: &str| -> Result<Mat> { mat(f, &gpu, name, &loaded) };
        let mut loops = Vec::new();
        for l in g.core.0..g.core.1 {
            let n = |s: &str| format!("blk.{l}.loop_{s}.weight");
            let ctrl = ["q", "k", "v", "ffn"];
            let part = |what: &str| -> Result<[Mat; 4]> {
                let v: Vec<Mat> = ctrl.iter().map(|c| read(&n(&format!("{c}_ctrl_{what}")))).collect::<Result<_>>()?;
                <[Mat; 4]>::try_from(v).map_err(|_| e("four controllers"))
            };
            let down = part("down")?;
            let gain = part("gain")?;
            let forget = part("forget")?;
            loops.push(Loop { msg_q: read(&n("msg_query"))?.buf, down, gain, forget });
        }
        let tok_norm = read("token_embd_norm.weight")?.buf;
        let output_norm = read("output_norm.weight")?.buf;
        let output_res = read("output_res_query.weight")?.buf;
        let output = read(if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" })?;
        if output.rows != g.vocab_pad || g.vocab > g.vocab_pad {
            return Err(e(format!("the output matrix has {} rows for a vocabulary of {} (padded {})", output.rows, g.vocab, g.vocab_pad)));
        }
        gpu.sync()?;
        let arena = Arena::new(&gpu, arena_bytes(&g, chunk))?;
        let w16 = DevBuf::new(&gpu, w16_bytes)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("inkling2: {} layers ({} applications a token: [{}, {}) x {}), {:.1} GiB on {} in {load_s:.0} s; prompt chunks of {chunk}", g.layers,
                    g.order().len(), g.core.0, g.core.1, g.passes, gib(loaded.get()), gpu.name));
        Ok(Inkling2 { g, file: f, gpu, arena, w16, k, silo, layers, loops, tok_norm, output_norm, output_res, output, embd, chunk, work: Mutex::new(()),
                      prof: std::env::var("NS_INK2_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(Default::default())), load_s, load_bytes: loaded.get() })
    }

    /// A kernel call, its return code checked (NS_INK2_PROFILE=1: timed, the queue drained before and after)
    fn op(&self, name: &'static str, f: impl FnOnce() -> c_int) -> Result<()> {
        let Some(p) = &self.prof else { return ffi::check(f(), name) };
        self.gpu.sync()?;
        let t0 = Instant::now();
        ffi::check(f(), name)?;
        self.gpu.sync()?;
        let mut m = p.lock().unwrap();
        let ent = m.entry(name).or_insert((0.0, 0));
        ent.0 += t0.elapsed().as_secs_f64();
        ent.1 += 1;
        Ok(())
    }

    /// The profile's lines (`report`): each kernel's time in all and a token
    pub fn profile_lines(&self, tokens: usize) -> Vec<String> {
        let Some(p) = &self.prof else { return Vec::new() };
        let m = p.lock().unwrap();
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        v.iter().map(|(n, (s, c))| format!("[profile {n:<16} {s:>8.3} s {c:>8} calls {:>8.3} ms/token]", s * 1000.0 / (tokens + 1) as f64)).collect()
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
    pub fn has_draft(&self) -> bool {
        false
    }
    pub fn max_verify(&self) -> usize {
        1
    }

    fn raw(&self) -> *mut std::ffi::c_void {
        self.gpu.raw()
    }

    fn f32(&self, n: usize) -> Result<DevBuf> {
        self.ab(n * 4)
    }

    /// n bytes of the arena - never an allocation of its own past its end: such a buffer would be freed when it goes
    /// out of scope while kernels queued on it have not run (a page fault on the GPU)
    fn ab(&self, n: usize) -> Result<DevBuf> {
        let at = self.arena.mark().next_multiple_of(256);
        if at + n > self.arena.len() {
            return Err(e(format!("inkling2: the work memory ({:.0} MiB) is short of {:.1} MiB more for this pass (NS_INK2_CHUNK smaller)",
                                 self.arena.len() as f64 / (1 << 20) as f64, (at + n - self.arena.len()) as f64 / (1 << 20) as f64)));
        }
        self.arena.bytes(n)
    }

    /// y [t, w.rows] = x [t, w.cols] . W^T: decode rows through the stored blocks, prompt rows through half
    fn matmul(&self, w: &Mat, x: &DevBuf, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        let ty = w.ty.code() as c_int;
        if t <= MAXV {
            // SAFETY: the matrix's blocks, x [t, cols], y [t, rows].
            return self.op("mv", || unsafe { (k.mv)(self.raw(), ty, w.buf.ptr(), x.fp(), y.fp(), kk, n, t as i64) });
        }
        if w.ty == GType::F32 {
            // SAFETY: x [t, cols], the matrix [rows, cols] float32, y [t, rows].
            return self.op("gemm", || unsafe { (k.gemm)(self.raw(), t as i64, n, kk, x.fp(), kk, w.buf.fp(), y.fp(), n, 0) });
        }
        let elems = w.rows * w.cols;
        if elems * 2 > self.w16.len {
            return Err(e(format!("a {}x{} matrix past the half buffer", w.rows, w.cols)));
        }
        // SAFETY: the blocks -> rows x cols halfs in w16; x [t, cols] -> half; y [t, rows] = xh . w16^T.
        self.op("expand", || unsafe { (k.dequant_f16)(self.raw(), ty, w.buf.ptr(), elems as i64, self.w16.ptr().cast()) })?;
        // x in half only for this product (the arena's later work reuses it: the queue is in order)
        let mark = self.arena.mark();
        let xh = self.ab(t * w.cols * 2)?;
        self.op("to half", || unsafe { (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (t * w.cols) as i64) })?;
        self.op("gemm f16", || unsafe { (k.gemm_f16)(self.raw(), t as i64, n, kk, xh.ptr().cast(), kk, self.w16.ptr().cast(), y.fp(), n, 0) })?;
        self.arena.rewind(mark);
        Ok(())
    }

    /// x [t, w.cols] . W^T into a new buffer
    fn mm(&self, w: &Mat, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let y = self.f32(t * w.rows)?;
        self.matmul(w, x, t, &y)?;
        Ok(y)
    }

    fn rms(&self, x: &DevBuf, w: &DevBuf, t: usize) -> Result<DevBuf> {
        let c = self.g.n_embd;
        let y = self.f32(t * c)?;
        // SAFETY: x and y [t, c], w c floats.
        self.op("rms norm", || unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), t as i64, c as i64, self.g.eps) })?;
        Ok(y)
    }

    /// The silo when decode rows use it
    fn dec(&self, t: usize) -> Option<&'static ffi::Silo> {
        self.silo.filter(|_| t <= MAXV)
    }

    /// x [t, n] quantized for the silo's decode products (None: prompt rows, or no silo)
    fn xq(&self, x: &DevBuf, n: usize, t: usize) -> Result<Option<DevBuf>> {
        let Some(s) = self.dec(t) else { return Ok(None) };
        // SAFETY: arithmetic only; then x [t, n] -> its quants and scales.
        let q = self.ab(unsafe { (s.xq_bytes)(t as i64, n as i64) })?;
        self.op("xq", || unsafe { (s.xq)(self.raw(), x.fp(), q.ptr(), t as i64, n as i64) })?;
        Ok(Some(q))
    }

    /// The RMS norm (weight w) of x [t, C], and for decode rows with the silo its quantized form too (one launch)
    fn rms_q(&self, x: &DevBuf, w: &DevBuf, t: usize) -> Result<(DevBuf, Option<DevBuf>)> {
        let Some(s) = self.dec(t) else { return Ok((self.rms(x, w, t)?, None)) };
        let c = self.g.n_embd;
        let y = self.f32(t * c)?;
        // SAFETY: arithmetic only; then x [t, c] normed into y and its quants.
        let q = self.ab(unsafe { (s.xq_bytes)(t as i64, c as i64) })?;
        self.op("rms xq", || unsafe { (s.rms_xq)(self.raw(), x.fp(), w.fp(), y.fp(), q.ptr(), t as i64, c as i64, self.g.eps) })?;
        Ok((y, Some(q)))
    }

    /// Several products of decode rows in one launch when the silo takes them all (each its matrix, input and quantized
    /// input), else each alone
    fn mm_many(&self, jobs: &[(&Mat, &DevBuf, Option<&DevBuf>)], t: usize) -> Result<Vec<DevBuf>> {
        if let Some(s) = self.dec(t) {
            // SAFETY: arithmetic only.
            let all = jobs.len() <= 8 && jobs.iter().all(|(w, _, q)| q.is_some() && unsafe { (s.mv_supported)(w.ty.code() as c_int, w.cols as i64) } != 0);
            if all {
                let ys: Vec<DevBuf> = jobs.iter().map(|(w, _, _)| self.f32(t * w.rows)).collect::<Result<_>>()?;
                let ty: Vec<c_int> = jobs.iter().map(|(w, _, _)| w.ty.code() as c_int).collect();
                let wp: Vec<*const std::ffi::c_void> = jobs.iter().map(|(w, _, _)| w.buf.ptr() as *const _).collect();
                let qp: Vec<*const std::ffi::c_void> = jobs.iter().map(|(_, _, q)| q.map_or(std::ptr::null(), |q| q.ptr() as *const _)).collect();
                let yp: Vec<*mut f32> = ys.iter().map(|y| y.fp()).collect();
                let rows: Vec<i64> = jobs.iter().map(|(w, _, _)| w.rows as i64).collect();
                let cols: Vec<i64> = jobs.iter().map(|(w, _, _)| w.cols as i64).collect();
                // SAFETY: each job's matrix, quantized input [t, cols] and output [t, rows], as the arrays say.
                self.op("mv many silo", || unsafe {
                    (s.mv_many)(self.raw(), jobs.len() as c_int, ty.as_ptr(), wp.as_ptr(), qp.as_ptr(), yp.as_ptr(), rows.as_ptr(), cols.as_ptr(), t as i64)
                })?;
                return Ok(ys);
            }
        }
        jobs.iter().map(|(w, x, q)| self.mmq(w, x, *q, t)).collect()
    }

    /// x [t, w.cols] . W^T: through the silo's products when x is quantized for them (xq) and they take the type
    fn mmq(&self, w: &Mat, x: &DevBuf, xq: Option<&DevBuf>, t: usize) -> Result<DevBuf> {
        if let (Some(s), Some(q)) = (self.dec(t), xq) {
            // SAFETY: arithmetic only.
            if unsafe { (s.mv_supported)(w.ty.code() as c_int, w.cols as i64) } != 0 {
                let y = self.f32(t * w.rows)?;
                // SAFETY: the matrix's blocks, x's quants, y [t, rows].
                self.op("mv silo", || unsafe { (s.mv)(self.raw(), w.ty.code() as c_int, w.buf.ptr(), q.ptr(), y.fp(), w.cols as i64, w.rows as i64, t as i64) })?;
                return Ok(y);
            }
        }
        self.mm(w, x, t)
    }

    /// The depth attention: `nq` mixes [nq, t, C] of the states (at most 8) by `query`
    fn mix(&self, states: &[&DevBuf], query: &DevBuf, nq: usize, t: usize) -> Result<DevBuf> {
        let c = self.g.n_embd;
        let out = self.f32(nq * t * c)?;
        let ptrs: Vec<*const f32> = states.iter().map(|s| s.fp() as *const f32).collect();
        // SAFETY: each state [t, c] on this GPU, query [nq, c], out [nq, t, c].
        self.op("mix", || unsafe { (self.k.mix)(self.raw(), ptrs.as_ptr(), ptrs.len() as i64, query.fp(), nq as i64, out.fp(), t as i64, c as i64, 1e-6) })?;
        Ok(out)
    }

    /// A loop controller on x [t, C] with its message
    fn control(&self, lp: &Loop, which: usize, x: &DevBuf, msg: &DevBuf, t: usize) -> Result<DevBuf> {
        let c = self.g.n_embd;
        let out = self.f32(t * c)?;
        // its work rewound after (the queue is in order)
        let mark = self.arena.mark();
        let z = self.f32(t * 2 * c)?;
        // SAFETY: x, msg [t, c]; z [t, 2c].
        self.op("ctrl z", || unsafe { (self.k.ctrl_z)(self.raw(), x.fp(), msg.fp(), z.fp(), t as i64, c as i64, 1e-6) })?;
        let u = self.mm(&lp.down[which], &z, t)?;
        let fo = self.mm(&lp.forget[which], &u, t)?;
        let gn = self.mm(&lp.gain[which], &u, t)?;
        // SAFETY: all [t, c].
        self.op("ctrl apply", || unsafe { (self.k.ctrl_apply)(self.raw(), x.fp(), msg.fp(), fo.fp(), gn.fp(), out.fp(), t as i64, c as i64) })?;
        self.arena.rewind(mark);
        Ok(out)
    }

    /// A short convolution of x [t, ch] with slot `ss`'s state group `grp` (dyn: [t, heads, 4] taps)
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, ss: &SlotState, grp: usize, w: &DevBuf, x: &DevBuf, dynt: Option<&DevBuf>, t: usize) -> Result<DevBuf> {
        let groups = self.g.conv_groups();
        let ch = groups[grp];
        let off: usize = groups[..grp].iter().sum();
        let state = ss.conv.view(3 * off * 4, 3 * ch * 4)?;
        let snap = if t <= MAXV { Some(ss.snap.view((3 + MAXV) * off * 4, (3 + t) * ch * 4)?) } else { None };
        let out = self.f32(t * ch)?;
        let taps = if dynt.is_some() { 2 * self.g.conv + 1 } else { self.g.conv + 1 };
        let inv = 1.0 / (taps as f32).sqrt();
        let heads = self.g.heads_kv as i64;
        // SAFETY: x / out [t, ch], the state [3, ch], w [ch, 4], dyn [t, heads, 4], snap [3 + t, ch].
        self.op("conv", || unsafe {
            (self.k.conv)(self.raw(), x.fp(), state.fp(), w.fp(), dynt.map_or(std::ptr::null(), |d| d.fp()), out.fp(),
                          snap.as_ref().map_or(std::ptr::null_mut(), |s| s.fp()), t as i64, ch as i64, heads, inv)
        })?;
        Ok(out)
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
        let src = self.ab(raw.len())?;
        src.write_async(0, &raw)?;
        let x = self.f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32.
        self.op("dequant", || unsafe { (self.k.dequant)(self.raw(), self.embd.ty.code() as c_int, src.ptr(), (tokens.len() * g.n_embd) as i64, x.fp()) })?;
        Ok(x)
    }

    /// The routed experts on h [t, C] (normed): their weighted sum [t, C]
    fn moe(&self, ly: &Layer, h: &DevBuf, hq: Option<&DevBuf>, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (c, kk, ff) = (g.n_embd, g.used, g.n_ff);
        let logits = self.mm(&ly.router, h, t)?;
        let ids = self.ab(t * kk * 4)?;
        let wts = self.f32(t * kk)?;
        // SAFETY: logits [t, E] -> ids, weights [t, K].
        self.op("route", || unsafe {
            (k.route)(self.raw(), logits.fp(), ly.exp_bias.as_ref().map_or(std::ptr::null(), |b| b.fp()), ids.ptr().cast(), wts.fp(), t as i64,
                      g.experts as i64, kk as i64, g.w_scale)
        })?;
        let out = self.f32(t * c)?;
        // the experts' work rewound after (the queue is in order)
        let mark = self.arena.mark();
        let (gu, dn) = (&ly.gu, &ly.down);
        if let (Some(s), Some(q)) = (self.dec(t), hq) {
            // SAFETY: arithmetic only.
            let takes = unsafe { (s.mv_supported)(gu.ty.code() as c_int, gu.cols as i64) != 0 && (s.mv_supported)(dn.ty.code() as c_int, dn.cols as i64) != 0 };
            if takes {
                // SAFETY: arithmetic only; then gate | up and SiTU into the act's quants, the down products weighted into out.
                let aq = self.ab(unsafe { (s.xq_bytes)((t * kk) as i64, ff as i64) })?;
                let sc = |e: &Experts| e.scale.as_ref().map_or(std::ptr::null(), |s| s.fp() as *const f32);
                self.op("experts gu silo", || unsafe {
                    (s.moe_gu)(self.raw(), gu.ty.code() as c_int, gu.buf.ptr(), gu.stride, sc(gu), ids.ptr().cast(), q.ptr(), aq.ptr(), t as i64, kk as i64,
                               gu.cols as i64, ff as i64, g.situ_beta, g.situ_lb)
                })?;
                self.op("experts down silo", || unsafe {
                    (s.moe_down)(self.raw(), dn.ty.code() as c_int, dn.buf.ptr(), dn.stride, sc(dn), ids.ptr().cast(), wts.fp(), aq.ptr(), out.fp(), t as i64,
                                 kk as i64, ff as i64, dn.rows as i64)
                })?;
                return Ok(out);
            }
        }
        if t <= MAXV {
            let y = self.f32(t * kk * gu.rows)?;
            let act = self.f32(t * kk * ff)?;
            let d = self.f32(t * kk * c)?;
            // SAFETY: each (row, expert) pair's product from its expert's blocks; SiTU; the down products; the sum.
            unsafe {
                self.op("experts gu", || (k.moe_mv)(self.raw(), gu.ty.code() as c_int, gu.buf.ptr(), gu.stride, gu.scale.as_ref().map_or(std::ptr::null(), |s| s.fp()),
                                                    ids.ptr().cast(), h.fp(), 0, y.fp(), t as i64, kk as i64, gu.cols as i64, gu.rows as i64))?;
                self.op("situ", || (k.situ)(self.raw(), y.fp(), std::ptr::null(), act.fp(), std::ptr::null_mut(), (t * kk) as i64, ff as i64, g.situ_beta, g.situ_lb))?;
                self.op("experts down", || (k.moe_mv)(self.raw(), dn.ty.code() as c_int, dn.buf.ptr(), dn.stride, dn.scale.as_ref().map_or(std::ptr::null(), |s| s.fp()),
                                                      ids.ptr().cast(), act.fp(), 1, d.fp(), t as i64, kk as i64, dn.cols as i64, dn.rows as i64))?;
                self.op("combine", || (k.combine)(self.raw(), d.fp(), wts.fp(), out.fp(), t as i64, kk as i64, c as i64))?;
            }
            return Ok(out);
        }
        // a prompt chunk: the rows grouped by expert on the host, each expert's rows one product
        let mut hid = vec![0u8; t * kk * 4];
        ids.read(0, &mut hid)?;
        let mut hw = vec![0u8; t * kk * 4];
        wts.read(0, &mut hw)?;
        let id: Vec<usize> = hid.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).collect();
        let wv: Vec<f32> = hw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let mut by: Vec<Vec<usize>> = vec![Vec::new(); g.experts];
        for (p, &x) in id.iter().enumerate() {
            if x >= g.experts {
                return Err(e(format!("the router chose expert {x}")));
            }
            by[x].push(p);
        }
        let m = t * kk;
        let mut tok = Vec::with_capacity(m);
        let mut rs = Vec::with_capacity(m);
        let mut at = vec![0usize; m];
        let mut runs = Vec::new();
        for (x, ps) in by.iter().enumerate() {
            if ps.is_empty() {
                continue;
            }
            runs.push((x, tok.len(), ps.len()));
            for &p in ps {
                at[p] = tok.len();
                tok.push((p / kk) as i32);
                rs.push(gu.scale_h[x]);
            }
        }
        // each token's entries: its K rows, weighted (the down product's scale folded in)
        let mut t_ptr = Vec::with_capacity(t + 1);
        let mut ent = Vec::with_capacity(m);
        let mut cw = Vec::with_capacity(m);
        for r in 0..t {
            t_ptr.push(ent.len() as i32);
            for j in 0..kk {
                let p = r * kk + j;
                ent.push(at[p] as i32);
                cw.push(wv[p] * dn.scale_h[id[p]]);
            }
        }
        t_ptr.push(ent.len() as i32);
        let up_i = |v: &[i32]| -> Result<DevBuf> {
            let b = self.ab(v.len() * 4)?;
            b.write_async(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(b)
        };
        let up_f = |v: &[f32]| -> Result<DevBuf> {
            let b = self.ab(v.len() * 4)?;
            b.write_async(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(b)
        };
        let (tok_d, rs_d, tp_d, ent_d, cw_d) = (up_i(&tok)?, up_f(&rs)?, up_i(&t_ptr)?, up_i(&ent)?, up_f(&cw)?);
        let xh = self.ab(m * c * 2)?;
        let y = self.f32(m * gu.rows)?;
        let act = self.ab(m * ff * 2)?;
        let d = self.f32(m * c)?;
        // SAFETY: rows of h gathered to half, [m, C]
        self.op("gather", || unsafe { (k.gather_f16)(self.raw(), h.fp(), tok_d.ptr().cast(), xh.ptr().cast(), m as i64, c as i64) })?;
        let (gn, gk) = (gu.rows as i64, gu.cols as i64);
        for &(x, r0, n) in &runs {
            let w = gu.buf.view(x * gu.stride, gu.stride)?;
            // SAFETY: expert x's gate|up -> half in w16; its rows' product.
            self.op("expand", || unsafe { (k.dequant_f16)(self.raw(), gu.ty.code() as c_int, w.ptr(), gn * gk, self.w16.ptr().cast()) })?;
            self.op("gemm f16", || unsafe {
                (k.gemm_f16)(self.raw(), n as i64, gn, gk, xh.ptr().cast::<u16>().add(r0 * c), gk, self.w16.ptr().cast(), y.fp().add(r0 * gu.rows), gn, 0)
            })?;
        }
        // SAFETY: [m, 2F] scaled by its expert's factor -> SiTU in half [m, F]
        self.op("situ", || unsafe { (k.situ)(self.raw(), y.fp(), rs_d.fp(), std::ptr::null_mut(), act.ptr().cast(), m as i64, ff as i64, g.situ_beta, g.situ_lb) })?;
        let (dn_n, dn_k) = (dn.rows as i64, dn.cols as i64);
        for &(x, r0, n) in &runs {
            let w = dn.buf.view(x * dn.stride, dn.stride)?;
            // SAFETY: expert x's down -> half in w16; its rows' product.
            self.op("expand", || unsafe { (k.dequant_f16)(self.raw(), dn.ty.code() as c_int, w.ptr(), dn_n * dn_k, self.w16.ptr().cast()) })?;
            self.op("gemm f16", || unsafe {
                (k.gemm_f16)(self.raw(), n as i64, dn_n, dn_k, act.ptr().cast::<u16>().add(r0 * ff), dn_k, self.w16.ptr().cast(), d.fp().add(r0 * c), dn_n, 0)
            })?;
        }
        out.fill(0)?;
        // SAFETY: each token's rows weighted into out [t, C]
        self.op("combine", || unsafe { (k.combine_rows)(self.raw(), d.fp(), tp_d.ptr().cast(), ent_d.ptr().cast(), cw_d.fp(), out.fp(), t as i64, c as i64) })?;
        self.arena.rewind(mark);
        Ok(out)
    }

    /// One layer application (slot `slot` of physical layer `l`, at `ip` in its pass) on the stream `cur`, its result
    /// into `dst`; `blocks` the depth attention's earlier states (a block's first layer adds `cur`); `msgs` [4, t, C]
    /// the loop's messages (passes 2 and 3)
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, ss: &SlotState, slot: usize, l: usize, cur: &DevBuf, dst: &DevBuf, blocks: &mut Vec<DevBuf>, msgs: Option<(&Loop, &DevBuf)>, t: usize, p0: usize,
             tap: Tap) -> Result<()> {
        let (g, k) = (&self.g, self.k);
        let ly = &self.layers[l];
        let c = g.n_embd;
        let tc = t * c * 4;
        let (hq, hk, hd) = (g.heads, g.heads_kv, g.head);
        let mark = self.arena.mark();
        // the attention's inputs: mixed over the depth (q, k, v on a full layer), then the loop's messages
        let mut st: Vec<&DevBuf> = blocks.iter().collect();
        st.push(cur);
        let mut inputs: Vec<DevBuf> = match &ly.attn_res {
            Some((q, nq)) => {
                let mixed = self.mix(&st, q, *nq, t)?;
                (0..3).map(|j| mixed.view(if *nq > 1 { j * tc } else { 0 }, tc)).collect::<Result<_>>()?
            }
            None => (0..3).map(|_| cur.view(0, tc)).collect::<Result<_>>()?,
        };
        if let Some((lp, m)) = msgs {
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = self.control(lp, j, inp, &m.view(j * tc, tc)?, t)?;
            }
        }
        // the normed inputs (and for decode rows with the silo their quantized forms)
        let (normed, nq): (Vec<DevBuf>, Vec<Option<DevBuf>>) = if ly.swa && msgs.is_none() {
            let (n0, q0) = self.rms_q(&inputs[0], &ly.norm[0], t)?;
            let qv = |q: &Option<DevBuf>| -> Result<Option<DevBuf>> { q.as_ref().map(|q| q.view(0, q.len)).transpose() };
            (vec![n0.view(0, tc)?, n0.view(0, tc)?, n0], vec![qv(&q0)?, qv(&q0)?, q0])
        } else {
            let v: Vec<(DevBuf, Option<DevBuf>)> = (0..3).map(|j| self.rms_q(&inputs[j], &ly.norm[j], t)).collect::<Result<_>>()?;
            v.into_iter().unzip()
        };
        let qd = hq * hd;
        let [q, kr, vr, r, gate]: [DevBuf; 5] = self.mm_many(&[(&ly.q, &normed[0], nq[0].as_ref()), (&ly.k, &normed[1], nq[1].as_ref()),
                                                               (&ly.v, &normed[2], nq[2].as_ref()), (&ly.rel, &normed[0], nq[0].as_ref()),
                                                               (&ly.gate, &normed[0], nq[0].as_ref())], t)?
            .try_into().map_err(|_| e("five products"))?;
        let sink = self.mm(&ly.sink, &normed[0], t)?;
        let kdyn = match &ly.k_dyn {
            Some(m) => Some(self.mm(m, &normed[1], t)?),
            None => None,
        };
        let vdyn = match &ly.v_dyn {
            Some(m) => Some(self.mm(m, &normed[2], t)?),
            None => None,
        };
        let kc = self.conv(ss, 0, &ly.convs[0], &kr, kdyn.as_ref(), t)?;
        let vc = self.conv(ss, 1, &ly.convs[1], &vr, vdyn.as_ref(), t)?;
        let alpha = if ly.swa { 0.0 } else { g.log_alpha };
        // SAFETY: the rows [t, heads, dims] in place.
        unsafe {
            self.op("clip", || (k.clip)(self.raw(), q.fp(), ly.q_clip.fp(), t as i64, hq as i64, hd as i64, g.qk_clip, alpha, g.log_floor, p0 as i64))?;
            self.op("clip", || (k.clip)(self.raw(), kc.fp(), std::ptr::null(), t as i64, hk as i64, hd as i64, g.qk_clip, 0.0, g.log_floor, p0 as i64))?;
            self.op("v norm", || (k.rms_rows)(self.raw(), vc.fp(), vc.fp(), (t * hk) as i64, hd as i64, g.eps))?;
            self.op("clip", || (k.clip)(self.raw(), r.fp(), ly.rel_clip.fp(), t as i64, hq as i64, g.rel_len as i64, g.rel_clip, alpha, g.log_floor, p0 as i64))?;
        }
        tap(&format!("Qcur-{slot}"), &q)?;
        tap(&format!("Kcur-{slot}"), &kc)?;
        tap(&format!("Vcur-{slot}"), &vc)?;
        tap(&format!("Rcur-{slot}"), &r)?;
        let att = self.f32(t * qd)?;
        let win = g.win(slot) as i64;
        let extent = if ly.swa { g.window } else { g.rel_extent } as i64;
        // SAFETY: the rows and caches as ink.h names them.
        unsafe {
            self.op("kv store", || (k.kv_store)(self.raw(), kc.fp(), vc.fp(), t as i64, hk as i64, hd as i64, ss.cap as i64, p0 as i64, ss.kc.ptr(), ss.vc.ptr()))?;
            let sc = (k.attn_scratch)(t as i64, hq as i64, hd as i64, p0 as i64, win) as usize;
            let part = if sc > 0 { Some(self.f32(sc)?) } else { None };
            self.op("attention", || (k.attn)(self.raw(), q.fp(), ss.kc.ptr(), ss.vc.ptr(), r.fp(), ly.rel_bias.fp(), extent, g.rel_len as i64, sink.fp(), att.fp(),
                                             t as i64, hq as i64, hk as i64, hd as i64, ss.cap as i64, p0 as i64, win, 1.0 / (hd as f32).sqrt(),
                                             part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
            self.op("gate", || (k.gate)(self.raw(), att.fp(), gate.fp(), (t * qd) as i64))?;
        }
        let aq = self.xq(&att, qd, t)?;
        let o = self.mmq(&ly.o, &att, aq.as_ref(), t)?;
        let o = self.conv(ss, 2, &ly.convs[2], &o, None, t)?;
        tap(&format!("attn_out-{slot}"), &o)?;
        // the first layer of a block starts the stream again (the old one joins the blocks)
        if l.is_multiple_of(g.block) {
            blocks.push(cur.view(0, tc)?);
            dst.copy_within(0, &o, 0, tc)?;
        } else {
            // SAFETY: [t, C] each.
            self.op("add", || unsafe { (k.add)(self.raw(), cur.fp(), o.fp(), dst.fp(), (t * c) as i64) })?;
        }
        tap(&format!("ffn_inp-{slot}"), dst)?;
        // the attention's work is done with (the queue is in order): the FFN's reuses it
        self.arena.rewind(mark);
        let mut st: Vec<&DevBuf> = blocks.iter().collect();
        st.push(dst);
        let mut m = self.mix(&st, &ly.ffn_res, 1, t)?;
        if let Some((lp, msg)) = msgs {
            m = self.control(lp, 3, &m, &msg.view(3 * tc, tc)?, t)?;
        }
        let (h, hq) = self.rms_q(&m, &ly.ffn_norm, t)?;
        let f = self.moe(ly, &h, hq.as_ref(), t)?;
        let f = self.conv(ss, 3, &ly.convs[3], &f, None, t)?;
        tap(&format!("ffn_out-{slot}"), &f)?;
        // SAFETY: [t, C] each.
        self.op("add", || unsafe { (k.add)(self.raw(), dst.fp(), f.fp(), dst.fp(), (t * c) as i64) })?;
        tap(&format!("l_out-{slot}"), dst)?;
        self.arena.rewind(mark);
        Ok(())
    }

    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        match self.pass_out(s, tokens, n_out, None, tap)? {
            Out::Logits(l) => Ok(l),
            Out::Ids(_) => Err(e("a pass's logits")),
        }
    }

    /// A pass's last `n_out` rows: their logits, or (`pick`: a decoder whose logits nothing reads) each one's token
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
        self.arena.reset();
        let p0 = s.pos;
        let c = g.n_embd;
        let tc = t * c * 4;
        let emb = self.embed(tokens)?;
        tap("inp_embd", &emb)?;
        let x0 = self.rms(&emb, &self.tok_norm, t)?;
        // the pass's persistent rows: the stream's two buffers, the prelude's output, the blocks the loop pushes, the
        // previous pass's bank and output, the messages
        let a = self.f32(t * c)?;
        let b = self.f32(t * c)?;
        let pre = self.f32(t * c)?;
        let s12 = self.f32(t * c)?;
        let s12_prev = self.f32(t * c)?;
        let s18 = self.f32(t * c)?;
        let pout = self.f32(t * c)?;
        let msgs = self.f32(4 * t * c)?;
        let (cs, ce) = g.core;
        let mut blocks: Vec<DevBuf> = Vec::new();
        // where the stream is now; the next layer writes the other buffer
        let mut cur: DevBuf = x0.view(0, tc)?;
        let other = |cur: &DevBuf| -> Result<DevBuf> { if cur.ptr() == a.ptr() { b.view(0, tc) } else { a.view(0, tc) } };
        let mut prelude_blocks = 0;
        for (slot, l, p) in g.order() {
            if l == cs {
                if p == 0 {
                    pre.copy_within(0, &cur, 0, tc)?;
                    prelude_blocks = blocks.len();
                } else {
                    // the bank: the previous pass's blocks after its first (its block at the loop's start is the prelude)
                    if blocks.len() != prelude_blocks + 2 {
                        return Err(e(format!("the loop's bank: {} blocks", blocks.len())));
                    }
                    s12_prev.copy_within(0, &s12, 0, tc)?;
                    pout.copy_within(0, &cur, 0, tc)?;
                    blocks.truncate(prelude_blocks);
                }
                cur = pre.view(0, tc)?;
            }
            if l > cs && l % g.block == 0 {
                // a block's first layer: the stream into a buffer of its own (it joins the blocks)
                let keep = if l < ce { &s12 } else { &s18 };
                keep.copy_within(0, &cur, 0, tc)?;
                cur = keep.view(0, tc)?;
            }
            let lp = if p > 0 {
                let lp = &self.loops[l - cs];
                let mk = self.arena.mark();
                let m = self.mix(&[&s12_prev, &pout], &lp.msg_q, 4, t)?;
                msgs.copy_within(0, &m, 0, 4 * tc)?;
                self.arena.rewind(mk);
                Some((lp, &msgs))
            } else {
                None
            };
            let dst = other(&cur)?;
            self.layer(&s.slots[slot], slot, l, &cur, &dst, &mut blocks, lp, t, p0, &mut *tap)?;
            cur = dst;
        }
        s.pos += t;
        s.last_rows = t;
        let n = n_out.clamp(1, t);
        let mut st: Vec<&DevBuf> = blocks.iter().collect();
        st.push(&cur);
        let fin = self.mix(&st, &self.output_res, 1, t)?;
        let last = fin.view((t - n) * c * 4, n * c * 4)?;
        let normed = self.rms(&last, &self.output_norm, n)?;
        tap("result_norm", &normed)?;
        let vp = self.output.rows;
        let v = g.vocab;
        let raw = self.f32(n * vp)?;
        for r0 in (0..n).step_by(MAXV) {
            let rn = (n - r0).min(MAXV);
            let x = normed.view(r0 * c * 4, rn * c * 4)?;
            let xq = self.xq(&x, c, rn)?;
            let y = self.mmq(&self.output, &x, xq.as_ref(), rn)?;
            raw.copy_within(r0 * vp * 4, &y, 0, rn * vp * 4)?;
        }
        let logits = self.f32(n * v)?;
        // SAFETY: n rows of vp -> n rows of v, soft-capped
        self.op("softcap", || unsafe { (self.k.softcap)(self.raw(), raw.fp(), logits.fp(), n as i64, v as i64, vp as i64, g.softcap) })?;
        tap("result_output", &logits)?;
        if let Some(how) = pick {
            let best = self.ab(n * 4)?;
            match how {
                // SAFETY: logits [n, vocab] -> n int32.
                None => self.op("argmax", || unsafe { (self.k.argmax)(self.raw(), logits.fp(), best.ptr().cast(), n as i64, v as i64) })?,
                Some(ds) => {
                    let pos0 = (s.pos - n) as i64;
                    // SAFETY: each row's draw (row i at its position) into n int32.
                    self.op("sample", || unsafe {
                        (self.k.sample)(self.raw(), logits.fp(), best.ptr().cast(), n as i64, v as i64, ds.temperature, ds.top_p, ds.seed, pos0)
                    })?;
                }
            }
            let mut bb = vec![0u8; n * 4];
            best.read(0, &mut bb)?;
            return Ok(Out::Ids(bb.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        }
        let all = logits.to_f32()?;
        Ok(Out::Logits(all.chunks(v).map(|c| c.to_vec()).collect()))
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        // checked against what is free first: past the card the xe driver spills into host memory and can take the
        // machine down
        let need = session_bytes(g, max_ctx, self.chunk);
        let (total, free) = self.gpu.memory()?;
        let free = free.unwrap_or(total);
        if need as u64 + (1536 << 20) > free {
            return Err(Error::no_room(format!("a session of {max_ctx} tokens needs {:.2} GiB on {}; {:.2} GiB are free (1.5 GiB kept spare)", gib(need as u64),
                                              self.gpu.name, gib(free))));
        }
        let conv: usize = g.conv_groups().iter().sum();
        let mut slots = Vec::with_capacity(g.slots());
        for sl in 0..g.slots() {
            let cap = g.cap(sl, max_ctx, self.chunk);
            let n = g.heads_kv * cap * g.head * 2;
            let st = DevBuf::new(&self.gpu, 3 * conv * 4)?;
            st.fill(0)?;
            slots.push(SlotState { kc: DevBuf::new(&self.gpu, n)?, vc: DevBuf::new(&self.gpu, n)?, cap, conv: st, snap: DevBuf::new(&self.gpu, (3 + MAXV) * conv * 4)? });
        }
        self.gpu.sync()?;
        Ok(Session { pos: 0, max_ctx, slots, last_rows: 0 })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        for sl in &s.slots {
            sl.conv.fill(0)?;
        }
        s.pos = 0;
        s.last_rows = 0;
        Ok(())
    }

    /// Positions [from, to) of one ring (`heads` of cap slots) into another (dcap slots), in runs that wrap neither
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

    /// The first position a ring with window `w` still reads at `pos`
    fn kept(w: usize, pos: usize) -> usize {
        if w > 0 { pos.saturating_sub(w - 1) } else { 0 }
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        for (sl, (d, sc)) in dst.slots.iter().zip(&src.slots).enumerate() {
            let from = Self::kept(self.g.win(sl), src.pos);
            self.copy_ring(&d.kc, d.cap, &sc.kc, sc.cap, self.g.heads_kv, from, src.pos)?;
            self.copy_ring(&d.vc, d.cap, &sc.vc, sc.cap, self.g.heads_kv, from, src.pos)?;
            d.conv.copy_within(0, &sc.conv, 0, d.conv.len)?;
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

    /// Back to the first `keep` rows of the last pass: the rings' later slots are written again; the convolutions'
    /// states from the pass's inputs (a decode pass keeps them)
    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep > s.last_rows {
            return Err(e(format!("a rollback to {keep} rows of a pass of {}", s.last_rows)));
        }
        if keep < s.last_rows {
            if s.last_rows > MAXV {
                return Err(e(format!("a rollback inside a prompt pass of {} rows", s.last_rows)));
            }
            self.conv_keep(s, keep)?;
        }
        s.pos -= s.last_rows - keep;
        s.last_rows = keep;
        Ok(())
    }

    /// Every slot's convolution states as after the first `keep` rows of the last decode pass
    fn conv_keep(&self, s: &Session, keep: usize) -> Result<()> {
        let groups = self.g.conv_groups();
        for sl in &s.slots {
            let mut off = 0;
            for &ch in &groups {
                let snap = sl.snap.view((3 + MAXV) * off * 4, (3 + MAXV) * ch * 4)?;
                let st = sl.conv.view(3 * off * 4, 3 * ch * 4)?;
                // SAFETY: snap [3 + rows, ch], the state [3, ch].
                self.op("conv keep", || unsafe { (self.k.conv_keep)(self.raw(), snap.fp(), st.fp(), keep as i64, ch as i64) })?;
                off += ch;
            }
        }
        Ok(())
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    /// The next committed token
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let _w = self.work.lock().unwrap();
        let out = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => smp.sample(&l),
            (None, Some(x)) => {
                // the token from the GPU when nothing reads the logits (greedy: the argmax; sampled: its draw)
                let pick = if smp.greedy() { Some(None) } else { smp.device().map(Some) };
                match self.pass_out(s, &[x], 1, pick, tap)? {
                    Out::Logits(mut l) => smp.sample(&l.pop().unwrap_or_default()),
                    Out::Ids(v) => *v.first().ok_or_else(|| e("no token"))?,
                }
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        d.next = Some(out);
        Ok(vec![out])
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let g = &self.g;
        let row = g.head * 2;
        let mut bufs = Vec::new();
        for (sl, st) in s.slots.iter().enumerate() {
            let from = Self::kept(g.win(sl), s.pos);
            let n = s.pos - from;
            for c in [&st.kc, &st.vc] {
                let mut b = vec![0u8; g.heads_kv * n * row];
                for h in 0..g.heads_kv {
                    let mut p = from;
                    while p < s.pos {
                        let si = p % st.cap;
                        let run = (s.pos - p).min(st.cap - si);
                        let at = (h * n + p - from) * row;
                        c.read((h * st.cap + si) * row, &mut b[at..at + run * row])?;
                        p += run;
                    }
                }
                bufs.push(b);
            }
            let mut cb = vec![0u8; st.conv.len];
            st.conv.read(0, &mut cb)?;
            bufs.push(cb);
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let g = &self.g;
        let row = g.head * 2;
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        if ck.bufs.len() != 3 * s.slots.len() {
            return Err(e("a checkpoint of another shape"));
        }
        let mut it = ck.bufs.iter();
        for (sl, st) in s.slots.iter().enumerate() {
            let from = Self::kept(g.win(sl), ck.pos);
            let n = ck.pos - from;
            for c in [&st.kc, &st.vc] {
                let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                if b.len() != g.heads_kv * n * row {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..g.heads_kv {
                    let mut p = from;
                    while p < ck.pos {
                        let si = p % st.cap;
                        let run = (ck.pos - p).min(st.cap - si);
                        let at = (h * n + p - from) * row;
                        c.write((h * st.cap + si) * row, &b[at..at + run * row])?;
                        p += run;
                    }
                }
            }
            let cb = it.next().ok_or_else(|| e("a short checkpoint"))?;
            if cb.len() != st.conv.len {
                return Err(e("a checkpoint of another shape"));
            }
            st.conv.write(0, cb)?;
        }
        s.pos = ck.pos;
        s.last_rows = 0;
        Ok(())
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        let (total, free) = self.gpu.memory().unwrap_or((0, None));
        vec![GpuInfo { index: self.gpu.index, name: self.gpu.name.clone(), pci: self.gpu.pci.clone(), total, free, layers: (0, self.g.layers as u64), expert_slots: 0,
                       host_slots: 0 }]
    }
}

/// A tensor read onto the GPU in its stored type: [rows (outputs), cols (inputs)] (a vector one row; an expert stack
/// [experts x rows, cols])
pub fn mat(f: &Gguf, gpu: &Arc<Gpu>, name: &str, loaded: &std::cell::Cell<u64>) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    let (rows, cols) = match t.shape.len() {
        1 => (1, t.shape[0] as usize),
        _ => (t.shape[..t.shape.len() - 1].iter().product::<u64>() as usize, t.shape[t.shape.len() - 1] as usize),
    };
    let ok = matches!(t.ty, GType::F32 | GType::BF16 | GType::Q8_0 | GType::NVFP4);
    if !ok {
        return Err(e(format!("{name}: {} - this engine reads F32, BF16, Q8_0 and NVFP4", t.ty.name())));
    }
    if t.shape.len() == 1 && t.ty != GType::F32 {
        return Err(e(format!("{name}: a {} vector (float32 expected)", t.ty.name())));
    }
    let raw = f.read(t).map_err(|x| e(x.0))?;
    loaded.set(loaded.get() + raw.len() as u64);
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    Ok(Mat { buf, ty: t.ty, rows, cols })
}
