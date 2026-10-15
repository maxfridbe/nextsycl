//! Nemotron-H MoE (`nemotron_h_moe`, llama.cpp's src/models/nemotron-h.cpp: Nemotron 3.5 Super 120B-A12B): 88 layers
//! of three kinds, each a pre-norm residual block - Mamba-2 mixers (an input projection to z | x B C | dt, a causal
//! conv + SiLU over x B C, the selective scan, a SiLU(z)-gated group RMS norm, the output projection), latent MoE
//! layers (a sigmoid router with a selection bias picks 22 of 512 experts, weights renormalized and scaled; the token
//! projected to a 1024-wide latent, each expert relu^2(up) then down there, the sum projected back; plus a relu^2
//! shared expert on the full width) and attention layers (GQA, no positions). The weights stay in their GGUF blocks on
//! the GPUs; every product goes through the Q8_1 kernels, a pass at most 8 rows.
//!
//! ```text
//!   x = embed(tokens)
//!   per layer:  x += mixer(attn_norm(x))      mixer: mamba2 | moe | attention
//!   logits = output(output_norm(x))
//! ```
//!
//! The appended MTP block (`nextn_predict_layers`) is not read yet: no drafts.

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

/// What a layer is
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mamba,
    Moe,
    Attn,
}

/// The model's shape, from the file's metadata
#[derive(Clone, Debug)]
pub struct Geometry {
    pub layers: usize,
    pub kinds: Vec<Kind>,
    pub n_embd: usize,
    pub vocab: usize,
    pub ctx: usize,
    pub eps: f32,
    pub heads: usize,
    pub heads_kv: usize,
    pub head: usize,
    pub conv: usize,
    pub state: usize,
    pub groups: usize,
    pub inner: usize,
    pub ssm_heads: usize,
    pub experts: usize,
    pub used: usize,
    pub latent: usize,
    pub n_ff_exp: usize,
    pub n_ff_shared: usize,
    pub w_scale: f32,
    pub w_norm: bool,
    pub nextn: usize,
}

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        let arch = f.architecture();
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {arch}.{k}"));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {arch}.{k}"));
        // a per-layer value: an array, or one value for every layer
        let per = |k: &str, n: usize| -> std::result::Result<Vec<usize>, String> {
            let v = f.arch_meta(k).ok_or_else(|| format!("no {arch}.{k}"))?;
            match v.as_array() {
                Some(a) => Ok(a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()),
                None => Ok(vec![v.as_u64().unwrap_or(0) as usize; n]),
            }
        };
        let nextn = u("nextn_predict_layers").unwrap_or(0);
        let blocks = u("block_count")?;
        let layers = blocks - nextn;
        let kv = per("attention.head_count_kv", blocks)?;
        let ff = per("feed_forward_length", blocks)?;
        if kv.len() < layers || ff.len() < layers {
            return Err(format!("{arch}: per-layer head_count_kv / feed_forward_length shorter than the {layers} layers"));
        }
        // a layer is a mixer when it has neither key heads nor a feed-forward (llama.cpp's is_recr)
        let kinds: Vec<Kind> = (0..layers).map(|l| if kv[l] > 0 { Kind::Attn } else if ff[l] > 0 { Kind::Moe } else { Kind::Mamba }).collect();
        let heads_kv = kv.iter().take(layers).copied().find(|v| *v > 0).unwrap_or(0);
        let eps = fl("attention.layer_norm_rms_epsilon").or_else(|_| fl("attention.layer_norm_epsilon"))?;
        let g = Geometry {
            layers,
            kinds,
            n_embd: u("embedding_length")?,
            vocab: f.tensor("token_embd.weight").map(|t| t.shape[0] as usize).ok_or("no token_embd.weight")?,
            ctx: u("context_length").unwrap_or(262_144),
            eps,
            heads: u("attention.head_count")?,
            heads_kv,
            head: u("attention.key_length")?,
            conv: u("ssm.conv_kernel")?,
            state: u("ssm.state_size")?,
            groups: u("ssm.group_count")?,
            inner: u("ssm.inner_size")?,
            ssm_heads: u("ssm.time_step_rank")?,
            experts: u("expert_count")?,
            used: u("expert_used_count")?,
            latent: u("moe_latent_size").unwrap_or(0),
            n_ff_exp: u("expert_feed_forward_length")?,
            n_ff_shared: u("expert_shared_feed_forward_length")?,
            w_scale: fl("expert_weights_scale").unwrap_or(1.0),
            w_norm: f.arch_meta("expert_weights_norm").and_then(|v| v.as_u64()).is_some_and(|v| v != 0),
            nextn,
        };
        if g.latent == 0 {
            return Err(format!("{arch}: no moe_latent_size (this engine reads latent-MoE files: Nemotron 3.5 Super)"));
        }
        if g.head != 128 || g.heads_kv == 0 || !g.heads.is_multiple_of(g.heads_kv) || !(g.heads / g.heads_kv).is_multiple_of(8) {
            return Err(format!("attention heads of {} ({} / {}): head 128 with a multiple of 8 query heads a key head", g.head, g.heads, g.heads_kv));
        }
        if !g.state.is_multiple_of(16) || g.state > 256 || !g.inner.is_multiple_of(g.ssm_heads) || !g.ssm_heads.is_multiple_of(g.groups) {
            return Err(format!("Mamba-2 of state {}, inner {}, {} heads in {} groups: state a multiple of 16 up to 256, inner a multiple of the heads, \
                                the heads of the groups", g.state, g.inner, g.ssm_heads, g.groups));
        }
        if !(2..=8).contains(&g.conv) {
            return Err(format!("a conv of {}: 2 to 8", g.conv));
        }
        if g.used == 0 || g.used > g.experts {
            return Err(format!("{} of {} experts a token", g.used, g.experts));
        }
        Ok(g)
    }

    /// x | B | C: the conv's channels
    pub fn conv_dim(&self) -> usize {
        self.inner + 2 * self.groups * self.state
    }

    /// z | x B C | dt: the mixer's input projection
    pub fn in_proj(&self) -> usize {
        2 * self.inner + 2 * self.groups * self.state + self.ssm_heads
    }

    pub fn count(&self, k: Kind) -> usize {
        self.kinds.iter().filter(|x| **x == k).count()
    }
}

/// A matrix [rows (outputs), cols (inputs)]: its stored blocks, or float32 (small or unquantized tensors)
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

/// A layer's experts in one buffer: each [rows, cols] in its blocks, `stride` bytes apart
struct Experts {
    buf: DevBuf,
    ty: GType,
    rows: usize,
    cols: usize,
    stride: usize,
}

struct Mamba {
    inp: Mat,
    conv_w: DevBuf,
    conv_b: DevBuf,
    dt_b: DevBuf,
    a: DevBuf,
    d: DevBuf,
    norm: DevBuf,
    out: Mat,
}

struct Attn {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
}

struct Moe {
    /// float32 [experts, n_embd]
    router: Mat,
    /// the selection bias (exp_probs_b), on the host (its routing) and the GPU (the silo's)
    bias: Vec<f32>,
    bias_dev: DevBuf,
    to_latent: Mat,
    from_latent: Mat,
    up: Experts,
    down: Experts,
    sh_up: Mat,
    sh_down: Mat,
    /// routed on the GPU by the silo (its downs packed for it); else on the host
    on_gpu: bool,
}

enum Mix {
    Mamba(Mamba),
    Moe(Moe),
    Attn(Attn),
}

struct Layer {
    norm: DevBuf,
    mix: Mix,
}

/// A conversation: per Mamba layer its conv inputs ([conv-1][channels]) and scan state ([heads][head dim][state])
/// float32, per attention layer its key / value cache ([kv heads][max_ctx][128] half); MoE layers keep nothing
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<LState>,
    last_rows: usize,
}

enum LState {
    Mamba { conv: DevBuf, s: DevBuf },
    Attn { k: DevBuf, v: DevBuf },
    None,
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
        w.write_all(b"NSNMH")?;
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
        if &magic != b"NSNMH" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a nemotronh checkpoint"));
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

/// One GPU's share: its layers [lb, le) and a pass's work memory
struct Dev {
    gpu: Arc<Gpu>,
    arena: Arena,
    /// the MoE combine's fixed tables for the silo's routing: row r's entries start at r * used ([CHUNK + 1]), entry j
    /// is row j of the experts' outputs ([CHUNK * used])
    tp: DevBuf,
    seq: DevBuf,
    /// a prompt pass's matrix expanded to half (an expert's two after the largest dense one)
    w16: DevBuf,
    lb: usize,
    le: usize,
}

/// The rows the decode products take (the Q8_1 kernels' widest); a pass of more rows is a prompt pass (half products)
const DEC: usize = 8;

/// NS_NH_CHUNK: a prompt pass's rows (default: chosen at load; with the silo only - without it passes of DEC rows)
fn chunk_setting() -> Option<usize> {
    std::env::var("NS_NH_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= DEC)
}

pub struct NemotronH<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    devs: Vec<Dev>,
    layer_dev: Vec<usize>,
    cur: std::sync::atomic::AtomicUsize,
    staging: Mutex<Vec<u8>>,
    k: &'static ffi::Api,
    /// the silo (kernels/llm/nemotronh/silo): a MoE layer's routing and experts in a few launches; None: the host's
    silo: Option<&'static ffi::Silo>,
    layers: Vec<Layer>,
    output_norm: DevBuf,
    output: Mat,
    embd: Tensor,
    work: Mutex<()>,
    /// a prompt pass's rows
    chunk: usize,
    /// NS_NH_PROFILE=1: each kernel waited for and timed by name (seconds, calls) - the wall time it costs, launch
    /// overhead and all (the waits slow the pass itself)
    prof: Option<Mutex<std::collections::BTreeMap<String, (f64, u64)>>>,
    load_s: f64,
    load_bytes: u64,
}

/// A pass's work memory for `t` rows: the activations (float32, and half copies for the products), the experts' rows,
/// the last rows' logits, with room to spare
fn arena_bytes(g: &Geometry, t: usize) -> usize {
    let e = t * g.used;
    let acts = t * (g.in_proj() + 2 * g.conv_dim() + 3 * g.inner + 8 * g.n_embd + 2 * g.n_ff_shared + 4 * g.latent + g.experts
        + g.heads * g.head * 2 + 2 * g.heads_kv * g.head) * 6 / 4;
    let exps = e * (g.n_ff_exp + g.latent) * 2 + e * 4;
    (acts + exps + DEC * g.vocab) * 4 + (64 << 20)
}

/// The half buffer a prompt pass's products expand a matrix into: the largest dense matrix (an expert's fit too)
fn w16_bytes(g: &Geometry) -> usize {
    let dense = [g.in_proj() * g.n_embd, g.n_embd * g.inner, g.n_ff_shared * g.n_embd, g.heads * g.head * g.n_embd].into_iter().max().unwrap_or(0);
    dense.max(g.n_ff_exp * g.latent) * 2
}

impl<'g> NemotronH<'g> {
    /// Loads the model over `gpus` in order, the layers split so each GPU keeps about the same memory free (NS_SPLIT:
    /// the first layer of each GPU after the first); the head on the last. `kv`: the context the sessions reserve
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), log: &mut dyn FnMut(String)) -> Result<NemotronH<'g>> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(e("no GPU"));
        }
        let g = Geometry::read(f).map_err(e)?;
        let k = ffi::api()?;
        let silo = ffi::silo()?;
        log(match silo {
            Some(s) => format!("nemotronh: the MoE layers on the GPU from its silo {}", s.path.display()),
            None => "nemotronh: no silo - the routing on the host, an expert a launch".into(),
        });
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        let layer_w: Vec<u64> = (0..g.layers).map(|l| {
            let p = format!("blk.{l}.");
            f.tensors.iter().filter(|t| t.name.starts_with(&p)).map(|t| t.bytes).sum()
        }).collect();
        let head_w: u64 = ["output.weight", "output_norm.weight"].iter().filter_map(|n| f.tensor(n)).map(|t| t.bytes).sum();
        let mut free = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let (total, fr) = gpu.memory()?;
            free.push(fr.unwrap_or(total));
        }
        let margin: u64 = 1536 << 20;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let ctx = if kv.0 > 0 { kv.0 } else { 32768 };
        let sessions = kv.1.max(1) as u64;
        // a layer's cost where it lands: its weights, its sessions' cache (attention) or states (Mamba)
        let cost: Vec<u64> = (0..g.layers).map(|l| layer_w[l] + sessions * match g.kinds[l] {
            Kind::Attn => (2 * g.heads_kv * ctx * g.head * 2) as u64,
            Kind::Mamba => ((g.inner * g.state + (g.conv - 1) * g.conv_dim()) * 4) as u64,
            Kind::Moe => 0,
        }).collect();
        let n = gpus.len();
        let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        // the layers over the GPUs for a prompt chunk of `chunk` rows: each GPU's room after its spare, work memory and
        // half buffer (the last's after the head too), the layers cut where each GPU's share of the room ends
        let plan = |chunk: usize| -> std::result::Result<Vec<(usize, usize)>, String> {
            let over = |i: usize| margin + (512 << 20) + arena_bytes(&g, chunk) as u64 + w16_bytes(&g) as u64 + if i + 1 == n { head_w } else { 0 };
            let room: Vec<i64> = (0..n).map(|i| free[i] as i64 - over(i) as i64).collect();
            let mut bounds = Vec::with_capacity(n);
            if let Some(sp) = &split_env {
                let mut lb = 0;
                for i in 0..n {
                    let le = if i + 1 == n { g.layers } else { *sp.get(i).ok_or("NS_SPLIT: a first layer for each GPU after the first")? };
                    bounds.push((lb, le.min(g.layers)));
                    lb = le.min(g.layers);
                }
            } else {
                let total: u64 = cost.iter().sum();
                let have: i64 = room.iter().sum();
                if total as i64 > have {
                    return Err(format!("its layers need {:.1} GiB at a {ctx}-token context; the GPUs have {:.1} GiB for them", gib(total), gib(have.max(0) as u64)));
                }
                let mut lb = 0;
                let mut acc_room = 0i64;
                let mut acc = 0u64;
                for (i, r) in room.iter().enumerate() {
                    acc_room += *r;
                    let target = (total as f64 * acc_room as f64 / have as f64) as u64;
                    let mut le = lb;
                    while le < g.layers && (i + 1 == n || acc + cost[le] <= target) {
                        acc += cost[le];
                        le += 1;
                    }
                    bounds.push((lb, le));
                    lb = le;
                }
            }
            for (i, &(lb, le)) in bounds.iter().enumerate() {
                let need: u64 = cost[lb..le].iter().sum();
                if need as i64 > room[i] {
                    return Err(format!("layers {lb}-{} need {:.1} GiB on {}; {:.1} GiB are left for them", le.max(lb + 1) - 1, gib(need), gpus[i].name,
                                       gib(room[i].max(0) as u64)));
                }
            }
            Ok(bounds)
        };
        // the prompt chunk: NS_NH_CHUNK, else the largest of 2,048 / 1,024 / 512 rows that fits (more rows a pass, more
        // rows an expert's product: 3.9K prompts 280 -> 530 tok/s from 512 to 2,048); without the silo, DEC
        let tries: Vec<usize> = if silo.is_none() { vec![DEC] } else { chunk_setting().map_or_else(|| vec![2048, 1024, 512], |c| vec![c]) };
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
        let (chunk, bounds) = found.ok_or_else(|| e(format!("{} over {} GPUs: {last_err}", f.architecture(), n)))?;
        let (gpus, bounds): (Vec<Arc<Gpu>>, Vec<(usize, usize)>) = gpus.iter().cloned().zip(bounds).filter(|(_, b)| b.1 > b.0).unzip();
        let mut layer_dev = vec![0usize; g.layers];
        for (i, &(lb, le)) in bounds.iter().enumerate() {
            for d in &mut layer_dev[lb..le] {
                *d = i;
            }
        }
        let lastg = gpus[gpus.len() - 1].clone();
        let loaded = std::cell::Cell::new(0u64);
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let gpu = &gpus[layer_dev[l]];
            let w = |s: &str| format!("blk.{l}.{s}.weight");
            let read = |name: &str| -> Result<Mat> { vec_mat(f, gpu, k, name, &loaded) };
            let norm = read(&w("attn_norm"))?.buf;
            let mix = match g.kinds[l] {
                Kind::Mamba => {
                    let m = Mamba {
                        inp: read(&w("ssm_in"))?,
                        conv_w: read(&w("ssm_conv1d"))?.buf,
                        conv_b: read(&format!("blk.{l}.ssm_conv1d.bias"))?.buf,
                        dt_b: read(&format!("blk.{l}.ssm_dt.bias"))?.buf,
                        a: read(&format!("blk.{l}.ssm_a"))?.buf,
                        d: read(&format!("blk.{l}.ssm_d"))?.buf,
                        norm: read(&w("ssm_norm"))?.buf,
                        out: read(&w("ssm_out"))?,
                    };
                    if m.inp.rows != g.in_proj() || m.out.cols != g.inner {
                        return Err(e(format!("layer {l}: ssm_in of {} rows (want {}), ssm_out of {} inputs (want {})", m.inp.rows, g.in_proj(), m.out.cols, g.inner)));
                    }
                    Mix::Mamba(m)
                }
                Kind::Attn => Mix::Attn(Attn { q: read(&w("attn_q"))?, k: read(&w("attn_k"))?, v: read(&w("attn_v"))?, o: read(&w("attn_output"))? }),
                Kind::Moe => {
                    let router = read(&w("ffn_gate_inp"))?;
                    let bias_t = f.tensor(&format!("blk.{l}.exp_probs_b.bias")).ok_or_else(|| e(format!("layer {l}: no exp_probs_b.bias")))?;
                    let raw = f.read(bias_t).map_err(|x| e(x.0))?;
                    let bias: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let bias_dev = DevBuf::new(gpu, bias.len() * 4)?;
                    bias_dev.write(0, &raw)?;
                    let m = Moe {
                        router,
                        bias,
                        bias_dev,
                        to_latent: read(&w("ffn_latent_down"))?,
                        from_latent: read(&w("ffn_latent_up"))?,
                        up: experts(f, gpu, &w("ffn_up_exps"), &loaded)?,
                        down: experts(f, gpu, &w("ffn_down_exps"), &loaded)?,
                        on_gpu: false,
                        sh_up: read(&w("ffn_up_shexp"))?,
                        sh_down: read(&w("ffn_down_shexp"))?,
                    };
                    if m.router.rows != g.experts || m.bias.len() != g.experts || m.up.rows != g.n_ff_exp || m.up.cols != g.latent || m.down.rows != g.latent
                        || m.down.cols != g.n_ff_exp {
                        return Err(e(format!("layer {l}: the experts' shapes ({} x {} up, {} x {} down, a router of {}) are not the metadata's", m.up.rows,
                                             m.up.cols, m.down.rows, m.down.cols, m.router.rows)));
                    }
                    let mut m = m;
                    // the silo takes both projections: the routing on the GPU, the downs repacked for its kernels
                    // SAFETY: arithmetic only.
                    if let Some(s) = silo.filter(|s| unsafe { (s.multi_supported)(m.up.ty.code() as i32, m.up.cols as i64) != 0
                                                             && (s.multi_supported)(m.down.ty.code() as i32, m.down.cols as i64) != 0 }) {
                        m.on_gpu = true;
                        let (ty, n) = (m.down.ty.code() as i32, m.down.cols as i64);
                        // SAFETY: arithmetic only.
                        if unsafe { (s.packed)(ty, n) } != 0 {
                            // SAFETY: arithmetic only.
                            let rb = unsafe { (s.packed_row_bytes)(ty, n) } as usize;
                            let rows = g.experts * m.down.rows;
                            let buf = DevBuf::new(gpu, rows * rb)?;
                            // SAFETY: the file's blocks (rows of n) into the packed rows.
                            ffi::check(unsafe { (s.pack)(gpu.raw(), ty, m.down.buf.ptr(), buf.ptr(), rows as i64, n) }, "packing the experts")?;
                            gpu.sync()?;
                            m.down = Experts { buf, ty: m.down.ty, rows: m.down.rows, cols: m.down.cols, stride: m.down.rows * rb };
                        }
                    }
                    for x in [&m.up, &m.down] {
                        // SAFETY: arithmetic only.
                        if unsafe { (k.mmvq_supported)(x.ty.code() as i32) } == 0 {
                            return Err(e(format!("layer {l}: experts in {} (the products take Q8_0, the K and IQ types, IQ4_NL, Q2_0)", x.ty.name())));
                        }
                    }
                    Mix::Moe(m)
                }
            };
            layers.push(Layer { norm, mix });
            if l % 11 == 10 {
                log(format!("nemotronh: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, gib(loaded.get()), t0.elapsed().as_secs_f64()));
            }
        }
        let output_norm = vec_mat(f, &lastg, k, "output_norm.weight", &loaded)?.buf;
        let output = vec_mat(f, &lastg, k, if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }, &loaded)?;
        let mut devs = Vec::with_capacity(gpus.len());
        for (gpu, &(lb, le)) in gpus.iter().zip(&bounds) {
            gpu.sync()?;
            let bytes = |v: Vec<i32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let tp = DevBuf::new(gpu, (chunk + 1) * 4)?;
            tp.write(0, &bytes((0..=chunk as i32).map(|r| r * g.used as i32).collect()))?;
            let seq = DevBuf::new(gpu, chunk * g.used * 4)?;
            seq.write(0, &bytes((0..(chunk * g.used) as i32).collect()))?;
            devs.push(Dev { gpu: gpu.clone(), arena: Arena::new(gpu, arena_bytes(&g, chunk))?, w16: DevBuf::new(gpu, w16_bytes(&g))?, tp, seq, lb, le });
        }
        let load_s = t0.elapsed().as_secs_f64();
        let wh = devs.iter().map(|d| format!("{} (layers {}-{})", d.gpu.name, d.lb, d.le - 1)).collect::<Vec<_>>().join(", ");
        log(format!("nemotronh: {} layers of {} ({} Mamba-2, {} MoE of {} / {} experts, {} attention), {:.1} GiB on {wh} in {load_s:.0} s; prompt chunks of {chunk}{}", g.layers,
                    g.n_embd, g.count(Kind::Mamba), g.count(Kind::Moe), g.used, g.experts, g.count(Kind::Attn), gib(loaded.get()),
                    if g.nextn > 0 { "; its MTP block is not read (no drafts)" } else { "" }));
        Ok(NemotronH { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(Vec::new()), k, silo, layers, output_norm, output, embd, work: Mutex::new(()), chunk,
                       prof: std::env::var("NS_NH_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(Default::default())), load_s, load_bytes: loaded.get() })
    }

    fn op(&self, name: &str, f: impl FnOnce() -> c_int) -> Result<()> {
        let Some(p) = &self.prof else { return ffi::check(f(), name) };
        self.dev().gpu.sync()?;
        let t0 = Instant::now();
        ffi::check(f(), name)?;
        self.dev().gpu.sync()?;
        let mut m = p.lock().unwrap();
        let e = m.entry(format!("{name} (GPU {})", self.cur.load(std::sync::atomic::Ordering::Relaxed))).or_insert((0.0, 0));
        e.0 += t0.elapsed().as_secs_f64();
        e.1 += 1;
        Ok(())
    }

    /// NS_NH_PROFILE's table: each kernel's share, the costliest first
    pub fn profile_lines(&self) -> Vec<String> {
        let Some(p) = &self.prof else { return Vec::new() };
        let m = p.lock().unwrap();
        let total: f64 = m.values().map(|v| v.0).sum();
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        let mut out = vec![format!("profile: {total:.3} s in kernels (each waited for)")];
        out.extend(v.iter().map(|(k, (s, n))| format!("  {k:<28} {s:>8.3} s {:>5.1}% {n:>8} calls {:>8.1} us/call", 100.0 * s / total.max(1e-9), 1e6 * s / *n as f64)));
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

    /// x [t, cols] -> its Q8_1 blocks (column j at j * cols / 32 blocks)
    fn quantize(&self, x: *const f32, cols: usize, t: usize) -> Result<DevBuf> {
        // SAFETY: arithmetic only.
        let q8 = unsafe { (self.k.q8_1_bytes)(cols as i64, t as i64) };
        let q = self.arena().bytes(q8)?;
        // SAFETY: x [t, cols] float32 -> its Q8_1 blocks.
        self.op("quantize", || unsafe { (self.k.quantize_q8_1)(self.raw(), x, q.ptr(), cols as i64, t as i64) })?;
        Ok(q)
    }

    /// y [t, w.rows] = x [t, w.cols] . W^T (`xq`: x's Q8_1 blocks when made already)
    fn matmul(&self, w: &Mat, x: &DevBuf, xq: Option<&DevBuf>, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        if w.ty == GType::F32 {
            // SAFETY: x [t, cols], the matrix [rows, cols] float32, y [t, rows].
            return self.op("gemm", || unsafe { (k.gemm)(self.raw(), t as i64, n, kk, x.fp(), kk, w.buf.fp(), y.fp(), n, 0) });
        }
        if t > DEC {
            return self.matmul16(w, x, t, y);
        }
        let own;
        let q = match xq {
            Some(q) => q,
            None => {
                own = self.quantize(x.fp(), w.cols, t)?;
                &own
            }
        };
        let name = if self.prof.is_some() { format!("mmvq {} {}x{} t{t}", w.ty.name(), w.rows, w.cols) } else { String::new() };
        // SAFETY: the matrix in its blocks, x's Q8_1 blocks, y [t, rows].
        self.op(&name, || unsafe { (k.mmvq)(self.raw(), w.ty.code() as i32, w.buf.ptr(), q.ptr(), y.fp(), kk, n, t as i64) })
    }

    /// The prompt pass's product: the matrix expanded to half (its type's own expansion, Q6_K's from this engine's
    /// kernels), x to half, then the half GEMM
    fn matmul16(&self, w: &Mat, x: &DevBuf, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let elems = w.rows * w.cols;
        let w16 = &self.dev().w16;
        if elems * 2 > w16.len {
            return Err(e(format!("a {}x{} matrix past the half buffer", w.rows, w.cols)));
        }
        let key = if self.prof.is_some() { format!("{} {}x{}", w.ty.name(), w.rows, w.cols) } else { String::new() };
        // SAFETY: the matrix's blocks -> rows x cols halfs.
        if w.ty == GType::Q6K {
            self.op(&format!("dequant16 {key}"), || unsafe { (k.q6k_f16)(self.raw(), w.buf.ptr(), elems as i64, w16.ptr().cast()) })?;
        } else {
            self.op(&format!("dequant16 {key}"), || unsafe { (k.dequant_f16)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems as i64, w16.ptr().cast()) })?;
        }
        let xh = self.arena().bytes(t * w.cols * 2)?;
        // SAFETY: x [t, cols] -> half; then y [t, rows] = xh . w16^T.
        self.op("to half", || unsafe { (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (t * w.cols) as i64) })?;
        self.op(&format!("gemm16 {key}"), || unsafe {
            (k.gemm_f16)(self.raw(), t as i64, w.rows as i64, w.cols as i64, xh.ptr().cast(), w.cols as i64, w16.ptr().cast(), y.fp(), w.rows as i64, 0)
        })
    }

    fn rms(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<()> {
        // SAFETY: x and y [rows, c], w c floats.
        self.op("rms norm", || unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, self.g.eps) })
    }

    fn add(&self, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        self.op("add", || unsafe { (self.k.add)(self.raw(), y.fp(), x.fp(), n as i64) })
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

    /// A Mamba-2 mixer on h [t, n_embd]: its output [t, n_embd]
    #[allow(clippy::too_many_arguments)]
    fn mamba(&self, m: &Mamba, conv: &DevBuf, s: &DevBuf, h: &DevBuf, t: usize, tap: Tap, l: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (ip, cd) = (g.in_proj(), g.conv_dim());
        let zx = self.arena().f32(t * ip)?;
        self.matmul(&m.inp, h, None, t, &zx)?;
        let co = self.arena().f32(t * cd)?;
        let y = self.arena().f32(t * g.inner)?;
        let yn = self.arena().f32(t * g.inner)?;
        let hp = g.inner / g.ssm_heads;
        // SAFETY: the buffers as named in nh.h; z at 0, x B C at inner, dt at inner + conv_dim of each zx row.
        unsafe {
            self.op("conv", || (k.conv)(self.raw(), zx.fp().add(g.inner), ip as i64, conv.fp(), m.conv_w.fp(), m.conv_b.fp(), co.fp(), t as i64, cd as i64,
                                        g.conv as i32))?;
            self.op("ssm scan", || (k.ssm)(self.raw(), co.fp(), cd as i64, zx.fp().add(g.inner + cd), ip as i64, m.dt_b.fp(), m.a.fp(), m.d.fp(), s.fp(),
                                           y.fp(), t as i64, g.ssm_heads as i64, hp as i64, g.state as i64, g.groups as i64))?;
            tap(&format!("mamba2_y_add_d-{l}"), &y)?;
            self.op("gated norm", || (k.gated_norm)(self.raw(), y.fp(), zx.fp(), ip as i64, m.norm.fp(), yn.fp(), t as i64, g.inner as i64,
                                                    (g.inner / g.groups) as i64, g.eps))?;
        }
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul(&m.out, &yn, None, t, &out)?;
        Ok(out)
    }

    /// An attention layer on h [t, n_embd] (no positions): its output [t, n_embd]
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, a: &Attn, kc: &DevBuf, vc: &DevBuf, cap: usize, h: &DevBuf, t: usize, p0: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        let hq8 = if t <= DEC { Some(self.quantize(h.fp(), g.n_embd, t)?) } else { None };
        let q = self.arena().f32(t * hq * d)?;
        let kr = self.arena().f32(t * hk * d)?;
        let vr = self.arena().f32(t * hk * d)?;
        self.matmul(&a.q, h, hq8.as_ref(), t, &q)?;
        self.matmul(&a.k, h, hq8.as_ref(), t, &kr)?;
        self.matmul(&a.v, h, hq8.as_ref(), t, &vr)?;
        let att = self.arena().f32(t * hq * d)?;
        // SAFETY: arithmetic only.
        let sc = unsafe { (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64) } as usize;
        let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
        // SAFETY: the rows and caches as named in nh.h.
        unsafe {
            self.op("kv store", || (k.kv_store)(self.raw(), kr.fp(), (hk * d) as i64, vr.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64, cap as i64,
                                                p0 as i64, kc.ptr(), vc.ptr()))?;
            self.op("attention", || (k.attn)(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64,
                                             p0 as i64, att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
        }
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul(&a.o, &att, None, t, &out)?;
        Ok(out)
    }

    /// A latent-MoE layer on h [t, n_embd]: the routed experts in the latent, projected back, plus the shared expert
    fn moe(&self, m: &Moe, h: &DevBuf, t: usize, tap: Tap, l: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let hq8 = if t <= DEC { Some(self.quantize(h.fp(), g.n_embd, t)?) } else { None };
        // the router (float32) and the choice on the host: sigmoid, the bias for the choice only, the chosen ones'
        // probabilities renormalized and scaled
        let logits = self.arena().f32(t * g.experts)?;
        self.matmul(&m.router, h, None, t, &logits)?;
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        let lat = self.arena().f32(t * g.latent)?;
        self.matmul(&m.to_latent, h, hq8.as_ref(), t, &lat)?;
        if let Some(s) = self.silo.filter(|_| m.on_gpu) {
            let mo = if t > DEC { self.moe_prompt(s, m, &logits, &lat, t)? } else { self.moe_gpu(s, m, &logits, &lat, t)? };
            return self.moe_tail(m, mo, h, hq8.as_ref(), t, tap, l);
        }
        if t > DEC {
            return Err(e(format!("a MoE layer of {t} rows without the silo (its passes are {DEC} rows)")));
        }
        let lg = logits.to_f32()?;
        let n_e = t * g.used;
        let mut ent: Vec<(usize, usize, f32)> = Vec::with_capacity(n_e);
        for r in 0..t {
            let row = &lg[r * g.experts..(r + 1) * g.experts];
            let p: Vec<f32> = row.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let mut idx: Vec<usize> = (0..g.experts).collect();
            // the choice by probability plus bias, highest first (the lower index on a tie)
            idx.sort_by(|a, b| (p[*b] + m.bias[*b]).total_cmp(&(p[*a] + m.bias[*a])).then(a.cmp(b)));
            idx.truncate(g.used);
            let mut w: Vec<f32> = idx.iter().map(|i| p[*i]).collect();
            if g.w_norm {
                let s: f32 = w.iter().sum();
                if s > 0.0 {
                    w.iter_mut().for_each(|x| *x /= s);
                }
            }
            for (i, x) in idx.into_iter().zip(w) {
                ent.push((r, i, x * g.w_scale));
            }
        }
        let latq = self.quantize(lat.fp(), g.latent, t)?;
        // SAFETY: arithmetic only.
        let (lq, uq) = unsafe { ((k.q8_1_bytes)(g.latent as i64, 1), (k.q8_1_bytes)(g.n_ff_exp as i64, 1)) };
        let u = self.arena().f32(n_e * g.n_ff_exp)?;
        for (j, (r, x, _)) in ent.iter().enumerate() {
            // SAFETY: expert x's up [n_ff, latent] in its blocks, token r's latent Q8_1, its row of u.
            self.op("expert up", || unsafe {
                (k.mmvq)(self.raw(), m.up.ty.code() as i32, m.up.buf.ptr().cast::<u8>().add(x * m.up.stride).cast(), latq.ptr().cast::<u8>().add(r * lq).cast(),
                         u.fp().add(j * g.n_ff_exp), g.latent as i64, g.n_ff_exp as i64, 1)
            })?;
        }
        // SAFETY: n floats.
        self.op("relu2", || unsafe { (k.relu2)(self.raw(), u.fp(), (n_e * g.n_ff_exp) as i64) })?;
        let uq8 = self.quantize(u.fp(), g.n_ff_exp, n_e)?;
        let dn = self.arena().f32(n_e * g.latent)?;
        for (j, (_, x, _)) in ent.iter().enumerate() {
            // SAFETY: expert x's down [latent, n_ff] in its blocks, entry j's Q8_1, its row of dn.
            self.op("expert down", || unsafe {
                (k.mmvq)(self.raw(), m.down.ty.code() as i32, m.down.buf.ptr().cast::<u8>().add(x * m.down.stride).cast(),
                         uq8.ptr().cast::<u8>().add(j * uq).cast(), dn.fp().add(j * g.latent), g.n_ff_exp as i64, g.latent as i64, 1)
            })?;
        }
        // the weighted sum per token (entries in token order)
        let mut tp = vec![0i32; t + 1];
        for (r, _, _) in &ent {
            tp[r + 1] += 1;
        }
        for r in 0..t {
            tp[r + 1] += tp[r];
        }
        let ids: Vec<i32> = (0..n_e as i32).collect();
        let ws: Vec<f32> = ent.iter().map(|x| x.2).collect();
        let bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (tpb, idb, wb) = (self.arena().bytes((t + 1) * 4)?, self.arena().bytes(n_e * 4)?, self.arena().bytes(n_e * 4)?);
        tpb.write(0, &bytes(&tp))?;
        idb.write(0, &bytes(&ids))?;
        wb.write(0, &ws.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        let mo = self.arena().f32(t * g.latent)?;
        mo.fill(0)?;
        // SAFETY: mo [t, latent], dn [entries, latent], the token pointers, entry ids and weights as named in glm.h.
        self.op("moe combine", || unsafe {
            (k.moe_combine)(self.raw(), mo.fp(), dn.fp(), tpb.ptr().cast(), idb.ptr().cast(), wb.fp(), t as i64, g.latent as i64)
        })?;
        self.moe_tail(m, mo, h, hq8.as_ref(), t, tap, l)
    }

    /// A MoE layer's routed experts over a prompt pass's rows: routed on the GPU, the choice read back and grouped by
    /// expert; per expert in use its rows gathered (half), its up and down expanded to half and multiplied over them,
    /// relu^2 between; the rows' weighted sums. The latent sum [t, latent]
    fn moe_prompt(&self, s: &ffi::Silo, m: &Moe, logits: &DevBuf, lat: &DevBuf, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let n_e = t * g.used;
        let ids = self.arena().bytes(n_e * 4)?;
        let ws = self.arena().f32(n_e)?;
        let cols = self.arena().bytes(n_e * 4)?;
        // SAFETY: the buffers as named in nhs.h.
        self.op("route", || unsafe {
            (s.route)(self.raw(), logits.fp(), m.bias_dev.fp(), t as i64, g.experts as i64, g.used as i64, g.w_norm as c_int, g.w_scale, ids.ptr().cast(),
                      ws.fp(), cols.ptr().cast())
        })?;
        let mut idb = vec![0u8; n_e * 4];
        ids.read(0, &mut idb)?;
        let chosen: Vec<usize> = idb.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
        // the entries in expert order (stable: rows ascending within an expert); each entry's place there
        let mut count = vec![0usize; g.experts + 1];
        for x in &chosen {
            count[x + 1] += 1;
        }
        for i in 0..g.experts {
            count[i + 1] += count[i];
        }
        let mut fill = count.clone();
        let mut rows = vec![0i32; n_e];
        let mut place = vec![0i32; n_e];
        for (j, x) in chosen.iter().enumerate() {
            rows[fill[*x]] = (j / g.used) as i32;
            place[j] = fill[*x] as i32;
            fill[*x] += 1;
        }
        let bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let rowb = self.arena().bytes(n_e * 4)?;
        rowb.write(0, &bytes(&rows))?;
        let placeb = self.arena().bytes(n_e * 4)?;
        placeb.write(0, &bytes(&place))?;
        let lat16 = self.arena().bytes(t * g.latent * 2)?;
        let xg = self.arena().bytes(n_e * g.latent * 2)?;
        let u = self.arena().f32(n_e * g.n_ff_exp)?;
        // SAFETY: the buffers as named in glm.h: the latent rows to half, then each entry's row gathered in expert order.
        unsafe {
            self.op("to half", || (k.to_f16)(self.raw(), lat.fp(), lat16.ptr().cast(), (t * g.latent) as i64))?;
            self.op("gather", || (k.gather_h)(self.raw(), lat16.ptr().cast(), rowb.ptr().cast(), xg.ptr().cast(), n_e as i64, g.latent as i64))?;
        }
        let w16 = &self.dev().w16;
        let u16 = self.arena().bytes(n_e * g.n_ff_exp * 2)?;
        let dn = self.arena().f32(n_e * g.latent)?;
        // SAFETY: arithmetic only.
        let packed = unsafe { (s.packed)(m.down.ty.code() as i32, g.n_ff_exp as i64) } != 0;
        // the up products, then relu^2 and half, then the down products: one expert at a time - its matrix expanded to
        // half, then multiplied over its rows while the half copy (5.5 MB) is still in the cache (a group of 64 expanded
        // in one launch was slower: 659 -> 596 tok/s, the copies read back from memory)
        for (stage, mat) in [(0, &m.up), (1, &m.down)] {
            if stage == 1 {
                // SAFETY: n floats, then to half.
                unsafe {
                    self.op("relu2", || (k.relu2)(self.raw(), u.fp(), (n_e * g.n_ff_exp) as i64))?;
                    self.op("to half", || (k.to_f16)(self.raw(), u.fp(), u16.ptr().cast(), (n_e * g.n_ff_exp) as i64))?;
                }
            }
            let (rows, cols) = (mat.rows, mat.cols);
            for x in 0..g.experts {
                let (a, b) = (count[x], count[x + 1]);
                if a == b {
                    continue;
                }
                // SAFETY: expert x -> half [rows][cols] in w16 (the down packed: the silo's expansion; else the type's
                // own); its rows (half) times it into their rows of the output.
                unsafe {
                    let src = mat.buf.ptr().cast::<u8>().add(x * mat.stride).cast();
                    if stage == 1 && packed {
                        self.op("expert down f16", || (s.packed_f16)(self.raw(), mat.ty.code() as i32, src, w16.ptr().cast(), rows as i64, cols as i64))?;
                    } else {
                        self.op(if stage == 0 { "expert up f16" } else { "expert down f16" },
                                || (k.dequant_f16)(self.raw(), mat.ty.code() as i32, src, (rows * cols) as i64, w16.ptr().cast()))?;
                    }
                    let (xin, out) = if stage == 0 { (xg.ptr().cast::<u16>().add(a * cols), u.fp().add(a * rows)) } else { (u16.ptr().cast::<u16>().add(a * cols), dn.fp().add(a * rows)) };
                    self.op(if stage == 0 { "expert up gemm16" } else { "expert down gemm16" }, || {
                        (k.gemm_f16)(self.raw(), (b - a) as i64, rows as i64, cols as i64, xin, cols as i64, w16.ptr().cast(), out, rows as i64, 0)
                    })?;
                }
            }
        }
        let mo = self.arena().f32(t * g.latent)?;
        mo.fill(0)?;
        // SAFETY: mo [t, latent], dn [entries, latent] in expert order, row r's entries r * used.. (their places), weights.
        self.op("moe combine", || unsafe {
            (k.moe_combine)(self.raw(), mo.fp(), dn.fp(), self.dev().tp.ptr().cast(), placeb.ptr().cast(), ws.fp(), t as i64, g.latent as i64)
        })?;
        Ok(mo)
    }

    /// A MoE layer's routed experts on the GPU (the silo): routing, every chosen expert's up in one launch, relu^2,
    /// every down in one, the weighted sum per row - no wait for the host. The latent sum [t, latent]
    fn moe_gpu(&self, s: &ffi::Silo, m: &Moe, logits: &DevBuf, lat: &DevBuf, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let n_e = t * g.used;
        let ids = self.arena().bytes(n_e * 4)?;
        let ws = self.arena().f32(n_e)?;
        let cols = self.arena().bytes(n_e * 4)?;
        // SAFETY: the buffers as named in nhs.h / glm.h.
        unsafe {
            self.op("route", || (s.route)(self.raw(), logits.fp(), m.bias_dev.fp(), t as i64, g.experts as i64, g.used as i64, g.w_norm as c_int, g.w_scale,
                                          ids.ptr().cast(), ws.fp(), cols.ptr().cast()))?;
            let lq = self.arena().bytes((s.q8_bytes)(g.latent as i64, t as i64))?;
            self.op("quantize", || (s.quantize_q8)(self.raw(), lat.fp(), lq.ptr(), g.latent as i64, t as i64))?;
            let u = self.arena().f32(n_e * g.n_ff_exp)?;
            self.op("experts up", || (s.multi)(self.raw(), m.up.ty.code() as i32, m.up.buf.ptr(), m.up.stride as i64, ids.ptr().cast(), cols.ptr().cast(),
                                               n_e as i64, lq.ptr(), t as i64, u.fp(), g.latent as i64, g.n_ff_exp as i64))?;
            self.op("relu2", || (k.relu2)(self.raw(), u.fp(), (n_e * g.n_ff_exp) as i64))?;
            let uq = self.arena().bytes((s.q8_bytes)(g.n_ff_exp as i64, n_e as i64))?;
            self.op("quantize", || (s.quantize_q8)(self.raw(), u.fp(), uq.ptr(), g.n_ff_exp as i64, n_e as i64))?;
            let dn = self.arena().f32(n_e * g.latent)?;
            let seq = &self.dev().seq;
            self.op("experts down", || (s.multi)(self.raw(), m.down.ty.code() as i32, m.down.buf.ptr(), m.down.stride as i64, ids.ptr().cast(),
                                                 seq.ptr().cast(), n_e as i64, uq.ptr(), n_e as i64, dn.fp(), g.n_ff_exp as i64, g.latent as i64))?;
            let mo = self.arena().f32(t * g.latent)?;
            mo.fill(0)?;
            self.op("moe combine", || (k.moe_combine)(self.raw(), mo.fp(), dn.fp(), self.dev().tp.ptr().cast(), seq.ptr().cast(), ws.fp(), t as i64,
                                                      g.latent as i64))?;
            Ok(mo)
        }
    }

    /// A MoE layer after its routed experts: the latent sum projected back, plus the shared expert
    #[allow(clippy::too_many_arguments)]
    fn moe_tail(&self, m: &Moe, mo: DevBuf, h: &DevBuf, hq8: Option<&DevBuf>, t: usize, tap: Tap, l: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        tap(&format!("ffn_moe_out-{l}"), &mo)?;
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul(&m.from_latent, &mo, None, t, &out)?;
        // the shared expert on the full width
        let su = self.arena().f32(t * g.n_ff_shared)?;
        self.matmul(&m.sh_up, h, hq8, t, &su)?;
        // SAFETY: n floats.
        self.op("relu2", || unsafe { (k.relu2)(self.raw(), su.fp(), (t * g.n_ff_shared) as i64) })?;
        let sd = self.arena().f32(t * g.n_embd)?;
        self.matmul(&m.sh_down, &su, None, t, &sd)?;
        tap(&format!("ffn_shexp-{l}"), &sd)?;
        self.add(&out, &sd, t * g.n_embd)?;
        Ok(out)
    }

    /// One pass of at most a chunk's rows: the last `n_out` rows' logits
    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        self.pass_in(s, tokens, n_out, tap)
    }

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
        let mut x = self.embed(tokens)?;
        let mut h = self.arena().f32(t * g.n_embd)?;
        for (l, (ly, st)) in self.layers.iter().zip(s.layers.iter()).enumerate() {
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                x = self.hop(&x, self.layer_dev[l])?;
                h = self.arena().f32(t * g.n_embd)?;
            }
            let mark = self.arena().mark();
            self.rms(&x, &ly.norm, &h, t, g.n_embd)?;
            tap(&format!("attn_norm-{l}"), &h)?;
            let out = match (&ly.mix, st) {
                (Mix::Mamba(m), LState::Mamba { conv, s: ss }) => {
                    let o = self.mamba(m, conv, ss, &h, t, &mut *tap, l)?;
                    tap(&format!("mamba_out-{l}"), &o)?;
                    o
                }
                (Mix::Attn(a), LState::Attn { k, v }) => {
                    let o = self.attention(a, k, v, s.max_ctx, &h, t, p0)?;
                    tap(&format!("attn_out-{l}"), &o)?;
                    o
                }
                (Mix::Moe(m), LState::None) => {
                    let o = self.moe(m, &h, t, &mut *tap, l)?;
                    tap(&format!("ffn_out-{l}"), &o)?;
                    o
                }
                _ => return Err(e(format!("layer {l}: its state and its weights disagree"))),
            };
            self.add(&x, &out, t * g.n_embd)?;
            tap(&format!("nemotron_h_block_out-{l}"), &x)?;
            self.arena().rewind(mark);
        }
        s.pos += t;
        s.last_rows = t;
        let n = n_out.clamp(1, t);
        let ne = g.n_embd;
        let last = x.view((t - n) * ne * 4, n * ne * 4)?;
        let normed = self.arena().f32(n * ne)?;
        self.rms(&last, &self.output_norm, &normed, n, ne)?;
        tap("result_norm", &normed)?;
        let logits = self.arena().f32(n * self.output.rows)?;
        self.matmul(&self.output, &normed, None, n, &logits)?;
        tap("result_output", &logits)?;
        let all = logits.to_f32()?;
        Ok(all.chunks(self.output.rows).map(|c| c.to_vec()).collect())
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        let kv = |n: usize| 2 * g.heads_kv * n * g.head * 2;
        let ms = (g.inner * g.state + (g.conv - 1) * g.conv_dim()) * 4;
        for d in &self.devs {
            let need: u64 = (d.lb..d.le).map(|l| match g.kinds[l] {
                Kind::Attn => kv(max_ctx) as u64,
                Kind::Mamba => ms as u64,
                Kind::Moe => 0,
            }).sum();
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
            layers.push(match g.kinds[l] {
                Kind::Attn => {
                    let n = g.heads_kv * max_ctx * g.head * 2;
                    LState::Attn { k: DevBuf::new(gpu, n)?, v: DevBuf::new(gpu, n)? }
                }
                Kind::Mamba => {
                    let conv = DevBuf::f32(gpu, (g.conv - 1) * g.conv_dim())?;
                    let s = DevBuf::f32(gpu, g.inner * g.state)?;
                    conv.fill(0)?;
                    s.fill(0)?;
                    LState::Mamba { conv, s }
                }
                Kind::Moe => LState::None,
            });
        }
        Ok(Session { pos: 0, max_ctx, layers, last_rows: 0 })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        for st in &s.layers {
            if let LState::Mamba { conv, s } = st {
                conv.fill(0)?;
                s.fill(0)?;
            }
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
                (LState::Attn { k: dk, v: dv }, LState::Attn { k: sk, v: sv }) => {
                    for h in 0..g.heads_kv {
                        let n = src.pos * g.head * 2;
                        dk.copy_within(h * dst.max_ctx * g.head * 2, sk, h * src.max_ctx * g.head * 2, n)?;
                        dv.copy_within(h * dst.max_ctx * g.head * 2, sv, h * src.max_ctx * g.head * 2, n)?;
                    }
                }
                (LState::Mamba { conv: dc, s: ds }, LState::Mamba { conv: sc, s: ss }) => {
                    dc.copy_within(0, sc, 0, sc.len)?;
                    ds.copy_within(0, ss, 0, ss.len)?;
                }
                (LState::None, LState::None) => {}
                _ => return Err(e("sessions of other shapes")),
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

    /// The Mamba states have no snapshots yet: only a rollback that keeps the whole last pass
    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep == s.last_rows {
            return Ok(());
        }
        Err(e("nemotronh: no rollback (the Mamba states keep no snapshots yet)"))
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let y = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => smp.sample(&l),
            (None, Some(x)) => {
                let l = self.pass(s, &[x], 1, tap)?.pop().unwrap_or_default();
                smp.sample(&l)
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        d.next = Some(y);
        Ok(vec![y])
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let g = &self.g;
        let mut bufs = Vec::new();
        for st in &s.layers {
            match st {
                LState::Attn { k, v } => {
                    for c in [k, v] {
                        let n = s.pos * g.head * 2;
                        let mut b = vec![0u8; g.heads_kv * n];
                        for h in 0..g.heads_kv {
                            c.read(h * s.max_ctx * g.head * 2, &mut b[h * n..(h + 1) * n])?;
                        }
                        bufs.push(b);
                    }
                }
                LState::Mamba { conv, s: ss } => {
                    for c in [conv, ss] {
                        let mut b = vec![0u8; c.len];
                        c.read(0, &mut b)?;
                        bufs.push(b);
                    }
                }
                LState::None => {}
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
        let mut it = ck.bufs.iter();
        for st in &s.layers {
            match st {
                LState::Attn { k, v } => {
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
                LState::Mamba { conv, s: ss } => {
                    for c in [conv, ss] {
                        let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                        if b.len() != c.len {
                            return Err(e("a checkpoint of another shape"));
                        }
                        c.write(0, b)?;
                    }
                }
                LState::None => {}
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

/// A tensor read onto the GPU: its blocks, or float32 for F32 / F16 / BF16
fn vec_mat(f: &Gguf, gpu: &Arc<Gpu>, k: &ffi::Api, name: &str, loaded: &std::cell::Cell<u64>) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    // outermost first: [rows (outputs), cols (inputs)]; a vector is one row
    let (rows, cols) = match t.shape.len() {
        1 => (1, t.shape[0] as usize),
        _ => (t.shape[0] as usize, t.shape[1..].iter().product::<u64>() as usize),
    };
    let raw = f.read(t).map_err(|x| e(x.0))?;
    loaded.set(loaded.get() + raw.len() as u64);
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

/// A layer's stacked experts [experts, rows, cols] onto the GPU in their blocks
fn experts(f: &Gguf, gpu: &Arc<Gpu>, name: &str, loaded: &std::cell::Cell<u64>) -> Result<Experts> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    if t.shape.len() != 3 {
        return Err(e(format!("{name}: [experts, rows, cols], not {:?}", t.shape)));
    }
    let (n, rows, cols) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
    let stride = t.ty.bytes((rows * cols) as u64).ok_or_else(|| e(format!("{name}: rows of {cols} are not whole {} blocks", t.ty.name())))? as usize;
    if stride * n != t.bytes as usize {
        return Err(e(format!("{name}: {} bytes, not {n} experts of {stride}", t.bytes)));
    }
    let raw = f.read(t).map_err(|x| e(x.0))?;
    loaded.set(loaded.get() + raw.len() as u64);
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    Ok(Experts { buf, ty: t.ty, rows, cols, stride })
}
