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
    /// the selection bias (exp_probs_b), on the host: the choice is made there
    bias: Vec<f32>,
    to_latent: Mat,
    from_latent: Mat,
    up: Experts,
    down: Experts,
    sh_up: Mat,
    sh_down: Mat,
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
    lb: usize,
    le: usize,
}

/// The rows a pass takes (the Q8_1 products' widest)
const CHUNK: usize = 8;

pub struct NemotronH<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    devs: Vec<Dev>,
    layer_dev: Vec<usize>,
    cur: std::sync::atomic::AtomicUsize,
    staging: Mutex<Vec<u8>>,
    k: &'static ffi::Api,
    layers: Vec<Layer>,
    output_norm: DevBuf,
    output: Mat,
    embd: Tensor,
    work: Mutex<()>,
    load_s: f64,
    load_bytes: u64,
}

/// A pass's work memory: the activations of CHUNK rows, the experts' rows, the logits, with room to spare
fn arena_bytes(g: &Geometry) -> usize {
    let t = CHUNK;
    let e = t * g.used;
    let acts = t * (g.in_proj() + 2 * g.conv_dim() + 3 * g.inner + 8 * g.n_embd + 2 * g.n_ff_shared + 4 * g.latent + g.experts
        + g.heads * g.head * 2 + 2 * g.heads_kv * g.head);
    let exps = e * (g.n_ff_exp + g.latent) * 2;
    (acts + exps + t * g.vocab) * 4 + (64 << 20)
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
        let over = |i: usize| margin + (512 << 20) + arena_bytes(&g) as u64 + if i + 1 == n { head_w } else { 0 };
        let room: Vec<i64> = (0..n).map(|i| free[i] as i64 - over(i) as i64).collect();
        let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        let mut bounds = Vec::with_capacity(n);
        if let Some(sp) = &split_env {
            let mut lb = 0;
            for i in 0..n {
                let le = if i + 1 == n { g.layers } else { *sp.get(i).ok_or_else(|| e("NS_SPLIT: a first layer for each GPU after the first"))? };
                bounds.push((lb, le.min(g.layers)));
                lb = le.min(g.layers);
            }
        } else {
            // the same share of each GPU's room: the layers in order, a GPU's share cut where its part of the total ends
            let total: u64 = cost.iter().sum();
            let have: i64 = room.iter().sum();
            if total as i64 > have {
                return Err(e(format!("{} needs {:.1} GiB for its layers at a {ctx}-token context; the GPUs have {:.1} GiB for them", f.architecture(),
                                     gib(total), gib(have.max(0) as u64))));
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
                return Err(e(format!("layers {lb}-{} need {:.1} GiB on {}; {:.1} GiB are left for them", le.max(lb + 1) - 1, gib(need), gpus[i].name,
                                     gib(room[i].max(0) as u64))));
            }
        }
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
                    let m = Moe {
                        router,
                        bias,
                        to_latent: read(&w("ffn_latent_down"))?,
                        from_latent: read(&w("ffn_latent_up"))?,
                        up: experts(f, gpu, &w("ffn_up_exps"), &loaded)?,
                        down: experts(f, gpu, &w("ffn_down_exps"), &loaded)?,
                        sh_up: read(&w("ffn_up_shexp"))?,
                        sh_down: read(&w("ffn_down_shexp"))?,
                    };
                    if m.router.rows != g.experts || m.bias.len() != g.experts || m.up.rows != g.n_ff_exp || m.up.cols != g.latent || m.down.rows != g.latent
                        || m.down.cols != g.n_ff_exp {
                        return Err(e(format!("layer {l}: the experts' shapes ({} x {} up, {} x {} down, a router of {}) are not the metadata's", m.up.rows,
                                             m.up.cols, m.down.rows, m.down.cols, m.router.rows)));
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
            devs.push(Dev { gpu: gpu.clone(), arena: Arena::new(gpu, arena_bytes(&g))?, lb, le });
        }
        let load_s = t0.elapsed().as_secs_f64();
        let wh = devs.iter().map(|d| format!("{} (layers {}-{})", d.gpu.name, d.lb, d.le - 1)).collect::<Vec<_>>().join(", ");
        log(format!("nemotronh: {} layers of {} ({} Mamba-2, {} MoE of {} / {} experts, {} attention), {:.1} GiB on {wh} in {load_s:.0} s{}", g.layers,
                    g.n_embd, g.count(Kind::Mamba), g.count(Kind::Moe), g.used, g.experts, g.count(Kind::Attn), gib(loaded.get()),
                    if g.nextn > 0 { "; its MTP block is not read (no drafts)" } else { "" }));
        Ok(NemotronH { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(Vec::new()), k, layers, output_norm, output, embd, work: Mutex::new(()),
                       load_s, load_bytes: loaded.get() })
    }

    fn op(&self, name: &str, f: impl FnOnce() -> c_int) -> Result<()> {
        ffi::check(f(), name)
    }

    pub fn load_seconds(&self) -> f64 {
        self.load_s
    }
    pub fn load_bytes(&self) -> u64 {
        self.load_bytes
    }
    pub fn prefill_chunk(&self) -> usize {
        CHUNK
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
        if t > CHUNK {
            return Err(e(format!("a product of {t} rows (at most {CHUNK})")));
        }
        let own;
        let q = match xq {
            Some(q) => q,
            None => {
                own = self.quantize(x.fp(), w.cols, t)?;
                &own
            }
        };
        // SAFETY: the matrix in its blocks, x's Q8_1 blocks, y [t, rows].
        self.op("mmvq", || unsafe { (k.mmvq)(self.raw(), w.ty.code() as i32, w.buf.ptr(), q.ptr(), y.fp(), kk, n, t as i64) })
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
        let hq8 = self.quantize(h.fp(), g.n_embd, t)?;
        let q = self.arena().f32(t * hq * d)?;
        let kr = self.arena().f32(t * hk * d)?;
        let vr = self.arena().f32(t * hk * d)?;
        self.matmul(&a.q, h, Some(&hq8), t, &q)?;
        self.matmul(&a.k, h, Some(&hq8), t, &kr)?;
        self.matmul(&a.v, h, Some(&hq8), t, &vr)?;
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
        let hq8 = self.quantize(h.fp(), g.n_embd, t)?;
        // the router (float32) and the choice on the host: sigmoid, the bias for the choice only, the chosen ones'
        // probabilities renormalized and scaled
        let logits = self.arena().f32(t * g.experts)?;
        self.matmul(&m.router, h, None, t, &logits)?;
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        let lat = self.arena().f32(t * g.latent)?;
        self.matmul(&m.to_latent, h, Some(&hq8), t, &lat)?;
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
        tap(&format!("ffn_moe_out-{l}"), &mo)?;
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul(&m.from_latent, &mo, None, t, &out)?;
        // the shared expert on the full width
        let su = self.arena().f32(t * g.n_ff_shared)?;
        self.matmul(&m.sh_up, h, Some(&hq8), t, &su)?;
        // SAFETY: n floats.
        self.op("relu2", || unsafe { (k.relu2)(self.raw(), su.fp(), (t * g.n_ff_shared) as i64) })?;
        let sd = self.arena().f32(t * g.n_embd)?;
        self.matmul(&m.sh_down, &su, None, t, &sd)?;
        tap(&format!("ffn_shexp-{l}"), &sd)?;
        self.add(&out, &sd, t * g.n_embd)?;
        Ok(out)
    }

    /// One pass of at most CHUNK rows: the last `n_out` rows' logits
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
        if t > CHUNK {
            return Err(e(format!("a pass of {t} rows; it takes {CHUNK}")));
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
        for c in tokens.chunks(CHUNK) {
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
