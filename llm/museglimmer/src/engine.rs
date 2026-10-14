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
    /// Q8_0 repacked by the silo (its quants, then its scales: silo.h)
    pub packed: bool,
}

struct Layer {
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

/// A conversation: per layer its key / value caches ([kv heads][slots][128] half, slot = position % slots)
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<(DevBuf, DevBuf, usize)>,
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
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), log: &mut dyn FnMut(String)) -> Result<MuseGlimmer<'g>> {
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
        let w16_bytes = [g.n_ff * g.n_embd, g.heads * g.head * g.n_embd].into_iter().max().unwrap_or(0) * 2;
        let mut free = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let (total, fr) = gpu.memory()?;
            free.push(fr.unwrap_or(total));
        }
        let margin: u64 = 1536 << 20;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let ctx = if kv.0 > 0 { kv.0 } else { 32768 };
        let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        // the plan for a chunk: the last GPU first (the server orders them weakest first), each taking the last layers
        // left while they fit; the first the rest
        let plan = |c: usize| -> std::result::Result<Vec<(usize, usize)>, String> {
            let n = gpus.len();
            let cost: Vec<u64> = (0..g.layers).map(|l| layer_w[l] + cache_bytes(&g, l, ctx, c) as u64).collect();
            let room: Vec<i64> = (0..n).map(|i| {
                let mut over = margin + (512 << 20) + w16_bytes as u64 + arena_bytes(&g, c) as u64;
                if i + 1 == n {
                    over += head_w;
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
        let (gpus, bounds): (Vec<Arc<Gpu>>, Vec<(usize, usize)>) = gpus.iter().cloned().zip(bounds).filter(|(_, b)| b.1 > b.0).unzip();
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
            let mut read = |name: &str| -> Result<Mat> { vec_mat(f, gpu, k, silo, name, &mut loaded) };
            layers.push(Layer {
                attn_norm: read(&n("attn_norm"))?.buf,
                q: read(&n("attn_q"))?,
                k: read(&n("attn_k"))?,
                v: read(&n("attn_v"))?,
                gate: read(&n("attn_gate"))?,
                o: read(&n("attn_output"))?,
                q_norm: read(&n("attn_q_norm"))?.buf,
                k_norm: read(&n("attn_k_norm"))?.buf,
                post_attn: read(&n("post_attention_norm"))?.buf,
                ffn_norm: read(&n("ffn_norm"))?.buf,
                ffn_gate: read(&n("ffn_gate"))?,
                ffn_up: read(&n("ffn_up"))?,
                ffn_down: read(&n("ffn_down"))?,
                post_ffn: read(&n("post_ffw_norm"))?.buf,
            });
            if l % 13 == 12 {
                log(format!("museglimmer: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, gib(loaded), t0.elapsed().as_secs_f64()));
            }
        }
        let output_norm = vec_mat(f, &lastg, k, silo, "output_norm.weight", &mut loaded)?.buf;
        let output = vec_mat(f, &lastg, k, silo, if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }, &mut loaded)?;
        let ones = DevBuf::f32(&gpus[0], g.n_embd)?;
        ones.write(0, &vec![1.0f32; g.n_embd].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let mut devs = Vec::with_capacity(gpus.len());
        for (gpu, &(lb, le)) in gpus.iter().zip(&bounds) {
            gpu.sync()?;
            devs.push(Dev { gpu: gpu.clone(), w16: DevBuf::new(gpu, w16_bytes)?, arena: Arena::new(gpu, arena_bytes(&g, chunk))?, lb, le });
        }
        let load_s = t0.elapsed().as_secs_f64();
        let wh = devs.iter().map(|d| if d.lb == 0 && d.le == g.layers { d.gpu.name.clone() } else { format!("{} (layers {}-{})", d.gpu.name, d.lb, d.le - 1) })
            .collect::<Vec<_>>().join(", ");
        log(format!("museglimmer: {} layers of {} ({} windowed), {:.1} GiB on {wh} in {load_s:.0} s; prompt chunks of {chunk}", g.layers, g.n_embd,
                    g.swa.iter().filter(|s| **s).count(), gib(loaded)));
        Ok(MuseGlimmer { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(Vec::new()), k, silo, layers, ones, output_norm, output, embd, chunk,
                         work: Mutex::new(()), prof: std::env::var("NS_MG_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(Default::default())), load_s,
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

    /// y [t, w.rows] = x [t, w.cols] . W^T
    fn matmul(&self, w: &Mat, x: &DevBuf, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        let (rows, cols) = (w.rows, w.cols);
        if w.ty == GType::F32 {
            // SAFETY: x [t, cols], the matrix [rows, cols] float32, y [t, rows].
            return self.op("gemm", || unsafe { (k.gemm)(self.raw(), t as i64, n, kk, x.fp(), kk, w.buf.fp(), y.fp(), n, 0) });
        }
        if let (true, Some(s)) = (w.packed, self.silo) {
            if t <= 8 {
                // SAFETY: arithmetic only.
                let xq = self.arena().bytes(unsafe { (s.xq_bytes)(kk, t as i64) })?;
                // SAFETY: x [t, cols] -> its split Q8 form; then the product from the repacked matrix.
                self.op("xq silo", || unsafe { (s.xq)(self.raw(), x.fp(), xq.ptr(), kk, t as i64) })?;
                return self.op("mmvq silo", || unsafe { (s.mmvq)(self.raw(), w.buf.ptr(), xq.ptr(), y.fp(), kk, n, t as i64) });
            }
            if rows * cols * 2 > self.dev().w16.len {
                return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
            }
            // SAFETY: the repacked matrix -> rows x cols halfs in w16.
            self.op("q8r f16 silo", || unsafe { (s.q8r_f16)(self.raw(), w.buf.ptr(), n, kk, self.dev().w16.ptr().cast()) })?;
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

    /// Layer `l`'s attention on h [t, n_embd] (normed): its output [t, n_embd], before the post norm
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, l: usize, a: &Layer, kc: &DevBuf, vc: &DevBuf, cap: usize, h: &DevBuf, t: usize, p0: usize, tap: Tap) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        let q = self.arena().f32(t * hq * d)?;
        let kr = self.arena().f32(t * hk * d)?;
        let vr = self.arena().f32(t * hk * d)?;
        let gate = self.arena().f32(t * hq * d)?;
        self.matmul(&a.q, h, t, &q)?;
        self.matmul(&a.k, h, t, &kr)?;
        self.matmul(&a.v, h, t, &vr)?;
        self.matmul(&a.gate, h, t, &gate)?;
        let rope = g.swa[l] as c_int;
        let win = g.win(l) as i64;
        let att = self.arena().f32(t * hq * d)?;
        // SAFETY: the rows and caches as named in mg.h.
        unsafe {
            self.op("q norm rope", || (k.qk_norm_rope)(self.raw(), q.fp(), (hq * d) as i64, q.fp(), (hq * d) as i64, t as i64, hq as i64, d as i64, a.q_norm.fp(),
                                                       g.eps, rope, g.theta, p0 as i64))?;
            self.op("k norm rope", || (k.qk_norm_rope)(self.raw(), kr.fp(), (hk * d) as i64, kr.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64, a.k_norm.fp(),
                                                       g.eps, rope, g.theta, p0 as i64))?;
            // llama.cpp's names: after the rotation on the windowed layers, after the norm on the full ones
            let stage = if g.swa[l] { "rope" } else { "normed" };
            tap(&format!("Qcur_{stage}-{l}"), &q)?;
            tap(&format!("Kcur_{stage}-{l}"), &kr)?;
            self.op("kv store", || (k.kv_store)(self.raw(), kr.fp(), vr.fp(), t as i64, hk as i64, d as i64, cap as i64, p0 as i64, kc.ptr(), vc.ptr()))?;
            let sc = (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64, win) as usize;
            let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
            self.op("attention", || (k.attn)(self.raw(), q.fp(), kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64, p0 as i64, win,
                                             att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
            tap(&format!("attn_out-{l}"), &att)?;
            self.op("output gate", || (k.gate_mul)(self.raw(), att.fp(), gate.fp(), (t * hq * d) as i64))?;
        }
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul(&a.o, &att, t, &out)?;
        Ok(out)
    }

    /// One pass of at most a chunk: the last `n_out` rows' logits
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
        let emb = self.embed(tokens)?;
        let mut x = self.arena().f32(t * ne)?;
        self.rms(&emb, &self.ones, &x, t, ne, g.eps)?;
        tap("embd_norm", &x)?;
        let mut h = self.arena().f32(t * ne)?;
        let mut ffn_g = self.arena().f32(t * g.n_ff)?;
        let mut ffn_u = self.arena().f32(t * g.n_ff)?;
        let mut tmp = self.arena().f32(t * ne)?;
        for (l, (ly, (kc, vc, cap))) in self.layers.iter().zip(s.layers.iter()).enumerate() {
            // the next GPU's layers: the rows cross over, its own buffers for them
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                x = self.hop(&x, self.layer_dev[l])?;
                h = self.arena().f32(t * ne)?;
                ffn_g = self.arena().f32(t * g.n_ff)?;
                ffn_u = self.arena().f32(t * g.n_ff)?;
                tmp = self.arena().f32(t * ne)?;
            }
            let mark = self.arena().mark();
            self.rms(&x, &ly.attn_norm, &h, t, ne, g.eps)?;
            let out = self.attention(l, ly, kc, vc, *cap, &h, t, p0, &mut *tap)?;
            self.rms(&out, &ly.post_attn, &tmp, t, ne, POST_EPS)?;
            self.add(&x, &tmp, t * ne)?;
            tap(&format!("ffn_inp-{l}"), &x)?;
            self.rms(&x, &ly.ffn_norm, &h, t, ne, g.eps)?;
            self.matmul(&ly.ffn_gate, &h, t, &ffn_g)?;
            self.matmul(&ly.ffn_up, &h, t, &ffn_u)?;
            let nf = t * g.n_ff;
            // SAFETY: n floats each.
            self.op("swiglu", || unsafe { (self.k.swiglu_clamp)(self.raw(), ffn_g.fp(), ffn_u.fp(), ffn_g.fp(), nf as i64, f32::INFINITY) })?;
            self.matmul(&ly.ffn_down, &ffn_g, t, &out)?;
            self.rms(&out, &ly.post_ffn, &tmp, t, ne, POST_EPS)?;
            self.add(&x, &tmp, t * ne)?;
            tap(&format!("l_out-{l}"), &x)?;
            self.arena().rewind(mark);
        }
        s.pos += t;
        let n = n_out.clamp(1, t);
        let last = x.view((t - n) * ne * 4, n * ne * 4)?;
        let normed = self.arena().f32(n * ne)?;
        self.rms(&last, &self.output_norm, &normed, n, ne, g.eps)?;
        tap("result_norm", &normed)?;
        let logits = self.arena().f32(n * self.output.rows)?;
        self.matmul(&self.output, &normed, n, &logits)?;
        let mut all = logits.to_f32()?;
        // the output multiplier, then the soft cap
        let (sc, cap) = (g.logit_scale, g.softcap);
        for v in all.iter_mut() {
            *v *= sc;
            if cap > 0.0 {
                *v = cap * (*v / cap).tanh();
            }
        }
        Ok(all.chunks(self.output.rows).map(|c| c.to_vec()).collect())
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
        Ok(Session { pos: 0, max_ctx, layers })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        s.pos = 0;
        Ok(())
    }

    /// Positions [from, to) of one cache (cap slots) into another (dcap slots), head by head, in runs that wrap neither
    #[allow(clippy::too_many_arguments)]
    fn copy_ring(&self, dst: &DevBuf, dcap: usize, src: &DevBuf, scap: usize, from: usize, to: usize) -> Result<()> {
        let g = &self.g;
        let row = g.head * 2;
        for h in 0..g.heads_kv {
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

    /// The positions of layer `l`'s cache a session at `pos` still reads: its window's, or all
    fn kept(&self, l: usize, pos: usize) -> usize {
        let w = self.g.win(l);
        if w > 0 { pos.saturating_sub(w - 1) } else { 0 }
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        for (l, (d, s)) in dst.layers.iter().zip(&src.layers).enumerate() {
            let from = self.kept(l, src.pos);
            self.copy_ring(&d.0, d.2, &s.0, s.2, from, src.pos)?;
            self.copy_ring(&d.1, d.2, &s.1, s.2, from, src.pos)?;
        }
        dst.pos = src.pos;
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
        if keep > s.pos {
            return Err(e(format!("a rollback to {keep} of a session at {}", s.pos)));
        }
        s.pos = keep;
        Ok(())
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    /// The next committed token: drawn from the logits of the last pass, the one before fed first
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let l = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => l,
            (None, Some(x)) => self.pass(s, &[x], 1, tap)?.pop().unwrap_or_default(),
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        let y = smp.sample(&l);
        d.next = Some(y);
        Ok(vec![y])
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let g = &self.g;
        let row = g.head * 2;
        let mut bufs = Vec::new();
        for (l, (kc, vc, cap)) in s.layers.iter().enumerate() {
            let from = self.kept(l, s.pos);
            let n = s.pos - from;
            for c in [kc, vc] {
                let mut b = vec![0u8; g.heads_kv * n * row];
                for h in 0..g.heads_kv {
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
        let g = &self.g;
        let row = g.head * 2;
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        let mut it = ck.bufs.iter();
        for (l, (kc, vc, cap)) in s.layers.iter().enumerate() {
            let from = self.kept(l, ck.pos);
            let n = ck.pos - from;
            for c in [kc, vc] {
                let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                if b.len() != g.heads_kv * n * row {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..g.heads_kv {
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
        return Ok(Mat { buf: out, ty: GType::F32, rows, cols, packed: false });
    }
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    if let (GType::Q8_0, Some(s), true) = (t.ty, silo, cols.is_multiple_of(32)) {
        // SAFETY: arithmetic only.
        let packed = DevBuf::new(gpu, unsafe { (s.q8r_bytes)(rows as i64, cols as i64) })?;
        // SAFETY: buf holds the matrix's Q8_0 blocks, packed its repacked size.
        ffi::check(unsafe { (s.q8r_pack)(gpu.raw(), buf.ptr(), rows as i64, cols as i64, packed.ptr()) }, name)?;
        gpu.sync()?;
        return Ok(Mat { buf: packed, ty: t.ty, rows, cols, packed: true });
    }
    Ok(Mat { buf, ty: t.ty, rows, cols, packed: false })
}
