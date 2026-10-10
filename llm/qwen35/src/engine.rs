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
    /// rows of the last forward pass (a rollback within them is not supported: no snapshots)
    last_rows: usize,
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

pub struct Qwen35<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    gpu: Arc<Gpu>,
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
    arena: Arena,
    /// the largest matrix expanded to half (a prompt chunk's products)
    w16: DevBuf,
    work: Mutex<()>,
    /// NS_Q35_PROFILE=1: every kernel timed on the device, with its bytes and arithmetic (probe.rs)
    probe: Option<Probe>,
    load_s: f64,
    load_bytes: u64,
}

fn chunk_size() -> usize {
    std::env::var("NS_Q35_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= 16).unwrap_or(512)
}

impl<'g> Qwen35<'g> {
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<Qwen35<'g>> {
        let t0 = Instant::now();
        let gpu = gpus.first().cloned().ok_or_else(|| e("no GPU"))?;
        if gpus.len() > 1 {
            log(format!("qwen35: one GPU ({}); the others are not used", gpu.name));
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
        let chunk = chunk_size();
        let w16_bytes = [g.n_ff * g.n_embd, g.heads * 2 * g.head * g.n_embd, g.conv_dim() * g.n_embd].into_iter().max().unwrap_or(0) * 2;
        let arena_bytes = chunk * (4 * g.n_embd + 2 * g.n_ff + g.conv_dim() + g.inner + 4 * g.v_heads * g.state + g.heads * 2 * g.head + 3 * g.heads * g.head) * 4
            + chunk * g.n_ff.max(g.conv_dim()) * 2 + 8 * g.vocab * 4 + (64 << 20);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        let need = weights + w16_bytes as u64 + arena_bytes as u64 + (1536 << 20);
        if need > free {
            return Err(e(format!("{} needs about {:.1} GiB on {} ({:.1} GiB of weights); {:.1} GiB are free", f.architecture(),
                                 need as f64 / (1u64 << 30) as f64, gpu.name, weights as f64 / (1u64 << 30) as f64, free as f64 / (1u64 << 30) as f64)));
        }
        let mut loaded = 0u64;
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let mut read = |name: &str| -> Result<Mat> { vec_mat(f, &gpu, k, name, &mut loaded) };
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
        let output_norm = vec_mat(f, &gpu, k, "output_norm.weight", &mut loaded)?.buf;
        let output = vec_mat(f, &gpu, k, if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }, &mut loaded)?;
        gpu.sync()?;
        let w16 = DevBuf::new(&gpu, w16_bytes)?;
        let arena = Arena::new(&gpu, arena_bytes)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("qwen35: {} layers of {} ({} full attention), {:.1} GiB on {} in {load_s:.0} s", g.layers, g.n_embd,
                    (0..g.layers).filter(|l| g.full(*l)).count(), loaded as f64 / (1u64 << 30) as f64, gpu.name));
        let probe = Probe::new(gpu.raw())?;
        Ok(Qwen35 { g, file: f, gpu, k, silo, layers, output_norm, output, embd, chunk, arena, w16, work: Mutex::new(()), probe, load_s, load_bytes: loaded })
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

    fn raw(&self) -> *mut std::ffi::c_void {
        self.gpu.raw()
    }

    /// y [t, w.rows] = x [t, w.cols] . W^T; `role` names it in the probe
    fn matmul(&self, role: &'static str, w: &Mat, x: &DevBuf, t: usize, y: &DevBuf) -> Result<()> {
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
            let qb = self.arena.bytes(q8)?;
            // SAFETY: x [t, cols] float32 -> its Q8_1 blocks.
            self.op("quantize", key, 4 * t * cols + q8, 0, || unsafe { (k.quantize_q8_1)(self.raw(), x.fp(), qb.ptr(), kk, t as i64) })?;
            // the silo's product for the types it covers (kernels/llm/qwen35/silo), the shared one otherwise
            // SAFETY: arithmetic only.
            let (f, name) = match self.silo {
                Some(s) if unsafe { (s.mmvq_supported)(w.ty.code() as i32, kk) } != 0 => (s.mmvq, "mmvq silo"),
                _ => (k.mmvq, "mmvq"),
            };
            // SAFETY: the matrix in its blocks, x's Q8_1 blocks, y [t, rows].
            return self.op(name, key, wb + q8 + 4 * t * rows, flops,
                           || unsafe { f(self.raw(), w.ty.code() as i32, w.buf.ptr(), qb.ptr(), y.fp(), kk, n, t as i64) });
        }
        // the matrix expanded to half (directly, or through float32), x to half, then the half product
        let elems = rows * cols;
        if elems * 2 > self.w16.len {
            return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
        }
        // SAFETY: the matrix's blocks -> rows x cols halfs in w16 (non-zero: a type it does not expand directly)
        let rc = self.op_rc("dequant f16", key, wb + 2 * elems, 0,
                            || unsafe { (k.dequant_f16)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems as i64, self.w16.ptr().cast()) })?;
        if rc != 0 {
            let f = self.arena.f32(elems)?;
            // SAFETY: the blocks -> float32, then -> half.
            self.op("dequant", key, wb + 4 * elems, 0, || unsafe { (k.dequant)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems, f.fp()) })?;
            self.op("to half", key, 6 * elems, 0, || unsafe { (k.to_f16)(self.raw(), f.fp(), self.w16.ptr().cast(), elems as i64) })?;
        }
        let xh = self.arena.bytes(t * cols * 2)?;
        // SAFETY: x [t, cols] -> half; then y [t, rows] = xh . w16^T.
        self.op("x to half", key, 6 * t * cols, 0, || unsafe { (k.to_f16)(self.raw(), x.fp(), xh.ptr().cast(), (t * cols) as i64) })?;
        self.op("gemm f16", key, 2 * t * cols + 2 * elems + 4 * t * rows, flops,
                || unsafe { (k.gemm_f16)(self.raw(), t as i64, n, kk, xh.ptr().cast(), kk, self.w16.ptr().cast(), y.fp(), n, 0) })
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
        let g = &self.g;
        let row = self.embd.ty.bytes(g.n_embd as u64).ok_or_else(|| e("the embedding's row size"))? as usize;
        let mut raw = vec![0u8; tokens.len() * row];
        for (i, t) in tokens.iter().enumerate() {
            if *t as usize >= g.vocab {
                return Err(e(format!("token {t} past the vocabulary ({})", g.vocab)));
            }
            self.file.read_into(&self.embd, *t as u64 * row as u64, &mut raw[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
        }
        let src = self.arena.bytes(raw.len())?;
        src.write(0, &raw)?;
        let x = self.arena.f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32.
        self.op("dequant", String::new, raw.len() + 4 * tokens.len() * g.n_embd, 0,
                || unsafe { (self.k.dequant)(self.raw(), self.embd.ty.code() as i32, src.ptr(), tokens.len() * g.n_embd, x.fp()) })?;
        Ok(x)
    }

    /// A full-attention layer on h [t, n_embd]: its output [t, n_embd]
    #[allow(clippy::too_many_arguments)]
    fn full(&self, a: &Full, kc: &DevBuf, vc: &DevBuf, cap: usize, h: &DevBuf, t: usize, p0: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        let qf = self.arena.f32(t * hq * 2 * d)?;
        let kr = self.arena.f32(t * hk * d)?;
        let vr = self.arena.f32(t * hk * d)?;
        self.matmul("q", &a.q, h, t, &qf)?;
        self.matmul("k", &a.k, h, t, &kr)?;
        self.matmul("v", &a.v, h, t, &vr)?;
        let q = self.arena.f32(t * hq * d)?;
        let kn = self.arena.f32(t * hk * d)?;
        let att = self.arena.f32(t * hq * d)?;
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
                let out = self.arena.f32(t * g.n_embd)?;
                self.matmul("o", &a.o, &att, t, &out)?;
                return Ok(out);
            }
            if let Some(s) = self.silo.filter(|s| (s.attn_decode_supported)(t as i64, hq as i64, hk as i64, d as i64) != 0) {
                // the silo's decode attention: a lane a key for the scores, the keys over ~512 work-groups
                let sc = (s.attn_decode_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64) as usize;
                let part = if sc > 0 { Some(self.arena.f32(sc)?) } else { None };
                self.op("attention silo", none, keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                        || (s.attn_decode)(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64,
                                           p0 as i64, att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
                self.op("output gate", none, 12 * t * hq * d, 4 * t * hq * d,
                        || (k.gate_mul)(self.raw(), att.fp(), qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64))?;
                let out = self.arena.f32(t * g.n_embd)?;
                self.matmul("o", &a.o, &att, t, &out)?;
                return Ok(out);
            }
            let sc = (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64) as usize;
            let part = if sc > 0 { Some(self.arena.f32(sc)?) } else { None };
            self.op("attention", none, keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                    || (k.attn)(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64, p0 as i64,
                                att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
            self.op("output gate", none, 12 * t * hq * d, 4 * t * hq * d,
                    || (k.gate_mul)(self.raw(), att.fp(), qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64))?;
        }
        let out = self.arena.f32(t * g.n_embd)?;
        self.matmul("o", &a.o, &att, t, &out)?;
        Ok(out)
    }

    /// A DeltaNet layer on h [t, n_embd]: its output [t, n_embd]
    fn linear(&self, a: &Linear, s: &DevBuf, conv: &DevBuf, h: &DevBuf, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hv, hk, d, cd) = (g.v_heads, g.k_heads, g.state, g.conv_dim());
        let qkv = self.arena.f32(t * cd)?;
        let z = self.arena.f32(t * g.inner)?;
        let beta = self.arena.f32(t * hv)?;
        let alpha = self.arena.f32(t * hv)?;
        self.matmul("qkv", &a.qkv, h, t, &qkv)?;
        self.matmul("z", &a.z, h, t, &z)?;
        self.matmul("beta", &a.beta, h, t, &beta)?;
        self.matmul("alpha", &a.alpha, h, t, &alpha)?;
        let eg = self.arena.f32(t * hv * d)?;
        let co = self.arena.f32(t * cd)?;
        let q = self.arena.f32(t * hv * d)?;
        let kk = self.arena.f32(t * hv * d)?;
        let v = self.arena.f32(t * hv * d)?;
        let o = self.arena.f32(t * hv * d)?;
        let y = self.arena.f32(t * g.inner)?;
        let qd = (hk * d) as i64;
        let none = String::new;
        let thd = t * hv * d;
        // SAFETY: the buffers as named in glm.h / q35.h.
        unsafe {
            self.op("gates", none, 4 * (3 * t * hv + 2 * hv) + 4 * thd, 20 * t * hv,
                    || (k.gdn_gates)(self.raw(), alpha.fp(), beta.fp(), a.dt.fp(), a.a.fp(), eg.fp(), t as i64, hv as i64, d as i64))?;
            self.op("conv silu", none, 4 * (2 * t * cd + 2 * (g.conv - 1) * cd + g.conv * cd), 2 * g.conv * t * cd + 4 * t * cd,
                    || (k.conv_silu)(self.raw(), qkv.fp(), conv.fp(), a.conv.fp(), co.fp(), t as i64, cd as i64, g.conv as i32, std::ptr::null_mut()))?;
            self.op("expand q", none, 4 * t * hk * d + 4 * thd, 0, || (k.expand)(self.raw(), co.fp(), cd as i64, q.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand k", none, 4 * t * hk * d + 4 * thd, 0,
                    || (k.expand)(self.raw(), co.fp().add(qd as usize), cd as i64, kk.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand v", none, 8 * thd, 0, || (k.expand)(self.raw(), co.fp().add(2 * qd as usize), cd as i64, v.fp(), t as i64, hv as i64, hv as i64, d as i64))?;
            self.op("l2 q", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), q.fp(), (t * hv) as i64, d as i64, g.eps))?;
            self.op("l2 k", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), kk.fp(), (t * hv) as i64, d as i64, g.eps))?;
            // the state [hv][d][d] read and written once, each row's q, k, v, decay in and output out; per row and head
            // the state decayed, read against k, updated, read against q: ~8 d^2
            self.op("delta rule", none, 8 * hv * d * d + 4 * (5 * thd + t * hv), 8 * t * hv * d * d,
                    || (k.kda_scan)(self.raw(), q.fp(), kk.fp(), v.fp(), eg.fp(), beta.fp(), s.fp(), o.fp(), t as i64, hv as i64, d as i64, std::ptr::null_mut()))?;
            self.op("gated norm", none, 4 * (3 * t * g.inner + d), 10 * t * g.inner,
                    || (k.gdn_out)(self.raw(), o.fp(), z.fp(), g.inner as i64, a.norm.fp(), y.fp(), t as i64, hv as i64, d as i64, g.eps))?;
        }
        let out = self.arena.f32(t * g.n_embd)?;
        self.matmul("out", &a.out, &y, t, &out)?;
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
        self.arena.reset();
        let p0 = s.pos;
        self.phase("embed");
        let x = self.embed(tokens)?;
        let h = self.arena.f32(t * g.n_embd)?;
        let ffn_g = self.arena.f32(t * g.n_ff)?;
        let ffn_u = self.arena.f32(t * g.n_ff)?;
        let ffn = self.arena.f32(t * g.n_embd)?;
        // the arena past this point is the layer's own, made again each layer
        for (l, (ly, st)) in self.layers.iter().zip(s.layers.iter()).enumerate() {
            let mark = self.arena.mark();
            self.phase(if ly.full.is_some() { "attention" } else { "deltanet" });
            self.rms(&x, &ly.attn_norm, &h, t, g.n_embd)?;
            let out = match (st, &ly.full, &ly.linear) {
                (LState::Full { k, v }, Some(a), _) => self.full(a, k, v, s.max_ctx, &h, t, p0)?,
                (LState::Linear { s: rs, conv }, _, Some(a)) => self.linear(a, rs, conv, &h, t)?,
                _ => return Err(e(format!("layer {l}: its state and its weights disagree"))),
            };
            self.add(&x, &out, t * g.n_embd)?;
            self.phase("ffn");
            self.rms(&x, &ly.post_norm, &h, t, g.n_embd)?;
            self.matmul("gate", &ly.gate, &h, t, &ffn_g)?;
            self.matmul("up", &ly.up, &h, t, &ffn_u)?;
            let nf = t * g.n_ff;
            // SAFETY: n floats each.
            self.op("swiglu", String::new, 12 * nf, 6 * nf,
                    || unsafe { (self.k.swiglu_clamp)(self.raw(), ffn_g.fp(), ffn_u.fp(), ffn_g.fp(), nf as i64, f32::INFINITY) })?;
            self.matmul("down", &ly.down, &ffn_g, t, &ffn)?;
            self.add(&x, &ffn, t * g.n_embd)?;
            tap(&format!("l_out-{l}"), &x)?;
            self.arena.rewind(mark);
        }
        s.pos += t;
        s.last_rows = t;
        // the last rows through the output norm and head
        let n = n_out.clamp(1, t);
        let last = x.view((t - n) * g.n_embd * 4, n * g.n_embd * 4)?;
        let normed = self.arena.f32(n * g.n_embd)?;
        self.phase("head");
        self.rms(&last, &self.output_norm, &normed, n, g.n_embd)?;
        tap("result_norm", &normed)?;
        let logits = self.arena.f32(n * self.output.rows)?;
        self.matmul("output", &self.output, &normed, n, &logits)?;
        let all = logits.to_f32()?;
        if let Some(p) = &self.probe {
            p.settle(t)?;
        }
        Ok(all.chunks(self.output.rows).map(|c| c.to_vec()).collect())
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        // the caches and states, checked against what is free first: past the card the xe driver spills into host
        // memory (no out-of-memory error) and can take the machine down
        let full = (0..g.layers).filter(|l| g.full(*l)).count();
        let need = (full * 2 * g.heads_kv * max_ctx * g.head * 2 + (g.layers - full) * (g.v_heads * g.state * g.state + (g.conv - 1) * g.conv_dim()) * 4) as u64;
        let (total, free) = self.gpu.memory()?;
        let free = free.unwrap_or(total);
        if need + (1536 << 20) > free {
            return Err(e(format!("a session of {max_ctx} tokens needs {:.2} GiB on {}; {:.2} GiB are free (1.5 GiB kept spare)", need as f64 / (1u64 << 30) as f64,
                                 self.gpu.name, free as f64 / (1u64 << 30) as f64)));
        }
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            layers.push(if g.full(l) {
                let n = g.heads_kv * max_ctx * g.head * 2;
                LState::Full { k: DevBuf::new(&self.gpu, n)?, v: DevBuf::new(&self.gpu, n)? }
            } else {
                let s = DevBuf::f32(&self.gpu, g.v_heads * g.state * g.state)?;
                let conv = DevBuf::f32(&self.gpu, (g.conv - 1) * g.conv_dim())?;
                s.fill(0)?;
                conv.fill(0)?;
                LState::Linear { s, conv }
            });
        }
        Ok(Session { pos: 0, max_ctx, layers, last_rows: 0 })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        for st in &s.layers {
            if let LState::Linear { s, conv } = st {
                s.fill(0)?;
                conv.fill(0)?;
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
        dst.pos = src.pos;
        dst.last_rows = 0;
        Ok(())
    }

    pub fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        let mut logits = Vec::new();
        let mut done = 0;
        for c in tokens.chunks(self.chunk) {
            if done > 0 && stop() {
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
        if keep == s.last_rows {
            return Ok(());
        }
        Err(e("qwen35: a rollback inside a pass needs the recurrent states' snapshots (verify passes are off without a draft model)"))
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let logits = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => l,
            (None, Some(x)) => self.pass(s, &[x], 1, tap)?.pop().unwrap_or_default(),
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        let y = smp.sample(&logits);
        d.next = Some(y);
        Ok(vec![y])
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
        s.pos = ck.pos;
        s.last_rows = 0;
        Ok(())
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        let (total, free) = self.gpu.memory().unwrap_or((0, None));
        vec![GpuInfo { index: 0, name: self.gpu.name.clone(), pci: None, total, free, layers: (0, self.g.layers as u64), expert_slots: 0, host_slots: 0 }]
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
