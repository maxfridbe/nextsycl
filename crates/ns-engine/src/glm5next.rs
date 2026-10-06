//! GLM-5.3-Flash on one GPU, bring-up form: every step of `docs/glm5next.md` in order, float32 activations, the
//! plain kernels of `kernels/ns/glm.cpp`. Everything but the routed experts is uploaded once in its stored form and
//! expanded per use; the experts a batch routes to are read from the file per layer (the expert store replaces
//! this). Prompt only, from position 0; MLA attends to every earlier token (the indexer selects all of them up to
//! ~2,048 tokens of context, docs/glm5next.md).

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_core::{DevBuf, Error, Gpu, Ops, Result};
use ns_gguf::{GType, Gguf, Tensor};
use ns_model::glm5next::{Model, Role, Scheme};

use crate::Tap;

/// float32 values expanded per matrix chunk (128 MiB)
const SCRATCH: usize = 32 << 20;
/// the decode kernels take up to this many rows (tokens) at once
const MMVQ_COLS: usize = 8;
/// widest matrix input (MLA's output projection)
const MAX_COLS: usize = 16384;
/// expert slots are allocated in chunks of this size (single device allocations stay small)
const CHUNK: usize = 2 << 30;

/// A matrix on the GPU: `rows` x `cols` (a tensor's outer dimensions folded into rows), in its stored quantized
/// form, or - for the 16- and 32-bit ones - as float32.
struct Mat {
    buf: DevBuf,
    ty: GType,
    rows: usize,
    cols: usize,
    /// expanded at load (BF16 / F16 / F32 matrices: small, multiplied as they are)
    f32: bool,
}

/// The experts of a layer in a slot: [gate | up | down], each matrix's stored bytes.
struct ExpertParts {
    gate: (usize, usize, GType),
    up: (usize, usize, GType),
    down: (usize, usize, GType),
}

/// Routed experts resident on the GPU: fixed-size slots, least recently used out. Filled from the file.
struct ExpertCache {
    chunks: Vec<DevBuf>,
    slot_bytes: usize,
    per_chunk: usize,
    map: HashMap<(u64, u64), usize>,
    owner: Vec<Option<(u64, u64)>>,
    used: Vec<u64>,
    tick: u64,
    pub hits: u64,
    pub misses: u64,
}

impl Mat {
    fn row_bytes(&self) -> usize {
        self.ty.bytes(self.cols as u64).unwrap_or(0) as usize
    }
}

/// A conversation's state on the GPU: per KDA layer the recurrent state and the convolutions' last inputs, per MLA
/// layer the latent cache; `pos` tokens so far.
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<LayerState>,
}

enum LayerState {
    Kda { s: DevBuf, conv: [DevBuf; 3] },
    Mla { c: DevBuf },
}

pub struct Glm<'g> {
    pub m: Model<'g>,
    ops: Ops,
    mats: BTreeMap<(u64, Role), Mat>,
    vecs: BTreeMap<(u64, Role), DevBuf>,
    scratch: DevBuf,
    /// Q8_1 of up to MMVQ_COLS rows of MAX_COLS
    q8: DevBuf,
    experts: Mutex<ExpertCache>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// slots the expert cache holds
    pub expert_slots: usize,
}

const VECTORS: [Role; 18] = [Role::OutputNorm, Role::AttnNorm, Role::FfnNorm, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnBase, Role::HcFfnScale, Role::KdaQConv,
                             Role::KdaKConv, Role::KdaVConv, Role::KdaDtBias, Role::KdaA, Role::KdaONorm, Role::MlaQANorm, Role::MlaKvANorm,
                             Role::IdxKNorm, Role::IdxKNormBias, Role::RouterBias];

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

impl<'g> Glm<'g> {
    /// Loads everything but the routed experts (and the MTP block) onto `gpu`; `expert_bytes` of it for routed
    /// experts (None: what the card has free less 3 GiB).
    pub fn load(file: &'g Gguf, gpu: &Arc<Gpu>, expert_bytes: Option<usize>, log: &mut dyn FnMut(String)) -> Result<Glm<'g>> {
        let t0 = Instant::now();
        let m = Model::open(file).map_err(e)?;
        let ops = Ops { gpu: gpu.clone() };
        let scratch = DevBuf::f32(gpu, SCRATCH)?;
        let mut mats = BTreeMap::new();
        let mut vecs = BTreeMap::new();
        let mut bytes = 0u64;
        let upload = |t: &Tensor| -> Result<DevBuf> {
            let b = DevBuf::new(gpu, t.bytes as usize)?;
            b.write(0, &file.read(t).map_err(e)?)?;
            Ok(b)
        };
        let mut layers: Vec<(u64, Vec<Role>)> = vec![(0, vec![Role::TokenEmbd, Role::OutputNorm, Role::Output])];
        layers.extend((0..m.g.n_layer).map(|l| (l, m.roles(l))));
        for (l, roles) in layers {
            for r in roles {
                if matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown | Role::TokenEmbd) {
                    continue; // experts per batch; the embedding's rows are read per token
                }
                let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                let raw = upload(t)?;
                bytes += t.bytes;
                let n = t.elements() as usize;
                if VECTORS.contains(&r) || matches!(r, Role::MlaKB | Role::MlaVB) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    if r == Role::KdaA && m.scheme == Scheme::Ds4 {
                        // ds4 stores A_log; the gate wants A = -exp(A_log) (llama.cpp's ssm_a holds that already)
                        let v: Vec<f32> = f.to_f32()?.iter().map(|a| -a.exp()).collect();
                        f.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                        gpu.sync()?;
                    }
                    vecs.insert((l, r), f);
                } else {
                    let cols = *t.shape.last().unwrap_or(&1) as usize;
                    if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
                        let f = DevBuf::f32(gpu, n)?;
                        ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                        mats.insert((l, r), Mat { buf: f, ty: GType::F32, rows: n / cols, cols, f32: true });
                    } else {
                        mats.insert((l, r), Mat { buf: raw, ty: t.ty, rows: n / cols, cols, f32: false });
                    }
                }
            }
            if l % 5 == 4 {
                log(format!("loaded through block {l}: {:.2} GiB", bytes as f64 / (1u64 << 30) as f64));
            }
        }
        gpu.sync()?;
        let q8 = DevBuf::new(gpu, ops.q8_1_bytes(MAX_COLS, MMVQ_COLS))?;
        // the expert cache: slots of the largest layer's [gate | up | down], in 2 GiB chunks
        let slot_bytes = (m.g.n_dense..m.g.n_layer).map(|l| m.expert_bytes(l) as usize).max().unwrap_or(0).next_multiple_of(256);
        let budget = match expert_bytes {
            Some(b) => b,
            None => gpu.memory()?.1.map_or(8usize << 30, |f| (f as usize).saturating_sub(3 << 30)),
        };
        let per_chunk = (CHUNK / slot_bytes.max(1)).max(1);
        let total = budget / slot_bytes.max(1);
        let mut chunks = Vec::new();
        let mut have = 0;
        while have < total {
            let n = per_chunk.min(total - have);
            chunks.push(DevBuf::new(gpu, n * slot_bytes)?);
            have += n;
        }
        let experts = Mutex::new(ExpertCache {
            chunks, slot_bytes, per_chunk, map: HashMap::new(), owner: vec![None; have], used: vec![0; have], tick: 0, hits: 0, misses: 0,
        });
        log(format!("expert cache: {have} slots of {:.1} MiB ({:.1} GiB)", slot_bytes as f64 / 1048576.0, (have * slot_bytes) as f64 / (1u64 << 30) as f64));
        Ok(Glm { m, ops, mats, vecs, scratch, q8, experts, load_seconds: t0.elapsed().as_secs_f64(), load_bytes: bytes, expert_slots: have })
    }

    /// (hits, misses) of the expert cache so far
    pub fn expert_stats(&self) -> (u64, u64) {
        let c = self.experts.lock().unwrap();
        (c.hits, c.misses)
    }

    fn parts(&self, l: u64) -> Result<ExpertParts> {
        let mut off = 0;
        let mut part = |r: Role| -> Result<(usize, usize, GType)> {
            let t = self.m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
            let b = (t.bytes / self.m.g.n_expert) as usize;
            let p = (off, b, t.ty);
            off += b;
            Ok(p)
        };
        Ok(ExpertParts { gate: part(Role::ExpGate)?, up: part(Role::ExpUp)?, down: part(Role::ExpDown)? })
    }

    /// The slot holding expert `ex` of layer `l` (read from the file on a miss): (chunk, byte offset).
    fn expert_slot(&self, l: u64, ex: u64) -> Result<(usize, usize)> {
        let mut c = self.experts.lock().unwrap();
        if c.owner.is_empty() {
            return Err(Error("the expert cache has no slots (no room on the GPU)".into()));
        }
        c.tick += 1;
        let tick = c.tick;
        let slot = match c.map.get(&(l, ex)) {
            Some(&s) => {
                c.hits += 1;
                s
            }
            None => {
                c.misses += 1;
                // a free slot, else the least recently used
                let s = (0..c.owner.len()).min_by_key(|&i| if c.owner[i].is_none() { 0 } else { c.used[i] + 1 }).unwrap();
                if let Some(old) = c.owner[s].take() {
                    c.map.remove(&old);
                }
                let p = self.parts(l)?;
                let mut host = vec![0u8; p.down.0 + p.down.1];
                for (r, (o, n, _)) in [(Role::ExpGate, p.gate), (Role::ExpUp, p.up), (Role::ExpDown, p.down)] {
                    let t = self.m.tensor(l, r).ok_or("expert tensor")?;
                    self.m.file.read_into(t, ex * n as u64, &mut host[o..o + n]).map_err(e)?;
                }
                let (ch, at) = (s / c.per_chunk, (s % c.per_chunk) * c.slot_bytes);
                c.chunks[ch].write(at, &host)?;
                c.owner[s] = Some((l, ex));
                c.map.insert((l, ex), s);
                s
            }
        };
        c.used[slot] = tick;
        Ok((slot / c.per_chunk, (slot % c.per_chunk) * c.slot_bytes))
    }

    fn mat(&self, l: u64, r: Role) -> Result<&Mat> {
        self.mats.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: matrix {r:?} not loaded")))
    }
    fn vec(&self, l: u64, r: Role) -> Result<&DevBuf> {
        self.vecs.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: vector {r:?} not loaded")))
    }

    /// y[t rows from yoff, ldy apart] (+)= x . W^T. Float matrices multiply as they are; quantized ones from their
    /// blocks for up to MMVQ_COLS contiguous rows (decode), else expanded in row chunks.
    #[allow(clippy::too_many_arguments)]
    fn matmul(&self, w: &Mat, t: usize, x: (&DevBuf, usize, usize), y: (&DevBuf, usize, usize), acc: bool) -> Result<()> {
        if w.f32 {
            return self.ops.gemm_at(t, w.rows, w.cols, x, (&w.buf, 0), y, acc);
        }
        if !acc && t <= MMVQ_COLS && x.2 == w.cols && y.2 == w.rows && w.cols <= MAX_COLS && self.ops.mmvq_supported(w.ty.code()) {
            self.ops.quantize_q8_1((x.0, x.1), &self.q8, w.cols, t)?;
            return self.ops.mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.q8, (y.0, y.1), w.cols, w.rows, t);
        }
        let chunk = (SCRATCH / w.cols).max(1).min(w.rows);
        let rb = w.row_bytes();
        let mut r0 = 0;
        while r0 < w.rows {
            let r = chunk.min(w.rows - r0);
            self.ops.dequant(w.ty.code(), &w.buf, r0 * rb, r * rb, r * w.cols, &self.scratch)?;
            self.ops.gemm_at(t, r, w.cols, x, (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
            r0 += r;
        }
        Ok(())
    }

    /// y [t, rows] = x [t, cols] . W^T (contiguous)
    fn mm(&self, l: u64, r: Role, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let w = self.mat(l, r)?;
        let y = DevBuf::f32(&self.ops.gpu, t * w.rows)?;
        self.matmul(w, t, (x, 0, w.cols), (&y, 0, w.rows), false)?;
        Ok(y)
    }

    /// One routed expert's matrix (`r` = ExpGate / ExpUp / ExpDown of layer `l`, expert `ex`) applied to x [n, cols],
    /// from its slot in the expert cache.
    fn expert_mm(&self, l: u64, r: Role, ex: u64, x: &DevBuf, n: usize) -> Result<DevBuf> {
        let t = self.m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
        let (rows, cols) = (t.shape[1] as usize, t.shape[2] as usize);
        let p = self.parts(l)?;
        let (off, bytes, ty) = match r {
            Role::ExpGate => p.gate,
            Role::ExpUp => p.up,
            _ => p.down,
        };
        let (ch, at) = self.expert_slot(l, ex)?;
        let c = self.experts.lock().unwrap();
        let buf = &c.chunks[ch];
        let y = DevBuf::f32(&self.ops.gpu, n * rows)?;
        if n <= MMVQ_COLS && self.ops.mmvq_supported(ty.code()) {
            self.ops.quantize_q8_1((x, 0), &self.q8, cols, n)?;
            self.ops.mmvq(ty.code(), (buf, at + off), bytes, &self.q8, (&y, 0), cols, rows, n)?;
        } else {
            self.ops.dequant(ty.code(), buf, at + off, bytes, rows * cols, &self.scratch)?;
            self.ops.gemm_at(n, rows, cols, (x, 0, cols), (&self.scratch, 0), (&y, 0, rows), false)?;
        }
        Ok(y)
    }

    /// A new conversation, up to `max_ctx` tokens (MLA attends to every earlier token: exact up to ~2,048).
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.m.g;
        let gpu = &self.ops.gpu;
        let kw = (g.kda_heads * g.kda_dim) as usize;
        let mut layers = Vec::new();
        for l in 0..g.n_layer {
            layers.push(if g.is_mla(l) {
                LayerState::Mla { c: DevBuf::f32(gpu, max_ctx * g.kv_lora as usize)? }
            } else {
                let s = DevBuf::f32(gpu, (g.kda_heads * g.kda_dim * g.kda_dim) as usize)?;
                s.fill(0)?;
                let conv = [0, 1, 2].map(|_| DevBuf::f32(gpu, (g.kda_conv as usize - 1) * kw));
                let [a, b, c] = conv;
                let conv = [a?, b?, c?];
                for cb in &conv {
                    cb.fill(0)?;
                }
                LayerState::Kda { s, conv }
            });
        }
        Ok(Session { pos: 0, max_ctx, layers })
    }

    /// The next `tokens` of a conversation: the logits of the last one. `tap` sees each named step.
    pub fn forward(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        let g = &self.m.g;
        let gpu = &self.ops.gpu;
        let o = &self.ops;
        let t = tokens.len();
        let pos0 = sess.pos;
        if pos0 + t > sess.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", sess.max_ctx, t)));
        }
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let kw = (g.kda_heads * g.kda_dim) as usize;
        let (kh, kd) = (g.kda_heads as usize, g.kda_dim as usize);

        // the embedding rows, read from the file, expanded
        let emb = self.m.tensor(0, Role::TokenEmbd).ok_or("token_embd missing")?;
        let rb = emb.ty.bytes(g.n_embd).unwrap_or(0) as usize;
        let mut rows = vec![0u8; t * rb];
        for (i, tok) in tokens.iter().enumerate() {
            if *tok as u64 >= g.n_vocab {
                return Err(Error(format!("token {tok} outside the vocabulary")));
            }
            self.m.file.read_into(emb, *tok as u64 * rb as u64, &mut rows[i * rb..(i + 1) * rb]).map_err(e)?;
        }
        let raw = DevBuf::new(gpu, rows.len())?;
        raw.write(0, &rows)?;
        let x0 = DevBuf::f32(gpu, t * d)?;
        o.dequant(emb.ty.code(), &raw, 0, rows.len(), t * d, &x0)?;
        tap("inp_embd", &x0)?;
        // the 4 streams start as copies of the embedding
        let e0 = x0.to_f32()?;
        let mut xs = vec![0f32; t * 4 * d];
        for ti in 0..t {
            for s in 0..4 {
                xs[(ti * 4 + s) * d..(ti * 4 + s + 1) * d].copy_from_slice(&e0[ti * d..(ti + 1) * d]);
            }
        }
        let mut x = DevBuf::from_f32(gpu, &xs)?;
        tap("hc_init", &x)?;

        let flat = DevBuf::f32(gpu, t * 4 * d)?;
        let h = DevBuf::f32(gpu, t * d)?;
        let normed = DevBuf::f32(gpu, t * d)?;
        let post = DevBuf::f32(gpu, t * 4)?;
        let comb = DevBuf::f32(gpu, t * 16)?;

        // before a half: the mixes, h, post, comb; then the half's norm
        let hc_pre = |l: u64, fn_: Role, base: Role, scale: Role, x: &DevBuf, norm: Role| -> Result<()> {
            o.rms_norm(x, None, &flat, t, 4 * d, eps)?;
            let mixes = self.mm(l, fn_, &flat, t)?;
            o.hc_pre(&mixes, self.vec(l, scale)?, self.vec(l, base)?, x, &h, &post, &comb, t, d, g.hc_eps as f32, g.hc_iters as u32)?;
            o.rms_norm(&h, Some(self.vec(l, norm)?), &normed, t, d, eps)
        };

        for l in 0..g.n_layer {
            // ---- attention half
            hc_pre(l, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, &x, Role::AttnNorm)?;
            tap(&format!("attn_norm-{l}"), &normed)?;
            let att = if let LayerState::Kda { s: kstate, conv: cstate } = &sess.layers[l as usize] {
                let conv = |r: Role, w: Role, state: &DevBuf| -> Result<DevBuf> {
                    let p = self.mm(l, r, &normed, t)?;
                    let out = DevBuf::f32(gpu, t * kw)?;
                    o.conv_silu(&p, state, self.vec(l, w)?, &out, t, kw, g.kda_conv as usize)?;
                    Ok(out)
                };
                let q = conv(Role::KdaQ, Role::KdaQConv, &cstate[0])?;
                let k = conv(Role::KdaK, Role::KdaKConv, &cstate[1])?;
                let v = conv(Role::KdaV, Role::KdaVConv, &cstate[2])?;
                tap(&format!("kda_q_conv-{l}"), &q)?;
                tap(&format!("kda_k_conv-{l}"), &k)?;
                tap(&format!("kda_v_conv-{l}"), &v)?;
                o.l2_norm(&q, t * kh, kd, 1e-6)?;
                o.l2_norm(&k, t * kh, kd, 1e-6)?;
                let fa = self.mm(l, Role::KdaFA, &normed, t)?;
                let gate = self.mm(l, Role::KdaFB, &fa, t)?;
                o.kda_gate(&gate, self.vec(l, Role::KdaDtBias)?, self.vec(l, Role::KdaA)?, t, kh, kd, g.kda_gate_low as f32)?;
                tap(&format!("kda_g1-{l}"), &gate)?;
                let beta = self.mm(l, Role::KdaBeta, &normed, t)?;
                o.sigmoid(&beta, t * kh)?;
                tap(&format!("kda_beta-{l}"), &beta)?;
                let scan = DevBuf::f32(gpu, t * kw)?;
                o.kda_scan(&q, &k, &v, &gate, &beta, kstate, &scan, t, kh, kd)?;
                tap(&format!("kda_scan_out-{l}"), &scan)?;
                let ga = self.mm(l, Role::KdaGA, &normed, t)?;
                let g2 = self.mm(l, Role::KdaGB, &ga, t)?;
                tap(&format!("kda_g2-{l}"), &g2)?;
                let y = DevBuf::f32(gpu, t * kw)?;
                o.kda_out(&scan, &g2, self.vec(l, Role::KdaONorm)?, &y, t, kh, kd, eps)?;
                let out = self.mm(l, Role::KdaOut, &y, t)?;
                tap(&format!("kda_out-{l}"), &out)?;
                out
            } else {
                let LayerState::Mla { c: cache } = &sess.layers[l as usize] else { return Err(Error("layer state".into())) };
                let (nh, hd, lat) = (g.n_head as usize, g.head_dim as usize, g.kv_lora as usize);
                let qa = self.mm(l, Role::MlaQA, &normed, t)?;
                let qr = DevBuf::f32(gpu, t * g.q_lora as usize)?;
                o.rms_norm(&qa, Some(self.vec(l, Role::MlaQANorm)?), &qr, t, g.q_lora as usize, eps)?;
                tap(&format!("q_resid-{l}"), &qr)?;
                let q = self.mm(l, Role::MlaQB, &qr, t)?;
                let kv = self.mm(l, Role::MlaKvA, &normed, t)?;
                let c = DevBuf::f32(gpu, t * lat)?;
                o.rms_norm(&kv, Some(self.vec(l, Role::MlaKvANorm)?), &c, t, lat, eps)?;
                tap(&format!("kv_cmpr-{l}"), &c)?;
                cache.copy_within(pos0 * lat * 4, &c, 0, t * lat * 4)?;
                // the absorbed queries: per head, q~ = k_b[h] . q_h
                let kb = self.vec(l, Role::MlaKB)?;
                let qt = DevBuf::f32(gpu, t * nh * lat)?;
                for hh in 0..nh {
                    o.gemm_at(t, lat, hd, (&q, hh * hd, nh * hd), (kb, hh * lat * hd), (&qt, hh * lat, nh * lat), false)?;
                }
                let u = DevBuf::f32(gpu, t * nh * lat)?;
                o.mla_attend(&qt, cache, &u, t, nh, lat, pos0, 1.0 / (hd as f32).sqrt())?;
                let vb = self.vec(l, Role::MlaVB)?;
                let oh = DevBuf::f32(gpu, t * nh * hd)?;
                for hh in 0..nh {
                    o.gemm_at(t, hd, lat, (&u, hh * lat, nh * lat), (vb, hh * hd * lat), (&oh, hh * hd, nh * hd), false)?;
                }
                tap(&format!("kqv_out-{l}"), &oh)?;
                let out = self.mm(l, Role::MlaOut, &oh, t)?;
                tap(&format!("attn_out-{l}"), &out)?;
                out
            };
            let x1 = DevBuf::f32(gpu, t * 4 * d)?;
            o.hc_post(&att, &x, &post, &comb, &x1, t, d)?;
            tap(&format!("hc_attn_post-{l}"), &x1)?;

            // ---- feed-forward half
            hc_pre(l, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, &x1, Role::FfnNorm)?;
            tap(&format!("ffn_norm-{l}"), &normed)?;
            let lim = g.swiglu_limit as f32;
            let ffn = if !g.is_moe(l) {
                let gt = self.mm(l, Role::FfnGate, &normed, t)?;
                let up = self.mm(l, Role::FfnUp, &normed, t)?;
                let n = t * g.ffn_dense as usize;
                o.swiglu_clamp(&gt, &up, &gt, n, lim)?;
                self.mm(l, Role::FfnDown, &gt, t)?
            } else {
                self.moe(l, t, &normed, &mut *tap)?
            };
            tap(&format!("ffn_out-{l}"), &ffn)?;
            let x2 = DevBuf::f32(gpu, t * 4 * d)?;
            o.hc_post(&ffn, &x1, &post, &comb, &x2, t, d)?;
            tap(&format!("l_out-{l}"), &x2)?;
            x = x2;
        }

        // the head, for the last token
        let mean = DevBuf::f32(gpu, t * d)?;
        o.hc_mean(&x, &mean, t, d)?;
        let last = DevBuf::f32(gpu, d)?;
        let mv = mean.to_f32()?;
        last.write(0, &mv[(t - 1) * d..].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
        let out = DevBuf::f32(gpu, d)?;
        o.rms_norm(&last, Some(self.vec(0, Role::OutputNorm)?), &out, 1, d, eps)?;
        tap("result_norm", &out)?;
        let logits = self.mm(0, Role::Output, &out, 1)?;
        tap("result_output", &logits)?;
        sess.pos += t;
        logits.to_f32()
    }

    /// The MoE half of layer `l` on x [t, d]: the router (on the host), the routed experts grouped by expert, the
    /// shared expert.
    fn moe(&self, l: u64, t: usize, x: &DevBuf, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &self.ops;
        let gpu = &o.gpu;
        let (d, ne, used) = (g.n_embd as usize, g.n_expert as usize, g.n_expert_used as usize);
        let lim = g.swiglu_limit as f32;
        let logits = self.mm(l, Role::Router, x, t)?;
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        let lv = logits.to_f32()?;
        let bias = self.vec(l, Role::RouterBias)?.to_f32()?;
        // expert -> (token, weight)
        let mut by: BTreeMap<usize, Vec<(i32, f32)>> = BTreeMap::new();
        for ti in 0..t {
            let p: Vec<f32> = lv[ti * ne..(ti + 1) * ne].iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|a, b| (p[*b] + bias[*b]).total_cmp(&(p[*a] + bias[*a])));
            let sel = &order[..used];
            let sum: f32 = sel.iter().map(|i| p[*i]).sum::<f32>().max(6.103_516e-5); // the smallest normal half, as llama.cpp clamps
            for i in sel {
                let w = if g.expert_norm { p[*i] / sum } else { p[*i] } * g.expert_scale as f32;
                by.entry(*i).or_default().push((ti as i32, w));
            }
        }
        // the shared expert first, then each routed expert added into it
        let sg = self.mm(l, Role::ShGate, x, t)?;
        let su = self.mm(l, Role::ShUp, x, t)?;
        let f = g.ffn_expert as usize * g.n_expert_shared as usize;
        o.swiglu_clamp(&sg, &su, &sg, t * f, lim)?;
        let y = self.mm(l, Role::ShDown, &sg, t)?;
        for (ex, list) in &by {
            let n = list.len();
            let idx = DevBuf::new(gpu, n * 4)?;
            idx.write(0, &list.iter().flat_map(|(ti, _)| ti.to_le_bytes()).collect::<Vec<u8>>())?;
            let w = DevBuf::from_f32(gpu, &list.iter().map(|(_, w)| *w).collect::<Vec<f32>>())?;
            let xe = DevBuf::f32(gpu, n * d)?;
            o.gather(x, &idx, &xe, n, d)?;
            let gt = self.expert_mm(l, Role::ExpGate, *ex as u64, &xe, n)?;
            let up = self.expert_mm(l, Role::ExpUp, *ex as u64, &xe, n)?;
            o.swiglu_clamp(&gt, &up, &gt, n * g.ffn_expert as usize, lim)?;
            let dn = self.expert_mm(l, Role::ExpDown, *ex as u64, &gt, n)?;
            o.scatter_add(&y, &dn, &idx, &w, n, d)?;
            gpu.sync()?;
        }
        Ok(y)
    }
}
