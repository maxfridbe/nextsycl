//! The engine: the model's layers split over the GPUs (a stage each, `kernels/engines/qwen4exp/qwen.cpp`), every
//! weight and every routed expert in VRAM, the PLE rows read on the host, sessions as the stages' states.
//!
//! A pass is Strata's verify window: up to 8 tokens through every stage, the GDN recurrences and the indexer
//! advanced only when the window is committed - all of it (a prompt's windows), the accepted prefix (a verify pass,
//! `rollback`), or itself (a one-token window). A stage keeps its last window's per-token inputs for that commit, so a
//! window and its commit are adjacent: an uncommitted verify pass is committed whole before any other window runs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_core::{DevBuf, Gpu, Result};
use ns_gguf::{GType, Gguf, Tensor};
use ns_runtime::{GpuInfo, Sampler};

use crate::ffi;
use crate::model::{Model, Role};
use crate::ple;

/// tokens a window takes (the kernels' kVerifyMaxT)
pub const MAX_WINDOW: usize = 8;
/// the longest session a stage is sized for (the model's context)
const STAGE_CELLS: i64 = 262144;

fn err(s: impl Into<String>) -> ns_core::Error {
    ns_core::Error(s.into())
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// One GPU's layers and their weights
struct Stage {
    gpu: Arc<Gpu>,
    raw: ffi::Stage,
    lb: u64,
    le: u64,
    hand_in: *mut f32,
    hand_out: *mut f32,
    hand_floats: usize,
    weight_bytes: u64,
    expert_bytes: u64,
    // kept alive for `raw`
    _dense: Vec<DevBuf>,
    _experts: Vec<DevBuf>,
    _d_res: DevBuf,
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Ok(a) = ffi::api() {
            // SAFETY: the stage came from ns_qw_new; its buffers are dropped after it (fields below `raw`).
            unsafe { (a.free)(self.raw) };
        }
    }
}

/// A conversation: each stage's state, where it is, the tokens before it (the PLE's n-grams)
pub struct Session {
    pub states: Vec<ffi::State>,
    pub pos: usize,
    pub max_ctx: usize,
    pub prev: [i32; 2],
    /// a verify pass not committed yet: (window id, its tokens, its first position)
    pub open: Option<(u64, Vec<u32>, usize)>,
    /// false once dropped: an open pass of a dropped session is never committed
    alive: Arc<AtomicBool>,
}

// SAFETY: the states are device memory reached only through the engine, which serializes its calls.
unsafe impl Send for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
        if let Ok(a) = ffi::api() {
            for s in &self.states {
                // SAFETY: from ns_qw_state_new.
                unsafe { (a.state_free)(*s) };
            }
        }
    }
}

/// Generation: the logits to draw the next token from, or the drawn token not fed yet
pub struct Decoder {
    pub logits: Option<Vec<f32>>,
    pub next: Option<u32>,
}

impl Decoder {
    pub fn pending(&mut self) -> Option<u32> {
        self.logits = None;
        self.next.take()
    }
}

/// A session at a position, in host memory
pub struct Checkpoint {
    pub pos: usize,
    pub prev: [i32; 2],
    pub parts: Vec<Vec<u8>>,
    pub bytes: usize,
}

const CK_MAGIC: &[u8; 8] = b"nsqw0001";

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        w.write_all(CK_MAGIC)?;
        w.write_all(&(self.pos as u64).to_le_bytes())?;
        w.write_all(&self.prev[0].to_le_bytes())?;
        w.write_all(&self.prev[1].to_le_bytes())?;
        w.write_all(&(self.parts.len() as u64).to_le_bytes())?;
        for p in &self.parts {
            w.write_all(&(p.len() as u64).to_le_bytes())?;
            w.write_all(p)?;
        }
        Ok(())
    }
    pub fn read_from(r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        let bad = |s: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, s.to_string());
        let mut m = [0u8; 8];
        r.read_exact(&mut m)?;
        if &m != CK_MAGIC {
            return Err(bad("not a qwen4exp checkpoint"));
        }
        let mut u = [0u8; 8];
        let mut i = [0u8; 4];
        r.read_exact(&mut u)?;
        let pos = u64::from_le_bytes(u) as usize;
        r.read_exact(&mut i)?;
        let p0 = i32::from_le_bytes(i);
        r.read_exact(&mut i)?;
        let p1 = i32::from_le_bytes(i);
        r.read_exact(&mut u)?;
        let n = u64::from_le_bytes(u) as usize;
        if n > 64 {
            return Err(bad("a checkpoint of too many stages"));
        }
        let mut parts = Vec::with_capacity(n);
        let mut bytes = 0;
        for _ in 0..n {
            r.read_exact(&mut u)?;
            let len = u64::from_le_bytes(u) as usize;
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            bytes += len;
            parts.push(v);
        }
        Ok(Checkpoint { pos, prev: [p0, p1], parts, bytes })
    }
}

/// What the stages run on, serialized: the window in flight between a verify pass and its commit
struct Run {
    /// (window id, each stage's state, tokens, the session alive) of an uncommitted verify pass
    open: Option<(u64, Vec<usize>, usize, Arc<AtomicBool>)>,
    next_id: u64,
    /// the last window id committed whole by another call (a session's open pass found it so)
    flushed: Vec<u64>,
    staging: Vec<u8>,
}

pub struct Qwen<'g> {
    pub file: &'g Gguf,
    pub m: Model<'g>,
    stages: Vec<Stage>,
    table: ple::Table,
    api: &'static ffi::Api,
    run: Mutex<Run>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    pub vocab: usize,
}

// SAFETY: the stages' raw handles are used only under `run`'s lock (every GPU call goes through `window`, `commit`
// or the state calls, which take it); the rest is read-only after load.
unsafe impl Send for Qwen<'_> {}
unsafe impl Sync for Qwen<'_> {}

/// A tensor's bytes uploaded into `buf` at `at`
fn upload(f: &Gguf, t: &Tensor, buf: &DevBuf, at: usize) -> Result<()> {
    let b = f.read(t).map_err(|e| err(e.0))?;
    buf.write(at, &b)
}

/// The kernels' fixed geometry (Strata's artifact); the file must have it
fn check_geometry(m: &Model) -> Result<()> {
    let g = &m.g;
    let want = [
        ("hidden", g.n_embd, 2560),
        ("streams", g.hc, 4),
        ("hc rank", g.hc_rank, 320),
        ("experts", g.n_expert, 512),
        ("experts a token", g.n_expert_used, 10),
        ("expert FFN", g.ffn_expert, 640),
        ("shared FFN", g.ffn_shared, 640),
        ("GDN state", g.gdn_state, 128),
        ("GDN k heads", g.gdn_k_heads, 16),
        ("GDN v heads", g.gdn_v_heads, 48),
        ("GDN conv", g.gdn_conv, 4),
        ("QSA heads", g.n_head, 24),
        ("QSA KV heads", g.n_head_kv, 2),
        ("QSA head dim", g.head_dim, 256),
        ("rotary dims", g.n_rot, 64),
        ("indexer heads", g.idx_heads, 4),
        ("indexer dim", g.idx_dim, 128),
        ("indexer top k", g.idx_top_k, 2048),
        ("indexer block", g.idx_block, 4),
        ("PLE dim", g.ple_dim, 160),
        ("PLE rows a token", g.ple_heads(), 16),
    ];
    let bad: Vec<String> = want.iter().filter(|(_, have, want)| have != want).map(|(n, have, want)| format!("{n} {have} (the kernels: {want})")).collect();
    if !bad.is_empty() {
        return Err(err(format!("qwen4exp: this file's geometry is not the one the kernels are built for: {}", bad.join(", "))));
    }
    if (0..g.n_layer).any(|l| g.is_qsa(l) != (l % 4 == 3)) || g.ple_layers != [1] {
        return Err(err("qwen4exp: the kernels take QSA at every 4th layer (3, 7, ...) and the PLE at layer 1"));
    }
    if (g.rope_base - 1e7).abs() > 1.0 {
        return Err(err(format!("qwen4exp: rope base {} (the kernels' default is 1e7; scaling is not wired yet)", g.rope_base)));
    }
    Ok(())
}

/// The bytes of layer `l`'s weights other than the routed experts
fn dense_bytes(m: &Model, l: u64) -> u64 {
    m.roles(l).iter().filter(|r| !matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown)).filter_map(|r| m.tensor(l, *r)).map(|t| (t.bytes + 255) & !255).sum()
}

impl<'g> Qwen<'g> {
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), log: &mut dyn FnMut(String)) -> Result<Qwen<'g>> {
        let t0 = Instant::now();
        let api = ffi::api()?;
        let m = Model::open(f).map_err(|e| err(e.0))?;
        let errs = m.check();
        if !errs.is_empty() {
            return Err(err(format!("{}: {}", f.paths[0].display(), errs.join("; "))));
        }
        check_geometry(&m)?;
        let g = &m.g;
        let nl = g.n_layer;
        let lib = ns_core::api()?;
        for l in 0..nl {
            let (gu, d) = (m.t(l, Role::ExpGate).ty.code(), m.t(l, Role::ExpDown).ty.code());
            // SAFETY: plain values.
            if unsafe { (lib.moe_grouped_supported)(gu as i32, d as i32, g.n_embd as i64, g.ffn_expert as i64) } == 0 {
                return Err(err(format!("layer {l}: no grouped expert kernel for {} gate/up and {} down", m.t(l, Role::ExpGate).ty.name(),
                                       m.t(l, Role::ExpDown).ty.name())));
            }
        }

        // ---- the split: each GPU in turn takes layers while they fit (the first the embedding, the last the head)
        let embd_b = m.t(0, Role::TokenEmbd).bytes;
        let layer_b: Vec<u64> = (0..nl).map(|l| dense_bytes(&m, l) + m.expert_bytes(l) * g.n_expert).collect();
        // the sessions' state: per QSA layer ~1,184 B a cell (int8 K/V, scales, pooled keys), per GDN layer 3.1 MiB
        let (kv_tokens, kv_sessions) = if kv.0 > 0 { kv } else { (65536, 2) };
        let state_b = |lb: u64, le: u64| -> u64 {
            let q = (lb..le).filter(|l| g.is_qsa(*l)).count() as u64;
            let gd = (le - lb) - q;
            q * 1184 * kv_tokens as u64 + gd * (128 * 48 * 128 + 10240 * 3) * 4 * kv_sessions.max(1) as u64
        };
        let margin: u64 = std::env::var("NS_VRAM_GUARD_GIB").ok().and_then(|v| v.parse::<f64>().ok()).map_or(1536 << 20, |g| (g * (1u64 << 30) as f64) as u64);
        let window_b: u64 = 400 << 20;
        let split_env: Option<Vec<u64>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        let mut bounds = Vec::new();
        let mut lb = 0u64;
        for (i, gpu) in gpus.iter().enumerate() {
            let last = i + 1 == gpus.len();
            let le = if let Some(sp) = &split_env {
                if last { nl } else { *sp.get(i).ok_or_else(|| err("NS_SPLIT: a first layer for each GPU after the first"))? }
            } else if last {
                nl
            } else {
                let (total, free) = gpu.memory()?;
                let mut have = free.unwrap_or(total).saturating_sub(margin + window_b);
                if i == 0 {
                    have = have.saturating_sub(embd_b);
                }
                let mut le = lb;
                while le < nl && layer_b[le as usize] + state_b(lb, le + 1) - state_b(lb, le) <= have {
                    have -= layer_b[le as usize] + state_b(lb, le + 1) - state_b(lb, le);
                    le += 1;
                }
                le
            };
            if le <= lb && !last {
                return Err(err(format!("{}: no layer fits ({:.1} GiB free)", gpu.name, gib(gpu.memory()?.1.unwrap_or(0)))));
            }
            bounds.push((lb, le));
            lb = le;
        }
        if gpus.is_empty() || bounds.last().map(|b| b.1) != Some(nl) {
            return Err(err("qwen4exp: the layers do not fit these GPUs"));
        }
        let mut stages = Vec::new();
        let mut load_bytes = 0u64;
        for (i, (gpu, &(lb, le))) in gpus.iter().zip(&bounds).enumerate() {
            if lb == le {
                continue;
            }
            let st = Self::load_stage(f, &m, gpu, lb, le, i == 0, le == nl, api, log)?;
            load_bytes += st.weight_bytes + st.expert_bytes;
            stages.push(st);
        }
        let table = ple::Table::open(&m, f)?;
        let vocab = g.n_vocab as usize;
        let load_seconds = t0.elapsed().as_secs_f64();
        log(format!("qwen4exp: {} stages ({}), {:.2} GiB of weights in {:.1} s",
                    stages.len(),
                    stages.iter().map(|s| format!("{} layers {}-{}", s.gpu.name, s.lb, s.le - 1)).collect::<Vec<_>>().join(", "),
                    gib(load_bytes), load_seconds));
        Ok(Qwen { file: f, m, stages, table, api, run: Mutex::new(Run { open: None, next_id: 1, flushed: Vec::new(), staging: Vec::new() }), load_seconds,
                  load_bytes, vocab })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_stage(f: &Gguf, m: &Model, gpu: &Arc<Gpu>, lb: u64, le: u64, first: bool, last: bool, api: &ffi::Api, log: &mut dyn FnMut(String))
                  -> Result<Stage> {
        let g = &m.g;
        let t0 = Instant::now();
        let mut dense = Vec::new();
        let mut layers = Vec::new();
        let mut weight_bytes = 0u64;
        // a tensor's device pointer inside its layer's buffer
        let ptr = |buf: &DevBuf, off: &std::collections::HashMap<Role, usize>, r: Role| -> *const std::ffi::c_void {
            // SAFETY: an offset inside the buffer (laid out below)
            off.get(&r).map_or(std::ptr::null(), |o| unsafe { buf.ptr().cast::<u8>().add(*o).cast() })
        };
        for l in lb..le {
            let roles: Vec<Role> = m.roles(l).into_iter().filter(|r| !matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown)).collect();
            let mut off = std::collections::HashMap::new();
            let mut at = 0usize;
            for r in &roles {
                off.insert(*r, at);
                at += (m.t(l, *r).bytes as usize + 255) & !255;
            }
            let buf = DevBuf::new(gpu, at)?;
            for r in &roles {
                upload(f, m.t(l, *r), &buf, off[r])?;
            }
            weight_bytes += at as u64;
            let p = |r: Role| ptr(&buf, &off, r);
            let ty = |r: Role| m.tensor(l, r).map_or(-1, |t| t.ty.code() as i32);
            let mut x = ffi::Layer {
                hc_norm: [p(Role::HcAttnNorm).cast(), p(Role::HcFfnNorm).cast()],
                hc_down: [p(Role::HcAttnDown).cast(), p(Role::HcFfnDown).cast()],
                hc_up: [p(Role::HcAttnUp).cast(), p(Role::HcFfnUp).cast()],
                hc_inject: [p(Role::HcAttnInject).cast(), p(Role::HcFfnInject).cast()],
                router: p(Role::Router).cast(),
                sh_gate_inp: p(Role::ShGateInp).cast(),
                sh_gate_type: ty(Role::ShGate),
                sh_up_type: ty(Role::ShUp),
                sh_down_type: ty(Role::ShDown),
                sh_gate: p(Role::ShGate),
                sh_up: p(Role::ShUp),
                sh_down: p(Role::ShDown),
                gu_type: ty(Role::ExpGate),
                d_type: ty(Role::ExpDown),
                ..Default::default()
            };
            if g.is_qsa(l) {
                x.q_type = ty(Role::QsaQ);
                x.k_type = ty(Role::QsaK);
                x.v_type = ty(Role::QsaV);
                x.o_type = ty(Role::QsaOut);
                x.q_w = p(Role::QsaQ);
                x.k_w = p(Role::QsaK);
                x.v_w = p(Role::QsaV);
                x.o_w = p(Role::QsaOut);
                x.q_norm = p(Role::QsaQNorm).cast();
                x.k_norm = p(Role::QsaKNorm).cast();
                x.idx_q = p(Role::IdxQ).cast();
                x.idx_k = p(Role::IdxK).cast();
                x.idx_q_norm = p(Role::IdxQNorm).cast();
                x.idx_k_norm = p(Role::IdxKNorm).cast();
            } else {
                x.qkv_type = ty(Role::GdnQkv);
                x.z_type = ty(Role::GdnZ);
                x.out_type = ty(Role::GdnOut);
                x.qkv = p(Role::GdnQkv);
                x.z_w = p(Role::GdnZ);
                x.out_w = p(Role::GdnOut);
                x.alpha = p(Role::GdnAlpha).cast();
                x.beta = p(Role::GdnBeta).cast();
                x.conv = p(Role::GdnConv).cast();
                x.ssm_a = p(Role::GdnA).cast();
                x.dt_bias = p(Role::GdnDtBias).cast();
                x.ssm_norm = p(Role::GdnNorm).cast();
            }
            if g.has_ple(l) {
                x.ple_key = p(Role::PleKey).cast();
                x.ple_value = p(Role::PleValue).cast();
                x.ple_norm_key = p(Role::PleNormKey).cast();
                x.ple_norm_query = p(Role::PleNormQuery).cast();
                x.ple_norm_conv = p(Role::PleNormConv).cast();
                x.ple_conv = p(Role::PleConv).cast();
            }
            // the kernels read these forms (GSQ-RCO stores them so; another converter's file would need converting)
            for (r, want) in [(Role::Router, GType::BF16), (Role::HcAttnDown, GType::BF16), (Role::HcFfnNorm, GType::F32), (Role::ShGateInp, GType::BF16)] {
                if m.t(l, r).ty != want {
                    return Err(err(format!("{}: {} (the kernels read {})", m.t(l, r).name, m.t(l, r).ty.name(), want.name())));
                }
            }
            layers.push(x);
            dense.push(buf);
        }

        // ---- the routed experts: a blob each, [gate rows | up rows | down rows], a buffer a layer
        let ne = g.n_expert as usize;
        let mut experts = Vec::new();
        let mut blobs = Vec::new(); // (layer buffer, blob bytes)
        let mut expert_bytes = 0u64;
        for l in lb..le {
            let (tg, tu, td) = (m.t(l, Role::ExpGate), m.t(l, Role::ExpUp), m.t(l, Role::ExpDown));
            let (gb, ub, db) = ((tg.bytes as usize) / ne, (tu.bytes as usize) / ne, (td.bytes as usize) / ne);
            if gb != ub {
                return Err(err(format!("layer {l}: gate and up experts of different sizes")));
            }
            let blob = gb + ub + db;
            let buf = DevBuf::new(gpu, blob * ne)?;
            // a few experts at a time: read their three slices, then one upload
            let per = 32usize;
            let mut host = vec![0u8; blob * per];
            for e0 in (0..ne).step_by(per) {
                let n = per.min(ne - e0);
                for i in 0..n {
                    let e = e0 + i;
                    let b = &mut host[i * blob..(i + 1) * blob];
                    f.read_into(tg, (e * gb) as u64, &mut b[..gb]).map_err(|x| err(x.0))?;
                    f.read_into(tu, (e * ub) as u64, &mut b[gb..gb + ub]).map_err(|x| err(x.0))?;
                    f.read_into(td, (e * db) as u64, &mut b[gb + ub..]).map_err(|x| err(x.0))?;
                }
                buf.write(e0 * blob, &host[..n * blob])?;
            }
            expert_bytes += (blob * ne) as u64;
            blobs.push((buf.ptr() as u64, blob as u64));
            experts.push(buf);
        }
        // the residency table: slot (l - lb) * 512 + e, at cache_base + slot_off[slot]
        let base = blobs.iter().map(|b| b.0).min().unwrap_or(0);
        let mut slot_off = Vec::with_capacity(blobs.len() * ne);
        for (p, blob) in &blobs {
            for e in 0..ne as u64 {
                slot_off.push(p - base + e * blob);
            }
        }
        let mut res = vec![-1i32; g.n_layer as usize * ne];
        for l in lb..le {
            for e in 0..ne {
                res[l as usize * ne + e] = ((l - lb) as usize * ne + e) as i32;
            }
        }
        let d_res = DevBuf::new(gpu, res.len() * 4)?;
        // SAFETY: i32s as bytes
        d_res.write(0, unsafe { std::slice::from_raw_parts(res.as_ptr().cast::<u8>(), res.len() * 4) })?;

        // ---- the edges
        let mut edges = ffi::Edges::default();
        if first {
            let t = m.t(0, Role::TokenEmbd);
            let buf = DevBuf::new(gpu, t.bytes as usize)?;
            upload(f, t, &buf, 0)?;
            edges.embd_type = t.ty.code() as i32;
            edges.embd = buf.ptr();
            edges.embd_row = (t.bytes / g.n_vocab) as usize;
            weight_bytes += t.bytes;
            dense.push(buf);
        }
        if last {
            let roles = [Role::Output, Role::OutputHcNorm, Role::OutputHcDown, Role::OutputHcUp];
            let mut at = 0usize;
            let mut off = std::collections::HashMap::new();
            for r in roles {
                off.insert(r, at);
                at += (m.t(0, r).bytes as usize + 255) & !255;
            }
            let buf = DevBuf::new(gpu, at)?;
            for r in roles {
                upload(f, m.t(0, r), &buf, off[&r])?;
            }
            let p = |r: Role| ptr(&buf, &off, r);
            edges.out_type = m.t(0, Role::Output).ty.code() as i32;
            edges.out = p(Role::Output);
            edges.out_hc_norm = p(Role::OutputHcNorm).cast();
            edges.out_hc_down = p(Role::OutputHcDown).cast();
            edges.out_hc_up = p(Role::OutputHcUp).cast();
            edges.vocab = g.n_vocab as i64;
            weight_bytes += at as u64;
            dense.push(buf);
        }
        gpu.sync()?;
        let desc = ffi::Desc {
            lb: lb as i64,
            le: le as i64,
            n_layer: g.n_layer as i64,
            max_cells: STAGE_CELLS,
            layers: layers.as_ptr(),
            edges,
            d_res: d_res.ptr().cast(),
            cache_base: base as *const u8,
            slot_off: slot_off.as_ptr(),
            n_slots: slot_off.len() as i64,
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the description's pointers live through the call (it copies them); the buffers outlive the stage.
        ffi::check(unsafe { (api.new)(gpu.raw(), &desc, &mut raw) }, &format!("{}: the stage", gpu.name))?;
        let (mut hi, mut ho, mut hf) = (std::ptr::null_mut(), std::ptr::null_mut(), 0usize);
        // SAFETY: out-pointers to locals.
        ffi::check(unsafe { (api.buffers)(raw, &mut hi, &mut ho, &mut hf) }, "the stage's hand-off")?;
        let (total, free) = gpu.memory()?;
        log(format!("{}: layers {lb}-{} ({:.2} GiB dense, {:.2} GiB experts) in {:.1} s; {:.1} of {:.1} GiB free", gpu.name, le - 1, gib(weight_bytes),
                    gib(expert_bytes), t0.elapsed().as_secs_f64(), gib(free.unwrap_or(0)), gib(total)));
        Ok(Stage { gpu: gpu.clone(), raw, lb, le, hand_in: hi, hand_out: ho, hand_floats: hf, weight_bytes, expert_bytes, _dense: dense, _experts: experts,
                   _d_res: d_res })
    }

    // ---- sessions

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let max_ctx = max_ctx.clamp(MAX_WINDOW, STAGE_CELLS as usize);
        let _run = self.run.lock().unwrap();
        let mut states = Vec::new();
        for st in &self.stages {
            let mut s = std::ptr::null_mut();
            // SAFETY: a live stage; out-pointer to a local.
            if let Err(e) = ffi::check(unsafe { (self.api.state_new)(st.raw, max_ctx as i64, &mut s) }, &format!("{}: a session", st.gpu.name)) {
                for s in states {
                    // SAFETY: from state_new above.
                    unsafe { (self.api.state_free)(s) };
                }
                return Err(e);
            }
            states.push(s);
        }
        Ok(Session { states, pos: 0, max_ctx, prev: [-1, -1], open: None, alive: Arc::new(AtomicBool::new(true)) })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        self.settle(&mut run, s)?;
        for (st, x) in self.stages.iter().zip(&s.states) {
            // SAFETY: a live stage and its state.
            ffi::check(unsafe { (self.api.state_reset)(st.raw, *x) }, "resetting a session")?;
        }
        s.pos = 0;
        s.prev = [-1, -1];
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        self.settle(&mut run, dst)?;
        if src.open.is_some() {
            return Err(err("copying a session with an uncommitted verify pass"));
        }
        if dst.max_ctx < src.pos {
            return Err(err(format!("copying {} tokens into a session of {}", src.pos, dst.max_ctx)));
        }
        for ((st, d), s) in self.stages.iter().zip(&dst.states).zip(&src.states) {
            // SAFETY: live stage and states.
            ffi::check(unsafe { (self.api.state_copy)(st.raw, *d, *s, src.pos as i64) }, "copying a session")?;
        }
        dst.pos = src.pos;
        dst.prev = src.prev;
        Ok(())
    }

    /// A session's open verify pass, settled: committed whole by another call meanwhile (its prev / pos follow)
    fn settle(&self, run: &mut Run, s: &mut Session) -> Result<()> {
        if let Some((id, toks, _)) = &s.open {
            let committed = run.flushed.contains(id) || run.open.as_ref().is_none_or(|o| o.0 != *id);
            if committed {
                run.flushed.retain(|x| x != id);
                let mut p = s.prev;
                self.table.rows(toks, &mut p);
                s.prev = p;
                s.open = None;
            }
        }
        Ok(())
    }

    /// The open verify pass of another session, committed whole (its window data is about to be overwritten)
    fn flush(&self, run: &mut Run) -> Result<()> {
        if let Some((id, states, t, alive)) = run.open.take() {
            // (a dropped session's states are freed: nothing to commit; its drop and this run under no common lock, but
            // a session is dropped only by its owner, which is not inside an engine call then)
            if alive.load(Ordering::SeqCst) {
                for (st, x) in self.stages.iter().zip(&states) {
                    // SAFETY: the open session's states, alive (above).
                    ffi::check(unsafe { (self.api.commit)(st.raw, *x as ffi::State, t as i32) }, "committing a verify pass")?;
                }
                run.flushed.push(id);
            }
        }
        Ok(())
    }

    /// One window: `tokens` (1..8) at the session's position through every stage; logits of rows [from, T)
    fn window(&self, run: &mut Run, s: &mut Session, tokens: &[u32], from: usize) -> Result<Vec<f32>> {
        let t = tokens.len();
        if t == 0 || t > MAX_WINDOW {
            return Err(err(format!("a window of {t} tokens")));
        }
        if s.pos + t > s.max_ctx {
            return Err(err(format!("the session's context ({}) is full", s.max_ctx)));
        }
        let mut prev = s.prev;
        let rows = self.table.rows(tokens, &mut prev);
        let ple = self.table.gather(&rows)?;
        let toks: Vec<i32> = tokens.iter().map(|x| *x as i32).collect();
        let mut logits = vec![0f32; t.saturating_sub(from) * self.vocab];
        for (i, (st, x)) in self.stages.iter().zip(&s.states).enumerate() {
            if i > 0 {
                let p = &self.stages[i - 1];
                let n = t * p.hand_floats * 4;
                if run.staging.len() < n {
                    run.staging.resize(n, 0);
                }
                let lib = ns_core::api()?;
                // SAFETY: both hand-off buffers hold MAX_WINDOW tokens; the staging is n bytes; the call waits.
                let rc = unsafe { (lib.copy_peer)(st.gpu.raw(), st.hand_in.cast(), p.gpu.raw(), p.hand_out.cast(), n, run.staging.as_mut_ptr().cast()) };
                ffi::check(rc, "the hand-off between GPUs")?;
            }
            let has_ple = st.lb <= 1 && 1 < st.le;
            let last = i + 1 == self.stages.len();
            // SAFETY: a live stage and its state; the token and row arrays hold t entries; logits hold (t - from) rows.
            let rc = unsafe {
                (self.api.window)(st.raw, *x, t as i32, toks.as_ptr(), s.pos as i64, if has_ple { ple.as_ptr() } else { std::ptr::null() },
                                  if last { from as i32 } else { t as i32 }, if last && from < t { logits.as_mut_ptr() } else { std::ptr::null_mut() })
            };
            ffi::check(rc, &format!("{}: a window at {}", st.gpu.name, s.pos))?;
        }
        Ok(logits)
    }

    /// The last window's first `keep` tokens made permanent
    fn commit(&self, s: &mut Session, tokens: &[u32], keep: usize) -> Result<()> {
        for (st, x) in self.stages.iter().zip(&s.states) {
            // SAFETY: a live stage and the state its last window ran on.
            ffi::check(unsafe { (self.api.commit)(st.raw, *x, keep as i32) }, "committing a window")?;
        }
        let mut p = s.prev;
        self.table.rows(&tokens[..keep], &mut p);
        s.prev = p;
        s.pos += keep;
        Ok(())
    }

    /// Tokens read: windows of up to 8, each committed; the last token's logits
    pub fn forward(&self, s: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        self.settle(&mut run, s)?;
        if s.open.is_some() {
            return Err(err("a window on a session with an uncommitted verify pass"));
        }
        let mut last = Vec::new();
        let n = tokens.len();
        for (i, w) in tokens.chunks(MAX_WINDOW).enumerate() {
            let end = (i + 1) * MAX_WINDOW >= n;
            let l = self.window(&mut run, s, w, if end { w.len() - 1 } else { w.len() })?;
            self.commit(s, w, w.len())?;
            if end {
                last = l;
            }
        }
        Ok(last)
    }

    /// One verify pass (up to 8 tokens): the last `n_out` rows' logits, left open for `rollback`
    pub fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize) -> Result<Vec<Vec<f32>>> {
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        self.settle(&mut run, s)?;
        let t = tokens.len();
        if n_out > t || t > MAX_WINDOW {
            return Err(err(format!("a verify pass of {t} tokens, {n_out} rows out (at most {MAX_WINDOW})")));
        }
        let l = self.window(&mut run, s, tokens, t - n_out)?;
        let pos0 = s.pos;
        let id = run.next_id;
        run.next_id += 1;
        run.open = Some((id, s.states.iter().map(|x| *x as usize).collect(), t, s.alive.clone()));
        s.open = Some((id, tokens.to_vec(), pos0));
        s.pos += t;
        Ok(l.chunks(self.vocab).map(|r| r.to_vec()).collect())
    }

    /// Keeps the first `keep` tokens of the session's last verify pass
    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        let mut run = self.run.lock().unwrap();
        let Some((id, toks, pos0)) = s.open.take() else {
            // nothing open: a pass committed whole may keep all of itself
            if keep <= s.pos {
                return Ok(());
            }
            return Err(err("rollback without a verify pass"));
        };
        let mine = run.open.as_ref().is_some_and(|o| o.0 == id);
        if !mine {
            // committed whole by another call meanwhile
            run.flushed.retain(|x| *x != id);
            let mut p = s.prev;
            self.table.rows(&toks, &mut p);
            s.prev = p;
            if keep != toks.len() {
                return Err(err("the verify pass was committed whole before its rollback"));
            }
            return Ok(());
        }
        run.open = None;
        s.pos = pos0;
        if keep == 0 || keep > toks.len() {
            // (the pass's first token is the last accepted one: a pass always keeps it)
            self.commit(s, &toks, toks.len())?;
            return Err(err(format!("a rollback keeping {keep} of a {}-token pass", toks.len())));
        }
        self.commit(s, &toks, keep)
    }

    /// One-token passes of several conversations (one after another for now)
    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(sessions.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.forward(s, &[*t])?);
        }
        Ok(out)
    }

    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler) -> Result<Vec<u32>> {
        let logits = match d.logits.take() {
            Some(l) => l,
            None => {
                let t = d.next.take().ok_or_else(|| err("a decode step with nothing to feed"))?;
                self.forward(s, &[t])?
            }
        };
        let t = smp.sample(&logits);
        d.next = Some(t);
        Ok(vec![t])
    }

    // ---- checkpoints

    pub fn save_ref(&self, s: &Session) -> Result<Checkpoint> {
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        if s.open.is_some() {
            return Err(err("saving a session with an uncommitted verify pass"));
        }
        let mut parts = Vec::new();
        let mut bytes = 0;
        for (st, x) in self.stages.iter().zip(&s.states) {
            let mut n = 0u64;
            // SAFETY: a live state; out-pointer to a local.
            ffi::check(unsafe { (self.api.state_bytes)(*x, s.pos as i64, &mut n) }, "a checkpoint's size")?;
            let mut v = vec![0u8; n as usize];
            // SAFETY: v holds the n bytes the state says.
            ffi::check(unsafe { (self.api.state_save)(st.raw, *x, s.pos as i64, v.as_mut_ptr().cast()) }, "saving a session")?;
            bytes += v.len();
            parts.push(v);
        }
        Ok(Checkpoint { pos: s.pos, prev: s.prev, parts, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        self.settle(&mut run, s)?;
        if ck.parts.len() != self.stages.len() || ck.pos > s.max_ctx {
            return Err(err("a checkpoint of another layout (stages, context)"));
        }
        for ((st, x), p) in self.stages.iter().zip(&s.states).zip(&ck.parts) {
            let mut n = 0u64;
            // SAFETY: a live state; out-pointer to a local.
            ffi::check(unsafe { (self.api.state_bytes)(*x, ck.pos as i64, &mut n) }, "a checkpoint's size")?;
            if n as usize != p.len() {
                return Err(err("a checkpoint of another layout (bytes)"));
            }
            // SAFETY: p holds the bytes the state reads.
            ffi::check(unsafe { (self.api.state_load)(st.raw, *x, ck.pos as i64, p.as_ptr().cast()) }, "restoring a session")?;
        }
        s.pos = ck.pos;
        s.prev = ck.prev;
        s.open = None;
        Ok(())
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.stages
            .iter()
            .map(|st| {
                let (total, free) = st.gpu.memory().unwrap_or((0, None));
                GpuInfo { index: st.gpu.index, name: st.gpu.name.clone(), pci: st.gpu.pci.clone(), total, free, layers: (st.lb, st.le),
                          expert_slots: ((st.le - st.lb) * self.m.g.n_expert) as usize, host_slots: 0 }
            })
            .collect()
    }
}
