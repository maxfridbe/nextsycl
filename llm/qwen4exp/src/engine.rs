//! The engine: the model's layers split over the GPUs (a stage each, `kernels/llm/qwen4exp/qwen.cpp`), every
//! weight and every routed expert in VRAM, the PLE rows read on the host, sessions as the stages' states.
//!
//! A pass is Strata's verify window: up to 8 tokens through every stage, the GDN recurrences and the indexer
//! advanced only when the window is committed - all of it (a prompt's windows), the accepted prefix (a verify pass,
//! `rollback`), or itself (a one-token window). A stage keeps its last window's per-token inputs for that commit, so a
//! window and its commit are adjacent: an uncommitted verify pass is committed whole before any other window runs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu, Result};
use nextsycl_gguf::{GType, Gguf, Tensor};
use nextsycl_llm::{GpuInfo, Sampler};

use crate::ffi;
use crate::model::{Model, Role};
use crate::ple;

/// tokens a window takes (the kernels' kVerifyMaxT)
pub const MAX_WINDOW: usize = 8;

/// the speculative window with the draft layer (Strata's --spec 4: the token and up to 3 drafts; NS_QW_SPEC)
fn spec() -> usize {
    static S: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("NS_QW_SPEC").ok().and_then(|v| v.parse().ok()).filter(|n: &usize| (2..=MAX_WINDOW).contains(n)).unwrap_or(4))
}
/// NS_QW_PROFILE=1: each decode round's time by part, on stderr
fn profile() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("NS_QW_PROFILE").is_ok_and(|v| v == "1"))
}

/// NS_QW_COUPLED=0: a sampled request's draws on the host from the logits, its drafts the draft layer's argmax
/// (Strata's default); otherwise on the GPU with the drafts coupled to them (Strata's STRATA_SPEC_COUPLED=1)
fn coupled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("NS_QW_COUPLED").map_or(true, |v| v != "0"))
}

/// a draft goes into the window while at least this likely under the draft layer (Strata's --spec-min-p 0.5;
/// NS_QW_SPEC_MIN_P)
fn spec_min_p() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("NS_QW_SPEC_MIN_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5))
}

/// tokens a prompt-path chunk takes (NS_QW_CHUNK; Strata serves --prefill 4096): its buffers grow with it, ~0.4 MiB a
/// token on each GPU
pub fn prompt_chunk() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("NS_QW_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|n: &usize| *n >= 16).unwrap_or(2048))
}
/// the longest session a stage is sized for (the model's context)
const STAGE_CELLS: i64 = 262144;

fn err(s: impl Into<String>) -> nextsycl_core::Error {
    nextsycl_core::Error(s.into())
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
    /// the token embedding when this stage has it (the first)
    embd_ptr: *const std::ffi::c_void,
    vram_slots: usize,
    host_slots: usize,
    // kept alive for `raw`
    _dense: Vec<DevBuf>,
    _experts: Vec<DevBuf>,
    _hosts: Vec<nextsycl_core::HostBuf>,
    _d_mirror: Option<DevBuf>,
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

impl Session {
    /// this session among the live ones
    fn uid(&self) -> usize {
        Arc::as_ptr(&self.alive) as usize
    }
}

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

/// Generation: the logits to draw the next token from, or the drawn token not fed yet and the draft layer's guesses
/// after it (each with its probability under the draft layer)
pub struct Decoder {
    pub logits: Option<Vec<f32>>,
    pub next: Option<u32>,
    pub draft: bool,
    pub drafts: Vec<u32>,
    pub probs: Vec<f32>,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn new(logits: Option<Vec<f32>>, next: Option<u32>, draft: bool) -> Decoder {
        Decoder { logits, next, draft, drafts: Vec::new(), probs: Vec::new(), drafted: 0, accepted: 0 }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.logits = None;
        self.drafts.clear();
        self.probs.clear();
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
    /// the last window run: (the session, its first position, its tokens) - the drafter reads its residual rows
    last_window: Option<(usize, usize, usize)>,
    /// the sampler the last stage holds (None: greedy picks)
    sampling: Option<nextsycl_llm::DeviceSampling>,
}

pub struct Qwen<'g> {
    pub file: &'g Gguf,
    pub m: Model<'g>,
    stages: Vec<Stage>,
    table: ple::Table,
    api: &'static ffi::Api,
    run: Mutex<Run>,
    /// the draft layer is on the last stage
    pub mtp: bool,
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
    if g.n_expert != 512 && g.n_expert != 256 {
        return Err(err(format!("qwen4exp: {} experts a layer (the router's kernels take 512 or 256)", g.n_expert)));
    }
    if (0..g.n_layer).any(|l| g.is_qsa(l) != (l % 4 == 3)) || g.ple_layers != [1] {
        return Err(err("qwen4exp: the kernels take QSA at every 4th layer (3, 7, ...) and the PLE at layer 1"));
    }
    if (g.rope_base - 1e7).abs() > 1.0 {
        return Err(err(format!("qwen4exp: rope base {} (the kernels' default is 1e7; scaling is not wired yet)", g.rope_base)));
    }
    Ok(())
}

/// A control vector, Strata's --control-vector-scaled / --control-vector-layer-range / --cvec-mode / --cvec-dir as
/// NS_QW_CVEC (path:scale, comma-separated), NS_QW_CVEC_LAYERS (first,last), NS_QW_CVEC_MODE (project | add),
/// NS_QW_CVEC_DIR (per-layer | single:L): its directions (n_layer x 2560) and scales (n_layer), the mode; None without
/// A control vector's tables: directions, scales, mode
type CvecTables = (Vec<f32>, Vec<f32>, i32);

fn load_cvec(nl: u64, n: usize, log: &mut dyn FnMut(String)) -> Result<Option<CvecTables>> {
    let Some(spec) = std::env::var("NS_QW_CVEC").ok().filter(|v| !v.is_empty()) else { return Ok(None) };
    let l = nl as usize;
    let mut data = vec![0f32; l * n];
    let mut have = vec![false; l];
    for item in spec.split(',') {
        let (path, scale) = match item.rsplit_once(':') {
            Some((p, sc)) if sc.parse::<f32>().is_ok() => (p, sc.parse::<f32>().unwrap()),
            _ => (item, 1.0),
        };
        let f = Gguf::open(std::path::Path::new(path)).map_err(|e| err(format!("{path}: {}", e.0)))?;
        if f.architecture() != "controlvector" {
            return Err(err(format!("{path}: not a control vector GGUF")));
        }
        let mut found = 0;
        for t in &f.tensors {
            let Some(layer) = t.name.strip_prefix("direction.").and_then(|x| x.parse::<usize>().ok()) else { continue };
            if layer < 1 || layer >= l {
                continue;
            }
            if t.ty != GType::F32 || t.elements() != n as u64 {
                return Err(err(format!("{path}: {} must be {n} f32", t.name)));
            }
            let b = f.read(t).map_err(|e| err(e.0))?;
            for (j, c) in b.chunks(4).enumerate() {
                data[layer * n + j] += scale * f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            have[layer] = true;
            found += 1;
        }
        if found == 0 {
            return Err(err(format!("{path}: no direction.<layer> tensors")));
        }
    }
    let (mut first, mut last) = (1usize, l - 1);
    if let Ok(r) = std::env::var("NS_QW_CVEC_LAYERS") {
        let v: Vec<usize> = r.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 2 {
            first = v[0].max(1);
            last = v[1].min(l - 1);
        }
    }
    let mode = if std::env::var("NS_QW_CVEC_MODE").is_ok_and(|m| m == "add") { 1 } else { 0 };
    let single: Option<usize> = std::env::var("NS_QW_CVEC_DIR").ok().and_then(|d| d.strip_prefix("single:").and_then(|x| x.parse().ok()));
    let mut dir = vec![0f32; l * n];
    let mut sc = vec![0f32; l];
    let mut steered = 0;
    for layer in first..=last {
        let src = if mode == 0 { single.unwrap_or(layer) } else { layer };
        if src >= l || !have[src] {
            continue;
        }
        let d = &data[src * n..(src + 1) * n];
        if mode == 0 {
            let nrm = d.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
            if nrm <= 0.0 {
                continue;
            }
            sc[layer] = nrm as f32;
            for j in 0..n {
                dir[layer * n + j] = (d[j] as f64 / nrm) as f32;
            }
        } else {
            sc[layer] = 1.0;
            dir[layer * n..(layer + 1) * n].copy_from_slice(d);
        }
        steered += 1;
    }
    log(format!("qwen4exp: a control vector ({spec}), {} on {steered} layers in [{first}, {last}]", if mode == 0 { "projection" } else { "added" }));
    Ok(Some((dir, sc, mode)))
}

/// Strata's expert profile (tools/make_profile.py: `STRP`, version, n_layers, n_expert, slots, n_ranked, then the
/// ranked (layer, expert) pairs as u16 pairs, most routed first)
fn read_profile(path: &str, nl: u64, ne: usize) -> Result<Vec<(usize, usize)>> {
    let b = std::fs::read(path).map_err(|e| err(format!("{path}: {e}")))?;
    let u = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    if b.len() < 24 || &b[..4] != b"STRP" || u(8) as u64 != nl || u(12) as usize != ne {
        return Err(err(format!("{path}: not an expert profile of {nl} layers x {ne} experts")));
    }
    let n = u(20) as usize;
    if b.len() < 24 + n * 4 {
        return Err(err(format!("{path}: its ranked list is truncated")));
    }
    Ok((0..n).map(|i| {
        let at = 24 + i * 4;
        (u16::from_le_bytes([b[at], b[at + 1]]) as usize, u16::from_le_bytes([b[at + 2], b[at + 3]]) as usize)
    })
    .filter(|(l, e)| (*l as u64) < nl && *e < ne)
    .collect())
}

/// The float form the kernels read a role in (Strata's tools/iq_pack.py FORM table); None: as stored (the quantized
/// projections, served from the GGUF's own blocks)
fn form(r: Role) -> Option<GType> {
    use Role::*;
    match r {
        Router | ShGateInp | HcAttnDown | HcAttnUp | HcAttnInject | HcFfnDown | HcFfnUp | HcFfnInject | OutputHcDown | OutputHcUp | IdxK
        | IdxQ | GdnAlpha | GdnBeta | PleKey | PleValue => Some(GType::BF16),
        PleConv => Some(GType::F16),
        QsaQNorm | QsaKNorm | HcAttnNorm | HcFfnNorm | OutputHcNorm | IdxQNorm | IdxKNorm | PleNormConv | PleNormKey | PleNormQuery | GdnA
        | GdnConv | GdnDtBias | GdnNorm => Some(GType::F32),
        _ => None,
    }
}

/// A tensor's bytes in `want`'s form: as stored, or F32 -> BF16 when every value is exactly BF16 (Swift 1.5's routers),
/// F32 -> F16 rounded (the PLE conv, as Strata's loader narrows it), F16 / BF16 -> F32 widened; anything else refused
fn in_form(f: &Gguf, t: &Tensor, want: Option<GType>) -> Result<Vec<u8>> {
    let b = f.read(t).map_err(|e| err(e.0))?;
    let Some(want) = want else { return Ok(b) };
    if t.ty == want {
        return Ok(b);
    }
    let f32s = || b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    match (t.ty, want) {
        (GType::F32, GType::BF16) => {
            let mut out = Vec::with_capacity(b.len() / 2);
            for x in f32s() {
                let bits = x.to_bits();
                if bits & 0xffff != 0 {
                    return Err(err(format!("{}: F32 values that are not BF16 (the kernels read BF16; converting would round)", t.name)));
                }
                out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
            Ok(out)
        }
        (GType::F32, GType::F16) => Ok(f32s().flat_map(|x| f32_to_f16(x).to_le_bytes()).collect()),
        (GType::BF16, GType::F32) => Ok(b.chunks(2).flat_map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect()),
        _ => Err(err(format!("{}: {} (the kernels read {})", t.name, t.ty.name(), want.name()))),
    }
}

/// float -> half, round to nearest even (numpy's astype(float16))
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let mut m = b & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        m |= 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let rest = m & 0x1fff;
    let mut v = ((e16 as u32) << 10) | (m >> 13);
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// The bytes of layer `l`'s weights other than the routed experts
fn dense_bytes(m: &Model, l: u64) -> u64 {
    m.roles(l).iter().filter(|r| !matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown)).filter_map(|r| m.tensor(l, *r)).map(|t| (t.bytes + 255) & !255).sum()
}

impl<'g> Qwen<'g> {
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), draft: bool, log: &mut dyn FnMut(String)) -> Result<Qwen<'g>> {
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
        let lib = ffi::api()?;
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
        let window_b: u64 = 128 << 20; // the window buffers (~60 MiB) and the GEMM scratch
        let head_b = [Role::Output, Role::OutputHcNorm, Role::OutputHcDown, Role::OutputHcUp].iter().map(|r| m.t(0, *r).bytes).sum::<u64>();
        let mtp_b: u64 = if draft && std::env::var("NS_QW_MTP").is_ok_and(|d| !d.is_empty()) { 1100 << 20 } else { 0 };
        let prompt_b = prompt_chunk() as u64 * (420 << 10) + (128 << 20); // carve(): ~1.65 GiB at 4096, each stage its own
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
                let mut have = free.unwrap_or(total).saturating_sub(margin + window_b + prompt_b);
                if i == 0 {
                    have = have.saturating_sub(embd_b);
                }
                let mut le = lb;
                while le < nl && layer_b[le as usize] + state_b(lb, le + 1) - state_b(lb, le) <= have {
                    have -= layer_b[le as usize] + state_b(lb, le + 1) - state_b(lb, le);
                    le += 1;
                }
                // the whole model here: the head and the draft layer too, or the last layers go to the next GPU
                if le == nl {
                    while le > lb && head_b + mtp_b > have {
                        le -= 1;
                        have += layer_b[le as usize] + state_b(lb, le + 1) - state_b(lb, le);
                    }
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
        // ---- the experts in VRAM: every one, unless the last stage's do not all fit (one card holds 24-25 GiB of the
        // 33): then the most routed by the expert profile (NS_QW_EXPERT_PROFILE, Strata's STRP file), the rest in this
        // GPU's pinned host memory, read by the expert kernels over PCIe (Strata's single-card RAM mirror)
        let ne = g.n_expert as usize;
        let mut resident = vec![true; nl as usize * ne];
        // the last stage with layers (a model that fits the first GPU leaves the others none)
        let last_i = bounds.iter().rposition(|b| b.1 > b.0).unwrap_or(0);
        if let (Some(&(lb, le)), Some(gpu)) = (bounds.get(last_i), gpus.get(last_i)) {
            let (total, free) = gpu.memory()?;
            let mut have = free.unwrap_or(total).saturating_sub(margin + window_b + state_b(lb, le));
            if lb == 0 {
                have = have.saturating_sub(embd_b);
            }
            have = have.saturating_sub(head_b + mtp_b + prompt_b + (lb..le).map(|l| dense_bytes(&m, l)).sum::<u64>());
            let all: u64 = (lb..le).map(|l| m.expert_bytes(l) * g.n_expert).sum();
            if all > have {
                let ranked = match std::env::var("NS_QW_EXPERT_PROFILE").ok().filter(|p| !p.is_empty()) {
                    Some(path) => read_profile(&path, nl, ne)?,
                    None => {
                        log("qwen4exp: the experts do not all fit and no NS_QW_EXPERT_PROFILE ranks them: layer order".into());
                        (0..nl as usize).flat_map(|l| (0..ne).map(move |e| (l, e))).collect()
                    }
                };
                for l in lb..le {
                    for e in 0..ne {
                        resident[l as usize * ne + e] = false;
                    }
                }
                let mut used = 0u64;
                for (l, e) in ranked {
                    if (l as u64) < lb || (l as u64) >= le {
                        continue;
                    }
                    let b = m.expert_bytes(l as u64);
                    if used + b > have {
                        continue;
                    }
                    used += b;
                    resident[l * ne + e] = true;
                }
                let n_in = (lb..le).map(|l| (0..ne).filter(|e| resident[l as usize * ne + e]).count()).sum::<usize>();
                log(format!("qwen4exp: {} of {} experts in {}'s VRAM ({:.1} GiB), the rest ({:.1} GiB) in its pinned host memory",
                            n_in, (le - lb) as usize * ne, gpu.name, gib(used), gib(all - used)));
            }
        }
        let mut stages = Vec::new();
        let mut load_bytes = 0u64;
        for (i, (gpu, &(lb, le))) in gpus.iter().zip(&bounds).enumerate() {
            if lb == le {
                continue;
            }
            let st = Self::load_stage(f, &m, gpu, lb, le, i == 0, le == nl, &resident, api, log)?;
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
        // ---- a control vector, on every stage (before any session: its graphs hold where it applies)
        if let Some((dir, sc, mode)) = load_cvec(nl, g.n_embd as usize, log)? {
            for st in &stages {
                // SAFETY: a live stage; dir and sc hold n_layer x 2560 and n_layer floats.
                ffi::check(unsafe { (api.cvec_set)(st.raw, dir.as_ptr(), sc.as_ptr(), nl as i32, mode) }, "the control vector")?;
            }
        }
        // ---- the MTP draft layer (Strata's runtime directory: NS_QW_MTP), on the last stage beside the head
        let mut mtp = false;
        let mtp_dir = std::env::var("NS_QW_MTP").ok().filter(|d| !d.is_empty() && draft);
        if let (Some(dir), Some(last)) = (mtp_dir, stages.last_mut()) {
            let t = m.t(0, Role::TokenEmbd);
            // the drafter embeds its tokens: the embedding on its GPU (the first stage has its own)
            let (ety, eptr, erow) = if last.lb == 0 {
                (t.ty.code() as i32, last.embd_ptr, (t.bytes / g.n_vocab) as usize)
            } else {
                let buf = DevBuf::new(&last.gpu, t.bytes as usize)?;
                upload(f, t, &buf, 0)?;
                let p = buf.ptr();
                last._dense.push(buf);
                (t.ty.code() as i32, p as *const std::ffi::c_void, (t.bytes / g.n_vocab) as usize)
            };
            let c = std::ffi::CString::new(dir.clone()).map_err(|e| err(e.to_string()))?;
            // SAFETY: a live stage; the embedding stays alive with it (its buffers).
            ffi::check(unsafe { (api.mtp_load)(last.raw, c.as_ptr(), spec() as i32, 32768, ety, eptr, erow) }, &format!("{dir}: the draft layer"))?;
            mtp = true;
            log(format!("qwen4exp: the MTP draft layer from {dir} on {}", last.gpu.name));
        }
        Ok(Qwen { file: f, m, stages, table, api,
                  run: Mutex::new(Run { open: None, next_id: 1, flushed: Vec::new(), staging: Vec::new(), last_window: None, sampling: None }), mtp,
                  load_seconds, load_bytes, vocab })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_stage(f: &Gguf, m: &Model, gpu: &Arc<Gpu>, lb: u64, le: u64, first: bool, last: bool, resident: &[bool], api: &ffi::Api,
                  log: &mut dyn FnMut(String)) -> Result<Stage> {
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
            // each tensor in the form the kernels read (Strata's iq_pack FORM: a file that stores one otherwise is
            // converted where that is exact)
            let mut bytes = Vec::with_capacity(roles.len());
            for r in &roles {
                let b = in_form(f, m.t(l, *r), form(*r))?;
                off.insert(*r, at);
                at += (b.len() + 255) & !255;
                bytes.push(b);
            }
            let buf = DevBuf::new(gpu, at)?;
            for (r, b) in roles.iter().zip(&bytes) {
                buf.write(off[r], b)?;
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
            layers.push(x);
            dense.push(buf);
        }

        // ---- the routed experts: a blob each, [gate rows | up rows | down rows]; a layer's resident ones in a VRAM
        // buffer, the others in pinned host memory (the mirror)
        let ne = g.n_expert as usize;
        let mut experts = Vec::new();
        let mut hosts = Vec::new();
        let mut slots: Vec<u64> = Vec::new(); // each VRAM slot's address
        let mut res = vec![-1i32; g.n_layer as usize * ne];
        let mut mirror = vec![0u64; g.n_layer as usize * ne];
        let mut expert_bytes = 0u64;
        let mut host_bytes = 0u64;
        for l in lb..le {
            let (tg, tu, td) = (m.t(l, Role::ExpGate), m.t(l, Role::ExpUp), m.t(l, Role::ExpDown));
            let (gb, ub, db) = ((tg.bytes as usize) / ne, (tu.bytes as usize) / ne, (td.bytes as usize) / ne);
            if gb != ub {
                return Err(err(format!("layer {l}: gate and up experts of different sizes")));
            }
            let blob = gb + ub + db;
            let read = |e: usize, b: &mut [u8]| -> Result<()> {
                f.read_into(tg, (e * gb) as u64, &mut b[..gb]).map_err(|x| err(x.0))?;
                f.read_into(tu, (e * ub) as u64, &mut b[gb..gb + ub]).map_err(|x| err(x.0))?;
                f.read_into(td, (e * db) as u64, &mut b[gb + ub..]).map_err(|x| err(x.0))
            };
            let hot: Vec<usize> = (0..ne).filter(|e| resident[l as usize * ne + e]).collect();
            let cold: Vec<usize> = (0..ne).filter(|e| !resident[l as usize * ne + e]).collect();
            if !hot.is_empty() {
                let buf = DevBuf::new(gpu, blob * hot.len())?;
                // a few experts at a time: read their three slices, then one upload
                let per = 32usize;
                let mut host = vec![0u8; blob * per];
                for (c, group) in hot.chunks(per).enumerate() {
                    for (i, e) in group.iter().enumerate() {
                        read(*e, &mut host[i * blob..(i + 1) * blob])?;
                    }
                    buf.write(c * per * blob, &host[..group.len() * blob])?;
                }
                for (i, e) in hot.iter().enumerate() {
                    res[l as usize * ne + e] = slots.len() as i32;
                    slots.push(buf.ptr() as u64 + (i * blob) as u64);
                }
                expert_bytes += (blob * hot.len()) as u64;
                experts.push(buf);
            }
            if !cold.is_empty() {
                let mut hb = nextsycl_core::HostBuf::new(gpu, blob * cold.len())?;
                let base = hb.as_slice().as_ptr() as u64;
                for (i, e) in cold.iter().enumerate() {
                    read(*e, &mut hb.as_mut_slice()[i * blob..(i + 1) * blob])?;
                    mirror[l as usize * ne + e] = base + (i * blob) as u64;
                }
                host_bytes += (blob * cold.len()) as u64;
                hosts.push(hb);
            }
        }
        // the residency table: slot k at cache_base + slot_off[k]
        let base = slots.iter().copied().min().unwrap_or(0);
        let slot_off: Vec<u64> = slots.iter().map(|p| p - base).collect();
        let d_res = DevBuf::new(gpu, res.len() * 4)?;
        // SAFETY: i32s as bytes
        d_res.write(0, unsafe { std::slice::from_raw_parts(res.as_ptr().cast::<u8>(), res.len() * 4) })?;
        let d_mirror = if hosts.is_empty() {
            None
        } else {
            let b = DevBuf::new(gpu, mirror.len() * 8)?;
            // SAFETY: u64s as bytes
            b.write(0, unsafe { std::slice::from_raw_parts(mirror.as_ptr().cast::<u8>(), mirror.len() * 8) })?;
            Some(b)
        };

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
            n_expert: g.n_expert as i64,
            max_cells: STAGE_CELLS,
            layers: layers.as_ptr(),
            edges,
            d_res: d_res.ptr().cast(),
            cache_base: base as *const u8,
            slot_off: slot_off.as_ptr(),
            n_slots: slot_off.len() as i64,
            h_res: res.as_ptr(),
            mirror: d_mirror.as_ref().map_or(std::ptr::null(), |b| b.ptr().cast()),
            h_mirror: if d_mirror.is_some() { mirror.as_ptr() } else { std::ptr::null() },
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the description's pointers live through the call (it copies them); the buffers outlive the stage.
        ffi::check(unsafe { (api.new)(gpu.raw(), &desc, &mut raw) }, &format!("{}: the stage", gpu.name))?;
        let (mut hi, mut ho, mut hf) = (std::ptr::null_mut(), std::ptr::null_mut(), 0usize);
        // SAFETY: out-pointers to locals.
        ffi::check(unsafe { (api.buffers)(raw, &mut hi, &mut ho, &mut hf) }, "the stage's hand-off")?;
        let (total, free) = gpu.memory()?;
        log(format!("{}: layers {lb}-{} ({:.2} GiB dense, {:.2} GiB experts{}) in {:.1} s; {:.1} of {:.1} GiB free", gpu.name, le - 1,
                    gib(weight_bytes), gib(expert_bytes),
                    if host_bytes > 0 { format!(", {:.2} GiB in pinned host memory", gib(host_bytes)) } else { String::new() },
                    t0.elapsed().as_secs_f64(), gib(free.unwrap_or(0)), gib(total)));
        let host_slots = mirror.iter().filter(|a| **a != 0).count();
        Ok(Stage { gpu: gpu.clone(), raw, lb, le, hand_in: hi, hand_out: ho, hand_floats: hf, weight_bytes, expert_bytes, embd_ptr: edges.embd,
                   vram_slots: slot_off.len(), host_slots, _dense: dense, _experts: experts, _hosts: hosts, _d_mirror: d_mirror, _d_res: d_res })
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
        // its decode graphs recorded now (windows of 1..spec, the commit, the drafter's), not inside a decode round
        let t0 = Instant::now();
        for (st, x) in self.stages.iter().zip(&states) {
            // SAFETY: a live stage and its new state.
            if let Err(e) = ffi::check(unsafe { (self.api.state_warm)(st.raw, *x, spec() as i32) }, "recording a session's graphs") {
                for s in states {
                    // SAFETY: from state_new above.
                    unsafe { (self.api.state_free)(s) };
                }
                return Err(e);
            }
        }
        if profile() {
            eprintln!("[qw session of {max_ctx}: graphs recorded in {:.0} ms]", t0.elapsed().as_secs_f64() * 1e3);
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
        Ok(self.window_raw(run, s, tokens, from, false)?.0)
    }

    /// One window, each row's argmax picked on the GPU (the logits stay there)
    /// The picks of the next windows: the argmax (None), or draws of the request's chain on the GPU, Philox(seed,
    /// position), with the drafts coupled to them (ns_qw_set_sampling); set on the last stage when it changes
    fn set_sampling(&self, run: &mut Run, want: Option<nextsycl_llm::DeviceSampling>) -> Result<()> {
        if run.sampling == want {
            return Ok(());
        }
        let last = self.stages.last().ok_or_else(|| err("no stages"))?;
        let (t, p, seed) = want.map_or((0.0, 1.0, 0), |d| (d.temperature, d.top_p, d.seed));
        // SAFETY: a live stage; no window or round is in flight (the run lock is held, the last call synced).
        ffi::check(unsafe { (self.api.set_sampling)(last.raw, t, p, 64, 0.0, seed) }, "the sampler")?;
        run.sampling = want;
        Ok(())
    }

    /// The window's picks on the GPU: each row's argmax, or its draw (`set_sampling`)
    fn window_argmax(&self, run: &mut Run, s: &mut Session, tokens: &[u32]) -> Result<Vec<u32>> {
        Ok(self.window_raw(run, s, tokens, tokens.len(), true)?.1.iter().map(|x| (*x).max(0) as u32).collect())
    }

    fn window_raw(&self, run: &mut Run, s: &mut Session, tokens: &[u32], from: usize, argmax: bool) -> Result<(Vec<f32>, Vec<i32>)> {
        let t = tokens.len();
        if t == 0 || t > MAX_WINDOW {
            return Err(err(format!("a window of {t} tokens")));
        }
        if s.pos + t > s.max_ctx {
            return Err(err(format!("the session's context ({}) is full", s.max_ctx)));
        }
        let t_ple = Instant::now();
        let mut prev = s.prev;
        let rows = self.table.rows(tokens, &mut prev);
        let ple = self.table.gather(&rows)?;
        let toks: Vec<i32> = tokens.iter().map(|x| *x as i32).collect();
        let mut logits = vec![0f32; t.saturating_sub(from) * self.vocab];
        let mut ids = vec![0i32; if argmax { t } else { 0 }];
        let mut marks = vec![Instant::now()];
        for (i, (st, x)) in self.stages.iter().zip(&s.states).enumerate() {
            if i > 0 {
                let p = &self.stages[i - 1];
                let n = t * p.hand_floats * 4;
                if run.staging.len() < n {
                    run.staging.resize(n, 0);
                }
                let lib = nextsycl_core::api()?;
                // SAFETY: both hand-off buffers hold MAX_WINDOW tokens; the staging is n bytes; the call waits.
                let rc = unsafe { (lib.copy_peer)(st.gpu.raw(), st.hand_in.cast(), p.gpu.raw(), p.hand_out.cast(), n, run.staging.as_mut_ptr().cast()) };
                ffi::check(rc, "the hand-off between GPUs")?;
                marks.push(Instant::now());
            }
            let has_ple = st.lb <= 1 && 1 < st.le;
            let last = i + 1 == self.stages.len();
            // SAFETY: a live stage and its state; the token and row arrays hold t entries; logits hold (t - from) rows.
            let rc = unsafe {
                (self.api.window)(st.raw, *x, t as i32, toks.as_ptr(), s.pos as i64, if has_ple { ple.as_ptr() } else { std::ptr::null() },
                                  if last { from as i32 } else { t as i32 }, if last && from < t { logits.as_mut_ptr() } else { std::ptr::null_mut() },
                                  if last && argmax { ids.as_mut_ptr() } else { std::ptr::null_mut() })
            };
            ffi::check(rc, &format!("{}: a window at {}", st.gpu.name, s.pos))?;
            marks.push(Instant::now());
        }
        if profile() {
            let parts: Vec<String> = marks.windows(2).map(|w| format!("{:.2}", (w[1] - w[0]).as_secs_f64() * 1e3)).collect();
            eprintln!("[qw window T {t}: PLE {:.2} ms, then {} ms (stage / hand-off / stage)]", (marks[0] - t_ple).as_secs_f64() * 1e3, parts.join(" / "));
        }
        run.last_window = Some((s.uid(), s.pos, t));
        Ok((logits, ids))
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

    /// Tokens read: all but the last through the prompt path (in chunks), the last as a window (as Strata reads a
    /// prompt: its first window is the prompt's last token); the last token's logits. NS_QW_WINDOWS=1: every token
    /// through windows of 8 (the decode arithmetic throughout)
    pub fn forward(&self, s: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.feed_until(s, tokens, &|| false)?.1)
    }

    /// As `forward`, `stop()` asked between prompt chunks: (tokens read, the last one's logits - none when stopped)
    pub fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync)) -> Result<(usize, Vec<f32>)> {
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        self.settle(&mut run, s)?;
        if s.open.is_some() {
            return Err(err("a window on a session with an uncommitted verify pass"));
        }
        let n = tokens.len();
        if n == 0 {
            return Err(err("a pass of no tokens"));
        }
        if s.pos + n > s.max_ctx {
            return Err(err(format!("{} tokens past the session's context ({} of {} used)", n, s.pos, s.max_ctx)));
        }
        static WINDOWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let windows = *WINDOWS.get_or_init(|| std::env::var("NS_QW_WINDOWS").is_ok_and(|v| v == "1"));
        let head = if windows { 0 } else { n - 1 };
        let chunk = prompt_chunk();
        let mut c0 = 0;
        while c0 < head {
            if c0 > 0 && stop() {
                return Ok((c0, Vec::new()));
            }
            // Strata's first chunk is short (256) when the prompt is longer than twice that: the same boundaries
            let t = if c0 == 0 && head > 512 && chunk > 256 { 256 } else { chunk.min(head - c0) };
            self.prefill(&mut run, s, &tokens[c0..c0 + t], &tokens[c0 + 1..c0 + t + 1])?;
            c0 += t;
        }
        let mut last = Vec::new();
        let rest = &tokens[head..];
        let m = rest.len();
        for (i, w) in rest.chunks(MAX_WINDOW).enumerate() {
            let end = (i + 1) * MAX_WINDOW >= m;
            let l = self.window(&mut run, s, w, if end { w.len() - 1 } else { w.len() })?;
            self.commit(s, w, w.len())?;
            if end {
                last = l;
            }
        }
        Ok((n, last))
    }

    /// One prompt-path chunk through every stage, committed
    fn prefill(&self, run: &mut Run, s: &mut Session, tokens: &[u32], next: &[u32]) -> Result<()> {
        let t = tokens.len();
        run.last_window = None;
        let chunk = prompt_chunk().max(t);
        let mut prev = s.prev;
        let rows = self.table.rows(tokens, &mut prev);
        let ple = self.table.gather(&rows)?;
        let toks: Vec<i32> = tokens.iter().map(|x| *x as i32).collect();
        let mut r_prev: (*mut f32, Option<&Stage>) = (std::ptr::null_mut(), None);
        for (st, x) in self.stages.iter().zip(&s.states) {
            let mut r = std::ptr::null_mut();
            // SAFETY: a live stage; out-pointer to a local.
            ffi::check(unsafe { (self.api.prefill_buffers)(st.raw, chunk as i64, &mut r) }, &format!("{}: the prompt path", st.gpu.name))?;
            if let (p, Some(ps)) = r_prev {
                // the previous stage's residual (t x 4 x 2560 floats) into this one's
                let n = t * 4 * 2560 * 4;
                if run.staging.len() < n {
                    run.staging.resize(n, 0);
                }
                let lib = nextsycl_core::api()?;
                // SAFETY: both buffers hold `chunk` >= t tokens' rows; the staging n bytes; the call waits.
                let rc = unsafe { (lib.copy_peer)(st.gpu.raw(), r.cast(), ps.gpu.raw(), p.cast(), n, run.staging.as_mut_ptr().cast()) };
                ffi::check(rc, "the prompt's hand-off between GPUs")?;
            }
            let has_ple = st.lb <= 1 && 1 < st.le;
            // SAFETY: a live stage and its state; t tokens and t rows of PLE floats.
            let rc = unsafe { (self.api.prefill)(st.raw, *x, t as i64, toks.as_ptr(), s.pos as i64, if has_ple { ple.as_ptr() } else { std::ptr::null() }) };
            ffi::check(rc, &format!("{}: a prompt chunk at {}", st.gpu.name, s.pos))?;
            r_prev = (r, Some(st));
        }
        if self.mtp {
            // the draft layer's K/V for these cells: cell i pairs the final residual at i with the token at i + 1
            let nx: Vec<i32> = next.iter().map(|x| *x as i32).collect();
            let (st, x) = (self.stages.last().unwrap(), *s.states.last().unwrap());
            // SAFETY: a live stage and its state; t next tokens; the prompt path's rows are this chunk's.
            ffi::check(unsafe { (self.api.mtp_prefill)(st.raw, x, t as i64, s.pos as i64, nx.as_ptr()) }, "the draft layer's prompt K/V")?;
        }
        s.prev = prev;
        s.pos += t;
        Ok(())
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
        // a sampled request with nothing reading the logits: its draws on the GPU too (and coupled drafts)
        let dev = if smp.greedy() || !coupled() { None } else { smp.device() };
        let on_gpu = smp.greedy() || dev.is_some();
        if !(d.draft && self.mtp) {
            if d.logits.is_none() && on_gpu {
                // the pick on the GPU: the token's window, its argmax (or draw), committed
                let t = d.next.take().ok_or_else(|| err("a decode step with nothing to feed"))?;
                let mut run = self.run.lock().unwrap();
                self.set_sampling(&mut run, dev)?;
                self.flush(&mut run)?;
                self.settle(&mut run, s)?;
                if s.open.is_some() {
                    return Err(err("a decode step on a session with an uncommitted verify pass"));
                }
                let y = self.window_argmax(&mut run, s, &[t])?[0];
                self.commit(s, &[t], 1)?;
                d.next = Some(y);
                return Ok(vec![y]);
            }
            let logits = match d.logits.take() {
                Some(l) => l,
                None => {
                    let t = d.next.take().ok_or_else(|| err("a decode step with nothing to feed"))?;
                    self.forward(s, &[t])?
                }
            };
            let t = smp.sample(&logits);
            d.next = Some(t);
            return Ok(vec![t]);
        }
        // ---- with the draft layer (Strata's decode round): the token and the likely drafts as one window, its rows
        // drawn in order while each draws the draft that follows it, the accepted prefix committed, the next drafts
        let mut run = self.run.lock().unwrap();
        self.flush(&mut run)?;
        self.settle(&mut run, s)?;
        if s.open.is_some() {
            return Err(err("a decode step on a session with an uncommitted verify pass"));
        }
        self.set_sampling(&mut run, dev)?;
        if let Some(l) = d.logits.take() {
            // the prompt's last token was its first window: draft from it
            let t = smp.sample(&l);
            d.next = Some(t);
            self.draft_after(&mut run, s, d, &[t], s.pos.saturating_sub(1), 0)?;
            return Ok(vec![t]);
        }
        let x = d.next.take().ok_or_else(|| err("a decode step with nothing to feed"))?;
        let mut tw = 1;
        while tw < spec() && tw - 1 < d.drafts.len() && d.probs[tw - 1] >= spec_min_p() {
            tw += 1;
        }
        if s.pos + tw > s.max_ctx {
            tw = 1;
        }
        let mut window = vec![x];
        window.extend_from_slice(&d.drafts[..tw - 1]);
        let p0 = s.pos;
        let t0 = Instant::now();
        let greedy = on_gpu;
        let (logits, picks) = if greedy { (Vec::new(), self.window_argmax(&mut run, s, &window)?) } else { (self.window(&mut run, s, &window, 0)?, Vec::new()) };
        let t1 = Instant::now();
        let mut out = Vec::with_capacity(tw);
        for i in 0..tw {
            let t = if greedy { picks[i] } else { smp.sample(&logits[i * self.vocab..(i + 1) * self.vocab]) };
            out.push(t);
            if i + 1 >= tw || window[i + 1] != t {
                break;
            }
        }
        let a = out.len() - 1;
        let t2 = Instant::now();
        self.commit(s, &window, a + 1)?;
        let t3 = Instant::now();
        d.drafted += (tw - 1) as u64;
        d.accepted += a as u64;
        d.next = Some(out[a]);
        self.draft_after(&mut run, s, d, &out, p0, a)?;
        if profile() {
            let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
            eprintln!("[qw round at {p0}: T {tw}, kept {}, window {:.2} ms, sample {:.2}, commit {:.2}, draft {:.2} ({} drafts)]", a + 1,
                      ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, Instant::now()), d.drafts.len());
        }
        Ok(out)
    }

    /// The draft layer's round after the window just run (when it was this session's, at `p0`): the catch-up over
    /// rows [0, a] (row t: the residual at p0 + t, then `tokens[t]`), then its drafts (none otherwise)
    fn draft_after(&self, run: &mut Run, s: &mut Session, d: &mut Decoder, tokens: &[u32], p0: usize, a: usize) -> Result<()> {
        d.drafts.clear();
        d.probs.clear();
        if run.last_window.is_none_or(|(u, q, t)| u != s.uid() || q != p0 || t <= a) {
            return Ok(());
        }
        let toks: Vec<i32> = tokens.iter().map(|x| *x as i32).collect();
        let mut drafts = vec![0i32; MAX_WINDOW];
        let mut probs = vec![0f32; MAX_WINDOW];
        let mut n = 0i32;
        let (st, x) = (self.stages.last().unwrap(), *s.states.last().unwrap());
        // SAFETY: a live stage, its state; a + 1 tokens; drafts and probs hold MAX_WINDOW entries.
        let rc = unsafe {
            (self.api.mtp_draft)(st.raw, x, toks.as_ptr(), p0 as i64, a as i32, (spec() - 1) as i32, spec_min_p(), drafts.as_mut_ptr(),
                                 probs.as_mut_ptr(), &mut n)
        };
        ffi::check(rc, "the draft layer")?;
        for j in 0..n.max(0) as usize {
            d.drafts.push(drafts[j].max(0) as u32);
            d.probs.push(probs[j]);
        }
        Ok(())
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
                          expert_slots: st.vram_slots, host_slots: st.host_slots }
            })
            .collect()
    }
}
