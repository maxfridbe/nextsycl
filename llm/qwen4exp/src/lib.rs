//! Qwen3.8-Flash-Next (`qwen4exp`): its model description (`model`: the geometry from the file's metadata, every
//! tensor by role) and its engine, behind the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry
//! entry. Its kernels are Strata's SYCL port (`kernels/strata`, the user's intel-arc branch), wrapped in
//! `kernels/llm/qwen4exp/`.

pub mod engine;
mod ffi;
pub mod model;
mod ple;
pub mod tools;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session,
                 SessionState, Tap};

/// The model description's errors (a file's layout against what the engine needs)
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

pub type ModelResult<T> = std::result::Result<T, Error>;

/// The options it takes (`--opt-NAME`; each sets the variable named, which the engine and its kernels read at load)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "mtp", env: "NS_QW_MTP", value: "DIR", help: "the MTP draft layer (Strata's runtime directory)", at: At::Load },
    EngineOption { name: "spec", env: "NS_QW_SPEC", value: "N", help: "the speculative window: the token and up to N - 1 drafts (Strata's --spec)", at: At::Load },
    EngineOption { name: "spec-min-p", env: "NS_QW_SPEC_MIN_P", value: "P", help: "a draft enters the window while at least this likely (default 0.5)", at: At::Load },
    EngineOption { name: "coupled", env: "NS_QW_COUPLED", value: "0|1", help: "sampled drafts coupled to the sampled token, on the GPU (Strata's STRATA_SPEC_COUPLED)", at: At::Load },
    EngineOption { name: "cvec", env: "NS_QW_CVEC", value: "FILE:SCALE", help: "a control vector (the refusal projection, the speed projection)", at: At::Load },
    EngineOption { name: "cvec-mode", env: "NS_QW_CVEC_MODE", value: "project|add", help: "how the control vector applies", at: At::Load },
    EngineOption { name: "cvec-dir", env: "NS_QW_CVEC_DIR", value: "per-layer|single:L", help: "each layer's own direction, or layer L's for all", at: At::Load },
    EngineOption { name: "cvec-layers", env: "NS_QW_CVEC_LAYERS", value: "FIRST,LAST", help: "the layers it steers", at: At::Load },
    EngineOption { name: "chunk", env: "NS_QW_CHUNK", value: "TOKENS", help: "a prompt chunk's tokens (default 2048; ~0.4 MiB a token on each GPU)", at: At::Load },
    EngineOption { name: "windows", env: "NS_QW_WINDOWS", value: "0|1", help: "a prompt read in decode windows instead of chunks (a check)", at: At::Load },
    EngineOption { name: "expert-profile", env: "NS_QW_EXPERT_PROFILE", value: "FILE", help: "which experts fill VRAM first when not all fit", at: At::Load },
    EngineOption { name: "ple-prefetch", env: "NS_QW_PLE_PREFETCH", value: "0|1", help: "ask for a window's per-layer embedding rows together (default 1)", at: At::Load },
    EngineOption { name: "qw-profile", env: "NS_QW_PROFILE", value: "", help: "each decode round's time by part, on stderr (costs ~9%)", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["qwen4exp"], name: "Qwen3.8-Flash-Next (Gated DeltaNet + QSA, hyper-connections, PLE, 512 experts)", load, info: tools::info,
                 kernels: None, chat: nextsycl_tok::qwen_chat, options: OPTIONS }
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Qwen::load(f, gpus, o.kv, o.draft, log)?))
}

// ---- the contract's opaque handles over this engine's own types

impl SessionState for engine::Session {
    fn pos(&self) -> usize {
        self.pos
    }
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl DecoderState for engine::Decoder {
    fn drafts(&self) -> (u64, u64) {
        (self.drafted, self.accepted)
    }
    fn pending(&mut self) -> Option<u32> {
        engine::Decoder::pending(self)
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl CheckpointState for engine::Checkpoint {
    fn pos(&self) -> usize {
        self.pos
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        engine::Checkpoint::write_to(self, w)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn sess(s: &Session) -> Result<&engine::Session> {
    s.get().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))
}
fn sess_mut(s: &mut Session) -> Result<&mut engine::Session> {
    s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))
}

impl Engine for engine::Qwen<'_> {
    fn arch(&self) -> &'static str {
        "qwen4exp"
    }
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }
    fn load_bytes(&self) -> u64 {
        self.load_bytes
    }
    fn has_draft(&self) -> bool {
        self.mtp
    }
    fn prefill_chunk(&self) -> usize {
        engine::prompt_chunk()
    }
    fn max_verify(&self) -> usize {
        engine::MAX_WINDOW
    }
    fn cache_fingerprint(&self) -> String {
        format!("qwen4exp kv=q8 mtp={} nsqw1", self.mtp)
    }

    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Qwen::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Qwen::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        engine::Qwen::copy_session(self, sess_mut(dst)?, sess(src)?)
    }

    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(Engine::feed_until(self, s, tokens, &|| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync), _tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Qwen::feed_until(self, sess_mut(s)?, tokens, stop)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], _tap: Tap) -> Result<Vec<f32>> {
        engine::Qwen::forward(self, sess_mut(s)?, tokens)
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, _tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Qwen::forward_rows(self, sess_mut(s)?, tokens, n_out)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Qwen::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], _tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = sessions.iter_mut().map(|s| sess_mut(s)).collect::<Result<_>>()?;
        engine::Qwen::forward_batch(self, &mut own, tokens)
    }

    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(Some(logits), None, draft))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(None, Some(next), draft))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, _tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Qwen::step(self, sess_mut(s)?, d, smp)
    }

    fn save(&self, s: &Session) -> Result<Checkpoint> {
        // (a session with a verify pass open cannot be saved: the server saves between passes)
        let s = sess(s)?;
        if s.open.is_some() {
            return Err(nextsycl_core::Error("saving a session with an uncommitted verify pass".into()));
        }
        Ok(Checkpoint::new(self.save_ref(s)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::Qwen::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(r)?))
    }

    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Qwen::gpu_info(self)
    }
}
