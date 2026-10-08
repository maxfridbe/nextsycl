//! Qwen3.8-Flash-Next (`qwen4exp`): its model description (`model`: the geometry from the file's metadata, every
//! tensor by role) and its engine, behind the runtime's `Engine` contract (`ns-runtime`). `kind()` is its registry
//! entry. Its kernels are Strata's SYCL port (`kernels/strata`, the user's intel-arc branch), wrapped in
//! `kernels/engines/qwen4exp/`.

pub mod engine;
mod ffi;
pub mod model;
mod ple;
pub mod tools;

use std::any::Any;
use std::sync::Arc;

use ns_gguf::Gguf;
use ns_runtime::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session,
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

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["qwen4exp"], name: "Qwen3.8-Flash-Next (Gated DeltaNet + QSA, hyper-connections, PLE, 512 experts)", load, info: tools::info,
                 kernels: None, chat: ns_tok::qwen_chat }
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<ns_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Qwen::load(f, gpus, o.kv, log)?))
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
        (0, 0)
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
    s.get().ok_or_else(|| ns_core::Error("a session of another engine".into()))
}
fn sess_mut(s: &mut Session) -> Result<&mut engine::Session> {
    s.get_mut().ok_or_else(|| ns_core::Error("a session of another engine".into()))
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
        false
    }
    fn prefill_chunk(&self) -> usize {
        engine::prompt_chunk()
    }
    fn max_verify(&self) -> usize {
        engine::MAX_WINDOW
    }
    fn cache_fingerprint(&self) -> String {
        "qwen4exp kv=q8 nsqw1".into()
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

    fn decoder(&self, logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder { logits: Some(logits), next: None })
    }
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder { logits: None, next: Some(next) })
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, _tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| ns_core::Error("a decoder of another engine".into()))?;
        engine::Qwen::step(self, sess_mut(s)?, d, smp)
    }

    fn save(&self, s: &Session) -> Result<Checkpoint> {
        // (a session with a verify pass open cannot be saved: the server saves between passes)
        let s = sess(s)?;
        if s.open.is_some() {
            return Err(ns_core::Error("saving a session with an uncommitted verify pass".into()));
        }
        Ok(Checkpoint::new(self.save_ref(s)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| ns_core::Error("a checkpoint of another engine".into()))?;
        engine::Qwen::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(r)?))
    }

    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Qwen::gpu_info(self)
    }
}
