//! GLM-5.3-Flash (`glm5-next`): its model description (`model`: the geometry from the file's metadata, every tensor
//! by role) and its engine (`engine`: the forward passes, the expert store, the decode loop with its draft block,
//! the prompt pipeline - tuned for this model and these GPUs), behind the runtime's `Engine` contract (`ns-runtime`).
//! `kind()` is its registry entry. Its own kernels are in `kernels/engines/glm5next/`.

pub mod engine;
pub mod model;
pub mod tools;

use std::any::Any;
use std::sync::Arc;

use ns_gguf::Gguf;
use ns_runtime::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, ExpertStats, GpuInfo, LoadOptions, Result, Sampler,
                 Session, SessionState, Tap};

/// The model description's errors (a file's layout against what the engine needs)
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<ns_gguf::Error> for Error {
    fn from(e: ns_gguf::Error) -> Error {
        Error(e.0)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

pub type ModelResult<T> = std::result::Result<T, Error>;

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["glm5-next", "glm5next"], name: "GLM-5.3-Flash (KDA + MLA with a DSA indexer, 288 experts, MTP)", load, info: tools::info,
                 kernels: Some(tools::kernels), chat: ns_tok::glm_chat }
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<ns_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Glm::load(f, gpus, o.expert_bytes, o.mirror_bytes, o.draft, o.kv, log)?))
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
    fn lookup_drafts(&self) -> (u64, u64) {
        (self.ng_drafted, self.ng_accepted)
    }
    fn set_context(&mut self, tokens: &[u32]) {
        engine::Decoder::set_context(self, tokens)
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
    fn write_to(&self, mut w: &mut dyn std::io::Write) -> std::io::Result<()> {
        engine::Checkpoint::write_to(self, &mut w)
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

impl Engine for engine::Glm<'_> {
    fn arch(&self) -> &'static str {
        "glm5-next"
    }
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }
    fn load_bytes(&self) -> u64 {
        self.load_bytes
    }
    fn has_draft(&self) -> bool {
        self.mtp.is_some()
    }
    fn prefill_chunk(&self) -> usize {
        engine::prefill_chunk()
    }
    fn max_verify(&self) -> usize {
        // a session keeps a snapshot for each draft: NS_DRAFTS=2 or NS_NGRAM=K widen it (up to MAX_VERIFY)
        (engine::max_drafts() + 1).min(engine::MAX_VERIFY)
    }
    fn cache_fingerprint(&self) -> String {
        // nsck2: the indexer ring of 8 slots (nsck1's 4 restore wrong)
        format!("glm5-next kv={} mtp={} nsck2", if engine::kv_q8() { "q8" } else { "f16" }, self.mtp.is_some())
    }

    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Glm::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Glm::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        engine::Glm::copy_session(self, sess_mut(dst)?, sess(src)?)
    }

    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Glm::feed(self, sess_mut(s)?, tokens, tap)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Glm::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Glm::forward(self, sess_mut(s)?, tokens, tap)
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Glm::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Glm::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = sessions.iter_mut().map(|s| sess_mut(s)).collect::<Result<_>>()?;
        engine::Glm::forward_batch(self, &mut own, tokens, tap)
    }

    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(engine::Glm::decoder(self, logits, draft))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(engine::Glm::decoder_after(self, next, draft))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| ns_core::Error("a decoder of another engine".into()))?;
        engine::Glm::step(self, sess_mut(s)?, d, smp, tap)
    }

    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Glm::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| ns_core::Error("a checkpoint of another engine".into()))?;
        engine::Glm::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, mut r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(&mut r)?))
    }

    fn capture_attention(&self, on: bool) {
        engine::Glm::capture_attention(self, on)
    }
    fn take_attention(&self) -> Option<Vec<f32>> {
        engine::Glm::take_attention(self)
    }

    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Glm::gpu_info(self)
    }
    fn expert_stats(&self) -> ExpertStats {
        let (vram_hits, swapped_in, from_host, read_direct, prefetched, prefetch_used) = engine::Glm::expert_stats(self);
        ExpertStats { vram_hits, swapped_in, from_host, read_direct, prefetched, prefetch_used }
    }
    fn hold_arena(&self, on: bool) {
        engine::Glm::hold_arena(self, on)
    }
    fn save_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        engine::Glm::save_expert_profile(self, path)
    }
    fn report(&self, tokens: usize) -> Vec<String> {
        let mut v = Vec::new();
        for (name, secs, calls) in engine::Glm::profile(self) {
            v.push(format!("[profile {name:<34} {secs:>7.2} s  {calls:>6} calls  {:>8.2} ms/token]", secs * 1000.0 / (tokens + 1) as f64));
        }
        v.push(format!("[host waited {:.2} s for the routers' logits, chose the experts in {:.2} s]",
                       engine::ROUTER_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9,
                       engine::ROUTE_HOST_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9));
        for (i, (peak, spills)) in engine::Glm::arena_peaks(self).iter().enumerate() {
            v.push(format!("[arena {i}: peak {:.2} GiB, {spills} request(s) past it]", *peak as f64 / (1u64 << 30) as f64));
        }
        v
    }
}
