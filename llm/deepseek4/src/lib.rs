//! DeepSeek-V4-Flash (`deepseek4`): its model description (`model`: the geometry from the file's metadata, every tensor
//! by role) and its engine (`engine`: the forward passes - attention over a window of raw keys and the compressed
//! blocks, the compressors and the lightning indexer, the mHC streams - with glm5next's expert store and MoE paths),
//! behind the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry. Its own kernels are its silo,
//! `kernels/llm/deepseek4/silo/` (dist/silo/libnextsycl-deepseek4.so).

/// The architecture the file names, and the silo's name
pub const ARCH: &str = "deepseek4";

pub mod engine;
pub mod ffi;
pub mod model;
pub mod mtp;
pub mod ops;
pub mod tools;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, ExpertStats, GpuInfo, LoadOptions, Result, Sampler,
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

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

pub type ModelResult<T> = std::result::Result<T, Error>;

/// The options it takes (`--opt-NAME`; each sets the variable named, which the engine reads at load)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "prefill-chunk", env: "NS_PREFILL_CHUNK", value: "TOKENS", help: "a prompt chunk's tokens (default 4096)", at: At::Load },
    EngineOption { name: "pipeline", env: "NS_PIPELINE", value: "0|1", help: "a long prompt's chunks pipelined over two GPUs (default 1)", at: At::Load },
    EngineOption { name: "ngram", env: "NS_NGRAM", value: "K", help: "up to K tokens drafted from the context's repeats, verified in one pass (default 0)", at: At::Load },
    EngineOption { name: "prefetch", env: "NS_PREFETCH", value: "N", help: "experts swapped in ahead of their layer (default 8; 0 none)", at: At::Load },
    EngineOption { name: "lend", env: "NS_LEND", value: "0|1", help: "the prompt arena holds experts while no chunk needs it (default 1)", at: At::Load },
    EngineOption { name: "expert-profile", env: "NS_EXPERT_PROFILE", value: "FILE", help: "which experts fill VRAM first", at: At::Load },
    EngineOption { name: "arena-mib", env: "NS_ARENA_MIB", value: "MIB", help: "the prompt path's temporaries", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["deepseek4"], name: "DeepSeek-V4 (a window of raw keys + compressed blocks with a lightning indexer, mHC, 256 experts)", load,
                 info: tools::info, kernels: Some(tools::kernels), chat: nextsycl_tok::deepseek4_chat, options: OPTIONS, prepare: Some(mtp::prepare) }
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    // with a draft asked for and the sidecar there, the model again with it as a further shard (held for the process)
    let side = f.paths.first().and_then(|p| p.parent()).map(|d| d.join(mtp::SIDECAR)).filter(|p| p.is_file());
    let f: &'g Gguf = match (o.draft, side) {
        (true, Some(s)) if f.tensor("blk.43.nextn.e_proj.weight").is_none() => {
            log(format!("deepseek4: the draft block from {}", s.display()));
            Box::leak(Box::new(Gguf::open_with(&f.paths[0], &[s]).map_err(|e| nextsycl_core::Error(e.0))?))
        }
        _ => f,
    };
    Ok(Box::new(engine::Ds4::load(f, gpus, o.expert_bytes, o.mirror_bytes, o.kv, o.draft, log)?))
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
    s.get().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))
}
fn sess_mut(s: &mut Session) -> Result<&mut engine::Session> {
    s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))
}

impl Engine for engine::Ds4<'_> {
    fn arch(&self) -> &'static str {
        "deepseek4"
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
        // the prompt-lookup drafts' pass (NS_NGRAM)
        (engine::ngram_k() + 1).max(if self.mtp.is_some() { 3 } else { 1 }).min(engine::MAX_VERIFY)
    }
    fn cache_fingerprint(&self) -> String {
        format!("deepseek4 raw-ring={} nsd42{}", engine::raw_ring(), if self.mtp.is_some() { " mtp" } else { "" })
    }

    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Ds4::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Ds4::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        engine::Ds4::copy_session(self, sess_mut(dst)?, sess(src)?)
    }

    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Ds4::feed(self, sess_mut(s)?, tokens, tap)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Ds4::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Ds4::forward(self, sess_mut(s)?, tokens, tap)
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Ds4::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Ds4::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = sessions.iter_mut().map(|s| sess_mut(s)).collect::<Result<_>>()?;
        engine::Ds4::forward_batch(self, &mut own, tokens, tap)
    }

    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(engine::Ds4::decoder(self, logits, draft))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(engine::Ds4::decoder_after(self, next, draft))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Ds4::step(self, sess_mut(s)?, d, smp, tap)
    }

    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Ds4::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::Ds4::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, mut r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(&mut r)?))
    }

    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Ds4::gpu_info(self)
    }
    fn expert_stats(&self) -> ExpertStats {
        let (vram_hits, swapped_in, from_host, read_direct, prefetched, prefetch_used) = engine::Ds4::expert_stats(self);
        ExpertStats { vram_hits, swapped_in, from_host, read_direct, prefetched, prefetch_used }
    }
    fn hold_arena(&self, on: bool) {
        engine::Ds4::hold_arena(self, on)
    }
    fn save_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        engine::Ds4::save_expert_profile(self, path)
    }
    fn report(&self, tokens: usize) -> Vec<String> {
        let mut v = Vec::new();
        for (name, secs, calls) in engine::Ds4::profile(self) {
            v.push(format!("[profile {name:<34} {secs:>7.2} s  {calls:>6} calls  {:>8.2} ms/token]", secs * 1000.0 / (tokens + 1) as f64));
        }
        v.push(format!("[host waited {:.2} s for the routers' logits, chose the experts in {:.2} s]",
                       engine::ROUTER_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9,
                       engine::ROUTE_HOST_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9));
        for (i, (peak, spills)) in engine::Ds4::arena_peaks(self).iter().enumerate() {
            v.push(format!("[arena {i}: peak {:.2} GiB, {spills} request(s) past it]", *peak as f64 / (1u64 << 30) as f64));
        }
        v
    }
}
