//! DeepSeek-V3 / R1 (`deepseek2`): its model description (`model`), its engine (`engine`: MLA absorbed over an fp16
//! latent cache, the routed experts in a three-tier store - VRAM, pinned host memory, the file - read ahead of their
//! layers, `store`), behind the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry. Its own
//! kernels are in `kernels/llm/deepseek2/` (IQ1_S, the layer's experts in one launch, YaRN's RoPE, the cache rows).

pub mod cpu;
pub mod engine;
pub mod ffi;
pub mod iq1s_grid;
pub mod model;
pub mod ops;
pub mod store;
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
    EngineOption { name: "chunk", env: "NS_DS2_CHUNK", value: "TOKENS", help: "a prompt pass's tokens (default 1024)", at: At::Load },
    EngineOption { name: "arena-mib", env: "NS_DS2_ARENA_MIB", value: "MIB", help: "the passes' temporaries", at: At::Load },
    EngineOption { name: "groups", env: "NS_DS2_GROUPS", value: "G,K|0", help: "routing within the best K of G expert groups (default 8,4; 0 none)", at: At::Load },
    EngineOption { name: "prefetch", env: "NS_DS2_PREFETCH", value: "N", help: "experts a layer loads ahead of the next (default 0)", at: At::Load },
    EngineOption { name: "prefetch-p", env: "NS_DS2_PREFETCH_P", value: "P", help: "the guessed chance of use from which an expert is loaded ahead (0.5)", at: At::Load },
    EngineOption { name: "readers", env: "NS_DS2_READERS", value: "N", help: "threads reading experts from the file (default 16)", at: At::Load },
    EngineOption { name: "multi-max", env: "NS_DS2_MULTI_MAX", value: "TOKENS", help: "passes up to this many tokens run the experts in the decode kernels (32)", at: At::Load },
    EngineOption { name: "expert-profile", env: "NS_EXPERT_PROFILE", value: "FILE", help: "which experts fill VRAM and pinned memory first", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["deepseek2"], name: "DeepSeek-V3 / R1 (MLA with YaRN, 256 experts in VRAM, pinned memory and the file)", load, info: tools::info,
                 kernels: Some(tools::kernels), chat: nextsycl_tok::deepseek_chat, options: OPTIONS }
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Ds::load(f, gpus, o.expert_bytes, o.mirror_bytes, o.kv, log)?))
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

impl Engine for engine::Ds<'_> {
    fn arch(&self) -> &'static str {
        "deepseek2"
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
        engine::prefill_chunk()
    }
    fn max_verify(&self) -> usize {
        1
    }
    fn cache_fingerprint(&self) -> String {
        "deepseek2 f16-latents nsds1".into()
    }

    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Ds::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Ds::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        engine::Ds::copy_session(self, sess_mut(dst)?, sess(src)?)
    }

    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Ds::feed(self, sess_mut(s)?, tokens, tap)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Ds::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Ds::forward(self, sess_mut(s)?, tokens, tap)
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Ds::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Ds::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = sessions.iter_mut().map(|s| sess_mut(s)).collect::<Result<_>>()?;
        engine::Ds::forward_batch(self, &mut own, tokens, tap)
    }

    fn decoder(&self, logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(engine::Ds::decoder(self, logits))
    }
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(engine::Ds::decoder_after(self, next))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Ds::step(self, sess_mut(s)?, d, smp, tap)
    }

    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Ds::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::Ds::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, mut r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(&mut r)?))
    }

    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Ds::gpu_info(self)
    }
    fn expert_stats(&self) -> ExpertStats {
        let (s, _) = engine::Ds::expert_stats(self);
        ExpertStats { vram_hits: s.hits, swapped_in: s.from_host + s.from_file, from_host: s.from_host, read_direct: s.direct, prefetched: s.prefetched + s.file_prefetched,
                      prefetch_used: s.prefetch_used }
    }
    fn report(&self, tokens: usize) -> Vec<String> {
        let mut v = Vec::new();
        for (name, secs, calls) in engine::Ds::profile(self) {
            v.push(format!("[profile {name:<34} {secs:>7.2} s  {calls:>6} calls  {:>8.2} ms/token]", secs * 1000.0 / (tokens + 1) as f64));
        }
        let (s, file) = engine::Ds::expert_stats(self);
        v.push(format!("[experts: {} VRAM hits, {} on the CPU, {} up from pinned memory, {} from the file ({:.2} GiB read), {} read in place by prompts; ahead: {} up, {} from the file, {} promoted, {} of those used]",
                       s.hits, s.on_cpu, s.from_host, s.from_file, file as f64 / (1u64 << 30) as f64, s.direct, s.prefetched, s.file_prefetched, s.promoted, s.prefetch_used));
        v.push(format!("[host waited {:.2} s for the routers' logits, chose the experts in {:.2} s]",
                       self.router_wait.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9,
                       self.route_host.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9));
        v
    }
}
