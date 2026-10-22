//! DeepSeek-V4.1-Flash (`deepseek41`; its files say `deepseek4`, variant `v41_flash`): its model description (`model`:
//! the geometry from the metadata, every tensor by role, YoungAi's VQ expert blobs), its Engram (`engram`: the n-gram
//! hash and the table rows read from disk) and its engine (`engine`: the forward passes with the shared compressed keys
//! and indexer picks, the mHC halves collapsed by the previous half's mix, the VQ experts in deepseek4's expert store,
//! the three DSpark draft towers), behind the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry
//! entry. Its kernels are its silo, `kernels/llm/deepseek41/silo/` (dist/silo/libnextsycl-deepseek41.so).

/// The architecture this engine is, and the silo's name
pub const ARCH: &str = "deepseek41";

pub mod engine;
pub mod engram;
pub mod requant;
pub mod ffi;
pub mod model;
pub mod ops;
pub mod tools;
pub mod vision;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, ExpertStats, GpuInfo, LoadOptions, Result, Sampler, Session,
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

/// The options it takes (`--opt-NAME`; each sets the variable named, which the engine reads at load)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "engram-dir", env: "NS_DS41_ENGRAM_DIR", value: "DIR", help: "the two Engram shards' directory (default: deepseek-engram beside the model)", at: At::Load },
    EngineOption { name: "engram-readers", env: "NS_DS41_ENGRAM_READERS", value: "N", help: "Engram row reads in flight (default 48)", at: At::Load },
    EngineOption { name: "engram-cache", env: "NS_DS41_ENGRAM_CACHE", value: "ROWS", help: "Engram rows kept in host memory (default 262144)", at: At::Load },
    EngineOption { name: "mtp", env: "NS_DS41_MTP", value: "0|1", help: "the DSpark draft towers loaded and drafting (default 0: they cost more than they save on prose)", at: At::Load },
    EngineOption { name: "drafts", env: "NS_DS41_DRAFTS", value: "K", help: "the towers' drafts a verify pass checks (1..7, default 2)", at: At::Load },
    EngineOption { name: "pin-gib", env: "NS_DS41_PIN_GIB", value: "GIB", help: "pinned host memory for experts (default 40)", at: At::Load },
    EngineOption { name: "free-gib", env: "NS_DS41_FREE_GIB", value: "GIB", help: "VRAM each GPU keeps free beyond the plan (default 0.75: 2.4 GiB measured free at the peak)", at: At::Load },
    EngineOption { name: "prefill-chunk", env: "NS_PREFILL_CHUNK", value: "TOKENS", help: "a prompt chunk's tokens (default 2048)", at: At::Load },
    EngineOption { name: "pipeline", env: "NS_PIPELINE", value: "0|1", help: "a long prompt's chunks pipelined over two GPUs (default 1)", at: At::Load },
    EngineOption { name: "prefetch", env: "NS_PREFETCH", value: "N", help: "experts swapped in ahead of their layer (default 8; 0 none)", at: At::Load },
    EngineOption { name: "lend", env: "NS_LEND", value: "0|1", help: "the prompt arena holds experts while no chunk needs it (default 1)", at: At::Load },
    EngineOption { name: "expert-profile", env: "NS_EXPERT_PROFILE", value: "FILE", help: "which experts fill VRAM first", at: At::Load },
    EngineOption { name: "arena-mib", env: "NS_ARENA_MIB", value: "MIB", help: "the prompt path's temporaries", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["deepseek4.v41_flash", "deepseek41"], name: "DeepSeek-V4.1 (shared compressed keys and indexer picks, Engram, VQ experts, DSpark towers)", load,
                 info: tools::info, kernels: Some(tools::kernels), chat: nextsycl_tok::deepseek41_chat, options: OPTIONS, prepare: None }
}

/// The DSpark drafter file beside an Unsloth model (NS_DS41_DRAFTER names it; else `dspark-*.gguf` in the model's folder or
/// its parent's)
fn drafter(f: &Gguf) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NS_DS41_DRAFTER") {
        return Some(p.into()).filter(|p: &std::path::PathBuf| p.is_file());
    }
    let dir = f.paths.first()?.parent()?;
    [Some(dir), dir.parent()].into_iter().flatten().find_map(|d| {
        std::fs::read_dir(d).ok()?.flatten().map(|e| e.path()).find(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("dspark-") && n.ends_with(".gguf"))
        })
    })
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    // an Unsloth model with drafts asked for (NS_DS41_MTP=1) and its drafter file there: the model again with the drafter as
    // a further shard, its tensors renamed to the towers' (held for the process)
    let f: &'g Gguf = match (f.architecture() == "deepseek41" && o.draft && engine::mtp_on(), drafter(f)) {
        (true, Some(d)) if f.tensor("mtp.0.attn_norm.weight").is_none() => {
            log(format!("deepseek41: the DSpark drafter from {}", d.display()));
            Box::leak(Box::new(Gguf::open_with_map(&f.paths[0], &[(d, model::dflash_name as fn(&str) -> String)]).map_err(|e| nextsycl_core::Error(e.0))?))
        }
        _ => f,
    };
    Ok(Box::new(engine::Ds41::load(f, gpus, o.expert_bytes, o.mirror_bytes, o.kv, o.draft, log)?))
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

impl Engine for engine::Ds41<'_> {
    fn arch(&self) -> &'static str {
        ARCH
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
        engine::prefill_chunk()
    }
    fn max_verify(&self) -> usize {
        if self.mtp { engine::mtp_drafts() + 1 } else { 1 }
    }
    fn cache_fingerprint(&self) -> String {
        format!("deepseek41 raw-ring={} nsd41{}", engine::raw_ring(), if self.mtp { " mtp" } else { "" })
    }

    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Ds41::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Ds41::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        engine::Ds41::copy_session(self, sess_mut(dst)?, sess(src)?)
    }

    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Ds41::feed(self, sess_mut(s)?, tokens, tap)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Ds41::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        engine::Ds41::forward(self, sess_mut(s)?, tokens, tap)
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Ds41::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Ds41::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = sessions.iter_mut().map(|s| sess_mut(s)).collect::<Result<_>>()?;
        engine::Ds41::forward_batch(self, &mut own, tokens, tap)
    }

    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(engine::Ds41::decoder(self, logits, draft))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(engine::Ds41::decoder_after(self, next, draft))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Ds41::step(self, sess_mut(s)?, d, smp, tap)
    }

    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Ds41::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::Ds41::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, mut r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(&mut r)?))
    }

    fn image_marker(&self) -> Option<&str> {
        self.vision.as_ref().map(|v| v.marker.as_str())
    }
    fn image_tokens(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let v = self.vision.as_ref().ok_or_else(|| nextsycl_core::Error("no vision sidecar (NS_DS41_VISION)".into()))?;
        v.tokens(bytes).map_err(|e| nextsycl_core::Error(e.0))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Ds41::gpu_info(self)
    }
    fn expert_stats(&self) -> ExpertStats {
        let (vram_hits, swapped_in, from_host, read_direct, prefetched, prefetch_used) = engine::Ds41::expert_stats(self);
        ExpertStats { vram_hits, swapped_in, from_host, read_direct, prefetched, prefetch_used }
    }
    fn hold_arena(&self, on: bool) {
        engine::Ds41::hold_arena(self, on)
    }
    fn save_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        engine::Ds41::save_expert_profile(self, path)
    }
    fn report(&self, tokens: usize) -> Vec<String> {
        let mut v = Vec::new();
        for (name, secs, calls) in engine::Ds41::profile(self) {
            v.push(format!("[profile {name:<34} {secs:>7.2} s  {calls:>6} calls  {:>8.2} ms/token]", secs * 1000.0 / (tokens + 1) as f64));
        }
        let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9;
        v.push(format!("[host waited {:.2} s for the routers' logits, chose the experts in {:.2} s; {} experts read from the file in {:.2} s; decode's turns from \
                        logits to products queued {:.2} s]", ld(&engine::ROUTER_WAIT_NS), ld(&engine::ROUTE_HOST_NS), self.file_reads(), ld(&engine::FILE_NS),
                       ld(&engine::TURN_NS)));
        v.push(format!("[expert misses: waited {:.2} s for read-aheads, {:.2} s in the staged reads (of it {:.2} s for a staging slot), {:.2} s submitting the swaps, \
                        {:.2} s for copies the products needed]", ld(&engine::ENS_AHEAD_NS), ld(&engine::ENS_STAGE_NS), ld(&engine::ENS_SLOT_NS), ld(&engine::ENS_SWAP_NS),
                       ld(&engine::ENS_COPY_NS)));
        {
            let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
            v.push(format!("[cold experts left out (NS_DS41_COLD_SKIP): {}]", ld(&engine::COLD_SKIPPED)));
            v.push(format!("[decode's file reads ahead: {} one layer ahead, {} further, {} of them taken; {} read by the layer itself]",
                           ld(&engine::AHEAD_NEAR), ld(&engine::AHEAD_FAR), ld(&engine::AHEAD_TAKEN), ld(&engine::STAGED_READS)));
            v.push(format!("[cold experts (neither in VRAM nor host memory) of {} decode layers: {} layers have one, {} of them all under 0.1 of the routing weight; by share <0.05 / <0.1 / <0.15 / <0.2 / <0.3 / more: {}]",
                           ld(&engine::DEC_LAYERS), ld(&engine::COLD_LAYERS), ld(&engine::COLD_LAYERS_LOW),
                           engine::COLD_BINS.iter().map(ld).map(|x| x.to_string()).collect::<Vec<_>>().join(" / ")));
        }
        let gn = engine::GUESS_N.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64;
        v.push(format!("[the guess's k-th best expert among the next layer's picks (of {gn} decode layers), k = 1..16: {}]",
                       engine::GUESS_HITS.iter().map(|h| format!("{:.2}", h.load(std::sync::atomic::Ordering::Relaxed) as f64 / gn)).collect::<Vec<_>>().join(" ")));
        v.push(format!("[wall: embedding + engram start {:.2} s, layers {:.2} s, head {:.2} s, sampling {:.2} s - passes of every size, the prompt's too]",
                       ld(&engine::WALL_STREAMS_NS), ld(&engine::WALL_LAYERS_NS), ld(&engine::WALL_HEAD_NS), ld(&engine::WALL_SAMPLE_NS)));
        if let Some(en) = &self.engram {
            let (rows, cached, reads, bytes, mean, max, waited) = en.stats();
            v.push(format!("[engram: {rows} rows read ({reads} reads, {:.1} MB, {mean:.0} us each on average, {max:.0} us the longest), {cached} from the cache, \
                            the passes waited {waited:.2} s ({:.2} ms a pass token)]", bytes as f64 / 1e6, waited * 1e3 / (tokens + 1) as f64));
        }
        for (i, (peak, spills)) in engine::Ds41::arena_peaks(self).iter().enumerate() {
            v.push(format!("[arena {i}: peak {:.2} GiB, {spills} request(s) past it]", *peak as f64 / (1u64 << 30) as f64));
        }
        v
    }
}
