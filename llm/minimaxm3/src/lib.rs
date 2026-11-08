//! MiniMax-M3.1-Flash (`minimax-m3`): its model description (`model`: the geometry from the metadata, every tensor by
//! role) and its engine (`engine`: the forward pass with MiniMax Sparse Attention, the routed experts, the sessions
//! and checkpoints), behind the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry. Its own
//! kernels are in `kernels/llm/minimaxm3/`; the dequantization, GEMMs and norms are the glm5next engine's generic
//! kernels in the same library. docs/ports/minimax-m31.md has the notes.

mod cpu;
pub mod engine;
mod grids;
mod ffi;
pub mod model;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

/// The architecture this engine is, and the silo's name
pub const ARCH: &str = "minimaxm3";

/// The model description's errors (a file's layout against what the engine needs)
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type ModelResult<T> = std::result::Result<T, Error>;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_MM31_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default 4096: every pass streams the experts that are off the cards once)", at: At::Load },
    EngineOption { name: "vram-gib", env: "NS_MM31_VRAM_GIB", value: "GIB", help: "VRAM a card may fill with kept experts (default: what is free after the arena, the sessions' caches and a spare)", at: At::Load },
    EngineOption { name: "pin-gib", env: "NS_MM31_PIN_GIB", value: "GIB", help: "pinned host memory, in all, for more kept experts (default: what is available less 16, at most 40)", at: At::Load },
    EngineOption { name: "split", env: "NS_MM31_SPLIT", value: "LAYERS", help: "with two GPUs, the layers on the first, the B65 unless NS_MM31_ORDER=asis (default: 40%)", at: At::Load },
    EngineOption { name: "direct", env: "NS_MM31_DIRECT", value: "0|1", help: "1: O_DIRECT all the time; 0: never; default: at load, then through the page cache", at: At::Load },
    EngineOption { name: "splitread", env: "NS_MM31_SPLITREAD", value: "N", help: "pieces an expert matrix is read in at once, each its own request (default 8)", at: At::Load },
    EngineOption { name: "lane", env: "NS_MM31_LANE", value: "0|1", help: "the copy lane the pinned and file-read experts are copied into VRAM on (default 1)", at: At::Load },
    EngineOption { name: "split-attend", env: "NS_MM31_SPLIT_ATTEND", value: "0|1", help: "0: decode passes use the prompt-pass attention kernel instead of the silo's key-split one", at: At::Load },
    EngineOption { name: "fused", env: "NS_MM31_FUSED", value: "0|1", help: "0: decode experts run as the shared kernels' three products each, not the silo's fused gate+up and down launches", at: At::Load },
    EngineOption { name: "dynamic", env: "NS_MM31_DYNAMIC", value: "0|1", help: "0: the experts stay where the load put them; by default decode keeps each layer's most recently used experts in VRAM and the next in pinned memory", at: At::Load },
    EngineOption { name: "ahead", env: "NS_MM31_AHEAD", value: "N", help: "decode: experts of the layers ahead guessed and read ahead (default 0 = none, at most 12)", at: At::Load },
    EngineOption { name: "stream-dir", env: "NS_MM31_STREAM_DIR", value: "DIR", help: "a directory on another drive: a copy of a share of the experts that are off the cards is written there once and read beside the model's file", at: At::Load },
    EngineOption { name: "stream-share", env: "NS_MM31_STREAM_SHARE", value: "PERCENT", help: "with stream-dir, the share of the off-card experts read from that drive (default 50)", at: At::Load },
    EngineOption { name: "sparse-all", env: "NS_MM31_SPARSE_ALL", value: "0|1", help: "the first three layers take the indexer too (default 0: llama.cpp's graph)", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["minimax-m3"], name: "MiniMax-M3.1-Flash (GQA with MiniMax Sparse Attention, 128 experts + 1 shared)", load, info, kernels: None,
                 chat: nextsycl_tok::minimax_chat, options: OPTIONS, prepare: None }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = model::Geometry::from_gguf(f).map_err(|e| e.0)?;
    g.check(f).map_err(|e| e.0)?;
    Ok(format!("minimax-m3: {} layers of {} ({} dense, {} MoE: {} of {} experts of {} and a shared one; {} / {} heads of {}, {} rotated, rope {}; sparse attention: {} index heads, \
                {} blocks of {} and {} local), vocabulary {}",
               g.n_layer, g.n_embd, g.n_dense, g.n_layer - g.n_dense, g.n_expert_used, g.n_expert, g.n_ff_expert, g.n_head, g.n_kv_head, g.head_dim, g.n_rot, g.rope_base,
               g.indexer.n_head, g.indexer.top_k, g.indexer.block_size, g.indexer.local_blocks, g.n_vocab))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::MiniMax::load(f, gpus, o, log)?))
}

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

impl Engine for engine::MiniMax<'_> {
    fn arch(&self) -> &'static str {
        "minimax-m3"
    }
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }
    fn load_bytes(&self) -> u64 {
        self.load_bytes
    }
    fn has_draft(&self) -> bool {
        self.draft_count() > 0
    }
    fn prefill_chunk(&self) -> usize {
        engine::prefill_chunk()
    }
    fn max_verify(&self) -> usize {
        1
    }
    fn cache_fingerprint(&self) -> String {
        "minimax-m3 kv=f16 mm31ck1".to_string()
    }

    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::MiniMax::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::MiniMax::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        engine::MiniMax::copy_session(self, sess_mut(dst)?, sess(src)?)
    }

    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::MiniMax::feed_until(self, sess_mut(s)?, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::MiniMax::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::MiniMax::forward_rows(self, sess_mut(s)?, tokens, 1, tap)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::MiniMax::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::MiniMax::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = sessions.iter_mut().map(|s| sess_mut(s)).collect::<Result<_>>()?;
        engine::MiniMax::forward_batch(self, &mut own, tokens, tap)
    }

    fn decoder(&self, logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(logits))
    }
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::after(next))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::MiniMax::step(self, sess_mut(s)?, d, smp, tap)
    }

    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::MiniMax::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::MiniMax::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, mut r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(&mut r)?))
    }

    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::MiniMax::gpu_info(self)
    }
    fn expert_stats(&self) -> nextsycl_llm::ExpertStats {
        engine::MiniMax::expert_stats(self)
    }
    fn save_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        engine::MiniMax::save_expert_profile(self, path)
    }
    fn report(&self, _tokens: usize) -> Vec<String> {
        self.profile_lines()
    }
}
