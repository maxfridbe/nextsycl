//! Ricursive Ripple-65B (`ricursive`: a dense 65B with sliding-window RoPE layers and global NoPE layers, four norms a
//! block, an output gate and a value residual): its geometry from the file's metadata and its engine (`engine`: the
//! forward pass over layers split across the GPUs or streamed from pinned host memory, the rings and caches, the
//! sessions and checkpoints), behind the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry.
//! Its own kernels are in `kernels/llm/ricursive/` (the attention, the norms with the rotary positions, the gates), its
//! silo's in `kernels/llm/ricursive/silo/`; the products from the stored blocks are the glm5next engine's generic ones
//! in the same library.

mod chat;
pub mod engine;
mod ffi;
pub mod lookup;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

pub use chat::ripple_chat;

pub const ARCH: &str = "ricursive";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_RC_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default 2048 when they fit beside the weights and a session, else 1024 or 512)", at: At::Load },
    EngineOption { name: "host-layers", env: "NS_RC_HOST", value: "N", help: "layers kept in pinned host memory and read from it by the products, though VRAM has room (a measurement of streaming; layers that fit nowhere go there anyway)", at: At::Load },
    EngineOption { name: "stream", env: "NS_RC_STREAM", value: "direct|copy", help: "how a host layer is read: in place over PCIe (direct, the default) or copied to a staging buffer in VRAM first (copy)", at: At::Load },
    EngineOption { name: "host-spare-gib", env: "NS_RC_HOST_SPARE_GIB", value: "GIB", help: "host memory kept free whatever is pinned for streamed layers (default 8)", at: At::Load },
    EngineOption { name: "layers", env: "NS_RC_LAYERS", value: "N", help: "load only the first N layers, the head after them (a reference check on a truncated model)", at: At::Load },
    EngineOption { name: "flash", env: "NS_RC_FLASH", value: "0|1", help: "the silo's attention on XMX (default 1; 0 runs the shared kernel: a parity A/B)", at: At::Load },
    EngineOption { name: "mmvq", env: "NS_RC_MMVQ", value: "0|1", help: "0: every one-row pass through the half GEMM, not the Q8_1 decode products (a precision A/B; slow)", at: At::Load },
    EngineOption { name: "ngram", env: "NS_RC_NGRAM", value: "K", help: "prompt-lookup drafts: up to K (0..7) tokens that followed the last 3 earlier in the conversation, verified in one pass; exact (default 3; 0: off)", at: At::Load },
    EngineOption { name: "profile", env: "NS_RC_PROFILE", value: "0|1", help: "1: sync the GPU after each operation and print the time by operation every 32 one-row passes (slows the run)", at: At::Load },
    EngineOption { name: "dump", env: "NS_RC_DUMP", value: "DIR", help: "write every tapped tensor of a pass (embd, l_out-N, result_norm) there as float32 (a check against reference/ricursive-65b)", at: At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Ricursive Ripple (dense, sliding-window RoPE + global NoPE layers, gated attention, value residual: Ripple-65B ...)", load, info,
                 kernels: None, chat: ripple_chat, options: OPTIONS, prepare: None }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = engine::Geometry::read(f)?;
    let (bytes, types) = engine::check_tensors(f, &g)?;
    let global = g.swa.iter().filter(|s| !**s).count();
    Ok(format!("{ARCH}: {} layers of {} ({} sliding window of {} with rotary positions at {}, {global} global without; {} / {} heads of {}), a dense SwiGLU of {} clamped at {}, \
                embeddings scaled by {:.4}, vocabulary {}, context {}; {:.1} GiB of weights by tensor: {}", g.layers, g.n_embd, g.layers - global, g.window, g.theta, g.heads,
               g.heads_kv, g.head, g.n_ff, g.clamp, g.embed_scale, g.vocab, g.ctx, bytes as f64 / (1u64 << 30) as f64,
               types.iter().map(|(t, n)| format!("{t} x{n}")).collect::<Vec<_>>().join(", ")))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Ricursive::load(f, gpus, o.kv, log)?))
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
        (0, 0)
    }
    fn lookup_drafts(&self) -> (u64, u64) {
        (self.drafted, self.accepted)
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

impl Engine for engine::Ricursive<'_> {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        engine::Ricursive::load_seconds(self)
    }
    fn load_bytes(&self) -> u64 {
        engine::Ricursive::load_bytes(self)
    }
    fn has_draft(&self) -> bool {
        false
    }
    fn prefill_chunk(&self) -> usize {
        engine::Ricursive::prefill_chunk(self)
    }
    fn max_verify(&self) -> usize {
        // the prompt-lookup drafts' pass (NS_RC_NGRAM): a decode-width pass at most
        engine::MAX_ROWS
    }
    fn cache_fingerprint(&self) -> String {
        // checkpoints hold the attention caches in half, the sliding layers' last window and the global layers' all
        format!("ricursive f16-kv nsrc-1 w{}", self.g.window)
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Ricursive::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Ricursive::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let src = sess(src)?;
        engine::Ricursive::copy_session(self, sess_mut(dst)?, src)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::Ricursive::feed_until(self, sess_mut(s)?, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Ricursive::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::Ricursive::forward_rows(self, sess_mut(s)?, tokens, 1, tap)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Ricursive::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Ricursive::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = Vec::with_capacity(sessions.len());
        for s in sessions.iter_mut() {
            own.push(s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))?);
        }
        engine::Ricursive::forward_batch(self, &mut own, tokens, tap)
    }
    fn decoder(&self, logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(logits))
    }
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::after(next))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Ricursive::step(self, sess_mut(s)?, d, smp, tap)
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Ricursive::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::Ricursive::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(r)?))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Ricursive::gpu_info(self)
    }
    fn report(&self, _tokens: usize) -> Vec<String> {
        if self.streams() > 0 {
            vec![format!("ricursive: {} layers streamed from pinned host memory", self.streams())]
        } else {
            Vec::new()
        }
    }
}
