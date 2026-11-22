//! Inkling-2-Mini (`inkling2_loop`): its geometry from the file's metadata and its engine (`engine`: the looped
//! layers with their controllers, the depth attention, the relative-bias attention with sinks, the short
//! convolutions, the NVFP4 MoE, the sessions and checkpoints), behind the runtime's `Engine` contract
//! (`nextsycl-llm`). `kind()` is its registry entry. Its own kernels are in `kernels/llm/inkling2/`; the RMS norm and
//! the GEMMs are the glm5next engine's generic kernels in the same library.

pub mod engine;
mod ffi;
mod tools;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::EngineOption;
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

pub const ARCH: &str = "inkling2_loop";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_INK2_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default 2048 when it fits beside the weights and a session, else 1024 or 512)", at: nextsycl_llm::At::Load },
    EngineOption { name: "draft", env: "NS_INK2_DRAFT", value: "FILE", help: "the DSpark drafter's GGUF (default: the smallest dspark-*.gguf beside the model)", at: nextsycl_llm::At::Load },
    EngineOption { name: "drafts", env: "NS_INK2_DRAFTS", value: "N", help: "tokens drafted a step (default 5; at most 7; 0: none)", at: nextsycl_llm::At::Load },
    EngineOption { name: "draft-pmin", env: "NS_INK2_DRAFT_PMIN", value: "P", help: "drop a draft (and those after it) the drafter gives less than this probability (default 0.8)", at: nextsycl_llm::At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Inkling 2 (a looped MoE: depth attention, relative-bias attention with sinks, short convolutions, NVFP4 experts: Inkling-2-Mini)",
                 load, info, kernels: Some(tools::kernels), chat: nextsycl_tok::inkling_chat, options: OPTIONS, prepare: None }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = engine::Geometry::read(f)?;
    let windowed = g.swa.iter().filter(|s| **s).count();
    Ok(format!("{ARCH}: {} layers of {} (layers {}-{} run {} times: {} applications a token; {windowed} windowed over {}, {} full), {} / {} heads of {}, \
                {} experts ({} a token) of {}, depth blocks of {}, logits capped at {}, vocabulary {} (padded {}), context {}", g.layers, g.n_embd, g.core.0,
               g.core.1 - 1, g.passes, g.order().len(), g.window, g.layers - windowed, g.heads, g.heads_kv, g.head, g.experts, g.used, g.n_ff, g.block, g.softcap,
               g.vocab, g.vocab_pad, g.ctx))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Inkling2::load(f, gpus, o.kv, o.draft, log)?))
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

impl Engine for engine::Inkling2<'_> {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        engine::Inkling2::load_seconds(self)
    }
    fn load_bytes(&self) -> u64 {
        engine::Inkling2::load_bytes(self)
    }
    fn has_draft(&self) -> bool {
        engine::Inkling2::has_draft(self)
    }
    fn prefill_chunk(&self) -> usize {
        engine::Inkling2::prefill_chunk(self)
    }
    fn max_verify(&self) -> usize {
        engine::Inkling2::max_verify(self)
    }
    fn cache_fingerprint(&self) -> String {
        format!("inkling2 f16-kv conv-state nsink-1 chunk={}{}", self.prefill_chunk(), if self.has_draft() { " dspark" } else { "" })
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Inkling2::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Inkling2::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let src = sess(src)?;
        engine::Inkling2::copy_session(self, sess_mut(dst)?, src)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::Inkling2::feed_until(self, sess_mut(s)?, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Inkling2::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::Inkling2::forward_rows(self, sess_mut(s)?, tokens, 1, tap)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Inkling2::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Inkling2::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = Vec::with_capacity(sessions.len());
        for s in sessions.iter_mut() {
            own.push(s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))?);
        }
        engine::Inkling2::forward_batch(self, &mut own, tokens, tap)
    }
    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(logits, draft))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::after(next, draft))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Inkling2::step(self, sess_mut(s)?, d, smp, tap)
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Inkling2::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::Inkling2::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(r)?))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Inkling2::gpu_info(self)
    }
    fn report(&self, tokens: usize) -> Vec<String> {
        engine::Inkling2::profile_lines(self, tokens)
    }
}
