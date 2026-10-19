//! Dense Qwen3.5 / Qwen3.8 (`qwen35`: Qwen3.8-27B and its siblings): its geometry from the file's metadata and its
//! engine (`engine`: gated DeltaNet and gated full attention, the dense SwiGLU, the sessions and checkpoints), behind
//! the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry. Its own kernels are in
//! `kernels/llm/qwen35/`; the products from the stored blocks, the norms, the conv and the delta-rule scan are the
//! glm5next engine's generic kernels in the same library.

pub mod engine;
mod ffi;
mod probe;
pub mod vision;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

pub const ARCH: &str = "qwen35";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_Q35_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default 2048 when it fits beside the weights and a session, else 1024 or 512)", at: At::Load },
    EngineOption { name: "mtp", env: "NS_Q35_MTP", value: "FILE", help: "the MTP draft block's GGUF (default: the one in an MTP/ folder beside the model)", at: At::Load },
    EngineOption { name: "vision", env: "NS_Q35_VISION", value: "FILE|0", help: "the vision tower's mmproj GGUF (default: an mmproj*.gguf beside the model; 0: no pictures)", at: At::Load },
    EngineOption { name: "image-tokens", env: "NS_Q35_IMAGE_TOKENS", value: "N", help: "the most prompt positions a picture takes (default 4096; larger pictures are scaled down)", at: At::Load },
    EngineOption { name: "drafts", env: "NS_Q35_DRAFTS", value: "N", help: "tokens drafted a step with MTP (default 3, at most 7; 0: none)", at: At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Qwen3.5 / Qwen3.8 dense (gated DeltaNet + gated full attention, SwiGLU: Qwen3.8-27B ...)", load, info,
                 kernels: None, chat: nextsycl_tok::qwen_chat, options: OPTIONS, prepare: None }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = engine::Geometry::read(f)?;
    let full = (0..g.layers).filter(|l| g.full(*l)).count();
    Ok(format!("{ARCH}: {} layers of {} ({full} gated full attention: {} / {} heads of {}, rotary {} at {}; {} gated DeltaNet: {} / {} heads of {}), \
                a dense SwiGLU of {}, vocabulary {}, context {}", g.layers, g.n_embd, g.heads, g.heads_kv, g.head, g.n_rot, g.theta,
               g.layers - full, g.k_heads, g.v_heads, g.state, g.n_ff, g.vocab, g.ctx))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Qwen35::load(f, gpus, o.draft, o.kv, log)?))
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

impl Engine for engine::Qwen35<'_> {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        engine::Qwen35::load_seconds(self)
    }
    fn load_bytes(&self) -> u64 {
        engine::Qwen35::load_bytes(self)
    }
    fn has_draft(&self) -> bool {
        engine::Qwen35::has_draft(self)
    }
    fn prefill_chunk(&self) -> usize {
        engine::Qwen35::prefill_chunk(self)
    }
    fn max_verify(&self) -> usize {
        engine::Qwen35::max_verify(self)
    }
    fn cache_fingerprint(&self) -> String {
        // checkpoints hold the draft block's cache too when it is loaded
        if self.has_draft() { "qwen35 f16-kv nsq35-1 mtp".into() } else { "qwen35 f16-kv nsq35-1".into() }
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::Qwen35::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::Qwen35::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let src = sess(src)?;
        engine::Qwen35::copy_session(self, sess_mut(dst)?, src)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::Qwen35::feed_until(self, sess_mut(s)?, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::Qwen35::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::Qwen35::forward_rows(self, sess_mut(s)?, tokens, 1, tap)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::Qwen35::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::Qwen35::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = Vec::with_capacity(sessions.len());
        for s in sessions.iter_mut() {
            own.push(s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))?);
        }
        engine::Qwen35::forward_batch(self, &mut own, tokens, tap)
    }
    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(logits, draft))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::after(next, draft))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Qwen35::step(self, sess_mut(s)?, d, smp, tap)
    }
    fn steps_together(&self) -> bool {
        engine::Qwen35::steps_together(self)
    }
    fn step_many(&self, ss: &mut [&mut Session], ds: &mut [&mut Decoder], smps: &mut [&mut dyn Sampler], tap: Tap) -> Result<Vec<Vec<u32>>> {
        let mut own: Vec<&mut engine::Session> = Vec::with_capacity(ss.len());
        for s in ss.iter_mut() {
            own.push(s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))?);
        }
        let mut dec: Vec<&mut engine::Decoder> = Vec::with_capacity(ds.len());
        for d in ds.iter_mut() {
            dec.push(d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?);
        }
        engine::Qwen35::step_many(self, &mut own, &mut dec, smps, tap)
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Qwen35::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::Qwen35::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(r)?))
    }
    fn capture_attention(&self, on: bool) {
        engine::Qwen35::capture_attention(self, on)
    }
    fn take_attention(&self) -> Option<Vec<f32>> {
        engine::Qwen35::take_attention(self)
    }
    fn image_marker(&self) -> Option<&str> {
        self.vision.as_ref().map(|v| v.marker.as_str())
    }
    fn image_tokens(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let v = self.vision.as_ref().ok_or_else(|| nextsycl_core::Error("no vision tower (NS_Q35_VISION: the model's mmproj file)".into()))?;
        v.tokens(bytes)
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Qwen35::gpu_info(self)
    }
    fn report(&self, _tokens: usize) -> Vec<String> {
        engine::Qwen35::profile_lines(self)
    }
}
