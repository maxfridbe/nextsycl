//! Dense Qwen3.5 / Qwen3.8 (`qwen35`: Qwen3.8-27B and its siblings): its geometry from the file's metadata and its
//! engine (`engine`: gated DeltaNet and gated full attention, the dense SwiGLU, the sessions and checkpoints), behind
//! the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry. Its own kernels are in
//! `kernels/llm/qwen35/`; the products from the stored blocks, the norms, the conv and the delta-rule scan are the
//! glm5next engine's generic kernels in the same library.

pub mod engine;
mod ffi;
mod probe;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

pub const ARCH: &str = "qwen35";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_Q35_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default 512)", at: At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Qwen3.5 / Qwen3.8 dense (gated DeltaNet + gated full attention, SwiGLU: Qwen3.8-27B ...)", load, info,
                 kernels: None, chat: nextsycl_tok::qwen_chat, options: OPTIONS }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = engine::Geometry::read(f)?;
    let full = (0..g.layers).filter(|l| g.full(*l)).count();
    Ok(format!("{ARCH}: {} layers of {} ({full} gated full attention: {} / {} heads of {}, rotary {} at {}; {} gated DeltaNet: {} / {} heads of {}), \
                a dense SwiGLU of {}, vocabulary {}, context {}", g.layers, g.n_embd, g.heads, g.heads_kv, g.head, g.n_rot, g.theta,
               g.layers - full, g.k_heads, g.v_heads, g.state, g.n_ff, g.vocab, g.ctx))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::Qwen35::load(f, gpus, log)?))
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
        false
    }
    fn prefill_chunk(&self) -> usize {
        engine::Qwen35::prefill_chunk(self)
    }
    fn max_verify(&self) -> usize {
        1
    }
    fn cache_fingerprint(&self) -> String {
        "qwen35 f16-kv nsq35-1".into()
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
        Ok(engine::Qwen35::feed_until(self, sess_mut(s)?, tokens, &|| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
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
    fn decoder(&self, logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(logits))
    }
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::after(next))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::Qwen35::step(self, sess_mut(s)?, d, smp, tap)
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
    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::Qwen35::gpu_info(self)
    }
    fn report(&self, _tokens: usize) -> Vec<String> {
        engine::Qwen35::profile_lines(self)
    }
}
