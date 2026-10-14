//! Muse Glimmer 30B (`muse-glimmer`): its geometry from the file's metadata and its engine (`engine`: gated attention
//! over sliding windows and full layers, the dense SwiGLU, the post norms, the sessions and checkpoints), behind the
//! runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry. Its own kernels are in
//! `kernels/llm/museglimmer/`; the products from the stored blocks and the norms are the glm5next engine's generic
//! kernels in the same library.

pub mod engine;
mod ffi;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::EngineOption;
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

pub const ARCH: &str = "muse-glimmer";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_MG_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default 2048 when it fits beside the weights and a session, else 1024 or 512)", at: nextsycl_llm::At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Muse Glimmer (dense gated attention: sliding windows and full layers, SwiGLU, post norms: Muse-Glimmer-30B)", load, info,
                 kernels: None, chat: nextsycl_tok::muse_chat, options: OPTIONS, prepare: None }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = engine::Geometry::read(f)?;
    let windowed = g.swa.iter().filter(|s| **s).count();
    Ok(format!("{ARCH}: {} layers of {} ({windowed} over a window of {} with rotary positions at {}, {} full without), {} / {} heads of {}, \
                a dense SwiGLU of {}, logits x {} capped at {}, vocabulary {}, context {}", g.layers, g.n_embd, g.window, g.theta, g.layers - windowed,
               g.heads, g.heads_kv, g.head, g.n_ff, g.logit_scale, g.softcap, g.vocab, g.ctx))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::MuseGlimmer::load(f, gpus, o.kv, log)?))
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

impl Engine for engine::MuseGlimmer<'_> {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        engine::MuseGlimmer::load_seconds(self)
    }
    fn load_bytes(&self) -> u64 {
        engine::MuseGlimmer::load_bytes(self)
    }
    fn has_draft(&self) -> bool {
        false
    }
    fn prefill_chunk(&self) -> usize {
        engine::MuseGlimmer::prefill_chunk(self)
    }
    fn max_verify(&self) -> usize {
        8
    }
    fn cache_fingerprint(&self) -> String {
        format!("museglimmer f16-kv nsmg-1 chunk={}", self.prefill_chunk())
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::MuseGlimmer::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::MuseGlimmer::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let src = sess(src)?;
        engine::MuseGlimmer::copy_session(self, sess_mut(dst)?, src)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::MuseGlimmer::feed_until(self, sess_mut(s)?, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::MuseGlimmer::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::MuseGlimmer::forward_rows(self, sess_mut(s)?, tokens, 1, tap)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::MuseGlimmer::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::MuseGlimmer::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = Vec::with_capacity(sessions.len());
        for s in sessions.iter_mut() {
            own.push(s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))?);
        }
        engine::MuseGlimmer::forward_batch(self, &mut own, tokens, tap)
    }
    fn decoder(&self, logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(logits))
    }
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::after(next))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::MuseGlimmer::step(self, sess_mut(s)?, d, smp, tap)
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::MuseGlimmer::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::MuseGlimmer::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(r)?))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::MuseGlimmer::gpu_info(self)
    }
    fn report(&self, tokens: usize) -> Vec<String> {
        engine::MuseGlimmer::profile_lines(self, tokens)
    }
}
