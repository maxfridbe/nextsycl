//! Nemotron-H MoE (`nemotron_h_moe`: NVIDIA Nemotron 3.5 Super 120B-A12B): its geometry from the file's metadata and
//! its engine (`engine`: Mamba-2 mixers, latent MoE layers, attention layers without positions; the sessions and
//! checkpoints), behind the runtime's `Engine` contract (`nextsycl-llm`). `kind()` is its registry entry. Its own
//! kernels are in `kernels/llm/nemotronh/`; the products from the stored blocks, the RMS norm and the MoE combine are
//! the glm5next engine's generic kernels in the same library.

pub mod engine;
mod ffi;

use std::any::Any;
use std::sync::Arc;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{At, EngineOption};
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

pub const ARCH: &str = "nemotron_h_moe";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_NH_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default the largest of 2048, 1024, 512 that fits)", at: At::Load },
    EngineOption { name: "drafts", env: "NS_NH_DRAFTS", value: "N", help: "tokens drafted a step with the MTP block (default 2, at most 7; 0: none)", at: At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Nemotron-H MoE (Mamba-2 + latent MoE + attention: Nemotron 3.5 Super 120B-A12B)", load, info, kernels: None,
                 chat: nextsycl_tok::nemotron_chat, options: OPTIONS, prepare: None }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    use engine::Kind;
    let g = engine::Geometry::read(f)?;
    Ok(format!("{ARCH}: {} layers of {} ({} Mamba-2: {} heads of {} in {} groups, state {}, conv {}; {} latent MoE: {} of {} experts of {} in a latent of {}, \
                a shared expert of {}; {} attention: {} / {} heads of {}, no positions), vocabulary {}, context {}{}", g.layers, g.n_embd,
               g.count(Kind::Mamba), g.ssm_heads, g.inner / g.ssm_heads, g.groups, g.state, g.conv, g.count(Kind::Moe), g.used, g.experts, g.n_ff_exp,
               g.latent, g.n_ff_shared, g.count(Kind::Attn), g.heads, g.heads_kv, g.head, g.vocab, g.ctx,
               if g.nextn > 0 { format!(", {} MTP block(s)", g.nextn) } else { String::new() }))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextsycl_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(engine::NemotronH::load(f, gpus, o.draft, o.kv, log)?))
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

impl Engine for engine::NemotronH<'_> {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        engine::NemotronH::load_seconds(self)
    }
    fn load_bytes(&self) -> u64 {
        engine::NemotronH::load_bytes(self)
    }
    fn has_draft(&self) -> bool {
        engine::NemotronH::has_draft(self)
    }
    fn prefill_chunk(&self) -> usize {
        engine::NemotronH::prefill_chunk(self)
    }
    fn max_verify(&self) -> usize {
        engine::NemotronH::max_verify(self)
    }
    fn cache_fingerprint(&self) -> String {
        // checkpoints hold the draft block's cache too when it is loaded
        if self.has_draft() { "nemotronh f16-kv f32-ssm nsnmh-1 mtp".into() } else { "nemotronh f16-kv f32-ssm nsnmh-1".into() }
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(engine::NemotronH::session(self, max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        engine::NemotronH::reset_session(self, sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let src = sess(src)?;
        engine::NemotronH::copy_session(self, sess_mut(dst)?, src)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::NemotronH::feed_until(self, sess_mut(s)?, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        engine::NemotronH::feed_until(self, sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(engine::NemotronH::forward_rows(self, sess_mut(s)?, tokens, 1, tap)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        engine::NemotronH::forward_rows(self, sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        engine::NemotronH::rollback(self, sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = Vec::with_capacity(sessions.len());
        for s in sessions.iter_mut() {
            own.push(s.get_mut().ok_or_else(|| nextsycl_core::Error("a session of another engine".into()))?);
        }
        engine::NemotronH::forward_batch(self, &mut own, tokens, tap)
    }
    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::new(logits, draft))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(engine::Decoder::after(next, draft))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<engine::Decoder>().ok_or_else(|| nextsycl_core::Error("a decoder of another engine".into()))?;
        engine::NemotronH::step(self, sess_mut(s)?, d, smp, tap)
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(engine::NemotronH::save(self, sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck.get::<engine::Checkpoint>().ok_or_else(|| nextsycl_core::Error("a checkpoint of another engine".into()))?;
        engine::NemotronH::restore(self, sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(engine::Checkpoint::read_from(r)?))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        engine::NemotronH::gpu_info(self)
    }
    fn report(&self, _tokens: usize) -> Vec<String> {
        engine::NemotronH::profile_lines(self)
    }
}
