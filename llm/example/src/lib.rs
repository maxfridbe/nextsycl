//! A template LLM engine: every part of the `nextsycl_llm` contract in place and documented, nothing ported. Copy this
//! crate (and `kernels/llm/example/`) to port a model architecture - CONTRIBUTING.md walks through it.
//!
//! What a port fills in, in order:
//! 1. `model`: read the file's metadata and find every tensor by role (`nextsycl_gguf::Gguf`), checked, so a wrong
//!    file fails at load with a reason, not at the first pass.
//! 2. its kernels (`kernels/llm/<arch>/`, SYCL only) and their C ABI (`<arch>.h`), bound in `ffi.rs`.
//! 3. the engine below: load the weights onto the GPUs, then sessions, forward passes, decode, checkpoints.
//! 4. its registry entry (`kind()`), added to the program's `engines()` list.
//!
//! Until then `load` checks that the kernel library and the GPU work - it runs the example kernel - and fails with a
//! clear message. Nothing in the program calls the stubs below; they show what each method must do.

mod ffi;

use std::any::Any;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_gguf::Gguf;
use nextsycl_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, Error, GpuInfo, LoadOptions, Result, Sampler, Session,
                   SessionState, Tap};

/// The architecture this engine serves: the file's `general.architecture`
pub const ARCH: &str = "example";

/// This engine's registry entry (the program lists it in `engines()`)
pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Example (a template engine: nothing ported)", load, info, kernels: None, chat: nextsycl_tok::qwen_chat }
}

/// `nextsycl llm info`: the file's architecture and geometry, checked, without a GPU
fn info(f: &Gguf) -> std::result::Result<String, String> {
    Ok(format!("{ARCH}: a template engine - {} tensors in the file, none read", f.tensors.len()))
}

/// The path to the GPU, end to end: this kind's kernel library, this engine's own symbol, a kernel run on `gpu` and
/// its result read back - [1, 2, 3, 4] x 2 (`nextsycl llm selftest`)
pub fn selftest(gpu: &Arc<Gpu>) -> Result<Vec<f32>> {
    let k = ffi::api()?;
    let x = DevBuf::from_f32(gpu, &[1.0, 2.0, 3.0, 4.0])?;
    // SAFETY: four floats of device memory on this GPU.
    ffi::check(unsafe { (k.scale)(gpu.raw(), x.fp(), 4, 2.0) }, "the example kernel")?;
    gpu.sync()?;
    let y = x.to_f32()?;
    if y != [2.0, 4.0, 6.0, 8.0] {
        return Err(Error(format!("the example kernel computed {y:?}, not [2, 4, 6, 8]")));
    }
    Ok(y)
}

fn not_ported(what: &str) -> Error {
    Error(format!("the example engine is a template: {what} is not ported (CONTRIBUTING.md)"))
}

/// Load the model onto `gpus`. A port reads its weights here and plans the memory; the template proves the path to
/// the GPU - the kernel library, this engine's own symbol, a kernel run and its result - then stops.
fn load<'g>(_f: &'g Gguf, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    let y = selftest(gpu)?;
    log(format!("{ARCH}: the example kernel ran on {}: [1, 2, 3, 4] x 2 = {y:?}", gpu.name));
    Err(not_ported("loading a model"))
}

// ---- the engine's own state, behind the contract's opaque handles

/// A conversation's state on the GPUs (its caches) at `pos`
pub struct ExampleSession {
    pos: usize,
    max_ctx: usize,
}

impl SessionState for ExampleSession {
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

/// A decode in progress: the next token to feed, and the drafts (accepted, proposed) so far
pub struct ExampleDecoder {
    next: Option<u32>,
}

impl DecoderState for ExampleDecoder {
    fn drafts(&self) -> (u64, u64) {
        (0, 0)
    }
    fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A session's state as bytes (the prompt cache keeps these in memory and on disk)
pub struct ExampleCheckpoint {
    pos: usize,
}

impl CheckpointState for ExampleCheckpoint {
    fn pos(&self) -> usize {
        self.pos
    }
    fn bytes(&self) -> usize {
        0
    }
    fn write_to(&self, _w: &mut dyn std::io::Write) -> std::io::Result<()> {
        Err(std::io::Error::other("the example engine has no checkpoints"))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The engine: the loaded weights and plan on its GPUs
pub struct Example {
    gpus: Vec<Arc<Gpu>>,
    loaded: Instant,
}

impl Engine for Example {
    fn arch(&self) -> &'static str {
        ARCH
    }
    /// seconds the load took (reported by `status`)
    fn load_seconds(&self) -> f64 {
        self.loaded.elapsed().as_secs_f64()
    }
    /// whether a draft model (MTP) is loaded: `step` then proposes and verifies drafts
    fn has_draft(&self) -> bool {
        false
    }
    /// the prompt tokens one pass of `feed` reads
    fn prefill_chunk(&self) -> usize {
        1
    }
    /// the most tokens one verify pass takes (1 + drafts)
    fn max_verify(&self) -> usize {
        1
    }
    /// what makes a checkpoint of this engine reusable: the architecture, the file, the cache's number format
    fn cache_fingerprint(&self) -> String {
        format!("{ARCH} template")
    }
    /// a new conversation of up to `max_ctx` tokens
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(ExampleSession { pos: 0, max_ctx }))
    }
    /// back to position 0
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        s.get_mut::<ExampleSession>().ok_or_else(|| not_ported("a session of another engine"))?.pos = 0;
        Ok(())
    }
    /// `dst` becomes a copy of `src` (its caches up to `src.pos()`)
    fn copy_session(&self, _dst: &mut Session, _src: &Session) -> Result<()> {
        Err(not_ported("copy_session"))
    }
    /// read a prompt (all of it, in chunks of `prefill_chunk`): the logits after its last token
    fn feed(&self, _s: &mut Session, _tokens: &[u32], _tap: Tap) -> Result<Vec<f32>> {
        Err(not_ported("feed (the prompt path)"))
    }
    /// `feed`, stopping at a chunk's end when `stop()` says so (another request waits): tokens read, and the logits
    /// when it read them all
    fn feed_until(&self, _s: &mut Session, _tokens: &[u32], _stop: &(dyn Fn() -> bool + Sync), _tap: Tap) -> Result<(usize, Vec<f32>)> {
        Err(not_ported("feed_until"))
    }
    /// one pass over a few tokens (a decode or verify pass): the last row's logits
    fn forward(&self, _s: &mut Session, _tokens: &[u32], _tap: Tap) -> Result<Vec<f32>> {
        Err(not_ported("forward (the decode pass)"))
    }
    /// one pass, the last `n_out` rows' logits (a verify pass reads each row)
    fn forward_rows(&self, _s: &mut Session, _tokens: &[u32], _n_out: usize, _tap: Tap) -> Result<Vec<Vec<f32>>> {
        Err(not_ported("forward_rows"))
    }
    /// keep the first `keep` positions (drafts rejected): the state as if only they were read
    fn rollback(&self, _s: &mut Session, _keep: usize) -> Result<()> {
        Err(not_ported("rollback"))
    }
    /// one pass carrying one token of each session: each session's logits, each row exactly its own pass
    fn forward_batch(&self, _sessions: &mut [&mut Session], _tokens: &[u32], _tap: Tap) -> Result<Vec<Vec<f32>>> {
        Err(not_ported("forward_batch"))
    }
    /// a decode from the prompt's logits (`draft`: use the draft model when loaded)
    fn decoder(&self, _logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(ExampleDecoder { next: None })
    }
    /// a decode that starts by feeding `next`
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(ExampleDecoder { next: Some(next) })
    }
    /// one decode step: the tokens committed (one, or more when drafts are accepted), each drawn by `smp`
    fn step(&self, _s: &mut Session, _d: &mut Decoder, _smp: &mut dyn Sampler, _tap: Tap) -> Result<Vec<u32>> {
        Err(not_ported("step (decode)"))
    }
    /// the session's state as a checkpoint (the prompt cache)
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(ExampleCheckpoint { pos: s.pos() }))
    }
    /// a session back to a checkpoint's state
    fn restore(&self, _s: &mut Session, _ck: &Checkpoint) -> Result<()> {
        Err(not_ported("restore"))
    }
    /// a checkpoint written by `CheckpointState::write_to`
    fn read_checkpoint(&self, _r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Err(std::io::Error::other("the example engine has no checkpoints"))
    }
    /// each GPU as the engine uses it (`status`)
    fn gpu_info(&self) -> Vec<GpuInfo> {
        self.gpus.iter().map(|g| {
            let (total, free) = g.memory().unwrap_or((0, None));
            GpuInfo { index: g.index, name: g.name.clone(), pci: g.pci.clone(), total, free, layers: (0, 0), expert_slots: 0, host_slots: 0 }
        }).collect()
    }
}
