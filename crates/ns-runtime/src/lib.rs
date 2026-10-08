//! The contract between the server / command line and the engines. An engine is one model architecture's runtime,
//! tuned for it end to end - its own sessions, forward passes, decode loop, kernels and memory plan (`engines/<arch>`).
//! What they share is below: a trait the server drives (`Engine`), opaque handles for the state each engine keeps in
//! its own types (`Session`, `Decoder`, `Checkpoint`), how tokens are drawn (`Sampler`), and the registry entry an
//! engine provides (`EngineKind`), chosen by a file's `general.architecture`. Nothing here assumes a layer structure:
//! a new model brings its own engine rather than parameters for a generic one.

use std::any::Any;
use std::sync::Arc;

pub use ns_core::{DevBuf, Error, Gpu, Result};
use ns_gguf::Gguf;

/// Called with a tensor's name (llama.cpp's graph names: `attn_norm-3`, `l_out-44`, ...) and its float32 values on
/// the GPU, at each point a reference dump can be compared with.
pub type Tap<'a> = &'a mut dyn FnMut(&str, &DevBuf) -> Result<()>;

/// How an engine's decode step draws tokens. `sample` picks one from a row's logits (and records it - logprobs);
/// with `dist` (the distribution `sample` draws from: temperature and top-p applied; None when greedy) and `uniform`,
/// an engine may sample drafts from its draft model's distribution and accept them with min(1, p/q) (speculative
/// sampling: what is committed distributed exactly as plain sampling). `record` notes a token chosen that way. A plain
/// closure is a sampler without it.
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
    fn dist(&mut self, _logits: &[f32]) -> Option<Vec<(u32, f32)>> {
        None
    }
    fn uniform(&mut self) -> f32 {
        0.0
    }
    fn record(&mut self, _logits: &[f32], _token: u32) {}
}

impl<F: FnMut(&[f32]) -> u32> Sampler for F {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        self(logits)
    }
}

/// An engine's conversation state (its caches at a position), behind `Session`
pub trait SessionState: Any + Send {
    /// tokens read so far
    fn pos(&self) -> usize;
    /// the most it holds
    fn max_ctx(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A conversation's state, opaque to the server: only its engine reads it (`get` / `get_mut`)
pub struct Session(Box<dyn SessionState>);

impl Session {
    pub fn new<T: SessionState>(s: T) -> Session {
        Session(Box::new(s))
    }
    pub fn pos(&self) -> usize {
        self.0.pos()
    }
    pub fn max_ctx(&self) -> usize {
        self.0.max_ctx()
    }
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref()
    }
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.0.as_any_mut().downcast_mut()
    }
}

/// An engine's generation state for one conversation (drafts, pending token), behind `Decoder`
pub trait DecoderState: Any + Send {
    /// drafts verified, and accepted
    fn drafts(&self) -> (u64, u64);
    /// prompt-lookup drafts verified, and accepted (engines without them: none)
    fn lookup_drafts(&self) -> (u64, u64) {
        (0, 0)
    }
    /// the conversation so far (its prompt and what is committed), for drafts taken from it
    fn set_context(&mut self, _tokens: &[u32]) {}
    /// the committed token not fed yet, handed over (e.g. to a batch step) - its drafts dropped
    fn pending(&mut self) -> Option<u32>;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Generation from a read prompt, opaque to the server
pub struct Decoder(Box<dyn DecoderState>);

impl Decoder {
    pub fn new<T: DecoderState>(d: T) -> Decoder {
        Decoder(Box::new(d))
    }
    pub fn drafts(&self) -> (u64, u64) {
        self.0.drafts()
    }
    pub fn lookup_drafts(&self) -> (u64, u64) {
        self.0.lookup_drafts()
    }
    pub fn set_context(&mut self, tokens: &[u32]) {
        self.0.set_context(tokens)
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.0.pending()
    }
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.0.as_any_mut().downcast_mut()
    }
}

/// A conversation's whole state at a position, in host memory (the prompt cache's entries)
pub trait CheckpointState: Any + Send {
    fn pos(&self) -> usize;
    /// host bytes held
    fn bytes(&self) -> usize;
    /// as a file (the prompt cache's disk tier; its engine's `read_checkpoint` reads it back)
    fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()>;
    fn as_any(&self) -> &dyn Any;
}

pub struct Checkpoint(Box<dyn CheckpointState>);

impl Checkpoint {
    pub fn new<T: CheckpointState>(c: T) -> Checkpoint {
        Checkpoint(Box::new(c))
    }
    pub fn pos(&self) -> usize {
        self.0.pos()
    }
    pub fn bytes(&self) -> usize {
        self.0.bytes()
    }
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.0.write_to(w)
    }
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref()
    }
}

/// One GPU as an engine uses it
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub pci: Option<String>,
    pub total: u64,
    pub free: Option<u64>,
    /// the layers it runs [first, end)
    pub layers: (u64, u64),
    pub expert_slots: usize,
    pub host_slots: usize,
}

/// An MoE engine's expert store since load
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpertStats {
    pub vram_hits: u64,
    pub swapped_in: u64,
    pub from_host: u64,
    /// host-slot experts a prompt pass read in place
    pub read_direct: u64,
    pub prefetched: u64,
    pub prefetch_used: u64,
}

/// How an engine is loaded
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    /// VRAM for routed experts (None: the engine's own budget)
    pub expert_bytes: Option<usize>,
    /// pinned host memory for the experts that do not fit VRAM (None: what is free less a margin)
    pub mirror_bytes: Option<usize>,
    /// load the draft model (MTP) when the file has one
    pub draft: bool,
    /// the sessions' context to reserve: (tokens in all, sessions)
    pub kv: (usize, usize),
}

/// The runtime one model architecture brings. The server and the command line drive an engine only through this.
/// Every pass is exact in the sense the checks hold an engine to: a verify pass's rows equal one-token passes, a
/// batch's rows each conversation's own pass (`nextsycl spec-check`, `batch-check`).
pub trait Engine: Send + Sync {
    /// its `general.architecture`
    fn arch(&self) -> &'static str;
    /// seconds the load took
    fn load_seconds(&self) -> f64;
    /// weights' bytes it loaded onto the GPUs
    fn load_bytes(&self) -> u64 {
        0
    }
    /// whether it drafts (a draft model is loaded)
    fn has_draft(&self) -> bool;
    /// prompt tokens a forward pass takes at most (`feed` splits longer prompts)
    fn prefill_chunk(&self) -> usize;
    /// rows a verify pass takes at most (as this engine is configured)
    fn max_verify(&self) -> usize;
    /// what a checkpoint must match to be restored by another process of this engine (beside the model file):
    /// its cache form, its draft model, its format
    fn cache_fingerprint(&self) -> String;

    fn session(&self, max_ctx: usize) -> Result<Session>;
    fn reset_session(&self, s: &mut Session) -> Result<()>;
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()>;

    /// the prompt's `tokens` read: the last one's logits
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>>;
    /// as `feed`, `stop()` asked between chunks: (tokens read, the last one's logits)
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)>;
    /// one forward pass of at most a chunk: the last token's logits
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>>;
    /// one forward pass: the last `n_out` tokens' logits
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>>;
    /// keeps the first `keep` tokens of the last (verify) pass
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()>;
    /// a decode step of several conversations at once: `tokens[i]` the next of `sessions[i]`, each one's logits
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>>;

    /// generation from a prompt `feed` returned `logits` for (`draft`: with the draft model)
    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder;
    /// generation whose last committed token `next` is not fed yet
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder;
    /// the next committed token(s)
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>>;

    fn save(&self, s: &Session) -> Result<Checkpoint>;
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()>;
    /// a checkpoint `Checkpoint::write_to` wrote
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint>;

    /// LogProbChain: capture each one-token pass's attention (engines without: nothing)
    fn capture_attention(&self, _on: bool) {}
    fn take_attention(&self) -> Option<Vec<f32>> {
        None
    }

    fn gpu_info(&self) -> Vec<GpuInfo>;
    fn expert_stats(&self) -> ExpertStats {
        ExpertStats::default()
    }
    /// keep memory a long prompt needs while it is read in groups between decode steps
    fn hold_arena(&self, _on: bool) {}
    /// what decode asked for, for the next load (an MoE's expert profile); entries written
    fn save_profile(&self, _path: &std::path::Path) -> std::io::Result<usize> {
        Ok(0)
    }
    /// lines for the end of a run of `tokens` generated (its profile, memory peaks, waits)
    fn report(&self, _tokens: usize) -> Vec<String> {
        Vec::new()
    }
}

/// How an engine loads: the file, its GPUs, the options, a log
pub type LoadFn = for<'g> fn(&'g Gguf, &[Arc<Gpu>], &LoadOptions, &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>>;
/// An engine's kernel test: the file, a GPU index
pub type KernelsFn = fn(&std::path::Path, usize) -> std::result::Result<(), String>;

/// An engine's registry entry
pub struct EngineKind {
    /// the `general.architecture` values it serves
    pub archs: &'static [&'static str],
    /// what it is
    pub name: &'static str,
    /// load it on these GPUs
    pub load: LoadFn,
    /// a file's architecture and geometry, checked, without a GPU (`nextsycl info`)
    pub info: fn(&Gguf) -> std::result::Result<String, String>,
    /// its kernels against an exact path on the file's own matrices (`nextsycl kernels`), when it has such a test
    pub kernels: Option<KernelsFn>,
}

/// The entry of `kinds` serving `g`'s architecture
pub fn kind_for<'k>(kinds: &'k [EngineKind], g: &Gguf) -> std::result::Result<&'k EngineKind, String> {
    let arch = g.architecture();
    kinds.iter().find(|k| k.archs.contains(&arch)).ok_or_else(|| {
        let known: Vec<&str> = kinds.iter().flat_map(|k| k.archs.iter().copied()).collect();
        format!("architecture {arch:?} has no engine here (known: {})", known.join(", "))
    })
}
