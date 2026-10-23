//! DeepSeek-V4.1-Flash on one or more GPUs: `model.rs`'s outline in order, float32 activations. The layers are split
//! over the GPUs at a kv source (a source's compressed keys stay with the layers that read them; B70 + B65: 0-19 |
//! 20-39), each `Part` holding its layers' weights in their stored form (q4_K, fp8_32x32, float32 for the small and
//! BF16 ones), its expert stores (VRAM slots, pinned host slots, the rest read from the file: deepseek4's exclusive
//! two-level cache), and the token stream [T, 4, hidden] crossing to the next GPU through host memory - with the mix
//! the next half collapses it by ([T, 4] after the streams).
//!
//! The routed experts are VQ: a slot holds an expert's three DQV3 payloads as the file has them (gate | up | down); a
//! decode pass multiplies them straight from the indices (ds41.h: vq_mmv, every expert of a layer in one launch), a
//! prompt pass expands each to fp16 and runs oneMKL's half GEMM over its tokens.
//!
//! The draft (DSpark, three towers) runs on the last part: after each pass the committed rows' stream means at the
//! target layers go through main_proj into each tower's window ring; a draft block is [token, 4 noise] over the towers,
//! the head and the Markov bias, verified by the main model in one pass.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Error, Gpu, HostBuf, Result};

use crate::engram::{self, Engram};
use crate::model::{Model, Role};
use crate::ops::Ops;
use nextsycl_gguf::{GType, Tensor};

use nextsycl_llm::{GpuInfo, Sampler, Tap};

/// float32 values expanded per matrix chunk (128 MiB)
const SCRATCH: usize = 32 << 20;
/// the decode kernels take up to this many rows (tokens) at once
pub const MMVQ_COLS: usize = 8;
/// rows of a verify pass (the token and its drafts)
pub const MAX_VERIFY: usize = 8;
/// expert slots and the mirror are allocated in chunks of this size
const CHUNK: usize = 2 << 30;
/// the pinned rings' sizes
const UP_BYTES: usize = 8 << 20;
const BACK_BYTES: usize = 4 << 20;
/// the arena of decode and short prompt chunks; past it, a pass takes the big one back from the expert store
const SMALL_ARENA: usize = 512 << 20;
/// the indexer's scores per pass of rows (floats)
const IDX_S_FLOATS: usize = 32 << 20;
/// the draft towers' window (positions of main_x a tower's ring holds)
const MTP_WIN: usize = 128;
/// a pass's last rows whose stream means the towers keep
const MTP_KEEP: usize = MTP_WIN + MAX_VERIFY;

/// Time the host spent waiting for the routers' logits / choosing experts (all layers)
pub static ROUTER_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ROUTE_HOST_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Decode: the host's time from a layer's router logits to its expert products queued (the GPU has nothing queued
/// meanwhile but the shared expert), and of it the file reads
pub static TURN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static FILE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Decode's expert misses, by where the host waited (NS_PROFILE reports them): for a read ahead, in `stage_reads`
/// (the file reads and a staging slot's earlier copy), in `swap_in` (the copies' submission), for the copies of the
/// products that need an expert still arriving, and in `stage_slot` alone
pub static ENS_AHEAD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// decode's file reads ahead started (one layer ahead, further ahead), the ones a layer then took, and the misses a
/// layer read from the file itself (staged)
pub static AHEAD_NEAR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static AHEAD_FAR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static AHEAD_TAKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static STAGED_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENS_STAGE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENS_SWAP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENS_COPY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENS_SLOT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Decode's wall time (no GPU syncs added): in `forward_rows` the streams (embedding), the layers (all parts), the head,
/// and in `step` the sampler - NS_PROFILE reports them
/// How often the guess's k-th best expert for a layer (its router on the layer before's input) was among the experts the
/// layer then picked, over the decode passes: hits by rank, and the passes counted (NS_PROFILE reports the rates, for
/// GUESS_HIT)
pub static GUESS_HITS: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];
pub static GUESS_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Decode's routed experts held by neither VRAM nor host memory ("cold"): by their share of the layer's routing weight
/// (<0.05, <0.1, <0.15, <0.2, <0.3, more), the layers with one, and those whose cold ones are all under 0.1
pub static COLD_BINS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
pub static COLD_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static COLD_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static COLD_LAYERS_LOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DEC_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GUESS_LAST: Mutex<Vec<(u64, Vec<usize>)>> = Mutex::new(Vec::new());
pub static WALL_STREAMS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WALL_LAYERS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WALL_HEAD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WALL_SAMPLE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Prompt tokens per forward pass (NS_PREFILL_CHUNK, default 2048, 64..8192)
pub fn prefill_chunk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_PREFILL_CHUNK", 2048).clamp(64, 32768))
}

/// Bytes of each GPU's big arena (NS_ARENA_MIB; default 1 GiB or 0.5 MiB a prompt token)
pub fn arena_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("NS_ARENA_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(m) => m << 20,
        None => (1usize << 30).max(prefill_chunk() * (512 << 10)),
    })
}

/// A prompt pass's experts with at most this many tokens multiplied from their indices (the decode kernels), the rest
/// expanded to fp16 for a GEMM (NS_DS41_PROMPT_MMV, default 0: none - the XMX product from the indices, vq_gemm, is faster at every count)
fn prompt_mmv_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_MMV", 0))
}
/// The decode kernels' variant for a prompt's few-token experts (NS_DS41_PROMPT_MMV_V, default 3: a sub-group a row, the
/// fp16 codebook - its columns share each index's weights)
fn prompt_mmv_variant() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_MMV_V", 3) as i32)
}
/// A prompt pass's VRAM staging slots for the experts it reads from host memory or the file (NS_DS41_PROMPT_RING,
/// default 24; 0: read in place from host memory, the file's swapped in)
fn prompt_ring() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_RING", 24))
}
/// A prompt pass's experts with more tokens than NS_DS41_PROMPT_MMV and at most this many multiplied on XMX from their
/// indices (ns_ds41_vq_gemm), more expanded to fp16 for a GEMM (NS_DS41_VQ_GEMM_MAX, default 100000: all; 0: none)
fn vq_gemm_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_VQ_GEMM_MAX", 100_000))
}
/// A prompt pass's pinned slots for the experts only the file holds (NS_DS41_PROMPT_FILE_SLOTS, default 320, two halves of 160 for two layers; 0: read a
/// wave at a time on the host thread)
fn prompt_file_slots() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_FILE_SLOTS", 320))
}
/// NS_DS41_PIPE_LOG=1: each prompt chunk's time on each GPU of the pipeline; 2: and each layer's, its attention and
/// its experts apart (a GPU sync between them: no overlap across layers)
fn pipe_log() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_PIPE_LOG").is_ok_and(|v| v == "1" || v == "2"))
}
fn pipe_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_PIPE_LOG").is_ok_and(|v| v == "2"))
}
/// Decode's empty VRAM slots kept a store (NS_DS41_SPARE, default 8; 0: a swap moves its victim down first)
fn spare_slots() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_SPARE", 8))
}
/// NS_DS41_SKIP_HOST=1 (a measurement - wrong output): decode's cold-expert skip takes the pinned host tier's experts too, so
/// no expert is swapped in for the layer that asks for it - the ceiling of making its misses free
fn skip_host() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_SKIP_HOST", 0) != 0)
}
/// Decode's expert stores also take the VRAM the passes' reserve leaves free while decoding (NS_DS41_LEND_RESERVE, default
/// 1; 0: only the big arena is lent): ~6.5 GiB a card at a 16384-token chunk, given back before a pass that needs it
fn lend_extra() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_LEND_RESERVE", 1) != 0)
}
/// Decode's file reads further ahead: the routers of this many layers after the next run on a layer's input too, and
/// of each one's best NS_DS41_FILE_AHEAD_FAR_RANK (default 8) at most NS_DS41_FILE_AHEAD_FAR_TOP (default 3) that only the
/// file holds read ahead - the best are nearly always resident (NS_DS41_FILE_AHEAD_FAR,
/// default 2; 0: one layer ahead only). A drive read takes ~4 ms, a layer ~1: one layer ahead, decode waited ~10 ms a
/// token on reads already started
fn far_layers() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_FILE_AHEAD_FAR", 0))
}
fn far_rank() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_FILE_AHEAD_FAR_RANK", 8))
}
fn far_top() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_FILE_AHEAD_FAR_TOP", 3))
}
/// Decode's expert products against split-Q8 activations with dp4a (mmvq.cpp; NS_DS41_STD_Q8, default 1; 0: std.cpp's
/// float32 kernel)
fn std_q8() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_STD_Q8", 1) != 0)
}
/// A full host tier gives its least recently used expert back to the file for a VRAM expert moved down (NS_DS41_HOST_LRU=1;
/// default 0: the tier keeps the experts it was loaded with, a VRAM expert with no free host slot is dropped - the LRU
/// tier measured no gain: tg 12.2 against 12.9 tok/s over 1,024 tokens after a 2.5K-token prompt)
fn host_lru() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_HOST_LRU", 0) != 0)
}
/// Decode reads ahead at most this many of the next layer's likely experts that only the file holds (NS_DS41_FILE_AHEAD,
/// default 8; 0: none)
fn file_ahead() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_FILE_AHEAD", 8).min(STAGE / 2))
}
/// The grouped o_a of one decode row in one launch of the std kernel (NS_DS41_STD_OA=1; default 0: a launch a group of the
/// shared kernel - alone one launch is faster (98 against 138 us), end to end the eight overlap and it measured no gain)
fn std_oa() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_STD_OA").is_ok_and(|v| v == "1"))
}
/// Attention's and the shared expert's Q8_0 / Q6_K weights requantized to q4_K when loaded (NS_DS41_DENSE_Q4K=1; default 0):
/// half the bytes a decode token reads and the fused q4_K kernels apply (an approximation - measure the perplexity).
/// NS_DS41_DENSE_Q4K_ROLES: a comma list of qa, qb, kv, oa, ob, shgate, shup, shdown (default all; a Q6_K one of the eight
/// is requantized whatever the list: its shared product is slow). Measured alone, cold from VRAM, the shared Q8_0 product
/// was as fast as q4_K for q_a, kv, q_b and o_b (B70 us q4_K / Q8_0: 18 / 18, 8.4 / 8.0, 106 / 81, 89 / 84); in decode
/// (traced, GPU time a token) keeping those four in Q8_0 was slower, B70 6.5 -> 7.8 ms of dense products - and all eight
/// in q4_K beat only o_a / o_b (35.0 against 37.4 ms) and none (38.7)
fn dense_q4k() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_DENSE_Q4K").is_ok_and(|v| v == "1"))
}
fn dense_q4k_role(r: Role) -> bool {
    static V: std::sync::OnceLock<Vec<Role>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let v = std::env::var("NS_DS41_DENSE_Q4K_ROLES").unwrap_or_else(|_| "qa,qb,kv,oa,ob,shgate,shup,shdown".into());
        v.split(',').filter_map(|x| match x.trim() {
            "qa" => Some(Role::QA),
            "qb" => Some(Role::QB),
            "kv" => Some(Role::Kv),
            "oa" => Some(Role::OutA),
            "ob" => Some(Role::OutB),
            "shgate" => Some(Role::ShGate),
            "shup" => Some(Role::ShUp),
            "shdown" => Some(Role::ShDown),
            _ => None,
        }).collect()
    }).contains(&r)
}
/// ... for passes of at least this many tokens (NS_DS41_PROMPT_PREFETCH_MIN, default 2048): below, a layer's compute is too
/// short to hide anything and the early reads only contend with the layer's own (XQ, O_DIRECT reads: 2,561 tokens 16.6 ->
/// 13.3 s with it; 580 tokens 8.7 -> 11-15 s, fast 5.3 -> 12-29 s)
fn prompt_prefetch_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_PREFETCH_MIN", 2048))
}
/// A prompt pass starts reading the next layer's file experts while it computes this one (NS_DS41_PROMPT_PREFETCH, default 1)
fn prompt_prefetch() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_PROMPT_PREFETCH").map_or(true, |v| v != "0"))
}
/// Decode leaves out a routed expert held by neither VRAM nor host memory (nor already being read ahead) whose share of the
/// layer's routing weight is under this (NS_DS41_COLD_SKIP, default 0: none): its read is a ~4 ms stall for a layer, its
/// output a few percent of the layer's routed sum. An approximation - measure the perplexity (generate --score)
/// the Engram's rows through the shared dequant kernels as before IQ4_NL (NS_DS41_ENGRAM_SHARED=1; a measurement switch:
/// Q5_K / Q6_K tables only)
fn engram_shared() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_ENGRAM_SHARED").is_ok_and(|v| v == "1"))
}
/// The big arena's slots lent to a store at load filled from the profile then (NS_DS41_FILL_LENT=1; off by default:
/// exact decode over 4,000 tokens read as many experts from the file, 52.4K against 51.7K, tg 7.58 against 7.23)
fn fill_lent() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_FILL_LENT", 0) != 0)
}
fn cold_skip() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_COLD_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0))
}
/// ... the remaining experts' weights then scaled up by the left-out share (NS_DS41_COLD_RENORM, default 0; 1: on)
fn cold_renorm() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_COLD_RENORM").is_ok_and(|v| v == "1"))
}
/// A prompt pass (more than a decode step's rows) leaves out a file-only expert that gets at most NS_DS41_COLD_SKIP_PROMPT_MAX
/// tokens (default 4), all with a share of the layer's routing weight under NS_DS41_COLD_SKIP_PROMPT (default 0: none): a few
/// tokens' small contributions against a ~10 MB read. An approximation - measure the perplexity after such a prompt
fn cold_skip_prompt() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_COLD_SKIP_PROMPT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0))
}
fn cold_skip_prompt_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_COLD_SKIP_PROMPT_MAX", 4))
}
/// ... also one whose read ahead is on its way (NS_DS41_COLD_SKIP_AHEAD, default 1; 0: wait for those)
fn cold_skip_ahead() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_COLD_SKIP_AHEAD").map_or(true, |v| v != "0"))
}
/// The chance of use from which decode reads a file expert ahead (NS_DS41_FILE_AHEAD_P, default 0.4 against the measured
/// GUESS_HIT: guesses that miss take the drive from the reads that matter). At 0.2, 512 tokens of the fast setting started
/// 13,200 reads ahead and took 672 (~280 MB a token: the drive's whole rate); 0.4: 6,281 started, the same 670 taken,
/// tg 18.4-18.9 -> 21.4 tok/s; exact 10.6-11.7 -> 11.7-11.8 (0.6: no better)
fn file_ahead_p() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_FILE_AHEAD_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.4))
}
/// The index sources after the candidate source score only their candidate blocks (NS_DS41_IDX_CAND, default 1; 0:
/// every key, the others masked)
fn idx_cand() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_IDX_CAND").map_or(true, |v| v != "0"))
}
/// Decode routes one token on the GPU and queues its VRAM experts' products before the host reads the routing
/// (NS_DS41_GPU_ROUTE, default 1; off with the draft towers: verify passes route on the host)
fn gpu_route() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_GPU_ROUTE").map_or(true, |v| v != "0"))
}
/// Token slots of one batch of a prompt's few-token experts (their gate | up and h in the arena)
const PROMPT_MMV_SLOTS: usize = 1024;

/// Dense products of this many rows or more in fp16 on the XMX units (NS_DS41_DENSE_F16_MIN, default 64)
fn dense_f16_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_DENSE_F16_MIN", 64))
}

/// Experts swapped in ahead of their layer at most (NS_PREFETCH, default 8; 0 none), from a chance of use of
/// NS_PREFETCH_P (default 0.6)
fn prefetch_limit() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_PREFETCH", 8))
}
fn prefetch_p() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PREFETCH_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6))
}
/// How often the next layer's router run on this layer's input has its k-th best among the next layer's picks (the
/// glm5next / deepseek4 measurement was the first guess here; these are V4.1's own, IQ2_XXS, 11,362 decode layers, NS_PROFILE=1 reports them)
const GUESS_HIT: [f32; 16] = [0.97, 0.91, 0.84, 0.71, 0.55, 0.40, 0.26, 0.19, 0.13, 0.10, 0.08, 0.07, 0.05, 0.04, 0.04, 0.03];

/// Decode's q4_K products by this engine's kernel on float32 activations, o_a's groups in one launch (NS_DS41_Q4K=0:
/// the shared Q8_1 quantization and product, a launch a group)
fn q4k_own() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_Q4K").map_or(true, |v| v != "0"))
}

/// Decode's mHC mixes through the fused kernels (NS_HC_FUSED=0: the mixes, then the Sinkhorn)
fn hc_fused() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_HC_FUSED").map_or(true, |v| v != "0"))
}

/// Whether the big arena lends its memory to the expert store while no prompt chunk needs it (NS_LEND=0: no)
fn lend_arena() -> bool {
    std::env::var("NS_LEND").map_or(true, |v| v != "0")
}

/// Whether a prompt of several chunks on two GPUs runs as a pipeline (NS_PIPELINE=0: in turn)
fn pipeline() -> bool {
    std::env::var("NS_PIPELINE").map_or(true, |v| v != "0")
}

/// Whether the DSpark towers load and draft (NS_DS41_MTP=1; default off)
pub fn mtp_on() -> bool {
    std::env::var("NS_DS41_MTP").is_ok_and(|v| v == "1")
}

/// Drafts a verify pass checks (NS_DS41_DRAFTS, 1..7, default 2: the third is rarely kept)
pub fn mtp_drafts() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_DRAFTS", 2).clamp(1, MAX_VERIFY - 1))
}

/// VRAM each GPU keeps free beyond the plan (NS_DS41_FREE_GIB, default 0.75: the xe driver has no out-of-memory path; the plan reserves a pass generously - measured 2.4 GiB free at the peak)
fn keep_free() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| (std::env::var("NS_DS41_FREE_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.75) * (1u64 << 30) as f64) as usize)
}

/// The pinned host memory for experts in all (NS_DS41_PIN_GIB, default 40, at most what is available less 12 GiB)
fn pin_cap() -> usize {
    (std::env::var("NS_DS41_PIN_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(40.0) * (1u64 << 30) as f64) as usize
}

/// The expert profile (NS_EXPERT_PROFILE: "layer expert weight" lines)
fn expert_profile() -> Option<&'static HashMap<(u64, u64), f64>> {
    static V: std::sync::OnceLock<Option<HashMap<(u64, u64), f64>>> = std::sync::OnceLock::new();
    V.get_or_init(|| std::env::var("NS_EXPERT_PROFILE").ok().and_then(|p| read_profile(std::path::Path::new(&p)))).as_ref()
}

fn read_profile(path: &std::path::Path) -> Option<HashMap<(u64, u64), f64>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut m = HashMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut it = line.split_whitespace();
        if let (Some(l), Some(x), Some(w)) = (it.next(), it.next(), it.next()) {
            if let (Ok(l), Ok(x), Ok(w)) = (l.parse(), x.parse(), w.parse()) {
                m.insert((l, x), w);
            }
        }
    }
    Some(m)
}

/// `tokens` in chunks of `c`, the last two split evenly when the last would be under half a chunk
fn chunked(tokens: &[u32], c: usize) -> Vec<&[u32]> {
    let mut v: Vec<&[u32]> = tokens.chunks(c).collect();
    if v.len() >= 2 && v[v.len() - 1].len() < c / 2 {
        let n = v.len();
        let both = &tokens[(n - 2) * c..];
        let half = both.len().div_ceil(2);
        v.truncate(n - 2);
        v.push(&both[..half]);
        v.push(&both[half..]);
    }
    v
}

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn mem_available() -> usize {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map_or(0, |kb| kb * 1024)
}

/// The highest logit's token
pub fn argmax(l: &[f32]) -> u32 {
    l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

/// sqrt(softplus(z)): the router's score
fn sqrt_softplus(z: f32) -> f32 {
    (if z > 20.0 { z } else { z.exp().ln_1p() }).sqrt()
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A matrix on the GPU: rows x cols in its stored form (q4_K, fp8_32x32) or as float32 (the 16- and 32-bit ones)
struct Mat {
    buf: DevBuf,
    ty: GType,
    rows: usize,
    cols: usize,
}

impl Mat {
    fn f32(&self) -> bool {
        self.ty == GType::F32
    }
    /// rows [r0, r0 + n) as a matrix of their own (q4_K / float32 only: an fp8_32x32 tile spans 32 rows)
    fn rows_view(&self, r0: usize, n: usize) -> Result<Mat> {
        let rb = if self.f32() { self.cols * 4 } else { self.ty.bytes(self.cols as u64).ok_or("rows of this type")? as usize };
        Ok(Mat { buf: self.buf.view(r0 * rb, n * rb)?, ty: self.ty, rows: n, cols: self.cols })
    }
}

/// A layer's VQ experts: where each expert's three payloads lie in the blob (its span: gate | up | down back to back,
/// the same layout for every expert), the matrices' places in a span, the index width, the codebook on the GPU
struct LayerVq {
    blob: Tensor,
    /// each expert's span's first byte in the blob
    starts: Vec<u64>,
    span: usize,
    gate: usize,
    up: usize,
    down: usize,
    nbit: usize,
    /// the layer's E4M3 codebook bytes (on the GPU of the layer's part)
    cb_bytes: Vec<u8>,
    /// the llama.cpp form: the three matrices' tensors, types and bytes an expert (a span: gate | up | down back to back)
    std: Option<StdExp>,
}

/// A layer's experts in ggml's row types: three merged tensors [n_expert, rows, cols] (gate and up the same type)
struct StdExp {
    t: [Tensor; 3],
    ty: [GType; 3],
    /// each matrix's bytes in an expert
    mat: [usize; 3],
    /// the XQ file's: each expert its own types
    xq: Option<XqExp>,
}

/// The XQ file's layer: its experts in up to three types (index 0 Q3_K, 1 Q2_0, 2 PTQ1_0), a tensor a type and role
/// holding that type's experts in id order. An expert's slot holds its own three matrices back to back (`LayerVq::lay`),
/// in a store of its own size; `LayerVq`'s gate / up / down / span are the layer's widest.
struct XqExp {
    /// [type][gate, up, down]
    t: [[Option<Tensor>; 3]; 3],
    /// each expert's gate | up type, its index among them, its down type, its index among those
    ex: Vec<(u8, u32, u8, u32)>,
    /// a gate | up matrix's bytes and a down's, by type
    gu_mat: [usize; 3],
    down_mat: [usize; 3],
}

/// The XQ types' index (in the pointer's tag: the silo's mixed products, T_MIXED_GU / T_MIXED_DOWN)
const XQ_TYPES: [GType; 3] = [GType::Q3K, GType::Q2_0, GType::Ptq1_0];
/// the silo's type codes for a mixed product: gate | up, down
const T_MIXED_GU: u32 = 1000;
const T_MIXED_DOWN: u32 = 1001;

impl LayerVq {
    /// What an expert's entries carry in their first meta word (beside the column count, at most 8) for the mixed
    /// products: its gate | up type index in bits 8-11, its down's in 12-15 (0 for every other form)
    fn tag(&self, x: u64) -> i32 {
        match self.std.as_ref().and_then(|s| s.xq.as_ref()) {
            Some(q) => {
                let (gt, _, dt, _) = q.ex[x as usize];
                ((gt as i32) | ((dt as i32) << 4)) << 8
            }
            None => 0,
        }
    }
    /// Expert x's place in its slot: gate, up and down's first bytes and its span - the XQ file's its own types' (back to
    /// back), every other form's the layer's
    fn lay(&self, x: u64) -> (usize, usize, usize, usize) {
        match self.std.as_ref().and_then(|s| s.xq.as_ref()) {
            Some(q) => {
                let (gt, _, dt, _) = q.ex[x as usize];
                let g = q.gu_mat[gt as usize];
                (0, g, 2 * g, 2 * g + q.down_mat[dt as usize])
            }
            None => (self.gate, self.up, self.down, self.span),
        }
    }
    /// the slot size expert x is kept in (its store's)
    fn slot_of(&self, x: u64) -> usize {
        self.lay(x).3.next_multiple_of(256)
    }
    /// Expert x's matrix `which` (0 gate, 1 up, 2 down): its tensor, its first byte in it, its bytes, its type
    fn mat_of(&self, x: u64, which: usize) -> Option<(&Tensor, u64, usize, GType)> {
        let s = self.std.as_ref()?;
        match &s.xq {
            Some(q) => {
                let (gt, gi, dt, di) = q.ex[x as usize];
                let (k, i, bytes) = if which == 2 { (dt as usize, di as u64, q.down_mat[dt as usize]) } else { (gt as usize, gi as u64, q.gu_mat[gt as usize]) };
                Some((q.t[k][which].as_ref()?, i * bytes as u64, bytes, XQ_TYPES[k]))
            }
            None => Some((&s.t[which], x * s.mat[which] as u64, s.mat[which], s.ty[which])),
        }
    }
    /// where expert x's matrix `which` starts in its slot
    fn mat_at(&self, x: u64, which: usize) -> usize {
        let (g, u, d, _) = self.lay(x);
        [g, u, d][which]
    }
}

/// The XQ file's layer: the type maps, each type's tensors, every expert's place
fn layer_xq(m: &Model, l: u64) -> Result<LayerVq> {
    let g = &m.g;
    let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
    let ne = g.experts(l).0 as usize;
    let map = |role: &str| -> Result<Vec<u8>> {
        let name = format!("blk.{l}.ffn_exps_{role}_type");
        let t = m.file.tensor(&name).ok_or_else(|| Error(format!("{name} is missing")))?;
        let raw = m.file.read(t).map_err(e)?;
        let v: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if v.len() != ne {
            return Err(Error(format!("{name}: {} entries, {ne} experts", v.len())));
        }
        v.into_iter().map(|c| XQ_TYPES.iter().position(|t| t.code() == c as u32).map(|k| k as u8)
            .ok_or_else(|| Error(format!("{name}: type {c}, the engine takes Q3_K, Q2_0 and PTQ1_0")))).collect()
    };
    let (gu, dn) = (map("gu")?, map("down")?);
    let tname = |k: usize| ["q3_k", "q2_0", "ptq1_0"][k];
    let mut t: [[Option<Tensor>; 3]; 3] = Default::default();
    let (mut gu_mat, mut down_mat) = ([0usize; 3], [0usize; 3]);
    for k in 0..3 {
        gu_mat[k] = XQ_TYPES[k].bytes((f * d) as u64).ok_or("XQ: rows that are not whole blocks")? as usize;
        down_mat[k] = XQ_TYPES[k].bytes((d * f) as u64).ok_or("XQ: rows that are not whole blocks")? as usize;
        for (w, role) in ["gate", "up", "down"].iter().enumerate() {
            let want = (if w == 2 { &dn } else { &gu }).iter().filter(|x| **x as usize == k).count();
            let name = format!("blk.{l}.ffn_{role}_exps.{}.weight", tname(k));
            match m.file.tensor(&name) {
                Some(tt) => {
                    let mb = if w == 2 { down_mat[k] } else { gu_mat[k] };
                    if tt.ty != XQ_TYPES[k] || tt.bytes as usize != mb * want {
                        return Err(Error(format!("{name}: {} {} bytes, {want} experts of {mb} expected", tt.ty.name(), tt.bytes)));
                    }
                    t[k][w] = Some(tt.clone());
                }
                None if want == 0 => {}
                None => return Err(Error(format!("{name} is missing ({want} experts of the type)"))),
            }
        }
    }
    let (mut ngu, mut ndn) = ([0u32; 3], [0u32; 3]);
    let ex: Vec<(u8, u32, u8, u32)> = (0..ne).map(|x| {
        let (a, b) = (gu[x] as usize, dn[x] as usize);
        let r = (gu[x], ngu[a], dn[x], ndn[b]);
        ngu[a] += 1;
        ndn[b] += 1;
        r
    }).collect();
    let present = |v: &[u8], k: usize| v.iter().any(|x| *x as usize == k);
    let g_max = (0..3).filter(|&k| present(&gu, k)).map(|k| gu_mat[k]).max().unwrap_or(0);
    let d_max = (0..3).filter(|&k| present(&dn, k)).map(|k| down_mat[k]).max().unwrap_or(0);
    let first = t.iter().flatten().flatten().next().cloned().ok_or_else(|| Error(format!("block {l}: no expert tensors")))?;
    let ty = [XQ_TYPES[gu[0] as usize], XQ_TYPES[gu[0] as usize], XQ_TYPES[dn[0] as usize]];
    Ok(LayerVq { blob: first.clone(), starts: Vec::new(), span: 2 * g_max + d_max, gate: 0, up: g_max, down: 2 * g_max, nbit: 0, cb_bytes: Vec::new(),
                 std: Some(StdExp { t: [first.clone(), first.clone(), first], ty, mat: [g_max, g_max, d_max], xq: Some(XqExp { t, ex, gu_mat, down_mat }) }) })
}

/// The std form's layer: the merged tensors' types and each expert's three slabs
fn layer_std(m: &Model, l: u64) -> Result<LayerVq> {
    let g = &m.g;
    let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
    let get = |r: Role| m.tensor(l, r).cloned().ok_or_else(|| Error(format!("block {l}: {r:?} is missing")));
    let t = [get(Role::ExpGate)?, get(Role::ExpUp)?, get(Role::ExpDown)?];
    let ty = [t[0].ty, t[1].ty, t[2].ty];
    if ty[0] != ty[1] {
        return Err(Error(format!("block {l}: gate is {} and up {}: the products take one type for both", ty[0].name(), ty[1].name())));
    }
    let ne = g.experts(l).0 as usize;
    let mat = [t[0].bytes as usize / ne, t[1].bytes as usize / ne, t[2].bytes as usize / ne];
    for (i, (rows, cols)) in [(f, d), (f, d), (d, f)].into_iter().enumerate() {
        let want = ty[i].bytes((rows * cols) as u64).ok_or_else(|| Error(format!("block {l}: {} rows of {cols} are not whole blocks", ty[i].name())))? as usize;
        if mat[i] != want || t[i].bytes as usize != want * ne {
            return Err(Error(format!("block {l}: {}: {} bytes an expert, {want} expected", t[i].name, mat[i])));
        }
    }
    Ok(LayerVq { blob: t[0].clone(), starts: Vec::new(), span: mat.iter().sum(), gate: 0, up: mat[0], down: mat[0] + mat[1], nbit: 0, cb_bytes: Vec::new(),
                 std: Some(StdExp { t, ty, mat, xq: None }) })
}

/// Reads the slot table and checks the payloads: every expert's three payloads back to back in one layout
fn layer_vq(m: &Model, l: u64) -> Result<LayerVq> {
    if m.g.xq {
        return layer_xq(m, l);
    }
    if m.g.std {
        return layer_std(m, l);
    }
    let g = &m.g;
    let t = m.tensor(l, Role::Experts).ok_or_else(|| Error(format!("block {l}: the expert blob is missing")))?.clone();
    let (ne, _) = g.experts(l);
    let mut head = vec![0u8; 16 + ne as usize * 24];
    m.file.read_into(&t, 0, &mut head).map_err(e)?;
    let u32_at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    if u32_at(&head, 0) != 0x4C56_5144 || u32_at(&head, 4) != 3 || u32_at(&head, 12) as u64 != ne {
        return Err(Error(format!("block {l}: not a DQVL v3 blob of {ne} experts")));
    }
    let slot = |x: usize, w: usize| u64_at(&head, 16 + (x * 3 + w) * 8);
    let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
    let mut ph = [0u8; 32];
    m.file.read_into(&t, slot(0, 0), &mut ph).map_err(e)?;
    let nc = u16::from_le_bytes([ph[6], ph[7]]) as usize;
    let nc = if nc == 0 { 65536 } else { nc };
    let nbit = (usize::BITS - (nc - 1).leading_zeros()) as usize;
    let cb_off = u64_at(&ph, 24);
    let s0 = slot(0, 0);
    let (gate, up, down) = (0usize, (slot(0, 1) - s0) as usize, (slot(0, 2) - s0) as usize);
    let mrow = |cols: usize| cols / 8 * 12 / 8;
    let plen = |rows: usize, cols: usize| 32 + rows * 2 + rows * mrow(cols) + if nbit > 12 { rows * (cols / 64) } else { 0 } + 4;
    let span = down + plen(d, f);
    if up < plen(f, d) || down - up < plen(f, d) {
        return Err(Error(format!("block {l}: expert 0's payloads overlap")));
    }
    let mut starts = Vec::with_capacity(ne as usize);
    for x in 0..ne as usize {
        let st = slot(x, 0);
        if slot(x, 1) - st != up as u64 || slot(x, 2) - st != down as u64 || st + span as u64 > t.bytes {
            return Err(Error(format!("block {l}: expert {x}'s payloads are not laid out as expert 0's")));
        }
        starts.push(st);
    }
    // each payload's header: the matrix's shape and the layer's codebook
    for (o, rows, cols) in [(gate, f, d), (up, f, d), (down, d, f)] {
        let mut h = [0u8; 32];
        m.file.read_into(&t, s0 + o as u64, &mut h).map_err(e)?;
        if u32_at(&h, 0) != 0x3356_5144 || u16::from_le_bytes([h[4], h[5]]) != 8 || u32_at(&h, 8) as usize != rows || u32_at(&h, 12) as usize != cols
            || u64_at(&h, 24) != cb_off || (u32_at(&h, 16) & 2 != 0) != (nbit > 12) {
            return Err(Error(format!("block {l}: a payload is not DQV3 [{rows} x {cols}] on the layer's codebook")));
        }
    }
    let mut cb_bytes = vec![0u8; nc * 8];
    m.file.read_into(&t, cb_off, &mut cb_bytes).map_err(e)?;
    Ok(LayerVq { blob: t, starts, span, gate, up, down, nbit, cb_bytes, std: None })
}

/// Expert `ex` of `vq`'s layer from the file into `host` (its span)
fn read_expert(m: &Model, vq: &LayerVq, ex: u64, host: &mut [u8]) -> Result<()> {
    if vq.std.is_some() {
        for i in 0..3 {
            let (t, off, n, _) = vq.mat_of(ex, i).ok_or("an expert's matrix")?;
            let at = vq.mat_at(ex, i);
            read_slab(m, t, off, &mut host[at..at + n], false)?;
        }
        return Ok(());
    }
    m.file.read_into(&vq.blob, vq.starts[ex as usize], &mut host[..vq.span]).map_err(e)
}

/// `read_expert`, a ggml-typed expert's three slabs on threads of their own (a synchronous read of one expert is the wait of
/// a decode layer: 3 reads in flight keep the drive busier than 1 does)
fn read_expert_par(m: &Model, vq: &LayerVq, ex: u64, host: &mut [u8], force: bool) -> Result<()> {
    if vq.std.is_none() {
        return read_expert(m, vq, ex, host);
    }
    let (_, up, down, _) = vq.lay(ex);
    let (a, rest) = host.split_at_mut(up);
    let (b, c) = rest.split_at_mut(down - up);
    let res: Vec<Result<()>> = std::thread::scope(|sc| {
        let hs: Vec<_> = [(0usize, a), (1, b), (2, c)].into_iter().map(|(i, buf)| {
            sc.spawn(move || {
                let (t, off, n, _) = vq.mat_of(ex, i).ok_or_else(|| Error("an expert's matrix".into()))?;
                read_slab(m, t, off, &mut buf[..n], force)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(Error("an expert reader panicked".into())))).collect()
    });
    res.into_iter().collect()
}

/// Tensor t (n values) as q4_K: from the cache (NS_DS41_Q4K_CACHE, a directory; the file named by the model file, its
/// size, the tensor's offset and name, its size checked) or requantized and kept there - NS_DS41_DENSE_Q4K's
/// requantization was ~26 s of a load's CPU time for the XQ file (exact 16.7 s, fast 42.3 s)
fn q4k_of(m: &Model, t: &Tensor, n: usize, from: crate::requant::From) -> Result<Vec<u8>> {
    let want = n / 256 * 144;
    let path = std::env::var("NS_DS41_Q4K_CACHE").ok().filter(|d| !d.is_empty()).map(|d| {
        let f = &m.file.paths[t.shard];
        let len = std::fs::metadata(f).map_or(0, |x| x.len());
        let stem = f.file_name().map_or("model".into(), |s| s.to_string_lossy().into_owned());
        std::path::Path::new(&d).join(format!("{stem}.{len:x}.{:x}.{}.q4k", t.offset, t.name))
    });
    if let Some(b) = path.as_ref().and_then(|p| std::fs::read(p).ok()).filter(|b| b.len() == want) {
        return Ok(b);
    }
    let host = m.file.read(t).map_err(e)?;
    let q = crate::requant::to_q4k(&host, n, from);
    if let Some(p) = &path {
        // kept for the next load (a failure only costs the next load the requantization again)
        let tmp = p.with_extension("tmp");
        if p.parent().is_some_and(|d| std::fs::create_dir_all(d).is_ok()) && std::fs::write(&tmp, &q).is_ok() {
            let _ = std::fs::rename(&tmp, p);
        }
    }
    Ok(q)
}

/// The model's files opened for O_DIRECT (NS_DS41_ODIRECT=1; default 0: buffered), for the experts' reads: one cold expert
/// is faster read straight from the drive (IQ2_XXS, three slabs: 3.7 ms buffered, 2.5 ms direct) but decode then loses the
/// page cache's hits on the cold experts that repeat (13.6 -> 12.7 tok/s), so it is opt-in
type DirectSet = Option<Arc<Vec<std::fs::File>>>;
/// A prompt pass's expert reads O_DIRECT (NS_DS41_PROMPT_ODIRECT, default 1)
fn prompt_odirect() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_ODIRECT", 1) != 0)
}
/// `force`: whatever NS_DS41_ODIRECT says (the prompt path's reads, NS_DS41_PROMPT_ODIRECT, default 1: a prompt reads each
/// cold expert once, so no page cache to hit - and none of decode's cached cold experts pushed out by it)
fn direct_files(paths: &[std::path::PathBuf], force: bool) -> DirectSet {
    use std::os::unix::fs::OpenOptionsExt;
    static D: Mutex<Option<(Vec<std::path::PathBuf>, DirectSet)>> = Mutex::new(None);
    if !force && env_usize("NS_DS41_ODIRECT", 0) == 0 {
        return None;
    }
    let mut g = D.lock().unwrap();
    if let Some((p, f)) = g.as_ref() {
        if p == paths {
            return f.clone();
        }
    }
    let o_direct = if cfg!(target_arch = "aarch64") { 0o200000 } else { 0o40000 };
    let files: Option<Vec<std::fs::File>> = paths.iter().map(|p| std::fs::OpenOptions::new().read(true).custom_flags(o_direct).open(p).ok()).collect();
    let f = files.map(Arc::new);
    *g = Some((paths.to_vec(), f.clone()));
    f
}

/// A page-aligned buffer of a reader thread's own for O_DIRECT reads
struct DirectBuf {
    ptr: *mut u8,
    cap: usize,
}
impl DirectBuf {
    fn get(&mut self, n: usize) -> &mut [u8] {
        if n > self.cap {
            if !self.ptr.is_null() {
                // SAFETY: allocated below with this layout
                unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.cap, 4096).unwrap()) };
            }
            self.cap = n.next_multiple_of(1 << 20);
            // SAFETY: a non-zero size; freed above or in Drop with the same layout
            self.ptr = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(self.cap, 4096).unwrap()) };
        }
        // SAFETY: cap >= n bytes allocated, owned by this value
        unsafe { std::slice::from_raw_parts_mut(self.ptr, n) }
    }
}
impl Drop for DirectBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated in get with this layout
            unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.cap, 4096).unwrap()) };
        }
    }
}
thread_local! {
    static DBUF: std::cell::RefCell<DirectBuf> = const { std::cell::RefCell::new(DirectBuf { ptr: std::ptr::null_mut(), cap: 0 }) };
}

/// Bytes [at, at + dst.len()) of `file` (opened O_DIRECT) into dst: the 4 KiB-aligned span around them read into the
/// thread's own buffer, the wanted part copied out
fn pread_direct(file: &std::fs::File, at: u64, dst: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    let a = at & !4095;
    let end = (at + dst.len() as u64 + 4095) & !4095;
    DBUF.with(|b| {
        let mut b = b.borrow_mut();
        let buf = b.get((end - a) as usize);
        let want = (at - a) as usize + dst.len();
        let mut got = 0;
        // (the last bytes of a file are short of the aligned span: enough is enough)
        while got < want {
            let k = file.read_at(&mut buf[got..], a + got as u64)?;
            if k == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "past the end of the file"));
            }
            got += k;
        }
        dst.copy_from_slice(&buf[(at - a) as usize..want]);
        Ok(())
    })
}

/// A slab of expert tensor `t` (bytes [off, off + dst.len()) of it): O_DIRECT when the files opened so, else buffered
fn read_slab(m: &Model, t: &Tensor, off: u64, dst: &mut [u8], force: bool) -> Result<()> {
    if let Some(files) = direct_files(&m.file.paths, force) {
        return pread_direct(&files[t.shard], t.offset + off, dst).map_err(e);
    }
    m.file.read_into(t, off, dst).map_err(e)
}

/// The file reads of expert `x` of a layer into the memory at `dst` (its span): a VQ expert is one range of the blob, a
/// ggml-typed one three slabs, one in each merged tensor (a job each, on different threads)
fn expert_segs(vq: &LayerVq, x: u64, dst: usize) -> Vec<(usize, u64, usize, usize)> {
    match &vq.std {
        Some(_) => (0..3).filter_map(|i| {
            let (t, off, n, _) = vq.mat_of(x, i)?;
            Some((t.shard, t.offset + off, n, dst + vq.mat_at(x, i)))
        }).collect(),
        None => vec![(vq.blob.shard, vq.blob.offset + vq.starts[x as usize], vq.span, dst)],
    }
}

/// Where an expert is.
#[derive(Clone, Copy)]
enum Loc {
    V(usize),
    R(usize),
}

/// The routed experts of some of a GPU's layers (of one span size), in an exclusive two-level cache: VRAM slots and
/// pinned host slots hold different experts; a VRAM miss swaps (the least recently used VRAM expert down to a free
/// host slot, the wanted one up); what neither holds is read from the file (deepseek4's store).
struct Store {
    vram: Vec<DevBuf>,
    host: Vec<HostBuf>,
    slot_bytes: usize,
    per_chunk: usize,
    loc: HashMap<(u64, u64), Loc>,
    vowner: Vec<Option<(u64, u64)>>,
    vused: Vec<u64>,
    rfree: Vec<usize>,
    /// each host slot's expert and the tick it was last used (moved down or read in place); a ticket after the GPU's
    /// work so far, taken once a tick before a host slot is given up (products may still read it in place)
    rowner: Vec<Option<(u64, u64)>>,
    rused: Vec<u64>,
    rmark: Option<i64>,
    tick: u64,
    hits: u64,
    misses: u64,
    from_host: u64,
    from_file: u64,
    direct: u64,
    rwrite: HashMap<usize, i64>,
    /// the copy up last reading each host slot (a copy down into it waits for it)
    rread: HashMap<usize, i64>,
    /// the slots' payload addresses on the GPU (the slot table's entries are copied from here), and the zeroing of a
    /// slot's last expert in the table that a copy up into it waits for
    slotaddr: Option<DevBuf>,
    vzero: HashMap<usize, i64>,
    n_expert: usize,
    down_any: Option<i64>,
    vfill: HashMap<usize, i64>,
    last_up: Option<i64>,
    prefetched: HashMap<(u64, u64), i64>,
    usage: HashMap<(u64, u64), u64>,
    base: usize,
    lend: usize,
    lent: bool,
    /// the lent slots' chunk of `vram` (the big arena), then the passes' reserve's: xlend more slots after them, in chunk
    /// x_chunk, the first x_on of them in use (allocated from what decode leaves free, given back before a pass that needs it)
    lend_chunk: usize,
    xlend: usize,
    x_chunk: usize,
    x_on: usize,
    pf_issued: u64,
    pf_used: u64,
    /// pinned staging slots for experts read from the file (copied up from there on the copy lane, not through a
    /// queue drain), and the copy last reading each
    stage: HostBuf,
    stage_tk: Vec<Option<i64>>,
    /// a read ahead writing each staging slot, and the reads ahead by expert: (layer, expert) -> (slot, its read)
    stage_rd: Vec<Option<Arc<RDone>>>,
    ahead: HashMap<(u64, u64), (usize, Arc<RDone>)>,
    stage_next: usize,
    /// a staging slot holding an expert read for this step whose copy up is not queued yet (a read ahead taken, or
    /// stage_reads'): no other read may take it until then
    stage_held: Vec<bool>,
}

/// the file reads a layer takes at once (in parallel, into the staging slots)
const STAGE: usize = 16;

impl Store {
    /// The `k` slots to fill next: empty ones first, then the least recently used (a partial selection: a store has
    /// thousands of slots, a layer's misses a few - sorting them all was the host's turn between router and products)
    fn victims(&self, tick: u64, k: usize) -> Vec<usize> {
        let key = |i: usize| if self.vowner[i].is_none() { 0 } else { self.vused[i] + 1 };
        let mut order: Vec<usize> = (0..self.vowner.len()).filter(|&i| self.vused[i] != tick && self.usable(i)).collect();
        if k < order.len() {
            if k == 0 {
                return Vec::new();
            }
            order.select_nth_unstable_by_key(k - 1, |&i| (key(i), i));
            order.truncate(k);
        }
        order.sort_by_key(|&i| (key(i), i));
        order
    }
    fn vat(&self, s: usize) -> (usize, usize) {
        if s >= self.base {
            let k = s - self.base;
            if k < self.lend {
                return (self.lend_chunk, k * self.slot_bytes);
            }
            return (self.x_chunk, (k - self.lend) * self.slot_bytes);
        }
        (s / self.per_chunk, (s % self.per_chunk) * self.slot_bytes)
    }
    fn usable(&self, s: usize) -> bool {
        s < self.base || if s < self.base + self.lend { self.lent } else { s - self.base - self.lend < self.x_on }
    }
    /// The slots' payload addresses on the GPU, a u64 each
    fn slot_addrs(&self) -> Vec<u8> {
        (0..self.vowner.len()).flat_map(|s| {
            let (ch, o) = self.vat(s);
            (self.vram[ch].ptr() as u64 + o as u64).to_le_bytes()
        }).collect()
    }
    fn rat(&self, r: usize) -> (usize, usize) {
        (r / self.per_chunk, (r % self.per_chunk) * self.slot_bytes)
    }
}

/// One GPU's share of the model.
pub struct Part {
    ops: Ops,
    pub arena: Arena,
    big: DevBuf,
    small: DevBuf,
    pub layers: Range<u64>,
    mats: BTreeMap<(u64, Role), Mat>,
    vecs: BTreeMap<(u64, Role), DevBuf>,
    /// o_a's groups as matrices of their own
    groups: BTreeMap<u64, Vec<Mat>>,
    scratch: DevBuf,
    /// Q8_1 of up to MMVQ_COLS rows
    q8: DevBuf,
    /// the routers' biases on the host
    biases: Mutex<HashMap<u64, Arc<Vec<f32>>>>,
    /// per layer, the std kernel's table for the grouped o_a at one row (a Q8_0 / Q6_K o_a): 8 groups' pointers, then meta
    oa_tab: Mutex<HashMap<u64, Arc<DevBuf>>>,
    /// the next layer's file experts a prompt pass already started reading (layer, expert -> pinned slot and its reads)
    pre_reads: Mutex<Option<(u64, HashMap<u64, (usize, Arc<RDone>)>)>>,
    vq: HashMap<u64, LayerVq>,
    cbs: HashMap<u64, DevBuf>,
    /// the codebooks as float32 (the decode products' weights without a conversion each)
    cbf: HashMap<u64, DevBuf>,
    /// and as fp16 (exact), for the variants that gather 16 bytes an index
    cbh: HashMap<u64, DevBuf>,
    /// the decode products' kernel variant by index width: [gate | up, down] (measured at load: vq_pick)
    vq_pick: HashMap<usize, [i32; 2]>,
    /// the expert stores (one a slot size) and each (layer, slot size)'s - an expert's slot size is LayerVq::slot_of: one
    /// for every expert of a layer but the XQ file's, whose experts' sizes follow their types
    stores: Vec<Mutex<Store>>,
    store_of: HashMap<(u64, usize), usize>,
    /// the towers' experts, all in VRAM (the last part): a buffer a tower
    draft: HashMap<u64, DevBuf>,
    hold: std::sync::atomic::AtomicBool,
    /// VRAM held back at load for the sessions' caches (kept free when the passes' reserve is lent), and whether a lend
    /// of that reserve is due (after load and after each pass that took it back)
    kv_reserve: usize,
    x_due: std::sync::atomic::AtomicBool,
    x_first: std::sync::atomic::AtomicBool,
    pub expert_slots: usize,
    pub host_slots: usize,
    pub weight_bytes: u64,
    up: Mutex<(HostBuf, usize)>,
    back: Mutex<HostBuf>,
    /// the index sources' picks [prefill_chunk][top_k] and the candidate source's blocks [prefill_chunk][cand_top],
    /// kept for the layers after them in a pass
    sel: DevBuf,
    cand: DevBuf,
    /// the compressor's position terms (zero: V4.1 has none) for the pooling kernel
    zeros: DevBuf,
    /// a prompt pass's staging slots in VRAM for the experts in host memory or the file (copied up on the copy lane while
    /// the others are multiplied; none when NS_DS41_PROMPT_RING=0): the buffer and its slots' size
    ring: Option<(DevBuf, usize)>,
    /// the ticket after the last pass's readers of the ring (its next copies wait for it)
    ring_free: Mutex<Option<i64>>,
    /// decode's reads ahead of the file's experts (None: NS_DS41_FILE_AHEAD=0)
    reader: Option<FileReader>,
    /// the slot table the GPU routes decode by: [layer][expert] -> the VRAM payload's address, 0 when not in VRAM
    devtab: DevBuf,
    /// the XQ file's: [layer][expert] -> its type word (LayerVq::tag) and its up's place in its slot / 256 (bits 16-31)
    /// for the GPU routing's tables
    xtags: Option<DevBuf>,
    /// a prompt pass's pinned slots for the experts only the file holds: read on threads while the GPU multiplies the
    /// others, then copied up through the ring; the copy last reading each slot
    pstage: Option<PStage>,
}

/// A prompt pass's pinned slots for the file's experts: the buffer, a slot's bytes, the copy last reading each
type PStage = (Mutex<HostBuf>, usize, Mutex<Vec<Option<i64>>>);

/// Decode's file reads ahead: a few threads with their own handles on the model's files, reading an expert's bytes into
/// a pinned staging slot while the GPU works (the next layer's likely experts, from its router's guess)
struct FileReader {
    tx: Mutex<std::sync::mpsc::Sender<RJob>>,
}
struct RJob {
    /// (shard, file offset, bytes, destination address): a VQ expert is one range of the file, a ggml-typed one three
    /// (a job each, on different threads)
    seg: (usize, u64, usize, usize),
    /// a prompt pass's read: O_DIRECT (NS_DS41_PROMPT_ODIRECT), past the page cache
    direct: bool,
    done: Arc<RDone>,
}
/// The reads of one expert (a job each) done: the count left, and the first error
struct RDone {
    state: Mutex<(usize, Option<String>)>,
    cv: std::sync::Condvar,
}
impl RDone {
    fn new(reads: usize) -> RDone {
        RDone { state: Mutex::new((reads, None)), cv: std::sync::Condvar::new() }
    }
    fn finish(&self, r: std::result::Result<(), String>) {
        let mut g = self.state.lock().unwrap();
        g.0 -= 1;
        if let (Err(x), None) = (r, &g.1) {
            g.1 = Some(x);
        }
        if g.0 == 0 {
            self.cv.notify_all();
        }
    }
    fn is_done(&self) -> bool {
        self.state.lock().unwrap().0 == 0
    }
    fn wait(&self) -> Result<()> {
        let mut g = self.state.lock().unwrap();
        while g.0 > 0 {
            g = self.cv.wait(g).unwrap();
        }
        g.1.clone().map_or(Ok(()), |x| Err(Error(x)))
    }
}
impl FileReader {
    fn new(paths: &[std::path::PathBuf], threads: usize) -> Result<FileReader> {
        let files: Vec<Arc<std::fs::File>> = paths.iter().map(|p| std::fs::File::open(p).map(Arc::new).map_err(|e| Error(format!("{}: {e}", p.display()))))
            .collect::<Result<_>>()?;
        let (tx, rx) = std::sync::mpsc::channel::<RJob>();
        let rx = Arc::new(Mutex::new(rx));
        // decode's reads as NS_DS41_ODIRECT says; a prompt's past the page cache (NS_DS41_PROMPT_ODIRECT): buffered, a
        // 2,561-token prompt's ~70 GB of XQ experts went through the page cache beside ~45 GiB of pinned memory, and its
        // reclaim stalled the readers for seconds (the same prompt 16 s or 35 s)
        let direct = direct_files(paths, false);
        let pdirect = if prompt_odirect() { direct_files(paths, true) } else { direct.clone() };
        for _ in 0..threads {
            let (rx, files, direct, pdirect) = (rx.clone(), files.clone(), direct.clone(), pdirect.clone());
            std::thread::spawn(move || loop {
                let job = match rx.lock().unwrap().recv() {
                    Ok(j) => j,
                    Err(_) => return,
                };
                use std::os::unix::fs::FileExt;
                let (shard, at, len, dst) = job.seg;
                // SAFETY: the job's staging slot is its own (read_ahead) until its done flag is set and waited for
                // (stage_slot or ensure), the pinned buffer alive for the engine
                let b = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
                let r = match if job.direct { &pdirect } else { &direct } {
                    Some(d) => pread_direct(&d[shard], at, b),
                    None => files[shard].read_exact_at(b, at),
                };
                job.done.finish(r.map_err(|e| format!("reading an expert: {e}")));
            });
        }
        Ok(FileReader { tx: Mutex::new(tx) })
    }
}

/// Where a prompt pass reads an expert from (Part::locate)
enum Src {
    V(DevBuf),
    R(DevBuf),
    F,
}

/// The pass-wide state the attention layers hand down: whether the last index source picked (its top-k rows in
/// `Part::sel`) and whether the candidate source kept blocks (`Part::cand`)
#[derive(Clone, Copy, Default)]
struct Shared {
    sel: bool,
    cand: bool,
}

impl Part {
    fn mat(&self, l: u64, r: Role) -> Result<&Mat> {
        self.mats.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: matrix {r:?} not on this GPU")))
    }
    fn vec(&self, l: u64, r: Role) -> Result<&DevBuf> {
        self.vecs.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: vector {r:?} not on this GPU")))
    }

    /// `data` into dst in queue order: through the pinned ring, copied by a kernel (big ones by write_async)
    fn upload(&self, dst: &DevBuf, data: &[u8]) -> Result<()> {
        let n = data.len();
        let mut r = self.up.lock().unwrap();
        if n == 0 || !n.is_multiple_of(4) || n > r.0.len / 4 {
            drop(r);
            return dst.write_async(0, data);
        }
        if r.1 + n > r.0.len {
            self.ops.gpu.sync()?;
            r.1 = 0;
        }
        let off = r.1;
        r.0.as_mut_slice()[off..off + n].copy_from_slice(data);
        r.1 = (off + n).next_multiple_of(64);
        let src = r.0.device_view(off, n)?;
        self.ops.kcopy(dst, 0, &src, 0, n)?;
        Ok(())
    }

    /// Floats of device buffers read back through the pinned ring by kernels (None: they do not fit it)
    #[allow(clippy::type_complexity)]
    fn read_start(&self, xs: &[(&DevBuf, usize)]) -> Result<Option<(i64, Vec<(usize, usize)>)>> {
        let back = self.back.lock().unwrap();
        let total: usize = xs.iter().map(|(_, n)| (n * 4).next_multiple_of(64)).sum();
        if total > back.len {
            return Ok(None);
        }
        let (mut at, mut spans, mut tk) = (0, Vec::new(), 0);
        for (x, n) in xs {
            tk = self.ops.kcopy(&back.device_view(at, n * 4)?, 0, x, 0, n * 4)?;
            spans.push((at, n * 4));
            at += (n * 4).next_multiple_of(64);
        }
        Ok(Some((tk, spans)))
    }
    fn read_finish(&self, tk: i64, spans: &[(usize, usize)]) -> Result<Vec<Vec<f32>>> {
        self.ops.ticket_wait(tk)?;
        let back = self.back.lock().unwrap();
        Ok(spans.iter().map(|(a, b)| back.as_slice()[*a..a + b].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).collect())
    }
    fn read_f32(&self, x: &DevBuf, n: usize) -> Result<Vec<f32>> {
        match self.read_start(&[(x, n)])? {
            Some((tk, spans)) => Ok(self.read_finish(tk, &spans)?.into_iter().next().unwrap_or_default()),
            None => Ok(x.view(0, n * 4)?.to_f32()?),
        }
    }

    /// What decode leaves free of the passes' reserve, lent to the stores as more expert slots (NS_DS41_LEND_RESERVE): the
    /// VRAM free now less NS_DS41_FREE_GIB, the sessions' held-back caches and 1 GiB for decode's own allocations, shared
    /// by the stores' sizes. The first lend fixes the slot counts; a later one (after a pass took the reserve back) lends
    /// what fits of them. The new slots are filled at once from the pinned host tier, its first slots (the
    /// profile's next experts at load): the copies up beside the work, the host slots they free taking VRAM's victims later
    fn lend_reserve(&self) -> Result<()> {
        let Some(free) = self.ops.gpu.memory()?.1 else { return Ok(()) };
        let room = (free as usize).saturating_sub(keep_free() + self.kv_reserve + (1 << 30));
        let mut cs: Vec<_> = self.stores.iter().map(|s| s.lock().unwrap()).collect();
        if cs.iter().any(|c| c.x_on > 0) {
            return Ok(());
        }
        let weight: usize = cs.iter().map(|c| c.slot_bytes * c.base).sum::<usize>().max(1);
        // a store's share of the room, in slots (a later lend: at most the first's count)
        let want: Vec<usize> = cs.iter().map(|c| {
            let n = (room as u128 * (c.slot_bytes * c.base) as u128 / weight as u128) as usize / c.slot_bytes;
            if c.xlend > 0 { n.min(c.xlend) } else { n }
        }).collect();
        let mut lent = 0;
        for (c, &n) in cs.iter_mut().zip(&want) {
            let c = &mut **c;
            if n == 0 {
                continue;
            }
            let buf = DevBuf::new(&self.ops.gpu, n * c.slot_bytes)?;
            if c.xlend == 0 {
                c.vram.push(buf);
                c.x_chunk = c.vram.len() - 1;
                c.xlend = n;
                let nv = c.vowner.len();
                c.vowner.resize(nv + n, None);
                c.vused.resize(nv + n, 0);
            } else {
                let xc = c.x_chunk;
                c.vram[xc] = buf;
            }
            c.x_on = n;
            let addrs = c.slot_addrs();
            let b = DevBuf::new(&self.ops.gpu, addrs.len().max(8))?;
            b.write(0, &addrs)?;
            c.slotaddr = Some(b);
            // filled from the host tier's first slots
            let sb = c.slot_bytes;
            let x0 = c.base + c.lend;
            let from: Vec<(usize, (u64, u64))> = (0..c.rowner.len()).filter_map(|r| c.rowner[r].map(|v| (r, v))).take(n).collect();
            for (k, (r, key)) in from.into_iter().enumerate() {
                let sl = x0 + k;
                let (vch, vo) = c.vat(sl);
                let (rch, ro) = c.rat(r);
                let t = self.ops.stream_copy_on(1, &c.vram[vch], vo, &c.host[rch].device_view(ro, sb)?, 0, sb, &[c.rwrite.get(&r).copied()])?;
                c.rread.insert(r, t);
                c.rfree.push(r);
                c.rowner[r] = None;
                c.vfill.insert(sl, t);
                c.last_up = Some(t);
                c.vowner[sl] = Some(key);
                c.vused[sl] = c.tick;
                self.tab_set_after(c, key, sl, t)?;
                c.loc.insert(key, Loc::V(sl));
                // a product the host routes waits for the copy as for a prefetch's
                c.prefetched.insert(key, t);
            }
            lent += n * sb;
        }
        if lent > 0 && (std::env::var("NS_PROFILE").is_ok() || self.x_first.swap(false, std::sync::atomic::Ordering::Relaxed)) {
            eprintln!("[{}: {} GiB of the passes' reserve lent to the expert stores ({:.1} GiB was free): {} more slots]", self.ops.gpu.name,
                      format!("{:.1}", gib(lent)), gib(free as usize), want.iter().sum::<usize>());
        }
        Ok(())
    }

    /// The arena a pass of `t` tokens needs: the small one, or the big one (its memory then out of the stores)
    fn arena_for(&self, t: usize) -> Result<()> {
        let need = t * (arena_bytes() / prefill_chunk()).max(1) + (64 << 20);
        let lent_any = self.stores.iter().any(|s| s.lock().unwrap().lend > 0);
        if !lent_any {
            return Ok(());
        }
        let hold = self.hold.load(std::sync::atomic::Ordering::Relaxed);
        let lent = self.stores.iter().any(|s| s.lock().unwrap().lent);
        // lent at decode's first step (a prompt pass reads host-tier experts in place, by another kernel than a resident
        // one's: filling the slots during a prompt would change its rounding)
        if t <= MMVQ_COLS && lent && self.x_due.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.lend_reserve()?;
        }
        if need > SMALL_ARENA && lent {
            let after = Some(self.ops.mark()?);
            for st in &self.stores {
                let mut c = st.lock().unwrap();
                let c = &mut *c;
                if c.lend == 0 && c.x_on == 0 {
                    continue;
                }
                let sb = c.slot_bytes;
                let hi = c.base + c.lend + c.x_on;
                for s in c.base..hi {
                    let Some(v) = c.vowner[s].take() else { continue };
                    self.tab_zero(c, v)?;
                    c.loc.remove(&v);
                    c.prefetched.remove(&v);
                    let (vch, vo) = c.vat(s);
                    if let Some(f) = c.rfree.pop() {
                        let (rch, ro) = c.rat(f);
                        let t = self.ops.stream_copy_on(0, &c.host[rch].device_view(ro, sb)?, 0, &c.vram[vch], vo, sb, &[after, c.vfill.get(&s).copied(), c.rwrite.get(&f).copied()])?;
                        c.rwrite.insert(f, t);
                        c.down_any = Some(t);
                        c.loc.insert(v, Loc::R(f));
                        c.rowner[f] = Some(v);
                    }
                    c.vfill.remove(&s);
                }
                for t in [c.down_any, c.last_up].into_iter().flatten() {
                    self.ops.await_ticket(t)?;
                }
                c.lent = false;
            }
            self.ops.gpu.sync()?;
            // the passes' reserve given back: its slots' buffer freed (the pass allocates beyond its arena)
            for st in &self.stores {
                let mut c = st.lock().unwrap();
                if c.x_on > 0 {
                    let xc = c.x_chunk;
                    c.vram[xc] = DevBuf::new(&self.ops.gpu, 256)?;
                    c.x_on = 0;
                }
            }
            self.arena.set(self.big.view(0, self.big.len)?);
        } else if need <= SMALL_ARENA && !lent && !hold {
            for st in &self.stores {
                let mut c = st.lock().unwrap();
                if c.lend > 0 {
                    c.lent = true;
                }
            }
            self.arena.set(self.small.view(0, self.small.len)?);
            if lend_extra() {
                self.x_due.store(true, std::sync::atomic::Ordering::Relaxed);
                if t <= MMVQ_COLS && self.x_due.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    self.lend_reserve()?;
                }
            }
        }
        Ok(())
    }

    /// Layer l's router bias on the host
    fn router_bias(&self, l: u64) -> Result<Arc<Vec<f32>>> {
        if let Some(b) = self.biases.lock().unwrap().get(&l) {
            return Ok(b.clone());
        }
        let b = Arc::new(self.vec(l, Role::RouterBias)?.to_f32()?);
        self.biases.lock().unwrap().insert(l, b.clone());
        Ok(b)
    }

    /// y[t rows from y.1, y.2 apart] (+)= x . W^T: float matrices by oneMKL; q4_K from its blocks for up to MMVQ_COLS
    /// rows, else expanded; fp8_32x32 from its bytes for up to MMVQ_COLS rows, else expanded to fp16
    fn matmul(&self, w: &Mat, t: usize, x: (&DevBuf, usize, usize), y: (&DevBuf, usize, usize), acc: bool) -> Result<()> {
        let o = &self.ops;
        if w.f32() {
            if t <= MMVQ_COLS {
                for r in 0..t {
                    o.gemm_at(1, w.rows, w.cols, (x.0, x.1 + r * x.2, x.2), (&w.buf, 0), (y.0, y.1 + r * y.2, y.2), acc)?;
                }
                return Ok(());
            }
            return o.gemm_at(t, w.rows, w.cols, x, (&w.buf, 0), y, acc);
        }
        let contiguous = x.2 == w.cols && y.2 == w.rows;
        if w.ty == GType::Fp8B32 {
            if !acc && t <= MMVQ_COLS && contiguous {
                return o.fp8_mmv(&w.buf, w.rows, w.cols, (x.0, x.1, w.cols), None, (y.0, y.1, w.rows), t);
            }
            let x16 = self.arena.bytes(t * w.cols * 2)?;
            if x.2 == w.cols {
                o.to_f16(&x.0.view(x.1 * 4, t * w.cols * 4)?, &x16, t * w.cols)?;
            } else {
                for r in 0..t {
                    o.to_f16(&x.0.view((x.1 + r * x.2) * 4, w.cols * 4)?, &x16.view(r * w.cols * 2, w.cols * 2)?, w.cols)?;
                }
            }
            // whole 32-row tiles a chunk
            let chunk = ((SCRATCH * 2 / w.cols) / 32 * 32).max(32).min(w.rows);
            let mut r0 = 0;
            while r0 < w.rows {
                let r = chunk.min(w.rows - r0);
                o.fp8_dequant(&w.buf, 0, w.rows, w.cols, r0, r, &self.scratch, true)?;
                o.gemm_f16(t, r, w.cols, (&x16, 0, w.cols), (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
                r0 += r;
            }
            return Ok(());
        }
        if !acc && t <= MMVQ_COLS && w.ty == GType::Q4K && q4k_own() {
            return o.q4k_mv(&w.buf, w.rows, w.cols, x, None, y, t);
        }
        // a short prompt's rows 8 at a time through the decode product: below the fp16 GEMM's rows the expansion was
        // to float32, the whole matrix a pass (q_b: 168 MB written for a 21-token prompt)
        if !acc && t < dense_f16_min() && w.ty == GType::Q4K && q4k_own() {
            let mut r0 = 0;
            while r0 < t {
                let n = MMVQ_COLS.min(t - r0);
                o.q4k_mv(&w.buf, w.rows, w.cols, (x.0, x.1 + r0 * x.2, x.2), None, (y.0, y.1 + r0 * y.2, y.2), n)?;
                r0 += n;
            }
            return Ok(());
        }
        if !acc && t <= MMVQ_COLS && contiguous && o.mmvq_supported(w.ty.code()) {
            o.quantize_q8_1((x.0, x.1), &self.q8, w.cols, t)?;
            return o.mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.q8, (y.0, y.1), w.cols, w.rows, t);
        }
        let rb = w.ty.bytes(w.cols as u64).ok_or("matmul: a row of whole blocks")? as usize;
        if t >= dense_f16_min() && x.2 == w.cols && w.cols.is_multiple_of(256) {
            let x16 = self.arena.bytes(t * w.cols * 2)?;
            o.to_f16(&x.0.view(x.1 * 4, t * w.cols * 4)?, &x16, t * w.cols)?;
            let chunk = (SCRATCH * 2 / w.cols).max(1).min(w.rows);
            let mut r0 = 0;
            while r0 < w.rows {
                let r = chunk.min(w.rows - r0);
                o.dequant_mat(w.ty.code(), &w.buf, r0 * rb, r * rb, r, w.cols, &self.scratch, true)?;
                o.gemm_f16(t, r, w.cols, (&x16, 0, w.cols), (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
                r0 += r;
            }
            return Ok(());
        }
        let chunk = (SCRATCH / w.cols).max(1).min(w.rows);
        let mut r0 = 0;
        while r0 < w.rows {
            let r = chunk.min(w.rows - r0);
            o.dequant_mat(w.ty.code(), &w.buf, r0 * rb, r * rb, r, w.cols, &self.scratch, false)?;
            o.gemm_at(t, r, w.cols, x, (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
            r0 += r;
        }
        Ok(())
    }

    /// y [t, rows] = x [t, cols] . W^T
    fn mm(&self, l: u64, r: Role, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let w = self.mat(l, r)?;
        let y = self.arena.f32(t * w.rows)?;
        self.matmul(w, t, (x, 0, w.cols), (&y, 0, w.rows), false)?;
        Ok(y)
    }

    /// Layer l's decode products of its experts' gate | up (`down` false) or down matrices (vq_mmv's table): the VQ form's
    /// from the indices, the ggml-typed form's by dequantizing in the kernel
    #[allow(clippy::too_many_arguments)]
    fn exp_mmv(&self, l: u64, down: bool, table: &DevBuf, entries: usize, x: &DevBuf, y: (&DevBuf, usize), y_rows: usize, rows: usize, cols: usize, var: i32,
               lut: Option<(usize, usize)>, ncmax: usize) -> Result<()> {
        let v = &self.vq[&l];
        if let Some(s) = &v.std {
            // the XQ file's: each entry's type its own (in its first meta word); one column an entry on the split-Q8
            // products like the rest, else the float32 kernel
            if s.xq.is_some() {
                let ty = if down { T_MIXED_DOWN } else { T_MIXED_GU };
                if let Some((xc, _)) = lut.filter(|_| ncmax <= 1 && std_q8() && self.ops.std_mmvq_supported(ty, cols)) {
                    let xs = self.arena.bytes(self.ops.std_q8_bytes(cols, xc))?;
                    self.ops.std_quantize_q8(x, &xs, cols, xc)?;
                    return self.ops.std_mmvq(ty, table, entries, &xs, xc, y, y_rows, rows, cols);
                }
                return self.ops.std_mmv(ty, table, entries, x, y, y_rows, rows, cols, ncmax);
            }
            let ty = s.ty[if down { 2 } else { 0 }].code();
            // one column an entry (decode) with x's column count known: the split-Q8 products (NS_DS41_STD_Q8=0: float32)
            if let Some((xc, _)) = lut.filter(|_| ncmax <= 1 && std_q8() && self.ops.std_mmvq_supported(ty, cols)) {
                let xs = self.arena.bytes(self.ops.std_q8_bytes(cols, xc))?;
                self.ops.std_quantize_q8(x, &xs, cols, xc)?;
                return self.ops.std_mmvq(ty, table, entries, &xs, xc, y, y_rows, rows, cols);
            }
            return self.ops.std_mmv(ty, table, entries, x, y, y_rows, rows, cols, ncmax);
        }
        let cb = self.cbs.get(&l).ok_or_else(|| Error(format!("layer {l}: no codebook")))?;
        self.ops.vq_mmv(table, entries, (cb, vq_wide((&self.cbf, &self.cbh), l, var)), x, y, y_rows, rows, cols, v.nbit, var, lut)
    }

    /// Two products of the same x: at one decode row, both q4_K (or both float32) in one launch (NS_DS41_KV_FUSE=0: two)
    fn mm2(&self, (l1, r1): (u64, Role), (l2, r2): (u64, Role), x: &DevBuf, t: usize) -> Result<(DevBuf, DevBuf)> {
        let (w1, w2) = (self.mat(l1, r1)?, self.mat(l2, r2)?);
        if t == 1 && fused() && w1.cols == w2.cols {
            if w1.ty == GType::Q4K && w2.ty == GType::Q4K && q4k_own() && w1.rows % 4 == 0 && w1.cols % 256 == 0 {
                let (y1, y2) = (self.arena.f32(w1.rows)?, self.arena.f32(w2.rows)?);
                self.ops.q4k_mv2(&w1.buf, w1.rows, &w2.buf, w2.rows, w1.cols, x, &y1, &y2)?;
                return Ok((y1, y2));
            }
            if w1.f32() && w2.f32() && w1.cols % 4 == 0 {
                let (y1, y2) = (self.arena.f32(w1.rows)?, self.arena.f32(w2.rows)?);
                self.ops.gemv2(x, &w1.buf, w1.rows, &w2.buf, w2.rows, w1.cols, &y1, &y2)?;
                return Ok((y1, y2));
            }
        }
        Ok((self.mm(l1, r1, x, t)?, self.mm(l2, r2, x, t)?))
    }

    /// the store holding layer l's expert x
    fn store_ix(&self, l: u64, x: u64) -> Option<usize> {
        self.store_of.get(&(l, self.vq.get(&l)?.slot_of(x))).copied()
    }
    /// layer l's stores (several only for the XQ file)
    fn stores_of(&self, l: u64) -> Vec<usize> {
        let mut v: Vec<usize> = self.store_of.iter().filter(|((ly, _), _)| *ly == l).map(|(_, &i)| i).collect();
        v.sort_unstable();
        v
    }
    /// `xs`' positions grouped by their store, the stores in the order their first expert comes
    fn by_store(&self, l: u64, xs: &[u64]) -> Result<Vec<(usize, Vec<usize>)>> {
        let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
        for (k, &x) in xs.iter().enumerate() {
            let si = self.store_ix(l, x).ok_or_else(|| Error(format!("block {l}: no expert store on this GPU")))?;
            match out.iter_mut().find(|(i, _)| *i == si) {
                Some((_, v)) => v.push(k),
                None => out.push((si, vec![k])),
            }
        }
        Ok(out)
    }
    /// f on the store holding layer l's expert x (None: no store of the layer here)
    fn with_store<R>(&self, l: u64, x: u64, f: impl FnOnce(&mut Store) -> R) -> Option<R> {
        let si = self.store_ix(l, x)?;
        let mut c = self.stores[si].lock().unwrap();
        Some(f(&mut c))
    }

    /// `ensure` over the stores `need` spans (one but for the XQ file), the results in `need`'s order
    #[allow(clippy::type_complexity)]
    fn ensure(&self, m: &Model, l: u64, need: &[u64], promote: bool) -> Result<(Vec<DevBuf>, Vec<bool>, Option<i64>)> {
        if let Some(buf) = self.draft.get(&l) {
            let span = self.vq[&l].span.next_multiple_of(256);
            let v = need.iter().map(|x| buf.view(*x as usize * span, span)).collect::<Result<Vec<_>>>()?;
            return Ok((v, vec![false; need.len()], None));
        }
        let groups = self.by_store(l, need)?;
        if groups.len() == 1 {
            return self.ensure_in(m, groups[0].0, l, need, promote);
        }
        let mut slots: Vec<Option<DevBuf>> = (0..need.len()).map(|_| None).collect();
        let mut arriving = vec![false; need.len()];
        let mut last: Option<i64> = None;
        for (si, ks) in groups {
            let sub: Vec<u64> = ks.iter().map(|&k| need[k]).collect();
            let (v, a, t) = self.ensure_in(m, si, l, &sub, promote)?;
            for ((k, v), a) in ks.into_iter().zip(v).zip(a) {
                slots[k] = Some(v);
                arriving[k] = a;
            }
            last = match (last, t) {
                (Some(x), Some(y)) => Some(x.max(y)),
                (x, y) => x.or(y),
            };
        }
        Ok((slots.into_iter().map(|v| v.ok_or_else(|| Error("ensure: an expert without a slot".into()))).collect::<Result<_>>()?, arriving, last))
    }

    /// Where every expert of `need` (layer l) is read from, as a view of its slot: `promote` makes each VRAM-resident
    /// (decode), else host-slot experts are read in place. The second value marks the experts whose copy is on its way,
    /// the third the last copy's ticket.
    #[allow(clippy::type_complexity)]
    fn ensure_in(&self, m: &Model, si: usize, l: u64, need: &[u64], promote: bool) -> Result<(Vec<DevBuf>, Vec<bool>, Option<i64>)> {
        let vq = &self.vq[&l];
        let mut c = self.stores[si].lock().unwrap();
        let c = &mut *c;
        if c.vowner.len() < need.len() {
            return Err(Error(format!("the expert store has {} VRAM slots; one layer needs {}", c.vowner.len(), need.len())));
        }
        c.tick += 1;
        c.rmark = None;
        let tick = c.tick;
        if promote {
            for &x in need {
                *c.usage.entry((l, x)).or_insert(0) += 1;
            }
        }
        let mut missing = Vec::new();
        let mut pending: Vec<u64> = Vec::new();
        let mut last: Option<i64> = None;
        let mut reading: HashMap<usize, i64> = HashMap::new();
        for &ex in need {
            match c.loc.get(&(l, ex)).copied() {
                Some(Loc::V(s)) => {
                    c.hits += 1;
                    c.vused[s] = tick;
                    if let Some(tk) = c.prefetched.remove(&(l, ex)) {
                        c.pf_used += 1;
                        pending.push(ex);
                        last = Some(last.map_or(tk, |a| a.max(tk)));
                    }
                }
                Some(Loc::R(r)) if !promote => c.rused[r] = tick,
                _ => missing.push(ex),
            }
        }
        c.prefetched.retain(|k, _| k.0 != l);
        if !missing.is_empty() {
            let order = c.victims(tick, missing.len());
            // the file's experts read ahead (the layer before's guess): waited for, their slots taken as staged
            let mut pre: HashMap<u64, usize> = HashMap::new();
            for x in &missing {
                if c.loc.contains_key(&(l, *x)) {
                    continue;
                }
                if let Some((k, done)) = c.ahead.remove(&(l, *x)) {
                    let ta = Instant::now();
                    done.wait()?;
                    ENS_AHEAD_NS.fetch_add(ta.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    AHEAD_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    c.stage_rd[k] = None;
                    c.stage_held[k] = true;
                    pre.insert(*x, k);
                }
            }
            let rest: Vec<u64> = missing.iter().copied().filter(|x| !pre.contains_key(x)).collect();
            let ts = Instant::now();
            let mut staged = self.stage_reads(m, vq, c, l, &rest, false)?;
            ENS_STAGE_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            staged.extend(pre);
            let tw = Instant::now();
            // the staged ones first: their slots are held until copied up, and a miss beyond them reads into a free one
            let mut missing = missing;
            missing.sort_by_key(|x| !staged.contains_key(x));
            for (ex, &s) in missing.iter().zip(&order) {
                c.misses += 1;
                if let Some(tk) = self.swap_in(m, vq, c, l, *ex, s, &mut reading, &staged)? {
                    last = Some(last.map_or(tk, |a| a.max(tk)));
                    pending.push(*ex);
                }
                c.vused[s] = tick;
            }
            if promote {
                self.keep_spare(c, tick, &reading)?;
            }
            ENS_SWAP_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let sb = c.slot_bytes;
        let slots = need.iter().map(|ex| match c.loc[&(l, *ex)] {
            Loc::V(s) => {
                let (ch, o) = c.vat(s);
                c.vram[ch].view(o, sb)
            }
            Loc::R(r) => {
                c.direct += 1;
                c.rused[r] = tick;
                let (rch, ro) = c.rat(r);
                c.host[rch].device_view(ro, sb)
            }
        }).collect::<Result<Vec<_>>>()?;
        let arriving = need.iter().map(|ex| pending.contains(ex)).collect();
        if !promote {
            if let Some(t) = c.down_any.take() {
                self.ops.await_ticket(t)?;
            }
        }
        Ok((slots, arriving, last))
    }

    /// A prompt pass's experts of `need` (layer l) where they are, nothing moved: a VRAM slot, a pinned host slot, or the
    /// file only
    fn locate(&self, l: u64, need: &[u64]) -> Result<Vec<Src>> {
        let mut out: Vec<Option<Src>> = (0..need.len()).map(|_| None).collect();
        for (si, ks) in self.by_store(l, need)? {
        let mut c = self.stores[si].lock().unwrap();
        let c = &mut *c;
        c.tick += 1;
        c.rmark = None;
        let tick = c.tick;
        let sb = c.slot_bytes;
        for k in ks {
            let ex = need[k];
            out[k] = Some(match c.loc.get(&(l, ex)).copied() {
                Some(Loc::V(s)) => {
                    c.hits += 1;
                    c.vused[s] = tick;
                    let (ch, o) = c.vat(s);
                    Src::V(c.vram[ch].view(o, sb)?)
                }
                Some(Loc::R(r)) => {
                    c.direct += 1;
                    c.rused[r] = tick;
                    let (rch, ro) = c.rat(r);
                    Src::R(c.host[rch].device_view(ro, sb)?)
                }
                None => {
                    c.from_file += 1;
                    Src::F
                }
            });
        }
        // a copy into a VRAM slot or down to a host slot still on its way (decode's swaps) lands before a product reads it
        for t in [c.last_up, c.down_any.take()].into_iter().flatten() {
            self.ops.await_ticket(t)?;
        }
        }
        out.into_iter().map(|v| v.ok_or_else(|| Error("locate: an expert without a place".into()))).collect()
    }

    /// Experts (k, ex) of layer l only the file holds, their reads started on the reader pool into the pinned prompt slots
    /// (each slot's last copy up done first): k -> its slot and its reads' completion, not waited for here - stage_up waits
    /// for an expert just before it copies it up, so a wave starts as soon as its own experts are in. At most as many as
    /// there are slots; the rest are left to stage_up.
    fn start_file_reads(&self, l: u64, files: &[u64]) -> Result<HashMap<u64, (usize, Arc<RDone>)>> {
        let (Some((buf, sb, tks)), Some(rd)) = (&self.pstage, &self.reader) else { return Ok(HashMap::new()) };
        let vq = &self.vq[&l];
        // the slots are two halves, a layer's parity picking its own (the next layer's reads start while this one's are used)
        let total = tks.lock().unwrap().len();
        let (half, base_slot) = (total / 2, (l as usize % 2) * (total / 2));
        let n = files.len().min(half);
        {
            // a slot is read into again only after the copy up that last read it
            let mut t = tks.lock().unwrap();
            for tk in t.iter_mut().skip(base_slot).take(n) {
                if let Some(x) = tk.take() {
                    self.ops.ticket_wait(x)?;
                }
            }
        }
        let base = buf.lock().unwrap().as_mut_slice().as_mut_ptr() as usize;
        let mut out = HashMap::with_capacity(n);
        for (i, ex) in files.iter().take(n).enumerate() {
            let segs = expert_segs(vq, *ex, base + (base_slot + i) * sb);
            let done = Arc::new(RDone::new(segs.len()));
            for seg in segs {
                if rd.tx.lock().unwrap().send(RJob { seg, done: done.clone(), direct: true }).is_err() {
                    return Err(Error("the expert readers ended".into()));
                }
            }
            out.insert(*ex, (base_slot + i, done));
        }
        Ok(out)
    }

    /// Experts `ks` of layer l into the staging ring's slots `slots` (their sources `src`: host slots copied up, the
    /// file's read into pinned staging on threads first), each copy after `free[slot]`: their views, the last copy's ticket
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn stage_up(&self, m: &Model, l: u64, ks: &[usize], need: &[u64], src: &[Src], slots: &[usize], free: &[Option<i64>], read: &HashMap<usize, (usize, Arc<RDone>)>)
                -> Result<(Vec<DevBuf>, Option<i64>)> {
        let (ring, rs) = self.ring.as_ref().ok_or("no staging ring")?;
        let vq = &self.vq[&l];
        let mut last = None;
        let mut views = Vec::with_capacity(ks.len());
        // the file's experts read ahead (start_file_reads): waited for, copied up from their pinned slots
        let mut rest = Vec::new();
        for (&k, &s) in ks.iter().zip(slots) {
            if let (Src::F, Some((i, done)), Some((buf, sb, tks))) = (&src[k], read.get(&k), &self.pstage) {
                let tw = Instant::now();
                done.wait()?;
                FILE_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                let i = *i;
                let span = vq.lay(need[k]).3;
                let v = buf.lock().unwrap().device_view(i * sb, span)?;
                let t = self.ops.stream_copy_on(1, ring, s * rs, &v, 0, span, &[free[s]])?;
                tks.lock().unwrap()[i] = Some(t);
                last = Some(t);
            } else {
                rest.push((k, s));
            }
        }
        // the rest by store (the XQ file's experts in several): the file's read into its staging slots, then copied up
        let rest_x: Vec<u64> = rest.iter().map(|(k, _)| need[*k]).collect();
        for (si, ix) in self.by_store(l, &rest_x)? {
        let mut grp: Vec<(usize, usize)> = ix.iter().map(|&i| rest[i]).collect();
        let files: Vec<u64> = grp.iter().filter(|(k, _)| matches!(src[*k], Src::F)).map(|(k, _)| need[*k]).collect();
        let mut c = self.stores[si].lock().unwrap();
        let c = &mut *c;
        let staged = if files.is_empty() { HashMap::new() } else { self.stage_reads(m, vq, c, l, &files, true)? };
        let sb = c.slot_bytes;
        // the staged ones first (their slots held until copied up)
        grp.sort_by_key(|(k, _)| !staged.contains_key(&need[*k]));
        for &(k, s) in &grp {
            let span = vq.lay(need[k]).3;
            let from = match &src[k] {
                Src::R(h) => h.view(0, span)?,
                Src::F => {
                    let st = match staged.get(&need[k]) {
                        Some(&st) => st,
                        None => {
                            let st = self.stage_slot(c)?;
                            read_expert(m, vq, need[k], &mut c.stage.as_mut_slice()[st * sb..(st + 1) * sb])?;
                            st
                        }
                    };
                    let v = c.stage.device_view(st * sb, span)?;
                    let t = self.ops.stream_copy_on(1, ring, s * rs, &v, 0, span, &[free[s]])?;
                    c.stage_tk[st] = Some(t);
                    c.stage_held[st] = false;
                    last = Some(t);
                    continue;
                }
                Src::V(_) => return Err(Error("stage_up: an expert in VRAM".into())),
            };
            let t = self.ops.stream_copy_on(1, ring, s * rs, &from, 0, span, &[free[s]])?;
            last = Some(t);
        }
        }
        for &s in slots {
            views.push(ring.view(s * rs, *rs)?);
        }
        Ok((views, last))
    }

    /// Expert (l, ex) into VRAM slot s: the slot's expert down to a free host slot, the wanted one up from its host
    /// slot or the file (read on this thread: the queue first drained of the copies that could touch the slot)
    #[allow(clippy::too_many_arguments)]
    fn swap_in(&self, m: &Model, vq: &LayerVq, c: &mut Store, l: u64, ex: u64, s: usize, reading: &mut HashMap<usize, i64>, staged: &HashMap<u64, usize>)
               -> Result<Option<i64>> {
        let key = (l, ex);
        let (vch, vo) = c.vat(s);
        let sb = c.slot_bytes;
        let from = c.loc.get(&key).copied();
        let fill = c.vfill.get(&s).copied();
        let mut down: Option<i64> = None;
        let mut zero = c.vzero.remove(&s);
        if let Some(v) = c.vowner[s].take() {
            zero = Some(self.tab_zero(c, v)?);
            c.loc.remove(&v);
            c.prefetched.remove(&v);
            if let Some((f, gone)) = self.host_slot(c)? {
                let (rch, ro) = c.rat(f);
                let t = self.ops.stream_copy_on(0, &c.host[rch].device_view(ro, sb)?, 0, &c.vram[vch], vo, sb, &[reading.get(&f).copied(), c.rread.get(&f).copied(), fill, gone])?;
                down = Some(t);
                c.rwrite.insert(f, t);
                c.down_any = Some(t);
                c.loc.insert(v, Loc::R(f));
                c.rowner[f] = Some(v);
                c.rused[f] = c.tick;
            }
        }
        let up = match from {
            Some(Loc::R(r)) => {
                c.from_host += 1;
                let (rch, ro) = c.rat(r);
                let deps = [down, c.rwrite.get(&r).copied(), zero];
                let host = c.host[rch].device_view(ro, sb)?;
                let t = self.ops.stream_copy_on(1, &c.vram[vch], vo, &host, 0, vq.lay(ex).3, &deps)?;
                reading.insert(r, t);
                c.rread.insert(r, t);
                c.rfree.push(r);
                c.rowner[r] = None;
                c.vfill.insert(s, t);
                c.last_up = Some(t);
                Some(t)
            }
            _ => {
                c.from_file += 1;
                let k = match staged.get(&ex) {
                    Some(&k) => k,
                    None => {
                        let k = self.stage_slot(c)?;
                        read_expert(m, vq, ex, &mut c.stage.as_mut_slice()[k * sb..(k + 1) * sb])?;
                        k
                    }
                };
                let t = self.ops.stream_copy_on(1, &c.vram[vch], vo, &c.stage.device_view(k * sb, sb)?, 0, vq.lay(ex).3, &[down, fill, zero])?;
                c.stage_tk[k] = Some(t);
                c.stage_held[k] = false;
                c.vfill.insert(s, t);
                c.last_up = Some(t);
                Some(t)
            }
        };
        c.vowner[s] = Some(key);
        if let Some(t) = up {
            self.tab_set_after(c, key, s, t)?;
        }
        c.loc.insert(key, Loc::V(s));
        Ok(up)
    }

    /// A host slot for an expert moved down from VRAM: a free one, else (NS_DS41_HOST_LRU) the least recently used
    /// one's expert given back to the file (the loaded tier's least profiled first), with the ticket the copy into it
    /// waits for (products reading it in place)
    fn host_slot(&self, c: &mut Store) -> Result<Option<(usize, Option<i64>)>> {
        if let Some(f) = c.rfree.pop() {
            return Ok(Some((f, None)));
        }
        if !host_lru() {
            return Ok(None);
        }
        let tick = c.tick;
        let Some(f) = (0..c.rowner.len()).filter(|&i| c.rowner[i].is_some() && c.rused[i] != tick)
            .min_by_key(|&i| (c.rused[i], std::cmp::Reverse(i))) else { return Ok(None) };
        let v = c.rowner[f].take().unwrap();
        c.loc.remove(&v);
        if c.rmark.is_none() {
            c.rmark = Some(self.ops.mark()?);
        }
        Ok(Some((f, c.rmark)))
    }

    /// Decode's spare VRAM slots: an expert swapped in takes an empty slot, so its copy up waits for no copy down; the
    /// least recently used experts are moved down to host slots afterwards, off the products' path, until NS_DS41_SPARE
    /// slots are empty again (the next copy up into one waits for its copy down)
    fn keep_spare(&self, c: &mut Store, tick: u64, reading: &HashMap<usize, i64>) -> Result<()> {
        let want = spare_slots();
        if want == 0 {
            return Ok(());
        }
        let empty = (0..c.vowner.len()).filter(|&i| c.usable(i) && c.vowner[i].is_none()).count();
        if empty >= want {
            return Ok(());
        }
        let mut owned: Vec<usize> = (0..c.vowner.len()).filter(|&i| c.vused[i] != tick && c.usable(i) && c.vowner[i].is_some()).collect();
        let k = (want - empty).min(owned.len());
        if k == 0 {
            return Ok(());
        }
        owned.select_nth_unstable_by_key(k - 1, |&i| (c.vused[i], i));
        let sb = c.slot_bytes;
        for &sl in &owned[..k] {
            let Some(v) = c.vowner[sl].take() else { continue };
            let z = self.tab_zero(c, v)?;
            c.vzero.insert(sl, z);
            c.loc.remove(&v);
            c.prefetched.remove(&v);
            let (vch, vo) = c.vat(sl);
            if let Some((f, gone)) = self.host_slot(c)? {
                let (rch, ro) = c.rat(f);
                let t = self.ops.stream_copy_on(0, &c.host[rch].device_view(ro, sb)?, 0, &c.vram[vch], vo, sb,
                                                &[reading.get(&f).copied(), c.rread.get(&f).copied(), c.vfill.get(&sl).copied(), gone])?;
                c.rwrite.insert(f, t);
                c.down_any = Some(t);
                c.vfill.insert(sl, t);
                c.loc.insert(v, Loc::R(f));
                c.rowner[f] = Some(v);
                c.rused[f] = c.tick;
            }
        }
        Ok(())
    }

    /// An expert leaves VRAM: its slot table entry zeroed on the compute queue (a decode token routed after it does not
    /// read the slot); the copy up that refills the slot waits for this ticket (and so for the products before it)
    fn tab_zero(&self, c: &Store, key: (u64, u64)) -> Result<i64> {
        let at = (key.0 as usize * c.n_expert + key.1 as usize) * 8;
        self.ops.kcopy(&self.devtab, at, &self.zeros, 0, 8)
    }

    /// An expert arrives in slot s: its slot table entry set on the copy lane after its copy up (ticket `after`), so the
    /// table names it only once it is there
    fn tab_set_after(&self, c: &Store, key: (u64, u64), s: usize, after: i64) -> Result<()> {
        let Some(sa) = &c.slotaddr else { return Ok(()) };
        let at = (key.0 as usize * c.n_expert + key.1 as usize) * 8;
        self.ops.stream_copy_on(1, &self.devtab, at, sa, s * 8, 8, &[Some(after)])?;
        Ok(())
    }

    /// A free staging slot (the copy last reading it done)
    fn stage_slot(&self, c: &mut Store) -> Result<usize> {
        Ok(self.stage_slots(c, 1)?[0])
    }

    /// n different staging slots, each free (the copy last reading it done) and not held (an expert in it still to be
    /// copied up this step: taking it would hand that expert another's bytes - with XQ's mixed types, NaN): from the next
    /// in turn, first the ones no read ahead holds or whose read ahead has ended, then the others (their reads ahead
    /// waited for and forgotten) - taking slots strictly in turn waited on reads ahead nobody wanted, 3-5 ms a token
    fn stage_slots(&self, c: &mut Store, n: usize) -> Result<Vec<usize>> {
        let t0 = Instant::now();
        let order: Vec<usize> = (0..STAGE).map(|i| (c.stage_next + i) % STAGE).filter(|&k| !c.stage_held[k]).collect();
        let idle = |k: usize| c.stage_rd[k].as_ref().is_none_or(|r| r.is_done());
        let mut ks: Vec<usize> = order.iter().copied().filter(|&k| idle(k)).collect();
        ks.extend(order.iter().copied().filter(|&k| !idle(k)));
        if ks.len() < n {
            return Err(Error(format!("{n} staging slots wanted, {} free", ks.len())));
        }
        ks.truncate(n);
        for &k in &ks {
            if let Some(t) = c.stage_tk[k].take() {
                self.ops.ticket_wait(t)?;
            }
            if let Some(r) = c.stage_rd[k].take() {
                // a read ahead into it not taken: let it end, forget it
                let _ = r.wait();
                c.ahead.retain(|_, v| v.0 != k);
            }
        }
        if let Some(&k) = ks.last() {
            c.stage_next = k + 1;
        }
        ENS_SLOT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(ks)
    }

    /// Decode: the file's experts among `want` (layer l's likely ones) read ahead into staging slots, at most `limit`
    fn read_ahead(&self, l: u64, want: &[u64], limit: usize) -> Result<usize> {
        let Ok(groups) = self.by_store(l, want) else { return Ok(0) };
        let mut n = 0;
        for (si, ks) in groups {
            if n >= limit {
                break;
            }
            let sub: Vec<u64> = ks.iter().map(|&k| want[k]).collect();
            n += self.read_ahead_in(si, l, &sub, limit - n)?;
        }
        Ok(n)
    }
    fn read_ahead_in(&self, si: usize, l: u64, want: &[u64], limit: usize) -> Result<usize> {
        let Some(rd) = &self.reader else { return Ok(0) };
        let vq = &self.vq[&l];
        let mut c = self.stores[si].lock().unwrap();
        let c = &mut *c;
        let xs: Vec<u64> = want.iter().copied().filter(|x| !c.loc.contains_key(&(l, *x)) && !c.ahead.contains_key(&(l, *x))).take(limit).collect();
        let sb = c.slot_bytes;
        let mut n = 0;
        for x in xs {
            let k = self.stage_slot(c)?;
            let dst = c.stage.as_mut_slice().as_mut_ptr() as usize + k * sb;
            let segs = expert_segs(vq, x, dst);
            let done = Arc::new(RDone::new(segs.len()));
            for seg in segs {
                if rd.tx.lock().unwrap().send(RJob { seg, done: done.clone(), direct: false }).is_err() {
                    return Err(Error("the expert readers ended".into()));
                }
            }
            c.stage_rd[k] = Some(done.clone());
            c.ahead.insert((l, x), (k, done));
            n += 1;
        }
        Ok(n)
    }

    /// The experts of `missing` held by neither VRAM nor host memory read from the file into staging slots, up to
    /// STAGE of them at once on threads of their own: expert -> its slot
    fn stage_reads(&self, m: &Model, vq: &LayerVq, c: &mut Store, l: u64, missing: &[u64], prompt: bool) -> Result<HashMap<u64, usize>> {
        let free = c.stage_held.iter().filter(|h| !**h).count();
        let file: Vec<u64> = missing.iter().copied().filter(|x| !c.loc.contains_key(&(l, *x))).take(free).collect();
        let mut at = HashMap::new();
        if file.is_empty() {
            return Ok(at);
        }
        let t0 = Instant::now();
        let ks = self.stage_slots(c, file.len())?;
        STAGED_READS.fetch_add(file.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let sb = c.slot_bytes;
        let mut slices: Vec<Option<&mut [u8]>> = c.stage.as_mut_slice().chunks_mut(sb).map(Some).collect();
        let jobs: Vec<(u64, &mut [u8])> = file.iter().zip(&ks).map(|(x, k)| (*x, slices[*k].take().unwrap())).collect();
        let failed: Mutex<Option<String>> = Mutex::new(None);
        std::thread::scope(|sc| {
            for (x, buf) in jobs {
                let failed = &failed;
                sc.spawn(move || {
                    if let Err(e) = read_expert_par(m, vq, x, buf, prompt && prompt_odirect()) {
                        *failed.lock().unwrap() = Some(e.0);
                    }
                });
            }
        });
        if let Some(x) = failed.into_inner().unwrap() {
            return Err(Error(format!("reading experts: {x}")));
        }
        for (x, k) in file.iter().zip(ks) {
            c.stage_held[k] = true;
            at.insert(*x, k);
        }
        FILE_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(at)
    }

    /// Decode: the experts layer l is expected to want, swapped in now (only ones in host memory; at most `limit`)
    fn prefetch(&self, m: &Model, l: u64, want: &[u64], limit: usize) -> Result<()> {
        let Ok(groups) = self.by_store(l, want) else { return Ok(()) };
        let mut n = 0;
        for (si, ks) in groups {
            if n >= limit {
                break;
            }
            let sub: Vec<u64> = ks.iter().map(|&k| want[k]).collect();
            n += self.prefetch_in(m, si, l, &sub, limit - n)?;
        }
        Ok(())
    }
    fn prefetch_in(&self, m: &Model, si: usize, l: u64, want: &[u64], limit: usize) -> Result<usize> {
        let vq = &self.vq[&l];
        let mut c = self.stores[si].lock().unwrap();
        let c = &mut *c;
        let tick = c.tick;
        let cand: Vec<u64> = want.iter().copied().filter(|ex| matches!(c.loc.get(&(l, *ex)), Some(Loc::R(_)))).take(limit).collect();
        if cand.is_empty() {
            return Ok(0);
        }
        let order = c.victims(tick, cand.len());
        let mut reading: HashMap<usize, i64> = HashMap::new();
        let none = HashMap::new();
        for (ex, &s) in cand.iter().zip(&order) {
            if let Some(tk) = self.swap_in(m, vq, c, l, *ex, s, &mut reading, &none)? {
                c.prefetched.insert((l, *ex), tk);
                c.pf_issued += 1;
            }
            c.vused[s] = tick;
        }
        self.keep_spare(c, tick, &reading)?;
        Ok(cand.len())
    }
}

/// A source layer's compressed keys: the cache (fp16 [ctx / ratio + 1][512], normed, rotated, fp4-simulated), at
/// ratio 2 the ring of its last tokens' projections, and the indexer's keys (fp16 [ctx / ratio + 1][128])
struct Comp {
    ratio: usize,
    cache: DevBuf,
    ring: Option<DevBuf>,
    idx: DevBuf,
}

/// A layer's attention state: the raw keys' ring (fp16 [raw_ring()][512]), and at a kv source its compressor's
struct Att {
    raw: DevBuf,
    comp: Option<Comp>,
    /// a kv source whose readers continue on the next GPU (a split past it, at an index source): its compressed and
    /// indexer keys copied there each pass
    mirror: Option<Comp>,
}

/// the compressor's ring of a ratio-2 source: the block being built, and a verify pass rolled back
fn comp_ring(ratio: usize) -> usize {
    ratio + MAX_VERIFY
}

/// The raw keys' ring: a chunk and the window before it, and a verify pass rolled back
pub fn raw_ring() -> usize {
    prefill_chunk() + 128 + MAX_VERIFY
}

/// The towers' side of a conversation: each tower's window of main_x keys (fp16 [128][512], slot pos % 128), the last
/// pass's rows' stream means at the target layers ([3][MTP_KEEP][d]) and the ring slots the last pass overwrote
pub struct MtpS {
    rings: Vec<DevBuf>,
    mh: DevBuf,
    /// (first position, rows) of the last pass's rows in `mh`
    mh_rows: (usize, usize),
    /// the ring slots a verify pass's push overwrote: [tower][MAX_VERIFY][512] fp16, from position snap_pos
    snap: DevBuf,
    snap_at: Option<(usize, usize)>,
}

/// A conversation's state: per layer its attention state (on its layer's GPU); `pos` tokens so far, all of them in
/// `hist` (the Engram hashes the 3 before a token)
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<Att>,
    hist: Vec<u32>,
    /// where this turn's generated tokens begin in `hist` (the end of the last prompt read): DRY's history
    gen_from: usize,
    /// (first position, rows) of the last pass of 2..=MAX_VERIFY rows (a rollback's bounds)
    snapped: Option<(usize, usize)>,
    mtp: Option<MtpS>,
}

/// A conversation's state at a position, in host memory
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

impl Checkpoint {
    /// The checkpoint as a file: "NSD41", the position, the buffers
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(b"NSD41")?;
        u(w, self.pos)?;
        u(w, self.bufs.len())?;
        for b in &self.bufs {
            u(w, b.len())?;
            w.write_all(b)?;
        }
        Ok(())
    }

    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Checkpoint> {
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "not a checkpoint file");
        let u = |r: &mut dyn std::io::Read| -> std::io::Result<usize> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b) as usize)
        };
        let mut magic = [0u8; 5];
        r.read_exact(&mut magic)?;
        if &magic != b"NSD41" {
            return Err(bad());
        }
        let pos = u(r)?;
        let n = u(r)?;
        let mut bufs = Vec::with_capacity(n);
        let mut bytes = 0;
        for _ in 0..n {
            let len = u(r)?;
            if len > 1 << 34 {
                return Err(bad());
            }
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            bytes += len;
            bufs.push(v);
        }
        Ok(Checkpoint { pos, bufs, bytes })
    }
}

/// Generation from a fed prompt: `Ds41::step` commits the next token(s) each call. The last committed token is fed at
/// the start of the following call.
pub struct Decoder {
    logits: Vec<f32>,
    next: Option<u32>,
    /// the towers' drafts for the positions after `next`
    drafts: Vec<u32>,
    mtp: bool,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn pending(&mut self) -> Option<u32> {
        self.drafts.clear();
        self.next.take()
    }
}

pub struct Ds41<'g> {
    pub m: Model<'g>,
    pub parts: Vec<Part>,
    /// layer -> part
    owner: Vec<usize>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    pub engram: Option<Engram>,
    prof: Option<Mutex<BTreeMap<&'static str, (f64, u64)>>>,
    /// the towers are loaded (on the last part)
    pub mtp: bool,
    /// the kv source mirrored on the second GPU (the parts split at an index source after it)
    mirror_src: Option<u64>,
    /// a pipelined prompt's candidate blocks for the second GPU, a chunk's at a time in order (the first stage runs a
    /// chunk ahead: its blocks must not land while the second GPU still reads the chunk before's)
    cand_q: Mutex<std::collections::VecDeque<Vec<u8>>>,
    pipelining: std::sync::atomic::AtomicBool,
    /// the vision sidecar (NS_DS41_VISION), when there is one
    pub vision: Option<crate::vision::Vision>,
    /// DRY against repeated runs (NS_DS41_DRY) and the tokens that break a run
    dry: Option<(crate::dry::Dry, crate::dry::Vocab)>,
}

/// the decode products' kernel variants (ds41.h: ns_ds41_vq_mmv): 0 a sub-group a row (float32 codebook), 1 a lane a row
/// (the E4M3 bytes), 2 a lane a row (fp16 codebook), 3 a sub-group a row (fp16), 4 a lane a row (the bytes two at once),
/// 5 by lookup table (each group of 8 values of x against every codebook entry, in local memory)
const VQ_VARIANTS: usize = 6;

/// The widened codebook variant `var` reads: float32 (0), fp16 (2, 3), none (the bytes)
fn vq_wide<'a>(wide: (&'a HashMap<u64, DevBuf>, &'a HashMap<u64, DevBuf>), l: u64, var: i32) -> Option<&'a DevBuf> {
    match var {
        0 => wide.0.get(&l),
        2 | 3 => wide.1.get(&l),
        _ => None,
    }
}

/// The decode products' fastest kernel variant on this GPU for each index width of its layers, [gate | up, down]: both
/// timed on six of a layer's experts in VRAM, one token (NS_DS41_VQ_V=0..4 sets it instead). Which wins depends on the
/// GPU and the shape (B70 + B65: 3 or 4 for gate | up, 2 or 4 for down).
#[allow(clippy::too_many_arguments)]
fn vq_pick(o: &Ops, g: &crate::model::Geometry, vq: &HashMap<u64, LayerVq>, cbs: &HashMap<u64, DevBuf>, wide: (&HashMap<u64, DevBuf>, &HashMap<u64, DevBuf>),
           stores: &[Mutex<Store>],
           store_of: &HashMap<(u64, usize), usize>, draft: &HashMap<u64, DevBuf>, mtp: bool, log: &mut dyn FnMut(String)) -> Result<HashMap<usize, [i32; 2]>> {
    let mut out = HashMap::new();
    if let Some(v) = std::env::var("NS_DS41_VQ_V").ok().and_then(|v| v.parse::<i32>().ok()) {
        for l in vq.values().filter(|l| l.std.is_none()) {
            out.insert(l.nbit, [v, v]);
        }
        return Ok(out);
    }
    let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
    let mut layers: Vec<u64> = vq.keys().copied().collect();
    layers.sort();
    for l in layers {
        let v = &vq[&l];
        if v.std.is_some() || out.contains_key(&v.nbit) {
            continue;
        }
        // six of the layer's experts in VRAM: the store's slots it owns, or the towers' buffer
        let bases: Vec<u64> = match (draft.get(&l), store_of.get(&(l, v.span.next_multiple_of(256)))) {
            (Some(buf), _) => (0..6).map(|x| buf.ptr() as u64 + (x * v.span.next_multiple_of(256)) as u64).collect(),
            (None, Some(&si)) => {
                let c = stores[si].lock().unwrap();
                (0..c.base).filter(|s| c.vowner[*s].is_some_and(|k| k.0 == l)).take(6).map(|s| {
                    let (ch, at) = c.vat(s);
                    c.vram[ch].ptr() as u64 + at as u64
                }).collect()
            }
            _ => Vec::new(),
        };
        if bases.len() < 6 {
            continue;
        }
        let n = bases.len();
        let mut tb: Vec<u8> = Vec::new();
        let mut meta: Vec<i32> = Vec::new();
        for w in 0..2 {
            for (x, b) in bases.iter().enumerate() {
                tb.extend((b + if w == 0 { v.gate } else { v.up } as u64).to_le_bytes());
                meta.extend([1, (w * n + x) as i32, 0, 0, 0, 0, 0, 0, 0, 0]);
            }
        }
        tb.extend(i32s(&meta));
        let (gl, gmax) = crate::ops::lut_lists(1, (0..2 * n).map(|e| (0, e, 0))).ok_or("vq_pick: the lists")?;
        tb.extend(gl);
        let mut db: Vec<u8> = bases.iter().flat_map(|b| (b + v.down as u64).to_le_bytes()).collect();
        db.extend(i32s(&(0..n).flat_map(|x| [1, x as i32, x as i32, 0, 0, 0, 0, 0, 0, 0]).collect::<Vec<_>>()));
        let (dl, dmax) = crate::ops::lut_lists(n, (0..n).map(|e| (e, e, 0))).ok_or("vq_pick: the lists")?;
        db.extend(dl);
        let (gt, dt) = (DevBuf::new(&o.gpu, tb.len())?, DevBuf::new(&o.gpu, db.len())?);
        gt.write(0, &tb)?;
        dt.write(0, &db)?;
        let x = DevBuf::f32(&o.gpu, n * f.max(d))?;
        x.fill(0)?;
        let y = DevBuf::f32(&o.gpu, 2 * n * f.max(d))?;
        let mut pick = [0i32; 2];
        for (k, (tab, e, rows, cols, lut)) in [(&gt, 2 * n, f, d, (1, gmax)), (&dt, n, d, f, (n, dmax))].into_iter().enumerate() {
            let run = |var: i32, reps: usize| -> Result<f64> {
                let t0 = Instant::now();
                for _ in 0..reps {
                    o.vq_mmv(tab, e, (&cbs[&l], vq_wide(wide, l, var)), &x, (&y, 0), e, rows, cols, v.nbit, var, Some(lut))?;
                }
                o.gpu.sync()?;
                Ok(t0.elapsed().as_secs_f64() * 1e6 / reps as f64)
            };
            // the clocks up first, then each twice in turn
            let t0 = Instant::now();
            while t0.elapsed().as_secs_f64() < 0.15 {
                run(0, 10)?;
            }
            let mut best = [f64::MAX; VQ_VARIANTS];
            // not with the draft towers: the table is built a column at a time (a verify pass's rows cost each its own),
            // and one-token passes must use the verify passes' kernel (their rows bit-identical)
            let lut = !mtp && std::env::var("NS_DS41_VQ_LUT").map_or(true, |v| v != "0");
            for _ in 0..2 {
                for (var, b) in best.iter_mut().enumerate().filter(|(v, _)| lut || *v as i32 != crate::ops::VQ_LUT) {
                    *b = b.min(run(var as i32, 20)?);
                }
            }
            pick[k] = (0..VQ_VARIANTS).min_by(|a, b| best[*a].total_cmp(&best[*b])).unwrap_or(0) as i32;
            log(format!("{}: {}-bit VQ {} products: {} us (variants 0-{}) - taking {}", o.gpu.name, v.nbit, ["gate | up", "down"][k],
                        best.iter().map(|b| format!("{b:.0}")).collect::<Vec<_>>().join(" / "), VQ_VARIANTS - 1, pick[k]));
        }
        out.insert(v.nbit, pick);
    }
    Ok(out)
}

/// the small tensors kept as float32 vectors
const VECTORS: [Role; 15] = [Role::OutputNorm, Role::AttnNorm, Role::FfnNorm, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnBase, Role::HcFfnScale,
                             Role::QANorm, Role::KvNorm, Role::Sinks, Role::CompNorm, Role::IdxKNorm, Role::RouterBias, Role::MtpMainNorm, Role::MtpOutNorm];

impl Part {
    /// The part's layers onto its GPU (the head and the towers when `last`), its expert stores and pinned mirror.
    #[allow(clippy::too_many_arguments)]
    fn load(m: &Model, gpu: &Arc<Gpu>, layers: Range<u64>, last: bool, mtp: bool, expert_bytes: Option<usize>, mirror_bytes: usize, kv_reserve: usize,
            log: &mut dyn FnMut(String)) -> Result<Part> {
        let g = &m.g;
        let ops = Ops::new(gpu.clone())?;
        let scratch = DevBuf::f32(gpu, SCRATCH)?;
        let mut mats = BTreeMap::new();
        let mut vecs = BTreeMap::new();
        let mut bytes = 0u64;
        let towers: Vec<u64> = if last && mtp { (g.n_layer..g.n_layer + g.n_mtp).collect() } else { Vec::new() };
        let mut todo: Vec<(u64, Vec<Role>)> = Vec::new();
        if last {
            let mut v = vec![Role::OutputNorm, Role::Output];
            if mtp {
                v.extend([Role::MtpMainProj, Role::MtpMainNorm, Role::MtpMarkovEmbd, Role::MtpMarkovHead, Role::MtpConfidence, Role::MtpOutNorm]);
            }
            todo.push((0, v));
        }
        todo.extend(layers.clone().chain(towers.iter().copied()).map(|l| (l, m.roles(l))));
        for (l, roles) in todo {
            for r in roles {
                if matches!(r, Role::Experts | Role::ExpGate | Role::ExpUp | Role::ExpDown | Role::TokenEmbd | Role::EngramK) || (r == Role::EngramQ && (!g.std || g.xq)) {
                    continue; // the experts through the stores; the embedding's rows per token; q x k below
                }
                let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                let n = t.elements() as usize;
                let cols = *t.shape.last().unwrap_or(&1) as usize;
                let to_q4k = dense_q4k() && n.is_multiple_of(256) && cols.is_multiple_of(256)
                    && match t.ty {
                        GType::Q8_0 => dense_q4k_role(r),
                        // the shared Q6_K product is slow: a Q6_K one of the eight always
                        GType::Q6K => matches!(r, Role::QA | Role::QB | Role::Kv | Role::OutA | Role::OutB | Role::ShGate | Role::ShUp | Role::ShDown),
                        // the XQ file's Engram projection: as float32, 630 MB read a token at each Engram layer
                        GType::F16 => r == Role::EngramWkv,
                        _ => false,
                    };
                let (raw, ty) = if to_q4k {
                    let from = match t.ty {
                        GType::Q6K => crate::requant::From::Q6K,
                        GType::F16 => crate::requant::From::F16,
                        _ => crate::requant::From::Q8_0,
                    };
                    let q = q4k_of(m, t, n, from)?;
                    let raw = DevBuf::new(gpu, q.len())?;
                    raw.write(0, &q)?;
                    bytes += q.len() as u64;
                    (raw, GType::Q4K)
                } else {
                    let host = m.file.read(t).map_err(e)?;
                    let raw = DevBuf::new(gpu, t.bytes as usize)?;
                    raw.write(0, &host)?;
                    bytes += t.bytes;
                    (raw, t.ty)
                };
                if VECTORS.contains(&r) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    vecs.insert((l, r), f);
                } else if !to_q4k && matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    mats.insert((l, r), Mat { buf: f, ty: GType::F32, rows: n / cols, cols });
                } else if matches!(ty, GType::Q4K | GType::Fp8B32) || ops.mmvq_supported(ty.code()) {
                    mats.insert((l, r), Mat { buf: raw, ty, rows: n / cols, cols });
                } else {
                    return Err(Error(format!("{}: type {} is not one this engine multiplies", t.name, t.ty.name())));
                }
            }
            // the Engram's q x k (only ever used as their product)
            if g.is_engram(l) && layers.contains(&l) {
                let rd = |r: Role| -> Result<Vec<f32>> {
                    let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                    Ok(m.file.read(t).map_err(e)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
                };
                // the llama.cpp form stores the product already (engram_gate)
                let qk = if g.std && !g.xq {
                    rd(Role::EngramQ)?
                } else {
                    let (q, k) = (rd(Role::EngramQ)?, rd(Role::EngramK)?);
                    q.iter().zip(&k).map(|(a, b)| a * b).collect::<Vec<f32>>()
                };
                vecs.insert((l, Role::EngramQ), DevBuf::from_f32(gpu, &qk)?);
            }
        }
        gpu.sync()?;
        let mut groups = BTreeMap::new();
        for l in layers.clone().chain(towers.iter().copied()) {
            let w = mats.get(&(l, Role::OutA)).ok_or_else(|| Error(format!("block {l}: o_a missing")))?;
            if w.ty != GType::Fp8B32 {
                let gs = (0..g.o_groups as usize).map(|gi| w.rows_view(gi * g.o_lora as usize, g.o_lora as usize)).collect::<Result<Vec<_>>>()?;
                groups.insert(l, gs);
            }
        }
        let q8 = DevBuf::new(gpu, ops.q8_1_bytes(4 * g.n_embd as usize, MMVQ_COLS))?;
        // the experts: each layer's layout and codebook
        let mut vq = HashMap::new();
        let mut cbs = HashMap::new();
        let mut cbf = HashMap::new();
        let mut cbh = HashMap::new();
        for l in layers.clone().chain(towers.iter().copied()) {
            let v = layer_vq(m, l)?;
            if let Some(s) = &v.std {
                let types: Vec<(GType, u64)> = match &s.xq {
                    Some(q) => q.ex.iter().flat_map(|&(gt, _, dt, _)| [(XQ_TYPES[gt as usize], g.n_embd), (XQ_TYPES[dt as usize], g.ffn_expert)]).collect(),
                    None => vec![(s.ty[0], g.n_embd), (s.ty[2], g.ffn_expert)],
                };
                if !types.iter().all(|(t, c)| ops.std_supported(t.code(), *c as usize)) {
                    return Err(Error(format!("block {l}: experts of {} | {} are not taken by the kernels", s.ty[0].name(), s.ty[2].name())));
                }
                vq.insert(l, v);
                continue;
            }
            if !ops.vq_supported(g.ffn_expert as usize, g.n_embd as usize, v.nbit) {
                return Err(Error(format!("block {l}: {}-bit VQ experts are not taken by the kernels", v.nbit)));
            }
            let cb = DevBuf::new(gpu, v.cb_bytes.len())?;
            cb.write(0, &v.cb_bytes)?;
            let f = DevBuf::f32(gpu, v.cb_bytes.len())?;
            ops.vq_codebook_f32(&cb, v.cb_bytes.len(), &f)?;
            cbf.insert(l, f);
            let h = DevBuf::new(gpu, v.cb_bytes.len() * 2)?;
            ops.vq_codebook_f16(&cb, v.cb_bytes.len(), &h)?;
            cbh.insert(l, h);
            cbs.insert(l, cb);
            vq.insert(l, v);
        }
        // the towers' experts: all of them, in VRAM
        let mut draft = HashMap::new();
        let mut draft_bytes = 0;
        for &l in &towers {
            let v = &vq[&l];
            let span = v.span.next_multiple_of(256);
            let n_ex = if v.std.is_some() { g.mtp_expert as usize } else { v.starts.len() };
            let buf = DevBuf::new(gpu, span * n_ex)?;
            let mut host = vec![0u8; v.span];
            for x in 0..n_ex as u64 {
                read_expert(m, v, x, &mut host)?;
                buf.write(x as usize * span, &host)?;
            }
            draft_bytes += buf.len;
            draft.insert(l, buf);
        }
        if !towers.is_empty() {
            log(format!("{}: the {} draft towers' experts in VRAM ({:.2} GiB)", gpu.name, towers.len(), gib(draft_bytes)));
        }
        // the stores: one a slot size (LayerVq::slot_of: a layer's span, or for the XQ file each expert's own), VRAM and
        // the mirror shared by their experts' bytes. Without a profile, every layer the same share of VRAM, host memory
        // and the file (expert-major order: in layer order the last layers were the file's alone - a prompt pass read all
        // of their experts from the file)
        let by_layer = std::env::var("NS_DS41_PLACE").is_ok_and(|v| v == "layer");
        let keys: Vec<(u64, u64)> = if by_layer {
            layers.clone().flat_map(|l| (0..g.n_expert).map(move |x| (l, x))).collect()
        } else {
            (0..g.n_expert).flat_map(|x| layers.clone().map(move |l| (l, x))).collect()
        };
        let mut classes: BTreeMap<usize, Vec<(u64, u64)>> = BTreeMap::new();
        for k in keys {
            classes.entry(vq[&k.0].slot_of(k.1)).or_default().push(k);
        }
        let total_bytes: usize = classes.iter().map(|(sb, ks)| sb * ks.len()).sum::<usize>().max(1);
        let ring_slot = classes.keys().copied().max().unwrap_or(0);
        let ring_bytes = prompt_ring() * ring_slot;
        let budget = match expert_bytes {
            Some(b) => b,
            None => gpu.memory()?.1.map_or(8usize << 30, |f| {
                let pass = (900 << 20) + prefill_chunk() * (400 << 10);
                (f as usize).saturating_sub(pass + arena_bytes() + SMALL_ARENA + kv_reserve + keep_free() + ring_bytes + (512 << 20))
            }),
        };
        // with several stores and a profile, each gets the slots its experts take in the profile's ranking over all of
        // them (VRAM first, then host memory); without, each the share of its bytes
        let ranked: Option<HashMap<usize, (usize, usize)>> = expert_profile().filter(|_| classes.len() > 1).map(|prof| {
            let mut all: Vec<(usize, (u64, u64))> = classes.iter().flat_map(|(sb, ks)| ks.iter().map(move |k| (*sb, *k))).collect();
            all.sort_by_key(|(_, k)| std::cmp::Reverse(prof.get(k).copied().unwrap_or(0.0) as u64));
            let (mut out, mut used): (HashMap<usize, (usize, usize)>, usize) = (HashMap::new(), 0);
            for (sb, _) in all {
                used += sb;
                let e = out.entry(sb).or_insert((0, 0));
                if used <= budget {
                    e.0 += 1;
                } else if used <= budget + mirror_bytes {
                    e.1 += 1;
                } else {
                    break;
                }
            }
            out
        });
        let mut stores = Vec::new();
        let mut store_of = HashMap::new();
        let (mut nv_all, mut nr_all) = (0, 0);
        let t0 = Instant::now();
        for (slot_bytes, ks) in &classes {
            let slot_bytes = *slot_bytes;
            let n_exp = ks.len();
            let share = |b: usize| (b as u128 * (slot_bytes * n_exp) as u128 / total_bytes as u128) as usize;
            let per_chunk = (CHUNK / slot_bytes).max(1);
            let (nv, nr) = match ranked.as_ref() {
                Some(r) => r.get(&slot_bytes).copied().unwrap_or((0, 0)),
                None => (share(budget) / slot_bytes, share(mirror_bytes) / slot_bytes),
            };
            // a decode pass of 8 rows can want 48 of a layer's experts from one store
            let nv = nv.max(64).min(n_exp);
            let nr = if n_exp > nv { nr.min(n_exp - nv + 1) } else { 0 };
            let mut ls: Vec<u64> = ks.iter().map(|k| k.0).collect();
            ls.sort_unstable();
            ls.dedup();
            let mut vram = Vec::new();
            let mut have = 0;
            while have < nv {
                let k = per_chunk.min(nv - have);
                vram.push(DevBuf::new(gpu, k * slot_bytes)?);
                have += k;
            }
            let mut host = Vec::new();
            let mut have = 0;
            while have < nr {
                let k = per_chunk.min(nr - have);
                host.push(HostBuf::new(gpu, k * slot_bytes)?);
                have += k;
            }
            let mut all: Vec<(u64, u64)> = ks.clone();
            if let Some(prof) = expert_profile() {
                all.sort_by_key(|k| std::cmp::Reverse(prof.get(k).copied().unwrap_or(0.0) as u64));
            }
            let in_v: Vec<(u64, u64)> = all.iter().take(nv).copied().collect();
            let in_r: Vec<(u64, u64)> = all.iter().skip(nv).take(nr.saturating_sub(1)).copied().collect();
            let threads = 8;
            let failed: Mutex<Option<String>> = Mutex::new(None);
            {
                let views: Vec<Mutex<&mut [u8]>> = host.iter_mut().map(|c| Mutex::new(c.as_mut_slice())).collect();
                std::thread::scope(|sc| {
                    for th in 0..threads {
                        let (in_v, in_r, views, vq, failed, vram) = (&in_v, &in_r, &views, &vq, &failed, &vram);
                        sc.spawn(move || {
                            let mut buf = vec![0u8; slot_bytes];
                            let mut job = |i: usize, key: (u64, u64), to_v: bool| -> Result<()> {
                                let v = &vq[&key.0];
                                read_expert(m, v, key.1, &mut buf)?;
                                let (ch, o) = (i / per_chunk, (i % per_chunk) * slot_bytes);
                                let span = v.lay(key.1).3;
                                if to_v {
                                    vram[ch].write(o, &buf[..span])
                                } else {
                                    views[ch].lock().unwrap()[o..o + span].copy_from_slice(&buf[..span]);
                                    Ok(())
                                }
                            };
                            for (i, key) in in_v.iter().enumerate().skip(th).step_by(threads) {
                                if let Err(x) = job(i, *key, true) {
                                    *failed.lock().unwrap() = Some(x.0);
                                    return;
                                }
                            }
                            for (i, key) in in_r.iter().enumerate().skip(th).step_by(threads) {
                                if let Err(x) = job(i, *key, false) {
                                    *failed.lock().unwrap() = Some(x.0);
                                    return;
                                }
                            }
                        });
                    }
                });
            }
            if let Some(x) = failed.into_inner().unwrap() {
                return Err(Error(format!("loading the experts: {x}")));
            }
            let mut loc = HashMap::new();
            let mut vowner = vec![None; nv];
            for (i, k) in in_v.iter().enumerate() {
                loc.insert(*k, Loc::V(i));
                vowner[i] = Some(*k);
            }
            let mut rowner = vec![None; nr];
            for (i, k) in in_r.iter().enumerate() {
                loc.insert(*k, Loc::R(i));
                rowner[i] = Some(*k);
            }
            log(format!("{} (layers {:?}, {:.2} MiB experts): {} in VRAM ({:.1} GiB), {} in pinned host memory ({:.1} GiB), {} on the file only",
                        gpu.name, ls, slot_bytes as f64 / 1048576.0, in_v.len(), gib(nv * slot_bytes), in_r.len(), gib(nr * slot_bytes), n_exp - in_v.len() - in_r.len()));
            nv_all += nv;
            nr_all += nr;
            for l in &ls {
                store_of.insert((*l, slot_bytes), stores.len());
            }
            stores.push(Mutex::new(Store { vram, host, slot_bytes, per_chunk, loc, vowner, vused: vec![0; nv], rfree: (in_r.len()..nr).collect(), rowner, rused: vec![0; nr], rmark: None, tick: 0, hits: 0, rread: HashMap::new(),
                                           slotaddr: None, vzero: HashMap::new(), n_expert: g.n_expert as usize,
                                           misses: 0, from_host: 0, from_file: 0, direct: 0, rwrite: HashMap::new(), down_any: None, vfill: HashMap::new(),
                                           last_up: None, prefetched: HashMap::new(), usage: HashMap::new(), base: nv, lend: 0, lent: false, lend_chunk: 0, xlend: 0, x_chunk: 0, x_on: 0, pf_issued: 0,
                                           pf_used: 0, stage: HostBuf::new(gpu, STAGE * slot_bytes)?, stage_tk: vec![None; STAGE], stage_next: 0,
                                           stage_rd: vec![None; STAGE], ahead: HashMap::new(), stage_held: vec![false; STAGE] }));
        }
        log(format!("{}: {:.2} GiB of weights, the experts loaded in {:.1} s", gpu.name, gib(bytes as usize), t0.elapsed().as_secs_f64()));
        let big = DevBuf::new(gpu, arena_bytes())?;
        let small = DevBuf::new(gpu, SMALL_ARENA)?;
        // the big arena lent to the largest store while no prompt chunk needs it
        let mut lent = false;
        if lend_arena() {
            if let Some(st) = stores.iter().max_by_key(|s| { let c = s.lock().unwrap(); c.slot_bytes * c.vowner.len() }) {
                let mut c = st.lock().unwrap();
                let n = big.len / c.slot_bytes;
                if n > 0 {
                    let v = big.view(0, n * c.slot_bytes)?;
                    c.vram.push(v);
                    c.lend_chunk = c.vram.len() - 1;
                    let nv = c.vowner.len();
                    c.vowner.resize(nv + n, None);
                    c.vused.resize(nv + n, 0);
                    c.base = nv;
                    c.lend = n;
                    c.lent = true;
                    lent = true;
                    nv_all += n;
                    // filled now with the profile's next experts (in neither VRAM nor host memory): empty, they waited for
                    // decode's misses to fill them one file read at a time (NS_DS41_FILL_LENT=1; default off)
                    if fill_lent() {
                        let sb = c.slot_bytes;
                        let mut keys: Vec<(u64, u64)> = (0..g.n_expert).flat_map(|x| layers.clone().map(move |l| (l, x)))
                            .filter(|k| vq[&k.0].slot_of(k.1) == sb && !c.loc.contains_key(k)).collect();
                        if let Some(prof) = expert_profile() {
                            keys.sort_by_key(|k| std::cmp::Reverse(prof.get(k).copied().unwrap_or(0.0) as u64));
                        }
                        keys.truncate(n);
                        let (sb, lc) = (c.slot_bytes, c.lend_chunk);
                        let failed: Mutex<Option<String>> = Mutex::new(None);
                        {
                            let (keys, vq, failed, dst) = (&keys, &vq, &failed, &c.vram[lc]);
                            std::thread::scope(|sc| {
                                for th in 0..8 {
                                    sc.spawn(move || {
                                        let mut buf = vec![0u8; sb];
                                        for (i, key) in keys.iter().enumerate().skip(th).step_by(8) {
                                            let v = &vq[&key.0];
                                            if let Err(x) = read_expert(m, v, key.1, &mut buf).and_then(|_| dst.write(i * sb, &buf[..v.lay(key.1).3])) {
                                                *failed.lock().unwrap() = Some(x.0);
                                                return;
                                            }
                                        }
                                    });
                                }
                            });
                        }
                        if let Some(x) = failed.into_inner().unwrap() {
                            return Err(Error(format!("filling the lent slots: {x}")));
                        }
                        for (i, k) in keys.iter().enumerate() {
                            c.loc.insert(*k, Loc::V(nv + i));
                            c.vowner[nv + i] = Some(*k);
                        }
                        log(format!("{}: the big arena's {} lent slots filled from the profile ({:.1} GiB)", gpu.name, keys.len(), gib(keys.len() * sb)));
                    }
                }
            }
        }
        // the slot table: every expert in VRAM now
        let ntab = (g.n_layer as usize) * g.n_expert as usize;
        let devtab = DevBuf::new(gpu, ntab * 8)?;
        {
            let mut host = vec![0u64; ntab];
            for st in &stores {
                let c = st.lock().unwrap();
                for (s, o) in c.vowner.iter().enumerate() {
                    if let Some((l, x)) = o {
                        if *l < g.n_layer {
                            let (ch, at) = c.vat(s);
                            host[*l as usize * g.n_expert as usize + *x as usize] = c.vram[ch].ptr() as u64 + at as u64;
                        }
                    }
                }
            }
            devtab.write(0, &host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        let xtags = if g.xq {
            let ne = g.n_expert as usize;
            let mut host = vec![0i32; ntab];
            for (l, v) in &vq {
                if *l < g.n_layer {
                    for x in 0..ne {
                        // the type word, and the up matrix's place in the slot (in 256 bytes; down follows at twice it)
                        host[*l as usize * ne + x] = v.tag(x as u64) | ((v.lay(x as u64).1 / 256) << 16) as i32;
                    }
                }
            }
            let b = DevBuf::new(gpu, ntab * 4)?;
            b.write(0, &i32s(&host))?;
            Some(b)
        } else {
            None
        };
        // each store's slot addresses on the GPU (the slot table's entries are copied from them), lent slots too
        for st in &stores {
            let mut c = st.lock().unwrap();
            let addrs = c.slot_addrs();
            let b = DevBuf::new(gpu, addrs.len().max(8))?;
            b.write(0, &addrs)?;
            c.slotaddr = Some(b);
        }
        let arena = Arena::on(if lent { small.view(0, small.len)? } else { big.view(0, big.len)? });
        let k = g.idx_top_k as usize;
        let sel = DevBuf::new(gpu, prefill_chunk().max(MAX_VERIFY) * k * 4)?;
        let cand = DevBuf::new(gpu, if g.cand_top > 0 { prefill_chunk().max(MAX_VERIFY) * g.cand_top as usize * 4 } else { 4 })?;
        let zeros = DevBuf::f32(gpu, 4 * g.head_dim as usize)?;
        zeros.fill(0)?;
        let vq_pick = vq_pick(&ops, g, &vq, &cbs, (&cbf, &cbh), &stores, &store_of, &draft, mtp, log)?;
        let ring = if ring_bytes > 0 { Some((DevBuf::new(gpu, ring_bytes)?, ring_slot)) } else { None };
        let nfs = prompt_file_slots();
        let pstage = if ring_bytes > 0 && nfs > 0 { Some((Mutex::new(HostBuf::new(gpu, nfs * ring_slot)?), ring_slot, Mutex::new(vec![None; nfs]))) } else { None };
        Ok(Part { ops, arena, big, small, layers, mats, vecs, groups, scratch, q8, biases: Mutex::new(HashMap::new()), oa_tab: Mutex::new(HashMap::new()), pre_reads: Mutex::new(None), vq, cbs, cbf, cbh, vq_pick, stores, store_of, draft,
                  hold: std::sync::atomic::AtomicBool::new(false), kv_reserve, x_due: std::sync::atomic::AtomicBool::new(lent && lend_extra()), x_first: std::sync::atomic::AtomicBool::new(true), expert_slots: nv_all, host_slots: nr_all, weight_bytes: bytes,
                  up: Mutex::new((HostBuf::new(gpu, UP_BYTES)?, 0)), back: Mutex::new(HostBuf::new(gpu, BACK_BYTES)?), sel, cand, zeros, ring, ring_free: Mutex::new(None), pstage, devtab, xtags,
                  reader: if file_ahead() > 0 { Some(FileReader::new(&m.file.paths, env_usize("NS_DS41_READERS", 16).max(env_usize("NS_DS41_PROMPT_READERS", 32)))?) } else { None } })
    }
}

impl<'g> Ds41<'g> {
    /// Loads the model over `gpus`: the layers split at a kv source (by default the one nearest half the layers; NS_SPLIT
    /// moves it, to another source), the head and the towers on the last. Per GPU, `expert_bytes` of VRAM for routed
    /// experts (None: what is free less the passes' needs and NS_DS41_FREE_GIB); `mirror_bytes` of pinned host memory in
    /// all (None: NS_DS41_PIN_GIB, at most what is available less 12 GiB). `kv`: (context, sessions) held back.
    pub fn load(file: &'g nextsycl_gguf::Gguf, gpus: &[Arc<Gpu>], expert_bytes: Option<usize>, mirror_bytes: Option<usize>, kv: (usize, usize), draft: bool,
                log: &mut dyn FnMut(String)) -> Result<Ds41<'g>> {
        let t0 = Instant::now();
        let m = Model::open(file).map_err(e)?;
        let errs = m.check();
        if !errs.is_empty() {
            return Err(Error(format!("the file's layout: {}", errs.join("; "))));
        }
        if gpus.is_empty() {
            return Err(Error("no GPU given".into()));
        }
        let g = &m.g;
        let n = g.n_layer;
        let k = gpus.len();
        let mirror = mirror_bytes.unwrap_or_else(|| pin_cap().min(mem_available().saturating_sub(12 << 30)));
        // the boundaries: kv sources (a source's readers stay with it); even by count unless NS_SPLIT says
        let mut bounds: Vec<u64> = vec![0];
        let cuts: Vec<u64> = match std::env::var("NS_SPLIT") {
            Ok(v) => v.split(',').filter_map(|x| x.trim().parse::<u64>().ok()).scan(0, |a, c| { *a += c; Some(*a) }).collect(),
            // by the GPUs' compute units (a B70's 256 and a B65's 160: 24 / 16 layers, tg 17.4 -> 20.7 tok/s against halves;
            // their decode kernel times a layer measured 1 : 1.43), at the nearest kv source (two GPUs: an index source
            // too, its keys mirrored)
            Err(_) => {
                let rate: Vec<f64> = gpus.iter().map(|gp| nextsycl_core::gpu_units(gp.index).map_or(1.0, |(u, _)| (u as f64).max(1.0))).collect();
                let total: f64 = rate.iter().sum();
                let at: Vec<u64> = if k == 2 { g.kv_sources.iter().chain(&g.index_sources).copied().collect() } else { g.kv_sources.clone() };
                (1..k).map(|i| {
                    let want = n as f64 * rate[..i].iter().sum::<f64>() / total;
                    *at.iter().chain(std::iter::once(&0)).min_by(|a, b| (**a as f64 - want).abs().total_cmp(&(**b as f64 - want).abs())).unwrap_or(&0)
                }).collect()
            }
        };
        if cuts.len() + 1 != k {
            return Err(Error(format!("NS_SPLIT: {} layer counts for {k} GPUs", cuts.len())));
        }
        let mut mirror_src = None;
        for c in cuts {
            // a part begins at a kv source, before the compressed layers, or (two GPUs) at an index source: its source's
            // keys mirrored on the second GPU
            let at_index = k == 2 && g.index_sources.contains(&c) && g.kv_source(c).is_some();
            if c <= *bounds.last().unwrap() || c >= n || (g.kv_source(c).is_some() && !g.kv_sources.contains(&c) && !at_index) {
                return Err(Error(format!("a split at layer {c}: a part must begin at a kv source ({:?}), an index source ({:?}, two GPUs) or before the \
                                          compressed layers", g.kv_sources, g.index_sources)));
            }
            if at_index && !g.kv_sources.contains(&c) {
                mirror_src = g.kv_source(c);
            }
            bounds.push(c);
        }
        bounds.push(n);
        let mut parts = Vec::new();
        let mut owner = vec![0usize; n as usize];
        // the towers only when asked for (NS_DS41_MTP=1): on the B70 + B65 their drafts cost more than they save on
        // prose (a verify pass's rows cost nearly what one-token passes do: the decode products are ALU-bound), and
        // without them the last GPU holds ~2.9 GiB more experts
        let mtp = draft && g.n_mtp > 0 && mtp_on();
        let mut plans = Vec::new();
        for (i, gpu) in gpus.iter().enumerate() {
            let range = bounds[i]..bounds[i + 1];
            for l in range.clone() {
                owner[l as usize] = i;
            }
            let last = i + 1 == k;
            let share = mirror * (range.end - range.start) as usize / n as usize;
            // per session: the raw rings, the sources' compressed and index keys (fp16), the towers' rings
            let reserve: usize = range.clone().map(|l| {
                let mut b = raw_ring() * 512 * 2;
                if g.kv_sources.contains(&l) {
                    let r = g.ratio(l) as usize;
                    b += (kv.0 / r + 1) * (512 + g.idx_dim as usize) * 2;
                }
                b
            }).sum::<usize>() * kv.1 + if last && mtp { kv.1 * 3 * MTP_WIN * 1024 } else { 0 }
                + match mirror_src {
                    Some(sl) if i == 1 => (kv.0 / g.ratio(sl) as usize + 1) * (512 + g.idx_dim as usize) * 2 * kv.1,
                    _ => 0,
                };
            if reserve > 0 {
                log(format!("{}: {:.2} GiB held for the sessions' {} tokens", gpu.name, gib(reserve), kv.0 * kv.1));
            }
            plans.push((gpu, range, last, share, reserve));
        }
        // the parts load at once, a thread each (their experts' reads overlap: 20.1 s -> one GPU's time); each part's
        // lines logged after, in order
        let loaded: Vec<(Result<Part>, Vec<String>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = plans.into_iter().map(|(gpu, range, last, share, reserve)| {
                let m = &m;
                sc.spawn(move || {
                    let mut lines = Vec::new();
                    let p = Part::load(m, gpu, range, last, mtp, expert_bytes, share, reserve, &mut |s| lines.push(s));
                    (p, lines)
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| (Err(Error("a part's load panicked".into())), Vec::new()))).collect()
        });
        for (p, lines) in loaded {
            for l in lines {
                log(l);
            }
            parts.push(p?);
        }
        let engram = if g.engram_layers.is_empty() {
            None
        } else {
            let dir = engram::engram_dir(&file.paths[0]);
            let en = Engram::open(file, &dir)?;
            log(format!("the Engram: layers {:?}, tables in {}", en.layers, dir.display()));
            Some(en)
        };
        let vision = match crate::vision::sidecar(&file.paths[0]) {
            Some(path) => {
                let mut v = crate::vision::Vision::open(&path, g.n_layer, g.n_embd as usize).map_err(|x| Error(x.0))?;
                if let Some(m) = file.meta("tokenizer.ggml.tokens").and_then(|t| t.as_array()).and_then(|a| a.get(v.cfg.image_token as usize))
                    .and_then(|x| x.as_str()) {
                    v.marker = m.to_string();
                }
                log(format!("the vision sidecar: {} ({} ViT layers, at most {} positions an image)", path.display(), v.cfg.layers, v.cfg.max_tokens));
                Some(v)
            }
            None => None,
        };
        let load_bytes = parts.iter().map(|p| p.weight_bytes).sum();
        let prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(BTreeMap::new()));
        let dry = match crate::dry::Dry::from_env() {
            Some(d) => {
                let tok = nextsycl_tok::Tokenizer::from_gguf(file).map_err(Error)?;
                log(format!("DRY against repeated runs: multiplier {}, base {}, from {} tokens", d.multiplier, d.base, d.allowed));
                Some((d, crate::dry::vocab(tok.tokens.len(), |t| tok.decode(&[t]))))
            }
            None => None,
        };
        if let Some(sl) = mirror_src {
            log(format!("layers {}.. on the second GPU: kv source {sl}'s keys mirrored there", bounds[1]));
        }
        Ok(Ds41 { m, parts, owner, load_seconds: t0.elapsed().as_secs_f64(), load_bytes, engram, prof, mtp, mirror_src, cand_q: Mutex::new(Default::default()),
                  pipelining: std::sync::atomic::AtomicBool::new(false), vision, dry })
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.parts.iter().map(|p| {
            let (total, free) = p.ops.gpu.memory().unwrap_or((0, None));
            GpuInfo { index: p.ops.gpu.index, name: p.ops.gpu.name.clone(), pci: p.ops.gpu.pci.clone(), total, free, layers: (p.layers.start, p.layers.end),
                      expert_slots: p.expert_slots, host_slots: p.host_slots }
        }).collect()
    }

    /// (VRAM hits, swaps, of them from host memory, host-memory reads by prompt passes, prefetched, prefetches used)
    pub fn expert_stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        let mut a = (0, 0, 0, 0, 0, 0);
        for p in &self.parts {
            for s in &p.stores {
                let c = s.lock().unwrap();
                a = (a.0 + c.hits, a.1 + c.misses, a.2 + c.from_host, a.3 + c.direct, a.4 + c.pf_issued, a.5 + c.pf_used);
            }
        }
        a
    }

    /// experts read from the file on a miss
    pub fn file_reads(&self) -> u64 {
        self.parts.iter().flat_map(|p| p.stores.iter()).map(|s| s.lock().unwrap().from_file).sum()
    }

    pub fn arena_peaks(&self) -> Vec<(usize, usize)> {
        use std::sync::atomic::Ordering::Relaxed;
        self.parts.iter().map(|p| (p.arena.peak.load(Relaxed), p.arena.spills.load(Relaxed))).collect()
    }

    fn timed<T>(&self, p: &Part, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(pr) = &self.prof else { return f() };
        p.ops.gpu.sync()?;
        let t0 = Instant::now();
        let r = f()?;
        p.ops.gpu.sync()?;
        let mut m = pr.lock().unwrap();
        let x = m.entry(name).or_insert((0.0, 0));
        x.0 += t0.elapsed().as_secs_f64();
        x.1 += 1;
        Ok(r)
    }

    /// With NS_PROFILE=1: the time since `t0` (the GPU drained) added to `name`
    fn lap(&self, p: &Part, name: &'static str, t0: Instant) -> Result<()> {
        let Some(pr) = &self.prof else { return Ok(()) };
        p.ops.gpu.sync()?;
        let mut m = pr.lock().unwrap();
        let x = m.entry(name).or_insert((0.0, 0));
        x.0 += t0.elapsed().as_secs_f64();
        x.1 += 1;
        Ok(())
    }

    pub fn profile(&self) -> Vec<(&'static str, f64, u64)> {
        let Some(p) = &self.prof else { return Vec::new() };
        let mut v: Vec<_> = p.lock().unwrap().iter().map(|(k, (s, n))| (*k, *s, *n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    pub fn hold_arena(&self, on: bool) {
        for p in &self.parts {
            p.hold.store(on, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn part_of(&self, l: u64) -> &Part {
        &self.parts[self.owner.get(l as usize).copied().unwrap_or(self.parts.len() - 1)]
    }

    /// A new conversation, up to `max_ctx` tokens.
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.m.g;
        let mut layers = Vec::new();
        for l in 0..g.n_layer {
            let gpu = &self.part_of(l).ops.gpu;
            let raw = DevBuf::new(gpu, raw_ring() * 512 * 2)?;
            let comp = if g.kv_sources.contains(&l) {
                let r = g.ratio(l) as usize;
                let ring = if r > 1 {
                    let b = DevBuf::f32(gpu, comp_ring(r) * 2 * 512)?;
                    b.fill(0)?;
                    Some(b)
                } else {
                    None
                };
                Some(Comp { ratio: r, cache: DevBuf::new(gpu, (max_ctx / r + 1) * 512 * 2)?, ring, idx: DevBuf::new(gpu, (max_ctx / r + 1) * g.idx_dim as usize * 2)? })
            } else {
                None
            };
            let mirror = if self.mirror_src == Some(l) {
                let r = g.ratio(l) as usize;
                let gpu = &self.parts[1].ops.gpu;
                Some(Comp { ratio: r, cache: DevBuf::new(gpu, (max_ctx / r + 1) * 512 * 2)?, ring: None, idx: DevBuf::new(gpu, (max_ctx / r + 1) * g.idx_dim as usize * 2)? })
            } else {
                None
            };
            layers.push(Att { raw, comp, mirror });
        }
        let mtp = if self.mtp {
            let gpu = &self.parts.last().unwrap().ops.gpu;
            let rings = (0..g.n_mtp).map(|_| DevBuf::new(gpu, MTP_WIN * 512 * 2)).collect::<Result<Vec<_>>>()?;
            Some(MtpS { rings, mh: DevBuf::f32(gpu, 3 * MTP_KEEP * g.n_embd as usize)?, mh_rows: (0, 0), snap: DevBuf::new(gpu, g.n_mtp as usize * MAX_VERIFY * 1024)?,
                        snap_at: None })
        } else {
            None
        };
        Ok(Session { pos: 0, max_ctx, layers, hist: Vec::new(), gen_from: 0, snapped: None, mtp })
    }

    /// The raw ring's slots of the window before `pos` (at most 136 positions, in up to two runs)
    fn window_runs(pos: usize) -> Vec<(usize, usize)> {
        let r = raw_ring();
        let n = pos.min(128 + MAX_VERIFY);
        if n == 0 {
            return Vec::new();
        }
        let first = (pos - n) % r;
        let row = 512 * 2;
        if first + n <= r {
            vec![(first * row, n * row)]
        } else {
            vec![(first * row, (r - first) * row), (0, (first + n - r) * row)]
        }
    }

    /// The conversation's state at `sess.pos` in host memory: per layer the window's raw keys, at a source its
    /// compressed keys so far, its ring and its index keys; the towers' rings; the tokens
    pub fn save(&self, sess: &Session) -> Result<Checkpoint> {
        let n = sess.pos;
        let idx_d = self.m.g.idx_dim as usize;
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let mut take = |b: &DevBuf, at: usize, len: usize| -> Result<()> {
            let mut v = vec![0u8; len];
            if len > 0 {
                b.read(at, &mut v)?;
            }
            bufs.push(v);
            Ok(())
        };
        for a in &sess.layers {
            let runs = Self::window_runs(n);
            for k in 0..2 {
                let (at, len) = runs.get(k).copied().unwrap_or((0, 0));
                take(&a.raw, at, len)?;
            }
            if let Some(c) = &a.comp {
                take(&c.cache, 0, n / c.ratio * 512 * 2)?;
                take(&c.idx, 0, n / c.ratio * idx_d * 2)?;
                if let Some(r) = &c.ring {
                    take(r, 0, r.len)?;
                }
            }
        }
        if let Some(ms) = &sess.mtp {
            for r in &ms.rings {
                take(r, 0, r.len)?;
            }
        }
        bufs.push(i32s(&sess.hist[..n].iter().map(|t| *t as i32).collect::<Vec<_>>()));
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: n, bufs, bytes })
    }

    pub fn restore(&self, sess: &mut Session, ck: &Checkpoint) -> Result<()> {
        if ck.pos + 1 > sess.max_ctx {
            return Err(Error(format!("restore: a checkpoint of {} tokens into a session of {}", ck.pos, sess.max_ctx)));
        }
        let mut it = ck.bufs.iter();
        for a in &sess.layers {
            let runs = Self::window_runs(ck.pos);
            for k in 0..2 {
                let v = it.next().ok_or("restore: the checkpoint is short")?;
                if !v.is_empty() {
                    a.raw.write(runs.get(k).map_or(0, |r| r.0), v)?;
                }
            }
            if let Some(c) = &a.comp {
                let (cache, idx) = (it.next().ok_or("restore: the checkpoint is short")?, it.next().ok_or("restore: the checkpoint is short")?);
                for cc in [Some(c), a.mirror.as_ref()].into_iter().flatten() {
                    if !cache.is_empty() {
                        cc.cache.write(0, cache)?;
                    }
                    if !idx.is_empty() {
                        cc.idx.write(0, idx)?;
                    }
                }
                if let Some(r) = &c.ring {
                    let v = it.next().ok_or("restore: the checkpoint is short")?;
                    if !v.is_empty() {
                        r.write(0, v)?;
                    }
                }
            }
        }
        if let Some(ms) = &sess.mtp {
            for r in &ms.rings {
                let v = it.next().ok_or("restore: the checkpoint is short")?;
                if !v.is_empty() {
                    r.write(0, v)?;
                }
            }
        }
        let toks = it.next().ok_or("restore: the checkpoint has no tokens")?;
        sess.hist = toks.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
        sess.pos = ck.pos;
        sess.gen_from = ck.pos;
        sess.snapped = None;
        if let Some(ms) = sess.mtp.as_mut() {
            ms.mh_rows = (0, 0);
            ms.snap_at = None;
        }
        Ok(())
    }

    pub fn reset_session(&self, sess: &mut Session) -> Result<()> {
        sess.pos = 0;
        sess.gen_from = 0;
        sess.hist.clear();
        sess.snapped = None;
        if let Some(ms) = sess.mtp.as_mut() {
            ms.mh_rows = (0, 0);
            ms.snap_at = None;
        }
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if dst.max_ctx != src.max_ctx || dst.layers.len() != src.layers.len() {
            return Err(Error("copy_session: the sessions differ in shape".into()));
        }
        let idx_d = self.m.g.idx_dim as usize;
        for (d, s) in dst.layers.iter().zip(&src.layers) {
            for (at, len) in Self::window_runs(src.pos) {
                d.raw.copy_within(at, &s.raw, at, len)?;
            }
            if let (Some(dc), Some(sc)) = (&d.comp, &s.comp) {
                let nb = src.pos / sc.ratio;
                if nb > 0 {
                    dc.cache.copy_within(0, &sc.cache, 0, nb * 512 * 2)?;
                    dc.idx.copy_within(0, &sc.idx, 0, nb * idx_d * 2)?;
                }
                if let (Some(dr), Some(sr)) = (&dc.ring, &sc.ring) {
                    dr.copy_within(0, sr, 0, sr.len)?;
                }
            }
            if let (Some(dm), Some(sm)) = (&d.mirror, &s.mirror) {
                let nb = src.pos / sm.ratio;
                if nb > 0 {
                    dm.cache.copy_within(0, &sm.cache, 0, nb * 512 * 2)?;
                    dm.idx.copy_within(0, &sm.idx, 0, nb * idx_d * 2)?;
                }
            }
        }
        if let (Some(dm), Some(sm)) = (dst.mtp.as_mut(), src.mtp.as_ref()) {
            for (d, s) in dm.rings.iter().zip(&sm.rings) {
                d.copy_within(0, s, 0, s.len)?;
            }
            dm.mh_rows = (0, 0);
            dm.snap_at = None;
        }
        dst.hist = src.hist.clone();
        dst.pos = src.pos;
        dst.gen_from = src.gen_from;
        dst.snapped = None;
        Ok(())
    }

    /// The pass's new rows of kv source l's compressed and indexer keys (and the candidate blocks it kept, when it did)
    /// onto the second GPU: read back here (the host waits for the source anyway, at the next router), queued there
    /// through its pinned ring
    fn mirror_up(&self, l: u64, a: &Att, pos0: usize, t: usize, cand: bool) -> Result<()> {
        let (Some(c), Some(m)) = (&a.comp, &a.mirror) else { return Ok(()) };
        let g = &self.m.g;
        let p1 = &self.parts[1];
        let (j0, j1) = (pos0 / c.ratio, (pos0 + t) / c.ratio);
        let id = g.idx_dim as usize;
        let piece = 1 << 20;
        let mut v = Vec::new();
        for (src, dst, w) in [(&c.cache, &m.cache, 512 * 2), (&c.idx, &m.idx, id * 2)] {
            let (at, len) = (j0 * w, (j1 - j0) * w);
            if len == 0 {
                continue;
            }
            v.resize(len, 0);
            src.read(at, &mut v)?;
            for (k, ch) in v.chunks(piece).enumerate() {
                p1.upload(&dst.view(at + k * piece, ch.len())?, ch)?;
            }
        }
        let mut blocks = Vec::new();
        if cand && g.cand_source == Some(l) {
            blocks.resize(t * g.cand_top as usize * 4, 0);
            self.parts[0].cand.read(0, &mut blocks)?;
        }
        if self.pipelining.load(std::sync::atomic::Ordering::Relaxed) {
            self.cand_q.lock().unwrap().push_back(blocks);
        } else {
            self.cand_in(&blocks)?;
        }
        Ok(())
    }

    /// A pass's candidate blocks into the second GPU's (empty: none kept)
    fn cand_in(&self, blocks: &[u8]) -> Result<()> {
        let p1 = &self.parts[1];
        for (k, ch) in blocks.chunks(1 << 20).enumerate() {
            p1.upload(&p1.cand.view(k * (1 << 20), ch.len())?, ch)?;
        }
        Ok(())
    }

    /// Rows of a [n_vocab, cols] matrix read from the file (the embedding: q4_K; the Markov embedding: BF16), as
    /// float32 on p's GPU
    fn rows_of(&self, p: &Part, r: Role, tokens: &[u32]) -> Result<DevBuf> {
        let t = self.m.tensor(0, r).ok_or_else(|| Error(format!("{r:?} missing")))?;
        let cols = *t.shape.last().unwrap_or(&1);
        let rb = t.ty.bytes(cols).unwrap_or(0) as usize;
        let mut rows = vec![0u8; tokens.len() * rb];
        for (i, &tok) in tokens.iter().enumerate() {
            // an image span's position: the placeholder's row (its own rows are written over it: image_spans)
            let tok = if crate::vision::is_image(tok) { self.vision.as_ref().map_or(tok, |v| v.cfg.image_token) } else { tok };
            if tok as u64 >= self.m.g.n_vocab {
                return Err(Error(format!("token {tok} outside the vocabulary")));
            }
            self.m.file.read_into(t, tok as u64 * rb as u64, &mut rows[i * rb..(i + 1) * rb]).map_err(e)?;
        }
        let raw = DevBuf::new(&p.ops.gpu, rows.len())?;
        p.upload(&raw, &rows)?;
        let x = DevBuf::f32(&p.ops.gpu, tokens.len() * cols as usize)?;
        p.ops.dequant(t.ty.code(), &raw, 0, rows.len(), tokens.len() * cols as usize, &x)?;
        Ok(x)
    }

    /// An image's input rows [lh * lw][n_embd] on part p: the ViT over its patches (vision.py: the patch product, 32
    /// blocks of norm, attention with the 2D rotation, norm, SwiGLU; the last norm), then the aligner (3 x 3 patches a
    /// row, zero-padded, through two products with GELU between). Float32 on the GPU, the weights a layer at a time from
    /// the sidecar (bf16) into the prompt arena; kept with the image once made.
    pub fn image_rows(&self, p: &Part, img: &crate::vision::Img) -> Result<Arc<Vec<f32>>> {
        if let Some(r) = img.rows.lock().unwrap().as_ref() {
            return Ok(r.clone());
        }
        let vis = self.vision.as_ref().ok_or("an image span, and no vision sidecar (NS_DS41_VISION)")?;
        let px = img.patches.lock().unwrap().clone().ok_or("an image without its patches")?;
        let c = &vis.cfg;
        let o = &p.ops;
        let pl = img.plan;
        let (n, dim, nh, ff, pp) = (pl.vh * pl.vw, c.dim, c.heads, c.ff, 3 * c.patch * c.patch);
        let t0 = Instant::now();
        // the big arena: an image's temporaries run to ~0.5 GiB
        p.arena_for(prefill_chunk())?;
        let mk = p.arena.mark();
        let w = |name: &str, len: usize| -> Result<DevBuf> {
            let v = crate::vision::floats(&vis.file, name).map_err(|x| Error(x.0))?;
            if v.len() != len {
                return Err(Error(format!("{name}: {} values, {len} expected", v.len())));
            }
            let b = p.arena.f32(len)?;
            b.write(0, &f32s(&v))?;
            Ok(b)
        };
        let x = p.arena.f32(n * dim)?;
        {
            let xin = p.arena.f32(n * pp)?;
            xin.write(0, &f32s(&px))?;
            let (pw, pb) = (w("v.patch_embd.weight", dim * pp)?, w("v.patch_embd.bias", dim)?);
            o.gemm_at(n, dim, pp, (&xin, 0, pp), (&pw, 0), (&x, 0, dim), false)?;
            o.bias_act(&x, Some(&pb), n, dim, false)?;
        }
        let normed = p.arena.f32(n * dim)?;
        let qkv = p.arena.f32(n * 3 * dim)?;
        let kc = p.arena.f32(n * dim)?;
        let att = p.arena.f32(n * dim)?;
        // attention a block of query rows at a time: 16 heads x rows x n scores in 96 MB at most
        let rb = ((96usize << 20) / (4 * nh * n)).clamp(1, n);
        let s = p.arena.f32(nh * rb * n)?;
        let ob = p.arena.f32(nh * rb * 64)?;
        // the MLP a block of rows at a time
        let mb = 2048.min(n);
        let gu = p.arena.f32(mb * 2 * ff)?;
        let hb = p.arena.f32(mb * ff)?;
        let lm = p.arena.mark();
        let eps = 1e-6f32;
        for l in 0..c.layers {
            p.arena.rewind(lm);
            let pre = format!("v.blk.{l}");
            let (n1, n2) = (w(&format!("{pre}.attn_norm.weight"), dim)?, w(&format!("{pre}.ffn_norm.weight"), dim)?);
            let (wqkv, bqkv) = (w(&format!("{pre}.attn_qkv.weight"), 3 * dim * dim)?, w(&format!("{pre}.attn_qkv.bias"), 3 * dim)?);
            let (wo, bo) = (w(&format!("{pre}.attn_output.weight"), dim * dim)?, w(&format!("{pre}.attn_output.bias"), dim)?);
            let (wgu, wd) = (w(&format!("{pre}.ffn_gate_up.weight"), 2 * ff * dim)?, w(&format!("{pre}.ffn_down.weight"), dim * ff)?);
            o.rms_norm(&x, Some(&n1), &normed, n, dim, eps)?;
            o.gemm_at(n, 3 * dim, dim, (&normed, 0, dim), (&wqkv, 0), (&qkv, 0, 3 * dim), false)?;
            o.bias_act(&qkv, Some(&bqkv), n, 3 * dim, false)?;
            o.vit_rope(&qkv, &kc, n, nh, pl.vw, c.theta)?;
            for r0 in (0..n).step_by(rb) {
                let r = rb.min(n - r0);
                // per head: s = q k^T (q in qkv's rows, k a head's [n][64]), softmax / 8, then s v (v in qkv's rows)
                o.gemm_batch(nh, r, n, 64, (&qkv, r0 * 3 * dim, 3 * dim, 64), (&kc, 0, n * 64), (&s, 0, n, r * n))?;
                o.softmax_rows(&s, nh * r, n, 0.125)?;
                // each head's [r][64] apart (oneMKL takes no interleaved outputs), then into att's rows
                o.gemm_batch_nn(nh, r, 64, n, (&s, 0, n, r * n), (&qkv, 2 * dim, 3 * dim, 64), (&ob, 0, 64, r * 64))?;
                o.permute(&ob, &att.view(r0 * dim * 4, r * dim * 4)?, nh, r, 64)?;
            }
            o.gemm_at(n, dim, dim, (&att, 0, dim), (&wo, 0), (&normed, 0, dim), false)?;
            o.bias_act(&normed, Some(&bo), n, dim, false)?;
            o.add(&x, &normed, n * dim)?;
            o.rms_norm(&x, Some(&n2), &normed, n, dim, eps)?;
            for r0 in (0..n).step_by(mb) {
                let r = mb.min(n - r0);
                o.gemm_at(r, 2 * ff, dim, (&normed, r0 * dim, dim), (&wgu, 0), (&gu, 0, 2 * ff), false)?;
                o.silu_mul(&gu, &hb, r, ff)?;
                o.gemm_at(r, dim, ff, (&hb, 0, ff), (&wd, 0), (&att, r0 * dim, dim), false)?;
            }
            o.add(&x, &att, n * dim)?;
        }
        p.arena.rewind(lm);
        let nf = w("v.output_norm.weight", dim)?;
        o.rms_norm(&x, Some(&nf), &normed, n, dim, eps)?;
        let vout = normed.to_f32()?;
        p.arena.rewind(mk);
        // the aligner: [C, vh, vw] zero-padded to whole 3 x 3 cells, unfolded (a cell's features c-major: c * 9 + 3 ky +
        // kx), a cell a row in reading order
        let r = c.ratio;
        let (lh, lw, kin, d) = (pl.lh, pl.lw, dim * r * r, self.m.g.n_embd as usize);
        let mut un = vec![0f32; lh * lw * kin];
        for bi in 0..lh {
            for bj in 0..lw {
                let row = &mut un[(bi * lw + bj) * kin..(bi * lw + bj + 1) * kin];
                for ky in 0..r {
                    for kx in 0..r {
                        let (y, xx) = (bi * r + ky, bj * r + kx);
                        if y >= pl.vh || xx >= pl.vw {
                            continue;
                        }
                        let src = &vout[(y * pl.vw + xx) * dim..(y * pl.vw + xx + 1) * dim];
                        for (ci, v) in src.iter().enumerate() {
                            row[ci * r * r + ky * r + kx] = *v;
                        }
                    }
                }
            }
        }
        let l = lh * lw;
        let ui = p.arena.f32(l * kin)?;
        ui.write(0, &f32s(&un))?;
        let hid = p.arena.f32(l * d)?;
        let out = p.arena.f32(l * d)?;
        {
            let lm2 = p.arena.mark();
            let (w0, b0) = (w("mm.0.weight", d * kin)?, w("mm.0.bias", d)?);
            o.gemm_at(l, d, kin, (&ui, 0, kin), (&w0, 0), (&hid, 0, d), false)?;
            o.bias_act(&hid, Some(&b0), l, d, true)?;
            p.arena.rewind(lm2);
            let (w1, b1) = (w("mm.1.weight", d * d)?, w("mm.1.bias", d)?);
            o.gemm_at(l, d, d, (&hid, 0, d), (&w1, 0), (&out, 0, d), false)?;
            o.bias_act(&out, Some(&b1), l, d, false)?;
        }
        let rows = Arc::new(out.to_f32()?);
        p.arena.rewind(mk);
        // NS_DS41_VISION_DUMP=PREFIX: the patches, the ViT's output and the rows as raw float32 (a check against the
        // reference's)
        if let Ok(pre) = std::env::var("NS_DS41_VISION_DUMP") {
            for (name, v) in [("patches", &px[..]), ("vit", &vout[..]), ("rows", &rows[..])] {
                std::fs::write(format!("{pre}.{name}.f32"), f32s(v)).map_err(|x| Error(format!("NS_DS41_VISION_DUMP: {x}")))?;
            }
            eprintln!("[the image's patches [{n}][{pp}], ViT output [{n}][{dim}], rows [{l}][{d}] in {pre}.*.f32 (grid {} x {})]", pl.vh, pl.vw);
        }
        eprintln!("[an image of {} x {} patches -> {} x {} rows in {:.2} s]", pl.vh, pl.vw, lh, lw, t0.elapsed().as_secs_f64());
        *img.rows.lock().unwrap() = Some(rows.clone());
        *img.patches.lock().unwrap() = None;
        Ok(rows)
    }

    /// The input rows of an image span's positions among `tokens` written over x0's ([t][d]) rows
    fn image_spans(&self, p: &Part, tokens: &[u32], x0: &DevBuf) -> Result<()> {
        use crate::vision::{Slot, is_image};
        let Some(vis) = self.vision.as_ref() else {
            return if tokens.iter().any(|t| is_image(*t)) { Err(Error("an image span, and no vision sidecar (NS_DS41_VISION)".into())) } else { Ok(()) };
        };
        let d = self.m.g.n_embd as usize;
        let mut buf = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if !is_image(tokens[i]) {
                i += 1;
                continue;
            }
            // a run of one image's positions
            let key = crate::vision::key_of(tokens[i]);
            let mut j = i;
            while j < tokens.len() && is_image(tokens[j]) && crate::vision::key_of(tokens[j]) == key {
                j += 1;
            }
            let img = vis.image(tokens[i]).map_err(|x| Error(x.0))?;
            let needs_rows = tokens[i..j].iter().any(|t| matches!(img.plan.slot(crate::vision::pos_of(*t)), Slot::Row(_)));
            let rows = if needs_rows { Some(self.image_rows(p, &img)?) } else { None };
            buf.clear();
            for t in &tokens[i..j] {
                let pos = crate::vision::pos_of(*t);
                if pos >= img.plan.span() {
                    return Err(Error(format!("an image position {pos} past its span of {}", img.plan.span())));
                }
                match img.plan.slot(pos) {
                    Slot::Start => buf.extend_from_slice(&vis.start),
                    Slot::End => buf.extend_from_slice(&vis.end),
                    Slot::Newline => buf.extend_from_slice(&vis.newline),
                    Slot::Row(k) => buf.extend_from_slice(&rows.as_ref().ok_or("image rows")?[k * d..(k + 1) * d]),
                }
            }
            x0.view(i * d * 4, (j - i) * d * 4)?.write(0, &f32s(&buf))?;
            i = j;
        }
        Ok(())
    }

    /// The embedding rows on part `pi` as the 4 streams [t, 4, d], then the first mix [t, 4] (stream 0)
    fn streams(&self, pi: usize, tokens: &[u32], tap: Tap) -> Result<DevBuf> {
        let d = self.m.g.n_embd as usize;
        let t = tokens.len();
        let p = &self.parts[pi];
        let x0 = self.rows_of(p, Role::TokenEmbd, tokens)?;
        if tokens.iter().any(|t| crate::vision::is_image(*t)) {
            p.ops.gpu.sync()?;
            self.image_spans(p, tokens, &x0)?;
        }
        tap("inp_embd", &x0)?;
        let mut pre = vec![0f32; t * 4];
        for ti in 0..t {
            pre[ti * 4] = 1.0;
        }
        if t > MMVQ_COLS {
            let e0 = x0.to_f32()?;
            let mut xs = Vec::with_capacity(t * 4 * d + t * 4);
            for ti in 0..t {
                for _ in 0..4 {
                    xs.extend_from_slice(&e0[ti * d..(ti + 1) * d]);
                }
            }
            xs.extend_from_slice(&pre);
            return DevBuf::from_f32(&p.ops.gpu, &xs);
        }
        // each row 4 times on the GPU (kernel copies: no trip through the host behind the expert copies)
        let xi = DevBuf::f32(&p.ops.gpu, t * 4 * d + t * 4)?;
        for ti in 0..t {
            for s in 0..4 {
                p.ops.kcopy(&xi, (ti * 4 + s) * d * 4, &x0, ti * d * 4, d * 4)?;
            }
        }
        p.upload(&xi.view(t * 4 * d * 4, t * 16)?, &f32s(&pre))?;
        Ok(xi)
    }

    /// The carried state [t, 4, d] streams + [t, 4] mix onto part pi (through host memory)
    fn to_part(&self, pi: usize, x: &DevBuf, staging: &mut Vec<u8>) -> Result<DevBuf> {
        let p = &self.parts[pi];
        let next = DevBuf::new(&p.ops.gpu, x.len)?;
        if pi > 0 && x.len.is_multiple_of(4) {
            let src = &self.parts[pi - 1];
            if let Some((tk, spans)) = src.read_start(&[(x, x.len / 4)])? {
                let v = src.read_finish(tk, &spans)?.into_iter().next().unwrap_or_default();
                p.upload(&next, &f32s(&v))?;
                return Ok(next);
            }
        }
        staging.resize(staging.len().max(x.len), 0);
        next.copy_from_peer(x, staging)?;
        Ok(next)
    }

    /// Engram layer l on the streams x [t, 4, d] (rows: the layer's raw table rows [t * 24][264])
    fn engram_apply(&self, p: &Part, l: u64, x: &DevBuf, raw: &[u8], tokens: &[u32], tap: Tap) -> Result<()> {
        let g = &self.m.g;
        let t = tokens.len();
        let en = self.engram.as_ref().ok_or("the Engram is not loaded")?;
        let (d, hd, nc) = (g.n_embd as usize, en.hd, en.cols());
        let rb = p.arena.bytes(raw.len().next_multiple_of(4))?;
        let mut padded = raw.to_vec();
        padded.resize(raw.len().next_multiple_of(4), 0);
        // an image position takes nothing from the Engram (model.py: its gate shut): its rows zero, so its key and value
        // are zero and the gate adds nothing
        let per = raw.len() / t.max(1);
        for (i, tok) in tokens.iter().enumerate() {
            if crate::vision::is_image(*tok) {
                padded[i * per..(i + 1) * per].fill(0);
            }
        }
        p.upload(&rb, &padded)?;
        let rows = p.arena.f32(t * nc * hd)?;
        match en.gpu_type() {
            // the llama.cpp form's Q5_K / Q6_K / IQ4_NL rows: one block a row, expanded by the shared kernels
            // the silo's std kernels where they take the type (IQ4_NL: the UD-IQ1_M file's table), else the shared ones
            Some(ty) if engram_shared() => p.ops.dequant(ty.code(), &rb, 0, t * nc * en.rb, t * nc * hd, &rows)?,
            Some(ty) => p.ops.dequant_mat(ty.code(), &rb, 0, t * nc * en.rb, t * nc, hd, &rows, false)?,
            None => p.ops.engram_rows(&rb, &rows, t * nc, hd)?,
        }
        let kv = p.mm(l, Role::EngramWkv, &rows, t)?;
        p.ops.engram_gate(x, &kv, p.vec(l, Role::EngramQ)?, t, d, g.rms_eps as f32)?;
        tap(&format!("hc_engram-{l}"), &x.view(0, t * 4 * d * 4)?)?;
        Ok(())
    }

    /// The compressor of kv source l over this pass's x [t, d] (normed): the groups it completes - pooled at ratio 2,
    /// projected at 1 - normed; their index keys (wk, k_norm, rotated, fp4) and compressed keys (rotated, fp4 E4M3) into
    /// the caches
    #[allow(clippy::too_many_arguments)]
    fn compress(&self, p: &Part, l: u64, x: &DevBuf, c: &Comp, pos0: usize, t: usize, tap: Tap) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (hd, id, rd, eps) = (g.head_dim as usize, g.idx_dim as usize, g.rope_dim as usize, g.rms_eps as f32);
        let r = c.ratio;
        let (j0, j1) = (pos0 / r, (pos0 + t) / r);
        let nb = j1 - j0;
        let kv = p.mm(l, Role::CompKv, x, t)?;
        let lat = if r == 1 {
            kv
        } else {
            let sc = p.mm(l, Role::CompGate, x, t)?;
            let pooled = p.arena.f32(nb.max(1) * hd)?;
            o.pool(c.ring.as_ref().ok_or("a ratio-2 source without its ring")?, &kv, &sc, &p.zeros, &pooled, pos0, t, hd, r, false, comp_ring(r))?;
            pooled
        };
        if nb == 0 {
            return Ok(());
        }
        o.rms_norm(&lat, Some(p.vec(l, Role::CompNorm)?), &lat, nb, hd, eps)?;
        let rope = self.m.rope(l);
        let ik = p.mm(l, Role::IdxWk, &lat, nb)?;
        o.rms_norm(&ik, Some(p.vec(l, Role::IdxKNorm)?), &ik, nb, id, eps)?;
        o.rope(&ik, nb, 1, id, id, id - rd, rd, j0 * r, r, rope, false)?;
        o.quant_sim(&ik, nb, id, 32, 1)?;
        o.rows_f16(&ik, &c.idx, nb, j0, id)?;
        o.rope(&lat, nb, 1, hd, hd, hd - rd, rd, j0 * r, r, rope, false)?;
        o.quant_sim(&lat, nb, hd, 16, 2)?;
        tap(&format!("comp-{l}"), &lat.view(0, nb * hd * 4)?)?;
        o.rows_f16(&lat, &c.cache, nb, j0, hd)
    }

    /// Index source l's picks for the rows of a pass at pos0 (x their attention input, qr their q LoRA) into p.sel:
    /// false when every row reads all it sees (no more than top_k). The candidate source keeps each row's best blocks
    /// (p.cand) for the index sources after it, which pick inside them.
    #[allow(clippy::too_many_arguments)]
    fn select(&self, p: &Part, l: u64, x: &DevBuf, qr: &DevBuf, c: &Comp, pos0: usize, t: usize, sh: &mut Shared, tap: Tap) -> Result<bool> {
        let g = &self.m.g;
        let o = &p.ops;
        let (id, ih, kp, rd) = (g.idx_dim as usize, g.idx_heads as usize, g.idx_top_k as usize, g.rope_dim as usize);
        let r = c.ratio;
        let n = (pos0 + t) / r;
        let is_cand = g.cand_source == Some(l);
        if is_cand {
            sh.cand = false;
        }
        if n <= kp {
            return Ok(false);
        }
        let (bs, ct) = (g.cand_block as usize, g.cand_top as usize);
        let nb = n.div_ceil(bs);
        let use_cand = !is_cand && sh.cand && g.cand_source.is_some_and(|s| l > s);
        let rope = self.m.rope(l);
        let compact = use_cand && nb > ct && idx_cand() && ih == 32 && id == 128 && bs == 8 && n >= 8;
        let rows = (IDX_S_FLOATS / if compact { ct * bs } else { n }).clamp(1, t);
        for q0 in (0..t).step_by(rows) {
            let qn = rows.min(t - q0);
            let mk = p.arena.mark();
            let iq = p.mm(l, Role::IdxQB, &qr.view(q0 * g.q_lora as usize * 4, qn * g.q_lora as usize * 4)?, qn)?;
            o.rope(&iq, qn, ih, ih * id, id, id - rd, rd, pos0 + q0, 1, rope, false)?;
            o.quant_sim(&iq, qn * ih, id, 32, 1)?;
            // the head weights' positive scale ranks nothing differently: left out
            let w = p.mm(l, Role::IdxProj, &x.view(q0 * g.n_embd as usize * 4, qn * g.n_embd as usize * 4)?, qn)?;
            if compact {
                // only the candidate blocks scored (the masked form scored every key of the context)
                let cv = p.cand.view(q0 * ct * 4, qn * ct * 4)?;
                let sc = p.arena.f32(qn * ct * bs)?;
                o.idx_score_cand(&iq, &w, &c.idx, &cv, ct, &sc, qn, ih, id, n, pos0 + q0, r)?;
                let selc = p.arena.bytes(qn * kp * 4)?;
                o.topk(&sc, &selc, qn, ct * bs, ct * bs, kp)?;
                o.cand_pos(&selc, &cv, &p.sel.view(q0 * kp * 4, qn * kp * 4)?, qn, ct, kp, bs)?;
                p.arena.rewind(mk);
                continue;
            }
            let score = p.arena.f32(qn * n)?;
            self.timed(p, "indexer: scores (every key)", || o.idx_score(&iq, &w, &c.idx, &score, qn, ih, id, n, n, pos0 + q0, r))?;
            if !use_cand && nb > ct && ct >= kp && n >= idx_two_min() && ih == 32 && id == 128 && bs == 8 {
                // two levels: the blocks' maxima, the best ct blocks (the candidate source's candidates too), their keys
                // scored again (the same sums: a block one tile, as in the full scoring), the top-k among those. A key of
                // the top k lies in a block whose maximum is at least the k-th score, and at most k <= ct blocks have
                // one: the same picks
                let bmax = p.arena.f32(qn * nb)?;
                o.block_max(&score, &bmax, qn, n, n, bs, pos0 + q0, r)?;
                let own = if is_cand { None } else { Some(p.arena.bytes(qn * ct * 4)?) };
                let cv = match &own {
                    Some(b) => b.view(0, qn * ct * 4)?,
                    None => p.cand.view(q0 * ct * 4, qn * ct * 4)?,
                };
                o.topk(&bmax, &cv, qn, nb, nb, ct)?;
                if is_cand {
                    sh.cand = true;
                }
                if qn == t {
                    tap(&format!("indexer_score-{l}"), &score)?;
                }
                let sc = p.arena.f32(qn * ct * bs)?;
                o.idx_score_cand(&iq, &w, &c.idx, &cv, ct, &sc, qn, ih, id, n, pos0 + q0, r)?;
                let selc = p.arena.bytes(qn * kp * 4)?;
                o.topk(&sc, &selc, qn, ct * bs, ct * bs, kp)?;
                o.cand_pos(&selc, &cv, &p.sel.view(q0 * kp * 4, qn * kp * 4)?, qn, ct, kp, bs)?;
                p.arena.rewind(mk);
                continue;
            }
            if is_cand && nb > ct {
                let bmax = p.arena.f32(qn * nb)?;
                o.block_max(&score, &bmax, qn, n, n, bs, pos0 + q0, r)?;
                o.topk(&bmax, &p.cand.view(q0 * ct * 4, qn * ct * 4)?, qn, nb, nb, ct)?;
                sh.cand = true;
            } else if use_cand && nb > ct {
                let mark = p.arena.bytes(qn * nb)?;
                o.cand_mask(&score, &p.cand.view(q0 * ct * 4, qn * ct * 4)?, &mark, qn, n, n, ct, bs)?;
            }
            if qn == t {
                tap(&format!("indexer_score-{l}"), &score)?;
            }
            self.timed(p, "indexer: top-k (every key)", || o.topk(&score, &p.sel.view(q0 * kp * 4, qn * kp * 4)?, qn, n, n, kp))?;
            p.arena.rewind(mk);
        }
        Ok(true)
    }

    /// The attention half of layer l on x [t, d] (normed) at pos0: the window's keys into the ring, a source's
    /// compressor, an index source's picks, then per block of query rows the attention and the grouped output
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, p: &Part, l: u64, x: &DevBuf, layers: &[Att], pos0: usize, t: usize, sh: &mut Shared, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let eps = g.rms_eps as f32;
        let (nh, hd, rd, ql) = (g.n_head as usize, g.head_dim as usize, g.rope_dim as usize, g.q_lora as usize);
        let (ng, ol) = (g.o_groups as usize, g.o_lora as usize);
        let gw = nh * hd / ng;
        let rope = self.m.rope(l);
        let st = &layers[l as usize];
        let (qr, kv) = p.mm2((l, Role::QA), (l, Role::Kv), x, t)?;
        o.rms_norm(&qr, Some(p.vec(l, Role::QANorm)?), &qr, t, ql, eps)?;
        let rr = raw_ring();
        if kv_fused() {
            o.kv_post(&kv, p.vec(l, Role::KvNorm)?, t, hd, eps, rd, pos0, rope, &st.raw, rr)?;
            tap(&format!("kv-{l}"), &kv)?;
        } else {
            o.rms_norm(&kv, Some(p.vec(l, Role::KvNorm)?), &kv, t, hd, eps)?;
            o.rope(&kv, t, 1, hd, hd, hd - rd, rd, pos0, 1, rope, false)?;
            o.quant_sim(&kv, t, hd, 32, 0)?;
            tap(&format!("kv-{l}"), &kv)?;
            o.ring_put(&kv, &st.raw, t, pos0, rr, hd)?;
        }
        // the compressed keys this layer reads: its source's
        let src = g.kv_source(l);
        if let Some(c) = &st.comp {
            self.timed(p, "attention: compressor", || self.compress(p, l, x, c, pos0, t, &mut *tap))?;
        }
        let comp = src.and_then(|s| {
            let a = &layers[s as usize];
            if self.owner[s as usize] != self.owner[l as usize] { a.mirror.as_ref() } else { a.comp.as_ref() }
        });
        if g.index_sources.contains(&l) {
            let c = comp.ok_or("an index source without its kv source's keys")?;
            let name = if g.cand_source == Some(l) { "attention: indexer (candidate source)" } else { "attention: indexer" };
            sh.sel = self.timed(p, name, || self.select(p, l, x, &qr, c, pos0, t, sh, &mut *tap))?;
        }
        let kp = g.idx_top_k as usize;
        let rb = if t <= MMVQ_COLS { t } else { 1024.min(t) };
        let oa = p.arena.f32(t * ng * ol)?;
        let q = p.arena.f32(rb * nh * hd)?;
        let att = p.arena.f32(rb * nh * hd)?;
        let og = p.arena.f32(rb * ng * ol)?;
        let sinks = p.vec(l, Role::Sinks)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let qb = p.mat(l, Role::QB)?;
        for r0 in (0..t).step_by(rb) {
            let tr = rb.min(t - r0);
            p.matmul(qb, tr, (&qr, r0 * ql, ql), (&q, 0, nh * hd), false)?;
            o.rope(&q, tr, nh, nh * hd, hd, hd - rd, rd, pos0 + r0, 1, rope, false)?;
            if r0 == 0 && tr == t {
                tap(&format!("q-{l}"), &q)?;
            }
            let compv = comp.map(|c| (&c.cache, c.ratio, c.cache.len / (512 * 2)));
            let sel = if comp.is_some() && sh.sel { Some((p.sel.view(r0 * kp * 4, tr * kp * 4)?, kp)) } else { None };
            self.timed(p, "attention: attend", || {
                o.attend(&q, &st.raw, rr, compv, sel.as_ref().map(|(s, k)| (s, *k)), sinks, &att, tr, nh, pos0 + r0, g.window as usize, scale)
            })?;
            if kv_fused() {
                o.att_post(&att, tr, nh, hd, rd, pos0 + r0, rope)?;
            } else {
                o.rope(&att, tr, nh, nh * hd, hd, hd - rd, rd, pos0 + r0, 1, rope, true)?;
                if r0 == 0 && tr == t {
                    tap(&format!("attn_derope-{l}"), &att)?;
                }
                o.quant_sim(&att, tr * nh, hd, 32, 0)?;
            }
            self.out_a(p, l, &att, &q, &og, &oa, r0, tr)?;
        }
        let out = p.mm(l, Role::OutB, &oa, t)?;
        let _ = gw;
        tap(&format!("attn_out-{l}"), &out)?;
        Ok(out)
    }

    /// The grouped o_a: att [tr][groups][gw] -> oa rows from r0 ([t][groups][o_lora]); `perm` / `og` scratch
    #[allow(clippy::too_many_arguments)]
    fn out_a(&self, p: &Part, l: u64, att: &DevBuf, perm: &DevBuf, og: &DevBuf, oa: &DevBuf, r0: usize, tr: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (ng, ol, gw) = (g.o_groups as usize, g.o_lora as usize, (g.n_head * g.head_dim / g.o_groups) as usize);
        match p.groups.get(&l) {
            // one decode row, a Q8_0 / Q6_K o_a: all the groups in one launch of the std kernel (each group an entry: its rows'
            // pointer, input vector g of att's row, output slice g of oa's) - eight launches and two permutes before
            Some(_) if tr == 1 && std_oa() && o.std_supported(p.mat(l, Role::OutA)?.ty.code(), gw) => {
                let w = p.mat(l, Role::OutA)?;
                let tab = {
                    let mut m = p.oa_tab.lock().unwrap();
                    match m.get(&l) {
                        Some(t) => t.clone(),
                        None => {
                            let rb = w.ty.bytes(w.cols as u64).ok_or("o_a: a row of whole blocks")? as usize;
                            let mut tb: Vec<u8> = (0..ng).flat_map(|gi| (w.buf.ptr() as u64 + (gi * ol * rb) as u64).to_le_bytes()).collect();
                            for gi in 0..ng as i32 {
                                for v in [1i32, gi, gi, 0, 0, 0, 0, 0, 0, 0] {
                                    tb.extend(v.to_le_bytes());
                                }
                            }
                            let t = DevBuf::new(&o.gpu, tb.len())?;
                            t.write(0, &tb)?;
                            let t = Arc::new(t);
                            m.insert(l, t.clone());
                            t
                        }
                    }
                };
                o.std_mmv(w.ty.code(), &tab, ng, att, (oa, r0 * ng * ol), ng, ol, gw, 1)
            }
            Some(_) if tr < dense_f16_min() && q4k_own() && p.mat(l, Role::OutA)?.ty == GType::Q4K => {
                // every group in one launch: rows [g o_lora, ..) read their heads [g gw, ..) of att's row; 8 rows of att
                // a launch
                let w = p.mat(l, Role::OutA)?;
                let mut a = 0;
                while a < tr {
                    let n = MMVQ_COLS.min(tr - a);
                    o.q4k_mv(&w.buf, w.rows, w.cols, (att, a * ng * gw, ng * gw), Some((ol, gw)), (oa, (r0 + a) * ng * ol, ng * ol), n)?;
                    a += n;
                }
                Ok(())
            }
            Some(gm) => {
                let perm = perm.view(0, tr * ng * gw * 4)?;
                o.permute(att, &perm, tr, ng, gw)?;
                for (gi, w) in gm.iter().enumerate() {
                    p.matmul(w, tr, (&perm, gi * tr * gw, gw), (og, gi * tr * ol, ol), false)?;
                }
                o.permute(og, &oa.view(r0 * ng * ol * 4, tr * ng * ol * 4)?, ng, tr, ol)
            }
            None if tr <= MMVQ_COLS => {
                // fp8 o_a (the towers'), every group in one launch as the q4_K one
                let w = p.mat(l, Role::OutA)?;
                o.fp8_mmv(&w.buf, w.rows, w.cols, (att, 0, ng * gw), Some((ol, gw)), (oa, r0 * ng * ol, ng * ol), tr)
            }
            None => {
                // fp8 o_a (the towers'): each group's rows [g * o_lora, ..) expanded (whole 32-row tiles) against its heads
                let w = p.mat(l, Role::OutA)?;
                let w16 = p.arena.bytes(ol * gw * 2)?;
                let x16 = p.arena.bytes(tr * gw * 2)?;
                let perm = perm.view(0, tr * ng * gw * 4)?;
                o.permute(att, &perm, tr, ng, gw)?;
                for gi in 0..ng {
                    o.fp8_dequant(&w.buf, 0, w.rows, w.cols, gi * ol, ol, &w16, true)?;
                    o.to_f16(&perm.view(gi * tr * gw * 4, tr * gw * 4)?, &x16, tr * gw)?;
                    o.gemm_f16(tr, ol, gw, (&x16, 0, gw), (&w16, 0), (og, gi * tr * ol, ol), false)?;
                }
                o.permute(og, &oa.view(r0 * ng * ol * 4, tr * ng * ol * 4)?, ng, tr, ol)
            }
        }
    }

    /// The mixes of a half of block l on the streams x: post, comb and this half's pre (for the half after); the
    /// half's input collapsed by `pre_in` (the half before's), normed into `normed`
    #[allow(clippy::too_many_arguments)]
    fn hc_half(&self, p: &Part, l: u64, fn_: Role, base: Role, scale: Role, norm: Role, x: &DevBuf, pre_in: &DevBuf, ws: &[DevBuf; 6], t: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let [h, normed, post, comb, pre_out, mixes] = ws;
        let w = p.mat(l, fn_)?;
        if !w.f32() || w.rows != 24 {
            return Err(Error(format!("block {l}: the mHC mixes are float32 [24, 4 d]")));
        }
        if t <= MMVQ_COLS && hc_fused() && fused() {
            // decode widths: the mixes, the Sinkhorn, h by the half before's mix and its norm in two launches
            return o.hc_pre_in(x, pre_in, &w.buf, p.vec(l, scale)?, p.vec(l, base)?, p.vec(l, norm)?, h, post, comb, pre_out, normed,
                               &p.arena.f32(t * 32 * 25)?, t, d, eps, g.hc_eps as f32, g.hc_iters as u32);
        }
        if t <= MMVQ_COLS && hc_fused() {
            // decode widths: the mixes and the Sinkhorn in glm5next's two fused launches (its h and norm, by this half's
            // own mix, are replaced below)
            o.hc_pre_fused(x, &w.buf, p.vec(l, scale)?, p.vec(l, base)?, p.vec(l, norm)?, h, post, comb, pre_out, normed, &p.arena.f32(t * 32 * 25)?, t, d,
                           eps, g.hc_eps as f32, g.hc_iters as u32)?;
        } else {
            o.hc_mix(x, &w.buf, mixes, &p.arena.f32(t * 32 * 25)?, t, 4 * d, eps)?;
            o.hc_pre(mixes, p.vec(l, scale)?, p.vec(l, base)?, x, h, post, comb, pre_out, t, d, g.hc_eps as f32, g.hc_iters as u32)?;
        }
        o.hc_collapse(x, pre_in, h, t, d)?;
        o.rms_norm(h, Some(p.vec(l, norm)?), normed, t, d, eps)
    }

    /// One run of layers (`run`: its part, first, end) over the carried state x ([t, 4, d] streams, [t, 4] mix) of the
    /// pass's `tokens` at pos0; the Engram's rows taken at its layers; the target layers' stream means kept for the towers
    #[allow(clippy::too_many_arguments)]
    fn run_layers(&self, layers: &[Att], run: (usize, u64, u64), x: DevBuf, pos0: usize, tokens: &[u32], eng: &[(usize, Vec<u64>, Arc<engram::Fetch>)],
                  mut mh: Option<&mut MtpS>, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let t = tokens.len();
        let d = g.n_embd as usize;
        let p = &self.parts[run.0];
        let o = &p.ops;
        let gpu = &o.gpu;
        let sz = t * 4 * d + t * 4;
        let mut ring = vec![x, DevBuf::f32(gpu, sz)?, DevBuf::f32(gpu, sz)?];
        let mut ix = 0usize;
        // h, normed, post, comb, pre_out, mixes; and the attention's pre (the FFN's input mix)
        let ws = [DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * 4)?, DevBuf::f32(gpu, t * 16)?, DevBuf::f32(gpu, t * 4)?,
                  DevBuf::f32(gpu, t * 24)?];
        let attn_pre = DevBuf::f32(gpu, t * 4)?;
        let mut sh = Shared::default();
        // a part after the candidate source on another GPU: its blocks (copied over) hold when that source kept them
        if let (Some(cs), Some(ms)) = (g.cand_source, self.mirror_src) {
            if cs == ms && run.1 > cs && self.owner[cs as usize] != run.0 {
                let n = (pos0 + t) / g.ratio(cs) as usize;
                sh.cand = n > g.idx_top_k as usize && n.div_ceil(g.cand_block as usize) > g.cand_top as usize;
            }
        }
        let keep = t.min(MTP_KEEP);
        for l in run.1..run.2 {
            let lt0 = Instant::now();
            p.arena.reset();
            let xs = ring[ix].view(0, t * 4 * d * 4)?;
            let pre_in = ring[ix].view(t * 4 * d * 4, t * 16)?;
            let (x1, x2) = (&ring[(ix + 1) % 3], &ring[(ix + 2) % 3]);
            tap(&format!("l_in-{l}"), &xs)?;
            if l > 0 {
                tap(&format!("pre_mix-{l}"), &pre_in)?;
            }
            if let Some(k) = g.engram_layers.iter().position(|e| *e == l) {
                let en = self.engram.as_ref().ok_or("the Engram is not loaded")?;
                let (_, rows, f) = eng.iter().find(|(li, _, _)| *li == k).ok_or("the Engram's rows were not fetched")?;
                let raw = self.timed(p, "engram: waiting for rows", || en.take(k, rows, f))?;
                self.timed(p, "engram", || self.engram_apply(p, l, &xs, &raw, tokens, &mut *tap))?;
            }
            if let (Some(ms), Some(k)) = (mh.as_deref_mut(), g.mtp_targets.iter().position(|x| *x == l)) {
                let mean = p.arena.f32(t * d)?;
                o.hc_mean(&xs, &mean, t, d)?;
                o.kcopy(&ms.mh, (k * MTP_KEEP) * d * 4, &mean, (t - keep) * d * 4, keep * d * 4)?;
            }
            let (normed, post, comb) = (&ws[1], &ws[2], &ws[3]);
            // each half's pre written where it is read next (no copies): the attention half's into attn_pre, the FFN
            // half's into the next streams' tail (it rides with them to the next layer and part)
            let v = |b: &DevBuf| b.view(0, b.len);
            let ws_a = [v(&ws[0])?, v(&ws[1])?, v(&ws[2])?, v(&ws[3])?, v(&attn_pre)?, v(&ws[5])?];
            let ws_f = [v(&ws[0])?, v(&ws[1])?, v(&ws[2])?, v(&ws[3])?, x2.view(t * 4 * d * 4, t * 16)?, v(&ws[5])?];
            self.timed(p, "hc", || self.hc_half(p, l, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::AttnNorm, &xs, &pre_in, &ws_a, t))?;
            tap(&format!("attn_norm-{l}"), normed)?;
            let att = self.timed(p, "attention", || self.attention(p, l, normed, layers, pos0, t, &mut sh, &mut *tap))?;
            if self.mirror_src == Some(l) {
                self.timed(p, "mirror", || self.mirror_up(l, &layers[l as usize], pos0, t, sh.cand))?;
            }
            o.hc_post(&att, &xs, post, comb, x1, t, d)?;
            tap(&format!("hc_attn_post-{l}"), &x1.view(0, t * 4 * d * 4)?)?;
            self.timed(p, "hc", || self.hc_half(p, l, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, Role::FfnNorm, x1, &attn_pre, &ws_f, t))?;
            tap(&format!("ffn_norm-{l}"), normed)?;
            let split = (t > MMVQ_COLS && pipe_split()).then(|| -> Result<Instant> {
                gpu.sync()?;
                Ok(Instant::now())
            }).transpose()?;
            let ffn = self.moe(p, l, t, normed, tokens, &mut *tap)?;
            if let Some(t1) = split {
                gpu.sync()?;
                let (a, m) = (t1.duration_since(lt0).as_secs_f64(), t1.elapsed().as_secs_f64());
                eprintln!("[layer {l} ({t} tokens): attention and the rest {:.1} ms, experts {:.1} ms]", a * 1e3, m * 1e3);
            }
            tap(&format!("ffn_out-{l}"), &ffn)?;
            o.hc_post(&ffn, x1, post, comb, x2, t, d)?;
            tap(&format!("l_out-{l}"), &x2.view(0, t * 4 * d * 4)?)?;
            ix = (ix + 2) % 3;
        }
        Ok(ring.swap_remove(ix))
    }

    /// The head on the last part for the last `n_out` rows of x: the streams collapsed by the carried mix, the norm,
    /// the output matrix
    fn head(&self, x: &DevBuf, t: usize, n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let g = &self.m.g;
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        p.arena.reset();
        let xv = x.view((t - n_out) * 4 * d * 4, n_out * 4 * d * 4)?;
        let pre = x.view((t * 4 * d + (t - n_out) * 4) * 4, n_out * 16)?;
        let h = p.arena.f32(n_out * d)?;
        o.hc_collapse(&xv, &pre, &h, n_out, d)?;
        o.rms_norm(&h, Some(p.vec(0, Role::OutputNorm)?), &h, n_out, d, eps)?;
        tap("result_norm", &h)?;
        let logits = p.mm(0, Role::Output, &h, n_out)?;
        tap("result_output", &logits)?;
        let v = p.read_f32(&logits, n_out * g.n_vocab as usize)?;
        Ok(v.chunks(g.n_vocab as usize).map(|c| c.to_vec()).collect())
    }

    /// The layers in order as runs on one part: (part, first layer, end)
    fn runs(&self) -> Vec<(usize, u64, u64)> {
        let mut v: Vec<(usize, u64, u64)> = Vec::new();
        for (l, &pi) in self.owner.iter().enumerate() {
            match v.last_mut() {
                Some(r) if r.0 == pi => r.2 = l as u64 + 1,
                _ => v.push((pi, l as u64, l as u64 + 1)),
            }
        }
        v
    }

    /// The Engram's rows of the pass's tokens (positions pos0..), on their way: (engram layer, rows, the fetch)
    fn engram_start(&self, hist: &[u32], pos0: usize) -> Vec<(usize, Vec<u64>, Arc<engram::Fetch>)> {
        let Some(en) = &self.engram else { return Vec::new() };
        en.rows(hist, pos0).into_iter().enumerate().map(|(k, rows)| {
            let f = en.fetch(k, &rows);
            (k, rows, f)
        }).collect()
    }

    pub fn feed(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.feed_until(sess, tokens, &|_| false, tap)?.1)
    }

    pub fn feed_until(&self, sess: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        let chunks = chunked(tokens, prefill_chunk());
        let runs = self.runs();
        let out = if chunks.len() >= 2 && runs.len() == 2 && pipeline() {
            self.feed_pipelined(sess, &chunks, (runs[0], runs[1]), stop, tap)?
        } else {
            let (mut fed, mut logits) = (0, Vec::new());
            for (i, c) in chunks.iter().enumerate() {
                if i > 0 && stop(fed) {
                    break;
                }
                logits = self.forward(sess, c, &mut *tap)?;
                fed += c.len();
            }
            (fed, logits)
        };
        // what is generated after a prompt is DRY's history
        sess.gen_from = sess.pos;
        Ok(out)
    }

    /// `logits` as the sampler sees them: DRY's penalties (NS_DS41_DRY) against this turn's generated tokens `hist`
    fn shaped<'a>(&self, logits: &'a [f32], hist: &[u32]) -> std::borrow::Cow<'a, [f32]> {
        match &self.dry {
            None => std::borrow::Cow::Borrowed(logits),
            Some((d, br)) => {
                let mut l = logits.to_vec();
                crate::dry::penalize(&mut l, hist, br, d);
                std::borrow::Cow::Owned(l)
            }
        }
    }

    /// `feed` on two GPUs with the chunks in a pipeline: a thread runs the first GPU's layers a chunk ahead while this
    /// one runs the second's, the head and the towers' push. Each GPU keeps the chunks' order: the same result.
    fn feed_pipelined(&self, sess: &mut Session, chunks: &[&[u32]], runs: ((usize, u64, u64), (usize, u64, u64)), stop: &(dyn Fn(usize) -> bool + Sync),
                      tap: Tap) -> Result<(usize, Vec<f32>)> {
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        if sess.pos + total > sess.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", sess.max_ctx, total)));
        }
        let rows = chunks.iter().map(|c| c.len()).max().unwrap_or(0);
        for p in &self.parts {
            p.arena_for(rows)?;
        }
        let start = sess.pos;
        sess.hist.truncate(start);
        for c in chunks {
            sess.hist.extend_from_slice(c);
        }
        let Session { layers, hist, mtp, .. } = sess;
        let layers: &[Att] = layers;
        let hist: &[u32] = hist;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<u8>>>(1);
        self.pipelining.store(true, std::sync::atomic::Ordering::Relaxed);
        self.cand_q.lock().unwrap().clear();
        let done = std::thread::scope(|sc| -> Result<(usize, Vec<f32>)> {
            let ahead = sc.spawn(move || {
                let mut none = |_: &str, _: &DevBuf| Ok(());
                let mut at = start;
                // each chunk's Engram rows asked for a chunk ahead (the reads of a chunk's rows outlast its first layer)
                let mut next = chunks.first().map(|c| self.engram_start(&hist[..at + c.len()], at));
                for (i, c) in chunks.iter().enumerate() {
                    if i > 0 && stop(chunks[..i].iter().map(|c| c.len()).sum()) {
                        return;
                    }
                    let t0 = Instant::now();
                    let eng = next.take().unwrap_or_default();
                    if let Some(n) = chunks.get(i + 1) {
                        next = Some(self.engram_start(&hist[..at + c.len() + n.len()], at + c.len()));
                    }
                    let r = self.streams(runs.0.0, c, &mut none).and_then(|x| self.run_layers(layers, runs.0, x, at, c, &eng, None, &mut none)).and_then(|x| {
                        let mut v = vec![0u8; x.len];
                        x.read(0, &mut v)?;
                        Ok(v)
                    });
                    if pipe_log() {
                        eprintln!("[pipeline: chunk {i} ({} tokens) on the first GPU in {:.3} s]", c.len(), t0.elapsed().as_secs_f64());
                    }
                    let failed = r.is_err();
                    if tx.send(r).is_err() || failed {
                        return;
                    }
                    at += c.len();
                }
            });
            let mut out = Ok(Vec::new());
            let mut at = start;
            let p1 = &self.parts[runs.1.0];
            for c in chunks.iter() {
                let tw = Instant::now();
                let Ok(v) = rx.recv() else { break };
                let waited = tw.elapsed().as_secs_f64();
                let step = (|| -> Result<Vec<f32>> {
                    let v = v?;
                    if self.mirror_src.is_some() {
                        let blocks = self.cand_q.lock().unwrap().pop_front().unwrap_or_default();
                        self.cand_in(&blocks)?;
                    }
                    let x = DevBuf::new(&p1.ops.gpu, v.len())?;
                    x.write(0, &v)?;
                    let eng = Vec::new();
                    let x = self.run_layers(layers, runs.1, x, at, c, &eng, mtp.as_mut(), &mut *tap)?;
                    let l = self.head(&x, c.len(), 1, &mut *tap)?.pop().unwrap_or_default();
                    if let Some(ms) = mtp.as_mut() {
                        self.mtp_push(ms, at, c.len(), false)?;
                    }
                    Ok(l)
                })();
                if pipe_log() {
                    eprintln!("[pipeline: chunk at {at} on the second GPU in {:.3} s after waiting {waited:.3} s]", tw.elapsed().as_secs_f64() - waited);
                }
                match step {
                    Ok(l) => out = Ok(l),
                    Err(e) => {
                        out = Err(e);
                        break;
                    }
                }
                at += c.len();
            }
            drop(rx);
            ahead.join().map_err(|_| Error("the pipeline's first stage panicked".into()))?;
            out.map(|l| (at - start, l))
        });
        self.pipelining.store(false, std::sync::atomic::Ordering::Relaxed);
        let done = done?;
        sess.pos += done.0;
        sess.hist.truncate(sess.pos);
        sess.snapped = None;
        Ok(done)
    }

    pub fn forward(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.forward_rows(sess, tokens, 1, tap)?.pop().unwrap_or_default())
    }

    /// One pass of at most a chunk: the logits of the last `n_out` tokens (1..=8). The towers' rings take the pass's
    /// rows (a verify pass's overwritten slots kept for `rollback`).
    pub fn forward_rows(&self, sess: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        if t == 0 || t > prefill_chunk() || n_out == 0 || n_out > t.min(MMVQ_COLS) {
            return Err(Error(format!("forward: {t} tokens, {n_out} outputs")));
        }
        for p in &self.parts {
            p.arena_for(t)?;
        }
        let pos0 = sess.pos;
        if pos0 + t > sess.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", sess.max_ctx, t)));
        }
        sess.hist.truncate(pos0);
        sess.hist.extend_from_slice(tokens);
        let tw = Instant::now();
        let eng = self.engram_start(&sess.hist, pos0);
        let mut x = self.streams(0, tokens, &mut *tap)?;
        WALL_STREAMS_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let tw = Instant::now();
        let mut staging = Vec::new();
        for (k, run) in self.runs().into_iter().enumerate() {
            if k > 0 {
                x = self.to_part(run.0, &x, &mut staging)?;
            }
            let last = k + 1 == self.parts.len();
            x = self.run_layers(&sess.layers, run, x, pos0, tokens, &eng, if last { sess.mtp.as_mut() } else { None }, &mut *tap)?;
        }
        WALL_LAYERS_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let tw = Instant::now();
        let out = self.head(&x, t, n_out, &mut *tap)?;
        WALL_HEAD_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if let Some(ms) = sess.mtp.as_mut() {
            self.mtp_push(ms, pos0, t, (2..=MAX_VERIFY).contains(&t))?;
        }
        sess.pos += t;
        sess.snapped = (2..=MAX_VERIFY).contains(&t).then_some((pos0, t));
        Ok(out)
    }

    /// Keeps only the first `keep` tokens of the last (verify) pass: the position back, the towers' ring slots the
    /// rest overwrote restored (the main caches need nothing: positions bound what is read)
    pub fn rollback(&self, sess: &mut Session, keep: usize) -> Result<()> {
        let (pos0, t) = sess.snapped.take().ok_or("rollback: the last pass was not a verify pass")?;
        if keep == 0 || keep >= t {
            return Err(Error(format!("rollback: keep {keep} of {t} rows")));
        }
        sess.pos = pos0 + keep;
        sess.hist.truncate(sess.pos);
        if let Some(ms) = sess.mtp.as_mut() {
            if let Some((p0, n)) = ms.snap_at.take() {
                let p = self.parts.last().unwrap();
                for (ti, ring) in ms.rings.iter().enumerate() {
                    for i in keep..n {
                        let slot = (p0 + i) % MTP_WIN;
                        p.ops.kcopy(ring, slot * 1024, &ms.snap, (ti * MAX_VERIFY + i) * 1024, 1024)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The pass's rows (their stream means at the target layers, kept in ms.mh) into each tower's window: main_x =
    /// main_norm(main_proj(the means)), each tower's key kv_norm(wkv(main_x)) rotated at the row's position, fp8;
    /// with `snap`, the slots overwritten kept first
    fn mtp_push(&self, ms: &mut MtpS, pos0: usize, t: usize, snap: bool) -> Result<()> {
        let g = &self.m.g;
        let (d, hd, rd, eps) = (g.n_embd as usize, g.head_dim as usize, g.rope_dim as usize, g.rms_eps as f32);
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        let keep = t.min(MTP_KEEP);
        let first = pos0 + t - keep;
        ms.mh_rows = (first, keep);
        let mk = p.arena.mark();
        // [3][keep][d] -> [keep][3][d]
        let cat = p.arena.f32(keep * 3 * d)?;
        let src = if keep == MTP_KEEP {
            ms.mh.view(0, 3 * keep * d * 4)?
        } else {
            let s = p.arena.f32(3 * keep * d)?;
            for k in 0..3 {
                o.kcopy(&s, k * keep * d * 4, &ms.mh, k * MTP_KEEP * d * 4, keep * d * 4)?;
            }
            s
        };
        o.permute(&src, &cat, 3, keep, d)?;
        let mx = p.mm(0, Role::MtpMainProj, &cat, keep)?;
        o.rms_norm(&mx, Some(p.vec(0, Role::MtpMainNorm)?), &mx, keep, d, eps)?;
        // only the last 128 matter to a window
        let (n, skip) = (keep.min(MTP_WIN), keep.saturating_sub(MTP_WIN));
        let mxv = mx.view(skip * d * 4, n * d * 4)?;
        ms.snap_at = None;
        if snap && n <= MAX_VERIFY {
            for (ti, ring) in ms.rings.iter().enumerate() {
                for i in 0..n {
                    let slot = (first + skip + i) % MTP_WIN;
                    o.kcopy(&ms.snap, (ti * MAX_VERIFY + i) * 1024, ring, slot * 1024, 1024)?;
                }
            }
            ms.snap_at = Some((first + skip, n));
        }
        for (ti, ring) in ms.rings.iter().enumerate() {
            let l = g.n_layer + ti as u64;
            let kv = p.mm(l, Role::Kv, &mxv, n)?;
            o.rms_norm(&kv, Some(p.vec(l, Role::KvNorm)?), &kv, n, hd, eps)?;
            o.rope(&kv, n, 1, hd, hd, hd - rd, rd, first + skip, 1, self.m.rope(l), false)?;
            o.quant_sim(&kv, n, hd, 32, 0)?;
            o.ring_put(&kv, ring, n, first + skip, MTP_WIN, hd)?;
        }
        p.arena.rewind(mk);
        Ok(())
    }

    /// The towers' draft after the committed token `tail` (fed next, at position `pos`): the block [tail, noise x 4]
    /// through the three towers, the head, the Markov bias token by token; up to `k` tokens for the positions after
    /// `tail`'s
    fn mtp_draft(&self, ms: &MtpS, tail: u32, pos: usize, k: usize, tap: Tap) -> Result<Vec<u32>> {
        let g = &self.m.g;
        let (d, hd, rd, eps) = (g.n_embd as usize, g.head_dim as usize, g.rope_dim as usize, g.rms_eps as f32);
        let (nh, ql) = (g.n_head as usize, g.q_lora as usize);
        let b = g.mtp_block as usize;
        let pi = self.parts.len() - 1;
        let p = &self.parts[pi];
        let o = &p.ops;
        let t0 = Instant::now();
        let mut toks = vec![g.mtp_noise; b];
        toks[0] = tail;
        let mut x = self.streams(pi, &toks, &mut *tap)?;
        let sz = b * 4 * d + b * 4;
        let x2 = DevBuf::f32(&o.gpu, sz)?;
        let ws = [DevBuf::f32(&o.gpu, b * d)?, DevBuf::f32(&o.gpu, b * d)?, DevBuf::f32(&o.gpu, b * 4)?, DevBuf::f32(&o.gpu, b * 16)?, DevBuf::f32(&o.gpu, b * 4)?,
                  DevBuf::f32(&o.gpu, b * 24)?];
        let attn_pre = DevBuf::f32(&o.gpu, b * 4)?;
        let x1 = DevBuf::f32(&o.gpu, sz)?;
        let n_ring = pos.min(MTP_WIN);
        // every block row reads the whole window and the whole block: the keys as one list, picked all by each row
        let nk = n_ring + b;
        let mut sel = Vec::with_capacity(b * nk);
        for _ in 0..b {
            sel.extend((0..nk as i32).collect::<Vec<_>>());
        }
        let selb = DevBuf::new(&o.gpu, b * nk * 4)?;
        p.upload(&selb, &i32s(&sel))?;
        let dummy = DevBuf::new(&o.gpu, 1024)?;
        for (ti, ring) in ms.rings.iter().enumerate() {
            let l = g.n_layer + ti as u64;
            p.arena.reset();
            let xs = x.view(0, b * 4 * d * 4)?;
            let pre_in = x.view(b * 4 * d * 4, b * 16)?;
            let (normed, post, comb, pre_out) = (&ws[1], &ws[2], &ws[3], &ws[4]);
            self.timed(p, "draft: hc", || self.hc_half(p, l, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::AttnNorm, &xs, &pre_in, &ws, b))?;
            o.kcopy(&attn_pre, 0, pre_out, 0, b * 16)?;
            let ta = Instant::now();
            // the attention: q and the block's keys from the block, the window's from the ring
            let rope = self.m.rope(l);
            let qr = p.mm(l, Role::QA, normed, b)?;
            o.rms_norm(&qr, Some(p.vec(l, Role::QANorm)?), &qr, b, ql, eps)?;
            let q = p.mm(l, Role::QB, &qr, b)?;
            o.rope(&q, b, nh, nh * hd, hd, hd - rd, rd, pos, 1, rope, false)?;
            let kv = p.mm(l, Role::Kv, normed, b)?;
            o.rms_norm(&kv, Some(p.vec(l, Role::KvNorm)?), &kv, b, hd, eps)?;
            o.rope(&kv, b, 1, hd, hd, hd - rd, rd, pos, 1, rope, false)?;
            o.quant_sim(&kv, b, hd, 32, 0)?;
            // a spare row a block row past the list: the kernel's rows-seen check (each row sees nk + 1.. of them)
            let keys = p.arena.bytes((nk + b + 1) * hd * 2)?;
            if n_ring > 0 {
                o.kcopy(&keys, 0, ring, 0, n_ring * hd * 2)?;
            }
            o.rows_f16(&kv, &keys, b, n_ring, hd)?;
            let att = p.arena.f32(b * nh * hd)?;
            // as the attention's compressed keys at ratio 1, every row seeing more than nk of them and picking all
            o.attend(&q, &dummy, 1, Some((&keys, 1, nk + b + 1)), Some((&selb, nk)), p.vec(l, Role::Sinks)?, &att, b, nh, nk, 0, 1.0 / (hd as f32).sqrt())?;
            o.rope(&att, b, nh, nh * hd, hd, hd - rd, rd, pos, 1, rope, true)?;
            o.quant_sim(&att, b * nh, hd, 32, 0)?;
            let oa = p.arena.f32(b * g.o_groups as usize * g.o_lora as usize)?;
            let og = p.arena.f32(b * g.o_groups as usize * g.o_lora as usize)?;
            self.out_a(p, l, &att, &q, &og, &oa, 0, b)?;
            let a = p.mm(l, Role::OutB, &oa, b)?;
            self.lap(p, "draft: attention", ta)?;
            o.hc_post(&a, &xs, post, comb, &x1, b, d)?;
            self.hc_half(p, l, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, Role::FfnNorm, &x1, &attn_pre, &ws, b)?;
            let ffn = self.timed(p, "draft: moe", || self.moe(p, l, b, normed, &[], &mut *tap))?;
            o.hc_post(&ffn, &x1, post, comb, &x2, b, d)?;
            o.kcopy(&x2, b * 4 * d * 4, pre_out, 0, b * 16)?;
            o.kcopy(&x, 0, &x2, 0, sz)?;
        }
        // the head: the streams by the last mix, the towers' norm, the model's output matrix
        p.arena.reset();
        let h = p.arena.f32(b * d)?;
        o.hc_collapse(&x.view(0, b * 4 * d * 4)?, &x.view(b * 4 * d * 4, b * 16)?, &h, b, d)?;
        o.rms_norm(&h, Some(p.vec(0, Role::MtpOutNorm)?), &h, b, d, eps)?;
        // only the rows a draft is taken from (row i: the token after position pos + i)
        let k = k.min(b);
        let logits = self.timed(p, "draft: head", || p.mm(0, Role::Output, &h, k))?;
        tap("mtp_logits", &logits)?;
        let vocab = g.n_vocab as usize;
        // the Markov bias, on the GPU: the previous token's (tail, then each draft) rank-256 embedding through its head,
        // added to its row, the argmax the next previous token
        let toks = p.arena.bytes((k + 1) * 4)?;
        p.upload(&toks, &(tail as i32).to_le_bytes())?;
        let emb = p.mat(0, Role::MtpMarkovEmbd)?;
        let me = p.arena.f32(emb.cols)?;
        for i in 0..k {
            o.gather_rows(&emb.buf, emb.rows, &toks, i * 4, 1, emb.cols, &me)?;
            let bias = p.mm(0, Role::MtpMarkovHead, &me, 1)?;
            o.markov_step((&logits, i * vocab), &bias, vocab, &toks, (i + 1) * 4)?;
        }
        let tv = p.read_f32(&toks, k + 1)?;
        let _ = (t0, &mut x);
        Ok(tv[1..].iter().map(|f| f.to_bits()).collect())
    }

    pub fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder { logits, next: None, drafts: Vec::new(), mtp: draft && self.mtp, drafted: 0, accepted: 0 }
    }

    pub fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder { logits: Vec::new(), next: Some(next), drafts: Vec::new(), mtp: draft && self.mtp, drafted: 0, accepted: 0 }
    }

    /// The next committed token(s): the pending token with the towers' drafts in one pass, each draft kept exactly when
    /// the token drawn for its position is the draft itself (what is committed is distributed as plain sampling)
    pub fn step(&self, sess: &mut Session, dec: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let room = sess.max_ctx.saturating_sub(sess.pos + 1);
        let (out, tail) = match dec.next {
            Some(x) if dec.mtp && !dec.drafts.is_empty() && room > 0 => {
                let mut dr = std::mem::take(&mut dec.drafts);
                dr.truncate(room.min(mtp_drafts()));
                let mut toks = vec![x];
                toks.extend_from_slice(&dr);
                let rows = self.forward_rows(sess, &toks, toks.len(), &mut *tap)?;
                let mut out = Vec::with_capacity(toks.len());
                // row i follows toks[..=i]: the history to there
                let at0 = sess.pos - toks.len();
                for (i, r) in rows.iter().enumerate() {
                    let gen = sess.gen_from.min(at0 + i + 1);
                    let y = smp.sample(&self.shaped(r, &sess.hist[gen..at0 + i + 1]));
                    out.push(y);
                    if i < dr.len() {
                        dec.drafted += 1;
                        if y == dr[i] {
                            dec.accepted += 1;
                            continue;
                        }
                    }
                    break;
                }
                if out.len() < toks.len() {
                    self.rollback(sess, out.len())?;
                }
                let tail = *out.last().unwrap();
                (out, tail)
            }
            _ => {
                if let Some(x) = dec.next.take() {
                    dec.logits = self.forward(sess, &[x], &mut *tap)?;
                }
                let ts = Instant::now();
                let y = smp.sample(&self.shaped(&dec.logits, &sess.hist[sess.gen_from.min(sess.pos)..sess.pos]));
                WALL_SAMPLE_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                (vec![y], y)
            }
        };
        dec.drafts = match (dec.mtp, sess.mtp.as_ref()) {
            (true, Some(ms)) if sess.pos + 2 <= sess.max_ctx => self.timed(self.parts.last().unwrap(), "draft", || self.mtp_draft(ms, tail, sess.pos, mtp_drafts(), &mut *tap))?,
            _ => Vec::new(),
        };
        dec.next = Some(tail);
        Ok(out)
    }

    /// One decode step of several conversations: each its own pass (the same result as alone)
    pub fn forward_batch(&self, sess: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != sess.len() {
            return Err(Error(format!("forward_batch: {} tokens for {} sessions", tokens.len(), sess.len())));
        }
        sess.iter_mut().zip(tokens).map(|(s, t)| self.forward(s, &[*t], &mut *tap)).collect()
    }

    /// The MoE half of block l on x [t, d]: the router (on the host), the shared expert, the routed experts grouped by
    /// expert - from their indices at decode widths, expanded to fp16 for a prompt
    /// `tokens`: the pass's (an image position routes by the layer's image bias); empty: no image positions
    fn moe(&self, p: &Part, l: u64, t: usize, x: &DevBuf, tokens: &[u32], tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let d = g.n_embd as usize;
        let f = g.ffn_expert as usize;
        let (ne, used) = { let (a, b) = g.experts(l); (a as usize, b as usize) };
        let lim = g.swiglu_limit as f32;
        let pf = prefetch_limit();
        let next_main = l + 1 < g.n_layer && !g.is_mtp(l) && p.layers.contains(&(l + 1));
        let (logits, guess) = if t <= MMVQ_COLS && pf > 0 && next_main {
            let (a, b) = p.mm2((l, Role::Router), (l + 1, Role::Router), x, t)?;
            (a, Some(b))
        } else {
            (p.mm(l, Role::Router, x, t)?, None)
        };
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        // decode: the routers of the layers after the next on this layer's input, for file reads further ahead
        let far: Vec<(u64, DevBuf)> = if t == 1 && guess.is_some() && file_ahead() > 0 {
            (2..=1 + far_layers() as u64).map(|j| l + j).filter(|&lj| lj < g.n_layer && !g.is_mtp(lj) && p.layers.contains(&lj))
                .map(|lj| Ok((lj, p.mm(lj, Role::Router, x, t)?))).collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        // one decode token routed on the GPU: its VRAM experts' products queued now, from the slot table; the host
        // reads the routing (ids, weights, which had no VRAM slot) after, for the rest and its bookkeeping
        let img_any = tokens.iter().any(|t| crate::vision::is_image(*t));
        let groute = t == 1 && gpu_route() && !self.mtp && !p.draft.contains_key(&l) && !p.stores_of(l).is_empty() && used <= 8 && !img_any;
        let mut early: Option<(DevBuf, DevBuf, DevBuf)> = None;
        let mut early_read = None;
        if groute {
            let vq = &p.vq[&l];
            // the table names an expert only once its copy up landed (tab_set_after): nothing to wait for here
            let dummy = p.stores[p.stores_of(l)[0]].lock().unwrap().vram.first().map_or(0, |b| b.ptr() as u64);
            let gtab = p.arena.bytes(2 * used * 48 + 33 * 4)?;
            let dtab = p.arena.bytes(used * 48 + used * 33 * 4)?;
            let info = p.arena.bytes(3 * used * 4)?;
            let ctab = p.arena.bytes((2 + 2 * used) * 4)?;
            let tab = p.devtab.view(l as usize * ne * 8, ne * 8)?;
            let tags = p.xtags.as_ref().map(|b| b.view(l as usize * ne * 4, ne * 4)).transpose()?;
            o.route1(&logits, p.vec(l, Role::RouterBias)?, ne, used, g.expert_scale as f32, &tab, dummy, (vq.gate, vq.up, vq.down), &gtab, &dtab, &info, &ctab, tags.as_ref())?;
            // the routing read back ahead of the products (the host's turn runs while they do)
            let mut down: Vec<(&DevBuf, usize)> = vec![(&info, 3 * used)];
            if let Some(gl) = &guess {
                down.push((gl, t * ne));
            }
            down.extend(far.iter().map(|(_, b)| (b, t * ne)));
            early_read = Some(p.read_start(&down)?);
            let (dn, gu, hset) = (p.arena.f32(used * d)?, p.arena.f32(2 * used * f)?, p.arena.f32(used * f)?);
            let nbit = vq.nbit;
            let pick = p.vq_pick.get(&nbit).copied().unwrap_or([0, 0]);
            p.exp_mmv(l, false, &gtab, 2 * used, x, (&gu, 0), 2 * used, f, d, pick[0], Some((1, 2 * used)), 1)?;
            o.swiglu_clamp(&gu.view(0, used * f * 4)?, &gu.view(used * f * 4, used * f * 4)?, &hset, used * f, lim)?;
            p.exp_mmv(l, true, &dtab, used, &hset, (&dn, 0), used, d, f, pick[1], Some((used, 1)), 1)?;
            early = Some((info, dn, ctab));
        }
        let pending = match early_read {
            Some(r) => r,
            None => {
                let mut down: Vec<(&DevBuf, usize)> = vec![(&logits, t * ne)];
                if let Some(gl) = &guess {
                    down.push((gl, t * ne));
                }
                down.extend(far.iter().map(|(_, b)| (b, t * ne)));
                p.read_start(&down)?
            }
        };
        // the shared expert, queued while the host picks the routed ones
        let y = {
            let (sg, su) = p.mm2((l, Role::ShGate), (l, Role::ShUp), x, t)?;
            o.swiglu_clamp(&sg, &su, &sg, t * f, lim)?;
            p.mm(l, Role::ShDown, &sg, t)?
        };
        let tw = Instant::now();
        let (lv, guess, farv) = match pending {
            Some((tk, spans)) => {
                let mut v = p.read_finish(tk, &spans)?.into_iter();
                (v.next().unwrap_or_default(), v.next(), v.collect::<Vec<_>>())
            }
            None => (logits.to_f32()?, match &guess { Some(gl) => Some(gl.to_f32()?), None => None },
                     far.iter().map(|(_, b)| b.to_f32()).collect::<Result<Vec<_>>>()?),
        };
        ROUTER_WAIT_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let turn = Instant::now();
        let bias = p.router_bias(l)?;
        // image positions: the layer's image bias (model.py: bias_vl)
        let bias_vl = if img_any { self.vision.as_ref().and_then(|v| v.bias_vl.get(&l).cloned()) } else { None };
        let th = Instant::now();
        let pick = |ti: usize| -> Vec<(usize, f32)> {
            let pr: Vec<f32> = lv[ti * ne..(ti + 1) * ne].iter().map(|z| sqrt_softplus(*z)).collect();
            let bias = match &bias_vl {
                Some(b) if tokens.get(ti).is_some_and(|t| crate::vision::is_image(*t)) => b,
                _ => &bias,
            };
            let key = |i: usize| pr[i] + bias[i];
            let mut order: Vec<usize> = (0..ne).collect();
            order.select_nth_unstable_by(used - 1, |a, c| key(*c).total_cmp(&key(*a)).then(a.cmp(c)));
            let mut s = order[..used].to_vec();
            s.sort_by(|a, c| key(*c).total_cmp(&key(*a)).then(a.cmp(c)));
            let sum: f32 = s.iter().map(|i| pr[*i]).sum::<f32>() + 1e-20;
            s.iter().map(|&i| (i, pr[i] / sum * g.expert_scale as f32)).collect()
        };
        let mut emiss: Vec<bool> = Vec::new();
        let rows: Vec<Vec<(usize, f32)>> = if early.is_some() {
            // the GPU's routing: ids ascending, their weights, which had no VRAM slot
            let ids: Vec<usize> = lv[..used].iter().map(|v| v.to_bits() as i32 as usize).collect();
            emiss = lv[2 * used..3 * used].iter().map(|v| v.to_bits() != 0).collect();
            vec![ids.iter().zip(&lv[used..2 * used]).map(|(&i, &w)| (i, f32::from_bits(w.to_bits()))).collect()]
        } else if t >= 256 {
            let parts = 16.min(t);
            let per = t.div_ceil(parts);
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..parts).map(|k| { let pick = &pick; sc.spawn(move || (k * per..((k + 1) * per).min(t)).map(pick).collect::<Vec<_>>()) }).collect();
                hs.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
            })
        } else {
            (0..t).map(pick).collect()
        };
        let mut by: BTreeMap<usize, Vec<(i32, f32)>> = BTreeMap::new();
        for (ti, sel) in rows.into_iter().enumerate() {
            for (i, w) in sel {
                by.entry(i).or_default().push((ti as i32, w));
            }
        }
        ROUTE_HOST_NS.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if t == 1 && self.prof.is_some() && !p.draft.contains_key(&l) {
            if !p.stores_of(l).is_empty() {
                let (mut any, mut low) = (false, true);
                DEC_LAYERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                for (ex, v) in &by {
                    if p.with_store(l, *ex as u64, |c| c.loc.contains_key(&(l, *ex as u64))).unwrap_or(false) {
                        continue;
                    }
                    let frac = v[0].1 / g.expert_scale as f32;
                    let bin = [0.05, 0.1, 0.15, 0.2, 0.3].iter().position(|b| frac < *b).unwrap_or(5);
                    COLD_BINS[bin].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    any = true;
                    low &= frac < 0.1;
                }
                if any {
                    COLD_LAYERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if low {
                        COLD_LAYERS_LOW.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
        tap(&format!("ffn_shexp-{l}"), &y)?;
        // cold experts of small weight left out (NS_DS41_COLD_SKIP): their rows of the products stay zero
        let mut skip: Vec<bool> = vec![false; by.len()];
        let prompt_skip = t > MMVQ_COLS && cold_skip_prompt() > 0.0 && p.ring.is_some() && !p.draft.contains_key(&l);
        if ((cold_skip() > 0.0 && t <= MMVQ_COLS) || prompt_skip) && by.len() > 1 && !p.draft.contains_key(&l) {
            let (thr, maxn) = if t <= MMVQ_COLS { (cold_skip(), usize::MAX) } else { (cold_skip_prompt(), cold_skip_prompt_max()) };
            for (k, (ex, v)) in by.iter().enumerate() {
                let key = (l, *ex as u64);
                let Some((cold, ahead)) = p.with_store(l, key.1, |c| {
                    (if skip_host() && t <= MMVQ_COLS { !matches!(c.loc.get(&key), Some(Loc::V(_))) } else { !c.loc.contains_key(&key) }, c.ahead.contains_key(&key))
                }) else { continue };
                if cold && (t > MMVQ_COLS || cold_skip_ahead() || !ahead) && v.len() <= maxn && v.iter().all(|x| x.1 / (g.expert_scale as f32) < thr) {
                    skip[k] = true;
                }
            }
            if skip.iter().all(|s| *s) {
                skip.fill(false);
            }
        }
        let skipped = skip.iter().filter(|s| **s).count();
        COLD_SKIPPED.fetch_add(skipped as u64, std::sync::atomic::Ordering::Relaxed);
        // the kept experts' weight scale: 1, or total / kept when renormalizing
        let wscale = if skipped > 0 && cold_renorm() {
            let (all, kept): (f32, f32) = by.values().zip(&skip).fold((0.0, 0.0), |(a, k), (v, sk)| (a + v[0].1, if *sk { k } else { k + v[0].1 }));
            if kept > 0.0 { all / kept } else { 1.0 }
        } else {
            1.0
        };
        // (a prompt pass keeps every expert in `need`: its rows line up with `lists`; the skipped are left out of what is staged)
        let need: Vec<u64> = by.keys().enumerate().filter(|(k, _)| !skip[*k] || t > MMVQ_COLS).map(|(_, e)| *e as u64).collect();
        if t == 1 && self.prof.is_some() {
            let mut gl = GUESS_LAST.lock().unwrap();
            if let Some(i) = gl.iter().position(|(ly, _)| *ly == l) {
                let (_, order) = gl.swap_remove(i);
                GUESS_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                for (rank, e) in order.iter().enumerate() {
                    if need.contains(&(*e as u64)) {
                        GUESS_HITS[rank].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
        let promote = t <= MMVQ_COLS;
        let staging = !promote && p.ring.is_some() && !p.draft.contains_key(&l);
        let (slots_of, arriving, copies) =
            if staging { (Vec::new(), Vec::new(), None) } else { self.timed(p, "experts: misses", || p.ensure(&self.m, l, &need, promote))? };
        // the slots of the experts left out: a view of the first one's (never read)
        let (slots_of, arriving) = if skipped > 0 && !slots_of.is_empty() {
            let (mut so, mut ar) = (Vec::with_capacity(skip.len()), Vec::with_capacity(skip.len()));
            let (mut it, mut ia) = (slots_of.into_iter(), arriving.into_iter());
            let mut keep0: Option<DevBuf> = None;
            for sk in skip.iter() {
                if *sk {
                    so.push(None);
                    ar.push(false);
                } else {
                    let b = it.next().ok_or("ensure: a slot short")?;
                    ar.push(ia.next().unwrap_or(false));
                    if keep0.is_none() {
                        keep0 = Some(b.view(0, b.len)?);
                    }
                    so.push(Some(b));
                }
            }
            let k0 = keep0.ok_or("every expert left out")?;
            (so.into_iter().map(|b| b.map_or_else(|| k0.view(0, k0.len), Ok)).collect::<Result<Vec<_>>>()?, ar)
        } else {
            (slots_of, arriving)
        };
        let vq = &p.vq[&l];
        let nbit = vq.nbit;
        let total: usize = by.values().map(|v| v.len()).sum();
        let lists: Vec<&Vec<(i32, f32)>> = by.values().collect();
        // each list's expert (its pointers' type tag, the XQ file's)
        let exids: Vec<u64> = by.keys().map(|e| *e as u64).collect();
        // the combine: per token its rows of dn, with their weights, in expert order
        let mut first = Vec::with_capacity(lists.len());
        let mut e0 = 0usize;
        for list in &lists {
            first.push(e0);
            e0 += list.len();
        }
        let dn = match &early {
            Some((_, dn, _)) => dn.view(0, dn.len)?,
            None => p.arena.f32(total * d)?,
        };
        if t > MMVQ_COLS {
            // a prompt pass: the rows of the experts left out zero, all at once
            if skipped > 0 {
                dn.fill(0)?;
            }
        } else {
            for (k, sk) in skip.iter().enumerate() {
                if *sk {
                    let n = lists[k].len() * d * 4;
                    p.upload(&dn.view(first[k] * d * 4, n)?, &vec![0u8; n])?;
                }
            }
        }
        // the GPU routing's combine weights: the skipped ones zero (their rows are too), the kept scaled
        if let (Some((_, _, ctab)), true) = (&early, wscale != 1.0) {
            let ws: Vec<f32> = lists.iter().zip(&skip).map(|(v, sk)| if *sk { 0.0 } else { v[0].1 * wscale }).collect();
            p.upload(&ctab.view((2 + used) * 4, used * 4)?, &f32s(&ws))?;
        }
        if t <= MMVQ_COLS {
            self.timed(p, "experts: products", || {
                let gu = p.arena.f32(2 * total * f)?;
                // routed on the GPU: only the experts it found no VRAM slot for are left
                let (now, later): (Vec<usize>, Vec<usize>) = (0..lists.len()).filter(|&k| !skip[k] && (early.is_none() || emiss.get(k).copied().unwrap_or(true))).partition(|&k| !arriving[k]);
                for (set, wait) in [(&now, false), (&later, true)] {
                    if set.is_empty() {
                        continue;
                    }
                    if wait {
                        if let Some(tk) = copies {
                            let tc = Instant::now();
                            o.await_ticket(tk)?;
                            ENS_COPY_NS.fetch_add(tc.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    // gate and up entries (an expert's tokens as its columns), then down
                    let mut ptrs: Vec<u64> = Vec::new();
                    let mut meta: Vec<i32> = Vec::new();
                    let mut dptr: Vec<u64> = Vec::new();
                    let mut dmeta: Vec<i32> = Vec::new();
                    for which in 0..2 {
                        for &k in set.iter() {
                            let (g0, u0, _, _) = vq.lay(exids[k]);
                            ptrs.push(slots_of[k].ptr() as u64 + if which == 0 { g0 } else { u0 } as u64);
                            let mut m = vec![lists[k].len() as i32 | vq.tag(exids[k]), (which * total + first[k]) as i32];
                            m.extend(lists[k].iter().map(|(ti, _)| *ti));
                            m.resize(10, 0);
                            meta.extend(m);
                        }
                    }
                    for &k in set.iter() {
                        dptr.push(slots_of[k].ptr() as u64 + vq.lay(exids[k]).2 as u64);
                        let mut m = vec![lists[k].len() as i32 | vq.tag(exids[k]), first[k] as i32];
                        m.extend((0..lists[k].len()).map(|j| (first[k] + j) as i32));
                        m.resize(10, 0);
                        dmeta.extend(m);
                    }
                    let mut tb: Vec<u8> = ptrs.iter().flat_map(|v| v.to_le_bytes()).collect();
                    tb.extend(i32s(&meta));
                    let mut db: Vec<u8> = dptr.iter().flat_map(|v| v.to_le_bytes()).collect();
                    db.extend(i32s(&dmeta));
                    let mut pick = p.vq_pick.get(&nbit).copied().unwrap_or([0, 0]);
                    // the lookup-table variant's lists: gate | up entries by token (the column of x), downs by their row of h
                    let (mut glut, mut dlut) = (None, None);
                    if pick[0] == crate::ops::VQ_LUT {
                        let mut pairs = Vec::new();
                        for w in 0..2 {
                            for (i, &k) in set.iter().enumerate() {
                                for (q, (ti, _)) in lists[k].iter().enumerate() {
                                    pairs.push((*ti as usize, w * set.len() + i, q));
                                }
                            }
                        }
                        match crate::ops::lut_lists(t, pairs) {
                            Some((v, mx)) => {
                                tb.extend(v);
                                glut = Some((t, mx));
                            }
                            None => pick[0] = 0,
                        }
                    }
                    if pick[1] == crate::ops::VQ_LUT {
                        let mut pairs = Vec::new();
                        for (i, &k) in set.iter().enumerate() {
                            for q in 0..lists[k].len() {
                                pairs.push((first[k] + q, i, q));
                            }
                        }
                        match crate::ops::lut_lists(total, pairs) {
                            Some((v, mx)) => {
                                db.extend(v);
                                dlut = Some((total, mx));
                            }
                            None => pick[1] = 0,
                        }
                    }
                    // the std kernels' split-Q8 path needs only x's column counts
                    if p.vq[&l].std.is_some() {
                        glut = glut.or(Some((t, 0)));
                        dlut = dlut.or(Some((total, 0)));
                    }
                    let gtab = p.arena.bytes(tb.len())?;
                    p.upload(&gtab, &tb)?;
                    let dtab = p.arena.bytes(db.len())?;
                    p.upload(&dtab, &db)?;
                    p.exp_mmv(l, false, &gtab, ptrs.len(), x, (&gu, 0), 2 * total, f, d, pick[0], glut, lists.iter().map(|v| v.len()).max().unwrap_or(1))?;
                    // h = silu(gate) up, in place of the gates (rows of the set only: the same op on the others is a no-op
                    // since they are computed once - the whole range is cheap and simplest)
                    let (gv, uv) = (gu.view(0, total * f * 4)?, gu.view(total * f * 4, total * f * 4)?);
                    let hset = p.arena.f32(total * f)?;
                    o.swiglu_clamp(&gv, &uv, &hset, total * f, lim)?;
                    p.exp_mmv(l, true, &dtab, dptr.len(), &hset, (&dn, 0), total, d, f, pick[1], dlut, lists.iter().map(|v| v.len()).max().unwrap_or(1))?;
                    if !wait {
                        TURN_NS.fetch_add(turn.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Ok(())
            })?;
        } else if staging {
            self.timed(p, "experts: prompt products", || {
                // the experts in VRAM multiplied while those in host memory and the file are copied up into the staging
                // ring, half of it a wave: a wave's copies queued while the one before is multiplied
                let src = p.locate(l, &need)?;
                let mut views: Vec<Option<DevBuf>> = src.iter().map(|s| match s {
                    Src::V(v) => v.view(0, v.len).map(Some),
                    _ => Ok(None),
                }).collect::<Result<_>>()?;
                let resident: Vec<usize> = (0..need.len()).filter(|&k| views[k].is_some()).collect();
                let mut away: Vec<usize> = (0..need.len()).filter(|&k| matches!(src[k], Src::R(_))).collect();
                away.extend((0..need.len()).filter(|&k| matches!(src[k], Src::F) && !skip[k]));
                let k_ring = p.ring.as_ref().map_or(0, |r| r.0.len / r.1);
                let w = (k_ring / 2).max(1);
                let waves: Vec<&[usize]> = away.chunks(w).collect();
                let mut free: Vec<Option<i64>> = vec![*p.ring_free.lock().unwrap(); k_ring];
                // the file's experts read on threads meanwhile (the reads end before the first wave holding one is copied)
                let files: Vec<(usize, u64)> = away.iter().filter(|&&k| matches!(src[k], Src::F)).map(|&k| (k, need[k])).collect();
                if pipe_split() {
                    eprintln!("[layer {l}: experts {} in VRAM, {} in host memory, {} from the file]", resident.len(), away.len() - files.len(), files.len());
                }
                // the file's experts: all their reads started now, on the reader pool (a wave below waits only for its own)
                // (those the layer before already started are there; the rest start now)
                let mut by_ex: HashMap<u64, (usize, Arc<RDone>)> = match p.pre_reads.lock().unwrap().take() {
                    Some((pl, m)) if pl == l => m,
                    _ => HashMap::new(),
                };
                let todo: Vec<u64> = files.iter().map(|f| f.1).filter(|ex| !by_ex.contains_key(ex)).collect();
                if !todo.is_empty() {
                    by_ex.extend(p.start_file_reads(l, &todo)?);
                }
                let read: HashMap<usize, (usize, Arc<RDone>)> = files.iter().filter_map(|(k, ex)| by_ex.get(ex).map(|v| (*k, v.clone()))).collect();
                // the next layer's file experts read meanwhile: a prompt of more than a few tokens uses nearly all of them
                if prompt_prefetch() && t >= prompt_prefetch_min() && p.layers.contains(&(l + 1)) && p.reader.is_some() {
                    if !p.stores_of(l + 1).is_empty() {
                        let nxt: Vec<u64> = (0..ne as u64).filter(|ex| !p.with_store(l + 1, *ex, |c| c.loc.contains_key(&(l + 1, *ex))).unwrap_or(true)).collect();
                        if !nxt.is_empty() {
                            let m = p.start_file_reads(l + 1, &nxt)?;
                            *p.pre_reads.lock().unwrap() = Some((l + 1, m));
                        }
                    }
                }
                let mut pending: Option<Option<i64>> = None;
                if let Some(w0) = waves.first() {
                    let slots: Vec<usize> = (0..w0.len()).collect();
                    let (v, tk) = p.stage_up(&self.m, l, w0, &need, &src, &slots, &free, &read)?;
                    for (&k, b) in w0.iter().zip(v) {
                        views[k] = Some(b);
                    }
                    pending = Some(tk);
                }
                let ts = Instant::now();
                self.prompt_set(p, l, x, &lists, &exids, &first, &views, &resident, &dn, total)?;
                if pipe_split() {
                    o.gpu.sync()?;
                    eprintln!("[layer {l}: the {} experts in VRAM multiplied in {:.1} ms]", resident.len(), ts.elapsed().as_secs_f64() * 1e3);
                }
                let tw = Instant::now();
                for (wi, wave) in waves.iter().enumerate() {
                    if let Some(Some(tk)) = pending.take() {
                        o.await_ticket(tk)?;
                    }
                    self.prompt_set(p, l, x, &lists, &exids, &first, &views, wave, &dn, total)?;
                    let done = Some(o.mark()?);
                    free[(wi % 2) * w..(wi % 2) * w + wave.len()].fill(done);
                    if let Some(next) = waves.get(wi + 1) {
                        let h = (wi + 1) % 2;
                        let slots: Vec<usize> = (h * w..h * w + next.len()).collect();
                        let (v, tk) = p.stage_up(&self.m, l, next, &need, &src, &slots, &free, &read)?;
                        for (&k, b) in next.iter().zip(v) {
                            views[k] = Some(b);
                        }
                        pending = Some(tk);
                    }
                }
                if pipe_split() {
                    o.gpu.sync()?;
                    eprintln!("[layer {l}: {} waves of the others copied up and multiplied in {:.1} ms]", waves.len(), tw.elapsed().as_secs_f64() * 1e3);
                }
                // the ring's last readers: before the next layer's copies
                *p.ring_free.lock().unwrap() = Some(o.mark()?);
                Ok(())
            })?;
        } else {
            self.timed(p, "experts: prompt products", || {
                if let Some(tk) = copies {
                    o.await_ticket(tk)?;
                }
                let views: Vec<Option<DevBuf>> = slots_of.iter().map(|v| v.view(0, v.len).map(Some)).collect::<Result<_>>()?;
                let all: Vec<usize> = (0..lists.len()).collect();
                self.prompt_set(p, l, x, &lists, &exids, &first, &views, &all, &dn, total)
            })?;
        }
        let mut t_ptr = vec![0i32];
        let mut ent = Vec::with_capacity(total);
        let mut wv = Vec::with_capacity(total);
        let mut per_tok: Vec<Vec<(usize, f32)>> = vec![Vec::new(); t];
        for (k, list) in lists.iter().enumerate() {
            for (j, (ti, w)) in list.iter().enumerate() {
                per_tok[*ti as usize].push((first[k] + j, if skip[k] { 0.0 } else { *w * wscale }));
            }
        }
        for v in &per_tok {
            for (r, w) in v {
                ent.push(*r as i32);
                wv.push(*w);
            }
            t_ptr.push(ent.len() as i32);
        }
        if let Some((_, _, ctab)) = &early {
            // the GPU's own combine table (the same rows and weights, in expert order)
            o.moe_combine(&y, &dn, &ctab.view(0, (2 + used) * 4)?, &ctab.view((2 + used) * 4, used * 4)?, t, total, d)?;
        } else {
            let cbuf = p.arena.bytes((t + 1 + total) * 4)?;
            p.upload(&cbuf, &i32s(&t_ptr.iter().chain(&ent).copied().collect::<Vec<_>>()))?;
            let wb = p.arena.f32(total)?;
            p.upload(&wb, &f32s(&wv))?;
            o.moe_combine(&y, &dn, &cbuf, &wb, t, total, d)?;
        }
        if let Some(gv) = guess {
            let gb = p.router_bias(l + 1)?;
            let mut miss: BTreeMap<usize, f32> = BTreeMap::new();
            for ti in 0..t {
                let pr: Vec<f32> = gv[ti * ne..(ti + 1) * ne].iter().map(|z| sqrt_softplus(*z)).collect();
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|a, b| (pr[*b] + gb[*b]).total_cmp(&(pr[*a] + gb[*a])));
                if t == 1 && self.prof.is_some() {
                    GUESS_LAST.lock().unwrap().push((l + 1, order[..GUESS_HIT.len()].to_vec()));
                }
                for (rank, &e) in order.iter().take(GUESS_HIT.len()).enumerate() {
                    *miss.entry(e).or_insert(1.0) *= 1.0 - GUESS_HIT[rank];
                }
            }
            let th = prefetch_p();
            let mut all: Vec<(usize, f32)> = miss.into_iter().map(|(e, q)| (e, 1.0 - q)).collect();
            all.sort_by(|a, b| b.1.total_cmp(&a.1));
            let want: Vec<u64> = all.iter().filter(|x| x.1 >= th).map(|(e, _)| *e as u64).collect();
            self.timed(p, "experts: prefetch", || p.prefetch(&self.m, l + 1, &want, pf))?;
            if file_ahead() > 0 {
                // a read costs the drive and a slot, not PCIe or VRAM: a lower bar than the copies'
                let fth = file_ahead_p();
                let fwant: Vec<u64> = all.iter().filter(|x| x.1 >= fth).map(|(e, _)| *e as u64).collect();
                let k = p.read_ahead(l + 1, &fwant, file_ahead())?;
                AHEAD_NEAR.fetch_add(k as u64, std::sync::atomic::Ordering::Relaxed);
                // and the guesses further ahead: their best few, their reads given more time than one layer's
                for ((lj, _), fv) in far.iter().zip(&farv) {
                    let fb = p.router_bias(*lj)?;
                    let key: Vec<f32> = fv.iter().zip(fb.iter()).map(|(z, b)| sqrt_softplus(*z) + b).collect();
                    let mut order: Vec<usize> = (0..ne).collect();
                    order.sort_by(|a, b| key[*b].total_cmp(&key[*a]));
                    let top: Vec<u64> = order.iter().take(far_rank()).map(|e| *e as u64).collect();
                    let k = p.read_ahead(*lj, &top, far_top())?;
                    AHEAD_FAR.fetch_add(k as u64, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        Ok(y)
    }

    /// A prompt pass's products of experts `set` (their payloads at views[k]) for their tokens (lists, first: as moe):
    /// the few-token ones from their indices, the others expanded to fp16 for a GEMM; into dn's rows
    #[allow(clippy::too_many_arguments)]
    fn prompt_set(&self, p: &Part, l: u64, x: &DevBuf, lists: &[&Vec<(i32, f32)>], exids: &[u64], first: &[usize], views: &[Option<DevBuf>], set: &[usize], dn: &DevBuf,
                  total: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
        let lim = g.swiglu_limit as f32;
        let vq = &p.vq[&l];
        if set.is_empty() {
            return Ok(());
        }
            // experts with few tokens straight from their indices (the decode kernels, up to 8 tokens an entry): an
            // expansion to fp16 writes and reads its whole matrix whatever its tokens
            // (the ggml-typed experts always: their products from the blocks read the matrix once, an expansion twice and more)
            let few = if vq.std.is_some() { MMVQ_COLS } else { prompt_mmv_max() };
            let small: Vec<usize> = set.iter().copied().filter(|&k| lists[k].len() <= few).collect();
            let mut at = 0;
            while at < small.len() {
                // a batch of experts, at most PROMPT_MMV_SLOTS of their tokens
                let mut end = at;
                let mut slots = 0;
                while end < small.len() && (slots + lists[small[end]].len() <= PROMPT_MMV_SLOTS || end == at) {
                    slots += lists[small[end]].len();
                    end += 1;
                }
                let batch = &small[at..end];
                at = end;
                let mk = p.arena.mark();
                let gu = p.arena.f32(2 * slots * f)?;
                let hh = p.arena.f32(slots * f)?;
                let (mut gp, mut gm, mut dp, mut dm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let mut slot = 0;
                for &k in batch {
                    let base = views[k].as_ref().ok_or("an expert not in place")?.ptr() as u64;
                    let tag = vq.tag(exids[k]);
                    let (g0, u0, d0, _) = vq.lay(exids[k]);
                    for (ci, ch) in lists[k].chunks(MMVQ_COLS).enumerate() {
                        let j0 = ci * MMVQ_COLS;
                        for (w, off) in [(0usize, g0), (1, u0)] {
                            gp.push(base + off as u64);
                            let mut m = vec![ch.len() as i32 | tag, (w * slots + slot) as i32];
                            m.extend(ch.iter().map(|(ti, _)| *ti));
                            m.resize(10, 0);
                            gm.extend(m);
                        }
                        dp.push(base + d0 as u64);
                        let mut m = vec![ch.len() as i32 | tag, (first[k] + j0) as i32];
                        m.extend((0..ch.len()).map(|c| (slot + c) as i32));
                        m.resize(10, 0);
                        dm.extend(m);
                        slot += ch.len();
                    }
                }
                let mut tb: Vec<u8> = gp.iter().flat_map(|v| v.to_le_bytes()).collect();
                tb.extend(i32s(&gm));
                let mut db: Vec<u8> = dp.iter().flat_map(|v| v.to_le_bytes()).collect();
                db.extend(i32s(&dm));
                let gtab = p.arena.bytes(tb.len())?;
                p.upload(&gtab, &tb)?;
                let dtab = p.arena.bytes(db.len())?;
                p.upload(&dtab, &db)?;
                let var = prompt_mmv_variant();
                p.exp_mmv(l, false, &gtab, gp.len(), x, (&gu, 0), 2 * slots, f, d, var, None, 8)?;
                o.swiglu_clamp(&gu.view(0, slots * f * 4)?, &gu.view(slots * f * 4, slots * f * 4)?, &hh, slots * f, lim)?;
                p.exp_mmv(l, true, &dtab, dp.len(), &hh, (dn, 0), total, d, f, var, None, 8)?;
                p.arena.rewind(mk);
            }
            let lists_big: Vec<usize> = set.iter().copied().filter(|&k| lists[k].len() > few).collect();
            if lists_big.is_empty() {
                return Ok(());
            }
            // fp16 x and h readable to whole tiles of 64 tokens (the XMX product's)
            let nmax = lists_big.iter().map(|&k| lists[k].len()).max().unwrap_or(1).next_multiple_of(64);
            let xh = p.arena.bytes(nmax * d * 2)?;
            let gu = p.arena.f32(nmax * 2 * f)?;
            let hh = p.arena.bytes(nmax * f * 2)?;
            // the ggml-typed experts are always expanded to fp16 for a GEMM (no product from their blocks on XMX yet)
            let gmax = if vq.std.is_some() { 0 } else { vq_gemm_max() };
            let w16 = if lists_big.iter().any(|&k| lists[k].len() > gmax) { Some(p.arena.bytes(2 * f * d * 2)?) } else { None };
            let toks: Vec<i32> = lists.iter().flat_map(|l| l.iter().map(|(ti, _)| *ti)).collect();
            let tb = p.arena.bytes(total * 4)?;
            p.upload(&tb, &i32s(&toks))?;
            let (cb, cbh) = (p.cbs.get(&l), p.cbh.get(&l));
            let nbit = vq.nbit;
            for &k in &lists_big {
                let n = lists[k].len();
                let buf = views[k].as_ref().ok_or("an expert not in place")?;
                let tv = tb.view(first[k] * 4, n * 4)?;
                o.gather_f16(x, &tv, &xh, n, d)?;
                match &w16 {
                    Some(w16) if n > gmax => {
                        // gate | up as rows [0, f) and [f, 2f) of w16, then the down
                        let deq = |which: usize, rows: usize, cols: usize, dst: &DevBuf| -> Result<()> {
                            let (g0, u0, d0, sp) = vq.lay(exids[k]);
                            let (at, bytes) = [(g0, u0 - g0), (u0, d0 - u0), (d0, sp - d0)][which];
                            match &vq.std {
                                Some(_) => {
                                    let (_, _, n, ty) = vq.mat_of(exids[k], which).ok_or("an expert's matrix")?;
                                    o.std_dequant(ty.code(), buf, at, n.min(bytes), rows, cols, dst, true)
                                }
                                None => o.vq_dequant(buf, at, bytes, cb.ok_or("no codebook")?, rows, cols, nbit, dst, true),
                            }
                        };
                        deq(0, f, d, &w16.view(0, f * d * 2)?)?;
                        deq(1, f, d, &w16.view(f * d * 2, f * d * 2)?)?;
                        o.gemm_f16(n, 2 * f, d, (&xh, 0, d), (w16, 0), (&gu, 0, 2 * f), false)?;
                        o.swiglu_gu_f16(&gu, &hh, n, f, lim)?;
                        deq(2, d, f, &w16.view(0, d * f * 2)?)?;
                        o.gemm_f16(n, d, f, (&hh, 0, f), (w16, 0), (dn, first[k] * d, d), false)?;
                    }
                    _ => {
                        let (gv, uv, dv) = (buf.view(vq.gate, vq.up - vq.gate)?, buf.view(vq.up, vq.down - vq.up)?, buf.view(vq.down, vq.span - vq.down)?);
                        let cbh = cbh.ok_or("no fp16 codebook")?;
                        o.vq_gemm(&gv, Some(&uv), f, 2 * f, d, nbit, cbh, &xh, d, n, (&gu, 0), 2 * f)?;
                        o.swiglu_gu_f16(&gu, &hh, n, f, lim)?;
                        o.vq_gemm(&dv, None, d, d, f, nbit, cbh, &hh, f, n, (dn, first[k] * d), d)?;
                    }
                }
            }
        Ok(())
    }

    /// The decode requests per expert since load blended into the profile at `path` (old weights halved)
    pub fn save_expert_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let mut prof: HashMap<(u64, u64), f64> = read_profile(path).unwrap_or_default();
        for v in prof.values_mut() {
            *v *= 0.5;
        }
        let mut n = 0;
        for p in &self.parts {
            for s in &p.stores {
                let c = s.lock().unwrap();
                for (k, u) in &c.usage {
                    *prof.entry(*k).or_insert(0.0) += *u as f64;
                    n += 1;
                }
            }
        }
        if n == 0 {
            return Ok(0);
        }
        let mut v: Vec<_> = prof.into_iter().filter(|(_, w)| *w >= 0.5).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out = String::from("# nextsycl expert profile: layer expert weight (decode requests, older halved each save)\n");
        for ((l, x), w) in &v {
            out += &format!("{l} {x} {w:.1}\n");
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(&tmp, path)?;
        Ok(v.len())
    }
}

/// Decode's fused chains (NS_DS41_KV_FUSE=0: the separate kernels; always those without the simulation): the window
/// keys' norm, rotation, simulation and ring write in one launch (ns_ds41_kv_post), the attention output's rotation back
/// and simulation in one (ns_ds41_att_post), an mHC half's mixes, collapse and norm in two (ns_ds41_hc_pre_in); and
/// at one row two products of the same input in one launch (Part::mm2: q_a | kv, the shared gate | up, the router | the
/// next layer's guess)
/// Keys from which an index source's top-k is taken in two levels (NS_DS41_IDX_TWO, default 196,608; 0: never): below
/// it one radix select over every key was as fast (B70: 1.6 against 2.0 ms at 131K keys and 256 rows, 1.7 against 1.4
/// at 262K; B65 2.8 / 3.0 and 3.4 / 1.9)
fn idx_two_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match env_usize("NS_DS41_IDX_TWO", 196_608) {
        0 => usize::MAX,
        v => v,
    })
}
fn fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NS_DS41_KV_FUSE").map_or(true, |v| v != "0"))
}
/// The fused chains that end in the simulation (kv_post, att_post): not without it
fn kv_fused() -> bool {
    fused() && crate::ops::qsim_on()
}
