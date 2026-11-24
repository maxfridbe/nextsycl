//! The card's memory for one clip: what the cap is, what a clip's denoiser and decoder hold on the card, whether a clip
//! fits beside the loaded model, and the longest that does.
//!
//! 2026-11-23: a B70 worker that had made 30 clips failed every clip after with "nsd_alloc: the device refused N MiB"
//! (the card at 32.4 of 32.8 GB) until it was unloaded by hand, and 294-328 frame clips were only refused once they
//! reached an allocation. Three things answer it: the kernels never take the card's last 1.5 GiB and give back their
//! caches before refusing (kernels/diffusion/nsd.cpp, `cap_for` here mirrors its cap); a clip is checked against the
//! card before its denoiser is made, here, and refused with the longest that fits (or run with the MLP in row chunks
//! when that is what makes it fit); and the daemon recycles a worker the card refused memory.

use crate::dit::{Config, Scratch};
use crate::jobs::{gib, temporal_shape};

/// The card's memory every engine leaves free (the box's rule; the kernels' NSD_MEM_RESERVE_MB)
pub const RESERVE: u64 = 1536 << 20;

/// The reserve the kernels were given (NSD_MEM_RESERVE_MB), else `RESERVE`
pub fn reserve() -> u64 {
    std::env::var("NSD_MEM_RESERVE_MB").ok().and_then(|v| v.parse::<u64>().ok()).map_or(RESERVE, |m| m << 20)
}

/// The kernels' cap on their book for a card of `total` bytes: the share `frac` of it, never into the reserve (every
/// card, shared or not - kernels/diffusion/nsd.cpp `init`)
pub fn cap_for(total: u64, frac: f64, reserve: u64) -> u64 {
    ((total as f64 * frac) as u64).min(total.saturating_sub(reserve))
}

/// What can still be had for a clip: the book's room under the cap, and the card's free memory over the reserve (memory
/// outside the book - the runtime's, oneDNN's, another process's - shows there), whichever is less. `reused`: the
/// kernels' scratch already grown (in the book and on the card), which the clip's own kernel calls take over
pub fn room(cap: u64, used: u64, card_free: Option<u64>, reserve: u64, reused: u64) -> u64 {
    let book = cap.saturating_sub(used.saturating_sub(reused));
    card_free.map_or(book, |f| book.min((f + reused).saturating_sub(reserve)))
}

/// Card memory a process holds outside its kernels' book: the card's used memory less the book (the runtime's pools,
/// oneDNN's kernels and reordered weights, contexts - and, on a card shared with another process, that process's)
pub fn outside_book(card_total: u64, card_free: u64, used: u64) -> u64 {
    card_total.saturating_sub(card_free).saturating_sub(used)
}

/// The most the kernels' per-shape caches may hold outside the book before a clip's end gives them back (nsd_trim).
/// 2026-11-23 a worker's grew 0.1-0.17 GiB a clip (every clip's sequence a new length: new primitives) and the card
/// refused memory after the 30th clip with ~5 GiB outside the book; a fresh worker holds ~0.5-1 GiB there.
pub const OUTSIDE_LIMIT: u64 = 2 << 30;

/// Whether the end of a clip gives the caches back: they hold more than `OUTSIDE_LIMIT` outside the book
pub fn should_trim(outside: u64) -> bool {
    outside > OUTSIDE_LIMIT
}

/// The MLP's rows at a time when a clip does not fit with the whole sequence at once (regen2k's choice: the two widest
/// buffers 8192 rows instead of every token's)
pub const MLP_CHUNK: usize = 8192;

/// Card memory the kernels and the runtime take beside the buffers counted below, for any sequence: oneDNN's
/// primitives and scratchpads, new shapes' kernels during the clip, the estimate's error. The card's reserve (1.5 GiB)
/// is kept on top of it.
pub const MARGIN: u64 = 512 << 20;

/// The rest a token costs, measured: the final layer's and the sampler's per-row buffers, the rotation tables. On the
/// B70 (2026-11-23, 832x480, 3 pictures, the book above the loaded model at its peak): 41725 tokens 9.43 GiB and
/// 47779 tokens 10.78 GiB at once (242.5 KB a token against 233.7 counted), 73865 tokens with the MLP 8192 rows at a time
/// 11.43 GiB (154 KB a token against 140.5 counted and 0.82 GiB of chunk buffers)
pub const EXTRA_PER_TOKEN: usize = 12_000;

/// Bytes the denoiser holds for a `tokens`-long sequence with the MLP `mlp_rows` at a time: its stream and scratch
/// (`Scratch::bytes`), and the kernels' scratch - the SageAttention path's int8 q and k, its 16-bit v and result
/// (6 bytes x heads x head_dim a token) and the int8 linears' quantized input (the widest input of a linear: the
/// attention output, or the MLP's inner width over the rows it runs at once)
pub fn denoise_bytes(cfg: &Config, tokens: usize, mlp_rows: usize) -> u64 {
    let w = cfg.heads * cfg.head_dim;
    let mr = mlp_rows.min(tokens);
    let stream = 2 * tokens * cfg.hidden;
    let attention = 6 * w * tokens;
    let linears = (w * tokens).max(cfg.ffn * mr);
    (Scratch::bytes(cfg, tokens, mlp_rows) + stream + attention + linears + EXTRA_PER_TOKEN * tokens) as u64 + MARGIN
}

/// Bytes the video decoder holds: its weights (its file's size: they are kept in the file's type) and a batch of tiles'
/// activations, which does not grow with the clip (the decoder works through it a batch at a time)
pub fn decode_bytes(vae_file: u64) -> u64 {
    vae_file + DECODE_ACTIVATIONS + MARGIN
}

/// A batch of tiles' activations in the video decoder (the B70, 832x480: the book 5.2 GiB above its clip-to-clip level
/// while a 124-frame clip decoded, 4.5 GiB of it the weights)
pub const DECODE_ACTIVATIONS: u64 = 1 << 30;

/// A clip's sequence: `other` tokens that do not grow with it (text, references, keyframes), its audio rows and its
/// video's 2x2 patches
pub fn clip_tokens(frames_seconds: f64, width: usize, height: usize, other: usize) -> usize {
    let (_, lt, at) = temporal_shape(frames_seconds);
    other + lt * (height / 32) * (width / 32) + 2 * at
}

/// How a clip will run, when it fits
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub tokens: usize,
    /// the MLP's rows at a time (the sequence's length: all at once)
    pub mlp_rows: usize,
    /// the most the clip holds on the card at once
    pub need: u64,
}

/// The clip's plan within `room` bytes: the whole sequence at once, else the MLP in chunks, else None
pub fn plan(cfg: &Config, tokens: usize, vae_file: u64, room: u64) -> Option<Plan> {
    let dec = decode_bytes(vae_file);
    [tokens, MLP_CHUNK].into_iter().filter(|r| *r <= tokens).map(|mlp_rows| Plan { tokens, mlp_rows, need: denoise_bytes(cfg, tokens, mlp_rows).max(dec) })
        .find(|p| p.need <= room)
}

/// The longest clip (in frames on the model's 17k + 5 grid) that fits in `room`, with `other` tokens beside it; None
/// when not even the shortest does
pub fn max_frames(cfg: &Config, width: usize, height: usize, other: usize, vae_file: u64, room: u64) -> Option<usize> {
    let mut best = None;
    let mut frames = 5;
    while frames <= 24 * 600 {
        let tokens = clip_tokens(frames as f64 / 24.0, width, height, other);
        if plan(cfg, tokens, vae_file, room).is_none() {
            break;
        }
        best = Some(frames);
        frames += 17;
    }
    best
}

/// The refusal of a clip that does not fit: what it needs, what can be had, the longest that fits
pub fn too_long(cfg: &Config, frames: usize, width: usize, height: usize, other: usize, vae_file: u64, room: u64) -> String {
    let tokens = clip_tokens(frames as f64 / 24.0, width, height, other);
    let need = denoise_bytes(cfg, tokens, MLP_CHUNK.min(tokens)).max(decode_bytes(vae_file));
    let most = match max_frames(cfg, width, height, other, vae_file, room) {
        Some(f) => format!("at most {f} frames ({:.2} s) fit", f as f64 / 24.0),
        None => "not even the shortest clip fits".into(),
    };
    format!("{frames} frames at {width}x{height} ({tokens} tokens) are too long for this card: the clip needs about {:.1} GiB beside the model, \
             {:.1} GiB can be had (the card keeps {:.1} GiB free); {most}", gib(need), gib(room), gib(reserve()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MiniMax H3's denoiser (minimax_h3_ref2va_pruned_int8_convrot: hidden 5376, 56 heads of 128, MLP 14336, 50 blocks)
    fn h3() -> Config {
        Config { hidden: 5376, heads: 56, head_dim: 128, ffn: 14336, rot_dim: 64, modalities: 3, t_dim: 256, norm_eps: 1e-6, qk_eps: 1e-6, blocks: 50 }
    }
    const GIB: u64 = 1 << 30;
    /// the B70's (and the B65's) global memory as the runtime reports it: 31.89 GiB
    const B70: u64 = 34_242_297_856;
    const VAE: u64 = 4_831_838_208; // minimax_h3_video_vae_fp16: 4.50 GiB

    #[test]
    fn every_card_gets_a_cap_under_the_reserve() {
        // 0.94 of a 32 GB card is already under total - 1.5 GiB; a smaller share of the reserve decides on a small card
        let c = cap_for(B70, 0.94, RESERVE);
        assert_eq!(c, (B70 as f64 * 0.94) as u64);
        assert!(B70 - c >= RESERVE);
        assert_eq!(cap_for(B70, 0.97, RESERVE), B70 - RESERVE, "a bigger share stops at the reserve");
        assert_eq!(cap_for(8 * GIB, 0.97, RESERVE), 8 * GIB - RESERVE);
        assert_eq!(cap_for(GIB, 0.94, RESERVE), 0, "a card smaller than the reserve gets nothing");
    }

    #[test]
    fn room_is_the_book_or_the_card_whichever_is_less() {
        let cap = cap_for(B70, 0.94, RESERVE);
        // the 2026-11-23 worker: 20 GiB in its book, the card 32.4 of 32.8 GB used - the book alone saw 10 GiB of room
        let used = 20 * GIB;
        assert_eq!(room(cap, used, None, RESERVE, 0), cap - used);
        assert_eq!(room(cap, used, Some(GIB / 3), RESERVE, 0), 0, "a card under its reserve has no room whatever the book says");
        assert_eq!(room(cap, used, Some(4 * GIB), RESERVE, 0), 4 * GIB - RESERVE);
        assert_eq!(room(cap, used, Some(20 * GIB), RESERVE, 0), cap - used, "the book's cap still holds on an empty card");
        assert_eq!(room(cap, cap + GIB, Some(20 * GIB), RESERVE, 0), 0);
        // grown scratch the clip reuses counts as room on both sides
        assert_eq!(room(cap, used, None, RESERVE, 2 * GIB), cap - used + 2 * GIB);
        assert_eq!(room(cap, used, Some(4 * GIB), RESERVE, 2 * GIB), 6 * GIB - RESERVE);
    }

    #[test]
    fn the_caches_go_back_once_they_hold_too_much_outside_the_book() {
        // production's B70 worker, 2026-11-23 20:40: 20.21 GiB in the book, 21.32 used on the card - 1.1 outside: kept
        let total = B70;
        let book = (20.21 * GIB as f64) as u64;
        let fresh = outside_book(total, total - (21.32 * GIB as f64) as u64, book);
        assert!((GIB..GIB + GIB / 5).contains(&fresh), "{fresh}");
        assert!(!should_trim(fresh));
        // the failing worker of 19:28: the card at 32,413 MiB with ~26 GiB in the book
        let stuck = outside_book(total, total - (32_413u64 << 20), 26 * GIB);
        assert!(should_trim(stuck), "{stuck}");
        assert_eq!(outside_book(total, total, 3 * GIB), 0, "a book over the card's used memory (a stale reading) is no outside memory");
    }

    #[test]
    fn the_clips_of_the_night_tokens() {
        // 832x480 with three 640x640 pictures; 25705 tokens for 192 frames with 1635 text-and-vision tokens in the log
        let other = 1635 + 3 * 400;
        assert_eq!(clip_tokens(8.0, 832, 480, other), 25705);
        assert_eq!(clip_tokens(5.17, 832, 480, other) - other, 17622 - 1578 - 1200);
    }

    #[test]
    fn denoise_memory_grows_with_the_sequence_and_chunks_save() {
        let cfg = h3();
        let a = denoise_bytes(&cfg, 20_000, 20_000);
        let b = denoise_bytes(&cfg, 40_000, 40_000);
        assert!(b > a && (b - MARGIN) == 2 * (a - MARGIN), "linear in tokens");
        // about 245 KB a token at once (242.5 measured)
        let per = (a - MARGIN) / 20_000;
        assert!((240_000..250_000).contains(&per), "{per}");
        assert!(denoise_bytes(&cfg, 40_000, MLP_CHUNK) + 2 * GIB < b, "the MLP in chunks saves over 2 GiB at 40k tokens");
        // a short sequence: chunking changes nothing
        assert_eq!(denoise_bytes(&cfg, 4000, MLP_CHUNK), denoise_bytes(&cfg, 4000, 4000));
    }

    #[test]
    fn a_plan_takes_the_whole_sequence_then_chunks_then_refuses() {
        let cfg = h3();
        let t = 30_000;
        let whole = denoise_bytes(&cfg, t, t);
        let p = plan(&cfg, t, VAE, whole).unwrap();
        assert_eq!((p.mlp_rows, p.need), (t, whole));
        let p = plan(&cfg, t, VAE, whole - 1).unwrap();
        assert_eq!(p.mlp_rows, MLP_CHUNK);
        assert!(plan(&cfg, t, VAE, denoise_bytes(&cfg, t, MLP_CHUNK) - 1).is_none());
        // a short clip: the decoder is the larger need
        let p = plan(&cfg, 5000, VAE, 30 * GIB).unwrap();
        assert_eq!(p.need, decode_bytes(VAE));
    }

    #[test]
    fn the_measured_clips_fit_their_estimates() {
        // the B70 runs of 2026-11-23 (the book above the model at the clip's peak): every estimate covers what was used,
        // and none by more than the margin and 0.5 GiB
        let cfg = h3();
        for (tokens, rows, used_gib) in [(41_725, 41_725, 9.43), (47_779, 47_779, 10.78), (73_865, MLP_CHUNK, 11.43)] {
            let used = (used_gib * GIB as f64) as u64;
            let est = denoise_bytes(&cfg, tokens, rows);
            assert!(est >= used && est - used <= MARGIN + GIB / 2, "{tokens} tokens: estimate {est}, used {used}");
        }
    }

    #[test]
    fn the_longest_clip_that_fits() {
        let cfg = h3();
        let other = 1635 + 1200;
        let r = 10 * GIB;
        let f = max_frames(&cfg, 832, 480, other, VAE, r).unwrap();
        assert_eq!(f % 17, 5, "on the model's frame grid");
        assert!(plan(&cfg, clip_tokens(f as f64 / 24.0, 832, 480, other), VAE, r).is_some());
        assert!(plan(&cfg, clip_tokens((f + 17) as f64 / 24.0, 832, 480, other), VAE, r).is_none());
        assert!(max_frames(&cfg, 832, 480, other, VAE, GIB).is_none(), "no clip fits in 1 GiB");
        let msg = too_long(&cfg, f + 17, 832, 480, other, VAE, r);
        assert!(msg.contains("too long for this card") && msg.contains(&format!("at most {f} frames")), "{msg}");
        assert!(!msg.contains("nsd_alloc"), "not an allocation failure: the daemon keeps the worker");
    }
}
