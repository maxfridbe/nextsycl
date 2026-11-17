//! ZONOS3-Mini (Zyphra, Apache-2.0): multilingual speech at 44.1 kHz, in a preset voice or a voice cloned from a
//! recording. An 11-layer backbone of 1024 (gated DeltaNet mixers with a global-attention layer in four, Full AttnRes,
//! a top-2-of-64 latent MoE) reads SuperBPE text after a voice - a speaker row (a 192-d MECT-B2 embedding, projected)
//! or a reference prefix (a recording's codes) - and three condition rows (loudness, bandwidth, speaking rate), and
//! writes ten 4096-code books at 15 Hz in the delay pattern; the ZAC codec's MR-iSTFT decoder makes them 44.1 kHz.
//!
//! ```text
//!   prompt (prompt.rs): [speaker row | reference prefix] [3 condition rows] BOS text EOS [3 silence rows]
//!   loop (lm.rs + sample.rs): a row of ten codes a step, codebook c of frame f on row f + c; the first end-of-audio
//!     fixes the last frame, eleven rows more drain the delay (speaker CFG: a twin without the voice, in lockstep)
//!   decode (codec.rs): the rows undelayed, the silence frames decoded ahead and cut, -3 dB and the soft limiter
//! ```
//!
//! Voices: a preset's codes (.codes.npy: the reference prefix, the checkpoint's default speaker mode) or its embedding
//! (.embedding.npy: the speaker row); a recording by the speaker encoder (spk.rs: the speaker row). Steering (steer.rs)
//! doses named directions. The files are Zyphra's GGUFs (`Zyphra/ZONOS3-Mini`, gguf/): the backbone, a decoder
//! (zac-keep2: the default, 2 layers a stage; zac: the full one), the speaker encoder and the steering bundle.
//! Raw-reference cloning (encoder.rs: the reference encoder and its SSL teacher, loaded for an encode), Zyphra's text
//! normalizer and markup (markup.rs), LACI and short texts (laci.rs), streaming decode (stream.rs) and speech editing
//! (edit.rs with FuseAlign's word timings, align.rs, and the edit LoRA in lm.rs).
//!
//! Origin: a port of Zyphra's reference runtime (github.com/Zyphra/ZONOS3-Mini: zonos3.reference, the ggml server
//! and the MLX server; MIT, the model Apache-2.0) to Rust and this project's SYCL kernels (kernels/audio/zonos3), by
//! way of our sister project nextcuda's port (audio/zonos3 there: the same host code, its CUDA kernels rewritten in
//! SYCL here). Not ported: CJK word segmentation for edits, accent adapters (none ship: accents are steering
//! directions).

#![allow(clippy::too_many_arguments, clippy::type_complexity, clippy::needless_range_loop)]

pub mod check;
pub mod align;
pub mod codec;
pub mod edit;
pub mod encoder;
mod ffi;
pub mod laci;
pub mod lm;
pub mod markup;
pub mod ops;
pub mod prompt;
pub mod sample;
pub mod spk;
pub mod steer;
pub mod stream;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech, Step};
use nextsycl_core::Gpu;
use nextsycl_tok::Tokenizer;

use crate::codec::Decoder;
use crate::lm::{Cond, Lm};
use crate::ops::{Ops, Weights};
use crate::prompt::Voice;
use crate::sample::{Rng, Sampling};
use crate::spk::SpeakerEncoder;
use crate::steer::Steer;

pub const ARCH: &str = lm::ARCH;
pub const ROLES: &[&str] = &["model", "decoder", "decoder-full", "speaker-encoder", "steer", "voice-*"];
pub const RATE: u32 = 44100;
/// code frames a second (2940 samples each)
pub const FPS: f32 = 15.0;
/// the GGUF servers' frame budget (the generation registry's max_frames)
pub const MAX_FRAMES: usize = 2000;
/// a text longer than this many characters is said in stretches of whole sentences
const MAX_CHARS: usize = 400;
/// the silence between stretches, seconds
const GAP: f32 = 0.25;
/// the soft limiter's knee (-1 dBFS)
const KNEE: f32 = 0.891_250_9;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "speaker-mode", env: "", value: "reference|embedding", help: "how the voice comes in: its codes as a reference prefix (reference, the model's default) or its speaker embedding (embedding; steering's voice controls need it). A recording is cloned by its embedding unless speaker-mode is reference: then by its own codes through the reference encoder (raw-reference cloning, its first 60 s; needs zac-encoder.gguf and ssl-teacher.gguf, about 6.3 GiB more on the card while it encodes)", at: At::Request },
    EngineOption { name: "controls", env: "", value: "ID=DOSE,...", help: "steering: doses of the bundle's directions, e.g. delivery.tempo=2 (faster), mood.valence=3, emotions.happy=4, delivery.whisper=3; identity.*, pitch.*, accents_*, texture.* and room.* move the speaker embedding (speaker-mode embedding or a recording)", at: At::Request },
    EngineOption { name: "speaking-rate", env: "", value: "BYTES/S", help: "the speaking-rate condition: UTF-8 bytes of text a second (about 14 is ordinary English; unset: the model's choice)", at: At::Request },
    EngineOption { name: "loudness", env: "", value: "LUFS", help: "the loudness condition (-30 to -15; unset: unknown)", at: At::Request },
    EngineOption { name: "fmax", env: "", value: "HZ", help: "the bandwidth condition (8000 to 20000; unset: unknown)", at: At::Request },
    EngineOption { name: "decoder", env: "", value: "keep2|full", help: "the codec decoder: keep2 (the default: two layers a stage) or full (the full-size head, when its file is there)", at: At::Request },
    EngineOption { name: "temperature", env: "", value: "0.75", help: "sampling temperature", at: At::Request },
    EngineOption { name: "top-k", env: "", value: "80", help: "sampling top-k", at: At::Request },
    EngineOption { name: "top-p", env: "", value: "0.69", help: "sampling top-p", at: At::Request },
    EngineOption { name: "repetition-penalty", env: "", value: "2.4", help: "the repetition penalty over the last 96 frames of the first six codebooks", at: At::Request },
    EngineOption { name: "normalize", env: "", value: "on|off", help: "Zyphra's text normalizer (zyphra-textnorm) in the request's language (on; auto: the script decides, Latin English): numbers, money, dates, units said as words; off: the text as written", at: At::Request },
    EngineOption { name: "laci", env: "", value: "on|off", help: "LACI (off): the alignment heads' reading of the text checked every step; a skipped or repeated stretch, or a take ending early, is rolled back and said again (at most 8 times) under an attention mask - needs cfg 1 and the language's heads (laci/ beside the model)", at: At::Request },
    EngineOption { name: "short-text", env: "", value: "auto|off", help: "a text of a word or two (auto): said with a pause and a filler sentence after it under LACI, and cut where the filler starts - alone the model often says it wrong or runs on; off: as it is", at: At::Request },
    EngineOption { name: "level", env: "", value: "on|off", help: "loudness leveling (on): the take brought to -19 dBFS of speech and held there (the model drifts quieter through a long take with some voices, and a preset keeps its recording's level); off: the decoder's level less 3 dB, as Zyphra serves it", at: At::Request },
    EngineOption { name: "silence", env: "", value: "0..30", help: "codec-silence frames after the text (3: 0.2 s, decoded ahead and cut)", at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "ZONOS3-Mini (multilingual speech at 44.1 kHz: preset voices, voices cloned from a recording, steering; Apache-2.0)", roles: ROLES,
                load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Zonos3::load(files, gpu, o, log)?))
}

/// Zyphra's gguf/ folder as downloaded (backbone-*.gguf, zac-keep2.gguf, zac.gguf, spk-encoder.gguf, steer.gguf,
/// voices/<Language>/<Name>_*.codes.npy / .embedding.npy): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    // the edit backbone first: the release LM with the phoneme, start and pause rows (and speech editing's vocabulary)
    let model = ["backbone-ptq-edit.gguf", "backbone-ptq.gguf", "backbone-f16.gguf", "backbone-q8_0.gguf"].iter().map(|n| dir.join(n)).find(|p| p.is_file())
        .ok_or_else(|| Error(format!("{}: no backbone-ptq.gguf", dir.display())))?;
    f.insert("model".into(), model);
    for (role, name) in [("decoder", "zac-keep2.gguf"), ("decoder-full", "zac.gguf"), ("speaker-encoder", "spk-encoder.gguf"), ("steer", "steer.gguf"),
                         ("edit-adapter", "edit/accents.gguf"), ("zac-encoder", "zac-encoder.gguf"), ("ssl-teacher", "ssl-teacher.gguf"),
                         ("aligner", "fusealign-multilingual-ns4-step60000-f32.gguf")] {
        if dir.join(name).is_file() {
            f.insert(role.into(), dir.join(name));
        }
    }
    // LACI's alignment heads, a file a language (Zyphra's web/laci)
    for e in std::fs::read_dir(dir.join("laci")).into_iter().flatten().flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if let Some(stem) = n.strip_suffix(".json").filter(|s| s.starts_with("zonos3_mini_heads")) {
            f.insert(format!("laci-{stem}"), e.path());
        }
    }
    if !f.contains_key("decoder") {
        return Err(Error(format!("{}: no zac-keep2.gguf", dir.display())));
    }
    let mut stack = vec![dir.join("voices")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let (stem, kind) = match (n.strip_suffix(".codes.npy"), n.strip_suffix(".embedding.npy")) {
                (Some(s), _) => (s, ""),
                (_, Some(s)) => (s, "-emb"),
                _ => continue,
            };
            let name = stem.split('_').next().unwrap_or(stem).to_lowercase();
            f.insert(format!("voice-{name}{kind}"), p.clone());
        }
    }
    Ok(f)
}

/// A preset voice's files
#[derive(Clone, Debug, Default)]
pub struct VoiceFiles {
    pub codes: Option<PathBuf>,
    pub embedding: Option<PathBuf>,
}

pub struct Zonos3 {
    pub ops: Ops,
    pub lm: Lm,
    pub tok: Tokenizer,
    /// the default decoder first
    pub decoders: Vec<Decoder>,
    /// the full-size decoder's file (zac.gguf), loaded on first use: its matrices take 4 GiB on the card
    full_path: Option<PathBuf>,
    full: std::sync::OnceLock<Decoder>,
    pub spk: Option<SpeakerEncoder>,
    pub steer: Option<Steer>,
    pub voices: BTreeMap<String, VoiceFiles>,
    /// LACI's alignment heads by file name (zonos3_mini_heads_de.json ...)
    pub laci: BTreeMap<String, laci::Heads>,
    /// raw-reference cloning's files (zac-encoder.gguf, ssl-teacher.gguf): loaded for an encode and freed after it
    /// (6.3 GiB on the card), the codes kept per recording
    enc_paths: Option<(PathBuf, PathBuf)>,
    /// FuseAlign (speech editing's word timings): loaded for an edit and freed after it
    aligner_path: Option<PathBuf>,
    ref_codes: Mutex<Vec<(u64, Vec<[u32; 10]>)>>,
    /// one request at a time on the card
    lock: Mutex<()>,
    load_s: f64,
}

/// A request's own option
fn given<'a>(req: &'a AudioRequest, name: &str) -> Option<&'a str> {
    req.extra.get(name).map(String::as_str).filter(|s| !s.trim().is_empty())
}

fn num<T: std::str::FromStr>(req: &AudioRequest, name: &str) -> Result<Option<T>> {
    given(req, name).map(|v| v.trim().parse::<T>().map_err(|_| Error(format!("option {name}: {v:?} is not a number")))).transpose()
}

/// The stretches a long text is said in: whole sentences, each at most `max` characters (a sentence longer than that
/// is cut at its last comma, else its last space, before the limit)
pub fn stretches(text: &str, max: usize) -> Vec<String> {
    let text = text.trim();
    if text.chars().count() <= max {
        return vec![text.to_string()];
    }
    let mut sentences = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        cur.push(*c);
        let end = matches!(c, '.' | '!' | '?' | '\n' | '。' | '！' | '？' | '…');
        if end && chars.get(i + 1).is_none_or(|n| n.is_whitespace() || !c.is_ascii()) {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur);
    }
    let mut out: Vec<String> = Vec::new();
    for s in sentences {
        let mut s = s.trim().to_string();
        while s.chars().count() > max {
            let head: String = s.chars().take(max).collect();
            let cut = head.rfind([',', ';', ':']).or_else(|| head.rfind(' ')).filter(|i| *i > 0).map_or(head.len(), |i| i + 1);
            out.push(s[..cut].trim().to_string());
            s = s[cut..].trim().to_string();
        }
        if s.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.chars().count() + 1 + s.chars().count() <= max => {
                last.push(' ');
                last.push_str(&s);
            }
            _ => out.push(s),
        }
    }
    out
}

/// The complete frames of delayed rows (`np` prefill rows first; frame f's codebook c on row f + c) before the take's
/// end: its aligned end frame `eos` (counted from the first generated row) and any end of audio drawn in a row
pub fn frames_before_end(buffer: &[[u32; 10]], ncb: usize, eoa: u32, eos: Option<usize>, np: usize) -> usize {
    let mut target = buffer.len().saturating_sub(ncb - 1);
    if let Some(e) = eos {
        target = target.min(e + np);
    }
    for (i, r) in buffer.iter().enumerate() {
        if let Some(mc) = (0..ncb).rev().find(|cb| r[*cb] == eoa) {
            target = target.min(i.saturating_sub(mc));
        }
    }
    target
}

/// The output stage: the decoder's headroom as gain, then the soft limiter above -1 dBFS
pub fn output(pcm: &mut [f32], headroom_db: f32) {
    let g = 10f32.powf(-headroom_db / 20.0);
    let w = 1.0 - KNEE;
    for x in pcm.iter_mut() {
        let v = *x * g;
        *x = if v.abs() > KNEE { v.signum() * (KNEE + w * ((v.abs() - KNEE) / w).tanh()) } else { v };
    }
}

/// The level speech is brought to: active-speech RMS in dBFS (about -18 LUFS; the platform's other speech engines
/// deliver -19 to -22)
pub const TARGET_DBFS: f32 = -19.0;

/// Loudness leveling: ZONOS3-Mini drifts quieter through a long take with some voices (Zyphra's own runtime does the
/// same on the same embedding: -26 -> -32 dB, -30 -> -42 dB over 27 s), and a preset's level is its recording's (Savannah
/// -29 dB). The level of the speech around each 50 ms frame (active frames only, a 3 s window) is brought to `target`
/// by a gain that moves slowly (0.5 s smoothing, -12 to +50 dB), so a sentence keeps its own dynamics while the take as
/// a whole holds its level. Pauses keep the gain of the speech before them.
pub fn level(pcm: &mut [f32], rate: u32, target: f32) {
    let hop = (rate as usize / 20).max(1);
    let nf = pcm.len().div_ceil(hop);
    if nf == 0 {
        return;
    }
    let energy: Vec<f64> = (0..nf).map(|f| {
        let s = &pcm[f * hop..((f + 1) * hop).min(pcm.len())];
        s.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / s.len().max(1) as f64
    }).collect();
    let db = |e: f64| 10.0 * (e + 1e-12).log10();
    // active: within 35 dB of the loud frames around it (the 95th percentile of the 10 s about the frame: a take that
    // fades 40 dB and more keeps its tail - a gate against the whole take's loud frames took a faded tail for silence
    // and left it 5 dB low, platform job f77a24fb), above digital silence
    let active = active_frames(&energy);
    if !active.iter().any(|a| *a) {
        return;
    }
    let half = 30; // frames: 1.5 s either side
    let (mut sum, mut cnt) = (vec![0f64; nf + 1], vec![0usize; nf + 1]);
    for f in 0..nf {
        sum[f + 1] = sum[f] + if active[f] { energy[f] } else { 0.0 };
        cnt[f + 1] = cnt[f] + active[f] as usize;
    }
    let mut gain_db = vec![0f32; nf];
    let mut last: Option<f32> = None;
    for f in 0..nf {
        let (lo, hi) = (f.saturating_sub(half), (f + half + 1).min(nf));
        let n = cnt[hi] - cnt[lo];
        // a window with too little speech keeps the gain before it (the first speech's for a leading pause)
        let g = if n >= 4 { Some((target - db((sum[hi] - sum[lo]) / n as f64) as f32).clamp(-12.0, MAX_GAIN_DB)) } else { None };
        if g.is_some() {
            last = g;
        }
        gain_db[f] = g.or(last).unwrap_or(f32::NAN);
    }
    let first = gain_db.iter().copied().find(|g| g.is_finite()).unwrap_or(0.0);
    gain_db.iter_mut().filter(|g| !g.is_finite()).for_each(|g| *g = first);
    // smoothed forwards and backwards (no lag either way): a 0.5 s time constant
    let a = (-1.0f32 / 10.0).exp();
    for f in 1..nf {
        gain_db[f] = a * gain_db[f - 1] + (1.0 - a) * gain_db[f];
    }
    for f in (0..nf - 1).rev() {
        gain_db[f] = a * gain_db[f + 1] + (1.0 - a) * gain_db[f];
    }
    // per sample, between frame centres
    for (i, x) in pcm.iter_mut().enumerate() {
        let p = (i as f32 / hop as f32 - 0.5).max(0.0);
        let f0 = (p as usize).min(nf - 1);
        let f1 = (f0 + 1).min(nf - 1);
        let t = p - f0 as f32;
        let g = gain_db[f0] * (1.0 - t) + gain_db[f1] * t;
        *x *= 10f32.powf(g / 20.0);
    }
}

/// The most the leveling raises (a take that fades to -64 dBFS of speech comes back)
pub const MAX_GAIN_DB: f32 = 50.0;

/// The 50 ms frames of speech among `energy` (mean squares): above -80 dBFS and within 35 dB of the 95th percentile of
/// the 10 s about each frame
pub fn active_frames(energy: &[f64]) -> Vec<bool> {
    let db = |e: f64| 10.0 * (e + 1e-12).log10();
    let nf = energy.len();
    let half = 100;
    (0..nf).map(|f| {
        let (lo, hi) = (f.saturating_sub(half), (f + half + 1).min(nf));
        let mut w: Vec<f64> = energy[lo..hi].to_vec();
        w.sort_by(|a, b| b.total_cmp(a));
        let loud = db(w[(w.len() / 20).min(w.len() - 1)]);
        db(energy[f]) > -80.0 && db(energy[f]) > loud - 35.0
    }).collect()
}

/// A control span placed in a take: its start and end frame and its hidden deltas by source
pub type SpanSched = (f32, f32, Vec<(usize, Vec<f32>)>);

/// The stretches of a (normalized, marked) text as character ranges: whole sentences, each at most `max` characters
/// (a longer sentence cut at its last comma, else space, before the limit)
pub fn stretch_ranges(text: &str, max: usize) -> Vec<(usize, usize)> {
    let ch: Vec<char> = text.chars().collect();
    let n = ch.len();
    let mut ends = Vec::new();
    for i in 0..n {
        let end = matches!(ch[i], '.' | '!' | '?' | '\n' | '。' | '！' | '？' | '…');
        if end && ch.get(i + 1).is_none_or(|c| c.is_whitespace() || !ch[i].is_ascii() || (markup::PAUSE0..='\u{E00A}').contains(c)) {
            ends.push(i + 1);
        }
    }
    if ends.last() != Some(&n) {
        ends.push(n);
    }
    let trim = |a: usize, b: usize| -> (usize, usize) {
        let (mut a, mut b) = (a, b);
        while a < b && ch[a].is_whitespace() {
            a += 1;
        }
        while b > a && ch[b - 1].is_whitespace() {
            b -= 1;
        }
        (a, b)
    };
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for e in ends {
        let (mut a, b) = trim(start, e);
        start = e;
        while b - a > max {
            let cut = (a + 1..a + max).rev().find(|i| matches!(ch[*i - 1], ',' | ';' | ':')).or_else(|| (a + 1..a + max).rev().find(|i| ch[*i].is_whitespace()))
                .unwrap_or(a + max);
            out.push(trim(a, cut));
            a = trim(cut, b).0;
        }
        if a >= b {
            continue;
        }
        match out.last_mut() {
            Some(last) if b - last.0 <= max && last.1 <= a => last.1 = b,
            _ => out.push((a, b)),
        }
    }
    out.retain(|(a, b)| b > a);
    out
}

/// One stretch's generation: the rows drawn (each ten codes) and the aligned end frame
pub struct Take {
    pub rows: Vec<[u32; 10]>,
    pub eos: Option<usize>,
}

/// A take's extras: LACI (its heads, and a short text's filler token) and a callback that gets the generated rows
/// that are final (no later rollback can change them), then all of them with the end frame when the take is done
#[derive(Default)]
pub struct TakeOpts<'a> {
    pub laci: Option<(&'a laci::Heads, Option<usize>)>,
    #[allow(clippy::type_complexity)]
    pub on_rows: Option<&'a mut dyn FnMut(&[[u32; 10]], Option<usize>) -> Result<()>>,
}

impl Zonos3 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Zonos3> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let role = |r: &str| files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let (model, dec) = (role("model")?, role("decoder")?);
        // float32 mixers and head (~4x their stored size), the decoder's tables and matrices (float32), a long take's
        // activations
        let need = size(&model) / 2 + 600 * (1 << 20) + size(&dec) * 2 + (1u64 << 30);
        let (total, free) = gpu.memory().map_err(ops::e)?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let w = Weights::open(&model)?;
        let mut lm = Lm::load(&ops, &w, log)?;
        // the speech-edit adapter (edit/accents.gguf) on the edit backbone (its span-end and marker rows)
        if let (Some(p), true) = (files.get("edit-adapter"), lm.c.audio_vocab > lm.c.pad as usize + 1) {
            lm.attach_edit(&ops, &Weights::open(p)?)?;
            log(format!("{ARCH}: speech-edit adapter, {:.0} MiB", lm.edit.as_ref().map_or(0, |e| e.bytes) as f64 / (1u64 << 20) as f64));
        }
        let tok = Tokenizer::from_hf_json_text_keep_gaps(&lm.c.tokenizer).map_err(Error)?;
        let decoders = vec![Decoder::load(&ops, &Weights::open(&dec)?, log)?];
        let spk = match files.get("speaker-encoder") {
            Some(p) => Some(SpeakerEncoder::load(&ops, &Weights::open(p)?, log)?),
            None => None,
        };
        let steer = match files.get("steer") {
            Some(p) => Some(Steer::load(&Weights::open(p)?)?),
            None => None,
        };
        let mut voices: BTreeMap<String, VoiceFiles> = BTreeMap::new();
        for (r, p) in files {
            let Some(name) = r.strip_prefix("voice-") else { continue };
            match name.strip_suffix("-emb") {
                Some(n) => voices.entry(n.to_string()).or_default().embedding = Some(p.clone()),
                None => voices.entry(name.to_string()).or_default().codes = Some(p.clone()),
            }
        }
        let mut laci = BTreeMap::new();
        for (r, p) in files {
            if r.starts_with("laci-") {
                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let text = std::fs::read_to_string(p).map_err(|x| Error(format!("{}: {x}", p.display())))?;
                laci.insert(name.clone(), laci::Heads::parse(&text, &name)?);
            }
        }
        gpu.sync().map_err(ops::e)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH} on {} in {load_s:.1} s: {} voices, decoder {}{}, speaker encoder {}, steering {}", gpu.name, voices.len(), decoders[0].variant,
                    if files.contains_key("decoder-full") { " (full_mr on request)" } else { "" },
                    if spk.is_some() { "yes" } else { "no" }, if steer.is_some() { "yes" } else { "no" }));
        Ok(Zonos3 { ops, lm, tok, decoders, full_path: files.get("decoder-full").cloned(), full: std::sync::OnceLock::new(), spk, steer, voices,
                    enc_paths: files.get("zac-encoder").cloned().zip(files.get("ssl-teacher").cloned()), ref_codes: Mutex::new(Vec::new()), laci, aligner_path: files.get("aligner").cloned(),
                    lock: Mutex::new(()), load_s })
    }

    pub fn decoder(&self, req: &AudioRequest) -> Result<&Decoder> {
        match given(req, "decoder") {
            None => Ok(&self.decoders[0]),
            Some(d) => {
                let want = match d.trim() {
                    "keep2" | "keep2_plain" => "keep2_plain",
                    "full" | "full_mr" => "full_mr",
                    x => return Err(Error(format!("decoder {x:?}: keep2 or full"))),
                };
                if let Some(d) = self.decoders.iter().find(|x| x.variant == want) {
                    return Ok(d);
                }
                if want == "full_mr" {
                    if let Some(d) = self.full.get() {
                        return Ok(d);
                    }
                    let p = self.full_path.as_ref().ok_or_else(|| Error("decoder full: this model has no zac.gguf".into()))?;
                    let d = Decoder::load(&self.ops, &Weights::open(p)?, &mut |m| eprintln!("{m}"))?;
                    let _ = self.full.set(d);
                    return self.full.get().ok_or_else(|| Error("decoder full: not loaded".into()));
                }
                Err(Error(format!("decoder {want} is not loaded")))
            }
        }
    }

    /// A recording's own codes [frames, 10] (raw-reference cloning: its first 60 s through the reference encoder),
    /// kept for the last 64 recordings
    pub fn reference_codes(&self, samples: &[f32], rate: u32, log: &mut dyn FnMut(String)) -> Result<Vec<[u32; 10]>> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        rate.hash(&mut h);
        samples.iter().for_each(|x| x.to_bits().hash(&mut h));
        let key = h.finish();
        if let Some((_, c)) = self.ref_codes.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|(k, _)| *k == key) {
            return Ok(c.clone());
        }
        let (ep, sp) = self.enc_paths.as_ref().ok_or_else(|| Error(format!("{ARCH}: raw-reference cloning needs zac-encoder.gguf and ssl-teacher.gguf")))?;
        // mono, peak-normalized when it clips (Zyphra's prepare_reference)
        let peak = samples.iter().fold(0f32, |m, x| m.max(x.abs()));
        let pcm: Vec<f32> = if peak > 1.0 { samples.iter().map(|x| x / peak).collect() } else { samples.to_vec() };
        let t0 = Instant::now();
        let enc = encoder::RefEncoder::load(&self.ops, &Weights::open(ep)?, &Weights::open(sp)?, log)?;
        let t1 = Instant::now();
        let codes = enc.encode(&self.ops, &pcm, rate)?;
        drop(enc);
        log(format!("{ARCH}: reference encoded: {} frames in {:.1} s (+ {:.1} s loading the encoder)", codes.len(), t1.elapsed().as_secs_f64(),
                    (t1 - t0).as_secs_f64()));
        let mut cache = self.ref_codes.lock().unwrap_or_else(|p| p.into_inner());
        cache.push((key, codes.clone()));
        if cache.len() > 64 {
            cache.remove(0);
        }
        Ok(codes)
    }

    /// The request's voice: a recording (its embedding, or with speaker-mode reference its own codes), a preset (its
    /// codes, or its embedding), or the default preset
    pub fn voice(&self, req: &AudioRequest, embedding_mode: bool) -> Result<Voice> {
        if let Some(r) = &req.reference {
            if !embedding_mode && given(req, "speaker-mode") == Some("reference") {
                if r.samples.len() < r.rate.max(1) as usize / 2 {
                    return Err(Error("the recording is shorter than half a second".into()));
                }
                return Ok(Voice::Reference(self.reference_codes(&r.samples, r.rate, &mut |m| eprintln!("{m}"))?));
            }
            let spk = self.spk.as_ref().ok_or_else(|| Error(format!("{ARCH}: cloning a recording needs the speaker encoder (spk-encoder.gguf)")))?;
            if r.samples.len() < r.rate.max(1) as usize / 2 {
                return Err(Error("the recording is shorter than half a second".into()));
            }
            return Ok(Voice::Embedding(spk.embed(&self.ops, &r.samples, r.rate)?));
        }
        let name = match req.voice.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) => v.to_lowercase(),
            None if self.voices.is_empty() => return Ok(Voice::None),
            None => ["savannah", "wyatt"].iter().find(|n| self.voices.contains_key(**n)).map(|s| s.to_string())
                .unwrap_or_else(|| self.voices.keys().next().cloned().unwrap_or_default()),
        };
        if name == "none" {
            return Ok(Voice::None);
        }
        let v = self.voices.get(&name).ok_or_else(|| Error(format!("no voice {name:?} (this model's: {})", self.voices.keys().cloned().collect::<Vec<_>>().join(", "))))?;
        let read = |p: &PathBuf| std::fs::read(p).map_err(|x| Error(format!("{}: {x}", p.display())));
        match (embedding_mode, &v.codes, &v.embedding) {
            (false, Some(c), _) => Ok(Voice::Reference(prompt::read_codes(&read(c)?)?)),
            (_, _, Some(e)) => Ok(Voice::Embedding(prompt::read_embedding(&read(e)?)?)),
            (true, Some(_), None) => Err(Error(format!("voice {name} has no embedding file: speaker-mode reference"))),
            _ => Err(Error(format!("voice {name} has no files"))),
        }
    }

    /// One stretch: the prompt's rows through the backbone, then a row of codes a step until the end frame drains
    #[allow(clippy::too_many_arguments)]
    pub fn take(&self, ids: &[u32], voice: &Voice, buckets: &[Option<usize>], deltas: &[(usize, Vec<f32>)], spans: &[SpanSched], cfg: f32, s: &Sampling,
                silence: usize, max_frames: usize, rng: &mut Rng, progress: &mut dyn FnMut(usize) -> Result<()>) -> Result<(Take, Vec<[u32; 10]>)> {
        self.take_with(ids, voice, buckets, deltas, spans, cfg, s, silence, max_frames, rng, progress, TakeOpts::default())
    }

    /// `take` with LACI (and a short text's cut) and a callback that gets the generated rows as they become final
    #[allow(clippy::too_many_arguments)]
    pub fn take_with(&self, ids: &[u32], voice: &Voice, buckets: &[Option<usize>], deltas: &[(usize, Vec<f32>)], spans: &[SpanSched], cfg: f32, s: &Sampling,
                     silence: usize, max_frames: usize, rng: &mut Rng, progress: &mut dyn FnMut(usize) -> Result<()>, mut o: TakeOpts)
                     -> Result<(Take, Vec<[u32; 10]>)> {
        let c = &self.lm.c;
        let ids = ids.to_vec();
        let p = prompt::build(c, &ids, voice, buckets, silence)?;
        let cond = Cond {
            speaker: p.speaker.as_ref().map(|(r, v)| Ok::<_, Error>((*r, self.lm.project_speaker(v)?))).transpose()?,
            buckets: p.buckets.clone(),
            deltas: Vec::new(),
        };
        let cap = p.rows + max_frames + 2;
        let w = c.row_width();
        let ncb = c.ncb;
        let np = p.prefill.len();
        // LACI: the text span of the prompt (BOS..EOS rows), the session over the stream
        let mut session = None;
        let mut text_span = (0usize, 0usize);
        if let Some((heads, short)) = o.laci {
            let col: Vec<u32> = (0..p.rows).map(|r| p.ids[r * w + ncb]).collect();
            let start = col.iter().position(|t| *t == c.bos).ok_or_else(|| Error("LACI: the prompt has no text BOS row".into()))?;
            let end = start + 1 + col[start + 1..].iter().position(|t| *t == c.eos).ok_or_else(|| Error("LACI: the prompt has no text EOS row".into()))?;
            text_span = (start, end - start + 1);
            session = Some(laci::Session::new(heads.clone(), &p.prefill, rng.seed(), max_frames, ncb, o.on_rows.is_none(), short));
        }
        let probing = session.is_some();
        let probe_heads: Vec<(usize, usize)> = o.laci.map(|(h, _)| h.heads.clone()).unwrap_or_default();
        let new_seq = |kept: &[[u32; 10]]| -> Result<lm::Seq> {
            let mut all = p.ids.clone();
            for r in kept {
                all.extend(r[..ncb].iter().copied().chain(std::iter::once(c.text_vocab)));
            }
            let mut sq = self.lm.seq(&self.ops, cap, p.rows + kept.len())?;
            if probing {
                sq.probe = Some(lm::Probe::new(text_span.0, text_span.1, probe_heads.clone()));
            }
            self.lm.forward(&self.ops, &mut sq, &all, &cond, false)?;
            Ok(sq)
        };
        let mut seq = new_seq(&[])?;
        // speaker CFG: the same text without the voice, every condition unknown (never under LACI)
        let guided = cfg != 1.0 && !matches!(voice, Voice::None) && session.is_none();
        let mut twin = None;
        if guided {
            let tp = prompt::build(c, &ids, &Voice::None, &[], silence)?;
            let tc = Cond { speaker: None, buckets: tp.buckets.clone(), deltas: Vec::new() };
            let mut ts = self.lm.seq(&self.ops, tp.rows + max_frames + 2, tp.rows)?;
            self.lm.forward(&self.ops, &mut ts, &tp.ids, &tc, false)?;
            twin = Some((ts, tc));
        }
        let mut step_cond = Cond { speaker: None, buckets: Vec::new(), deltas: deltas.to_vec() };
        // the stream: the silence-prefill rows, then the generated ones
        let mut stream: Vec<[u32; 10]> = p.prefill.clone();
        let (mut eos, mut countdown) = (None::<usize>, 0usize);
        let mut step = 0usize;
        let mut ctl = laci::Control::default();
        let mut sent = 0usize;
        loop {
            let mut logits = self.lm.logits(&seq, 1)?;
            if let Some((ts, _)) = &twin {
                let u = self.lm.logits(ts, 1)?;
                for (l, u) in logits.iter_mut().zip(&u) {
                    *l = u + cfg * (*l - u);
                }
            }
            if let Some(ss) = session.as_mut() {
                // the readout of the last decode step first; a stop set there makes the row sampled now the last
                if step > 0 {
                    if let Some((pos, m)) = seq.probe.as_ref().and_then(|pr| pr.readout) {
                        ss.on_step(&mut ctl, text_span.1, step, pos, m)?;
                        if let Some(pr) = seq.probe.as_mut() {
                            pr.windows = ctl.mask.clone().unwrap_or_default();
                        }
                    }
                }
                let boost = ss.stop_boost();
                if boost != 0.0 {
                    for cb in 0..ncb {
                        logits[cb * c.audio_vocab + c.eoa as usize] += boost;
                    }
                }
            }
            let total = stream.len() - np;
            // the cold start: a codebook whose delayed stream has not begun is padding (the stream row counts the
            // silence rows; a prompt that ends in a recording's rows has begun)
            let stream_row = if p.audio_prefix && p.prefill.is_empty() { usize::MAX } else { stream.len() };
            let r = sample::row(&logits, ncb, s, c.codebook, c.pad, c.eoa, &stream[np..], stream_row, rng);
            stream.push(r);
            if eos.is_none() {
                if let Some(mc) = (0..ncb).rev().find(|cb| r[*cb] == c.eoa) {
                    eos = Some(total.saturating_sub(mc));
                    countdown = ncb + 1;
                }
            }
            let attempt_rows = stream.len() - session.as_ref().map_or(np, |ss| ss.attempt_start);
            let mut finished = stream.len() - np >= max_frames || attempt_rows >= max_frames;
            if countdown > 0 {
                countdown -= 1;
                if countdown == 0 {
                    finished = true;
                }
            }
            if let Some(ss) = session.as_mut() {
                ss.append(r, eos.map(|e| e + np));
                finished |= ctl.stop;
            }
            if (stream.len() - np).is_multiple_of(15) || finished {
                progress(stream.len() - np)?;
            }
            if finished {
                if let Some(ss) = session.as_mut() {
                    if let Some(cont) = ss.finish_attempt() {
                        let f = ss.failures.last().cloned();
                        eprintln!("{ARCH}: LACI {} (bin {}) at row {}, back to row {}, retry {}", f.as_ref().map_or("?", |f| f.kind), f.as_ref().map_or(0, |f| f.bin),
                                  f.as_ref().map_or(0, |f| f.detect_row), cont.n_rows, ss.attempts);
                        stream.truncate(cont.n_rows);
                        *rng = Rng::new(cont.seed);
                        seq = new_seq(&stream[np..])?;
                        if let Some(pr) = seq.probe.as_mut() {
                            pr.windows = ss.probe_mask().unwrap_or_default();
                        }
                        ctl = laci::Control::default();
                        eos = None;
                        countdown = 0;
                        step = 0;
                        continue;
                    }
                    eprintln!("{ARCH}: LACI {}", ss.summary());
                }
                break;
            }
            // the rows no later rollback can change go downstream
            let fin = match session.as_mut() {
                Some(ss) => ss.drain(false).1,
                None => stream.len(),
            };
            if let Some(cb) = o.on_rows.as_mut() {
                if fin > np + sent {
                    sent = fin - np;
                    cb(&stream[np..fin], None)?;
                }
            }
            let ids: Vec<u32> = r[..ncb].iter().copied().chain(std::iter::once(c.text_vocab)).collect();
            // the request's constant deltas and the spans active at this generated frame (Zyphra's steer_fn)
            if !spans.is_empty() {
                let f = (stream.len() - np) as f32;
                let mut d = deltas.to_vec();
                for (start, end, sd) in spans {
                    let w = markup::span_weight(*start, *end, f);
                    if w <= 0.0 {
                        continue;
                    }
                    for (src, v) in sd {
                        match d.iter_mut().find(|(s0, _)| s0 == src) {
                            Some((_, acc)) => acc.iter_mut().zip(v).for_each(|(a, b)| *a += w * b),
                            None => d.push((*src, v.iter().map(|b| w * b).collect())),
                        }
                    }
                }
                step_cond.deltas = d;
            }
            self.lm.forward(&self.ops, &mut seq, &ids, &step_cond, false)?;
            step += 1;
            if let Some((ts, _)) = twin.as_mut() {
                self.lm.forward(&self.ops, ts, &ids, &step_cond, false)?;
            }
        }
        let mut rows: Vec<[u32; 10]> = stream[np..].to_vec();
        if let Some(ss) = session.as_ref() {
            if let Some(e) = ss.eos_frame() {
                eos = Some(e);
                rows.truncate((e + ncb).min(rows.len()));
            }
        }
        if let Some(cb) = o.on_rows.as_mut() {
            cb(&rows, Some(eos.unwrap_or(rows.len())))?;
        }
        Ok((Take { rows, eos }, p.prefill))
    }

    /// Frames [from, to) of delayed rows (frame f's codebook c on row f + c), codes outside the codebook as 0
    fn undelay(&self, buffer: &[[u32; 10]], from: usize, to: usize) -> Vec<[u32; 10]> {
        let c = &self.lm.c;
        (from..to).map(|f| {
            let mut fr = [0u32; 10];
            for (cb, x) in fr.iter_mut().enumerate().take(c.ncb) {
                let v = buffer.get(f + cb).map_or(c.pad, |r| r[cb]);
                *x = if (v as usize) < c.codebook { v } else { 0 };
            }
            fr
        }).collect()
    }

    /// The final frame count of a take's buffer (prefill + rows): complete frames, before its end and any end of audio
    fn target_frames(&self, buffer: &[[u32; 10]], eos: Option<usize>, np: usize) -> usize {
        frames_before_end(buffer, self.lm.c.ncb, self.lm.c.eoa, eos, np)
    }

    /// A take's rows to PCM: undelayed (frame f's codebook c on row f + c), the complete frames before the end, the
    /// silence frames decoded ahead and their samples cut
    pub fn decode_take(&self, dec: &Decoder, take: &Take, prefill: &[[u32; 10]]) -> Result<Vec<f32>> {
        let c = &self.lm.c;
        let ncb = c.ncb;
        let buffer: Vec<[u32; 10]> = prefill.iter().chain(&take.rows).copied().collect();
        let mut target = buffer.len().saturating_sub(ncb - 1);
        if let Some(e) = take.eos {
            target = target.min(e + prefill.len());
        }
        for (i, r) in buffer.iter().enumerate() {
            if let Some(mc) = (0..ncb).rev().find(|cb| r[*cb] == c.eoa) {
                target = target.min(i.saturating_sub(mc));
            }
        }
        if target <= prefill.len() {
            return Ok(Vec::new());
        }
        let codes = self.undelay(&buffer, 0, target);
        let pcm = dec.decode(&self.ops, &codes)?;
        Ok(pcm[(prefill.len() * dec.hop).min(pcm.len())..].to_vec())
    }
}

impl AudioEngine for Zonos3 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: 600.0, steps: 0, cfg: 1.0, rate: RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: self.voices.keys().cloned().collect(),
            languages: ["en", "fr", "es", "de", "it", "pt", "nl", "pl", "ru", "sv", "tr", "uk", "cs", "da", "fi", "el", "hu", "no", "ro", "sk", "bg", "zh",
                        "ja", "ko", "ar", "he", "vi", "id", "th", "tl", "ms", "ka", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "ur"].iter().map(|s| s.to_string()).collect(),
            instructions: false,
            design: false,
            clone: self.spk.is_some(),
            clone_needs_text: false,
            clone_takes_text: false,
            direction: false,
            events: Vec::new(),
            requires: None,
            enrolled: false,
            speakers: 0,
        })
    }
    fn markup(&self) -> Vec<String> {
        // pauses, the start marker, IPA and steering spans: the server would make their brackets pauses and say the
        // words ("pause two")
        markup::KEEP.iter().map(|k| k.to_string()).collect()
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        self.run(req, progress, None)
    }

    fn streams(&self) -> bool {
        true
    }

    fn edits(&self) -> bool {
        self.lm.edit.is_some() && self.aligner_path.is_some() && self.enc_paths.is_some() && self.spk.is_some()
    }

    fn edit(&self, req: &nextsycl_audio::EditRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<nextsycl_audio::Edited> {
        let _one = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        self.edit_speech(req, progress)
    }

    fn generate_streamed(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>, chunk: &mut dyn FnMut(&[f32]) -> Result<()>)
                         -> Result<Audio> {
        self.run(req, progress, Some(chunk))
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {}: backbone {:.0} MiB, decoders {}, {} voices", self.ops.gpu.name, self.lm.bytes as f64 / (1u64 << 20) as f64,
                     self.decoders.iter().map(|d| d.variant.clone()).collect::<Vec<_>>().join(", "), self.voices.len())]
    }
}

impl Zonos3 {
    /// Speech editing (edit.rs): the regions of `req.original` the target changes said anew by the edit LM, the rest
    /// of the recording's codes kept, the whole decoded once
    fn edit_speech(&self, req: &nextsycl_audio::EditRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<nextsycl_audio::Edited> {
        use crate::edit as ed;
        let t0 = Instant::now();
        let c = &self.lm.c;
        let span_end = c.pad + 1;
        if self.lm.edit.is_none() || c.audio_vocab <= span_end as usize {
            return Err(Error(format!("{ARCH}: speech editing needs the edit backbone (backbone-ptq-edit.gguf) and its adapter (edit/accents.gguf)")));
        }
        let ap = self.aligner_path.as_ref().ok_or_else(|| Error(format!("{ARCH}: speech editing needs FuseAlign (fusealign-multilingual-ns4-step60000-f32.gguf)")))?;
        let spk = self.spk.as_ref().ok_or_else(|| Error(format!("{ARCH}: speech editing needs the speaker encoder")))?;
        let (orig, target) = (req.original.trim(), req.target.trim());
        if orig.is_empty() || target.is_empty() {
            return Err(Error("an edit needs the recording's transcript (original) and the words it should say (target)".into()));
        }
        let pcm = if req.rate == RATE { req.samples.clone() } else { spk::resample_any(&req.samples, req.rate, RATE) };
        let duration = pcm.len() as f64 / RATE as f64;
        if duration > 30.0 {
            return Err(Error(format!("the recording is {duration:.1} s; an edit takes at most 30 s (FuseAlign aligns 30 s at once)")));
        }
        if duration < 0.5 {
            return Err(Error("the recording is shorter than half a second".into()));
        }
        let step = |p: &mut dyn FnMut(Step) -> Result<()>, phase: &'static str, at: u32, of: u32| p(Step { phase, at, of, seconds: t0.elapsed().as_secs_f64() });
        // the words of the original, placed
        step(progress, "align", 0, 1)?;
        let placed = {
            let fa = align::FuseAlign::load(&self.ops, &Weights::open(ap)?)?;
            fa.align(&self.ops, &pcm, RATE, &ed::align_tokens(orig))?
        };
        let words = ed::aligned_words(&placed.iter().map(|p| (p.word.clone(), p.start, p.end, p.presence)).collect::<Vec<_>>());
        // the recording's codes and its voice
        step(progress, "encode", 0, 1)?;
        let codes = self.reference_codes(&pcm, RATE, &mut |m| eprintln!("{m}"))?;
        let emb = spk.embed(&self.ops, &pcm, RATE)?;
        let total = codes.len();
        let (tall, twords) = ed::target_tokens(target);
        let target_words: Vec<String> = target.split_whitespace().map(str::to_string).collect();
        let spans = ed::diff_spans(&words, &tall, total, 0.08, 3)?;
        ed::validate_spans(&spans, total, 3)?;
        let windowed = total as f64 / ed::FRAME_RATE > 16.0;
        let windows = if windowed {
            ed::plan_windows(&spans, &words, &target_words, &twords, total, 4.0, target)?
        } else {
            vec![ed::Window { start: 0, end: total, regions: (0..spans.len()).collect(), text: target.to_string() }]
        };
        let expected = ed::expected_seconds(&spans, &words, &twords);
        let max_span = 30.0;
        let min_frames: Vec<usize> = expected.iter().map(|s| ((0.7 * s * ed::FRAME_RATE) as usize).min((max_span * ed::FRAME_RATE) as usize)).collect();
        let max_tokens = ((max_span * ed::FRAME_RATE) as usize + c.ncb + 1).min(MAX_FRAMES);
        let s = Sampling { temperature: num(&AudioRequest { extra: req.extra.clone(), ..Default::default() }, "temperature")?.unwrap_or(0.75), ..Sampling::default() };
        let spec = markup::Spec::read(&c.spec);
        let ereq = AudioRequest { extra: req.extra.clone(), ..Default::default() };
        let normalize = match given(&ereq, "normalize") {
            None | Some("on") | Some("1") | Some("true") => true,
            Some("off") | Some("0") | Some("false") => false,
            Some(v) => return Err(Error(format!("normalize {v:?}: on or off"))),
        };
        let lang = markup::grammar_language(req.language.as_deref(), target);
        let mut fills: Vec<Vec<[u32; 10]>> = vec![Vec::new(); spans.len()];
        let mut ended = vec![false; spans.len()];
        let voice = Voice::Embedding(emb);
        for w in &windows {
            let wcodes = &codes[w.start..w.end];
            let rel: Vec<ed::Span> = w.regions.iter().map(|k| (spans[*k].0 - w.start, spans[*k].1)).collect();
            // the window's text normalized in the request's language, as Zyphra's EditEngine.fill does
            // (text_normalization, default on; option normalize off: as written)
            let wtext = markup::normalize(&w.text, if normalize { &lang } else { "none" });
            let ids = markup::text_ids(&self.tok, c, &spec, &wtext)?;
            let p = prompt::build(c, &ids, &voice, &[], 0)?;
            let cond = Cond { speaker: p.speaker.as_ref().map(|(r, v)| Ok::<_, Error>((*r, self.lm.project_speaker(v)?))).transpose()?, buckets: p.buckets.clone(),
                              deltas: Vec::new() };
            let mut wfills: Vec<Vec<[u32; 10]>> = Vec::new();
            for (ri, region) in w.regions.iter().enumerate() {
                step(progress, "tokens", *region as u32, spans.len() as u32)?;
                let pid = ed::edit_prompt(&p.ids, wcodes, &rel, &wfills, c.ncb, c.pad, c.text_vocab, c.eoa, span_end)?;
                let rows = pid.len() / c.row_width();
                if rows + max_tokens > c.ctx {
                    return Err(Error(format!("the edit prompt is {rows} rows; the model holds {}", c.ctx)));
                }
                // NS_Z3_EDIT_DUMP=<dir>: the first region's prompt for the reference (ids.npy, ids.cond.json)
                if let (Ok(d), true) = (std::env::var("NS_Z3_EDIT_DUMP"), *region == 0) {
                    let d = std::path::Path::new(&d);
                    let f: Vec<f32> = pid.iter().map(|x| *x as f32).collect();
                    check::npy(&d.join("ids.npy"), &[rows, c.row_width()], &f)?;
                    let (srow, semb) = match (&p.speaker, &voice) {
                        (Some((r, _)), Voice::Embedding(e)) => (*r as i64, e.clone()),
                        _ => (-1, Vec::new()),
                    };
                    let j = serde_json::json!({"condition_buckets": p.buckets.iter().map(|b| b.1).collect::<Vec<_>>(), "condition_row": p.buckets.first().map_or(0, |b| b.0),
                                               "prefill_frames": 0, "speaker_embedding": semb, "speaker_row": srow,
                                               "speaker_rows": i64::from(srow >= 0)});
                    std::fs::write(d.join("ids.cond.json"), j.to_string()).map_err(|x| Error(x.to_string()))?;
                }
                let mut seq = self.lm.seq_edit(&self.ops, rows + max_tokens + 2, rows)?;
                self.lm.forward(&self.ops, &mut seq, &pid, &cond, false)?;
                let mut rng = Rng::new(req.seed);
                let mut gen: Vec<[u32; 10]> = Vec::new();
                let (mut eos, mut countdown) = (None::<usize>, 0usize);
                loop {
                    let mut logits = self.lm.logits(&seq, 1)?;
                    // span-end waits for min_frames (+ the codebook's delay)
                    for cb in 0..c.ncb {
                        if gen.len() < min_frames[*region] + cb {
                            logits[cb * c.audio_vocab + span_end as usize] = f32::NEG_INFINITY;
                        }
                    }
                    let r = sample::row(&logits, c.ncb, &s, c.codebook, c.pad, span_end, &gen, gen.len(), &mut rng);
                    let row = gen.len();
                    gen.push(r);
                    if eos.is_none() {
                        if let Some(mc) = (0..c.ncb).rev().find(|cb| r[*cb] == span_end) {
                            eos = Some(row.saturating_sub(mc));
                            countdown = c.ncb + 1;
                        }
                    }
                    let mut done = gen.len() >= max_tokens;
                    if countdown > 0 {
                        countdown -= 1;
                        done |= countdown == 0;
                    }
                    if done {
                        break;
                    }
                    let next: Vec<u32> = r[..c.ncb].iter().copied().chain(std::iter::once(c.text_vocab)).collect();
                    self.lm.forward(&self.ops, &mut seq, &next, &Cond::default(), false)?;
                }
                // NS_Z3_EDIT_DUMP: the first region's generated rows as fed back (rows.npy, for `check --stages forced`
                // with the file `edit` there) and the window it was made from (codes.npy, edit.json)
                if let (Ok(d), true) = (std::env::var("NS_Z3_EDIT_DUMP"), *region == 0) {
                    let d = std::path::Path::new(&d);
                    let f: Vec<f32> = gen.iter().flat_map(|r| r.iter().map(|x| *x as f32).chain(std::iter::once(c.text_vocab as f32))).collect();
                    check::npy(&d.join("rows.npy"), &[gen.len(), c.row_width()], &f)?;
                    let f: Vec<f32> = wcodes.iter().flat_map(|r| r.iter().map(|x| *x as f32)).collect();
                    check::npy(&d.join("codes.npy"), &[wcodes.len(), c.ncb], &f)?;
                    let j = serde_json::json!({"text": w.text, "spans": rel, "min_frames": min_frames[*region], "eos_frame": eos, "seed": req.seed,
                                               "max_tokens": max_tokens});
                    std::fs::write(d.join("edit.json"), j.to_string()).map_err(|x| Error(x.to_string()))?;
                    std::fs::write(d.join("edit"), b"").map_err(|x| Error(x.to_string()))?;
                }
                gen.pop();
                let (fill, end) = ed::extract_fill(&gen, eos, c.ncb, c.codebook as u32);
                eprintln!("{ARCH}: edit region {} of {}: {} frames (expected {:.2} s, at least {} frames){}", region + 1, spans.len(), fill.len(), expected[*region],
                          min_frames[*region], if end { "" } else { ", no span-end" });
                fills[*region] = fill.clone();
                ended[*region] = end;
                wfills.push(fill);
                let _ = ri;
            }
        }
        step(progress, "decode", 0, 1)?;
        let spliced = ed::splice(&codes, &spans, &fills);
        let dec = &self.decoders[0];
        let mut out = dec.decode(&self.ops, &spliced)?;
        output(&mut out, dec.headroom_db);
        let mut regions = Vec::new();
        let mut shift: i64 = 0;
        let tok_words = |a: f64, b: f64| -> String { words.iter().filter(|w| w.end > a && w.start < b).map(|w| w.word.clone()).collect::<Vec<_>>().join(" ") };
        for (j, (st, len)) in spans.iter().enumerate() {
            let n = fills[j].len() as i64;
            let (o0, o1) = (*st as f64 / ed::FRAME_RATE, (st + len) as f64 / ed::FRAME_RATE);
            let e0 = (*st as i64 + shift) as f64 / ed::FRAME_RATE;
            regions.push(nextsycl_audio::EditedRegion { original: (o0, o1), edited: (e0, e0 + n as f64 / ed::FRAME_RATE), old_words: tok_words(o0, o1), new_words: String::new() });
            shift += n - *len as i64;
        }
        eprintln!("{ARCH}: edit of {duration:.1} s: {} regions, {:.1} s by {:.2} s", spans.len(), out.len() as f64 / RATE as f64, t0.elapsed().as_secs_f64());
        Ok(nextsycl_audio::Edited { audio: Audio { rate: dec.rate, channels: 1, samples: out }, regions })
    }

    /// The request's sound; `sink`: handed over in pieces as it is made (streaming)
    fn run(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>, sink: Option<&mut dyn FnMut(&[f32]) -> Result<()>>) -> Result<Audio> {
        let _one = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let text = req.prompt.trim();
        if text.is_empty() {
            return Err(Error("no text to say".into()));
        }
        let c = &self.lm.c;
        let embedding_mode = match given(req, "speaker-mode") {
            None | Some("reference") => false,
            Some("embedding") => true,
            Some(m) => return Err(Error(format!("speaker-mode {m:?}: reference or embedding"))),
        };
        let controls = given(req, "controls").map(steer::parse_controls).transpose()?.unwrap_or_default();
        let raw_controls = controls.iter().any(|(id, d)| *d != 0.0 && self.steer.as_ref().is_some_and(|s| s.dirs.iter().any(|x| &x.id == id && x.source.is_none())));
        progress(Step { phase: "voice", at: 0, of: 1, seconds: 0.0 })?;
        let mut voice = self.voice(req, embedding_mode || raw_controls)?;
        let mut deltas = Vec::new();
        if !controls.is_empty() {
            let st = self.steer.as_ref().ok_or_else(|| Error(format!("{ARCH}: controls need the steering bundle (steer.gguf)")))?;
            let emb = match &voice {
                Voice::Embedding(v) => Some(v.as_slice()),
                _ => None,
            };
            let r = st.resolve(&controls, emb, c.dim)?;
            if let Some(e) = r.embedding {
                voice = Voice::Embedding(e);
            }
            deltas = r.deltas;
        }
        let s = Sampling {
            temperature: num(req, "temperature")?.unwrap_or(0.75),
            topk: num(req, "top-k")?.unwrap_or(80),
            top_p: num(req, "top-p")?.unwrap_or(0.69),
            repetition_penalty: num(req, "repetition-penalty")?.unwrap_or(2.4),
            ..Sampling::default()
        };
        let silence_given: Option<usize> = num(req, "silence")?;
        let silence = silence_given.unwrap_or(3);
        if silence > 30 {
            return Err(Error("silence: 0 to 30 frames".into()));
        }
        let mut buckets: Vec<Option<usize>> = vec![None; c.cond_names.len()];
        for (i, name) in c.cond_names.iter().enumerate() {
            let opt = match name.as_str() {
                "speaking_rate" => "speaking-rate",
                n => n,
            };
            if let Some(v) = num::<f64>(req, opt)? {
                buckets[i] = Some(prompt::bucket(&c.cond_edges[i], v));
            }
        }
        let leveled = match given(req, "level") {
            None | Some("on") | Some("1") | Some("true") => true,
            Some("off") | Some("0") | Some("false") => false,
            Some(v) => return Err(Error(format!("level {v:?}: on or off"))),
        };
        let cfg = req.cfg.unwrap_or(1.0);
        if !cfg.is_finite() || cfg < 0.0 {
            return Err(Error(format!("cfg {cfg}: a scale of 0 or more (1: none)")));
        }
        let dec = self.decoder(req)?;
        // `seconds` caps the whole when given; without it a long text is said to its end (600 s at most)
        let seconds = req.seconds.unwrap_or(self.defaults().max_seconds).clamp(1.0, self.defaults().max_seconds);
        // the markup, the normalizer (the request's language; option normalize off: as written), the spans
        let normalize = match given(req, "normalize") {
            None | Some("on") | Some("1") | Some("true") => true,
            Some("off") | Some("0") | Some("false") => false,
            Some(v) => return Err(Error(format!("normalize {v:?}: on or off"))),
        };
        let parsed = markup::parse(text)?;
        let lang = markup::grammar_language(req.language.as_deref(), text);
        let normed = if normalize { markup::normalize(&parsed.text, &lang) } else { markup::normalize(&parsed.text, "none") };
        let (clean, spans) = markup::take_spans(&normed, &parsed.spans)?;
        if !spans.is_empty() && self.steer.is_none() {
            return Err(Error(format!("{ARCH}: [steer] spans need the steering bundle (steer.gguf)")));
        }
        let spec = markup::Spec::read(&c.spec);
        let global_pace = self.steer.as_ref().map_or(1.0, |st| st.duration_factor(&controls));
        let chars: Vec<char> = clean.chars().collect();
        let parts = stretch_ranges(&clean, MAX_CHARS);
        let mut rng = Rng::new(req.seed);
        // LACI (option laci on) and short texts (option short-text auto, the default): the language's alignment heads
        let server_lang = laci::server_language(&lang);
        let heads = self.laci.get(laci::heads_file(&server_lang));
        let laci_on = match given(req, "laci") {
            None | Some("off") | Some("0") | Some("false") => false,
            Some("on") | Some("1") | Some("true") => true,
            Some(v) => return Err(Error(format!("laci {v:?}: on or off"))),
        };
        let short_auto = match given(req, "short-text") {
            None | Some("auto") | Some("on") => true,
            Some("off") => false,
            Some(v) => return Err(Error(format!("short-text {v:?}: auto or off"))),
        };
        if laci_on && heads.is_none() {
            return Err(Error(format!("laci: no alignment heads for language {server_lang} (the model's laci/ files: {})",
                                     if self.laci.is_empty() { "none - pull the model again".to_string() } else { self.laci.keys().cloned().collect::<Vec<_>>().join(", ") })));
        }
        if laci_on && cfg != 1.0 {
            return Err(Error("laci: needs cfg 1 (the probe reads the conditional take alone)".into()));
        }
        let mut sink = sink;
        let mut out_stream = sink.as_mut().map(|s| stream::Out { level: leveled.then(|| stream::Leveler::new(dec.rate, TARGET_DBFS)), headroom_db: dec.headroom_db,
                                                                sink: &mut **s, all: Vec::new() });
        let mut out = Vec::new();
        let mut frames_done = 0usize;
        let budget = (seconds * FPS) as usize;
        for (i, (a, b)) in parts.iter().copied().enumerate() {
            let part: String = chars[a..b].iter().collect();
            let mut ids = markup::text_ids(&self.tok, c, &spec, &part)?;
            let mut short = None;
            if let Some(h) = heads.filter(|_| short_auto && cfg == 1.0 && spans.iter().all(|s| s.start >= b || s.end <= a)) {
                let payload = |t: &str| -> Result<Vec<u32>> {
                    let v = markup::text_ids(&self.tok, c, &spec, t)?;
                    Ok(v[1..v.len() - 1].to_vec())
                };
                let blank = |id: u32| id >= c.offset && self.tok.decode(&[id - c.offset]).trim().is_empty();
                if let Some((full, filler)) = laci::short_prompt(&part, &server_lang, &payload, &blank)? {
                    eprintln!("{ARCH}: short text {part:?}: under LACI with the filler, cut where it starts (token {filler})");
                    ids = markup::text_ids(&self.tok, c, &spec, &full)?;
                    short = Some((h, Some(filler)));
                }
            }
            let laci_opt = short.or_else(|| heads.filter(|_| laci_on).map(|h| (h, None)));
            // this stretch's spans, shifted to it and placed by its characters
            let local: Vec<markup::Span> = spans.iter().filter(|s| s.start < b && s.end > a)
                .map(|s| markup::Span { start: s.start.max(a) - a, end: s.end.min(b) - a, controls: s.controls.clone() }).collect();
            let mut sched: Vec<SpanSched> = Vec::new();
            if let Some(st) = &self.steer {
                let place = markup::schedule(&local, b - a, markup::FRAMES_PER_CHAR * global_pace, &|ctl| st.duration_factor(ctl));
                for (sp, (f0, f1)) in local.iter().zip(place) {
                    sched.push((f0, f1, st.span_deltas(&sp.controls, c.dim)?));
                }
            }
            // a stretch that starts with the start marker was trained without the silence prefill
            let silence = if part.starts_with(markup::START) { silence_given.unwrap_or(0) } else { silence };
            let left = budget.saturating_sub(frames_done);
            if left < 15 {
                break;
            }
            // a take that never ends (no end-of-audio drawn) stops at four rows a byte of its text and a few
            // seconds more: the slowest steered pace says about three
            let max_frames = left.min(MAX_FRAMES).min(4 * part.len() + 90);
            if let Some(o) = out_stream.as_mut() {
                if i > 0 {
                    o.send(&vec![0.0f32; (GAP * RATE as f32) as usize])?;
                }
            }
            // streaming: the final rows decoded as they come (prefix decode behind a left context)
            let silence_rows = prompt::build(c, &ids, &voice, &buckets, silence)?.prefill;
            let mut prefix = stream::Prefix::new(silence_rows.len());
            let streaming = out_stream.is_some();
            let mut on_rows = |rows: &[[u32; 10]], end: Option<usize>| -> Result<()> {
                let Some(o) = out_stream.as_mut() else { return Ok(()) };
                let buffer: Vec<[u32; 10]> = silence_rows.iter().chain(rows).copied().collect();
                // complete frames, but none past an end of audio already drawn (Zyphra's PrefixDecoder.ready: the rows
                // of the delay countdown after it are not speech - a frame past the end used as the right context of the
                // last one made the stream's last frame differ from the whole take's, up to 266 of 32768 at seed 11)
                let complete = self.target_frames(&buffer, None, silence_rows.len());
                let last = end.map(|e| self.target_frames(&buffer, Some(e), silence_rows.len()));
                if let Some((from, emitted, to, end)) = prefix.due(complete, last) {
                    let pcm = dec.decode(&self.ops, &self.undelay(&buffer, from, end))?;
                    let a = ((emitted - from) * dec.hop).min(pcm.len());
                    let z = ((to - from) * dec.hop).min(pcm.len());
                    prefix.emitted = to;
                    o.send(&pcm[a..z])?;
                }
                Ok(())
            };
            let opts = TakeOpts { laci: laci_opt, on_rows: if streaming { Some(&mut on_rows) } else { None } };
            let (take, prefill) = self.take_with(&ids, &voice, &buckets, &deltas, &sched, cfg, &s, silence, max_frames, &mut rng, &mut |n| {
                progress(Step { phase: "tokens", at: (frames_done + n) as u32, of: budget as u32, seconds: t0.elapsed().as_secs_f64() })
            }, opts)?;
            frames_done += take.rows.len();
            let t_gen = t0.elapsed().as_secs_f64();
            if out_stream.is_some() {
                eprintln!("{ARCH}: stretch {}/{} streamed: {} rows by {:.2} s", i + 1, parts.len(), take.rows.len(), t_gen);
                continue;
            }
            progress(Step { phase: "decode", at: i as u32, of: parts.len() as u32, seconds: t_gen })?;
            let mut pcm = self.decode_take(dec, &take, &prefill)?;
            eprintln!("{ARCH}: stretch {}/{}: {} rows ({:.1} s of sound), generated by {:.2} s, decoded ({}) by {:.2} s", i + 1, parts.len(), take.rows.len(),
                      pcm.len() as f64 / RATE as f64, t_gen, dec.variant, t0.elapsed().as_secs_f64());
            if !leveled {
                output(&mut pcm, dec.headroom_db);
            }
            if !out.is_empty() {
                out.extend(std::iter::repeat_n(0.0f32, (GAP * RATE as f32) as usize));
            }
            out.extend(pcm);
        }
        if let Some(mut o) = out_stream {
            o.finish()?;
            out = o.all;
        } else if leveled {
            level(&mut out, RATE, TARGET_DBFS);
            output(&mut out, 0.0);
        }
        progress(Step { phase: "decode", at: parts.len() as u32, of: parts.len() as u32, seconds: t0.elapsed().as_secs_f64() })?;
        Ok(Audio { rate: dec.rate, channels: 1, samples: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_decodes_no_frame_past_an_end_of_audio_drawn() {
        // 3 prefill rows, then rows whose codebook 4 draws the end of audio at row 20 (frame 16): before the take
        // finishes (the delay countdown) the rows after it must not count as complete frames
        let (eoa, pad) = (4096u32, 4097u32);
        let mut rows: Vec<[u32; 10]> = (0..25).map(|i| [i as u32 % 4096; 10]).collect();
        rows[20][4] = eoa;
        for r in rows.iter_mut().skip(21) {
            r[4] = pad;
        }
        assert_eq!(frames_before_end(&rows[..20], 10, eoa, None, 3), 11, "no end yet: every complete frame");
        assert_eq!(frames_before_end(&rows, 10, eoa, None, 3), 16, "the end drawn: frames before it only");
        assert_eq!(frames_before_end(&rows, 10, eoa, Some(12), 3), 15, "the aligned end frame wins when earlier");
    }

    #[test]
    fn long_texts_are_said_in_sentences() {
        assert_eq!(stretches("  Short one. ", 400), vec!["Short one.".to_string()]);
        let t = "First sentence here. Second one is a bit longer! Third? Fourth.";
        let s = stretches(t, 30);
        assert!(s.iter().all(|x| x.chars().count() <= 30), "{s:?}");
        assert_eq!(s.join(" "), t);
        let long = "word ".repeat(50);
        let s = stretches(&long, 40);
        assert!(s.iter().all(|x| x.chars().count() <= 40 && !x.is_empty()));
    }

    #[test]
    fn leveling_holds_a_fading_take_at_the_target() {
        // 20 s of a tone fading from -26 to -42 dB, with a pause in the middle
        let rate = 44100;
        let mut v: Vec<f32> = (0..20 * rate).map(|i| {
            let t = i as f32 / rate as f32;
            let amp = 10f32.powf((-26.0 - 0.8 * t) / 20.0) * 2f32.sqrt();
            if (9.0..10.0).contains(&t) { 0.0 } else { amp * (2.0 * std::f32::consts::PI * 220.0 * t).sin() }
        }).collect();
        level(&mut v, rate as u32, TARGET_DBFS);
        let rms = |a: usize, b: usize| 20.0 * ((v[a * rate..b * rate].iter().map(|x| x * x).sum::<f32>() / ((b - a) * rate) as f32).sqrt()).log10();
        for (a, b) in [(1, 3), (5, 8), (11, 14), (16, 19)] {
            assert!((rms(a, b) - TARGET_DBFS).abs() < 1.0, "{a}-{b} s: {}", rms(a, b));
        }
        assert!(v[(9.5 * rate as f32) as usize].abs() < 1e-6, "the pause stays silent");
        let mut quiet = vec![0f32; 1000];
        level(&mut quiet, rate as u32, TARGET_DBFS);
        assert!(quiet.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn leveling_brings_back_a_tail_faded_45_db() {
        // 30 s of a tone fading from -24 to -69 dB: the old gate (35 dB under the whole take's loud frames) took the
        // tail for silence
        let rate = 44100;
        let mut v: Vec<f32> = (0..30 * rate).map(|i| {
            let t = i as f32 / rate as f32;
            10f32.powf((-24.0 - 1.5 * t) / 20.0) * 2f32.sqrt() * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
        }).collect();
        level(&mut v, rate as u32, TARGET_DBFS);
        for s in [5usize, 15, 25] {
            let w = &v[s * rate..(s + 2) * rate];
            let db = 10.0 * (w.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / w.len() as f64).log10();
            assert!((db + 19.0).abs() < 1.5, "{s} s: {db}");
        }
    }

    #[test]
    fn the_limiter_bends_only_above_the_knee() {
        let mut v = vec![0.5, -0.5, 1.5, -3.0];
        output(&mut v, 0.0);
        assert_eq!(v[0], 0.5);
        assert_eq!(v[1], -0.5);
        assert!(v[2] > KNEE && v[2] < 1.0);
        assert!(v[3] < -KNEE && v[3] >= -1.0);
        let mut g = vec![1.0];
        output(&mut g, 3.0);
        assert!((g[0] - 0.70794576).abs() < 1e-5);
    }

    #[test]
    fn a_folder_of_ggufs_is_found() {
        let d = std::env::temp_dir().join(format!("z3files-{}", std::process::id()));
        std::fs::create_dir_all(d.join("voices/English")).unwrap();
        for f in ["backbone-ptq.gguf", "zac-keep2.gguf", "spk-encoder.gguf", "voices/English/Savannah_en-American_F_1.codes.npy",
                  "voices/English/Savannah_en-American_F_1.embedding.npy"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let f = files_in(&d).unwrap();
        std::fs::remove_dir_all(&d).ok();
        assert!(f.contains_key("model") && f.contains_key("decoder") && f.contains_key("speaker-encoder"));
        assert!(!f.contains_key("decoder-full"));
        assert!(f["voice-savannah"].ends_with("Savannah_en-American_F_1.codes.npy"));
        assert!(f["voice-savannah-emb"].ends_with("Savannah_en-American_F_1.embedding.npy"));
    }
}
