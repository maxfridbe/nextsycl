//! OmniVoice (k2-fsa / Xiaomi) on Intel Arc (SYCL): zero-shot speech in 646 languages at 24 kHz - a voice cloned from a
//! recording and its transcript, a voice designed from attributes (gender, age, pitch, whisper, an English accent, a
//! Chinese dialect), or the model's own pick; non-verbal tags in the text (`[laughter]`, `[sigh]` ...).
//!
//! ```text
//!   the prompt (text.rs): style tokens (<|denoise|> with a recording, the language, the attributes), the transcript and
//!   text, a recording's codes (codec.rs: Higgs Audio v2's encoder), T mask codes for the target (T from the text's
//!   length at the recording's pace, or a fixed pace: upstream's duration estimate)
//!   -> 32 steps of masked diffusion (lm.rs: Qwen3-0.6B, bidirectional): a pass over the prompt and over the target
//!      alone (classifier-free guidance 2.0); each masked position's best code and its log probability; the codebooks
//!      penalised by depth (5 a codebook), Gumbel noise on the positions (temperature 5), the best k unmasked (k by a
//!      shifted schedule, t_shift 0.1) - after the last step every code is set
//!   -> the codec's decoder: 25 frames a second of 8 codes to 24 kHz; silences over 0.5 s shortened, the level set (a
//!      clone's like its recording, else a peak of 0.5), 0.1 s fades and padding
//! ```
//!
//! A text longer than 30 s (by the estimate) is said in chunks of about 15 s at sentence ends, each with the
//! recording (or, without one, the first chunk's sound as the voice), cross-faded.
//!
//! The weights are omnivoice.cpp's GGUFs of the checkpoint (`nextsycl models pull omnivoice`: the backbone Q8_0 and the
//! codec Q8_0 with its convolutions half; any of the release's variants loads); `check` compares the stages with the
//! reference's dumps (reference/omnivoice/ref.py: upstream's own code on the checkpoint, float32 on the CPU).
//!
//! License: the weights are CC-BY-NC 4.0 (non-commercial; the training data's terms); upstream's code is Apache 2.0, the
//! codec's (Boson AI's Higgs Audio v2 tokenizer) Apache 2.0.

pub mod check;
pub mod codec;
mod ffi;
pub mod lm;
pub mod ops;
pub mod post;
pub mod text;
pub mod weights;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech, Step};
use nextsycl_core::{DevBuf, Gpu};

use crate::codec::Codec;
use crate::lm::{Lm, Taps, Work};
use crate::ops::{bytes, ibytes, read_f32, Ops};
use crate::text::{Durations, Languages, Text};
use crate::weights::Weights;

pub const ARCH: &str = "omnivoice";
pub const ROLES: &[&str] = &["model", "codec"];
/// frames a second (24,000 / 960)
pub const FPS: f32 = 25.0;
/// the longest answer (a long text goes in chunks; this bounds a request)
const MAX_SECONDS: f32 = 600.0;
/// the longest recording to clone (upstream warns past 20 s: slower, and a worse clone)
const MAX_REFERENCE_S: f32 = 30.0;
/// what a load asks beyond the weights (bytes): a short request's passes. A long one takes more - 0.8 GiB over the 1.6
/// loaded at a 29 s recording cloned into 31 s of speech (B65, 2026-11-16) - which the front door keeps for it (the
/// card's vram_gib 2.6, then the peak it measures); the load must not ask more than that plan, or a start the front
/// door placed would end while loading (2026-11-16: it asked weights + 2 GiB and refused on a B70 with 2.9 GiB free)
const LOAD_MARGIN: u64 = 1 << 28;
/// a `[sigh]`'s time in the length estimate (seconds), and the shortest answer without a recording: the two departures
/// from upstream's estimate that keep short lines' words (text::Durations::frames; measured 2026-11-16, see
/// docs/ports/omnivoice.md)
pub const SIGH_S: f64 = 0.8;
pub const FLOOR_S: f64 = 2.0;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "speed", env: "NS_OV_SPEED", value: "1.0", help: "how fast it speaks: the estimated length divided by this (upstream's speed)",
                   at: At::Request },
    EngineOption { name: "duration", env: "NS_OV_DURATION", value: "seconds", help: "a fixed length in seconds instead of the estimate (upstream's duration)",
                   at: At::Request },
    EngineOption { name: "sigh-seconds", env: "NS_OV_SIGH_SECONDS", value: "0.8",
                   help: "the time a [sigh] gets in the length estimate (0: upstream's, its letters - short lines then lose their words)", at: At::Request },
    EngineOption { name: "min-seconds", env: "NS_OV_MIN_SECONDS", value: "2.0",
                   help: "the shortest answer without a recording (0: upstream's - one-word lines then often come out as a hum)", at: At::Request },
    EngineOption { name: "t-shift", env: "NS_OV_T_SHIFT", value: "0.1", help: "the unmasking schedule's shift (smaller: more codes set late)", at: At::Request },
    EngineOption { name: "layer-penalty", env: "NS_OV_LAYER_PENALTY", value: "5.0", help: "how much later codebooks wait to be unmasked", at: At::Request },
    EngineOption { name: "position-temperature", env: "NS_OV_POSITION_TEMPERATURE", value: "5.0",
                   help: "randomness in which positions are unmasked first (0: by score alone, deterministic)", at: At::Request },
    EngineOption { name: "denoise", env: "NS_OV_DENOISE", value: "1|0", help: "with a recording: ask for clean speech (the <|denoise|> token)", at: At::Request },
    EngineOption { name: "postprocess", env: "NS_OV_POSTPROCESS", value: "1|0", help: "shorten silences over 0.5 s in the answer", at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "OmniVoice (speech in 646 languages, 24 kHz: voices cloned from a recording and its transcript, or designed from attributes; non-verbal tags)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(OmniVoice::load(files, gpu, o, log)?))
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// How a request is made: upstream's generation config
#[derive(Clone, Debug)]
pub struct Gen {
    pub steps: usize,
    pub guidance: f32,
    pub t_shift: f32,
    pub layer_penalty: f32,
    pub position_temperature: f32,
    pub denoise: bool,
    pub postprocess: bool,
    pub speed: f64,
    pub duration: Option<f32>,
    /// a `[sigh]`'s own time in the length estimate (seconds; 0: upstream's, its letters) - text::Durations::frames
    pub sigh_s: f64,
    /// the shortest target without a recording (seconds; 0: upstream's, no floor) - text::Durations::frames
    pub floor_s: f64,
}

impl Default for Gen {
    fn default() -> Gen {
        Gen { steps: 32, guidance: 2.0, t_shift: 0.1, layer_penalty: 5.0, position_temperature: 5.0, denoise: true, postprocess: true, speed: 1.0, duration: None,
              sigh_s: SIGH_S, floor_s: FLOOR_S }
    }
}

/// A recording made ready to clone: its codes [8][frames] (codebook-major), its transcript, its level
#[derive(Clone, Debug)]
pub struct Voice {
    pub codes: Vec<i32>,
    pub frames: usize,
    pub text: String,
    pub rms: Option<f32>,
}

/// One pass's prompt: the token ids ahead of the audio, the audio rows' codes, the target's length
pub struct Prompt {
    /// the style and text tokens (each repeated over the codebooks in the model's input)
    pub tokens: Vec<u32>,
    /// the recording's codes [8][r] ahead of the target (codebook-major), and r
    pub ref_codes: Vec<i32>,
    pub ref_frames: usize,
    /// the target's frames
    pub t: usize,
}

impl Prompt {
    pub fn audio_start(&self) -> usize {
        self.tokens.len()
    }
    pub fn len(&self) -> usize {
        self.tokens.len() + self.ref_frames + self.t
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a check keeps of a run: the first step's input rows, first layer, final norm and logits; a middle step's
/// logits
#[derive(Default)]
pub struct Keep {
    pub emb0: Vec<f32>,
    pub taps: Taps,
    pub logits0: Vec<f32>,
    pub logits_mid: Vec<f32>,
    pub ids_mid: Vec<i32>,
}

/// A small, seeded generator (SplitMix64) for the positions' Gumbel noise
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// upstream's _get_time_steps in float32 (torch.linspace, then the shift), and the codes to unmask at each step
pub fn schedule(total: usize, steps: usize, t_shift: f32) -> Vec<usize> {
    let n = steps + 1;
    let step = 1.0f32 / steps as f32;
    // torch.linspace (CPU): start + i step for the first half, end - (n - 1 - i) step for the rest
    let ts: Vec<f32> = (0..n).map(|i| if i < n / 2 { i as f32 * step } else { 1.0 - (n - 1 - i) as f32 * step })
        .map(|t| t_shift * t / (1.0 + (t_shift - 1.0) * t)).collect();
    let mut rem = total;
    let mut out = Vec::with_capacity(steps);
    for s in 0..steps {
        let k = if s == steps - 1 { rem } else { ((total as f64 * (ts[s + 1] as f64 - ts[s] as f64)).ceil() as usize).min(rem) };
        out.push(k);
        rem -= k;
    }
    out
}

pub struct OmniVoice {
    pub ops: Ops,
    pub text: Text,
    pub langs: Languages,
    pub durs: Durations,
    pub base: Weights,
    pub lm: Lm,
    pub codec: Codec,
    pub codebooks: usize,
    pub vocab: usize,
    pub mask: usize,
    loaded: Instant,
    load_s: f64,
    quant: String,
}

/// A pass's buffers on the card for one prompt
pub struct Seq {
    pub work: Work,
    /// the text rows' embeddings [audio_start, hidden]
    text_rows: DevBuf,
    rows: DevBuf,
    ids: DevBuf,
    tok: DevBuf,
    score: DevBuf,
    pub l: usize,
    pub t: usize,
    pub audio_start: usize,
}

impl OmniVoice {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<OmniVoice> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let (model, codecf) = (role(files, "model")?, role(files, "codec")?);
        let base = Weights::open(&model, "omnivoice-lm")?;
        let cw = Weights::open(&codecf, "omnivoice-tokenizer")?;
        // the backbone half, the embeddings table float32, the codec float32 (1.6 GiB loaded, any variant), and a short
        // request's passes (LOAD_MARGIN). Short of it: no room, not a failure
        let need: u64 = base.g.tensors.iter().filter(|t| t.name != "llm.embed_tokens.weight").map(|t| t.elements() * if t.name.starts_with("llm.layers") { 2 } else { 4 })
            .sum::<u64>() + cw.g.tensors.iter().map(|t| t.elements() * 4).sum::<u64>() + LOAD_MARGIN;
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error::no_room(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                              free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let text = Text::new(&base.g)?;
        let codebooks = base.num("omnivoice.num_audio_codebook")? as usize;
        let vocab = base.num("omnivoice.audio_vocab_size")? as usize;
        let mask = base.num("omnivoice.audio_mask_id")? as usize;
        let lm = Lm::load(&ops, &base)?;
        if lm.heads_n != codebooks * vocab {
            return Err(Error(format!("audio heads of {} rows, {codebooks} codebooks of {vocab} expected", lm.heads_n)));
        }
        log(format!("backbone: {} layers of {} in {:.1} s", lm.layers(), lm.hidden, t0.elapsed().as_secs_f64()));
        let codec = Codec::load(&ops, &cw, true)?;
        if codec.codebooks != codebooks || codec.codes + 1 != vocab {
            return Err(Error(format!("the codec's {} codebooks of {} do not fit the model's {codebooks} of {vocab} (with the mask)", codec.codebooks, codec.codes)));
        }
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let after = gpu.memory()?.1.unwrap_or(free);
        // the file's type: its layers' (a mix names its first projection's and down projection's)
        let ty = |n: &str| base.tensor(n).map(|t| t.ty.name().to_string()).unwrap_or_default();
        let (q, d) = (ty("llm.layers.0.self_attn.q_proj.weight"), ty("llm.layers.0.mlp.down_proj.weight"));
        let quant = if q == d { q } else { format!("{q} / {d}") };
        log(format!("{ARCH} ({quant}) on {} in {load_s:.0} s ({:.1} GiB on the card)", gpu.name, free.saturating_sub(after) as f64 / (1u64 << 30) as f64));
        Ok(OmniVoice { ops, text, langs: Languages::new(), durs: Durations::new(), base, lm, codec, codebooks, vocab, mask, loaded: Instant::now(), load_s, quant })
    }

    /// A recording to its codes, as upstream's create_voice_clone_prompt (a transcript given): 24 kHz, a quiet one
    /// raised to an RMS of 0.1, silences over 0.2 s shortened and the ends trimmed, cut to whole frames, encoded
    pub fn voice(&self, samples: &[f32], rate: u32, transcript: &str, taps: Option<&mut codec::EncTaps>) -> Result<(Voice, Vec<f32>)> {
        let mut w = post::resample(samples, rate, codec::RATE);
        let rms = (w.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / w.len().max(1) as f64).sqrt() as f32;
        if rms > 0.0 && rms < 0.1 {
            w.iter_mut().for_each(|v| *v *= 0.1 / rms);
        }
        let mut w = post::remove_silence(&w, codec::RATE, 200, 100, 200);
        w.truncate(w.len() / codec::HOP * codec::HOP);
        if w.is_empty() {
            return Err(Error("the recording is silence (nothing left after trimming its silences)".into()));
        }
        let (codes, frames) = self.codec.encode(&self.ops, &w, taps)?;
        Ok((Voice { codes, frames, text: text::add_punctuation(transcript), rms: Some(rms) }, w))
    }

    /// The prompt for `text` (the target of `t` frames): style tokens, the transcript and text, the recording's codes
    pub fn prompt(&self, text: &str, voice: Option<&Voice>, lang: Option<&str>, instruct: Option<&str>, denoise: bool, t: usize) -> Prompt {
        let mut tokens = self.text.style(denoise && voice.is_some(), lang, instruct);
        tokens.extend(self.text.text(voice.map(|v| v.text.as_str()), text));
        Prompt { tokens, ref_codes: voice.map(|v| v.codes.clone()).unwrap_or_default(), ref_frames: voice.map_or(0, |v| v.frames), t }
    }

    /// The buffers of a pass over `p`, with its text rows' embeddings and rotary table in place
    pub fn seq(&self, p: &Prompt) -> Result<Seq> {
        let (g, h) = (&self.ops.gpu, self.lm.hidden);
        let (l, t, a) = (p.len(), p.t, p.audio_start());
        let work = self.lm.work(&self.ops, l + t, t)?;
        // the text rows: the embedding's rows of the tokens
        let mut rows = Vec::with_capacity(a * h);
        let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
        for id in &p.tokens {
            if !cache.contains_key(id) {
                let r = self.base.rows(&self.ops, "llm.embed_tokens.weight", *id as usize, *id as usize + 1)?;
                cache.insert(*id, r);
            }
            rows.extend_from_slice(&cache[id]);
        }
        let text_rows = DevBuf::f32(g, (a * h).max(1))?;
        text_rows.write(0, &bytes(&rows))?;
        let pos: Vec<usize> = (0..l).chain(0..t).collect();
        work.cs.write(0, &bytes(&self.lm.rope_rows(&pos)))?;
        let n = l + t - a;
        let ri: Vec<i32> = (a..l + t).map(|r| r as i32).collect();
        let rbuf = DevBuf::new(g, n * 4)?;
        rbuf.write(0, &ibytes(&ri))?;
        let c = self.codebooks;
        Ok(Seq { work, text_rows, rows: rbuf, ids: DevBuf::new(g, n * c * 4)?, tok: DevBuf::new(g, c * t * 4)?, score: DevBuf::f32(g, c * t)?, l, t, audio_start: a })
    }

    /// One pass with the audio rows' codes `audio` ([rows][codebooks] for rows audio_start..l + t: the conditional
    /// sequence's recording and target, then the unconditional target): the best codes [c][t] and their scores
    pub fn step(&self, s: &Seq, audio: &[i32], guidance: f32, keep: Option<&mut Keep>, logits_to: Option<&mut Vec<f32>>) -> Result<(Vec<i32>, Vec<f32>)> {
        let (ops, h, c) = (&self.ops, self.lm.hidden, self.codebooks);
        let w = &s.work;
        if s.audio_start > 0 {
            w.x.copy_within(0, &s.text_rows, 0, s.audio_start * h * 4)?;
        }
        s.ids.write(0, &ibytes(audio))?;
        ops.audio_embed(&self.lm.audio_emb, c, self.vocab, h, &s.ids, &s.rows, s.l + s.t - s.audio_start, w.x.fp(), h)?;
        let mut keep = keep;
        if let Some(k) = keep.as_deref_mut() {
            ops.gpu.sync()?;
            k.emb0 = read_f32(&w.x, 0, (s.l + s.t) * h)?;
        }
        self.lm.pass(ops, w, s.l, s.t, keep.map(|k| &mut k.taps))?;
        if let Some(v) = logits_to {
            ops.gpu.sync()?;
            *v = read_f32(&w.logits, 0, 2 * s.t * self.lm.heads_n)?;
        }
        ops.score(w.logits.fp(), s.t, c, self.vocab, self.mask, guidance, &s.tok, &s.score)?;
        ops.gpu.sync()?;
        let mut tb = vec![0u8; c * s.t * 4];
        s.tok.read(0, &mut tb)?;
        let tok = tb.chunks_exact(4).map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
        Ok((tok, read_f32(&s.score, 0, c * s.t)?))
    }

    /// The audio rows' codes of a step: the recording's and the target's (conditional), the target's again
    fn audio_rows(&self, p: &Prompt, tokens: &[i32]) -> Vec<i32> {
        let (c, t, r) = (self.codebooks, p.t, p.ref_frames);
        let mut v = Vec::with_capacity((r + 2 * t) * c);
        for f in 0..r {
            v.extend((0..c).map(|k| p.ref_codes[k * r + f]));
        }
        for _ in 0..2 {
            for f in 0..t {
                v.extend((0..c).map(|k| tokens[k * t + f]));
            }
        }
        v
    }

    /// The target's codes [c][t] (codebook-major) by upstream's iterative unmasking
    pub fn codes(&self, p: &Prompt, gen: &Gen, rng: &mut Rng, mut keep: Option<&mut Keep>, progress: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<i32>> {
        let (c, t) = (self.codebooks, p.t);
        let s = self.seq(p)?;
        let mut tokens = vec![self.mask as i32; c * t];
        let sched = schedule(c * t, gen.steps, gen.t_shift);
        let mid = gen.steps / 2;
        for (step, k) in sched.iter().enumerate() {
            let audio = self.audio_rows(p, &tokens);
            let mut lg = Vec::new();
            let want = keep.is_some() && (step == 0 || step == mid);
            if want && step == mid {
                if let Some(kp) = keep.as_deref_mut() {
                    kp.ids_mid = tokens.clone();
                }
            }
            let (pred, score) = self.step(&s, &audio, gen.guidance, if step == 0 { keep.as_deref_mut() } else { None }, if want { Some(&mut lg) } else { None })?;
            if let Some(kp) = keep.as_deref_mut() {
                if step == 0 {
                    kp.logits0 = lg;
                } else if step == mid {
                    kp.logits_mid = lg;
                }
            }
            progress(step + 1)?;
            if *k == 0 {
                continue;
            }
            // the masked positions by score: deeper codebooks penalised, Gumbel noise on the positions
            let mut cand: Vec<(f32, usize)> = (0..c * t).filter(|i| tokens[*i] == self.mask as i32).map(|i| {
                let mut v = score[i] - (i / t) as f32 * gen.layer_penalty;
                if gen.position_temperature > 0.0 {
                    let u = rng.uniform();
                    v = v / gen.position_temperature - (-(u + 1e-10).ln() + 1e-10).ln();
                }
                (v, i)
            }).collect();
            cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (_, i) in cand.into_iter().take(*k) {
                tokens[i] = pred[i];
            }
        }
        if tokens.contains(&(self.mask as i32)) {
            return Err(Error("codes left masked after the last step".into()));
        }
        Ok(tokens)
    }

    /// What the request asks for, checked
    fn options(&self, req: &AudioRequest) -> Result<Gen> {
        let mut g = Gen::default();
        let get = |k: &str| req.extra.get(k).map(String::as_str);
        let num = |k: &str| -> Result<Option<f64>> { get(k).map(|v| v.parse::<f64>().map_err(|_| Error(format!("{k}: {v}? (a number)")))).transpose() };
        let flag = |k: &str| -> Result<Option<bool>> {
            get(k).map(|v| match v {
                "1" | "true" | "on" => Ok(true),
                "0" | "false" | "off" => Ok(false),
                _ => Err(Error(format!("{k}: {v}? (1 or 0)"))),
            }).transpose()
        };
        if let Some(s) = req.steps {
            if !(1..=128).contains(&s) {
                return Err(Error(format!("steps {s}: 1 to 128 (32 by default; 16 is faster)")));
            }
            g.steps = s as usize;
        }
        if let Some(c) = req.cfg {
            if !(0.0..=10.0).contains(&c) {
                return Err(Error(format!("cfg {c}: 0 to 10 (2 by default)")));
            }
            g.guidance = c;
        }
        if let Some(v) = num("speed")? {
            if !(0.25..=4.0).contains(&v) {
                return Err(Error(format!("speed {v}: 0.25 to 4")));
            }
            g.speed = v;
        }
        if let Some(v) = num("duration")? {
            if !(0.2..=MAX_SECONDS as f64).contains(&v) {
                return Err(Error(format!("duration {v}: 0.2 to {MAX_SECONDS} s")));
            }
            g.duration = Some(v as f32);
        }
        if let Some(v) = num("sigh-seconds")? {
            if !(0.0..=5.0).contains(&v) {
                return Err(Error(format!("sigh-seconds {v}: 0 to 5")));
            }
            g.sigh_s = v;
        }
        if let Some(v) = num("min-seconds")? {
            if !(0.0..=10.0).contains(&v) {
                return Err(Error(format!("min-seconds {v}: 0 to 10")));
            }
            g.floor_s = v;
        }
        if let Some(v) = num("t-shift")? {
            if !(v > 0.0 && v <= 1.0) {
                return Err(Error(format!("t-shift {v}: in (0, 1]")));
            }
            g.t_shift = v as f32;
        }
        if let Some(v) = num("layer-penalty")? {
            if !(0.0..=50.0).contains(&v) {
                return Err(Error(format!("layer-penalty {v}: 0 to 50")));
            }
            g.layer_penalty = v as f32;
        }
        if let Some(v) = num("position-temperature")? {
            if !(0.0..=50.0).contains(&v) {
                return Err(Error(format!("position-temperature {v}: 0 to 50")));
            }
            g.position_temperature = v as f32;
        }
        if let Some(v) = flag("denoise")? {
            g.denoise = v;
        }
        if let Some(v) = flag("postprocess")? {
            g.postprocess = v;
        }
        Ok(g)
    }

    /// The target's frames for `text`: a fixed duration, else the estimate (at the recording's pace) / speed, at most
    /// `cap` seconds
    fn frames_for(&self, text: &str, voice: Option<&Voice>, g: &Gen, cap: Option<f32>) -> usize {
        let mut t = match g.duration {
            Some(d) => ((d * FPS) as usize).max(1),
            None => self.durs.frames(text, voice.map(|v| v.text.as_str()), voice.map(|v| v.frames), g.speed, g.sigh_s, g.floor_s),
        };
        if let Some(c) = cap {
            t = t.min(((c * FPS) as usize).max(1));
        }
        t
    }

    /// The whole request: codes chunk by chunk, the sound, the post-processing
    #[allow(clippy::too_many_arguments)]
    pub fn say(&self, text: &str, voice: Option<&Voice>, lang: Option<&str>, instruct: Option<&str>, g: &Gen, cap: Option<f32>, seed: u64, t0: Instant,
               progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<f32>> {
        let mut rng = Rng::new(seed);
        let est = self.frames_for(text, voice, g, None);
        let long = g.duration.is_none() && cap.is_none_or(|c| c > 30.0) && est as f32 > 30.0 * FPS;
        let pieces: Vec<String> = if long {
            // about 15 s a chunk: characters by the text's frames a character
            let per = est as f64 / text.chars().count().max(1) as f64;
            text::chunks(text, ((15.0 * FPS as f64) / per) as usize, 3)
        } else {
            vec![text.to_string()]
        };
        let n = pieces.len();
        let mut sounds = Vec::new();
        let mut first: Option<Voice> = None;
        for (i, piece) in pieces.iter().enumerate() {
            // without a recording, the first chunk's sound is the voice of the rest
            let v = voice.or(first.as_ref());
            let pc = if n > 1 && g.duration.is_none() { None } else { cap };
            let mut gi = g.clone();
            if n > 1 {
                gi.duration = None;
            }
            let t = self.frames_for(piece, v, &gi, pc);
            let p = self.prompt(piece, v, lang, instruct, g.denoise, t);
            let steps = g.steps as u32;
            let codes = self.codes(&p, g, &mut rng, None, &mut |s| {
                progress(Step { phase: "steps", at: i as u32 * steps + s as u32, of: n as u32 * steps, seconds: t0.elapsed().as_secs_f64() })
            })?;
            progress(Step { phase: "decode", at: i as u32, of: n as u32, seconds: t0.elapsed().as_secs_f64() })?;
            sounds.push(self.codec.decode(&self.ops, &codes, t)?);
            if voice.is_none() && i == 0 && n > 1 {
                first = Some(Voice { codes, frames: t, text: piece.clone(), rms: None });
            }
        }
        progress(Step { phase: "decode", at: n as u32, of: n as u32, seconds: t0.elapsed().as_secs_f64() })?;
        let raw = post::cross_fade(&sounds, codec::RATE);
        Ok(self.finish(&raw, voice.and_then(|v| v.rms), g))
    }

    /// upstream's _post_process_audio: silences over 0.5 s shortened, the level, the fades and padding
    pub fn finish(&self, raw: &[f32], ref_rms: Option<f32>, g: &Gen) -> Vec<f32> {
        let mut x = if g.postprocess { post::remove_silence(raw, codec::RATE, 500, 100, 100) } else { raw.to_vec() };
        post::level(&mut x, ref_rms);
        post::fade_and_pad(&x, codec::RATE, 0.1, 0.1)
    }

    /// The request's text, recording, language and attributes, checked
    #[allow(clippy::type_complexity)]
    fn spec(&self, req: &AudioRequest) -> Result<(String, Option<Voice>, Option<String>, Option<String>)> {
        let text = req.prompt.trim();
        if text.is_empty() {
            return Err(Error("the text to say is empty".into()));
        }
        if let Some(v) = req.voice.as_deref().filter(|v| !v.is_empty()) {
            return Err(Error(format!("voice {v}: {ARCH} has no built-in voices (clone a recording with its transcript, design one with instructions, or leave \
                                      voice out for the model's own)")));
        }
        let lang = match req.language.as_deref().map(str::trim).filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("auto")) {
            Some(l) => Some(self.langs.resolve(Some(l)).ok_or_else(|| Error(format!("language {l}: not one of {ARCH}'s 646 (a name - english, \
                                                                                         cantonese, swahili ... - or its id: en, yue, sw; or auto)")))?),
            None => None,
        };
        let instruct = match req.instructions.as_deref() {
            Some(i) => text::resolve_instruct(i, text)?,
            None => None,
        };
        let voice = match &req.reference {
            Some(r) => {
                let transcript = r.text.as_deref().map(str::trim).filter(|t| !t.is_empty())
                    .ok_or_else(|| Error("ref_text: the recording's transcript (OmniVoice clones from a recording and what it says)".into()))?;
                let secs = r.samples.len() as f32 / r.rate.max(1) as f32;
                if secs > MAX_REFERENCE_S {
                    return Err(Error(format!("the recording is {secs:.1} s: up to {MAX_REFERENCE_S:.0} s (3-10 s of one speaker clones best)")));
                }
                Some(self.voice(&r.samples, r.rate, transcript, None)?.0)
            }
            None => None,
        };
        Ok((text.to_string(), voice, lang, instruct))
    }
}

impl AudioEngine for OmniVoice {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 15.0, max_seconds: MAX_SECONDS, steps: 32, cfg: 2.0, rate: codec::RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: Vec::new(),
            languages: self.langs.names.clone(),
            instructions: true,
            design: true,
            clone: true,
            clone_needs_text: true,
            clone_takes_text: true,
            direction: true,
            events: text::EVENTS.iter().map(|e| e.to_string()).collect(),
            requires: None,
            enrolled: false,
            speakers: 1,
        })
    }
    fn markup(&self) -> Vec<String> {
        // the non-verbal tags and the bracketed pronunciations are read only as written: the server's normalisation
        // would make "[laughter]" a pause and the word "laughter", which the model then says
        text::KEEP.iter().map(|k| k.to_string()).collect()
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let t0 = Instant::now();
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: 0.0 })?;
        let g = self.options(req)?;
        let (text, voice, lang, instruct) = self.spec(req)?;
        if let Some(s) = req.seconds {
            if s <= 0.0 {
                return Err(Error(format!("{s} s: a length above zero")));
            }
        }
        let cap = req.seconds.map(|s| s.min(MAX_SECONDS)).or(Some(MAX_SECONDS));
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let samples = self.say(&text, voice.as_ref(), lang.as_deref(), instruct.as_deref(), &g, cap, req.seed, t0, progress)?;
        Ok(Audio { rate: codec::RATE, channels: 1, samples })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} ({}, loaded {:.0} s ago)", self.ops.gpu.name, self.quant, self.loaded.elapsed().as_secs_f64())]
    }
}

/// A directory with the two GGUFs (as `models pull` lays them out, or omnivoice.cpp's models/): its files by role -
/// the Q8_0 pair when there are several
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let list: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf")).collect();
    let pick = |prefix: &str| -> Result<PathBuf> {
        list.iter().filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(prefix)))
            .min_by_key(|p| (!p.to_string_lossy().contains("Q8_0"), (*p).clone())).cloned()
            .ok_or_else(|| Error(format!("{}: no {prefix}*.gguf", dir.display())))
    };
    let mut f = ModelFiles::new();
    f.insert("model".into(), pick("omnivoice-base")?);
    f.insert("codec".into(), pick("omnivoice-tokenizer")?);
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_unmasks_every_code() {
        let s = schedule(8 * 143, 32, 0.1);
        assert_eq!(s.len(), 32);
        assert_eq!(s.iter().sum::<usize>(), 8 * 143);
        // t_shift 0.1: few codes early, many late
        assert!(s[0] < s[31]);
        let one = schedule(10, 1, 0.1);
        assert_eq!(one, vec![10]);
    }

    #[test]
    fn the_rng_is_uniform_and_seeded() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        let v: Vec<f32> = (0..1000).map(|_| a.uniform()).collect();
        assert!(v.iter().all(|x| (0.0..1.0).contains(x)));
        assert_eq!(v[10], (0..11).map(|_| b.uniform()).last().unwrap());
        let mean = v.iter().sum::<f32>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05);
    }
}
