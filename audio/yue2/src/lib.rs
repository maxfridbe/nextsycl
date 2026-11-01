//! YuE2 (M-A-P's YuE2-3B, non-commercial: CC BY-NC 4.0) on SYCL: a whole song with vocals from a style prompt and
//! lyrics, 48 kHz stereo. One AR-NAR Mixture-of-Transformers backbone (lm.rs) writes a score and the semantic codes
//! autoregressively, then makes the acoustic latents by flow matching; YuE2-Vae (vae.rs) turns them into sound.
//!
//! ```text
//!   style + lyrics -> prompt.rs: [EOD] instruction / [Tags] style / [Lyrics] lyrics (the cot mode's instruction)
//!   cot full | melody: the score (ABC text: melody and chords, or melody alone) drawn after ABC_START, until ABC_END
//!                      (or given: a cover's melody, an earlier song's score with new words)
//!   the semantic codes, 25 a second, until MUSIC_END (guided against the instruction alone when cfg is not 1)
//!   the flow: 32 midpoint steps from torch.randn noise, the NAR path over the codes' AR rows -> latents [frames, 64]
//!   -> the VAE, tiled -> stereo
//! ```
//!
//! A cover (a recording given: `AudioRequest::source`) gets its score from SheetSage2 (sheet/: the recording to a lead
//! sheet in ABC - the melody alone for cot melody, the default for a cover; melody and chords for cot full), loaded
//! when the model's files include it (the `sheetsage` role, on the second card when there are two), and sings it with
//! the request's style and words: the recording itself never reaches the music stages, as upstream's cover.
//!
//! The weights are scragnog/YuE2-GGUF's: the language model Q8_0 (every matrix kept in its blocks on the card), the
//! VAE float32. The whole of it is resident: ~3.9 GB of language model, the song's caches (half, 112 KiB a token) and
//! the flow's rows on the first card, the VAE on the second when there is one. `check` compares the stages with the
//! dumps of reference/yue2/ref.py (upstream's yue2_infer wheel, as it is).

pub mod check;
mod ffi;
pub mod lm;
pub mod prompt;
pub mod sample;
pub mod sheet;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Step};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::torch_rng::Mt19937;
use nextsycl_gguf::Gguf;
use nextsycl_tok::Tokenizer;

use crate::lm::{Cache, Lm, Scratch, CONTEXT, HIDDEN, LATENT};
use crate::prompt::{Cot, ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END, MUSIC_START};
use crate::sample::{Rng, Sampling};
use crate::sheet::SheetSage;
use crate::vae::{Vae, RATE};

pub const ARCH: &str = "yue2";
/// the files: the language model, the VAE, and SheetSage2 (optional: covers - a recording's score - when present)
pub const ROLES: &[&str] = &["lm", "vae", "sheetsage"];
/// codes a second
pub const FPS: f32 = 25.0;
/// the longest song: the semantic phase's 9000 codes
pub const MAX_SECONDS: f32 = 360.0;
pub const ODE_STEPS: u32 = 32;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "cot", env: "NS_YUE2_COT", value: "full|melody|off", help: "what the model writes before the music: a score of melody and chords (full, the default), a melody alone (melody: covers), or none (off)", at: At::Both },
    EngineOption { name: "abc", env: "", value: "TEXT", help: "a score (ABC) to sing over instead of one drawn (cot melody or full): a cover's melody, or the score of an earlier song (its `score`) to sing it again with new words - same seed and style", at: At::Request },
    EngineOption { name: "sing-lead", env: "", value: "1|0", help: "a cover of an instrumental recording: its lead line (SheetSage2's Ins voice) moved to the voice YuE2 sings (on; 0: the score as transcribed - an instrumental cover)", at: At::Request },
    EngineOption { name: "score-tokens", env: "", value: "4096", help: "the most tokens the score may take (32 to 4096; a longer score is cut and closed)", at: At::Request },
    EngineOption { name: "score-only", env: "", value: "true|false", help: "with a recording (source): only its score - SheetSage2's transcription, the melody for cot melody, melody and chords for cot full - comes back (the answer's score), no song is made", at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "YuE2-3B (songs with vocals from a style and lyrics: an AR-NAR Mixture-of-Transformers, a score first, flow-matched latents, 48 kHz stereo; CC BY-NC 4.0)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    Ok(Box::new(Yue2::load(files, gpus, o, log)?))
}

/// A directory holding the GGUFs (yue2-lm-*.gguf, yue2-vae-standard-*.gguf): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let find = |pre: &str| -> Option<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(pre) && n.ends_with(".gguf"))).collect();
        v.sort();
        // the Q8_0 first: the file this engine reads
        v.iter().find(|p| p.to_string_lossy().contains("q8_0")).cloned().or_else(|| v.first().cloned())
    };
    let mut f = ModelFiles::new();
    f.insert("lm".into(), find("yue2-lm-").ok_or_else(|| Error(format!("{}: no yue2-lm-*.gguf", dir.display())))?);
    f.insert("vae".into(), find("yue2-vae-standard").ok_or_else(|| Error(format!("{}: no yue2-vae-standard-*.gguf", dir.display())))?);
    if let Some(p) = find("sheetsage2-") {
        f.insert("sheetsage".into(), p);
    }
    Ok(f)
}

pub fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: plain floats as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
}

const GIB: f64 = (1u64 << 30) as f64;

/// What a song is made from, once the request is read
#[derive(Clone, Debug)]
pub struct Song {
    pub cot: Cot,
    pub style: String,
    pub lyrics: String,
    /// the score given (its text), cot melody or full
    pub abc: Option<String>,
    pub seed: u64,
    /// the semantic phase's guidance (1: one branch)
    pub cfg: f32,
    pub max_codes: usize,
    pub max_score: usize,
    pub steps: usize,
}

/// What the stages made (the check and the score read it)
#[derive(Default, Clone)]
pub struct Made {
    pub prefix: Vec<u32>,
    pub abc: Vec<u32>,
    pub sem_prefix: Vec<u32>,
    pub codes: Vec<u32>,
    pub latents: Vec<f32>,
}

pub struct Yue2 {
    pub lm: Lm,
    pub vae: Vae,
    /// SheetSage2, when its file is given: a recording's score (covers)
    pub sheet: Option<SheetSage>,
    pub tok: Tokenizer,
    settings: LoadOptions,
    work: Mutex<Scratch>,
    vwork: Mutex<Scratch>,
    last: Mutex<Option<String>>,
    load_s: f64,
}

impl Yue2 {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Yue2> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
        let vgpu = gpus.get(1).unwrap_or(gpu);
        let path = |r: &str| files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
        let (lmp, vp) = (path("lm")?, path("vae")?);
        let sp = files.get("sheetsage").cloned();
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        // the plan: the language model's file (the position table stays on the host) and a minute's caches and flow
        // rows on its card, the VAE and a tile's activations on its own (or the same); SheetSage2 beside the VAE (its
        // 1.3 GB and a window's encoder activations and attention scores, ~1 GiB - it runs before the song, so the
        // VAE's tiles never overlap it)
        let lm_need = size(&lmp) + (1u64 << 30);
        let v_need = size(&vp) + (1u64 << 30) + sp.as_deref().map_or(0, |p| size(p) + (1u64 << 30));
        let plan: Vec<(&Arc<Gpu>, u64)> = if Arc::ptr_eq(gpu, vgpu) { vec![(gpu, lm_need + v_need)] } else { vec![(gpu, lm_need), (vgpu, v_need)] };
        for (g, need) in &plan {
            let (total, free) = g.memory()?;
            let free = free.unwrap_or(total);
            if *need > free {
                return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", *need as f64 / GIB, g.name, free as f64 / GIB)));
            }
        }
        let f = Gguf::open(&lmp).map_err(|e| Error(e.0))?;
        // the file carries Qwen's tiktoken pattern in its own key and no tokenizer.ggml.pre: Qwen2's split
        let tok = Tokenizer::from_gguf_as(&f, "qwen2").map_err(Error)?;
        if tok.tokens.len() != MUSIC_START as usize || tok.id("<|endoftext|>") != Some(EOD) {
            return Err(Error(format!("{}: not YuE2's tokenizer ({} tokens, 151851 expected)", lmp.display(), tok.tokens.len())));
        }
        let lm = Lm::load(&f, gpu, log)?;
        let vf = Gguf::open(&vp).map_err(|e| Error(e.0))?;
        let vae = Vae::load(&vf, vgpu)?;
        let sheet = match &sp {
            Some(p) => {
                let s = SheetSage::load(p, vgpu)?;
                log(format!("{ARCH}: SheetSage2 (covers: a recording's score) {:.2} GiB on {}", s.bytes as f64 / GIB, vgpu.name));
                Some(s)
            }
            None => None,
        };
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: the language model {:.2} GiB on {}, the VAE {:.2} GiB on {} - loaded in {load_s:.1} s",
                    lm.bytes as f64 / GIB, gpu.name, vae.bytes as f64 / GIB, vgpu.name));
        Ok(Yue2 { lm, vae, sheet, tok, settings: o.clone(), work: Mutex::new(Scratch::default()), vwork: Mutex::new(Scratch::default()), last: Mutex::new(None),
                  load_s })
    }

    fn opt(&self, req: &AudioRequest, name: &str, env: &str) -> Option<String> {
        req.extra.get(name).cloned().or_else(|| if env.is_empty() { None } else { self.settings.setting(env) }).filter(|v| !v.trim().is_empty())
    }

    /// The request as a song: the pipeline's defaults where it says nothing (cot full, guidance 1 - 1.01 for cot off -,
    /// up to six minutes: the model ends the song, 32 ODE steps)
    pub fn song(&self, req: &AudioRequest) -> Result<Song> {
        // a cover's own default: the melody (upstream's recommendation - the recording's chords are the new style's)
        let cot_s = req.extra.get("cot").cloned().filter(|v| !v.trim().is_empty())
            .or_else(|| req.source.as_ref().map(|_| "melody".to_string()))
            .or_else(|| self.opt(req, "cot", "NS_YUE2_COT"))
            .unwrap_or_else(|| "full".into());
        let cot = Cot::parse(&cot_s).ok_or_else(|| Error(format!("cot {cot_s}: full, melody or off")))?;
        let abc = self.opt(req, "abc", "").map(|a| a.trim().to_string());
        if abc.is_some() && cot == Cot::Off {
            return Err(Error("abc: a score is sung with cot melody or full (cot off writes none)".into()));
        }
        if req.source.is_some() {
            if abc.is_some() {
                return Err(Error("a recording (source) and a score (abc): the recording's score is its transcription - give one or the other".into()));
            }
            if cot == Cot::Off {
                return Err(Error("a cover sings over the recording's score: cot melody (its melody, the default) or full (melody and chords)".into()));
            }
            if self.sheet.is_none() {
                return Err(Error("a recording (source) needs SheetSage2 (the model's sheetsage file: nextsycl models pull yue2-3b-q8)".into()));
            }
        }
        let style = req.prompt.trim().to_string();
        if style.is_empty() {
            return Err(Error("the style (instructions) is empty: genre, mood, voice, instruments".into()));
        }
        let lyrics = req.lyrics.clone().unwrap_or_default();
        let seconds = req.seconds.unwrap_or(MAX_SECONDS);
        if !(1.0..=MAX_SECONDS).contains(&seconds) {
            return Err(Error(format!("{seconds} s: from 1 to {MAX_SECONDS} s (the song may end sooner: the model decides)")));
        }
        let max_score = match self.opt(req, "score-tokens", "") {
            Some(v) => v.trim().parse::<usize>().map_err(|_| Error(format!("score-tokens {v}: a whole number")))?.clamp(32, sample::ABC.max_tokens),
            None => sample::ABC.max_tokens,
        };
        let cfg = req.cfg.unwrap_or(cot.default_cfg());
        if !(0.0..=20.0).contains(&cfg) {
            return Err(Error(format!("cfg {cfg}: 0 to 20")));
        }
        Ok(Song { cot, style, lyrics, abc, seed: req.seed, cfg, max_codes: ((seconds * FPS).ceil() as usize).clamp(1, sample::SEMANTIC.max_tokens), max_score,
                  steps: req.steps.unwrap_or(ODE_STEPS).clamp(1, 200) as usize })
    }

    /// The score's draw from the first phase's prompt in `c` (its last row's logits given): its tokens
    fn draw_score(&self, s: &mut Scratch, c: &mut Cache, first: Vec<f32>, song: &Song, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>)
                  -> Result<Vec<u32>> {
        let samp = Sampling { max_tokens: song.max_score, min_tokens: sample::ABC.min_tokens.min(song.max_score), ..sample::ABC };
        let mut rng = Rng::new(song.seed);
        let (mut hist, mut out) = (Vec::new(), Vec::new());
        let mut logits = first;
        for step in 0..samp.max_tokens {
            // the candidates: the text tokens, then ABC_END
            let mut sc: Vec<f32> = logits[..EOD as usize].to_vec();
            sc.push(logits[ABC_END as usize]);
            let end = EOD as usize;
            let t = sample::draw(&mut sc, end, &samp, &hist, step, 1, &mut rng);
            if t == end {
                break;
            }
            hist.push(t);
            out.push(t as u32);
            progress(Step { phase: "score", at: out.len() as u32, of: samp.max_tokens as u32, seconds: t0.elapsed().as_secs_f64() })?;
            if step + 1 == samp.max_tokens {
                break;
            }
            self.lm.step(s, &mut [c], &[t as u32])?;
            let x = s.buf("dx").ptr();
            logits = self.lm.logits(s, x, 1, 0, ABC_END as usize + 1)?;
        }
        Ok(out)
    }

    /// The semantic codes, from the prompts in `pos` (and `neg`, guided) whose last rows' logits are given
    #[allow(clippy::too_many_arguments)]
    fn draw_codes(&self, s: &mut Scratch, pos: &mut Cache, mut neg: Option<&mut Cache>, first: (Vec<f32>, Option<Vec<f32>>), song: &Song, t0: Instant,
                  progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<u32>> {
        // NS_YUE2_MIN_CODES (debug): the song may not end before this many codes - a worst case for measuring memory, no song to keep
        let floor = std::env::var("NS_YUE2_MIN_CODES").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(sample::SEMANTIC.min_tokens);
        let samp = Sampling { max_tokens: song.max_codes, min_tokens: floor.min(song.max_codes), ..sample::SEMANTIC };
        let mut rng = Rng::new(song.seed);
        let keep = if song.cot == Cot::Off { 3 } else { 1 };
        let n = CODEC_SIZE as usize + 1;
        let (mut c, mut u) = first;
        let mut hist: Vec<usize> = Vec::new();
        for step in 0..samp.max_tokens {
            // the logits over MUSIC_END (index 0) and the codes (1 + code), guided: u + cfg (c - u)
            let mut sc: Vec<f32> = match &u {
                Some(u) => c.iter().zip(u).map(|(c, u)| u + song.cfg * (c - u)).collect(),
                None => c.clone(),
            };
            let t = sample::draw(&mut sc, 0, &samp, &hist, step, keep, &mut rng);
            if t == 0 {
                break;
            }
            hist.push(t);
            progress(Step { phase: "tokens", at: hist.len() as u32, of: samp.max_tokens as u32, seconds: t0.elapsed().as_secs_f64() })?;
            if step + 1 == samp.max_tokens {
                break;
            }
            let tok = CODEC_OFFSET + t as u32 - 1;
            match neg.as_deref_mut() {
                Some(nc) => {
                    self.lm.step(s, &mut [&mut *pos, nc], &[tok, tok])?;
                    let x = s.buf("dx").ptr();
                    let both = self.lm.logits(s, x, 2, MUSIC_END as usize, n)?;
                    c = both[..n].to_vec();
                    u = Some(both[n..].to_vec());
                }
                None => {
                    self.lm.step(s, &mut [&mut *pos], &[tok])?;
                    let x = s.buf("dx").ptr();
                    c = self.lm.logits(s, x, 1, MUSIC_END as usize, n)?;
                }
            }
        }
        Ok(hist.into_iter().map(|t| t as u32 - 1).collect())
    }

    /// The flow over one chunk: from `noise` [frames, 64] under the AR rows in `c` (the chunk's prompt, codes and
    /// MUSIC_END), `steps` midpoint steps: the latents [frames, 64] on the host (nar.py's CachedNAR.solve)
    pub fn solve(&self, s: &mut Scratch, c: &Cache, noise: &[f32], steps: usize, mut on_step: impl FnMut(usize, &DevBuf, &DevBuf) -> Result<()>) -> Result<Vec<f32>> {
        let g = &self.lm.gpu;
        let frames = noise.len() / LATENT;
        let pos = self.lm.positions(frames + 2)?;
        let state = DevBuf::from_f32(g, noise)?;
        let mid = DevBuf::f32(g, noise.len())?;
        let v = DevBuf::f32(g, noise.len())?;
        let dt = 1.0 / steps as f64;
        let logit = |t: f64| (t / (1.0 - t)).ln().clamp(-20.0, 20.0);
        let temb = |raw: f64| DevBuf::from_f32(g, &self.lm.time_embedding(raw));
        for step in 0..steps {
            let t = 1.0 - step as f64 * dt;
            let e = temb(logit(t))?;
            self.lm.velocity(s, c, &state, frames, &e, &pos, v.ptr())?;
            self.lm.axpy(mid.ptr(), state.ptr(), v.ptr(), noise.len(), -(dt / 2.0) as f32)?;
            let e = temb(logit(t - dt / 2.0))?;
            self.lm.velocity(s, c, &mid, frames, &e, &pos, v.ptr())?;
            self.lm.axpy(state.ptr(), state.ptr(), v.ptr(), noise.len(), -dt as f32)?;
            on_step(step, &mid, &state)?;
        }
        let out = state.to_f32()?;
        if out.iter().any(|x| !x.is_finite()) {
            return Err(Error("the flow made non-finite latents".into()));
        }
        Ok(out)
    }

    /// The whole song's latents: the score, the codes, the flow (`made` what each stage made)
    pub fn latents(&self, song: &Song, made: &mut Made, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<()> {
        let mut s = self.work.lock().unwrap_or_else(|p| p.into_inner());
        let el = || t0.elapsed().as_secs_f64();
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: el() })?;
        made.prefix = prompt::prefix(&self.tok, song.cot, &song.style, &song.lyrics);
        let given: Option<Vec<u32>> = song.abc.as_ref().map(|a| prompt::encode(&self.tok, a));
        if let Some(g) = &given {
            if g.is_empty() || g.iter().any(|t| *t >= EOD) {
                return Err(Error("abc: an empty score, or one past the text vocabulary".into()));
            }
        }
        let score_room = match (&given, song.cot) {
            (_, Cot::Off) => 0,
            (Some(g), _) => g.len() + 2,
            (None, _) => song.max_score + 2,
        };
        let sem_upper = made.prefix.len() + score_room;
        if sem_upper + song.max_codes > CONTEXT {
            return Err(Error(format!("the prompt ({} tokens with its score) and {} codes are past the model's context ({CONTEXT}): shorter lyrics or a shorter song",
                                     sem_upper, song.max_codes)));
        }
        // the song's cache: the prompt, the score, the codes, MUSIC_END, then the flow's rows (one chunk), within the context
        let cap = (sem_upper + 2 * song.max_codes + 4).min(CONTEXT);
        let mut pos = self.lm.cache(cap)?;
        let n_sem = CODEC_SIZE as usize + 1;
        let first_logits = match (song.cot, &given) {
            (Cot::Off, _) => {
                made.sem_prefix = made.prefix.clone();
                let x = self.lm.prefill(&mut s, &mut pos, &made.prefix, &[made.prefix.len() - 1])?;
                self.lm.logits_of(&mut s, &x, MUSIC_END as usize, n_sem)?
            }
            (_, Some(g)) => {
                made.abc = g.clone();
                made.sem_prefix = prompt::semantic(&made.prefix, song.cot, g);
                let x = self.lm.prefill(&mut s, &mut pos, &made.sem_prefix, &[made.sem_prefix.len() - 1])?;
                self.lm.logits_of(&mut s, &x, MUSIC_END as usize, n_sem)?
            }
            (_, None) => {
                let x = self.lm.prefill(&mut s, &mut pos, &made.prefix, &[made.prefix.len() - 1])?;
                let first = self.lm.logits_of(&mut s, &x, 0, ABC_END as usize + 1)?;
                progress(Step { phase: "prompt", at: 1, of: 1, seconds: el() })?;
                made.abc = self.draw_score(&mut s, &mut pos, first, song, t0, progress)?;
                made.sem_prefix = prompt::semantic(&made.prefix, song.cot, &made.abc);
                // the score closed in the cache: its last token (unless the draw stopped at the budget before feeding it),
                // ABC_END and MUSIC_START
                let fed = pos.len - made.prefix.len();
                let rest = &made.sem_prefix[made.prefix.len() + fed..];
                let x = self.lm.prefill(&mut s, &mut pos, rest, &[rest.len() - 1])?;
                self.lm.logits_of(&mut s, &x, MUSIC_END as usize, n_sem)?
            }
        };
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: el() })?;
        let mut neg = if song.cfg != 1.0 {
            let np = prompt::negative(&self.tok, song.cot, &made.abc);
            let mut nc = self.lm.cache(np.len() + song.max_codes + 1)?;
            let x = self.lm.prefill(&mut s, &mut nc, &np, &[np.len() - 1])?;
            let l = self.lm.logits_of(&mut s, &x, MUSIC_END as usize, n_sem)?;
            Some((nc, l))
        } else {
            None
        };
        let (nc, ul) = match neg.as_mut() {
            Some((c, l)) => (Some(c), Some(std::mem::take(l))),
            None => (None, None),
        };
        made.codes = self.draw_codes(&mut s, &mut pos, nc, (first_logits, ul), song, t0, progress)?;
        drop(neg);
        if made.codes.is_empty() {
            return Err(Error("the model ended the song before its first code".into()));
        }
        // the flow: one noise draw for the whole song, cut at the chunks
        let frames = made.codes.len();
        let noise = Mt19937::new(song.seed).randn(frames * LATENT);
        let chunks = prompt::chunks(frames, made.sem_prefix.len(), CONTEXT);
        let mut lat = Vec::with_capacity(frames * LATENT);
        let total = chunks.len() * song.steps;
        for (ci, (a, b)) in chunks.iter().enumerate() {
            let codes: Vec<u32> = made.codes[*a..*b].iter().map(|c| c + CODEC_OFFSET).collect();
            let fed = pos.len.saturating_sub(made.sem_prefix.len());
            if chunks.len() == 1 && fed <= frames && pos.cap >= made.sem_prefix.len() + 2 * frames + 3 {
                // the generation's cache is the chunk's rows already (they are the same causal rows): the codes not fed
                // yet (the last, when the budget ended the song) and MUSIC_END after them
                let mut rest = codes[fed..].to_vec();
                rest.push(MUSIC_END);
                self.lm.prefill(&mut s, &mut pos, &rest, &[])?;
            } else {
                pos.len = 0;
                let mut ids = made.sem_prefix.clone();
                ids.extend_from_slice(&codes);
                ids.push(MUSIC_END);
                if ids.len() + (b - a) + 2 > pos.cap {
                    pos = self.lm.cache(ids.len() + (b - a) + 2)?;
                }
                self.lm.prefill(&mut s, &mut pos, &ids, &[])?;
            }
            let part = self.solve(&mut s, &pos, &noise[a * LATENT..b * LATENT], song.steps, |i, _, _| {
                progress(Step { phase: "flow", at: (ci * song.steps + i + 1) as u32, of: total as u32, seconds: el() })
            })?;
            lat.extend(part);
        }
        made.latents = lat;
        s.clear();
        Ok(())
    }

    /// The latents to stereo [2, n] channel-major (the VAE's card)
    pub fn sound(&self, lat: &[f32], t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<f32>> {
        let mut s = self.vwork.lock().unwrap_or_else(|p| p.into_inner());
        let out = self.vae.decode(&mut s, lat, &mut |a, b| progress(Step { phase: "decode", at: a as u32, of: b as u32, seconds: t0.elapsed().as_secs_f64() }));
        s.clear();
        out
    }
}

impl AudioEngine for Yue2 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: MAX_SECONDS, max_seconds: MAX_SECONDS, steps: ODE_STEPS, cfg: 1.0, rate: RATE }
    }
    fn lyrics(&self) -> bool {
        true
    }
    fn sources(&self) -> bool {
        self.sheet.is_some()
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let t0 = Instant::now();
        let mut song = self.song(req)?;
        let mut made = Made::default();
        *self.last.lock().unwrap_or_else(|p| p.into_inner()) = None;
        if let (Some(src), Some(sheet)) = (&req.source, &self.sheet) {
            // a cover: the recording's score (its melody for cot melody, melody and chords for full), sung in the new style
            let tr = sheet.transcribe(&sheet::wave::prepare(&src.samples, src.channels as usize, src.rate), progress)?;
            let s = if song.cot == Cot::Melody { &tr.melody } else { &tr.full };
            let abc = s.abc.clone().ok_or_else(|| Error(format!("the recording's score: {}", s.error.as_deref().unwrap_or("none was made"))))?;
            // an instrumental recording's tune is in the Ins voice: moved to the Vocal voice, the one YuE2 sings (else it
            // makes an instrumental: measured on a bluegrass Jingle Bells, Whisper heard no words in any of three styles)
            let abc = if self.opt(req, "sing-lead", "").is_some_and(|v| matches!(v.trim(), "0" | "false" | "off")) { abc } else { prompt::sung_lead(&abc).0 };
            *self.last.lock().unwrap_or_else(|p| p.into_inner()) = Some(abc.clone());
            if self.opt(req, "score-only", "").is_some_and(|v| v.trim() == "true" || v.trim() == "1") {
                return Ok(Audio { rate: RATE, channels: 2, samples: Vec::new() });
            }
            song.abc = Some(abc.trim().to_string());
        } else if self.opt(req, "score-only", "").is_some_and(|v| v.trim() == "true" || v.trim() == "1") {
            return Err(Error("score-only: a recording's score - give the recording (source)".into()));
        }
        self.latents(&song, &mut made, t0, progress)?;
        // a debug dump of what the stages made (raw little-endian): two runs compared stage by stage
        if let Ok(d) = std::env::var("NS_YUE2_DUMP") {
            let p = Path::new(&d);
            let ints = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(p.join("abc.u32"), ints(&made.abc)).map_err(|e| Error(e.to_string()))?;
            std::fs::write(p.join("codes.u32"), ints(&made.codes)).map_err(|e| Error(e.to_string()))?;
            std::fs::write(p.join("latents.f32"), f32_bytes(&made.latents)).map_err(|e| Error(e.to_string()))?;
        }
        if song.cot != Cot::Off {
            *self.last.lock().unwrap_or_else(|p| p.into_inner()) = Some(String::from_utf8_lossy(&self.tok.decode_bytes(&made.abc)).into_owned());
        }
        let wav = self.sound(&made.latents, t0, progress)?;
        let n = wav.len() / 2;
        let samples = (0..n).flat_map(|i| [wav[i], wav[n + i]]).collect();
        Ok(Audio { rate: RATE, channels: 2, samples })
    }
    fn score(&self) -> Option<String> {
        self.last.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    fn report(&self) -> Vec<String> {
        let on = if Arc::ptr_eq(&self.lm.gpu, &self.vae.gpu) { self.lm.gpu.name.clone() } else { format!("{} + {}", self.lm.gpu.name, self.vae.gpu.name) };
        let sheet = self.sheet.as_ref().map_or(String::new(), |s| format!("; SheetSage2 for covers, {:.2} GiB on {}", s.bytes as f64 / GIB, s.gpu.name));
        vec![format!("{ARCH} on {on} (the language model Q8_0, {:.2} GiB{sheet}; loaded in {:.0} s)", self.lm.bytes as f64 / GIB, self.load_s)]
    }
}

/// HIDDEN re-exported for the check
pub const WIDTH: usize = HIDDEN;
