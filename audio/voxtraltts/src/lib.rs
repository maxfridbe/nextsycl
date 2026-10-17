//! Voxtral 4B TTS 2603 (Mistral) on Intel Arc (SYCL): speech at 24 kHz in nine languages (English, French, German,
//! Spanish, Italian, Portuguese, Dutch, Arabic, Hindi) in one of its 20 preset voices.
//!
//! ```text
//!   the prompt (prompt.rs): <s> [BEGIN_AUDIO] the voice's rows [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
//!   -> the backbone (lm.rs, Ministral 3B: 26 layers of 3,072): one hidden state a frame (12.5 a second)
//!   -> the acoustic transformer (flow.rs): the semantic code (its head's argmax; [END_AUDIO] ends the speech) and 36
//!      acoustic values by a 7-step Euler flow from noise, guided (cfg 1.2) against a pass without the hidden state
//!   -> the frame's 37 code embeddings summed: the next position's input
//!   -> at the end the Voxtral codec's decoder (codec.rs): 240 samples a frame
//! ```
//!
//! The weights are CrispASR's GGUF of the checkpoint (`nextsycl models pull voxtral-4b-tts`: q8_0; f16 for parity),
//! expanded to half at load; the Tekken vocabulary and the preset voices are in it. `check` compares the stages with
//! the reference's dumps (reference/voxtraltts/ref.py: upstream's own code, float32 on the CPU).
//!
//! What the open release does not do: the checkpoint has the codec's decoder only - no encoder - so no voice is cloned
//! from a recording (vllm-omni refuses ref_audio for it: "The open-source variant only supports preset voices"); its
//! prompt has no instruction, so no voice is designed or directed, and no vocal events are read. A request's style is
//! its preset voice (casual, cheerful, neutral; a language's own speakers) and the guidance.
//!
//! License: the weights, the voices and what they make are CC BY-NC 4.0 (non-commercial).

pub mod check;
pub mod codec;
mod ffi;
pub mod flow;
pub mod lm;
pub mod ops;
pub mod prompt;
pub mod sample;
pub mod weights;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech, Step};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_tok::Tokenizer;

use crate::codec::Codec;
use crate::flow::{Flow, ACOUSTIC, END, LEVELS, SPECIAL};
use crate::lm::Lm;
use crate::ops::Ops;
use crate::prompt::Specials;
use crate::sample::Rng;
use crate::weights::Weights;

pub const ARCH: &str = "voxtral-tts";
/// the GGUF's general.architecture
pub const GGUF_ARCH: &str = "voxtral_tts";
pub const ROLES: &[&str] = &["model"];
pub const FPS: f32 = 12.5;
pub const RATE: u32 = 24_000;
/// the longest run (frames: 160 s)
const MAX_FRAMES: usize = 2000;
/// upstream's guidance (vllm-omni's _DEFAULT_CFG_ALPHA) and Euler steps (its parser's n_decoding_steps)
pub const CFG: f32 = 1.2;
pub const STEPS: usize = 7;
/// the voice a request that names none speaks in
pub const DEFAULT_VOICE: &str = "neutral_female";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "steps", env: "NS_VXT_STEPS", value: "7", help: "the flow's Euler steps a frame (upstream's 7; fewer is faster, the model is trained for 7)",
                   at: At::Both },
    EngineOption { name: "enrolled", env: "NS_VXT_ENROLLED", value: "FILE.npy",
                   help: "experimental: a voice's fitted codes [F, 37] (reference/voxtraltts/enroll.py) as the voice instead of a preset", at: At::Both },
];

/// The languages it speaks, each with the preset voice a request in it that names none gets
pub const LANGUAGES: &[(&str, &str, &str)] = &[
    ("english", "en", "neutral_female"),
    ("french", "fr", "fr_female"),
    ("german", "de", "de_female"),
    ("spanish", "es", "es_female"),
    ("italian", "it", "it_female"),
    ("portuguese", "pt", "pt_female"),
    ("dutch", "nl", "nl_female"),
    ("arabic", "ar", "ar_male"),
    ("hindi", "hi", "hi_female"),
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Voxtral 4B TTS (speech in nine languages in 20 preset voices; no cloning, design or instructions in the open weights)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(VoxtralTts::load(files, gpu, o, log)?))
}

pub struct VoxtralTts {
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    pub sp: Specials,
    pub w: Weights,
    pub bb: Lm,
    pub flow: Flow,
    pub codec: Codec,
    /// the codebooks' embeddings, half [9,088, 3,072]: codebook 0's (8,194 ids) then the 36 acoustic ones' (23 each)
    table: DevBuf,
    /// the preset voices, in the file's order
    pub voices: Vec<String>,
    loaded: Instant,
    load_s: f64,
}

/// What a request asks for, checked
#[derive(Clone, Debug)]
pub struct Spec {
    pub text: String,
    pub voice: String,
    /// an enrolled voice's codes [F][37] (as the model emits them), its prompt rows made from them
    pub enrolled: Option<Vec<Vec<i32>>>,
    pub cfg: f32,
    pub steps: usize,
}

/// What a check forces: the frames' codes (teacher forcing) and their starting noise
#[derive(Default)]
pub struct Force<'a> {
    pub codes: Option<&'a [Vec<i32>]>,
    pub noise: Option<&'a [Vec<f32>]>,
}

/// What a check keeps of a run: the prompt's ids and rows, each frame's hidden state, semantic logits and flow result
#[derive(Default)]
pub struct Keep {
    pub ids: Vec<u32>,
    pub rows: Vec<f32>,
    pub hidden: Vec<Vec<f32>>,
    pub sem: Vec<Vec<f32>>,
    pub lat: Vec<Vec<f32>>,
}

fn given<'a>(req: &'a AudioRequest, name: &str) -> Option<&'a str> {
    req.extra.get(name).map(String::as_str)
}

/// A language's preset voice (full name or code, any case)
pub fn language_voice(lang: &str) -> Option<&'static str> {
    let l = lang.trim().to_lowercase();
    LANGUAGES.iter().find(|(n, c, _)| *n == l || *c == l).map(|x| x.2)
}

impl VoxtralTts {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<VoxtralTts> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let model = files.get("model").cloned().ok_or_else(|| Error(format!("{ARCH}: no model file")))?;
        let w = Weights::open(&model)?;
        // on the card: the matrices in half, the norms and the codec's convolutions in float32; the text embedding, the
        // voices, the vocabulary and the semantic codebook stay on the host; work room (the cache, the codec's chunk): 2 GiB
        let host = |n: &str| n == "token_embd.weight" || n.starts_with("voice.") || n.starts_with("tokenizer.") || n == "codec.semantic_cb.weight";
        let on_card: u64 = w.g.tensors.iter().filter(|t| !host(&t.name)).map(|t| t.elements() * if t.shape.len() == 2 { 2 } else { 4 }).sum();
        let need = on_card + (2u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let (tok, sp) = prompt::tokenizer(&w)?;
        let eps = w.num("llm.norm_eps")? as f32;
        let bb = Lm::load(&ops, &w, "", eps, &lm::rope_freqs(w.num("llm.rope_theta")?))?;
        if bb.hidden != w.num("llm.dim")? as usize || bb.layers() != w.num("llm.n_layers")? as usize {
            return Err(Error(format!("the backbone's {} layers of {} are not the file's", bb.layers(), bb.hidden)));
        }
        log(format!("backbone {} layers of {} in {:.1} s", bb.layers(), bb.hidden, t0.elapsed().as_secs_f64()));
        let flow = Flow::load(&ops, &w)?;
        let codec = Codec::load(&ops, &w)?;
        let table = w.mat(&ops, &["audio_embd.weight"])?;
        if table.k != bb.hidden || table.n < SPECIAL + flow.codes + ACOUSTIC * (LEVELS + SPECIAL) || flow.dim != bb.hidden || codec.codes != flow.codes {
            return Err(Error("the audio embedding, the acoustic transformer and the codec do not fit the backbone".into()));
        }
        let voices = w.texts("voxtral_tts.voice_names");
        for v in &voices {
            w.tensor(&format!("voice.{v}"))?;
        }
        if !voices.iter().any(|v| v == DEFAULT_VOICE) {
            return Err(Error(format!("no voice {DEFAULT_VOICE} in the file")));
        }
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let after = gpu.memory()?.1.unwrap_or(free);
        log(format!("{ARCH} on {} in {load_s:.0} s ({:.1} GiB of weights on the card, {} voices)", gpu.name,
                    free.saturating_sub(after) as f64 / (1u64 << 30) as f64, voices.len()));
        Ok(VoxtralTts { ops, nsd, tok, sp, w, bb, flow, codec, table: table.w, voices, loaded: Instant::now(), load_s })
    }

    /// The preset voice a request names (any case), else its language's, else the default
    pub fn voice_of(&self, req: &AudioRequest) -> Result<String> {
        if let Some(v) = req.voice.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            return self.voices.iter().find(|x| x.eq_ignore_ascii_case(v)).cloned()
                .ok_or_else(|| Error(format!("voice {v}: not one of {ARCH}'s preset voices ({})", self.voices.join(", "))));
        }
        match req.language.as_deref().map(str::trim).filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("auto")) {
            Some(l) => language_voice(l).map(str::to_string).ok_or_else(|| {
                Error(format!("language {l}: not one {ARCH} speaks ({}, or auto)", LANGUAGES.iter().map(|x| x.0).collect::<Vec<_>>().join(", ")))
            }),
            None => Ok(DEFAULT_VOICE.to_string()),
        }
    }

    /// What the request asks for, checked
    pub fn spec(&self, req: &AudioRequest) -> Result<Spec> {
        if req.prompt.trim().is_empty() {
            return Err(Error("the text to say is empty".into()));
        }
        if req.reference.is_some() {
            return Err(Error(format!("{ARCH} does not clone a recording: the open weights have no audio encoder (a preset voice: {})", self.voices.join(", "))));
        }
        if req.instructions.as_deref().is_some_and(|i| !i.trim().is_empty()) {
            return Err(Error(format!("{ARCH} takes no instructions (no voice design or direction): its style is the preset voice")));
        }
        let cfg = req.cfg.unwrap_or(CFG);
        if !(cfg.is_finite() && (0.0..=10.0).contains(&cfg)) {
            return Err(Error(format!("cfg {cfg}: 0 to 10 (1 unguided; upstream 1.2)")));
        }
        let steps = match given(req, "steps") {
            Some(v) => v.parse::<usize>().ok().filter(|s| (1..=64).contains(s)).ok_or_else(|| Error(format!("steps {v}: 1 to 64")))?,
            // the request's own steps (`--steps`, an API request's "steps"): the same Euler steps
            None => match req.steps {
                Some(n) if (1..=64).contains(&n) => n as usize,
                Some(n) => return Err(Error(format!("steps {n}: 1 to 64"))),
                None => STEPS,
            },
        };
        let enrolled = match given(req, "enrolled") {
            Some(p) => Some(enrolled_codes(Path::new(p))?),
            None => None,
        };
        let voice = if enrolled.is_some() { "enrolled".to_string() } else { self.voice_of(req)? };
        Ok(Spec { text: req.prompt.clone(), voice, enrolled, cfg, steps })
    }

    /// The prompt's ids and its rows [L, hidden] (host): the voice's rows at its [AUDIO] positions
    pub fn prompt(&self, s: &Spec) -> Result<(Vec<u32>, Vec<f32>)> {
        let h = self.bb.hidden;
        let voice = match &s.enrolled {
            Some(c) => self.enrolled_rows(c)?,
            None => self.w.all(&format!("voice.{}", s.voice))?,
        };
        let n = voice.len() / h;
        let ids = prompt::ids(&self.tok, &self.sp, &s.text, n);
        let mut rows = Vec::with_capacity(ids.len() * h);
        let mut v = 0;
        for id in &ids {
            if *id == self.sp.audio && v < n {
                rows.extend_from_slice(&voice[v * h..(v + 1) * h]);
                v += 1;
            } else {
                rows.extend(self.w.rows("token_embd.weight", *id as usize, *id as usize + 1)?);
            }
        }
        Ok((ids, rows))
    }

    /// An enrolled voice's prompt rows [F + 1, hidden]: each frame's codes embedded and summed, then the end-of-audio
    /// frame (semantic [END_AUDIO], the acoustic ones [EMPTY_AUDIO]) - upstream's encode_waveforms + encode_tokens
    pub fn enrolled_rows(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let h = self.bb.hidden;
        let out = DevBuf::f32(&self.ops.gpu, h)?;
        let tmp = DevBuf::f32(&self.ops.gpu, h)?;
        let mut eoa = vec![0i32; 1 + ACOUSTIC];
        eoa[0] = END as i32;
        let mut rows = Vec::with_capacity((codes.len() + 1) * h);
        for f in codes.iter().chain(std::iter::once(&eoa)) {
            self.frame_rows(f, out.fp(), &tmp)?;
            self.ops.gpu.sync()?;
            rows.extend(out.to_f32()?);
        }
        Ok(rows)
    }

    /// A frame's 37 rows of the audio embedding summed into `out` (codebook 0 from 0, codebook i > 0 from
    /// 8,194 + 23 (i - 1))
    fn frame_rows(&self, codes: &[i32], out: *mut f32, tmp: &DevBuf) -> Result<()> {
        let h = self.bb.hidden;
        let (s0, a) = ((SPECIAL + self.flow.codes) as i32, (SPECIAL + LEVELS) as i32);
        let idx: Vec<i32> = codes.iter().enumerate().map(|(i, c)| if i == 0 { (*c).clamp(0, s0 - 1) } else { s0 + (i as i32 - 1) * a + (*c).clamp(0, a - 1) }).collect();
        for (i, part) in idx.chunks(8).enumerate() {
            if i == 0 {
                self.ops.embed(&self.table, h, part, out, h, 1)?;
            } else {
                self.ops.embed(&self.table, h, part, tmp.fp(), h, 1)?;
                self.ops.add(out, tmp.fp(), h)?;
            }
        }
        Ok(())
    }

    /// The frames' codes [T][37] (as the model emits them: the specials' 2 on each). `force`: a check's codes and noise;
    /// `keep`: what a check compares
    #[allow(clippy::too_many_arguments)]
    pub fn frames(&self, s: &Spec, max: usize, seed: u64, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>, force: &Force,
                  mut keep: Option<&mut Keep>) -> Result<Vec<Vec<i32>>> {
        let (ops, nsd, h) = (&self.ops, &self.nsd, self.bb.hidden);
        let (ids, rows) = self.prompt(s)?;
        let l = ids.len();
        let max = max.min(MAX_FRAMES);
        let c = self.bb.cache(ops, 1, l, l + max + 1)?;
        c.x.write(0, &bytes(&rows))?;
        self.bb.pass(ops, nsd, &c, l, 0)?;
        let hid = DevBuf::f32(&ops.gpu, h)?;
        let tmp = DevBuf::f32(&ops.gpu, h)?;
        hid.copy_within(0, &c.out, (l - 1) * h * 4, h * 4)?;
        if let Some(k) = keep.as_deref_mut() {
            k.ids = ids;
            k.rows = rows;
        }
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let work = self.flow.work(ops, s.steps)?;
        let mut rng = Rng::new(seed);
        let mut codes: Vec<Vec<i32>> = Vec::new();
        for f in 0..max {
            let logits = self.flow.semantic_logits(ops, &work, hid.fp())?;
            let sem = flow::semantic(&logits, self.flow.codes);
            if let Some(k) = keep.as_deref_mut() {
                k.hidden.push(hid.to_f32()?);
                k.sem.push(logits);
            }
            let forced = force.codes.map(|fc| fc.get(f));
            let sem = match forced {
                Some(Some(r)) => r[0] as usize,
                Some(None) => END,
                None => sem,
            };
            if sem == END {
                break;
            }
            let noise = match force.noise.and_then(|n| n.get(f)) {
                Some(n) => n.clone(),
                None => rng.normal(ACOUSTIC),
            };
            let lat = self.flow.acoustic(ops, nsd, &work, hid.fp(), &noise, s.cfg, None)?;
            let mut frame = vec![sem as i32];
            frame.extend(flow::quantize(&lat));
            if let Some(Some(r)) = forced {
                frame = r.clone();
            }
            if let Some(k) = keep.as_deref_mut() {
                k.lat.push(lat);
            }
            // the frame back into the backbone: the next hidden state
            self.frame_rows(&frame, c.x.fp(), &tmp)?;
            self.bb.pass(ops, nsd, &c, 1, l + f)?;
            hid.copy_within(0, &c.out, 0, h * 4)?;
            codes.push(frame);
            progress(Step { phase: "tokens", at: codes.len() as u32, of: max as u32, seconds: t0.elapsed().as_secs_f64() })?;
        }
        ops.gpu.sync()?;
        if codes.is_empty() {
            return Err(Error("the model ended the speech before its first frame".into()));
        }
        Ok(codes)
    }
}

impl AudioEngine for VoxtralTts {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: MAX_FRAMES as f32 / FPS, steps: STEPS as u32, cfg: CFG, rate: RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: self.voices.clone(),
            languages: LANGUAGES.iter().map(|x| x.0.to_string()).collect(),
            instructions: false,
            design: false,
            clone: false,
            clone_needs_text: false,
            clone_takes_text: false,
            direction: false,
            events: Vec::new(),
            requires: None,
            enrolled: true,
        })
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
        let s = self.spec(req)?;
        let seconds = req.seconds.unwrap_or(self.defaults().seconds);
        if seconds <= 0.0 {
            return Err(Error(format!("{seconds} s: a length above zero")));
        }
        let max = ((seconds * FPS) as usize).clamp(1, MAX_FRAMES);
        let codes = self.frames(&s, max, req.seed, t0, progress, &Force::default(), None)?;
        let samples = self.codec.decode(&self.ops, &self.nsd, &codes, &mut |at, of| {
            progress(Step { phase: "decode", at: at as u32, of: of as u32, seconds: t0.elapsed().as_secs_f64() })
        })?;
        Ok(Audio { rate: RATE, channels: 1, samples })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} (half, loaded {:.0} s ago)", self.ops.gpu.name, self.loaded.elapsed().as_secs_f64())]
    }
}

/// An enrolled voice's codes: a .npy [F, 37] of whole numbers (enroll.py's), each in its codebook's range
pub fn enrolled_codes(path: &Path) -> Result<Vec<Vec<i32>>> {
    let (shape, v) = check::read_npy(path)?;
    if shape.len() != 2 || shape[1] != 1 + ACOUSTIC || shape[0] == 0 || shape[0] > 1000 {
        return Err(Error(format!("{}: codes [frames, {}] expected (1 to 1,000 frames), not {shape:?}", path.display(), 1 + ACOUSTIC)));
    }
    let codes: Vec<Vec<i32>> = v.chunks(1 + ACOUSTIC).map(|f| f.iter().map(|x| *x as i32).collect()).collect();
    if codes.iter().any(|f| f[0] < SPECIAL as i32 || f[1..].iter().any(|c| *c < SPECIAL as i32 || *c >= (SPECIAL + LEVELS) as i32)) {
        return Err(Error(format!("{}: codes outside their codebooks (semantic >= 2, acoustic 2..22)", path.display())));
    }
    Ok(codes)
}

/// A directory with the GGUF (q8_0 preferred, as `models pull` lays it out or by hand): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let gguf: PathBuf = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf") && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("voxtral-4b-tts")))
        .min_by_key(|p| (!p.to_string_lossy().contains("q8_0"), p.clone())).ok_or_else(|| Error(format!("{}: no voxtral-4b-tts*.gguf", dir.display())))?;
    f.insert("model".into(), gguf);
    Ok(f)
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_pick_their_voice() {
        assert_eq!(language_voice("French"), Some("fr_female"));
        assert_eq!(language_voice("ar"), Some("ar_male"));
        assert_eq!(language_voice("chinese"), None);
        assert_eq!(LANGUAGES.len(), 9);
    }
}
