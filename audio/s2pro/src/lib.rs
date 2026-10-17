//! Fish Audio S2 Pro on Intel Arc (SYCL): speech in about 80 languages at 44.1 kHz - a random voice, or one cloned from
//! a recording and its transcript; steered by free-form inline tags in the text (`[whisper]`, `[laughing]`,
//! `[excited tone]`, `[pause]` ...); several speakers in one text (`<|speaker:N|>`).
//!
//! ```text
//!   the prompt (prompt.rs): fish-speech's conversation - a system turn (with a recording: its transcript and its
//!   frames), the text as the user turn, "<|im_start|>assistant\n<|voice|>"; a row's embedding is its token's, and a
//!   frame's the sum of its token's and its ten codes' rows, / sqrt(11)
//!   -> the slow model (lm.rs: Qwen3, 36 layers of 2,560): a frame's first code from its normed last hidden state, the
//!      tied embedding's 4,096 semantic rows and <|im_end|> (the end)
//!   -> the fast model (4 layers of 2,560) over [the hidden state, code 0, ..., code 8]: the other nine codes
//!   -> the frame fed back to the slow model; at the end Fish's codec (codec.rs) to 44.1 kHz
//! ```
//!
//! The weights are s2.cpp's GGUF of the checkpoint (`nextsycl models pull s2-pro`: Q8_0, kept as Q8_0 on the card; the
//! F16 file for checks), with the codec in the same file; the tokenizer is the checkpoint's tokenizer.json. `check`
//! compares the stages with the reference's dumps (reference/s2pro/ref.py: fish-speech's own code on the checkpoint,
//! float32 on the CPU).
//!
//! License: the weights and what they make are under the Fish Audio Research License (research and non-commercial use
//! only; commercial use needs Fish Audio's written license).

pub mod check;
pub mod codec;
mod ffi;
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
use crate::lm::{Cache, Lm};
use crate::ops::{fp, rope_table, Mat, Ops};
use crate::prompt::{Ids, Row, Spec};
use crate::sample::{Draw, Rng};
use crate::weights::Weights;

pub const ARCH: &str = "s2-pro";
/// the GGUF's general.architecture (s2.cpp's converter)
pub const ARCH_GGUF: &str = "fish-speech";
pub const ROLES: &[&str] = &["model", "tokenizer"];
/// frames a second: 44,100 / 2,048
pub const FPS: f32 = codec::RATE as f32 / codec::HOP as f32;
/// the longest clip (120 s), and the slow model's context
const MAX_FRAMES: usize = 2584;
const MAX_SEQ: usize = 32768;
/// the longest recording to clone (its encoder runs over it whole)
const MAX_REFERENCE_S: f32 = 30.0;
/// rows a prompt pass (the prompt goes in chunks)
const PREFILL: usize = 256;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "greedy", env: "NS_S2P_GREEDY", value: "0|1", help: "always the most likely code (a deterministic read; flatter, may not end)",
                   at: At::Request },
    EngineOption { name: "temperature", env: "NS_S2P_TEMPERATURE", value: "0.7", help: "the sampling temperature (slow and fast models)", at: At::Request },
    EngineOption { name: "top-p", env: "NS_S2P_TOP_P", value: "0.7", help: "the sampling top-p (slow and fast models)", at: At::Request },
    EngineOption { name: "top-k", env: "NS_S2P_TOP_K", value: "30", help: "the sampling top-k (slow and fast models)", at: At::Request },
];

/// Inline tags the model card lists as common (S2 Pro reads free-form ones too: any short description in brackets)
pub const EVENTS: &[&str] = &[
    "[pause]", "[short pause]", "[emphasis]", "[laughing]", "[chuckle]", "[sigh]", "[inhale]", "[exhale]", "[clearing throat]", "[tsk]", "[whisper]",
    "[low voice]", "[shouting]", "[screaming]", "[excited]", "[excited tone]", "[sad]", "[angry]", "[surprised]", "[shocked]", "[delight]", "[panting]",
    "[singing]", "[volume up]", "[volume down]", "[low volume]", "[echo]", "[with strong accent]", "[audience laughter]", "[interrupting]",
];

/// The languages its card names first (it reads about 80, from the script)
const LANGUAGES: &[&str] = &["japanese", "english", "chinese", "korean", "spanish", "portuguese", "arabic", "russian", "french", "german"];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Fish Audio S2 Pro (speech in ~80 languages, 44.1 kHz: voices cloned from a recording, free-form inline tags, several speakers)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(S2Pro::load(files, gpu, o, log)?))
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

fn given<'a>(req: &'a AudioRequest, name: &str) -> Option<&'a str> {
    req.extra.get(name).map(String::as_str)
}

pub struct S2Pro {
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    pub ids: Ids,
    pub w: Weights,
    pub slow: Lm,
    pub fast: Lm,
    /// the slow model's head: the tied embedding's semantic rows (4,096), then <|im_end|>'s
    head: Mat,
    fast_out: Mat,
    /// the fast model's code embeddings, half [4,096, 2,560]
    fast_emb: DevBuf,
    /// the rotary table (θ 1e6, heads of 128) of the whole context
    cs: DevBuf,
    /// codes a frame (10) and a codebook's (4,096)
    pub groups: usize,
    pub codes: usize,
    pub codec: Codec,
    loaded: Instant,
    load_s: f64,
}

/// What a check keeps of a run: the prompt's rows and input, each frame's slow logits [4,097] and fast logits [9][4,096]
#[derive(Default)]
pub struct Keep {
    pub rows: Vec<Row>,
    pub prompt: Vec<f32>,
    pub logits: Vec<Vec<f32>>,
    pub fast: Vec<Vec<Vec<f32>>>,
}

impl S2Pro {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<S2Pro> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let (model, tokf) = (role(files, "model")?, role(files, "tokenizer")?);
        let w = Weights::open(&model)?;
        // the file's matrices as stored (the embeddings stay on disk; the codec in float32: twice its F16), work room 2 GiB
        let need: u64 = w.g.tensors.iter().filter(|t| t.name != "embeddings.weight" && t.name != "codebook_embeddings.weight")
            .map(|t| if t.name.starts_with("c.") || t.shape.len() == 1 { t.elements() * 4 } else { t.bytes }).sum::<u64>() + (2u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_hf_json(&tokf).map_err(Error)?;
        let ids = Ids::of(&tok)?;
        let sb = w.num("semantic_begin_id")? as u32;
        if ids.semantic0 != sb {
            return Err(Error(format!("{}: <|semantic:0|> is {}, the model's semantic ids begin at {sb}: not this model's tokenizer", tokf.display(), ids.semantic0)));
        }
        let groups = w.num("num_codebooks")? as usize;
        let codes = w.num("codebook_size")? as usize;
        let eps = w.num("attention.layer_norm_rms_epsilon")? as f32;
        let slow = Lm::load(&ops, &w, "layers.", "norm", w.flag("attention_qk_norm"), eps)?;
        log(format!("slow model: {} layers of {} in {:.1} s", slow.layers(), slow.hidden, t0.elapsed().as_secs_f64()));
        let fast = Lm::load(&ops, &w, "fast_layers.", "fast_norm", w.flag("fast_attention_qk_norm"), w.num("fast_layer_norm_rms_eps")? as f32)?;
        if w.flag("fast_project_in") || fast.hidden != slow.hidden {
            return Err(Error("the fast model's projection in (a fast model narrower than the slow one) is not one this engine runs".into()));
        }
        let im_end = ids.im_end as usize;
        let head = w.mat_rows(&ops, &[("embeddings.weight", Some(sb as usize..sb as usize + codes)), ("embeddings.weight", Some(im_end..im_end + 1))])?;
        let fast_out = w.mat(&ops, &["fast_output.weight"])?;
        let fe = w.f32("fast_embeddings.weight")?;
        let fast_emb = DevBuf::new(gpu, fe.len() * 2)?;
        {
            let stage = DevBuf::from_f32(gpu, &fe)?;
            ops.to_half(stage.fp(), fast_emb.ptr(), fe.len())?;
            gpu.sync()?;
        }
        if fast_out.n != codes || fe.len() != codes * fast.hidden || head.k != slow.hidden {
            return Err(Error("the fast model's head or embeddings, or the slow head, do not fit the models".into()));
        }
        let theta = w.num("rope.freq_base")?;
        if (w.num("fast_rope_freq_base")? - theta).abs() > 1e-3 {
            return Err(Error("the fast model's RoPE base differs from the slow model's: not a checkpoint this engine runs".into()));
        }
        let cs = DevBuf::from_f32(gpu, &rope_table(MAX_SEQ, lm::HEAD, theta))?;
        let codec = Codec::load(gpu, &w)?;
        if codec.sizes()[0] != codes || codec.sizes().len() != groups {
            return Err(Error(format!("the codec's codebooks {:?} do not fit the model's {groups} of {codes}", codec.sizes())));
        }
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let after = gpu.memory()?.1.unwrap_or(free);
        log(format!("{ARCH} on {} in {load_s:.0} s ({:.1} GiB on the card)", gpu.name, free.saturating_sub(after) as f64 / (1u64 << 30) as f64));
        Ok(S2Pro { ops, nsd, tok, ids, w, slow, fast, head, fast_out, fast_emb, cs, groups, codes, codec, loaded: Instant::now(), load_s })
    }

    /// The model's semantic id of code 0
    pub fn semantic0(&self) -> u32 {
        self.ids.semantic0
    }

    /// What the request asks for, checked: the text, the instruction, the recording (encoded) and its transcript
    pub fn spec(&self, req: &AudioRequest) -> Result<Spec> {
        if req.prompt.trim().is_empty() {
            return Err(Error("the text to say is empty".into()));
        }
        if let Some(v) = req.voice.as_deref().filter(|v| !v.is_empty()) {
            return Err(Error(format!("voice {v}: {ARCH} has no built-in voices (clone a recording with its transcript, or leave voice out)")));
        }
        let instruction = req.instructions.as_deref().map(str::trim).filter(|i| !i.is_empty()).map(str::to_string);
        let reference = match &req.reference {
            Some(r) => {
                let text = r.text.as_deref().map(str::trim).filter(|t| !t.is_empty())
                    .ok_or_else(|| Error("ref_text: the recording's exact transcript (S2 Pro clones from a recording and what it says)".into()))?;
                let secs = r.samples.len() as f32 / r.rate.max(1) as f32;
                if secs > MAX_REFERENCE_S {
                    return Err(Error(format!("the recording is {secs:.1} s: up to {MAX_REFERENCE_S:.0} s (10-30 s of one speaker clones best)")));
                }
                let wav = resample(&r.samples, r.rate, codec::RATE);
                Some((text.to_string(), self.codec.encode(&self.nsd, &wav)?))
            }
            None => None,
        };
        Ok(Spec { text: req.prompt.clone(), instruction, reference })
    }

    pub fn drawing(&self, req: &AudioRequest) -> Result<Draw> {
        let mut d = if given(req, "greedy").is_some_and(|v| v == "1") { Draw::GREEDY } else { Draw::DEFAULT };
        let num = |k: &str| -> Result<Option<f32>> { given(req, k).map(|v| v.parse().map_err(|_| Error(format!("{k}: {v}?")))).transpose() };
        if let Some(v) = num("temperature")? {
            if !(v > 0.0 && v < 2.0) {
                return Err(Error(format!("temperature {v}: in (0, 2)")));
            }
            d.temperature = v;
        }
        if let Some(v) = num("top-p")? {
            if !(v > 0.0 && v <= 1.0) {
                return Err(Error(format!("top-p {v}: in (0, 1]")));
            }
            d.top_p = v;
        }
        if let Some(v) = given(req, "top-k") {
            d.top_k = v.parse().map_err(|_| Error(format!("top-k: {v}?")))?;
        }
        Ok(d)
    }

    /// Rows' input embeddings [n, 2,560] (host): a token's row of the embedding; a frame's adds its ten codes' rows of
    /// the codebook embeddings (codebook i's code c: row c + 4,096 i) and divides by sqrt(11)
    pub fn embed_rows(&self, rows: &[Row]) -> Result<Vec<f32>> {
        let h = self.slow.hidden;
        let mut out = Vec::with_capacity(rows.len() * h);
        let scale = 1.0 / ((self.groups + 1) as f32).sqrt();
        for r in rows {
            let mut x = self.w.rows("embeddings.weight", r.token as usize, r.token as usize + 1)?;
            if let Some(codes) = &r.codes {
                if codes.len() != self.groups {
                    return Err(Error(format!("a frame of {} codes, {} expected", codes.len(), self.groups)));
                }
                for (i, c) in codes.iter().enumerate() {
                    let row = (*c).clamp(0, self.codes as i32 - 1) as usize + i * self.codes;
                    let e = self.w.rows("codebook_embeddings.weight", row, row + 1)?;
                    x.iter_mut().zip(&e).for_each(|(a, b)| *a += b);
                }
                x.iter_mut().for_each(|a| *a *= scale);
            }
            out.extend(x);
        }
        Ok(out)
    }

    /// Rows through the slow model from position `p0`, in chunks; the last one's normed hidden state into `last`
    fn prefill(&self, c: &Cache, x: &[f32], p0: usize, last: &DevBuf) -> Result<()> {
        let h = self.slow.hidden;
        let n = x.len() / h;
        let mut s = 0;
        while s < n {
            let e = (s + PREFILL).min(n);
            c.x.write(0, &bytes(&x[s * h..e * h]))?;
            self.slow.pass(&self.ops, &self.nsd, c, e - s, p0 + s, &self.cs)?;
            if e == n {
                last.copy_within(0, &c.out, (e - s - 1) * h * 4, h * 4)?;
            }
            s = e;
        }
        Ok(())
    }

    /// The frames' codes [T][10]. `force`: a check's codes to follow (each frame's taken from it, not drawn); `keep`:
    /// what a check compares
    #[allow(clippy::too_many_arguments)]
    pub fn frames(&self, s: &Spec, d: Draw, max: usize, seed: u64, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>,
                  force: Option<&[Vec<i32>]>, mut keep: Option<&mut Keep>) -> Result<Vec<Vec<i32>>> {
        let (ops, nsd, g, h) = (&self.ops, &self.nsd, &self.ops.gpu, self.slow.hidden);
        let system = prompt::system(&self.tok, &self.ids, s.reference.as_ref());
        let users: Vec<Vec<Row>> = s.batches().iter().map(|b| prompt::user(&self.tok, &self.ids, b)).collect();
        let positions = system.len() + users.iter().map(|u| u.len() + 2).sum::<usize>() + max + 1;
        if positions > MAX_SEQ {
            return Err(Error(format!("the prompt and {max} frames are {positions} positions; the model reads {MAX_SEQ}")));
        }
        let c = self.slow.cache(ops, 1, PREFILL, positions)?;
        let fc = self.fast.cache(ops, 1, 2, self.groups)?;
        let last = DevBuf::f32(g, h)?;
        let n0 = self.head.n;
        let logits = DevBuf::f32(g, n0.max(self.codes))?;
        let read = |n: usize| -> Result<Vec<f32>> {
            g.sync()?;
            let mut v = vec![0u8; n * 4];
            logits.read(0, &mut v)?;
            Ok(v.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect())
        };
        let mut rng = Rng::new(seed);
        let mut out: Vec<Vec<i32>> = Vec::new();
        let (mut pos, mut open) = (0, false);
        let end = self.codes;
        for (bi, user) in users.iter().enumerate() {
            if out.len() >= max {
                break;
            }
            let start = pos;
            let mut rows = if bi == 0 { system.clone() } else { Vec::new() };
            if open {
                rows.extend(prompt::end(&self.ids));
            }
            rows.extend(user.iter().cloned());
            let x = self.embed_rows(&rows)?;
            if let Some(k) = keep.as_deref_mut() {
                k.rows.extend(rows.iter().cloned());
                k.prompt.extend_from_slice(&x);
            }
            self.prefill(&c, &x, pos, &last)?;
            pos += rows.len();
            progress(Step { phase: "prompt", at: bi as u32 + 1, of: users.len() as u32, seconds: t0.elapsed().as_secs_f64() })?;
            let mut window: Vec<usize> = Vec::new();
            let mut made = 0;
            while out.len() < max {
                // the first code: the head over the slow model's last hidden state
                self.head.gemv(ops, last.fp(), 1, logits.fp(), n0, false)?;
                let l0 = read(n0)?;
                if let Some(k) = keep.as_deref_mut() {
                    k.logits.push(l0.clone());
                }
                let c0 = match force {
                    Some(r) => r.get(out.len()).map_or(end, |f| f[0] as usize),
                    None => sample::pick_ras(&l0, d, &window, |i| i < end, &mut rng),
                };
                if c0 >= end {
                    break;
                }
                if made > 0 {
                    window.push(c0);
                    if window.len() > sample::WINDOW {
                        window.remove(0);
                    }
                }
                // the fast model: [hidden, E(c0)] at positions 0, 1; then a code a position
                fc.x.copy_within(0, &last, 0, h * 4)?;
                ops.embed(&self.fast_emb, h, &[c0 as i32], fp(&fc.x, h), h, 1)?;
                self.fast.pass(ops, nsd, &fc, 2, 0, &self.cs)?;
                let mut frame = vec![c0 as i32];
                let mut fl = Vec::new();
                for cb in 1..self.groups {
                    let row = if cb == 1 { 1 } else { 0 };
                    self.fast_out.gemv(ops, fp(&fc.out, row * h), 1, logits.fp(), self.codes, false)?;
                    let l = read(self.codes)?;
                    let code = match force {
                        Some(r) => r[out.len()][cb],
                        None => sample::pick(&l, d, &mut rng) as i32,
                    };
                    if keep.is_some() {
                        fl.push(l);
                    }
                    frame.push(code);
                    if cb + 1 < self.groups {
                        ops.embed(&self.fast_emb, h, &[code.clamp(0, self.codes as i32 - 1)], fc.x.fp(), h, 1)?;
                        self.fast.pass(ops, nsd, &fc, 1, cb + 1, &self.cs)?;
                    }
                }
                if let Some(k) = keep.as_deref_mut() {
                    k.fast.push(fl);
                }
                // the frame fed back
                let x = self.embed_rows(&[Row { token: self.ids.semantic0 + c0 as u32, codes: Some(frame.clone()) }])?;
                c.x.write(0, &bytes(&x))?;
                self.slow.pass(ops, nsd, &c, 1, pos, &self.cs)?;
                last.copy_within(0, &c.out, 0, h * 4)?;
                pos += 1;
                made += 1;
                out.push(frame);
                progress(Step { phase: "tokens", at: out.len() as u32, of: max as u32, seconds: t0.elapsed().as_secs_f64() })?;
            }
            if made == 0 {
                // upstream drops a turn that made no audio: back to before it (the system turn stays)
                pos = start + if bi == 0 { system.len() } else { 0 };
            } else {
                open = true;
            }
        }
        g.sync()?;
        if out.is_empty() {
            return Err(Error("the model ended the speech before its first frame".into()));
        }
        Ok(out)
    }
}

/// Mono samples at `rate` to `to` (a windowed-sinc resampler: Hann, 32 zero crossings, cut at the lower Nyquist)
pub fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() || rate == 0 {
        return x.to_vec();
    }
    let ratio = to as f64 / rate as f64;
    let cut = ratio.min(1.0);
    let zc = 32.0;
    let half = (zc / cut).ceil() as isize;
    let n = (x.len() as f64 * ratio).round() as usize;
    (0..n).map(|i| {
        let center = i as f64 / ratio;
        let c0 = center.floor() as isize;
        let mut s = 0.0;
        for j in c0 - half..=c0 + half {
            if j < 0 || j as usize >= x.len() {
                continue;
            }
            let d = (j as f64 - center) * cut;
            if d.abs() >= zc {
                continue;
            }
            let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
            let win = 0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos();
            s += x[j as usize] as f64 * sinc * win * cut;
        }
        s as f32
    }).collect()
}

impl AudioEngine for S2Pro {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: MAX_FRAMES as f32 / FPS, steps: 0, cfg: 1.0, rate: codec::RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: Vec::new(),
            languages: LANGUAGES.iter().map(|l| l.to_string()).collect(),
            instructions: true,
            design: false,
            clone: true,
            clone_needs_text: true,
            clone_takes_text: true,
            direction: true,
            events: EVENTS.iter().map(|e| e.to_string()).collect(),
            requires: None,
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
        let d = self.drawing(req)?;
        let seconds = req.seconds.unwrap_or(self.defaults().seconds);
        if seconds <= 0.0 {
            return Err(Error(format!("{seconds} s: a length above zero")));
        }
        let max = ((seconds * FPS) as usize).clamp(1, MAX_FRAMES);
        let codes = self.frames(&s, d, max, req.seed, t0, progress, None, None)?;
        progress(Step { phase: "decode", at: 0, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let samples = self.codec.decode(&self.nsd, &codes)?;
        progress(Step { phase: "decode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        Ok(Audio { rate: codec::RATE, channels: 1, samples })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} (Q8_0 / half, loaded {:.0} s ago)", self.ops.gpu.name, self.loaded.elapsed().as_secs_f64())]
    }
}

/// A directory with the GGUF (q8_0 preferred) and tokenizer.json (as `models pull` lays them out, or by hand): its
/// files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let gguf = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf") && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("s2-pro")))
        // the q8_0 file when there are several (the card's), else the first by name
        .min_by_key(|p| (!p.to_string_lossy().contains("q8_0"), p.clone())).ok_or_else(|| Error(format!("{}: no s2-pro*.gguf", dir.display())))?;
    f.insert("model".into(), gguf);
    let t = dir.join("tokenizer.json");
    if !t.exists() {
        return Err(Error(format!("{}: no tokenizer.json", dir.display())));
    }
    f.insert("tokenizer".into(), t);
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
    fn a_frame_is_2048_samples_of_44_1_khz() {
        assert!((FPS - 21.533).abs() < 1e-3);
        assert!((MAX_FRAMES as f32 / FPS - 120.0).abs() < 0.05);
        let x: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.05).sin()).collect();
        assert_eq!(resample(&x, 24000, 44100).len(), 44100);
        assert_eq!(resample(&x, 44100, 44100), x);
    }
}
