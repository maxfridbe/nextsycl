//! Breeze TTS 2 (BreezeBlue, a CSM fork) on Intel Arc (SYCL): English and Chinese speech at 24 kHz - in a voice
//! designed from a description, cloned from a recording and its exact transcript, or cloned and directed by an
//! instruction (tone, emotion, pace); vocal events inline in the text ("(laugh)", "[笑]").
//!
//! ```text
//!   the prompt (prompt.rs): text segments through the T5Gemma2 text encoder (text.rs) and its projection; a
//!   recording's frames as their summed codebook embeddings
//!   -> the backbone (lm.rs, Qwen3, 28 layers of 2,048): a frame's first code from its last hidden state (2,052 ids:
//!      the codec's 2,048, three reserved, the end), guided by a branch without the instruction (cfg)
//!   -> the depth decoder (12 layers of 1,024) over [the hidden state, code 0, ...]: the other fifteen codes
//!   -> the frame fed back (its 16 embeddings summed); at the end the 12 Hz codec's decoder (nextsycl-codec12hz)
//! ```
//!
//! The weights are CrispASR's GGUF of the checkpoint (`nextsycl models pull breeze-tts-2`: q8_0, or q4_k), expanded
//! to half at load; the tokenizer is the checkpoint's tokenizer.json; the codec is Qwen's speech tokenizer (the same
//! file as Qwen3-TTS's). `check` compares the stages with the reference's dumps (reference/breezetts2/ref.py:
//! upstream's own code on the checkpoint, float32 on the CPU).
//!
//! License: the weights and what they make are BreezeBlue's Research and Non-Commercial License 1.1; the inference
//! code upstream is Apache-2.0.

pub mod check;
mod ffi;
pub mod lm;
pub mod ops;
pub mod prompt;
pub mod sample;
pub mod text;
pub mod weights;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech, Step};
use nextsycl_codec12hz::{Codec, Encoder};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_tok::Tokenizer;

use crate::lm::{Cache, Lm};
use crate::ops::{fp, Mat, Ops};
use crate::prompt::{Segment, Spec};
use crate::sample::{Draw, Rng};
use crate::text::TextEncoder;
use crate::weights::Weights;

pub const ARCH: &str = "breeze-tts-2";
pub const ROLES: &[&str] = &["model", "tokenizer", "codec"];
pub const FPS: f32 = 12.5;
/// upstream's limits (infer.py): 1,500 frames (2 minutes), 2,048 positions with the prompt
const MAX_FRAMES: usize = 1500;
const MAX_SEQ: usize = 2048;
/// the guidance a request with an instruction gets when it names none (upstream's recommendation)
pub const CFG: f32 = 4.0;
const PENALTY: f32 = 1.1;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "greedy", env: "NS_BZ2_GREEDY", value: "0|1", help: "always the most likely code (a deterministic read; flatter, may not end)",
                   at: At::Request },
    EngineOption { name: "temperature", env: "NS_BZ2_TEMPERATURE", value: "0.9", help: "the sampling temperature (backbone and depth decoder)", at: At::Request },
    EngineOption { name: "top-k", env: "NS_BZ2_TOP_K", value: "50", help: "the sampling top-k (backbone and depth decoder)", at: At::Request },
];

/// The vocal events the model knows inline in the text (upstream's examples): English in parentheses, Chinese in
/// square brackets
pub const EVENTS: &[&str] = &["(laugh)", "(sigh)", "(cough)", "(clears throat)", "[笑]", "[叹气]", "[咳嗽]", "[清嗓子]"];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Breeze TTS 2 (speech in English and Chinese: voices designed from a description, cloned from a recording, or cloned and directed; vocal events)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(BreezeTts2::load(files, gpu, o, log)?))
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

fn given<'a>(req: &'a AudioRequest, name: &str) -> Option<&'a str> {
    req.extra.get(name).map(String::as_str)
}

pub struct BreezeTts2 {
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    pub w: Weights,
    pub te: TextEncoder,
    pub bb: Lm,
    pub dd: Lm,
    /// the codebooks' embeddings, half [16 x 2,051, 2,048] (the backbone's and the depth decoder's, tied)
    table: DevBuf,
    /// the backbone's head (2,052 ids), the depth decoder's input projection (2,048 -> 1,024) and its fifteen heads
    head0: Mat,
    proj: Mat,
    heads: Vec<Mat>,
    pub vocab: usize,
    pub groups: usize,
    /// the codec's codes a codebook (2,048: ids [codes, vocab) are reserved)
    pub codes: usize,
    pub codec: Codec,
    pub encoder: Encoder,
    loaded: Instant,
    load_s: f64,
}

/// How a run's codes are drawn
#[derive(Clone, Copy, Debug)]
pub struct Drawing {
    pub backbone: Draw,
    pub depth: Draw,
    pub penalty: f32,
}

/// What a check keeps of a run: each frame's guided backbone logits and depth logits [15][2,051]
#[derive(Default)]
pub struct Keep {
    pub prompts: Vec<Vec<f32>>,
    pub te0: Vec<f32>,
    pub logits: Vec<Vec<f32>>,
    pub depth: Vec<Vec<Vec<f32>>>,
}

impl BreezeTts2 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<BreezeTts2> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let (model, tokf, codec) = (role(files, "model")?, role(files, "tokenizer")?, role(files, "codec")?);
        let w = Weights::open(&model)?;
        // the file's matrices in half (Q8_0 / Q4_K expand to twice their F16 size; the text embedding stays on disk),
        // the codec in float32, work room: 2 GiB
        let te_embd = w.tensor("te.token_embd.weight")?.bytes;
        let expanded: u64 = w.g.tensors.iter().filter(|t| t.name != "te.token_embd.weight").map(|t| t.elements() * if t.shape.len() > 1 { 2 } else { 4 }).sum();
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let need = expanded + size(&codec) + (2u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_hf_json(&tokf).map_err(Error)?;
        if tok.id("<|AUDIO|>") != Some(262144) || tok.id("<ins_bos>") != Some(262156) {
            return Err(Error(format!("{}: not Breeze TTS 2's tokenizer (no <|AUDIO|> at 262144, <ins_bos> at 262156)", tokf.display())));
        }
        let te = TextEncoder::load(&ops, &w)?;
        log(format!("text encoder in {:.1} s (its {:.0} MB embedding read from the file)", t0.elapsed().as_secs_f64(), te_embd as f64 / 1e6));
        let bb = Lm::load(&ops, &w, "backbone.", true, w.num("bb.rms_norm_eps")? as f32, &lm_rope(w.num("bb.rope_theta")?))?;
        let dd_inv = match w.g.meta("breeze.dd.rope_scaling_type").and_then(|v| v.as_str()) {
            Some("llama3") => lm::llama3_inv_freq(w.num("dd.rope_theta")?, lm::HEAD, w.num("dd.rope_scaling_factor")?, w.num("dd.rope_low_freq_factor")?,
                                                  w.num("dd.rope_high_freq_factor")?, w.num("dd.rope_orig_max_pos")?),
            _ => lm_rope(w.num("dd.rope_theta")?),
        };
        let dd = Lm::load(&ops, &w, "depth.", false, w.num("dd.rms_norm_eps")? as f32, &dd_inv)?;
        log(format!("backbone {} layers of {}, depth decoder {} layers of {} in {:.1} s", bb.layers(), bb.hidden, dd.layers(), dd.hidden, t0.elapsed().as_secs_f64()));
        let groups = w.num("num_codebooks")? as usize;
        let vocab = w.num("audio_vocab_size")? as usize;
        let codes = w.num("codec_codebook_size")? as usize;
        let table = w.mat(&ops, &["backbone.audio_embd.weight"])?;
        if table.n != groups * vocab || table.k != bb.hidden {
            return Err(Error(format!("backbone.audio_embd: [{}, {}], [{}, {}] expected", table.n, table.k, groups * vocab, bb.hidden)));
        }
        let heads = (0..groups - 1).map(|i| w.mat(&ops, &[&format!("depth.cb_head.{i}.weight")])).collect::<Result<Vec<_>>>()?;
        let head0 = w.mat(&ops, &["backbone.codebook0_head.weight"])?;
        let proj = w.mat(&ops, &["depth.projection.weight"])?;
        if te.out_width() != bb.hidden || proj.n != dd.hidden || head0.n != vocab + 1 {
            return Err(Error("the text projection, depth projection or backbone head does not fit the decoders".into()));
        }
        let cw = nextsycl_codec12hz::Weights::open(&codec)?;
        let codec = Codec::load(gpu, &cw)?;
        let encoder = Encoder::load(gpu, &cw)?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let after = gpu.memory()?.1.unwrap_or(free);
        log(format!("{ARCH} on {} in {load_s:.0} s ({:.1} GiB of weights on the card)", gpu.name, free.saturating_sub(after) as f64 / (1u64 << 30) as f64));
        Ok(BreezeTts2 { ops, nsd, tok, w, te, bb, dd, table: table.w, head0, proj, heads, vocab, groups, codes, codec, encoder, loaded: Instant::now(), load_s })
    }

    /// What the request asks for, checked: the text, the instruction, the recording (encoded) and its transcript
    pub fn spec(&self, req: &AudioRequest) -> Result<Spec> {
        if req.prompt.trim().is_empty() {
            return Err(Error("the text to say is empty".into()));
        }
        if let Some(v) = req.voice.as_deref().filter(|v| !v.is_empty()) {
            return Err(Error(format!("voice {v}: {ARCH} has no built-in voices (describe one with instructions, or clone a recording)")));
        }
        let instruction = req.instructions.as_deref().map(str::trim).filter(|i| !i.is_empty()).map(str::to_string);
        let reference = match &req.reference {
            Some(r) => {
                let text = r.text.as_deref().map(str::trim).filter(|t| !t.is_empty())
                    .ok_or_else(|| Error("ref_text: the recording's exact transcript (Breeze TTS 2 clones from a recording and what it says)".into()))?;
                let wav = nextsycl_codec12hz::resample(&r.samples, r.rate);
                Some((text.to_string(), self.encoder.encode(&self.nsd, &wav)?))
            }
            None => None,
        };
        let cfg = if instruction.is_some() { req.cfg.unwrap_or(CFG) } else { 1.0 };
        if !(cfg.is_finite() && cfg > 0.0) {
            return Err(Error(format!("cfg {cfg}: above zero")));
        }
        Ok(Spec { text: req.prompt.clone(), instruction, reference, cfg })
    }

    pub fn drawing(&self, req: &AudioRequest) -> Result<Drawing> {
        let mut d = if given(req, "greedy").is_some_and(|v| v == "1") { Draw::GREEDY } else { Draw::DEFAULT };
        if let Some(v) = given(req, "temperature") {
            d.temperature = v.parse().map_err(|_| Error(format!("temperature: {v}?")))?;
        }
        if let Some(v) = given(req, "top-k") {
            d.top_k = v.parse().map_err(|_| Error(format!("top-k: {v}?")))?;
        }
        Ok(Drawing { backbone: d, depth: d, penalty: PENALTY })
    }

    /// The sum of the codebook rows `idx` (of the combined table) into `out` [hidden] (the kernel sums 8 at a time)
    fn sum_rows(&self, idx: &[i32], out: *mut f32, tmp: &DevBuf) -> Result<()> {
        let h = self.bb.hidden;
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

    /// A frame's rows of the table: codebook g's code c at c + g * vocab
    fn frame_idx(&self, codes: &[i32]) -> Vec<i32> {
        codes.iter().enumerate().map(|(g, c)| (*c).clamp(0, self.vocab as i32 - 1) + (g * self.vocab) as i32).collect()
    }

    /// A prompt's rows [L, hidden] (host); `te0`: the text encoder's output of its first text segment
    pub fn prompt_rows(&self, segs: &[Segment], mut te0: Option<&mut Vec<f32>>) -> Result<Vec<f32>> {
        let h = self.bb.hidden;
        let mut rows = Vec::new();
        let one = DevBuf::f32(&self.ops.gpu, h)?;
        let tmp = DevBuf::f32(&self.ops.gpu, h)?;
        for s in segs {
            match s {
                Segment::Text(t) => {
                    let ids = prompt::text_ids(&self.tok, t)?;
                    rows.extend(self.te.encode(&self.ops, &self.nsd, &self.w, &ids, te0.take())?);
                }
                Segment::Audio(frames) => {
                    for f in frames.iter().map(Vec::as_slice).chain(std::iter::once(&vec![0; self.groups][..])) {
                        if f.len() != self.groups {
                            return Err(Error(format!("a recording's frame of {} codes, {} expected", f.len(), self.groups)));
                        }
                        self.sum_rows(&self.frame_idx(f), one.fp(), &tmp)?;
                        self.ops.gpu.sync()?;
                        rows.extend(one.to_f32()?);
                    }
                }
            }
        }
        Ok(rows)
    }

    /// The frames' codes [T][16]. `force`: a check's codes to follow (each frame's taken from it, not drawn);
    /// `keep`: what a check compares
    #[allow(clippy::too_many_arguments)]
    pub fn frames(&self, s: &Spec, d: &Drawing, max: usize, seed: u64, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>,
                  force: Option<&[Vec<i32>]>, mut keep: Option<&mut Keep>) -> Result<Vec<Vec<i32>>> {
        let (ops, nsd, h) = (&self.ops, &self.nsd, self.bb.hidden);
        let branches = s.branches();
        let nb = branches.len();
        // each branch's prompt through the backbone: its cache and last hidden state
        let mut caches: Vec<(Cache, usize)> = Vec::new();
        let lasts: Vec<DevBuf> = (0..nb).map(|_| DevBuf::f32(&ops.gpu, h)).collect::<Result<_>>()?;
        let mut longest = 0;
        for (bi, segs) in branches.iter().enumerate() {
            let te0 = keep.as_deref_mut().filter(|_| bi == 0).map(|k| &mut k.te0);
            let rows = self.prompt_rows(segs, te0)?;
            let l = rows.len() / h;
            longest = longest.max(l);
            if l + 1 >= MAX_SEQ {
                return Err(Error(format!("the prompt is {l} positions; the model reads {MAX_SEQ} with the speech")));
            }
            let c = self.bb.cache(ops, 1, l, (l + max + 2).min(MAX_SEQ))?;
            c.x.write(0, &crate::bytes(&rows))?;
            self.bb.pass(ops, nsd, &c, l, 0)?;
            lasts[bi].copy_within(0, &c.out, (l - 1) * h * 4, h * 4)?;
            if let Some(k) = keep.as_deref_mut() {
                k.prompts.push(rows);
            }
            caches.push((c, l));
        }
        let max = max.min(MAX_SEQ - 1 - longest);
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let dc = self.dd.cache(ops, nb, 2 * nb, self.groups)?;
        let logits = DevBuf::f32(&ops.gpu, nb * (self.vocab + 1))?;
        let e = DevBuf::f32(&ops.gpu, nb * h)?;
        let tmp = DevBuf::f32(&ops.gpu, h)?;
        let dh = self.dd.hidden;
        let mut rng = Rng::new(seed);
        let mut codes: Vec<Vec<i32>> = Vec::new();
        let mut seen: Vec<i32> = Vec::new();
        let eos = self.vocab as i32;
        let reserved = (self.codes, self.vocab);
        let read = |n: usize, w: usize| -> Result<Vec<Vec<f32>>> {
            ops.gpu.sync()?;
            let all = logits.to_f32()?;
            Ok((0..n).map(|b| all[b * w..(b + 1) * w].to_vec()).collect())
        };
        for f in 0..max {
            // the first code: the backbone's head on each branch's last hidden state, guided
            for (bi, last) in lasts.iter().enumerate() {
                ops.gemv(last.fp(), 1, &self.head0, fp(&logits, bi * (self.vocab + 1)), self.head0.n, false)?;
            }
            let ls = read(nb, self.vocab + 1)?;
            let mut l0 = sample::guided(&ls[0], ls.get(1).map(Vec::as_slice), s.cfg);
            if let Some(k) = keep.as_deref_mut() {
                k.logits.push(l0.clone());
            }
            sample::repetition_penalty(&mut l0, &seen, d.penalty);
            sample::suppress(&mut l0, reserved.0, reserved.1);
            let c0 = match force {
                Some(fc) => fc.get(f).map_or(eos, |r| r[0]),
                None => sample::pick(&l0, d.backbone, &mut rng) as i32,
            };
            if c0 == eos {
                break;
            }
            // the depth decoder: [projection(hidden), projection(E(c0))] for each branch at positions 0, 1
            ops.embed(&self.table, h, &[c0], e.fp(), h, 1)?;
            for (bi, last) in lasts.iter().enumerate() {
                ops.gemv(last.fp(), 1, &self.proj, fp(&dc.x, 2 * bi * dh), dh, false)?;
                ops.gemv(e.fp(), 1, &self.proj, fp(&dc.x, (2 * bi + 1) * dh), dh, false)?;
            }
            self.dd.pass(ops, nsd, &dc, 2, 0)?;
            let mut frame = vec![c0];
            let mut dl = Vec::new();
            for g in 1..self.groups {
                let m = &self.heads[g - 1];
                for bi in 0..nb {
                    // the first step's rows are 2 a branch (its second the one read); later ones 1
                    let row = if g == 1 { 2 * bi + 1 } else { bi };
                    ops.gemv(fp(&dc.out, row * dh), 1, m, fp(&logits, bi * self.vocab), m.n, false)?;
                }
                let ls = read(nb, self.vocab)?;
                let mut lg = sample::guided(&ls[0], ls.get(1).map(Vec::as_slice), s.cfg);
                sample::suppress(&mut lg, reserved.0, reserved.1);
                let code = match force {
                    Some(fc) => fc[f][g],
                    None => sample::pick(&lg, d.depth, &mut rng) as i32,
                };
                if keep.is_some() {
                    dl.push(lg);
                }
                frame.push(code);
                if g + 1 < self.groups {
                    ops.embed(&self.table, h, &[code + (g * self.vocab) as i32], e.fp(), h, nb)?;
                    ops.gemv(e.fp(), nb, &self.proj, dc.x.fp(), dh, false)?;
                    self.dd.pass(ops, nsd, &dc, 1, g + 1)?;
                }
            }
            if let Some(k) = keep.as_deref_mut() {
                k.depth.push(dl);
            }
            seen.push(c0);
            // the frame fed back to each branch
            let idx = self.frame_idx(&frame);
            for ((c, l), last) in caches.iter().zip(&lasts) {
                self.sum_rows(&idx, c.x.fp(), &tmp)?;
                self.bb.pass(ops, nsd, c, 1, l + f)?;
                last.copy_within(0, &c.out, 0, h * 4)?;
            }
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

/// Plain RoPE frequencies for heads of 128
fn lm_rope(theta: f64) -> Vec<f32> {
    (0..lm::HEAD / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / lm::HEAD as f64)) as f32).collect()
}

impl AudioEngine for BreezeTts2 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: MAX_FRAMES as f32 / FPS, steps: 0, cfg: CFG, rate: nextsycl_codec12hz::RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: Vec::new(),
            languages: vec!["chinese".into(), "english".into()],
            instructions: true,
            design: true,
            clone: true,
            clone_needs_text: true,
            clone_takes_text: true,
            direction: true,
            events: EVENTS.iter().map(|e| e.to_string()).collect(),
            requires: None,
            speakers: 1,
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
        let codes = self.frames(&s, &d, max, req.seed, t0, progress, None, None)?;
        progress(Step { phase: "decode", at: 0, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let samples = self.codec.decode(&self.nsd, &codes)?;
        progress(Step { phase: "decode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        Ok(Audio { rate: nextsycl_codec12hz::RATE, channels: 1, samples })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} (half, loaded {:.0} s ago)", self.ops.gpu.name, self.loaded.elapsed().as_secs_f64())]
    }
}

/// A directory with the GGUF (q8_0 preferred), tokenizer.json and the codec's model.safetensors (as `models pull` lays them out, or by
/// hand): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let gguf = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf") && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("breeze-tts-2")))
        // the q8_0 file when there are several (the card's), else the first by name
        .min_by_key(|p| (!p.to_string_lossy().contains("q8_0"), p.clone())).ok_or_else(|| Error(format!("{}: no breeze-tts-2*.gguf", dir.display())))?;
    f.insert("model".into(), gguf);
    for (role, name) in [("tokenizer", "tokenizer.json"), ("codec", "audio_tokenizer/model.safetensors")] {
        let p = dir.join(name);
        if !p.exists() {
            return Err(Error(format!("{}: no {name}", dir.display())));
        }
        f.insert(role.into(), p);
    }
    Ok(f)
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
