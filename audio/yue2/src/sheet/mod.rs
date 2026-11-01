//! SheetSage2 (M-A-P's music transcription model, non-commercial: CC BY-NC 4.0) on SYCL: a recording to its lead
//! sheet as ABC - the score a YuE2 cover sings over (cot melody: the melody alone; cot full: melody and chords).
//!
//! ```text
//!   the recording -> wave.rs: FFmpeg's mono 24 kHz (sqrt(1/2) (L + R), libswresample's Kaiser sinc)
//!   windows of 300 s (stitch.rs: every 100 s, 200 s of overlap) -> each:
//!     mel.rs: log-mel [30000, 128] -> mert.rs: MERT-v2 (ConvNeXt to 25 Hz, 24 conformer blocks), the softmax layer mix,
//!     the projection -> memory [7500, 512]
//!     decoder.rs: BART's decoder, greedy under tokens.rs' grammar, from sos + the prompts + out (+ the overlap's events)
//!   the events stitched -> notation.rs: beats, measures, the two melody voices (Vocal, Ins), keys, chords -> ABC
//! ```
//!
//! The model file is scragnog's sheetsage2-f16.gguf (HOT-Step's conversion of the Comfy-Org merge: the encoder's
//! attention adapters already folded in, every matrix half from the bf16 source - exact - the rest float32). The
//! reference is upstream's code as it is (m-a-p/SheetSage2 rev b83c3d79, reference/yue2/sheetsage_ref.py), run with its
//! defaults: the full prompts, bf16 autocast, the default preset.

pub mod check;
pub mod decoder;
pub mod mel;
pub mod mert;
pub mod notation;
pub mod spell;
pub mod stitch;
pub mod tokens;
pub mod wave;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{Error, Result, Step};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_gguf::Gguf;

use crate::lm::Scratch;
use decoder::{Decoder, CONTEXT};
use mert::Mert;
use tokens::{Event, Grammar, EOS, OUT};

/// the window, its overlap and look-ahead (the default preset)
pub const WINDOW: f64 = 300.0;
pub const OVERLAP: f64 = 200.0;
pub const LOOKAHEAD: f64 = 100.0;

/// A transcription: the ABC (with chords and without), what the model wrote, and the times
#[derive(Clone, Debug, Default)]
pub struct Transcription {
    pub full: notation::Sheet,
    pub melody: notation::Sheet,
    /// each window's tokens (prefix, the events, eos)
    pub tokens: Vec<Vec<u32>>,
    /// the stitched events
    pub events: Vec<Event>,
    pub duration: f64,
    pub warnings: Vec<String>,
    pub encode_seconds: f64,
    pub decode_seconds: f64,
}

pub struct SheetSage {
    pub mert: Mert,
    pub dec: Decoder,
    pub gpu: Arc<Gpu>,
    work: Mutex<Scratch>,
    pub bytes: u64,
}

impl SheetSage {
    pub fn load(path: &Path, gpu: &Arc<Gpu>) -> Result<SheetSage> {
        let f = Gguf::open(path).map_err(|e| Error(e.0))?;
        if f.architecture() != "sheetsage2" {
            return Err(Error(format!("{}: a {} file; SheetSage2's is arch sheetsage2", path.display(), f.architecture())));
        }
        let fp = f.arch_meta("vocab_fingerprint").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        if fp != "5ba3325af0344c7f" || f.arch_meta("vocab_size").and_then(|v| v.as_u64()) != Some(tokens::VOCAB as u64) {
            return Err(Error(format!("{}: vocabulary {fp:?}, SheetSage2 v1's 5ba3325af0344c7f (31678 tokens) expected", path.display())));
        }
        if f.arch_meta("encoder.attention_lora_merged").and_then(|v| v.as_u64()) == Some(0) {
            return Err(Error(format!("{}: the encoder's adapters are not merged into it", path.display())));
        }
        let mert = Mert::load(&f, gpu)?;
        let dec = Decoder::load(&f, gpu)?;
        gpu.sync()?;
        let bytes = mert.bytes + dec.bytes;
        Ok(SheetSage { mert, dec, gpu: gpu.clone(), work: Mutex::new(Scratch::default()), bytes })
    }

    /// The prompts transcribe asks for (FULL_TASK_PROMPTS)
    pub fn prompts() -> Vec<u32> {
        tokens::normalize(&tokens::FULL).unwrap_or_default()
    }

    /// The features of the window of `wave` from `start` seconds (300 s, zero-padded past the recording's end)
    pub fn window_features(&self, wave: &[f32], start: f64) -> Vec<f32> {
        let off = (start * wave::RATE as f64).round_ties_even() as usize;
        let count = (WINDOW * wave::RATE as f64).round_ties_even() as usize;
        let mut seg = vec![0f32; count];
        if off < wave.len() {
            let n = (wave.len() - off).min(count);
            seg[..n].copy_from_slice(&wave[off..off + n]);
        }
        self.mert.front.features(&seg)
    }

    /// The greedy decode of one window (constrained_prompt_generate): from `prefix` (sos, the prompts, out, maybe an
    /// overlap's events), a time token past `stop` seconds ends it with eos
    pub fn generate(&self, s: &mut Scratch, memory: &DevBuf, prefix: &[u32], stop: Option<f64>, progress: &mut dyn FnMut(usize) -> Result<()>)
                    -> Result<Vec<u32>> {
        let mut prefix = prefix.to_vec();
        if prefix.last() == Some(&EOS) {
            prefix.pop();
        }
        if prefix.first() != Some(&tokens::SOS) {
            return Err(Error("the generation's prefix must begin with <|sos|>".into()));
        }
        let out_at = prefix.iter().position(|t| *t == OUT).ok_or_else(|| Error("the prefix has no <|out|>".into()))?;
        let mut g = Grammar::default();
        for t in &prefix[out_at + 1..] {
            g.update(*t).map_err(Error)?;
        }
        let mut st = self.dec.start(s, memory, CONTEXT)?;
        let mut logits = self.dec.forward(s, &mut st, &prefix, &[prefix.len() - 1])?;
        let mut out = prefix.clone();
        let mut cur = prefix.len();
        while cur < CONTEXT {
            let t = g.pick(&logits);
            out.push(t);
            let mut done = g.update(t).map_err(Error)?;
            if !done && (tokens::TIME0..tokens::METER0).contains(&t) {
                if let Some(stop) = stop {
                    if (t - tokens::TIME0) as f64 / tokens::TIME_HZ as f64 >= stop {
                        out.push(EOS);
                        done = true;
                    }
                }
            }
            cur += 1;
            if cur.is_multiple_of(64) {
                progress(cur)?;
            }
            if done || cur >= CONTEXT {
                break;
            }
            logits = self.dec.forward(s, &mut st, &[t], &[0])?;
        }
        if out.last() != Some(&EOS) {
            out.push(EOS);
        }
        Ok(out)
    }

    /// The whole recording (24 kHz mono: wave::prepare) to its lead sheet, window by window as pipeline_sheetsage2.py
    pub fn transcribe(&self, wave: &[f32], progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Transcription> {
        let t0 = Instant::now();
        if wave.len() < 1025 || wave.iter().any(|x| !x.is_finite()) {
            return Err(Error("the recording must have at least 1025 finite samples at 24 kHz".into()));
        }
        let duration = wave.len() as f64 / wave::RATE as f64;
        let plan = stitch::plan(duration, WINDOW, OVERLAP, LOOKAHEAD).map_err(Error)?;
        let prompts = Self::prompts();
        let mut s = self.work.lock().unwrap_or_else(|p| p.into_inner());
        let mut tr = Transcription { duration, ..Default::default() };
        let mut stitched: Vec<Event> = Vec::new();
        let n = plan.len() as u32;
        for (i, w) in plan.iter().enumerate() {
            progress(Step { phase: "transcribe", at: i as u32, of: n, seconds: t0.elapsed().as_secs_f64() })?;
            let (mut prefix, mut base) = (tokens::prompt_prefix(&prompts), 0i64);
            if i > 0 {
                if let Some((p, b)) = stitch::overlap_prefix(&stitched, &prompts, w.start, w.prefix_end).map_err(Error)? {
                    if p.len() >= CONTEXT - 128 {
                        return Err(Error("the overlap's prefix fills the decoder's context".into()));
                    }
                    prefix = p;
                    base = b;
                }
            }
            let te = Instant::now();
            let mel = self.window_features(wave, w.start);
            let memory = self.mert.encode(&mut s, &mel, &mut |_, _, _| Ok(()))?;
            self.gpu.sync()?;
            tr.encode_seconds += te.elapsed().as_secs_f64();
            let td = Instant::now();
            let stop = w.generation_stop.unwrap_or_else(|| (duration - w.start).min(WINDOW));
            let toks = self.generate(&mut s, &memory, &prefix, Some(stop), &mut |c| {
                progress(Step { phase: "transcribe", at: i as u32, of: n, seconds: t0.elapsed().as_secs_f64() }).map(|_| {
                    let _ = c;
                })
            })?;
            tr.decode_seconds += td.elapsed().as_secs_f64();
            if toks.len() > CONTEXT {
                tr.warnings.push(format!("Window {} reached the token limit; inspect its token coverage", i + 1));
            }
            let (decoded, warn) = tokens::decode_generated(&toks).map_err(Error)?;
            if let Some(wr) = warn {
                tr.warnings.push(wr);
            }
            let map = stitch::TimeMap::new(&decoded, WINDOW);
            stitched.extend(stitch::stitched(&decoded, &map, w, duration, i, base));
            tr.tokens.push(toks);
        }
        s.clear();
        drop(s);
        stitched.sort_by(|a, b| a.time.unwrap_or(0.0).total_cmp(&b.time.unwrap_or(0.0)).then(a.global_subbeat.cmp(&b.global_subbeat)));
        tr.full = notation::export(&stitched, duration, false);
        tr.melody = notation::export(&stitched, duration, true);
        tr.events = stitched;
        progress(Step { phase: "transcribe", at: n, of: n, seconds: t0.elapsed().as_secs_f64() })?;
        Ok(tr)
    }
}
