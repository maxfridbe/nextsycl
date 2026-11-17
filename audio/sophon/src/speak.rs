//! The voice: Flash's talker (Qwen3-TTS-12Hz-1.7B-Base fine-tuned to perform sound tags) reading the brain's reply
//! while it is written (the release's talker.py: SophonTTSTalker), then the 12 Hz codec.
//!
//! ```text
//!   prompt: proj("<|im_start|>assistant\n")                                         3 rows
//!           tts_pad x 5, tts_bos  +  codec [think, think_bos, language, think_eos, x-vector, pad]
//!           proj(first text token) + codec bos
//!   a frame: codes drawn from the last hidden state, then the talker fed the frame's embeddings plus the next text
//!            token's projection - tts_eos once the reply is whole, tts_pad after it; with no text in hand it waits
//!            (never pads while the brain still writes), and the end of speech is held back until tts_eos is fed
//! ```
//!
//! Text ids are the thinker's tokenizer's (the same BPE as the talker's). A voice is an x-vector: a voices/<name>.pt
//! of the release (2,048 floats), or a recording's (speaker.rs: the checkpoint's ECAPA encoder). 24 kHz out.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_codec12hz::{Codec, Weights as CodecWeights};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;

use crate::ops::{fp, Ops, Shards};
use crate::sample::{Draw, Rng};
use crate::speaker::Speaker;
use crate::talker::{Drawing, Talker};

pub const RATE: u32 = 24000;
pub const FPS: f32 = 12.5;
/// frames a text token at most after the text has ended (talker.py's runaway guard: 12 a token + 60)
const FRAMES_PER_TOKEN: usize = 12;
/// the first audio after this many frames (0.64 s), then the codec's own chunk points
const STREAM_FIRST: usize = 6;

/// talker/config.json's ids
#[derive(Clone, Debug)]
pub struct Config {
    pub tts_bos: u32,
    pub tts_eos: u32,
    pub tts_pad: u32,
    pub pad: i32,
    pub bos: i32,
    pub eos: i32,
    pub think: i32,
    pub think_bos: i32,
    pub think_eos: i32,
    pub vocab: usize,
    pub languages: Vec<(String, i32)>,
}

impl Config {
    pub fn read(p: &Path) -> Result<Config> {
        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
            .map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let t = &c["talker_config"];
        let int = |v: &serde_json::Value, k: &str| v[k].as_i64().ok_or_else(|| Error(format!("{}: no {k}", p.display())));
        Ok(Config {
            tts_bos: int(&c, "tts_bos_token_id")? as u32,
            tts_eos: int(&c, "tts_eos_token_id")? as u32,
            tts_pad: int(&c, "tts_pad_token_id")? as u32,
            pad: int(t, "codec_pad_id")? as i32,
            bos: int(t, "codec_bos_id")? as i32,
            eos: int(t, "codec_eos_token_id")? as i32,
            think: int(t, "codec_think_id")? as i32,
            think_bos: int(t, "codec_think_bos_id")? as i32,
            think_eos: int(t, "codec_think_eos_id")? as i32,
            vocab: int(t, "vocab_size")? as usize,
            languages: t["codec_language_id"].as_object().into_iter().flatten().filter_map(|(n, v)| Some((n.clone(), v.as_i64()? as i32))).collect(),
        })
    }

    /// The codec id of a language code ("en", "zh", ...): talker.py's LANG table, English when unknown
    pub fn language(&self, code: &str) -> i32 {
        let name = match code {
            "zh" => "chinese",
            "ja" => "japanese",
            "ko" => "korean",
            "de" => "german",
            "fr" => "french",
            "es" => "spanish",
            "it" => "italian",
            "pt" => "portuguese",
            "ru" => "russian",
            _ => "english",
        };
        let get = |n: &str| self.languages.iter().find(|(k, _)| k == n).map(|(_, v)| *v);
        get(name).or_else(|| get("english")).unwrap_or(2050)
    }
}

/// A release voice: voices/<name>.pt - a PyTorch zip whose one storage (`.../data/0`) is the x-vector, float32
pub fn read_pt_xvector(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let bad = || Error(format!("{}: not a PyTorch zip with one float32 storage", p.display()));
    let u16at = |k: usize| -> Option<usize> { Some(u16::from_le_bytes([*b.get(k)?, *b.get(k + 1)?]) as usize) };
    let u32at = |k: usize| -> Option<usize> { Some(u32::from_le_bytes([*b.get(k)?, *b.get(k + 1)?, *b.get(k + 2)?, *b.get(k + 3)?]) as usize) };
    // the end of central directory record, then its entries (the sizes are there: PyTorch writes data descriptors,
    // so its local headers carry none)
    let eocd = (0..b.len().saturating_sub(21)).rev().take(1 << 16).find(|&i| b[i..i + 4] == [0x50, 0x4b, 0x05, 0x06]).ok_or_else(bad)?;
    let (count, mut at) = (u16at(eocd + 10).ok_or_else(bad)?, u32at(eocd + 16).ok_or_else(bad)?);
    for _ in 0..count {
        if b.get(at..at + 4) != Some(&[0x50, 0x4b, 0x01, 0x02]) {
            return Err(bad());
        }
        let (method, size) = (u16at(at + 10).ok_or_else(bad)?, u32at(at + 20).ok_or_else(bad)?);
        let (nlen, xlen, clen, local) = (u16at(at + 28).ok_or_else(bad)?, u16at(at + 30).ok_or_else(bad)?, u16at(at + 32).ok_or_else(bad)?,
                                         u32at(at + 42).ok_or_else(bad)?);
        let name = String::from_utf8_lossy(b.get(at + 46..at + 46 + nlen).ok_or_else(bad)?).to_string();
        if name.ends_with("/data/0") {
            if method != 0 || size % 4 != 0 {
                return Err(Error(format!("{}: its storage is compressed or not float32", p.display())));
            }
            let data = local + 30 + u16at(local + 26).ok_or_else(bad)? + u16at(local + 28).ok_or_else(bad)?;
            let s = b.get(data..data + size).ok_or_else(bad)?;
            return Ok(s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
        }
        at += 46 + nlen + xlen + clen;
    }
    Err(Error(format!("{}: no tensor storage (data/0) in it", p.display())))
}

/// What the brain hands the talker: the reply's speakable text ids so far, and whether it is whole
pub struct Feed {
    pub ids: Vec<u32>,
    pub done: bool,
}

/// How a speech went
#[derive(Clone, Copy, Debug, Default)]
pub struct Spoken {
    pub frames: usize,
    pub samples: usize,
    /// seconds from the start to the first audio handed out
    pub first_audio: f64,
    pub seconds: f64,
}

/// What a check catches on the way (check.rs's talker stage): the prompt's rows, the first codebook's logits after
/// it, every frame's codes
#[derive(Default)]
pub struct Probe {
    pub prefill: Vec<f32>,
    pub logits0: Vec<f32>,
    pub codes: Vec<Vec<i32>>,
}

pub struct Talk {
    pub ops: Ops,
    pub nsd: Nsd,
    pub talker: Talker,
    pub codec: Codec,
    pub conf: Config,
    pub speaker: Option<Speaker>,
    weights: Shards,
    pub voices: PathBuf,
}

impl Talk {
    pub fn load(model: &Path, config: &Path, codec: &Path, voices: &Path, gpu: &Arc<Gpu>, int8: bool, log: &mut dyn FnMut(String)) -> Result<Talk> {
        let t0 = Instant::now();
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let conf = Config::read(config)?;
        let weights = Shards::open(&[model.to_path_buf()])?;
        let talker = Talker::load(&ops, &weights, int8, log)?;
        let speaker = if weights.has("speaker_encoder.fc.weight") { Some(Speaker::load(&weights)?) } else { None };
        let codec = Codec::load(gpu, &CodecWeights::open(codec)?)?;
        codec.decode_bucketed(&nsd, &[vec![0; 16]])?;
        gpu.sync()?;
        log(format!("talker (Qwen3-TTS 12 Hz, {}) and codec on {} in {:.0} s", if int8 { "int8" } else { "half" }, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(Talk { ops, nsd, talker, codec, conf, speaker, weights, voices: voices.to_path_buf() })
    }

    /// The x-vector of a release voice: voices/<name>_<lang>.pt, else voices/<name>.pt (talker.py's lookup)
    pub fn voice(&self, name: &str, lang: &str) -> Result<Vec<f32>> {
        let ok = !name.is_empty() && name.len() <= 64 && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "_.-".contains(c));
        if !ok {
            return Err(Error(format!("voice {name:?}: lower-case letters, digits, _ . -")));
        }
        for n in [format!("{name}_{lang}"), name.to_string()] {
            let p = self.voices.join(format!("{n}.pt"));
            if p.exists() {
                return read_pt_xvector(&p);
            }
        }
        Err(Error(format!("voice {name:?}: not in {} ({})", self.voices.display(), self.voice_names().join(", "))))
    }

    /// The release's voices (their names, the language suffix taken off)
    pub fn voice_names(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(&self.voices).into_iter().flatten().flatten()
            .filter_map(|e| e.file_name().to_str()?.strip_suffix(".pt").map(|n| n.trim_end_matches("_en").trim_end_matches("_zh").to_string()))
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// A recording's x-vector (a clone: 3-15 s of one speaker, any rate)
    pub fn clone_voice(&self, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
        let sp = self.speaker.as_ref().ok_or_else(|| Error("the talker has no speaker encoder (no clones)".into()))?;
        if (samples.len() as f64) < rate as f64 {
            return Err(Error("a voice recording under 1 s".into()));
        }
        let keep = (rate as usize * 15).min(samples.len());
        let wav = crate::speaker::resample(&samples[..keep], rate);
        sp.embed(&wav)
    }

    /// Speak the text `next` hands out as it comes (it blocks until there is more, or the end), in `xvec`'s voice:
    /// samples to `out` as they are made; `stop` asked between frames (a cancel: true ends it)
    #[allow(clippy::too_many_arguments)]
    pub fn speak(&self, xvec: &[f32], lang: &str, draw: Draw, seed: u64, next: &mut dyn FnMut(usize) -> Feed, out: &mut dyn FnMut(&[f32]) -> Result<()>,
                 stop: &dyn Fn() -> bool) -> Result<Spoken> {
        self.speak_probed(xvec, lang, draw, seed, next, out, stop, None)
    }

    /// `speak`, `probe` catching the prompt, the first logits and the codes
    #[allow(clippy::too_many_arguments)]
    pub fn speak_probed(&self, xvec: &[f32], lang: &str, draw: Draw, seed: u64, next: &mut dyn FnMut(usize) -> Feed,
                        out: &mut dyn FnMut(&[f32]) -> Result<()>, stop: &dyn Fn() -> bool, mut probe: Option<&mut Probe>) -> Result<Spoken> {
        let t0 = Instant::now();
        let (ops, nsd, t, c) = (&self.ops, &self.nsd, &self.talker, &self.conf);
        let h = t.lm.hidden;
        if xvec.len() != h {
            return Err(Error(format!("a voice of {} values; the talker takes {h}", xvec.len())));
        }
        // the first text token (or the reply's end without one)
        let mut feed = next(1);
        while feed.ids.is_empty() && !feed.done {
            if stop() {
                return Ok(Spoken::default());
            }
            feed = next(1);
        }
        if feed.ids.is_empty() {
            return Ok(Spoken::default());
        }
        let (max_tokens, max_frames) = (1024, 2400);
        let ss = t.session(ops, 16, max_frames)?;
        let proj = |ids: &[u32]| -> Result<Vec<f32>> {
            let mut rows = Vec::with_capacity(ids.len() * self.talker.fc1.k);
            for &i in ids {
                rows.extend(self.weights.rows_f32("talker.model.text_embedding.weight", i as usize, i as usize + 1)?);
            }
            t.project_text(ops, nsd, &ss, &rows)
        };
        let specials = proj(&[c.tts_bos, c.tts_eos, c.tts_pad, 151644, 77091, 198])?;
        let row = |i: usize| &specials[i * h..(i + 1) * h];
        let (tts_bos, tts_eos, tts_pad) = (row(0).to_vec(), row(1).to_vec(), row(2).to_vec());
        let codec_ids = [c.think, c.think_bos, c.language(lang), c.think_eos];
        let mut codec_rows: Vec<Vec<f32>> = codec_ids.iter().map(|i| t.codec_sum(ops, &ss, &[*i])).collect::<Result<_>>()?;
        codec_rows.push(xvec.to_vec());
        codec_rows.push(t.codec_sum(ops, &ss, &[c.pad])?);
        let bos = t.codec_sum(ops, &ss, &[c.bos])?;
        let mut rows: Vec<f32> = specials[3 * h..6 * h].to_vec();
        let n = codec_rows.len();
        for (i, cr) in codec_rows.iter().enumerate() {
            let base = if i + 1 < n { &tts_pad } else { &tts_bos };
            rows.extend(base.iter().zip(cr).map(|(a, b)| a + b));
        }
        let first = proj(&feed.ids[..1])?;
        rows.extend(first.iter().zip(&bos).map(|(a, b)| a + b));
        let l = rows.len() / h;
        t.prefill(ops, nsd, &ss, &rows)?;
        if let Some(p) = probe.as_deref_mut() {
            p.prefill = rows.clone();
            p.logits0 = t.logits0(ops, &ss)?;
        }
        // the text rows fed a frame each, on the card: [tokens + eos, then pad]
        let feed_buf = DevBuf::f32(&ops.gpu, (max_tokens + 2) * h)?;
        let pad_at = max_tokens + 1;
        feed_buf.write(pad_at * h * 4, &crate::f32_bytes(&tts_pad))?;
        let mut fed = 1usize; // text ids projected so far (the first is in the prompt)
        let mut rows_in = 0usize; // rows in feed_buf
        let mut eos_in = false;
        let mut done = feed.done;
        let mut total_ids = feed.ids.len();
        let add = |ids: &[u32], from: usize, rows_in: &mut usize| -> Result<()> {
            if from < ids.len() {
                let p = proj(&ids[from..])?;
                let k = (ids.len() - from).min(max_tokens - *rows_in);
                feed_buf.write(*rows_in * h * 4, &crate::f32_bytes(&p[..k * h]))?;
                *rows_in += k;
            }
            Ok(())
        };
        add(&feed.ids, fed, &mut rows_in)?;
        fed = feed.ids.len();
        let mut rng = Rng::new(seed);
        let mut codes: Vec<Vec<i32>> = Vec::new();
        let mut seen = Vec::new();
        let mut sent = 0;
        let mut spoken = Spoken::default();
        let mut lead = true;
        let mut cut = 0usize;
        for f in 0..max_frames {
            if stop() {
                break;
            }
            // the row this frame feeds: wait for text while the reply is still being written
            while f >= rows_in && !eos_in {
                if done {
                    feed_buf.write(rows_in * h * 4, &crate::f32_bytes(&tts_eos))?;
                    rows_in += 1;
                    eos_in = true;
                    break;
                }
                let more = next(fed + 1);
                if stop() {
                    break;
                }
                done = more.done;
                total_ids = more.ids.len();
                add(&more.ids, fed, &mut rows_in)?;
                fed = more.ids.len().max(fed);
            }
            if stop() {
                break;
            }
            let at = if f < rows_in { f } else { pad_at };
            let d = Drawing {
                talker: draw,
                predictor: draw,
                repetition_penalty: if draw.sample { 1.05 } else { 1.0 },
                eos: c.eos,
                suppress_from: c.vocab - 1024,
                // the end of speech held back while text is still to come
                min_frames: if eos_in { 2 } else { usize::MAX },
            };
            let guard = FRAMES_PER_TOKEN * total_ids.max(1) + 60;
            let fr = if eos_in && f > guard { None } else { t.frame(ops, nsd, &ss, l + f, fp(&feed_buf, at * h), &d, &seen, &mut rng, None)? };
            let Some(cd) = fr else { break };
            seen.push(cd[0]);
            codes.push(cd);
            if Codec::stream_due(codes.len(), sent, STREAM_FIRST) {
                let mut wav = self.codec.decode_from(nsd, &codes, sent)?;
                sent = codes.len();
                if lead {
                    // the talker starts a reply with a little silence: dropped (talker.py's LEAD_TRIM, up to 1 s)
                    let nz = wav.iter().position(|v| v.abs() > 1e-3);
                    if nz.is_none() && cut + wav.len() <= RATE as usize {
                        cut += wav.len();
                        continue;
                    }
                    let k = nz.unwrap_or(0).saturating_sub((0.02 * RATE as f32) as usize).min((RATE as usize).saturating_sub(cut));
                    wav.drain(..k.min(wav.len()));
                    lead = false;
                }
                if spoken.first_audio == 0.0 && !wav.is_empty() {
                    spoken.first_audio = t0.elapsed().as_secs_f64();
                }
                spoken.samples += wav.len();
                out(&wav)?;
            }
        }
        if !stop() && sent < codes.len() {
            let wav = self.codec.decode_from(nsd, &codes, sent)?;
            if spoken.first_audio == 0.0 && !wav.is_empty() {
                spoken.first_audio = t0.elapsed().as_secs_f64();
            }
            spoken.samples += wav.len();
            out(&wav)?;
        }
        spoken.frames = codes.len();
        if let Some(p) = probe {
            p.codes = codes;
        }
        spoken.seconds = t0.elapsed().as_secs_f64();
        Ok(spoken)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zip as PyTorch writes one: stored entries with data descriptors (no sizes in the local headers)
    fn torch_zip(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        let entries: Vec<(&str, Vec<u8>)> = vec![("v/data.pkl", vec![0x80, 2, 0x2e]), ("v/data/0", values.iter().flat_map(|x| x.to_le_bytes()).collect())];
        for (name, data) in &entries {
            let off = out.len() as u32;
            out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04, 20, 0, 0x08, 0x08, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&[0u8; 12]);
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            out.extend_from_slice(&[0x50, 0x4b, 0x07, 0x08, 0, 0, 0, 0]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02, 20, 0, 20, 0, 0x08, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0u8; 12]);
            central.extend_from_slice(&off.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 2, 0, 2, 0]);
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn a_release_voice_is_read() {
        let p = std::env::temp_dir().join(format!("sophon-voice-{}.pt", std::process::id()));
        std::fs::write(&p, torch_zip(&[0.5, -1.25, 3.0])).unwrap();
        let v = read_pt_xvector(&p);
        std::fs::remove_file(&p).ok();
        assert_eq!(v.unwrap(), vec![0.5, -1.25, 3.0]);
    }
}
