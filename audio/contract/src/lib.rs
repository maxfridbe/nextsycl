//! The contract between the audio server / command line and the audio engines. An engine is one model architecture's
//! whole pipeline - from a description (and lyrics, for songs) to a waveform - with its own kernels
//! (`kernels/audio/<arch>`) and memory plan. The program drives it only through `AudioEngine`; it is chosen by the
//! architecture its registry entry names (`AudioKind`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use nextsycl_core::options::Given;
pub use nextsycl_core::{At, EngineOption, Error, Gpu, Result};

/// One generation: what to make and its settings. `None` fields take the engine's defaults.
#[derive(Clone, Debug, Default)]
pub struct AudioRequest {
    /// the description: genre, mood, tempo and key, the voice, the instruments, the arrangement
    pub prompt: String,
    /// the words to sing, structure tags (`[verse]`, `[chorus]` ...) on lines of their own
    pub lyrics: Option<String>,
    /// at most this long (a song may end sooner: the model decides)
    pub seconds: Option<f32>,
    /// denoising steps (of each window, for a model that makes the sound in windows)
    pub steps: Option<u32>,
    /// guidance scale (1.0: none)
    pub cfg: Option<f32>,
    pub seed: u64,
    /// the engine's own options for this request, by name (`--opt-NAME`, an API request's `options`), checked
    /// against those it declares for requests
    pub extra: Given,
    /// speech (an engine whose `speech` is Some; `prompt` is then the text to say): a built-in voice
    pub voice: Option<String>,
    /// speech: the language of the text ("auto" or None: the engine's guess)
    pub language: Option<String>,
    /// speech: how to say it, or the voice to make up (a designed voice)
    pub instructions: Option<String>,
    /// speech: a recording whose voice to speak in (a clone)
    pub reference: Option<Reference>,
}

/// A recording to clone a voice from: mono samples, their rate, and what is said in it (a transcript makes a
/// closer clone; without one only the voice's timbre is taken)
#[derive(Clone, Debug, Default)]
pub struct Reference {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub text: Option<String>,
}

/// What a speech engine takes
#[derive(Clone, Debug, Default)]
pub struct Speech {
    /// its built-in voices (`AudioRequest::voice`)
    pub voices: Vec<String>,
    /// the languages it speaks (besides "auto")
    pub languages: Vec<String>,
    /// whether it styles the speech by `instructions`
    pub instructions: bool,
    /// whether it makes a voice from `instructions` alone (no built-in voice, no recording)
    pub design: bool,
    /// whether it clones a `reference` recording (and whether it needs the recording's transcript)
    pub clone: bool,
    pub clone_needs_text: bool,
    /// whether a transcript of the recording makes a closer clone
    pub clone_takes_text: bool,
}

/// The engine's own defaults, shown by `engines` and used for the request's `None` fields
#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    pub seconds: f32,
    /// the longest it makes
    pub max_seconds: f32,
    pub steps: u32,
    pub cfg: f32,
    /// samples a second of what it returns
    pub rate: u32,
}

/// Progress, reported as the engine goes: `phase` is the engine's ("prompt", "tokens", "flow", "decode" ...), `at`
/// of `of` its units (frames, steps, windows)
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub phase: &'static str,
    pub at: u32,
    pub of: u32,
    /// seconds since the request started
    pub seconds: f64,
}

/// A waveform: `channels` interleaved, in [-1, 1]
#[derive(Clone, Debug, Default)]
pub struct Audio {
    pub rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Audio {
    /// seconds of sound
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.channels.max(1) as f64 / self.rate.max(1) as f64
    }

    /// As a 16-bit PCM WAV file's bytes, `info` (name, text) pairs in a LIST INFO chunk (INAM, ICMT, ...)
    pub fn wav(&self, info: &[(&str, &str)]) -> Vec<u8> {
        let data: Vec<u8> = self.samples.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect();
        let mut list = Vec::new();
        for (id, text) in info {
            let mut t = text.as_bytes().to_vec();
            t.push(0);
            if t.len() % 2 == 1 {
                t.push(0);
            }
            list.extend_from_slice(&id.as_bytes()[..4]);
            list.extend_from_slice(&(t.len() as u32).to_le_bytes());
            list.extend_from_slice(&t);
        }
        let ch = self.channels.max(1);
        let mut out = Vec::with_capacity(data.len() + list.len() + 64);
        let riff = 4 + (8 + 16) + (8 + data.len()) + if list.is_empty() { 0 } else { 12 + list.len() };
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(riff as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&ch.to_le_bytes());
        out.extend_from_slice(&self.rate.to_le_bytes());
        out.extend_from_slice(&(self.rate * ch as u32 * 2).to_le_bytes());
        out.extend_from_slice(&(ch * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        if !list.is_empty() {
            out.extend_from_slice(b"LIST");
            out.extend_from_slice(&((4 + list.len()) as u32).to_le_bytes());
            out.extend_from_slice(b"INFO");
            out.extend_from_slice(&list);
        }
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    pub fn write_wav(&self, path: &Path, info: &[(&str, &str)]) -> std::result::Result<(), String> {
        std::fs::write(path, self.wav(info)).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// A WAV file's samples (PCM 8 / 16 / 24 / 32-bit or float 32 / 64; the channels mixed to mono) and their rate
pub fn decode_wav(b: &[u8]) -> std::result::Result<(Vec<f32>, u32), String> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let u16at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
    let u32at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
    let (mut fmt, mut ch, mut rate, mut bits) = (0, 0, 0, 0);
    let mut i = 12;
    while i + 8 <= b.len() {
        let (id, len) = (&b[i..i + 4], u32at(i + 4));
        let body = i + 8;
        match id {
            b"fmt " if body + 16 <= b.len() => {
                fmt = u16at(body);
                ch = u16at(body + 2);
                rate = u32at(body + 4);
                bits = u16at(body + 14);
                // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes
                if fmt == 0xFFFE && body + 26 <= b.len() {
                    fmt = u16at(body + 24);
                }
            }
            b"data" => {
                if ch == 0 || rate == 0 {
                    return Err("no fmt chunk ahead of the samples".into());
                }
                let end = (body + len).min(b.len());
                let d = &b[body..end];
                let one: fn(&[u8]) -> f32 = match (fmt, bits) {
                    (1, 8) => |c| (c[0] as f32 - 128.0) / 128.0,
                    (1, 16) => |c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0,
                    (1, 24) => |c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0,
                    (1, 32) => |c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0,
                    (3, 32) => |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    (3, 64) => |c| f64::from_le_bytes(c.try_into().expect("8 bytes")) as f32,
                    _ => return Err(format!("format {fmt} at {bits} bits: PCM or float expected")),
                };
                let w = bits / 8;
                let samples = d.chunks_exact(w * ch).map(|f| f.chunks_exact(w).map(one).sum::<f32>() / ch as f32).collect();
                return Ok((samples, rate as u32));
            }
            _ => {}
        }
        i = body + len + (len & 1);
    }
    Err("no data chunk".into())
}

/// A WAV file's LIST INFO entries and its length in seconds (from the chunks ahead of its samples: what `Audio::wav`
/// writes; only the head of the file is read)
pub fn wav_info(path: &Path) -> std::result::Result<(Vec<(String, String)>, f64), String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut b = vec![0u8; 1 << 16];
    let n = f.read(&mut b).map_err(|e| format!("{}: {e}", path.display()))?;
    b.truncate(n);
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("{}: not a WAV file", path.display()));
    }
    let u32at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
    let (mut info, mut rate, mut block, mut seconds) = (Vec::new(), 0usize, 0usize, 0f64);
    let mut i = 12;
    while i + 8 <= b.len() {
        let (id, len) = (&b[i..i + 4], u32at(i + 4));
        let body = i + 8;
        match id {
            b"fmt " if body + 16 <= b.len() => {
                rate = u32at(body + 4);
                block = u16::from_le_bytes([b[body + 12], b[body + 13]]) as usize;
            }
            b"LIST" if body + 4 <= b.len() && &b[body..body + 4] == b"INFO" => {
                let mut j = body + 4;
                while j + 8 <= (body + len).min(b.len()) {
                    let (k, l) = (String::from_utf8_lossy(&b[j..j + 4]).into_owned(), u32at(j + 4));
                    let end = (j + 8 + l).min(b.len());
                    let text = String::from_utf8_lossy(&b[j + 8..end]).trim_end_matches('\0').to_string();
                    info.push((k, text));
                    j = j + 8 + l + (l & 1);
                }
            }
            b"data" => {
                if rate > 0 && block > 0 {
                    seconds = len as f64 / (rate * block) as f64;
                }
                break;
            }
            _ => {}
        }
        i = body + len + (len & 1);
    }
    Ok((info, seconds))
}

/// A model's files by role (the engine says which it needs)
pub type ModelFiles = BTreeMap<String, PathBuf>;

/// How an engine is loaded
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// settings by name (a registry entry's `env`, the --opt-NAMEs taken at load): what the process environment would
    /// hold when the engine is served; an engine reads a setting here first, then from the environment
    pub settings: BTreeMap<String, String>,
}

impl LoadOptions {
    /// A setting: from `settings`, else the environment
    pub fn setting(&self, name: &str) -> Option<String> {
        self.settings.get(name).cloned().or_else(|| std::env::var(name).ok())
    }
}

/// The runtime one audio architecture brings
pub trait AudioEngine: Send + Sync {
    /// its architecture (the registry entry's "arch")
    fn arch(&self) -> &'static str;
    fn defaults(&self) -> Defaults;
    /// whether it sings words (`AudioRequest::lyrics`)
    fn lyrics(&self) -> bool {
        false
    }
    /// a speech engine's voices and abilities (None: not speech - songs, sounds)
    fn speech(&self) -> Option<Speech> {
        None
    }
    /// the options it takes beyond the request's fields (its kind's `options`)
    fn options(&self) -> &'static [EngineOption] {
        &[]
    }
    /// seconds the load took
    fn load_seconds(&self) -> f64;
    /// make the request's sound, reporting as it goes; an error from `progress` stops it (a cancel)
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio>;
    /// lines for `status`: its GPUs, memory, what is loaded where
    fn report(&self) -> Vec<String> {
        Vec::new()
    }
}

/// How an engine loads: its files, its GPUs, the options, a log
pub type LoadFn = fn(&ModelFiles, &[Arc<Gpu>], &LoadOptions, &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>>;

/// An engine's registry entry
pub struct AudioKind {
    /// the architectures it serves (a registry entry's "arch")
    pub archs: &'static [&'static str],
    /// what it is
    pub name: &'static str,
    /// the file roles it needs (a role ending in `*`: one or more, `<role>-1`, `<role>-2` ... - a sharded file)
    pub roles: &'static [&'static str],
    pub load: LoadFn,
    /// the options it takes (`--opt-NAME`: nextsycl_core::options) - at load as the variables they name (in
    /// `LoadOptions::settings` too), per request in `AudioRequest::extra`
    pub options: &'static [EngineOption],
}

/// The entry of `kinds` serving `arch`
pub fn kind_for<'k>(kinds: &'k [AudioKind], arch: &str) -> std::result::Result<&'k AudioKind, String> {
    kinds.iter().find(|k| k.archs.contains(&arch)).ok_or_else(|| {
        let known: Vec<&str> = kinds.iter().flat_map(|k| k.archs.iter().copied()).collect();
        format!("audio architecture {arch:?} has no engine here (known: {})", known.join(", "))
    })
}

/// The files of a sharded role, in order: `role` itself, or `role-1`, `role-2` ...
pub fn shards(files: &ModelFiles, role: &str) -> Vec<PathBuf> {
    if let Some(p) = files.get(role) {
        return vec![p.clone()];
    }
    (1..).map_while(|i| files.get(&format!("{role}-{i}")).cloned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wav_has_its_header_and_info() {
        let a = Audio { rate: 44100, channels: 2, samples: vec![0.0, 1.0, -1.0, 0.5] };
        let w = a.wav(&[("INAM", "x")]);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes([w[4], w[5], w[6], w[7]]) as usize, w.len() - 8);
        let d = w.windows(4).position(|x| x == b"data").unwrap();
        assert_eq!(u32::from_le_bytes([w[d + 4], w[d + 5], w[d + 6], w[d + 7]]), 8);
        assert_eq!(i16::from_le_bytes([w[d + 10], w[d + 11]]), 32767);
        assert!(w.windows(4).any(|x| x == b"INAM"));
        let p = std::env::temp_dir().join(format!("nsaudio-{}.wav", std::process::id()));
        a.write_wav(&p, &[("INAM", "a song"), ("ICMT", "{\"seed\":7}")]).unwrap();
        let (info, secs) = wav_info(&p).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(info, vec![("INAM".to_string(), "a song".to_string()), ("ICMT".to_string(), "{\"seed\":7}".to_string())]);
        // and back: the two channels mixed
        let (s, rate) = decode_wav(&w).unwrap();
        assert_eq!(rate, 44100);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 0.5).abs() < 1e-3 && (s[1] + 0.25).abs() < 1e-3);
        assert!((secs - 2.0 / 44100.0).abs() < 1e-9);
    }

    #[test]
    fn shards_are_found_in_order() {
        let mut f = ModelFiles::new();
        f.insert("lm-2".into(), "b".into());
        f.insert("lm-1".into(), "a".into());
        assert_eq!(shards(&f, "lm"), vec![PathBuf::from("a"), PathBuf::from("b")]);
        f.insert("v".into(), "c".into());
        assert_eq!(shards(&f, "v"), vec![PathBuf::from("c")]);
    }
}
