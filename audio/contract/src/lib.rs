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
