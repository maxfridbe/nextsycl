//! Saved voices: a recording (cloned from a sample, or made by a voice-design model) with its transcript, kept by
//! name so a speech request can say `"voice": "NAME"` to any model that clones.
//!
//! ```text
//!   <audio output>/voices/<name>/sample.wav    the recording (mono, its own rate)
//!                                /voice.json    {name, text (what it says), description, language, kind: clone |
//!                                               design, seconds, rate, created, from}
//!                                /enrolled/<arch>.npy   the voice fitted for a model without an encoder (Voxtral
//!                                               TTS: `nextsycl audio voice enroll`), its report beside it (.json)
//! ```
//!
//! The audio server reads them (the output directory is mounted into its container); the front door and the
//! command line write them.

use std::path::{Path, PathBuf};

use nextsycl_audio::{Audio, Reference};
use serde_json::{json, Value};

/// The library beside an audio output directory
pub struct Library {
    pub dir: PathBuf,
}

/// The shortest and longest recording kept (seconds): long enough to carry a voice, short enough for a prompt
const MIN_S: f64 = 1.0;
const MAX_S: f64 = 60.0;

/// A voice's name: lowercase letters, digits, '-' and '_', 1-40 long, a letter first
pub fn check_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty() && name.len() <= 40 && name.starts_with(|c: char| c.is_ascii_lowercase())
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    if ok { Ok(()) } else { Err(format!("voice name {name:?}: lowercase letters, digits, - and _ (a letter first, at most 40)")) }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Library {
    /// The library of an audio output directory
    pub fn of(out_dir: &Path) -> Library {
        Library { dir: out_dir.join("voices") }
    }

    /// Every voice's record, by name (with `enrolled`: the architectures it is enrolled for)
    pub fn list(&self) -> Vec<Value> {
        let mut v: Vec<Value> = std::fs::read_dir(&self.dir).into_iter().flatten().flatten()
            .filter_map(|e| std::fs::read_to_string(e.path().join("voice.json")).ok())
            .filter_map(|t| serde_json::from_str::<Value>(&t).ok())
            .map(|mut r| {
                let name = r["name"].as_str().unwrap_or("").to_string();
                r["enrolled"] = json!(self.enrolled(&name));
                r
            })
            .collect();
        v.sort_by(|a: &Value, b: &Value| a["name"].as_str().cmp(&b["name"].as_str()));
        v
    }

    /// A voice's record
    pub fn get(&self, name: &str) -> Option<Value> {
        check_name(name).ok()?;
        serde_json::from_str(&std::fs::read_to_string(self.dir.join(name).join("voice.json")).ok()?).ok()
    }

    /// Its recording's path
    pub fn sample_path(&self, name: &str) -> Option<PathBuf> {
        check_name(name).ok()?;
        let p = self.dir.join(name).join("sample.wav");
        p.exists().then_some(p)
    }

    /// Where a voice's enrolment for an architecture goes (`<voice>/enrolled/<arch>.npy`)
    pub fn enrolled_file(&self, name: &str, arch: &str) -> Option<PathBuf> {
        check_name(name).ok()?;
        let ok = !arch.is_empty() && arch.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        ok.then(|| self.dir.join(name).join("enrolled").join(format!("{arch}.npy")))
    }

    /// A voice's enrolment for an architecture, when it has one
    pub fn enrolled_path(&self, name: &str, arch: &str) -> Option<PathBuf> {
        self.enrolled_file(name, arch).filter(|p| p.is_file())
    }

    /// The architectures a voice is enrolled for
    pub fn enrolled(&self, name: &str) -> Vec<String> {
        if check_name(name).is_err() {
            return Vec::new();
        }
        let mut v: Vec<String> = std::fs::read_dir(self.dir.join(name).join("enrolled")).into_iter().flatten().flatten()
            .filter_map(|e| e.file_name().to_str()?.strip_suffix(".npy").filter(|n| !n.starts_with('.')).map(str::to_string)).collect();
        v.sort();
        v
    }

    /// A voice as a request's reference: the recording and its transcript
    pub fn reference(&self, name: &str) -> Result<Reference, String> {
        let v = self.get(name).ok_or_else(|| format!("no saved voice {name:?} (nextsycl audio voice list)"))?;
        let p = self.sample_path(name).ok_or_else(|| format!("voice {name}: its sample.wav is missing"))?;
        let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let (samples, rate) = nextsycl_audio::decode_wav(&b).map_err(|e| format!("{}: {e}", p.display()))?;
        Ok(Reference { samples, rate, text: v["text"].as_str().filter(|t| !t.trim().is_empty()).map(str::to_string) })
    }

    /// Keep a recording (any format decode_audio reads) under `name`: its record returned. `replace`: over a voice of
    /// that name.
    #[allow(clippy::too_many_arguments)]
    pub fn add(&self, name: &str, recording: &[u8], text: Option<&str>, description: Option<&str>, language: Option<&str>, kind: &str, from: &str,
               replace: bool) -> Result<Value, String> {
        check_name(name)?;
        let d = self.dir.join(name);
        if d.join("voice.json").exists() && !replace {
            return Err(format!("a voice {name:?} exists already (remove it first, or replace it)"));
        }
        let (samples, rate) = nextsycl_audio::decode_audio(recording)?;
        let seconds = samples.len() as f64 / rate.max(1) as f64;
        if seconds < MIN_S {
            return Err(format!("the recording is {seconds:.1} s; a voice needs at least {MIN_S:.0} s (3-15 s is best)"));
        }
        // a long one is cut: the prompt carries all of it
        let keep = samples.len().min((MAX_S * rate as f64) as usize);
        let a = Audio { rate, channels: 1, samples: samples[..keep].to_vec() };
        std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
        // a new recording: the enrolments fitted to the old one no longer hold
        let _ = std::fs::remove_dir_all(d.join("enrolled"));
        a.write_wav(&d.join("sample.wav"), &[("INAM", name)])?;
        let rec = json!({
            "name": name, "text": text.map(str::trim).filter(|t| !t.is_empty()), "description": description, "language": language,
            "kind": kind, "seconds": (a.seconds() * 100.0).round() / 100.0, "rate": rate, "created": now(), "from": from,
        });
        std::fs::write(d.join("voice.json"), serde_json::to_string_pretty(&rec).unwrap_or_default()).map_err(|e| format!("{}: {e}", d.display()))?;
        Ok(rec)
    }

    /// A voice's record with one key set (its owner ...): the record
    pub fn tag(&self, name: &str, key: &str, value: Value) -> Result<Value, String> {
        let mut v = self.get(name).ok_or_else(|| format!("no saved voice {name:?}"))?;
        v[key] = value;
        let f = self.dir.join(name).join("voice.json");
        std::fs::write(&f, serde_json::to_string_pretty(&v).unwrap_or_default()).map_err(|e| format!("{}: {e}", f.display()))?;
        Ok(v)
    }

    pub fn remove(&self, name: &str) -> Result<(), String> {
        check_name(name)?;
        let d = self.dir.join(name);
        if !d.join("voice.json").exists() {
            return Err(format!("no saved voice {name:?}"));
        }
        std::fs::remove_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))
    }
}

/// What a designed voice reads for its sample when no text is given (about ten seconds)
pub fn design_text(language: Option<&str>) -> Option<&'static str> {
    match language.map(str::to_lowercase).as_deref() {
        None | Some("" | "auto" | "english") => Some(
            "Hello there. This is how I sound when I read aloud: a short note, a long story, or anything in between. Thank you for listening."),
        Some("chinese") => Some("你好。这是我朗读时的声音：一张便条，一个长长的故事，或者介于两者之间的任何内容。谢谢你的聆听。"),
        Some("german") => Some("Hallo. So klinge ich, wenn ich vorlese: eine kurze Notiz, eine lange Geschichte oder alles dazwischen. Danke fürs Zuhören."),
        Some("french") => Some("Bonjour. Voici ma voix quand je lis à voix haute : une courte note, une longue histoire, ou tout ce qui se trouve entre les deux. Merci de m'écouter."),
        Some("spanish") => Some("Hola. Así sueno cuando leo en voz alta: una nota breve, una historia larga o cualquier cosa intermedia. Gracias por escuchar."),
        Some("japanese") => Some("こんにちは。これが私の朗読の声です。短いメモでも、長い物語でも、その間の何でも読みます。聞いてくれてありがとう。"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_a_round_trip() {
        assert!(check_name("narrator-2").is_ok());
        for bad in ["", "Narrator", "2fast", "a b", "../x", &"x".repeat(41)] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
        let dir = std::env::temp_dir().join(format!("nsvoices-{}", std::process::id()));
        let lib = Library::of(&dir);
        let wav = Audio { rate: 16000, channels: 1, samples: (0..32000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect() }.wav(&[]);
        let r = lib.add("anna", &wav, Some(" Hello. "), None, Some("english"), "clone", "test", false).unwrap();
        assert_eq!(r["text"], "Hello.");
        assert_eq!(lib.tag("anna", "owner", json!("film-app")).unwrap()["owner"], "film-app");
        assert_eq!(lib.get("anna").unwrap()["owner"], "film-app");
        assert!(lib.add("anna", &wav, None, None, None, "clone", "test", false).is_err());
        assert_eq!(lib.list().len(), 1);
        // an enrolment is listed, and goes when the recording is replaced
        let e = lib.enrolled_file("anna", "voxtral-tts").unwrap();
        assert!(lib.enrolled_path("anna", "voxtral-tts").is_none());
        std::fs::create_dir_all(e.parent().unwrap()).unwrap();
        std::fs::write(&e, b"x").unwrap();
        assert_eq!(lib.enrolled("anna"), vec!["voxtral-tts".to_string()]);
        assert_eq!(lib.list()[0]["enrolled"], json!(["voxtral-tts"]));
        assert!(lib.enrolled_file("anna", "../x").is_none());
        lib.add("anna", &wav, Some("Hello."), None, Some("english"), "clone", "test", true).unwrap();
        assert!(lib.enrolled("anna").is_empty());
        let rf = lib.reference("anna").unwrap();
        assert_eq!((rf.rate, rf.samples.len(), rf.text.as_deref()), (16000, 32000, Some("Hello.")));
        assert!(lib.add("short", &wav[..1000], None, None, None, "clone", "test", false).is_err());
        lib.remove("anna").unwrap();
        assert!(lib.list().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
