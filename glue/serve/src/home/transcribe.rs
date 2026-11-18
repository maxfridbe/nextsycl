//! Saved voices heard once: a voice saved without what it says (a recording added with no text: the film app's,
//! the audio page's, `nextsycl audio voice add`), or with a text but never heard, is transcribed by the enabled
//! recogniser (Whisper large-v3-turbo, word times on) and the transcript kept with it (`voices::Library::transcribed`),
//! so a model that clones in context (Sophon-Realtime's Base talker) has a transcript, and the words' times to cut
//! the recording by, done once.
//!
//! ```text
//!   audio.voices.add without text  -> heard now (a thread of its own)
//!   the front door's start (+60 s), then every 10 min -> every voice that needs it (voices::needs_transcript)
//!   each: an audio.transcribe job (label "voice transcript: NAME", priority -5, words on), one at a time; the
//!         record's text filled when it had none, its words' times and who heard it kept
//! ```
//!
//! Here, not in the engines: a speech engine runs in the audio server, which has no recogniser and does not call the
//! front door; the front door holds the backplane (the recogniser loads where there is room, as any job) and owns
//! the voice library's writes. NS_VOICE_TRANSCRIBE=0 leaves the voices as they are. A voice that failed is tried
//! again after six hours (a voice replaced since is a new one).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use serde_json::json;

use super::life::s;
use super::{now, Home};
use crate::voices::needs_transcript;

/// One pass at a time (the add's and the timer's)
static RUNNING: AtomicBool = AtomicBool::new(false);
/// voice (dir, name, created) -> when it last failed
fn failed() -> &'static Mutex<HashMap<String, f64>> {
    static F: OnceLock<Mutex<HashMap<String, f64>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(HashMap::new()))
}
const AGAIN_S: f64 = 6.0 * 3600.0;
const EVERY_S: u64 = 600;
const FIRST_S: u64 = 60;

struct Pass;
impl Drop for Pass {
    fn drop(&mut self) {
        RUNNING.store(false, Ordering::SeqCst);
    }
}

impl Home {
    fn transcribing(&self) -> bool {
        self.o.cfg.get("NS_VOICE_TRANSCRIBE").is_none_or(|v| v.trim() != "0")
    }

    /// The timer: a pass a minute after the start, then every ten minutes
    pub(super) fn voice_ears_loop(self: Arc<Self>) {
        std::thread::sleep(Duration::from_secs(FIRST_S));
        loop {
            self.hear_voices();
            std::thread::sleep(Duration::from_secs(EVERY_S));
        }
    }

    /// A pass now, on its own thread (a voice was just saved)
    pub(super) fn hear_voices_soon(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || me.hear_voices());
    }

    /// Every saved voice that needs it, heard by the recogniser one at a time; how many were kept
    pub(super) fn hear_voices(self: &Arc<Self>) -> usize {
        if !self.transcribing() || RUNNING.swap(true, Ordering::SeqCst) {
            return 0;
        }
        let _pass = Pass;
        let lib = self.voices();
        let mut kept = 0;
        for v in lib.list() {
            if !needs_transcript(&v) {
                continue;
            }
            let name = s(&v, "name");
            let key = format!("{}/{name}/{}", lib.dir.display(), v["created"]);
            if failed().lock().unwrap().get(&key).is_some_and(|t| now() - t < AGAIN_S) {
                continue;
            }
            // a recogniser enabled (none: nothing to do until one is)
            let all = self.models();
            let Ok(id) = self.recogniser(&json!({}), &all) else { return kept };
            let Some(bytes) = lib.sample_path(&name).and_then(|p| std::fs::read(p).ok()) else { continue };
            let mut b = json!({"audio": crate::http::base64(&bytes), "words": true, "model": id, "label": format!("voice transcript: {name}"),
                               "priority": -5});
            if let Some(l) = v["language"].as_str().filter(|l| !l.trim().is_empty()) {
                b["language"] = json!(l);
            }
            let r = self.submit("audio.transcribe", &b).map_err(|(st, m)| format!("{st}: {m}")).and_then(|r| {
                let words = r["words"].as_array().cloned().unwrap_or_default();
                lib.transcribed(&name, &v["created"], r["text"].as_str().unwrap_or(""), &words, &id)
            });
            match r {
                Ok(rec) => {
                    kept += 1;
                    eprintln!("voices: {name} heard by {id} ({} words): {:?}", rec["words"].as_array().map_or(0, |w| w.len()),
                              rec["transcribed"]["text"].as_str().unwrap_or("").chars().take(80).collect::<String>());
                }
                Err(e) => {
                    failed().lock().unwrap().insert(key, now());
                    eprintln!("voices: {name} not heard ({e}); again in {} h", AGAIN_S / 3600.0);
                }
            }
        }
        kept
    }
}

#[cfg(test)]
mod tests {
    use crate::voices::needs_transcript;
    use serde_json::json;

    #[test]
    fn which_voices_are_heard() {
        assert!(needs_transcript(&json!({"name": "a", "seconds": 9.0})));
        assert!(needs_transcript(&json!({"name": "a", "seconds": 9.0, "text": "  "})));
        assert!(needs_transcript(&json!({"name": "a", "seconds": 9.0, "text": "Hello."})));
        assert!(!needs_transcript(&json!({"name": "a", "seconds": 9.0, "text": "Hello.", "words": [{"start": 0.0, "end": 0.4, "text": "Hello."}]})));
        assert!(needs_transcript(&json!({"name": "a", "seconds": 38.8, "text": "Hello."})));
        assert!(!needs_transcript(&json!({"name": "a", "seconds": 38.8, "text": "Hello.", "words": [{"start": 0.0, "end": 0.4, "text": "Hello."}]})));
    }
}
