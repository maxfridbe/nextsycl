//! The registry and the settings as the front door sees them: through `Store`, which the program implements (glue
//! depends on foundation and contracts only: CONTRIBUTING.md) - the same calls as nextsycl-models' registry.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// The model registry and the settings file
pub trait Store: Send + Sync {
    /// every entry
    fn models(&self) -> Result<Vec<Value>, String>;
    /// change an entry (the studio's chat modes kept in step): the entry as saved
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String>;
    /// the registry's file
    fn registry_file(&self) -> PathBuf;
    /// a setting (the environment over the files)
    fn get(&self, name: &str) -> Option<String>;
    /// whether the environment sets it
    fn in_env(&self, name: &str) -> bool;
    /// the settings file the page edits
    fn file(&self) -> PathBuf;
    /// an entry's prompt guide (Markdown), from the model cards
    fn guide(&self, entry: &Value) -> Option<String>;
    /// the model cards (each as JSON; none: no cards to read) - what an entry registered before a card key existed
    /// lacks (`headroom_gib`)
    fn cards(&self) -> Vec<Value> {
        Vec::new()
    }
    /// a card's files as `nextsycl models pull` lays them out: (path under the models folder, bytes); empty: no such
    /// card (pulls.rs: what a download is and how far it is)
    fn card_files(&self, _id: &str) -> Vec<(String, u64)> {
        Vec::new()
    }
}

impl<T: Store + ?Sized> Store for Box<T> {
    fn models(&self) -> Result<Vec<Value>, String> {
        (**self).models()
    }
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String> {
        (**self).update(id, f)
    }
    fn registry_file(&self) -> PathBuf {
        (**self).registry_file()
    }
    fn get(&self, name: &str) -> Option<String> {
        (**self).get(name)
    }
    fn in_env(&self, name: &str) -> bool {
        (**self).in_env(name)
    }
    fn file(&self) -> PathBuf {
        (**self).file()
    }
    fn guide(&self, entry: &Value) -> Option<String> {
        (**self).guide(entry)
    }
    fn cards(&self) -> Vec<Value> {
        (**self).cards()
    }
    fn card_files(&self, id: &str) -> Vec<(String, u64)> {
        (**self).card_files(id)
    }
}

/// An entry's kind: "llm", "image", "video", "audio", "model3d" or "lora" (entries from before kinds are llm)
pub fn kind_of(m: &Value) -> &str {
    m.get("kind").and_then(|v| v.as_str()).unwrap_or("llm")
}

pub fn all(s: &dyn Store) -> Result<Vec<Value>, String> {
    s.models()
}

pub fn find(s: &dyn Store, id: &str) -> Result<Option<Value>, String> {
    Ok(s.models()?.into_iter().find(|m| m["id"] == id))
}

pub fn update(s: &dyn Store, id: &str, mut f: impl FnMut(&mut Value)) -> Result<Value, String> {
    s.update(id, &mut f)
}

pub fn registry(s: &dyn Store) -> PathBuf {
    s.registry_file()
}

/// The cards a `gpus` value names: "0", "0,1" (one copy over both), "0|1" (one of them, the first preferred); "all"
/// and anything else name none here (the caller knows how many cards there are)
pub fn gpu_list(g: &str) -> Vec<usize> {
    g.split([',', ' ', '|']).filter_map(|x| x.trim().parse().ok()).collect()
}

/// Whether an entry's cards are a choice of one (`gpus: '0|1'`): a chat model that fits a card loads whole on
/// whichever of them has the room, the first preferred - not spread over them as '0,1' is
pub fn one_of(m: &Value) -> bool {
    m.get("gpus").and_then(Value::as_str).is_some_and(|g| g.contains('|'))
}

/// The card the front door chose for a one-of chat model, handed to `llm start` beside the registry `file` (the
/// studio's start command names the model only)
pub fn chat_gpu_file(registry_file: &Path) -> PathBuf {
    registry_file.with_file_name("chat-gpu.json")
}

/// The front door's choice of card for chat model `id`, written for `llm start`
pub fn write_chat_gpu(s: &dyn Store, id: &str, gpu: usize) -> Result<(), String> {
    let f = chat_gpu_file(&s.registry_file());
    std::fs::write(&f, serde_json::json!({"id": id, "gpu": gpu, "at": unix_s()}).to_string()).map_err(|e| format!("{}: {e}", f.display()))
}

/// The card the hand-off beside `registry_file` names for `id`, when it is that model's and fresh (written in the
/// last 10 minutes: a start follows the choice in seconds; an older one was a choice for another time)
pub fn chat_gpu_in(registry_file: &Path, id: &str) -> Option<usize> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(chat_gpu_file(registry_file)).ok()?).ok()?;
    chat_gpu_of(&v, id, unix_s())
}

/// `chat_gpu_in` for the front door's registry
pub fn chat_gpu(s: &dyn Store, id: &str) -> Option<usize> {
    chat_gpu_in(&s.registry_file(), id)
}

fn chat_gpu_of(v: &Value, id: &str, now: u64) -> Option<usize> {
    let at = v["at"].as_u64()?;
    if v["id"] != id || now.saturating_sub(at) >= 600 {
        return None;
    }
    v["gpu"].as_u64().map(|g| g as usize)
}

fn unix_s() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The host paths an entry names: its file, its files by role, settings that are paths (`path`, `path:scale`)
pub fn paths(m: &Value) -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from(m.get("file").and_then(Value::as_str).unwrap_or(""))];
    v.extend(m["files"].as_object().into_iter().flatten().filter_map(|(_, p)| p.as_str()).map(PathBuf::from));
    for val in m.get("env").and_then(Value::as_object).into_iter().flatten().filter_map(|(_, v)| v.as_str()) {
        for part in val.split(',') {
            let p = match part.rsplit_once(':') {
                Some((p, scale)) if scale.parse::<f64>().is_ok() => p,
                _ => part,
            };
            if p.starts_with('/') && Path::new(p).exists() {
                v.push(PathBuf::from(p));
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::{chat_gpu_of, gpu_list, one_of};
    use serde_json::json;

    #[test]
    fn a_choice_of_cards_reads_apart_from_a_span() {
        assert_eq!(gpu_list("0|1"), vec![0, 1]);
        assert_eq!(gpu_list("1|0"), vec![1, 0], "the order is the preference");
        assert_eq!(gpu_list("0,1"), vec![0, 1]);
        assert_eq!(gpu_list("all"), Vec::<usize>::new());
        assert!(one_of(&json!({"gpus": "0|1"})));
        assert!(!one_of(&json!({"gpus": "0,1"})) && !one_of(&json!({"gpus": "all"})) && !one_of(&json!({})));
    }

    #[test]
    fn llm_start_takes_the_front_doors_card_only_when_it_is_that_models_and_fresh() {
        let v = json!({"id": "swift", "gpu": 1, "at": 1000});
        assert_eq!(chat_gpu_of(&v, "swift", 1030), Some(1));
        assert_eq!(chat_gpu_of(&v, "other", 1030), None, "another model's choice");
        assert_eq!(chat_gpu_of(&v, "swift", 1000 + 600), None, "ten minutes old");
        assert_eq!(chat_gpu_of(&json!({"id": "swift", "at": 1000}), "swift", 1001), None);
    }
}
