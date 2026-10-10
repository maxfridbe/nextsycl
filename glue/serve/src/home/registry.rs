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
}

/// An entry's kind: "llm", "image", "video", "audio" or "lora" (entries from before kinds are llm)
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
