//! Settings: the environment, then `nextsycl.conf` beside the repository (one level above dist/), then
//! `~/.config/nextsycl.conf`; the first that has a name wins. The files are `NAME=value` lines (`#` comments, quotes
//! around the value optional) - the same names as the environment variables (sycl-h3's scheme).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct Config {
    files: BTreeMap<String, String>,
    /// dist/: where this program and the kernel library live
    pub dist: PathBuf,
}

fn parse(text: &str, into: &mut BTreeMap<String, String>) {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            let v = v.trim();
            let v = v.strip_prefix('"').and_then(|v| v.strip_suffix('"')).or_else(|| v.strip_prefix('\'').and_then(|v| v.strip_suffix('\''))).unwrap_or(v);
            into.entry(k.trim().to_string()).or_insert_with(|| v.to_string());
        }
    }
}

impl Config {
    pub fn load() -> Config {
        let exe = std::env::current_exe().ok().and_then(|p| p.canonicalize().ok()).unwrap_or_default();
        let dist = std::env::var("NS_DIST").map(PathBuf::from).unwrap_or_else(|_| exe.parent().map(Path::to_path_buf).unwrap_or_default());
        let mut files = BTreeMap::new();
        let mut candidates = vec![dist.join("../nextsycl.conf")];
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".config/nextsycl.conf"));
        }
        for f in candidates {
            if let Ok(t) = std::fs::read_to_string(&f) {
                parse(&t, &mut files);
            }
        }
        Config { files, dist }
    }

    pub fn get(&self, name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty()).or_else(|| self.files.get(name).cloned().filter(|v| !v.is_empty()))
    }

    /// Every setting whose name starts with `prefix` (the environment's over the files'), as (name, value)
    pub fn with_prefix(&self, prefix: &str) -> Vec<(String, String)> {
        let mut m: BTreeMap<String, String> = self.files.iter().filter(|(k, v)| k.starts_with(prefix) && !v.is_empty()).map(|(k, v)| (k.clone(), v.clone())).collect();
        for (k, v) in std::env::vars() {
            if k.starts_with(prefix) && !v.is_empty() {
                m.insert(k, v);
            }
        }
        m.into_iter().collect()
    }

    pub fn or(&self, name: &str, default: &str) -> String {
        self.get(name).unwrap_or_else(|| default.to_string())
    }

    /// The directory holding the server's control socket, on the host.
    pub fn socket_dir(&self) -> PathBuf {
        if let Some(d) = self.get("NS_SOCKET_DIR") {
            return PathBuf::from(d);
        }
        let base = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(base).join("nextsycl")
    }

    pub fn socket(&self) -> PathBuf {
        self.socket_dir().join("nextsycl.sock")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conf_lines() {
        let mut m = BTreeMap::new();
        parse("# a comment\nNS_MODELS=/m\nexport NS_PORT=\"9000\"\n  NS_CTX = '4096'\nNS_MODELS=/later\n", &mut m);
        assert_eq!(m["NS_MODELS"], "/m");
        assert_eq!(m["NS_PORT"], "9000");
        assert_eq!(m["NS_CTX"], "4096");
    }
}
