//! The catalog of supported models (`catalog.json`, built into the program; `NS_CATALOG` names another): every file a
//! model needs as a direct link pinned to a revision, with its size and SHA-256, and the settings its registry entry
//! gets. `nextsycl models search` lists it; `nextsycl models pull <id>` downloads what is missing, checks it, and
//! registers the model.
//!
//! Files are keyed once and shared by id (two models naming the same file download it once); a file with the same
//! content already on the disk under another name (the Coder's shard 2 is the original's) is hard-linked, not
//! fetched again. Some parts are made locally rather than downloaded (Qwen3.8-Flash-Next's MTP draft layer is built
//! by Strata's tools): a model lists them under "local", and `pull` uses one it finds (the setting in nextsycl.conf,
//! or another registry entry's) or registers the model without it.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;

const BUILT_IN: &str = include_str!("../catalog.json");

/// The catalog: its files by key, its models
pub struct Catalog {
    pub files: Map<String, Value>,
    pub models: Vec<Value>,
}

impl Catalog {
    /// `NS_CATALOG` (a file), else the one built in
    pub fn load(cfg: &Config) -> Result<Catalog, String> {
        let text = match cfg.get("NS_CATALOG") {
            Some(p) => std::fs::read_to_string(&p).map_err(|e| format!("{p}: {e}"))?,
            None => BUILT_IN.to_string(),
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("the catalog: {e}"))?;
        Ok(Catalog { files: v["files"].as_object().cloned().unwrap_or_default(), models: v["models"].as_array().cloned().unwrap_or_default() })
    }

    pub fn model(&self, id: &str) -> Option<&Value> {
        self.models.iter().find(|m| m["id"] == id)
    }

    /// A model's files: (role, key, the file's entry)
    fn files_of<'a>(&'a self, m: &'a Value) -> Vec<(&'a str, &'a str, &'a Value)> {
        m["files"].as_object().map_or_else(Vec::new, |o| {
            o.iter().filter_map(|(role, key)| {
                let k = key.as_str()?;
                Some((role.as_str(), k, self.files.get(k)?))
            }).collect()
        })
    }

    /// Bytes to download for a model (each distinct file once)
    pub fn bytes(&self, m: &Value) -> u64 {
        let keys: BTreeSet<&str> = self.files_of(m).iter().map(|f| f.1).collect();
        keys.iter().filter_map(|k| self.files.get(*k)?["size"].as_u64()).sum()
    }
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// `nextsycl models search [TEXT] [--kind K]`
pub fn search(cfg: &Config, installed: &[Value], text: Option<&str>, kind: Option<&str>) -> Result<(), String> {
    let cat = Catalog::load(cfg)?;
    let want = text.map(str::to_lowercase);
    println!("{:<44} {:<6} {:>8}  {:<6} {:<10} TITLE", "ID", "KIND", "SIZE", "GPUS", "INSTALLED");
    for m in &cat.models {
        let id = s(m, "id");
        if kind.is_some_and(|k| s(m, "kind") != k) {
            continue;
        }
        if let Some(w) = &want {
            let hay = format!("{id} {} {}", s(m, "title"), s(m, "about")).to_lowercase();
            if !hay.contains(w.as_str()) {
                continue;
            }
        }
        let have = if installed.iter().any(|x| x["id"] == id.as_str()) { "yes" } else { "-" };
        println!("{:<44} {:<6} {:>7.1}G  {:<6} {:<10} {}", id, s(m, "kind"), gib(cat.bytes(m)), s(m, "gpus"), have, s(m, "title"));
    }
    Ok(())
}

/// A file's SHA-256, hex, reading it once
fn sha256_of(p: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", p.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `name` of `size` bytes somewhere under `dir` (a copy the machine already has)
fn find_copy(dir: &Path, name: &str, size: u64, depth: usize) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        let Ok(md) = std::fs::metadata(&p) else { continue };
        if md.is_dir() {
            subdirs.push(p);
        } else if p.file_name().is_some_and(|n| n == name) && md.len() == size {
            return Some(p);
        }
    }
    if depth == 0 {
        return None;
    }
    subdirs.iter().find_map(|d| find_copy(d, name, size, depth - 1))
}

/// One file in place at `target`: there already (the right size; `verify` hashes it too), linked from a file with the
/// same content (another catalog path under `root`, or a copy found under `from`), or downloaded and checked
fn fetch(cat: &Catalog, root: &Path, key: &str, f: &Value, from: &[PathBuf], verify: bool) -> Result<PathBuf, String> {
    let target = root.join(s(f, "path"));
    let size = f["size"].as_u64().unwrap_or(0);
    let sha = s(f, "sha256");
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if std::fs::metadata(&target).is_ok_and(|m| m.len() == size) {
        if verify && sha256_of(&target)? != sha {
            return Err(format!("{}: not the file the catalog names (SHA-256 differs)", target.display()));
        }
        println!("  {key}: have it ({})", target.display());
        return Ok(target);
    }
    if let Some(d) = target.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    // the same content under another catalog path, or a copy elsewhere on the machine (by name and size)
    let same = cat.files.iter().filter(|(k, o)| *k != key && s(o, "sha256") == sha).map(|(_, o)| root.join(s(o, "path")))
        .find(|p| std::fs::metadata(p).is_ok_and(|m| m.len() == size));
    let copy = same.or_else(|| from.iter().find_map(|d| find_copy(d, &name, size, 4)));
    if let Some(src) = copy {
        if verify && sha256_of(&src)? != sha {
            return Err(format!("{}: not the file the catalog names (SHA-256 differs)", src.display()));
        }
        let _ = std::fs::remove_file(&target);
        if std::fs::hard_link(&src, &target).is_err() {
            std::os::unix::fs::symlink(&src, &target).map_err(|e| format!("{}: {e}", target.display()))?;
        }
        println!("  {key}: linked from {}", src.display());
        return Ok(target);
    }
    let part = target.with_file_name(format!("{name}.part"));
    let url = s(f, "url");
    println!("  {key}: downloading {:.2} GiB\n    {url}", gib(size));
    let st = Command::new("curl").args(["-fL", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o"]).arg(&part).arg(&url).status()
        .map_err(|e| format!("curl: {e}"))?;
    if !st.success() {
        return Err(format!("{url}: the download failed (run pull again: it resumes)"));
    }
    let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if got != size {
        return Err(format!("{}: {got} bytes, the catalog says {size}", part.display()));
    }
    print!("    checking its SHA-256 ... ");
    let h = sha256_of(&part)?;
    if h != sha {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{url}: SHA-256 {h}, the catalog says {sha} (the partial file is removed: pull again)"));
    }
    println!("ok");
    std::fs::rename(&part, &target).map_err(|e| format!("{}: {e}", target.display()))?;
    Ok(target)
}

/// The registry entry a pulled model gets. `local`: the made-locally parts found (role -> path)
fn entry(m: &Value, paths: &BTreeMap<String, PathBuf>, local: &BTreeMap<String, String>) -> Value {
    let sub = |v: &str| -> Option<String> {
        let mut out = v.to_string();
        while let Some(a) = out.find('{') {
            let b = out[a..].find('}')? + a;
            let role = &out[a + 1..b];
            let p = paths.get(role).map(|p| p.to_string_lossy().into_owned()).or_else(|| local.get(role).cloned())?;
            out.replace_range(a..=b, &p);
        }
        Some(out)
    };
    let mut env = Map::new();
    for (k, v) in m["env"].as_object().into_iter().flatten() {
        // a setting naming a part that is not here (a local one not found) is left out
        if let Some(x) = v.as_str().and_then(sub) {
            env.insert(k.clone(), Value::String(x));
        }
    }
    let main = ["model", "transformer", "denoiser", "lora"].iter().find_map(|r| paths.get(*r)).or_else(|| paths.values().next());
    let files: Map<String, Value> = paths.iter().map(|(r, p)| (r.clone(), Value::String(p.to_string_lossy().into_owned()))).collect();
    let mut e = json!({
        "id": m["id"], "kind": m["kind"], "arch": m["arch"], "title": m["title"],
        "file": main.map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
        "files": files, "gpus": m.get("gpus").cloned().unwrap_or(json!("0")), "env": env, "enabled": true,
        "catalog": m["id"],
    });
    for k in ["ctx", "tools", "tasks", "base", "defaults"] {
        if let Some(v) = m.get(k) {
            e[k] = v.clone();
        }
    }
    e
}

/// A made-locally part's path: `NS_<ROLE>` style settings in nextsycl.conf are the engine's own (the MTP layer:
/// NS_QW_MTP), else what another registry entry uses for the same setting
fn find_local(cfg: &Config, installed: &[Value], m: &Value, role: &str) -> Option<String> {
    let setting = m["env"].as_object()?.iter().find(|(_, v)| v.as_str() == Some(&format!("{{{role}}}")))?.0.clone();
    if let Some(v) = cfg.get(&setting).filter(|v| Path::new(v).exists()) {
        return Some(v);
    }
    installed.iter().filter_map(|x| x["env"][&setting].as_str()).find(|p| Path::new(p).exists()).map(str::to_string)
}

/// `nextsycl models pull <id>... [--dir DIR] [--from DIR]... [--verify]`: each model's missing files downloaded (or
/// linked), checked, and the model registered. Returns the entries to add.
pub fn pull(cfg: &Config, installed: &[Value], ids: &[String], raw: &[String]) -> Result<Vec<Value>, String> {
    let cat = Catalog::load(cfg)?;
    let opt = |k: &str| raw.iter().position(|a| a == k).and_then(|i| raw.get(i + 1)).cloned();
    let root = PathBuf::from(opt("--dir").or_else(|| cfg.get("NS_MODELS")).ok_or("set NS_MODELS (where models go) or pass --dir DIR")?);
    let from: Vec<PathBuf> = raw.iter().enumerate().filter(|(_, a)| *a == "--from").filter_map(|(i, _)| raw.get(i + 1)).map(PathBuf::from).collect();
    let verify = raw.iter().any(|a| a == "--verify");
    let mut out = Vec::new();
    for id in ids {
        let m = cat.model(id).ok_or_else(|| format!("{id}: not in the catalog (nextsycl models search)"))?;
        if installed.iter().any(|x| x["id"] == id.as_str()) && !raw.iter().any(|a| a == "--again") {
            println!("{id}: installed already (nextsycl models list; --again pulls it anew)");
            continue;
        }
        println!("{id}: {} ({:.1} GiB) into {}", s(m, "title"), gib(cat.bytes(m)), root.display());
        let mut paths = BTreeMap::new();
        for (role, key, f) in cat.files_of(m) {
            paths.insert(role.to_string(), fetch(&cat, &root, key, f, &from, verify)?);
        }
        let mut local = BTreeMap::new();
        for (role, about) in m["local"].as_object().into_iter().flatten() {
            match find_local(cfg, installed, m, role) {
                Some(p) => {
                    println!("  {role}: {p}");
                    local.insert(role.clone(), p);
                }
                None => println!("  {role}: not found - {}", about.as_str().unwrap_or("")),
            }
        }
        out.push(entry(m, &paths, &local));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_catalog_is_whole() {
        let v: Value = serde_json::from_str(BUILT_IN).unwrap();
        let files = v["files"].as_object().unwrap();
        for (k, f) in files {
            assert!(s(f, "url").starts_with("https://"), "{k}: a direct link");
            assert_eq!(s(f, "sha256").len(), 64, "{k}: a SHA-256");
            assert!(f["size"].as_u64().unwrap_or(0) > 0, "{k}: a size");
            assert!(!s(f, "path").is_empty() && !s(f, "path").starts_with('/'), "{k}: a relative path");
        }
        let mut ids = BTreeSet::new();
        for m in v["models"].as_array().unwrap() {
            let id = s(m, "id");
            assert!(ids.insert(id.clone()), "{id} twice");
            assert!(["llm", "image", "video", "audio", "lora"].contains(&s(m, "kind").as_str()), "{id}: a kind");
            for (role, key) in m["files"].as_object().unwrap() {
                assert!(files.contains_key(key.as_str().unwrap()), "{id}: {role} names no file");
            }
            // every {role} in a setting is a file or a local part of the model
            for v in m["env"].as_object().into_iter().flatten().filter_map(|(_, v)| v.as_str()) {
                if let (Some(a), Some(b)) = (v.find('{'), v.find('}')) {
                    let role = &v[a + 1..b];
                    assert!(m["files"].get(role).is_some() || m["local"].get(role).is_some(), "{id}: {{{role}}}");
                }
            }
        }
    }

    #[test]
    fn an_entry_takes_the_files_and_drops_missing_local_parts() {
        let m = json!({"id": "x", "kind": "llm", "arch": "a", "title": "X", "gpus": "0", "ctx": "4096",
                       "env": {"A": "{model}", "B": "{mtp}", "C": "plain"}});
        let paths = BTreeMap::from([("model".to_string(), PathBuf::from("/m/x.gguf"))]);
        let e = entry(&m, &paths, &BTreeMap::new());
        assert_eq!(e["file"], "/m/x.gguf");
        assert_eq!(e["env"], json!({"A": "/m/x.gguf", "C": "plain"}));
        let e = entry(&m, &paths, &BTreeMap::from([("mtp".to_string(), "/d/rt".to_string())]));
        assert_eq!(e["env"]["B"], "/d/rt");
    }
}
