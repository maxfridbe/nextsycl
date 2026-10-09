//! `nextsycl models`: the models this machine serves, in one registry (NS_REGISTRY, a JSON file; default
//! ~/.config/nextsycl/models.json). An entry is what `nextsycl llm start <id>` runs: its GGUF file (the first shard), its
//! GPUs and session contexts, its engine settings (NS_QW_* and the like), whether it is offered (enabled), and what a
//! client may send it (tools, background tasks). Its id is the model id clients see and the name it is served under.
//!
//! The registry is the one place models are added: the Open WebUI switcher lists its enabled entries, and with
//! NS_STUDIO_MODES (the H3 studio's llm-modes file) every change rewrites that file's nextsycl entries (one mode a
//! model, `nextsycl llm start <id>`; the studio reads the file when it starts).
//!
//! ```text
//!   nextsycl models list [--json]
//!   nextsycl models add <id> <file.gguf> [--title T] [--gpu 0[,1] | all] [--ctx N[,M...]] [--set NAME=VALUE]...
//!                       [--no-tools] [--no-tasks] [--disabled]
//!   nextsycl models search [TEXT] [--kind llm|image|video|audio|lora]     the catalog of supported models
//!   nextsycl models pull <id>... [--dir DIR] [--from DIR]... [--verify] [--again]
//!                                           a catalog model's files downloaded (or linked), checked, registered
//!   nextsycl models pull <id> <url | hf:org/repo/path/file.gguf> [--dir DIR] [add's options]   a file outside it
//!   nextsycl models remove <id> [--files]      (--files: the model's GGUF shards are deleted too)
//!   nextsycl models enable <id> | disable <id>
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};

use crate::config::Config;

/// What the program can serve from a file: the engine's name for it (`Err` when no engine of this build takes it).
/// The program passes its engines' check, so this crate depends on no engine.
pub type Describe<'a> = &'a dyn Fn(&nextsycl_gguf::Gguf) -> Result<String, String>;

/// An entry's kind: "llm", "image", "video", "audio" or "lora" (entries from before kinds are llm)
pub fn kind_of(m: &Value) -> &str {
    m.get("kind").and_then(|v| v.as_str()).unwrap_or("llm")
}

/// The registry's file
pub fn registry(cfg: &Config) -> PathBuf {
    cfg.get("NS_REGISTRY").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())).join(".config/nextsycl/models.json")
    })
}

fn load(path: &Path) -> Result<Vec<Value>, String> {
    match std::fs::read_to_string(path) {
        Ok(t) => {
            let v: Value = serde_json::from_str(&t).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(v.get("models").and_then(Value::as_array).cloned().unwrap_or_default())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

fn save(cfg: &Config, path: &Path, models: &[Value]) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(&json!({ "models": models })).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, text + "\n").map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    sync_modes(cfg, models)
}

/// Every entry
pub fn all(cfg: &Config) -> Result<Vec<Value>, String> {
    load(&registry(cfg))
}

/// The entry `id`
pub fn find(cfg: &Config, id: &str) -> Result<Option<Value>, String> {
    Ok(load(&registry(cfg))?.into_iter().find(|m| m["id"] == id))
}

fn s(m: &Value, k: &str) -> String {
    m.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

/// A model file's shards (a split GGUF names them -0000N-of-0000M), the first one first
fn shards(first: &Path) -> Vec<PathBuf> {
    let name = first.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some(i) = name.find("-00001-of-") {
        let of = &name[i + 10..i + 15];
        if let Ok(n) = of.parse::<usize>() {
            return (1..=n).map(|k| first.with_file_name(format!("{}-{k:05}-of-{of}{}", &name[..i], &name[i + 15..]))).collect();
        }
    }
    vec![first.to_path_buf()]
}

/// The host paths an entry names (its file, settings that are paths: `path`, `path:scale`, comma lists): what the
/// container must see, at the same paths
pub fn paths(m: &Value) -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from(s(m, "file"))];
    // an entry of several files (pulled from the catalog: by role)
    v.extend(m["files"].as_object().into_iter().flatten().filter_map(|(_, p)| p.as_str()).map(PathBuf::from));
    if let Some(env) = m.get("env").and_then(Value::as_object) {
        for val in env.values().filter_map(Value::as_str) {
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
    }
    v
}

/// Every file of an entry: its file's shards, and the files it lists by role
fn files_of(m: &Value) -> Vec<PathBuf> {
    let mut v = shards(Path::new(&s(m, "file")));
    for p in m["files"].as_object().into_iter().flatten().filter_map(|(_, p)| p.as_str()).map(PathBuf::from) {
        if !v.contains(&p) {
            v.push(p);
        }
    }
    v
}

fn bytes_of(m: &Value) -> u64 {
    files_of(m).iter().filter_map(|p| std::fs::metadata(p).ok()).map(|x| x.len()).sum()
}

/// `nextsycl models ...`
pub fn cmd(cfg: &Config, raw: &[String], describe: Describe) -> Result<(), String> {
    let path = registry(cfg);
    let mut models = load(&path)?;
    let opt = |k: &str| raw.iter().position(|a| a == k).and_then(|i| raw.get(i + 1)).cloned();
    let flag = |k: &str| raw.iter().any(|a| a == k);
    let pos: Vec<&String> = {
        // the positional arguments: those that are not an option or an option's value
        let with_value = ["--title", "--gpu", "--ctx", "--set", "--dir", "--from", "--kind"];
        let mut v = Vec::new();
        let mut i = 1;
        while i < raw.len() {
            if with_value.contains(&raw[i].as_str()) {
                i += 2;
                continue;
            }
            if !raw[i].starts_with("--") {
                v.push(&raw[i]);
            }
            i += 1;
        }
        v
    };
    match raw.first().map(String::as_str) {
        None | Some("list") => {
            if flag("--json") {
                println!("{}", serde_json::to_string_pretty(&json!({ "models": models })).unwrap_or_default());
                return Ok(());
            }
            if models.is_empty() {
                println!("no models yet ({}): nextsycl models add <id> <file.gguf>", path.display());
                return Ok(());
            }
            println!("{:<44} {:<6} {:<8} {:>8}  {:<8} {:<16} TITLE", "ID", "KIND", "STATE", "SIZE", "GPUS", "CONTEXT");
            for m in &models {
                let present = Path::new(&s(m, "file")).exists();
                let state = if !present { "missing" } else if m["enabled"] == false { "disabled" } else { "enabled" };
                println!("{:<44} {:<6} {:<8} {:>7.1}G  {:<8} {:<16} {}", s(m, "id"), kind_of(m), state, bytes_of(m) as f64 / (1u64 << 30) as f64, s(m, "gpus"),
                         s(m, "ctx"), s(m, "title"));
            }
            Ok(())
        }
        Some("add") => {
            let (id, file) = match (pos.first(), pos.get(1)) {
                (Some(i), Some(f)) => ((*i).clone(), (*f).clone()),
                _ => return Err("nextsycl models add <id> <file.gguf> [options]".into()),
            };
            add(cfg, &path, &mut models, &id, &file, raw, describe)?;
            println!("added {id}");
            Ok(())
        }
        Some("search") => crate::catalog::search(cfg, &models, pos.first().map(|s| s.as_str()), opt("--kind").as_deref()),
        // a catalog model (or several), or `pull <id> <url | hf:org/repo/path/file.gguf>` for a file outside it
        Some("pull") if pos.len() == 2 && (pos[1].starts_with("hf:") || pos[1].contains("://")) => {
            cmd(cfg, &[&["download".to_string()][..], &raw[1..]].concat(), describe)
        }
        Some("pull") => {
            if pos.is_empty() {
                return Err("nextsycl models pull <id>... [--dir DIR] [--from DIR]... [--verify] [--again]   (nextsycl models search)".into());
            }
            let ids: Vec<String> = pos.iter().map(|s| s.to_string()).collect();
            for e in crate::catalog::pull(cfg, &models, &ids, raw)? {
                let id = s(&e, "id");
                if kind_of(&e) == "llm" {
                    // the file is one this program serves
                    let g = nextsycl_gguf::Gguf::open(Path::new(&s(&e, "file"))).map_err(|x| x.0)?;
                    describe(&g)?;
                }
                models.retain(|m| m["id"] != id.as_str());
                models.push(e);
                save(cfg, &path, &models)?;
                println!("registered {id}");
            }
            Ok(())
        }
        Some("download") => {
            let (id, src) = match (pos.first(), pos.get(1)) {
                (Some(i), Some(u)) => ((*i).clone(), (*u).clone()),
                _ => return Err("nextsycl models download <id> <url | hf:org/repo/path/file.gguf> [--dir DIR] [options]".into()),
            };
            let url = match src.strip_prefix("hf:") {
                Some(r) => {
                    let mut it = r.splitn(3, '/');
                    let (org, repo, file) = (it.next().unwrap_or(""), it.next().unwrap_or(""), it.next().unwrap_or(""));
                    if file.is_empty() {
                        return Err("hf:org/repo/path/file.gguf".into());
                    }
                    format!("https://huggingface.co/{org}/{repo}/resolve/main/{file}")
                }
                None => src.clone(),
            };
            let dir = PathBuf::from(opt("--dir").unwrap_or_else(|| format!("{}/{id}", cfg.or("NS_MODELS", "."))));
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let name = url.rsplit('/').next().unwrap_or("model.gguf").split('?').next().unwrap_or("model.gguf").to_string();
            let first = dir.join(&name);
            // every shard of a split file (-00001-of-0000N), each resumed where a download stopped
            for target in shards(&first) {
                let fname = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let u = format!("{}/{fname}", url.rsplit_once('/').map_or("", |x| x.0));
                println!("downloading {u}\n  -> {}", target.display());
                let st = Command::new("curl").args(["-fL", "-C", "-", "--retry", "5", "-o"]).arg(&target).arg(&u).status()
                    .map_err(|e| format!("curl: {e}"))?;
                if !st.success() {
                    return Err(format!("{u}: the download failed (it resumes when run again)"));
                }
            }
            add(cfg, &path, &mut models, &id, &first.to_string_lossy(), raw, describe)?;
            println!("downloaded and added {id}");
            Ok(())
        }
        Some("remove") => {
            let id = pos.first().ok_or("nextsycl models remove <id> [--files]")?.to_string();
            let i = models.iter().position(|m| m["id"] == id.as_str()).ok_or_else(|| format!("no model {id}"))?;
            let m = models.remove(i);
            if flag("--files") {
                // a file another entry still uses stays
                let kept: Vec<PathBuf> = models.iter().flat_map(files_of).collect();
                for p in files_of(&m).into_iter().filter(|p| !kept.contains(p)) {
                    match std::fs::remove_file(&p) {
                        Ok(()) => println!("deleted {}", p.display()),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(format!("{}: {e}", p.display())),
                    }
                }
            }
            save(cfg, &path, &models)?;
            println!("removed {id}{}", if flag("--files") { " and its files" } else { " (its files stay; --files deletes them)" });
            Ok(())
        }
        Some(c @ ("enable" | "disable")) => {
            let id = pos.first().ok_or_else(|| format!("nextsycl models {c} <id>"))?.to_string();
            let m = models.iter_mut().find(|m| m["id"] == id.as_str()).ok_or_else(|| format!("no model {id}"))?;
            m["enabled"] = Value::Bool(c == "enable");
            save(cfg, &path, &models)?;
            println!("{id}: {c}d");
            Ok(())
        }
        Some(other) => Err(format!("nextsycl models {other}: list | add | download | remove | enable | disable")),
    }
}

fn add(cfg: &Config, path: &Path, models: &mut Vec<Value>, id: &str, file: &str, raw: &[String], describe: Describe) -> Result<(), String> {
    if id.is_empty() || id.contains(['/', ' ', '"']) {
        return Err(format!("{id:?}: an id is one word (it is the model id clients see)"));
    }
    let file = std::fs::canonicalize(file).map_err(|e| format!("{file}: {e}"))?;
    // the file must be one this program serves
    let g = nextsycl_gguf::Gguf::open(&file).map_err(|e| e.0)?;
    let engine = describe(&g)?;
    let opt = |k: &str| raw.iter().position(|a| a == k).and_then(|i| raw.get(i + 1)).cloned();
    let mut env = Map::new();
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == "--set" {
            let kv = raw.get(i + 1).ok_or("--set NAME=VALUE")?;
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("--set {kv}: NAME=VALUE"))?;
            env.insert(k.to_string(), Value::String(v.to_string()));
            i += 1;
        }
        i += 1;
    }
    let entry = json!({
        "id": id,
        "kind": "llm",
        "title": opt("--title").unwrap_or_else(|| format!("{} ({})", g.meta("general.name").and_then(|v| v.as_str()).unwrap_or(id), engine)),
        "file": file.to_string_lossy(),
        "arch": g.architecture(),
        "gpus": opt("--gpu").unwrap_or_else(|| "all".into()),
        "ctx": opt("--ctx").unwrap_or_else(|| "65536".into()),
        "env": env,
        "enabled": !raw.iter().any(|a| a == "--disabled"),
        "tools": !raw.iter().any(|a| a == "--no-tools"),
        "tasks": !raw.iter().any(|a| a == "--no-tasks"),
    });
    match models.iter_mut().find(|m| m["id"] == id) {
        Some(m) => *m = entry,
        None => models.push(entry),
    }
    save(cfg, path, models)
}

/// NS_STUDIO_MODES: the H3 studio's llm-modes file gets one mode a registry model (named by its id, `nextsycl start
/// <id>`), replacing the modes nextsycl wrote before; the studio's own modes stay
fn sync_modes(cfg: &Config, models: &[Value]) -> Result<(), String> {
    let Some(file) = cfg.get("NS_STUDIO_MODES") else { return Ok(()) };
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
    let mut d: Value = serde_json::from_str(&text).map_err(|e| format!("{file}: {e}"))?;
    let exe = cfg.dist.join("nextsycl");
    let ids: Vec<String> = models.iter().map(|m| s(m, "id")).collect();
    let mut modes: Vec<Value> = d["modes"].as_array().cloned().unwrap_or_default();
    modes.retain(|m| m["managed_by"] != "nextsycl" && !ids.contains(&s(m, "name")));
    // the studio's chat modes: the language models
    for m in models.iter().filter(|m| m["enabled"] != false && kind_of(m) == "llm") {
        modes.push(json!({
            "name": s(m, "id"),
            "title": format!("{} (nextsycl)", s(m, "title")),
            "url": format!("http://127.0.0.1:{}", cfg.or("NS_PORT", "8085")),
            "health": "/health",
            "up_codes": [200],
            "start": format!("{} {} start {} >/dev/null", exe.display(), kind_of(m), s(m, "id")),
            "stop": format!("{} stop >/dev/null", exe.display()),
            "managed_by": "nextsycl",
        }));
    }
    d["modes"] = Value::Array(modes);
    let tmp = format!("{file}.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&d).map_err(|e| e.to_string())? + "\n").map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, &file).map_err(|e| format!("{file}: {e}"))?;
    println!("{file}: the studio's nextsycl modes rewritten (it reads the file when it starts)");
    Ok(())
}
