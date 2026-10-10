//! The settings file (`nextsycl.conf` beside the repository) from the page and the API: every setting the program
//! reads, what it does and what reads it; edits keep the file's comments and order (a setting changed in place, a new
//! one added at the end, a removed one dropped), a copy of the file before each save (`nextsycl.conf.bak-<time>`,
//! the last 10 kept). The environment wins over the file (a service's `Environment=`): such a setting shows as set
//! there. What reads a changed setting has to restart to see it: `restart` does that for this page, the switcher,
//! the studio, the video daemon and the image and music servers.

use serde_json::{json, Map, Value};

use super::{now, Home};

/// A setting: its name, group, default, what it does, and what reads it (serve, switch, studio, video, llm, image,
/// audio, cli)
pub struct Setting {
    pub name: &'static str,
    pub group: &'static str,
    pub default: &'static str,
    pub what: &'static str,
    pub readers: &'static [&'static str],
}

const fn st(name: &'static str, group: &'static str, default: &'static str, what: &'static str, readers: &'static [&'static str]) -> Setting {
    Setting { name, group, default, what, readers }
}

pub const SETTINGS: &[Setting] = &[
    st("NS_MODELS", "box", "", "host directory with the model files (seen as /models in the containers)", &["llm", "image", "audio", "video"]),
    st("NS_REGISTRY", "box", "~/.config/nextsycl/models.json", "the model registry (nextsycl models)", &["serve", "switch", "studio", "llm", "image", "audio", "video"]),
    st("NS_CATALOG", "box", "built in", "another catalog of downloadable models (nextsycl models search | pull)", &["cli"]),
    st("NS_SOCKET_DIR", "box", "$XDG_RUNTIME_DIR/nextsycl", "where the control sockets live", &["serve", "llm", "video", "studio"]),
    st("NS_IMAGE", "box", "localhost/h3-build", "the container image with the oneAPI runtime", &["llm", "image", "audio", "video"]),
    st("NS_CONTAINER_ENGINE", "box", "podman", "podman or docker", &["llm", "image", "audio", "video"]),
    st("NS_MOUNTS", "box", "", "more read-only directories for the containers, host:inside[,...]", &["llm"]),
    st("NS_CORS", "box", "", "web pages that may call the servers from a browser, beyond the loopback ones", &["llm", "image", "audio"]),
    st("NS_SERVE_PORT", "front door", "8000", "this page's port", &["serve"]),
    st("NS_API_TOKEN", "front door", "made once (serve.json)", "the API's bearer token", &["serve"]),
    st("NS_IDLE_MINUTES", "front door", "10", "an enabled model unloads after this long without a request (0: never)", &["serve", "switch"]),
    st("NS_SWITCH_PORT", "front door", "8001", "the model switcher's port (Open WebUI's endpoint)", &["serve"]),
    st("NS_IMAGE_PORT", "front door", "8086", "the image server's port", &["serve"]),
    st("NS_AUDIO_PORT", "front door", "8087", "the music server's port", &["serve"]),
    st("NS_CHAT_UI_PORT", "front door", "8080", "Open WebUI's port (a link)", &["serve"]),
    st("NS_HOST", "chat", "127.0.0.1", "the chat server's address (0.0.0.0: the network)", &["llm"]),
    st("NS_PORT", "chat", "8085", "the chat server's port", &["llm", "serve", "switch"]),
    st("NS_GPUS", "chat", "all", "the chat server's GPUs when a model does not say (\"0 1\")", &["llm"]),
    st("NS_CTX", "chat", "65536", "tokens of context a session, or one a session: 262144,32768", &["llm"]),
    st("NS_PARALLEL", "chat", "2", "requests decoded together", &["llm"]),
    st("NS_MAX_TOKENS", "chat", "to the context's end", "tokens a request without max_tokens may make", &["llm"]),
    st("NS_EFFORT", "chat", "low", "default reasoning effort", &["llm"]),
    st("NS_NO_MTP", "chat", "", "1: decode without the draft block", &["llm"]),
    st("NS_KV", "chat", "q8", "the latent cache's form: q8 or f16", &["llm"]),
    st("NS_PROMPT_CACHE_MIB", "chat", "4096", "host memory for the prompt cache (0: off)", &["llm"]),
    st("NS_CACHE_DIR", "chat", "", "where prompt-cache checkpoints go when pushed out of memory", &["llm"]),
    st("NS_CACHE_DISK_GIB", "chat", "", "their disk budget", &["llm"]),
    st("NS_CACHE_TTL_HOURS", "chat", "", "how long they are kept", &["llm"]),
    st("NS_KEEP_REQUESTS", "chat", "100", "ended requests kept for ps and inspect", &["llm"]),
    st("NS_PREFILL_CHUNK", "chat", "", "prompt tokens a prefill step takes", &["llm"]),
    st("NS_VRAM_GUARD_GIB", "chat", "", "VRAM the chat server leaves free", &["llm"]),
    st("NS_STUDIO_MODES", "chat", "", "the studio's chat modes file, kept in step with the registry", &["cli", "serve"]),
    st("NS_IMAGE_OUT", "images", "~/.local/share/nextsycl/images", "where pictures are saved", &["image"]),
    st("NS_AUDIO_OUT", "music", "~/.local/share/nextsycl/audio", "where songs are saved", &["audio"]),
    st("NS_VIDEO_MODEL", "video", "", "the video daemon's default model (serve: the enabled video models decide)", &["video"]),
    st("NS_VIDEO_ENGINES", "video", "", "the video daemon's other models (serve: the enabled video models decide)", &["video"]),
    st("NS_VIDEO_GPUS", "video", "", "the video daemon's GPUs (serve: the enabled video models' GPUs)", &["video"]),
    st("NS_VIDEO_SHARED_GPUS", "video", "", "GPUs shared with the chat model: a clip there stops chat first", &["video"]),
    st("NS_VIDEO_IDLE", "video", "600", "seconds before an idle video worker lets its card go", &["video"]),
    st("NS_VIDEO_GPU_LOCK", "video", "", "a lock file shared with the card's other users", &["video"]),
    st("NS_VIDEO_LLM_SWITCHER", "video", "", "the chat switch the daemon asks before taking a shared card", &["video"]),
    st("NS_VIDEO_OUT", "video", "", "clips", &["studio", "video"]),
    st("NS_VIDEO_MODELS_DIR", "video", "", "the video models' directory", &["video"]),
    st("NS_VIDEO_STUDIO_DIR", "video", "", "the studio's state (queue, projects)", &["studio"]),
    st("NS_VIDEO_LLM_MODES", "video", "", "the studio's chat modes file", &["studio"]),
    st("NS_VIDEO_TEMPLATES", "video", "", "prompt templates", &["studio"]),
    st("NS_VIDEO_CHARACTERS", "video", "", "characters for the speech tool", &["studio", "cli"]),
    st("NS_VIDEO_GPUSTAT", "video", "/run/gpustat.json", "the GPU telemetry file", &["studio"]),
    st("NS_VIDEO_LISTEN", "video", "127.0.0.1", "the studio's address", &["studio"]),
    st("NS_VIDEO_PORT", "video", "8090", "the studio's port", &["studio", "serve"]),
    st("NS_VIDEO_STUDIO", "video", "http://127.0.0.1:8095", "the studio the tools talk to", &["cli"]),
];

/// Engine settings by prefix (passed to the engine's container)
pub const PREFIXES: &[(&str, &str, &[&str])] = &[
    ("NS_QW_", "the qwen4exp chat engine's settings (docs/engines.md)", &["llm"]),
    ("NS_QI_", "the Qwen-Image engine's settings", &["image"]),
    ("NS_MM3_", "the MiniMax Music engine's settings", &["audio"]),
    ("NSD_", "the diffusion kernels' settings (image, music, video)", &["image", "audio", "video"]),
    ("H3_", "the H3 engine's settings", &["video"]),
];

fn valid_name(k: &str) -> bool {
    !k.is_empty() && k.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_') && !k.starts_with(|c: char| c.is_ascii_digit())
}

/// `NAME=value` lines of a settings text, in order (comments and blanks left out)
fn entries(text: &str) -> Vec<(String, String)> {
    text.lines().filter_map(|l| {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            return None;
        }
        let (k, v) = l.strip_prefix("export ").unwrap_or(l).split_once('=')?;
        let v = v.trim();
        let v = v.strip_prefix('"').and_then(|x| x.strip_suffix('"')).or_else(|| v.strip_prefix('\'').and_then(|x| x.strip_suffix('\''))).unwrap_or(v);
        Some((k.trim().to_string(), v.to_string()))
    }).collect()
}

/// A text checked line by line: every line blank, a comment or NAME=value
fn check(text: &str) -> Result<(), String> {
    for (i, l) in text.lines().enumerate() {
        let t = l.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        match t.strip_prefix("export ").unwrap_or(t).split_once('=') {
            Some((k, v)) if valid_name(k.trim()) && !v.contains('\n') => {}
            _ => return Err(format!("line {}: {l:?} - a line is NAME=value (capitals, digits, _), a # comment or blank", i + 1)),
        }
    }
    Ok(())
}

/// `set` applied to `text`: a name with a value changed in place (its first line; later duplicates dropped), added at
/// the end when new; a name with null removed
fn apply(text: &str, set: &Map<String, Value>) -> Result<String, String> {
    for (k, v) in set {
        if !valid_name(k) {
            return Err(format!("{k:?}: a setting's name is capitals, digits and _"));
        }
        if let Some(s) = v.as_str() {
            if s.contains('\n') {
                return Err(format!("{k}: one line"));
            }
        } else if !v.is_null() {
            return Err(format!("{k}: a string, or null to remove it"));
        }
    }
    let mut done: Vec<&str> = Vec::new();
    let mut out = Vec::new();
    for l in text.lines() {
        let t = l.trim();
        let name = (!t.starts_with('#')).then(|| t.strip_prefix("export ").unwrap_or(t).split_once('=').map(|(k, _)| k.trim())).flatten();
        match name.and_then(|n| set.get_key_value(n)) {
            Some((k, v)) => {
                if done.contains(&k.as_str()) {
                    continue; // a duplicate line of a set name
                }
                done.push(k);
                if let Some(s) = v.as_str() {
                    out.push(format!("{k}={s}"));
                }
            }
            None => out.push(l.to_string()),
        }
    }
    for (k, v) in set {
        if let (false, Some(s)) = (done.contains(&k.as_str()), v.as_str()) {
            out.push(format!("{k}={s}"));
        }
    }
    let mut t = out.join("\n");
    t.push('\n');
    Ok(t)
}

/// What reads `name`
fn readers(name: &str) -> Vec<&'static str> {
    SETTINGS.iter().find(|s| s.name == name).map(|s| s.readers.to_vec())
        .or_else(|| PREFIXES.iter().find(|(p, _, _)| name.starts_with(p)).map(|(_, _, r)| r.to_vec()))
        .unwrap_or_default()
}

impl Home {
    fn conf_path(&self) -> std::path::PathBuf {
        let p = self.o.cfg.file();
        std::fs::canonicalize(&p).unwrap_or(p)
    }

    /// The file, every known setting (its value in the file, what the program sees, its default and readers), the
    /// file's settings the program does not know
    pub(crate) fn config(&self) -> Value {
        let path = self.conf_path();
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let file: Vec<(String, String)> = entries(&text);
        let in_file = |k: &str| file.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        let settings: Vec<Value> = SETTINGS.iter().map(|s| {
            let env = self.o.cfg.in_env(s.name);
            json!({"name": s.name, "group": s.group, "default": s.default, "what": s.what, "readers": s.readers,
                   "file": in_file(s.name), "env": env.then(|| std::env::var(s.name).unwrap_or_default())})
        }).collect();
        let known = |k: &str| SETTINGS.iter().any(|s| s.name == k);
        let extra: Vec<Value> = file.iter().filter(|(k, _)| !known(k)).map(|(k, v)| {
            let p = PREFIXES.iter().find(|(p, _, _)| k.starts_with(p));
            json!({"name": k, "file": v, "what": p.map(|x| x.1).unwrap_or("not a setting this program reads"), "readers": readers(k)})
        }).collect();
        json!({"path": path, "text": text, "settings": settings, "other": extra,
               "prefixes": PREFIXES.iter().map(|(p, w, r)| json!({"prefix": p, "what": w, "readers": r})).collect::<Vec<_>>()})
    }

    /// Save the settings: `text` (the whole file) or `set` ({NAME: value | null}); the file copied first. Answers
    /// what changed and what has to restart to see it
    pub(crate) fn config_set(&self, b: &Value) -> Result<Value, (u16, String)> {
        let path = self.conf_path();
        let old = std::fs::read_to_string(&path).unwrap_or_default();
        let new = match (b["text"].as_str(), b["set"].as_object()) {
            (Some(t), None) => {
                check(t).map_err(|e| (400, e))?;
                if t.ends_with('\n') { t.to_string() } else { format!("{t}\n") }
            }
            (None, Some(set)) => apply(&old, set).map_err(|e| (400, e))?,
            _ => return Err((400, "give text (the whole file) or set ({NAME: value | null})".into())),
        };
        let (a, z) = (entries(&old), entries(&new));
        let get = |v: &[(String, String)], k: &str| v.iter().rev().find(|(n, _)| n == k).map(|(_, x)| x.clone());
        let mut names: Vec<String> = a.iter().chain(&z).map(|(k, _)| k.clone()).collect();
        names.sort();
        names.dedup();
        let changed: Vec<String> = names.into_iter().filter(|k| get(&a, k) != get(&z, k)).collect();
        if changed.is_empty() {
            return Ok(json!({"changed": [], "restart": []}));
        }
        // the copy, then the file (written whole, renamed into place)
        let dir = path.parent().map(std::path::Path::to_path_buf).unwrap_or_default();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "nextsycl.conf".into());
        if !old.is_empty() {
            std::fs::write(dir.join(format!("{name}.bak-{}", now() as u64)), &old).map_err(|e| (500, format!("the copy: {e}")))?;
            // the last 10 copies kept
            let mut baks: Vec<_> = std::fs::read_dir(&dir).into_iter().flatten().flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}.bak-1"))).map(|e| e.path()).collect();
            baks.sort();
            for p in baks.iter().take(baks.len().saturating_sub(10)) {
                let _ = std::fs::remove_file(p);
            }
        }
        let tmp = dir.join(format!("{name}.tmp"));
        std::fs::write(&tmp, &new).and_then(|_| std::fs::rename(&tmp, &path)).map_err(|e| (500, format!("{}: {e}", path.display())))?;
        let mut restart: Vec<&str> = changed.iter().flat_map(|k| readers(k)).collect();
        restart.sort();
        restart.dedup();
        let env: Vec<&String> = changed.iter().filter(|k| self.o.cfg.in_env(k)).collect();
        Ok(json!({"changed": changed, "restart": restart, "path": path,
                  "note": if env.is_empty() { Value::Null } else { json!(format!("{env:?} are set in this service's environment, which wins over the file")) }}))
    }

    /// Restart what reads the settings: serve (this page: systemd brings it back), switch, studio (their services
    /// do), video (the daemon, when no clip renders), image / audio (stopped: the next request loads them), llm (the
    /// chat model reloaded)
    pub(crate) fn restart(&self, what: &str) -> Result<Value, (u16, String)> {
        let kill = |pattern: &str| -> Result<(), (u16, String)> {
            // the process by its exact command (the services run as this user)
            let out = std::process::Command::new("pgrep").args(["-u", &std::env::var("USER").unwrap_or_default(), "-f", pattern]).output()
                .map_err(|e| (500, e.to_string()))?;
            let pids: Vec<String> = String::from_utf8_lossy(&out.stdout).split_whitespace().map(str::to_string)
                .filter(|p| *p != std::process::id().to_string()).collect();
            if pids.is_empty() {
                return Err((404, format!("nothing runs as {pattern}")));
            }
            std::process::Command::new("kill").args(&pids).status().map_err(|e| (500, e.to_string()))?;
            Ok(())
        };
        match what {
            "serve" => {
                // after the answer: systemd's Restart=always brings it back (the servers it started stay up)
                std::thread::spawn(|| {
                    std::thread::sleep(std::time::Duration::from_millis(700));
                    std::process::exit(0);
                });
                Ok(json!({"restarting": "serve"}))
            }
            "switch" => kill("nextsycl switch --port").map(|_| json!({"restarting": "switch"})),
            "studio" => {
                let pidf = std::path::PathBuf::from(self.o.cfg.get("NS_SOCKET_DIR").unwrap_or_else(|| "/tmp".into())).join("studio.pid");
                let pid = std::fs::read_to_string(&pidf).map_err(|e| (404, format!("{}: {e}", pidf.display())))?;
                std::process::Command::new("kill").arg(pid.trim()).status().map_err(|e| (500, e.to_string()))?;
                Ok(json!({"restarting": "studio"}))
            }
            "video" => {
                self.exec("video stop (settings changed)".into(), &["video".into(), "stop".into()]).map_err(|e| (500, e))?;
                self.video_apply(false).map_err(|e| (500, e))?;
                Ok(json!({"restarted": "video"}))
            }
            k @ ("image" | "audio") => {
                let _ = self.exec(format!("{k} stop (settings changed)"), &[k.into(), "stop".into()]);
                Ok(json!({"stopped": k, "note": "the next request loads it with the new settings"}))
            }
            "llm" => {
                let all = self.models();
                let Some(id) = self.chat_loaded(&all) else { return Ok(json!({"note": "no chat model is loaded: the next one loads with the new settings"})) };
                self.unload(&id)?;
                self.load(&id)?;
                Ok(json!({"reloaded": id}))
            }
            other => Err((400, format!("restart {other:?}: serve, switch, studio, video, image, audio or llm"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_keep_the_rest() {
        let text = "# box\nNS_A=1\n\n# b\nNS_B=x\nNS_A=dup\n";
        let set: Map<String, Value> = serde_json::from_str(r#"{"NS_A": "2", "NS_B": null, "NS_C": "new"}"#).unwrap();
        assert_eq!(apply(text, &set).unwrap(), "# box\nNS_A=2\n\n# b\nNS_C=new\n");
        assert!(check("NS_A=1\n# c\n\nexport NS_B=\"q\"").is_ok());
        assert!(check("ns_a=1").is_err() && check("NS A=1").is_err() && check("just words").is_err());
        assert_eq!(entries("export NS_B=\"q\"\nNS_C='r'"), vec![("NS_B".into(), "q".into()), ("NS_C".into(), "r".into())]);
    }
}
