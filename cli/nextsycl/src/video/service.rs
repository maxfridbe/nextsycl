//! The video service from the host (H3's `sycl-h3 start | stop | gpus | logs`): the daemon in its container
//! (`nextsycl-video`), the GPUs, the logs. The models are registry entries; their files, the registry and the output
//! directory are mounted at their own paths, so a job's paths are the same inside and out.
//!
//! Settings (environment, or nextsycl.conf):
//!
//! ```text
//!   NS_VIDEO_MODEL        the model jobs get unless they name one ("engine"): a registry id (default: the first video
//!                         model)
//!   NS_VIDEO_ENGINES      the other models a job may name, space-separated ids (e.g. "minimax-h3-q4km minimax-h3-q6k")
//!   NS_VIDEO_OUT          where clips go (default ~/.local/share/nextsycl/video)
//!   NS_VIDEO_GPUS         GPUs to serve, e.g. "0 1" (default all)   NS_VIDEO_SHARED_GPUS  GPUs the hooks are about
//!   NS_VIDEO_IDLE         seconds without a job before an engine unloads; 0 = never (default 600)
//!   NS_VIDEO_GPU_LOCK     a lock file shared with the GPU's other users
//!   NS_VIDEO_LLM_SWITCHER a front end's model switcher URL: its model stops before loading, comes back after
//! ```

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

use super::client;
use crate::container::{mount, Ce, SOCKET_DIR_IN};

pub const ENGINE: &str = "nextsycl-video";

pub fn socket(cfg: &Config) -> PathBuf {
    cfg.socket_dir().join("video.sock")
}

fn socket_ready(cfg: &Config) -> bool {
    std::os::unix::net::UnixStream::connect(socket(cfg)).is_ok()
}

/// Where clips go: NS_VIDEO_OUT, else ~/.local/share/nextsycl/video
pub fn out_dir(cfg: &Config) -> PathBuf {
    cfg.get("NS_VIDEO_OUT").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(format!("{}/.local/share/nextsycl/video", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())))
    })
}

fn wait_until(what: &str, secs: u64, mut ok: impl FnMut() -> bool) -> Result<(), String> {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        if ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err(format!("{what} did not come up in {secs} s (nextsycl video logs)"))
}

fn repeated(raw: &[String], name: &str) -> Vec<String> {
    let mut v = Vec::new();
    let mut it = raw.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(x) = it.next() {
                v.push(x.clone());
            }
        }
    }
    v
}

/// `nextsycl video start [--model ID] [--engine ID]... [--all | --gpu N ...] [--shared-gpu N ...]`: the daemon's
/// container
pub fn start(cfg: &Config, raw: &[String]) -> Result<(), String> {
    let all = raw.iter().any(|a| a == "--all");
    let gpus = repeated(raw, "--gpu");
    if all && !gpus.is_empty() {
        return Err("give --all or --gpu N ..., not both".into());
    }
    let gpus: Vec<String> = if all || !gpus.is_empty() { gpus } else { cfg.or("NS_VIDEO_GPUS", "").split_whitespace().map(String::from).collect() };
    let shared = repeated(raw, "--shared-gpu");
    let shared: Vec<String> = if shared.is_empty() { cfg.or("NS_VIDEO_SHARED_GPUS", "").split_whitespace().map(String::from).collect() } else { shared };
    for g in gpus.iter().chain(&shared) {
        g.parse::<usize>().map_err(|_| format!("{g}: not a GPU number (nextsycl video gpus)"))?;
    }
    // the models: the default and the others a job may name, every one a registry entry
    let model_id = super::opt(raw, "--model").map(str::to_string).or_else(|| cfg.get("NS_VIDEO_MODEL"));
    let (m, _) = super::model(cfg, model_id.as_deref())?;
    let id = m["id"].as_str().unwrap_or("").to_string();
    let mut others = repeated(raw, "--engine");
    if others.is_empty() {
        others = cfg.or("NS_VIDEO_ENGINES", "").split_whitespace().map(String::from).collect();
    }
    let mut entries: Vec<Value> = vec![m.clone()];
    for o in &others {
        entries.push(super::model(cfg, Some(o))?.0);
    }
    // the --opt-NAMEs the engine takes at load: checked here, passed as the variables they set
    let (_, opts) = super::kind_and_options(&m)?;

    let ce = Ce::new(cfg)?;
    if ce.running(ENGINE) {
        println!("the video engine is already running");
        return client::status(&["--no-stream".into()], &|| String::new());
    }
    ce.need_image()?;
    for f in ["nextsycl", "libnextsycl-video.so"] {
        if !cfg.dist.join(f).exists() {
            return Err(format!("{}: not built yet (./build.sh)", cfg.dist.join(f).display()));
        }
    }
    ce.remove(ENGINE); // a stopped one from before
    let out = out_dir(cfg);
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let sock_dir = cfg.socket_dir();
    std::fs::create_dir_all(&sock_dir).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(socket(cfg));

    let mut args: Vec<String> = vec!["run".into(), "-d".into(), "--init".into(), "--name".into(), ENGINE.into(), "--network".into(), "host".into()];
    // a stop waits for the running job's next step boundary (about a second at production size), then unloads
    args.extend(["--stop-timeout".into(), "150".into()]);
    args.extend(ce.user_args());
    args.extend(ce.gpu_args());
    args.extend(mount(&cfg.dist, "/app", true));
    // the registry, the models' and their LoRAs' files, read-only at their own paths; the clips, writable
    let reg = models::registry(cfg);
    let mut paths = vec![reg.clone()];
    for e in &entries {
        paths.extend(models::paths(e));
    }
    for l in models::all(cfg)?.iter().filter(|l| models::kind_of(l) == "lora" && l["arch"] == m["arch"]) {
        paths.extend(models::paths(l));
    }
    let mut dirs = std::collections::BTreeSet::new();
    for p in paths {
        let d = if p.is_dir() { p } else { p.parent().map(PathBuf::from).unwrap_or_default() };
        if !d.as_os_str().is_empty() && d.exists() && dirs.insert(d.clone()) {
            args.extend(mount(&d, &d.to_string_lossy(), true));
        }
    }
    args.extend(mount(&out, &out.to_string_lossy(), false));
    args.extend(mount(&sock_dir, SOCKET_DIR_IN, false));
    let mut daemon_args: Vec<String> = vec!["--socket".into(), format!("{SOCKET_DIR_IN}/video.sock"), "--model".into(), id.clone(),
                                           "--idle".into(), cfg.or("NS_VIDEO_IDLE", "600")];
    for o in &others {
        daemon_args.extend(["--engine".into(), o.clone()]);
    }
    for g in &gpus {
        daemon_args.extend(["--gpu".into(), g.clone()]);
    }
    for g in &shared {
        daemon_args.extend(["--shared-gpu".into(), g.clone()]);
    }
    if let Some(lock) = cfg.get("NS_VIDEO_GPU_LOCK") {
        let dir = PathBuf::from(&lock).parent().map(|d| d.to_path_buf()).unwrap_or_default();
        args.extend(mount(&dir, &dir.to_string_lossy(), false));
        daemon_args.extend(["--gpu-lock".into(), lock]);
    }
    if let Some(sw) = cfg.get("NS_VIDEO_LLM_SWITCHER") {
        daemon_args.extend(["--llm-switcher".into(), sw]);
    }
    args.extend(["-e".into(), format!("NS_REGISTRY={}", reg.display()), "-e".into(), "ONEAPI_DEVICE_SELECTOR=level_zero:*".into()]);
    // the engine's own settings: from the configuration (NSD_*, H3_*), then the --opt-NAMEs
    for (k, v) in cfg.with_prefix("NSD_").into_iter().chain(cfg.with_prefix("H3_")) {
        args.extend(["-e".into(), format!("{k}={v}")]);
    }
    for (k, v) in &opts.settings {
        args.extend(["-e".into(), format!("{k}={v}")]);
    }
    args.extend([ce.image.clone(), "bash".into(), "-c".into(),
                 "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl video daemon \"$@\"".into(), "nextsycl".into()]);
    args.extend(daemon_args);
    let st = ce.cmd().args(&args).stdout(Stdio::null()).status().map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("{} run failed", ce.bin));
    }
    wait_until("the video engine", 60, || socket_ready(cfg))?;
    client::status(&["--no-stream".into()], &|| String::new())
}

/// `nextsycl video stop`: the running jobs stop at their next step boundary, the engines unload, the daemon ends
pub fn stop(cfg: &Config) -> Result<(), String> {
    let ce = Ce::new(cfg)?;
    if ce.running(ENGINE) {
        let _ = client::post("/engine/shutdown", None);
        let t0 = Instant::now();
        while ce.running(ENGINE) && t0.elapsed() < Duration::from_secs(150) {
            std::thread::sleep(Duration::from_millis(500));
        }
        if ce.running(ENGINE) {
            ce.stop(ENGINE, 150); // the backstop: SIGTERM means the same to the daemon
        }
        ce.remove(ENGINE);
        println!("video engine: stopped");
    } else {
        println!("video engine: not running");
    }
    Ok(())
}

/// `nextsycl video gpus`: from the running daemon, or a short-lived container
pub fn gpus(cfg: &Config, raw: &[String]) -> Result<(), String> {
    if raw.iter().any(|a| a == "--json") {
        return super::gpus(raw); // in the container (the daemon asks it so)
    }
    let list = match client::get("/engine/gpus") {
        Ok(v) => v,
        Err(_) => {
            let ce = Ce::new(cfg)?;
            ce.need_image()?;
            let mut args: Vec<String> = vec!["run".into(), "--rm".into()];
            args.extend(ce.user_args());
            args.extend(ce.gpu_args());
            args.extend(mount(&cfg.dist, "/app", true));
            args.extend(["-e".into(), "ONEAPI_DEVICE_SELECTOR=level_zero:*".into()]);
            args.extend([ce.image.clone(), "bash".into(), "-c".into(),
                         "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl video gpus --json".into()]);
            let out = ce.cmd().args(&args).stderr(Stdio::inherit()).output().map_err(|e| e.to_string())?;
            if !out.status.success() {
                return Err("could not list the GPUs".into());
            }
            serde_json::from_slice(&out.stdout).map_err(|e| format!("nextsycl video gpus: {e}"))?
        }
    };
    println!("{:<4} {:<34} {:>8}  {:<14} SERVED", "GPU", "NAME", "MEMORY", "PCI");
    for g in list.as_array().cloned().unwrap_or_default() {
        println!("{:<4} {:<34} {:>5.1}GiB  {:<14} {}", g["index"].to_string(), g["name"].as_str().unwrap_or("?"), g["mem_gib"].as_f64().unwrap_or(0.0),
                 g["pci"].as_str().unwrap_or(""), match g["served"].as_bool() { Some(true) => "yes", Some(false) => "no", None => "-" });
    }
    Ok(())
}

/// `nextsycl video logs`: the daemon's, followed
pub fn logs(cfg: &Config) -> Result<(), String> {
    let ce = Ce::new(cfg)?;
    let st = ce.cmd().args(["logs", "-f", ENGINE]).status().map_err(|e| e.to_string())?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("no log for {ENGINE} (is it running?)"))
    }
}
