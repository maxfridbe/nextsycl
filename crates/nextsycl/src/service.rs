//! The server as a service (sycl-h3's start / stop / logs): `nextsycl start` runs `nextsycl serve` in a container
//! with the GPUs, the models and the control socket's directory mounted; `stop` asks it to end once no request runs.
//!
//! ```text
//!   nextsycl (you) ---- start/stop (podman) ----> [nextsycl]   nextsycl serve: the model on the GPUs
//!        |                                             ^ Unix socket (JSON over HTTP): status, ps, cache, chat
//!        +---------- status/ps/cache/chat -------------+
//!   clients (Open WebUI, ...) ------------ OpenAI API on TCP (NS_HOST:NS_PORT) ------^
//! ```

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use crate::client;
use crate::config::Config;
use crate::container::{mount, Ce, SERVER, SOCKET_DIR_IN};

fn need_dist(cfg: &Config, file: &str) -> Result<PathBuf, String> {
    let p = cfg.dist.join(file);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("{}: not built yet (./build.sh)", p.display()))
    }
}

fn socket_ready(cfg: &Config) -> bool {
    std::os::unix::net::UnixStream::connect(cfg.socket()).is_ok()
}

/// `--name value` from the arguments, else the setting, else the default
fn arg(raw: &[String], flag: &str, cfg: &Config, setting: &str, default: &str) -> String {
    raw.iter().position(|a| a == flag).and_then(|i| raw.get(i + 1)).cloned().unwrap_or_else(|| cfg.or(setting, default))
}

/// `nextsycl start [--gpu N ...] [--model PATH] [--port N] [--host H] [--ctx N] [--name ID] [--effort E]
/// [--prompt-cache-mib N] [--no-mtp]`
pub fn start(cfg: &Config, raw: &[String]) -> Result<(), String> {
    let ce = Ce::new(cfg)?;
    if ce.running(SERVER) {
        println!("the server is already running");
        return client::status(&["--no-stream".into()]);
    }
    ce.need_image()?;
    need_dist(cfg, "nextsycl")?;
    need_dist(cfg, "libnextsycl.so")?;
    let models = cfg.get("NS_MODELS").ok_or("set NS_MODELS (the host directory with the model files, seen as /models)")?;
    // --gpu 0 --gpu 1, or NS_GPUS="0 1", or every GPU
    let mut gpus: Vec<String> = Vec::new();
    let mut it = raw.iter();
    while let Some(a) = it.next() {
        if a == "--gpu" {
            gpus.push(it.next().ok_or("--gpu N")?.clone());
        }
    }
    if gpus.is_empty() {
        gpus = cfg.or("NS_GPUS", "all").split([' ', ',']).filter(|s| !s.is_empty()).map(String::from).collect();
    }
    ce.remove(SERVER); // a stopped one from before
    let sock_dir = cfg.socket_dir();
    std::fs::create_dir_all(&sock_dir).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(cfg.socket());

    let mut args: Vec<String> = vec!["run".into(), "-d".into(), "--name".into(), SERVER.into(), "--network".into(), "host".into()];
    // a stop waits for the request running to finish (a long answer: minutes)
    args.extend(["--stop-timeout".into(), "300".into()]);
    args.extend(ce.user_args());
    args.extend(ce.gpu_args());
    args.extend(mount(&cfg.dist, "/app", true));
    args.extend(mount(&PathBuf::from(&models), "/models", true));
    args.extend(mount(&sock_dir, SOCKET_DIR_IN, false));
    // the prompt cache's disk tier (NS_CACHE_DIR, default ~/.cache/nextsycl/prompts; NS_CACHE_DISK_GIB, default 32,
    // 0 = none): checkpoints pushed out of memory - a 256K prompt's is ~3.5 GiB
    let disk_gib = cfg.or("NS_CACHE_DISK_GIB", "32");
    let cache_dir = (disk_gib.trim() != "0").then(|| {
        PathBuf::from(cfg.get("NS_CACHE_DIR").unwrap_or_else(|| format!("{}/.cache/nextsycl/prompts", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))))
    });
    if let Some(d) = &cache_dir {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        args.extend(mount(d, "/cache", false));
    }
    args.extend(["-e".into(), "NEXTSYCL_LIB=/app/libnextsycl.so".into(), "-e".into(), "ONEAPI_DEVICE_SELECTOR=level_zero:*".into()]);
    // engine settings the server reads from its environment (NS_KV=q8: the latent cache in q8)
    for k in ["NS_KV"] {
        if let Some(v) = cfg.get(k) {
            args.extend(["-e".into(), format!("{k}={v}")]);
        }
    }
    args.extend([ce.image.clone(), "bash".into(), "-c".into(),
                 "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl serve \"$@\"".into(), "nextsycl".into()]);
    args.push(arg(raw, "--model", cfg, "NS_MODEL", "/models/glm53-iq2/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf"));
    args.extend(["--gpu".into(), gpus.join(",")]);
    for (flag, setting, default) in [("--host", "NS_HOST", "127.0.0.1"), ("--port", "NS_PORT", "8085"), ("--ctx", "NS_CTX", "65536"),
                                     ("--name", "NS_NAME", "glm-5.3-flash-uncensored"), ("--effort", "NS_EFFORT", "low"),
                                     ("--prompt-cache-mib", "NS_PROMPT_CACHE_MIB", "4096"), ("--keep-requests", "NS_KEEP_REQUESTS", "100"),
                                     ("--parallel", "NS_PARALLEL", "2")] {
        args.extend([flag.into(), arg(raw, flag, cfg, setting, default)]);
    }
    if raw.iter().any(|a| a == "--no-mtp") || cfg.get("NS_NO_MTP").is_some_and(|v| v == "1") {
        args.push("--no-mtp".into());
    }
    if let Some(m) = cfg.get("NS_MAX_TOKENS") {
        args.extend(["--max-tokens".into(), m]);
    }
    if let Some(c) = cfg.get("NS_CORS") {
        args.extend(["--cors".into(), c]);
    }
    if cache_dir.is_some() {
        args.extend(["--cache-dir".into(), "/cache".into(), "--cache-disk-gib".into(), disk_gib, "--cache-ttl-hours".into(), cfg.or("NS_CACHE_TTL_HOURS", "24")]);
    }
    args.extend(["--socket".into(), format!("{SOCKET_DIR_IN}/nextsycl.sock")]);
    let st = ce.cmd().args(&args).stdout(Stdio::null()).status().map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("{} run failed", ce.bin));
    }
    // the model loads before the socket answers (~20 s for the IQ2 file on two cards)
    let t0 = Instant::now();
    while !socket_ready(cfg) {
        if !ce.running(SERVER) {
            return Err("the server ended while loading (nextsycl logs)".into());
        }
        if t0.elapsed() > Duration::from_secs(600) {
            return Err("the server did not come up in 600 s (nextsycl logs)".into());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    client::status(&["--no-stream".into()])
}

/// `nextsycl stop`: the server ends once no request runs (never inside a GPU kernel); SIGTERM after 300 s means the
/// same to it.
pub fn stop(cfg: &Config) -> Result<(), String> {
    let ce = Ce::new(cfg)?;
    if !ce.running(SERVER) {
        ce.remove(SERVER);
        println!("server: not running");
        return Ok(());
    }
    let _ = client::post("/server/shutdown", None);
    let t0 = Instant::now();
    while ce.running(SERVER) && t0.elapsed() < Duration::from_secs(300) {
        std::thread::sleep(Duration::from_millis(500));
    }
    if ce.running(SERVER) {
        ce.stop(SERVER, 300);
    }
    ce.remove(SERVER);
    println!("server: stopped");
    Ok(())
}

/// `nextsycl logs [--no-follow]`
pub fn logs(cfg: &Config, raw: &[String]) -> Result<(), String> {
    let ce = Ce::new(cfg)?;
    let mut a = vec!["logs".to_string()];
    if !raw.iter().any(|x| x == "--no-follow") {
        a.push("-f".into());
    }
    a.push(SERVER.into());
    let st = ce.cmd().args(&a).status().map_err(|e| e.to_string())?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("no log for {SERVER} (is it running?)"))
    }
}
