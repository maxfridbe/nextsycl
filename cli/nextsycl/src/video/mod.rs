//! `nextsycl video ...`: the video commands. The engine's jobs (H3's: generate, encode, decode, denoise,
//! check-block, bench-blocks) run here in-process with `job <kind> --here` (a one-shot: load, run, exit), or go to
//! the daemon's queue; `worker` is the per-GPU process the daemon starts.

pub mod client;
pub mod plan;
pub mod service;
pub mod tools;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use nextsycl_models::{config::Config, registry as models};
use nextsycl_video::{JobCtl, LoadOptions, ModelFiles, VideoEngine, VideoKind};
use serde_json::{json, Value};

const USAGE: &str = "the service (the daemon in its container; models are registry ids):
nextsycl video start [--model ID] [--engine ID]... [--all | --gpu N ...] [--shared-gpu N ...]
nextsycl video serve [--bind ADDR] [--port N]   the studio: the web front end and its clip queue (default :8095)
nextsycl video stop [--web | --all] | status [--no-stream] | logs [--web] | gpus | unload [--gpu N]
the jobs:
nextsycl video job <kind> [--KEY VALUE]... [--engine ID] [--gpu N] [-f]   queue one (-f: follow its log)
nextsycl video job <kind> [--model ID] [--KEY VALUE]... [--json SPEC|@FILE] [--gpu N] --here
                    a job in this process instead: load, run, print its result (kinds: nextsycl video engines)
nextsycl video ps [-a] | inspect <id> | cancel <id>... | rm <id>...
the studio's tools (they talk to it as the front end does):
nextsycl video speech <text file> [options] | scene <scene.json> [options] | join <prefix> [options] | speechpct <clip>...
nextsycl video plan measure [--gpu N ...] [--engine NAME ...] [--tokens 2048,...] [--no-clip] | plan show
nextsycl video engines | selftest [--gpu N]
in the service's container: nextsycl video daemon --socket PATH --model ID [--engine ID]... [--gpu N]... [--shared-gpu N]...
                    [--idle S] [--gpu-lock FILE] [--llm-switcher URL] [--threads N];  nextsycl video worker --gpu N --model ID";

/// The video engines this program has (video/<arch>)
pub fn engines() -> Vec<VideoKind> {
    vec![nextsycl_video_h3::kind(), nextsycl_video_example::kind()]
}

pub(crate) fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// The registered video model `id`, or the first one: (entry, files by role)
pub fn model(cfg: &Config, id: Option<&str>) -> Result<(Value, ModelFiles), String> {
    let all = models::all(cfg)?;
    let m = match id {
        Some(id) => all.into_iter().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id} (nextsycl models search --kind video)"))?,
        None => all.into_iter().find(|m| models::kind_of(m) == "video").ok_or("no video model (nextsycl models pull minimax-h3)")?,
    };
    if models::kind_of(&m) != "video" {
        return Err(format!("{} is a {} model, not a video one", m["id"].as_str().unwrap_or("?"), models::kind_of(&m)));
    }
    let files: ModelFiles = m["files"].as_object().into_iter().flatten().filter_map(|(r, p)| Some((r.clone(), PathBuf::from(p.as_str()?)))).collect();
    Ok((m, files))
}

/// The entry's engine and its load options: the entry's settings, then the --opt-NAMEs it takes at load
pub fn kind_and_options(m: &Value) -> Result<(&'static VideoKind, LoadOptions), String> {
    let arch = m["arch"].as_str().unwrap_or("");
    let ks: &'static [VideoKind] = Box::leak(engines().into_boxed_slice());
    let k = nextsycl_video::kind_for(ks, arch)?;
    let mut o = LoadOptions::default();
    for (key, v) in m["env"].as_object().into_iter().flatten() {
        if let Some(v) = v.as_str() {
            o.settings.insert(key.clone(), v.to_string());
        }
    }
    for (var, v) in crate::opt_env(k.options, nextsycl_core::At::Load, k.name)? {
        o.settings.insert(var, v);
    }
    Ok((k, o))
}

/// The engine of entry `m`, loaded on GPU `gpu`
pub fn load(m: &Value, files: &ModelFiles, gpu: usize, log: &mut dyn FnMut(String)) -> Result<Box<dyn VideoEngine>, String> {
    nextsycl_core::use_kind("video");
    let (k, o) = kind_and_options(m)?;
    // the engine's own variables (the kernels read them with getenv): set before anything opens the GPU
    for (var, v) in &o.settings {
        if var.starts_with("NSD_") || var.starts_with("H3_") {
            std::env::set_var(var, v);
        }
    }
    let g = nextsycl_core::Gpu::open(gpu).map_err(|e| e.0)?;
    (k.load)(files, &[g], &o, log).map_err(|e| e.0)
}

/// A job's spec from the command line: `{"kind": <kind>}`, then --json (text, or @FILE), then every --KEY VALUE that
/// is not the command's own (numbers as numbers, a bare --FLAG as true)
pub fn spec(kind: &str, args: &[String], own: &[&str]) -> Result<Value, String> {
    let mut s = json!({ "kind": kind });
    if let Some(j) = opt(args, "--json") {
        let text = match j.strip_prefix('@') {
            Some(f) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
            None => j.to_string(),
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("--json: {e}"))?;
        for (k, x) in v.as_object().into_iter().flatten() {
            s[k] = x.clone();
        }
    }
    let args = nextsycl_core::options::strip(args);
    let mut i = 0;
    while i < args.len() {
        let Some(k) = args[i].strip_prefix("--") else {
            i += 1;
            continue;
        };
        let takes_value = args.get(i + 1).is_some_and(|v| !v.starts_with("--"));
        if own.contains(&k) || k == "json" {
            i += if takes_value { 2 } else { 1 };
            continue;
        }
        if takes_value {
            let v = &args[i + 1];
            s[k] = v.parse::<f64>().map_or_else(|_| json!(v), |n| json!(n));
            i += 2;
        } else {
            s[k] = json!(true);
            i += 1;
        }
    }
    Ok(s)
}

/// `job <kind> --here`: load the engine, run the job, print its result
fn job_here(cfg: &Config, kind: &str, args: &[String]) -> Result<(), String> {
    let (m, files) = model(cfg, opt(args, "--model"))?;
    let gpu: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let spec = spec(kind, args, &["model", "gpu", "here"])?;
    let t0 = std::time::Instant::now();
    let mut log = |l: String| eprintln!("{l}");
    let e = load(&m, &files, gpu, &mut log)?;
    if !e.jobs().contains(&kind) {
        return Err(format!("{}: no job {kind} (it runs {})", e.arch(), e.jobs().join(", ")));
    }
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let cancel = AtomicBool::new(false);
    let mut progress = |done: usize, total: usize| eprintln!("progress {done}/{total}");
    let t1 = std::time::Instant::now();
    let v = e.run_job(&spec, &mut JobCtl { log: &mut log, cancel: &cancel, progress: Some(&mut progress) }).map_err(|e| e.0)?;
    eprintln!("{kind} in {:.1} s", t1.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    Ok(())
}

/// `video gpus [--json]`: the GPUs there are (without opening them)
pub(crate) fn gpus(args: &[String]) -> Result<(), String> {
    nextsycl_core::use_kind("video");
    let list = nextsycl_video_h3::device::Device::list().map_err(|e| e.0)?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if args.iter().any(|a| a == "--json") {
        let v: Vec<Value> = list.iter().map(|g| json!({"index": g.index, "name": g.name, "mem_gib": gib(g.mem_bytes), "pci": g.pci})).collect();
        println!("{}", Value::from(v));
    } else {
        println!("{:<4} {:<34} {:>8}  PCI", "GPU", "NAME", "MEMORY");
        for g in &list {
            println!("{:<4} {:<34} {:>5.1}GiB  {}", g.index, g.name, gib(g.mem_bytes), g.pci);
        }
    }
    Ok(())
}

/// The values of a repeatable option (`--gpu 0 --gpu 1`)
fn repeated(args: &[String], name: &str) -> Result<Vec<String>, String> {
    let mut v = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            v.push(it.next().ok_or_else(|| format!("{name} needs a value"))?.clone());
        }
    }
    Ok(v)
}

/// `video daemon`: the queue and the GPU slots (nextsycl_serve::video::daemon), the models by registry id
fn daemon(cfg: &Config, args: &[String]) -> Result<(), String> {
    use nextsycl_serve::video::daemon::{serve, Options};
    let num = |v: &str, what: &str| v.parse::<usize>().map_err(|_| format!("{what} {v}: a number"));
    let file_of = |id: &str| -> Result<(String, PathBuf), String> {
        let (_, files) = model(cfg, Some(id))?;
        Ok((id.to_string(), files.get("denoiser").cloned().ok_or_else(|| format!("{id}: no denoiser file"))?))
    };
    let (name, path) = file_of(opt(args, "--model").ok_or("daemon needs --model ID")?)?;
    let engines = repeated(args, "--engine")?.iter().map(|e| file_of(e)).collect::<Result<Vec<_>, _>>()?;
    let gpus = repeated(args, "--gpu")?.iter().map(|g| num(g, "--gpu")).collect::<Result<Vec<_>, _>>()?;
    let shared = repeated(args, "--shared-gpu")?.iter().map(|g| num(g, "--shared-gpu")).collect::<Result<Vec<_>, _>>()?;
    let idle = opt(args, "--idle").map_or(Ok(600), |v| num(v, "--idle"))?;
    serve(Options {
        socket: PathBuf::from(opt(args, "--socket").ok_or("daemon needs --socket PATH")?),
        model: path,
        model_name: name,
        engines,
        gpus: (!gpus.is_empty()).then_some(gpus),
        shared_gpus: (!shared.is_empty()).then_some(shared),
        idle: (idle > 0).then(|| std::time::Duration::from_secs(idle as u64)),
        gpu_lock: opt(args, "--gpu-lock").map(PathBuf::from),
        llm_switcher: opt(args, "--llm-switcher").map(str::to_string),
        threads: opt(args, "--threads").map_or(Ok(8), |v| num(v, "--threads"))?,
    })
    .map_err(|e| e.0)
}

fn emit(v: Value) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

/// `video worker --gpu N --model ID`: the process that holds GPU N for the daemon (H3's `h3d worker`), speaking JSON
/// lines - requests on stdin, events on stdout:
///
/// ```text
///   in:   {"job": 3, "spec": {"kind": ...}}  {"cancel": 3}  {"exit": true} (or stdin closed)
///   out:  {"event": "engine", "state": ...}  {"event": "ready", "info": {...}}  {"event": "log", "job": 3, "line": ...}
///         {"event": "progress", "job": 3, "done": 12, "total": 100}  {"event": "stats", ...} (once a second)
///         {"event": "result", "job": 3, "ok": true, "value": {...}} | {"event": "result", "job": 3, "ok": false, "error": ...}
/// ```
///
/// It waits until its card has the room before it loads (never into a card still occupied), keeps the model for the
/// next jobs, and stops a job only between its steps.
fn worker(cfg: &Config, args: &[String]) -> Result<(), String> {
    use std::io::BufRead;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{mpsc, Arc};
    use std::time::Duration;
    let gpu: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).ok_or("worker needs --gpu N")?;
    let id = opt(args, "--model").ok_or("worker needs --model ID")?;
    let (mut m, files) = model(cfg, Some(id))?;
    if let Some(t) = opt(args, "--threads") {
        m["env"]["H3_THREADS"] = json!(t);
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let current = Arc::new(AtomicU64::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let abort = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<(u64, Value)>();
    {
        let (current, cancel, abort) = (current.clone(), cancel.clone(), abort.clone());
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                if let Some(j) = v.get("cancel").and_then(Value::as_u64) {
                    if current.load(Ordering::SeqCst) == j {
                        cancel.store(true, Ordering::SeqCst);
                    }
                } else if v.get("exit").is_some() {
                    break;
                } else if let (Some(j), Some(spec)) = (v.get("job").and_then(Value::as_u64), v.get("spec")) {
                    let _ = tx.send((j, spec.clone()));
                }
            }
            abort.store(true, Ordering::SeqCst);
            cancel.store(true, Ordering::SeqCst);
        });
    }
    // the card's room first: the denoiser and 8 GiB of work (the card's size at most)
    nextsycl_core::use_kind("video");
    let den: u64 = files.get("denoiser").and_then(|p| std::fs::metadata(p).ok()).map_or(0, |m| m.len());
    {
        // one context to ask the card with, let go before the engine opens its own
        let g = nextsycl_core::Gpu::open(gpu).map_err(|e| e.0)?;
        let total = g.memory().map_or(u64::MAX, |(t, _)| t);
        let need = (den + (8u64 << 30)).min(total);
        loop {
            if abort.load(Ordering::SeqCst) {
                return Ok(());
            }
            match g.memory().ok().and_then(|(_, f)| f) {
                Some(free) if free < need => {
                    emit(json!({"event": "engine", "state": format!("waiting for the GPU: {:.1} GiB free, {:.1} GiB needed", gib(free), gib(need))}));
                    std::thread::sleep(Duration::from_secs(5));
                }
                _ => break,
            }
        }
    }
    emit(json!({"event": "engine", "state": "loading"}));
    let mut log = |l: String| {
        eprintln!("[worker] {l}");
        emit(json!({"event": "engine", "state": "loading", "line": l}));
    };
    let e: Arc<Box<dyn VideoEngine>> = Arc::new(load(&m, &files, gpu, &mut log)?);
    let stats = e.device_stats();
    emit(json!({"event": "ready", "info": {"gpu": gpu, "model": id, "arch": e.arch(), "load_seconds": e.load_seconds(),
                                           "gib_in_use": stats.map(|s| gib(s.0)), "gib_cap": stats.map(|s| gib(s.1))}}));
    let stats_stop = Arc::new(AtomicBool::new(false));
    let stats_thread = {
        let (e, stop) = (e.clone(), stats_stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                if let Some((used, cap, free)) = e.device_stats() {
                    emit(json!({"event": "stats", "gib_in_use": gib(used), "gib_cap": gib(cap), "gib_free_card": free.map(gib)}));
                }
                std::thread::sleep(Duration::from_secs(1));
            }
        })
    };
    loop {
        let (job, spec) = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(j) => j,
            Err(mpsc::RecvTimeoutError::Timeout) if !abort.load(Ordering::SeqCst) => continue,
            _ => break,
        };
        cancel.store(false, Ordering::SeqCst);
        current.store(job, Ordering::SeqCst);
        let mut log = |l: String| {
            eprintln!("[job {job}] {l}");
            emit(json!({"event": "log", "job": job, "line": l}));
        };
        let mut progress = |done: usize, total: usize| emit(json!({"event": "progress", "job": job, "done": done, "total": total}));
        let out = e.run_job(&spec, &mut JobCtl { log: &mut log, cancel: &cancel, progress: Some(&mut progress) });
        current.store(0, Ordering::SeqCst);
        match out {
            Ok(v) => emit(json!({"event": "result", "job": job, "ok": true, "value": v})),
            Err(err) => emit(json!({"event": "result", "job": job, "ok": false, "error": err.0, "cancelled": cancel.load(Ordering::SeqCst)})),
        }
        if abort.load(Ordering::SeqCst) {
            break;
        }
    }
    stats_stop.store(true, Ordering::SeqCst);
    let _ = stats_thread.join(); // it holds the engine: gone before the engine can go
    drop(e); // every GPU buffer, then the context; the process ends right after
    emit(json!({"event": "engine", "state": "unloaded"}));
    Ok(())
}

/// `nextsycl video <command>`
pub fn cmd(cfg: &Config, args: &[String], selftest: impl Fn(&[String]) -> Result<(), String>) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    let _ = client::DAEMON.set(nextsycl_serve::http::Target::Unix(service::socket(cfg)));
    match args.first().map(String::as_str) {
        Some("engines") => {
            super::list_engines(engines().iter().map(|k| (k.archs.join(", "), k.name, k.options.to_vec())));
            Ok(())
        }
        Some("selftest") => selftest(rest),
        Some("gpus") => service::gpus(cfg, rest),
        Some("start") => service::start(cfg, rest),
        Some("stop") => {
            let (web, all) = (rest.iter().any(|a| a == "--web"), rest.iter().any(|a| a == "--all"));
            if web || all {
                service::stop_web(cfg);
            }
            if !web || all {
                service::stop(cfg)?;
            }
            Ok(())
        }
        Some("logs") if rest.iter().any(|a| a == "--web") => service::logs_web(cfg),
        Some("logs") => service::logs(cfg),
        Some("serve") => service::serve(cfg, rest),
        Some("speech") => tools::speech(cfg, rest).map_err(|e| e.0),
        Some("scene") => tools::scene(cfg, rest).map_err(|e| e.0),
        Some("join") => tools::join(cfg, rest).map_err(|e| e.0),
        Some("speechpct") => tools::speechpct(rest).map_err(|e| e.0),
        Some("plan") => plan::run(cfg, rest).map_err(|e| e.0),
        Some("studio") => service::studio(rest),
        Some("status") => client::status(rest, &|| service::web_line(cfg) + "\n"),
        Some("ps") => client::ps(rest),
        Some("inspect") => client::inspect(rest),
        Some("cancel") => client::cancel(rest),
        Some("rm") => client::rm(rest),
        Some("unload") => client::unload(rest),
        Some("daemon") => daemon(cfg, rest),
        Some("worker") => worker(cfg, rest),
        Some("job") => {
            let kind = rest.first().filter(|k| !k.starts_with("--")).ok_or(USAGE)?;
            if rest.iter().any(|a| a == "--here") {
                job_here(cfg, kind, &rest[1..])
            } else {
                client::add(rest)
            }
        }
        _ => Err(USAGE.into()),
    }
}
