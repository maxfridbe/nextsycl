//! `nextsycl video ...`: the video commands. The engine's jobs (H3's: generate, encode, decode, denoise,
//! check-block, bench-blocks) run here in-process with `job <kind> --here` (a one-shot: load, run, exit), or go to
//! the daemon's queue; `worker` is the per-GPU process the daemon starts.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use nextsycl_models::{config::Config, registry as models};
use nextsycl_video::{JobCtl, LoadOptions, ModelFiles, VideoEngine, VideoKind};
use serde_json::{json, Value};

const USAGE: &str = "nextsycl video job <kind> [--model ID] [--KEY VALUE]... [--json SPEC|@FILE] [--gpu N] --here
                    a job in this process: load, run, print its result (kinds: nextsycl video engines)
nextsycl video engines | selftest [--gpu N]";

/// The video engines this program has (video/<arch>)
pub fn engines() -> Vec<VideoKind> {
    vec![nextsycl_video_h3::kind(), nextsycl_video_example::kind()]
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
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

/// `nextsycl video <command>`
pub fn cmd(cfg: &Config, args: &[String], selftest: impl Fn(&[String]) -> Result<(), String>) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("engines") => {
            super::list_engines(engines().iter().map(|k| (k.archs.join(", "), k.name, k.options.to_vec())));
            Ok(())
        }
        Some("selftest") => selftest(rest),
        Some("job") => {
            let kind = rest.first().filter(|k| !k.starts_with("--")).ok_or(USAGE)?;
            if rest.iter().any(|a| a == "--here") {
                job_here(cfg, kind, &rest[1..])
            } else {
                Err("the daemon's queue comes next; --here runs the job in this process".into())
            }
        }
        _ => Err(USAGE.into()),
    }
}
