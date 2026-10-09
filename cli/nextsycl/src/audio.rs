//! `nextsycl audio ...`: the audio commands. `gen` runs the engine in this process (inside the image); `check`
//! compares an engine's stages with a reference's dumps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_audio::{AudioRequest, ModelFiles};
use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

const USAGE: &str = "nextsycl audio gen \"<description>\" [--lyrics TEXT | --lyrics-file FILE] [--model ID | --dir DIR] [--seconds N] [--steps N]
                   [--cfg X] [--seed N] [--set NAME=VALUE]... [--out FILE|DIR] [--gpu N]
nextsycl audio check <dump dir> [--model ID | --dir DIR] [--stages ar,dit,voc,chunks] [--gpu N]
nextsycl audio engines | selftest [--gpu N]";

/// The audio engines this program has (audio/<arch>)
pub fn engines() -> Vec<nextsycl_audio::AudioKind> {
    vec![nextsycl_audio_minimaxmusic3::kind(), nextsycl_audio_example::kind()]
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// The registered audio model `id` (or the first one), or `--dir`: a MiniMax Music 3 checkpoint's directory as
/// downloaded (its diffusers layout): (entry, files by role)
fn model(cfg: &Config, args: &[String]) -> Result<(Value, ModelFiles), String> {
    if let Some(d) = opt(args, "--dir") {
        let files = nextsycl_audio_minimaxmusic3::files_in(Path::new(d)).map_err(|e| e.0)?;
        let id = Path::new(d).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        return Ok((serde_json::json!({"id": id, "kind": "audio", "arch": nextsycl_audio_minimaxmusic3::ARCH, "env": {}}), files));
    }
    let all = models::all(cfg)?;
    let m = match opt(args, "--model") {
        Some(id) => all.into_iter().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id} (nextsycl models search --kind audio)"))?,
        None => all.into_iter().find(|m| models::kind_of(m) == "audio").ok_or("no audio model (nextsycl models pull minimax-music3)")?,
    };
    if models::kind_of(&m) != "audio" {
        return Err(format!("{} is a {} model, not an audio one", m["id"].as_str().unwrap_or("?"), models::kind_of(&m)));
    }
    let files: ModelFiles = m["files"].as_object().into_iter().flatten().filter_map(|(r, p)| Some((r.clone(), PathBuf::from(p.as_str()?)))).collect();
    Ok((m, files))
}

/// The load options: the registry entry's settings, `--set NAME=VALUE`s, the --opt-NAMEs taken at load
fn options(m: &Value, args: &[String]) -> Result<nextsycl_audio::LoadOptions, String> {
    let mut settings = std::collections::BTreeMap::new();
    for (k, v) in m["env"].as_object().into_iter().flatten() {
        if let Some(v) = v.as_str() {
            settings.insert(k.clone(), v.to_string());
        }
    }
    for (i, a) in args.iter().enumerate() {
        if a == "--set" {
            let (k, v) = args.get(i + 1).and_then(|x| x.split_once('=')).ok_or("--set NAME=VALUE")?;
            settings.insert(k.to_string(), v.to_string());
        }
    }
    let ks = engines();
    let k = nextsycl_audio::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    for (var, v) in super::opt_env(k.options, nextsycl_core::At::Load, k.name)? {
        settings.insert(var, v);
    }
    Ok(nextsycl_audio::LoadOptions { settings })
}

fn gpu(args: &[String]) -> Result<Arc<nextsycl_core::Gpu>, String> {
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    nextsycl_core::Gpu::open(i).map_err(|e| e.0)
}

fn num<T: std::str::FromStr>(args: &[String], k: &str) -> Result<Option<T>, String> {
    opt(args, k).map(|s| s.parse().map_err(|_| format!("{k}: {s}?"))).transpose()
}

fn gen(cfg: &Config, args: &[String]) -> Result<(), String> {
    let prompt = args.first().filter(|p| !p.starts_with("--")).ok_or(USAGE)?.clone();
    let lyrics = match (opt(args, "--lyrics"), opt(args, "--lyrics-file")) {
        (Some(l), _) => Some(l.replace("\\n", "\n")),
        (None, Some(f)) => Some(std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?),
        (None, None) => None,
    };
    nextsycl_core::use_kind("audio");
    let (m, files) = model(cfg, args)?;
    let ks = engines();
    let k = nextsycl_audio::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = (k.load)(&files, &[g], &options(&m, args)?, &mut log).map_err(|e| e.0)?;
    let seed: u64 = num(args, "--seed")?.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_nanos() as u64 % 100_000).unwrap_or(0)
    });
    let req = AudioRequest {
        prompt: prompt.clone(),
        lyrics: lyrics.clone(),
        seconds: num(args, "--seconds")?,
        steps: num(args, "--steps")?,
        cfg: num(args, "--cfg")?,
        seed,
        extra: nextsycl_core::options::Given::default(),
    };
    let t0 = std::time::Instant::now();
    let mut phase = "";
    let mut mark = 0f64;
    let audio = e.generate(&req, &mut |s| {
        if s.phase != phase {
            if !phase.is_empty() {
                eprintln!();
            }
            phase = s.phase;
            mark = s.seconds;
        }
        let rate = if s.phase == "tokens" && s.seconds > mark { s.at as f64 / (s.seconds - mark) } else { 0.0 };
        match s.phase {
            "tokens" => eprint!("\rtokens {}/{} ({:.1} s of sound)  {rate:.1} frames/s  {:.0} s   ", s.at, s.of, s.at as f64 / 25.0, s.seconds),
            _ => eprint!("\r{} {}/{}  {:.0} s   ", s.phase, s.at, s.of, s.seconds),
        }
        Ok(())
    }).map_err(|e| e.0)?;
    eprintln!();
    let id = m["id"].as_str().unwrap_or("audio");
    let name = format!("{id}-{seed}.wav");
    let path = match opt(args, "--out").map(PathBuf::from) {
        Some(o) if o.is_dir() => o.join(name),
        Some(o) => o,
        None => PathBuf::from(name),
    };
    let comment = format!("model {id}, seed {seed}, steps {}, cfg {}", req.steps.unwrap_or(e.defaults().steps), req.cfg.unwrap_or(e.defaults().cfg));
    let lyr = lyrics.unwrap_or_default();
    audio.write_wav(&path, &[("INAM", &prompt), ("ICMT", &comment), ("ILYR", &lyr)])?;
    println!("saved {}  ({:.1} s of sound in {:.1} s)", path.display(), audio.seconds(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn check(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = Path::new(args.first().filter(|a| !a.starts_with("--")).ok_or(USAGE)?);
    nextsycl_core::use_kind("audio");
    let (m, files) = model(cfg, args)?;
    if m["arch"] != nextsycl_audio_minimaxmusic3::ARCH {
        return Err(format!("check knows {} only", nextsycl_audio_minimaxmusic3::ARCH));
    }
    let stages: Vec<&str> = opt(args, "--stages").unwrap_or("ar,dit,voc").split(',').collect();
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = nextsycl_audio_minimaxmusic3::MiniMaxMusic3::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
    let worst = nextsycl_audio_minimaxmusic3::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?;
    println!("worst relative error {worst:.2e}");
    Ok(())
}

/// `nextsycl audio <command>`
pub fn cmd(cfg: &Config, args: &[String], selftest: impl Fn(&[String]) -> Result<(), String>) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("engines") => {
            super::list_engines(engines().iter().map(|k| (k.archs.join(", "), k.name, k.options.to_vec())));
            Ok(())
        }
        Some("selftest") => selftest(rest),
        Some("gen") => gen(cfg, rest),
        Some("check") => check(cfg, rest),
        _ => Err(USAGE.into()),
    }
}
