//! `nextsycl audio ...`: the audio commands. `gen` runs the engine in this process (inside the image); `check`
//! compares an engine's stages with a reference's dumps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_audio::{AudioRequest, ModelFiles};
use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

const USAGE: &str = "nextsycl audio gen \"<description>\" [--lyrics TEXT | --lyrics-file FILE] [--model ID | --dir DIR] [--seconds N] [--steps N]
                   [--cfg X] [--seed N] [--set NAME=VALUE]... [--opt-NAME VALUE]... [--out FILE|DIR] [--gpu N]
nextsycl audio gen \"<text to say>\" --model qwen3-tts-... [--voice NAME] [--language L] [--instructions TEXT]
                   [--ref-audio FILE.wav [--ref-text TEXT]] [--seconds N] [--seed N] [--out FILE|DIR]   (speech)
nextsycl audio serve [MODEL] [--port 8087] [--host 127.0.0.1] [--gpu N] [--set NAME=VALUE]... [--wfe] [--out DIR] [--cors ORIGIN]
nextsycl audio start [serve's options]   (in the background) | ps | logs [-f] | stop
nextsycl audio voice list                                   the saved voices (spoken by name: --voice NAME, \"voice\": NAME)
nextsycl audio voice add NAME --sample FILE [--text \"what it says\"] [--description D] [--language L] [--replace]
                   a voice cloned from a recording (WAV, MP3, FLAC, Ogg, M4A; 3-15 s of one speaker)
nextsycl audio voice design NAME --instructions \"the voice\" [--text \"what the sample says\"] [--language L] [--seed N] [--replace]
                   a voice made up from a description (the front door's voice-design model reads a sample)
nextsycl audio voice rm NAME            (addvoice / clonevoice NAME ... = voice add; designvoice = voice design)
nextsycl audio check <dump dir> [--model ID | --dir DIR] [--stages ar,dit,voc,chunks | prompt,frames,codec] [--gpu N]
nextsycl audio engines | selftest [--gpu N]";

/// The audio engines this program has (audio/<arch>)
pub fn engines() -> Vec<nextsycl_audio::AudioKind> {
    vec![nextsycl_audio_minimaxmusic3::kind(), nextsycl_audio_qwen3tts::kind(), nextsycl_audio_voxcpm2::kind(), nextsycl_audio_example::kind()]
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// The registered audio model `id` (or the first one), or `--dir`: a checkpoint's directory as downloaded (MiniMax
/// Music 3's diffusers layout, or Qwen3-TTS's: its config.json and speech_tokenizer/): (entry, files by role)
fn model(cfg: &Config, args: &[String]) -> Result<(Value, ModelFiles), String> {
    if let Some(d) = opt(args, "--dir") {
        let dir = Path::new(d);
        let (arch, files) = if dir.join("audiovae.pth").exists() {
            (nextsycl_audio_voxcpm2::ARCH, nextsycl_audio_voxcpm2::files_in(dir).map_err(|e| e.0)?)
        } else if dir.join("speech_tokenizer").is_dir() {
            (nextsycl_audio_qwen3tts::ARCH, nextsycl_audio_qwen3tts::files_in(dir).map_err(|e| e.0)?)
        } else {
            (nextsycl_audio_minimaxmusic3::ARCH, nextsycl_audio_minimaxmusic3::files_in(dir).map_err(|e| e.0)?)
        };
        let id = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        return Ok((serde_json::json!({"id": id, "kind": "audio", "arch": arch, "env": {}}), files));
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
    let mut voice = opt(args, "--voice").map(str::to_string);
    let reference = match opt(args, "--ref-audio") {
        Some(f) => {
            let b = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
            let (samples, rate) = nextsycl_audio::decode_audio(&b).map_err(|e| format!("{f}: {e}"))?;
            Some(nextsycl_audio::Reference { samples, rate, text: opt(args, "--ref-text").map(str::to_string) })
        }
        // a saved voice: its recording and transcript (for a model that clones)
        None => match (&voice, e.speech()) {
            (Some(v), Some(sp)) if !sp.voices.iter().any(|x| x.eq_ignore_ascii_case(v)) => {
                let lib = nextsycl_serve::voices::Library::of(&out_dir(cfg, &[]));
                if lib.get(v).is_none() {
                    return Err(format!("voice {v}: not one of the model's ({}) nor a saved one (nextsycl audio voice list)", sp.voices.join(", ")));
                }
                if !sp.clone {
                    return Err(format!("voice {v} is a saved voice: speak it with a model that clones (--model qwen3-tts-base)"));
                }
                let r = lib.reference(v)?;
                voice = None;
                Some(r)
            }
            _ => None,
        },
    };
    // the request's own --opt-NAMEs (those taken at load went into the load options)
    let given: nextsycl_core::options::Given = nextsycl_core::options::given(args).into_iter()
        .filter(|(n, _)| k.options.iter().any(|o| (o.name == n.as_str() || o.env == n.as_str()) && o.at != nextsycl_core::At::Load)).collect();
    nextsycl_core::options::resolve(&given, k.options, nextsycl_core::At::Request, k.name)?;
    let req = AudioRequest {
        prompt: prompt.clone(),
        lyrics: lyrics.clone(),
        seconds: num(args, "--seconds")?,
        steps: num(args, "--steps")?,
        cfg: num(args, "--cfg")?,
        seed,
        extra: nextsycl_core::options::by_name(&given, k.options),
        voice: voice.clone(),
        language: opt(args, "--language").map(str::to_string),
        instructions: opt(args, "--instructions").map(str::to_string),
        reference,
    };
    let t0 = std::time::Instant::now();
    // frames a second of sound: speech 12.5, songs 25
    let fps = match e.arch() {
        "voxcpm2" => 6.25,
        _ if e.speech().is_some() => 12.5,
        _ => 25.0,
    };
    let mut phase = "";
    let mut mark = 0f64;
    // each phase's time (the time up to a report is its phase's) and its last count, in the order first seen
    let mut spans: Vec<(&'static str, f64, u32)> = Vec::new();
    let mut last = 0f64;
    let audio = e.generate(&req, &mut |s| {
        if s.phase != phase {
            if !phase.is_empty() {
                eprintln!();
            }
            phase = s.phase;
            mark = s.seconds;
        }
        let i = spans.iter().position(|x| x.0 == s.phase).unwrap_or_else(|| {
            spans.push((s.phase, 0.0, 0));
            spans.len() - 1
        });
        spans[i].1 += s.seconds - last;
        spans[i].2 = s.at;
        last = s.seconds;
        let rate = if s.phase == "tokens" && s.seconds > mark { s.at as f64 / (s.seconds - mark) } else { 0.0 };
        match s.phase {
            "tokens" => eprint!("\rtokens {}/{} ({:.1} s of sound)  {rate:.1} frames/s  {:.0} s   ", s.at, s.of, s.at as f64 / fps, s.seconds),
            _ => eprint!("\r{} {}/{}  {:.0} s   ", s.phase, s.at, s.of, s.seconds),
        }
        Ok(())
    }).map_err(|e| e.0)?;
    eprintln!();
    let parts: Vec<String> = spans.iter().map(|(p, t, n)| {
        let t = *t;
        match *p {
            "tokens" => format!("tokens {n} frames in {t:.1} s ({:.1} frames/s)", *n as f64 / t.max(1e-9)),
            "flow" => format!("flow {n} steps in {t:.1} s"),
            "decode" => format!("decode {n} windows in {t:.1} s"),
            p => format!("{p} {t:.1} s"),
        }
    }).collect();
    eprintln!("{}", parts.join(" · "));
    let id = m["id"].as_str().unwrap_or("audio");
    let name = format!("{id}-{seed}.wav");
    let path = match opt(args, "--out").map(PathBuf::from) {
        Some(o) if o.is_dir() => o.join(name),
        Some(o) => o,
        None => PathBuf::from(name),
    };
    let comment = match e.speech() {
        Some(_) => format!("model {id}, seed {seed}, voice {}, language {}, instructions {}", req.voice.as_deref().unwrap_or("-"),
                           req.language.as_deref().unwrap_or("auto"), req.instructions.as_deref().unwrap_or("-")),
        None => format!("model {id}, seed {seed}, steps {}, cfg {}", req.steps.unwrap_or(e.defaults().steps), req.cfg.unwrap_or(e.defaults().cfg)),
    };
    let lyr = lyrics.unwrap_or_default();
    audio.write_wav(&path, &[("INAM", &prompt), ("ICMT", &comment), ("ILYR", &lyr)])?;
    println!("saved {}  ({:.1} s of sound in {:.1} s)", path.display(), audio.seconds(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn check(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = Path::new(args.first().filter(|a| !a.starts_with("--")).ok_or(USAGE)?);
    nextsycl_core::use_kind("audio");
    let (m, files) = model(cfg, args)?;
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let worst = if m["arch"] == nextsycl_audio_qwen3tts::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("prompt,frames,codec").split(',').collect();
        let e = nextsycl_audio_qwen3tts::Qwen3Tts::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_qwen3tts::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_voxcpm2::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("prompt,prefill,patches,vae").split(',').collect();
        let e = nextsycl_audio_voxcpm2::VoxCpm2::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_voxcpm2::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_minimaxmusic3::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("ar,dit,voc").split(',').collect();
        let e = nextsycl_audio_minimaxmusic3::MiniMaxMusic3::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_minimaxmusic3::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else {
        return Err(format!("check knows {}, {} and {} only", nextsycl_audio_minimaxmusic3::ARCH, nextsycl_audio_qwen3tts::ARCH, nextsycl_audio_voxcpm2::ARCH));
    };
    println!("worst relative error {worst:.2e}");
    Ok(())
}

/// The music server's container (`audio serve` in the foreground, `audio start` in the background)
const SERVED: crate::served::Served = crate::served::Served { name: "nextsycl-audio", kind: "audio", port: "8087" };

/// Where songs go: --out, NS_AUDIO_OUT, ~/.local/share/nextsycl/audio
fn out_dir(cfg: &Config, args: &[String]) -> PathBuf {
    opt(args, "--out").map(PathBuf::from).or_else(|| cfg.get("NS_AUDIO_OUT").map(PathBuf::from)).unwrap_or_else(|| {
        PathBuf::from(format!("{}/.local/share/nextsycl/audio", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())))
    })
}

/// `nextsycl audio serve`: from the host, the build image as container `nextsycl-audio` (the GPU, the program and its
/// libraries, the registry, the model's files read-only, the output directory) running this command inside with
/// `--here`; there, the engine loaded and served (nextsycl_serve::audio)
fn serve(cfg: &Config, args: &[String], detach: bool) -> Result<(), String> {
    let id = args.first().filter(|a| !a.starts_with("--")).map(String::as_str);
    let mut margs = args.to_vec();
    if let Some(i) = id {
        margs.extend(["--model".to_string(), i.to_string()]);
    }
    let (m, files) = model(cfg, &margs)?;
    let model_id = m["id"].as_str().unwrap_or("audio").to_string();
    let out = out_dir(cfg, args);
    if args.iter().any(|a| a == "--here") {
        return serve_here(&m, files, args, out, cfg);
    }
    use crate::container::{mount, Ce};
    // the engine options checked here, before a container starts (they travel inside with the other arguments)
    options(&m, args)?;
    let ce = Ce::new(cfg)?;
    ce.need_image()?;
    for f in ["nextsycl", "libnextsycl-audio.so"] {
        if !cfg.dist.join(f).exists() {
            return Err(format!("{}: not built yet (./build.sh)", cfg.dist.join(f).display()));
        }
    }
    if args.iter().any(|a| a == "--wfe") && !cfg.dist.join("wfe/audio/index.html").exists() {
        return Err(format!("{}: the web front end is not built yet (./build.sh wfe)", cfg.dist.join("wfe").display()));
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    const NAME: &str = SERVED.name;
    if ce.running(NAME) {
        return Err(format!("{NAME} is running already ({} stop {NAME})", ce.bin));
    }
    ce.remove(NAME);
    // --init: the server is not PID 1, so a stop's SIGTERM ends it (PID 1 ignores signals it does not handle)
    let mut a = SERVED.run_args(detach, args);
    a.extend(ce.user_args());
    a.extend(ce.gpu_args());
    a.extend(mount(&cfg.dist, "/app", true));
    let reg = models::registry(cfg);
    let mut dirs = std::collections::BTreeSet::new();
    let mut paths: Vec<PathBuf> = files.values().cloned().collect();
    if let Some(d) = opt(args, "--dir") {
        paths.push(PathBuf::from(d));
    }
    paths.push(reg.clone());
    for p in paths {
        let d = if p.is_dir() { p } else { p.parent().map(PathBuf::from).unwrap_or_default() };
        if !d.as_os_str().is_empty() && d.exists() && dirs.insert(d.clone()) {
            a.extend(mount(&d, &d.to_string_lossy(), true));
        }
    }
    a.extend(mount(&out, &out.to_string_lossy(), false));
    a.extend(["-e".into(), format!("NS_REGISTRY={}", reg.display()), "-e".into(), "ONEAPI_DEVICE_SELECTOR=level_zero:*".into()]);
    // the engines' own settings from the configuration (NS_MM3_INT8=1 ...)
    for (k, v) in cfg.with_prefix("NS_MM3_").into_iter().chain(cfg.with_prefix("NS_Q3T_")).chain(cfg.with_prefix("NS_VCP_")).chain(cfg.with_prefix("NSD_")) {
        a.extend(["-e".into(), format!("{k}={v}")]);
    }
    a.extend([ce.image.clone(), "bash".into(), "-c".into(),
              "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl audio serve \"$@\"".into(), "nextsycl".into()]);
    if opt(args, "--dir").is_none() {
        a.push(model_id);
    }
    let mut it = args.iter().skip(if id.is_some() { 1 } else { 0 });
    while let Some(x) = it.next() {
        if x == "--out" || x == "--model" {
            it.next();
            continue;
        }
        a.push(x.clone());
    }
    a.extend(["--out".into(), out.to_string_lossy().into_owned(), "--here".into()]);
    SERVED.launch(&ce, &a, detach)
}

fn serve_here(m: &Value, files: ModelFiles, args: &[String], out: PathBuf, cfg: &Config) -> Result<(), String> {
    nextsycl_core::use_kind("audio");
    let ks = engines();
    let k = nextsycl_audio::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = (k.load)(&files, std::slice::from_ref(&g), &options(m, args)?, &mut log).map_err(|e| e.0)?;
    let wfe = args.iter().any(|a| a == "--wfe").then(|| cfg.dist.join("wfe"));
    let cors: Vec<String> = opt(args, "--cors").map(str::to_string).or(cfg.get("NS_CORS")).map(|c| c.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let addr = format!("{}:{}", opt(args, "--host").unwrap_or("127.0.0.1"), opt(args, "--port").unwrap_or("8087"));
    let mut srv = nextsycl_serve::audio::AudioServer::new(m["id"].as_str().unwrap_or("audio").to_string(), g, e, Some(out), wfe, cors)?;
    srv.idle = crate::served::idle_exit(args)?;
    Arc::new(srv).run(&addr)
}

/// `nextsycl audio voice ...`: the saved voices (beside the audio output: list, add and remove here; design through
/// the front door, whose voice-design model reads the sample)
fn voice(cfg: &Config, args: &[String]) -> Result<(), String> {
    let lib = nextsycl_serve::voices::Library::of(&out_dir(cfg, args));
    let name = args.get(1).filter(|a| !a.starts_with("--")).map(String::as_str).or_else(|| opt(args, "--name")).unwrap_or("");
    let replace = args.iter().any(|a| a == "--replace");
    match args.first().map(String::as_str) {
        Some("list" | "ls") | None => {
            let list = lib.list();
            if list.is_empty() {
                println!("no saved voices ({}); nextsycl audio voice add NAME --sample FILE", lib.dir.display());
            }
            for v in list {
                let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
                println!("{:<16} {:<6} {:>5.1} s  {}{}", s("name"), s("kind"), v["seconds"].as_f64().unwrap_or(0.0),
                         if s("description").is_empty() { String::new() } else { format!("{} - ", s("description")) },
                         if s("text").is_empty() { "(no transcript: an x-vector clone)".into() } else { format!("\"{}\"", s("text")) });
            }
            Ok(())
        }
        Some("add") => {
            let f = opt(args, "--sample").ok_or("--sample FILE: the recording to clone")?;
            let b = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
            let v = lib.add(name, &b, opt(args, "--text"), opt(args, "--description"), opt(args, "--language"), "clone",
                            &format!("a recording ({})", Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()), replace)?;
            println!("saved voice {name}: {:.1} s at {} Hz{}", v["seconds"].as_f64().unwrap_or(0.0), v["rate"],
                     if v["text"].is_null() { " (no --text: cloned from its timbre alone; with the transcript the clone is closer)" } else { "" });
            Ok(())
        }
        Some("design") => {
            // through the front door: it loads the voice-design model and saves what it reads
            let ins = opt(args, "--instructions").or_else(|| opt(args, "--description")).ok_or("--instructions \"the voice: age, gender, timbre, pace ...\"")?;
            let mut b = serde_json::json!({"name": name, "instructions": ins, "replace": replace});
            for (k, f) in [("text", "--text"), ("language", "--language"), ("model", "--model")] {
                if let Some(v) = opt(args, f) {
                    b[k] = serde_json::json!(v);
                }
            }
            if let Some(n) = num::<u64>(args, "--seed")? {
                b["seed"] = serde_json::json!(n);
            }
            let reg = models::registry(cfg);
            let token = cfg.get("NS_API_TOKEN").or_else(|| {
                std::fs::read_to_string(reg.with_file_name("serve.json")).ok()
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v["token"].as_str().map(str::to_string))
            }).ok_or("the front door's token (NS_API_TOKEN, or serve.json beside the registry): is nextsycl serve running?")?;
            let port = cfg.get("NS_SERVE_PORT").unwrap_or_else(|| "8000".into());
            let t = nextsycl_serve::http::Target::Tcp(format!("127.0.0.1:{port}"));
            eprintln!("designing {name} through the front door (:{port}) ...");
            let r = nextsycl_serve::http::rpc(&t, &token, "audio.voices.design", &b, 900)?;
            println!("saved voice {name}: {:.1} s, seed {}  - \"{}\"", r["voice"]["seconds"].as_f64().unwrap_or(0.0), r["seed"],
                     r["voice"]["text"].as_str().unwrap_or(""));
            Ok(())
        }
        Some("rm" | "remove" | "delete") => {
            lib.remove(name)?;
            println!("removed voice {name}");
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
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
        Some("voice" | "voices") => voice(cfg, rest),
        Some("addvoice" | "clonevoice") => voice(cfg, &[&["add".to_string()], rest].concat()),
        Some("designvoice") => voice(cfg, &[&["design".to_string()], rest].concat()),
        Some("serve") => serve(cfg, rest, false),
        Some("start") => SERVED.start(cfg, rest, |d| serve(cfg, rest, d)),
        Some("ps") => SERVED.ps(cfg),
        Some("stop") => SERVED.stop(cfg),
        Some("logs") => SERVED.logs(cfg, rest),
        _ => Err(USAGE.into()),
    }
}
