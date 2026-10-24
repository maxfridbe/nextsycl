//! `nextsycl audio ...`: the audio commands. `gen` runs the engine in this process (inside the image); `check`
//! compares an engine's stages with a reference's dumps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_audio::{AudioRequest, ModelFiles};
use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

const USAGE: &str = "nextsycl audio gen \"<description>\" [--lyrics TEXT | --lyrics-file FILE] [--model ID | --dir DIR] [--seconds N] [--steps N]
                   [--cfg X] [--seed N] [--set NAME=VALUE]... [--opt-NAME VALUE]... [--out FILE|DIR] [--gpu N | --gpus A,B]
nextsycl audio gen \"<text to say>\" --model qwen3-tts-... [--voice NAME] [--language L] [--instructions TEXT]
                   [--ref-audio FILE.wav [--ref-text TEXT]] [--cfg X] [--seconds N] [--seed N] [--out FILE|DIR]   (speech)
nextsycl audio gen \"Speaker 1: ...\\nSpeaker 2: ...\" --model vibevoice-1.5b [--ref-audio S1.wav --ref-audio S2.wav ... |
                   --voices NAME,NAME ...]   (a multi-speaker script: a recording or saved voice a speaker, in order)
nextsycl audio gen \"<description>\" --model acestep-1.5 --source FILE --opt-task cover | repaint | edit | lego [--opt-repaint-start S]
                   [--opt-repaint-end S] [--opt-cover-strength X] [--opt-source-lyrics TEXT] ...   (a song made from a recording)
nextsycl audio serve [MODEL] [--port 8087] [--host 127.0.0.1] [--gpu N | --gpus A,B] [--set NAME=VALUE]... [--wfe] [--out DIR] [--cors ORIGIN]
nextsycl audio start [serve's options] [--slot N]   (in the background; slot N: another server beside the first) | ps | logs [-f] | stop  [--slot N]
nextsycl audio voice list                                   the saved voices (spoken by name: --voice NAME, \"voice\": NAME)
nextsycl audio voice add NAME --sample FILE [--text \"what it says\"] [--description D] [--language L] [--replace]
                   a voice cloned from a recording (WAV, MP3, FLAC, Ogg, M4A; 3-15 s of one speaker)
nextsycl audio voice design NAME --instructions \"the voice\" [--text \"what the sample says\"] [--language L] [--seed N] [--replace]
                   a voice made up from a description (the front door's voice-design model reads a sample)
nextsycl audio voice enroll NAME [--gpu N] [--steps 2000] [--replace] [--direct]
                   a saved voice fitted for Voxtral TTS (experimental: no encoder in its open weights; PyTorch XPU) - a front-door
                   job that waits for a card like a load (audio.voices.enroll); --direct: here, now (the door runs that)
nextsycl audio voice rm NAME            (addvoice / clonevoice NAME ... = voice add; designvoice = voice design)
nextsycl audio transcribe FILE [--model ID | --dir DIR] [--language L] [--translate] [--no-words] [--prompt TEXT] [--opt-NAME VALUE]...
                   [--json] [--gpu N]   (speech to text: a recogniser, whisper-large-v3-turbo; any rate, WAV, MP3, FLAC, Ogg, M4A)
nextsycl audio check <dump dir> [--model ID | --dir DIR] [--stages ar,dit,voc,chunks | prompt,frames,codec | mel,encoder,logits,transcribe | prompt,lm,head,codec,run | prompt,frames,flow,codec] [--gpu N]
nextsycl audio engines | selftest [--gpu N]";

/// The audio engines this program has (audio/<arch>)
pub fn engines() -> Vec<nextsycl_audio::AudioKind> {
    vec![nextsycl_audio_minimaxmusic3::kind(), nextsycl_audio_qwen3tts::kind(), nextsycl_audio_voxcpm2::kind(), nextsycl_audio_breezetts2::kind(),
         nextsycl_audio_vibevoice::kind(), nextsycl_audio_s2pro::kind(),
         nextsycl_audio_voxtraltts::kind(), nextsycl_audio_whisper::kind(), nextsycl_audio_acestep::kind(), nextsycl_audio_example::kind()]
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// The registered audio model `id` (or the first one), or `--dir`: a checkpoint's directory as downloaded (MiniMax
/// Music 3's diffusers layout, or Qwen3-TTS's: its config.json and speech_tokenizer/): (entry, files by role)
fn model(cfg: &Config, args: &[String]) -> Result<(Value, ModelFiles), String> {
    if let Some(d) = opt(args, "--dir") {
        let dir = Path::new(d);
        let whisper = std::fs::read_to_string(dir.join("config.json")).is_ok_and(|c| c.contains("\"model_type\": \"whisper\""));
        let (arch, files) = if whisper {
            (nextsycl_audio_whisper::ARCH, nextsycl_audio_whisper::files_in(dir).map_err(|e| e.0)?)
        } else if dir.join("acestep-v15-turbo").is_dir() {
            (nextsycl_audio_acestep::ARCH, nextsycl_audio_acestep::files_in(dir).map_err(|e| e.0)?)
        } else if let Ok(f) = nextsycl_audio_breezetts2::files_in(dir) {
            (nextsycl_audio_breezetts2::ARCH, f)
        } else if let Ok(f) = nextsycl_audio_vibevoice::files_in(dir) {
            (nextsycl_audio_vibevoice::ARCH, f)
        } else if let Ok(f) = nextsycl_audio_s2pro::files_in(dir) {
            (nextsycl_audio_s2pro::ARCH, f)
        } else if let Ok(f) = nextsycl_audio_voxtraltts::files_in(dir) {
            (nextsycl_audio_voxtraltts::ARCH, f)
        } else if dir.join("audiovae.pth").exists() {
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
    // not while its files move to another drive (nextsycl models move)
    crate::drives::not_moving(cfg, m["id"].as_str().unwrap_or(""))?;
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
    // the --opt-NAMEs for the load: a request's own (a song's bpm, task ...) are left for the request
    let request_only = |n: &str| k.options.iter().any(|o| (o.name == n || o.env == n) && o.at == nextsycl_core::At::Request);
    let load: nextsycl_core::options::Given = super::engine_opts().iter().filter(|(n, _)| !request_only(n)).map(|(a, b)| (a.clone(), b.clone())).collect();
    for (var, v) in nextsycl_core::options::resolve(&load, k.options, nextsycl_core::At::Load, k.name)? {
        settings.insert(var, v);
    }
    Ok(nextsycl_audio::LoadOptions { settings })
}

fn gpu(args: &[String]) -> Result<Arc<nextsycl_core::Gpu>, String> {
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    nextsycl_core::Gpu::open(i).map_err(|e| e.0)
}

/// The GPUs an engine loads on: `--gpus A,B` (an engine that splits itself across cards: MiniMax Music's language
/// model on A, its flow and decoder on B), else `--gpu N`
fn gpus(args: &[String]) -> Result<Vec<Arc<nextsycl_core::Gpu>>, String> {
    match opt(args, "--gpus") {
        Some(l) => l.split(',').map(|x| {
            let i: usize = x.trim().parse().map_err(|_| format!("--gpus {l}: GPU indices, comma-separated"))?;
            nextsycl_core::Gpu::open(i).map_err(|e| e.0)
        }).collect(),
        None => Ok(vec![gpu(args)?]),
    }
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
    let g = gpus(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = (k.load)(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
    let seed: u64 = num(args, "--seed")?.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_nanos() as u64 % 100_000).unwrap_or(0)
    });
    let mut voice = opt(args, "--voice").map(str::to_string);
    // a multi-speaker script's voices: --ref-audio once a speaker (speaker 1's first), or --voices NAME,NAME (saved)
    let recordings: Vec<&str> = args.windows(2).filter(|w| w[0] == "--ref-audio").map(|w| w[1].as_str()).collect();
    let read_ref = |f: &str, text: Option<String>| -> Result<nextsycl_audio::Reference, String> {
        let b = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
        let (samples, rate) = nextsycl_audio::decode_audio(&b).map_err(|e| format!("{f}: {e}"))?;
        Ok(nextsycl_audio::Reference { samples, rate, text })
    };
    let mut references = Vec::new();
    for f in recordings.iter().skip(1) {
        references.push(read_ref(f, None)?);
    }
    let mut saved: Vec<String> = opt(args, "--voices").map(|v| v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    if !saved.is_empty() {
        if !recordings.is_empty() || voice.is_some() {
            return Err("--voices names every speaker's saved voice: not with --voice or --ref-audio".into());
        }
        if e.speech().is_none_or(|sp| sp.speakers < saved.len()) {
            return Err(format!("--voices: {} voices, and {} speaks {} at most", saved.len(), m["id"].as_str().unwrap_or("this model"),
                               e.speech().map_or(0, |sp| sp.speakers.max(1))));
        }
        let lib = nextsycl_serve::voices::Library::of(&out_dir(cfg, &[]));
        for v in saved.iter().skip(1) {
            references.push(lib.reference(v)?);
        }
        voice = Some(saved.remove(0));
    }
    if !references.is_empty() && e.speech().is_none_or(|sp| sp.speakers < references.len() + 1) {
        return Err(format!("{} recordings, and {} speaks one voice (a model of several speakers: vibevoice-1.5b)", references.len() + 1,
                           m["id"].as_str().unwrap_or("this model")));
    }
    let reference = match recordings.first() {
        Some(f) => Some(read_ref(f, opt(args, "--ref-text").map(str::to_string))?),
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
        references,
        source: match opt(args, "--source") {
            Some(f) => Some(nextsycl_audio::read_audio(&std::fs::read(f).map_err(|e| format!("{f}: {e}"))?).map_err(|e| format!("{f}: {e}"))?),
            None => None,
        },
    };
    if req.source.is_some() && !e.sources() {
        return Err(format!("--source: {} makes no song from a recording (acestep does: --opt-task cover | repaint | edit)", e.arch()));
    }
    let t0 = std::time::Instant::now();
    // frames a second of sound: speech 12.5 (S2 Pro 21.5), songs 25
    let fps = match e.arch() {
        "voxcpm2" => 6.25,
        "vibevoice" => 7.5,
        nextsycl_audio_s2pro::ARCH => nextsycl_audio_s2pro::FPS as f64,
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

/// `nextsycl audio transcribe FILE`: the recording written down by a recogniser in this process (inside the image)
fn transcribe(cfg: &Config, args: &[String]) -> Result<(), String> {
    let file = args.first().filter(|p| !p.starts_with("--")).ok_or(USAGE)?.clone();
    nextsycl_core::use_kind("audio");
    let mut margs = args.to_vec();
    let all = models::all(cfg)?;
    // no model named: the first registered recogniser
    if opt(args, "--model").is_none() && opt(args, "--dir").is_none() {
        let m = all.iter().find(|m| m["arch"] == nextsycl_audio_whisper::ARCH).ok_or("no recogniser registered (nextsycl models pull whisper-large-v3-turbo)")?;
        margs.extend(["--model".to_string(), m["id"].as_str().unwrap_or("").to_string()]);
    }
    let (m, files) = model(cfg, &margs)?;
    let ks = engines();
    let k = nextsycl_audio::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    let gs = gpus(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = (k.load)(&files, &gs, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
    if e.recogniser().is_none() {
        return Err(format!("{} makes sound; transcribe needs a recogniser (--model whisper-large-v3-turbo)", m["id"].as_str().unwrap_or("?")));
    }
    let b = std::fs::read(&file).map_err(|e| format!("{file}: {e}"))?;
    let (samples, rate) = nextsycl_audio::decode_audio(&b).map_err(|e| format!("{file}: {e}"))?;
    let given: nextsycl_core::options::Given = nextsycl_core::options::given(args).into_iter()
        .filter(|(n, _)| k.options.iter().any(|o| (o.name == n.as_str() || o.env == n.as_str()) && o.at != nextsycl_core::At::Load)).collect();
    nextsycl_core::options::resolve(&given, k.options, nextsycl_core::At::Request, k.name)?;
    let req = nextsycl_audio::TranscribeRequest {
        samples,
        rate,
        language: opt(args, "--language").map(str::to_string),
        translate: args.iter().any(|a| a == "--translate"),
        words: !args.iter().any(|a| a == "--no-words"),
        prompt: opt(args, "--prompt").map(str::to_string),
        extra: nextsycl_core::options::by_name(&given, k.options),
    };
    let t0 = std::time::Instant::now();
    let t = e.transcribe(&req, &mut |s| {
        eprint!("\r{} {}/{} frames  {:.1} s   ", s.phase, s.at, s.of, s.seconds);
        Ok(())
    }).map_err(|e| e.0)?;
    eprintln!();
    let took = t0.elapsed().as_secs_f64();
    if args.iter().any(|a| a == "--json") {
        let timed = |v: &[nextsycl_audio::Timed]| -> Vec<serde_json::Value> {
            v.iter().map(|x| serde_json::json!({"start": x.start, "end": x.end, "text": x.text, "probability": x.probability})).collect()
        };
        println!("{}", serde_json::json!({"text": t.text, "language": t.language, "duration": t.duration, "took": took,
                                          "segments": timed(&t.segments), "words": timed(&t.words)}));
    } else {
        for s in &t.segments {
            println!("[{:8.2} - {:8.2}] {}", s.start, s.end, s.text);
        }
        eprintln!("{:.1} s of recording ({}) in {took:.1} s, {} segments, {} words", t.duration, t.language, t.segments.len(), t.words.len());
    }
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
    } else if m["arch"] == nextsycl_audio_breezetts2::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("prompt,frames,codec").split(',').collect();
        let e = nextsycl_audio_breezetts2::BreezeTts2::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_breezetts2::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_vibevoice::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("prompt,lm,head,codec,run").split(',').collect();
        let e = nextsycl_audio_vibevoice::VibeVoice::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_vibevoice::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_s2pro::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("prompt,frames,codec").split(',').collect();
        let e = nextsycl_audio_s2pro::S2Pro::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_s2pro::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_voxtraltts::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("prompt,frames,flow,codec").split(',').collect();
        let e = nextsycl_audio_voxtraltts::VoxtralTts::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_voxtraltts::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_minimaxmusic3::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("ar,dit,voc").split(',').collect();
        let e = nextsycl_audio_minimaxmusic3::MiniMaxMusic3::load_on(&files, &gpus(args)?, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_minimaxmusic3::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_acestep::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("text,te,enc,velocity,sample,vae").split(',').collect();
        let e = nextsycl_audio_acestep::AceStep::load(&files, &gpus(args)?, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_acestep::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else if m["arch"] == nextsycl_audio_whisper::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("mel,encoder,logits,transcribe").split(',').collect();
        let e = nextsycl_audio_whisper::Whisper::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
        nextsycl_audio_whisper::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?
    } else {
        return Err(format!("check knows {}, {}, {}, {}, {}, {}, {}, {} and {} only", nextsycl_audio_minimaxmusic3::ARCH, nextsycl_audio_qwen3tts::ARCH,
                           nextsycl_audio_voxcpm2::ARCH, nextsycl_audio_breezetts2::ARCH, nextsycl_audio_vibevoice::ARCH, nextsycl_audio_s2pro::ARCH,
                           nextsycl_audio_voxtraltts::ARCH, nextsycl_audio_whisper::ARCH, nextsycl_audio_acestep::ARCH));
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
    let name = SERVED.name_for(args)?;
    if ce.running(&name) {
        return Err(format!("{name} is running already ({} stop {name})", ce.bin));
    }
    ce.remove(&name);
    // --init: the server is not PID 1, so a stop's SIGTERM ends it (PID 1 ignores signals it does not handle)
    let mut a = SERVED.run_args(detach, args)?;
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
    for (k, v) in cfg.with_prefix("NS_MM3_").into_iter().chain(cfg.with_prefix("NS_Q3T_")).chain(cfg.with_prefix("NS_VCP_")).chain(cfg.with_prefix("NS_BZ2_")).chain(cfg.with_prefix("NS_WH_"))
        .chain(cfg.with_prefix("NSD_")) {
        a.extend(["-e".into(), format!("{k}={v}")]);
    }
    a.extend([ce.image.clone(), "bash".into(), "-c".into(),
              "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl audio serve \"$@\"".into(), "nextsycl".into()]);
    if opt(args, "--dir").is_none() {
        a.push(model_id);
    }
    let mut it = args.iter().skip(if id.is_some() { 1 } else { 0 });
    while let Some(x) = it.next() {
        if x == "--out" || x == "--model" || x == "--slot" {
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
    let gs = gpus(args)?;
    let g = gs[0].clone();
    let mut log = |s: String| eprintln!("{s}");
    let e = (k.load)(&files, &gs, &options(m, args)?, &mut log).map_err(|e| e.0)?;
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
                let enrolled: Vec<&str> = v["enrolled"].as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect();
                println!("{:<16} {:<6} {:>5.1} s  {}{}{}", s("name"), s("kind"), v["seconds"].as_f64().unwrap_or(0.0),
                         if s("description").is_empty() { String::new() } else { format!("{} - ", s("description")) },
                         if s("text").is_empty() { "(no transcript: an x-vector clone)".into() } else { format!("\"{}\"", s("text")) },
                         if enrolled.is_empty() { String::new() } else { format!("  [enrolled: {}]", enrolled.join(", ")) });
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
            let (t, token) = door(cfg)?;
            eprintln!("designing {name} through the front door ...");
            let r = nextsycl_serve::http::rpc(&t, &token, "audio.voices.design", &b, 900)?;
            println!("saved voice {name}: {:.1} s, seed {}  - \"{}\"", r["voice"]["seconds"].as_f64().unwrap_or(0.0), r["seed"],
                     r["voice"]["text"].as_str().unwrap_or(""));
            Ok(())
        }
        Some("enroll") => enroll(cfg, &lib, name, args),
        Some("rm" | "remove" | "delete") => {
            lib.remove(name)?;
            println!("removed voice {name}");
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

/// The front door (NS_SERVE_PORT on this machine) and its API token (NS_API_TOKEN, or serve.json beside the registry)
fn door(cfg: &Config) -> Result<(nextsycl_serve::http::Target, String), String> {
    let reg = models::registry(cfg);
    let token = cfg.get("NS_API_TOKEN").or_else(|| {
        std::fs::read_to_string(reg.with_file_name("serve.json")).ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v["token"].as_str().map(str::to_string))
    }).ok_or("the front door's token (NS_API_TOKEN, or serve.json beside the registry): is nextsycl serve running?")?;
    let port = cfg.get("NS_SERVE_PORT").unwrap_or_else(|| "8000".into());
    Ok((nextsycl_serve::http::Target::Tcp(format!("127.0.0.1:{port}")), token))
}

/// `nextsycl audio voice enroll NAME` through the front door: its job (audio.voices.enroll), which waits for a card
/// like a load does, followed until it ends
fn enroll_via_door(cfg: &Config, name: &str, args: &[String]) -> Result<(), String> {
    let (t, token) = door(cfg)?;
    let label = format!("enroll-{name}-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()));
    let mut b = serde_json::json!({"name": name, "wait": false, "label": label});
    if let Some(g) = num::<u64>(args, "--gpu")? {
        b["gpu"] = serde_json::json!(g);
    }
    if let Some(n) = num::<u64>(args, "--steps")? {
        b["steps"] = serde_json::json!(n);
    }
    if let Some(m) = opt(args, "--model") {
        b["model"] = serde_json::json!(m);
    }
    if args.iter().any(|a| a == "--replace") {
        b["replace"] = serde_json::json!(true);
    }
    nextsycl_serve::http::rpc(&t, &token, "audio.voices.enroll", &b, 60)?;
    eprintln!("enrolling {name} through the front door (job {label}) ...");
    let mut last = String::new();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let j = nextsycl_serve::http::rpc(&t, &token, "audio.status", &serde_json::json!({"label": label}), 30)?;
        let now = match j["state"].as_str().unwrap_or("") {
            "done" => {
                let r = &j["result"];
                println!("enrolled {name} for {} on GPU {} in {} s (experimental: Voxtral speaks it as \"voice\": \"{name}\")",
                         r["model"].as_str().unwrap_or("?"), r["gpu"], r["seconds"]);
                return Ok(());
            }
            "failed" | "cancelled" => return Err(j["error"]["message"].as_str().unwrap_or("the enrolment failed").to_string()),
            _ => j["waiting_for"].as_str().map(str::to_string).or_else(|| {
                let p = &j["progress"];
                p["at"].as_u64().map(|a| format!("step {a} of {}", p["of"]))
            }).unwrap_or_else(|| j["stage"].as_str().unwrap_or("").to_string()),
        };
        if now != last {
            eprintln!("  {now}");
            last = now;
        }
    }
}

/// The card memory an enrolment takes (GiB; the decoder's fit of a 15 s clip on a B65: ~6), and the margin kept free
const ENROLL_GIB: f64 = 8.0;

/// `nextsycl audio voice enroll NAME`: a saved voice fitted for Voxtral TTS (experimental; no encoder in its open
/// weights) - reference/voxtraltts/enroll.py in the build image with PyTorch XPU on one card, its codes kept with the
/// voice (`<voice>/enrolled/voxtral-tts.npy`, the report and the decoder's reconstruction beside them)
fn enroll(cfg: &Config, lib: &nextsycl_serve::voices::Library, name: &str, args: &[String]) -> Result<(), String> {
    let arch = opt(args, "--arch").unwrap_or(nextsycl_audio_voxtraltts::ARCH);
    if arch != nextsycl_audio_voxtraltts::ARCH {
        return Err(format!("--arch {arch}: only {} enrolls voices", nextsycl_audio_voxtraltts::ARCH));
    }
    // through the front door (a job that waits for a card as a load does) unless --direct (the door's own run)
    if !args.iter().any(|a| a == "--direct") {
        return enroll_via_door(cfg, name, args);
    }
    // (--direct is itself GPU work: the door runs it once it has the card; by hand, hold the GPU run lock around it)
    let sample = lib.sample_path(name).ok_or_else(|| format!("no saved voice {name:?} (nextsycl audio voice list)"))?;
    let dir = |k: &str, what: &str| -> Result<PathBuf, String> {
        cfg.get(k).map(PathBuf::from).filter(|p| p.is_dir()).ok_or_else(|| format!("{k}: {what} (nextsycl.conf)"))
    };
    let ckpt = dir("NS_VXT_CHECKPOINT", "the mistralai/Voxtral-4B-TTS-2603 checkout (consolidated.safetensors, params.json)")?;
    let omni = dir("NS_VXT_VLLM_OMNI", "a vllm-omni checkout (its voxtral_tts code is the decoder enroll.py runs)")?;
    let torch = dir("NS_VXT_TORCH", "PyTorch with XPU, installed with pip --target (the image's Python)")?;
    let tool = cfg.get("NS_VXT_ENROLL_TOOL").map(PathBuf::from).unwrap_or_else(|| cfg.dist.join("../reference/voxtraltts"));
    if !tool.join("enroll.py").is_file() {
        return Err(format!("{}: no enroll.py (NS_VXT_ENROLL_TOOL: the repository's reference/voxtraltts)", tool.display()));
    }
    let gpu: usize = num(args, "--gpu")?.unwrap_or(0);
    // the card's memory by the front door's count: the fit runs only with its share and 1.5 GiB more free
    let port = cfg.get("NS_SERVE_PORT").unwrap_or_else(|| "8000".into());
    let st = nextsycl_serve::http::call_for(&nextsycl_serve::http::Target::Tcp(format!("127.0.0.1:{port}")), "GET", "/api/state", None, 5)
        .map_err(|e| format!("the front door (:{port}) did not answer ({e}): its card count is the memory check"))?;
    let card = st["cards"].as_array().and_then(|c| c.get(gpu)).ok_or_else(|| format!("no GPU {gpu}"))?;
    let free = (card["vram_total_mb"].as_f64().unwrap_or(0.0) - card["vram_used_mb"].as_f64().unwrap_or(0.0)) / 1024.0;
    if free < ENROLL_GIB + 1.5 {
        return Err(format!("GPU {gpu} has {free:.1} GiB free; an enrolment needs {ENROLL_GIB} + 1.5 (another card: --gpu N)"));
    }
    let vdir = sample.parent().ok_or("the voice's directory")?.to_path_buf();
    let out = vdir.join("enrolled");
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let ce = crate::container::Ce::new(cfg)?;
    ce.need_image()?;
    let mut a: Vec<String> = vec!["run".into(), "--rm".into(), "--name".into(), format!("nextsycl-enroll-{gpu}"), "--security-opt".into(), "label=disable".into()];
    a.extend(ce.gpu_args());
    for (host, inside, ro) in [(&tool, "/tool", true), (&ckpt, "/ckpt", true), (&omni, "/omni", true), (&torch, "/torch", true), (&vdir, "/voice", false)] {
        a.extend(crate::container::mount(host, inside, ro));
    }
    for e in [format!("ZE_AFFINITY_MASK={gpu}"), "PYTHONPATH=/torch".into(), "PYTHONDONTWRITEBYTECODE=1".into(), "OMP_NUM_THREADS=8".into()] {
        a.extend(["-e".into(), e]);
    }
    let steps = num::<u32>(args, "--steps")?.unwrap_or(2000).to_string();
    let seconds = opt(args, "--seconds").unwrap_or("15").to_string();
    a.extend([ce.image.clone(), "bash".into(), "-c".into(), "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec python3 \"$@\"".into(), "enroll".into(),
              "/tool/enroll.py".into(), "/ckpt".into(), "--code".into(), "/omni".into(), "--audio".into(), "/voice/sample.wav".into(),
              "--out".into(), format!("/voice/enrolled/.{arch}.npy"), "--report".into(), format!("/voice/enrolled/.{arch}.json"),
              "--device".into(), "xpu".into(), "--steps".into(), steps, "--seconds".into(), seconds]);
    eprintln!("enrolling {name} for {arch} on GPU {gpu} (PyTorch XPU; a minute or two) ...");
    let t0 = std::time::Instant::now();
    let st = ce.cmd().args(&a).status().map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("the enrolment ended ({st})"));
    }
    for ext in ["npy", "json", "wav"] {
        let (from, to) = (out.join(format!(".{arch}.{ext}")), out.join(format!("{arch}.{ext}")));
        if from.exists() {
            std::fs::rename(&from, &to).map_err(|e| format!("{}: {e}", to.display()))?;
        }
    }
    println!("enrolled {name} for {arch} in {:.0} s: {} (experimental: Voxtral speaks it as \"voice\": \"{name}\")", t0.elapsed().as_secs_f64(),
             out.join(format!("{arch}.npy")).display());
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
        Some("transcribe" | "stt") => transcribe(cfg, rest),
        Some("check") => check(cfg, rest),
        Some("voice" | "voices") => voice(cfg, rest),
        Some("addvoice" | "clonevoice") => voice(cfg, &[&["add".to_string()], rest].concat()),
        Some("designvoice") => voice(cfg, &[&["design".to_string()], rest].concat()),
        Some("serve") => serve(cfg, rest, false),
        Some("start") => SERVED.start(cfg, rest, |d| serve(cfg, rest, d)),
        Some("ps") => SERVED.ps(cfg, rest),
        Some("stop") => SERVED.stop(cfg, rest),
        Some("logs") => SERVED.logs(cfg, rest),
        _ => Err(USAGE.into()),
    }
}
