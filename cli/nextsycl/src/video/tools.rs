//! The tools the films are made with, talking to the studio the way the front end does (`/rpc/generate`):
//!
//! - `speech`: a speech text -> clips sized to their words, cut only at sentence ends or a `||` beat, each a character
//!   speaking to camera, chained to the clip before (picture, sound tail, exposure), optionally cutting between two
//!   cameras with a match-on-action turn;
//! - `scene`: a scene file (the front end's export) -> the queue;
//! - `join`: a series of finished clips -> one film, frame-exact, crossfaded (or overlapped by the audio anchor's
//!   length), loudness-matched, its sync checked;
//! - `speechpct`: how much of each clip is speech.
//!
//! They are the legacy Python tools, ported (H3's sycl-h3 tools as they were); two of their bugs are fixed:
//! abbreviations ("Dr. Hale") no longer end a sentence, and `--paused` holds the project before its first clip is
//! queued.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use nextsycl_models::config::Config;
use nextsycl_serve::video::studio::http::{self as h3_http, Error, Result, Target};
use nextsycl_serve::video::studio::pyround;
use serde_json::{json, Value};

/// `--name value` / `--flag` options.
struct Opts {
    pos: Vec<String>,
    kv: BTreeMap<String, Vec<String>>,
}

impl Opts {
    fn parse(raw: &[String], flags: &[&str]) -> Opts {
        let (mut pos, mut kv) = (Vec::new(), BTreeMap::<String, Vec<String>>::new());
        let mut it = raw.iter().peekable();
        while let Some(a) = it.next() {
            match a.strip_prefix("--") {
                Some(k) if flags.contains(&k) => {
                    // a flag with an optional number after it (--realism [S])
                    let v = it.peek().filter(|n| n.parse::<f64>().is_ok() && k == "realism").map(|n| n.to_string());
                    if v.is_some() {
                        it.next();
                    }
                    kv.entry(k.into()).or_default().push(v.unwrap_or_default());
                }
                Some(k) => {
                    let v = it.next().cloned().unwrap_or_default();
                    kv.entry(k.into()).or_default().push(v);
                }
                None => pos.push(a.clone()),
            }
        }
        Opts { pos, kv }
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).and_then(|v| v.last()).map(String::as_str)
    }
    fn all(&self, k: &str) -> Vec<String> {
        self.kv.get(k).cloned().unwrap_or_default()
    }
    fn has(&self, k: &str) -> bool {
        self.kv.contains_key(k)
    }
    fn f(&self, k: &str, d: f64) -> f64 {
        self.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    fn i(&self, k: &str, d: i64) -> i64 {
        self.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    fn s(&self, k: &str, d: &str) -> String {
        self.get(k).unwrap_or(d).to_string()
    }
}

fn studio_url(cfg: &Config, o: &Opts) -> String {
    o.get("studio").map(str::to_string).or_else(|| cfg.get("NS_VIDEO_STUDIO")).unwrap_or_else(|| "http://127.0.0.1:8095".into())
}

fn rpc(base: &str, method: &str, body: &Value) -> Result<Value> {
    let (host, path) = h3_http::split_url(&format!("{}/rpc/{method}", base.trim_end_matches('/')))?;
    let v = h3_http::call(&Target::Tcp(host), "POST", &path, Some(body))?;
    if v["ok"] == true {
        Ok(v["result"].clone())
    } else {
        Err(Error(format!("{method}: {}", v["error"]["message"].as_str().unwrap_or("failed"))))
    }
}

fn wc(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Sentences: runs of text ending in . ! or ? (text after the last one is dropped, as the original does).
fn sentences(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_end = false;
    for c in t.chars() {
        let end = matches!(c, '.' | '!' | '?');
        if in_end && !end {
            out.push(std::mem::take(&mut cur));
            in_end = false;
        }
        if end && cur.is_empty() {
            continue; // terminal marks with no text before them
        }
        cur.push(c);
        in_end |= end;
    }
    if in_end {
        out.push(cur);
    }
    out
}

fn snap(sec: f64) -> f64 {
    let mut n = (pyround(sec * 24.0) as i64).max(5);
    while n % 17 != 5 {
        n += 1;
    }
    n as f64 / 24.0
}

const BEAT: char = '\u{0}';
const ABBREV: [&str; 12] = ["Dr", "Mr", "Mrs", "Ms", "St", "Lt", "Cmdr", "Capt", "Jr", "Sr", "vs", "etc"];

/// A speech text -> the lines of its clips (with the pause mark between beats).
pub fn speech_lines(text: &str, mark: &str, max_words: usize, min_frag: usize) -> Vec<String> {
    let mut text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    // the abbreviation fix before the blocks are cut (the original applied it to nothing)
    for w in ABBREV {
        text = text.replace(&format!("{w}. "), &format!("{w}<DOT> "));
    }
    let blocks: Vec<String> = text.split("|||").map(|b| b.trim().to_string()).filter(|b| !b.is_empty()).collect();
    let blocks = if blocks.is_empty() { vec![text.clone()] } else { blocks };
    let mut lines = Vec::new();
    for block in blocks {
        let block = block.replace(mark, &BEAT.to_string());
        let mut units: Vec<(String, bool)> = Vec::new();
        for sent in sentences(&block).iter().map(|x| x.replace("<DOT>", ".").trim().to_string()) {
            for (k, seg) in sent.split(BEAT).map(str::trim).enumerate() {
                if seg.is_empty() {
                    continue;
                }
                if wc(seg) <= max_words {
                    units.push((seg.to_string(), k > 0));
                    continue;
                }
                // a sentence too long for one clip: split at commas, merge fragments too short to stand alone
                let mut parts: Vec<String> = Vec::new();
                let mut cur = String::new();
                for w in seg.split_whitespace() {
                    if !cur.is_empty() {
                        cur.push(' ');
                    }
                    cur.push_str(w);
                    if w.ends_with(',') {
                        parts.push(std::mem::take(&mut cur));
                    }
                }
                if !cur.is_empty() {
                    parts.push(cur);
                }
                let mut merged: Vec<String> = Vec::new();
                for p in parts {
                    match merged.last_mut() {
                        Some(m) if wc(&p) < min_frag || wc(m) < min_frag => {
                            m.push(' ');
                            m.push_str(&p);
                        }
                        _ => merged.push(p),
                    }
                }
                for (m, piece) in merged.into_iter().enumerate() {
                    units.push((piece, k > 0 && m == 0));
                }
            }
        }
        let (mut cur, mut beats): (Vec<String>, Vec<bool>) = (Vec::new(), Vec::new());
        let flush = |cur: &mut Vec<String>, beats: &mut Vec<bool>, lines: &mut Vec<String>| {
            let mut out: Vec<String> = Vec::new();
            for (i, p) in cur.iter().enumerate() {
                if i > 0 && beats[i] {
                    out.push(mark.to_string());
                }
                out.push(p.clone());
            }
            lines.push(out.join(" "));
            cur.clear();
            beats.clear();
        };
        for (u, beat) in units {
            if !cur.is_empty() && wc(&format!("{} {u}", cur.join(" "))) > max_words {
                flush(&mut cur, &mut beats, &mut lines);
            }
            let had = !cur.is_empty();
            cur.push(u);
            beats.push(beat && had);
        }
        if !cur.is_empty() {
            flush(&mut cur, &mut beats, &mut lines);
        }
    }
    lines
}

const LOCKED: &str = "The camera sits in one fixed position on a tripod for the whole shot, at a constant distance from him, and the shot size is identical in the first frame and in the last. ";

fn camera(sec: f64, pan: &str) -> String {
    if sec < 7.0 {
        return LOCKED.into();
    }
    if sec < 11.0 {
        return format!("{LOCKED}As he speaks he turns his head very slightly, no more than a few degrees, and settles it back. ");
    }
    format!("The camera sits almost still on a tripod, easing from {pan} by a hair across the whole shot, a movement so small it is only just noticeable, holding the same distance and the same shot size in the first frame and in the last. His head turns slowly to follow that drift, keeping his eyes on the lens. ")
}

fn cameras(k: &str) -> (&'static str, &'static str) {
    match k {
        "A" => ("a medium shot from a camera set squarely in front of him, level with his eyes", "to his right"),
        _ => ("a three-quarter medium shot from a second camera set well round to his right, so that he is seen at an angle rather than straight on", "back to his left"),
    }
}

const SPEECH_HELP: &str = "nextsycl video speech <text file> [options]  - a speech as sized, chained character clips
  --character NAME (from --characters FILE or NS_VIDEO_CHARACTERS; data)   --words-per-s R (the character's)
  --speech-frac 0.57  --max-words 28  --min-frag 5  --min-s 4  --max-s 15  --steps 8
  --width 896 --height 672  --chain-mode png|video|latent|none  --cond-noise-aug X
  --cameras 1|2  --switch-every 2  --anchor chain|hub  --no-end-anchor  --chain-every N
  --match-exposure (default) | --no-match-exposure
  --audio-anchor prev|first|none|FILE  --audio-anchor-s 1.0   (the previous clip's sound tail at frame 0)
  --guide-frames N   motion guide: the previous clip's last N frames as a moving keyframe at frame 0 (new)
  --voice-ref FILE (up to 3, in the clips directory: <Audio j>, the voice lock)
  --realism [S] | --no-realism  --lora PATH[:S] ...  --trigger WORD  --upscale 1.5
  --look-on PHRASE  --look-on-direction TEXT  --voice TEXT  --scene TEXT  --soundscape TEXT  --final-direction TEXT
  --pause-mark '||'  --pause-on TEXT  --start-after TEXT  --start-n 1  --from 1  --count 0  --first-frame FILE
  --label Speech  --project P  --batch B  --paused  --dry  --studio URL";

/// `nextsycl video speech`
pub fn speech(cfg: &Config, raw: &[String]) -> Result<()> {
    let o = Opts::parse(raw, &["realism", "no-realism", "no-end-anchor", "match-exposure", "no-match-exposure", "paused", "dry", "help"]);
    if o.has("help") || o.pos.is_empty() {
        println!("{SPEECH_HELP}");
        return Ok(());
    }
    let chars_path = o.get("characters").map(PathBuf::from).or_else(|| cfg.get("NS_VIDEO_CHARACTERS").map(PathBuf::from)).unwrap_or_else(|| cfg.dist.join("../docs/video/characters.example.json"));
    let chars: Value = serde_json::from_slice(&std::fs::read(&chars_path).map_err(|e| Error(format!("{}: {e}", chars_path.display())))?).map_err(|e| Error(format!("{}: {e}", chars_path.display())))?;
    let name = o.s("character", "data");
    let c = chars.get(&name).ok_or_else(|| Error(format!("no character {name:?} in {} (there are: {:?})", chars_path.display(), chars.as_object().map(|m| m.keys().collect::<Vec<_>>()))))?;
    let cs = |k: &str| c[k].as_str().unwrap_or("").to_string();
    let mut text = std::fs::read_to_string(&o.pos[0])?;
    if let Some(sa) = o.get("start-after") {
        let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let i = t.find(sa).ok_or_else(|| Error(format!("--start-after not found: {}", sa.chars().take(40).collect::<String>())))?;
        text = t[i + sa.len()..].trim().to_string();
    }
    let mark = o.s("pause-mark", "||");
    let lines = speech_lines(&text, &mark, o.i("max-words", 28) as usize, o.i("min-frag", 5) as usize);
    let rate = o.get("words-per-s").and_then(|v| v.parse().ok()).unwrap_or_else(|| c["rate"].as_f64().unwrap_or(3.4));
    let frac = o.f("speech-frac", 0.57);
    let clip_seconds = |line: &str| -> f64 {
        let w = wc(&line.replace(&mark, " ")) as f64;
        let need = w / (rate * frac) + line.matches(&mark).count() as f64 * 0.5;
        snap(need.max(o.f("min-s", 4.0)).min(o.f("max-s", 15.0)))
    };
    // the character's realism LoRA unless the command line says otherwise
    let realism = if o.has("no-realism") {
        0.0
    } else if o.has("realism") {
        o.get("realism").and_then(|v| v.parse().ok()).unwrap_or(0.5)
    } else {
        c["realism"].as_f64().unwrap_or(0.0)
    };
    let mut loras = o.all("lora");
    let mut trigger = o.get("trigger").map(str::to_string);
    if realism > 0.0 {
        loras.push(format!("/models/loras/h3-realism-people-t2v-i2v-r2v.safetensors:{realism}"));
        trigger.get_or_insert_with(|| "r34l1sm".into());
    }
    let upscale = Some(o.f("upscale", 1.5)).filter(|u| *u > 1.0);
    let label = o.s("label", "Speech");
    let project = o.get("project").unwrap_or(&label).trim().to_string();
    let stamp = Command::new("date").arg("+%m%d-%H%M").output().map(|x| String::from_utf8_lossy(&x.stdout).trim().to_string()).unwrap_or_default();
    let batch = o.get("batch").map(str::to_string).unwrap_or_else(|| format!("{label}-{stamp}")).trim().to_string();
    let ncam = o.i("cameras", 1);
    let switch_every = o.i("switch-every", 2).max(1) as usize;
    let hub = o.s("anchor", "chain") == "hub";
    let end_anchor = !o.has("no-end-anchor");
    let chain_mode = o.s("chain-mode", "png");
    let chain_every = o.i("chain-every", 0) as usize;
    let match_exposure = !o.has("no-match-exposure");
    let audio_anchor = o.s("audio-anchor", "prev");
    let file_anchor = !matches!(audio_anchor.as_str(), "none" | "prev" | "first");
    let start_n = o.i("start-n", 1) as usize;
    let keys: Vec<Option<&str>> = (0..lines.len()).map(|i| (ncam == 2).then(|| if (i / switch_every).is_multiple_of(2) { "A" } else { "B" })).collect();
    let mut jobs = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let sec = clip_seconds(line);
        let segs: Vec<&str> = line.split(mark.as_str()).map(str::trim).filter(|x| !x.is_empty()).collect();
        let mut speech = format!("says: <d>[English] {}</d>", segs[0]);
        for seg in &segs[1..] {
            speech += &format!(" He holds silent for half a second, then continues: <d>[English] {seg}</d>");
        }
        let last = i + 1 == lines.len();
        let mut ending = if i % 3 == 2 || last { cs("tilt") } else { cs("hold") };
        if hub {
            ending = if i % 3 == 2 {
                format!("{} and settles back to exactly the pose he began in, facing the camera squarely, perfectly still, as the shot ends.", cs("tilt").replace("as the shot ends.", "then straightens his head again"))
            } else {
                "He finishes the line and settles back to exactly the pose he began in, facing the camera squarely at the same distance, perfectly still, as the shot ends.".into()
            };
        }
        if ncam == 2 && !last && keys[i + 1] != keys[i] {
            ending = format!("As the shot ends he begins to turn his head slowly {}, the movement only just started as the shot finishes.", cameras(keys[i].unwrap()).1);
        }
        if let (Some(f), true) = (o.get("final-direction"), last) {
            ending = f.trim().to_string();
        }
        let is_pause = o.get("pause-on").is_some_and(|p| line.starts_with(p));
        let mut opener = if is_pause { cs("pause") } else { cs("look") };
        if let Some(lo) = o.get("look-on").filter(|lo| line.to_lowercase().contains(&lo.to_lowercase())) {
            let beat = o.get("look-on-direction").map(str::to_string).unwrap_or_else(|| format!("on the words \"{lo}\" he turns his head to face the lens squarely and holds his eyes on the camera through the phrase"));
            opener = format!("{}, {},", opener.trim_end().trim_end_matches(','), beat.trim().trim_end_matches('.').trim_start());
        }
        let mut cam = if is_pause { String::new() } else { camera(sec, if i % 2 == 0 { "left to right" } else { "right to left" }) };
        if hub && !is_pause {
            cam = camera(sec.min(10.0), "");
        }
        let framing = match keys[i] {
            Some(k) => {
                let mut f = format!("This is {}, and that framing holds for the whole shot, his shoulders and the bridge rail visible. ", cameras(k).0);
                if i > 0 && keys[i - 1] != keys[i] {
                    f += "He is already mid-turn as the shot opens, and he completes that turn smoothly, his eyes settling on this camera, and holds there. ";
                }
                f
            }
            None => format!("{} ", cs("frame")),
        };
        let trig = trigger.as_ref().map(|t| format!("{}, ", t.trim())).unwrap_or_default();
        let style = c["style"].as_str().unwrap_or("Live-action, cinematic");
        let prompt = format!(
            "integrated_multimodal_description: [Shot 1] {trig}{style}, a medium shot frames {} {} {framing}{cam}{opener} and {}, {speech} {ending}\n\noverall_soundscape: {}\n\nnon_diegetic_music: None.",
            cs("who"), o.get("scene").map(str::to_string).unwrap_or_else(|| cs("scene")), o.get("voice").map(str::to_string).unwrap_or_else(|| cs("voice")),
            o.get("soundscape").map(str::to_string).unwrap_or_else(|| cs("sound")));
        let first = (chain_mode != "none" && i > 0).then(|| if ncam == 2 { "prev_cam" } else if hub || (chain_every > 0 && i % chain_every == 0) { "first" } else { "prev" });
        let n = i + start_n;
        let short: String = line.replace(&mark, "").trim().chars().take(32).collect::<String>().trim_end_matches([' ', ',', '.']).to_string();
        let mut j = json!({
            "project": project, "batch": batch, "prompt": prompt, "seconds": sec, "steps": o.i("steps", 8), "seed": 0, "te": "teacher",
            "engine": o.s("engine", "Q8_0"), "width": o.i("width", 896), "height": o.i("height", 672), "chain_mode": chain_mode,
            "cond_noise_aug": o.get("cond-noise-aug").and_then(|v| v.parse::<f64>().ok()), "first_frame": first, "camera": keys[i],
            "last_frame": (hub && end_anchor && i > 0).then_some("first"), "upscale": upscale, "loras": loras,
            "first_audio": match audio_anchor.as_str() { "none" => None, _ if file_anchor => Some(audio_anchor.clone()), _ => (i > 0).then(|| audio_anchor.clone()) },
            "first_audio_s": o.f("audio-anchor-s", 1.0), "ref_audios": o.all("voice-ref"),
            "exposure_ref": (match_exposure && i > 0).then_some("first"), "queue": true,
            "label": format!("{label} {n:02}/{}: {short}...", lines.len() + start_n - 1),
        });
        if let (Some(g), true) = (o.get("guide-frames").and_then(|v| v.parse::<usize>().ok()), i > 0 && chain_mode != "none") {
            j["guide_clip"] = json!(format!("{}:{g}:0", if ncam == 2 { "prev_cam" } else { "prev" }));
        }
        jobs.push((j, line.clone()));
    }
    let total: f64 = jobs.iter().map(|j| j.0["seconds"].as_f64().unwrap_or(0.0)).sum();
    println!("{} clips, {total:.0}s of video ({:.1} min), project {project:?}, batch {batch:?}{}, {chain_mode} chaining{}{}{}{}{}",
             jobs.len(), total / 60.0, if o.has("paused") { " [PAUSED]" } else { "" },
             if ncam == 2 { format!(", 2 cameras switching every {switch_every}") } else { String::new() },
             if hub { format!(", hub-anchored to clip 1{}", if end_anchor { " (both ends)" } else { "" }) } else { String::new() },
             if loras.is_empty() { String::new() } else { format!(", {} lora(s)", loras.len()) },
             if audio_anchor != "none" { format!(", audio anchored to {audio_anchor} ({}s)", o.f("audio-anchor-s", 1.0)) } else { String::new() },
             if match_exposure { ", exposure matched" } else { "" });
    println!("{:>3} {:>5} {:>6} {:>6}  line", "#", "secs", "words", "beats");
    for (k, (j, line)) in jobs.iter().enumerate() {
        println!("{:3} {:5.2} {:6} {:6} {:>2}  {}", k + start_n, j["seconds"].as_f64().unwrap_or(0.0), wc(&line.replace(&mark, " ")), line.matches(&mark).count(),
                 j["camera"].as_str().unwrap_or("-"), line.replace(&mark, " | ").chars().take(56).collect::<String>());
    }
    if o.has("dry") {
        return Ok(());
    }
    let base = studio_url(cfg, &o);
    if o.has("paused") {
        // held before the first clip is queued (the original held it after, so clip 1 could already be running)
        rpc(&base, "queue.pause", &json!({"project": project}))?;
    }
    let from_n = o.i("from", 1) as usize;
    let count = o.i("count", 0) as usize;
    let end = if count > 0 { from_n + count - 1 } else { jobs.len() };
    for (k, (mut j, _)) in jobs.into_iter().enumerate() {
        let i = k + 1;
        if i < from_n {
            continue;
        }
        if i > end {
            break;
        }
        if let (Some(ff), true) = (o.get("first-frame"), i == from_n) {
            j["first_frame"] = json!(ff);
        }
        let r = rpc(&base, "generate", &j).map_err(|e| Error(format!("clip {i}: {e}")))?;
        println!("queued {i} {}", r["position"].as_u64().map_or("RUNNING".into(), |p| p.to_string()));
    }
    if o.has("paused") {
        println!("project {project:?} held: resume it in the front end or with nextsycl video scene / the front end");
    }
    Ok(())
}

/// `nextsycl video scene <scene.json>`: queue a scene file.
pub fn scene(cfg: &Config, raw: &[String]) -> Result<()> {
    let o = Opts::parse(raw, &["dry", "paused"]);
    let path = o.pos.first().ok_or("nextsycl video scene <scene.json> [--project P] [--label L] [--from N --count N] [--cams AB] [--skip 1,2] [--paused] [--dry]")?;
    let doc: Value = serde_json::from_slice(&std::fs::read(path)?).map_err(|e| Error(format!("{path}: {e}")))?;
    if doc["version"].as_i64() != Some(1) {
        return Err(Error(format!("unsupported scene version {}", doc["version"])));
    }
    let project = o.get("project").map(str::to_string).or_else(|| doc["project"].as_str().map(str::to_string)).unwrap_or_else(|| "Scene".into());
    let skip: Vec<i64> = o.get("skip").map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let from = o.i("from", 0);
    let count = o.i("count", 0);
    let end = if count > 0 { from + count - 1 } else { i64::MAX };
    let cams = o.get("cams");
    let mut clips = doc["clips"].as_array().cloned().unwrap_or_default();
    clips.sort_by_key(|c| c["n"].as_i64().unwrap_or(0));
    let base = studio_url(cfg, &o);
    if o.has("paused") && !o.has("dry") {
        rpc(&base, "queue.pause", &json!({"project": project}))?;
    }
    for c in clips {
        let n = c["n"].as_i64().unwrap_or(0);
        if n < from || n > end || skip.contains(&n) || cams.is_some_and(|cs| !cs.contains(c["camera"].as_str().unwrap_or(""))) {
            continue;
        }
        let mut item = doc["defaults"].clone();
        if !item.is_object() {
            item = json!({});
        }
        for (k, v) in c.as_object().into_iter().flatten() {
            if k != "n" {
                item[k] = v.clone();
            }
        }
        if let (Some(new), Some(old)) = (o.get("label"), doc["label"].as_str()) {
            if let Some(l) = item["label"].as_str() {
                item["label"] = json!(l.replacen(old, new, 1));
            }
        }
        item["project"] = json!(project);
        if item["batch"].is_null() {
            item["batch"] = json!(doc["batch"].as_str().unwrap_or(&project));
        }
        item["queue"] = json!(true);
        if o.has("dry") {
            println!("{n:3} {}", item["label"].as_str().unwrap_or(""));
            continue;
        }
        let r = rpc(&base, "generate", &item).map_err(|e| Error(format!("clip {n}: {e}")))?;
        println!("queued {n} {}", r["position"].as_u64().map_or("RUNNING".into(), |p| p.to_string()));
    }
    Ok(())
}

fn ffprobe(path: &Path, stream: &str, entry: &str) -> Option<f64> {
    let o = Command::new("ffprobe").args(["-v", "error", "-select_streams", stream, "-count_packets", "-show_entries", &format!("stream={entry}"), "-of", "csv=p=0"]).arg(path).output().ok()?;
    String::from_utf8_lossy(&o.stdout).trim().split(',').next()?.trim().parse().ok()
}

/// `nextsycl video join <prefix>`: a series of finished clips -> one film.
pub fn join(cfg: &Config, raw: &[String]) -> Result<()> {
    let o = Opts::parse(raw, &["denoise"]);
    let prefix = o.pos.first().ok_or("nextsycl video join <label prefix> [--from 1 --to 999] [--xfade-frames 4] [--audio-anchor-s S] [--match-loudness -18] [--denoise] [--fade-out S] [--out FILE] [--out-dir DIR]")?.clone();
    let dir = o.get("out-dir").map(PathBuf::from).unwrap_or_else(|| super::service::out_dir(cfg));
    let fps = o.f("fps", 24.0);
    let (from, to) = (o.i("from", 1), o.i("to", 999));
    // the newest finished clip of each number
    let mut names: Vec<String> = std::fs::read_dir(&dir)?.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("h3_") && n.ends_with(".json")).collect();
    names.sort();
    let mut items: BTreeMap<i64, (PathBuf, Value)> = BTreeMap::new();
    for n in names {
        let Ok(side) = serde_json::from_slice::<Value>(&std::fs::read(dir.join(&n)).unwrap_or_default()) else { continue };
        let label = side["job"]["label"].as_str().unwrap_or("");
        let Some(rest) = label.strip_prefix(&format!("{prefix} ")) else { continue };
        let Some(num) = rest.split(|c: char| !c.is_ascii_digit()).next().and_then(|d| d.parse::<i64>().ok()) else { continue };
        if side["times"]["TOTAL"].as_f64().is_some_and(|t| t > 0.0) && (from..=to).contains(&num) {
            items.insert(num, (dir.join(n.replace(".json", ".mp4")), side));
        }
    }
    if items.is_empty() {
        return Err(Error(format!("no finished clips labelled {prefix:?} NN in {}", dir.display())));
    }
    let files: Vec<PathBuf> = items.values().map(|v| v.0.clone()).collect();
    let nframes: Vec<i64> = files.iter().map(|f| ffprobe(f, "v:0", "nb_read_packets").unwrap_or(0.0) as i64).collect();
    // the overlap: the audio anchor's length, when the clips were made with one
    let anchor_s = match o.get("audio-anchor-s") {
        Some(v) => v.parse().unwrap_or(0.0),
        None => {
            let anchored: Vec<Option<f64>> = items.values().skip(1).filter(|v| !v.1["job"]["first_audio"].is_null()).map(|v| v.1["job"]["first_audio_s"].as_f64()).collect();
            if anchored.is_empty() { 0.0 } else { anchored.iter().flatten().next().copied().unwrap_or(1.0) }
        }
    };
    let k = if anchor_s > 0.0 { pyround(anchor_s * fps / 2.0) as i64 } else { o.i("xfade-frames", 4).max(0) };
    let n = files.len();
    if 2 * k >= *nframes.iter().min().unwrap_or(&0) {
        return Err(Error(format!("a {k}-frame overlap does not fit the shortest clip ({} frames)", nframes.iter().min().unwrap_or(&0))));
    }
    let d = 2.0 * k as f64 / fps;
    let target = o.f("match-loudness", -18.0);
    let mut fc: Vec<String> = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let nf = nframes[i];
        fc.push(format!("[{i}:v]trim=start_frame={}:end_frame={},setpts=PTS-STARTPTS[v{i}]", if i > 0 { k } else { 0 }, if i + 1 < n { nf - k } else { nf }));
        let mut gain = String::new();
        if target != 0.0 {
            let e = Command::new("ffmpeg").args(["-hide_banner", "-vn", "-i"]).arg(f).args(["-af", "ebur128=framelog=quiet", "-f", "null", "-"]).output().ok();
            let l = e.and_then(|e| {
                let t = String::from_utf8_lossy(&e.stderr).into_owned();
                let i = t.rfind("I:")?;
                t[i + 2..].split_whitespace().next()?.parse::<f64>().ok()
            });
            if let Some(l) = l.filter(|l| *l > -70.0) {
                gain = format!(",volume={}dB", pyround((target - l) * 100.0) / 100.0);
            }
        }
        fc.push(format!("[{i}:a]apad,atrim=end={:.6}{gain},asetpts=PTS-STARTPTS[a{i}]", nf as f64 / fps));
    }
    fc.push(format!("{}concat=n={n}:v=1:a=0[vout]", (0..n).map(|i| format!("[v{i}]")).collect::<String>()));
    if n == 1 || k == 0 {
        fc.push(format!("{}concat=n={n}:v=0:a=1[aout]", (0..n).map(|i| format!("[a{i}]")).collect::<String>()));
    } else {
        let mut prev = "a0".to_string();
        for i in 1..n {
            let out = if i + 1 == n { "aout".to_string() } else { format!("ax{i}") };
            fc.push(format!("[{prev}][a{i}]acrossfade=d={d:.6}:c1=tri:c2=tri[{out}]"));
            prev = out;
        }
    }
    let (mut vlab, mut alab) = ("vout".to_string(), "aout".to_string());
    if target != 0.0 || o.has("denoise") {
        let dn = if o.has("denoise") { format!("highpass=f=85,afftdn=nr={}:nf={}:tn=1,", o.f("denoise-nr", 12.0), o.f("denoise-nf", -32.0)) } else { String::new() };
        fc.push(format!("[aout]{dn}alimiter=limit=0.891:level=disabled[aoutl]"));
        alab = "aoutl".into();
    }
    let total_frames: i64 = nframes.iter().sum::<i64>() - 2 * k * (n as i64 - 1);
    let total = total_frames as f64 / fps;
    let fade = o.f("fade-out", 0.0);
    if fade > 0.0 {
        let st = (total - fade).max(0.0);
        fc.push(format!("[{vlab}]fade=t=out:st={st:.3}:d={fade}[voutf]"));
        fc.push(format!("[{alab}]afade=t=out:st={st:.3}:d={fade}[aoutf]"));
        vlab = "voutf".into();
        alab = "aoutf".into();
    }
    let (first_n, last_n) = (*items.keys().next().unwrap(), *items.keys().last().unwrap());
    let out = o.get("out").map(PathBuf::from).unwrap_or_else(|| dir.join(format!("{}_join_{first_n:02}-{last_n:02}.mp4", prefix.to_lowercase().replace(' ', "_"))));
    let part = out.with_extension("part.mp4");
    let mut c = Command::new("ffmpeg");
    c.args(["-y", "-loglevel", "error"]);
    for f in &files {
        c.arg("-i").arg(f);
    }
    c.args(["-filter_complex", &fc.join(";"), "-map", &format!("[{vlab}]"), "-map", &format!("[{alab}]"), "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", &format!("{fps}"), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-f", "mp4"]).arg(&part);
    println!("{n} clips ({first_n}-{last_n}), {k}-frame overlaps{} -> {total:.2} s", if anchor_s > 0.0 { format!(" (audio anchor {anchor_s} s)") } else { String::new() });
    if !c.status()?.success() {
        return Err(Error("ffmpeg failed".into()));
    }
    std::fs::rename(&part, &out)?;
    let vd = ffprobe(&out, "v:0", "duration").unwrap_or(0.0);
    let ad = ffprobe(&out, "a:0", "duration").unwrap_or(0.0);
    let vf = ffprobe(&out, "v:0", "nb_read_packets").unwrap_or(0.0) as i64;
    let ok = (ad - vd).abs() <= 1.5 / fps && vf == total_frames;
    println!("{}: video {vd:.3} s ({vf} frames, want {total_frames}), audio {ad:.3} s - {}", out.display(), if ok { "SYNC: OK" } else { "*** OUT OF SYNC ***" });
    if ok { Ok(()) } else { Err(Error("out of sync".into())) }
}

/// `nextsycl video speechpct <clip.mp4>...`
pub fn speechpct(raw: &[String]) -> Result<()> {
    if raw.is_empty() {
        return Err(Error("nextsycl video speechpct <clip.mp4>...".into()));
    }
    println!("{:<34} {:>7} {:>8} {:>6} {:>6} {:>6}", "clip", "secs", "speech", "%", "lead", "tail");
    for f in raw {
        match nextsycl_serve::video::studio::speech::analyse(Path::new(f)) {
            Some(s) => println!("{:<34} {:>7.2} {:>8.2} {:>6.1} {:>6.2} {:>6.2}  floor {:.1} dB, peak {:.1} dB", Path::new(f).file_name().unwrap_or_default().to_string_lossy(),
                                s.seconds, s.speech_s, s.speech_pct, s.lead_silence, s.tail_silence, s.floor_db, s.ceil_db),
            None => println!("{f}: no sound"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_cutting() {
        let t = "I am Ada. I was taught by Dr. Hale, who gave me a cat, Pip, and a violin, which I play every evening at exactly seven. || Then I read. ||| The end.";
        let l = speech_lines(t, "||", 12, 5);
        // "Dr. Hale" does not end a sentence; the long sentence splits at commas, no stranded fragments; ||| breaks hard
        assert!(l.iter().any(|x| x.contains("Dr. Hale")), "{l:?}");
        assert_eq!(l.last().unwrap(), "The end.");
        assert!(l.iter().all(|x| wc(x) <= 14), "{l:?}");
        assert_eq!(snap(4.0), 107.0 / 24.0);
        assert_eq!(sentences("A b. C d! no end"), vec!["A b.", " C d!"]);
    }
}
