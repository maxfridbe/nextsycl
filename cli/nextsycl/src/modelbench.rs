//! `nextsycl models bench <id> [--copies 1,2,3] [--beside <id>...] [--requests N] [--gpu N] [--write]`: measures a
//! model's concurrency profile and proposes the `concurrency` / `coexistence` block for its model card.
//!
//! It starts n copies of the model on one card (separate servers on ports of its own), sends a small standard request
//! set (`standard`: one place per kind) to every copy at once, and records the total throughput, the wall time of each
//! request, the peak VRAM of the card and any failure. With `--beside` it also runs the model next to each named model
//! (each alone, then both together) to see whether either collapses or the pair fails. The decisions are
//! `backplane_scheduler::profile::{recommend, scaling_of, verdict}` (tested with synthetic numbers there).
//!
//! Every GPU run holds the run lock (NS_GPU_LOCK, crate::gpulock): the command re-runs itself under flock, and while it
//! is held the front door loads nothing beside it. What is already loaded stays, so put the box in maintenance and let
//! the cards empty first. It never edits a card unless `--write` is given, which puts the proposed block into the card's file.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use backplane_scheduler::profile::{self, Beside, Mode, ScalePoint};
use nextsycl_models::config::Config;
use nextsycl_serve::http::{call_for, Target};
use serde_json::{json, Value};

/// The first port the bench's servers use (its own, away from the real servers')
const PORT0: u16 = 28100;

/// A kind's standard request: where it goes, what it sends, and what one answer is worth
#[derive(Debug)]
struct Std {
    path: &'static str,
    body: Value,
    /// the unit of throughput
    unit: &'static str,
    /// units one answer is
    units: f64,
}

/// The standard small request of a model, one place for every kind; Err says why a model has none yet
fn standard(entry: &Value) -> Result<Std, String> {
    let (kind, arch) = (entry["kind"].as_str().unwrap_or("llm"), entry["arch"].as_str().unwrap_or(""));
    match kind {
        "image" if arch.contains("ocr") || arch.contains("esrgan") || arch == "waifu2x" || arch == "seedvr2" || arch == "starvector" || arch == "olmocr2" => {
            Err(format!("{arch} takes a picture in: no standard request for it yet"))
        }
        "image" => Ok(Std { path: "/v1/images/generations", body: json!({"prompt": "a red fox sitting in fresh snow at dawn, photograph", "size": "512x512", "n": 1}), unit: "images", units: 1.0 }),
        "audio" if arch == "whisper" => Err("a recogniser takes a recording in: no standard request for it yet".into()),
        "audio" if entry["speech"].is_array() => Ok(Std {
            path: "/v1/audio/speech",
            body: json!({"input": "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.", "response_format": "wav"}),
            unit: "clips",
            units: 1.0,
        }),
        // (a song is the speech route: input the lyrics, instructions the music)
        "audio" => Ok(Std { path: "/v1/audio/speech", body: json!({"input": "[instrumental]", "instructions": "a calm piano melody", "seconds": 15}), unit: "songs", units: 1.0 }),
        "llm" => Ok(Std {
            path: "/v1/chat/completions",
            body: json!({"messages": [{"role": "user", "content": "Write two sentences about the sea."}], "max_tokens": 96, "stream": false}),
            unit: "answers",
            units: 1.0,
        }),
        other => Err(format!("{other} models have no standard request yet (the video daemon and the 3D servers need inputs): measure by hand")),
    }
}

/// One server the bench started
struct Copy {
    kind: String,
    slot: usize,
    port: u16,
}

fn opt_list(args: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = args.iter().skip_while(|a| *a != flag).skip(1);
    for a in it.by_ref() {
        if a.starts_with("--") {
            break;
        }
        out.push(a.clone());
    }
    out
}

fn opt(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn run_exe(exe: &str, args: &[String]) -> Result<(), String> {
    let st = Command::new(exe).args(args).stdin(std::process::Stdio::null()).status().map_err(|e| format!("{exe}: {e}"))?;
    if st.success() { Ok(()) } else { Err(format!("{exe} {}: {st}", args.join(" "))) }
}

fn start(exe: &str, entry: &Value, gpu: usize, slot: usize) -> Result<Copy, String> {
    let kind = entry["kind"].as_str().unwrap_or("llm").to_string();
    let port = PORT0 + slot as u16;
    let id = entry["id"].as_str().unwrap_or("");
    let mut a = vec![kind.clone(), "start".into(), id.into(), "--gpu".into(), gpu.to_string(), "--port".into(), port.to_string(), "--host".into(), "127.0.0.1".into()];
    if kind != "llm" {
        a.extend(["--slot".into(), slot.to_string()]);
    }
    run_exe(exe, &a)?;
    let t0 = Instant::now();
    loop {
        let up = if kind == "llm" {
            call_for(&Target::Tcp(format!("127.0.0.1:{port}")), "GET", "/v1/models", None, 3).is_ok()
        } else {
            call_for(&Target::Tcp(format!("127.0.0.1:{port}")), "GET", "/api/info", None, 3).ok().is_some_and(|i| i["model"] == id && i["loading"] != true)
        };
        if up {
            return Ok(Copy { kind, slot, port });
        }
        if t0.elapsed() > Duration::from_secs(900) {
            return Err(format!("{id} did not answer in 15 minutes on port {port}"));
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

fn stop(exe: &str, c: &Copy) {
    let a: Vec<String> = if c.kind == "llm" { vec!["llm".into(), "stop".into()] } else { vec![c.kind.clone(), "stop".into(), "--slot".into(), c.slot.to_string()] };
    let _ = run_exe(exe, &a);
}

/// What one run of requests measured
#[derive(Debug, Default)]
struct Run {
    /// units a minute, summed over the copies
    per_min: f64,
    wall_s: Vec<f64>,
    /// each copy's own units a minute (in the order given)
    per_copy: Vec<f64>,
    failures: usize,
    peak_vram_mb: u64,
}

/// `requests` standard requests to each of `copies` at once, one after another within a copy; the card's peak VRAM
/// sampled meanwhile
fn drive(copies: &[(Copy, Arc<Std>)], requests: usize, card: usize) -> Run {
    let stop_sampling = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let sampler = {
        let (stop, peak) = (stop_sampling.clone(), peak.clone());
        std::thread::spawn(move || {
            let pcis = nextsycl_serve::gpustat::find_cards();
            let Some(pci) = pcis.get(card) else { return };
            let mut c = nextsycl_serve::gpustat::Card::open(pci, None);
            while !stop.load(Ordering::Relaxed) {
                let (v, _) = c.sample(Duration::from_secs(5));
                peak.fetch_max(v["vram_used_mb"].as_u64().unwrap_or(0), Ordering::Relaxed);
                std::thread::sleep(Duration::from_secs(1));
            }
        })
    };
    let times = Arc::new(Mutex::new(Vec::new()));
    let failures = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let threads: Vec<_> = copies.iter().map(|(c, std)| {
        let (target, std, times, failures) = (Target::Tcp(format!("127.0.0.1:{}", c.port)), std.clone(), times.clone(), failures.clone());
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut ok = 0usize;
            for _ in 0..requests {
                let t = Instant::now();
                match call_for(&target, "POST", std.path, Some(&std.body), 1800) {
                    Ok(_) => {
                        ok += 1;
                        times.lock().unwrap().push(t.elapsed().as_secs_f64());
                    }
                    Err(e) => {
                        eprintln!("  request failed: {e}");
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            // this copy's units a minute over its own time
            std.units * ok as f64 / (start.elapsed().as_secs_f64() / 60.0).max(1e-9)
        })
    }).collect();
    let per_copy: Vec<f64> = threads.into_iter().map(|t| t.join().unwrap_or(0.0)).collect();
    let _ = t0;
    stop_sampling.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let wall_s = times.lock().unwrap().clone();
    Run { per_min: per_copy.iter().sum(), per_copy, wall_s, failures: failures.load(Ordering::Relaxed) as usize, peak_vram_mb: peak.load(Ordering::Relaxed) }
}

/// The block a measurement proposes: from the copies' points, the largest count that ran clean, and the verdicts
/// beside other models
fn propose(points: &[ScalePoint], clean_max: u32, besides: &[(String, Beside)], card: &str, date: &str, method: &str) -> (Value, Value) {
    let max = clean_max.max(1);
    let rec = profile::recommend(points).min(max);
    let scaling = profile::scaling_of(points);
    let mut mode = Mode::Shared;
    let mut why = Vec::new();
    for (other, b) in besides {
        let (m, w) = profile::verdict(b);
        if m == Mode::Exclusive {
            mode = Mode::Exclusive;
            why.push(format!("beside {other}: {w}"));
        }
    }
    let tested: Vec<&str> = besides.iter().map(|(o, _)| o.as_str()).collect();
    let concurrency = json!({
        "max_copies_per_card": max, "recommended_copies": rec,
        "scaling": scaling.iter().filter(|(n, _)| **n <= max).map(|(n, x)| (n.to_string(), json!(x))).collect::<serde_json::Map<_, _>>(),
        "measured": {"date": date, "card": card, "method": method, "notes": if tested.is_empty() { String::new() } else { format!("co-running tested with {}", tested.join(", ")) }},
    });
    let coexistence = match mode {
        Mode::Exclusive => json!({"mode": "exclusive", "exclusive_reason": why.join("; ")}),
        Mode::Shared => json!({"mode": "shared", "shared_with": []}),
    };
    (concurrency, coexistence)
}

/// A card file's text with its top-level `concurrency:` and `coexistence:` blocks replaced (or added at the end)
fn replace_blocks(text: &str, concurrency: &Value, coexistence: &Value) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut skipping = false;
    for l in text.lines() {
        let top = !l.is_empty() && !l.starts_with(' ') && !l.starts_with('\t') && !l.starts_with('#');
        if top {
            skipping = l.starts_with("concurrency:") || l.starts_with("coexistence:");
        }
        if !skipping {
            out.push(l);
        }
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    format!("{}\n{}\n{}\n", out.join("\n"), block("concurrency", concurrency), block("coexistence", coexistence))
}

/// The proposal into the card's file when `write` (`--write`, nothing else, touches a card); the file must stay a valid card
fn write_card(path: &Path, c: &Value, x: &Value, write: bool) -> Result<bool, String> {
    if !write {
        return Ok(false);
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let new = replace_blocks(&text, c, x);
    let v: Value = serde_yaml::from_str(&new).map_err(|e| format!("the card no longer parses: {e}"))?;
    profile::Profile::from_card(&v).map_err(|e| format!("the proposed block is not valid: {e}"))?;
    std::fs::write(path, new).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(true)
}

/// A block as YAML (JSON flow style is YAML)
fn block(key: &str, v: &Value) -> String {
    format!("{key}: {}", serde_json::to_string(v).unwrap_or_default())
}

pub fn cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let usage = "nextsycl models bench <id> [--copies 1,2,3] [--beside <id>...] [--requests N] [--gpu N] [--write]";
    let id = args.get(1).filter(|a| !a.starts_with("--")).ok_or(usage)?.clone();
    // every GPU run holds the run lock
    if std::env::var("NS_BENCH_LOCKED").ok().as_deref() != Some("1") {
        let lock = nextsycl_serve::gpulock::path(cfg.get("NS_GPU_LOCK")).ok_or("NS_GPU_LOCK is off: a bench needs the run lock, so the front door loads nothing beside it")?;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut c = Command::new("flock");
        c.arg(lock).arg(exe).arg("models").args(args).env("NS_BENCH_LOCKED", "1");
        let st = c.status().map_err(|e| format!("flock: {e} (every GPU run holds NS_GPU_LOCK)"))?;
        return if st.success() { Ok(()) } else { Err(format!("under flock: {st}")) };
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?.to_string_lossy().into_owned();
    let copies: Vec<u32> = opt(args, "--copies").unwrap_or_else(|| "1,2".into()).split(',').map(|n| n.trim().parse().map_err(|_| format!("--copies {n}: numbers like 1,2,3"))).collect::<Result<_, _>>()?;
    let requests: usize = opt(args, "--requests").map_or(Ok(6), |v| v.parse().map_err(|_| "--requests: a number".to_string()))?;
    let gpu: usize = opt(args, "--gpu").map_or(Ok(0), |v| v.parse().map_err(|_| "--gpu: a card number".to_string()))?;
    let besides = opt_list(args, "--beside");
    let entry_of = |m: &str| -> Result<Value, String> {
        nextsycl_models::registry::find(cfg, m)?.ok_or_else(|| format!("no model {m} registered (nextsycl models list)"))
    };
    let me = entry_of(&id)?;
    let std_me = Arc::new(standard(&me)?);
    let card = nextsycl_serve::gpustat::find_cards().get(gpu).map(|p| nextsycl_serve::gpustat::Card::open(p, None).name).unwrap_or_else(|| format!("GPU {gpu}"));
    eprintln!("bench {id} on GPU {gpu} ({card}): copies {copies:?}, {requests} requests each, unit {}", std_me.unit);

    // 1. scaling: n copies at once
    let mut points = Vec::new();
    let mut report = Vec::new();
    let mut clean_max = 1;
    for &n in &copies {
        let mut started: Vec<Copy> = Vec::new();
        let mut ok = true;
        for slot in 0..n as usize {
            match start(&exe, &me, gpu, slot) {
                Ok(c) => started.push(c),
                Err(e) => {
                    eprintln!("  {n} copies: copy {} did not start: {e}", slot + 1);
                    ok = false;
                    break;
                }
            }
        }
        let run = if ok {
            let set: Vec<(Copy, Arc<Std>)> = started.into_iter().map(|c| (c, std_me.clone())).collect();
            let r = drive(&set, requests, gpu);
            for (c, _) in &set {
                stop(&exe, c);
            }
            Some(r)
        } else {
            for c in &started {
                stop(&exe, c);
            }
            None
        };
        match run {
            Some(r) if r.failures == 0 => {
                clean_max = clean_max.max(n);
                points.push(ScalePoint { copies: n, total_per_min: r.per_min });
                report.push(json!({"copies": n, "per_min": r.per_min, "wall_s": r.wall_s, "peak_vram_mb": r.peak_vram_mb, "failures": 0}));
            }
            Some(r) => report.push(json!({"copies": n, "per_min": r.per_min, "failures": r.failures, "peak_vram_mb": r.peak_vram_mb, "note": "failures: not counted toward the maximum"})),
            None => report.push(json!({"copies": n, "failed": "did not start (out of memory?)"})),
        }
    }

    // 2. co-residency: alone, the other alone, both together
    let mut beside_results: Vec<(String, Beside)> = Vec::new();
    let alone = points.iter().find(|p| p.copies == 1).map_or(0.0, |p| p.total_per_min);
    for other in &besides {
        let oe = entry_of(other)?;
        let std_o = Arc::new(standard(&oe)?);
        let mut oc = start(&exe, &oe, gpu, 1)?;
        oc.slot = 1;
        let o_alone = drive(&[(Copy { kind: oc.kind.clone(), slot: oc.slot, port: oc.port }, std_o.clone())], requests, gpu).per_min;
        let mine = start(&exe, &me, gpu, 0);
        let (with_other, other_with, failed) = match mine {
            Ok(m) => {
                let r = drive(&[(Copy { kind: m.kind.clone(), slot: m.slot, port: m.port }, std_me.clone()), (Copy { kind: oc.kind.clone(), slot: oc.slot, port: oc.port }, std_o.clone())], requests, gpu);
                stop(&exe, &m);
                (r.per_copy.first().copied().unwrap_or(0.0), r.per_copy.get(1).copied().unwrap_or(0.0), r.failures > 0)
            }
            Err(e) => {
                eprintln!("  {id} did not start beside {other}: {e}");
                (0.0, 0.0, true)
            }
        };
        stop(&exe, &oc);
        report.push(json!({"beside": other, "alone": alone, "with_other": with_other, "other_alone": o_alone, "other_with": other_with, "failed": failed}));
        beside_results.push((other.clone(), Beside { alone, with_other, other_alone: o_alone, other_with, failed }));
    }

    let date = Command::new("date").arg("+%F").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let (c, x) = propose(&points, clean_max, &beside_results, &card, &date, "nextsycl models bench");
    println!("{}", serde_json::to_string_pretty(&json!({"model": id, "runs": report})).unwrap_or_default());
    println!("\n# proposed for model-cards/{}/{id}.yml (not written{})", me["kind"].as_str().unwrap_or(""), if args.iter().any(|a| a == "--write") { "; --write given" } else { ": add --write" });
    println!("{}\n{}", block("concurrency", &c), block("coexistence", &x));
    let path = nextsycl_models::catalog::cards_dir(cfg).join(me["kind"].as_str().unwrap_or("llm")).join(format!("{id}.yml"));
    if write_card(&path, &c, &x, args.iter().any(|a| a == "--write"))? {
        eprintln!("written to {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_kind_has_a_standard_request_in_one_place() {
        let img = standard(&json!({"kind": "image", "arch": "qwenimage21"})).unwrap();
        assert_eq!((img.path, img.unit), ("/v1/images/generations", "images"));
        assert!(standard(&json!({"kind": "image", "arch": "paddleocrvl"})).is_err());
        assert_eq!(standard(&json!({"kind": "audio", "arch": "voxcpm2", "speech": ["clone"]})).unwrap().path, "/v1/audio/speech");
        assert_eq!(standard(&json!({"kind": "audio", "arch": "acestep"})).unwrap().unit, "songs");
        assert!(standard(&json!({"kind": "audio", "arch": "whisper"})).is_err());
        assert_eq!(standard(&json!({"kind": "llm", "arch": "qwen35"})).unwrap().path, "/v1/chat/completions");
        assert!(standard(&json!({"kind": "video"})).unwrap_err().contains("no standard request"));
    }

    #[test]
    fn a_card_is_touched_only_with_write_and_only_with_a_valid_block() {
        let d = std::env::temp_dir().join(format!("nextsycl-bench-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("x.yml");
        let card = "id: x\nkind: image\nvram_gib: 8\n";
        std::fs::write(&f, card).unwrap();
        let (c, x) = (json!({"max_copies_per_card": 2, "recommended_copies": 1}), json!({"mode": "shared", "shared_with": []}));
        assert!(!write_card(&f, &c, &x, false).unwrap());
        assert_eq!(std::fs::read_to_string(&f).unwrap(), card, "no --write: not a byte changed");
        assert!(write_card(&f, &json!({"max_copies_per_card": 0}), &x, true).unwrap_err().contains("not valid"));
        assert_eq!(std::fs::read_to_string(&f).unwrap(), card, "a bad block: not written");
        assert!(write_card(&f, &c, &x, true).unwrap());
        let v: Value = serde_yaml::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        assert_eq!((v["vram_gib"].as_i64(), v["concurrency"]["max_copies_per_card"].as_i64()), (Some(8), Some(2)));
        std::fs::remove_dir_all(&d).unwrap();
    }

    fn pts(v: &[(u32, f64)]) -> Vec<ScalePoint> {
        v.iter().map(|(c, t)| ScalePoint { copies: *c, total_per_min: *t }).collect()
    }

    #[test]
    fn the_proposal_follows_the_scaling_and_the_co_running() {
        let ok = Beside { alone: 10.0, with_other: 9.0, other_alone: 6.0, other_with: 5.0, failed: false };
        let (c, x) = propose(&pts(&[(1, 10.0), (2, 11.8)]), 2, &[("whisper".into(), ok)], "RTX 5070", "2026-10-31", "bench");
        assert_eq!((c["max_copies_per_card"].as_u64(), c["recommended_copies"].as_u64()), (Some(2), Some(1)));
        assert_eq!(c["scaling"]["2"], 1.18);
        assert!(c["measured"]["notes"].as_str().unwrap().contains("whisper"));
        assert_eq!(x["mode"], "shared");
        // 3 copies ran clean and pay: 1.6x then 2.1x
        let (c, _) = propose(&pts(&[(1, 10.0), (2, 16.0), (3, 21.0)]), 3, &[], "c", "d", "m");
        assert_eq!(c["recommended_copies"], 3);
        // a copy count that failed is not the maximum, even if a later point exists
        let (c, _) = propose(&pts(&[(1, 10.0), (2, 16.0)]), 2, &[], "c", "d", "m");
        assert_eq!(c["max_copies_per_card"], 2);
        // a collapse beside another model: exclusive, with the reason
        let bad = Beside { with_other: 3.0, ..ok };
        let (_, x) = propose(&pts(&[(1, 10.0)]), 1, &[("trellis2".into(), bad)], "c", "d", "m");
        assert_eq!(x["mode"], "exclusive");
        assert!(x["exclusive_reason"].as_str().unwrap().contains("beside trellis2"), "{x}");
        // and the block is a valid card block
        let v = json!({"concurrency": propose(&pts(&[(1, 10.0), (2, 16.0)]), 2, &[], "c", "d", "m").0, "coexistence": x});
        assert!(profile::Profile::from_card(&v).is_ok(), "{v}");
    }

    #[test]
    fn write_replaces_the_blocks_and_keeps_the_rest_of_the_card() {
        let card = "id: x\nkind: image\nconcurrency:\n  max_copies_per_card: 1\n  recommended_copies: 1\nvram_gib: 8\ncoexistence:\n  mode: shared\nfiles:\n  model: k\n";
        let c = json!({"max_copies_per_card": 2, "recommended_copies": 1});
        let x = json!({"mode": "shared", "shared_with": []});
        let out = replace_blocks(card, &c, &x);
        assert!(out.contains("vram_gib: 8") && out.contains("files:\n  model: k"));
        assert_eq!(out.matches("concurrency:").count(), 1);
        assert_eq!(out.matches("coexistence:").count(), 1);
        assert!(out.contains("concurrency: {\"max_copies_per_card\":2,\"recommended_copies\":1}"));
        // a card without the blocks gets them at the end
        let out = replace_blocks("id: y\nkind: audio\n", &c, &x);
        assert!(out.ends_with("coexistence: {\"mode\":\"shared\",\"shared_with\":[]}\n"));
        let v: Value = serde_yaml::from_str(&out).unwrap();
        assert!(profile::Profile::from_card(&v).is_ok());
    }
}
