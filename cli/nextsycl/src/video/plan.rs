//! `nextsycl video plan`: what a denoiser step costs on each of this box's GPUs, measured, for the studio's canvas table
//! (`/api/plan`, the front end's CanvasPicker).
//!
//!     nextsycl video plan measure [--gpu N ...] [--engine NAME ...] [--tokens 2048,4096,...] [--no-clip] [--no-cells]
//!     nextsycl video plan show
//!
//! For every GPU the engine serves, `measure` queues `bench-blocks` over a range of token counts (the 50 blocks of
//! one step, pinned to that GPU), then one short real clip: a clip's step is the 50 blocks plus its own small extras
//! (embeddings, the final layer, the sampler), so the clip's median step over the bench at the clip's token count is
//! the GPU's step factor. The result goes to `plan.json` in the studio's directory, which the studio reads on every
//! `/api/plan` - a new measurement shows without a restart. It also benches every cell of the canvas table (each
//! canvas at each of LENGTHS, at the cell's exact token count): the cell's own step time and peak memory, or "over"
//! when the engine refuses it (it counts its allocations and refuses past its cap - no stall). GPUs run their jobs side by side; one the engine shares
//! with another program is handed over the usual way (lock file, model switch).
//!
//! Each denoiser the engine has (INT8, and the GGUF forms of NS_VIDEO_ENGINES: `--engine NAME` for some) is measured on its
//! own: one entry per GPU and engine (`"engine"`; an entry without one is INT8, from before). A GPU's jobs are queued
//! engine after engine, so its worker changes model once per engine.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nextsycl_serve::video::studio::http::{Error, Result};
use serde_json::{json, Value};

use super::client;
use nextsycl_models::config::Config;

/// The token counts measured by default: every canvas the studio offers falls in this range from 1 s up to the
/// longest clip that fits.
pub const TOKENS: [u64; 11] = [2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 47104];

/// The canvas table's clip lengths (wfe CanvasPicker's LENGTHS): every canvas at each is a measured cell.
pub const LENGTHS: [f64; 5] = [5.0, 8.0, 10.0, 12.0, 15.0];

/// Latent tokens of a canvas at a clip length - the front end's tokensFor, the legacy server's tokens_for.
pub fn tokens_for(w: i64, h: i64, seconds: f64) -> u64 {
    let mut n = ((seconds * 24.0).round() as i64).max(5);
    while n % 17 != 5 {
        n += 1;
    }
    let ltf = if n <= 5 { 2 } else { (n - 5) / 17 * 5 + 2 };
    (ltf * (w * h / 1024) + ((n as f64 / 24.0 * 40.0).round() as i64) + 10 + 336) as u64
}

/// The calibration clip: the production canvas, short, no upscale (only its sampling is used).
const CLIP: (u64, u64, u64, u64) = (768, 576, 3, 4); // width, height, seconds, steps

pub fn file(cfg: &Config) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(cfg.get("NS_VIDEO_STUDIO_DIR").unwrap_or_else(|| format!("{home}/.local/share/nextsycl/studio"))).join("plan.json")
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn add(spec: Value) -> Result<u64> {
    client::post("/engine/jobs", Some(&spec))?["id"].as_u64().ok_or_else(|| Error("no job id in the answer".into()))
}

/// Wait for a job to end; its record, or the reason it did not finish.
fn wait(id: u64) -> Result<Value> {
    loop {
        let j = client::get(&format!("/engine/jobs/{id}"))?;
        match j["state"].as_str().unwrap_or("") {
            "done" => return Ok(j),
            s @ ("failed" | "cancelled") => return Err(Error(format!("job {id} {s}: {}", j["error"].as_str().unwrap_or("")))),
            _ => std::thread::sleep(Duration::from_secs(2)),
        }
    }
}

/// Seconds per step at `t` tokens: linear between the measured points, the end segments' slope beyond them.
pub fn interp(points: &[(f64, f64)], t: f64) -> f64 {
    match points.len() {
        0 => 0.0,
        1 => points[0].1 * t / points[0].0,
        n => {
            let k = points.iter().position(|p| p.0 >= t).unwrap_or(n - 1).clamp(1, n - 1);
            let (a, b) = (points[k - 1], points[k]);
            a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0)
        }
    }
}

/// The least-squares fit of s = a t^2 + b t (the shape attention and the linears give), for clients that want a
/// curve rather than the points.
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let (mut s4, mut s3, mut s2, mut y2, mut y1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(t, s) in points {
        s4 += t.powi(4);
        s3 += t.powi(3);
        s2 += t * t;
        y2 += s * t * t;
        y1 += s * t;
    }
    let det = s4 * s2 - s3 * s3;
    if det.abs() < 1e-30 {
        return (0.0, 0.0);
    }
    ((y2 * s2 - y1 * s3) / det, (s4 * y1 - s3 * y2) / det)
}

/// The steps' times of a clip's log: `tokens : N in the denoiser` and `sampled: k steps in X s (t1 t2 ...)`.
fn clip_steps(job: &Value) -> Option<(f64, Vec<f64>)> {
    let log: Vec<&str> = job["log"].as_array()?.iter().filter_map(Value::as_str).collect();
    let tokens = log.iter().find_map(|l| l.strip_prefix("tokens : ")?.split_whitespace().next()?.parse().ok())?;
    let s = log.iter().find(|l| l.starts_with("sampled:"))?;
    let times = s[s.find('(')? + 1..s.rfind(')')?].split_whitespace().filter_map(|v| v.parse().ok()).collect();
    Some((tokens, times))
}

/// One GPU's queued jobs for one engine: (tokens, job id) for each bench, and the calibration clip's job.
struct Queued {
    gpu: u64,
    engine: String,
    bench: Vec<(u64, u64)>,
    clip: Option<u64>,
    /// the canvas table's cells: (key "WxH|S", tokens, job) - one job per distinct token count
    cells: Vec<(String, u64, u64)>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn run(cfg: &Config, raw: &[String]) -> Result<()> {
    match raw.first().map(String::as_str) {
        Some("measure") => measure(cfg, &raw[1..]),
        Some("show") | None => {
            let p = file(cfg);
            let v: Value = serde_json::from_slice(&std::fs::read(&p).map_err(|e| Error(format!("{}: {e} (nextsycl video plan measure)", p.display())))?)
                .map_err(|e| Error(format!("{}: {e}", p.display())))?;
            for g in v["gpus"].as_array().into_iter().flatten() {
                let scale = g["step_scale"].as_f64().unwrap_or(1.0);
                println!("GPU {} {}, {} - a clip's step is {scale:.3} x the blocks (from a {:.0}-token clip)", g["gpu"],
                         g["name"].as_str().unwrap_or("?"), g["engine"].as_str().unwrap_or("INT8"), g["clip_tokens"].as_f64().unwrap_or(0.0));
                for p in g["points"].as_array().into_iter().flatten() {
                    let (t, s) = (p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0));
                    println!("  {t:>6.0} tokens  {s:6.2} s (blocks)  {:6.2} s per clip step", s * scale);
                }
            }
            Ok(())
        }
        Some(other) => Err(Error(format!("nextsycl video plan: measure or show, not {other:?}"))),
    }
}

fn measure(cfg: &Config, raw: &[String]) -> Result<()> {
    let (mut gpus, mut engines, mut tokens, mut clip, mut cells) = (Vec::<u64>::new(), Vec::<String>::new(), TOKENS.to_vec(), true, true);
    let mut it = raw.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--gpu" => gpus.push(it.next().and_then(|v| v.parse().ok()).ok_or("--gpu N")?),
            "--engine" => engines.push(it.next().ok_or("--engine NAME")?.to_uppercase()),
            "--tokens" => {
                tokens = it.next().ok_or("--tokens 2048,4096,...")?.split(',').map(|v| v.trim().parse().map_err(|_| Error(format!("{v}: not a token count")))).collect::<Result<_>>()?
            }
            "--no-clip" => clip = false,
            "--no-cells" => cells = false,
            other => return Err(Error(format!("nextsycl video plan measure: unknown option {other}"))),
        }
    }
    let status = client::get("/engine/status")?;
    let served: Vec<Value> = status["gpus"].as_array().cloned().unwrap_or_default();
    if gpus.is_empty() {
        gpus = served.iter().filter_map(|g| g["gpu"].as_u64()).collect();
    }
    // the daemon's models are registry ids; the plan and the front end name them by quant (INT8, Q6_K, Q4_K_M)
    let names = super::service::engine_names(cfg);
    let quant = |id: &str| names.iter().find(|(_, i)| i == id).map_or_else(|| id.to_string(), |(q, _)| q.clone());
    let id_of = |q: &str| names.iter().find(|(n, _)| n == q).map_or_else(|| q.to_string(), |(_, i)| i.clone());
    let have: Vec<String> = status["engines"].as_array().into_iter().flatten().filter_map(|e| e["name"].as_str().map(&quant)).collect();
    let have = if have.is_empty() { vec!["INT8".to_string()] } else { have };
    if engines.is_empty() {
        engines = have.clone();
    }
    if let Some(bad) = engines.iter().find(|e| !have.contains(e)) {
        return Err(Error(format!("--engine {bad}: the engine has {have:?}")));
    }
    // every job first, so the GPUs measure side by side
    let mut queued: Vec<Queued> = Vec::new();
    for &g in &gpus {
      for engine in &engines {
        let id = id_of(engine);
        let en = id.as_str();
        let bench = tokens.iter().map(|&t| Ok((t, add(json!({"kind": "bench-blocks", "tokens": t, "gpu": g, "engine": en}))?))).collect::<Result<Vec<_>>>()?;
        let c = if clip {
            let (w, h, s, steps) = CLIP;
            Some(add(json!({"kind": "generate", "gpu": g, "engine": en, "prompt": "a lighthouse on a cliff at dusk, waves breaking below, a slow push in",
                            "width": w, "height": h, "seconds": s, "steps": steps, "seed": 0, "upscale": 1,
                            "out": super::service::out_dir(cfg).join(format!("plan-measure-gpu{g}-{engine}.mp4"))}))?)
        } else {
            None
        };
        let mut cell_jobs: Vec<(String, u64, u64)> = Vec::new();
        if cells {
            let mut by_tokens: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
            for (w, h, _, _) in nextsycl_serve::video::studio::library::CANVASES {
                for sec in LENGTHS {
                    let t = tokens_for(w, h, sec);
                    let id = match by_tokens.get(&t) {
                        Some(id) => *id,
                        None => {
                            let id = add(json!({"kind": "bench-blocks", "tokens": t, "gpu": g, "engine": en}))?;
                            by_tokens.insert(t, id);
                            id
                        }
                    };
                    cell_jobs.push((format!("{w}x{h}|{sec}"), t, id));
                }
            }
        }
        eprintln!("GPU {g}, {engine}: {} bench jobs{}{} queued", bench.len(), if c.is_some() { ", a calibration clip" } else { "" },
                  if cells { ", the canvas table's cells" } else { "" });
        queued.push(Queued { gpu: g, engine: engine.clone(), bench, clip: c, cells: cell_jobs });
      }
    }
    let mut out = Vec::new();
    for Queued { gpu: g, engine, bench, clip: c, cells: cell_jobs } in queued {
        let who = format!("GPU {g}, {engine}");
        let mut points = Vec::new();
        for (t, id) in bench {
            let j = wait(id)?;
            let s = j["result"]["seconds"].as_f64().ok_or_else(|| Error(format!("job {id}: no seconds in its result")))?;
            eprintln!("{who}: {t:>6} tokens  {s:6.2} s/step (blocks), {:.1} GiB", j["result"]["gib_in_use"].as_f64().unwrap_or(0.0));
            points.push((t as f64, s));
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut scale, mut ctok) = (1.0, Value::Null);
        if let Some(id) = c {
            let j = wait(id)?;
            let (t, times) = clip_steps(&j).ok_or_else(|| Error(format!("job {id}: no step times in its log")))?;
            // the first step carries one-time work (kernels' first use); the rest are the steady state
            let steady = median(if times.len() > 1 { times[1..].to_vec() } else { times });
            scale = steady / interp(&points, t);
            ctok = json!(t);
            eprintln!("{who}: a {t}-token clip steps in {steady:.2} s: {scale:.3} x the blocks");
        }
        // the cells: the bench of each distinct token count once, read for every cell that has it
        let mut cell_out = serde_json::Map::new();
        let mut seen: std::collections::BTreeMap<u64, Value> = std::collections::BTreeMap::new();
        for (key, t, id) in cell_jobs {
            let v = match seen.get(&t) {
                Some(v) => v.clone(),
                None => {
                    let v = match wait(id) {
                        Ok(j) => {
                            let s = j["result"]["seconds"].as_f64().unwrap_or(0.0) * scale;
                            let gib = j["result"]["gib_in_use"].as_f64().unwrap_or(0.0);
                            eprintln!("{who}: cell tokens {t:>6}: {s:6.2} s per clip step, {gib:.1} GiB");
                            json!({"tokens": t, "s_per_step": (s * 100.0).round() / 100.0, "gib": (gib * 10.0).round() / 10.0})
                        }
                        // only the engine's refusal of the size is "over"; anything else (a stopped daemon) ends the run
                        // rather than writing a table of false overs
                        Err(e) if e.0.contains("cap") || e.0.contains("memory") || e.0.contains("refused") => {
                            eprintln!("{who}: cell tokens {t:>6}: does not fit ({e})");
                            json!({"tokens": t, "over": true})
                        }
                        Err(e) => return Err(Error(format!("{who}, a cell of {t} tokens: {e} - nothing written"))),
                    };
                    seen.insert(t, v.clone());
                    v
                }
            };
            cell_out.insert(key, v);
        }
        let name = served.iter().find(|s| s["gpu"].as_u64() == Some(g)).and_then(|s| s["name"].as_str()).unwrap_or("?").to_string();
        let shared = served.iter().find(|s| s["gpu"].as_u64() == Some(g)).map(|s| s["shared"] == true).unwrap_or(false);
        let (a, b) = fit(&points.iter().map(|&(t, s)| (t, s * scale)).collect::<Vec<_>>());
        out.push(json!({"gpu": g, "engine": engine, "name": name, "shared": shared, "step_scale": scale, "clip_tokens": ctok,
                        "points": points.iter().map(|&(t, s)| json!([t, s])).collect::<Vec<_>>(), "step_a": a, "step_b": b,
                        "cells": Value::Object(cell_out)}));
    }
    let p = file(cfg);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    // keep the (GPU, engine) entries measured before that this run did not cover
    let covered = |o: &Value| gpus.contains(&o["gpu"].as_u64().unwrap_or(u64::MAX)) && engines.iter().any(|e| e == o["engine"].as_str().unwrap_or("INT8"));
    let mut all: Vec<Value> = std::fs::read(&p).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["gpus"].as_array().cloned()).unwrap_or_default()
        .into_iter().filter(|o| !covered(o)).collect();
    all.extend(out);
    for o in &mut all {
        if o["engine"].is_null() {
            o["engine"] = json!("INT8");
        }
    }
    all.sort_by_key(|g| (g["gpu"].as_u64().unwrap_or(0), g["engine"].as_str().unwrap_or("").to_string()));
    let text = serde_json::to_string_pretty(&json!({"measured": now(), "gpus": all})).map_err(|e| Error(e.to_string()))?;
    std::fs::write(&p, text + "\n")?;
    println!("written to {}", p.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_is_linear_between_points_and_extends_the_ends() {
        let p = [(1000.0, 1.0), (2000.0, 3.0), (4000.0, 7.0)];
        assert!((interp(&p, 1500.0) - 2.0).abs() < 1e-9);
        assert!((interp(&p, 4000.0) - 7.0).abs() < 1e-9);
        assert!((interp(&p, 5000.0) - 9.0).abs() < 1e-9); // the last segment's slope
        assert!((interp(&p, 500.0) - 0.0).abs() < 1e-9); // the first segment's slope
    }

    #[test]
    fn fit_recovers_a_quadratic() {
        let p: Vec<(f64, f64)> = [2048.0, 8192.0, 16384.0, 47104.0].iter().map(|&t| (t, 2e-8 * t * t + 3e-4 * t)).collect();
        let (a, b) = fit(&p);
        assert!((a - 2e-8).abs() < 1e-12 && (b - 3e-4).abs() < 1e-8, "{a} {b}");
    }

    #[test]
    fn tokens_match_the_front_end() {
        // the front end's tokensFor and the legacy server's tokens_for give these
        assert_eq!(tokens_for(768, 576, 5.0), 16537);
        assert_eq!(tokens_for(768, 576, 15.0), 47173);
    }

    #[test]
    fn clip_steps_reads_the_log() {
        let j = json!({"log": ["tokens : 9178 in the denoiser", "sampled: 3 steps in 9.0 s (4.10 2.45 2.47)"]});
        assert_eq!(clip_steps(&j), Some((9178.0, vec![4.10, 2.45, 2.47])));
    }
}
