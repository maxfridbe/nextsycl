//! `nextsycl image ...`: the image commands. `gen --local` runs the engine in this process (inside the image; the
//! server path comes with `image serve`); `check` compares an engine's stages with a reference's dumps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_image::{ImageRequest, LoraUse, ModelFiles, Sampler, Schedule};
use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

const USAGE: &str = "nextsycl image gen \"<prompt>\" [--model ID] [--size WxH | --aspect W:H] [--steps N] [--seed N] [--n N] [--sampler S]
                   [--schedule S] [--lora NAME[:SCALE]]... [--out FILE|DIR] [--rgba] [--local] [--gpu N]
nextsycl image check <dump dir> [--model ID] [--stages te,dit,steps,vae] [--lora NAME[:SCALE]]... [--gpu N]
nextsycl image engines | selftest [--gpu N]";

/// The image engines this program has (image/<arch>)
pub fn engines() -> Vec<nextsycl_image::ImageKind> {
    vec![nextsycl_image_qwenimage21::kind(), nextsycl_image_example::kind()]
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// The registered image model `id`, or the first one: (entry, files by role)
fn model(cfg: &Config, id: Option<&str>) -> Result<(Value, ModelFiles), String> {
    let all = models::all(cfg)?;
    let m = match id {
        Some(id) => all.into_iter().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id} (nextsycl models search --kind image)"))?,
        None => all.into_iter().find(|m| models::kind_of(m) == "image").ok_or("no image model (nextsycl models pull qwen-image-2.1-q8)")?,
    };
    if models::kind_of(&m) != "image" {
        return Err(format!("{} is a {} model, not an image one", m["id"].as_str().unwrap_or("?"), models::kind_of(&m)));
    }
    let files: ModelFiles = m["files"].as_object().into_iter().flatten().filter_map(|(r, p)| Some((r.clone(), PathBuf::from(p.as_str()?)))).collect();
    Ok((m, files))
}

/// Every `--lora NAME[:SCALE]`: a registered LoRA's id or a file
fn loras(cfg: &Config, args: &[String]) -> Result<Vec<LoraUse>, String> {
    let all = models::all(cfg)?;
    let find = |n: &str| -> Option<PathBuf> {
        all.iter()
            .find(|m| m["id"] == n && models::kind_of(m) == "lora")
            .and_then(|m| m["file"].as_str().map(PathBuf::from))
            .or_else(|| Path::new(n).is_file().then(|| PathBuf::from(n)))
    };
    args.iter().enumerate().filter(|(_, a)| *a == "--lora").map(|(i, _)| {
        let v = args.get(i + 1).ok_or("--lora NAME[:SCALE]")?;
        LoraUse::parse(v, &find)
    }).collect()
}

fn load(cfg: &Config, args: &[String]) -> Result<(Value, Box<dyn nextsycl_image::ImageEngine>), String> {
    nextsycl_core::use_kind("image");
    let (m, files) = model(cfg, opt(args, "--model"))?;
    let arch = m["arch"].as_str().unwrap_or("");
    let ks = engines();
    let k = nextsycl_image::kind_for(&ks, arch)?;
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let gpu = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
    let mut log = |s: String| eprintln!("{s}");
    let o = nextsycl_image::LoadOptions { merge_loras: loras(cfg, args)? };
    let e = (k.load)(&files, &[gpu], &o, &mut log).map_err(|e| e.0)?;
    Ok((m, e))
}

/// `--size WxH`, or `--aspect W:H` at about the default's area (sides multiples of 16)
fn size(args: &[String], d: (u32, u32)) -> Result<(u32, u32), String> {
    if let Some(s) = opt(args, "--size") {
        let (w, h) = s.split_once('x').ok_or("--size WxH")?;
        return Ok((w.parse().map_err(|_| "--size WxH")?, h.parse().map_err(|_| "--size WxH")?));
    }
    if let Some(a) = opt(args, "--aspect") {
        let (w, h) = a.split_once(':').ok_or("--aspect W:H")?;
        let (w, h): (f64, f64) = (w.parse().map_err(|_| "--aspect W:H")?, h.parse().map_err(|_| "--aspect W:H")?);
        let area = d.0 as f64 * d.1 as f64;
        let hh = (area * h / w).sqrt();
        let r = |x: f64| ((x / 16.0).round() as u32 * 16).max(16);
        return Ok((r(hh * w / h), r(hh)));
    }
    Ok(d)
}

fn gen(cfg: &Config, args: &[String]) -> Result<(), String> {
    let prompt = args.first().filter(|p| !p.starts_with("--")).ok_or(USAGE)?.clone();
    // the server path comes with `image serve`; until then every generation runs here
    let (m, e) = load(cfg, args)?;
    let d = e.defaults();
    let (width, height) = size(args, (d.width, d.height))?;
    let seed: u64 = opt(args, "--seed").map(|s| s.parse().map_err(|_| "--seed N")).transpose()?.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_nanos() as u64 % 100_000).unwrap_or(0)
    });
    let req = ImageRequest {
        prompt: prompt.clone(),
        width: Some(width),
        height: Some(height),
        steps: opt(args, "--steps").map(|s| s.parse().map_err(|_| "--steps N")).transpose()?,
        seed,
        n: opt(args, "--n").map(|s| s.parse().map_err(|_| "--n N")).transpose()?.unwrap_or(1),
        sampler: opt(args, "--sampler").map(|s| Sampler::parse(s).ok_or(format!("--sampler: {s}?"))).transpose()?,
        schedule: opt(args, "--schedule").map(|s| Schedule::parse(s).ok_or(format!("--schedule: {s}?"))).transpose()?,
        rgba: args.iter().any(|a| a == "--rgba"),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let mut last = 0f64;
    let pics = e.generate(&req, &mut |s| {
        if s.at > 0 {
            let per = (s.seconds - last).max(0.0);
            last = s.seconds;
            eprint!("\rpicture {}  step {}/{}  {:.2} s/step  {:.0} s   ", s.picture + 1, s.at, s.of, per, s.seconds);
        } else {
            last = s.seconds;
            eprintln!("prompt encoded in {:.1} s", s.seconds);
        }
    }).map_err(|e| e.0)?;
    eprintln!();
    let id = m["id"].as_str().unwrap_or("image");
    let out = opt(args, "--out").map(PathBuf::from);
    let mut saved = Vec::new();
    for (i, p) in pics.iter().enumerate() {
        let name = format!("{id}-{seed}-{}.png", i + 1);
        let path = match &out {
            Some(o) if o.is_dir() => o.join(name),
            Some(o) if pics.len() == 1 => o.clone(),
            Some(o) => o.with_file_name(format!("{}-{}.png", o.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), i + 1)),
            None => PathBuf::from(name),
        };
        let lora_list: Vec<String> = loras(cfg, args)?.iter().map(|l| format!("{}:{}", l.name, l.scale)).collect();
        let meta = [("prompt", prompt.clone()), ("model", id.to_string()), ("seed", (seed + i as u64).to_string()),
                    ("size", format!("{}x{}", p.width, p.height)), ("steps", req.steps.unwrap_or(d.steps).to_string()),
                    ("loras", lora_list.join(","))];
        p.write_png(&path, &meta)?;
        saved.push(path.display().to_string());
    }
    println!("saved {}  ({}x{}, {:.1} s, settings in the PNG)", saved.join(", "), width, height, t0.elapsed().as_secs_f64());
    Ok(())
}

fn check(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = Path::new(args.first().ok_or(USAGE)?);
    nextsycl_core::use_kind("image");
    let (m, files) = model(cfg, opt(args, "--model"))?;
    if m["arch"] != nextsycl_image_qwenimage21::ARCH {
        return Err(format!("check knows {} only", nextsycl_image_qwenimage21::ARCH));
    }
    let stages: Vec<&str> = opt(args, "--stages").unwrap_or("te,dit,vae").split(',').collect();
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let gpu: Arc<nextsycl_core::Gpu> = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = nextsycl_image_qwenimage21::QwenImage21::load(&files, &gpu, &loras(cfg, args)?, &mut log).map_err(|e| e.0)?;
    let worst = nextsycl_image_qwenimage21::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?;
    println!("worst relative error {worst:.2e}");
    Ok(())
}

/// `nextsycl image <command>`
pub fn cmd(cfg: &Config, args: &[String], selftest: impl Fn(&[String]) -> Result<(), String>) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("engines") => {
            super::list_engines(engines().iter().map(|k| (k.archs.join(", "), k.name)));
            Ok(())
        }
        Some("selftest") => selftest(rest),
        Some("gen") => gen(cfg, rest),
        Some("check") => check(cfg, rest),
        _ => Err(USAGE.into()),
    }
}
