//! `nextsycl image ...`: the image commands. `gen --local` runs the engine in this process (inside the image; the
//! server path comes with `image serve`); `check` compares an engine's stages with a reference's dumps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_image::{ImageRequest, LoraUse, ModelFiles, Sampler, Schedule};
use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

const USAGE: &str = "nextsycl image edit \"<instructions>\" --image PICTURE [--image REFERENCE]... [gen's options]
nextsycl image gen \"<prompt>\" [--model ID] [--size WxH | --aspect W:H] [--steps N] [--seed N] [--n N] [--sampler S]
                   [--schedule S] [--lora NAME[:SCALE]]... [--set NAME=VALUE]... [--out FILE|DIR] [--rgba] [--local] [--gpu N]
nextsycl image serve [MODEL] [--port 8086] [--host 127.0.0.1] [--gpu N] [--lora NAME[:SCALE]]... [--set NAME=VALUE]...
                     [--wfe] [--out DIR] [--cors ORIGIN]
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

/// The load options: every `--lora` and the settings of the model's registry entry and of those LoRAs' entries (a
/// few-step LoRA brings its sigma preset), then `--set NAME=VALUE`s
fn options(cfg: &Config, m: &Value, args: &[String]) -> Result<nextsycl_image::LoadOptions, String> {
    let merge_loras = loras(cfg, args)?;
    let all = models::all(cfg)?;
    let mut settings = std::collections::BTreeMap::new();
    let mut take = |e: &Value| {
        for (k, v) in e["env"].as_object().into_iter().flatten() {
            if let Some(v) = v.as_str() {
                settings.insert(k.clone(), v.to_string());
            }
        }
    };
    take(m);
    for l in &merge_loras {
        if let Some(e) = all.iter().find(|e| e["id"] == l.name.as_str()) {
            take(e);
        }
    }
    for (i, a) in args.iter().enumerate() {
        if a == "--set" {
            let (k, v) = args.get(i + 1).and_then(|x| x.split_once('=')).ok_or("--set NAME=VALUE")?;
            settings.insert(k.to_string(), v.to_string());
        }
    }
    // the --opt-NAMEs the engine takes at load, by the variables they name (an unknown one: its options listed)
    let arch = m["arch"].as_str().unwrap_or("");
    let ks = engines();
    let k = nextsycl_image::kind_for(&ks, arch)?;
    for (var, v) in super::opt_env(k.options, nextsycl_core::At::Load, k.name)? {
        settings.insert(var, v);
    }
    Ok(nextsycl_image::LoadOptions { merge_loras, settings })
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
    let o = options(cfg, &m, args)?;
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
    gen_or_edit(cfg, args, false)
}

/// `nextsycl image edit "<instructions>" --image P [--image R]...`: the first picture changed as described, the others
/// as references ("put the hat from picture 2 on the fox")
fn edit(cfg: &Config, args: &[String]) -> Result<(), String> {
    gen_or_edit(cfg, args, true)
}

fn gen_or_edit(cfg: &Config, args: &[String], editing: bool) -> Result<(), String> {
    let prompt = args.first().filter(|p| !p.starts_with("--")).ok_or(USAGE)?.clone();
    let pics: Vec<nextsycl_image::Picture> = args.iter().enumerate().filter(|(_, a)| *a == "--image")
        .map(|(i, _)| args.get(i + 1).ok_or("--image PICTURE".to_string()).and_then(|p| nextsycl_image::Picture::read_png(Path::new(p))))
        .collect::<Result<_, _>>()?;
    if editing && pics.is_empty() {
        return Err("image edit: --image PICTURE (a PNG)".into());
    }
    // the server path comes with `image serve`; until then every generation runs here
    let (m, e) = load(cfg, args)?;
    let d = e.defaults();
    // an edit's size: its picture's aspect unless one is given
    let given = opt(args, "--size").is_some() || opt(args, "--aspect").is_some();
    let (width, height) = size(args, (d.width, d.height))?;
    let seed: u64 = opt(args, "--seed").map(|s| s.parse().map_err(|_| "--seed N")).transpose()?.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_nanos() as u64 % 100_000).unwrap_or(0)
    });
    let req = ImageRequest {
        prompt: prompt.clone(),
        width: (!editing || given).then_some(width),
        height: (!editing || given).then_some(height),
        steps: opt(args, "--steps").map(|s| s.parse().map_err(|_| "--steps N")).transpose()?,
        seed,
        n: opt(args, "--n").map(|s| s.parse().map_err(|_| "--n N")).transpose()?.unwrap_or(1),
        sampler: opt(args, "--sampler").map(|s| Sampler::parse(s).ok_or(format!("--sampler: {s}?"))).transpose()?,
        schedule: opt(args, "--schedule").map(|s| Schedule::parse(s).ok_or(format!("--schedule: {s}?"))).transpose()?,
        rgba: args.iter().any(|a| a == "--rgba"),
        negative: opt(args, "--negative").map(str::to_string),
        cfg: opt(args, "--cfg").map(|s| s.parse().map_err(|_| "--cfg X")).transpose()?,
        edit: editing.then(|| nextsycl_image::Edit { image: pics[0].clone(), refs: pics[1..].to_vec(), mask: None, strength: 1.0 }),
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
                    ("edit", if editing { format!("{} picture(s)", pics.len()) } else { String::new() }),
                    ("loras", lora_list.join(","))];
        p.write_png(&path, &meta)?;
        saved.push(path.display().to_string());
    }
    let (pw, ph) = pics_size(&saved);
    println!("saved {}  ({pw}x{ph}, {:.1} s, settings in the PNG)", saved.join(", "), t0.elapsed().as_secs_f64());
    Ok(())
}

/// The first saved picture's size
fn pics_size(saved: &[String]) -> (u32, u32) {
    saved.first().and_then(|p| nextsycl_image::Picture::read_png(Path::new(p)).ok()).map_or((0, 0), |p| (p.width, p.height))
}

fn check(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = Path::new(args.first().ok_or(USAGE)?);
    nextsycl_core::use_kind("image");
    let (m, files) = model(cfg, opt(args, "--model"))?;
    if m["arch"] != nextsycl_image_qwenimage21::ARCH {
        return Err(format!("check knows {} only", nextsycl_image_qwenimage21::ARCH));
    }
    let stages: Vec<&str> = opt(args, "--stages").unwrap_or("te,dit,vae").split(',').collect(); // + edit-pre, edit-te, edit-vae, edit-dit
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let gpu: Arc<nextsycl_core::Gpu> = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = nextsycl_image_qwenimage21::QwenImage21::load(&files, &gpu, &options(cfg, &m, args)?, &mut log).map_err(|e| e.0)?;
    let worst = nextsycl_image_qwenimage21::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?;
    println!("worst relative error {worst:.2e}");
    Ok(())
}

/// The LoRAs registered for the model's architecture
fn loras_for(cfg: &Config, m: &Value) -> Result<Vec<Value>, String> {
    Ok(models::all(cfg)?.into_iter().filter(|e| models::kind_of(e) == "lora" && e["arch"] == m["arch"] && e["enabled"] != false).collect())
}

/// Where pictures go: --out, NS_IMAGE_OUT, ~/.local/share/nextsycl/images
fn out_dir(cfg: &Config, args: &[String]) -> PathBuf {
    opt(args, "--out").map(PathBuf::from).or_else(|| cfg.get("NS_IMAGE_OUT").map(PathBuf::from)).unwrap_or_else(|| {
        PathBuf::from(format!("{}/.local/share/nextsycl/images", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())))
    })
}

/// `nextsycl image serve`: from the host, the build image as container `nextsycl-image` (the GPU, the program and its
/// libraries, the registry, the model's and its LoRAs' files read-only, the output directory) running this command
/// inside with `--here`; there, the engine loaded and served (nextsycl_serve::image)
fn serve(cfg: &Config, args: &[String]) -> Result<(), String> {
    let id = args.first().filter(|a| !a.starts_with("--")).map(String::as_str).or_else(|| opt(args, "--model"));
    let (m, files) = model(cfg, id)?;
    let model_id = m["id"].as_str().unwrap_or("image").to_string();
    let out = out_dir(cfg, args);
    if args.iter().any(|a| a == "--here") {
        return serve_here(cfg, &m, files, args, out);
    }
    use crate::container::{mount, Ce};
    // the engine options checked here, before a container starts (they travel inside with the other arguments)
    options(cfg, &m, args)?;
    let ce = Ce::new(cfg)?;
    ce.need_image()?;
    for f in ["nextsycl", "libnextsycl-image.so"] {
        if !cfg.dist.join(f).exists() {
            return Err(format!("{}: not built yet (./build.sh)", cfg.dist.join(f).display()));
        }
    }
    if args.iter().any(|a| a == "--wfe") && !cfg.dist.join("wfe/image/index.html").exists() {
        return Err(format!("{}: the web front end is not built yet (./build.sh wfe)", cfg.dist.join("wfe").display()));
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    const NAME: &str = "nextsycl-image";
    if ce.running(NAME) {
        return Err(format!("{NAME} is running already ({} stop {NAME})", ce.bin));
    }
    ce.remove(NAME);
    // --init: the server is not PID 1, so a stop's SIGTERM ends it (PID 1 ignores signals it does not handle)
    let mut a: Vec<String> = vec!["run".into(), "--rm".into(), "--init".into(), "--name".into(), NAME.into(), "--network".into(), "host".into(),
                                  "--stop-timeout".into(), "120".into()];
    a.extend(ce.user_args());
    a.extend(ce.gpu_args());
    a.extend(mount(&cfg.dist, "/app", true));
    let reg = models::registry(cfg);
    let mut dirs = std::collections::BTreeSet::new();
    let mut paths = models::paths(&m);
    for l in loras_for(cfg, &m)? {
        paths.extend(models::paths(&l));
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
    // the engines' own settings from the configuration (NS_QI_INT8=1 ...)
    for (k, v) in cfg.with_prefix("NS_QI_").into_iter().chain(cfg.with_prefix("NSD_")) {
        a.extend(["-e".into(), format!("{k}={v}")]);
    }
    a.extend([ce.image.clone(), "bash".into(), "-c".into(),
              "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl image serve \"$@\"".into(), "nextsycl".into()]);
    a.push(model_id);
    let mut it = args.iter().skip(if id.is_some() && !args.first().is_some_and(|x| x.starts_with("--")) { 1 } else { 0 });
    while let Some(x) = it.next() {
        if x == "--out" || x == "--model" {
            it.next();
            continue;
        }
        a.push(x.clone());
    }
    a.extend(["--out".into(), out.to_string_lossy().into_owned(), "--here".into()]);
    let st = ce.cmd().args(&a).status().map_err(|e| e.to_string())?;
    if st.success() { Ok(()) } else { Err(format!("the image server ended ({st})")) }
}

fn serve_here(cfg: &Config, m: &Value, files: ModelFiles, args: &[String], out: PathBuf) -> Result<(), String> {
    use nextsycl_serve::image::{ImageServer, KnownLora};
    nextsycl_core::use_kind("image");
    let arch = m["arch"].as_str().unwrap_or("").to_string();
    let ks = engines();
    let load_fn = nextsycl_image::kind_for(&ks, &arch)?.load;
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let gpu = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
    let first = options(cfg, m, args)?;
    let loras = loras_for(cfg, m)?;
    let known: Vec<KnownLora> = loras.iter().filter_map(|l| Some(KnownLora {
        id: l["id"].as_str()?.to_string(), path: PathBuf::from(l["file"].as_str()?), title: l["title"].as_str().unwrap_or("").to_string(),
    })).collect();
    // the settings without any LoRA's: a reload adds those of the LoRAs it merges
    let mut base = first.clone();
    base.merge_loras.clear();
    let lora_env: Vec<(String, Vec<(String, String)>)> = loras.iter().map(|l| (
        l["id"].as_str().unwrap_or("").to_string(),
        l["env"].as_object().into_iter().flatten().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect(),
    )).collect();
    for (id, env) in &lora_env {
        if first.merge_loras.iter().all(|u| &u.name != id) {
            for (k, _) in env {
                if !m["env"].get(k.as_str()).is_some() && !args.iter().any(|a| a.starts_with(&format!("{k}="))) {
                    base.settings.remove(k);
                }
            }
        }
    }
    let g = gpu.clone();
    let loader: nextsycl_serve::image::Loader = Box::new(move |use_: &[LoraUse], log: &mut dyn FnMut(String)| {
        let mut o = base.clone();
        o.merge_loras = use_.to_vec();
        for u in use_ {
            if let Some((_, env)) = lora_env.iter().find(|(id, _)| *id == u.name) {
                o.settings.extend(env.iter().cloned());
            }
        }
        load_fn(&files, std::slice::from_ref(&g), &o, log).map_err(|e| e.0)
    });
    let wfe = args.iter().any(|a| a == "--wfe").then(|| cfg.dist.join("wfe"));
    let cors: Vec<String> = opt(args, "--cors").or(cfg.get("NS_CORS").as_deref()).map(|c| c.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    let addr = format!("{}:{}", opt(args, "--host").unwrap_or("127.0.0.1"), opt(args, "--port").unwrap_or("8086"));
    let srv = ImageServer::new(m["id"].as_str().unwrap_or("image").to_string(), gpu, loader, first.merge_loras, known, Some(out), wfe, cors)?;
    std::sync::Arc::new(srv).run(&addr)
}

/// `nextsycl image <command>`
pub fn cmd(cfg: &Config, args: &[String], selftest: impl Fn(&[String]) -> Result<(), String>) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("engines") => {
            super::list_engines(engines().iter().map(|k| (k.archs.join(", "), k.name, k.options.to_vec())));
            Ok(())
        }
        Some("selftest") => selftest(rest),
        Some("gen") => gen(cfg, rest),
        Some("edit") => edit(cfg, rest),
        Some("check") => check(cfg, rest),
        Some("serve") => serve(cfg, rest),
        _ => Err(USAGE.into()),
    }
}
