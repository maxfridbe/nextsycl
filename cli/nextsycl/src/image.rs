//! `nextsycl image ...`: the image commands. `gen --local` runs the engine in this process (inside the image; the
//! server path comes with `image serve`); `check` compares an engine's stages with a reference's dumps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_image::{ImageRequest, LoraUse, ModelFiles, Sampler, Schedule};
use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

const USAGE: &str = "nextsycl image edit \"<instructions>\" --image PICTURE [--image REFERENCE]... [gen's options]
nextsycl image gen \"<prompt>\" [--model ID] [--size WxH | --aspect W:H] [--steps N] [--seed N] [--n N] [--sampler S]
                   [--schedule S] [--lora NAME[:SCALE]]... [--request-lora NAME[:SCALE]]... [--set NAME=VALUE]... [--out FILE|DIR] [--rgba] [--local] [--gpu N]
                   (--lora: merged into the weights at load; --request-lora: beside them for this request, engines that take it)
nextsycl image serve [MODEL] [--port 8086] [--host 127.0.0.1] [--gpu N] [--lora NAME[:SCALE]]... [--set NAME=VALUE]...
                     [--wfe] [--out DIR] [--cors ORIGIN]
nextsycl image start [serve's options] [--slot N]   (in the background; slot N: another server beside the first) | ps | logs [-f] | stop  [--slot N]
nextsycl image ocr PICTURE|PDF... [--one-pass] [--no-columns] [--model ID|FILE.gguf] [--prompt TEXT] [--no-tiles] [--no-figures] [--max N] [--temp T] [--out FILE] [--gpu N] [--dump-views DIR]
nextsycl image check <dump dir> [--model ID|FILE.gguf] [--stages te,dit,steps,vae] [--lora NAME[:SCALE]]... [--tokens N] [--gpu N]
nextsycl image upscale PICTURE [--model real-esrgan|waifu2x | --dir NCNN_DIR] [--variant V] [--scale N] [--noise N] [--tile N] [--out FILE] [--gpu N]
nextsycl image upscale-check PICTURE REFERENCE.png [upscale's options]   (against realesrgan- / waifu2x-ncnn-vulkan's output: PSNR, worst pixel)
nextsycl image vectorize PICTURE [--model starvector-1b-im2svg | --dir CHECKPOINT_DIR] [--max-length N] [--beams N] [--greedy] [--top-p P] [--seed S] [--out FILE.svg] [--gpu N]
nextsycl image vectorize PICTURE --check REFERENCE_DIR [--stages pixels,encoder,prefill,logits,greedy] [--dir ...]   (against reference/starvector/ref.py's dumps)
nextsycl image engines | selftest [--gpu N]";

/// The image engines this program has (image/<arch>)
pub fn engines() -> Vec<nextsycl_image::ImageKind> {
    vec![nextsycl_image_qwenimage21::kind(), nextsycl_image_unlimitedocr::kind(), nextsycl_image_paddleocrvl::kind(), nextsycl_image_olmocr2::kind(), nextsycl_image_hunyuanimage21::kind(),
         nextsycl_image_upscale::kind(), nextsycl_image_starvector::kind(), nextsycl_image_mingimage::kind(), nextsycl_image_mossimage::kind(), nextsycl_image_example::kind()]
}

/// `--gpu 0` or `--gpu 0,1` (an engine that splits over two cards takes the second too)
fn gpus(args: &[String]) -> Result<Vec<std::sync::Arc<nextsycl_core::Gpu>>, String> {
    let list = opt(args, "--gpu").unwrap_or("0");
    let mut v = Vec::new();
    for i in list.split(',').filter(|x| !x.trim().is_empty()) {
        let i: usize = i.trim().parse().map_err(|_| format!("--gpu {list}: card numbers, comma separated"))?;
        v.push(nextsycl_core::Gpu::open(i).map_err(|e| e.0)?);
    }
    if v.is_empty() {
        v.push(nextsycl_core::Gpu::open(0).map_err(|e| e.0)?);
    }
    Ok(v)
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// The registered image model `id`, or the first one: (entry, files by role). A GGUF file named instead of an id is
/// read as it is: its architecture, and an OCR decoder's projector (mmproj-*.gguf) found beside it
fn model(cfg: &Config, id: Option<&str>) -> Result<(Value, ModelFiles), String> {
    if let Some(p) = id.map(Path::new).filter(|p| p.is_file()) {
        let g = nextsycl_gguf::Gguf::open(p).map_err(|e| e.0)?;
        let mut files = ModelFiles::new();
        files.insert("model".into(), p.to_path_buf());
        if let Some(mm) = std::fs::read_dir(p.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?.flatten().map(|e| e.path())
            .find(|q| q != p && q.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("mmproj") && n.ends_with(".gguf"))) {
            files.insert("mmproj".into(), mm);
        }
        // a tokenizer.json beside it (PaddleOCR-VL reads the original one)
        let tj = p.with_file_name("tokenizer.json");
        if tj.is_file() {
            files.insert("tokenizer".into(), tj);
        }
        let m = serde_json::json!({"id": p.file_stem().and_then(|s| s.to_str()).unwrap_or("model"), "arch": g.architecture(), "kind": "image"});
        return Ok((m, files));
    }
    let all = models::all(cfg)?;
    let m = match id {
        Some(id) => all.into_iter().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id} (nextsycl models search --kind image)"))?,
        None => all.into_iter().find(|m| models::kind_of(m) == "image").ok_or("no image model (nextsycl models pull qwen-image-2.1-q8)")?,
    };
    if models::kind_of(&m) != "image" {
        return Err(format!("{} is a {} model, not an image one", m["id"].as_str().unwrap_or("?"), models::kind_of(&m)));
    }
    // not while its files move to another drive (nextsycl models move)
    crate::drives::not_moving(cfg, m["id"].as_str().unwrap_or(""))?;
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
    loras_by(cfg, args, "--lora")
}

/// Every `FLAG NAME[:SCALE]` (--lora: merged at load; --request-lora: the request's own, beside the weights)
fn loras_by(cfg: &Config, args: &[String], flag: &str) -> Result<Vec<LoraUse>, String> {
    let all = models::all(cfg)?;
    let find = |n: &str| -> Option<PathBuf> {
        all.iter()
            .find(|m| m["id"] == n && models::kind_of(m) == "lora")
            .and_then(|m| m["file"].as_str().map(PathBuf::from))
            .or_else(|| Path::new(n).is_file().then(|| PathBuf::from(n)))
    };
    args.iter().enumerate().filter(|(_, a)| *a == flag).map(|(i, _)| {
        let v = args.get(i + 1).ok_or_else(|| format!("{flag} NAME[:SCALE]"))?;
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

/// `nextsycl image ocr PICTURE...`: the pages read here (an OCR model), the text streamed to stdout (and --out)
fn ocr(cfg: &Config, args: &[String]) -> Result<(), String> {
    let mut pages: Vec<nextsycl_image::Picture> = Vec::new();
    for p in args.iter().take_while(|a| !a.starts_with("--")) {
        let b = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
        if nextsycl_image::Picture::is_pdf(&b) {
            // a PDF: its pages at 150 dpi
            pages.extend(nextsycl_image::Picture::from_pdf(&b, 150.0, 64).map_err(|e| format!("{p}: {e}"))?.0);
        } else {
            pages.push(nextsycl_image::Picture::decode(&b).map_err(|e| format!("{p}: {e}"))?);
        }
    }
    if pages.is_empty() {
        return Err(USAGE.into());
    }
    // the views as the model sees them, written as float32 planes (a check of the preprocessing) - no model needed
    if let Some(dir) = opt(args, "--dump-views") {
        let seen = nextsycl_image_unlimitedocr::prep::see(&pages, !args.iter().any(|a| a == "--no-tiles"));
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let put = |name: String, v: &[f32]| std::fs::write(Path::new(dir).join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        for (i, t) in seen.tiles.iter().enumerate() {
            put(format!("tile-{i}.f32"), &t.planes).map_err(|e| e.to_string())?;
        }
        for (i, g) in seen.globals.iter().enumerate() {
            put(format!("global-{i}.f32"), &g.planes).map_err(|e| e.to_string())?;
        }
        println!("grid {}x{}: {} tiles, {} global views", seen.grid.0, seen.grid.1, seen.tiles.len(), seen.globals.len());
        return Ok(());
    }
    let (_, e) = load(cfg, args)?;
    if !e.reads() {
        return Err(format!("{} makes pictures; an OCR model reads them (--model)", e.arch()));
    }
    let req = nextsycl_image::OcrRequest {
        pages,
        prompt: opt(args, "--prompt").unwrap_or("").to_string(),
        tiles: Some(!args.iter().any(|a| a == "--no-tiles")),
        figures: Some(!args.iter().any(|a| a == "--no-figures")),
        each_page: Some(!args.iter().any(|a| a == "--one-pass")),
        columns: Some(!args.iter().any(|a| a == "--no-columns")),
        max_tokens: opt(args, "--max").and_then(|v| v.parse().ok()).unwrap_or(0),
        temperature: opt(args, "--temp").and_then(|v| v.parse().ok()).unwrap_or(0.0),
        ngram: opt(args, "--ngram").and_then(|v| v.parse().ok()),
        ..Default::default()
    };
    use std::io::Write;
    let mut streamed = String::new();
    let out = e.ocr(&req, &mut |t| {
        streamed.push_str(t);
        print!("{t}");
        let _ = std::io::stdout().flush();
    }).map_err(|e| e.0)?;
    println!();
    // the pictures inside the page are read after it: the whole reading again, with them in
    if out.text != streamed {
        println!("--- with the text inside its pictures ---\n{}", out.text);
    }
    eprintln!("[{} prompt tokens in {:.2} s ({:.0} tok/s); {} written in {:.2} s ({:.1} tok/s)]", out.prompt_tokens, out.prompt_seconds,
              out.prompt_tokens as f64 / out.prompt_seconds.max(1e-9), out.tokens, out.seconds, out.tokens as f64 / out.seconds.max(1e-9));
    if let Some(f) = opt(args, "--out") {
        std::fs::write(f, &out.text).map_err(|e| format!("{f}: {e}"))?;
    }
    Ok(())
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
        edit: editing.then(|| nextsycl_image::Edit { image: pics[0].clone(), refs: pics[1..].to_vec(), mask: None, strength: 1.0, condition: true }),
        loras: loras_by(cfg, args, "--request-lora")?,
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
        Ok(())
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

/// An ncnn bundle's files by role, as the upscalers' cards name them: a Real-ESRGAN `models/` directory
/// (`realesrgan-x4plus.param` ...) or waifu2x's directory of `models-*/` (`models-cunet/noise1_scale2.0x_model.param` ...)
fn ncnn_files(dir: &Path) -> Result<ModelFiles, String> {
    let mut files = ModelFiles::new();
    let mut add = |sub: Option<&str>, d: &Path| -> Result<(), String> {
        for e in std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))?.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".param") || name.ends_with(".bin") {
                files.insert(sub.map(|s| format!("{s}/{name}")).unwrap_or(name), p);
            }
        }
        Ok(())
    };
    add(None, dir)?;
    for e in std::fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if e.path().is_dir() && (n.starts_with("models-") || n == "models") {
            if n == "models" {
                add(None, &e.path())?;
            } else {
                add(Some(&n), &e.path())?;
            }
        }
    }
    if files.is_empty() {
        return Err(format!("{}: no .param / .bin files (a realesrgan models/ directory, or waifu2x's with models-*/)", dir.display()));
    }
    Ok(files)
}

/// `image upscale` and `image upscale-check`: the engine in this process
fn upscale_cmd(cfg: &Config, args: &[String], check: bool) -> Result<(), String> {
    let pos: Vec<&String> = {
        let mut v = Vec::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a.starts_with("--") {
                it.next();
            } else {
                v.push(a);
            }
        }
        v
    };
    let input = pos.first().ok_or(USAGE)?;
    nextsycl_core::use_kind("image");
    let files = match opt(args, "--dir") {
        Some(d) => ncnn_files(Path::new(d))?,
        None => model(cfg, Some(opt(args, "--model").unwrap_or("real-esrgan")))?.1,
    };
    let gs = gpus(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let t0 = std::time::Instant::now();
    let e = nextsycl_image_upscale::Upscaler::load(&files, &gs, &mut log).map_err(|e| e.0)?;
    eprintln!("loaded in {:.2} s", t0.elapsed().as_secs_f64());
    let pic = nextsycl_image::Picture::decode(&std::fs::read(input).map_err(|e| format!("{input}: {e}"))?)?;
    let req = nextsycl_image::UpscaleRequest {
        picture: pic,
        variant: opt(args, "--variant").map(str::to_string),
        scale: opt(args, "--scale").and_then(|v| v.parse().ok()),
        noise: opt(args, "--noise").and_then(|v| v.parse().ok()),
        tile: opt(args, "--tile").and_then(|v| v.parse().ok()),
    };
    use nextsycl_image::ImageEngine;
    let runs: usize = opt(args, "--runs").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let mut out = None;
    for r in 0..runs {
        let (p, info) = e.upscale(&req, &mut |_| Ok(())).map_err(|e| e.0)?;
        // (memory: nvidia-smi's per-process figure; the card's free memory moves with every other program on it)
        eprintln!("run {}: {} {}x{} -> {}x{} ({}x{}, {} tiles of {}) in {:.3} s on {}",
                  r + 1, info.variant, req.picture.width, req.picture.height, p.width, p.height, info.scale,
                  info.noise.map(|n| format!(" denoise {n}")).unwrap_or_default(), info.tiles, info.tile, info.seconds, gs[0].name);
        out = Some(p);
    }
    let p = out.unwrap();
    if let Some(o) = opt(args, "--out") {
        p.write_png(Path::new(o), &[]).map_err(|e| format!("{o}: {e}"))?;
        eprintln!("saved {o}");
    }
    if check {
        let r = pos.get(1).ok_or("upscale-check PICTURE REFERENCE.png")?;
        let want = nextsycl_image::Picture::decode(&std::fs::read(r).map_err(|e| format!("{r}: {e}"))?)?;
        if (want.width, want.height) != (p.width, p.height) {
            return Err(format!("the reference is {}x{}, ours {}x{}", want.width, want.height, p.width, p.height));
        }
        let c = p.channels.min(want.channels) as usize;
        let (pc, wc) = (p.channels as usize, want.channels as usize);
        let (mut se, mut worst, mut same, mut near) = (0f64, 0i32, 0usize, 0usize);
        let n = (p.width * p.height) as usize;
        for i in 0..n {
            for ch in 0..c {
                let d = p.data[i * pc + ch] as i32 - want.data[i * wc + ch] as i32;
                se += (d * d) as f64;
                worst = worst.max(d.abs());
                same += (d == 0) as usize;
                near += (d.abs() <= 2) as usize;
            }
        }
        let mse = se / (n * c) as f64;
        let psnr = if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
        println!("against {r}: PSNR {psnr:.2} dB, worst {worst}/255, {:.2}% values equal, {:.2}% within 2", 100.0 * same as f64 / (n * c) as f64,
                 100.0 * near as f64 / (n * c) as f64);
    }
    Ok(())
}

/// `image vectorize`: StarVector in this process (`--check DIR`: its stages against the reference's dumps)
fn vectorize_cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--greedy" {
            continue;
        }
        if a.starts_with("--") {
            it.next();
        } else {
            pos.push(a.clone());
        }
    }
    let input = pos.first().ok_or(USAGE)?;
    nextsycl_core::use_kind("image");
    let files = match opt(args, "--dir") {
        Some(d) => nextsycl_image_starvector::files_in(Path::new(d)).map_err(|e| e.0)?,
        None => model(cfg, Some(opt(args, "--model").unwrap_or("starvector-1b-im2svg")))?.1,
    };
    let gs = gpus(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = nextsycl_image_starvector::StarVector::load(&files, &gs[0], &mut log).map_err(|e| e.0)?;
    if let Some(d) = opt(args, "--check") {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("pixels,encoder,prefill,logits,greedy").split(',').collect();
        let worst = nextsycl_image_starvector::check::run(&e, Path::new(d), Path::new(input), &stages, &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        return Ok(());
    }
    let pic = nextsycl_image::Picture::decode(&std::fs::read(input).map_err(|e| format!("{input}: {e}"))?)?;
    let greedy = args.iter().any(|a| a == "--greedy");
    let req = nextsycl_image::VectorizeRequest {
        picture: pic,
        max_length: opt(args, "--max-length").and_then(|v| v.parse().ok()).unwrap_or(0),
        beams: if greedy { Some(1) } else { opt(args, "--beams").and_then(|v| v.parse().ok()) },
        sample: if greedy { Some(false) } else { None },
        top_p: opt(args, "--top-p").and_then(|v| v.parse().ok()),
        seed: opt(args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(0),
        ..Default::default()
    };
    use nextsycl_image::ImageEngine;
    let out = e.vectorize(&req, &mut |_| Ok(())).map_err(|e| e.0)?;
    eprintln!("{} tokens in {:.2} s after {:.2} s for the picture and prompt ({:.1} tokens/s){} on {}", out.tokens, out.seconds, out.prompt_seconds,
              out.tokens as f64 / out.seconds.max(1e-9), if out.closed { ", ended by </svg>" } else { ", stopped at the length limit" }, gs[0].name);
    match opt(args, "--out") {
        Some(o) => {
            std::fs::write(o, &out.svg).map_err(|e| format!("{o}: {e}"))?;
            eprintln!("saved {o}");
        }
        None => println!("{}", out.svg),
    }
    Ok(())
}

fn check(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = Path::new(args.first().ok_or(USAGE)?);
    nextsycl_core::use_kind("image");
    let (m, files) = model(cfg, opt(args, "--model"))?;
    if nextsycl_image_unlimitedocr::ARCHS.contains(&m["arch"].as_str().unwrap_or("")) {
        let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
        let gpu = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
        let mut log = |s: String| eprintln!("{s}");
        let e = nextsycl_image_unlimitedocr::UnlimitedOcr::load(&files, &gpu, &mut log).map_err(|e| e.0)?;
        let n: usize = opt(args, "--tokens").and_then(|v| v.parse().ok()).unwrap_or(64);
        let worst = nextsycl_image_unlimitedocr::check::run(&e, dir, n, &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        return Ok(());
    }
    if nextsycl_image_paddleocrvl::ARCHS.contains(&m["arch"].as_str().unwrap_or("")) {
        let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
        let gpu = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
        let mut log = |s: String| eprintln!("{s}");
        let e = nextsycl_image_paddleocrvl::PaddleOcrVl::load(&files, &gpu, &mut log).map_err(|e| e.0)?;
        let n: usize = opt(args, "--tokens").and_then(|v| v.parse().ok()).unwrap_or(64);
        let pic = opt(args, "--picture").map(|p| nextsycl_image::Picture::read_png(Path::new(p))).transpose()?;
        let worst = nextsycl_image_paddleocrvl::check::run(&e, dir, pic.as_ref(), n, &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        return Ok(());
    }
    if nextsycl_image_olmocr2::ARCHS.contains(&m["arch"].as_str().unwrap_or("")) {
        let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
        let gpu = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
        let mut log = |s: String| eprintln!("{s}");
        let e = nextsycl_image_olmocr2::OlmOcr::load(&files, &gpu, &mut log).map_err(|e| e.0)?;
        let n: usize = opt(args, "--tokens").and_then(|v| v.parse().ok()).unwrap_or(32);
        let pic = opt(args, "--picture").map(|p| nextsycl_image::Picture::read_png(Path::new(p))).transpose()?;
        let worst = nextsycl_image_olmocr2::check::run(&e, dir, pic.as_ref(), n, &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        return Ok(());
    }
    if m["arch"] == nextsycl_image_hunyuanimage21::ARCH {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("te,byt5,dit,vae").split(',').collect(); // + steps
        let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
        let gpu = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
        let mut log = |s: String| eprintln!("{s}");
        let e = nextsycl_image_hunyuanimage21::HunyuanImage21::load(&files, &gpu, &options(cfg, &m, args)?, &mut log).map_err(|e| e.0)?;
        let worst = nextsycl_image_hunyuanimage21::check::run(&e, dir, &stages, &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        return Ok(());
    }
    if m["arch"].as_str() == Some(nextsycl_image_mingimage::ARCH) {
        let gs = gpus(args)?;
        let mut log = |s: String| eprintln!("{s}");
        let e = nextsycl_image_mingimage::MingImage::load(&files, &gs, &mut log).map_err(|e| e.0)?;
        let st = opt(args, "--stages").unwrap_or("tokens,te,dit,steps,vae");
        let worst = nextsycl_image_mingimage::check::run(&e, dir, &st.split(',').collect::<Vec<_>>(), &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        return Ok(());
    }
    if m["arch"].as_str() == Some(nextsycl_image_mossimage::ARCH) {
        let gs = gpus(args)?;
        let mut log = |s: String| eprintln!("{s}");
        let e = nextsycl_image_mossimage::MossImage::load(&files, &gs, &Default::default(), &mut log).map_err(|e| e.0)?;
        let st = opt(args, "--stages").unwrap_or("sigmas,dit,steps");
        let worst = nextsycl_image_mossimage::check::run(&e, dir, &st.split(',').collect::<Vec<_>>(), &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        return Ok(());
    }
    if m["arch"] != nextsycl_image_qwenimage21::ARCH {
        return Err(format!("check knows {}, {}, {}, {} and the OCR engines", nextsycl_image_qwenimage21::ARCH, nextsycl_image_hunyuanimage21::ARCH, nextsycl_image_mingimage::ARCH,
                           nextsycl_image_mossimage::ARCH));
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
/// The image server's container (`image serve` in the foreground, `image start` in the background)
const SERVED: crate::served::Served = crate::served::Served { name: "nextsycl-image", kind: "image", port: "8086" };

fn serve(cfg: &Config, args: &[String]) -> Result<(), String> {
    run(cfg, args, false)
}

fn run(cfg: &Config, args: &[String], detach: bool) -> Result<(), String> {
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
    if args.iter().any(|a| a == "--wfe") && !nextsycl_serve::wfe::exists(&cfg.dist.join("wfe"), "image/index.html") {
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
    for (k, v) in cfg.with_prefix("NS_QI_").into_iter().chain(cfg.with_prefix("NS_HI_")).chain(cfg.with_prefix("NS_MOSS_")).chain(cfg.with_prefix("NSD_")) {
        a.extend(["-e".into(), format!("{k}={v}")]);
    }
    a.extend([ce.image.clone(), "bash".into(), "-c".into(),
              "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl image serve \"$@\"".into(), "nextsycl".into()]);
    a.push(model_id);
    let mut it = args.iter().skip(if id.is_some() && !args.first().is_some_and(|x| x.starts_with("--")) { 1 } else { 0 });
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
        env: l["env"].as_object().into_iter().flatten().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect(),
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
    let mut srv = ImageServer::new(m["id"].as_str().unwrap_or("image").to_string(), gpu, loader, first.merge_loras, known, Some(out), wfe, cors)?;
    srv.idle = crate::served::idle_exit(args)?;
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
        Some("upscale") => upscale_cmd(cfg, rest, false),
        Some("upscale-check") => upscale_cmd(cfg, rest, true),
        Some("vectorize") => vectorize_cmd(cfg, rest),
        Some("ocr") => ocr(cfg, rest),
        Some("serve") => serve(cfg, rest),
        Some("start") => SERVED.start(cfg, rest, |d| run(cfg, rest, d)),
        Some("ps") => SERVED.ps(cfg, rest),
        Some("stop") => SERVED.stop(cfg, rest),
        Some("logs") => SERVED.logs(cfg, rest),
        _ => Err(USAGE.into()),
    }
}
