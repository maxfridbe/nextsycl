//! `nextsycl model3d ...`: the 3D commands. `gen` and `rig` run an engine in this process (inside the image); `serve`
//! runs the 3D server in a container; `tool` runs the CPU mesh tools (no GPU, anywhere).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_model3d::{MeshRequest, ModelFiles, RigRequest, View};
use nextsycl_models::{config::Config, registry as models};
use serde_json::Value;

const USAGE: &str = "nextsycl model3d gen PICTURE [PICTURE[:SIDE]]... [--model ID] [--texture | --no-texture] [--seed N] [--steps N] [--cfg X]
                   [--resolution N] [--faces N] [--texture-size N] [--mesh FILE.glb|.obj (paint it like the picture; a model that paints)] [--keep-background] [--set NAME=VALUE]... [--opt-NAME VALUE]...
                   [--out FILE.glb|DIR] [--gpu N]                 a mesh from a picture (or several views: front, back, left, right)
nextsycl model3d rig MESH.glb|.obj [--model ID] [--prompt \"the motion\"] [--frames N] [--fps X] [--seed N] [--out FILE.glb|DIR] [--gpu N]
                   a skeleton and skin weights (UniRig, RigNet), or motion from words (AnimateAnyMesh)
nextsycl model3d serve [MODEL] [--port 8088] [--host 127.0.0.1] [--gpu N] [--set NAME=VALUE]... [--wfe] [--out DIR] [--cors ORIGIN]
nextsycl model3d start [serve's options] [--slot N] | ps | logs [-f] | stop [--slot N]
nextsycl model3d tool decimate IN OUT --faces N          quadric decimation (positions and colours kept)
nextsycl model3d tool remesh IN OUT [--edge X] [--passes N]   isotropic remeshing (even triangles)
nextsycl model3d tool uv IN OUT [--size N]               a fresh atlas, the old texture or colours baked across
nextsycl model3d tool convert IN OUT                     GLB <-> OBJ (by the extensions)
nextsycl model3d check <dump dir> [--model ID] [--stages pre,cond,dit,steps,vae,grid] [--picture P] [--gpu N]
                   an engine's stages against a reference's dumps (reference/hunyuan3d); a paint dump's stages:
                   maps,pic,dino,enc,dual,unet,unipc,steps,decode
nextsycl model3d check <dump dir> --model ID [--stages a,b,...] [--mesh FILE.obj] [--prompt TEXT] [--gpu N]
                   an engine's stages against its reference's dumps (reference/<arch>/)
nextsycl model3d check DUMP_DIR [--model ID] [--stages pre,remover,dino,ss,ssdec,shape,cascade,tex,sdec,tdec] [--full] [--picture FILE] [--gpu N]
                   an engine's stages against reference/trellis2/ref.py's dumps
nextsycl model3d engines | selftest [--gpu N]";

/// The 3D engines this program has (model3d/<arch>)
pub fn engines() -> Vec<nextsycl_model3d::Model3dKind> {
    vec![nextsycl_model3d_trellis2::kind(), nextsycl_model3d_trellis2::kind_pixal(), nextsycl_model3d_hunyuan3d::kind(), nextsycl_model3d_unirig::kind(),
         nextsycl_model3d_rignet::kind(), nextsycl_model3d_animateanymesh::kind(), nextsycl_model3d_example::kind()]
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn num<T: std::str::FromStr>(args: &[String], k: &str) -> Result<Option<T>, String> {
    opt(args, k).map(|s| s.parse().map_err(|_| format!("{k}: {s}?"))).transpose()
}

/// The arguments that are not options or option values (the files)
fn positional(args: &[String]) -> Vec<String> {
    const FLAGS: &[&str] = &["--texture", "--no-texture", "--keep-background", "--here", "--wfe", "-f", "--full"];
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a.starts_with("--") || a == "-f" {
            if !FLAGS.contains(&a.as_str()) {
                it.next();
            }
            continue;
        }
        out.push(a.clone());
    }
    out
}

/// The registered 3D model `id` (or the first one that does `want`: "generate" or "rig"): (entry, files by role)
fn model(cfg: &Config, args: &[String], want: &str) -> Result<(Value, ModelFiles), String> {
    let all = models::all(cfg)?;
    let m = match opt(args, "--model") {
        Some(id) => all.into_iter().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id} (nextsycl models search --kind model3d)"))?,
        None => all.into_iter().find(|m| models::kind_of(m) == "model3d" && (want.is_empty() || m["does"].as_array().is_none_or(|d| d.iter().any(|x| x == want))))
            .ok_or_else(|| format!("no 3D model that does {want} (nextsycl models search --kind model3d)"))?,
    };
    if models::kind_of(&m) != "model3d" {
        return Err(format!("{} is a {} model, not a 3D one", m["id"].as_str().unwrap_or("?"), models::kind_of(&m)));
    }
    // not while its files move to another drive (nextsycl models move)
    crate::drives::not_moving(cfg, m["id"].as_str().unwrap_or(""))?;
    let files: ModelFiles = m["files"].as_object().into_iter().flatten().filter_map(|(r, p)| Some((r.clone(), PathBuf::from(p.as_str()?)))).collect();
    Ok((m, files))
}

/// The load options: the registry entry's settings, `--set NAME=VALUE`s, the --opt-NAMEs taken at load
fn options(m: &Value, args: &[String]) -> Result<nextsycl_model3d::LoadOptions, String> {
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
    let k = nextsycl_model3d::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    // a command here loads and asks at once: every option it takes is checked (a per-request one such as TRELLIS's
    // --opt-fov goes in the request's `extra`), those set at load also set their variables
    for (var, v) in super::opt_env(k.options, nextsycl_core::At::Both, k.name)? {
        settings.insert(var, v);
    }
    Ok(nextsycl_model3d::LoadOptions { settings })
}

fn gpu(args: &[String]) -> Result<Arc<nextsycl_core::Gpu>, String> {
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    nextsycl_core::Gpu::open(i).map_err(|e| e.0)
}

fn seed(args: &[String]) -> Result<u64, String> {
    Ok(num(args, "--seed")?.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_nanos() as u64 % 100_000).unwrap_or(0)
    }))
}

/// Where a result goes: --out FILE.glb, or a name in --out DIR (else the 3D output directory)
fn out_file(cfg: &Config, args: &[String], stem: &str) -> Result<PathBuf, String> {
    match opt(args, "--out") {
        Some(o) if o.ends_with(".glb") => Ok(PathBuf::from(o)),
        o => {
            let d = o.map(PathBuf::from).unwrap_or_else(|| out_dir(cfg, &[]));
            std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
            Ok(d.join(format!("{stem}.glb")))
        }
    }
}

fn progress(st: nextsycl_model3d::Step) -> nextsycl_core::Result<()> {
    eprintln!("[{:>6.1} s] {} {}/{}", st.seconds, st.phase, st.at, st.of);
    Ok(())
}

fn load(cfg: &Config, args: &[String], want: &str) -> Result<(Value, Box<dyn nextsycl_model3d::Model3dEngine>), String> {
    nextsycl_core::use_kind("model3d");
    let (m, files) = model(cfg, args, want)?;
    let ks = engines();
    let k = nextsycl_model3d::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = (k.load)(&files, &[g], &options(&m, args)?, &mut log).map_err(|e| e.0)?;
    Ok((m, e))
}

fn gen(cfg: &Config, args: &[String]) -> Result<(), String> {
    let pics = positional(args);
    if pics.is_empty() {
        return Err(USAGE.into());
    }
    let mut views = Vec::new();
    for p in &pics {
        // PICTURE:SIDE (a path may hold colons: only a known side after the last one counts)
        let (path, side) = match p.rsplit_once(':') {
            Some((a, s)) if ["front", "back", "left", "right", "top", "bottom"].contains(&s) || s.parse::<f32>().is_ok() => (a, Some(s.to_string())),
            _ => (p.as_str(), None),
        };
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        views.push(View { image: nextsycl_model3d::read_picture(&bytes).map_err(|e| format!("{path}: {e}"))?, side });
    }
    // --mesh FILE.glb|.obj: paint that mesh (an engine that paints) instead of making a shape
    let given = match opt(args, "--mesh") {
        Some(f) => Some(nextsycl_model3d::read_mesh(&std::fs::read(f).map_err(|e| format!("{f}: {e}"))?).map_err(|e| format!("{f}: {e}"))?),
        None => None,
    };
    let (m, e) = load(cfg, args, if given.is_some() { "texture" } else { "generate" })?;
    let a = e.abilities();
    let texture = if args.iter().any(|x| x == "--no-texture") { false } else { args.iter().any(|x| x == "--texture") || a.texture };
    let seed = seed(args)?;
    let req = MeshRequest { views, mesh: given, texture, seed, steps: num(args, "--steps")?, cfg: num(args, "--cfg")?, resolution: num(args, "--resolution")?,
                            faces: num(args, "--faces")?, texture_size: num(args, "--texture-size")?,
                            remove_background: Some(!args.iter().any(|x| x == "--keep-background")),
                            extra: nextsycl_core::options::by_name(super::engine_opts(), e.options()) };
    let t0 = std::time::Instant::now();
    let mesh = e.generate(&req, &mut progress).map_err(|e| e.0)?;
    for l in e.report() {
        eprintln!("{l}");
    }
    let out = out_file(cfg, args, &format!("{}-{seed}", m["id"].as_str().unwrap_or("mesh")))?;
    std::fs::write(&out, nextsycl_model3d::mesh::glb::write(&mesh)?).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("{} ({} triangles{}, {:.1} s, seed {seed})", out.display(), mesh.triangles(),
             if mesh.material.as_ref().is_some_and(|m| m.base_color.is_some()) { ", textured" } else { "" }, t0.elapsed().as_secs_f64());
    Ok(())
}

fn rig(cfg: &Config, args: &[String]) -> Result<(), String> {
    let path = positional(args).into_iter().next().ok_or(USAGE)?;
    let mesh = nextsycl_model3d::read_mesh(&std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))?;
    let (m, e) = load(cfg, args, "rig")?;
    let seed = seed(args)?;
    let req = RigRequest { mesh, seed, prompt: opt(args, "--prompt").map(str::to_string), frames: num(args, "--frames")?, fps: num(args, "--fps")?,
                           steps: num(args, "--steps")?, cfg: num(args, "--cfg")?, extra: nextsycl_core::options::by_name(super::engine_opts(), e.options()) };
    let t0 = std::time::Instant::now();
    let out_mesh = e.rig(&req, &mut progress).map_err(|e| e.0)?;
    let stem = Path::new(&path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "mesh".into());
    let out = out_file(cfg, args, &format!("{stem}-{}", m["id"].as_str().unwrap_or("rig")))?;
    std::fs::write(&out, nextsycl_model3d::mesh::glb::write(&out_mesh)?).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("{} ({} joints, {} frames, {:.1} s)", out.display(), out_mesh.skin.as_ref().map_or(0, |s| s.joints.len()),
             out_mesh.animation.as_ref().map_or(0, |a| a.frames.len()), t0.elapsed().as_secs_f64());
    Ok(())
}

/// `nextsycl model3d check DIR` for Hunyuan3D: its stages against reference/hunyuan3d's dumps
fn check_hunyuan3d(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = positional(args).into_iter().next().ok_or(USAGE)?;
    nextsycl_core::use_kind("model3d");
    let (m, files) = model(cfg, args, "generate")?;
    let arch = m["arch"].as_str().unwrap_or("");
    if ![nextsycl_model3d_hunyuan3d::ARCH_21, nextsycl_model3d_hunyuan3d::ARCH_MINI].contains(&arch) {
        return Err(format!("check knows Hunyuan3D only, not {arch}"));
    }
    let model_file = files.get("model").ok_or("the entry has no model file")?;
    let st = nextsycl_gguf::safetensors::SafeTensors::open(model_file).map_err(|e| e.0)?;
    let a = if arch == nextsycl_model3d_hunyuan3d::ARCH_MINI { nextsycl_model3d_hunyuan3d::MINI } else { nextsycl_model3d_hunyuan3d::V21 };
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = nextsycl_model3d_hunyuan3d::Hunyuan3d::load(a, &files, &st, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
    let stages: Vec<String> = opt(args, "--stages").unwrap_or("all").split(',').map(str::to_string).collect();
    let pic = opt(args, "--picture").map(|p| std::fs::read(p).map_err(|e| format!("{p}: {e}")).and_then(|b| nextsycl_model3d::read_picture(&b))).transpose()?;
    // the paint model's dumps (reference/hunyuan3d/ref_paint.py) hold the views' maps; the shape's do not
    let d = Path::new(&dir);
    let worst = if d.join("normal.npy").exists() {
        let stages = if stages.iter().any(|s| s == "all") { vec!["paint".to_string()] } else { stages };
        nextsycl_model3d_hunyuan3d::check::run_paint(&e, d, &stages, pic.as_ref(), &mut log)
    } else {
        nextsycl_model3d_hunyuan3d::check::run(&e, d, &stages, pic.as_ref(), &mut log)
    }
    .map_err(|e| e.0)?;
    println!("worst relative error {worst:.2e}");
    Ok(())
}

fn read_any(p: &str) -> Result<nextsycl_model3d::Mesh, String> {
    nextsycl_model3d::read_mesh(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))
}

fn write_any(p: &str, m: &nextsycl_model3d::Mesh) -> Result<(), String> {
    let bytes = if p.ends_with(".obj") { nextsycl_model3d::mesh::obj::write(m).into_bytes() } else { nextsycl_model3d::mesh::glb::write(m)? };
    std::fs::write(p, bytes).map_err(|e| format!("{p}: {e}"))
}

/// `nextsycl model3d tool ...`: the CPU mesh tools
fn tool(args: &[String]) -> Result<(), String> {
    use nextsycl_model3d::mesh::tools;
    let files = positional(&args[1.min(args.len())..]);
    let (Some(input), Some(output)) = (files.first(), files.get(1)) else { return Err(USAGE.into()) };
    let m = read_any(input)?;
    let before = m.triangles();
    let t0 = std::time::Instant::now();
    let out = match args.first().map(String::as_str) {
        Some("decimate") => {
            let faces: usize = num(args, "--faces")?.ok_or("--faces N")?;
            let mat = m.material.clone();
            let mut d = tools::decimate(&m, faces);
            d.material = mat;
            d.compute_normals();
            d
        }
        Some("remesh") => {
            let mut r = tools::remesh(&m, num(args, "--edge")?, num(args, "--passes")?.unwrap_or(5));
            r.compute_normals();
            r
        }
        Some("uv") => {
            let size: u32 = num(args, "--size")?.unwrap_or(2048);
            let mut u = tools::unwrap(&m, size, 4);
            if m.material.as_ref().is_some_and(|x| x.base_color.is_some()) || !m.colors.is_empty() {
                u.material = Some(tools::bake(&u, &m, size)?);
            }
            if u.normals.is_empty() {
                u.compute_normals();
            }
            u
        }
        Some("convert") => m,
        _ => return Err(USAGE.into()),
    };
    write_any(output, &out)?;
    println!("{output}: {before} -> {} triangles, {} vertices ({:.2} s)", out.triangles(), out.positions.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// `nextsycl model3d check DIR --model ID`: an engine's stages against the reference's dumps in DIR
fn check(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = PathBuf::from(positional(args).into_iter().next().ok_or(USAGE)?);
    nextsycl_core::use_kind("model3d");
    let (m, files) = model(cfg, args, "")?;
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let stages = |all: &'static str| -> Vec<String> { opt(args, "--stages").unwrap_or(all).split(',').map(str::to_string).collect() };
    let worst = match m["arch"].as_str().unwrap_or("") {
        nextsycl_model3d_hunyuan3d::ARCH_21 | nextsycl_model3d_hunyuan3d::ARCH_MINI => return check_hunyuan3d(cfg, args),
        nextsycl_model3d_trellis2::ARCH | nextsycl_model3d_trellis2::ARCH_PIXAL => return check_trellis2(cfg, args),
        nextsycl_model3d_animateanymesh::ARCH => {
            let e = nextsycl_model3d_animateanymesh::AnimateAnyMesh::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
            let st = stages("embed,fps,enc,clip,dit,dec");
            let st: Vec<&str> = st.iter().map(String::as_str).collect();
            nextsycl_model3d_animateanymesh::check::run(&e, &dir, &st, opt(args, "--mesh").map(Path::new), opt(args, "--prompt"), &mut log)
                .map_err(|e| e.0)?
        }
        nextsycl_model3d_unirig::ARCH => {
            let e = nextsycl_model3d_unirig::UniRig::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
            let st = stages("encoder,logits,skin");
            let st: Vec<&str> = st.iter().map(String::as_str).collect();
            nextsycl_model3d_unirig::check::run(&e, &dir, &st, opt(args, "--mesh").map(Path::new), &mut log).map_err(|e| e.0)?
        }
        nextsycl_model3d_rignet::ARCH => {
            let e = nextsycl_model3d_rignet::RigNet::load(&files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
            let st = stages("joints,root,bones,skin");
            let st: Vec<&str> = st.iter().map(String::as_str).collect();
            nextsycl_model3d_rignet::check::run(&e, &dir, &st, opt(args, "--mesh").map(Path::new), &mut log).map_err(|e| e.0)?
        }
        a => return Err(format!("check knows {}, {}, {}, {} and {} (not {a})", nextsycl_model3d_hunyuan3d::ARCH_21, nextsycl_model3d_hunyuan3d::ARCH_MINI,
                                nextsycl_model3d_animateanymesh::ARCH, nextsycl_model3d_unirig::ARCH, nextsycl_model3d_rignet::ARCH)),
    };
    println!("worst relative error {worst:.2e}");
    Ok(())
}

/// `nextsycl model3d check DIR`: the engine's stages against a reference's dumps (TRELLIS.2 / Pixal3D)
fn check_trellis2(cfg: &Config, args: &[String]) -> Result<(), String> {
    let dir = Path::new(positional(args).into_iter().next().ok_or(USAGE)?.as_str()).to_path_buf();
    nextsycl_core::use_kind("model3d");
    let (m, files) = model(cfg, args, "generate")?;
    let arch = m["arch"].as_str().unwrap_or("");
    if arch != nextsycl_model3d_trellis2::ARCH && arch != nextsycl_model3d_trellis2::ARCH_PIXAL {
        return Err(format!("check knows {} and {}, not {arch}", nextsycl_model3d_trellis2::ARCH, nextsycl_model3d_trellis2::ARCH_PIXAL));
    }
    let arch = if arch == nextsycl_model3d_trellis2::ARCH { nextsycl_model3d_trellis2::ARCH } else { nextsycl_model3d_trellis2::ARCH_PIXAL };
    let stages: Vec<&str> = opt(args, "--stages").unwrap_or("dino,ss,ssdec,shape,tex,sdec,tdec").split(',').collect();
    let pic = opt(args, "--picture").map(|p| -> Result<nextsycl_model3d_trellis2::Picture, String> {
        nextsycl_model3d_trellis2::Picture::decode(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?)
    }).transpose()?;
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = nextsycl_model3d_trellis2::Trellis2::load(arch, &files, &g, &options(&m, args)?, &mut log).map_err(|e| e.0)?;
    let worst = nextsycl_model3d_trellis2::check::run(&e, &dir, pic.as_ref(), &stages, args.iter().any(|a| a == "--full"), &mut log).map_err(|e| e.0)?;
    println!("worst relative error {worst:.2e}");
    Ok(())
}

/// The 3D server's container (`model3d serve` in the foreground, `model3d start` in the background)
const SERVED: crate::served::Served = crate::served::Served { name: "nextsycl-model3d", kind: "model3d", port: "8088" };

/// Where meshes go: --out, NS_MODEL3D_OUT, ~/.local/share/nextsycl/model3d
fn out_dir(cfg: &Config, args: &[String]) -> PathBuf {
    opt(args, "--out").map(PathBuf::from).or_else(|| cfg.get("NS_MODEL3D_OUT").map(PathBuf::from)).unwrap_or_else(|| {
        PathBuf::from(format!("{}/.local/share/nextsycl/model3d", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())))
    })
}

/// `nextsycl model3d serve`: from the host, the build image as container `nextsycl-model3d` running this command
/// inside with `--here`; there, the engine loaded and served (nextsycl_serve::model3d)
fn serve(cfg: &Config, args: &[String], detach: bool) -> Result<(), String> {
    let id = args.first().filter(|a| !a.starts_with("--")).map(String::as_str);
    let mut margs = args.to_vec();
    if let Some(i) = id {
        margs.extend(["--model".to_string(), i.to_string()]);
    }
    let (m, files) = model(cfg, &margs, "")?;
    let model_id = m["id"].as_str().unwrap_or("model3d").to_string();
    let out = out_dir(cfg, args);
    if args.iter().any(|a| a == "--here") {
        return serve_here(&m, files, args, out, cfg);
    }
    use crate::container::{mount, Ce};
    options(&m, args)?;
    let ce = Ce::new(cfg)?;
    ce.need_image()?;
    for f in ["nextsycl", "libnextsycl-model3d.so"] {
        if !cfg.dist.join(f).exists() {
            return Err(format!("{}: not built yet (./build.sh)", cfg.dist.join(f).display()));
        }
    }
    if args.iter().any(|a| a == "--wfe") && !cfg.dist.join("wfe/model3d/index.html").exists() {
        return Err(format!("{}: the web front end is not built yet (./build.sh wfe)", cfg.dist.join("wfe").display()));
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let name = SERVED.name_for(args)?;
    if ce.running(&name) {
        return Err(format!("{name} is running already ({} stop {name})", ce.bin));
    }
    ce.remove(&name);
    let mut a = SERVED.run_args(detach, args)?;
    a.extend(ce.user_args());
    a.extend(ce.gpu_args());
    a.extend(mount(&cfg.dist, "/app", true));
    let reg = models::registry(cfg);
    let mut dirs = std::collections::BTreeSet::new();
    let mut paths: Vec<PathBuf> = files.values().cloned().collect();
    paths.push(reg.clone());
    for p in paths {
        let d = if p.is_dir() { p } else { p.parent().map(PathBuf::from).unwrap_or_default() };
        if !d.as_os_str().is_empty() && d.exists() && dirs.insert(d.clone()) {
            a.extend(mount(&d, &d.to_string_lossy(), true));
        }
    }
    a.extend(mount(&out, &out.to_string_lossy(), false));
    a.extend(["-e".into(), format!("NS_REGISTRY={}", reg.display()), "-e".into(), "ONEAPI_DEVICE_SELECTOR=level_zero:*".into()]);
    // the engines' own settings from the configuration
    for (k, v) in cfg.with_prefix("NS_M3D_").into_iter().chain(cfg.with_prefix("NSD_")) {
        a.extend(["-e".into(), format!("{k}={v}")]);
    }
    a.extend([ce.image.clone(), "bash".into(), "-c".into(),
              "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/nextsycl model3d serve \"$@\"".into(), "nextsycl".into()]);
    a.push(model_id);
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
    nextsycl_core::use_kind("model3d");
    let ks = engines();
    let k = nextsycl_model3d::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    let g = gpu(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = (k.load)(&files, std::slice::from_ref(&g), &options(m, args)?, &mut log).map_err(|e| e.0)?;
    let wfe = args.iter().any(|a| a == "--wfe").then(|| cfg.dist.join("wfe"));
    let cors: Vec<String> = opt(args, "--cors").map(str::to_string).or(cfg.get("NS_CORS")).map(|c| c.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let addr = format!("{}:{}", opt(args, "--host").unwrap_or("127.0.0.1"), opt(args, "--port").unwrap_or("8088"));
    let mut srv = nextsycl_serve::model3d::Model3dServer::new(m["id"].as_str().unwrap_or("model3d").to_string(), g, e, Some(out), wfe, cors)?;
    srv.idle = crate::served::idle_exit(args)?;
    Arc::new(srv).run(&addr)
}

/// `nextsycl model3d <command>`
pub fn cmd(cfg: &Config, args: &[String], selftest: impl Fn(&[String]) -> Result<(), String>) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("engines") => {
            super::list_engines(engines().iter().map(|k| (k.archs.join(", "), k.name, k.options.to_vec())));
            Ok(())
        }
        Some("selftest") => selftest(rest),
        Some("gen" | "generate") => gen(cfg, rest),
        Some("rig" | "animate") => rig(cfg, rest),
        Some("tool" | "tools") => tool(rest),
        Some("check") => check(cfg, rest),
        Some("serve") => serve(cfg, rest, false),
        Some("start") => SERVED.start(cfg, rest, |d| serve(cfg, rest, d)),
        Some("ps") => SERVED.ps(cfg, rest),
        Some("stop") => SERVED.stop(cfg, rest),
        Some("logs") => SERVED.logs(cfg, rest),
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_told_from_options() {
        let a: Vec<String> = ["a.png", "--seed", "4", "b.png:back", "--texture", "c.png", "--out", "x.glb"].iter().map(|s| s.to_string()).collect();
        assert_eq!(positional(&a), vec!["a.png", "b.png:back", "c.png"]);
    }
}
