//! Stage by stage against Tencent's own code (reference/hunyuan3d/ref_mini.py, ref_21.py: float32 on the CPU, the
//! same files). Each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   pre    the picture                -> pixels.npy      the cut-out, recentred, resized, normalized picture
//!   cond   pixels.npy                 -> cond.npy        DINOv2's last hidden state
//!   dit    x_<i> + cond.npy at sigma  -> v_<i>.npy       the (guided) velocity of every step; 2.1 also each pass
//!                                     -> d0_*.npy, b0.npy   the stream after the first block (step 0)
//!   steps  noise.npy + cond.npy       -> latents.npy     the whole sampler
//!   vae    latents.npy                -> vae_tokens.npy  the decoder's transformer
//!          + queries.npy              -> logits.npy      the geometry decoder at fixed points
//!   grid   latents.npy                -> grid.npy        the dense grid at the dump's resolution, then its mesh
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result, Texture};

use crate::ops::read_f32;
use crate::{Dit, Hunyuan3d, ARCH_MINI};

/// [n, c, h, w] -> [n, h, w, c]
pub fn nhwc(v: &[f32], n: usize, c: usize, hw: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for i in 0..n {
        for ch in 0..c {
            for p in 0..hw {
                o[(i * hw + p) * c + ch] = v[(i * c + ch) * hw + p];
            }
        }
    }
    o
}

/// [n, h, w, c] -> [n, c, h, w]
pub fn nchw(v: &[f32], n: usize, c: usize, hw: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for i in 0..n {
        for ch in 0..c {
            for p in 0..hw {
                o[(i * c + ch) * hw + p] = v[(i * hw + p) * c + ch];
            }
        }
    }
    o
}

/// The paint stages against reference/hunyuan3d/ref_paint.py's dumps in `dir` (its shape as `mesh.glb` beside
/// them), each from the reference's own input:
///
/// ```text
///   maps    mesh.glb             -> normal.npy, position.npy   the six views' maps (rendered at 2,048, resized)
///   pic     the picture          -> ref.npy                    resized, on white
///   dino    ref.npy              -> dino.npy; dino.npy -> dino_proj.npy
///   enc     ref.npy, the maps    -> ref_latents.npy, embeds_normal.npy, embeds_position.npy
///   dual    ref_latents.npy      -> cond_<layer>.npy            the reference UNet's stored states
///   unet    x_0.npy (+ all the reference's conditions) -> out_0.npy (36 pictures), v_0.npy (guided)
///   unipc   v_i, x_i             -> latents_i                   the sampler alone
///   steps   noise.npy            -> latents_<last>.npy          the whole denoising
///   decode  latents_<last>.npy   -> images.npy                  the twelve views
/// ```
pub fn run_paint(e: &Hunyuan3d, dir: &Path, stages: &[String], picture: Option<&Texture>, log: &mut dyn FnMut(String)) -> Result<f64> {
    use crate::paint::{self, nn::Mem, unet::{BRANCHES, MATS, VIEWS}, SIZE};
    let p = e.paint.as_ref().ok_or_else(|| Error("the paint model is not loaded (2.1 with paint-unet, paint-vae, paint-dino)".into()))?;
    let npy = |name: &str| read_npy(&dir.join(name)).map(|x| x.1);
    let has = |s: &str| stages.iter().any(|x| x == s || x == "paint");
    let ops = &e.ops;
    let mem = Mem::new(&ops.gpu, 3 << 30, 4 << 30)?;
    let mut worst = 0f64;
    let mut keep = |r: f64| worst = worst.max(r);
    let lat = 64usize;
    let hw = lat * lat;
    let maps_ref = |name: &str| -> Result<Vec<Vec<f32>>> {
        let v = npy(name)?;
        Ok(v.chunks_exact(SIZE * SIZE * 3).map(|c| c.iter().map(|x| (x * 255.0).round()).collect()).collect())
    };
    if has("maps") {
        let mesh = nextsycl_model3d::read_mesh(&std::fs::read(dir.join("mesh.glb")).map_err(|e| Error(format!("mesh.glb: {e}")))?).map_err(Error)?;
        let t = std::time::Instant::now();
        let scene = paint::render::Scene::new(&mesh);
        let (n, ps) = paint::maps(&scene);
        log(format!("maps: rendered in {:.2} s", t.elapsed().as_secs_f64()));
        let flat = |m: &[Vec<f32>]| m.iter().flatten().map(|x| x / 255.0).collect::<Vec<f32>>();
        keep(line("normal maps", &flat(&n), &npy("normal.npy")?, log)?);
        keep(line("position maps", &flat(&ps), &npy("position.npy")?, log)?);
        let off = flat(&ps).iter().zip(npy("position.npy")?).filter(|(a, b)| (**a - b).abs() > 1.5 / 255.0).count();
        log(format!("position maps: {off} values more than one level off"));
    }
    if has("pic") {
        if let Some(t) = picture {
            let pic = paint::reference_picture(t);
            keep(line("picture", &pic.iter().map(|x| x / 255.0).collect::<Vec<_>>(), &npy("ref.npy")?, log)?);
        }
    }
    let ref_pic: Vec<f32> = npy("ref.npy")?.iter().map(|x| (x * 255.0).round()).collect();
    let dino_ref = npy("dino.npy")?;
    let dino_ref_dev = DevBuf::from_f32(&ops.gpu, &dino_ref)?;
    if has("dino") {
        let s = p.dino_states(ops, &ref_pic)?;
        keep(line("dino", &read_f32(&s, dino_ref.len())?, &dino_ref, log)?);
        let pr = p.project(ops, &dino_ref_dev)?;
        let want = npy("dino_proj.npy")?;
        let per = want.len() / 3;
        keep(line("dino proj (zero)", &crate::ops::read_half(&pr[0], per)?, &want[..per], log)?);
        keep(line("dino proj", &crate::ops::read_half(&pr[1], per)?, &want[2 * per..], log)?);
    }
    let (nref, pref) = (maps_ref("normal.npy")?, maps_ref("position.npy")?);
    if has("enc") {
        let t = std::time::Instant::now();
        let (rl, nl, pl) = p.encode_all(ops, &mem, &nref, &pref, &ref_pic)?;
        log(format!("enc: 13 pictures in {:.2} s", t.elapsed().as_secs_f64()));
        keep(line("ref latent", &nchw(&rl, 1, 4, hw), &npy("ref_latents.npy")?, log)?);
        keep(line("normal latents", &nchw(&nl, VIEWS, 4, hw), &npy("embeds_normal.npy")?, log)?);
        keep(line("position latents", &nchw(&pl, VIEWS, 4, hw), &npy("embeds_position.npy")?, log)?);
    }
    let meta: serde_json::Value = std::fs::read(dir.join("meta.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let steps = meta["steps"].as_u64().unwrap_or(15) as usize;
    if has("dual") || has("unet") || has("steps") {
        let rl = nhwc(&npy("ref_latents.npy")?, 1, 4, hw);
        let nl = nhwc(&npy("embeds_normal.npy")?, VIEWS, 4, hw);
        let pl = nhwc(&npy("embeds_position.npy")?, VIEWS, 4, hw);
        let pos01: Vec<Vec<f32>> = pref.iter().map(|m| m.iter().map(|x| x / 255.0).collect()).collect();
        let t = std::time::Instant::now();
        let proj = p.project(ops, &dino_ref_dev)?;
        let mut cond = p.prepare(ops, &mem, &rl, &proj, &pos01)?;
        log(format!("conditions (the reference UNet, the keys) in {:.2} s", t.elapsed().as_secs_f64()));
        if has("dual") {
            for name in ["down_0_0_0", "mid_0_0", "up_3_2_0"] {
                let st = cond.states.get(name).ok_or_else(|| Error(format!("no state {name}")))?;
                let want = npy(&format!("cond_{name}.npy"))?;
                keep(line(&format!("dual {name}"), &crate::ops::read_half(st, want.len())?, &want, log)?);
            }
        }
        let sched = paint::unipc::UniPc::new(steps);
        if has("unet") {
            let x0 = nhwc(&npy("x_0.npy")?, MATS * VIEWS, 4, hw);
            let t = std::time::Instant::now();
            let (v, raw) = p.guided(ops, &mem, &mut cond, &x0, &nl, &pl, sched.timesteps[0] as f32)?;
            log(format!("unet: a step (36 pictures) in {:.2} s", t.elapsed().as_secs_f64()));
            // the reference's batch is (branch, material, view)
            let per = hw * 4;
            let mut ord = vec![0f32; raw.len()];
            for m in 0..MATS {
                for b in 0..BRANCHES {
                    for vi in 0..VIEWS {
                        let src = ((m * BRANCHES + b) * VIEWS + vi) * per;
                        let dst = ((b * MATS + m) * VIEWS + vi) * per;
                        ord[dst..dst + per].copy_from_slice(&raw[src..src + per]);
                    }
                }
            }
            keep(line("unet out (36)", &nchw(&ord, MATS * BRANCHES * VIEWS, 4, hw), &npy("out_0.npy")?, log)?);
            keep(line("unet v_0 (guided)", &nchw(&v, MATS * VIEWS, 4, hw), &npy("v_0.npy")?, log)?);
        }
        if has("steps") {
            let mut s = paint::unipc::UniPc::new(steps);
            let mut x = nhwc(&npy("noise.npy")?, MATS * VIEWS, 4, hw);
            let t = std::time::Instant::now();
            for i in 0..steps {
                let (v, _) = p.guided(ops, &mem, &mut cond, &x, &nl, &pl, s.timesteps[i] as f32)?;
                x = s.step(&v, &x);
                if let Ok(w) = npy(&format!("latents_{i}.npy")) {
                    keep(line(&format!("latents {i}"), &nchw(&x, MATS * VIEWS, 4, hw), &w, log)?);
                }
            }
            log(format!("steps: {steps} in {:.2} s", t.elapsed().as_secs_f64()));
        }
    }
    if has("unipc") {
        let mut s = paint::unipc::UniPc::new(steps);
        for i in 0..steps {
            let (Ok(v), Ok(x), Ok(w)) = (npy(&format!("v_{i}.npy")), npy(&format!("x_{i}.npy")), npy(&format!("latents_{i}.npy"))) else { break };
            let got = s.step(&v, &x);
            keep(line(&format!("unipc {i}"), &got, &w, log)?);
        }
    }
    if has("decode") {
        let last = npy(&format!("latents_{}.npy", steps - 1))?;
        let z = nhwc(&last, MATS * VIEWS, 4, hw);
        let want = npy("images.npy")?;
        let per = SIZE * SIZE * 3;
        let t = std::time::Instant::now();
        let mut got = Vec::new();
        for i in 0..MATS * VIEWS {
            got.extend(p.vae.decode(ops, &mem, &z[i * hw * 4..(i + 1) * hw * 4], lat, lat)?);
        }
        log(format!("decode: 12 views in {:.2} s", t.elapsed().as_secs_f64()));
        keep(line("views", &got, &want[..per * MATS * VIEWS], log)?);
    }
    Ok(worst)
}

/// A little-endian float32 .npy: (shape, values)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if !head.contains("'<f4'") || head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: {head} (float32, C order expected)", path.display())));
    }
    let shape: Vec<usize> = head
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let v = b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((shape, v))
}

/// How far `got` is from `want`: (relative RMS error, max abs error, cosine)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
        mx = mx.max((g - w).abs());
    }
    ((e2 / w2.max(1e-30)).sqrt(), mx, dot / (g2.sqrt() * w2.sqrt()).max(1e-30))
}

fn line(what: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> Result<f64> {
    if got.len() != want.len() {
        return Err(Error(format!("{what}: {} values, the reference {}", got.len(), want.len())));
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<16} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.6}  ({} values)", want.len()));
    Ok(rel)
}

/// Every stage in `stages` against the dumps in `dir` (`picture`: the reference's input, for "pre"); the worst
/// relative error
pub fn run(e: &Hunyuan3d, dir: &Path, stages: &[String], picture: Option<&Texture>, log: &mut dyn FnMut(String)) -> Result<f64> {
    let npy = |name: &str| read_npy(&dir.join(name)).map(|x| x.1);
    let meta: serde_json::Value = std::fs::read(dir.join("meta.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let has = |s: &str| stages.iter().any(|x| x == s || x == "all");
    let gpu = e.ops.gpu.clone();
    let mut worst = 0f64;
    let mut keep = |r: f64| worst = worst.max(r);
    if has("pre") {
        match picture {
            Some(t) => {
                let (px, note) = crate::picture::prepare(t, true, e.arch.proc_size, e.arch.enc_size, crate::BORDER)?;
                log(format!("pre: {note}"));
                keep(line("pre pixels", &px, &npy("pixels.npy")?, log)?);
            }
            None => log("pre: no picture given (--picture): skipped".into()),
        }
    }
    let cond_ref = npy("cond.npy")?;
    if has("cond") {
        let t = std::time::Instant::now();
        let c = e.condition(&npy("pixels.npy")?)?;
        let got = read_f32(&c, cond_ref.len())?;
        log(format!("cond: DINOv2 in {:.2} s", t.elapsed().as_secs_f64()));
        keep(line("cond", &got, &cond_ref, log)?);
    }
    let cond = DevBuf::from_f32(&gpu, &cond_ref)?;
    let noise = npy("noise.npy")?;
    let tokens = noise.len() / e.vae.channels;
    let sigmas = npy("sigmas.npy")?;
    let cfg = meta["guidance"].as_f64().unwrap_or(e.arch.cfg as f64) as f32;
    if has("dit") || has("steps") {
        let mut st = e.run_state(&cond, tokens)?;
        if has("dit") {
            let v = DevBuf::new(&gpu, noise.len() * 4)?;
            let n = noise.len();
            for i in 0..sigmas.len() - 1 {
                let Ok(xi) = npy(&format!("x_{i}.npy")) else { break };
                let x = DevBuf::from_f32(&gpu, &xi)?;
                if i == 0 {
                    // the stream after the first block
                    match (&e.dit, &mut st.dit) {
                        (Dit::Flux(f), crate::RunDit::Flux(sc)) => {
                            f.forward(&e.ops, sc, &x, sigmas[0], cfg, &v, Some(0))?;
                            let (txt, img) = f.stream(sc)?;
                            if let (Ok(wt), Ok(wi)) = (npy("d0_txt.npy"), npy("d0_img.npy")) {
                                keep(line("dit block 0 txt", &txt, &wt, log)?);
                                keep(line("dit block 0 img", &img, &wi, log)?);
                            }
                        }
                        (Dit::Plain(p), crate::RunDit::Plain(sc)) => {
                            p.forward(&e.ops, sc, &x, sigmas[0], 0, &v, Some(0))?;
                            if let Ok(w) = npy("b0.npy") {
                                keep(line("dit block 0", &p.stream(sc)?, &w, log)?);
                            }
                            for (which, name) in [(0, "vc_0.npy"), (1, "vu_0.npy")] {
                                if let Ok(w) = npy(name) {
                                    p.forward(&e.ops, sc, &x, sigmas[0], which, &v, None)?;
                                    keep(line(if which == 0 { "dit cond pass" } else { "dit uncond pass" }, &read_f32(&v, n)?, &w, log)?);
                                }
                            }
                        }
                        _ => return Err(Error("the DiT and its scratch disagree".into())),
                    }
                }
                let t = std::time::Instant::now();
                e.velocity(&mut st, &x, sigmas[i], cfg, &v)?;
                let got = read_f32(&v, n)?;
                let secs = t.elapsed().as_secs_f64();
                keep(line(&format!("dit v_{i} ({secs:.2} s)"), &got, &npy(&format!("v_{i}.npy"))?, log)?);
            }
        }
        if has("steps") {
            let t = std::time::Instant::now();
            let got = e.sample(&mut st, &noise, &sigmas, cfg, &mut |_, _| Ok(()))?;
            log(format!("steps: {} in {:.2} s", sigmas.len() - 1, t.elapsed().as_secs_f64()));
            keep(line("latents", &got, &npy("latents.npy")?, log)?);
        }
    }
    let latents = npy("latents.npy")?;
    if has("vae") {
        let t = std::time::Instant::now();
        let toks = e.vae.tokens(&e.ops, &latents)?;
        let got = read_f32(&toks, tokens * e.vae.width)?;
        log(format!("vae: transformer in {:.2} s", t.elapsed().as_secs_f64()));
        keep(line("vae tokens", &got, &npy("vae_tokens.npy")?, log)?);
        let data = e.vae.data(&e.ops, &toks, tokens)?;
        let sc = e.vae.query_scratch(&e.ops, 8192)?;
        let got = e.vae.query_points(&e.ops, &data, &sc, &npy("queries.npy")?)?;
        let want = npy("logits.npy")?;
        keep(line("vae logits", &got, &want, log)?);
        let agree = got.iter().zip(&want).filter(|(a, b)| (**a > 0.0) == (**b > 0.0)).count();
        log(format!("vae inside / outside agree at {agree} of {} points", want.len()));
    }
    if has("grid") {
        let res = meta["res"].as_u64().unwrap_or(128) as usize;
        let t = std::time::Instant::now();
        let got = e.decode_dense(&latents, res, e.vae.chunk(), &mut |_, _| Ok(()))?;
        log(format!("grid {res}: {:.2} s", t.elapsed().as_secs_f64()));
        let want = npy("grid.npy")?;
        keep(line("grid logits", &got, &want, log)?);
        let flips = got.iter().zip(&want).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
        log(format!("grid: {flips} of {} points on the other side of the surface", want.len()));
        // coarse to fine: the same surface from a share of the points
        let t = std::time::Instant::now();
        let (fast, queried) = e.decode_fast(&latents, res, e.vae.chunk(), &mut |_, _| Ok(()))?;
        let flips = fast.iter().zip(&want).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
        log(format!("grid {res} coarse to fine: {:.2} s, {queried} points queried ({:.0} %), {flips} on the other side of the surface", t.elapsed().as_secs_f64(),
                    100.0 * queried as f64 / want.len() as f64));
        let t = std::time::Instant::now();
        let m = Hunyuan3d::mesh(&got, res);
        log(format!("mesh: {} vertices, {} triangles in {:.2} s (the reference's skimage: {} vertices, {} faces)", m.positions.len(), m.triangles(),
                    t.elapsed().as_secs_f64(), meta["vertices"], meta["faces"]));
    }
    if e.arch.name == ARCH_MINI && meta["tokens"].as_u64().is_some_and(|t| t as usize != e.arch.tokens) {
        log(format!("note: the reference sampled {} tokens, the engine's default is {}", meta["tokens"], e.arch.tokens));
    }
    Ok(worst)
}
