//! Stage by stage against the reference (reference/budget3d/ref.py: Budget3D's and Idol's own modules on the same files,
//! float32 on the CPU): each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   noise    the seed (sample.json's)                 -> z_noise, cam_noise (PyTorch's generator, value for value)
//!   ctx      cond_f1, cond_f2                         -> ctx_pos, ctx_neg (Turbo: ctx_slow, ctx_fast)
//!   layout   z_noise                                  -> b<B>_s4, b<B>_s2 (scores), b<B>_idx4 / idx2 / idx1
//!   tokens   z_noise at the reference's layouts       -> b<B>_tok
//!   call     z_noise, cam_noise, the reference's contexts and layouts, t = 1, 0.6, 0.3
//!                                                     -> b<B>_k<K>_{nref, joint, head, vtok, vpts, v, vcam}
//!   sample   cond_f1 / f2 and the noise, sample.json's steps and budget -> sample_z<i>, sample_cam<i>, the layouts
//!   dec      sample_z                                 -> dec_l<l>_logits (at the reference's cells), gs_feat (at its
//!                                                        anchors), the Gaussians of its features (gs_xyz ...)
//! ```
//! and reference/budget3d/encoders.py's (another directory: the picture side):
//! ```text
//!   pre      picture.png                              -> pre (the framed picture; BiRefNet when the model has it)
//!   enc      pre                                      -> enc_dino, enc_vae (the encoders' tokens)
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::layout::Layout;
use crate::{decoder, dit, Budget3d, Params};

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
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let v = b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((shape, v))
}

/// (relative RMS error, max abs error, cosine)
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

struct Log<'a> {
    out: &'a mut dyn FnMut(String),
    worst: f64,
}

impl Log<'_> {
    fn line(&mut self, what: &str, got: &[f32], want: &[f32]) -> f64 {
        if got.len() != want.len() {
            (self.out)(format!("{what:<22} {} values here, {} in the reference", got.len(), want.len()));
            self.worst = f64::INFINITY;
            return f64::INFINITY;
        }
        let (rel, mx, cos) = compare(got, want);
        (self.out)(format!("{what:<22} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.6}  ({} values)", want.len()));
        self.worst = self.worst.max(rel);
        rel
    }
    fn say(&mut self, s: String) {
        (self.out)(s);
    }
}

fn idx(v: &[f32]) -> Vec<usize> {
    v.iter().map(|x| *x as usize).collect()
}

/// The reference's layout `pre` (pre_idx4, pre_idx2, pre_idx1)
fn ref_layout(d: &Path, pre: &str) -> Result<Layout> {
    let r = |n: &str| -> Result<Vec<usize>> { Ok(idx(&read_npy(&d.join(format!("{pre}_{n}.npy")))?.1)) };
    let (i4, i2, i1) = (r("idx4")?, r("idx2")?, r("idx1")?);
    let k4 = crate::layout::N_CUBE - i4.len();
    let k2 = i1.len();
    Layout::new(i4, i2, i1, k4, k2)
}

/// The budgets the reference dumped (b100, b50, ...), in percent
fn budgets(d: &Path) -> Vec<usize> {
    let mut b: Vec<usize> = std::fs::read_dir(d).into_iter().flatten().flatten().filter_map(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        n.strip_suffix("_tok.npy").and_then(|s| s.strip_prefix('b')).and_then(|s| s.parse().ok())
    }).collect();
    b.sort_unstable_by(|a, b| b.cmp(a));
    b
}

/// The picture side: Idol's framing and the two encoders
fn encoders(e: &Budget3d, d: &Path, stages: &[&str], lg: &mut Log) -> Result<f64> {
    let want = read_npy(&d.join("pre.npy"))?.1;
    if stages.contains(&"pre") {
        let b = std::fs::read(d.join("picture.png")).map_err(|x| Error(format!("{}: {x}", d.join("picture.png").display())))?;
        let pic = crate::Picture::decode(&b).map_err(Error)?;
        let r = if crate::prep::has_real_alpha(&pic) { None } else { e.remover_for_check()? };
        lg.say(format!("pre: {} x {}, {}", pic.width, pic.height, if crate::prep::has_real_alpha(&pic) { "its own alpha" } else if r.is_some() {
            "cut out by BiRefNet" } else { "no remover: taken whole" }));
        let t0 = std::time::Instant::now();
        let pre = crate::prep::prepare(&pic, r.as_deref(), 1.2)?;
        lg.say(format!("pre: framed in {:.2} s", t0.elapsed().as_secs_f64()));
        let got: Vec<f32> = pre.data.iter().map(|v| *v as f32).collect();
        lg.line("framed picture", &got, &want);
        let off = got.iter().zip(&want).filter(|(a, b)| (*a - *b).abs() > 2.0).count();
        lg.say(format!("framed picture        {off} of {} values more than 2 levels off", want.len()));
    }
    if stages.contains(&"enc") {
        let pre = crate::Picture { width: 1024, height: 1024, channels: 3, data: want.iter().map(|v| *v as u8).collect() };
        let t0 = std::time::Instant::now();
        let (f1, f2) = e.encode(&pre)?;
        lg.say(format!("enc: both encoders in {:.2} s (loads included)", t0.elapsed().as_secs_f64()));
        lg.line("DINOv3 tokens", &f1, &read_npy(&d.join("enc_dino.npy"))?.1);
        lg.line("FLUX.2 VAE tokens", &f2, &read_npy(&d.join("enc_vae.npy"))?.1);
    }
    Ok(lg.worst)
}

/// The stages asked for, against the dumps in `d`; the worst relative error
pub fn run(e: &Budget3d, d: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut lg = Log { out: log, worst: 0.0 };
    let npy = |n: &str| -> Result<Vec<f32>> { Ok(read_npy(&d.join(format!("{n}.npy")))?.1) };
    let has = |n: &str| d.join(format!("{n}.npy")).is_file();
    if stages.contains(&"pre") || stages.contains(&"enc") {
        return encoders(e, d, stages, &mut lg);
    }
    let cx = &e.cx;
    let g = &cx.gpu;
    let turbo = e.dit.turbo;
    let info: serde_json::Value = std::fs::read(d.join("sample.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let seed = info["seed"].as_u64().unwrap_or(42);
    if info["turbo"].as_bool().is_some_and(|t| t != turbo) {
        return Err(Error(format!("the dumps are of {} model, this one is {}", if turbo { "the 50-step" } else { "the Turbo" }, if turbo { "Turbo" } else { "the 50-step" })));
    }
    let (f1, f2) = (npy("cond_f1")?, npy("cond_f2")?);
    let ctx_names: &[&str] = if turbo { &["ctx_slow", "ctx_fast"] } else { &["ctx_pos", "ctx_neg"] };
    if stages.contains(&"noise") {
        let (z, cam) = crate::noise(seed);
        lg.line("noise z", &z, &npy("z_noise")?);
        lg.line("noise cam", &cam, &npy("cam_noise")?);
    }
    let w1 = e.dit.work(g, crate::layout::N_SOBOL, 1)?;
    if stages.contains(&"ctx") {
        let zero1 = vec![0f32; f1.len()];
        let zero2 = vec![0f32; f2.len()];
        for (i, n) in ctx_names.iter().enumerate() {
            let c = if turbo { e.dit.context(cx, &w1, &f1, &f2, i)? } else if i == 0 { e.dit.context(cx, &w1, &f1, &f2, 0)? } else {
                e.dit.context(cx, &w1, &zero1, &zero2, 0)?
            };
            lg.line(n, &c.to_f32()?, &npy(n)?);
        }
    }
    drop(w1);
    let zn = npy("z_noise")?;
    let camn = npy("cam_noise")?;
    for b in budgets(d) {
        let pre = format!("b{b}");
        let rl = ref_layout(d, &pre)?;
        if stages.contains(&"layout") {
            let (s4, s2) = e.dit.octree.detail_scores(&zn, dit::IN);
            let f = |v: &[f64]| v.iter().map(|x| *x as f32).collect::<Vec<f32>>();
            lg.line(&format!("{pre} scores 4"), &f(&s4), &npy(&format!("{pre}_s4"))?);
            lg.line(&format!("{pre} scores 2"), &f(&s2), &npy(&format!("{pre}_s2"))?);
            let mine = e.dit.octree.build(&s4, &s2, rl.tokens())?;
            lg.say(format!("{pre} layout            {} (k4 {}, k2 {} here; reference k4 {}, k2 {})", if mine == rl { "identical" } else { "DIFFERENT" }, mine.k4, mine.k2,
                           rl.k4, rl.k2));
            if mine != rl {
                lg.worst = f64::INFINITY;
            }
        }
        if !stages.contains(&"tokens") && !stages.contains(&"call") {
            continue;
        }
        let n = rl.tokens();
        let seqs = if turbo { 1 } else { 2 };
        let w = e.dit.work(g, n, seqs)?;
        if stages.contains(&"tokens") {
            e.dit.tokens(cx, &w, &zn, &rl)?;
            lg.line(&format!("{pre} tokens"), &w.tokens()?, &npy(&format!("{pre}_tok"))?);
        }
        if stages.contains(&"call") {
            let ctxs: Vec<DevBuf> = ctx_names.iter().map(|c| -> Result<DevBuf> { DevBuf::from_f32(g, &npy(c)?) }).collect::<Result<_>>()?;
            for (k, t) in [1.0f32, 0.6, 0.3].iter().enumerate() {
                let p = format!("{pre}_k{k}");
                if !has(&format!("{p}_v")) {
                    continue;
                }
                let ctx: Vec<&DevBuf> = if turbo { vec![&ctxs[dit::phase(true, *t)]] } else { ctxs.iter().collect() };
                let mut probe = dit::Probe::default();
                let (v, vc) = e.dit.velocity(cx, &w, &zn, &camn, *t, &rl, &ctx, 3.0, Some(&mut probe))?;
                lg.line(&format!("{p} noise refiner"), &probe.nref, &npy(&format!("{p}_nref"))?);
                lg.line(&format!("{p} joint"), &probe.joint, &npy(&format!("{p}_joint"))?);
                lg.line(&format!("{p} head"), &probe.head, &npy(&format!("{p}_head"))?);
                lg.line(&format!("{p} v tokens"), &probe.vtok, &npy(&format!("{p}_vtok"))?);
                if has(&format!("{p}_vpts")) {
                    lg.line(&format!("{p} v points"), &probe.vpts, &npy(&format!("{p}_vpts"))?);
                }
                lg.line(&format!("{p} velocity"), &v, &npy(&format!("{p}_v"))?);
                lg.line(&format!("{p} camera"), &vc, &npy(&format!("{p}_vcam"))?);
            }
        }
    }
    if stages.contains(&"sample") && has("sample_z") {
        let steps = info["steps"].as_u64().unwrap_or(6) as usize;
        let budget = info["budget"].as_f64().unwrap_or(0.5);
        let p = Params { steps, cfg: if turbo { 0.0 } else { 3.0 }, budget, gaussians: 262_144, margin: 1.2, seed };
        let mut lines = Vec::new();
        let t0 = std::time::Instant::now();
        let (z, cam, _) = e.sample_with(&f1, &f2, &p, &mut |_, _| Ok(()), &mut |i, z, cam, l| {
            let pre = format!("sample_layout{i}");
            let same = ref_layout(d, &pre).ok().map(|r| r == *l);
            lines.push((i, z.to_vec(), cam.to_vec(), same));
        })?;
        lg.say(format!("sample: {steps} steps at budget {budget} in {:.2} s", t0.elapsed().as_secs_f64()));
        for (i, z, cam, same) in lines {
            if let Some(s) = same {
                lg.say(format!("sample layout {i}        {}", if s { "identical" } else { "DIFFERENT" }));
            }
            lg.line(&format!("sample z{i}"), &z, &npy(&format!("sample_z{i}"))?);
            lg.line(&format!("sample cam{i}"), &cam, &npy(&format!("sample_cam{i}"))?);
        }
        lg.line("sample z", &z, &npy("sample_z")?);
        lg.line("sample cam", &cam, &npy("sample_cam")?);
    }
    if stages.contains(&"dec") && has("sample_z") {
        let lat = decoder::latent(g, &npy("sample_z")?)?;
        let kvs = e.dec.kvs(cx, &lat)?;
        for l in 1..=decoder::LEVELS {
            let n = format!("dec_l{l}");
            if !has(&format!("{n}_x")) {
                continue;
            }
            let x: Vec<[f32; 3]> = npy(&format!("{n}_x"))?.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            let got = e.dec.logits(cx, &x, 1 << l, &kvs)?;
            lg.line(&format!("octree level {l} logits"), &got, &npy(&format!("{n}_logits"))?);
        }
        let anchors: Vec<[f32; 3]> = npy("dec_anchors")?.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let t0 = std::time::Instant::now();
        let feats = e.gs.features(cx, &lat, &anchors)?;
        lg.say(format!("Gaussian decoder: {} anchors in {:.2} s", anchors.len(), t0.elapsed().as_secs_f64()));
        let want = npy("gs_feat")?;
        lg.line("gaussian features", &feats, &want);
        let s = e.gs.gaussians(&anchors, &want);
        let flat3 = |v: &[[f32; 3]]| v.iter().flatten().copied().collect::<Vec<f32>>();
        lg.line("gaussians xyz", &flat3(&s.positions), &npy("gs_xyz")?);
        lg.line("gaussians colour", &flat3(&s.dc), &npy("gs_fdc")?);
        lg.line("gaussians scale", &flat3(&s.scales), &npy("gs_scaling")?);
        lg.line("gaussians rotation", &s.rotations.iter().flatten().copied().collect::<Vec<f32>>(), &npy("gs_rotation")?);
        lg.line("gaussians opacity", &s.opacity, &npy("gs_opacity")?);
        // our own octree draw: as many anchors, inside the unit cube
        let mut gen = nextsycl_diffusion::torch_rng::Mt19937::new(seed);
        let mut rng = || ((gen.next_u32() >> 8) as f32) / (1u32 << 24) as f32;
        let mine = e.dec.anchors(cx, &lat, anchors.len(), &mut rng, |_, _, _| {})?;
        let inside = mine.iter().all(|a| a.iter().all(|v| (0.0..=1.0).contains(v)));
        let mean = |v: &[[f32; 3]], c: usize| v.iter().map(|a| a[c] as f64).sum::<f64>() / v.len().max(1) as f64;
        lg.say(format!("octree anchors        {} here ({}), mean ({:.3}, {:.3}, {:.3}); reference mean ({:.3}, {:.3}, {:.3})", mine.len(),
                       if inside { "inside the unit cube" } else { "SOME OUTSIDE" }, mean(&mine, 0), mean(&mine, 1), mean(&mine, 2), mean(&anchors, 0),
                       mean(&anchors, 1), mean(&anchors, 2)));
    }
    Ok(lg.worst)
}
