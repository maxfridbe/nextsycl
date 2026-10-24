//! The engine against reference/mingimage/ref.py's dumps (inclusionAI's DiT class and the text side line for line,
//! on the CPU in float32, on the same files): each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   tokens  the prompt's template             -> ids.npy (exactly)
//!   te      ids.npy                           -> te_final [T, 2048], cap [256, 2560], direct [L, 3840]
//!   dit     cap, direct (the reference's)     -> cr_out (the context refiner); at steps 0 and 5 lat_i -> nr_out (step
//!                                                0), l0_out (step 0), vel_i
//!   steps   lat_0 through every step          -> lat_final
//!   vae     lat_final                         -> rgba [4, H, W]; our picture of our latents written as ours.png
//! ```

use std::path::Path;

use nextsycl_image::{Error, Picture, Result};

use crate::{dit, sigmas, te, MingImage};

pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: Fortran order", path.display())));
    }
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v: Vec<f64> = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

fn f32s(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

/// (relative L2 error, max |difference|, cosine)
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
    ((e2 / w2.max(1e-30)).sqrt(), mx, dot / (w2.sqrt() * g2.sqrt()).max(1e-30))
}

fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    let (rel, mx, cos) = compare(got, want);
    log(format!("{name:<24} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values)", want.len().min(got.len())));
    rel
}

pub fn run(m: &MingImage, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, f32s(v))) };
    let mut worst = 0f64;
    let ids: Vec<u32> = read_npy(&dir.join("ids.npy"))?.1.iter().map(|x| *x as u32).collect();
    let l = ids.iter().position(|i| *i == 157158).ok_or_else(|| Error("ids.npy: no <image> token".into()))?;
    if stages.contains(&"tokens") {
        let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
        let (_, ours) = m.te.ids(&prompt);
        let same = ours == ids;
        log(format!("tokens                   {} ({} vs {} ids{})", if same { "identical" } else { "DIFFER" }, ours.len(), ids.len(),
                    ours.iter().zip(&ids).position(|(a, b)| a != b).map_or(String::new(), |p| format!(", first at {p}"))));
        if !same {
            worst = worst.max(1.0);
        }
    }
    if stages.contains(&"te") {
        let mut tr = te::Trace::default();
        let c = m.te.encode_ids(&ids[..l], &ids, Some(&mut tr))?;
        let fin = tr.fin.clone();
        // each layer against the reference's, when it dumped them (te_x{i}, te_route{i}): the template's rows and the
        // query tokens' apart, and how many of the experts chosen differ
        let tl = l + 1;
        let mut mi = 0;
        for (i, x) in tr.layers.iter().enumerate() {
            let Ok((_, want)) = get(&format!("te_x{i}")) else { break };
            let (rt, _, _) = compare(&x[..tl * te::HIDDEN], &want[..tl * te::HIDDEN]);
            let (rq, _, _) = compare(&x[tl * te::HIDDEN..(tl + te::QUERY) * te::HIDDEN], &want[tl * te::HIDDEN..(tl + te::QUERY) * te::HIDDEN]);
            let routes = if i > 0 && mi < tr.routes.len() {
                let ours = &tr.routes[mi];
                mi += 1;
                match read_npy(&dir.join(format!("te_route{i}.npy"))) {
                    Ok((_, w)) => {
                        let (mut tdiff, mut qdiff) = (0usize, 0usize);
                        for tok in 0..ours.len() / 8 {
                            let a: std::collections::BTreeSet<u32> = ours[tok * 8..tok * 8 + 8].iter().copied().collect();
                            let b: std::collections::BTreeSet<u32> = w[tok * 8..tok * 8 + 8].iter().map(|v| *v as u32).collect();
                            let d = a.difference(&b).count();
                            if tok < tl { tdiff += d } else { qdiff += d }
                        }
                        format!("  experts differing: {tdiff} (template), {qdiff} (query)")
                    }
                    Err(_) => String::new(),
                }
            } else {
                String::new()
            };
            log(format!("te layer {i:<2}  template rel {rt:.2e}  query rel {rq:.2e}{routes}"));
        }
        worst = worst.max(note("te final hidden", &fin, &get("te_final")?.1, log));
        worst = worst.max(note("te query rows", &fin[(l + 1) * te::HIDDEN..(l + 1 + te::QUERY) * te::HIDDEN], &get("te_final")?.1[(l + 1) * te::HIDDEN..(l + 1 + te::QUERY) * te::HIDDEN], log));
        worst = worst.max(note("cap (connector)", &c.cap, &get("cap")?.1, log));
        worst = worst.max(note("direct", &c.direct, &get("direct")?.1, log));
    }
    let cap = get("cap")?.1;
    let direct = get("direct")?.1;
    let sig = get("sigmas")?.1;
    let (lshape, lat0) = get("lat_0")?;
    let (lh, lw) = (lshape[1], lshape[2]);
    let hw = (lh / 2, lw / 2);
    let n = hw.0 * hw.1;
    if stages.contains(&"dit") {
        let mut g = dit::Grabs::default();
        let pre = m.dit.prefix(&m.nsd, &cap, &direct, l, Some(&mut g))?;
        let c = pre.rows;
        worst = worst.max(note("context refiner", &g.cr_out, &get("cr_out")?.1[..c * dit::DIM], log));
        let mods = m.dit.mods(&m.nsd, &sig)?;
        for i in [0usize, 5] {
            let (_, li) = get(&format!("lat_{i}"))?;
            let lat = nextsycl_core::DevBuf::from_f32(&m.dit_gpu(), &dit::patchify(&li, lh, lw))?;
            let mut gi = dit::Grabs::default();
            let v = m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[i], if i == 0 { Some(&mut gi) } else { None })?.to_f32()?;
            if i == 0 {
                worst = worst.max(note("noise refiner", &gi.nr_out, &get("nr_out")?.1[..n * dit::DIM], log));
                let want = get("l0_out")?.1;
                worst = worst.max(note("layer 0 (image rows)", &gi.l0_out[..n * dit::DIM], &want[..n * dit::DIM], log));
                worst = worst.max(note("layer 0 (caption rows)", &gi.l0_out[n * dit::DIM..(n + c) * dit::DIM], &want[n * dit::DIM..(n + c) * dit::DIM], log));
            }
            let want = dit::patchify(&get(&format!("vel_{i}"))?.1, lh, lw);
            worst = worst.max(note(&format!("velocity, step {i}"), &v, &want, log));
        }
    }
    let mut ours_lat = None;
    if stages.contains(&"steps") {
        let pre = m.dit.prefix(&m.nsd, &cap, &direct, l, None)?;
        let t0 = std::time::Instant::now();
        let lat = m.sample(&pre, None, 1.0, &lat0, lh, lw, &sig, &mut |_| Ok(()))?;
        log(format!("{} steps in {:.1} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents after the steps", &lat, &get("lat_final")?.1, log));
        ours_lat = Some(lat);
    }
    if stages.contains(&"vae") {
        let (_, lf) = get("lat_final")?;
        let (img, pw, ph) = m.vae.decode_planes(&m.nsd, &lf, lh, lw)?;
        let k = crate::ffi::api()?;
        let f = nextsycl_core::DevBuf::f32(&m.dit_gpu(), pw * ph * 4)?;
        // SAFETY: ph x pw x 4 halves in, as many floats out.
        crate::ffi::check(unsafe { (k.to_float)(m.dit_gpu().raw(), img.ptr(), f.fp(), (pw * ph * 4) as i64) }, "to float")?;
        let v = f.to_f32()?;
        // ours [H, W, 4] (unclamped) -> [4, H, W], clamped as the reference's
        let planes: Vec<f32> = (0..4).flat_map(|c| (0..pw * ph).map(move |p| (c, p))).map(|(c, p)| v[p * 4 + c].clamp(-1.0, 1.0)).collect();
        worst = worst.max(note("vae (rgba)", &planes, &get("rgba")?.1, log));
        let lat = ours_lat.unwrap_or(lf);
        let (rgba, pw, ph) = m.vae.decode(&m.nsd, &lat, lh, lw)?;
        let pic = Picture { width: pw as u32, height: ph as u32, channels: 4, data: rgba };
        pic.write_png(&dir.join("ours.png"), &[]).map_err(Error)?;
        let alpha_min = pic.data.chunks_exact(4).map(|c| c[3]).min().unwrap_or(255);
        log(format!("ours.png: {pw}x{ph}, alpha down to {alpha_min}"));
    }
    if stages.contains(&"full") {
        // the whole pipeline on our side - our text encoder's condition too - from the reference's starting noise: the
        // picture to put beside ref.png
        let c = m.te.encode_ids(&ids[..l], &ids, None)?;
        let pre = m.dit.prefix(&m.nsd, &c.cap, &c.direct, c.direct_rows, None)?;
        let t0 = std::time::Instant::now();
        let lat = m.sample(&pre, None, 1.0, &lat0, lh, lw, &sig, &mut |_| Ok(()))?;
        log(format!("full: {} steps in {:.1} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents (all ours)", &lat, &get("lat_final")?.1, log));
        let (rgba, pw, ph) = m.vae.decode(&m.nsd, &lat, lh, lw)?;
        let pic = Picture { width: pw as u32, height: ph as u32, channels: 4, data: rgba };
        pic.write_png(&dir.join("ours_full.png"), &[]).map_err(Error)?;
        let want = get("rgba")?.1;
        let ours: Vec<f32> = (0..4).flat_map(|ch| (0..pw * ph).map(move |p| (ch, p))).map(|(ch, p)| pic.data[p * 4 + ch] as f32 / 127.5 - 1.0).collect();
        note("picture (all ours)", &ours, &want, log);
    }
    let _ = sigmas;
    Ok(worst)
}

impl MingImage {
    pub(crate) fn dit_gpu(&self) -> std::sync::Arc<nextsycl_core::Gpu> {
        self.nsd.gpu.clone()
    }
}
