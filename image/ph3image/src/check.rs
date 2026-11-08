//! The engine against reference/p-h3-image/ref.py's dumps (diffusers' pipeline on the CPU in float32, the full model
//! in its diffusers form): each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   te     the prompt's tokens (prompt.txt -> ids), then the reference's ids -> hs_13 .. hs_36, cond; an edit: each
//!          picture's vision tokens (vis_img_I -> vis_emb_I, vis_deep_I_J) first
//!   venc   (an edit) ref_img_I -> the conditioning latents ref_lat_I
//!   dit    the layout (pos, tags_s), the draws (noise_v, noise_a from the seed), the refined text (txt), then each
//!          step's velocities from the reference's own latents (lat_I, aud_I -> vel_I, avel_I)
//!   steps  the reference's noise and conditioning through all the steps -> lat_final
//!   vae    lat_final -> raw (the decoder's output) and the picture
//! ```

use std::path::Path;

use nextsycl_h3::denoiser::{pack_audio, patchify, unpack_audio, unpatchify, Conditions, Denoiser, RefIn, Schedule, Shape};
use nextsycl_image::{Error, Picture, Result};

use crate::{h3e, te, PH3Image, SIGMAS};

/// A .npy file: its shape and values (float32, float64, int32 or uint8)
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
    } else if head.contains("'<f8'") {
        d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'|u1'") {
        d.iter().map(|v| *v as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
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

struct Dump<'a>(&'a Path);

impl Dump<'_> {
    fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (s, v) = read_npy(&self.0.join(format!("{name}.npy")))?;
        Ok((s, v.into_iter().map(|x| x as f32).collect()))
    }
    fn has(&self, name: &str) -> bool {
        self.0.join(format!("{name}.npy")).exists()
    }
}

fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    if got.len() != want.len() {
        log(format!("{name:14} SIZE {} vs {}", got.len(), want.len()));
        return f64::INFINITY;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{name:14} rel {rel:.3e}  max {mx:.3e}  cos {cos:.6}"));
    rel
}

/// The reference's pictures (uint8 [h, w, 3] dumps)
fn picture(d: &Dump, name: &str) -> Result<Picture> {
    let (s, v) = read_npy(&d.0.join(format!("{name}.npy")))?;
    Ok(Picture { width: s[1] as u32, height: s[0] as u32, channels: 3, data: v.into_iter().map(|x| x as u8).collect() })
}

/// Runs `stages` (te, venc, dit, steps, vae) against the dumps in `dir`; the worst relative error
pub fn run(e: &PH3Image, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let d = Dump(dir);
    let mut worst = 0f64;
    let mut pics: Vec<Picture> = Vec::new();
    while d.has(&format!("vis_img_{}", pics.len())) {
        pics.push(picture(&d, &format!("vis_img_{}", pics.len()))?);
    }
    if stages.contains(&"te") {
        let mut te = e.text_encoder(log)?;
        if !pics.is_empty() {
            te.enc.load_vision(log)?;
        }
        let (_, ids) = d.f32("ids")?;
        let ids: Vec<u32> = ids.iter().map(|v| *v as u32).collect();
        if pics.is_empty() {
            let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|er| Error(er.to_string()))?;
            let mine = te.enc.tokenize(&prompt);
            log(format!("tokens         {} of {} the same{}", mine.iter().zip(&ids).filter(|(a, b)| a == b).count(), ids.len(),
                        if mine.len() == ids.len() { "" } else { " (LENGTHS DIFFER)" }));
            if mine != ids {
                worst = f64::INFINITY;
            }
        }
        // an edit: the pictures through the tower as the processor showed them (vis_img_I: already at their size)
        let vision = te.enc.vision.as_ref();
        let mut seen = Vec::new();
        for (i, p) in pics.iter().enumerate() {
            let v = vision.ok_or_else(|| Error("no vision tower".into()))?;
            let px: Vec<f32> = p.data.iter().map(|x| *x as f32).collect();
            let s = v.see(&te.nsd, &px, p.height as usize, p.width as usize)?;
            worst = worst.max(note(&format!("vis_emb_{i}"), &s.tokens.to_f32()?, &d.f32(&format!("vis_emb_{i}"))?.1, log));
            for (j, ds) in s.deep.iter().enumerate() {
                worst = worst.max(note(&format!("vis_deep_{i}_{j}"), &ds.to_f32()?, &d.f32(&format!("vis_deep_{i}_{j}"))?.1, log));
            }
            seen.push(s);
        }
        let mut taps: Vec<usize> = vec![1];
        taps.extend(te::TAPS);
        let hs = te.enc.encode_taps(&te.nsd, &ids, &seen.iter().collect::<Vec<_>>(), &taps, true)?;
        for (k, h) in taps.iter().zip(&hs) {
            worst = worst.max(note(&format!("hs_{k}"), &h.to_f32()?, &d.f32(&format!("hs_{k}"))?.1, log));
        }
        let cond = te.connect(&hs[1..], ids.len())?;
        worst = worst.max(note("cond", &cond, &d.f32("cond")?.1, log));
    }
    let mut refs: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    let mut i = 0;
    while d.has(&format!("ref_lat_{i}")) {
        let (s, v) = d.f32(&format!("ref_lat_{i}"))?;
        refs.push((v, s[1], s[2]));
        i += 1;
    }
    if stages.contains(&"venc") {
        let mut imgs = Vec::new();
        while d.has(&format!("ref_img_{}", imgs.len())) {
            imgs.push(picture(&d, &format!("ref_img_{}", imgs.len()))?);
        }
        let first = imgs.first().map(|p| (p.width, p.height));
        let mine = e.reference_latents(&imgs, first)?;
        for (k, ((m, _, _), (r, _, _))) in mine.iter().zip(&refs).enumerate() {
            worst = worst.max(note(&format!("ref_lat_{k}"), m, r, log));
        }
    }
    if stages.contains(&"dit") || stages.contains(&"steps") {
        let (_, cond) = d.f32("cond")?;
        let (_, tags) = d.f32("tags")?;
        let tags: Vec<i32> = tags.iter().map(|v| *v as i32).collect();
        let (_, lat_final) = d.f32("lat_final")?;
        let (sz, _) = d.f32("z")?;
        let (lh, lw) = if sz.len() == 3 { (sz[1], sz[2]) } else { return Err(Error("z.npy: [24, h, w]".into())) };
        let (_, sig) = d.f32("sigmas")?;
        worst = worst.max(note("sigmas", &SIGMAS, &sig, log));
        let seed: u64 = std::env::var("NS_PH3_CHECK_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let text = e.refine(&cond)?;
        if stages.contains(&"dit") {
            worst = worst.max(note("txt", &text.to_f32().map_err(h3e)?, &d.f32("txt")?.1, log));
            // the draws: the engine's torch generator against the reference's
            let mut g = nextsycl_h3::noise::Mt19937::new(seed);
            let (_, nv) = d.f32("noise_v")?;
            let (_, na) = d.f32("noise_a")?;
            let ncv: usize = refs.iter().map(|(_, h, w)| h / 2 * w / 2).sum();
            let mut mine_v: Vec<f32> = Vec::new();
            for (lat, h, w) in &refs {
                let n = g.randn(lat.len());
                let mixed: Vec<f32> = lat.iter().zip(&n).map(|(l, n)| 0.999 * l + (1.0 - 0.999f32) * n).collect();
                mine_v.extend(patchify(&mixed, 24, 1, *h, *w));
            }
            mine_v.extend(patchify(&g.randn(24 * lh * lw), 24, 1, lh, lw));
            worst = worst.max(note("noise_v", &mine_v, &nv, log));
            worst = worst.max(note("noise_a", &g.randn(na.len()), &na, log));
            // teacher-forced: each step's velocity from the reference's latents
            let mixed: Vec<RefIn> = {
                let rows = &nv[..ncv * 96];
                let mut out = Vec::new();
                let mut at = 0;
                for (_, h, w) in &refs {
                    let n = h / 2 * w / 2;
                    out.push(RefIn::Image { latent: unpatchify(&rows[at * 96..(at + n) * 96], 24, 1, *h, *w), h: *h, w: *w });
                    at += n;
                }
                out
            };
            let cnd = Conditions { refs: mixed, premixed: true, seed, ..Conditions::default() };
            let shape = Shape { t: 1, h: lh, w: lw, audio_t: 2 };
            let schedule = Schedule { shift_video: 1.0, shift_audio: 1.0, audio_scale: 1.0 };
            let mut dn = Denoiser::new(&e.blocks, &e.outer, e.refine(&cond)?, Some(tags.clone()), shape, schedule, &cnd).map_err(h3e)?;
            dn.lora = e.lora.as_ref();
            let (_, pos) = d.f32("pos")?;
            worst = worst.max(note("pos", &dn.layout.positions.iter().map(|v| *v as f32).collect::<Vec<_>>(), &pos, log));
            for (i, sigma) in SIGMAS[..SIGMAS.len() - 1].iter().enumerate() {
                if !d.has(&format!("lat_{i}")) {
                    continue;
                }
                let (_, lat) = d.f32(&format!("lat_{i}"))?;
                let (_, aud) = d.f32(&format!("aud_{i}"))?;
                let xv = unpatchify(&lat[ncv * 96..], 24, 1, lh, lw);
                let xa = unpack_audio(&aud, 32, 2);
                let (vv, va) = dn.net(&xv, &xa, *sigma, &mut |_| Ok(())).map_err(h3e)?;
                let v: Vec<f32> = patchify(&vv, 24, 1, lh, lw).into_iter().map(|x| -x).collect();
                let a: Vec<f32> = pack_audio(&va, 32, 2).into_iter().map(|x| -x).collect();
                let (_, rv) = d.f32(&format!("vel_{i}"))?;
                let (_, ra) = d.f32(&format!("avel_{i}"))?;
                worst = worst.max(note(&format!("vel_{i}"), &v, &rv[ncv * 96..], log));
                worst = worst.max(note(&format!("avel_{i}"), &a, &ra, log));
            }
        }
        if stages.contains(&"steps") {
            let t0 = std::time::Instant::now();
            let lat = e.sample(text, &tags, &refs, lh, lw, seed, &SIGMAS, &mut |_| Ok(()))?;
            log(format!("steps          {:.1} s", t0.elapsed().as_secs_f64()));
            worst = worst.max(note("lat_final", &lat, &lat_final, log));
        }
    }
    if stages.contains(&"silo") {
        // the silo's attention against the shared kernels' Sage form, on the same rows (an 8 MP picture's 8,318 tokens of
        // 56 heads of 128, the reference's text-to-image activations' scale: normal draws)
        use nextsycl_h3::device::Tensor;
        use nextsycl_h3::dtype::DType;
        use nextsycl_h3::ops::{self, Rows};
        let (t, h, dd) = (8318usize, e.blocks.cfg.heads, e.blocks.cfg.head_dim);
        let w = h * dd;
        let mut g = nextsycl_h3::noise::Mt19937::new(1);
        let vals = g.randn(t * 3 * w);
        let qkv = Tensor::from_bytes(&e.dev, DType::BF16, &[t, 3 * w], &nextsycl_h3::denoiser::bf16_bytes(&vals)).map_err(h3e)?;
        let part = |i: usize| Rows { t: &qkv, offset: i * w, stride: 3 * w, tokens: t, heads: h, dim: dd };
        let (a, b) = (Tensor::new(&e.dev, DType::BF16, &[t, w]).map_err(h3e)?, Tensor::new(&e.dev, DType::BF16, &[t, w]).map_err(h3e)?);
        ops::attention(part(0), part(1), part(2), &a).map_err(h3e)?;
        let f = e.blocks.attention.as_ref().ok_or_else(|| Error("no silo".into()))?;
        if !f(part(0), part(1), part(2), &b).map_err(h3e)? {
            return Err(Error("the silo did not take the shape".into()));
        }
        worst = worst.max(note("silo attn", &b.to_f32().map_err(h3e)?, &a.to_f32().map_err(h3e)?, log));
        ops::attention(part(0), part(1), part(2), &b).map_err(h3e)?;
        note("shared again", &b.to_f32().map_err(h3e)?, &a.to_f32().map_err(h3e)?, log);
    }
    if stages.contains(&"vae") {
        let (s, lat) = d.f32("lat_final")?;
        let (lh, lw) = (s[1], s[2]);
        let t0 = std::time::Instant::now();
        let raw = e.decoder.decode_raw(&lat, lh, lw, &mut |_, _| Ok(())).map_err(h3e)?;
        log(format!("vae            {:.2} s", t0.elapsed().as_secs_f64()));
        worst = worst.max(note("raw", &raw, &d.f32("raw")?.1, log));
        let mine = crate::vae::to_rgb8(&raw, lh * 16, lw * 16);
        let refpng = Picture::read_png(&dir.join("ref.png")).map_err(Error)?;
        let (same, off1) = (mine.iter().zip(&refpng.data).filter(|(a, b)| a == b).count(), mine.iter().zip(&refpng.data).filter(|(a, b)| a.abs_diff(**b) <= 1).count());
        log(format!("picture        {:.2} % of the bytes equal, {:.2} % within 1", 100.0 * same as f64 / mine.len() as f64, 100.0 * off1 as f64 / mine.len() as f64));
        let out = dir.join("nextsycl.png");
        Picture { width: (lw * 16) as u32, height: (lh * 16) as u32, channels: 3, data: mine }.write_png(&out, &[]).map_err(Error)?;
        log(format!("wrote {}", out.display()));
    }
    Ok(worst)
}
