//! The engine against reference/seedvr2/ref.py's dumps (the official code on the CPU in float32, numz's fp16 weights
//! upcast): each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   resize    the clip's frames (a directory of PNGs)  -> input
//!   enc       input                                    -> cond (the latent mean times 0.9152)
//!   dit       noise, cond                              -> blk_N (the video tokens after block N), v; noise - v -> x0
//!   dec       x0                                       -> dec
//!   colorfix  dec, input                               -> out
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_image::{Error, Picture, Result};

use crate::vae::Vol;
use crate::{dit, ffi, SeedVr2};

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
    Ok((shape, b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
}

/// cosine, relative error (|a - b| / |b|) and the largest difference
pub fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut ab, mut aa, mut bb, mut dd, mut worst) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        dd += (x - y) * (x - y);
        worst = worst.max((x - y).abs());
    }
    (ab / (aa.sqrt() * bb.sqrt()).max(1e-30), (dd / bb.max(1e-30)).sqrt(), worst)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| dit::bf16(*x).to_le_bytes()).collect()
}

fn vol_f32(v: &Vol) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; v.t * v.h * v.w * v.c * 2];
    v.buf.read(0, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect())
}

/// Runs the stages; the worst relative error. `frames`: the clip's PNG frames (the resize stage), `variant` the DiT's
pub fn run(e: &SeedVr2, dir: &Path, frames: Option<&Path>, variant: &'static str, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let card = &e.cards[0];
    let (gpu, nsd) = (&card.gpu, &card.nsd);
    let k = ffi::api()?;
    let mut worst = 0f64;
    let mut report = |what: &str, a: &[f32], b: &[f32], log: &mut dyn FnMut(String)| -> Result<()> {
        if a.len() != b.len() {
            return Err(Error(format!("{what}: {} values against the reference's {}", a.len(), b.len())));
        }
        let (cos, rel, max) = compare(a, b);
        log(format!("{what:<10} cosine {cos:.6}  relative error {rel:.3e}  largest difference {max:.3e}"));
        worst = worst.max(rel);
        Ok(())
    };
    if stages.contains(&"bench") {
        // what the card reaches: a large bfloat16 GEMM (the XMX), a large copy on the card (its memory), pinned host
        // memory up (PCIe) - the roofline the profile's rows are read against
        let (pf, pb) = crate::prof::peaks(gpu);
        let n = 8192usize;
        let (a, b, c) = (DevBuf::new(gpu, n * n * 2)?, DevBuf::new(gpu, n * n * 2)?, DevBuf::new(gpu, n * n * 2)?);
        a.fill(0x3c)?;
        b.fill(0x3c)?;
        nsd.linear(a.ptr(), nextsycl_diffusion::kernels::Dt::BF16, n, n, b.ptr(), n, nextsycl_diffusion::kernels::none(), c.ptr(), nextsycl_diffusion::kernels::Dt::BF16)?;
        nsd.wait()?;
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            nsd.linear(a.ptr(), nextsycl_diffusion::kernels::Dt::BF16, n, n, b.ptr(), n, nextsycl_diffusion::kernels::none(), c.ptr(), nextsycl_diffusion::kernels::Dt::BF16)?;
        }
        nsd.wait()?;
        let tf = 5.0 * 2.0 * (n * n * n) as f64 / t0.elapsed().as_secs_f64() / 1e12;
        let big = 2usize << 30;
        let (x, y) = (DevBuf::new(gpu, big)?, DevBuf::new(gpu, big)?);
        y.copy_within(0, &x, 0, big)?;
        gpu.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            y.copy_within(0, &x, 0, big)?;
        }
        gpu.sync()?;
        let gb = 5.0 * 2.0 * big as f64 / t0.elapsed().as_secs_f64() / 1e9;
        let host = nextsycl_core::HostBuf::new(gpu, big)?;
        y.write(0, host.as_slice())?;
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            y.write(0, host.as_slice())?;
        }
        let up = 3.0 * big as f64 / t0.elapsed().as_secs_f64() / 1e9;
        log(format!("bench on {}: GEMM {n}^3 bf16 {tf:.1} TFLOPS ({:.0}% of {pf:.1}); copy on the card {gb:.0} GB/s read+write ({:.0}% of {pb:.0}); pinned host -> card {up:.1} GB/s",
                    gpu.name, 100.0 * tf / pf, 100.0 * gb / pb));
        if stages.len() == 1 {
            return Ok(0.0);
        }
    }
    let (ishape, input) = read_npy(&dir.join("input.npy"))?;
    let (t, h, w) = (ishape[0], ishape[1], ishape[2]);
    let put = |v: &[f32], t: usize, h: usize, w: usize, c: usize| -> Result<Vol> {
        let x = Vol::new(gpu, t, h, w, c)?;
        x.buf.write(0, &bf16_bytes(v))?;
        Ok(x)
    };
    if stages.contains(&"resize") {
        let dirp = frames.ok_or_else(|| Error("resize: the clip's frames (--frames DIR of PNGs)".into()))?;
        let mut names: Vec<_> = std::fs::read_dir(dirp).map_err(|e| Error(e.to_string()))?.flatten().map(|d| d.path())
            .filter(|p| p.extension().is_some_and(|x| x == "png")).collect();
        names.sort();
        let pics: Vec<Picture> = names.iter().take(t).map(|p| Picture::read_png(p).map_err(Error)).collect::<Result<_>>()?;
        let (sw, sh) = (pics[0].width, pics[0].height);
        // the reference resized to w x h (its --size: each side its own scale, no crop)
        let (rw, rh, cx, cy) = (w, h, 0, 0);
        let rgb: Vec<u8> = pics.iter().flat_map(|p| p.data.chunks_exact(p.channels as usize).flat_map(|c| [c[0], c[1], c[2]])).collect();
        let up = DevBuf::new(gpu, rgb.len())?;
        up.write(0, &rgb)?;
        let out = Vol::new(gpu, t, h, w, 3)?;
        // SAFETY: t frames of sh x sw x 3 bytes in, t x h x w x 3 out.
        ffi::check(unsafe {
            (k.resize)(gpu.raw(), up.ptr().cast(), t as i64, sh as i64, sw as i64, rh as i64, rw as i64, cy as i64, cx as i64, h as i64, w as i64, out.buf.ptr())
        }, "resize")?;
        nsd.wait()?;
        report("resize", &vol_f32(&out)?, &input, log)?;
    }
    if stages.contains(&"enc") {
        let (_, cond) = read_npy(&dir.join("cond.npy"))?;
        let t0 = std::time::Instant::now();
        let mine = card.vae.encode(nsd, &put(&input, t, h, w, 3)?)?;
        log(format!("encode {} frames of {w}x{h}: {:.2} s", t, t0.elapsed().as_secs_f64()));
        report("enc", &mine, &cond, log)?;
    }
    let (lshape, cond) = read_npy(&dir.join("cond.npy"))?;
    let (lt, lh, lw) = (lshape[0], lshape[1], lshape[2]);
    if stages.contains(&"dit") {
        let (_, noise) = read_npy(&dir.join("noise.npy"))?;
        let d = e.dit_for(card, variant, log)?;
        let s = d.shape(lt, lh, lw)?;
        let mut taps = Vec::new();
        for i in 0..64 {
            let p = dir.join(format!("blk_{i}.npy"));
            if p.is_file() {
                taps.push((i, read_npy(&p)?.1));
            }
        }
        let t0 = std::time::Instant::now();
        let mut lines = Vec::new();
        let v = d.forward(nsd, &s, &noise, &cond, &mut |i, xv, _| {
            if let Some((_, r)) = taps.iter().find(|(j, _)| *j == i) {
                let mine = xv.to_f32()?;
                lines.push((format!("blk_{i}"), compare(&mine, r)));
            }
            Ok(())
        })?;
        log(format!("dit {lt}x{lh}x{lw}: {:.2} s", t0.elapsed().as_secs_f64()));
        for (what, (cos, rel, max)) in lines {
            log(format!("{what:<10} cosine {cos:.6}  relative error {rel:.3e}  largest difference {max:.3e}"));
        }
        let (_, rv) = read_npy(&dir.join("v.npy"))?;
        report("v", &v, &rv, log)?;
        let (_, rx) = read_npy(&dir.join("x0.npy"))?;
        let x0: Vec<f32> = noise.iter().zip(&v).map(|(a, b)| a - b).collect();
        report("x0", &x0, &rx, log)?;
    }
    if stages.contains(&"dec") {
        let (_, x0) = read_npy(&dir.join("x0.npy"))?;
        let (_, rd) = read_npy(&dir.join("dec.npy"))?;
        let mut mine = Vec::new();
        let t0 = std::time::Instant::now();
        card.vae.decode(nsd, &x0, lt, lh, lw, &mut |f| {
            mine.extend(vol_f32(&f)?);
            Ok(())
        })?;
        log(format!("decode {lt} latent frames: {:.2} s", t0.elapsed().as_secs_f64()));
        mine.truncate(rd.len());
        report("dec", &mine, &rd, log)?;
    }
    if stages.contains(&"colorfix") {
        let (_, rd) = read_npy(&dir.join("dec.npy"))?;
        let (_, ro) = read_npy(&dir.join("out.npy"))?;
        let n = rd.len() / (h * w * 3);
        let (dv, iv) = (put(&rd, n, h, w, 3)?, put(&input[..n * h * w * 3], n, h, w, 3)?);
        let out = Vol::new(gpu, n, h, w, 3)?;
        let tmp = DevBuf::f32(gpu, 2 * n * h * w * 3)?;
        // SAFETY: n frames of h x w x 3 in each and out, the scratch twice that in floats.
        ffi::check(unsafe { (k.colorfix)(gpu.raw(), dv.buf.ptr(), iv.buf.ptr(), n as i64, h as i64, w as i64, tmp.fp(), out.buf.ptr()) }, "colour fix")?;
        nsd.wait()?;
        report("colorfix", &vol_f32(&out)?, &ro, log)?;
    }
    Ok(worst)
}
