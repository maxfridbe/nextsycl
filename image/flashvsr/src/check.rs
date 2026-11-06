//! The engine against reference/flashvsr/ref.py's dumps (the official pipeline on the CPU in float32): the same
//! prepared frames (lq_u8) and the same noise, the stages compared as they come.
//!
//! ```text
//!   c0, c1  lq (the LQ projection's tokens), b0 / b15 / b29 (the residual stream after those blocks),
//!           lat (noise - the DiT's output)
//!   frames  dec (the decoder's frames, before the colour fix), out (the bytes, after it: PSNR)
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_image::{Error, Picture, Result};

use crate::resize::Geometry;
use crate::{FlashVsr, Hooks, Settings};

// (a port of nextcuda's check, reading the same dumps)

/// A .npy's shape and values (float32 or uint8)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if head.contains("'|u1'") {
        d.iter().map(|x| *x as f32).collect()
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

fn halves(v: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    v.read(0, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
}

fn floats(v: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 4];
    v.read(0, &mut raw)?;
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// window order -> (frame, y, x) order, rows of c
fn natural(v: &[f32], f: usize, hp: usize, wp: usize, c: usize) -> Vec<f32> {
    let nbw = wp / 8;
    let nb = (hp / 8) * nbw;
    let mut out = vec![0f32; v.len()];
    for t in 0..f {
        for y in 0..hp {
            for x in 0..wp {
                let w = (((t / 2) * nb + (y / 8) * nbw + x / 8) << 7) + (t & 1) * 64 + (y & 7) * 8 + (x & 7);
                let n = (t * hp + y) * wp + x;
                out[n * c..(n + 1) * c].copy_from_slice(&v[w * c..(w + 1) * c]);
            }
        }
    }
    out
}

struct Check<'a> {
    dir: &'a Path,
    noise: Vec<f32>,
    nshape: Vec<usize>,
    hp: usize,
    wp: usize,
    dec: Vec<f32>,
    dshape: Vec<usize>,
    worst: f64,
    dec_err: (f64, f64, usize),
    log: &'a mut dyn FnMut(String),
}

impl Check<'_> {
    fn note(&mut self, name: &str, got: &[f32], want: &[f32]) {
        let (rel, mx, cos) = compare(got, want);
        (self.log)(format!("{name:<10} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values)", want.len().min(got.len())));
        self.worst = self.worst.max(rel);
    }
}

impl Hooks for Check<'_> {
    fn noise(&mut self, t: usize, h8: usize, w8: usize) -> Option<Vec<f32>> {
        let (tt, hh, ww) = (self.nshape[1], self.nshape[2], self.nshape[3]);
        if hh != h8 || ww != w8 || t >= tt {
            return None;
        }
        let mut v = vec![0f32; 16 * h8 * w8];
        for c in 0..16 {
            v[c * h8 * w8..(c + 1) * h8 * w8].copy_from_slice(&self.noise[(c * tt + t) * h8 * w8..(c * tt + t + 1) * h8 * w8]);
        }
        Some(v)
    }

    fn tensor(&mut self, c: usize, name: &str, v: &DevBuf, rows: usize) -> Result<()> {
        if c > 1 {
            return Ok(());
        }
        let f = if c == 0 { 6 } else { 2 };
        match name {
            "lq" => {
                let got = floats(v, rows * 1536)?;
                let (_, want) = read_npy(&self.dir.join(format!("c{c}_lq.npy")))?;
                self.note(&format!("c{c} lq"), &got, &want);
            }
            "lat" => {
                let got = floats(v, rows)?;
                let (_, x) = read_npy(&self.dir.join(format!("c{c}_x.npy")))?;
                let (_, p) = read_npy(&self.dir.join(format!("c{c}_pred.npy")))?;
                let want: Vec<f32> = x.iter().zip(&p).map(|(a, b)| a - b).collect();
                self.note(&format!("c{c} lat"), &got, &want);
            }
            b => {
                let path = self.dir.join(format!("c{c}_{b}.npy"));
                if path.exists() {
                    let got = natural(&floats(v, rows * 1536)?, f, self.hp, self.wp, 1536);
                    let (_, want) = read_npy(&path)?;
                    self.note(&format!("c{c} {b}"), &got, &want);
                }
            }
        }
        Ok(())
    }

    fn active(&self) -> bool {
        true
    }

    fn decoded(&mut self, i: usize, v: &DevBuf) -> Result<()> {
        let (t, h, w) = (self.dshape[1], self.dshape[2], self.dshape[3]);
        if i >= t {
            return Ok(());
        }
        let got = halves(v, h * w * 3)?;
        // reference [3, T, H, W] in [-1, 1]; ours [H, W, 3] in [0, 1]
        let mut want = vec![0f32; h * w * 3];
        for c in 0..3 {
            for p in 0..h * w {
                want[p * 3 + c] = (self.dec[(c * t + i) * h * w + p] + 1.0) / 2.0;
            }
        }
        let (rel, mx, _) = compare(&got, &want);
        self.dec_err.0 = self.dec_err.0.max(rel);
        self.dec_err.1 = self.dec_err.1.max(mx);
        self.dec_err.2 += 1;
        if i.is_multiple_of(8) {
            (self.log)(format!("frame {i:<4} decoded rel {rel:.2e}  max {mx:.3e}"));
        }
        Ok(())
    }
}

/// Runs the dumps in `dir` through the engine; the worst relative error of the stages, and each frame's PSNR against
/// the reference's bytes (logged). Our frames are written to dir/ours_NNN.png (the dump's folder: keep it off git - a
/// film's frames).
pub fn run(e: &FlashVsr, dir: &Path, log: &mut dyn FnMut(String)) -> Result<f64> {
    let (fshape, lq) = read_npy(&dir.join("lq_u8.npy"))?;
    let (nshape, noise) = read_npy(&dir.join("noise.npy"))?;
    // the decoder's frames come last in the reference's run: a check of the DiT can go before they are there
    let (dshape, dec) = read_npy(&dir.join("dec.npy")).unwrap_or((vec![3, 0, 0, 0], Vec::new()));
    let (oshape, out) = read_npy(&dir.join("out.npy")).unwrap_or((vec![3, 0, 0, 0], Vec::new()));
    let (nf, th, tw) = (fshape[0], fshape[1], fshape[2]);
    let geo = Geometry { w: tw, h: th, sw: tw, sh: th, tw, th, left: 0, top: 0 };
    // the reference's real frames: its output's count less nothing (it pads past the end as we do)
    let real = if oshape[1] > 0 { oshape[1].min(nf) } else { nf - 4 };
    log(format!("{nf} prepared frames {tw}x{th}, {real} out; noise {nshape:?}"));
    let frames: Vec<Vec<u8>> = (0..nf).map(|i| lq[i * th * tw * 3..(i + 1) * th * tw * 3].iter().map(|v| *v as u8).collect()).collect();
    let mut it = frames.into_iter().take(real);
    let mut ours: Vec<Picture> = Vec::new();
    let mut ch = Check { dir, noise, nshape, hp: th / 16, wp: tw / 16, dec, dshape, worst: 0.0, dec_err: (0.0, 0.0, 0), log };
    // the reference ran FlashVSR's own sparse_ratio, 2.0
    let set = Settings { sparse_ratio: 2.0, ..Settings::default() };
    e.run(geo, &set, true, &mut || Ok(it.next()), &mut |p| {
        ours.push(p);
        Ok(())
    }, &mut ch)?;
    let (worst, de) = (ch.worst, ch.dec_err);
    log(format!("decoded frames: {} compared, worst rel {:.2e}, worst max {:.3e}", de.2, de.0, de.1));
    let (t, h, w) = (oshape[1], oshape[2], oshape[3]);
    let mut psnrs = Vec::new();
    for (i, p) in ours.iter().enumerate().take(t) {
        let mut se = 0f64;
        for c in 0..3 {
            for q in 0..h * w {
                let r = (((out[(c * t + i) * h * w + q] + 1.0) * 127.5).clamp(0.0, 255.0)) as u8 as f64;
                let d = p.data[q * 3 + c] as f64 - r;
                se += d * d;
            }
        }
        let mse = se / (h * w * 3) as f64;
        psnrs.push(if mse == 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() });
        let _ = p.write_png(&dir.join(format!("ours_{i:03}.png")), &[]);
    }
    let mn = psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = psnrs.iter().sum::<f64>() / psnrs.len().max(1) as f64;
    log(format!("{} frames out: PSNR against the reference's bytes mean {avg:.2} dB, worst {mn:.2} dB", ours.len()));
    Ok(worst.max(de.0))
}
