//! The depth network against reference/depthanything/ref.py's dumps (transformers' DepthAnythingForDepthEstimation on
//! the CPU in float32), each stage fed the reference's own input so its error is its own:
//!
//! ```text
//!   input   the picture -> the preparation                               against input.npy
//!   emb     input.npy -> the embeddings                                  against emb.npy
//!   taps    ... the four taps through the final layer norm               against tap0..3.npy
//!   fused   ... the neck's last fusion                                   against fused3.npy
//!   pred    ... the head                                                 against pred.npy
//!   depth   the picture the whole way, resized to its own size          against depth.npy, and depth8 against depth8.png
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_image::{Error, Picture, Result};

use crate::net::DepthNet;
use crate::{depth8, prepare};

fn npy(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 12 || &b[1..6] != b"NUMPY" {
        return Err(Error(format!("{}: not a .npy file", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = b.get(at..at + hl).ok_or_else(|| Error(format!("{}: cut short", path.display())))?;
    if !String::from_utf8_lossy(head).contains("'<f4'") {
        return Err(Error(format!("{}: not float32", path.display())));
    }
    Ok(b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// (max |difference|, relative L2 error)
pub fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    if a.len() != b.len() {
        return (f64::INFINITY, f64::INFINITY);
    }
    let (mut mx, mut e2, mut w2) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        mx = mx.max(d.abs());
        e2 += d * d;
        w2 += (*y as f64) * (*y as f64);
    }
    (mx, (e2 / w2.max(1e-30)).sqrt())
}

pub struct Row {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub grid: (usize, usize),
    pub input: (f64, f64),
    pub emb: (f64, f64),
    pub taps: [(f64, f64); 4],
    pub fused: (f64, f64),
    pub pred: (f64, f64),
    pub depth: (f64, f64),
    /// depth8 against depth8.png: share of equal bytes, largest difference
    pub depth8: (f64, i32),
    pub seconds: f64,
}

impl Row {
    pub fn line(&self) -> String {
        let t = self.taps.iter().map(|t| t.1).fold(0.0, f64::max);
        format!("{:<16} {:>3}x{:<3} (net {}x{}) input max {:.1e} rel {:.1e} | emb rel {:.1e} | taps worst rel {:.1e} | fused rel {:.1e} | pred max {:.2e} rel {:.1e} | depth max {:.2e} rel {:.1e} | depth8 {:.4}% equal, max {} | {:.3} s",
                self.name, self.width, self.height, self.grid.1 * 14, self.grid.0 * 14, self.input.0, self.input.1, self.emb.1, t, self.fused.1, self.pred.0,
                self.pred.1, self.depth.0, self.depth.1, self.depth8.0 * 100.0, self.depth8.1, self.seconds)
    }
}

pub fn run(net: &DepthNet, dir: &Path, log: &mut dyn FnMut(String)) -> Result<Vec<Row>> {
    let mut subs: Vec<_> = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subs.sort();
    let mut rows = Vec::new();
    for d in subs {
        let pic = Picture::read_png(&d.join("picture.png")).map_err(Error)?;
        let want = npy(&d.join("input.npy"))?;
        let (mine, w, h) = prepare(&pic);
        let input = diff(&mine, &want);
        let (emb, taps, fused, pred) = net.stages(&want, h, w)?;
        let tap = |i: usize| -> Result<(f64, f64)> { Ok(diff(&taps[i], &npy(&d.join(format!("tap{i}.npy")))?)) };
        let t0 = Instant::now();
        let depth = net.depth_at(&mine, h, w, pic.width as usize, pic.height as usize)?;
        let seconds = t0.elapsed().as_secs_f64();
        let want_depth = npy(&d.join("depth.npy"))?;
        let g = Picture::read_png(&d.join("depth8.png")).map_err(Error)?;
        let gc = g.channels as usize;
        let want8: Vec<u8> = g.data.chunks_exact(gc).map(|v| v[0]).collect();
        let mine8 = &depth8(&[&depth])[0];
        // against the reference's own depth scaled the same way, so the 8-bit bytes differ only by the network
        let ref8 = &depth8(&[&want_depth])[0];
        debug_assert_eq!(ref8.len(), want8.len());
        let eq = mine8.iter().zip(&want8).filter(|(a, b)| a == b).count();
        let r = Row { name: d.file_name().unwrap().to_string_lossy().into_owned(), width: pic.width, height: pic.height, grid: (h / 14, w / 14), input,
                      emb: diff(&emb, &npy(&d.join("emb.npy"))?), taps: [tap(0)?, tap(1)?, tap(2)?, tap(3)?], fused: diff(&fused, &npy(&d.join("fused3.npy"))?),
                      pred: diff(&pred, &npy(&d.join("pred.npy"))?), depth: diff(&depth, &want_depth),
                      depth8: (eq as f64 / want8.len().max(1) as f64, mine8.iter().zip(&want8).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0)), seconds };
        log(r.line());
        rows.push(r);
    }
    Ok(rows)
}
