//! The removers against reference/rembg/ref.py's dumps (on the CPU in float32: rembg 2.0.67 on onnxruntime for
//! IS-Net, BiRefNet's own code for BiRefNet): each picture's stages, the network fed the reference's own input, so a
//! stage's error is its own.
//!
//! ```text
//!   input  the picture -> the model's preparation                              against input.npy
//!   net    input.npy -> the prediction (IS-Net: the sigmoid; BiRefNet: logits)  against pred.npy / logits.npy
//!   mask   the whole way, the picture -> the 8-bit mask at its size             against mask.png (equal bytes)
//!   pp     the mask -> post_process_mask                                        against mask_pp.png
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_image::{Error, Picture, Result};
use nextsycl_remover::Remover;

use crate::{isnet::IsNet, post_process_mask, IsNetEngine};

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

fn grey(path: &Path) -> Result<Picture> {
    let p = Picture::read_png(path).map_err(Error)?;
    // a grey PNG reads as RGB: its first channel
    let c = p.channels as usize;
    Ok(Picture { width: p.width, height: p.height, channels: 1, data: p.data.chunks_exact(c).map(|v| v[0]).collect() })
}

/// (max |difference|, relative L2 error)
fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut mx, mut e2, mut w2) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        mx = mx.max(d.abs());
        e2 += d * d;
        w2 += (*y as f64) * (*y as f64);
    }
    (mx, (e2 / w2.max(1e-30)).sqrt())
}

/// (share of equal bytes, the largest difference)
fn bytes(a: &[u8], b: &[u8]) -> (f64, i32) {
    let eq = a.iter().zip(b).filter(|(x, y)| x == y).count();
    (eq as f64 / b.len().max(1) as f64, a.iter().zip(b).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0))
}

/// One picture's comparison
pub struct Row {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// the preparation: (max, relative)
    pub input: (f64, f64),
    /// the network on the reference's input: (max, relative)
    pub net: (f64, f64),
    pub net_seconds: f64,
    /// the mask the whole way: (share equal, largest difference)
    pub mask: (f64, i32),
    /// post_process_mask of it: (share equal, largest difference)
    pub pp: (f64, i32),
    /// the whole way (preparation, network, mask), and post_process_mask on top
    pub seconds: f64,
    pub pp_seconds: f64,
}

impl Row {
    pub fn line(&self) -> String {
        // pixels that differ, of the picture's
        let off = |share: f64| ((1.0 - share) * (self.width as f64 * self.height as f64)).round() as u64;
        format!("{:<13} {:>4}x{:<4}  input max {:.1e} rel {:.1e} | net max {:.2e} rel {:.1e} ({:.3} s) | mask {:.4}% equal ({} off), max {} | post-processed {:.4}% equal ({} off) | whole {:.3} s (+ pp {:.3} s)",
                self.name, self.width, self.height, self.input.0, self.input.1, self.net.0, self.net.1, self.net_seconds, self.mask.0 * 100.0, off(self.mask.0), self.mask.1,
                self.pp.0 * 100.0, off(self.pp.0), self.seconds, self.pp_seconds)
    }
}

/// The picture folders in `dir`, sorted
fn folders(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut subs: Vec<_> = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subs.sort();
    Ok(subs)
}

/// (share of equal bytes, the largest difference)
type Bytes = (f64, i32);

/// A mask's comparisons: (the mask, post_process_mask of it, its seconds)
fn masks(mine: &Picture, d: &Path) -> Result<(Bytes, Bytes, f64)> {
    let m = bytes(&mine.data, &grey(&d.join("mask.png"))?.data);
    let t = Instant::now();
    let pp = post_process_mask(mine);
    let secs = t.elapsed().as_secs_f64();
    Ok((m, bytes(&pp.data, &grey(&d.join("mask_pp.png"))?.data), secs))
}

/// IS-Net on every picture folder in `dir` (ref.py isnet's dumps)
pub fn isnet(net: &IsNet, dir: &Path, log: &mut dyn FnMut(String)) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for d in folders(dir)? {
        let pic = Picture::read_png(&d.join("picture.png")).map_err(Error)?;
        let want = npy(&d.join("input.npy"))?;
        let input = diff(&IsNetEngine::input(&pic), &want);
        let t0 = Instant::now();
        let pred = net.forward(&want)?;
        let net_seconds = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mine = IsNetEngine::mask_of(&net.forward(&IsNetEngine::input(&pic))?, pic.width, pic.height);
        let seconds = t1.elapsed().as_secs_f64();
        let (mask, pp, pp_seconds) = masks(&mine, &d)?;
        let r = Row { name: d.file_name().unwrap().to_string_lossy().into_owned(), width: pic.width, height: pic.height, input,
                      net: diff(&pred, &npy(&d.join("pred.npy"))?), net_seconds, mask, pp, seconds, pp_seconds };
        log(r.line());
        rows.push(r);
    }
    Ok(rows)
}

/// BiRefNet on every picture folder in `dir` (ref.py birefnet's dumps)
pub fn birefnet(r: &Remover, dir: &Path, log: &mut dyn FnMut(String)) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for d in folders(dir)? {
        let pic = Picture::read_png(&d.join("picture.png")).map_err(Error)?;
        let want = npy(&d.join("input.npy"))?;
        let input = diff(&Remover::input(&pic), &want);
        let t0 = Instant::now();
        let (logits, _) = r.logits(&want, false)?;
        let net_seconds = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mine = r.mask(&pic)?;
        let seconds = t1.elapsed().as_secs_f64();
        let (mask, pp, pp_seconds) = masks(&mine, &d)?;
        let row = Row { name: d.file_name().unwrap().to_string_lossy().into_owned(), width: pic.width, height: pic.height, input,
                        net: diff(&logits, &npy(&d.join("logits.npy"))?), net_seconds, mask, pp, seconds, pp_seconds };
        log(row.line());
        rows.push(row);
    }
    Ok(rows)
}
