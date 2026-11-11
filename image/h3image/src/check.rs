//! The engine against reference/h3-image/ref.py's dumps (ComfyUI's model written out in torch, on the CPU in float32,
//! from the bf16 checkpoint): each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   te     the prompt (prompt.txt) through H3's text encoder -> cond.npy, tags.npy: the reference's input (the text
//!          encoder is the video engine's, checked in its port)
//!   dit    the refined text (txt), the positions (pos), the noise from the seed (noise), then each step's velocity
//!          from the reference's own latents (lat_I -> vel_I, the stream after the blocks at step 0 -> blk_49)
//!   steps  the reference's noise through all the steps -> lat_final
//!   vae    lat_final -> raw (the decoder's output) and the picture (nextsycl.png beside the dumps)
//! ```

use std::path::Path;

use nextsycl_image::{Error, Picture, Result};

use crate::{h3e, net, H3Image};

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

/// A .npy file of float32 (`descr` '<f4') or int32 ('<i4') values
pub fn write_npy(path: &Path, shape: &[usize], data: &[u8], descr: &str) -> Result<()> {
    let dims = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut head = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {dims}, }}");
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(head.len() as u16).to_le_bytes());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(data);
    std::fs::write(path, out).map_err(|e| Error(format!("{}: {e}", path.display())))
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

/// Runs `stages` (te, dit, steps, vae) against the dumps in `dir`; the worst relative error
pub fn run(e: &H3Image, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let d = Dump(dir);
    let mut worst = 0f64;
    let seed: u64 = std::env::var("NS_H3I_CHECK_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    if stages.contains(&"te") {
        let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|er| Error(format!("{}: {er}", dir.join("prompt.txt").display())))?;
        let c = e.encode(prompt.trim_end_matches('\n'), &[], log)?;
        let t = c.tags.len();
        write_npy(&dir.join("cond.npy"), &[t, c.rows.len() / t], &c.rows.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), "<f4")?;
        write_npy(&dir.join("tags.npy"), &[t], &c.tags.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), "<i4")?;
        log(format!("te             {t} tokens in {:.1} s -> cond.npy, tags.npy", c.seconds));
    }
    if stages.contains(&"dit") || stages.contains(&"steps") {
        let (_, cond) = d.f32("cond")?;
        let tags: Vec<i32> = d.f32("tags")?.1.iter().map(|v| *v as i32).collect();
        let (sn, noise) = d.f32("noise")?;
        let (lh, lw) = (sn[1], sn[2]);
        let (_, sig) = d.f32("sigmas")?;
        let text = e.refine(&cond)?;
        if stages.contains(&"dit") {
            worst = worst.max(note("txt", &text.to_f32().map_err(h3e)?, &d.f32("txt")?.1, log));
            worst = worst.max(note("noise", &nextsycl_h3::noise::Mt19937::new(seed).randn(24 * lh * lw), &noise, log));
            let mut run = net::Run::new(&e.blocks, &e.outer, e.refine(&cond)?, Some(tags.clone()), lh, lw, &[], seed).map_err(h3e)?;
            let (_, pos) = d.f32("pos")?;
            worst = worst.max(note("pos", &run.layout.positions.iter().map(|v| *v as f32).collect::<Vec<_>>(), &pos, log));
            if d.has("rows") {
                let rows: Vec<f32> = run.timesteps(sig[0]).rows.iter().map(|v| *v as f32).collect();
                worst = worst.max(note("rows", &rows, &d.f32("rows")?.1, log));
            }
            for i in 0..sig.len() - 1 {
                if !d.has(&format!("lat_{i}")) {
                    continue;
                }
                let (_, lat) = d.f32(&format!("lat_{i}"))?;
                let t0 = std::time::Instant::now();
                let v = run.net(&lat, sig[i], sig[i + 1], &mut |_| Ok(())).map_err(h3e)?;
                let secs = t0.elapsed().as_secs_f64();
                if d.has(&format!("head_{i}")) {
                    let want = d.f32(&format!("head_{i}"))?.1;
                    let got = net::head_weights((sig[i] * 1000.0) / 1000.0, sig[i + 1]);
                    worst = worst.max(note(&format!("head_{i}"), &got, &want, log));
                }
                if i == 0 && d.has("blk_49") {
                    let (_, blk) = d.f32("blk_49")?;
                    note("blk_49", &run.stream().map_err(h3e)?, &blk, log);
                }
                worst = worst.max(note(&format!("vel_{i} ({secs:.1} s)"), &v, &d.f32(&format!("vel_{i}"))?.1, log));
            }
        }
        if stages.contains(&"steps") {
            let t0 = std::time::Instant::now();
            let lat = e.sample(text, &tags, &[], lh, lw, seed, &sig, &mut |_| Ok(()))?;
            log(format!("steps          {:.1} s", t0.elapsed().as_secs_f64()));
            worst = worst.max(note("lat_final", &lat, &d.f32("lat_final")?.1, log));
            let out = dir.join("lat_engine.npy");
            write_npy(&out, &[24, lh, lw], &lat.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), "<f4")?;
        }
    }
    if stages.contains(&"expand") {
        worst = worst.max(expansions(e, log)?);
    }
    if stages.contains(&"vae") {
        let (s, lat) = d.f32("lat_final")?;
        let (lh, lw) = (s[1], s[2]);
        let t0 = std::time::Instant::now();
        let raw = e.decoder.decode_raw(&lat, lh, lw, &mut |_, _| Ok(())).map_err(h3e)?;
        log(format!("vae            {:.2} s", t0.elapsed().as_secs_f64()));
        worst = worst.max(note("raw", &raw, &d.f32("raw")?.1, log));
        let mine = crate::vae::to_rgb8(&raw, lh * 16, lw * 16);
        if d.has("rgb") {
            let (_, rgb) = d.f32("rgb")?;
            let (h, w) = (lh * 16, lw * 16);
            let theirs: Vec<u8> = (0..h * w * 3).map(|i| (rgb[(i % 3) * h * w + i / 3] * 255.0).round() as u8).collect();
            let (same, off1) = (mine.iter().zip(&theirs).filter(|(a, b)| a == b).count(), mine.iter().zip(&theirs).filter(|(a, b)| a.abs_diff(**b) <= 1).count());
            log(format!("picture        {:.2} % of the bytes equal, {:.2} % within 1", 100.0 * same as f64 / mine.len() as f64, 100.0 * off1 as f64 / mine.len() as f64));
        }
        let out = dir.join("nextsycl.png");
        Picture { width: (lw * 16) as u32, height: (lh * 16) as u32, channels: 3, data: mine }.write_png(&out, &[]).map_err(Error)?;
        log(format!("wrote {}", out.display()));
    }
    Ok(worst)
}

/// The bfloat16 values of two expansions that differ, by bit pattern (0: the same bits)
pub fn differing(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count() + a.len().abs_diff(b.len())
}

/// The `expand` stage (no dumps): the silo's expansion of the denoiser's own block matrices against the shared
/// kernels' (a Q3_K matrix against its Q6_K form, read again from the file and transcoded on the host), bit for bit,
/// for blocks 0, 10, 20, 30, 40 and the last; the share of differing values (0 when exact)
fn expansions(e: &H3Image, log: &mut dyn FnMut(String)) -> Result<f64> {
    use nextsycl_h3::device::Tensor;
    use nextsycl_h3::dit::Weight;
    use nextsycl_h3::dtype::DType;
    use nextsycl_h3::gguf::GType;
    use nextsycl_h3::safetensors::Checkpoint;
    let Some(s) = crate::ffi::silo().map_err(|er| Error(er.0))? else {
        log("expand         no silo: nothing to compare".into());
        return Ok(0.0);
    };
    let Some(wbuf) = e.blocks.expansion_buffer() else {
        log("expand         an int8 checkpoint: nothing expanded".into());
        return Ok(0.0);
    };
    let other = Tensor::new(&e.dev, DType::BF16, &wbuf.shape).map_err(h3e)?;
    let file = Checkpoint::open(&e.ck.path).map_err(h3e)?;
    let n_blocks = e.blocks.blocks.len();
    let (mut total, mut diff) = (0usize, 0usize);
    for bi in [0, 10, 20, 30, 40, n_blocks - 1].into_iter().filter(|b| *b < n_blocks) {
        for (j, w) in e.blocks.blocks[bi].linears().into_iter().enumerate() {
            let Weight::K(l) = w else { continue };
            if !crate::ffi::expands(s, l.kind) {
                log(format!("expand         {:?}: not one the silo reads", l.kind));
                return Ok(0.0);
            }
            let n = l.n * l.k;
            crate::ffi::expand_into(e.gpu(), s, &l.blocks, l.kind, n, wbuf).map_err(h3e)?;
            if l.kind == GType::Q3K {
                let name = format!("blocks.{bi}.{}", ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"][j]) + ".weight";
                let q6 = Tensor::from_bytes(&e.dev, DType::U8, &[GType::Q6K.bytes(n)], &file.read(&name).map_err(h3e)?).map_err(h3e)?;
                nextsycl_h3::ops::expand_into(&q6, GType::Q6K, n, &other).map_err(h3e)?;
            } else {
                nextsycl_h3::ops::expand_into(&l.blocks, l.kind, n, &other).map_err(h3e)?;
            }
            let (a, b) = (wbuf.to_f32().map_err(h3e)?, other.to_f32().map_err(h3e)?);
            let dn = differing(&a[..n], &b[..n]);
            log(format!("expand         block {bi:2} matrix {j} ({:?}, {} x {}): {dn} of {n} values differ", l.kind, l.n, l.k));
            total += n;
            diff += dn;
        }
    }
    log(format!("expand         {diff} of {total} values differ"));
    Ok(diff as f64 / total.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differing_counts_bit_patterns() {
        assert_eq!(differing(&[1.0, -0.0, f32::NAN], &[1.0, 0.0, f32::NAN]), 1);
        assert_eq!(differing(&[1.0, 2.0], &[1.0]), 1);
        assert_eq!(differing(&[], &[]), 0);
    }

    #[test]
    fn npy_round_trip() {
        let p = std::env::temp_dir().join(format!("h3i-npy-{}.npy", std::process::id()));
        let v = [1.5f32, -2.0, 3.25, 0.0, 7.0, 8.0];
        write_npy(&p, &[2, 3], &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), "<f4").unwrap();
        let (s, got) = read_npy(&p).unwrap();
        assert_eq!((s, got), (vec![2, 3], v.iter().map(|x| *x as f64).collect::<Vec<_>>()));
        write_npy(&p, &[3], &[1i32, -4, 9].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), "<i4").unwrap();
        assert_eq!(read_npy(&p).unwrap(), (vec![3], vec![1.0, -4.0, 9.0]));
        let _ = std::fs::remove_file(p);
    }
}
