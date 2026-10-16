//! Stage by stage against the reference (reference/animateanymesh/ref.py: the official modules in float32 on the
//! CPU). Each stage starts from the reference's own input, so a difference shows where it is born:
//!
//! ```text
//!   prep    the OBJ (--mesh)              -> verts.npy, faces.npy (merged, normalized, Blender's frame)
//!   embed   verts + faces                 -> embed0.npy, embed_nb.npy (the neighbour attention)
//!   fps     embed_nb                      -> fps_raw.npy (farthest points: how many agree)
//!   enc     embed_nb + fps                -> f0.npy (the rest latents, rescaled)
//!   clip    the prompt (--prompt), tokens -> tokens.npy, clip.npy
//!   dit     dit_x + dit_t + tokens        -> dit_out.npy (one forward of the guided pair)
//!   ode     noise + f0 + tokens           -> traj_S.npy, samples.npy (the whole flow)
//!   dec     samples + embed_nb            -> out.npy (16 frames)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::{dit, prep, vae, AnimateAnyMesh};

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

pub fn line(what: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    if got.len() != want.len() {
        log(format!("{what:<12} {} values here, {} in the reference", got.len(), want.len()));
        return f64::INFINITY;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<12} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.7}  ({} values)", want.len()));
    rel
}

fn ints(v: &[f32]) -> Vec<i32> {
    v.iter().map(|x| *x as i32).collect()
}

/// The reference's prepared mesh (verts.npy, faces.npy)
fn prepared(dir: &Path) -> Result<prep::Prepared> {
    let (_, v) = read_npy(&dir.join("verts.npy"))?;
    let (_, f) = read_npy(&dir.join("faces.npy"))?;
    let verts: Vec<[f32; 3]> = v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let faces = f.chunks_exact(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect();
    Ok(prep::Prepared { map: (0..verts.len() as u32).collect(), verts, faces, center: [0.0; 3], den: 1.0 })
}

/// The reference's shape: its embeddings after the neighbour attention
fn ref_shape(m: &AnimateAnyMesh, dir: &Path) -> Result<vae::Shape> {
    let (s, e) = read_npy(&dir.join("embed_nb.npy"))?;
    let (_, v) = read_npy(&dir.join("verts.npy"))?;
    Ok(vae::Shape { n: v.len() / 3, l: s[0], embed: DevBuf::from_f32(&m.ops.gpu, &e)? })
}

/// Every stage of `stages` against the dumps in `dir`; the worst relative error
pub fn run(m: &AnimateAnyMesh, dir: &Path, stages: &[&str], mesh: Option<&Path>, prompt: Option<&str>, log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let mut note = |r: f64| worst = worst.max(if r.is_nan() { f64::INFINITY } else { r });
    if stages.contains(&"prep") {
        let path = mesh.ok_or_else(|| Error("the prep stage reads the mesh: --mesh FILE.obj".into()))?;
        let text = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let mesh = nextsycl_model3d::read_mesh(&text).map_err(Error)?;
        let p = prep::prepare(&mesh);
        let (_, v) = read_npy(&dir.join("verts.npy"))?;
        let (_, f) = read_npy(&dir.join("faces.npy"))?;
        let got: Vec<f32> = p.verts.iter().flatten().copied().collect();
        note(line("prep verts", &got, &v, log));
        let mut want: Vec<[u32; 3]> = f.chunks_exact(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect();
        want.sort();
        log(format!("prep faces   {} here, {} in the reference, {}", p.faces.len(), want.len(), if p.faces == want { "the same" } else { "DIFFERENT" }));
        if p.faces != want {
            note(f64::INFINITY);
        }
    }
    if stages.contains(&"embed") {
        let p = prepared(dir)?;
        let (sh, e0) = m.shape(&p)?;
        let (_, want0) = read_npy(&dir.join("embed0.npy"))?;
        note(line("embed0", &e0.to_f32()?, &want0, log));
        let (_, want) = read_npy(&dir.join("embed_nb.npy"))?;
        note(line("embed_nb", &sh.embed.to_f32()?, &want, log));
    }
    if stages.contains(&"fps") {
        let sh = ref_shape(m, dir)?;
        let (_, want) = read_npy(&dir.join("fps_raw.npy"))?;
        let want = ints(&want);
        let t0 = Instant::now();
        let (raw, _) = m.sample_tokens(&sh, want.len(), 0)?;
        let same = raw.iter().zip(&want).take_while(|(a, b)| a == b).count();
        log(format!("fps          the first {same} of {} indices agree ({:.2} s){}", want.len(), t0.elapsed().as_secs_f64(),
                    if same < want.len() { format!("; then {} here, {} there", raw[same], want[same]) } else { String::new() }));
        if same < want.len() {
            note(1.0);
        }
    }
    if stages.contains(&"enc") {
        let sh = ref_shape(m, dir)?;
        let (_, idx) = read_npy(&dir.join("fps.npy"))?;
        let f0 = m.f0(&sh, &ints(&idx))?;
        let (_, want) = read_npy(&dir.join("f0.npy"))?;
        note(line("enc f0", &f0.to_f32()?, &want, log));
    }
    let (_, tokens) = read_npy(&dir.join("tokens.npy"))?;
    let ids: Vec<u32> = tokens.iter().map(|x| *x as u32).collect();
    if stages.contains(&"clip") {
        if let Some(p) = prompt {
            let got = m.tok.encode(p);
            let same = got == ids;
            log(format!("tokens       {}", if same { "the same".to_string() } else { format!("DIFFERENT: {got:?} here, {ids:?} there") }));
            if !same {
                note(f64::INFINITY);
            }
        }
        let h = m.clip.encode(&m.ops, &ids)?;
        let (_, want) = read_npy(&dir.join("clip.npy"))?;
        note(line("clip", &h.to_f32()?, &want, log));
    }
    let text = || -> Result<dit::Text> {
        let h = m.clip.encode(&m.ops, &ids)?;
        m.dit.text(&m.ops, &h)
    };
    if stages.contains(&"dit") {
        let text = text()?;
        let (s, x) = read_npy(&dir.join("dit_x.npy"))?;
        let (_, t) = read_npy(&dir.join("dit_t.npy"))?;
        let k = s[1];
        let xb = DevBuf::from_f32(&m.ops.gpu, &x)?;
        let w = m.dit.work(&m.ops, k)?;
        let out = m.ops.buf(2 * k * dit::CHANNELS)?;
        m.dit.forward(&m.ops, &w, &text, &xb, t[0], &out)?;
        let (_, want) = read_npy(&dir.join("dit_out.npy"))?;
        note(line("dit", &out.to_f32()?, &want, log));
    }
    if stages.contains(&"ode") {
        let text = text()?;
        let (_, noise) = read_npy(&dir.join("noise.npy"))?;
        let (_, f0) = read_npy(&dir.join("f0.npy"))?;
        let (_, times) = read_npy(&dir.join("times.npy"))?;
        let steps = times.len() - 1;
        let grid = crate::linspace(times.len());
        log(format!("ode grid     {}", if grid == times { "the same".to_string() } else { "DIFFERENT".to_string() }));
        let f0b = DevBuf::from_f32(&m.ops.gpu, &f0)?;
        let t0 = Instant::now();
        let mut lines = Vec::new();
        let y = m.sample(&text, &f0b, &noise, steps, 3.0, &mut |_| Ok(()), t0, &mut |s, y| {
            let p = dir.join(format!("traj_{s}.npy"));
            if p.exists() {
                let (_, want) = read_npy(&p)?;
                let got = y.to_f32()?;
                let (rel, mx, cos) = compare(&got, &want);
                lines.push((s, rel, mx, cos));
            }
            Ok(())
        })?;
        for (s, rel, mx, cos) in lines {
            log(format!("{:<12} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.7}", format!("ode step {s}")));
        }
        let f = m.factors;
        let mut got = y.to_f32()?;
        for (r, f0r) in got.chunks_exact_mut(dit::CHANNELS).zip(f0.chunks_exact(vae::LATENT)) {
            for c in 0..dit::CHANNELS {
                r[c] = if c < vae::LATENT { f0r[c] * f.f0_std + f.f0_mean } else { r[c] * f.ft_std + f.ft_mean };
            }
        }
        let (_, want) = read_npy(&dir.join("samples.npy"))?;
        log(format!("ode          {steps} steps in {:.2} s", t0.elapsed().as_secs_f64()));
        note(line("samples", &got, &want, log));
    }
    if stages.contains(&"dec") {
        let sh = ref_shape(m, dir)?;
        let (_, s) = read_npy(&dir.join("samples.npy"))?;
        // samples.npy is rescaled back; the engine's decode takes the flow's scale
        let f = m.factors;
        let flow: Vec<f32> = s.chunks_exact(dit::CHANNELS).flat_map(|r| {
            (0..dit::CHANNELS).map(move |c| if c < vae::LATENT { (r[c] - f.f0_mean) / f.f0_std } else { (r[c] - f.ft_mean) / f.ft_std })
        }).collect();
        let off = m.decode(&sh, &flow)?;
        let (_, v) = read_npy(&dir.join("verts.npy"))?;
        let n = v.len() / 3;
        let mut got = vec![0f32; vae::FRAMES * n * 3];
        for t in 0..vae::FRAMES {
            for i in 0..n {
                for c in 0..3 {
                    got[(t * n + i) * 3 + c] = v[i * 3 + c] + off[i * 3 * vae::FRAMES + 3 * t + c];
                }
            }
        }
        let (_, want) = read_npy(&dir.join("out.npy"))?;
        note(line("dec frames", &got, &want, log));
    }
    Ok(worst)
}
