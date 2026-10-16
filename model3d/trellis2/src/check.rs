//! Stage by stage against the reference (reference/trellis2/ref.py: TRELLIS.2's own modules on the same files, float32 on
//! the CPU): each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   pre      the picture (--picture)              -> pre.png (byte for byte)
//!   remover  rm_in.npy                            -> rm_x1, rm_x4 (the encoder), rm_logits (and rm_in from the picture)
//!   dino     pre.png                              -> dino512_in, cond512 (cond1024 when dumped)
//!   ss       ss_noise, ss_t0, cond512             -> ss_in0, ss_blk0, ss_blk29, ss_call0 (and ss_call1, uncond); the
//!                                                    12 steps from the noise -> ss_x<i>, ss_z
//!   ssdec    ss_z                                 -> ss_logits, ss_coords
//!   shape    shape_noise, ss_coords, cond512      -> shape_in0 ... shape_call0, the steps -> shape_x<i>, shape_slat
//!   tex      tex_noise, shape_slat                -> tex_call0, the steps -> tex_slat
//!   sdec     shape_slat at ss_coords              -> sdec_sub<i> (the subdivision logits and the voxels kept),
//!                                                    sdec_up<i> (matched by coordinate), sdec_h, mesh_v / mesh_f counts
//!   tdec     tex_slat with the shape's tree       -> tdec
//! ```
//! (`--full` runs the samplers' steps too; without it a stage checks its first call.)

use std::collections::HashMap;
use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_diffusion::Picture;
use nextsycl_model3d::{Error, Result};

use crate::{dit, dino, sampler, sconv, ssdec, vae, Trellis2};

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
            (self.out)(format!("{what:<18} {} values here, {} in the reference", got.len(), want.len()));
            self.worst = f64::INFINITY;
            return f64::INFINITY;
        }
        let (rel, mx, cos) = compare(got, want);
        (self.out)(format!("{what:<18} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.6}  ({} values)", want.len()));
        self.worst = self.worst.max(rel);
        rel
    }
    fn say(&mut self, s: String) {
        (self.out)(s);
    }
}

fn npy(dir: &Path, name: &str) -> Result<Vec<f32>> {
    Ok(read_npy(&dir.join(format!("{name}.npy")))?.1)
}

fn coords_npy(dir: &Path, name: &str) -> Result<Vec<[i32; 3]>> {
    Ok(npy(dir, name)?.chunks_exact(3).map(|c| [c[0] as i32, c[1] as i32, c[2] as i32]).collect())
}

/// [C, H, W] planes to [H * W, C] rows
fn chw_to_hwc(v: &[f32], c: usize) -> Vec<f32> {
    let n = v.len() / c;
    (0..n * c).map(|i| v[(i % c) * n + i / c]).collect()
}

/// [N, C] rows to [C, N] planes
fn hwc_to_chw(v: &[f32], c: usize) -> Vec<f32> {
    let n = v.len() / c;
    (0..n * c).map(|i| v[(i % n) * c + i / n]).collect()
}

/// Features of two voxel sets compared where their coordinates agree: (ours, theirs) aligned, missing, extra
fn align(ours: &[[i32; 3]], of: &[f32], theirs: &[[i32; 3]], tf: &[f32], c: usize) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let at: HashMap<[i32; 3], usize> = theirs.iter().enumerate().map(|(i, x)| (*x, i)).collect();
    let (mut a, mut b, mut extra) = (Vec::new(), Vec::new(), 0);
    for (i, x) in ours.iter().enumerate() {
        match at.get(x) {
            Some(&j) => {
                a.extend_from_slice(&of[i * c..(i + 1) * c]);
                b.extend_from_slice(&tf[j * c..(j + 1) * c]);
            }
            None => extra += 1,
        }
    }
    let missing = theirs.len() - (ours.len() - extra);
    (a, b, missing, extra)
}

/// Every stage in `stages` against the dumps in `dir`; the worst relative error
pub fn run(e: &Trellis2, dir: &Path, picture: Option<&Picture>, stages: &[&str], full: bool, log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut l = Log { out: log, worst: 0.0 };
    let cx = &e.cx;
    let g = &cx.gpu;
    let has = |n: &str| dir.join(format!("{n}.npy")).exists();
    let pre = Picture::read_png(&dir.join("pre.png")).map_err(Error)?;
    if stages.contains(&"pre") {
        let pic = picture.ok_or_else(|| Error("pre: the original picture is needed (--picture)".into()))?;
        let mine = e.prepare(pic, true)?;
        let same = mine.data.iter().zip(&pre.data).filter(|(a, b)| a == b).count();
        l.say(format!("pre                {}x{} here, {}x{} in the reference; {} of {} bytes equal", mine.width, mine.height, pre.width, pre.height, same,
                      pre.data.len()));
        if (mine.width, mine.height) != (pre.width, pre.height) {
            l.worst = f64::INFINITY;
        } else {
            let a: Vec<f32> = mine.data.iter().map(|v| *v as f32).collect();
            let b: Vec<f32> = pre.data.iter().map(|v| *v as f32).collect();
            l.line("pre (bytes)", &a, &b);
        }
    }
    if stages.contains(&"remover") && has("rm_in") {
        let r = nextsycl_remover::Remover::load(e.files.get("remover").ok_or_else(|| Error("no remover file".into()))?, g)?;
        let input = chw_to_hwc(&npy(dir, "rm_in")?, 3);
        if let Some(pic) = picture {
            l.line("rm input", &nextsycl_remover::Remover::input(pic), &input);
        }
        let (logits, probe) = r.logits(&input, true)?;
        let p = probe.unwrap();
        l.line("rm x1", &p.x1, &chw_to_hwc(&npy(dir, "rm_x1")?, p.x1.len() / (256 * 256)));
        l.line("rm x4", &p.x4, &chw_to_hwc(&npy(dir, "rm_x4")?, p.x4.len() / (32 * 32)));
        let want = npy(dir, "rm_logits")?;
        l.line("rm logits", &logits, &want);
        let flips = logits.iter().zip(&want).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
        l.say(format!("rm mask            {flips} of {} pixels on the other side of 0.5", want.len()));
    }
    if stages.contains(&"dino") {
        let d = e.dino()?;
        for size in [512usize, 1024] {
            if !has(&format!("cond{size}")) {
                continue;
            }
            let want_in = npy(dir, &format!("dino{size}_in"))?;
            l.line(&format!("dino{size} input"), &dino::input(&pre, size), &want_in);
            let f = d.encode(cx, &want_in, size)?.to_f32()?;
            l.line(&format!("cond{size}"), &f, &npy(dir, &format!("cond{size}"))?);
        }
    }
    let cond = |name: &str, m: &dit::Dit| -> Result<(DevBuf, usize)> {
        let v = npy(dir, name)?;
        Ok((m.cond(g, &v)?, v.len() / m.cond_ch))
    };
    // a flow stage: its first calls against the dump, then (full) the sampler
    let flow = |l: &mut Log, name: &str, m: &dit::Dit, p: &sampler::Params, coords: &[[i32; 3]], x0: Vec<f32>, cn: &str, concat: Option<&[f32]>,
                    to_ref: &dyn Fn(&[f32]) -> Vec<f32>| -> Result<Option<Vec<f32>>> {
        let c = cond(cn, m)?;
        let n = coords.len();
        let w = m.work(g, n, c.1)?;
        let cs = DevBuf::from_f32(g, &dit::rope(coords))?;
        let t = npy(dir, &format!("{name}_t0"))?[0];
        let input = match concat {
            Some(cc) => (0..n).flat_map(|i| x0[i * 32..(i + 1) * 32].iter().chain(&cc[i * 32..(i + 1) * 32]).copied().collect::<Vec<_>>()).collect(),
            None => x0.clone(),
        };
        let mut probe = dit::Probe::default();
        let v = m.forward(cx, &w, &input, t, &c.0, c.1, &cs, None, Some(&mut probe))?;
        l.line(&format!("{name} input layer"), &probe.input, &npy(dir, &format!("{name}_in0"))?);
        l.line(&format!("{name} block 0"), &probe.block0, &npy(dir, &format!("{name}_blk0"))?);
        l.line(&format!("{name} block 29"), &probe.last, &npy(dir, &format!("{name}_blk29"))?);
        l.line(&format!("{name} call 0"), &to_ref(&v), &npy(dir, &format!("{name}_call0"))?);
        if has(&format!("{name}_call1")) && p.guidance != 1.0 {
            let neg = DevBuf::new(g, c.1 * m.cond_ch * 2)?;
            neg.fill(0)?;
            let v1 = m.forward(cx, &w, &input, t, &neg, c.1, &cs, None, None)?;
            l.line(&format!("{name} call 1"), &to_ref(&v1), &npy(dir, &format!("{name}_call1"))?);
            // as the sampler runs it: the zero condition's cross-attention as its value bias
            let rows = m.const_cross(cx)?;
            let v2 = m.forward_with(cx, &w, &input, t, dit::Cross::Const(&rows), &cs, None, None)?;
            l.line(&format!("{name} call 1 (bias)"), &to_ref(&v2), &npy(dir, &format!("{name}_call1"))?);
        }
        drop(w);
        if !full {
            return Ok(None);
        }
        let out = e.flow(m, p, coords, x0, &c, concat, None, |_, _| Ok(()))?;
        let last = format!("{name}_x{}", p.steps);
        if has(&last) {
            l.line(&format!("{name} sample"), &to_ref(&out), &npy(dir, &last)?);
        }
        Ok(Some(out))
    };
    if stages.contains(&"ss") {
        let m = e.ss_flow()?;
        let x0 = chw_to_hwc(&npy(dir, "ss_noise")?, 8);
        let grid = sconv::grid(16);
        let r = flow(&mut l, "ss", &m, &e.cfg.ss, &grid, x0, "cond512", None, &|v| hwc_to_chw(v, 8))?;
        if let Some(z) = r {
            l.line("ss z", &hwc_to_chw(&z, 8), &npy(dir, "ss_z")?);
        }
    }
    if stages.contains(&"ssdec") {
        let (d, logits) = e.ss_dec()?.decode(cx, &npy(dir, "ss_z")?)?;
        let want = npy(dir, "ss_logits")?;
        l.line("ss logits", &logits, &want);
        let flips = logits.iter().zip(&want).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
        let mine = ssdec::occupied(d, &logits, 32);
        let theirs = coords_npy(dir, "ss_coords")?;
        l.say(format!("ss coords          {} here, {} in the reference; {flips} of {} cells on the other side of 0 at 64^3", mine.len(), theirs.len(), want.len()));
    }
    let coords = if has("ss_coords") { coords_npy(dir, "ss_coords")? } else { Vec::new() };
    let id = |v: &[f32]| v.to_vec();
    if stages.contains(&"shape") && has("shape_noise") {
        let m = e.shape_flow(false)?;
        let r = flow(&mut l, "shape", &m, &e.cfg.shape, &coords, npy(dir, "shape_noise")?, "cond512", None, &id)?;
        if let Some(mut s) = r {
            s.chunks_exact_mut(32).for_each(|row| row.iter_mut().enumerate().for_each(|(c, x)| *x = *x * e.cfg.shape_std[c] + e.cfg.shape_mean[c]));
            l.line("shape slat", &s, &npy(dir, "shape_slat")?);
        }
    }
    let hr = has("hr_coords");
    let (tcoords, sname, cname) = if hr { (coords_npy(dir, "hr_coords")?, "shape_hr_slat", "cond1024") } else { (coords.clone(), "shape_slat", "cond512") };
    if stages.contains(&"cascade") && hr {
        let up = e.shape_dec()?.run(cx, &coords, &npy(dir, "shape_slat")?, &vae::Plan { guide: None, upsample: Some(4), keep_ups: false }, |_, _| Ok(()))?;
        let want = coords_npy(dir, "hr_up")?;
        let mut a = up.coords.clone();
        let mut b = want.clone();
        a.sort_unstable();
        b.sort_unstable();
        l.say(format!("hr upsample        {} voxels here, {} in the reference, {}", a.len(), b.len(), if a == b { "the same set" } else { "different sets" }));
        let res = 1024;
        let (q, r) = crate::cascade_coords(&want, 512, res, 49152);
        l.say(format!("hr coords          {} at {r} from the reference's voxels; {} in the reference ({})", q.len(), tcoords.len(),
                      if q == tcoords { "the same" } else { "different" }));
        let m = e.shape_flow(true)?;
        let r = flow(&mut l, "shape_hr", &m, &e.cfg.shape, &tcoords, npy(dir, "shape_hr_noise")?, "cond1024", None, &id)?;
        if let Some(mut s) = r {
            s.chunks_exact_mut(32).for_each(|row| row.iter_mut().enumerate().for_each(|(c, x)| *x = *x * e.cfg.shape_std[c] + e.cfg.shape_mean[c]));
            l.line("shape hr slat", &s, &npy(dir, "shape_hr_slat")?);
        }
    }
    if stages.contains(&"tex") && has("tex_noise") {
        let m = e.tex_flow(hr)?;
        let sn: Vec<f32> = npy(dir, sname)?.chunks_exact(32)
            .flat_map(|r| r.iter().enumerate().map(|(c, x)| (x - e.cfg.shape_mean[c]) / e.cfg.shape_std[c]).collect::<Vec<_>>()).collect();
        let r = flow(&mut l, "tex", &m, &e.cfg.tex, &tcoords, npy(dir, "tex_noise")?, cname, Some(&sn), &id)?;
        if let Some(mut s) = r {
            s.chunks_exact_mut(32).for_each(|row| row.iter_mut().enumerate().for_each(|(c, x)| *x = *x * e.cfg.tex_std[c] + e.cfg.tex_mean[c]));
            l.line("tex slat", &s, &npy(dir, "tex_slat")?);
        }
    }
    if stages.contains(&"pixal") && has("px_ss_global") {
        // reference/pixal3d/ref.py's dumps: each stage's projected condition, then its flow's first calls on them
        let info: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("px_info.json")).map_err(|e| Error(e.to_string()))?)
            .map_err(|e| Error(e.to_string()))?;
        let cam = crate::pixal::Camera::new(info["fov"].as_f64().unwrap_or(crate::pixal::DEFAULT_FOV));
        l.say(format!("pixal camera       distance {:.6} here, {:.6} in the reference", cam.distance, info["distance"].as_f64().unwrap_or(0.0)));
        let conds = crate::pixal::Conds::new(e, &pre, &cam, true)?;
        for (name, size, grid, wide) in [("ss", 512, 16, false), ("shape512", 512, 32, true), ("shape1024", 1024, 64, true), ("tex", 1024, 64, true)] {
            let coords = match name {
                "ss" => sconv::grid(16),
                "shape512" => coords.clone(),
                _ => coords_npy(dir, "hr_coords")?,
            };
            let (glob, proj) = conds.project(e, size, &coords, grid, wide)?;
            let half_f32 = |b: &DevBuf| -> Result<Vec<f32>> {
                let f = DevBuf::f32(g, b.len / 2)?;
                cx.k.convert(b.ptr(), nextsycl_diffusion::kernels::Dt::F16, b.len / 2, f.ptr(), nextsycl_diffusion::kernels::Dt::F32)?;
                f.to_f32()
            };
            l.line(&format!("px {name} global"), &half_f32(&glob.unwrap().0)?, &npy(dir, &format!("px_{name}_global"))?);
            l.line(&format!("px {name} proj"), &half_f32(proj.as_ref().unwrap())?, &npy(dir, &format!("px_{name}_proj"))?);
            // the flow on the reference's own condition
            let m = match name {
                "ss" => e.ss_flow()?,
                "shape512" => e.shape_flow(false)?,
                "shape1024" => e.shape_flow(true)?,
                _ => e.tex_flow(true)?,
            };
            let gl = crate::w::dev_half(g, &npy(dir, &format!("px_{name}_global"))?)?;
            let pj = crate::w::dev_half(g, &npy(dir, &format!("px_{name}_proj"))?)?;
            let n = coords.len();
            let noise = npy(dir, &format!("px_{name}_noise"))?;
            let x = if name == "ss" { chw_to_hwc(&noise, 8) } else { noise };
            let input: Vec<f32> = if name == "tex" {
                let cc = npy(dir, "px_tex_concat")?;
                (0..n).flat_map(|i| x[i * 32..(i + 1) * 32].iter().chain(&cc[i * 32..(i + 1) * 32]).copied().collect::<Vec<_>>()).collect()
            } else {
                x
            };
            let w = m.work(g, n, 5)?;
            let cs = DevBuf::from_f32(g, &dit::rope(&coords))?;
            let mut probe = dit::Probe::default();
            let v = m.forward(cx, &w, &input, 1000.0, &gl, 5, &cs, Some(&pj), Some(&mut probe))?;
            let to_ref = |v: &[f32]| if name == "ss" { hwc_to_chw(v, 8) } else { v.to_vec() };
            let p = format!("px_{name}");
            l.line(&format!("{p} block 0"), &probe.block0, &npy(dir, &format!("{p}_blk0"))?);
            l.line(&format!("{p} block 29"), &probe.last, &npy(dir, &format!("{p}_blk29"))?);
            l.line(&format!("{p} call 0"), &to_ref(&v), &npy(dir, &format!("{p}_call0"))?);
            let (zg, zp) = (DevBuf::new(g, gl.len)?, DevBuf::new(g, pj.len)?);
            zg.fill(0)?;
            zp.fill(0)?;
            let v1 = m.forward(cx, &w, &input, 1000.0, &zg, 5, &cs, Some(&zp), None)?;
            l.line(&format!("{p} call 1"), &to_ref(&v1), &npy(dir, &format!("{p}_call1"))?);
        }
    }
    if stages.contains(&"sdec") || stages.contains(&"tdec") {
        let res = if hr { 1024 } else { 512 };
        let sd = e.shape_dec()?;
        let dec = sd.run(cx, &tcoords, &npy(dir, sname)?, &vae::Plan { guide: None, upsample: None, keep_ups: true }, |_, _| Ok(()))?;
        drop(sd);
        for i in 0..dec.subs.len() {
            let (rc, rs) = (coords_npy(dir, &format!("sdec_c{i}"))?, npy(dir, &format!("sdec_sub{i}"))?);
            let (a, b, miss, extra) = align(&dec.level_coords[i], &dec.subs[i], &rc, &rs, 8);
            let flips = a.iter().zip(&b).filter(|(x, y)| (**x > 0.0) != (**y > 0.0)).count();
            l.line(&format!("sdec sub {i}"), &a, &b);
            l.say(format!("                   level {i}: {} voxels here, {} there ({miss} missing, {extra} extra); {flips} of {} children decided otherwise",
                          dec.level_coords[i].len(), rc.len(), b.len()));
            let next = if i + 1 < dec.subs.len() { coords_npy(dir, &format!("sdec_c{}", i + 1))? } else { coords_npy(dir, "sdec_coords")? };
            let mine_next = if i + 1 < dec.level_coords.len() { &dec.level_coords[i + 1] } else { &dec.coords };
            let c = dec.ups[i].len() / mine_next.len().max(1);
            let (a, b, _, _) = align(mine_next, &dec.ups[i], &next, &npy(dir, &format!("sdec_up{i}"))?, c);
            l.line(&format!("sdec up {i}"), &a, &b);
        }
        let (a, b, miss, extra) = align(&dec.coords, &dec.feats, &coords_npy(dir, "sdec_coords")?, &npy(dir, "sdec_h")?, 7);
        l.line("sdec h", &a, &b);
        l.say(format!("                   {} voxels here ({miss} missing, {extra} extra against the reference)", dec.coords.len()));
        // the same decoder on the reference's own subdivisions: its arithmetic alone, no decision flipped
        let mut tree = crate::sparse::Tree::default();
        for i in 0..dec.subs.len() {
            let (rc, rs) = (coords_npy(dir, &format!("sdec_c{i}"))?, npy(dir, &format!("sdec_sub{i}"))?);
            tree.levels.push(crate::sparse::subdivide(&rc, |p, o| rs[p * 8 + o] > 0.0));
        }
        let sd = e.shape_dec()?;
        let rt = sd.run(cx, &tcoords, &npy(dir, sname)?, &vae::Plan { guide: Some(&tree), upsample: None, keep_ups: true }, |_, _| Ok(()))?;
        drop(sd);
        for i in 0..rt.ups.len() {
            let next = if i + 1 < tree.levels.len() { format!("sdec_c{}", i + 1) } else { "sdec_coords".into() };
            if tree.levels[i].coords != coords_npy(dir, &next)? {
                l.say(format!("sdec tree {i}       the reference's subdivision does not give its own next voxels"));
            }
            l.line(&format!("sdec up {i} (ref tree)"), &rt.ups[i], &npy(dir, &format!("sdec_up{i}"))?);
        }
        l.line("sdec h (ref tree)", &rt.feats, &npy(dir, "sdec_h")?);
        let mesh = crate::mesh::dual_grid(&dec.coords, &dec.feats, &dec.nmap, res);
        let (rv, rf) = (npy(dir, "mesh_v")?.len() / 3, npy(dir, "mesh_f")?.len() / 3);
        l.say(format!("mesh               {} vertices, {} triangles here; {rv} vertices (before dropping unused), {rf} triangles in the reference",
                      mesh.positions.len(), mesh.triangles()));
        if stages.contains(&"tdec") && has("tdec") {
            let td = e.tex_dec()?;
            let tv = td.run(cx, &tcoords, &npy(dir, "tex_slat")?, &vae::Plan { guide: Some(&dec.tree), upsample: None, keep_ups: false }, |_, _| Ok(()))?;
            let mine: Vec<f32> = tv.feats.iter().map(|v| v * 0.5 + 0.5).collect();
            let (a, b, _, _) = align(&tv.coords, &mine, &coords_npy(dir, "tdec_coords")?, &npy(dir, "tdec")?, 6);
            l.line("tdec", &a, &b);
        }
    }
    Ok(l.worst)
}
