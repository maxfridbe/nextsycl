//! Stage by stage against the reference (reference/unimate/ref.py: UniMate's modules transcribed, float32 on the CPU).
//! Each stage starts from the reference's own input, so a difference shows where it is born:
//!
//! ```text
//!   t5      the prompt (meta.json) and the joint names  -> tokens.npy, t5.npy, caption.npy, names_emb.npy
//!   pass    in_*.npy + noise.npy + caption.npy          -> pass_{y,tokens0,block0,block9,cond,uncond,guided}.npy (t = 0.37)
//!   ode     the same inputs                             -> ode.npy (the whole flow: dopri5, 128 evaluations in the reference)
//!   dec     ode.npy                                     -> motion.npy, quats.npy, root_pos.npy
//!   prep    a rigged GLB (--mesh)                       -> the dump's own skeleton inputs (the example assets' cond)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_model3d::{Error, Result};

use crate::net::{Cond, SkelIn, Trace, FEAT, J};
use crate::{rig, t5, UniMate};

/// A little-endian .npy: (shape, values) of a float32 or int64 array
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: {head} (C order expected)", path.display())));
    }
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let data = &b[at + hl..];
    let v = if head.contains("'<f4'") {
        data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if head.contains("'<i8'") {
        data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f32).collect()
    } else {
        return Err(Error(format!("{}: {head} (float32 or int64 expected)", path.display())));
    };
    Ok((shape, v))
}

/// How far `got` is from `want`: (relative RMS error, max abs error, cosine)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (f64::from(*g), f64::from(*w));
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
        log(format!("{what:<14} {} values here, {} in the reference", got.len(), want.len()));
        return f64::INFINITY;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<14} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.7}  ({} values)", want.len()));
    rel
}

fn npy(dir: &Path, name: &str) -> Result<Vec<f32>> {
    Ok(read_npy(&dir.join(format!("{name}.npy")))?.1)
}

fn ints(v: &[f32]) -> Vec<i32> {
    v.iter().map(|x| *x as i32).collect()
}

/// The reference's skeleton inputs (the dump's in_*.npy, padded to 71 joints)
fn ref_skel(dir: &Path, n: usize, names_emb: Vec<f32>) -> Result<SkelIn> {
    Ok(SkelIn {
        n,
        tpos: npy(dir, "in_tpos")?,
        tpos_parents: npy(dir, "in_tpos_parents")?,
        depths: ints(&npy(dir, "in_depths")?),
        spectral: npy(dir, "in_spectral")?,
        graph_dist: ints(&npy(dir, "in_graph_dist")?),
        relations: ints(&npy(dir, "in_relations")?),
        names_emb,
    })
}

fn meta(dir: &Path) -> Result<serde_json::Value> {
    let p = dir.join("meta.json");
    serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?).map_err(|e| Error(format!("{}: {e}", p.display())))
}

fn padded_names(emb: &[f32]) -> Vec<f32> {
    let mut v = vec![0f32; J * t5::DIM];
    v[..emb.len()].copy_from_slice(emb);
    v
}

/// Every stage of `stages` against the dumps in `dir`; the worst relative error
pub fn run(m: &UniMate, dir: &Path, stages: &[&str], mesh: Option<&Path>, prompt: Option<&str>, log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let mut note = |r: f64| worst = worst.max(if r.is_nan() { f64::INFINITY } else { r });
    let meta = meta(dir)?;
    let n = meta["joints"].as_u64().unwrap_or(0) as usize;
    let cfg = meta["cfg"].as_f64().unwrap_or(3.0) as f32;
    let want_prompt = meta["prompt"].as_str().unwrap_or("").to_string();
    let prompt = prompt.unwrap_or(&want_prompt);

    if stages.contains(&"t5") {
        let ids = m.tok.encode(prompt);
        let want: Vec<u32> = npy(dir, "tokens")?.iter().map(|x| *x as u32).collect();
        let same = ids == want;
        log(format!("tokens         {}", if same { "the same".to_string() } else { format!("DIFFERENT: {ids:?} here, {want:?} there") }));
        if !same {
            note(f64::INFINITY);
        }
        let t0 = Instant::now();
        let h = m.t5.encode(&m.ops, &want)?;
        note(line("t5 hidden", &h, &npy(dir, "t5")?, log));
        note(line("caption", &t5::pooled(&h), &npy(dir, "caption")?, log));
        let names: Vec<String> = meta["names"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect());
        let got = m.name_embeddings(&names)?;
        note(line("names_emb", &got, &npy(dir, "names_emb")?, log));
        log(format!("t5 stage took {:.2} s ({} names)", t0.elapsed().as_secs_f64(), names.len()));
    }

    let skel = || -> Result<(SkelIn, Cond)> {
        let s = ref_skel(dir, n, padded_names(&npy(dir, "names_emb")?))?;
        let c = m.net.cond(&m.ops, &s)?;
        Ok((s, c))
    };
    let caption = || npy(dir, "caption");

    if stages.contains(&"pass") {
        let (_, c) = skel()?;
        let x = npy(dir, "noise")?;
        let cap = m.net.caption(&caption()?, false);
        let unc = m.net.caption(&caption()?, true);
        let w = m.net.work(&m.ops)?;
        let mut tr = Trace::default();
        let oc = m.net.forward(&m.ops, &w, &c, &cap, &x, 0.37, Some(&mut tr))?;
        note(line("pass y", &tr.y, &npy(dir, "pass_y")?, log));
        note(line("pass tokens0", &tr.tokens0, &npy(dir, "pass_tokens0")?, log));
        note(line("pass block0", &tr.block0, &npy(dir, "pass_block0")?, log));
        note(line("pass block9", &tr.block_last, &npy(dir, "pass_block9")?, log));
        note(line("pass cond", &oc, &npy(dir, "pass_cond")?, log));
        let ou = m.net.forward(&m.ops, &w, &c, &unc, &x, 0.37, None)?;
        note(line("pass uncond", &ou, &npy(dir, "pass_uncond")?, log));
        let g: Vec<f32> = ou.iter().zip(&oc).map(|(u, c)| u + cfg * (c - u)).collect();
        note(line("pass guided", &g, &npy(dir, "pass_guided")?, log));
    }

    let want_ode = dir.join("ode.npy");
    let mut ode_out: Option<Vec<f32>> = None;
    if stages.contains(&"ode") {
        let (_, c) = skel()?;
        let t0 = Instant::now();
        let (x, st) = m.sample(&c, &caption()?, &npy(dir, "noise")?, cfg, &mut |_| Ok(()))?;
        let ref_nfe = meta["nfe"].as_u64().unwrap_or(0) as usize;
        let ref_steps = (ref_nfe.saturating_sub(2)) / 6;
        let rejected = st.steps.iter().filter(|s| !s.2).count();
        log(format!("ode            {} evaluations here, {} in the reference ({} steps, {} rejected here vs {} steps there); {:.2} s",
                    st.nfe, ref_nfe, st.steps.len(), rejected, ref_steps, t0.elapsed().as_secs_f64()));
        if st.nfe != ref_nfe {
            note(1.0);
        }
        if want_ode.exists() {
            note(line("ode result", &x, &npy(dir, "ode")?, log));
        }
        ode_out = Some(x);
    }

    if stages.contains(&"dec") {
        let x = match ode_out {
            Some(x) => x,
            None => npy(dir, "ode")?,
        };
        let parents: Vec<i32> = meta["parents"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|p| p.as_i64()).map(|p| p as i32).collect());
        let (mean, std): (Vec<f64>, Vec<f64>) = (npy(dir, "in_mean")?.iter().map(|v| f64::from(*v)).collect(), npy(dir, "in_std")?.iter().map(|v| f64::from(*v)).collect());
        let d = crate::motion::decode(&x, n, &parents, &mean, &std, None);
        note(line("dec motion", &d.features, &npy(dir, "motion")?, log));
        let q: Vec<f32> = d.quats.iter().flatten().copied().collect();
        note(line("dec quats", &q, &npy(dir, "quats")?, log));
        let r: Vec<f32> = d.root.iter().flatten().copied().collect();
        note(line("dec root", &r, &npy(dir, "root_pos")?, log));
    }

    if stages.contains(&"prep") {
        let path = mesh.ok_or_else(|| Error("the prep stage reads the rigged GLB: --mesh FILE.glb".into()))?;
        let bytes = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let mesh = nextsycl_model3d::read_mesh(&bytes).map_err(Error)?;
        let skin = mesh.skin.as_ref().ok_or_else(|| Error(format!("{}: no skin", path.display())))?;
        let p = rig::prepare(skin, rig::Facing::Auto, rig::NameMode::Auto)?;
        let ref_parents: Vec<i32> = meta["parents"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|p| p.as_i64()).map(|p| p as i32).collect());
        log(format!("prep           {} joints here, {} in the reference; facing from {}", p.n, ref_parents.len(), p.facing_note));
        log(format!("prep parents   {}", if p.parents == ref_parents { "the same" } else { "DIFFERENT (the joint order is not the reference's)" }));
        let ref_names: Vec<String> = meta["names"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect());
        let same = p.labels.iter().zip(&ref_names).filter(|(a, b)| a == b).count();
        log(format!("prep names     {same} of {} labels equal the reference's cleaned names ({} recognized by the rules)", ref_names.len(), p.recognized));
        for (a, b) in p.labels.iter().zip(&ref_names).filter(|(a, b)| a != b) {
            log(format!("               {a:?} here, {b:?} there"));
        }
        let si = rig::skel_in(&p, &m.stats, &vec![0f32; p.n * t5::DIM]);
        if p.parents == ref_parents {
            note(line("prep tpos", &si.tpos[..n * FEAT], &npy(dir, "in_tpos")?[..n * FEAT], log));
            let (rel, dist) = (ints(&npy(dir, "in_relations")?), ints(&npy(dir, "in_graph_dist")?));
            log(format!("prep relations {}", if si.relations == rel { "the same" } else { "DIFFERENT" }));
            log(format!("prep distances {}", if si.graph_dist == dist { "the same" } else { "DIFFERENT" }));
            log(format!("prep depths    {}", if si.depths == ints(&npy(dir, "in_depths")?) { "the same" } else { "DIFFERENT" }));
            if si.relations != rel || si.graph_dist != dist {
                note(1.0);
            }
        } else {
            note(1.0);
        }
    }
    Ok(worst)
}
