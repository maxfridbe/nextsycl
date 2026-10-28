//! UniMate (Friedrich-M/UniMate, SIGGRAPH Asia 2026; code MIT, checkpoints CC BY-NC 4.0): the skeleton of a rigged
//! mesh set in motion by a sentence - 60 frames of joint rotations and a root path, written as a glTF skeletal
//! animation on the input's own rig.
//!
//! The pipeline (inference/sample.py), float32 as the reference:
//! 1. the rig in the model's terms (`rig`): breadth-first joint order, the canonical rest pose, labels from the joint
//!    names by rules (no LLM), the graph's relations, distances, depths and spectral features;
//! 2. flan-T5-base (`t5`): the prompt's pooled embedding and each joint label's;
//! 3. the flow (`net`, `ode`): from noise, torchdiffeq's adaptive Dormand-Prince over the velocity network with
//!    classifier-free guidance 3 against the empty text, 71 joints x 12 features x 60 frames;
//! 4. the features as an animation (`motion`): de-normalized, the root's path, each joint's rotation, back in the
//!    rig's own frame.
//!
//! Its kernels: `kernels/model3d/unimate/unimate.cpp` (norms with the adaLN modulation, attention with a graph bias,
//! rotary embeddings, the gated units) and the shared diffusion kernels' float32 linear layers.

pub mod check;
mod eig;
mod ffi;
pub mod motion;
pub mod names;
pub mod net;
pub mod ode;
pub mod ops;
pub mod rig;
pub mod t5;
mod vocab;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::Gpu;
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::mesh::SkelMotion;
use nextsycl_model3d::{Abilities, At, Defaults, EngineOption, Error, LoadOptions, Mesh, Model3dEngine, Model3dKind, ModelFiles, Result, RigRequest, Step};

use ops::{Ops, Store};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "unimate";

/// The files it needs, by role: the checkpoint (EMA weights), its run's config.json and dataset statistics, and
/// flan-T5-base's weights, config and tokenizer
pub const ROLES: &[&str] = &["ckpt", "config", "stats", "t5", "t5-config", "tokenizer"];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption {
        name: "facing",
        env: "NS_M3D_UNIMATE_FACING",
        value: "auto|+z|-z|+x|-x",
        help: "which way the rig looks (default auto: the sides in the joint names, else the mirror symmetry, else +Z)",
        at: At::Both,
    },
    EngineOption {
        name: "names",
        env: "NS_M3D_UNIMATE_NAMES",
        value: "auto|structure",
        help: "auto: the joint names cleaned by rules (unrecognized: Bone); structure: no names, the model sees the graph alone",
        at: At::Both,
    },
];

/// The model's fixed clip: 60 frames at 30 fps
pub const FRAMES: usize = net::T;
pub const FPS: f32 = 30.0;
/// The checkpoint's tensors in `parameters()` order: name and shape (the EMA copy is a list in this order)
const PARAM_ORDER: &str = include_str!("param_order.txt");

pub fn kind() -> Model3dKind {
    Model3dKind {
        archs: &[ARCH],
        name: "UniMate (the skeleton of a rigged mesh set in motion by a sentence: flan-T5, a graph spatio-temporal flow transformer; 60 frames as glTF skeletal animation)",
        roles: ROLES,
        load,
        options: OPTIONS,
    }
}

pub struct UniMate {
    pub ops: Ops,
    pub tok: t5::T5Tok,
    pub t5: t5::T5,
    pub net: net::Net,
    pub stats: rig::Stats,
    work: Mutex<net::Work>,
    facing: rig::Facing,
    names: rig::NameMode,
    load_s: f64,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(UniMate::load(files, gpu, o, log)?))
}

/// config.json must be the run this engine is: the graph attention, the adaLN text, 71 joints of 12 features, 60 frames
fn check_config(path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let j: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let m = &j["model"];
    let d = &j["dataset"];
    let ok = m["attention"] == "graph" && m["text_cond"] == "adaln" && m["latent_dim"] == 512 && m["num_layers"] == 10 && m["num_heads"] == 8
        && m["ff_size"] == 2048 && d["max_joints"] == 71 && d["max_motion_length"] == 60 && d["feature_len"] == 12 && m["num_tpos_queries"] == 4;
    if ok {
        Ok(())
    } else {
        Err(Error(format!("{}: not the graph_adaln run (attention graph, text_cond adaln, 512 wide, 10 layers, 71 joints, 60 frames) this engine implements", path.display())))
    }
}

impl UniMate {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<UniMate> {
        let t0 = Instant::now();
        let missing = nextsycl_model3d::missing(files, ROLES);
        if !missing.is_empty() {
            return Err(Error(format!("{ARCH}: no {} file", missing.join(", "))));
        }
        check_config(role(files, "config")?)?;
        let ops = Ops::new(gpu)?;
        // the EMA weights, in parameters() order
        let pth = Pth::open(role(files, "ckpt")?).map_err(|e| Error(e.0))?;
        let mut store = Store::new(gpu);
        for (i, line) in PARAM_ORDER.lines().enumerate() {
            let mut it = line.split_whitespace();
            let name = it.next().ok_or_else(|| Error("param_order.txt".into()))?;
            let shape: Vec<usize> = it.map(|d| d.parse().unwrap_or(0)).collect();
            let key = format!("ema_state_dict.shadow_params.{i}");
            let t = pth.need(&key).map_err(|e| Error(format!("{}: {} (the EMA weights of UniMate's checkpoint)", e.0, pth.path.display())))?;
            if t.shape != shape {
                return Err(Error(format!("{}: {key} is {:?}, {name} should be {shape:?}", pth.path.display(), t.shape)));
            }
            store.map.insert(name.to_string(), (shape, pth.f32(&key).map_err(|e| Error(e.0))?));
        }
        let net = net::Net::load(&ops, store)?;
        let work = net.work(&ops)?;
        let tok = t5::T5Tok::open(role(files, "tokenizer")?)?;
        check_t5_config(role(files, "t5-config")?)?;
        let t5 = t5::T5::load(&ops, role(files, "t5")?)?;
        let stats = rig::Stats::open(role(files, "stats")?)?;
        let facing = o.setting("NS_M3D_UNIMATE_FACING").map(|v| rig::Facing::parse(&v)).transpose()?.unwrap_or(rig::Facing::Auto);
        let names = o.setting("NS_M3D_UNIMATE_NAMES").map(|v| rig::NameMode::parse(&v)).transpose()?.unwrap_or(rig::NameMode::Auto);
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: loaded on {} in {load_s:.1} s (flan-T5-base, the graph_adaln denoiser; float32)", gpu.name));
        Ok(UniMate { ops, tok, t5, net, stats, work: Mutex::new(work), facing, names, load_s })
    }

    /// The pooled T5 embedding of a text (zeros for an empty one) [768]
    pub fn text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tok.encode(text);
        if text.trim().is_empty() {
            return Ok(vec![0.0; t5::DIM]);
        }
        Ok(t5::pooled(&self.t5.encode(&self.ops, &ids)?))
    }

    /// The joints' name embeddings [n, 768] (a label once, however many joints carry it)
    pub fn name_embeddings(&self, labels: &[String]) -> Result<Vec<f32>> {
        let mut cache: std::collections::HashMap<&str, Vec<f32>> = std::collections::HashMap::new();
        let mut out = Vec::with_capacity(labels.len() * t5::DIM);
        for l in labels {
            if !cache.contains_key(l.as_str()) {
                cache.insert(l, self.text(l)?);
            }
            out.extend_from_slice(&cache[l.as_str()]);
        }
        Ok(out)
    }

    /// The flow from `noise` [71, 12, 60] to the sample at t = 1, with guidance `cfg` (<= 1: the text alone)
    pub fn sample(&self, c: &net::Cond, caption: &[f32], noise: &[f32], cfg: f32, progress: &mut dyn FnMut(usize) -> Result<()>) -> Result<(Vec<f32>, ode::Stats)> {
        let w = self.work.lock().map_err(|_| Error("the work buffers are poisoned".into()))?;
        let cap = self.net.caption(caption, false);
        let unc = self.net.caption(caption, true);
        let mut calls = 0usize;
        let mut drift = |t: f32, x: &[f32]| -> Result<Vec<f32>> {
            calls += 1;
            let cond = self.net.forward(&self.ops, &w, c, &cap, x, t, None)?;
            let out = if cfg > 1.0 {
                let un = self.net.forward(&self.ops, &w, c, &unc, x, t, None)?;
                un.iter().zip(&cond).map(|(u, c)| u + cfg * (c - u)).collect()
            } else {
                cond
            };
            progress(calls)?;
            Ok(out)
        };
        ode::dopri5(&mut drift, noise, 1.0, 1e-3, 1e-6)
    }
}

fn check_t5_config(path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let j: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let ok = j["d_model"] == 768 && j["num_layers"] == 12 && j["num_heads"] == 12 && j["d_ff"] == 2048 && j["d_kv"] == 64
        && j["feed_forward_proj"] == "gated-gelu" && j["relative_attention_num_buckets"] == 32 && j["relative_attention_max_distance"] == 128;
    if ok {
        Ok(())
    } else {
        Err(Error(format!("{}: not flan-T5-base's config (768 wide, 12 layers, 12 heads, gated GELU)", path.display())))
    }
}

/// Quaternions (x, y, z, w) of two frames blended: normalized linear with the shorter arc
fn nlerp(a: [f32; 4], b: [f32; 4], w: f32) -> [f32; 4] {
    let s = if a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>() < 0.0 { -1.0 } else { 1.0 };
    let q = [0, 1, 2, 3].map(|i| a[i] * (1.0 - w) + s * b[i] * w);
    let l = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    q.map(|v| v / l)
}

/// `n` frames over the span of the clip's (rotations by joint and the root path, blended linearly)
fn resample(rot: &[Vec<[f32; 4]>], root: &[[f32; 3]], n: usize) -> (Vec<Vec<[f32; 4]>>, Vec<[f32; 3]>) {
    if n == rot.len() || rot.len() < 2 {
        return (rot.to_vec(), root.to_vec());
    }
    let mut r = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n);
    for j in 0..n {
        let t = if n > 1 { j as f32 * (rot.len() - 1) as f32 / (n - 1) as f32 } else { 0.0 };
        let a = (t.floor() as usize).min(rot.len() - 2);
        let w = t - a as f32;
        r.push(rot[a].iter().zip(&rot[a + 1]).map(|(x, y)| nlerp(*x, *y, w)).collect());
        p.push([0, 1, 2].map(|c| root[a][c] + (root[a + 1][c] - root[a][c]) * w));
    }
    (r, p)
}

impl Model3dEngine for UniMate {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn abilities(&self) -> Abilities {
        Abilities { animate: true, ..Default::default() }
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: 0, cfg: 3.0, frames: FRAMES as u32, fps: FPS, ..Default::default() }
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn rig(&self, req: &RigRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        let t0 = Instant::now();
        let prompt = req.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty())
            .ok_or_else(|| Error(format!("{ARCH} animates a rig from a description: give a prompt (\"An object walks forward.\")")))?;
        let skin = req.mesh.skin.as_ref().ok_or_else(|| Error(format!("{ARCH} animates the skeleton of a rigged mesh: this one has no skin (rig it first: unirig, rignet)")))?;
        let facing = req.extra.get("facing").map(|v| rig::Facing::parse(v)).transpose()?.unwrap_or(self.facing);
        let names = req.extra.get("names").map(|v| rig::NameMode::parse(v)).transpose()?.unwrap_or(self.names);
        let d = self.defaults();
        let cfg = req.cfg.unwrap_or(d.cfg);
        let frames = req.frames.unwrap_or(d.frames).clamp(2, 600) as usize;
        let fps = req.fps.unwrap_or(d.fps);
        progress(Step { phase: "encode", at: 0, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let p = rig::prepare(skin, facing, names)?;
        eprintln!("{ARCH}: {} joints, {} of their names recognized; facing from {}", p.n, p.recognized, p.facing_note);
        let caption = self.text(prompt)?;
        let name_emb = self.name_embeddings(&p.labels)?;
        let cond = self.net.cond(&self.ops, &rig::skel_in(&p, &self.stats, &name_emb))?;
        progress(Step { phase: "encode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let noise = ops::noise(req.seed, net::J * net::FEAT * net::T);
        let (x, st) = self.sample(&cond, &caption, &noise, cfg, &mut |calls| {
            progress(Step { phase: "motion", at: calls as u32, of: calls.max(if cfg > 1.0 { 256 } else { 128 }) as u32, seconds: t0.elapsed().as_secs_f64() })
        })?;
        eprintln!("{ARCH}: the flow took {} evaluations ({} steps, {} rejected) in {:.2} s", st.nfe, st.steps.len(), st.steps.iter().filter(|s| !s.2).count(), t0.elapsed().as_secs_f64());
        let (mean, std) = self.stats.rows(p.n);
        let dec = motion::decode(&x, p.n, &p.parents, &mean, &std, Some(&p.canon.r));

        // back onto the rig's own joints and frame: the root's travel (canonical units) in the rig's units
        let mut newidx = vec![0usize; p.n];
        for (k, &o) in p.order.iter().enumerate() {
            newidx[o] = k;
        }
        let rest = p.tpos[0];
        let head = skin.joints[p.order[0]].head;
        let rot: Vec<Vec<[f32; 4]>> = (0..FRAMES).map(|t| {
            (0..p.n).map(|i| {
                let q = dec.quats[t * p.n + newidx[i]];
                [q[1], q[2], q[3], q[0]]
            }).collect()
        }).collect();
        let root: Vec<[f32; 3]> = dec.root.iter().map(|r| {
            let dl = p.canon.back_delta([f64::from(r[0]), f64::from(r[1]) - rest[1], f64::from(r[2])]);
            [head[0] + dl[0] as f32, head[1] + dl[1] as f32, head[2] + dl[2] as f32]
        }).collect();
        let (rot, root) = resample(&rot, &root, frames);
        progress(Step { phase: "decode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let mut m = req.mesh.clone();
        if let Some(s) = m.skin.as_mut() {
            s.motion = Some(SkelMotion { fps, rotations: rot, root, root_joint: p.order[0] });
        }
        if m.normals.len() != m.positions.len() {
            m.compute_normals();
        }
        Ok(m)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {}", self.ops.gpu.name)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parameter_list_matches_the_model() {
        let n = PARAM_ORDER.lines().count();
        assert_eq!(n, 363);
        assert!(PARAM_ORDER.starts_with("cond_embedder.weight 512 768"));
    }

    #[test]
    fn resampling_keeps_the_ends() {
        let rot: Vec<Vec<[f32; 4]>> = (0..60).map(|i| vec![[0.0, 0.0, (i as f32 * 0.01).sin(), (i as f32 * 0.01).cos()]]).collect();
        let root: Vec<[f32; 3]> = (0..60).map(|i| [0.0, 0.0, i as f32]).collect();
        let (r, p) = resample(&rot, &root, 119);
        assert_eq!(r.len(), 119);
        assert_eq!(p[118][2], 59.0);
        assert_eq!(r[0][0], rot[0][0]);
        assert!((p[1][2] - 0.5).abs() < 1e-6);
    }
}
