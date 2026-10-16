//! AnimateAnyMesh (Wu et al., ICCV 2025; github.com/JarrentWu1031/AnimateAnyMesh, Apache-2.0): a static mesh and a
//! sentence in, 16 frames of motion out - a morph animation on the input's own vertices.
//!
//! The pipeline (test_drive.py), all float32 as the reference:
//! 1. the mesh merged, recentred and scaled in Blender's frame (`prep`);
//! 2. DyMeshVAE's encoder (`vae`): point embeddings, attention over each vertex's mesh neighbours, farthest point
//!    sampling of the embeddings into max(512, N / 8) trajectory tokens, eight cross-attention blocks - their rest
//!    latents f0;
//! 3. the prompt through SD 1.5's CLIP-L text tower (`clip`);
//! 4. the rectified flow (`dit`): from noise, the midpoint rule over 63 steps (torchdiffeq's fixed grid of 64 times),
//!    classifier-free guidance 3 against the empty prompt, the rest channels clamped to f0 at every evaluation;
//! 5. DyMeshVAE's decoder: 16 frames of offsets for every vertex, back in the input's frame and scale.
//!
//! Its kernels: `kernels/model3d/animateanymesh/aam.cpp` (norms, attention dense and over a CSR of neighbours,
//! farthest points, the solver's steps) and the shared diffusion kernels' float32 linear layers.

pub mod check;
pub mod clip;
pub mod dit;
mod ffi;
pub mod ops;
pub mod prep;
pub mod prof;
pub mod vae;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::{
    mesh::Morph, Abilities, At, Defaults, EngineOption, Error, LoadOptions, Mesh, Model3dEngine, Model3dKind, ModelFiles, Result, RigRequest, Step,
};

use ops::{Ops, Weights};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "animateanymesh";

/// The files it needs, by role: the RF model (the MMDiT and CLIP's text tower), the VAE, its latents' statistics
/// (dvae_factors), and CLIP's tokenizer
pub const ROLES: &[&str] = &["rf", "vae", "factors", "vocab", "merges"];

pub const OPTIONS: &[EngineOption] = &[EngineOption {
    name: "traj",
    env: "NS_M3D_AAM_TRAJ",
    value: "N",
    help: "trajectory tokens the VAE samples (default max(512, vertices / 8), the reference's)",
    at: At::Both,
}];

/// The trajectory tokens of the reference at least
const MIN_TRAJ: usize = 512;
/// Padded rows the reference's VAE always sees
const MIN_ROWS: usize = 4096;

pub fn kind() -> Model3dKind {
    Model3dKind {
        archs: &[ARCH],
        name: "AnimateAnyMesh (a mesh set in motion by a sentence: CLIP-L, DyMeshVAE, a rectified-flow MMDiT; 16 frames as morph targets)",
        roles: ROLES,
        load,
        options: OPTIONS,
    }
}

/// The latents' rescaling (dvae_factors): f0 mean / std, ft mean / std
#[derive(Clone, Copy, Debug)]
pub struct Factors {
    pub f0_mean: f32,
    pub f0_std: f32,
    pub ft_mean: f32,
    pub ft_std: f32,
}

pub struct AnimateAnyMesh {
    pub ops: Ops,
    pub tok: clip::ClipTok,
    pub clip: clip::Clip,
    pub dit: dit::Dit,
    pub vae: vae::Vae,
    pub factors: Factors,
    traj: Option<usize>,
    load_s: f64,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(AnimateAnyMesh::load(files, gpu, o, log)?))
}

impl AnimateAnyMesh {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<AnimateAnyMesh> {
        let t0 = Instant::now();
        let missing = nextsycl_model3d::missing(files, ROLES);
        if !missing.is_empty() {
            return Err(Error(format!("{ARCH}: no {} file", missing.join(", "))));
        }
        let ops = Ops::new(gpu)?;
        let rf = Pth::open(role(files, "rf")?).map_err(|e| Error(e.0))?;
        let clip_prefix = rf.prefix_of("clip_text_model.text_model.final_layer_norm.weight")
            .map(|p| format!("{p}clip_text_model.text_model."))
            .or_else(|| rf.prefix_of("clip_text_model.final_layer_norm.weight").map(|p| format!("{p}clip_text_model.")))
            .ok_or_else(|| Error(format!("{}: no CLIP text model inside (not AnimateAnyMesh's RF checkpoint)", rf.path.display())))?;
        let dit_prefix = rf.prefix_of("input_proj.weight").ok_or_else(|| Error(format!("{}: no input_proj (not AnimateAnyMesh's RF checkpoint)", rf.path.display())))?;
        let clip = clip::Clip::load(&Weights { pth: &rf, prefix: clip_prefix, gpu })?;
        let dit = dit::Dit::load(&Weights { pth: &rf, prefix: dit_prefix, gpu })?;
        let vp = Pth::open(role(files, "vae")?).map_err(|e| Error(e.0))?;
        let vae_prefix = vp.prefix_of("point_embed.basis").ok_or_else(|| Error(format!("{}: no point_embed (not DyMeshVAE)", vp.path.display())))?;
        let vae = vae::Vae::load(&Weights { pth: &vp, prefix: vae_prefix, gpu })?;
        let fpath = role(files, "factors")?;
        let fj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fpath).map_err(|e| Error(format!("{}: {e}", fpath.display())))?)
            .map_err(|e| Error(format!("{}: {e}", fpath.display())))?;
        let f = |k: &str| fj[k].as_f64().map(|x| x as f32).ok_or_else(|| Error(format!("{}: no {k}", fpath.display())));
        let factors = Factors { f0_mean: f("f0_mean")?, f0_std: f("f0_std")?, ft_mean: f("ft_mean")?, ft_std: f("ft_std")? };
        let tok = clip::ClipTok::open(role(files, "vocab")?, role(files, "merges")?)?;
        let traj = o.setting("NS_M3D_AAM_TRAJ").map(|v| v.parse::<usize>().map_err(|_| Error(format!("NS_M3D_AAM_TRAJ: {v:?} is not a count")))).transpose()?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let silo = ops.k.silo().map_or_else(|| "no silo: the engine's own kernels".to_string(), |s| format!("silo {}", s.path.display()));
        log(format!("{ARCH}: loaded on {} in {load_s:.1} s (CLIP-L, the MMDiT, DyMeshVAE; float32; {silo})", gpu.name));
        Ok(AnimateAnyMesh { ops, tok, clip, dit, vae, factors, traj, load_s })
    }

    /// The encoder's view of a prepared mesh: its rows' embeddings after the neighbour attention
    pub fn shape(&self, p: &prep::Prepared) -> Result<(vae::Shape, DevBuf)> {
        let n = p.verts.len();
        let l = n.max(MIN_ROWS);
        let mut pos = p.verts.clone();
        pos.resize(l, [0.0; 3]);
        let embed = self.vae.embed0(&self.ops, &pos)?;
        let e0 = self.ops.buf(l * vae::DIM)?;
        e0.copy_within(0, &embed, 0, l * vae::DIM * 4)?;
        let (rowptr, cols) = p.csr(l);
        let rowptr = ops::dev_i32(&self.ops.gpu, &rowptr)?;
        let cols = ops::dev_i32(&self.ops.gpu, &cols)?;
        self.vae.neighbours(&self.ops, &embed, l, &rowptr, &cols)?;
        Ok((vae::Shape { n, l, embed }, e0))
    }

    /// The trajectory tokens of a shape: farthest points of its vertices' embeddings, the first one replaced by a
    /// random vertex (as the reference's replace_negative_indices does: it takes index 0 for "none"), and random
    /// vertices past the vertex count
    pub fn sample_tokens(&self, sh: &vae::Shape, k: usize, seed: u64) -> Result<(Vec<i32>, Vec<i32>)> {
        let kk = k.min(sh.n);
        let dist = self.ops.buf(sh.n)?;
        let idx = DevBuf::new(&self.ops.gpu, kk * 4)?;
        self.ops.k.fps(&sh.embed, sh.n, vae::DIM, kk, &dist, &idx)?;
        let mut b = vec![0u8; kk * 4];
        idx.read(0, &mut b)?;
        let raw: Vec<i32> = b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut rng = ops::Rng::new(seed ^ 0x5eed_f00d);
        let mut out = raw.clone();
        out.resize(k, 0);
        for (i, x) in out.iter_mut().enumerate() {
            if i == 0 || i >= kk {
                *x = (rng.unit() * sh.n as f64) as i32;
            }
        }
        Ok((raw, out))
    }

    /// The rest latents of the tokens, rescaled: f0 [K, 32] (what the flow is clamped to)
    pub fn f0(&self, sh: &vae::Shape, idx: &[i32]) -> Result<DevBuf> {
        let k = idx.len();
        let d = ops::dev_i32(&self.ops.gpu, idx)?;
        let x0 = self.vae.encode(&self.ops, sh, &d, k)?;
        let f = self.factors;
        let v: Vec<f32> = x0.to_f32()?.iter().map(|x| (x - f.f0_mean) / f.f0_std).collect();
        DevBuf::from_f32(&self.ops.gpu, &v)
    }

    /// The guided pair's text for `prompt`
    pub fn text(&self, prompt: &str) -> Result<dit::Text> {
        let ids = self.tok.encode(prompt);
        let h = self.clip.encode(&self.ops, &ids)?;
        self.dit.text(&self.ops, &h)
    }

    /// The flow from `noise` [K, 64] to the sample [K, 64] (rescaled latents): the midpoint rule on the grid of
    /// `steps + 1` times, the rest channels clamped to f0. `trace` sees the state after each step.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, text: &dit::Text, f0: &DevBuf, noise: &[f32], steps: usize, cfg: f32, progress: &mut dyn FnMut(Step) -> Result<()>,
                  t0: Instant, trace: &mut dyn FnMut(usize, &DevBuf) -> Result<()>) -> Result<DevBuf> {
        let c = dit::CHANNELS;
        let k = noise.len() / c;
        let o = &self.ops;
        let w = self.dit.work(o, k)?;
        let y = DevBuf::from_f32(&o.gpu, noise)?;
        let (ym, x2, out, f1, f2) = (o.buf(k * c)?, o.buf(2 * k * c)?, o.buf(2 * k * c)?, o.buf(k * c)?, o.buf(k * c)?);
        let ts = linspace(steps + 1);
        let n = k * c;
        let flow = |y: &DevBuf, t: f32, f: &DevBuf| -> Result<()> {
            o.k.flow_in(f0, y, k, c, vae::LATENT, &x2)?;
            self.dit.forward(o, &w, text, &x2, t, &out)?;
            o.k.cfg(&out, n, cfg, f)
        };
        prof::reset_clock(&o.nsd)?;
        for i in 0..steps {
            let (ta, tb) = (ts[i], ts[i + 1]);
            let dt = tb - ta;
            let half = 0.5 * dt;
            flow(&y, ta, &f1)?;
            o.k.axpy(&y, &f1, half, n, &ym)?;
            flow(&ym, ta + half, &f2)?;
            o.k.axpy(&y, &f2, dt, n, &ym)?;
            y.copy_within(0, &ym, 0, n * 4)?;
            trace(i + 1, &y)?;
            prof::mark(&o.nsd, "solver")?;
            progress(Step { phase: "motion", at: i as u32 + 1, of: steps as u32, seconds: t0.elapsed().as_secs_f64() })?;
        }
        prof::report(&mut |s| eprintln!("{s}"));
        prof::used(&o.nsd, "at the flow's end (its buffers held)");
        Ok(y)
    }

    /// The decoded offsets [N, 48] of a sample [K, 64] (rescaled latents, as the flow leaves them)
    pub fn decode(&self, sh: &vae::Shape, sample: &[f32]) -> Result<Vec<f32>> {
        let c = dit::CHANNELS;
        let l = vae::LATENT;
        let k = sample.len() / c;
        let f = self.factors;
        let (mut x0, mut xt) = (Vec::with_capacity(k * l), Vec::with_capacity(k * l));
        for r in sample.chunks_exact(c) {
            x0.extend(r[..l].iter().map(|x| x * f.f0_std + f.f0_mean));
            xt.extend(r[l..].iter().map(|x| x * f.ft_std + f.ft_mean));
        }
        let x0 = DevBuf::from_f32(&self.ops.gpu, &x0)?;
        let xt = DevBuf::from_f32(&self.ops.gpu, &xt)?;
        self.vae.decode(&self.ops, sh, &x0, &xt, k)?.to_f32()
    }
}

/// torch.linspace(0, 1, n) in float32: the first half counted up from 0, the rest down from 1 (one rounding each:
/// PyTorch's vector kernel fuses the product and the sum)
pub fn linspace(n: usize) -> Vec<f32> {
    if n < 2 {
        return vec![0.0; n];
    }
    let step = 1.0f32 / (n - 1) as f32;
    let half = n / 2;
    (0..n).map(|i| if i < half { step * i as f32 } else { (-step).mul_add((n - 1 - i) as f32, 1.0) }).collect()
}

/// Frames resampled in time (linearly): `n` frames over the same span as `frames`
fn resample(frames: &[Vec<[f32; 3]>], n: usize) -> Vec<Vec<[f32; 3]>> {
    if n == frames.len() || frames.len() < 2 {
        return frames.to_vec();
    }
    (0..n).map(|j| {
        let t = if n > 1 { j as f32 * (frames.len() - 1) as f32 / (n - 1) as f32 } else { 0.0 };
        let a = (t.floor() as usize).min(frames.len() - 2);
        let w = t - a as f32;
        frames[a].iter().zip(&frames[a + 1]).map(|(p, q)| [0, 1, 2].map(|c| p[c] + (q[c] - p[c]) * w)).collect()
    }).collect()
}

impl Model3dEngine for AnimateAnyMesh {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn abilities(&self) -> Abilities {
        Abilities { animate: true, ..Default::default() }
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: 63, cfg: 3.0, frames: vae::FRAMES as u32, fps: 10.0, ..Default::default() }
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
            .ok_or_else(|| Error(format!("{ARCH} sets a mesh in motion from a description: give a prompt (\"a dragon flapping its wings\")")))?;
        let d = self.defaults();
        let steps = req.steps.unwrap_or(d.steps).clamp(1, 1000) as usize;
        let cfg = req.cfg.unwrap_or(d.cfg);
        let frames = req.frames.unwrap_or(d.frames).clamp(2, 240) as usize;
        let fps = req.fps.unwrap_or(d.fps);
        if req.mesh.positions.is_empty() || req.mesh.indices.is_empty() {
            return Err(Error("an empty mesh".into()));
        }
        let p = prep::prepare(&req.mesh);
        let n = p.verts.len();
        let k = match req.extra.get("traj") {
            Some(v) => v.parse::<usize>().map_err(|_| Error(format!("traj: {v:?} is not a count")))?,
            None => self.traj.unwrap_or_else(|| MIN_TRAJ.max(n / 8)),
        }.max(1);
        progress(Step { phase: "encode", at: 0, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let (sh, _) = self.shape(&p)?;
        let (_, idx) = self.sample_tokens(&sh, k, req.seed)?;
        let f0 = self.f0(&sh, &idx)?;
        let text = self.text(prompt)?;
        progress(Step { phase: "encode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let noise = ops::noise(req.seed, k * dit::CHANNELS);
        let y = self.sample(&text, &f0, &noise, steps, cfg, progress, t0, &mut |_, _| Ok(()))?;
        // the result: f0 for the rest channels, the flow's for the motion
        let mut s = y.to_f32()?;
        let f0h = f0.to_f32()?;
        for (r, f) in s.chunks_exact_mut(dit::CHANNELS).zip(f0h.chunks_exact(vae::LATENT)) {
            r[..vae::LATENT].copy_from_slice(f);
        }
        let off = self.decode(&sh, &s)?;
        let mut out = Vec::with_capacity(vae::FRAMES);
        for t in 0..vae::FRAMES {
            let merged: Vec<[f32; 3]> = (0..n).map(|v| {
                let o = &off[v * 3 * vae::FRAMES + 3 * t..];
                [0, 1, 2].map(|c| p.verts[v][c] + o[c])
            }).collect();
            out.push(p.map.iter().map(|&m| p.unmap(merged[m as usize])).collect::<Vec<_>>());
        }
        progress(Step { phase: "decode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let mut m = req.mesh.clone();
        m.animation = Some(Morph { fps, frames: resample(&out, frames) });
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
    fn the_time_grid_is_torchs() {
        let t = linspace(64);
        assert_eq!(t[0], 0.0);
        assert_eq!(t[63], 1.0);
        assert_eq!(t[1], 1.0 / 63.0);
        assert_eq!(t[62], 1.0 - 1.0 / 63.0);
        // where a separate product and sum round twice, torch's (and this) rounds once
        assert_eq!(t[41], 0.650_793_6);
    }

    #[test]
    fn resampling_keeps_the_ends() {
        let f: Vec<Vec<[f32; 3]>> = (0..16).map(|i| vec![[i as f32, 0.0, 0.0]]).collect();
        let r = resample(&f, 31);
        assert_eq!(r.len(), 31);
        assert_eq!(r[0][0][0], 0.0);
        assert_eq!(r[30][0][0], 15.0);
        assert!((r[1][0][0] - 0.5).abs() < 1e-6);
    }
}
