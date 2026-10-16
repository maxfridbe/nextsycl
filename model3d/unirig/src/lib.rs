//! UniRig (VAST-AI, SIGGRAPH 2025; github.com/VAST-AI-Research/UniRig, MIT): a skeleton and skin weights for a mesh.
//!
//! The pipeline (launch/inference/generate_skeleton.sh, generate_skin.sh, merge.sh), float32 throughout:
//! 1. the mesh in Blender's frame, merged, normals (`prep`);
//! 2. the skeleton: 65,536 samples, Michelangelo's shape encoder (`encoder`), the OPT-350m-shaped decoder (`opt`) over
//!    UniRig's skeleton tokens with their grammar (`tokens`), beam search with sampling (`search`), the tokens read
//!    back into joints and parents;
//! 3. the skin: the voxel prior (`voxel`: a solid voxelization and geodesic distances from each joint), 32,768
//!    samples, Point Transformer V3 (`ptv3`), the bone encoder, mesh-bone attention and the skin head (`skin`);
//! 4. the samples' weights carried to the mesh's vertices (`reskin`: the median of the 7 nearest samples, a diffusion
//!    over the mesh's edges), then to the user's own vertices; four a vertex.

pub mod check;
pub mod encoder;
mod ffi;
pub mod ops;
pub mod opt;
pub mod prep;
pub mod prof;
pub mod search;
pub mod tokens;

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::{Error, LoadOptions, Model3dEngine, Model3dKind, ModelFiles, Result};

use ops::{Ops, Rng, Weights};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "unirig";

/// The files it needs, by role: the skeleton model's and the skin model's Lightning checkpoints
pub const ROLES: &[&str] = &["skeleton", "skin"];

pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "UniRig (a skeleton from an autoregressive decoder, skin weights from Point Transformer V3)", roles: ROLES, load,
                  options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(UniRig::load(files, gpu, o, log)?))
}

/// The skeleton stage's points
pub const SKELETON_POINTS: usize = 65_536;
pub const SKIN_POINTS: usize = 32_768;
pub const VERTEX_SAMPLES: usize = 8_192;

pub struct UniRig {
    pub ops: Ops,
    pub enc: encoder::Encoder,
    pub opt: opt::Opt,
    load_s: f64,
}

fn open(files: &ModelFiles, role: &str) -> Result<Pth> {
    let p = files.get(role).ok_or_else(|| Error(format!("{ARCH}: no {role} file")))?;
    Pth::open(p).map_err(|e| Error(e.0))
}

impl UniRig {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<UniRig> {
        let t0 = Instant::now();
        let ops = Ops::new(gpu)?;
        let sk = open(files, "skeleton")?;
        let pre = sk.prefix_of("model.output_proj.weight").map(|p| format!("{p}model."))
            .ok_or_else(|| Error(format!("{}: not UniRig's skeleton checkpoint (no model.output_proj)", sk.path.display())))?;
        let w = Weights { pth: &sk, prefix: pre.clone(), gpu };
        let enc = encoder::Encoder::load(&Weights { pth: &sk, prefix: format!("{pre}mesh_encoder."), gpu })?;
        let opt = opt::Opt::load(&w)?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let silo = ops.k.silo().map_or_else(|| "no silo: the engine's own kernels".to_string(), |s| format!("silo {}", s.path.display()));
        log(format!("{ARCH}: loaded on {} in {load_s:.1} s (float32; {silo})", gpu.name));
        Ok(UniRig { ops, enc, opt, load_s })
    }

    /// The decoder's prefix [1024 + 2, 1024]: the encoder's latents through output_proj, then bos and the class token
    pub fn prefix(&self, pts: &[[f32; 3]], nrm: &[[f32; 3]], queries: &[usize]) -> Result<(DevBuf, DevBuf)> {
        let o = &self.ops;
        let lat = self.enc.encode(o, pts, nrm, queries)?;
        let t = encoder::TOKENS;
        let x = o.buf((t + 2) * opt::WIDTH)?;
        o.linear(&lat, t, &self.opt.output_proj, &x)?;
        let s = self.opt.embed(o, &START)?;
        x.copy_within(t * opt::WIDTH * 4, &s, 0, 2 * opt::WIDTH * 4)?;
        Ok((lat, x))
    }

    /// The logits [tokens.len() + 1, 267] of the prefix's last position and every token after it (teacher forcing)
    pub fn logits(&self, prefix: &DevBuf, tokens: &[u32]) -> Result<Vec<f32>> {
        let o = &self.ops;
        let l0 = encoder::TOKENS + START.len();
        let l = l0 + tokens.len();
        let x = o.buf(l * opt::WIDTH)?;
        x.copy_within(0, prefix, 0, l0 * opt::WIDTH * 4)?;
        if !tokens.is_empty() {
            let e = self.opt.embed(o, tokens)?;
            x.copy_within(l0 * opt::WIDTH * 4, &e, 0, tokens.len() * opt::WIDTH * 4)?;
        }
        let mut pool = self.opt.pool(o, l)?;
        self.opt.prefill(o, &x, l, &mut pool, tokens.len() + 1)
    }

    /// The generated tokens (eos last) after the prefix: greedy (`s.beams` 1, no sampling) or beam search
    pub fn generate(&self, prefix: &DevBuf, s: search::Settings, rng: &mut Rng, progress: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<u32>> {
        let o = &self.ops;
        let l0 = encoder::TOKENS + START.len();
        let nb = s.beams.max(1);
        let max_new = s.max_new.min(self.opt.max_positions() - l0);
        // the pool: the prefix's slots, then one a beam a step - as many steps as the card holds (1.5 GiB kept free)
        let free = o.gpu.memory()?.1.unwrap_or(8 << 30) as usize;
        let room = free.saturating_sub(3 << 29) / opt::Opt::pool_bytes(1);
        let cap = (l0 + nb * max_new).min(room);
        if cap < l0 + nb * 8 {
            return Err(Error(format!("not enough free memory on {} for the decoder's cache", o.gpu.name)));
        }
        let max_new = max_new.min((cap - l0) / nb);
        let mut pool = self.opt.pool(o, cap)?;
        let x = o.buf(l0 * opt::WIDTH)?;
        x.copy_within(0, prefix, 0, l0 * opt::WIDTH * 4)?;
        let first = self.opt.prefill(o, &x, l0, &mut pool, 1)?;
        drop(x);
        let stride = l0 + max_new + 1;
        // each beam's slots: the prefix's, then its own
        let mut hist: Vec<Vec<i32>> = vec![(0..l0 as i32).collect(); nb];
        let slots = DevBuf::new(&o.gpu, nb * stride * 4)?;
        if nb == 1 && !s.sample {
            let mut seq: Vec<u32> = Vec::new();
            let mut logits = first;
            loop {
                let t = search::greedy(&logits, &START, &seq, &s);
                seq.push(t);
                progress(seq.len())?;
                if t == tokens::EOS || seq.len() >= max_new {
                    return Ok(seq);
                }
                hist[0].push(pool.used as i32);
                slots.write(0, &ops::i32_bytes(&hist[0]))?;
                logits = self.opt.step(o, &[t], l0 + seq.len() - 1, &mut pool, &slots, stride)?;
            }
        }
        let mut beams = search::Beams::new(search::Settings { max_new, ..s }, &START);
        let mut logits: Vec<f32> = (0..nb).flat_map(|_| first.iter().copied()).collect();
        loop {
            let st = beams.step(&logits, rng);
            progress(beams.running[0].len())?;
            if beams.done {
                return Ok(beams.best());
            }
            // the new running beams follow their parents' slots, then a slot of their own
            let old = hist.clone();
            let base = pool.used as i32;
            for (b, &p) in st.parents.iter().enumerate() {
                hist[b] = old[p].clone();
                hist[b].push(base + b as i32);
            }
            let pos = l0 + beams.running[0].len() - 1;
            let mut flat = vec![0i32; nb * stride];
            for (b, h) in hist.iter().enumerate() {
                flat[b * stride..b * stride + h.len()].copy_from_slice(h);
            }
            slots.write(0, &ops::i32_bytes(&flat))?;
            logits = self.opt.step(o, &st.tokens, pos, &mut pool, &slots, stride)?;
        }
    }
}

/// The tokens every skeleton starts with: bos, and the class UniRig's inference asks for (articulationxl)
pub const START: [u32; 2] = [tokens::BOS, tokens::CLS_ARTICULATIONXL];

impl Model3dEngine for UniRig {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn abilities(&self) -> nextsycl_model3d::Abilities {
        nextsycl_model3d::Abilities { rig: true, ..Default::default() }
    }
    fn defaults(&self) -> nextsycl_model3d::Defaults {
        nextsycl_model3d::Defaults::default()
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
}
