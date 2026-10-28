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
pub mod reskin;
pub mod ptv3;
pub mod search;
pub mod skin;
pub mod tokens;
pub mod voxel;

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::mesh::{tools, Joint, Mesh, Skin};
use nextsycl_model3d::{At, EngineOption, Error, LoadOptions, Model3dEngine, Model3dKind, ModelFiles, Result, RigRequest, Step};

use ops::{Ops, Rng, Weights};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "unirig";

/// The files it needs, by role: the skeleton model's and the skin model's Lightning checkpoints
pub const ROLES: &[&str] = &["skeleton", "skin"];

pub const OPTIONS: &[EngineOption] = &[EngineOption {
    name: "beams",
    env: "NS_M3D_UNIRIG_BEAMS",
    value: "N",
    help: "the skeleton decoder's beams (default 15, sampled, as UniRig's inference; 1: greedy - the same skeleton every time)",
    at: At::Both,
}];

pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "UniRig (a skeleton from an autoregressive decoder, skin weights from Point Transformer V3)", roles: ROLES, load,
                  options: OPTIONS }
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
    pub skin: skin::SkinModel,
    beams: usize,
    load_s: f64,
}

/// A rig in UniRig's frames: the skeleton (Blender's frame, the mesh's scale) and the weights of the merged mesh's
/// vertices [V, J]
pub struct Rig {
    pub joints: Vec<[f32; 3]>,
    pub parents: Vec<Option<usize>>,
    pub weights: Vec<f32>,
}

fn open(files: &ModelFiles, role: &str) -> Result<Pth> {
    let p = files.get(role).ok_or_else(|| Error(format!("{ARCH}: no {role} file")))?;
    Pth::open(p).map_err(|e| Error(e.0))
}

impl UniRig {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, lo: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<UniRig> {
        let t0 = Instant::now();
        let ops = Ops::new(gpu)?;
        let sk = open(files, "skeleton")?;
        let pre = sk.prefix_of("model.output_proj.weight").map(|p| format!("{p}model."))
            .ok_or_else(|| Error(format!("{}: not UniRig's skeleton checkpoint (no model.output_proj)", sk.path.display())))?;
        let w = Weights { pth: &sk, prefix: pre.clone(), gpu };
        let enc = encoder::Encoder::load(&Weights { pth: &sk, prefix: format!("{pre}mesh_encoder."), gpu })?;
        let opt = opt::Opt::load(&w)?;
        let sp = open(files, "skin")?;
        let pre = sp.prefix_of("model.qmesh.weight").map(|p| format!("{p}model."))
            .ok_or_else(|| Error(format!("{}: not UniRig's skin checkpoint (no model.qmesh)", sp.path.display())))?;
        let skin = skin::SkinModel::load(&Weights { pth: &sp, prefix: pre, gpu })?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let silo = ops.k.silo().map_or_else(|| "no silo: the engine's own kernels".to_string(), |s| format!("silo {}", s.path.display()));
        log(format!("{ARCH}: loaded on {} in {load_s:.1} s (float32; {silo})", gpu.name));
        let beams = lo.setting("NS_M3D_UNIRIG_BEAMS").map(|v| v.parse::<usize>().map_err(|_| Error(format!("NS_M3D_UNIRIG_BEAMS: {v:?}?")))).transpose()?
            .unwrap_or(15).max(1);
        Ok(UniRig { ops, enc, opt, skin, beams, load_s })
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
        // the pool: the prefix's slots, then one a beam a step - grown as the sequences do, as far as the card holds
        // (1.5 GiB kept free)
        let free = o.gpu.memory()?.1.unwrap_or(8 << 30) as usize;
        let room = free.saturating_sub(3 << 29) / opt::Opt::pool_bytes(1);
        let most = (l0 + nb * max_new).min(room);
        if most < l0 + nb * 8 {
            return Err(Error(format!("not enough free memory on {} for the decoder's cache", o.gpu.name)));
        }
        let max_new = max_new.min((most - l0) / nb);
        let mut pool = self.opt.pool(o, (l0 + nb * 192).min(most))?;
        let x = o.buf(l0 * opt::WIDTH)?;
        x.copy_within(0, prefix, 0, l0 * opt::WIDTH * 4)?;
        let first = self.opt.prefill(o, &x, l0, &mut pool, 1)?;
        drop(x);
        prof::mark(&o.nsd, "decoder prefill")?;
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
                if pool.used + 1 > pool.cap {
                    let cap = (pool.cap * 2).min(most);
                self.opt.grow(o, &mut pool, cap)?;
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
            prof::mark(&o.nsd, "decoder search (host)")?;
            progress(beams.running[0].len())?;
            if beams.done {
                prof::used(&o.nsd, "at the skeleton decoder's end");
                prof::report(&mut |s| eprintln!("{s}"));
                return Ok(beams.best());
            }
            if pool.used + nb > pool.cap {
                let cap = (pool.cap * 2).min(most);
                self.opt.grow(o, &mut pool, cap)?;
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

impl UniRig {
    /// The whole rig of a merged mesh (Blender's frame): its skeleton, then its weights
    pub fn rig_shape(&self, sh: &prep::Shape, beams: usize, rng: &mut Rng, progress: &mut dyn FnMut(Step) -> Result<()>, t0: Instant) -> Result<Rig> {
        let at = |phase: &'static str, a: usize, of: usize| Step { phase, at: a as u32, of: of as u32, seconds: t0.elapsed().as_secs_f64() };
        // the skeleton
        let (c, s) = prep::normalization(sh.v.iter().copied());
        let nv: Vec<[f32; 3]> = sh.v.iter().map(|p| prep::apply(*p, c, s)).collect();
        let sm = prep::sample(&nv, &sh.vn, &sh.f, &sh.fn_, SKELETON_POINTS, VERTEX_SAMPLES, rng);
        let (ind, fps) = encoder::queries(&sm.pts, rng);
        let q: Vec<usize> = fps.iter().map(|&i| ind[i]).collect();
        progress(at("skeleton", 0, 1))?;
        prof::reset_clock(&self.ops.nsd)?;
        let (_, prefix) = self.prefix(&sm.pts, &sm.nrm, &q)?;
        prof::mark(&self.ops.nsd, "skeleton encoder")?;
        let st = if beams <= 1 { search::Settings { beams: 1, sample: false, ..Default::default() } } else { search::Settings { beams, ..Default::default() } };
        let gen = self.generate(&prefix, st, rng, &mut |n| progress(at("skeleton", n, n + 1)))?;
        drop(prefix);
        let mut ids = START.to_vec();
        ids.extend(&gen);
        if ids.last() != Some(&tokens::EOS) {
            ids.push(tokens::EOS);
        }
        let sk = tokens::detokenize(&ids).map_err(|e| Error(format!("the decoder's skeleton: {e}")))?;
        let joints: Vec<[f32; 3]> = sk.joints.iter().map(|j| [0, 1, 2].map(|k| j[k] * s + c[k])).collect();
        let nj = joints.len();
        progress(at("skin", 0, 3))?;
        // the skin: its frame takes the joints too
        let (c2, s2) = prep::normalization(sh.v.iter().copied().chain(joints.iter().copied()));
        let nv2: Vec<[f32; 3]> = sh.v.iter().map(|p| prep::apply(*p, c2, s2)).collect();
        let nj2: Vec<[f32; 3]> = joints.iter().map(|p| prep::apply(*p, c2, s2)).collect();
        let prior = voxel::voxel_skin(&nv2, &sh.f, &nj2, 196);
        progress(at("skin", 1, 3))?;
        let sm2 = prep::sample(&nv2, &sh.vn, &sh.f, &sh.fn_, SKIN_POINTS, VERTEX_SAMPLES, rng);
        let prior_s = sm2.carry(&sh.f, &prior, nj);
        let (ind2, fps2) = encoder::queries(&sm2.pts, rng);
        let q2: Vec<usize> = fps2.iter().map(|&i| ind2[i]).collect();
        let mut perm = [0usize, 1, 2, 3];
        for i in (1..4).rev() {
            perm.swap(i, ((rng.unit() * (i + 1) as f64) as usize).min(i));
        }
        let gc = ptv3::grid_coords(&sm2.pts);
        let lay = ptv3::Layout { orders: ptv3::orders(&gc, perm), gc };
        let soft = self.skin.predict(&self.ops, &sm2.pts, &sm2.nrm, &q2, &lay, &nj2, &prior_s, None)?;
        let w = skin::masked(&soft, &prior_s, &sk.parents, 3.0);
        progress(at("skin", 2, 3))?;
        let weights = reskin::reskin(&sm2.pts, &w, &nv2, &sh.f, &sk.parents);
        progress(at("skin", 3, 3))?;
        Ok(Rig { joints, parents: sk.parents, weights })
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
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn rig(&self, req: &RigRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        let t0 = Instant::now();
        if req.mesh.positions.is_empty() || req.mesh.indices.len() < 3 {
            return Err(Error("an empty mesh".into()));
        }
        let beams = match req.extra.get("beams") {
            Some(v) => v.parse::<usize>().map_err(|_| Error(format!("beams: {v:?}?")))?.max(1),
            None => self.beams,
        };
        let sh = prep::shape(&req.mesh, 50_000);
        let mut rng = Rng::new(req.seed);
        let rig = self.rig_shape(&sh, beams, &mut rng, progress, t0)?;
        let nj = rig.joints.len();
        // the weights of the user's own vertices: by the merge's map, or (a decimated copy) by closest points
        let dense: Vec<f32> = if sh.decimated {
            let idx: Vec<u32> = sh.f.iter().flatten().copied().collect();
            let targets: Vec<[f32; 3]> = req.mesh.positions.iter().map(|p| prep::to_blender(*p)).collect();
            tools::skin::transfer(&sh.v, &idx, &rig.weights, nj, &targets)
        } else {
            sh.map.iter().flat_map(|&m| rig.weights[m as usize * nj..(m as usize + 1) * nj].iter().copied()).collect()
        };
        let joints = rig.joints.iter().zip(&rig.parents).enumerate()
            .map(|(i, (j, p))| Joint { name: format!("bone_{i}"), parent: *p, head: prep::from_blender(*j) }).collect();
        let mut m = req.mesh.clone();
        m.skin = Some(Skin { joints, weights: tools::skin::top4(&dense, nj), motion: None });
        if m.normals.len() != m.positions.len() {
            m.compute_normals();
        }
        m.validate().map_err(Error)?;
        Ok(m)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {}", self.ops.gpu.name)]
    }
}
