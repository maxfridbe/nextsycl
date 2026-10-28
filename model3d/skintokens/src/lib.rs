//! SkinTokens / TokenRig (VAST-AI, 2026; github.com/VAST-AI-Research/SkinTokens, MIT): a skeleton and its skin weights for a
//! mesh, written by one autoregressive decoder as a single token sequence - float32 throughout.
//!
//! The pipeline (demo.py, src/model/tokenrig.py):
//! 1. the mesh in Blender's frame, normalized into [-1, 1], 54,000 samples (`prep`): 16,384 of its vertices and the rest on
//!    its surface, each with a normal;
//! 2. Michelangelo's shape encoder (`encoder`): 512 latents, through `output_proj` into the decoder's width;
//! 3. a Qwen3-shaped decoder (`qwen`) over those latents and the start tokens writes the skeleton's tokens (`tokens`: the
//!    joints' positions in 256 bins, branches, ends), then four skin tokens a joint - codes of the skin VAE's FSQ codebook -
//!    by beam search with sampling (`search`);
//! 4. the skin VAE (`vae`): its condition encoder reads the samples, its decoder turns each joint's four codes into that
//!    joint's weight at every sample; the weights are carried to the mesh's vertices by the 8 nearest samples (`knn`), four
//!    a vertex.

pub mod check;
pub mod encoder;
mod ffi;
pub mod knn;
pub mod ops;
pub mod prep;
pub mod qwen;
pub mod search;
pub mod tokens;
pub mod vae;

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::mesh::{tools, Joint, Mesh, Skin};
use nextsycl_model3d::{At, EngineOption, Error, LoadOptions, Model3dEngine, Model3dKind, ModelFiles, Result, RigRequest, Step};

use ops::{Ops, Rng, Weights};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "skintokens";

/// The files it needs, by role: the TokenRig checkpoint (which carries the skin VAE too). The standalone skin VAE checkpoint is
/// the same weights (bfloat16-rounded in the first), so it is optional and unused at inference.
pub const ROLES: &[&str] = &["tokenrig", "skinvae?"];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption {
        name: "beams",
        env: "NS_M3D_SKINTOKENS_BEAMS",
        value: "N",
        help: "the decoder's beams (default 10, sampled, as the demo; 1: greedy - the same rig every time)",
        at: At::Both,
    },
    EngineOption {
        name: "skin-end",
        env: "NS_M3D_SKINTOKENS_SKIN_END",
        value: "upstream|fix",
        help: "where the skin tokens end: upstream (default) ends them one token early, so the last joint's fourth token is the end token and reads as code 0; fix lets the model say it",
        at: At::Both,
    },
];

pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "SkinTokens (TokenRig: a skeleton and skin weights as one token sequence)", roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(SkinTokens::load(files, gpu, o, log)?))
}

/// The surface samples every stage reads
pub const POINTS: usize = 54_000;
pub const VERTEX_SAMPLES: usize = 16_384;
/// The nearest samples a vertex's weights come from
pub const NEIGHBOURS: usize = 8;
/// The tokens every sequence starts with: bos, and the class the demo asks for (articulation)
pub const START: [u32; 2] = [tokens::BOS, tokens::CLS_ARTICULATION];

pub struct SkinTokens {
    pub ops: Ops,
    pub enc: encoder::Encoder,
    pub qwen: qwen::Qwen,
    pub vae: vae::Vae,
    beams: usize,
    fix_end: bool,
    load_s: f64,
}

/// A rig in TokenRig's frames: the skeleton (Blender's frame, the mesh's scale) and the weights of the mesh's vertices [V, J]
pub struct Rig {
    pub joints: Vec<[f32; 3]>,
    pub parents: Vec<Option<usize>>,
    pub weights: Vec<f32>,
}

fn parse_end(v: &str) -> Result<bool> {
    match v {
        "upstream" => Ok(false),
        "fix" => Ok(true),
        _ => Err(Error(format!("skin-end: {v:?}? (upstream or fix)"))),
    }
}

impl SkinTokens {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, lo: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<SkinTokens> {
        let t0 = Instant::now();
        let ops = Ops::new(gpu)?;
        let p = files.get("tokenrig").ok_or_else(|| Error(format!("{ARCH}: no tokenrig file")))?;
        let pth = Pth::open(p).map_err(|e| Error(e.0))?;
        let pre = pth.prefix_of("output_proj.0.weight").ok_or_else(|| Error(format!("{}: not TokenRig's checkpoint (no output_proj)", pth.path.display())))?;
        let silo = ops.k.silo().is_some();
        let w = |sub: &str| Weights { pth: &pth, prefix: format!("{pre}{sub}"), gpu, silo };
        let enc = encoder::Encoder::load(&w(""))?;
        let qwen = qwen::Qwen::load(&w(""))?;
        let vae = vae::Vae::load(&w("vae.model."))?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let silo = ops.k.silo().map_or_else(|| "no silo: the engine's own kernels".to_string(), |s| format!("silo {}", s.path.display()));
        log(format!("{ARCH}: loaded on {} in {load_s:.1} s (float32; {silo})", gpu.name));
        let beams = lo.setting("NS_M3D_SKINTOKENS_BEAMS").map(|v| v.parse::<usize>().map_err(|_| Error(format!("NS_M3D_SKINTOKENS_BEAMS: {v:?}?")))).transpose()?
            .unwrap_or(10).max(1);
        let fix_end = lo.setting("NS_M3D_SKINTOKENS_SKIN_END").map(|v| parse_end(&v)).transpose()?.unwrap_or(false);
        Ok(SkinTokens { ops, enc, qwen, vae, beams, fix_end, load_s })
    }

    /// The decoder's prefix [512 + 2, 896]: the encoder's latents through output_proj, then bos and the class token
    pub fn prefix(&self, pts: &[[f32; 3]], nrm: &[[f32; 3]], queries: &[usize]) -> Result<(DevBuf, DevBuf)> {
        let o = &self.ops;
        let lat = self.enc.encode(o, pts, nrm, queries)?;
        let t = encoder::TOKENS;
        let c = self.qwen.project(o, &lat, t)?;
        let x = o.buf((t + START.len()) * qwen::WIDTH)?;
        x.copy_within(0, &c, 0, t * qwen::WIDTH * 4)?;
        let s = self.qwen.embed(o, &START)?;
        x.copy_within(t * qwen::WIDTH * 4, &s, 0, START.len() * qwen::WIDTH * 4)?;
        Ok((lat, x))
    }

    /// The logits [tokens.len() + 1, 33036] of the prefix's last position and every token after it (teacher forcing)
    pub fn logits(&self, prefix: &DevBuf, tokens: &[u32]) -> Result<Vec<f32>> {
        let o = &self.ops;
        let l0 = encoder::TOKENS + START.len();
        let l = l0 + tokens.len();
        let x = o.buf(l * qwen::WIDTH)?;
        x.copy_within(0, prefix, 0, l0 * qwen::WIDTH * 4)?;
        if !tokens.is_empty() {
            let e = self.qwen.embed(o, tokens)?;
            x.copy_within(l0 * qwen::WIDTH * 4, &e, 0, tokens.len() * qwen::WIDTH * 4)?;
        }
        let mut pool = self.qwen.pool(o, l)?;
        self.qwen.prefill(o, &x, l, &mut pool, tokens.len() + 1)
    }

    /// The generated tokens (the model's eos last) after the prefix: greedy (`s.beams` 1, no sampling) or beam search
    pub fn generate(&self, prefix: &DevBuf, s: search::Settings, rng: &mut Rng, progress: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<u32>> {
        let o = &self.ops;
        let l0 = encoder::TOKENS + START.len();
        let nb = s.beams.max(1);
        let max_new = s.max_new.min(self.qwen.max_positions() - l0);
        // the pool: the prefix's slots, then one a beam a step - grown as the sequences do, as far as the card holds
        // (1.5 GiB kept free)
        let free = o.gpu.memory()?.1.unwrap_or(8 << 30) as usize;
        let room = free.saturating_sub(3 << 29) / qwen::Qwen::pool_bytes(1);
        let most = (l0 + nb * max_new).min(room);
        if most < l0 + nb * 8 {
            return Err(Error(format!("not enough free memory on {} for the decoder's cache", o.gpu.name)));
        }
        let max_new = max_new.min((most - l0) / nb);
        let mut pool = self.qwen.pool(o, (l0 + nb * 192).min(most))?;
        let x = o.buf(l0 * qwen::WIDTH)?;
        x.copy_within(0, prefix, 0, l0 * qwen::WIDTH * 4)?;
        let first = self.qwen.prefill(o, &x, l0, &mut pool, 1)?;
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
                if t == tokens::EOS_MODEL || seq.len() >= max_new {
                    return Ok(seq);
                }
                if pool.used + 1 > pool.cap {
                    let cap = (pool.cap * 2).min(most);
                    self.qwen.grow(o, &mut pool, cap)?;
                }
                hist[0].push(pool.used as i32);
                slots.write(0, &ops::i32_bytes(&hist[0]))?;
                logits = self.qwen.step(o, &[t], l0 + seq.len() - 1, &mut pool, &slots, stride)?;
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
            if pool.used + nb > pool.cap {
                let cap = (pool.cap * 2).min(most);
                self.qwen.grow(o, &mut pool, cap)?;
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
            logits = self.qwen.step(o, &st.tokens, pos, &mut pool, &slots, stride)?;
        }
    }

    /// The skeleton and the skin tokens of a generated sequence: (the skeleton, the skin tokens' FSQ codes [4 J])
    pub fn read(&self, gen: &[u32]) -> Result<(tokens::Skeleton, Vec<u32>)> {
        let mut ids = START.to_vec();
        ids.extend(gen);
        let w = ids.iter().position(|&t| t == tokens::EOS).ok_or_else(|| Error("the decoder never ended the skeleton".into()))?;
        let sk = tokens::detokenize(&ids[..=w]).map_err(|e| Error(format!("the decoder's skeleton: {e}")))?;
        let j = sk.joints.len();
        let skin = &ids[w + 1..];
        if skin.len() < j * tokens::PER_JOINT {
            return Err(Error(format!("the decoder ended with {} skin tokens, a rig of {j} joints needs {}", skin.len(), j * tokens::PER_JOINT)));
        }
        // a code is the token less the skeleton's tokens; the model's own eos (as the last token) reads as code 32,768 = 0
        Ok((sk, skin[..j * tokens::PER_JOINT].iter().map(|&t| t.saturating_sub(tokens::SKIN0)).collect()))
    }

    /// The weights [N, J] of the surface samples for the skin codes of `j` joints
    pub fn decode_skin(&self, codes: &[u32], j: usize, pts: &[[f32; 3]], nrm: &[[f32; 3]], rng: &mut Rng, progress: &mut dyn FnMut(usize) -> Result<()>)
                       -> Result<Vec<f32>> {
        let o = &self.ops;
        let (ind, fps) = encoder::subset_fps(pts, vae::COND_SUBSET, vae::COND_TOKENS, rng);
        let cond = self.vae.cond(o, pts, nrm, &ind, &fps)?;
        let z = self.vae.codes(codes);
        Ok(self.vae.decode(o, &z, j, &cond, pts, nrm, progress)?.0)
    }

    /// The whole rig of a mesh (Blender's frame)
    pub fn rig_shape(&self, sh: &prep::Shape, s: search::Settings, rng: &mut Rng, progress: &mut dyn FnMut(Step) -> Result<()>, t0: Instant) -> Result<Rig> {
        let at = |phase: &'static str, a: usize, of: usize| Step { phase, at: a as u32, of: of as u32, seconds: t0.elapsed().as_secs_f64() };
        let (c, sc) = prep::normalization(sh.v.iter().copied());
        let nv: Vec<[f32; 3]> = sh.v.iter().map(|p| prep::apply(*p, c, sc)).collect();
        let sm = prep::sample(&nv, &sh.vn, &sh.f, &sh.fn_, POINTS, VERTEX_SAMPLES, rng);
        let (ind, fps) = encoder::queries(&sm.pts, rng);
        let q: Vec<usize> = fps.iter().map(|&i| ind[i]).collect();
        progress(at("skeleton", 0, 1))?;
        let (_, prefix) = self.prefix(&sm.pts, &sm.nrm, &q)?;
        let gen = self.generate(&prefix, s, rng, &mut |n| progress(at("skeleton", n, n + 1)))?;
        drop(prefix);
        let (sk, codes) = self.read(&gen)?;
        let j = sk.joints.len();
        let joints: Vec<[f32; 3]> = sk.joints.iter().map(|p| [0, 1, 2].map(|k| p[k] * sc + c[k])).collect();
        progress(at("skin", 0, j))?;
        let skin = self.decode_skin(&codes, j, &sm.pts, &sm.nrm, rng, &mut |n| progress(at("skin", n, j)))?;
        let weights = knn::carry(&sm.pts, &skin, j, &nv, NEIGHBOURS);
        Ok(Rig { joints, parents: sk.parents, weights })
    }
}

impl Model3dEngine for SkinTokens {
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
        let fix_end = match req.extra.get("skin-end") {
            Some(v) => parse_end(v)?,
            None => self.fix_end,
        };
        let s = if beams <= 1 { search::Settings { beams: 1, sample: false, fix_end, ..Default::default() } } else { search::Settings { beams, fix_end, ..Default::default() } };
        let sh = prep::shape(&req.mesh);
        let mut rng = Rng::new(req.seed);
        let rig = self.rig_shape(&sh, s, &mut rng, progress, t0)?;
        let nj = rig.joints.len();
        let joints = rig.joints.iter().zip(&rig.parents).enumerate()
            .map(|(i, (j, p))| Joint { name: format!("bone_{i}"), parent: *p, head: prep::from_blender(*j) }).collect();
        let mut m = req.mesh.clone();
        m.skin = Some(Skin { joints, weights: tools::skin::top4(&rig.weights, nj) });
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
