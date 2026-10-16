//! Hunyuan3D 2.1's paint model (`hunyuan3d-paintpbr-v2-1`): PBR textures for a mesh from the picture it was made
//! from. The mesh is seen from six fixed cameras; its normal and position maps there (`render`) and the picture
//! condition a multiview diffusion (`unet`: Stable Diffusion 2.1 grown into a 2.5D, two-material model; `sdvae`;
//! `unipc`) that draws the six views twice - albedo and metallic-roughness - at 512 px; the views are baked back into
//! a UV atlas and the unseen texels filled (`bake`).
//!
//! The pipeline, as Tencent's textureGenPipeline (with its multiviewDiffusionNet and HunyuanPaintPipeline):
//!
//! ```text
//!   maps   6 x (normal, position) rendered at 2,048, resized (Pillow, bicubic) to 512; the picture resized to 512
//!          on white; DINOv2-giant of the picture at 224 (resize 256, centre crop), projected to 1,028 tokens
//!   latents of the picture, the normal and the position maps (the VAE encoder's mean)
//!   reference: unet_dual on the picture's latent at t = 0 -> every block's normalized input
//!   15 UniPC steps: the UNet on [latent | normal | position] x (material, branch, view) = 36 pictures;
//!          v = u + 3 (ref - u) + 3 (full - ref)
//!   decode 12 views; bake albedo and metallic-roughness (no RealESRGAN: the 512 px views resized to 2,048)
//! ```

pub mod bake;
pub mod nn;
pub mod pil;
pub mod render;
pub mod sdvae;
pub mod unet;
pub mod unipc;

use std::collections::BTreeMap;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_mesh::{Material, Mesh, Texture};

use crate::dino::Dino;
use crate::ops::{read_f32, Lin, Ops, Weights};
use nn::{Map, Mem};
use unet::{Cond, Unet, BRANCHES, MATS, VIEWS};

pub const SIZE: usize = 512;
const LAT: usize = SIZE / 8;
const RENDER: usize = 2048;
const GUIDANCE: f32 = 3.0;
const DINO_SIZE: usize = 224;
/// arena sizes: a pass's kept maps and a block's temporaries
const KEEP: usize = 3 << 30;
const TMP: usize = 4 << 30;

pub struct Paint {
    pub unet: Unet,
    pub dual: Unet,
    pub vae: sdvae::SdVae,
    pub dino: Dino,
    proj: Lin,
    proj_norm: (DevBuf, DevBuf),
    /// the learned text embeddings: albedo, metallic-roughness, the reference's [77, 1024] half
    text: [DevBuf; 3],
    pub bytes: usize,
}

/// What a texture run makes along the way (for `check` and for looking at)
#[derive(Default)]
pub struct Trace {
    pub normal: Vec<Vec<f32>>,
    pub position: Vec<Vec<f32>>,
    pub picture: Vec<f32>,
    /// the decoded views: albedo x 6, metallic-roughness x 6 (8-bit values [512, 512, 3])
    pub views: Vec<Vec<f32>>,
}

fn half_rows(w: &Weights, name: &str) -> Result<DevBuf> {
    let m = w.host_mat(name)?;
    let b = DevBuf::new(&w.gpu, m.v.len() * 2)?;
    // SAFETY: u16s viewed as bytes.
    b.write(0, unsafe { std::slice::from_raw_parts(m.v.as_ptr().cast::<u8>(), m.v.len() * 2) })?;
    Ok(b)
}

impl Paint {
    /// The paint model: its UNet checkpoint (a PyTorch .bin), its VAE (.bin), DINOv2-giant (safetensors)
    pub fn load(unet: &SafeTensors, vae: &SafeTensors, dino: &SafeTensors, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Paint> {
        let t0 = std::time::Instant::now();
        let uw = Weights::new(unet, "unet.", gpu);
        let main = Unet::load(&uw, true)?;
        let text = [half_rows(&uw, "learned_text_clip_albedo")?, half_rows(&uw, "learned_text_clip_mr")?, half_rows(&uw, "learned_text_clip_ref")?];
        let proj = uw.lin("image_proj_model_dino.proj")?;
        let proj_norm = (uw.vec("image_proj_model_dino.norm.weight")?, uw.vec("image_proj_model_dino.norm.bias")?);
        let dw = Weights::new(unet, "unet_dual.", gpu);
        let dual = Unet::load(&dw, false)?;
        let vw = Weights::new(vae, "", gpu);
        let vae = sdvae::SdVae::load(&vw)?;
        let gw = Weights::new(dino, "", gpu);
        let dino = Dino::load(&gw, DINO_SIZE)?;
        let bytes = uw.bytes.get() + dw.bytes.get() + vw.bytes.get() + gw.bytes.get();
        log(format!("paint: loaded in {:.1} s ({:.2} GiB of weights: the 2.5D UNet and its reference UNet, the VAE, DINOv2-giant)",
                    t0.elapsed().as_secs_f64(), bytes as f64 / (1u64 << 30) as f64));
        Ok(Paint { unet: main, dual, vae, dino, proj, proj_norm, text, bytes })
    }

    /// DINOv2-giant's last hidden state of the picture (8-bit values [512, 512, 3]): float32 [257, 1536]
    pub fn dino_states(&self, ops: &Ops, picture: &[f32]) -> Result<DevBuf> {
        self.dino.forward(ops, &dino_pixels(picture))
    }

    /// DINOv2-giant's tokens projected (Linear 1536 -> 4 x 1024, LayerNorm): [unconditional (of zeros), the
    /// picture's], [1028, 1024] half each
    pub fn project(&self, ops: &Ops, states: &DevBuf) -> Result<[DevBuf; 2]> {
        let rows = self.dino.tokens();
        let c = self.dino.width;
        let sh = DevBuf::new(&ops.gpu, rows * c * 2)?;
        ops.to_half(states.ptr(), sh.ptr(), rows * c)?;
        let zero = DevBuf::new(&ops.gpu, rows * c * 2)?;
        zero.fill(0)?;
        let mut out = Vec::new();
        // the projections stay until the queue is done with them
        let mut proj = Vec::new();
        for x in [&zero, &sh] {
            let p = DevBuf::new(&ops.gpu, rows * self.proj.n * 4)?;
            ops.lin(&self.proj, x.ptr(), rows, p.ptr(), Dt::F32)?;
            let tokens = rows * self.proj.n / 1024;
            let o = DevBuf::new(&ops.gpu, tokens * 1024 * 2)?;
            ops.ln(p.ptr(), false, tokens, 1024, 1e-5, self.proj_norm.0.ptr(), self.proj_norm.1.ptr(), none(), none(), o.ptr(), true)?;
            out.push(o);
            proj.push(p);
        }
        ops.gpu.sync()?;
        drop(proj);
        let b = out.pop().expect("two");
        let a = out.pop().expect("two");
        Ok([a, b])
    }

    /// The latents (host [64, 64, 4] each) of the picture, the normal maps and the position maps (8-bit values)
    pub fn encode_all(&self, ops: &Ops, mem: &Mem, normal: &[Vec<f32>], position: &[Vec<f32>], picture: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let to01 = |v: &[f32]| v.iter().map(|x| x / 255.0).collect::<Vec<f32>>();
        let ref_lat = self.vae.encode(ops, mem, &to01(picture), SIZE, SIZE)?;
        let mut nlat = Vec::new();
        let mut plat = Vec::new();
        for i in 0..VIEWS {
            nlat.extend(self.vae.encode(ops, mem, &to01(&normal[i]), SIZE, SIZE)?);
            plat.extend(self.vae.encode(ops, mem, &to01(&position[i]), SIZE, SIZE)?);
        }
        Ok((ref_lat, nlat, plat))
    }

    /// The conditions of a generation: maps' and picture's latents, the reference states, the keys of every block
    pub fn conditions(&self, ops: &Ops, mem: &Mem, normal: &[Vec<f32>], position: &[Vec<f32>], picture: &[f32]) -> Result<(Cond, Vec<f32>, Vec<f32>)> {
        let (ref_lat, nlat, plat) = self.encode_all(ops, mem, normal, position, picture)?;
        let states = self.dino_states(ops, picture)?;
        let pos01: Vec<Vec<f32>> = position.iter().map(|p| p.iter().map(|x| x / 255.0).collect()).collect();
        let cond = self.prepare(ops, mem, &ref_lat, &self.project(ops, &states)?, &pos01)?;
        Ok((cond, nlat, plat))
    }

    /// The main passes' conditions from the picture's latent, DINO's projected tokens and the position maps (in
    /// [0, 1]): the reference UNet's states, every block's keys and values, the rotations
    pub fn prepare(&self, ops: &Ops, mem: &Mem, ref_lat: &[f32], dino: &[DevBuf; 2], pos01: &[Vec<f32>]) -> Result<Cond> {
        let mut cond = Cond { text: BTreeMap::new(), text_rows: 77, refkv: BTreeMap::new(), dino: BTreeMap::new(), dino_rows: 0, rope: BTreeMap::new(), states: BTreeMap::new() };
        // the reference UNet on the picture's latent at t = 0
        self.dual.text_kv(ops, &self.text[2..3], 77, &mut cond)?;
        let x = Map { buf: nn::upload_half(&mem.keep, ref_lat)?, n: 1, h: LAT, w: LAT, c: 4 };
        self.dual.forward(ops, mem, &x, 0.0, &mut cond, None)?;
        mem.keep.reset();
        // the main UNet's keys: text per material, the reference's, DINO's
        cond.text.clear();
        self.unet.text_kv(ops, &self.text[0..2], 77, &mut cond)?;
        self.unet.ref_kv(ops, &mut cond)?;
        let rows = dino[0].len / 2 / 1024;
        self.unet.dino_kv(ops, dino, rows, &mut cond)?;
        // the multiview rotation by level: (cos, sin) per row of the 36-picture batch and pair
        for (g, vres) in [(LAT, LAT * 8), (LAT / 2, LAT * 4), (LAT / 4, LAT * 2), (LAT / 8, LAT)] {
            let idx = voxel_indices(pos01, SIZE, g, vres);
            let table = rope_table(&idx);
            let mut all = Vec::with_capacity(table.len() * MATS * BRANCHES);
            for _ in 0..MATS * BRANCHES {
                all.extend_from_slice(&table);
            }
            cond.rope.insert(VIEWS * g * g, DevBuf::from_f32(&ops.gpu, &all)?);
        }
        ops.gpu.sync()?;
        Ok(cond)
    }

    #[allow(clippy::too_many_arguments)]
    /// The guided v of latents (host [12 = (material, view), 64, 64, 4]) at timestep t
    pub fn guided(&self, ops: &Ops, mem: &Mem, cond: &mut Cond, lat: &[f32], nlat: &[f32], plat: &[f32], t: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        let px = LAT * LAT;
        let n = MATS * BRANCHES * VIEWS;
        let mut input = vec![0f32; n * px * 12];
        for m in 0..MATS {
            for b in 0..BRANCHES {
                for v in 0..VIEWS {
                    let img = (m * BRANCHES + b) * VIEWS + v;
                    let src = (m * VIEWS + v) * px * 4;
                    for p in 0..px {
                        let o = (img * px + p) * 12;
                        input[o..o + 4].copy_from_slice(&lat[src + p * 4..src + p * 4 + 4]);
                        input[o + 4..o + 8].copy_from_slice(&nlat[(v * px + p) * 4..(v * px + p) * 4 + 4]);
                        input[o + 8..o + 12].copy_from_slice(&plat[(v * px + p) * 4..(v * px + p) * 4 + 4]);
                    }
                }
            }
        }
        let x = Map { buf: nn::upload_half(&mem.keep, &input)?, n, h: LAT, w: LAT, c: 12 };
        let out = DevBuf::new(&ops.gpu, n * px * 4 * 4)?;
        self.unet.forward(ops, mem, &x, t, cond, Some(&out))?;
        mem.keep.reset();
        let raw = read_f32(&out, n * px * 4)?;
        let per = px * 4;
        let mut v = vec![0f32; MATS * VIEWS * per];
        for m in 0..MATS {
            for vi in 0..VIEWS {
                let at = |b: usize| ((m * BRANCHES + b) * VIEWS + vi) * per;
                let (u, r, f) = (at(0), at(1), at(2));
                let dst = (m * VIEWS + vi) * per;
                for k in 0..per {
                    let (uu, rr, ff) = (raw[u + k], raw[r + k], raw[f + k]);
                    v[dst + k] = uu + GUIDANCE * (rr - uu) + GUIDANCE * (ff - rr);
                }
            }
        }
        Ok((v, raw))
    }

    /// Textures for `mesh` (the shape, as generated) from `picture` (the request's, RGBA): the mesh with an atlas and
    /// its PBR material
    #[allow(clippy::too_many_arguments)]
    pub fn paint(&self, ops: &Ops, mesh: &Mesh, picture: &Texture, seed: u64, steps: usize, tsize: u32, trace: Option<&mut Trace>,
                 progress: &mut dyn FnMut(&'static str, u32, u32) -> Result<()>) -> Result<Mesh> {
        let mem = Mem::new(&ops.gpu, KEEP, TMP)?;
        progress("views", 0, 1)?;
        let scene = render::Scene::new(mesh);
        let (normal, position) = maps(&scene);
        let pic = reference_picture(picture);
        progress("views", 1, 1)?;
        let (mut cond, nlat, plat) = self.conditions(ops, &mem, &normal, &position, &pic)?;
        let mut sched = unipc::UniPc::new(steps);
        let mut lat = crate::noise(seed, MATS * VIEWS * LAT * LAT * 4);
        for i in 0..steps {
            let (v, _) = self.guided(ops, &mem, &mut cond, &lat, &nlat, &plat, sched.timesteps[i] as f32)?;
            lat = sched.step(&v, &lat);
            progress("texture", i as u32 + 1, steps as u32)?;
        }
        let per = LAT * LAT * 4;
        let mut views = Vec::new();
        for i in 0..MATS * VIEWS {
            let img = self.vae.decode(ops, &mem, &lat[i * per..(i + 1) * per], LAT, LAT)?;
            views.push(img.into_iter().map(|x| (x * 255.0).round()).collect::<Vec<f32>>());
            progress("decode", i as u32 + 1, (MATS * VIEWS) as u32)?;
        }
        if mem.keep.spills.load(std::sync::atomic::Ordering::Relaxed) + mem.tmp.spills.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            ops.gpu.sync()?;
        }
        drop(cond);
        // bake: the views at 2,048 (Pillow's bicubic in place of RealESRGAN), the albedo and the metallic-roughness
        let big: Vec<Vec<f32>> = views.iter().map(|v| pil::resize(v, SIZE, SIZE, 3, RENDER, RENDER, pil::Filter::Bicubic)).collect();
        let mut m = nextsycl_mesh::tools::unwrap(mesh, tsize, 4);
        m.compute_normals();
        let mut textures = Vec::new();
        for (k, set) in big.chunks(VIEWS).enumerate() {
            let (rgb, seen, tex) = bake::bake(&m, &scene, set, RENDER, tsize);
            textures.push(bake::inpaint(&m, &rgb, &seen, &tex, tsize));
            progress("bake", k as u32 + 1, 2)?;
        }
        let mr = textures.pop().expect("two");
        let albedo = textures.pop().expect("two");
        // glTF's metallic-roughness: roughness in G, metallic in B (Tencent's mr picture: metallic R, roughness G)
        let mut packed = Texture::new(tsize, tsize);
        for (o, i) in packed.rgba.chunks_exact_mut(4).zip(mr.rgba.chunks_exact(4)) {
            o.copy_from_slice(&[0, i[1], i[0], 255]);
        }
        m.material = Some(Material { base_color: Some(albedo), metallic_roughness: Some(packed), metallic: 1.0, roughness: 1.0, ..Default::default() });
        if let Some(t) = trace {
            *t = Trace { normal, position, picture: pic, views };
        }
        Ok(m)
    }
}

/// The six views' normal and position maps as the model takes them: 8-bit values [512, 512, 3]
pub fn maps(scene: &render::Scene) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (mut n, mut p) = (Vec::new(), Vec::new());
    for i in 0..VIEWS {
        let v = render::view(scene, render::ELEVS[i], render::AZIMS[i], RENDER);
        n.push(pil::resize(&render::normal_map(scene, &v), RENDER, RENDER, 3, SIZE, SIZE, pil::Filter::Bicubic));
        p.push(pil::resize(&render::position_map(scene, &v), RENDER, RENDER, 3, SIZE, SIZE, pil::Filter::Bicubic));
    }
    (n, p)
}

/// The picture as the paint model takes it: resized to 512 (Pillow, bicubic, alpha premultiplied) and put on white;
/// 8-bit values [512, 512, 3]
pub fn reference_picture(t: &Texture) -> Vec<f32> {
    let rgba: Vec<f32> = t.rgba.iter().map(|&b| b as f32).collect();
    let r = pil::resize_rgba(&rgba, t.width as usize, t.height as usize, SIZE, SIZE, pil::Filter::Bicubic);
    pil::on_white(&r)
}

/// DINOv2-giant's input from the 512 px picture (its image processor: shortest side to 256, bicubic; a centre crop
/// of 224; ImageNet's normalization): [3, 224, 224]
fn dino_pixels(picture: &[f32]) -> Vec<f32> {
    let r = pil::resize(picture, SIZE, SIZE, 3, 256, 256, pil::Filter::Bicubic);
    let off = (256 - DINO_SIZE) / 2;
    let mut out = vec![0f32; 3 * DINO_SIZE * DINO_SIZE];
    for y in 0..DINO_SIZE {
        for x in 0..DINO_SIZE {
            for c in 0..3 {
                out[c * DINO_SIZE * DINO_SIZE + y * DINO_SIZE + x] = r[((y + off) * 256 + x + off) * 3 + c] / 255.0;
            }
        }
    }
    crate::dino::normalize(&mut out, DINO_SIZE);
    out
}

/// The position maps' voxel indices at a grid of g x g cells a view (Tencent's compute_discrete_voxel_indice): each
/// cell's mean position over its foreground pixels (not pure white), zero when fewer than 1/16 of them are, scaled
/// to `vres` levels; [views x g x g, 3]
pub fn voxel_indices(pos01: &[Vec<f32>], size: usize, g: usize, vres: usize) -> Vec<[usize; 3]> {
    let cell = size / g;
    let thres = (cell * cell) / 16;
    let mut out = Vec::with_capacity(pos01.len() * g * g);
    for p in pos01 {
        for cy in 0..g {
            for cx in 0..g {
                let (mut s, mut n) = ([0f64; 3], 0usize);
                for y in cy * cell..(cy + 1) * cell {
                    for x in cx * cell..(cx + 1) * cell {
                        let px = &p[(y * size + x) * 3..(y * size + x) * 3 + 3];
                        if px.iter().all(|&v| v != 1.0) {
                            for k in 0..3 {
                                s[k] += px[k] as f64;
                            }
                            n += 1;
                        }
                    }
                }
                let mean = if n < thres { [0f64; 3] } else { s.map(|v| v / n.max(1) as f64) };
                out.push(mean.map(|m| (m.clamp(0.0, 1.0) * (vres - 1) as f64).round() as usize));
            }
        }
    }
    out
}

/// The multiview attention's rotation for tokens at voxel indices: (cos, sin) per token and pair, [tokens, 32, 2],
/// the pairs ordered as the q / k rows were (x: 12 pairs, y: 12, z: 8; frequencies 10000^(-2i/24), 10000^(-2i/16))
pub fn rope_table(idx: &[[usize; 3]]) -> Vec<f32> {
    let fxy: Vec<f64> = (0..12).map(|i| 1.0 / 10000f64.powf(2.0 * i as f64 / 24.0)).collect();
    let fz: Vec<f64> = (0..8).map(|i| 1.0 / 10000f64.powf(2.0 * i as f64 / 16.0)).collect();
    let mut out = Vec::with_capacity(idx.len() * 64);
    for p in idx {
        for pair in 0..32 {
            let a = if pair < 12 {
                p[0] as f64 * fxy[pair]
            } else if pair < 24 {
                p[1] as f64 * fxy[pair - 12]
            } else {
                p[2] as f64 * fz[pair - 24]
            };
            out.push(a.cos() as f32);
            out.push(a.sin() as f32);
        }
    }
    out
}

impl Paint {
    /// Errors for files that are not the paint model's
    pub fn check_files(unet: &SafeTensors) -> Result<()> {
        if unet.tensor("unet.learned_text_clip_albedo").is_none() || unet.tensor("unet_dual.conv_in.weight").is_none() {
            return Err(Error(format!("{}: not Hunyuan3D-Paint's UNet (no learned material tokens, no reference UNet)", unet.path.display())));
        }
        Ok(())
    }
}
