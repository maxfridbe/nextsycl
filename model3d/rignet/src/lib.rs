//! RigNet (Xu et al., SIGGRAPH 2020; github.com/zhan-xu/RigNet): a skeleton and skin weights for a character mesh.
//! **GPLv3**, as RigNet's code (a commercial licence is offered by UMass): this crate is a port of that code and is kept
//! apart from the MIT ones; a program built with it is GPLv3.
//!
//! The pipeline (quick_start.py):
//! 1. the mesh welded and decimated to about 4,000 triangles (the authors remesh with MeshLab or open3d's quadric
//!    decimation; here `nextsycl_mesh::tools::decimate`, also quadric, so the remeshed vertices differ), normalized
//!    (height 1, feet at 0, centred);
//! 2. its geometry on the CPU (`geom`): the mesh's edges, surface geodesics (Poisson-disk samples, Dijkstra), geodesic
//!    edges, a solid voxelization;
//! 3. GMEdgeNet (JointNet, `net`, the GPU): every vertex's displacement toward a joint and its attention; then the
//!    joints by weighted mean-shift, a density threshold and non-maximum suppression, symmetrized (`skel`);
//! 4. RootNet picks the root, BoneNet scores every pair of joints; the symmetric Prim tree over their costs;
//! 5. SkinNet: each vertex's 5 nearest bones by volumetric geodesic distance (ray visibility, `geom`) to weights,
//!    smoothed over the one-ring;
//! 6. the weights carried back to the user's own vertices (`nextsycl_mesh::tools::skin`), four a vertex.
//!
//! Where it differs from the reference by choice: the joint encoders' farthest point sampling starts at the first
//! joint (torch_cluster starts at a random one); bones that share a parent bind to that parent (the reference adds a
//! duplicate joint a bone for Maya); the volumetric geodesics are computed on every vertex (the reference's demo
//! subsamples 1,500).

pub mod check;
mod ffi;
pub mod geom;
pub mod net;
pub mod npy;
pub mod pointnet;
pub mod skel;
pub mod zip;

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::Gpu;
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::mesh::{tools, Joint, Skin};
use nextsycl_model3d::{
    Abilities, At, Defaults, EngineOption, Error, LoadOptions, Mesh, Model3dEngine, Model3dKind, ModelFiles, Result, RigRequest, Step,
};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "rignet";

/// The files it needs, by role: the authors' trained_models.zip (the four networks' checkpoints are read from it)
pub const ROLES: &[&str] = &["models"];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "faces", env: "NS_M3D_RIGNET_FACES", value: "N", help: "triangles the mesh is decimated to before rigging (default 4000, the reference's)", at: At::Both },
    EngineOption { name: "bandwidth", env: "NS_M3D_RIGNET_BANDWIDTH", value: "X", help: "mean-shift bandwidth of the joints (default: the learned 0.0429; the authors tune 0.03 - 0.06 per model)", at: At::Both },
    EngineOption { name: "threshold", env: "NS_M3D_RIGNET_THRESHOLD", value: "X", help: "density threshold of the joints (default 1e-5; lower: more joints)", at: At::Both },
];

pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "RigNet (a skeleton and skin weights: GMEdgeNet, RootNet, BoneNet, SkinNet; GPLv3)", roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(RigNet::load(files, gpu, o, log)?))
}

/// The settings of a rig
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub faces: usize,
    pub bandwidth: Option<f32>,
    pub threshold: f32,
}

pub struct RigNet {
    pub ops: net::Ops,
    pub joint: net::JointNet,
    pub root: net::RootNet,
    pub bone: net::BoneNet,
    pub skin: net::SkinNet,
    settings: Settings,
    load_s: f64,
}

/// The mesh as the networks take it, and what the CPU made of it
pub struct Prepared {
    /// the decimated mesh, normalized
    pub pos: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub pivot: [f32; 3],
    pub scale: f32,
    pub tpl: Vec<(u32, u32)>,
    pub geo: Vec<(u32, u32)>,
    pub geodesic: geom::Geodesic,
    pub vox: geom::Vox,
}

/// What a rig makes, on the decimated mesh (normalized)
pub struct Rigged {
    /// joints in tree order (root first, parents before children)
    pub joints: Vec<[f64; 3]>,
    pub parents: Vec<Option<usize>>,
    /// weights [vertices of the decimated mesh, joints]
    pub weights: Vec<f32>,
}

fn setting<T: std::str::FromStr>(o: &LoadOptions, name: &str) -> Result<Option<T>> {
    o.setting(name).map(|v| v.parse::<T>().map_err(|_| Error(format!("{name}: {v:?}?")))).transpose()
}

impl RigNet {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<RigNet> {
        let t0 = Instant::now();
        let zipf = files.get("models").ok_or_else(|| Error(format!("{ARCH}: no models file (trained_models.zip)")))?;
        let ops = net::Ops::new(gpu)?;
        let scratch = zip::Scratch::new()?;
        let open = |net: &str| -> Result<Pth> {
            let p = scratch.extract(zipf, &format!("checkpoints/{net}/model_best.pth.tar"), &format!("{net}.pth.tar"))?;
            Pth::open(&p).map_err(|e| Error(e.0))
        };
        let joint = net::JointNet::load(&open("gcn_meanshift")?, gpu)?;
        let root = net::RootNet::load(&open("rootnet")?, gpu)?;
        let bone = net::BoneNet::load(&open("bonenet")?, gpu)?;
        let skin = net::SkinNet::load(&open("skinnet")?, gpu)?;
        drop(scratch);
        let settings = Settings {
            faces: setting(o, "NS_M3D_RIGNET_FACES")?.unwrap_or(4000),
            bandwidth: setting(o, "NS_M3D_RIGNET_BANDWIDTH")?,
            threshold: setting(o, "NS_M3D_RIGNET_THRESHOLD")?.unwrap_or(1e-5),
        };
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: loaded on {} in {load_s:.1} s (JointNet bandwidth {:.4}; GPLv3)", gpu.name, joint.bandwidth));
        Ok(RigNet { ops, joint, root, bone, skin, settings, load_s })
    }

    /// The decimated mesh's geometry (welded, decimated to `faces` triangles, normalized)
    pub fn prepare(&self, m: &Mesh, faces: usize, seed: u64) -> Result<Prepared> {
        let (mut w, _) = tools::weld(m, 1e-7);
        tools::compact(&mut w);
        let mut d = if w.triangles() > faces { tools::decimate(&w, faces) } else { w };
        tools::compact(&mut d);
        if d.positions.len() < 4 || d.triangles() < 2 {
            return Err(Error("the mesh has too few triangles to rig".into()));
        }
        let faces: Vec<[u32; 3]> = d.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        Ok(self.prepare_geometry(&d.positions, &faces, seed))
    }

    /// The geometry of a mesh already at the networks' size
    pub fn prepare_geometry(&self, v: &[[f32; 3]], faces: &[[u32; 3]], seed: u64) -> Prepared {
        let (pos, pivot, scale) = geom::normalize(v);
        let vn = geom::vertex_normals(&pos, faces);
        let mut rng = geom::Rng::new(seed ^ 0x51_9e7);
        let (pts, nrm) = geom::poisson_disk(&pos, &vn, faces, 4000, &mut rng);
        let geodesic = geom::surface_geodesic(&pos, &pts, &nrm);
        let geo = geom::geo_edges(&geodesic, &mut rng);
        let tpl = geom::tpl_edges(pos.len(), faces);
        let vox = geom::voxelize(&pos, faces, 88);
        Prepared { pos, faces: faces.to_vec(), pivot, scale, tpl, geo, geodesic, vox }
    }

    pub fn mesh_in(&self, p: &Prepared) -> Result<net::MeshIn> {
        let pos: Vec<f32> = p.pos.iter().flatten().copied().collect();
        Ok(net::MeshIn { pos: self.ops.up(&pos)?, tpl: net::Graph::new(&self.ops, p.pos.len(), &p.tpl)?, geo: net::Graph::new(&self.ops, p.pos.len(), &p.geo)? })
    }

    /// The connection tree of the joints: RootNet's root, BoneNet's pair probabilities, the symmetric Prim tree over
    /// their costs (parents, -1 for the root; the root)
    pub fn skeleton(&self, mi: &net::MeshIn, p: &Prepared, joints: &[[f64; 3]]) -> Result<(Vec<i64>, usize)> {
        let jf: Vec<[f32; 3]> = joints.iter().map(|j| j.map(|x| x as f32)).collect();
        let shape = self.root.shape(&self.ops, mi)?;
        let root_logits = self.root.run(&self.ops, &shape, &jf)?;
        let root = (0..jf.len()).max_by(|a, b| root_logits[*a].total_cmp(&root_logits[*b]).then(b.cmp(a))).unwrap_or(0);
        let (pairs, rows) = skel::pairs(joints, &p.vox);
        let shape = self.bone.shape(&self.ops, mi)?;
        let logits = self.bone.run(&self.ops, &shape, &jf, &rows)?;
        let prob: Vec<f32> = logits.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let cost = skel::costs(joints, &pairs, &prob, &p.vox);
        Ok(skel::prim_symmetric(&cost, root, joints))
    }

    /// The skin weights [n, joints] of the decimated mesh for a skeleton in tree order
    pub fn skin_weights(&self, mi: &net::MeshIn, p: &Prepared, joints: &[[f64; 3]], parents: &[Option<usize>]) -> Result<Vec<f32>> {
        let (bones, leaf, bind) = skel::bones(joints, parents);
        if bones.is_empty() {
            return Ok(vec![1.0; p.pos.len()]);
        }
        let geo = geom::volumetric_geodesic(&p.pos, &p.faces, &bones, &p.geodesic);
        let (input, ids, mask) = skel::skin_input(&p.pos, &geo, &bones, &leaf);
        let logits = self.skin.run(&self.ops, mi, &input)?;
        let ring = one_ring(p.pos.len(), &p.tpl);
        let wb = skel::skin_weights(&logits, &ids, &mask, bones.len(), &ring);
        // the bones' weights to their joints
        let nj = joints.len();
        let mut out = vec![0f32; p.pos.len() * nj];
        for v in 0..p.pos.len() {
            for (b, &j) in bind.iter().enumerate() {
                out[v * nj + j] += wb[v * bones.len() + b];
            }
        }
        Ok(out)
    }

    /// The whole rig of a prepared mesh
    pub fn rig_prepared(&self, p: &Prepared, s: Settings, progress: &mut dyn FnMut(&'static str, u32) -> Result<()>) -> Result<Rigged> {
        let mi = self.mesh_in(p)?;
        let (disp, attn) = self.joint.run(&self.ops, &mi)?;
        let bw = s.bandwidth.unwrap_or(self.joint.bandwidth) as f64;
        let joints = skel::joints_from(&p.pos, &disp, &attn, &p.vox, bw, s.threshold as f64);
        if joints.is_empty() {
            return Err(Error(format!("{ARCH}: no joints found (try a lower --opt-threshold or another --opt-bandwidth)")));
        }
        progress("skeleton", 1)?;
        let (parent, root) = self.skeleton(&mi, p, &joints)?;
        let (order, parents) = skel::tree_order(&parent, root);
        let joints: Vec<[f64; 3]> = order.iter().map(|&o| joints[o]).collect();
        progress("skeleton", 2)?;
        let weights = self.skin_weights(&mi, p, &joints, &parents)?;
        progress("skin", 1)?;
        Ok(Rigged { joints, parents, weights })
    }
}

/// Each vertex's neighbours over the mesh's edges (itself apart)
pub fn one_ring(n: usize, tpl: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut r: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in tpl {
        if a != b {
            r[a as usize].push(b);
            r[b as usize].push(a);
        }
    }
    for x in &mut r {
        x.sort_unstable();
        x.dedup();
    }
    r
}

impl Model3dEngine for RigNet {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn abilities(&self) -> Abilities {
        Abilities { rig: true, ..Default::default() }
    }
    fn defaults(&self) -> Defaults {
        Defaults { faces: self.settings.faces as u32, ..Default::default() }
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn rig(&self, req: &RigRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        let t0 = Instant::now();
        let num = |k: &str| -> Result<Option<f32>> { req.extra.get(k).map(|v| v.parse::<f32>().map_err(|_| Error(format!("{k}: {v:?}?")))).transpose() };
        let s = Settings {
            faces: num("faces")?.map_or(self.settings.faces, |f| f as usize).max(100),
            bandwidth: num("bandwidth")?.or(self.settings.bandwidth),
            threshold: num("threshold")?.unwrap_or(self.settings.threshold),
        };
        if req.mesh.positions.is_empty() || req.mesh.indices.is_empty() {
            return Err(Error("an empty mesh".into()));
        }
        progress(Step { phase: "mesh", at: 0, of: 1, seconds: 0.0 })?;
        let p = self.prepare(&req.mesh, s.faces, req.seed)?;
        progress(Step { phase: "mesh", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let r = self.rig_prepared(&p, s, &mut |phase, at| progress(Step { phase, at, of: if phase == "skin" { 1 } else { 2 }, seconds: t0.elapsed().as_secs_f64() }))?;
        // back to the input's frame and vertices
        let unmap = |q: [f32; 3]| [0, 1, 2].map(|c| q[c] / p.scale + p.pivot[c]);
        let dec: Vec<[f32; 3]> = p.pos.iter().map(|q| unmap(*q)).collect();
        let flat: Vec<u32> = p.faces.iter().flatten().copied().collect();
        let nj = r.joints.len();
        let dense = tools::skin::transfer(&dec, &flat, &r.weights, nj, &req.mesh.positions);
        let joints = r.joints.iter().zip(&r.parents).enumerate().map(|(i, (j, par))| Joint {
            name: if par.is_none() { "root".to_string() } else { format!("joint_{i}") },
            parent: *par,
            head: unmap(j.map(|x| x as f32)),
        }).collect();
        let mut m = req.mesh.clone();
        m.skin = Some(Skin { joints, weights: tools::skin::top4(&dense, nj) });
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
