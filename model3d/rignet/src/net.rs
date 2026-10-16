//! RigNet's four networks on the GPU (models/*.py): every MLP layer is Linear -> ReLU -> BatchNorm (eval: folded to a
//! per-channel affine after the ReLU); a GCU is two EdgeConvs (the mesh's edges and the geodesic ones, max over each
//! node's messages, self-loops included) and an MLP. The joint encoders' PointNet++ layers (farthest points, radius
//! neighbourhoods, k-nearest interpolation) pick their points on the host (`pointnet`) and run their MLPs here.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{Dt, Nsd};
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::{Error, Result};

use crate::ffi::{K, SIGMOID, TANH};
use crate::pointnet;

/// The kernels on the engine's GPU
pub struct Ops {
    pub nsd: Nsd,
    pub k: K,
    pub gpu: Arc<Gpu>,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { nsd: Nsd::new(gpu)?, k: K::new(gpu)?, gpu: gpu.clone() })
    }
    pub fn buf(&self, n: usize) -> Result<DevBuf> {
        DevBuf::f32(&self.gpu, n.max(1))
    }
    pub fn up(&self, v: &[f32]) -> Result<DevBuf> {
        if v.is_empty() {
            return self.buf(1);
        }
        DevBuf::from_f32(&self.gpu, v)
    }
    pub fn up_i32(&self, v: &[i32]) -> Result<DevBuf> {
        let b = DevBuf::new(&self.gpu, v.len().max(1) * 4)?;
        if !v.is_empty() {
            b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        Ok(b)
    }
    /// The first n floats of a buffer, on the host
    pub fn down(&self, b: &DevBuf, n: usize) -> Result<Vec<f32>> {
        let mut v = b.to_f32()?;
        v.truncate(n);
        Ok(v)
    }
}

/// A linear layer: weight [n, k], bias [n]
pub struct Lin {
    w: DevBuf,
    b: DevBuf,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    pub fn run(&self, o: &Ops, x: &DevBuf, m: usize) -> Result<DevBuf> {
        if x.floats() < m * self.k {
            return Err(Error(format!("a linear layer [{}, {}] on {} floats for {m} rows", self.n, self.k, x.floats())));
        }
        let out = o.buf(m * self.n)?;
        if m > 0 {
            o.nsd.linear(x.ptr(), Dt::F32, m, self.k, self.w.ptr(), self.n, self.b.ptr(), out.ptr(), Dt::F32)?;
        }
        Ok(out)
    }
}

/// One checkpoint's weights (its state dict)
pub struct W<'a> {
    pth: &'a Pth,
    prefix: String,
    gpu: &'a Arc<Gpu>,
}

impl<'a> W<'a> {
    pub fn new(pth: &'a Pth, probe: &str, gpu: &'a Arc<Gpu>) -> Result<W<'a>> {
        let prefix = pth.prefix_of(probe).ok_or_else(|| Error(format!("{}: no {probe} (not this RigNet network's checkpoint)", pth.path.display())))?;
        Ok(W { pth, prefix, gpu })
    }
    pub fn has(&self, name: &str) -> bool {
        self.pth.tensor(&format!("{}{name}", self.prefix)).is_some()
    }
    pub fn host(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let full = format!("{}{name}", self.prefix);
        let t = self.pth.need(&full).map_err(|e| Error(e.0))?;
        Ok((t.shape.clone(), self.pth.f32(&full).map_err(|e| Error(e.0))?))
    }
    pub fn lin(&self, name: &str) -> Result<Lin> {
        let (s, w) = self.host(&format!("{name}.weight"))?;
        let (sb, b) = self.host(&format!("{name}.bias"))?;
        if s.len() != 2 || sb != [s[0]] {
            return Err(Error(format!("{name}: weight {s:?}, bias {sb:?}")));
        }
        Ok(Lin { w: DevBuf::from_f32(self.gpu, &w)?, b: DevBuf::from_f32(self.gpu, &b)?, n: s[0], k: s[1] })
    }
    /// A BatchNorm in eval form, as the affine (scale, shift) it is
    pub fn bn(&self, name: &str) -> Result<(DevBuf, DevBuf)> {
        let (_, w) = self.host(&format!("{name}.weight"))?;
        let (_, b) = self.host(&format!("{name}.bias"))?;
        let (_, m) = self.host(&format!("{name}.running_mean"))?;
        let (_, v) = self.host(&format!("{name}.running_var"))?;
        let s: Vec<f32> = w.iter().zip(&v).map(|(w, v)| w / (v + 1e-5).sqrt()).collect();
        let t: Vec<f32> = b.iter().zip(&m).zip(&s).map(|((b, m), s)| b - m * s).collect();
        Ok((DevBuf::from_f32(self.gpu, &s)?, DevBuf::from_f32(self.gpu, &t)?))
    }
    /// models/gcn_basic_modules.py MLP: layers `{name}.{i}.0` (Linear) and `{name}.{i}.2` (BatchNorm)
    pub fn mlp(&self, name: &str) -> Result<Mlp> {
        let mut layers = Vec::new();
        while self.has(&format!("{name}.{}.0.weight", layers.len())) {
            let i = layers.len();
            layers.push((self.lin(&format!("{name}.{i}.0"))?, self.bn(&format!("{name}.{i}.2"))?));
        }
        if layers.is_empty() {
            return Err(Error(format!("{}: no MLP {name}", self.pth.path.display())));
        }
        Ok(Mlp { layers })
    }
    pub fn gcu(&self, name: &str) -> Result<Gcu> {
        Ok(Gcu { tpl: self.mlp(&format!("{name}.edge_conv_tpl.nn"))?, geo: self.mlp(&format!("{name}.edge_conv_geo.nn"))?, mlp: self.mlp(&format!("{name}.mlp"))? })
    }
}

/// Linear -> ReLU -> BatchNorm, layer after layer
pub struct Mlp {
    layers: Vec<(Lin, (DevBuf, DevBuf))>,
}

impl Mlp {
    pub fn out(&self) -> usize {
        self.layers.last().map_or(0, |l| l.0.n)
    }
    pub fn input(&self) -> usize {
        self.layers.first().map_or(0, |l| l.0.k)
    }
    pub fn run(&self, o: &Ops, x: &DevBuf, m: usize) -> Result<DevBuf> {
        let mut h: Option<DevBuf> = None;
        for (l, (s, t)) in &self.layers {
            let y = l.run(o, h.as_ref().unwrap_or(x), m)?;
            o.k.relu_affine(&y, m, l.n, Some((s, t)))?;
            h = Some(y);
        }
        h.ok_or_else(|| Error("an empty MLP".into()))
    }
    /// On host rows: upload, run, download
    pub fn run_host(&self, o: &Ops, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if x.len() != m * self.input() {
            return Err(Error(format!("an MLP of {} inputs given {} values for {m} rows", self.input(), x.len())));
        }
        let y = self.run(o, &o.up(x)?, m)?;
        o.down(&y, m * self.out())
    }
}

/// A graph on the GPU: edges src -> dst sorted by dst (each node's messages together), one self-loop a node
pub struct Graph {
    pub n: usize,
    pub e: usize,
    src: DevBuf,
    dst: DevBuf,
    rowptr: DevBuf,
}

impl Graph {
    /// From directed edges (src, dst) over n nodes: self-loops removed, then one added a node (EdgeConv's
    /// remove_self_loops / add_self_loops)
    pub fn new(o: &Ops, n: usize, edges: &[(u32, u32)]) -> Result<Graph> {
        let mut by: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(s, d) in edges {
            if s != d {
                by[d as usize].push(s);
            }
        }
        let (mut src, mut dst, mut rowptr) = (Vec::new(), Vec::new(), vec![0i32]);
        for (d, ss) in by.iter().enumerate() {
            for &s in ss.iter().chain(std::iter::once(&(d as u32))) {
                src.push(s as i32);
                dst.push(d as i32);
            }
            rowptr.push(src.len() as i32);
        }
        Ok(Graph { n, e: src.len(), src: o.up_i32(&src)?, dst: o.up_i32(&dst)?, rowptr: o.up_i32(&rowptr)? })
    }
}

/// Two EdgeConvs (the mesh's edges, the geodesic ones) side by side, then an MLP
pub struct Gcu {
    tpl: Mlp,
    geo: Mlp,
    mlp: Mlp,
}

impl Gcu {
    pub fn out(&self) -> usize {
        self.mlp.out()
    }
    pub fn run(&self, o: &Ops, x: &DevBuf, c: usize, tpl: &Graph, geo: &Graph) -> Result<DevBuf> {
        let n = tpl.n;
        let half = self.tpl.out();
        let cat = o.buf(n * 2 * half)?;
        for (k, (g, nn)) in [(tpl, &self.tpl), (geo, &self.geo)].into_iter().enumerate() {
            let ef = o.buf(g.e * 2 * c)?;
            o.k.edge_feat(x, c, &g.src, &g.dst, g.e, &ef)?;
            let h = nn.run(o, &ef, g.e)?;
            o.k.seg_max(&h, half, &g.rowptr, n, &cat, 2 * half, k * half)?;
        }
        self.mlp.run(o, &cat, n)
    }
}

/// Columns side by side: `parts` of (buffer, its columns), m rows each (columns 0 rows: a row broadcast)
fn cat(o: &Ops, m: usize, parts: &[(&DevBuf, usize, bool)]) -> Result<(DevBuf, usize)> {
    let w: usize = parts.iter().map(|p| p.1).sum();
    let out = o.buf(m * w)?;
    let mut off = 0;
    for (b, c, broadcast) in parts {
        o.k.copy_cols(b, if *broadcast { 0 } else { *c }, m, *c, &out, w, off)?;
        off += c;
    }
    Ok((out, w))
}

/// The max over all rows of x [m, c]: [c]
fn global_max(o: &Ops, x: &DevBuf, m: usize, c: usize) -> Result<DevBuf> {
    let rp = o.up_i32(&[0, m as i32])?;
    let out = o.buf(c)?;
    o.k.seg_max(x, c, &rp, 1, &out, c, 0)?;
    Ok(out)
}

/// The mesh's inputs on the GPU: positions and the two graphs
pub struct MeshIn {
    pub pos: DevBuf,
    pub tpl: Graph,
    pub geo: Graph,
}

/// models/GCN.py JointPredNet (positions only): per-vertex outputs
pub struct JointPred {
    gcu: [Gcu; 3],
    glb: Mlp,
    tr: Mlp,
    last: Lin,
}

impl JointPred {
    fn load(w: &W, p: &str) -> Result<JointPred> {
        Ok(JointPred {
            gcu: [w.gcu(&format!("{p}.gcu_1"))?, w.gcu(&format!("{p}.gcu_2"))?, w.gcu(&format!("{p}.gcu_3"))?],
            glb: w.mlp(&format!("{p}.mlp_glb"))?,
            tr: w.mlp(&format!("{p}.mlp_tramsform.0"))?,
            last: w.lin(&format!("{p}.mlp_tramsform.2"))?,
        })
    }
    fn run(&self, o: &Ops, m: &MeshIn) -> Result<DevBuf> {
        let n = m.tpl.n;
        let x1 = self.gcu[0].run(o, &m.pos, 3, &m.tpl, &m.geo)?;
        let c1 = self.gcu[0].out();
        let x2 = self.gcu[1].run(o, &x1, c1, &m.tpl, &m.geo)?;
        let c2 = self.gcu[1].out();
        let x3 = self.gcu[2].run(o, &x2, c2, &m.tpl, &m.geo)?;
        let c3 = self.gcu[2].out();
        let (c, _) = cat(o, n, &[(&x1, c1, false), (&x2, c2, false), (&x3, c3, false)])?;
        let x4 = self.glb.run(o, &c, n)?;
        let g = global_max(o, &x4, n, self.glb.out())?;
        let (x5, _) = cat(o, n, &[(&g, self.glb.out(), true), (&m.pos, 3, false), (&x1, c1, false), (&x2, c2, false), (&x3, c3, false)])?;
        let h = self.tr.run(o, &x5, n)?;
        self.last.run(o, &h, n)
    }
}

/// models/GCN.py JOINTNET_MASKNET_MEANSHIFT: displacement (tanh), attention (sigmoid), the learned bandwidth
pub struct JointNet {
    joint: JointPred,
    mask: JointPred,
    pub bandwidth: f32,
}

impl JointNet {
    pub fn load(pth: &Pth, gpu: &Arc<Gpu>) -> Result<JointNet> {
        let w = W::new(pth, "jointnet.gcu_1.mlp.0.0.weight", gpu)?;
        let (_, b) = w.host("bandwidth")?;
        Ok(JointNet { joint: JointPred::load(&w, "jointnet")?, mask: JointPred::load(&w, "masknet")?, bandwidth: b.first().copied().unwrap_or(0.04) })
    }
    /// (displacement [n, 3], attention [n]) on the host
    pub fn run(&self, o: &Ops, m: &MeshIn) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = m.tpl.n;
        let d = self.joint.run(o, m)?;
        o.k.act(&d, n * 3, TANH)?;
        let a = self.mask.run(o, m)?;
        o.k.act(&a, n, SIGMOID)?;
        Ok((o.down(&d, n * 3)?, o.down(&a, n)?))
    }
}

/// The shape encoders of RootNet and BoneNet: three GCUs, an MLP, the max over the vertices
pub struct ShapeEnc {
    gcu: [Gcu; 3],
    glb: Mlp,
}

impl ShapeEnc {
    fn load(w: &W) -> Result<ShapeEnc> {
        Ok(ShapeEnc { gcu: [w.gcu("shape_encoder.gcu_1")?, w.gcu("shape_encoder.gcu_2")?, w.gcu("shape_encoder.gcu_3")?], glb: w.mlp("shape_encoder.mlp_glb")? })
    }
    pub fn run(&self, o: &Ops, m: &MeshIn) -> Result<Vec<f32>> {
        let n = m.tpl.n;
        let x1 = self.gcu[0].run(o, &m.pos, 3, &m.tpl, &m.geo)?;
        let x2 = self.gcu[1].run(o, &x1, self.gcu[0].out(), &m.tpl, &m.geo)?;
        let x3 = self.gcu[2].run(o, &x2, self.gcu[1].out(), &m.tpl, &m.geo)?;
        let (c, _) = cat(o, n, &[(&x1, self.gcu[0].out(), false), (&x2, self.gcu[1].out(), false), (&x3, self.gcu[2].out(), false)])?;
        let x4 = self.glb.run(o, &c, n)?;
        let g = global_max(o, &x4, n, self.glb.out())?;
        o.down(&g, self.glb.out())
    }
}

/// A PointNet++ set abstraction (SAModule): farthest points (`ratio`), each one's neighbours within `r`, PointConv's
/// MLP over [x_j, pos_j - pos_i] with the max
fn set_abstraction(o: &Ops, nn: &Mlp, x: Option<&[f32]>, cx: usize, pos: &[[f32; 3]], ratio: f32, r: f32) -> Result<(Vec<f32>, Vec<[f32; 3]>)> {
    let idx = pointnet::fps(pos, ratio);
    let edges = pointnet::radius_edges(pos, &idx, r, 64);
    let mut rows = Vec::with_capacity(edges.len() * (cx + 3));
    let mut rowptr = vec![0i32];
    for (i, nb) in edges.iter().enumerate() {
        let pi = pos[idx[i]];
        for &j in nb {
            if let Some(x) = x {
                rows.extend_from_slice(&x[j * cx..(j + 1) * cx]);
            }
            rows.extend((0..3).map(|c| pos[j][c] - pi[c]));
        }
        rowptr.push((rowptr.last().copied().unwrap_or(0) as usize + nb.len()) as i32);
    }
    let e = rowptr.last().copied().unwrap_or(0) as usize;
    let h = nn.run(o, &o.up(&rows)?, e)?;
    let m = idx.len();
    let out = o.buf(m * nn.out())?;
    o.k.seg_max(&h, nn.out(), &o.up_i32(&rowptr)?, m, &out, nn.out(), 0)?;
    Ok((o.down(&out, m * nn.out())?, idx.iter().map(|&i| pos[i]).collect()))
}

/// GlobalSAModule: the MLP over [x, pos], the max over the points
fn global_sa(o: &Ops, nn: &Mlp, x: &[f32], cx: usize, pos: &[[f32; 3]]) -> Result<Vec<f32>> {
    let rows: Vec<f32> = pos.iter().enumerate().flat_map(|(i, p)| x[i * cx..(i + 1) * cx].iter().copied().chain(p.iter().copied())).collect();
    let h = nn.run(o, &o.up(&rows)?, pos.len())?;
    let g = global_max(o, &h, pos.len(), nn.out())?;
    o.down(&g, nn.out())
}

/// FPModule: features interpolated from `src` points to `dst` ones (k nearest, inverse squared distance), the skip
/// features after them, the MLP
#[allow(clippy::too_many_arguments)]
fn feature_propagation(o: &Ops, nn: &Mlp, x: &[f32], c: usize, src: &[[f32; 3]], dst: &[[f32; 3]], skip: &[f32], cs: usize, k: usize) -> Result<Vec<f32>> {
    let xi = pointnet::knn_interpolate(x, c, src, dst, k);
    let rows: Vec<f32> = (0..dst.len()).flat_map(|i| xi[i * c..(i + 1) * c].iter().copied().chain(skip[i * cs..(i + 1) * cs].iter().copied())).collect();
    nn.run_host(o, &rows, dst.len())
}

/// models/ROOT_GCN.py ROOTNET: a root score a joint
pub struct RootNet {
    shape: ShapeEnc,
    sa: [Mlp; 3],
    fp: [Mlp; 3],
    back: Mlp,
    last: Lin,
}

impl RootNet {
    pub fn load(pth: &Pth, gpu: &Arc<Gpu>) -> Result<RootNet> {
        let w = W::new(pth, "shape_encoder.gcu_1.mlp.0.0.weight", gpu)?;
        let j = "joint_encoder";
        Ok(RootNet {
            shape: ShapeEnc::load(&w)?,
            sa: [w.mlp(&format!("{j}.sa1_joint.conv.local_nn"))?, w.mlp(&format!("{j}.sa2_joint.conv.local_nn"))?, w.mlp(&format!("{j}.sa3_joint.nn"))?],
            fp: [w.mlp(&format!("{j}.fp3_joint.nn"))?, w.mlp(&format!("{j}.fp2_joint.nn"))?, w.mlp(&format!("{j}.fp1_joint.nn"))?],
            back: w.mlp("back_layers.0")?,
            last: w.lin("back_layers.1")?,
        })
    }
    /// The root logits [J] of `joints` (shape: the shape encoder's output, `shape`)
    pub fn run(&self, o: &Ops, shape: &[f32], joints: &[[f32; 3]]) -> Result<Vec<f32>> {
        let x0: Vec<f32> = joints.iter().map(|p| p[0].abs()).collect();
        let (x1, p1) = set_abstraction(o, &self.sa[0], Some(&x0), 1, joints, 0.999, 0.4)?;
        let (x2, p2) = set_abstraction(o, &self.sa[1], Some(&x1), self.sa[0].out(), &p1, 0.33, 0.6)?;
        let x3 = global_sa(o, &self.sa[2], &x2, self.sa[1].out(), &p2)?;
        let f3 = feature_propagation(o, &self.fp[0], &x3, self.sa[2].out(), &[[0.0; 3]], &p2, &x2, self.sa[1].out(), 1)?;
        let f2 = feature_propagation(o, &self.fp[1], &f3, self.fp[0].out(), &p2, &p1, &x1, self.sa[0].out(), 3)?;
        let f1 = feature_propagation(o, &self.fp[2], &f2, self.fp[1].out(), &p1, joints, &x0, 1, 3)?;
        let cf = self.fp[2].out();
        let rows: Vec<f32> = (0..joints.len()).flat_map(|i| shape.iter().copied().chain(f1[i * cf..(i + 1) * cf].iter().copied())).collect();
        let h = self.back.run(o, &o.up(&rows)?, joints.len())?;
        let y = self.last.run(o, &h, joints.len())?;
        o.down(&y, joints.len())
    }
    pub fn shape(&self, o: &Ops, m: &MeshIn) -> Result<Vec<f32>> {
        self.shape.run(o, m)
    }
}

/// models/PairCls_GCN.py PairCls: a connection logit a pair of joints
pub struct BoneNet {
    shape: ShapeEnc,
    sa: [Mlp; 3],
    expand: Mlp,
    mix: Mlp,
    last: Lin,
}

impl BoneNet {
    pub fn load(pth: &Pth, gpu: &Arc<Gpu>) -> Result<BoneNet> {
        let w = W::new(pth, "shape_encoder.gcu_1.mlp.0.0.weight", gpu)?;
        let j = "joint_encoder";
        Ok(BoneNet {
            shape: ShapeEnc::load(&w)?,
            sa: [w.mlp(&format!("{j}.sa1_module_joints.conv.local_nn"))?, w.mlp(&format!("{j}.sa2_module_joints.conv.local_nn"))?,
                 w.mlp(&format!("{j}.sa3_module_joints.nn"))?],
            expand: w.mlp("expand_joint_feature.0")?,
            mix: w.mlp("mix_transform.0")?,
            last: w.lin("mix_transform.2")?,
        })
    }
    pub fn shape(&self, o: &Ops, m: &MeshIn) -> Result<Vec<f32>> {
        self.shape.run(o, m)
    }
    /// The logits [P] of `pairs` (rows [joint a, joint b, distance, inside proportion]: 8 values)
    pub fn run(&self, o: &Ops, shape: &[f32], joints: &[[f32; 3]], pairs: &[f32]) -> Result<Vec<f32>> {
        let p = pairs.len() / 8;
        let (x1, p1) = set_abstraction(o, &self.sa[0], None, 0, joints, 0.999, 0.4)?;
        let (x2, p2) = set_abstraction(o, &self.sa[1], Some(&x1), self.sa[0].out(), &p1, 0.33, 0.6)?;
        let jf = global_sa(o, &self.sa[2], &x2, self.sa[1].out(), &p2)?;
        let pf = self.expand.run_host(o, pairs, p)?;
        let ce = self.expand.out();
        let rows: Vec<f32> = (0..p).flat_map(|i| shape.iter().chain(&jf).copied().chain(pf[i * ce..(i + 1) * ce].iter().copied())).collect();
        let h = self.mix.run(o, &o.up(&rows)?, p)?;
        let y = self.last.run(o, &h, p)?;
        o.down(&y, p)
    }
}

/// models/SKINNING.py SKINNET (nearest_bone 5, Dg and Lf): a logit for each of a vertex's 5 nearest bones
pub struct SkinNet {
    t1: Mlp,
    gcu: [Gcu; 3],
    t2: Mlp,
    cls: [(Lin, Option<(DevBuf, DevBuf)>); 3],
}

impl SkinNet {
    pub fn load(pth: &Pth, gpu: &Arc<Gpu>) -> Result<SkinNet> {
        let w = W::new(pth, "gcu1.mlp.0.0.weight", gpu)?;
        Ok(SkinNet {
            t1: w.mlp("multi_layer_tranform1")?,
            gcu: [w.gcu("gcu1")?, w.gcu("gcu2")?, w.gcu("gcu3")?],
            t2: w.mlp("multi_layer_tranform2")?,
            cls: [(w.lin("cls_branch.0")?, Some(w.bn("cls_branch.2")?)), (w.lin("cls_branch.3")?, Some(w.bn("cls_branch.5")?)), (w.lin("cls_branch.6")?, None)],
        })
    }
    /// The logits [n, 5] from the vertices' inputs [n, 43] (position, then 5 x (bone, 1 / distance, is leaf))
    pub fn run(&self, o: &Ops, m: &MeshIn, input: &[f32]) -> Result<Vec<f32>> {
        let n = m.tpl.n;
        let x0 = self.t1.run(o, &o.up(input)?, n)?;
        let x1 = self.gcu[0].run(o, &x0, self.t1.out(), &m.tpl, &m.geo)?;
        let g = self.t2.run(o, &x1, n)?;
        let g = global_max(o, &g, n, self.t2.out())?;
        let x2 = self.gcu[1].run(o, &x1, self.gcu[0].out(), &m.tpl, &m.geo)?;
        let x3 = self.gcu[2].run(o, &x2, self.gcu[1].out(), &m.tpl, &m.geo)?;
        let (mut h, _) = cat(o, n, &[(&x3, self.gcu[2].out(), false), (&g, self.t2.out(), true)])?;
        for (l, bn) in &self.cls {
            let y = l.run(o, &h, n)?;
            if let Some((s, t)) = bn {
                o.k.relu_affine(&y, n, l.n, Some((s, t)))?;
            }
            h = y;
        }
        o.down(&h, n * self.cls[2].0.n)
    }
}
