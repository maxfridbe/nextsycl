//! What UniMate needs to know of a rig, from the rig alone (no LLM, no human review: the reference's
//! data_process/rig_preprocess labels joints and picks the facing with a vision-language model and a review; here the
//! rules): the skeleton in the order the model was trained on (breadth first, big subtrees first), the canonical pose
//! (facing +Z, the root at the XZ origin, scaled so the leaf-to-leaf diameter is 2, standing on y = 0), the joint labels
//! of `names`, and the graph's relations, distances, depths and spectral features.

use std::collections::VecDeque;
use std::path::Path;

use nextsycl_model3d::mesh::Skin;
use nextsycl_model3d::{Error, Result};

use crate::eig::eigh;
use crate::names::{self, Side};
use crate::net::{SkelIn, FEAT, J};

/// A rig's joints past which UniMate cannot go
pub const MAX_JOINTS: usize = J;
const TARGET_DIAMETER: f64 = 2.0;
const MIN_DIAMETER: f64 = 1e-6;
const FREQS: usize = 8;

/// How the facing is decided
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Facing {
    /// the sides in the joints' names, else the rig's mirror symmetry, else +Z
    Auto,
    /// the rig looks along this horizontal direction
    Dir([f64; 2]),
}

impl Facing {
    pub fn parse(s: &str) -> Result<Facing> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Facing::Auto,
            "+z" | "z" => Facing::Dir([0.0, 1.0]),
            "-z" => Facing::Dir([0.0, -1.0]),
            "+x" | "x" => Facing::Dir([1.0, 0.0]),
            "-x" => Facing::Dir([-1.0, 0.0]),
            o => return Err(Error(format!("facing: {o:?} is not auto, +z, -z, +x or -x"))),
        })
    }
}

/// Where the joint labels come from
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameMode {
    /// the rig's names cleaned by the rules; a name nothing recognizes is "Bone" (the root: "Root")
    Auto,
    /// no names: the model sees the structure alone (graph distances, edge types, depths, the rest pose)
    Structure,
}

impl NameMode {
    pub fn parse(s: &str) -> Result<NameMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(NameMode::Auto),
            "structure" | "none" => Ok(NameMode::Structure),
            o => Err(Error(format!("names: {o:?} is not auto or structure"))),
        }
    }
}

/// The map from the rig's frame to the canonical one: p_c = s (R p - (rx, 0, rz)) - (0, gy, 0)
#[derive(Clone, Copy, Debug)]
pub struct Canon {
    pub r: [[f64; 3]; 3],
    pub scale: f64,
    pub rx: f64,
    pub rz: f64,
    pub gy: f64,
}

impl Canon {
    pub fn forward(&self, p: [f64; 3]) -> [f64; 3] {
        let q = mat_vec(&self.r, p);
        [self.scale * (q[0] - self.rx), self.scale * q[1] - self.gy, self.scale * (q[2] - self.rz)]
    }

    /// A displacement of the canonical frame in the rig's
    pub fn back_delta(&self, d: [f64; 3]) -> [f64; 3] {
        let v = [d[0] / self.scale, d[1] / self.scale, d[2] / self.scale];
        // R^T v
        [0, 1, 2].map(|i| (0..3).map(|k| self.r[k][i] * v[k]).sum())
    }
}

fn mat_vec(m: &[[f64; 3]; 3], p: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2])
}

/// A rig in the model's terms
#[derive(Debug)]
pub struct Prepared {
    pub n: usize,
    /// the model's joint k is the skin's joint order[k]
    pub order: Vec<usize>,
    /// parents in the model's order (-1: the root)
    pub parents: Vec<i32>,
    /// the labels in the model's order ("" when the names are left out)
    pub labels: Vec<String>,
    /// how many joints the rules recognized
    pub recognized: usize,
    pub canon: Canon,
    /// the canonical rest positions, in the model's order
    pub tpos: Vec<[f64; 3]>,
    /// how the facing was decided
    pub facing_note: String,
}

/// The skeleton's tree in BFS order, children by (larger subtree first, then shorter bone)
fn bfs_order(parents: &[Option<usize>], len: &[f64]) -> Vec<usize> {
    let n = parents.len();
    let mut kids = vec![Vec::new(); n];
    let mut root = 0;
    for (i, p) in parents.iter().enumerate() {
        match p {
            Some(p) => kids[*p].push(i),
            None => root = i,
        }
    }
    let mut size = vec![1usize; n];
    // parents come first in a Skin, so a reverse pass sums the subtrees
    for i in (0..n).rev() {
        if let Some(p) = parents[i] {
            size[p] += size[i];
        }
    }
    for k in &mut kids {
        k.sort_by(|&a, &b| size[b].cmp(&size[a]).then(len[a].total_cmp(&len[b])));
    }
    let mut order = Vec::with_capacity(n);
    let mut q = VecDeque::from([root]);
    while let Some(u) = q.pop_front() {
        order.push(u);
        q.extend(kids[u].iter().copied());
    }
    order
}

/// The leaf-to-leaf diameter of the tree weighted by bone length (two breadth-first passes)
fn diameter(parents: &[i32], pos: &[[f64; 3]]) -> f64 {
    let n = parents.len();
    let mut adj = vec![Vec::new(); n];
    for (i, &p) in parents.iter().enumerate() {
        if p >= 0 {
            let d = dist3(pos[i], pos[p as usize]);
            adj[p as usize].push((i, d));
            adj[i].push((p as usize, d));
        }
    }
    let far = |s: usize| -> (usize, f64) {
        let mut d = vec![f64::NAN; n];
        d[s] = 0.0;
        let mut q = VecDeque::from([s]);
        let (mut best, mut bd) = (s, 0.0);
        while let Some(u) = q.pop_front() {
            for &(v, w) in &adj[u] {
                if d[v].is_nan() {
                    d[v] = d[u] + w;
                    if d[v] > bd {
                        bd = d[v];
                        best = v;
                    }
                    q.push_back(v);
                }
            }
        }
        (best, bd)
    };
    let (u, _) = far(0);
    far(u).1
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The rotation about Y that turns `f` (a horizontal direction, x z) to +Z
fn yaw_to_z(f: [f64; 2]) -> [[f64; 3]; 3] {
    let l = (f[0] * f[0] + f[1] * f[1]).sqrt();
    let (c, s) = if l > 1e-9 { (f[1] / l, -f[0] / l) } else { (1.0, 0.0) };
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// The facing of a rig: (the horizontal direction it looks along, how it was found)
fn facing_of(pos: &[[f64; 3]], parents: &[i32], sides: &[Option<Side>]) -> ([f64; 2], String) {
    let n = pos.len();
    // 1. the sides the names give: forward = Y x (right - left)
    let mean = |s: Side| -> Option<[f64; 3]> {
        let v: Vec<[f64; 3]> = (0..n).filter(|&i| sides[i] == Some(s)).map(|i| pos[i]).collect();
        (!v.is_empty()).then(|| [0, 1, 2].map(|c| v.iter().map(|p| p[c]).sum::<f64>() / v.len() as f64))
    };
    if let (Some(l), Some(r)) = (mean(Side::Left), mean(Side::Right)) {
        let a = [r[0] - l[0], r[2] - l[2]];
        if (a[0] * a[0] + a[1] * a[1]).sqrt() > 1e-6 {
            return ([a[1], -a[0]], "the sides in the joints' names".into());
        }
    }
    // 2. the mirror symmetry: the horizontal axis a that maps the joints onto themselves best
    let c: [f64; 3] = [0, 1, 2].map(|k| pos.iter().map(|p| p[k]).sum::<f64>() / n as f64);
    let diam = diameter(parents, pos).max(1e-9);
    let err = |th: f64| -> f64 {
        let a = [th.cos(), th.sin()];
        let mut tot = 0.0;
        for p in pos {
            let t = (p[0] - c[0]) * a[0] + (p[2] - c[2]) * a[1];
            let m = [p[0] - 2.0 * t * a[0], p[1], p[2] - 2.0 * t * a[1]];
            tot += pos.iter().map(|q| dist3(*q, m)).fold(f64::INFINITY, f64::min);
        }
        tot / n as f64 / diam
    };
    let steps = 90;
    let (mut best, mut be) = (0.0, f64::INFINITY);
    for k in 0..steps {
        let th = std::f64::consts::PI * k as f64 / steps as f64;
        let e = err(th);
        if e < be {
            (best, be) = (th, e);
        }
    }
    let mut h = std::f64::consts::PI / steps as f64 / 2.0;
    while h > 1e-4 {
        for th in [best - h, best + h] {
            let e = err(th);
            if e < be {
                (best, be) = (th, e);
            }
        }
        h /= 2.0;
    }
    if be > 0.02 {
        return ([0.0, 1.0], format!("+Z (the rig is not mirror symmetric: error {be:.3} of its diameter)"));
    }
    // the mirror axis a is the lateral one; the facing is perpendicular to it, and which way is told by the feet
    let f = [-best.sin(), best.cos()];
    let low: f64 = pos.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let high: f64 = pos.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
    let mut has_child = vec![false; n];
    for &p in parents {
        if p >= 0 {
            has_child[p as usize] = true;
        }
    }
    let mut vote = 0.0;
    for i in 0..n {
        if !has_child[i] && parents[i] >= 0 && pos[i][1] < low + 0.2 * (high - low) {
            let p = pos[parents[i] as usize];
            vote += (pos[i][0] - p[0]) * f[0] + (pos[i][2] - p[2]) * f[1];
        }
    }
    if vote.abs() > 1e-3 * diam {
        let s = vote.signum();
        return ([f[0] * s, f[1] * s], format!("the mirror symmetry (error {be:.3}) and the feet pointing forward"));
    }
    // no feet to tell: the direction nearer +Z
    let s = if f[1] >= 0.0 { 1.0 } else { -1.0 };
    ([f[0] * s, f[1] * s], format!("the mirror symmetry (error {be:.3}); which way it looks could not be told, +Z side taken"))
}

/// The rig in the model's terms
pub fn prepare(skin: &Skin, facing: Facing, mode: NameMode) -> Result<Prepared> {
    let n = skin.joints.len();
    if n < 2 {
        return Err(Error("UniMate animates a skeleton of at least two joints".into()));
    }
    if n > MAX_JOINTS {
        return Err(Error(format!("UniMate takes at most {MAX_JOINTS} joints; this rig has {n} (reduce it: fewer bones)")));
    }
    let roots = skin.joints.iter().filter(|j| j.parent.is_none()).count();
    if roots != 1 {
        return Err(Error(format!("UniMate needs one root joint; this rig has {roots}")));
    }
    let parents: Vec<Option<usize>> = skin.joints.iter().map(|j| j.parent).collect();
    if parents.iter().enumerate().any(|(i, p)| p.is_some_and(|p| p >= i)) {
        return Err(Error("the skeleton lists a joint before its parent".into()));
    }
    let head = |i: usize| skin.joints[i].head.map(f64::from);
    let len: Vec<f64> = (0..n).map(|i| parents[i].map_or(0.0, |p| dist3(head(i), head(p)))).collect();
    let order = bfs_order(&parents, &len);
    let mut newidx = vec![0usize; n];
    for (k, &o) in order.iter().enumerate() {
        newidx[o] = k;
    }
    let mp: Vec<i32> = order.iter().map(|&o| parents[o].map_or(-1, |p| newidx[p] as i32)).collect();
    let pos: Vec<[f64; 3]> = order.iter().map(|&o| head(o)).collect();

    // the labels
    let mut labels = Vec::with_capacity(n);
    let mut sides = Vec::with_capacity(n);
    let mut recognized = 0;
    for (k, &o) in order.iter().enumerate() {
        let l = names::clean(&skin.joints[o].name);
        if l.is_some() {
            recognized += 1;
        }
        sides.push(l.as_ref().and_then(|l| l.side));
        labels.push(match mode {
            NameMode::Structure => String::new(),
            NameMode::Auto => l.map(|l| l.text()).unwrap_or_else(|| if k == 0 { "Root".to_string() } else { "Bone".to_string() }),
        });
    }

    // the canonical pose: facing, root to the XZ origin, diameter 2, on the ground
    let (f, facing_note) = match facing {
        Facing::Dir(d) => (d, "the facing option".to_string()),
        Facing::Auto => facing_of(&pos, &mp, &sides),
    };
    let r = yaw_to_z(f);
    let rotated: Vec<[f64; 3]> = pos.iter().map(|p| mat_vec(&r, *p)).collect();
    let d = diameter(&mp, &pos);
    if !d.is_finite() || d <= MIN_DIAMETER {
        return Err(Error(format!("degenerate skeleton: leaf-to-leaf diameter {d:.3e} (zero-length bones?)")));
    }
    let scale = TARGET_DIAMETER / d;
    let (rx, rz) = (rotated[0][0], rotated[0][2]);
    let gy = rotated.iter().map(|p| scale * p[1]).fold(f64::INFINITY, f64::min);
    let canon = Canon { r, scale, rx, rz, gy };
    let tpos: Vec<[f64; 3]> = pos.iter().map(|p| canon.forward(*p)).collect();
    Ok(Prepared { n, order, parents: mp, labels, recognized, canon, tpos, facing_note })
}

/// The graph's edge classes (0 self, 1 parent, 2 child, 3 sibling, 4 none, 5 a leaf's own) and BFS distances (to 5)
pub fn relations(parents: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let n = parents.len();
    let mut has_child = vec![false; n];
    for &p in parents {
        if p >= 0 {
            has_child[p as usize] = true;
        }
    }
    let mut rel = vec![4i32; n * n];
    for i in 0..n {
        for j in 0..n {
            rel[i * n + j] = if i == j {
                0
            } else if parents[j] == i as i32 {
                2
            } else if parents[i] >= 0 && j as i32 == parents[i] {
                1
            } else if parents[i] >= 0 && parents[j] == parents[i] {
                3
            } else {
                4
            };
        }
        if !has_child[i] {
            rel[i * n + i] = 5;
        }
    }
    let mut adj = vec![Vec::new(); n];
    for (c, &p) in parents.iter().enumerate() {
        if p >= 0 {
            adj[c].push(p as usize);
            adj[p as usize].push(c);
        }
    }
    let mut dist = vec![5i32; n * n];
    for s in 0..n {
        let mut d = vec![i32::MAX; n];
        d[s] = 0;
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            if d[u] >= 5 {
                continue;
            }
            for &v in &adj[u] {
                if d[v] > d[u] + 1 {
                    d[v] = d[u] + 1;
                    q.push_back(v);
                }
            }
        }
        for v in 0..n {
            dist[s * n + v] = d[v].min(5);
        }
    }
    (rel, dist)
}

/// Depth of each joint below the root
pub fn depths(parents: &[i32]) -> Vec<i32> {
    let mut d = vec![0i32; parents.len()];
    for i in 0..parents.len() {
        if parents[i] >= 0 {
            d[i] = d[parents[i] as usize] + 1;
        }
    }
    d
}

/// Eigenvectors 1..k of the symmetric normalized Laplacian (each L2-normalized, zero-padded to 8 columns), [n, 8]. The
/// basis of a degenerate eigenspace (a symmetric skeleton's) is arbitrary, as numpy's is; the model's SignNet takes any sign.
pub fn spectral(parents: &[i32]) -> Vec<f32> {
    let n = parents.len();
    let mut a = vec![0f64; n * n];
    for (c, &p) in parents.iter().enumerate() {
        if p >= 0 {
            a[p as usize * n + c] = 1.0;
            a[c * n + p as usize] = 1.0;
        }
    }
    let deg: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[i * n + j]).sum()).collect();
    let dinv: Vec<f64> = deg.iter().map(|d| if *d > 0.0 { d.powf(-0.5) } else { 0.0 }).collect();
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            l[i * n + j] = f64::from(i == j) - dinv[i] * a[i * n + j] * dinv[j];
        }
    }
    let (_, v) = eigh(&l, n);
    let k = (n - 1).min(FREQS);
    let mut out = vec![0f32; n * FREQS];
    for c in 0..k {
        let norm = (0..n).map(|i| v[i * n + c + 1].powi(2)).sum::<f64>().sqrt();
        for i in 0..n {
            out[i * FREQS + c] = (if norm > 1e-12 { v[i * n + c + 1] / norm } else { v[i * n + c + 1] }) as f32;
        }
    }
    out
}

/// The statistics the model's features are normalized with (dataset_stats.npy): the root's and the joints' (12 each)
#[derive(Clone, Debug)]
pub struct Stats {
    pub mean_root: [f64; FEAT],
    pub std_root: [f64; FEAT],
    pub mean_local: [f64; FEAT],
    pub std_local: [f64; FEAT],
}

impl Stats {
    /// Read from the .npy holding a pickled dict, without running the pickle: each array is found after its key as the
    /// next 96-byte string (12 float64)
    pub fn open(path: &Path) -> Result<Stats> {
        let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let get = |key: &str| -> Result<[f64; FEAT]> {
            let at = find(&b, key.as_bytes(), 0).ok_or_else(|| Error(format!("{}: no {key}", path.display())))?;
            let mut from = at + key.len();
            loop {
                let p = find(&b, &[0x43, 96], from).ok_or_else(|| Error(format!("{}: no data after {key}", path.display())))?;
                if p + 2 + 96 <= b.len() {
                    let v: Vec<f64> = b[p + 2..p + 2 + 96].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect();
                    if v.iter().all(|x| x.is_finite()) {
                        let mut out = [0f64; FEAT];
                        out.copy_from_slice(&v);
                        return Ok(out);
                    }
                }
                from = p + 1;
            }
        };
        Ok(Stats { mean_root: get("mean_root")?, std_root: get("std_root")?, mean_local: get("mean_local")?, std_local: get("std_local")? })
    }

    /// The normalization rows for `n` joints padded to 71: (mean, std), the root's first; padding mean 0, std 1
    pub fn rows(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut mean, mut std) = (vec![0f64; J * FEAT], vec![1f64; J * FEAT]);
        for j in 0..n {
            let (m, s) = if j == 0 { (&self.mean_root, &self.std_root) } else { (&self.mean_local, &self.std_local) };
            mean[j * FEAT..(j + 1) * FEAT].copy_from_slice(m);
            std[j * FEAT..(j + 1) * FEAT].copy_from_slice(s);
        }
        (mean, std)
    }
}

fn find(h: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    h.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

/// The model's inputs for a prepared rig; `names_emb` [n, 768] the joints' pooled name embeddings (zeros: no names)
pub fn skel_in(p: &Prepared, stats: &Stats, names_emb: &[f32]) -> SkelIn {
    let n = p.n;
    let (mean, std) = stats.rows(n);
    let mut tpos = vec![0f32; J * FEAT];
    for j in 0..n {
        // [position, the identity rotation as 6D, three zeros], normalized
        let row = [p.tpos[j][0], p.tpos[j][1], p.tpos[j][2], 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        for c in 0..FEAT {
            let v = (row[c] - mean[j * FEAT + c]) / std[j * FEAT + c];
            tpos[j * FEAT + c] = if v.is_finite() { v as f32 } else { 0.0 };
        }
    }
    let mut parents_t = vec![0f32; J * FEAT];
    for j in 0..n {
        let src = if p.parents[j] >= 0 { p.parents[j] as usize } else { j };
        parents_t[j * FEAT..(j + 1) * FEAT].copy_from_slice(&tpos[src * FEAT..(src + 1) * FEAT]);
    }
    let (rel, dist) = relations(&p.parents);
    let (mut relations_p, mut dist_p) = (vec![0i32; J * J], vec![0i32; J * J]);
    for i in 0..n {
        for j in 0..n {
            relations_p[i * J + j] = rel[i * n + j];
            dist_p[i * J + j] = dist[i * n + j];
        }
    }
    let mut depth = vec![0i32; J];
    depth[..n].copy_from_slice(&depths(&p.parents));
    let sp = spectral(&p.parents);
    let mut spectral_p = vec![0f32; J * FREQS];
    spectral_p[..n * FREQS].copy_from_slice(&sp);
    let mut names = vec![0f32; J * crate::t5::DIM];
    names[..names_emb.len()].copy_from_slice(names_emb);
    SkelIn { n, tpos, tpos_parents: parents_t, depths: depth, spectral: spectral_p, graph_dist: dist_p, relations: relations_p, names_emb: names }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextsycl_model3d::mesh::Joint;

    fn joint(name: &str, parent: Option<usize>, head: [f32; 3]) -> Joint {
        Joint { name: name.into(), parent, head }
    }

    /// A little biped looking along +X (its left, by the names, at +Z)
    fn biped() -> Skin {
        Skin {
            joints: vec![
                joint("Hips", None, [0.0, 1.0, 0.0]),
                joint("Spine", Some(0), [0.0, 1.4, 0.0]),
                joint("Head", Some(1), [0.0, 1.8, 0.0]),
                joint("LeftUpLeg", Some(0), [0.0, 0.9, 0.15]),
                joint("LeftFoot", Some(3), [0.0, 0.1, 0.15]),
                joint("RightUpLeg", Some(0), [0.0, 0.9, -0.15]),
                joint("RightFoot", Some(5), [0.0, 0.1, -0.15]),
            ],
            weights: Vec::new(),
            motion: None,
        }
    }

    #[test]
    fn the_names_set_the_facing_and_the_pose_is_canonical() {
        let p = prepare(&biped(), Facing::Auto, NameMode::Auto).unwrap();
        // right = -z, left = +z: forward = Y x (right - left) = Y x (0, 0, -0.3) = (-0.3, 0, 0): looks along -X
        assert!(p.facing_note.contains("names"), "{}", p.facing_note);
        assert_eq!(p.parents[0], -1);
        // canonical: root at the XZ origin, feet on the ground, diameter 2
        assert!(p.tpos[0][0].abs() < 1e-9 && p.tpos[0][2].abs() < 1e-9);
        let low = p.tpos.iter().map(|t| t[1]).fold(f64::INFINITY, f64::min);
        assert!(low.abs() < 1e-9);
        assert!((diameter(&p.parents, &p.tpos) - 2.0).abs() < 1e-9);
        // the left leg is on +X in the canonical frame
        let lf = p.labels.iter().position(|l| l == "Left Foot").unwrap();
        let rf = p.labels.iter().position(|l| l == "Right Foot").unwrap();
        assert!(p.tpos[lf][0] > 0.0 && p.tpos[rf][0] < 0.0);
        // the canonical frame maps back
        let back = p.canon.back_delta([0.0, 0.0, p.canon.scale]);
        assert!((back[0].hypot(back[2]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn structure_mode_drops_the_names_and_generic_names_become_bone() {
        let mut s = biped();
        for (i, j) in s.joints.iter_mut().enumerate() {
            j.name = format!("bone_{i}");
        }
        let p = prepare(&s, Facing::Dir([0.0, 1.0]), NameMode::Auto).unwrap();
        assert_eq!(p.recognized, 0);
        assert_eq!(p.labels[0], "Root");
        assert!(p.labels[1..].iter().all(|l| l == "Bone"));
        let q = prepare(&s, Facing::Auto, NameMode::Structure).unwrap();
        assert!(q.labels.iter().all(String::is_empty));
        // the symmetry finds the lateral axis (z) and the feet-less rig falls to the +Z side of it
        assert!(q.facing_note.contains("symmetr"), "{}", q.facing_note);
    }

    #[test]
    fn relations_depths_and_spectrum() {
        let parents = [-1, 0, 1, 0, 3, 0, 5];
        let (rel, dist) = relations(&parents);
        assert_eq!(rel[0], 0);
        assert_eq!(rel[1], 2); // joint 1 is a child of 0: from 0's row, "child"
        assert_eq!(rel[7], 1); // from 1: its parent is 0
        assert_eq!(rel[3 * 7 + 1], 3); // 3 and 1 are siblings
        assert_eq!(rel[2 * 7 + 2], 5); // a leaf
        assert_eq!(dist[2 * 7 + 6], 4);
        assert_eq!(dist[2 * 7 + 1], 1);
        assert_eq!(depths(&parents), vec![0, 1, 2, 1, 2, 1, 2]);
        let s = spectral(&parents);
        for c in 0..6 {
            let norm: f32 = (0..7).map(|i| s[i * 8 + c] * s[i * 8 + c]).sum();
            assert!((norm - 1.0).abs() < 1e-5);
        }
        assert_eq!(s[6], 0.0);
        assert_eq!(s[7], 0.0);
    }

    #[test]
    fn too_big_and_rooted_wrong() {
        let mut s = biped();
        s.joints[3].parent = None;
        assert!(prepare(&s, Facing::Auto, NameMode::Auto).is_err());
        let big = Skin { joints: (0..72).map(|i| joint("b", if i == 0 { None } else { Some(0) }, [i as f32, 0.0, 0.0])).collect(), weights: Vec::new(), motion: None };
        assert!(prepare(&big, Facing::Auto, NameMode::Auto).unwrap_err().0.contains("71"));
    }
}
