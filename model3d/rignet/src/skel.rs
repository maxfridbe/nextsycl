//! From network outputs to a skeleton and skin, on the CPU, as RigNet's numpy code does: the joints by weighted
//! mean-shift of the shifted vertices, a density threshold and non-maximum suppression (utils/cluster_utils.py,
//! quick_start.predict_joints), symmetrized (mst_utils.flip); the joint pairs' inputs; the connection costs and the
//! symmetric Prim spanning tree (mst_utils.primMST_symmetry); the bones (common_ops.get_bones); SkinNet's per-vertex
//! inputs and its weights' smoothing (run_skinning.post_filter).

use crate::geom::Vox;

fn d2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|c| (a[c] - b[c]) * (a[c] - b[c])).sum()
}

/// Weighted mean-shift (cluster_utils.meanshift_cluster): points move 0.3 of the way to their kernel-weighted means,
/// until they move less than 1e-3 in all or `max_iter` - 1 rounds
pub fn meanshift(pts: &mut [[f64; 3]], bandwidth: f64, w: &[f64], max_iter: usize) {
    let n = pts.len();
    let mut it = 1;
    let mut diff = 1e10;
    let b2 = bandwidth * bandwidth;
    while diff > 1e-3 && it < max_iter {
        // K[i][j] = max(b^2 - |p_i - p_j|^2, 0) * w[i]; a point's new place: its column of K, normalized, over the points
        let p: &[[f64; 3]] = pts;
        let colsum = crate::geom::par_rows(n, |j| (0..n).map(|i| (b2 - d2(p[i], p[j])).max(0.0) * w[i]).sum::<f64>());
        let next = crate::geom::par_rows(n, |a| {
            let mut m = [0f64; 3];
            for b in 0..n {
                let k = (b2 - d2(p[b], p[a])).max(0.0) * w[b] / (colsum[a] + 1e-10);
                for c in 0..3 {
                    m[c] += k * p[b][c];
                }
            }
            [0, 1, 2].map(|c| 0.3 * (m[c] - p[a][c]) + p[a][c])
        });
        diff = next.iter().zip(pts.iter()).map(|(a, b)| d2(*a, *b)).sum::<f64>().sqrt();
        pts.copy_from_slice(&next);
        it += 1;
    }
}

/// Each point's density: the sum over all points of max(b^2 - d^2, 0)
pub fn density(pts: &[[f64; 3]], bandwidth: f64) -> Vec<f64> {
    let b2 = bandwidth * bandwidth;
    crate::geom::par_rows(pts.len(), |j| pts.iter().map(|p| (b2 - d2(*p, pts[j])).max(0.0)).sum())
}

/// cluster_utils.nms_meanshift: from the densest down, a point suppresses those within the bandwidth
pub fn nms(pts: &[[f64; 3]], dens: &[f64], bandwidth: f64) -> Vec<[f64; 3]> {
    let n = pts.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| dens[a].total_cmp(&dens[b]).then(a.cmp(&b)));
    order.reverse();
    let mut unique = vec![true; n];
    for &i in &order {
        if unique[i] {
            for j in 0..n {
                if d2(pts[j], pts[i]).sqrt() <= bandwidth {
                    unique[j] = false;
                }
            }
            unique[i] = true;
        }
    }
    (0..n).filter(|&i| unique[i]).map(|i| pts[i]).collect()
}

/// mst_utils.flip: the joints left of the middle (x < -0.02), the middle ones (on x = 0), the left ones mirrored
pub fn flip(j: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let left: Vec<[f64; 3]> = j.iter().filter(|p| p[0] < -2e-2).copied().collect();
    let mid = j.iter().filter(|p| p[0].abs() <= 2e-2).map(|p| [0.0, p[1], p[2]]);
    let right = left.iter().map(|p| [-p[0], p[1], p[2]]);
    left.iter().copied().chain(mid).chain(right).collect()
}

/// quick_start.predict_joints from the network's outputs: shifted vertices inside the volume with attention over 1e-3,
/// mirrored, mean-shifted, kept above the density threshold, suppressed, symmetrized
pub fn joints_from(pos: &[[f32; 3]], disp: &[f32], attn: &[f32], vox: &Vox, bandwidth: f64, threshold: f64) -> Vec<[f64; 3]> {
    let mut p = Vec::new();
    let mut w = Vec::new();
    for (i, q) in pos.iter().enumerate() {
        let y = [q[0] + disp[i * 3], q[1] + disp[i * 3 + 1], q[2] + disp[i * 3 + 2]];
        if vox.inside(y) && attn[i] > 1e-3 {
            p.push(y.map(|x| x as f64));
            w.push(attn[i] as f64);
        }
    }
    let m = p.len();
    for i in 0..m {
        p.push([-p[i][0], p[i][1], p[i][2]]);
        w.push(w[i]);
    }
    meanshift(&mut p, bandwidth, &w, 40);
    let dens = density(&p, bandwidth);
    let sum: f64 = dens.iter().sum();
    let keep: Vec<usize> = (0..p.len()).filter(|&i| dens[i] / sum > threshold).collect();
    let pk: Vec<[f64; 3]> = keep.iter().map(|&i| p[i]).collect();
    let dk: Vec<f64> = keep.iter().map(|&i| dens[i]).collect();
    flip(&nms(&pk, &dk, bandwidth))
}

/// mst_generate.sample_on_bone: points every ~0.01 from a to b (b included, a not)
pub fn sample_on_bone(a: [f64; 3], b: [f64; 3]) -> Vec<[f64; 3]> {
    let len = d2(a, b).sqrt();
    let steps = (len / 0.01).round_ties_even() as usize;
    (1..=steps).map(|i| [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) / (steps as f64 + 1e-30) * i as f64)).collect()
}

fn inside(vox: &Vox, p: [f64; 3]) -> bool {
    vox.inside(p.map(|x| x as f32))
}

/// Every pair (i < j) of joints, in order: [joint i, joint j, distance, the share of the bone's samples inside] (the
/// BoneNet input; quick_start.predict_joints's pair_attr less its last column)
pub fn pairs(j: &[[f64; 3]], vox: &Vox) -> (Vec<(usize, usize)>, Vec<f32>) {
    let mut ps = Vec::new();
    let mut rows = Vec::new();
    for a in 0..j.len() {
        for b in a + 1..j.len() {
            let s = sample_on_bone(j[a], j[b]);
            let inn = s.iter().filter(|p| inside(vox, **p)).count();
            ps.push((a, b));
            rows.extend(j[a].iter().chain(j[b].iter()).map(|x| *x as f32));
            rows.push(d2(j[a], j[b]).sqrt() as f32);
            rows.push((inn as f64 / (s.len() as f64 + 1e-10)) as f32);
        }
    }
    (ps, rows)
}

/// The connection costs: -log of the symmetric probabilities, raised for bones that leave the volume and halved for
/// pairs on the middle line (mst_utils.increase_cost_for_outside_bone)
pub fn costs(j: &[[f64; 3]], pairs: &[(usize, usize)], prob: &[f32], vox: &Vox) -> Vec<f64> {
    let n = j.len();
    let mut p = vec![0f64; n * n];
    for (k, &(a, b)) in pairs.iter().enumerate() {
        p[a * n + b] = prob[k] as f64;
    }
    let mut c: Vec<f64> = (0..n * n).map(|k| -((p[k] + p[(k % n) * n + k / n]) + 1e-10).ln()).collect();
    for a in 0..n {
        for b in a + 1..n {
            let s = sample_on_bone(j[a], j[b]);
            let out = s.iter().filter(|q| !inside(vox, **q)).count();
            if out > 1 {
                c[a * n + b] = 2.0 * out as f64;
                c[b * n + a] = 2.0 * out as f64;
            }
            if j[a][0].abs() < 2e-2 && j[b][0].abs() < 2e-2 {
                c[a * n + b] *= 0.5;
                c[b * n + a] *= 0.5;
            }
        }
    }
    c
}

/// mst_utils.primMST_symmetry: Prim's tree from the root (moved to the nearest joint on the middle line), a left
/// joint taken with its mirror (and the mirror's parent mirrored). (parents, -1 for the root; the root)
pub fn prim_symmetric(g: &[f64], root: usize, j: &[[f64; 3]]) -> (Vec<i64>, usize) {
    let n = j.len();
    let left: Vec<usize> = (0..n).filter(|&i| j[i][0] < -2e-2).collect();
    let mid: Vec<usize> = (0..n).filter(|&i| j[i][0].abs() <= 2e-2).collect();
    let right: Vec<usize> = (0..n).filter(|&i| j[i][0] > 2e-2).collect();
    let mut mirror = vec![usize::MAX; n];
    for (a, b) in left.iter().zip(&right) {
        mirror[*a] = *b;
        mirror[*b] = *a;
    }
    let mut init = root;
    if !mid.contains(&init) && !mid.is_empty() {
        let d = |m: usize| d2(j[m], j[init]).sqrt();
        init = *mid.iter().min_by(|a, b| d(**a).total_cmp(&d(**b))).unwrap_or(&mid[0]);
    }
    let side = |i: i64| -> u8 {
        if i < 0 { 3 } else if left.contains(&(i as usize)) { 0 } else if mid.contains(&(i as usize)) { 1 } else if right.contains(&(i as usize)) { 2 } else { 3 }
    };
    let mut key = vec![f64::MAX; n];
    let mut parent = vec![i64::MIN; n];
    let mut done = vec![false; n];
    key[init] = 0.0;
    parent[init] = -1;
    while done.iter().any(|d| !d) {
        let Some(u) = (0..n).filter(|&v| !done[v] && key[v] < f64::MAX).min_by(|a, b| key[*a].total_cmp(&key[*b]).then(a.cmp(b))) else {
            // what Prim cannot reach (no positive cost to it): hung on the root
            for v in 0..n {
                if !done[v] {
                    done[v] = true;
                    parent[v] = init as i64;
                }
            }
            break;
        };
        let mut u2 = None;
        if side(u as i64) != 1 && side(u as i64) != 3 && mirror[u] != usize::MAX {
            let ps = side(parent[u]);
            if ps <= 2 {
                let m = mirror[u];
                u2 = Some(m);
                if !done[m] {
                    done[m] = true;
                    parent[m] = if ps == 1 { parent[u] } else { mirror[parent[u] as usize] as i64 };
                    key[m] = g[m * n + parent[m] as usize];
                }
            }
        }
        done[u] = true;
        for v in 0..n {
            if g[u * n + v] > 0.0 && !done[v] && key[v] > g[u * n + v] {
                key[v] = g[u * n + v];
                parent[v] = u as i64;
            }
            if let Some(m) = u2 {
                if g[m * n + v] > 0.0 && !done[v] && key[v] > g[m * n + v] {
                    key[v] = g[m * n + v];
                    parent[v] = m as i64;
                }
            }
        }
    }
    (parent, init)
}

/// The skeleton as RigNet's tree holds it: joints in breadth-first order from the root, each one's children in index
/// order (loadSkel_recur, get_joint_dict). (order: the old indices; parents in the new order)
pub fn tree_order(parent: &[i64], root: usize) -> (Vec<usize>, Vec<Option<usize>>) {
    let n = parent.len();
    let mut order = vec![root];
    let mut at = vec![usize::MAX; n];
    at[root] = 0;
    let mut k = 0;
    while k < order.len() {
        let p = order[k];
        for c in 0..n {
            if parent[c] == p as i64 && at[c] == usize::MAX {
                at[c] = order.len();
                order.push(c);
            }
        }
        k += 1;
    }
    let parents = order.iter().map(|&o| (parent[o] >= 0 && at[parent[o] as usize] != usize::MAX).then(|| at[parent[o] as usize])).collect();
    (order, parents)
}

/// common_ops.get_bones over joints in tree order: breadth first, a bone (parent, child) for every child and a
/// "leaf" bone (child, child) after each childless one. (bones [a | b], is leaf, the joint each bone's weight binds to:
/// its parent end, a leaf bone its joint)
pub fn bones(pos: &[[f64; 3]], parents: &[Option<usize>]) -> (Vec<[f32; 6]>, Vec<bool>, Vec<usize>) {
    let n = pos.len();
    let kids = |p: usize| (0..n).filter(move |&c| parents[c] == Some(p));
    let (mut b, mut leaf, mut bind) = (Vec::new(), Vec::new(), Vec::new());
    let mut level = vec![0usize];
    while !level.is_empty() {
        let mut next = Vec::new();
        for &p in &level {
            for c in kids(p) {
                next.push(c);
                b.push([pos[p][0], pos[p][1], pos[p][2], pos[c][0], pos[c][1], pos[c][2]].map(|x| x as f32));
                leaf.push(false);
                bind.push(p);
                if kids(c).next().is_none() {
                    b.push([pos[c][0], pos[c][1], pos[c][2], pos[c][0], pos[c][1], pos[c][2]].map(|x| x as f32));
                    leaf.push(true);
                    bind.push(c);
                }
            }
        }
        level = next;
    }
    (b, leaf, bind)
}

/// SkinNet's input of each vertex [n, 43] (position, then its 5 nearest bones by volumetric geodesic distance: the bone,
/// 1 / distance, is leaf; past the bone count the nearest again, masked) and the bones' ids and mask [n, 5]
pub fn skin_input(pos: &[[f32; 3]], geo: &[f32], bones: &[[f32; 6]], leaf: &[bool]) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
    let nb = bones.len();
    let mut input = Vec::with_capacity(pos.len() * 43);
    let (mut ids, mut mask) = (Vec::new(), Vec::new());
    for (v, p) in pos.iter().enumerate() {
        let row = &geo[v * nb..(v + 1) * nb];
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|a, b| row[*a].total_cmp(&row[*b]).then(a.cmp(b)));
        input.extend_from_slice(p);
        for i in 0..5 {
            let (b, m) = if i < nb { (order[i], 1.0) } else { (order[0], 0.0) };
            input.extend_from_slice(&bones[b]);
            input.push((1.0 / (row[b] as f64 + 1e-10)) as f32);
            input.push(if leaf[b] { 1.0 } else { 0.0 });
            ids.push(if i < nb { b } else { 0 });
            mask.push(m);
        }
    }
    (input, ids, mask)
}

/// The full weights [n, bones] from SkinNet's logits [n, 5]: softmax, masked, placed (a masked slot, when there are
/// fewer than 5 bones, writes its 0 over bone 0, as the reference does), averaged over each vertex's one-ring
/// (post_filter: the neighbours alone), cut below 0.35 of each vertex's largest, renormalized
pub fn skin_weights(logits: &[f32], ids: &[usize], mask: &[f32], nb: usize, ring: &[Vec<u32>]) -> Vec<f32> {
    let n = logits.len() / 5;
    let mut full = vec![0f64; n * nb];
    for v in 0..n {
        let l = &logits[v * 5..v * 5 + 5];
        let mx = l.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let e: Vec<f64> = l.iter().map(|x| ((x - mx) as f64).exp()).collect();
        let s: f64 = e.iter().sum();
        for k in 0..5 {
            full[v * nb + ids[v * 5 + k]] = e[k] / s * mask[v * 5 + k] as f64;
        }
    }
    let mut out = vec![0f32; n * nb];
    for v in 0..n {
        let nbh = &ring[v];
        if nbh.is_empty() {
            for b in 0..nb {
                out[v * nb + b] = full[v * nb + b] as f32;
            }
            continue;
        }
        for b in 0..nb {
            out[v * nb + b] = (nbh.iter().map(|&u| full[u as usize * nb + b]).sum::<f64>() / nbh.len() as f64) as f32;
        }
    }
    for row in out.chunks_mut(nb.max(1)) {
        let mx = row.iter().fold(0f32, |a, b| a.max(*b));
        for x in row.iter_mut() {
            if *x < mx * 0.35 {
                *x = 0.0;
            }
        }
        let s: f32 = row.iter().sum();
        for x in row.iter_mut() {
            *x /= s + 1e-10;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meanshift_gathers_clusters_and_nms_keeps_one_each() {
        let mut p = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.012, 0.0], [1.0, 1.0, 1.0], [1.01, 1.0, 1.0]];
        let w = vec![1.0; 5];
        meanshift(&mut p, 0.05, &w, 40);
        assert!(d2(p[0], p[1]) < 1e-4 && d2(p[3], p[4]) < 1e-4);
        let d = density(&p, 0.05);
        let k = nms(&p, &d, 0.05);
        assert_eq!(k.len(), 2);
    }

    #[test]
    fn flip_mirrors_the_left() {
        let f = flip(&[[0.3, 1.0, 0.0], [-0.2, 0.5, 0.0], [0.01, 0.2, 0.0]]);
        assert_eq!(f, vec![[-0.2, 0.5, 0.0], [0.0, 0.2, 0.0], [0.2, 0.5, 0.0]]);
    }

    #[test]
    fn prim_builds_a_symmetric_tree() {
        // left arm 0, spine 1 (root) and 2 (head), right arm 3
        let j = [[-0.3, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.5, 0.0], [0.3, 1.0, 0.0]];
        let n = 4;
        let mut g = vec![10.0; n * n];
        let set = |g: &mut Vec<f64>, a: usize, b: usize, c: f64| {
            g[a * n + b] = c;
            g[b * n + a] = c;
        };
        set(&mut g, 0, 1, 1.0);
        set(&mut g, 1, 2, 1.0);
        set(&mut g, 1, 3, 5.0);
        let (parent, root) = prim_symmetric(&g, 2, &j);
        assert_eq!(root, 2);
        // the right arm follows the left one's parent although its own edge is dear
        assert_eq!(parent, vec![1, 2, -1, 1]);
        let (order, parents) = tree_order(&parent, root);
        assert_eq!(order, vec![2, 1, 0, 3]);
        assert_eq!(parents, vec![None, Some(0), Some(1), Some(1)]);
        let pos: Vec<[f64; 3]> = order.iter().map(|&o| j[o]).collect();
        let (b, leaf, bind) = bones(&pos, &parents);
        // 2-1, 1-0, leaf 0, 1-3, leaf 3
        assert_eq!(b.len(), 5);
        assert_eq!(leaf, vec![false, false, true, false, true]);
        assert_eq!(bind, vec![0, 1, 2, 1, 3]);
    }

    #[test]
    fn skin_weights_smooth_and_cut() {
        // three vertices in a row, five bones; vertex 0 all on bone 0, vertex 2 all on bone 1
        let z = -50.0;
        let logits = vec![0.0, z, z, z, z, 0.0, 0.0, z, z, z, 0.0, z, z, z, z];
        let ids = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 0, 2, 3, 4];
        let mask = vec![1.0; 15];
        let ring = vec![vec![1], vec![0, 2], vec![1]];
        let w = skin_weights(&logits, &ids, &mask, 5, &ring);
        // vertex 1: the mean of vertex 0 and vertex 2: half each
        assert!((w[5] - 0.5).abs() < 1e-4 && (w[6] - 0.5).abs() < 1e-4);
        // vertex 0 takes vertex 1's (half and half)
        assert!((w[0] - 0.5).abs() < 1e-4 && (w[1] - 0.5).abs() < 1e-4);
        assert_eq!(sample_on_bone([0.0; 3], [0.05, 0.0, 0.0]).len(), 5);
    }
}
