//! The surface voxels of a mesh (sparc3d.ops.voxelize: MeshVoxelizer and adaptive_vox, its Triton kernels as float32
//! loops): the mesh centred on its eps-expanded box and scaled so it fills the grid less a 3-voxel margin, every voxel
//! a triangle's box covers tested by the separating-axis theorem (the box's 3 axes, the 9 edge cross products, the
//! triangle's plane), the hits made unique by their key (z << 22 | y << 11 | x) and returned in that key's order.
//! adaptive_vox shrinks the grid by 5 % while more voxels than the budget come out.

use std::thread;

/// (the voxels, the grid it settled at)
pub fn adaptive(pos: &[[f32; 3]], tris: &[[u32; 3]], resolution: usize, budget: usize) -> Result<(Vec<[i32; 3]>, usize), String> {
    let mut scale = 1.0f64;
    loop {
        let res = (resolution as f64 * scale) as usize;
        if res < 7 {
            return Err(format!("the mesh does not fit {budget} voxels on any grid down to 7 (from {resolution})"));
        }
        let v = voxelize(pos, tris, res);
        if v.len() <= budget {
            return Ok((v, res));
        }
        scale *= 0.95;
    }
}

/// The mesh's vertices in voxel units for a grid of `res` (MeshVoxelizer.vertex_to_voxel)
pub fn normalized(pos: &[[f32; 3]], res: usize) -> Vec<[f32; 3]> {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in pos {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let eps = 1e-6f32;
    let (lo, hi) = (lo.map(|v| v - eps), hi.map(|v| v + eps));
    let center = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let range = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
    let s = (res as f32 - 3.0 * 2.0 - 2.0 * eps) / range;
    let half = res as f32 / 2.0;
    pos.iter().map(|p| [(p[0] - center[0]) * s + half, (p[1] - center[1]) * s + half, (p[2] - center[2]) * s + half]).collect()
}

/// Whether a triangle (its corners less the voxel's centre) overlaps the unit voxel (Triton's _triangle_overlaps_voxel)
fn overlaps(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> bool {
    for a in 0..3 {
        if !(v0[a].min(v1[a]).min(v2[a]) <= 0.5 && v0[a].max(v1[a]).max(v2[a]) >= -0.5) {
            return false;
        }
    }
    let e = [sub(v1, v0), sub(v2, v1), sub(v0, v2)];
    let axis = |ax: f32, ay: f32, az: f32| -> bool {
        let p0 = v0[0] * ax + v0[1] * ay + v0[2] * az;
        let p1 = v1[0] * ax + v1[1] * ay + v1[2] * az;
        let p2 = v2[0] * ax + v2[1] * ay + v2[2] * az;
        let r = 0.5 * (ax.abs() + ay.abs() + az.abs());
        p0.min(p1).min(p2) <= r && p0.max(p1).max(p2) >= -r
    };
    for ed in &e {
        if !(axis(0.0, ed[2], -ed[1]) && axis(-ed[2], 0.0, ed[0]) && axis(ed[1], -ed[0], 0.0)) {
            return false;
        }
    }
    let (e0, e1) = (e[0], e[1]);
    let n = [e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]];
    let off = -(n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2]);
    let rad = 0.5 * (n[0].abs() + n[1].abs() + n[2].abs());
    off.abs() <= rad
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// The surface voxels of the mesh on a grid of `res`, sorted by (z << 22 | y << 11 | x)
pub fn voxelize(pos: &[[f32; 3]], tris: &[[u32; 3]], res: usize) -> Vec<[i32; 3]> {
    let nv = normalized(pos, res);
    let threads = thread::available_parallelism().map_or(8, |n| n.get()).min(32);
    let chunk = tris.len().div_ceil(threads).max(1);
    let mut keys: Vec<u64> = thread::scope(|s| {
        let hs: Vec<_> = tris.chunks(chunk).map(|part| {
            let nv = &nv;
            s.spawn(move || {
                let mut out = Vec::new();
                for t in part {
                    let (a, b, c) = (nv[t[0] as usize], nv[t[1] as usize], nv[t[2] as usize]);
                    let lo = |i: usize| (a[i].min(b[i]).min(c[i]).floor() as i32).max(0);
                    let hi = |i: usize| (a[i].max(b[i]).max(c[i]).floor() as i32).min(res as i32 - 1);
                    for z in lo(2)..=hi(2) {
                        for y in lo(1)..=hi(1) {
                            for x in lo(0)..=hi(0) {
                                let ctr = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                                if overlaps(sub(a, ctr), sub(b, ctr), sub(c, ctr)) {
                                    out.push(((z as u64) << 22) | ((y as u64) << 11) | x as u64);
                                }
                            }
                        }
                    }
                }
                out
            })
        }).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    });
    keys.sort_unstable();
    keys.dedup();
    let m = (1u64 << 11) - 1;
    keys.iter().map(|k| [(k & m) as i32, ((k >> 11) & m) as i32, ((k >> 22) & m) as i32]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_fills_one_layer() {
        // a unit square in the z = 0 plane: on a grid of 16 it spans 10 voxels a side, one layer thick (or two where it
        // lies on a boundary)
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let v = voxelize(&pos, &tris, 16);
        assert!(v.len() >= 100, "{}", v.len());
        assert!(v.iter().all(|c| c[2] == 7 || c[2] == 8));
        // sorted by z, then y, then x
        let k: Vec<u64> = v.iter().map(|c| ((c[2] as u64) << 22) | ((c[1] as u64) << 11) | c[0] as u64).collect();
        assert!(k.windows(2).all(|w| w[0] < w[1]));
        let (_, res) = adaptive(&pos, &tris, 16, 60).unwrap();
        assert!(res < 16);
    }
}
