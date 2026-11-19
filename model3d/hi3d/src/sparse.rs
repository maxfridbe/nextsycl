//! Voxel bookkeeping on the host: neighbour maps (spconv's submanifold pairs as a [n, 27] table), the transformer's
//! paired downsample (sparc3d.nn.sparse.spatial: parents by unique codes, the mean of their children, the inverse
//! for the paired upsample), PTv3's z-order serialization and its patch padding.

use std::collections::HashMap;

/// A voxel's key (coordinates up to 2^20 a side)
pub fn key(c: [i32; 3]) -> u64 {
    ((c[0] as u64 & 0xf_ffff) << 40) | ((c[1] as u64 & 0xf_ffff) << 20) | (c[2] as u64 & 0xf_ffff)
}

/// [n, 27] rows: tap (a, b, c) (t = 9 a + 3 b + c) is the voxel at (x + a - 1, y + b - 1, z + c - 1), -1 where none
/// (the weight's [Co, 3, 3, 3, Ci] order: the gather's)
pub fn nmap(coords: &[[i32; 3]]) -> Vec<i32> {
    let at: HashMap<u64, i32> = coords.iter().enumerate().map(|(i, c)| (key(*c), i as i32)).collect();
    let mut out = vec![-1i32; coords.len() * 27];
    for (i, c) in coords.iter().enumerate() {
        for t in 0..27 {
            let d = [(t / 9) as i32 - 1, (t / 3 % 3) as i32 - 1, (t % 3) as i32 - 1];
            let n = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
            if n.iter().all(|v| *v >= 0) {
                if let Some(j) = at.get(&key(n)) {
                    out[i * 27 + t] = *j;
                }
            }
        }
    }
    out
}

/// The paired downsample by 2: the parents (unique halved coordinates, in (x, y, z) order as the reference's sorted
/// codes), each child's parent (the upsample's gather), and the children of each parent as CSR (ptr, idx) for the mean
pub struct Down {
    pub parents: Vec<[i32; 3]>,
    pub inverse: Vec<i32>,
    pub ptr: Vec<i32>,
    pub idx: Vec<i32>,
}

pub fn downsample(coords: &[[i32; 3]]) -> Down {
    let half: Vec<[i32; 3]> = coords.iter().map(|c| [c[0] / 2, c[1] / 2, c[2] / 2]).collect();
    let mut parents = half.clone();
    parents.sort_unstable();
    parents.dedup();
    let at: HashMap<[i32; 3], i32> = parents.iter().enumerate().map(|(i, c)| (*c, i as i32)).collect();
    let inverse: Vec<i32> = half.iter().map(|c| at[c]).collect();
    let mut count = vec![0i32; parents.len() + 1];
    for p in &inverse {
        count[*p as usize + 1] += 1;
    }
    for i in 0..parents.len() {
        count[i + 1] += count[i];
    }
    let ptr = count.clone();
    let mut fill = count;
    let mut idx = vec![0i32; coords.len()];
    for (j, p) in inverse.iter().enumerate() {
        idx[fill[*p as usize] as usize] = j as i32;
        fill[*p as usize] += 1;
    }
    Down { parents, inverse, ptr, idx }
}

/// PTv3's z-order code of a voxel (Point Cloud's xyz2key: bit i of x at 3 i + 2, of y at 3 i + 1, of z at 3 i)
pub fn zorder(c: [i32; 3], depth: u32) -> u64 {
    let mut k = 0u64;
    for i in 0..depth {
        k |= ((c[0] as u64 >> i) & 1) << (3 * i + 2) | ((c[1] as u64 >> i) & 1) << (3 * i + 1) | ((c[2] as u64 >> i) & 1) << (3 * i);
    }
    k
}

/// Point.serialization for one order ("z", or "z-trans": x and y swapped first): the voxels sorted by their codes
/// (the codes are unique) and each voxel's place in that order
pub fn serialize(coords: &[[i32; 3]], trans: bool) -> (Vec<i32>, Vec<i32>) {
    let mx = coords.iter().flat_map(|c| c.iter()).copied().max().unwrap_or(0).max(0) as u32;
    let depth = 32 - mx.leading_zeros();
    let codes: Vec<u64> = coords.iter().map(|c| zorder(if trans { [c[1], c[0], c[2]] } else { *c }, depth)).collect();
    let mut order: Vec<i32> = (0..coords.len() as i32).collect();
    order.sort_unstable_by_key(|i| codes[*i as usize]);
    let mut inverse = vec![0i32; coords.len()];
    for (p, i) in order.iter().enumerate() {
        inverse[*i as usize] = p as i32;
    }
    (order, inverse)
}

/// SerializedAttention's padding for n voxels in patches of `patch`: the rows (into the serialized order) of each
/// padded place - an n that is not a whole number of patches fills its last patch with the rows one patch earlier (so
/// the last patch is the last `patch` voxels); n up to one patch is one patch of n. (windows, patch size, rows)
pub fn padding(n: usize, patch: usize) -> (usize, usize, Vec<usize>) {
    if n <= patch {
        return (1, n, (0..n).collect());
    }
    let padded = n.div_ceil(patch) * patch;
    let mut pad: Vec<usize> = (0..padded).collect();
    if !n.is_multiple_of(patch) {
        for (i, p) in pad.iter_mut().enumerate().skip(n) {
            *p = i - patch;
        }
    }
    (padded / patch, patch, pad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_and_downsample() {
        let c = vec![[0, 0, 0], [1, 0, 0], [0, 0, 1], [3, 3, 3]];
        let m = nmap(&c);
        assert_eq!(m[13], 0); // the centre tap is the voxel itself
        assert_eq!(m[9 * 2 + 3 + 1], 1); // (x + 1, y, z)
        assert_eq!(m[9 + 3 + 2], 2); // (x, y, z + 1)
        assert_eq!(m[3 * 27 + 13], 3);
        assert!(m[3 * 27..4 * 27].iter().enumerate().all(|(t, v)| t == 13 || *v == -1));
        let d = downsample(&c);
        assert_eq!(d.parents, vec![[0, 0, 0], [1, 1, 1]]);
        assert_eq!(d.inverse, vec![0, 0, 0, 1]);
        assert_eq!(d.ptr, vec![0, 3, 4]);
        assert_eq!(d.idx, vec![0, 1, 2, 3]);
    }

    #[test]
    fn zorder_interleaves_and_pads() {
        assert_eq!(zorder([1, 0, 0], 1), 4);
        assert_eq!(zorder([0, 1, 0], 1), 2);
        assert_eq!(zorder([0, 0, 1], 1), 1);
        assert_eq!(zorder([2, 0, 0], 2), 32);
        let (w, s, p) = padding(100, 48);
        assert_eq!((w, s, p.len()), (3, 48, 144));
        // the last patch holds the last 48 voxels: 52..100 (52..96 copied in the padded places 100..144)
        let mut last: Vec<usize> = p[96..].to_vec();
        last.sort();
        assert_eq!(last, (52..100).collect::<Vec<_>>());
        assert_eq!(padding(30, 48), (1, 30, (0..30).collect()));
    }
}
