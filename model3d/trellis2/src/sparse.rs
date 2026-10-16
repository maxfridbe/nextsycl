//! The sparse voxel structures, on the host: a coordinate map (open addressing), the 27-neighbour maps the
//! submanifold convolutions gather through, and the channel-to-space subdivisions - each level's children in the
//! reference's order (parent by parent, octant by octant: x + s % 2, y + s / 2 % 2, z + s / 4) with their neighbour maps
//! derived from the parents' (a child's neighbour is a child of its parent's neighbour), so a level of millions of
//! voxels needs no hashing. The texture decoder replays the shape decoder's subdivisions (`Tree`).

/// Neighbour tap t = kd * 9 + kh * 3 + kw is the voxel at (x + kd - 1, y + kh - 1, z + kw - 1) (flex_gemm's weight
/// [Co, Kd, Kh, Kw, Ci] order)
pub fn tap(t: usize) -> [i32; 3] {
    [(t / 9) as i32 - 1, (t / 3 % 3) as i32 - 1, (t % 3) as i32 - 1]
}

fn key(c: [i32; 3]) -> u64 {
    // 21 bits an axis, offset so -1 fits; +1 so no key is 0 (the empty slot)
    let f = |v: i32| (v + 2) as u64 & 0x1f_ffff;
    ((f(c[0]) << 42) | (f(c[1]) << 21) | f(c[2])) + 1
}

/// Voxel coordinates to rows (open addressing, linear probing)
pub struct CoordMap {
    keys: Vec<u64>,
    vals: Vec<u32>,
    mask: usize,
}

impl CoordMap {
    pub fn new(coords: &[[i32; 3]]) -> CoordMap {
        let cap = (coords.len() * 2).next_power_of_two().max(16);
        let mut m = CoordMap { keys: vec![0; cap], vals: vec![0; cap], mask: cap - 1 };
        for (i, c) in coords.iter().enumerate() {
            let k = key(*c);
            let mut h = Self::hash(k) & m.mask;
            while m.keys[h] != 0 && m.keys[h] != k {
                h = (h + 1) & m.mask;
            }
            m.keys[h] = k;
            m.vals[h] = i as u32;
        }
        m
    }

    fn hash(k: u64) -> usize {
        (k.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 20) as usize
    }

    pub fn get(&self, c: [i32; 3]) -> Option<u32> {
        if c.iter().any(|v| *v < -1 || *v >= (1 << 21) - 2) {
            return None;
        }
        let k = key(c);
        let mut h = Self::hash(k) & self.mask;
        loop {
            let s = self.keys[h];
            if s == k {
                return Some(self.vals[h]);
            }
            if s == 0 {
                return None;
            }
            h = (h + 1) & self.mask;
        }
    }
}

/// Runs `f(row0, chunk)` over `out` in chunks of `per` rows on the machine's threads
pub fn par_rows<T: Send>(out: &mut [T], per: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let n = out.len() / per.max(1);
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(32);
    let rows = n.div_ceil(threads).max(1024);
    if n <= rows {
        f(0, out);
        return;
    }
    std::thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(rows * per).enumerate() {
            let f = &f;
            s.spawn(move || f(i * rows, chunk));
        }
    });
}

/// The [N, 27] neighbour map of a voxel set (-1: no voxel there)
pub fn neighbours(coords: &[[i32; 3]]) -> Vec<i32> {
    let map = CoordMap::new(coords);
    let mut out = vec![-1i32; coords.len() * 27];
    par_rows(&mut out, 27, |r0, chunk| {
        for (i, row) in chunk.chunks_exact_mut(27).enumerate() {
            let c = coords[r0 + i];
            for (t, o) in row.iter_mut().enumerate() {
                let d = tap(t);
                *o = map.get([c[0] + d[0], c[1] + d[1], c[2] + d[2]]).map_or(-1, |v| v as i32);
            }
        }
    });
    out
}

/// One channel-to-space subdivision: for each child its parent row and octant, the children's coordinates, each
/// parent's child rows by octant (-1: not subdivided there), and the children's neighbour map once a decoder made it
/// (the texture decoder, replaying the shape decoder's subdivisions, takes it from there)
#[derive(Clone, Debug, Default)]
pub struct Subdiv {
    pub parent: Vec<i32>,
    pub sub: Vec<i32>,
    pub coords: Vec<[i32; 3]>,
    pub children: Vec<[i32; 8]>,
    pub nmap: Option<std::sync::Arc<Vec<i32>>>,
}

/// The octant's offset: x + s % 2, y + s / 2 % 2, z + s / 4
pub fn octant(s: usize) -> [i32; 3] {
    [(s & 1) as i32, ((s >> 1) & 1) as i32, ((s >> 2) & 1) as i32]
}

/// The children of `coords` where `keep(parent, octant)` (the subdivision logits > 0), in the reference's order
pub fn subdivide(coords: &[[i32; 3]], keep: impl Fn(usize, usize) -> bool) -> Subdiv {
    let mut s = Subdiv { children: vec![[-1; 8]; coords.len()], ..Default::default() };
    for (p, c) in coords.iter().enumerate() {
        for o in 0..8 {
            if keep(p, o) {
                let d = octant(o);
                s.children[p][o] = s.parent.len() as i32;
                s.parent.push(p as i32);
                s.sub.push(o as i32);
                s.coords.push([2 * c[0] + d[0], 2 * c[1] + d[1], 2 * c[2] + d[2]]);
            }
        }
    }
    s
}

/// The children's neighbour map from their parents' (`parent_nmap` [Np, 27])
pub fn child_neighbours(parent_nmap: &[i32], s: &Subdiv) -> Vec<i32> {
    let n = s.parent.len();
    let mut out = vec![-1i32; n * 27];
    par_rows(&mut out, 27, |r0, chunk| {
        for (i, row) in chunk.chunks_exact_mut(27).enumerate() {
            let j = r0 + i;
            let (p, c) = (s.parent[j] as usize, s.coords[j]);
            let pp = [c[0] >> 1, c[1] >> 1, c[2] >> 1];
            for (t, o) in row.iter_mut().enumerate() {
                let d = tap(t);
                let q = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
                if q.iter().any(|v| *v < 0) {
                    continue;
                }
                let pc = [q[0] >> 1, q[1] >> 1, q[2] >> 1];
                let dp = [pc[0] - pp[0], pc[1] - pp[1], pc[2] - pp[2]];
                let tp = ((dp[0] + 1) * 9 + (dp[1] + 1) * 3 + (dp[2] + 1)) as usize;
                let np = parent_nmap[p * 27 + tp];
                if np < 0 {
                    continue;
                }
                let oct = ((q[0] & 1) | ((q[1] & 1) << 1) | ((q[2] & 1) << 2)) as usize;
                *o = s.children[np as usize][oct];
            }
        }
    });
    out
}

/// The shape decoder's subdivisions, level by level: what the texture decoder replays (its voxel tree is the shape's)
#[derive(Clone, Debug, Default)]
pub struct Tree {
    pub levels: Vec<Subdiv>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(coords: &[[i32; 3]]) -> Vec<i32> {
        let mut out = vec![-1; coords.len() * 27];
        for (i, c) in coords.iter().enumerate() {
            for t in 0..27 {
                let d = tap(t);
                let q = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
                if let Some(j) = coords.iter().position(|x| *x == q) {
                    out[i * 27 + t] = j as i32;
                }
            }
        }
        out
    }

    fn shell(r: i32) -> Vec<[i32; 3]> {
        let mut v = Vec::new();
        for x in 0..r {
            for y in 0..r {
                for z in 0..r {
                    let d = (((x - r / 2).pow(2) + (y - r / 2).pow(2) + (z - r / 2).pow(2)) as f32).sqrt();
                    if (d - r as f32 / 3.0).abs() < 1.0 {
                        v.push([x, y, z]);
                    }
                }
            }
        }
        v
    }

    #[test]
    fn coordinate_map_finds_every_voxel_and_nothing_else() {
        let c = shell(12);
        let m = CoordMap::new(&c);
        for (i, x) in c.iter().enumerate() {
            assert_eq!(m.get(*x), Some(i as u32));
        }
        assert_eq!(m.get([100, 100, 100]), None);
        assert_eq!(m.get([-1, 0, 0]), None);
    }

    #[test]
    fn neighbour_maps_match_a_search() {
        let c = shell(10);
        assert_eq!(neighbours(&c), brute(&c));
        // the centre tap is the voxel itself
        let n = neighbours(&c);
        assert!((0..c.len()).all(|i| n[i * 27 + 13] == i as i32));
    }

    #[test]
    fn children_neighbours_match_a_search() {
        let c = shell(8);
        let pn = neighbours(&c);
        // keep a pseudo-random half of the octants
        let s = subdivide(&c, |p, o| (p * 7 + o * 3) % 5 < 3);
        assert_eq!(s.parent.len(), s.coords.len());
        assert_eq!(child_neighbours(&pn, &s), brute(&s.coords));
        // children come parent by parent, octants ascending
        assert!(s.parent.windows(2).all(|w| w[0] <= w[1]));
        let j = 1 + s.parent.iter().position(|p| *p > 0).unwrap_or(0);
        assert_eq!(s.coords[s.children[s.parent[j] as usize][s.sub[j] as usize] as usize], s.coords[j]);
    }
}
