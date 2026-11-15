//! Budget3D's token layouts (budget3d/octree.py), on the host.
//!
//! Idol's 8192 latent tokens are a stratified 3-D Sobol set with exactly two points in each of the 16^3 cells (one in
//! each x-half), so the set is a relabelling of a three-level octree: a point (s = 1, 8192 of them), a cell of two
//! points (s = 2, 4096), a cube of 8 cells (s = 4, 512). A layout opens k4 of the 512 cubes into their cells and k2 of
//! those cells into their points: N = 512 + 7 k4 + k2 tokens, laid out [s = 4 | s = 2 | s = 1], each part in
//! ascending index order. The cubes and cells that open are the ones whose latent varies most (`detail_scores`); how
//! the budget is spent between the two levels comes from bisecting a shared offset of two merge thresholds.

use nextsycl_model3d::{Error, Result};

pub const N_SOBOL: usize = 8192;
pub const N_CELL: usize = 4096;
pub const N_CUBE: usize = 512;
/// base merge thresholds (tau_4, tau_2)
const THRESHOLDS: (f64, f64) = (2.1, 1.5);
const SCORE_SCALE: f64 = 0.08;

/// The Sobol set's octree order: `perm[j]` is the Idol token at octree position j (cell j / 2, slot j % 2); `inv` the
/// other way
pub struct Octree {
    pub perm: Vec<usize>,
    pub inv: Vec<usize>,
    /// the 8 cells of each cube, children ordered (a, b, c) over the x, y, z halves
    pub cube_cells: Vec<[usize; 8]>,
}

impl Octree {
    /// From Idol's positions (pos_pe [8192, 3] in [0, 1]); an error unless they are exactly two a 16^3 cell
    pub fn new(pos: &[f32]) -> Result<Octree> {
        if pos.len() != N_SOBOL * 3 {
            return Err(Error(format!("pos_pe: {} values, not 8192 x 3", pos.len())));
        }
        let key = |p: &[f32]| -> usize {
            let i = |v: f32| ((v * 16.0) as i64).clamp(0, 15) as usize;
            let hx = ((p[0] * 32.0) as i64).rem_euclid(2) as usize;
            ((i(p[0]) * 16 + i(p[1])) * 16 + i(p[2])) * 2 + hx
        };
        let keys: Vec<usize> = pos.chunks_exact(3).map(key).collect();
        let mut seen = vec![false; N_SOBOL];
        for &k in &keys {
            if seen[k] {
                return Err(Error("pos_pe is not an exact 16^3 x 2 stratified Sobol set: the Budget3D scale hierarchy does not apply".into()));
            }
            seen[k] = true;
        }
        // the keys are a permutation of 0..8192: the token with key j sits at octree position j
        let mut perm = vec![0usize; N_SOBOL];
        for (t, &k) in keys.iter().enumerate() {
            perm[k] = t;
        }
        let mut inv = vec![0usize; N_SOBOL];
        for (j, &t) in perm.iter().enumerate() {
            inv[t] = j;
        }
        let cube_cells = (0..N_CUBE).map(|j| {
            let (jx, jy, jz) = (j / 64, (j / 8) % 8, j % 8);
            let mut c = [0usize; 8];
            for (s, cell) in c.iter_mut().enumerate() {
                let (a, b, cc) = (s / 4, (s / 2) % 2, s % 2);
                *cell = ((2 * jx + a) * 16 + (2 * jy + b)) * 16 + (2 * jz + cc);
            }
            c
        }).collect();
        Ok(Octree { perm, inv, cube_cells })
    }

    /// Rows of `x` ([8192, c], Idol's order) in octree order
    pub fn to_octree(&self, x: &[f32], c: usize) -> Vec<f32> {
        let mut o = vec![0f32; N_SOBOL * c];
        for (j, &t) in self.perm.iter().enumerate() {
            o[j * c..(j + 1) * c].copy_from_slice(&x[t * c..(t + 1) * c]);
        }
        o
    }

    /// Rows of `x` (octree order) back in Idol's order
    pub fn from_octree(&self, x: &[f32], c: usize) -> Vec<f32> {
        let mut o = vec![0f32; N_SOBOL * c];
        for (j, &t) in self.perm.iter().enumerate() {
            o[t * c..(t + 1) * c].copy_from_slice(&x[j * c..(j + 1) * c]);
        }
        o
    }

    /// (score_4 [512], score_2 [4096]) of a latent `z` (Idol's order, [8192, c]): log2(1 + sqrt(mean_c var(z)) /
    /// 0.08) over the points a merged token would cover
    pub fn detail_scores(&self, z: &[f32], c: usize) -> (Vec<f64>, Vec<f64>) {
        let zo = self.to_octree(z, c);
        let bits = |v: f64| (1.0 + (v + 1e-8).sqrt() / SCORE_SCALE).log2();
        let spread = |pts: &[usize]| -> f64 {
            let n = pts.len() as f64;
            let mut tot = 0f64;
            for ch in 0..c {
                let m = pts.iter().map(|&p| zo[p * c + ch] as f64).sum::<f64>() / n;
                tot += pts.iter().map(|&p| (zo[p * c + ch] as f64 - m).powi(2)).sum::<f64>() / n;
            }
            tot / c as f64
        };
        let s2 = (0..N_CELL).map(|cell| bits(spread(&[2 * cell, 2 * cell + 1]))).collect();
        let s4 = self.cube_cells.iter().map(|cells| {
            let pts: Vec<usize> = cells.iter().flat_map(|&cl| [2 * cl, 2 * cl + 1]).collect();
            bits(spread(&pts))
        }).collect();
        (s4, s2)
    }

    /// The cells of an open cube may open: [4096] by cell
    fn eligible(&self, split4: &[bool]) -> Vec<bool> {
        let mut e = vec![false; N_CELL];
        for (j, cells) in self.cube_cells.iter().enumerate() {
            if split4[j] {
                for &c in cells {
                    e[c] = true;
                }
            }
        }
        e
    }

    /// The exact-budget layout from the scores: the k4 highest-scoring cubes open, then among their cells the k2
    /// highest (ties by index)
    pub fn build(&self, s4: &[f64], s2: &[f64], n_tokens: usize) -> Result<Layout> {
        let (k4, k2) = self.balance(s4, s2, n_tokens)?;
        let mut split4 = vec![false; N_CUBE];
        for j in top(s4, k4) {
            split4[j] = true;
        }
        let elig = self.eligible(&split4);
        let masked: Vec<f64> = s2.iter().zip(&elig).map(|(s, e)| if *e { *s } else { f64::NEG_INFINITY }).collect();
        let mut split2 = vec![false; N_CELL];
        for j in top(&masked, k2) {
            split2[j] = true;
        }
        let idx4 = (0..N_CUBE).filter(|j| !split4[*j]).collect();
        let idx2 = (0..N_CELL).filter(|j| elig[*j] && !split2[*j]).collect();
        let idx1 = (0..N_CELL).filter(|j| split2[*j]).collect();
        Layout::new(idx4, idx2, idx1, k4, k2)
    }

    /// How a budget is spent between the levels (balance_splits): the shared threshold offset bisected until the
    /// threshold rule lands on the token count, its open-cube count kept, realised exactly by rank
    fn balance(&self, s4: &[f64], s2: &[f64], n_tokens: usize) -> Result<(usize, usize)> {
        let n = snap(n_tokens);
        let (lo4, hi4) = k4_bounds(n)?;
        if lo4 == hi4 {
            return split_counts(n, Some(lo4));
        }
        let (t4, t2) = THRESHOLDS;
        let counts = |delta: f64| -> (f64, f64) {
            let split4: Vec<bool> = s4.iter().map(|s| *s > t4 + delta).collect();
            let elig = self.eligible(&split4);
            let k4 = split4.iter().filter(|x| **x).count() as f64;
            let k2 = s2.iter().zip(&elig).filter(|(s, e)| **e && **s > t2 + delta).count() as f64;
            (k4, N_CUBE as f64 + 7.0 * k4 + k2)
        };
        let (mut lo, mut hi) = (-16.0f64, 16.0f64);
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if counts(mid).1 > n as f64 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let k4 = counts(hi).0.round() as usize;
        split_counts(n, Some(k4))
    }
}

/// Indices of the k largest values, ties by index
fn top(v: &[f64], k: usize) -> Vec<usize> {
    let mut i: Vec<usize> = (0..v.len()).collect();
    i.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    i.truncate(k);
    i
}

/// A token count rounded to the nearest one a layout realises (512 or 519..8192)
pub fn snap(n: usize) -> usize {
    let n = n.clamp(N_CUBE, N_SOBOL);
    if N_CUBE < n && n < N_CUBE + 7 {
        if n - N_CUBE < 4 { N_CUBE } else { N_CUBE + 7 }
    } else {
        n
    }
}

/// The tokens of a budget (a fraction of 8192)
pub fn tokens_of(budget: f64) -> usize {
    snap((budget * N_SOBOL as f64).round().max(0.0) as usize)
}

/// The cube splits that can realise n tokens: ceil((n - 512) / 15) ..= min(512, (n - 512) / 7)
fn k4_bounds(n: usize) -> Result<(usize, usize)> {
    if n == N_CUBE {
        return Ok((0, 0));
    }
    let lo = (n - N_CUBE).div_ceil(15);
    let hi = N_CUBE.min((n - N_CUBE) / 7);
    if hi < lo {
        return Err(Error(format!("{n} tokens is not realisable; 512 or 519..8192")));
    }
    Ok((lo, hi))
}

/// (k4, k2) for a token count, k4 clamped into its range (none: the most cubes open)
fn split_counts(n: usize, k4: Option<usize>) -> Result<(usize, usize)> {
    let (lo, hi) = k4_bounds(n)?;
    let k4 = k4.map_or(hi, |k| k.clamp(lo, hi));
    let k2 = n - N_CUBE - 7 * k4;
    if k2 > 8 * k4 && n != N_CUBE {
        return Err(Error(format!("{n} tokens: k4 {k4}, k2 {k2}")));
    }
    Ok((k4, k2))
}

/// One layout: the cubes kept whole (s = 4), the cells kept whole (s = 2), the cells opened into their two points (s
/// = 1); the token sequence is [idx4 | idx2 | the points of idx1, two a cell]
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub idx4: Vec<usize>,
    pub idx2: Vec<usize>,
    pub idx1: Vec<usize>,
    pub k4: usize,
    pub k2: usize,
}

impl Layout {
    pub fn new(idx4: Vec<usize>, idx2: Vec<usize>, idx1: Vec<usize>, k4: usize, k2: usize) -> Result<Layout> {
        let l = Layout { idx4, idx2, idx1, k4, k2 };
        // every cell covered exactly once: by an s = 1 pair, an s = 2 token, or one of an s = 4 cube's 8
        if l.k2 + l.n2() + 8 * l.n4() != N_CELL || l.idx4.len() != N_CUBE - k4 || l.idx2.len() != 8 * k4 - k2 || l.idx1.len() != k2 {
            return Err(Error(format!("a layout of k4 {k4}, k2 {k2} with {} / {} / {} indices", l.idx4.len(), l.idx2.len(), l.idx1.len())));
        }
        Ok(l)
    }
    /// The 8192-token layout: every point its own token
    pub fn dense() -> Layout {
        Layout { idx4: Vec::new(), idx2: Vec::new(), idx1: (0..N_CELL).collect(), k4: N_CUBE, k2: N_CELL }
    }
    pub fn is_dense(&self) -> bool {
        self.k4 == N_CUBE && self.k2 == N_CELL
    }
    pub fn n4(&self) -> usize {
        self.idx4.len()
    }
    pub fn n2(&self) -> usize {
        self.idx2.len()
    }
    pub fn n1(&self) -> usize {
        2 * self.idx1.len()
    }
    pub fn tokens(&self) -> usize {
        self.n4() + self.n2() + self.n1()
    }
}

/// [4096, 3]: each 16^3 cell's centre in [0, 1]^3, in cell order
pub fn cell_centres() -> Vec<[f32; 3]> {
    grid(16)
}

/// [512, 3]: each 8^3 cube's centre
pub fn cube_centres() -> Vec<[f32; 3]> {
    grid(8)
}

fn grid(n: usize) -> Vec<[f32; 3]> {
    let g = |i: usize| (i as f32 + 0.5) / n as f32;
    (0..n * n * n).map(|j| [g(j / (n * n)), g((j / n) % n), g(j % n)]).collect()
}

/// A cube's children (cells) relative to its centre, the cube being [-1, 1]^3: (a, b, c) - 0.5
pub fn cube_child_offsets() -> Vec<[f32; 3]> {
    (0..8).map(|s| [(s / 4) as f32 - 0.5, ((s / 2) % 2) as f32 - 0.5, (s % 2) as f32 - 0.5]).collect()
}

/// A cell's two points: only their x-halves are known
pub fn cell_child_offsets() -> Vec<[f32; 3]> {
    vec![[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in Sobol set: two points a cell, one in each x-half
    fn positions() -> Vec<f32> {
        let mut p = Vec::new();
        for cell in 0..N_CELL {
            let (ix, iy, iz) = (cell / 256, (cell / 16) % 16, cell % 16);
            for h in 0..2 {
                p.extend([(ix as f32 + 0.25 + 0.5 * h as f32) / 16.0, (iy as f32 + 0.3) / 16.0, (iz as f32 + 0.6) / 16.0]);
            }
        }
        // a scramble, so the octree order is not the stored one
        let n = N_SOBOL;
        let mut out = vec![0f32; n * 3];
        for t in 0..n {
            let s = (t * 2741) % n;
            out[s * 3..s * 3 + 3].copy_from_slice(&p[t * 3..t * 3 + 3]);
        }
        out
    }

    #[test]
    fn the_octree_order_is_a_permutation_and_its_inverse() {
        let o = Octree::new(&positions()).unwrap();
        for t in 0..N_SOBOL {
            assert_eq!(o.perm[o.inv[t]], t);
        }
        let x: Vec<f32> = (0..N_SOBOL * 2).map(|v| v as f32).collect();
        assert_eq!(o.from_octree(&o.to_octree(&x, 2), 2), x);
        // a cell's two points are adjacent in octree order, the lower x-half first
        let p = positions();
        let (a, b) = (o.perm[10], o.perm[11]);
        assert!(p[a * 3] < p[b * 3]);
        assert!(Octree::new(&vec![0.5; N_SOBOL * 3]).is_err());
    }

    #[test]
    fn budgets_land_on_their_token_counts() {
        let o = Octree::new(&positions()).unwrap();
        let z: Vec<f32> = (0..N_SOBOL * 16).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
        let (s4, s2) = o.detail_scores(&z, 16);
        for b in [1.0, 0.5, 0.25, 0.0625, 0.07] {
            let n = tokens_of(b);
            let l = o.build(&s4, &s2, n).unwrap();
            assert_eq!(l.tokens(), n, "budget {b}");
            if b == 1.0 {
                assert!(l.is_dense());
                assert_eq!(l, Layout::dense());
            }
        }
        assert_eq!(snap(514), 512);
        assert_eq!(snap(517), 519);
        assert_eq!(tokens_of(0.5), 4096);
    }
}
