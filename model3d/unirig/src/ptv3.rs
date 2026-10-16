//! Point Transformer V3 as UniRig's skin model builds it (`ptv3obj`: PTv3Object.py): no pooling - five stages of
//! 3, 3, 3, 6 and 16 blocks at full resolution, 32 to 384 channels, heads of 16 features. The points on a grid of
//! 0.005; each block a submanifold sparse convolution (3 x 3 x 3, its positional encoding), attention within patches
//! of 1,024 consecutive points along one of four space-filling curves (z-order, Hilbert, and both with x and y
//! swapped), its q and k normalized per head, then an MLP. The stem: a 5 x 5 x 5 sparse convolution, batch norm, GELU,
//! plus a linear path.
//!
//! The attention's inputs are rounded to half first, as UniRig's flash attention takes them (`qkv.half()`); the rest
//! is float32.
//!
//! The sparse convolutions run as im2col (each point's neighbours' features in a row, `-1`: zeros) and one product.
//! spconv's handling of several points in one voxel is mirrored as reference/unirig/ref.py defines it: a tap reads the
//! first point of the cloud in the neighbouring voxel, the centre tap the point itself.

use std::collections::HashMap;

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::ffi::GELU;
use crate::ops::{dev_i32, Lin, Ln, Ops, Weights};

pub const GRID: f32 = 0.005;
const DEPTHS: [usize; 5] = [3, 3, 3, 6, 16];
const CHANNELS: [usize; 5] = [32, 64, 128, 256, 384];
const HEADS: [usize; 5] = [2, 4, 8, 16, 24];
pub const PATCH: usize = 1024;
pub const OUT: usize = 384;

// ---- serialization -------------------------------------------------------------------------------------------

/// Grid coordinates: trunc((p - min) / grid), float32 as the reference
pub fn grid_coords(pts: &[[f32; 3]]) -> Vec<[i64; 3]> {
    let mut lo = [f32::INFINITY; 3];
    for p in pts {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
        }
    }
    pts.iter().map(|p| [0, 1, 2].map(|c| ((p[c] - lo[c]) / GRID).trunc() as i64)).collect()
}

/// The z-order key (x bit i at 3i + 2, y at 3i + 1, z at 3i)
pub fn z_order(x: i64, y: i64, z: i64, depth: u32) -> i64 {
    let mut k = 0i64;
    for i in 0..depth {
        let m = 1i64 << i;
        k |= ((x & m) << (2 * i + 2)) | ((y & m) << (2 * i + 1)) | ((z & m) << (2 * i));
    }
    k
}

/// Skilling's Hilbert index of a point (pointcept's `hilbert.encode`, 3 dimensions of `bits` bits)
#[allow(clippy::needless_range_loop)]
pub fn hilbert(p: [i64; 3], bits: u32) -> i64 {
    let nb = bits as usize;
    // gray[d][b]: bit (bits - 1 - b) of coordinate d, the most significant first
    let mut g = [[false; 64]; 3];
    for d in 0..3 {
        for b in 0..nb {
            g[d][b] = (p[d] >> (nb - 1 - b)) & 1 == 1;
        }
    }
    for bit in 0..nb {
        for dim in 0..3 {
            let mask = g[dim][bit];
            for j in bit + 1..nb {
                g[0][j] ^= mask;
            }
            for j in bit + 1..nb {
                let flip = !mask && (g[0][j] ^ g[dim][j]);
                g[dim][j] ^= flip;
                g[0][j] ^= flip;
            }
        }
    }
    // interleave bit by bit (dims inner), then gray to binary (a running xor), most significant first
    let mut acc = false;
    let mut out = 0i64;
    for bit in 0..nb {
        for row in g.iter() {
            acc ^= row[bit];
            out = (out << 1) | acc as i64;
        }
    }
    out
}

/// The four orders' codes (z, z-trans, hilbert, hilbert-trans) of grid coordinates
pub fn codes(gc: &[[i64; 3]]) -> [Vec<i64>; 4] {
    let mx = gc.iter().flat_map(|c| c.iter().copied()).max().unwrap_or(0);
    let depth = 64 - (mx as u64).leading_zeros();
    [
        gc.iter().map(|c| z_order(c[0], c[1], c[2], depth)).collect(),
        gc.iter().map(|c| z_order(c[1], c[0], c[2], depth)).collect(),
        gc.iter().map(|c| hilbert(*c, depth)).collect(),
        gc.iter().map(|c| hilbert([c[1], c[0], c[2]], depth)).collect(),
    ]
}

/// The points' orders along each curve (stable on equal codes), the curves permuted by `perm` (the block i uses
/// orders[i % 4])
pub fn orders(gc: &[[i64; 3]], perm: [usize; 4]) -> Vec<Vec<usize>> {
    let c = codes(gc);
    perm.iter().map(|&k| {
        let mut o: Vec<usize> = (0..gc.len()).collect();
        o.sort_by_key(|&i| (c[k][i], i));
        o
    }).collect()
}

/// The neighbour table of a k x k x k submanifold convolution: [N, k^3], offsets x outermost; a tap names the first
/// point in that voxel (-1: none), the centre tap the point itself
pub fn neighbours(gc: &[[i64; 3]], k: i64) -> Vec<i32> {
    let mut first: HashMap<[i64; 3], i32> = HashMap::with_capacity(gc.len());
    for (i, c) in gc.iter().enumerate() {
        first.entry(*c).or_insert(i as i32);
    }
    let r = k / 2;
    let mut out = Vec::with_capacity(gc.len() * (k * k * k) as usize);
    for (i, c) in gc.iter().enumerate() {
        for a in -r..=r {
            for b in -r..=r {
                for d in -r..=r {
                    out.push(if a == 0 && b == 0 && d == 0 { i as i32 } else { *first.get(&[c[0] + a, c[1] + b, c[2] + d]).unwrap_or(&-1) });
                }
            }
        }
    }
    out
}

// ---- the network ---------------------------------------------------------------------------------------------

struct Block {
    conv: Lin,
    cpe_lin: Lin,
    cpe_ln: Ln,
    norm1: Ln,
    qkv: Lin,
    proj: Lin,
    norm2: Ln,
    fc1: Lin,
    fc2: Lin,
}

pub struct Ptv3 {
    stem: Lin,
    bn_scale: DevBuf,
    bn_shift: DevBuf,
    res: Lin,
    links: Vec<Option<Lin>>,
    blocks: Vec<Vec<Block>>,
}

/// What the serialization made (the caller's, or injected from a reference)
pub struct Layout {
    pub gc: Vec<[i64; 3]>,
    pub orders: Vec<Vec<usize>>,
}

impl Ptv3 {
    pub fn load(w: &Weights) -> Result<Ptv3> {
        let (s, stem) = w.host("embedding.stem.conv.weight")?;
        if s != [32, 5, 5, 5, 9] {
            return Err(Error(format!("embedding.stem.conv.weight is {s:?}")));
        }
        let (_, g) = w.host("embedding.stem.norm.weight")?;
        let (_, b) = w.host("embedding.stem.norm.bias")?;
        let (_, rm) = w.host("embedding.stem.norm.running_mean")?;
        let (_, rv) = w.host("embedding.stem.norm.running_var")?;
        let scale: Vec<f32> = g.iter().zip(&rv).map(|(g, v)| g / (v + 1e-3).sqrt()).collect();
        let shift: Vec<f32> = b.iter().zip(&rm).zip(&scale).map(|((b, m), s)| b - m * s).collect();
        let mut links = Vec::new();
        let mut blocks = Vec::new();
        for st in 0..5 {
            let c = CHANNELS[st];
            links.push(if st > 0 { Some(w.lin(&format!("enc.enc{st}.0"), c, CHANNELS[st - 1], true)?) } else { None });
            let mut v = Vec::new();
            for i in 0..DEPTHS[st] {
                let p = format!("enc.enc{st}.block{i}");
                v.push(Block {
                    conv: Lin { w: w.dev(&format!("{p}.cpe.0.weight"), &[c, 3, 3, 3, c])?, b: Some(w.dev(&format!("{p}.cpe.0.bias"), &[c])?), n: c, k: 27 * c },
                    cpe_lin: w.lin(&format!("{p}.cpe.1"), c, c, true)?,
                    cpe_ln: w.ln(&format!("{p}.cpe.2"), c)?,
                    norm1: w.ln(&format!("{p}.norm1.0"), c)?,
                    qkv: w.lin(&format!("{p}.attn.qkv"), 3 * c, c, true)?,
                    proj: w.lin(&format!("{p}.attn.proj"), c, c, true)?,
                    norm2: w.ln(&format!("{p}.norm2.0"), c)?,
                    fc1: w.lin(&format!("{p}.mlp.0.fc1"), 4 * c, c, true)?,
                    fc2: w.lin(&format!("{p}.mlp.0.fc2"), c, 4 * c, true)?,
                });
            }
            blocks.push(v);
        }
        Ok(Ptv3 {
            stem: Lin { w: DevBuf::from_f32(w.gpu, &stem)?, b: None, n: 32, k: 125 * 9 },
            bn_scale: DevBuf::from_f32(w.gpu, &scale)?,
            bn_shift: DevBuf::from_f32(w.gpu, &shift)?,
            res: w.lin("embedding.res_linear", 32, 9, true)?,
            links,
            blocks,
        })
    }

    /// out [n, l.n] = the sparse convolution of x [n, c] (neighbour table `nbr` [n, k], int32 on the GPU)
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, o: &Ops, x: &DevBuf, n: usize, c: usize, nbr: &DevBuf, k: usize, l: &Lin, out: &DevBuf) -> Result<()> {
        let rows = (64usize << 20) / (k * c * 4).max(1);
        let rows = rows.clamp(256, n);
        let col = o.buf(rows * k * c)?;
        for r0 in (0..n).step_by(rows) {
            let r = rows.min(n - r0);
            o.k.im2col(x, c, nbr, k, r0, r, &col)?;
            o.linear(&col, r, l, &out.view(r0 * l.n * 4, r * l.n * 4)?)?;
        }
        Ok(())
    }

    /// The features [n, 384] of the points (`feat` [n, 9]: position, normal, zeros) in the layout's orders
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, o: &Ops, feat: &[f32], lay: &Layout) -> Result<DevBuf> {
        let n = lay.gc.len();
        if !n.is_multiple_of(PATCH) {
            return Err(Error(format!("PTv3 takes whole patches of {PATCH} points, not {n}")));
        }
        let n5 = dev_i32(&o.gpu, &neighbours(&lay.gc, 5))?;
        let n3 = dev_i32(&o.gpu, &neighbours(&lay.gc, 3))?;
        let ords: Vec<DevBuf> = lay.orders.iter().map(|v| dev_i32(&o.gpu, &v.iter().map(|&i| i as i32).collect::<Vec<_>>())).collect::<Result<_>>()?;
        let f = DevBuf::from_f32(&o.gpu, feat)?;
        let cmax = OUT;
        let x = o.buf(n * cmax)?;
        let t = o.buf(n * cmax)?;
        let u = o.buf(n * 4 * cmax)?;
        let qkv = o.buf(n * 3 * cmax)?;
        let q2 = o.buf(n * 3 * cmax)?;
        let a = o.buf(n * cmax)?;
        let scratch = o.buf((24 * PATCH * PATCH * 2).min(64 << 20))?;
        // the stem
        self.conv(o, &f, n, 9, &n5, 125, &self.stem, &x)?;
        o.k.affine(&x, n, 32, &self.bn_scale, &self.bn_shift)?;
        o.k.act(&x, n * 32, GELU)?;
        o.linear(&f, n, &self.res, &t)?;
        o.k.add(&x, &t, n * 32)?;
        let mut c = 32;
        for st in 0..5 {
            if let Some(l) = &self.links[st] {
                o.linear(&x, n, l, &t)?;
                c = l.n;
                x.copy_within(0, &t, 0, n * c * 4)?;
            }
            let h = HEADS[st];
            let d = c / h;
            for (i, b) in self.blocks[st].iter().enumerate() {
                // the positional encoding
                self.conv(o, &x, n, c, &n3, 27, &b.conv, &t)?;
                o.linear(&t, n, &b.cpe_lin, &a)?;
                o.ln(&a, n, c, Some(&b.cpe_ln), &a)?;
                o.k.add(&x, &a, n * c)?;
                // attention in patches along the block's curve
                o.ln(&x, n, c, Some(&b.norm1), &t)?;
                o.linear(&t, n, &b.qkv, &qkv)?;
                o.k.gather(&qkv, 3 * c, &ords[i % 4], n, 3 * c, &q2)?;
                o.k.head_norm(&q2, n, 3 * c, 0, h, d, 1e-5)?;
                o.k.head_norm(&q2, n, 3 * c, c, h, d, 1e-5)?;
                // UniRig hands flash attention qkv.half(): the inputs rounded to half, the product float32
                o.k.round_half(&q2, n * 3 * c)?;
                let kv = q2.view(c * 4, q2.len - c * 4)?;
                let vv = q2.view(2 * c * 4, q2.len - 2 * c * 4)?;
                o.k.attention((&q2, 3 * c, d), (&kv, 3 * c, d), (&vv, 3 * c, d), n / PATCH, PATCH, PATCH, h, d, d, 1.0 / (d as f32).sqrt(), false,
                              &scratch, (&t, c))?;
                o.k.scatter(&t, &ords[i % 4], n, c, &a)?;
                o.linear(&a, n, &b.proj, &t)?;
                o.k.add(&x, &t, n * c)?;
                // the MLP
                o.ln(&x, n, c, Some(&b.norm2), &t)?;
                o.linear(&t, n, &b.fc1, &u)?;
                o.k.act(&u, n * 4 * c, GELU)?;
                o.linear(&u, n, &b.fc2, &t)?;
                o.k.add(&x, &t, n * c)?;
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves_order_a_cube() {
        // the 8 corners of a 2-cube: z-order interleaves x, y, z (x most significant)
        assert_eq!(z_order(1, 0, 0, 1), 4);
        assert_eq!(z_order(0, 1, 0, 1), 2);
        assert_eq!(z_order(0, 0, 1, 1), 1);
        // a Hilbert curve visits the 8 corners once each, neighbours one apart
        let mut by: Vec<(i64, [i64; 3])> = (0..8).map(|i| [(i >> 2) & 1, (i >> 1) & 1, i & 1]).map(|p| (hilbert(p, 1), p)).collect();
        by.sort();
        assert_eq!(by.iter().map(|x| x.0).collect::<Vec<_>>(), (0..8).collect::<Vec<_>>());
        for w in by.windows(2) {
            let d: i64 = (0..3).map(|c| (w[0].1[c] - w[1].1[c]).abs()).sum();
            assert_eq!(d, 1);
        }
    }

    #[test]
    fn hilbert_on_a_4_grid_steps_by_one() {
        let mut by: Vec<(i64, [i64; 3])> = (0..64).map(|i| [(i >> 4) & 3, (i >> 2) & 3, i & 3]).map(|p| (hilbert(p, 2), p)).collect();
        by.sort();
        assert_eq!(by.iter().map(|x| x.0).collect::<Vec<_>>(), (0..64).collect::<Vec<_>>());
        for w in by.windows(2) {
            let d: i64 = (0..3).map(|c| (w[0].1[c] - w[1].1[c]).abs()).sum();
            assert_eq!(d, 1);
        }
    }

    #[test]
    fn neighbours_take_the_first_point_of_a_voxel() {
        let gc = vec![[0, 0, 0], [1, 0, 0], [1, 0, 0], [5, 5, 5]];
        let nb = neighbours(&gc, 3);
        // point 0's +x tap (offset (1, 0, 0): index (2 * 3 + 1) * 3 + 1 = 22) is point 1, the first in that voxel
        assert_eq!(nb[22], 1);
        // point 2's centre is itself, its -x tap (index 4) point 0
        assert_eq!(nb[2 * 27 + 13], 2);
        assert_eq!(nb[2 * 27 + 4], 0);
        assert!(nb[3 * 27..4 * 27].iter().enumerate().all(|(k, &j)| if k == 13 { j == 3 } else { j == -1 }));
    }
}
