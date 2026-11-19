//! The sparse flow transformer (sparc3d's Sparc3DTransformer: TRELLIS's SLat flow with rotary positions and QK RMS
//! norm): the 32-channel latent on the structure's voxels in, its velocity out.
//!
//! ```text
//!   x [n, 32] -> input_layer (256) -> ResBlock(256) -> ResBlock(256 -> 2112, downsample by 2: the parents' mean, the
//!      reference's zero row counted)
//!     -> 24 blocks of 2112 (22 heads of 96) on the p parents: adaLN (shared + the block's own), self-attention with
//!        the rotary embedding of the parents' coordinates then the per-head RMS norm on q and k, cross-attention to
//!        the picture's 2,917 tokens (2048 wide), a GELU-tanh MLP of 8448
//!     -> ResBlock([h | skip] 4224 -> 256, upsample: each child its parent's row) -> ResBlock([h | skip] 512 -> 256)
//!     -> layer norm -> out_layer (32)
//!   ResBlock: silu(norm1(x)) -> conv1 -> norm2 * (1 + scale) + shift (from the timestep) -> silu -> conv2 -> silu
//!             -> linear, + x (or its linear skip)
//! ```
//!
//! float32 residual streams, half activations into the products (the fp16 weights as stored), float32 products where
//! the reference keeps float32 (the input and output layers, the timestep MLP and modulations on the host).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::ffi::Epi;
use crate::sconv::SConv;
use crate::w::{dev_half, dev_i32, silu, File, HostLin, Lin};
use crate::{sparse, Ctx};

pub const C: usize = 2112;
pub const HEADS: usize = 22;
pub const HD: usize = 96;
/// The heads as the attention runs them: padded to 128 (the flash kernel's width; q scaled by sqrt(128 / 96))
pub const HP: usize = 128;
const CP: usize = HEADS * HP;
const MLP: usize = 8448;
const EPS: f32 = 1e-6;

struct ResBlock {
    norm1: (DevBuf, DevBuf),
    conv1: SConv,
    conv2: SConv,
    linear: Lin,
    emb: HostLin,
    skip: Option<Lin>,
    ch: usize,
    hid: usize,
    out: usize,
    down: bool,
    up: bool,
}

impl ResBlock {
    fn load(f: &File, gpu: &Arc<Gpu>, p: &str, down: bool, up: bool) -> Result<ResBlock> {
        let conv1 = SConv::load(f, gpu, &format!("{p}.conv1.conv"))?;
        let conv2 = SConv::load(f, gpu, &format!("{p}.conv2.conv"))?;
        let linear = Lin::load(f, gpu, &format!("{p}.linear"), false)?;
        let skip = if f.has(&format!("{p}.skip_connection.weight")) { Some(Lin::load(f, gpu, &format!("{p}.skip_connection"), false)?) } else { None };
        let (ch, hid, out) = (conv1.ci, conv1.co, linear.n);
        Ok(ResBlock {
            norm1: (f.dev(gpu, &format!("{p}.norm1.weight"))?, f.dev(gpu, &format!("{p}.norm1.bias"))?),
            emb: HostLin::load(f, &format!("{p}.emb_layers.1"))?,
            conv1,
            conv2,
            linear,
            skip,
            ch,
            hid,
            out,
            down,
            up,
        })
    }
}

struct Block {
    modulation: Vec<f32>,
    qkv: Lin,
    q_gamma: DevBuf,
    k_gamma: DevBuf,
    out: Lin,
    norm2: (DevBuf, DevBuf),
    cq: Lin,
    ckv: Lin,
    cout: Lin,
    mlp0: Lin,
    mlp2: Lin,
}

pub struct Dit {
    input: Lin,
    t0: HostLin,
    t2: HostLin,
    ada: HostLin,
    inb: Vec<ResBlock>,
    blocks: Vec<Block>,
    outb: Vec<ResBlock>,
    out: Lin,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub bytes: usize,
}

/// A pass's voxels: the structure's (n, their neighbour map), the parents' (p, theirs), the downsample's maps, the
/// rotary phases of the parents
pub struct Geo {
    pub n: usize,
    pub p: usize,
    nmap: DevBuf,
    pnmap: DevBuf,
    ptr: DevBuf,
    idx: DevBuf,
    inv: DevBuf,
    cs: DevBuf,
}

/// What a call's cross-attention reads
pub enum Cross<'a> {
    /// every block's keys and values (`cache_cross`) for l tokens
    Cached(&'a [DevBuf], usize),
    /// the zero condition: each block's cross-attention output is one row (`const_cross`, [blocks, 2112] float32)
    Const(&'a DevBuf),
}

/// The first call's insides, for the check (the reference's hook names)
#[derive(Default)]
pub struct Probe {
    pub taps: Vec<(&'static str, Vec<f32>)>,
}

/// Hi3D's rotary phases ([n, 48, 2]: cos, sin) for coordinates: 16 frequencies 10000^(-j/16) per axis x, y, z, the
/// long-grid rescale when a coordinate reaches 64 (RotaryPositionEmbedder._update_freqs, train length 64)
pub fn rope(coords: &[[i32; 3]]) -> Vec<f32> {
    let fd = HD / 3 / 2;
    let mut freqs: Vec<f32> = (0..fd).map(|j| 1.0 / 10000f32.powf(j as f32 / fd as f32)).collect();
    let mx = coords.iter().flat_map(|c| c.iter()).copied().max().unwrap_or(0);
    if mx >= 64 {
        let (train, infer) = (64f32, ((64 + mx) / 2) as f32);
        for f in &mut freqs {
            let wl = *f * train / (2.0 * std::f32::consts::PI);
            let mut r = (wl - 1.0) / (24.0 - 1.0);
            r += (1.0 - r) * train / infer;
            if wl > 24.0 {
                r = 1.0;
            }
            if wl < 1.0 {
                r = train / infer;
            }
            *f *= r;
        }
    }
    let mut cs = vec![0f32; coords.len() * HD];
    for (i, c) in coords.iter().enumerate() {
        for a in 0..3 {
            for (j, f) in freqs.iter().enumerate() {
                let ang = c[a] as f32 * f;
                cs[i * HD + (a * fd + j) * 2] = ang.cos();
                cs[i * HD + (a * fd + j) * 2 + 1] = ang.sin();
            }
        }
    }
    cs
}

/// A buffer's half pointer at an offset (in halves)
fn hp(b: &DevBuf, off: usize) -> *mut std::ffi::c_void {
    // SAFETY: offsets inside the half buffers sized by the caller.
    unsafe { b.ptr().cast::<u16>().add(off).cast() }
}

impl Dit {
    pub fn load(f: &File, gpu: &Arc<Gpu>, mean: Vec<f32>, std: Vec<f32>) -> Result<Dit> {
        let nb = (0..).take_while(|i| f.has(&format!("blocks.{i}.modulation"))).count();
        if nb == 0 {
            return Err(Error(format!("{}: no Sparc3D transformer blocks (blocks.N.modulation)", f.path())));
        }
        let mut blocks = Vec::new();
        let mut bytes = 0;
        for i in 0..nb {
            let b = format!("blocks.{i}");
            let blk = Block {
                modulation: f.vec(&format!("{b}.modulation"))?,
                qkv: Lin::load(f, gpu, &format!("{b}.self_attn.to_qkv"), false)?,
                q_gamma: f.dev(gpu, &format!("{b}.self_attn.q_rms_norm.gamma"))?,
                k_gamma: f.dev(gpu, &format!("{b}.self_attn.k_rms_norm.gamma"))?,
                out: Lin::load(f, gpu, &format!("{b}.self_attn.to_out"), false)?,
                norm2: (f.dev(gpu, &format!("{b}.norm2.weight"))?, f.dev(gpu, &format!("{b}.norm2.bias"))?),
                cq: Lin::load(f, gpu, &format!("{b}.cross_attn.to_q"), false)?,
                ckv: Lin::load(f, gpu, &format!("{b}.cross_attn.to_kv"), false)?,
                cout: Lin::load(f, gpu, &format!("{b}.cross_attn.to_out"), false)?,
                mlp0: Lin::load(f, gpu, &format!("{b}.mlp.mlp.0"), false)?,
                mlp2: Lin::load(f, gpu, &format!("{b}.mlp.mlp.2"), false)?,
            };
            if blk.modulation.len() != 6 * C || blk.qkv.n != 3 * C || blk.mlp0.n != MLP || blk.ckv.k != 2048 {
                return Err(Error(format!("{}: block {i} is not Sparc3D's 2112-wide block", f.path())));
            }
            bytes += [&blk.qkv, &blk.out, &blk.cq, &blk.ckv, &blk.cout, &blk.mlp0, &blk.mlp2].iter().map(|l| l.bytes()).sum::<usize>();
            blocks.push(blk);
        }
        let inb = vec![ResBlock::load(f, gpu, "input_blocks.0", false, false)?, ResBlock::load(f, gpu, "input_blocks.1", true, false)?];
        let outb = vec![ResBlock::load(f, gpu, "out_blocks.0", false, true)?, ResBlock::load(f, gpu, "out_blocks.1", false, false)?];
        if inb[1].out != C || outb[0].ch != 2 * C || outb[1].out != 256 {
            return Err(Error(format!("{}: not the released I/O stages (256 -> 2112 -> 256)", f.path())));
        }
        let input = Lin::load(f, gpu, "input_layer", true)?;
        let out = Lin::load(f, gpu, "out_layer", true)?;
        if mean.len() != out.n || std.len() != out.n {
            return Err(Error(format!("latent statistics of {} channels for a {}-channel latent", mean.len(), out.n)));
        }
        Ok(Dit {
            t0: HostLin::load(f, "t_embedder.mlp.0")?,
            t2: HostLin::load(f, "t_embedder.mlp.2")?,
            ada: HostLin::load(f, "adaLN_modulation.1")?,
            input,
            inb,
            blocks,
            outb,
            out,
            mean,
            std,
            bytes,
        })
    }

    pub fn channels(&self) -> usize {
        self.input.k
    }

    /// The timestep's embedding (TimestepEmbedder: cos | sin of 128 frequencies, the MLP), for model time t (1000 sigma)
    fn t_emb(&self, t: f32) -> Vec<f32> {
        let half = 128;
        let ln = (10000f32).ln();
        let freqs: Vec<f32> = (0..half).map(|i| (-ln * i as f32 / half as f32).exp()).collect();
        let args: Vec<f32> = freqs.iter().map(|f| t * f).collect();
        let emb: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        self.t2.apply(&silu(&self.t0.apply(&emb)))
    }

    /// A pass's voxel geometry on the GPU
    pub fn geo(&self, gpu: &Arc<Gpu>, coords: &[[i32; 3]]) -> Result<Geo> {
        let d = sparse::downsample(coords);
        Ok(Geo {
            n: coords.len(),
            p: d.parents.len(),
            nmap: dev_i32(gpu, &sparse::nmap(coords))?,
            pnmap: dev_i32(gpu, &sparse::nmap(&d.parents))?,
            ptr: dev_i32(gpu, &d.ptr)?,
            idx: dev_i32(gpu, &d.idx)?,
            inv: dev_i32(gpu, &d.inverse)?,
            cs: DevBuf::from_f32(gpu, &rope(&d.parents))?,
        })
    }

    /// Every block's cross-attention keys and values for the picture's tokens `cond` (half [l, 2048])
    pub fn cache_cross(&self, cx: &Ctx, cond: &DevBuf, l: usize) -> Result<Vec<DevBuf>> {
        let kv = DevBuf::new(&cx.gpu, l * 2 * C * 2)?;
        self.blocks.iter().map(|b| {
            b.ckv.apply(cx, cond.ptr(), l, kv.ptr(), Dt::F16)?;
            // keys and values in heads of 128 (zero-padded) for the attention
            let kvp = DevBuf::new(&cx.gpu, l * 2 * CP * 2)?;
            cx.k.copy_heads(kv.ptr(), 2 * C, HD, l, 2 * HEADS, kvp.ptr(), 2 * CP, HP, 1.0)?;
            Ok(kvp)
        }).collect()
    }

    /// The zero condition's cross-attention output of every block: to_out(the value bias) [blocks, 2112] float32 (zero
    /// tokens give every key the key bias: the softmax is uniform and returns the value bias)
    pub fn const_cross(&self, cx: &Ctx) -> Result<DevBuf> {
        let out = DevBuf::f32(&cx.gpu, self.blocks.len() * C)?;
        for (i, b) in self.blocks.iter().enumerate() {
            let bias = b.ckv.bias.as_ref().map(|x| x.to_f32()).transpose()?.unwrap_or_else(|| vec![0.0; 2 * C]);
            let v = dev_half(&cx.gpu, &bias[C..2 * C])?;
            let row = DevBuf::f32(&cx.gpu, C)?;
            b.cout.apply(cx, v.ptr(), 1, row.ptr(), Dt::F32)?;
            out.copy_within(i * C * 4, &row, 0, C * 4)?;
        }
        Ok(out)
    }

    /// A residual block on x (float32 [rows, ch] at its input level): out float32 [rows', out]
    fn res(&self, cx: &Ctx, rb: &ResBlock, g: &Geo, x: &DevBuf, rows: usize, emb: &[f32]) -> Result<(DevBuf, usize)> {
        let (k, gpu) = (&cx.k, &cx.gpu);
        let e = rb.emb.apply(&silu(emb));
        let scale = DevBuf::from_f32(gpu, &e[..rb.hid])?;
        let shift = DevBuf::from_f32(gpu, &e[rb.hid..2 * rb.hid])?;
        // the paired resampling first
        let (xs, m, nmap) = if rb.down {
            let xd = DevBuf::f32(gpu, g.p * rb.ch)?;
            k.seg_mean(x.ptr(), Dt::F32, rb.ch, g.ptr.ptr().cast(), g.idx.ptr().cast(), g.p, xd.fp())?;
            (Some(xd), g.p, &g.pnmap)
        } else if rb.up {
            let xu = DevBuf::f32(gpu, g.n * rb.ch)?;
            k.gather_rows(x.ptr(), Dt::F32, rb.ch, g.inv.ptr().cast(), g.n, xu.ptr())?;
            (Some(xu), g.n, &g.nmap)
        } else {
            (None, rows, if rows == g.n { &g.nmap } else { &g.pnmap })
        };
        let xs = xs.as_ref().unwrap_or(x);
        let h = DevBuf::new(gpu, m * rb.ch.max(rb.hid) * 2)?;
        k.ln_mod(xs.ptr(), Dt::F32, m, rb.ch, EPS, rb.norm1.0.fp(), rb.norm1.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), Dt::F16)?;
        k.act(h.ptr(), Dt::F16, m * rb.ch, 0)?;
        let t = DevBuf::f32(gpu, m * rb.hid)?;
        rb.conv1.apply(cx, &h, m, nmap, t.ptr(), Dt::F32)?;
        k.ln_mod(t.ptr(), Dt::F32, m, rb.hid, EPS, std::ptr::null(), std::ptr::null(), scale.fp(), shift.fp(), h.ptr(), Dt::F16)?;
        k.act(h.ptr(), Dt::F16, m * rb.hid, 0)?;
        let h2 = DevBuf::new(gpu, m * rb.hid * 2)?;
        rb.conv2.apply(cx, &h, m, nmap, h2.ptr(), Dt::F16)?;
        k.act(h2.ptr(), Dt::F16, m * rb.hid, 0)?;
        let out = DevBuf::f32(gpu, m * rb.out)?;
        rb.linear.apply(cx, h2.ptr(), m, out.ptr(), Dt::F32)?;
        match &rb.skip {
            Some(sk) => {
                let xh = DevBuf::new(gpu, m * rb.ch * 2)?;
                k.convert(xs.ptr(), Dt::F32, m * rb.ch, xh.ptr(), Dt::F16)?;
                let tmp = DevBuf::f32(gpu, m * rb.out)?;
                sk.apply_epi(&cx.nsd, k, xh.ptr(), m, out.ptr(), Dt::F32, Epi::Resid(std::ptr::null()), Some(&tmp))?;
            }
            None => k.add(out.ptr(), Dt::F32, xs.ptr(), Dt::F32, m * rb.out)?,
        }
        Ok((out, m))
    }

    /// The velocity [n, 32] for x [n, 32] at model time t (1000 sigma)
    pub fn forward(&self, cx: &Ctx, g: &Geo, x: &[f32], t: f32, cross: Cross, mut probe: Option<&mut Probe>) -> Result<Vec<f32>> {
        let (nsd, k, gpu) = (&cx.nsd, &cx.k, &cx.gpu);
        let (n, p) = (g.n, g.p);
        if x.len() != n * self.input.k {
            return Err(Error(format!("the flow's input: {} values for {n} voxels of {}", x.len(), self.input.k)));
        }
        let mut tap = |name: &'static str, b: &DevBuf, len: usize| -> Result<()> {
            if let Some(pr) = probe.as_deref_mut() {
                let mut v = b.to_f32()?;
                v.truncate(len);
                pr.taps.push((name, v));
            }
            Ok(())
        };
        // NS_M3D_HI3D_PROFILE: each part's time (synced: slower), on stderr after the call
        let prof = crate::PROFILE.load(std::sync::atomic::Ordering::Relaxed);
        let mut laps: Vec<(&'static str, f64)> = Vec::new();
        let mut t0 = std::time::Instant::now();
        let mut lap = |name: &'static str| -> Result<()> {
            if prof {
                gpu.sync()?;
                let d = t0.elapsed().as_secs_f64();
                match laps.iter_mut().find(|l| l.0 == name) {
                    Some(l) => l.1 += d,
                    None => laps.push((name, d)),
                }
                t0 = std::time::Instant::now();
            }
            Ok(())
        };
        let emb = self.t_emb(t);
        let xd = DevBuf::from_f32(gpu, x)?;
        let h0 = DevBuf::f32(gpu, n * 256)?;
        self.input.apply(cx, xd.ptr(), n, h0.ptr(), Dt::F32)?;
        tap("input", &h0, n * 256)?;
        let (s0, _) = self.res(cx, &self.inb[0], g, &h0, n, &emb)?;
        tap("in0", &s0, n * 256)?;
        let (s1, _) = self.res(cx, &self.inb[1], g, &s0, n, &emb)?;
        tap("in1", &s1, p * C)?;
        lap("input blocks")?;
        // the transformer on the parents, its stream starting from the second skip
        let h = DevBuf::f32(gpu, p * C)?;
        h.copy_within(0, &s1, 0, p * C * 4)?;
        let ada = self.ada.apply(&silu(&emb));
        let mods: Vec<f32> = self.blocks.iter().flat_map(|b| b.modulation.iter().zip(&ada).map(|(a, b)| a + b).collect::<Vec<_>>()).collect();
        let mods = DevBuf::from_f32(gpu, &mods)?;
        let hn = DevBuf::new(gpu, p * C * 2)?;
        let qkv = DevBuf::new(gpu, p * 3 * C * 2)?;
        let att = DevBuf::new(gpu, p * C * 2)?;
        let mlp = DevBuf::new(gpu, p * MLP * 2)?;
        // the attention's padded heads: q | k | v, and its output
        let qkvp = DevBuf::new(gpu, p * 3 * CP * 2)?;
        let attp = DevBuf::new(gpu, p * CP * 2)?;
        let qscale = (HP as f32 / HD as f32).sqrt();
        let fused = k.silo().is_some();
        let o = DevBuf::f32(gpu, if fused { 1 } else { p * C })?;
        let nb = self.blocks.len();
        for (bi, b) in self.blocks.iter().enumerate() {
            // SAFETY: rows of the [blocks, 6, 2112] modulation table.
            let md = |j: usize| unsafe { mods.fp().add(bi * 6 * C + j * C) as *const f32 };
            k.ln_mod(h.ptr(), Dt::F32, p, C, EPS, std::ptr::null(), std::ptr::null(), md(1), md(0), hn.ptr(), Dt::F16)?;
            b.qkv.apply(cx, hn.ptr(), p, qkv.ptr(), Dt::F16)?;
            k.rope_rms(qkv.ptr(), p, HEADS, HD, 3 * C, g.cs.fp(), b.q_gamma.fp())?;
            k.rope_rms(hp(&qkv, C), p, HEADS, HD, 3 * C, g.cs.fp(), b.k_gamma.fp())?;
            lap("blocks' linears")?;
            k.copy_heads(qkv.ptr(), 3 * C, HD, p, HEADS, qkvp.ptr(), 3 * CP, HP, qscale)?;
            k.copy_heads(hp(&qkv, C), 3 * C, HD, p, 2 * HEADS, hp(&qkvp, CP), 3 * CP, HP, 1.0)?;
            attend(cx, qkvp.ptr(), p, 3 * CP, hp(&qkvp, CP), hp(&qkvp, 2 * CP), p, 3 * CP, attp.ptr())?;
            k.copy_heads(attp.ptr(), CP, HP, p, HEADS, att.ptr(), C, HD, 1.0)?;
            lap("self-attention")?;
            b.out.apply_epi(nsd, k, att.ptr(), p, h.ptr(), Dt::F32, Epi::Resid(md(2)), Some(&o))?;
            match cross {
                Cross::Cached(kvs, l) => {
                    let kv = &kvs[bi];
                    k.ln_mod(h.ptr(), Dt::F32, p, C, EPS, b.norm2.0.fp(), b.norm2.1.fp(), std::ptr::null(), std::ptr::null(), hn.ptr(), Dt::F16)?;
                    b.cq.apply(cx, hn.ptr(), p, att.ptr(), Dt::F16)?;
                    k.copy_heads(att.ptr(), C, HD, p, HEADS, qkvp.ptr(), CP, HP, qscale)?;
                    attend(cx, qkvp.ptr(), p, CP, kv.ptr(), hp(kv, CP), l, 2 * CP, attp.ptr())?;
                    k.copy_heads(attp.ptr(), CP, HP, p, HEADS, hn.ptr(), C, HD, 1.0)?;
                    b.cout.apply_epi(nsd, k, hn.ptr(), p, h.ptr(), Dt::F32, Epi::Resid(std::ptr::null()), Some(&o))?;
                }
                Cross::Const(rows) => {
                    // SAFETY: block bi's row of the [blocks, 2112] table.
                    k.add_row(h.fp(), p, C, unsafe { rows.fp().add(bi * C) })?;
                }
            }
            k.ln_mod(h.ptr(), Dt::F32, p, C, EPS, std::ptr::null(), std::ptr::null(), md(4), md(3), hn.ptr(), Dt::F16)?;
            b.mlp0.apply_epi(nsd, k, hn.ptr(), p, mlp.ptr(), Dt::F16, Epi::Gelu, None)?;
            b.mlp2.apply_epi(nsd, k, mlp.ptr(), p, h.ptr(), Dt::F32, Epi::Resid(md(5)), Some(&o))?;
            if bi == 0 {
                tap("blk0", &h, p * C)?;
            } else if bi == nb - 1 {
                tap("blk23", &h, p * C)?;
            }
        }
        lap("blocks' linears")?;
        drop((hn, qkv, att, mlp, o, qkvp, attp));
        // out block 0: [h | skip1] at the parents, upsampled
        let cat = DevBuf::f32(gpu, p * 2 * C)?;
        k.put_cols(h.ptr(), Dt::F32, p, C, cat.ptr(), Dt::F32, 2 * C, 0)?;
        k.put_cols(s1.ptr(), Dt::F32, p, C, cat.ptr(), Dt::F32, 2 * C, C)?;
        drop((h, s1));
        let (o0, _) = self.res(cx, &self.outb[0], g, &cat, p, &emb)?;
        drop(cat);
        tap("out0", &o0, n * 256)?;
        let cat = DevBuf::f32(gpu, n * 512)?;
        k.put_cols(o0.ptr(), Dt::F32, n, 256, cat.ptr(), Dt::F32, 512, 0)?;
        k.put_cols(s0.ptr(), Dt::F32, n, 256, cat.ptr(), Dt::F32, 512, 256)?;
        let (o1, _) = self.res(cx, &self.outb[1], g, &cat, n, &emb)?;
        tap("out1", &o1, n * 256)?;
        lap("out blocks")?;
        k.ln_mod(o1.ptr(), Dt::F32, n, 256, 1e-5, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), o1.ptr(), Dt::F32)?;
        let v = DevBuf::f32(gpu, n * self.out.n)?;
        self.out.apply(cx, o1.ptr(), n, v.ptr(), Dt::F32)?;
        let v = v.to_f32()?;
        lap("out")?;
        if prof {
            eprintln!("  dit call ({n} voxels, {p} tokens): {}", laps.iter().map(|(k, d)| format!("{k} {d:.3} s")).collect::<Vec<_>>().join(", "));
        }
        Ok(v)
    }
}

/// softmax(q k^T / sqrt(128)) v over the padded heads (the flash kernel; q comes scaled so this is the model's
/// softmax(q k^T / sqrt(96)) v)
#[allow(clippy::too_many_arguments)]
fn attend(cx: &Ctx, q: *const std::ffi::c_void, sq: usize, qs: usize, kk: *const std::ffi::c_void, v: *const std::ffi::c_void, skv: usize, kvs: usize,
          out: *mut std::ffi::c_void) -> Result<()> {
    cx.nsd.attention_qk(q, sq, qs, kk, v, skv, kvs, Dt::F16, HEADS, HP, out, Dt::F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_is_per_axis_and_rescales_long_grids() {
        let cs = rope(&[[0, 0, 0], [1, 2, 3]]);
        assert_eq!(cs.len(), 2 * 96);
        assert!(cs[..96].chunks(2).all(|p| p == [1.0, 0.0]));
        let r = &cs[96..];
        assert!((r[0] - 1f32.cos()).abs() < 1e-6 && (r[1] - 1f32.sin()).abs() < 1e-6);
        assert!((r[32] - 2f32.cos()).abs() < 1e-6);
        assert!((r[64] - 3f32.cos()).abs() < 1e-6);
        // a coordinate of 64 or more rescales the frequencies (every one of the 16 has under 24 turns over 64)
        let long = rope(&[[1, 0, 0], [100, 0, 0]]);
        assert!((long[0] - 1f32.cos()).abs() > 1e-4);
    }
}
