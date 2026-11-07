//! IS-Net (xuebinqin/DIS `ISNetDIS`, rembg's isnet-general-use) in float32 on the GPU: a stride-2 stem, the RSU
//! U-blocks (RSU7, 6, 5, 4 and the dilated 4F) down to 1/32 and back up, the first side output upsampled to the input
//! and through a sigmoid - what rembg reads (`output_image`).
//!
//! ```text
//!   x [1024, 1024, 3] -> conv_in (3x3, stride 2) -> stage1 RSU7 -> pool -> stage2 RSU6 -> pool -> stage3 RSU5 -> pool
//!   -> stage4 RSU4 -> pool -> stage5 RSU4F -> pool -> stage6 RSU4F
//!   -> stage5d RSU4F [up(6), 5] -> stage4d RSU4 [up, 4] -> stage3d RSU5 -> stage2d RSU6 -> stage1d RSU7 [up, 1]
//!   -> side1 (3x3 -> 1) -> up to 1024 -> sigmoid
//!   RSU-L(x): xin = rebn(x); encoder rebn + pool L-2 times, rebn at dilation 2; decoder rebn([up(d), skip]); + xin
//! ```
//!
//! The weights come from the ONNX file (its batch norms folded into the convolutions); each convolution is found by
//! the module path its output carries (`/stage1/rebnconv1/conv_s1/Conv_output_0`), its stride and dilation read from
//! its node. Convolutions are the shared vision im2col (dilated, strided) and the diffusion kernels' GEMM (oneDNN,
//! float32). After nextcuda's isnet.rs.

use std::collections::HashMap;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ffi::{self, check};
use crate::onnx;

pub const SIDE: usize = 1024;
/// bytes of unfolded columns at a time
const COL_BYTES: usize = 128 << 20;

struct Conv {
    w: DevBuf,
    b: DevBuf,
    co: usize,
    ci: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
}

/// A feature map on the card: float32 [h, w, c]
pub struct Map {
    pub buf: DevBuf,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

pub struct IsNet {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    convs: HashMap<String, Conv>,
    col: DevBuf,
    pub bytes: usize,
}

impl IsNet {
    pub fn load(m: &onnx::Model, gpu: &Arc<Gpu>) -> Result<IsNet> {
        let mut convs = HashMap::new();
        let mut bytes = 0;
        for n in m.nodes.iter().filter(|n| n.op == "Conv") {
            let out = n.outputs.first().ok_or_else(|| Error("isnet: a convolution with no output".into()))?;
            let path = out.trim_end_matches("/Conv_output_0").trim_end_matches("/conv_s1").trim_start_matches('/').to_string();
            let wt = n.inputs.get(1).and_then(|i| m.tensors.get(i)).ok_or_else(|| Error(format!("isnet: {path} has no weight")))?;
            let bt = m.tensors.get(n.inputs.get(2).map(String::as_str).unwrap_or("")).ok_or_else(|| Error(format!("isnet: {path} has no bias")))?;
            if wt.dims.len() != 4 {
                return Err(Error(format!("isnet: {path}: a weight of {:?}, not a 2-D convolution", wt.dims)));
            }
            let (co, ci, k) = (wt.dims[0], wt.dims[1], wt.dims[2]);
            if wt.dims[3] != k || n.ints.get("group").is_some_and(|g| g[0] != 1) || wt.data.len() != co * ci * k * k || bt.data.len() != co {
                return Err(Error(format!("isnet: {path}: a {:?} grouped, non-square or short convolution", wt.dims)));
            }
            // [co, ci, k, k] -> [co, k, k, ci]: the im2col's (ky, kx, c) order
            let mut r = vec![0f32; wt.data.len()];
            for o in 0..co {
                for c in 0..ci {
                    for t in 0..k * k {
                        r[(o * k * k + t) * ci + c] = wt.data[(o * ci + c) * k * k + t];
                    }
                }
            }
            let one = |key: &str| n.ints.get(key).and_then(|v| v.first().copied()).unwrap_or(1) as usize;
            let pad = n.ints.get("pads").and_then(|v| v.first().copied()).unwrap_or(0) as usize;
            bytes += r.len() * 4;
            convs.insert(path, Conv { w: DevBuf::from_f32(gpu, &r)?, b: DevBuf::from_f32(gpu, &bt.data)?, co, ci, k, stride: one("strides"), pad, dil: one("dilations") });
        }
        if !convs.contains_key("conv_in") || !convs.contains_key("side1") {
            return Err(Error("isnet: not IS-Net's graph (no conv_in, side1)".into()));
        }
        Ok(IsNet { gpu: gpu.clone(), nsd: Nsd::new(gpu)?, convs, col: DevBuf::new(gpu, COL_BYTES)?, bytes })
    }

    fn conv(&self, name: &str, x: &Map, relu: bool) -> Result<Map> {
        let v = ffi::vision()?;
        let r = ffi::api()?;
        let c = self.convs.get(name).ok_or_else(|| Error(format!("isnet: no convolution {name}")))?;
        if c.ci != x.c {
            return Err(Error(format!("isnet: {name} takes {} channels, given {}", c.ci, x.c)));
        }
        let ho = (x.h + 2 * c.pad - c.dil * (c.k - 1) - 1) / c.stride + 1;
        let wo = (x.w + 2 * c.pad - c.dil * (c.k - 1) - 1) / c.stride + 1;
        let out = DevBuf::f32(&self.gpu, ho * wo * c.co)?;
        let kk = c.k * c.k * c.ci;
        let tile = (COL_BYTES / (kk * 4)).max(1);
        let g = self.gpu.raw();
        let mut p0 = 0;
        while p0 < ho * wo {
            let n = tile.min(ho * wo - p0);
            // SAFETY: x holds h x w x c floats, col n x kk (COL_BYTES), out ho x wo x co.
            check(unsafe { (v.im2col)(g, x.buf.fp(), x.h as i64, x.w as i64, x.c as i64, c.k as i64, c.k as i64, c.stride as i64, c.pad as i64,
                                      c.dil as i64, ho as i64, wo as i64, p0 as i64, n as i64, self.col.fp()) }, "isnet im2col")?;
            // SAFETY: rows p0 .. p0 + n of out's ho * wo rows of co floats.
            let o = unsafe { out.fp().add(p0 * c.co) };
            self.nsd.linear(self.col.ptr(), Dt::F32, n, kk, c.w.ptr(), c.co, none(), o.cast(), Dt::F32)?;
            p0 += n;
        }
        // SAFETY: out holds ho x wo x co floats, the bias co.
        check(unsafe { (r.bias_act)(g, out.fp(), (ho * wo) as i64, c.co as i64, c.b.fp(), relu as i32) }, "isnet bias")?;
        Ok(Map { buf: out, h: ho, w: wo, c: c.co })
    }

    fn pool(&self, x: &Map) -> Result<Map> {
        let r = ffi::api()?;
        let (h, w) = (x.h.div_ceil(2), x.w.div_ceil(2));
        let out = DevBuf::f32(&self.gpu, h * w * x.c)?;
        // SAFETY: x holds h x w x c floats, out the pooled size.
        check(unsafe { (r.maxpool2)(self.gpu.raw(), x.buf.fp(), x.h as i64, x.w as i64, x.c as i64, out.fp()) }, "isnet pool")?;
        Ok(Map { buf: out, h, w, c: x.c })
    }

    /// [a upsampled to b's size, b] along the channels
    fn up_cat(&self, a: &Map, b: &Map) -> Result<Map> {
        let r = ffi::api()?;
        let v = ffi::vision()?;
        let c = a.c + b.c;
        let out = DevBuf::f32(&self.gpu, b.h * b.w * c)?;
        let g = self.gpu.raw();
        if a.h == b.h && a.w == b.w {
            // SAFETY: a holds h x w x a.c floats, out h x w x c.
            check(unsafe { (v.copy_cols)(g, a.buf.fp(), (b.h * b.w) as i64, a.c as i64, a.c as i64, out.fp(), c as i64) }, "isnet cat")?;
        } else {
            // SAFETY: a holds its h x w x a.c floats, out b's h x w x c.
            check(unsafe { (r.resize)(g, a.buf.fp(), a.h as i64, a.w as i64, a.c as i64, b.h as i64, b.w as i64, out.fp(), c as i64) }, "isnet up")?;
        }
        // SAFETY: b holds h x w x b.c floats, written after a's channels of each row of c.
        check(unsafe { (v.copy_cols)(g, b.buf.fp(), (b.h * b.w) as i64, b.c as i64, b.c as i64, out.fp().add(a.c), c as i64) }, "isnet cat")?;
        Ok(Map { buf: out, h: b.h, w: b.w, c })
    }

    fn add(&self, x: &Map, y: &Map) -> Result<()> {
        let v = ffi::vision()?;
        // SAFETY: x and y hold the same number of floats.
        check(unsafe { (v.add)(self.gpu.raw(), x.buf.fp(), y.buf.fp(), (x.h * x.w * x.c) as i64) }, "isnet add")
    }

    /// RSU-L (7, 6, 5, 4): an encoder of L - 1 levels (pooled between), the last at dilation 2, a decoder back up
    fn rsu(&self, p: &str, l: usize, x: &Map) -> Result<Map> {
        let xin = self.conv(&format!("{p}/rebnconvin"), x, true)?;
        let mut enc = vec![self.conv(&format!("{p}/rebnconv1"), &xin, true)?];
        for i in 2..l {
            let pooled = self.pool(enc.last().unwrap())?;
            enc.push(self.conv(&format!("{p}/rebnconv{i}"), &pooled, true)?);
        }
        let deep = self.conv(&format!("{p}/rebnconv{l}"), enc.last().unwrap(), true)?;
        let mut d = self.conv(&format!("{p}/rebnconv{}d", l - 1), &self.up_cat(&deep, &enc[l - 2])?, true)?;
        for i in (1..l - 1).rev() {
            d = self.conv(&format!("{p}/rebnconv{i}d"), &self.up_cat(&d, &enc[i - 1])?, true)?;
        }
        self.add(&d, &xin)?;
        Ok(d)
    }

    /// RSU4F: no pooling, dilations 1, 2, 4, 8 and back (read from the nodes)
    fn rsu4f(&self, p: &str, x: &Map) -> Result<Map> {
        let xin = self.conv(&format!("{p}/rebnconvin"), x, true)?;
        let h1 = self.conv(&format!("{p}/rebnconv1"), &xin, true)?;
        let h2 = self.conv(&format!("{p}/rebnconv2"), &h1, true)?;
        let h3 = self.conv(&format!("{p}/rebnconv3"), &h2, true)?;
        let h4 = self.conv(&format!("{p}/rebnconv4"), &h3, true)?;
        let d3 = self.conv(&format!("{p}/rebnconv3d"), &self.up_cat(&h4, &h3)?, true)?;
        let d2 = self.conv(&format!("{p}/rebnconv2d"), &self.up_cat(&d3, &h2)?, true)?;
        let d1 = self.conv(&format!("{p}/rebnconv1d"), &self.up_cat(&d2, &h1)?, true)?;
        self.add(&d1, &xin)?;
        Ok(d1)
    }

    /// The prediction (the sigmoid, [1024 * 1024]) for an input [1024, 1024, 3] (rembg's normalization)
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != SIDE * SIDE * 3 {
            return Err(Error(format!("isnet's input: {} values, not 1024 x 1024 x 3", input.len())));
        }
        let x = Map { buf: DevBuf::from_f32(&self.gpu, input)?, h: SIDE, w: SIDE, c: 3 };
        let hxin = self.conv("conv_in", &x, false)?;
        let h1 = self.rsu("stage1", 7, &hxin)?;
        let h2 = self.rsu("stage2", 6, &self.pool(&h1)?)?;
        let h3 = self.rsu("stage3", 5, &self.pool(&h2)?)?;
        let h4 = self.rsu("stage4", 4, &self.pool(&h3)?)?;
        let h5 = self.rsu4f("stage5", &self.pool(&h4)?)?;
        let h6 = self.rsu4f("stage6", &self.pool(&h5)?)?;
        let d5 = self.rsu4f("stage5d", &self.up_cat(&h6, &h5)?)?;
        let d4 = self.rsu("stage4d", 4, &self.up_cat(&d5, &h4)?)?;
        let d3 = self.rsu("stage3d", 5, &self.up_cat(&d4, &h3)?)?;
        let d2 = self.rsu("stage2d", 6, &self.up_cat(&d3, &h2)?)?;
        let d1 = self.rsu("stage1d", 7, &self.up_cat(&d2, &h1)?)?;
        let s1 = self.conv("side1", &d1, false)?;
        let r = ffi::api()?;
        let out = DevBuf::f32(&self.gpu, SIDE * SIDE)?;
        let g = self.gpu.raw();
        // SAFETY: s1 holds its h x w floats, out 1024 x 1024.
        check(unsafe { (r.resize)(g, s1.buf.fp(), s1.h as i64, s1.w as i64, 1, SIDE as i64, SIDE as i64, out.fp(), 1) }, "isnet up")?;
        // SAFETY: out holds 1024 x 1024 floats.
        check(unsafe { (r.sigmoid)(g, out.fp(), (SIDE * SIDE) as i64) }, "isnet sigmoid")?;
        self.nsd.wait()?;
        out.to_f32()
    }
}
