//! One ncnn graph on the GPU: its weights on the card (convolutions as half in PyTorch's [Co, Ci, kh, kw]; a
//! transposed convolution as the convolution of its zero-dilated input with the flipped kernel), its layers run in
//! the file's order over channels-last half activations. Blobs are counted: a buffer goes back to the pool when its
//! last reader has run (Split's outputs share their input's), so a 999-layer RRDB holds only the few it needs.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::ffi::{self, check};
use crate::ncnn::{Bin, Graph, Params};

/// An activation fused into a layer (ncnn's activation_type 9 and its parameters 10)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Act {
    /// 0 none, 1 relu, 2 leaky relu, 4 sigmoid (ncnn's numbers; 3, clip, is not used by these graphs)
    pub kind: i32,
    pub a: f32,
}

fn act_of(p: &Params) -> Result<Act> {
    let kind = p.int(9, 0);
    let a = match kind {
        0 | 1 | 4 => 0.0,
        2 => p.array(10).first().copied().unwrap_or(0.0),
        k => return Err(Error(format!("activation type {k} is not run here"))),
    };
    Ok(Act { kind, a })
}

struct Conv {
    ci: usize,
    co: usize,
    k: usize,
    stride: usize,
    /// top, left, bottom, right zero padding of its input
    pad: [usize; 4],
    act: Act,
    w: DevBuf,
    b: Option<DevBuf>,
    /// a transposed convolution: dilate the input by `stride` first (then the convolution's own stride is 1)
    dilate: usize,
}

enum Op {
    Input,
    Conv(Conv),
    PRelu { slopes: DevBuf, n: usize },
    PixelShuffle(usize),
    Interp { kind: i32, hs: f32, ws: f32, oh: usize, ow: usize },
    /// op 0 sum (with coefficients), 1 product, 2 max
    Eltwise { op: i32, coeffs: Vec<f32> },
    Concat,
    Split,
    Crop { x: usize, y: usize },
    Mean,
    Dense { i: usize, o: usize, w: DevBuf, b: Option<DevBuf>, act: Act },
    Scale,
}

struct Layer {
    op: Op,
    name: String,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

pub struct Net {
    layers: Vec<Layer>,
    /// how many layers read each blob
    readers: Vec<usize>,
    input: usize,
    output: usize,
    pub name: String,
}

/// A blob's value: a picture [h, w, c] half, or a vector [c] float32
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub vec: bool,
}

impl Shape {
    pub fn bytes(&self) -> usize {
        if self.vec {
            self.c * 4
        } else {
            self.h * self.w * self.c * 2
        }
    }
}

/// Buffers handed back and out again: a run's activations reuse the ones before them
pub struct Pool {
    gpu: Arc<Gpu>,
    free: Vec<DevBuf>,
}

impl Pool {
    pub fn new(gpu: &Arc<Gpu>) -> Pool {
        Pool { gpu: gpu.clone(), free: Vec::new() }
    }
    /// A buffer of at least `bytes` (the smallest that fits, else a new one)
    pub fn get(&mut self, bytes: usize) -> Result<DevBuf> {
        let best = self.free.iter().enumerate().filter(|(_, b)| b.len >= bytes).min_by_key(|(_, b)| b.len).map(|(i, _)| i);
        if let Some(i) = best {
            return Ok(self.free.swap_remove(i));
        }
        // rounded up so the edge tiles' slightly different sizes find the same buffers again
        let n = bytes.max(1).div_ceil(1 << 16) << 16;
        DevBuf::new(&self.gpu, n)
    }
    pub fn put(&mut self, b: DevBuf) {
        self.free.push(b);
    }
    /// Every free buffer released (after a request, when memory is wanted elsewhere)
    pub fn clear(&mut self) {
        self.free.clear();
    }
    pub fn held(&self) -> usize {
        self.free.iter().map(|b| b.len).sum()
    }
}

fn half_buf(gpu: &Arc<Gpu>, v: &[u16]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, v.len() * 2)?;
    let bytes: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
    b.write(0, &bytes)?;
    Ok(b)
}

fn conv_of(gpu: &Arc<Gpu>, l: &crate::ncnn::Layer, bin: &mut Bin, transposed: bool) -> Result<Conv> {
    let p = &l.params;
    let co = p.int(0, 0) as usize;
    let kw = p.int(1, 0) as usize;
    let kh = p.int(11, kw as i32) as usize;
    let dil = p.int(2, 1);
    let stride = p.int(3, 1) as usize;
    let pl = p.int(4, 0);
    let pt = p.int(14, pl);
    let pr = p.int(15, pl);
    let pb = p.int(16, pt);
    let bias = p.int(5, 0) != 0;
    let wsize = p.int(6, 0) as usize;
    if kh != kw || dil != 1 || p.int(12, 1) != 1 || p.int(13, stride as i32) != stride as i32 || p.int(7, 1) != 1 {
        return Err(Error(format!("{}: only square, undilated, ungrouped kernels are run here", l.name)));
    }
    if [pl, pt, pr, pb].iter().any(|v| *v < 0) {
        return Err(Error(format!("{}: SAME padding (-233) is not run here", l.name)));
    }
    let k = kw;
    if co == 0 || k == 0 || !wsize.is_multiple_of(co * k * k) {
        return Err(Error(format!("{}: {wsize} weights for {co} outputs of {k}x{k}", l.name)));
    }
    let ci = wsize / (co * k * k);
    let act = act_of(p).map_err(|e| Error(format!("{}: {}", l.name, e.0)))?;
    let w = bin.weight(wsize).map_err(|e| Error(format!("{}: {e}", l.name)))?;
    let b = if bias { Some(bin.floats(co).map_err(|e| Error(format!("{}: {e}", l.name)))?) } else { None };
    let mut w16 = w.to_f16();
    let (pad, dilate) = if transposed {
        // ncnn keeps a transposed convolution's kernel as [Co, Ci, kh, kw] (its converter swaps PyTorch's in and
        // out): as a convolution over the dilated input it is the same kernel flipped in both directions
        let mut f = vec![0u16; w16.len()];
        for o in 0..co {
            for i in 0..ci {
                for y in 0..k {
                    for x in 0..k {
                        f[((o * ci + i) * k + y) * k + x] = w16[((o * ci + i) * k + (k - 1 - y)) * k + (k - 1 - x)];
                    }
                }
            }
        }
        w16 = f;
        let full = k as i32 - 1;
        let (ot, ol) = (p.int(19, 0), p.int(18, 0));
        if [pl, pt, pr, pb].iter().any(|v| *v > full) {
            return Err(Error(format!("{}: a transposed convolution padded past its kernel", l.name)));
        }
        ([(full - pt) as usize, (full - pl) as usize, (full - pb + ot) as usize, (full - pr + ol) as usize], stride)
    } else {
        ([pt as usize, pl as usize, pb as usize, pr as usize], 1)
    };
    Ok(Conv {
        ci,
        co,
        k,
        stride: if transposed { 1 } else { stride },
        pad,
        act,
        w: half_buf(gpu, &w16)?,
        b: b.map(|v| DevBuf::from_f32(gpu, &v)).transpose()?,
        dilate,
    })
}

impl Net {
    /// The graph and its weights (the `.bin` read through, in the layers' order), onto `gpu`
    pub fn load(gpu: &Arc<Gpu>, g: &Graph, bin: &[u8], name: &str) -> Result<Net> {
        let mut r = Bin::new(bin);
        let mut layers = Vec::new();
        let mut input = None;
        for l in &g.layers {
            let p = &l.params;
            let e = |m: String| Error(format!("{name}: {}: {m}", l.name));
            let op = match l.kind.as_str() {
                "Input" => {
                    input = Some(l.outputs[0]);
                    Op::Input
                }
                "Convolution" => Op::Conv(conv_of(gpu, l, &mut r, false)?),
                "Deconvolution" => Op::Conv(conv_of(gpu, l, &mut r, true)?),
                "PReLU" => {
                    let n = p.int(0, 1).max(1) as usize;
                    Op::PRelu { slopes: DevBuf::from_f32(gpu, &r.floats(n).map_err(e)?)?, n }
                }
                "PixelShuffle" => {
                    if p.int(1, 0) != 0 {
                        return Err(e("PixelShuffle mode 1 (DCR) is not run here".into()));
                    }
                    Op::PixelShuffle(p.int(0, 1) as usize)
                }
                "Interp" => {
                    let kind = p.int(0, 0);
                    if !(1..=3).contains(&kind) || p.int(6, 0) != 0 {
                        return Err(e(format!("Interp type {kind} (align_corner {}) is not run here", p.int(6, 0))));
                    }
                    Op::Interp { kind, hs: p.float(1, 1.0), ws: p.float(2, 1.0), oh: p.int(3, 0) as usize, ow: p.int(4, 0) as usize }
                }
                "BinaryOp" => {
                    if p.int(1, 0) != 0 {
                        return Err(e("BinaryOp with a scalar is not run here".into()));
                    }
                    match p.int(0, 0) {
                        0 => Op::Eltwise { op: 0, coeffs: vec![] },
                        1 => Op::Eltwise { op: 0, coeffs: vec![1.0, -1.0] },
                        2 => Op::Eltwise { op: 1, coeffs: vec![] },
                        k => return Err(e(format!("BinaryOp {k} is not run here"))),
                    }
                }
                "Eltwise" => {
                    let op = match p.int(0, 1) {
                        0 => 1,
                        1 => 0,
                        2 => 2,
                        k => return Err(e(format!("Eltwise {k} is not run here"))),
                    };
                    Op::Eltwise { op, coeffs: p.array(1) }
                }
                "Concat" => {
                    if p.int(0, 0) != 0 {
                        return Err(e("Concat on an axis but channels is not run here".into()));
                    }
                    Op::Concat
                }
                "Split" => Op::Split,
                "Crop" => {
                    if l.inputs.len() != 2 {
                        return Err(e("Crop with one input (its own sizes) is not run here".into()));
                    }
                    Op::Crop { x: p.int(0, 0) as usize, y: p.int(1, 0) as usize }
                }
                "Pooling" => {
                    if p.int(0, 0) != 1 || p.int(4, 0) != 1 {
                        return Err(e("only global average pooling is run here".into()));
                    }
                    Op::Mean
                }
                "InnerProduct" => {
                    let o = p.int(0, 0) as usize;
                    let ws = p.int(2, 0) as usize;
                    if o == 0 || !ws.is_multiple_of(o) {
                        return Err(e(format!("{ws} weights for {o} outputs")));
                    }
                    let w = r.weight(ws).map_err(e)?.to_f32();
                    let b = if p.int(1, 0) != 0 { Some(r.floats(o).map_err(e)?) } else { None };
                    let act = act_of(p).map_err(|x| e(x.0))?;
                    Op::Dense { i: ws / o, o, w: DevBuf::from_f32(gpu, &w)?, b: b.map(|v| DevBuf::from_f32(gpu, &v)).transpose()?, act }
                }
                "Scale" => {
                    if p.int(0, 0) != -233 || p.int(1, 0) != 0 {
                        return Err(e("only Scale by a second blob, without bias, is run here".into()));
                    }
                    Op::Scale
                }
                k => return Err(e(format!("{k} is not run here"))),
            };
            layers.push(Layer { op, name: l.name.clone(), inputs: l.inputs.clone(), outputs: l.outputs.clone() });
        }
        if r.rest() != 0 {
            return Err(Error(format!("{name}: {} bytes of weights left over (the graph and the weights do not belong together)", r.rest())));
        }
        let input = input.ok_or_else(|| Error(format!("{name}: no Input layer")))?;
        let output = *layers.last().and_then(|l| l.outputs.first()).ok_or_else(|| Error(format!("{name}: no layers")))?;
        let mut readers = vec![0usize; g.blobs.len()];
        for l in &layers {
            for i in &l.inputs {
                readers[*i] += 1;
            }
        }
        Ok(Net { layers, readers, input, output, name: name.to_string() })
    }

    /// The graph over `x` [h, w, 3] half (taken over: it goes back to the pool when read): the output blob's buffer
    /// and shape (the caller puts it back in the pool)
    pub fn run(&self, gpu: &Arc<Gpu>, nsd: &Nsd, pool: &mut Pool, x: DevBuf, shape: Shape) -> Result<(DevBuf, Shape)> {
        let k = ffi::api()?;
        let gp = ffi::g(gpu);
        // storage: buffers by id, the readers left; a blob names its storage
        let mut store: Vec<Option<(DevBuf, usize)>> = Vec::new();
        let mut blob: Vec<Option<(usize, Shape)>> = vec![None; self.readers.len()];
        let readers = |b: usize| if b == self.output { self.readers[b] + 1 } else { self.readers[b] };
        let put = |store: &mut Vec<Option<(DevBuf, usize)>>, buf: DevBuf, refs: usize| -> usize {
            store.push(Some((buf, refs)));
            store.len() - 1
        };
        if let Some(dir) = std::env::var_os("NS_UP_DUMP") {
            let mut raw = vec![0u8; shape.bytes()];
            x.read(0, &mut raw)?;
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("input_{}x{}x{}.raw", shape.h, shape.w, shape.c)), &raw);
        }
        let first = put(&mut store, x, readers(self.input));
        blob[self.input] = Some((first, shape));
        for l in &self.layers {
            let e = |m: String| Error(format!("{}: {}: {m}", self.name, l.name));
            let ins: Vec<(usize, Shape)> = l.inputs.iter().map(|i| blob[*i].ok_or_else(|| e(format!("blob {i} has no value yet")))).collect::<Result<_>>()?;
            let ptr = |store: &Vec<Option<(DevBuf, usize)>>, s: usize| store[s].as_ref().map(|(b, _)| b.ptr()).unwrap_or(std::ptr::null_mut());
            let mut outs: Vec<(DevBuf, Shape)> = Vec::new();
            match &l.op {
                Op::Input => {}
                Op::Split => {
                    let (s, sh) = ins[0];
                    for o in &l.outputs {
                        if let Some(st) = store[s].as_mut() {
                            st.1 += readers(*o);
                        }
                        blob[*o] = Some((s, sh));
                    }
                }
                Op::Conv(c) => {
                    let (s, sh) = ins[0];
                    if sh.vec || sh.c != c.ci {
                        return Err(e(format!("{} channels in, the kernel takes {}", sh.c, c.ci)));
                    }
                    let mut src = ptr(&store, s) as *const _;
                    let (mut h, mut w) = (sh.h, sh.w);
                    let mut tmp: Vec<DevBuf> = Vec::new();
                    if c.dilate > 1 {
                        let (dh, dw) = ((h - 1) * c.dilate + 1, (w - 1) * c.dilate + 1);
                        let d = pool.get(dh * dw * c.ci * 2)?;
                        // SAFETY: the buffers hold the sizes named.
                        check(unsafe { (k.dilate)(gp, src, h as i64, w as i64, c.ci as i64, c.dilate as i64, d.ptr()) }, "dilate")?;
                        src = d.ptr() as *const _;
                        h = dh;
                        w = dw;
                        tmp.push(d);
                    }
                    let [pt, pl, pb, pr] = c.pad;
                    if pt + pl + pb + pr > 0 {
                        let (ph, pw) = (h + pt + pb, w + pl + pr);
                        let d = pool.get(ph * pw * c.ci * 2)?;
                        // SAFETY: as above.
                        check(unsafe { (k.pad)(gp, src, h as i64, w as i64, c.ci as i64, pt as i64, pl as i64, pb as i64, pr as i64, d.ptr()) }, "pad")?;
                        src = d.ptr() as *const _;
                        h = ph;
                        w = pw;
                        tmp.push(d);
                    }
                    if h < c.k || w < c.k {
                        return Err(e(format!("{h}x{w} is smaller than the {}x{} kernel", c.k, c.k)));
                    }
                    let (oh, ow) = ((h - c.k) / c.stride + 1, (w - c.k) / c.stride + 1);
                    let out = pool.get(oh * ow * c.co * 2)?;
                    let bias = c.b.as_ref().map(|b| b.ptr() as *const _).unwrap_or(std::ptr::null());
                    nsd.conv3d_ex(src, Dt::F16, (1, h, w), c.ci, c.w.ptr(), c.co, (1, c.k, c.k), (1, c.stride, c.stride), bias, out.ptr())
                        .map_err(|x| e(x.0))?;
                    if c.act.kind != 0 {
                        // SAFETY: out holds oh x ow x co halves.
                        check(unsafe { (k.act)(gp, out.ptr(), (oh * ow) as i64, c.co as i64, c.act.kind, c.act.a, std::ptr::null(), 0) }, "activation")?;
                    }
                    for t in tmp {
                        pool.put(t);
                    }
                    outs.push((out, Shape { h: oh, w: ow, c: c.co, vec: false }));
                }
                Op::PRelu { slopes, n } => {
                    let (s, sh) = ins[0];
                    let out = pool.get(sh.bytes())?;
                    out.copy_within(0, store[s].as_ref().map(|x| &x.0).unwrap(), 0, sh.bytes())?;
                    // SAFETY: out holds the shape's halves, slopes n floats.
                    check(unsafe { (k.act)(gp, out.ptr(), (sh.h * sh.w) as i64, sh.c as i64, 3, 0.0, slopes.ptr() as *const f32, *n as i64) }, "prelu")?;
                    outs.push((out, sh));
                }
                Op::PixelShuffle(r) => {
                    let (s, sh) = ins[0];
                    if sh.c % (r * r) != 0 {
                        return Err(e(format!("{} channels do not shuffle by {r}", sh.c)));
                    }
                    let c = sh.c / (r * r);
                    let out = pool.get(sh.h * r * sh.w * r * c * 2)?;
                    // SAFETY: as above.
                    check(unsafe { (k.pixel_shuffle)(gp, ptr(&store, s), sh.h as i64, sh.w as i64, c as i64, *r as i64, out.ptr()) }, "pixel shuffle")?;
                    outs.push((out, Shape { h: sh.h * r, w: sh.w * r, c, vec: false }));
                }
                Op::Interp { kind, hs, ws, oh, ow } => {
                    let (s, sh) = ins[0];
                    let ho = if *oh > 0 { *oh } else { (sh.h as f32 * hs) as usize };
                    let wo = if *ow > 0 { *ow } else { (sh.w as f32 * ws) as usize };
                    let out = pool.get(ho * wo * sh.c * 2)?;
                    // SAFETY: as above.
                    check(unsafe { (k.resize)(gp, ptr(&store, s), sh.h as i64, sh.w as i64, sh.c as i64, ho as i64, wo as i64, *kind, out.ptr()) }, "interp")?;
                    outs.push((out, Shape { h: ho, w: wo, c: sh.c, vec: false }));
                }
                Op::Eltwise { op, coeffs } => {
                    let sh = ins[0].1;
                    if ins.iter().any(|(_, s)| s.h != sh.h || s.w != sh.w || s.c != sh.c || s.vec) {
                        return Err(e(format!("inputs of different shapes: {:?}", ins.iter().map(|x| x.1).collect::<Vec<_>>())));
                    }
                    let n = (sh.h * sh.w * sh.c) as i64;
                    let co = |i: usize| coeffs.get(i).copied().unwrap_or(1.0);
                    let out = pool.get(sh.bytes())?;
                    // SAFETY: every input and out hold n halves.
                    check(unsafe { (k.add)(gp, ptr(&store, ins[0].0), ptr(&store, ins[1].0), n, *op, co(0), co(1), out.ptr()) }, "eltwise")?;
                    for (j, (s, _)) in ins.iter().enumerate().skip(2) {
                        // SAFETY: as above (out read and written in place, elementwise)
                        check(unsafe { (k.add)(gp, out.ptr(), ptr(&store, *s), n, *op, 1.0, co(j), out.ptr()) }, "eltwise")?;
                    }
                    outs.push((out, sh));
                }
                Op::Concat => {
                    let sh = ins[0].1;
                    let c: usize = ins.iter().map(|x| x.1.c).sum();
                    if ins.iter().any(|(_, s)| s.h != sh.h || s.w != sh.w || s.vec) {
                        return Err(e("concatenated pictures of different sizes".into()));
                    }
                    let out = pool.get(sh.h * sh.w * c * 2)?;
                    let mut off = 0;
                    for (s, si) in &ins {
                        // SAFETY: out holds P x c halves, the input P x si.c.
                        check(unsafe { (k.copy_channels)(gp, ptr(&store, *s), (sh.h * sh.w) as i64, si.c as i64, out.ptr(), c as i64, off as i64) }, "concat")?;
                        off += si.c;
                    }
                    outs.push((out, Shape { h: sh.h, w: sh.w, c, vec: false }));
                }
                Op::Crop { x, y } => {
                    let (s, sh) = ins[0];
                    let r = ins[1].1;
                    if x + r.w > sh.w || y + r.h > sh.h {
                        return Err(e(format!("a {}x{} crop at ({x}, {y}) of {}x{}", r.w, r.h, sh.w, sh.h)));
                    }
                    let out = pool.get(r.h * r.w * sh.c * 2)?;
                    // SAFETY: as above.
                    check(unsafe { (k.crop)(gp, ptr(&store, s), sh.w as i64, sh.c as i64, *x as i64, *y as i64, r.w as i64, r.h as i64, out.ptr()) }, "crop")?;
                    outs.push((out, Shape { h: r.h, w: r.w, c: sh.c, vec: false }));
                }
                Op::Mean => {
                    let (s, sh) = ins[0];
                    let out = pool.get(sh.c * 4)?;
                    // SAFETY: as above.
                    check(unsafe { (k.mean)(gp, ptr(&store, s), (sh.h * sh.w) as i64, sh.c as i64, out.ptr() as *mut f32) }, "mean")?;
                    outs.push((out, Shape { h: 1, w: 1, c: sh.c, vec: true }));
                }
                Op::Dense { i, o, w, b, act } => {
                    let (s, sh) = ins[0];
                    if !sh.vec || sh.c != *i {
                        return Err(e(format!("{} values in, the layer takes {i}", sh.c)));
                    }
                    let out = pool.get(o * 4)?;
                    let bias = b.as_ref().map(|b| b.ptr() as *const f32).unwrap_or(std::ptr::null());
                    // SAFETY: as above.
                    check(unsafe { (k.dense)(gp, ptr(&store, s) as *const f32, *i as i64, w.ptr() as *const f32, bias, *o as i64, act.kind, act.a,
                                             out.ptr() as *mut f32) }, "dense")?;
                    outs.push((out, Shape { h: 1, w: 1, c: *o, vec: true }));
                }
                Op::Scale => {
                    let (s, sh) = ins[0];
                    let (v, vs) = ins[1];
                    if !vs.vec || vs.c != sh.c {
                        return Err(e(format!("a scale of {} values for {} channels", vs.c, sh.c)));
                    }
                    let out = pool.get(sh.bytes())?;
                    out.copy_within(0, store[s].as_ref().map(|x| &x.0).unwrap(), 0, sh.bytes())?;
                    // SAFETY: as above.
                    check(unsafe { (k.scale)(gp, out.ptr(), (sh.h * sh.w) as i64, sh.c as i64, ptr(&store, v) as *const f32) }, "scale")?;
                    outs.push((out, sh));
                }
            }
            // what it read, counted down (a buffer whose readers are all done goes back)
            for (s, _) in &ins {
                let done = match store[*s].as_mut() {
                    Some(st) => {
                        st.1 -= 1;
                        st.1 == 0
                    }
                    None => return Err(e("a blob read after its buffer was freed".into())),
                };
                if done {
                    if let Some((b, _)) = store[*s].take() {
                        pool.put(b);
                    }
                }
            }
            if let Some(dir) = std::env::var_os("NS_UP_DUMP") {
                // a debugging aid: each layer's output as raw halves (or float32 for vectors), its shape in the name
                for ((buf, sh), o) in outs.iter().zip(&l.outputs) {
                    let mut raw = vec![0u8; sh.bytes()];
                    buf.read(0, &mut raw)?;
                    let f = std::path::Path::new(&dir).join(format!("{:04}_{}_{}x{}x{}{}.raw", o, l.name, sh.h, sh.w, sh.c, if sh.vec { "_f32" } else { "" }));
                    std::fs::write(&f, &raw).map_err(|x| e(format!("{}: {x}", f.display())))?;
                }
            }
            for ((buf, sh), o) in outs.into_iter().zip(&l.outputs) {
                let n = readers(*o);
                if n == 0 {
                    pool.put(buf);
                    continue;
                }
                let id = put(&mut store, buf, n);
                blob[*o] = Some((id, sh));
            }
        }
        let (s, sh) = blob[self.output].ok_or_else(|| Error(format!("{}: the output blob was never made", self.name)))?;
        let (buf, _) = store[s].take().ok_or_else(|| Error(format!("{}: the output's buffer is gone", self.name)))?;
        // whatever is still held (blobs nobody read) goes back
        for st in store.into_iter().flatten() {
            pool.put(st.0);
        }
        Ok((buf, sh))
    }
}
