//! Depth Anything V2 Small (`DepthAnythingForDepthEstimation`, LiheYoung's DA-V2, depth-anything's `-hf` weights) in
//! float32 on the GPU: a DINOv2-small ViT (12 layers of 6 heads, LayerScale, 14 x 14 patches, positions interpolated
//! to the picture's grid) read at layers 3, 6, 9, 12, a DPT neck (reassemble to 4x, 2x, 1x, 1/2 and four feature
//! fusions of 64 channels) and the depth head.
//!
//! ```text
//!   x [h, w, 3] (ImageNet-normalized, h and w multiples of 14) -> patches (14 x 14 conv) + cls + positions -> N + 1 tokens
//!   12 x { x += ls1 * attn(ln1(x)); x += ls2 * mlp(ln2(x)) }   taps after layers 3, 6, 9, 12 through the final layer norm
//!   reassemble: tokens[1:] as a [gh, gw] map -> 1x1 conv (48, 96, 192, 384) -> ConvT 4 | ConvT 2 | - | conv 3x3 s2
//!   neck convs 3x3 -> 64;  fusion: deepest first, each [+ residual unit(skip)] -> residual unit -> resize -> 1x1
//!   head: conv 3x3 (32) -> resize (align corners) to h x w -> conv 3x3 (32) ReLU -> conv 1x1 (1) ReLU
//! ```
//!
//! The kernels are `kernels/image/depthanything`; with the silo (kernels/image/depthanything/silo) the attention runs as
//! float32 products on the shared linear (its kernels lay the heads out, take the softmax and merge them), else the engine's
//! own per-query kernel (about 15 times slower at 2,550 tokens). The checkpoint is read by the
//! foundation's safetensors reader.

use std::path::Path;
use std::sync::{Arc, Mutex};

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi::{self, check};

/// NS_PROFILE=1: each op class's time in a pass (the GPU synced after each op), on stderr after the pass
fn profiling() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("NS_PROFILE").is_ok_and(|v| v == "1"))
}

thread_local! {
    static PROF: std::cell::RefCell<Prof> = std::cell::RefCell::new((std::time::Instant::now(), Vec::new()));
}

pub const PATCH: usize = 14;
const DIM: usize = 384;
const HEADS: usize = 6;
const LAYERS: usize = 12;
const MLP: usize = 1536;
const TAPS: [usize; 4] = [3, 6, 9, 12];
const NECK: [usize; 4] = [48, 96, 192, 384];
const FUSION: usize = 64;
const HEAD: usize = 32;
/// the position table's own grid (518 / 14)
const GRID: usize = 37;
/// bytes of unfolded columns at a time
const COL_BYTES: usize = 128 << 20;

struct Lin {
    w: DevBuf,
    n: usize,
    k: usize,
}

struct Conv {
    w: DevBuf,
    b: Option<DevBuf>,
    co: usize,
    ci: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

/// A feature map on the card: float32 [h, w, c]
pub struct Map {
    pub buf: DevBuf,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

struct Layer {
    n1: (DevBuf, DevBuf),
    qkv: Lin,
    qkv_b: DevBuf,
    proj: Lin,
    proj_b: DevBuf,
    ls1: DevBuf,
    n2: (DevBuf, DevBuf),
    fc1: Lin,
    fc1_b: DevBuf,
    fc2: Lin,
    fc2_b: DevBuf,
    ls2: DevBuf,
}

struct Residual {
    c1: Conv,
    c2: Conv,
}

struct Fusion {
    r1: Residual,
    r2: Residual,
    proj: Conv,
}

/// ConvTranspose2d with kernel = stride: the GEMM's weight [k k Co, Ci] and its bias
struct ConvT {
    w: DevBuf,
    b: DevBuf,
    k: usize,
    co: usize,
    ci: usize,
}

enum Resize {
    Up(ConvT),
    Same,
    Down(Conv),
}

/// The position table made for a grid (gh, gw), kept while pictures of that grid come
type PosTable = ((usize, usize), Arc<DevBuf>);

/// What NS_PROFILE=1 has counted in a pass: when the last tick was, and (op class, seconds, count)
type Prof = (std::time::Instant, Vec<(&'static str, f64, usize)>);

pub struct DepthNet {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    patch: Conv,
    cls: DevBuf,
    pos_host: Vec<f32>,
    pos: Mutex<Option<PosTable>>,
    layers: Vec<Layer>,
    ln: (DevBuf, DevBuf),
    reassemble: Vec<(Conv, Resize)>,
    neck: Vec<Conv>,
    fusion: Vec<Fusion>,
    head: (Conv, Conv, Conv),
    col: DevBuf,
    pub bytes: usize,
}

fn dev(gpu: &Arc<Gpu>, v: &[f32], bytes: &mut usize) -> Result<DevBuf> {
    *bytes += v.len() * 4;
    DevBuf::from_f32(gpu, v)
}

/// PyTorch's `interpolate(mode="bicubic", align_corners=False)` of a [sh, sw, c] table to [oh, ow, c] (A = -0.75, taps
/// clamped to the table, the source coordinate not clamped at 0): the position embeddings at another grid
pub fn bicubic(src: &[f32], sh: usize, sw: usize, c: usize, oh: usize, ow: usize) -> Vec<f32> {
    const A: f32 = -0.75;
    let c1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let axis = |i: usize, n_in: usize, n_out: usize| -> ([usize; 4], [f32; 4]) {
        let s = (n_in as f32 / n_out as f32) * (i as f32 + 0.5) - 0.5;
        let f = s.floor();
        let t = s - f;
        let at = |d: i64| ((f as i64 + d).clamp(0, n_in as i64 - 1)) as usize;
        ([at(-1), at(0), at(1), at(2)], [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)])
    };
    let mut out = vec![0f32; oh * ow * c];
    for y in 0..oh {
        let (yi, yw) = axis(y, sh, oh);
        for x in 0..ow {
            let (xi, xw) = axis(x, sw, ow);
            for ch in 0..c {
                let mut col = [0f32; 4];
                for (r, cv) in col.iter_mut().enumerate() {
                    *cv = (0..4).map(|q| src[(yi[r] * sw + xi[q]) * c + ch] * xw[q]).sum();
                }
                out[(y * ow + x) * c + ch] = (0..4).map(|r| col[r] * yw[r]).sum();
            }
        }
    }
    out
}

impl DepthNet {
    pub fn load(path: &Path, gpu: &Arc<Gpu>) -> Result<DepthNet> {
        let st = SafeTensors::open(path).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
        let mut bytes = 0usize;
        let shape_of = |n: &str| -> Result<Vec<usize>> {
            Ok(st.need(n).map_err(|e| Error(e.0))?.shape.iter().map(|d| *d as usize).collect())
        };
        // the Small size only: the larger ones (Base, Large, Giant) are CC-BY-NC and other shapes
        if shape_of("backbone.embeddings.cls_token")? != [1, 1, DIM] || st.tensor(&format!("backbone.encoder.layer.{LAYERS}.norm1.weight")).is_some()
            || shape_of("neck.convs.0.weight")? != [FUSION, NECK[0], 3, 3] {
            return Err(Error(format!("{}: not Depth Anything V2 Small (384 wide, 12 layers, neck 48-96-192-384, 64 fusion)", path.display())));
        }
        let f = |n: &str| st.f32(n).map_err(|e| Error(format!("{n}: {}", e.0)));
        let lin = |n: &str, out: usize, inp: usize, bytes: &mut usize| -> Result<Lin> {
            let v = f(&format!("{n}.weight"))?;
            if v.len() != out * inp {
                return Err(Error(format!("{n}.weight: {} values, not {out} x {inp}", v.len())));
            }
            Ok(Lin { w: dev(gpu, &v, bytes)?, n: out, k: inp })
        };
        // [co, ci, kh, kw] -> [co, kh, kw, ci]: the im2col's (ky, kx, c) order
        let conv = |n: &str, co: usize, ci: usize, k: usize, stride: usize, pad: usize, bias: bool, bytes: &mut usize| -> Result<Conv> {
            let v = f(&format!("{n}.weight"))?;
            if v.len() != co * ci * k * k {
                return Err(Error(format!("{n}.weight: {} values, not {co} x {ci} x {k} x {k}", v.len())));
            }
            let mut r = vec![0f32; v.len()];
            for o in 0..co {
                for c in 0..ci {
                    for t in 0..k * k {
                        r[(o * k * k + t) * ci + c] = v[(o * ci + c) * k * k + t];
                    }
                }
            }
            let b = if bias { Some(dev(gpu, &f(&format!("{n}.bias"))?, bytes)?) } else { None };
            Ok(Conv { w: dev(gpu, &r, bytes)?, b, co, ci, k, stride, pad })
        };
        let patch = conv("backbone.embeddings.patch_embeddings.projection", DIM, 3, PATCH, PATCH, 0, true, &mut bytes)?;
        let cls = dev(gpu, &f("backbone.embeddings.cls_token")?, &mut bytes)?;
        let pos_host = f("backbone.embeddings.position_embeddings")?;
        if pos_host.len() != (GRID * GRID + 1) * DIM {
            return Err(Error("position_embeddings: not 1 + 37 x 37 tokens of 384".into()));
        }
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let p = format!("backbone.encoder.layer.{i}");
            let a = format!("{p}.attention.attention");
            // q, k, v as one matrix and one bias, in that order
            let cat = |part: &str| -> Result<Vec<f32>> {
                let mut v = Vec::new();
                for t in ["query", "key", "value"] {
                    v.extend(f(&format!("{a}.{t}.{part}"))?);
                }
                Ok(v)
            };
            let qkv_w = cat("weight")?;
            if qkv_w.len() != 3 * DIM * DIM {
                return Err(Error(format!("{a}: q, k, v are not 384 x 384")));
            }
            layers.push(Layer {
                n1: (dev(gpu, &f(&format!("{p}.norm1.weight"))?, &mut bytes)?, dev(gpu, &f(&format!("{p}.norm1.bias"))?, &mut bytes)?),
                qkv: Lin { w: dev(gpu, &qkv_w, &mut bytes)?, n: 3 * DIM, k: DIM },
                qkv_b: dev(gpu, &cat("bias")?, &mut bytes)?,
                proj: lin(&format!("{p}.attention.output.dense"), DIM, DIM, &mut bytes)?,
                proj_b: dev(gpu, &f(&format!("{p}.attention.output.dense.bias"))?, &mut bytes)?,
                ls1: dev(gpu, &f(&format!("{p}.layer_scale1.lambda1"))?, &mut bytes)?,
                n2: (dev(gpu, &f(&format!("{p}.norm2.weight"))?, &mut bytes)?, dev(gpu, &f(&format!("{p}.norm2.bias"))?, &mut bytes)?),
                fc1: lin(&format!("{p}.mlp.fc1"), MLP, DIM, &mut bytes)?,
                fc1_b: dev(gpu, &f(&format!("{p}.mlp.fc1.bias"))?, &mut bytes)?,
                fc2: lin(&format!("{p}.mlp.fc2"), DIM, MLP, &mut bytes)?,
                fc2_b: dev(gpu, &f(&format!("{p}.mlp.fc2.bias"))?, &mut bytes)?,
                ls2: dev(gpu, &f(&format!("{p}.layer_scale2.lambda1"))?, &mut bytes)?,
            });
        }
        let ln = (dev(gpu, &f("backbone.layernorm.weight")?, &mut bytes)?, dev(gpu, &f("backbone.layernorm.bias")?, &mut bytes)?);
        let mut reassemble = Vec::new();
        for (i, &ch) in NECK.iter().enumerate() {
            let p = format!("neck.reassemble_stage.layers.{i}");
            let proj = conv(&format!("{p}.projection"), ch, DIM, 1, 1, 0, true, &mut bytes)?;
            let rs = match i {
                0 | 1 => {
                    let k = [4, 2][i];
                    // ConvTranspose2d's weight [ci, co, ky, kx] -> [(ky, kx, co), ci]
                    let v = f(&format!("{p}.resize.weight"))?;
                    if v.len() != ch * ch * k * k {
                        return Err(Error(format!("{p}.resize.weight: not {ch} x {ch} x {k} x {k}")));
                    }
                    let mut r = vec![0f32; v.len()];
                    for ci in 0..ch {
                        for co in 0..ch {
                            for t in 0..k * k {
                                r[(t * ch + co) * ch + ci] = v[(ci * ch + co) * k * k + t];
                            }
                        }
                    }
                    Resize::Up(ConvT { w: dev(gpu, &r, &mut bytes)?, b: dev(gpu, &f(&format!("{p}.resize.bias"))?, &mut bytes)?, k, co: ch, ci: ch })
                }
                2 => Resize::Same,
                _ => Resize::Down(conv(&format!("{p}.resize"), ch, ch, 3, 2, 1, true, &mut bytes)?),
            };
            reassemble.push((proj, rs));
        }
        let neck = NECK.iter().enumerate().map(|(i, &ch)| conv(&format!("neck.convs.{i}"), FUSION, ch, 3, 1, 1, false, &mut bytes)).collect::<Result<Vec<_>>>()?;
        let mut fusion = Vec::new();
        for i in 0..4 {
            let p = format!("neck.fusion_stage.layers.{i}");
            let res = |r: &str, bytes: &mut usize| -> Result<Residual> {
                Ok(Residual { c1: conv(&format!("{p}.{r}.convolution1"), FUSION, FUSION, 3, 1, 1, true, bytes)?,
                              c2: conv(&format!("{p}.{r}.convolution2"), FUSION, FUSION, 3, 1, 1, true, bytes)? })
            };
            fusion.push(Fusion { r1: res("residual_layer1", &mut bytes)?, r2: res("residual_layer2", &mut bytes)?,
                                 proj: conv(&format!("{p}.projection"), FUSION, FUSION, 1, 1, 0, true, &mut bytes)? });
        }
        let head = (conv("head.conv1", HEAD, FUSION, 3, 1, 1, true, &mut bytes)?, conv("head.conv2", HEAD, HEAD, 3, 1, 1, true, &mut bytes)?,
                    conv("head.conv3", 1, HEAD, 1, 1, 0, true, &mut bytes)?);
        Ok(DepthNet { gpu: gpu.clone(), nsd: Nsd::new(gpu)?, patch, cls, pos_host, pos: Mutex::new(None), layers, ln, reassemble, neck, fusion, head,
                      col: DevBuf::new(gpu, COL_BYTES)?, bytes })
    }

    /// The position embeddings at a grid (cls first): the table itself at 37 x 37, else its patches interpolated
    fn positions(&self, gh: usize, gw: usize) -> Result<Arc<DevBuf>> {
        let mut g = self.pos.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((k, b)) = g.as_ref() {
            if *k == (gh, gw) {
                return Ok(b.clone());
            }
        }
        let table = if (gh, gw) == (GRID, GRID) {
            self.pos_host.clone()
        } else {
            let mut v = self.pos_host[..DIM].to_vec();
            v.extend(bicubic(&self.pos_host[DIM..], GRID, GRID, DIM, gh, gw));
            v
        };
        let b = Arc::new(DevBuf::from_f32(&self.gpu, &table)?);
        *g = Some(((gh, gw), b.clone()));
        Ok(b)
    }

    /// With NS_PROFILE=1: the GPU waited for, the time since the last tick counted under `what`
    fn tick(&self, what: &'static str) {
        if !profiling() {
            return;
        }
        let _ = self.nsd.wait();
        PROF.with(|p| {
            let mut p = p.borrow_mut();
            let s = p.0.elapsed().as_secs_f64();
            match p.1.iter_mut().find(|e| e.0 == what) {
                Some(e) => {
                    e.1 += s;
                    e.2 += 1;
                }
                None => p.1.push((what, s, 1)),
            }
            p.0 = std::time::Instant::now();
        });
    }

    fn conv(&self, c: &Conv, x: &Map, act: i32) -> Result<Map> {
        self.tick("other");
        let v = ffi::vision()?;
        let r = ffi::api()?;
        if c.ci != x.c {
            return Err(Error(format!("depth: a convolution takes {} channels, given {}", c.ci, x.c)));
        }
        let ho = (x.h + 2 * c.pad - c.k) / c.stride + 1;
        let wo = (x.w + 2 * c.pad - c.k) / c.stride + 1;
        let out = DevBuf::f32(&self.gpu, ho * wo * c.co)?;
        let g = self.gpu.raw();
        if c.k == 1 && c.stride == 1 {
            self.nsd.linear(x.buf.ptr(), Dt::F32, ho * wo, c.ci, c.w.ptr(), c.co, none(), out.ptr(), Dt::F32)?;
        } else {
            let kk = c.k * c.k * c.ci;
            let tile = (COL_BYTES / (kk * 4)).max(1);
            let mut p0 = 0;
            while p0 < ho * wo {
                self.tick("conv 1x1 / alloc");
                let n = tile.min(ho * wo - p0);
                // SAFETY: x holds h x w x c floats, col n x kk (COL_BYTES), out ho x wo x co.
                check(unsafe { (v.im2col)(g, x.buf.fp(), x.h as i64, x.w as i64, x.c as i64, c.k as i64, c.k as i64, c.stride as i64, c.pad as i64, 1,
                                          ho as i64, wo as i64, p0 as i64, n as i64, self.col.fp()) }, "depth im2col")?;
                // SAFETY: rows p0 .. p0 + n of out's ho * wo rows of co floats.
                let o = unsafe { out.fp().add(p0 * c.co) };
                self.tick("im2col");
                self.nsd.linear(self.col.ptr(), Dt::F32, n, kk, c.w.ptr(), c.co, none(), o.cast(), Dt::F32)?;
                self.tick("conv gemm");
                p0 += n;
            }
        }
        self.tick("conv 1x1 / alloc");
        if c.b.is_some() || act != 0 {
            // SAFETY: out holds ho * wo rows of co floats, the bias co.
            check(unsafe { (r.bias_act)(g, out.fp(), (ho * wo) as i64, c.co as i64, c.b.as_ref().map_or(std::ptr::null(), |b| b.fp()), act) }, "depth bias")?;
        }
        self.tick("conv bias");
        Ok(Map { buf: out, h: ho, w: wo, c: c.co })
    }

    fn act(&self, x: &Map, act: i32) -> Result<Map> {
        let out = DevBuf::f32(&self.gpu, x.h * x.w * x.c)?;
        // SAFETY: x and out hold h x w x c floats.
        check(unsafe { (ffi::api()?.act)(self.gpu.raw(), x.buf.fp(), (x.h * x.w * x.c) as i64, act, out.fp()) }, "depth activation")?;
        self.tick("act");
        Ok(Map { buf: out, h: x.h, w: x.w, c: x.c })
    }

    fn add(&self, x: &Map, y: &Map) -> Result<()> {
        // SAFETY: x and y hold the same number of floats.
        check(unsafe { (ffi::vision()?.add)(self.gpu.raw(), x.buf.fp(), y.buf.fp(), (x.h * x.w * x.c) as i64) }, "depth add")?;
        self.tick("add");
        Ok(())
    }

    /// bilinear resize: align_corners True (the fusions' upsampling, the head's) or False (a residual to a size)
    fn resize(&self, x: &Map, h: usize, w: usize, align: bool) -> Result<Map> {
        let out = DevBuf::f32(&self.gpu, h * w * x.c)?;
        let g = self.gpu.raw();
        // SAFETY: x holds its h x w x c floats, out the new size.
        let rc = unsafe {
            if align {
                (ffi::vision()?.resize_ac)(g, x.buf.fp(), x.h as i64, x.w as i64, x.c as i64, h as i64, w as i64, out.fp(), x.c as i64)
            } else {
                (ffi::api()?.resize)(g, x.buf.fp(), x.h as i64, x.w as i64, x.c as i64, h as i64, w as i64, out.fp(), x.c as i64)
            }
        };
        check(rc, "depth resize")?;
        self.tick(if align { "resize (align)" } else { "resize" });
        Ok(Map { buf: out, h, w, c: x.c })
    }

    /// pre-activation residual unit: x + conv2(relu(conv1(relu(x))))
    fn residual(&self, r: &Residual, x: &Map) -> Result<Map> {
        let a = self.act(x, 1)?;
        let a = self.conv(&r.c1, &a, 1)?;
        let b = self.conv(&r.c2, &a, 0)?;
        self.add(&b, x)?;
        Ok(b)
    }

    fn linear(&self, x: *const std::ffi::c_void, m: usize, l: &Lin, bias: &DevBuf, act: i32) -> Result<DevBuf> {
        let out = DevBuf::f32(&self.gpu, m * l.n)?;
        self.nsd.linear(x, Dt::F32, m, l.k, l.w.ptr(), l.n, none(), out.ptr(), Dt::F32)?;
        // SAFETY: out holds m x n floats, the bias n.
        check(unsafe { (ffi::api()?.bias_act)(self.gpu.raw(), out.fp(), m as i64, l.n as i64, bias.fp(), act) }, "depth bias")?;
        Ok(out)
    }

    fn norm(&self, x: &DevBuf, m: usize, (w, b): &(DevBuf, DevBuf)) -> Result<DevBuf> {
        let out = DevBuf::f32(&self.gpu, m * DIM)?;
        // SAFETY: x and out hold m x 384 floats, the weights 384.
        check(unsafe { (ffi::api()?.layernorm)(self.gpu.raw(), x.fp(), m as i64, DIM as i64, w.fp(), b.fp(), 1e-6, out.fp()) }, "depth layer norm")?;
        Ok(out)
    }

    /// The transformer: tokens [1 + N, 384] -> the four taps (through the final layer norm)
    fn backbone(&self, input: &DevBuf, h: usize, w: usize, want_emb: bool) -> Result<(Vec<DevBuf>, Option<DevBuf>)> {
        let a = ffi::api()?;
        let (gh, gw) = (h / PATCH, w / PATCH);
        let n = gh * gw;
        let m = n + 1;
        let g = self.gpu.raw();
        // the patch convolution without its bias (the embeddings add it)
        let v = ffi::vision()?;
        let kk = PATCH * PATCH * 3;
        let patches = DevBuf::f32(&self.gpu, n * DIM)?;
        let mut p0 = 0;
        let tile = (COL_BYTES / (kk * 4)).max(1);
        while p0 < n {
            let t = tile.min(n - p0);
            // SAFETY: input holds h x w x 3 floats, col t x kk, patches n x 384.
            check(unsafe { (v.im2col)(g, input.fp(), h as i64, w as i64, 3, PATCH as i64, PATCH as i64, PATCH as i64, 0, 1, gh as i64, gw as i64,
                                      p0 as i64, t as i64, self.col.fp()) }, "depth patches")?;
            // SAFETY: rows p0 .. p0 + t of patches.
            let o = unsafe { patches.fp().add(p0 * DIM) };
            self.nsd.linear(self.col.ptr(), Dt::F32, t, kk, self.patch.w.ptr(), DIM, none(), o.cast(), Dt::F32)?;
            p0 += t;
        }
        let pos = self.positions(gh, gw)?;
        let x = DevBuf::f32(&self.gpu, m * DIM)?;
        // SAFETY: patches n x 384, the bias, cls and the table (1 + n) x 384, x the same.
        check(unsafe { (a.embed)(g, patches.fp(), self.patch.b.as_ref().expect("patch bias").fp(), self.cls.fp(), pos.fp(), n as i64, DIM as i64, x.fp()) },
              "depth embeddings")?;
        // (the embeddings kept for the check alone: a copy on the card is slow)
        let emb = if want_emb { Some(copy(&self.gpu, &x, 0, m * DIM)?) } else { None };
        // the silo's attention (as products) when the library is there and takes this head size (NS_SILO=0: the engine's own)
        // SAFETY: a live GPU context.
        let silo = ffi::silo()?.filter(|sl| unsafe { (sl.supported)(g, (DIM / HEADS) as i64) } == 1);
        let mut taps = Vec::new();
        self.tick("patches");
        for (i, l) in self.layers.iter().enumerate() {
            let y = self.norm(&x, m, &l.n1)?;
            self.tick("norm");
            let qkv = self.linear(y.ptr(), m, &l.qkv, &l.qkv_b, 0)?;
            self.tick("qkv");
            let att = DevBuf::f32(&self.gpu, m * DIM)?;
            if let Some(sl) = silo {
                // the attention as float32 products on the shared linear, a head at a time: s = q k^T, its scaled
                // softmax, o = s v (v transposed by the split) - 46 ms a layer with the per-query kernel at 2,550
                // tokens on a B65, a few with the products
                let d = DIM / HEADS;
                let (q, k, vt, o) = (DevBuf::f32(&self.gpu, m * DIM)?, DevBuf::f32(&self.gpu, m * DIM)?, DevBuf::f32(&self.gpu, m * DIM)?, DevBuf::f32(&self.gpu, m * DIM)?);
                let s = DevBuf::f32(&self.gpu, m * m)?;
                // SAFETY: qkv m x 1152; q, k, vt, o m x 384 each; s m x m.
                check(unsafe { (sl.split)(g, qkv.fp(), m as i64, HEADS as i64, d as i64, q.fp(), k.fp(), vt.fp()) }, "depth attention split")?;
                for h in 0..HEADS {
                    // SAFETY: head h's block of m x 64 (q, k, o) or 64 x m (vt) floats.
                    let (qh, kh, vh, oh) = unsafe { (q.fp().add(h * m * d), k.fp().add(h * m * d), vt.fp().add(h * m * d), o.fp().add(h * m * d)) };
                    self.nsd.linear(qh.cast(), Dt::F32, m, d, kh.cast(), m, none(), s.ptr(), Dt::F32)?;
                    // SAFETY: s m x m.
                    check(unsafe { (sl.softmax)(g, s.fp(), m as i64, m as i64, 1.0 / (d as f32).sqrt()) }, "depth attention softmax")?;
                    self.nsd.linear(s.ptr(), Dt::F32, m, m, vh.cast(), d, none(), oh.cast(), Dt::F32)?;
                }
                // SAFETY: o 6 x m x 64, att m x 384.
                check(unsafe { (sl.merge)(g, o.fp(), m as i64, HEADS as i64, d as i64, att.fp()) }, "depth attention merge")?;
            } else {
                // SAFETY: qkv m x 1152, att m x 384.
                check(unsafe { (a.attention)(g, qkv.fp(), m as i64, HEADS as i64, (DIM / HEADS) as i64, att.fp()) }, "depth attention")?;
            }
            self.tick("attention");
            let o = DevBuf::f32(&self.gpu, m * DIM)?;
            self.nsd.linear(att.ptr(), Dt::F32, m, DIM, l.proj.w.ptr(), DIM, none(), o.ptr(), Dt::F32)?;
            // SAFETY: x and o m x 384, bias and scale 384.
            check(unsafe { (a.ls_add)(g, x.fp(), o.fp(), l.proj_b.fp(), l.ls1.fp(), m as i64, DIM as i64) }, "depth residual")?;
            let y = self.norm(&x, m, &l.n2)?;
            self.tick("proj + norm");
            let hdn = self.linear(y.ptr(), m, &l.fc1, &l.fc1_b, 2)?;
            self.tick("fc1");
            let o = DevBuf::f32(&self.gpu, m * DIM)?;
            self.nsd.linear(hdn.ptr(), Dt::F32, m, MLP, l.fc2.w.ptr(), DIM, none(), o.ptr(), Dt::F32)?;
            self.tick("fc2");
            // SAFETY: as above.
            check(unsafe { (a.ls_add)(g, x.fp(), o.fp(), l.fc2_b.fp(), l.ls2.fp(), m as i64, DIM as i64) }, "depth residual")?;
            if TAPS.contains(&(i + 1)) {
                taps.push(self.norm(&x, m, &self.ln)?);
            }
        }
        Ok((taps, emb))
    }

    /// The neck and head from the taps: the depth [h, w] (the head's ReLU) and the last fusion's map
    fn decode(&self, taps: &[DevBuf], h: usize, w: usize) -> Result<(Map, Map)> {
        let a = ffi::api()?;
        let (gh, gw) = (h / PATCH, w / PATCH);
        let g = self.gpu.raw();
        let mut feats = Vec::new();
        for (i, t) in taps.iter().enumerate() {
            // the tokens after the cls, as a [gh, gw, 384] map
            self.tick("other");
            // (a view of the tap past its cls row: no copy)
            let tokens = Map { buf: t.view(DIM * 4, gh * gw * DIM * 4)?, h: gh, w: gw, c: DIM };
            self.tick("tokens copy");
            let (proj, rs) = &self.reassemble[i];
            let p = self.conv(proj, &tokens, 0)?;
            let r = match rs {
                Resize::Same => p,
                Resize::Down(c) => self.conv(c, &p, 0)?,
                Resize::Up(t) => {
                    let y = DevBuf::f32(&self.gpu, p.h * p.w * t.k * t.k * t.co)?;
                    self.nsd.linear(p.buf.ptr(), Dt::F32, p.h * p.w, t.ci, t.w.ptr(), t.k * t.k * t.co, none(), y.ptr(), Dt::F32)?;
                    let out = DevBuf::f32(&self.gpu, p.h * t.k * p.w * t.k * t.co)?;
                    // SAFETY: y holds h w k k Co floats, out (h k) (w k) Co.
                    self.tick("convT gemm");
                    check(unsafe { (a.shuffle)(g, y.fp(), p.h as i64, p.w as i64, t.k as i64, t.co as i64, t.b.fp(), out.fp()) }, "depth transposed conv")?;
                    self.tick("convT shuffle");
                    Map { buf: out, h: p.h * t.k, w: p.w * t.k, c: t.co }
                }
            };
            feats.push(self.conv(&self.neck[i], &r, 0)?);
        }
        // fusion: the deepest first, each upsampled to the next one's size (the last by 2)
        let mut fused: Option<Map> = None;
        for idx in 0..4 {
            let hs = &feats[3 - idx];
            let l = &self.fusion[idx];
            let x = match fused.take() {
                None => self.residual(&l.r2, hs)?,
                Some(f) => {
                    let res = if (f.h, f.w) != (hs.h, hs.w) { Some(self.resize(hs, f.h, f.w, false)?) } else { None };
                    let r = self.residual(&l.r1, res.as_ref().unwrap_or(hs))?;
                    self.add(&f, &r)?;
                    self.residual(&l.r2, &f)?
                }
            };
            let (oh, ow) = if idx < 3 { (feats[2 - idx].h, feats[2 - idx].w) } else { (x.h * 2, x.w * 2) };
            let up = self.resize(&x, oh, ow, true)?;
            fused = Some(self.conv(&l.proj, &up, 0)?);
        }
        let last = fused.expect("four fusions");
        let c1 = self.conv(&self.head.0, &last, 0)?;
        let up = self.resize(&c1, h, w, true)?;
        let c2 = self.conv(&self.head.1, &up, 1)?;
        let depth = self.conv(&self.head.2, &c2, 1)?;
        Ok((depth, last))
    }

    /// The depth [h, w] for an input [h, w, 3] (ImageNet-normalized; h and w multiples of 14, at least 14)
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> Result<Map> {
        if h == 0 || w == 0 || !h.is_multiple_of(PATCH) || !w.is_multiple_of(PATCH) || input.len() != h * w * 3 {
            return Err(Error(format!("depth: an input of {} values for {w} x {h} (sides must be multiples of 14)", input.len())));
        }
        // NS_PROFILE=1: the pass by part on stderr, the GPU synced at each boundary
        let prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "1");
        let t0 = std::time::Instant::now();
        self.tick("before");
        let x = DevBuf::from_f32(&self.gpu, input)?;
        let (taps, _) = self.backbone(&x, h, w, false)?;
        if prof {
            self.nsd.wait()?;
            eprintln!("depth profile {w}x{h}: backbone {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let t1 = std::time::Instant::now();
        let (d, _) = self.decode(&taps, h, w)?;
        if prof {
            self.tick("other");
            self.nsd.wait()?;
            eprintln!("depth profile {w}x{h}: neck and head {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);
            PROF.with(|p| {
                for (k, s, n) in p.borrow_mut().1.drain(..) {
                    eprintln!("depth profile   {k:<18} {:8.1} ms in {n}", s * 1e3);
                }
            });
        }
        Ok(d)
    }

    /// Every stage, for the check: (embeddings, the four taps, the last fusion channels-last, the depth), as host floats
    #[allow(clippy::type_complexity)]
    pub fn stages(&self, input: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>)> {
        let x = DevBuf::from_f32(&self.gpu, input)?;
        let (taps, emb) = self.backbone(&x, h, w, true)?;
        let (d, last) = self.decode(&taps, h, w)?;
        self.nsd.wait()?;
        Ok((emb.expect("asked for").to_f32()?, taps.iter().map(DevBuf::to_f32).collect::<Result<_>>()?, last.buf.to_f32()?, d.buf.to_f32()?))
    }

    /// The depth map resized to out_w x out_h (bilinear, half-pixel: torch's interpolate), as host floats
    pub fn depth_at(&self, input: &[f32], h: usize, w: usize, out_w: usize, out_h: usize) -> Result<Vec<f32>> {
        let d = self.forward(input, h, w)?;
        let r = if (out_h, out_w) == (h, w) { d } else { self.resize(&d, out_h, out_w, false)? };
        self.nsd.wait()?;
        r.buf.to_f32()
    }
}

/// `n` floats of `src` from float `at` in a new buffer (a copy on the card)
fn copy(gpu: &Arc<Gpu>, src: &DevBuf, at: usize, n: usize) -> Result<DevBuf> {
    let out = DevBuf::f32(gpu, n)?;
    out.copy_2d(0, n * 4, src, at * 4, n * 4, n * 4, 1)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_of_a_constant_is_that_constant_and_same_size_is_identity() {
        let c = vec![0.7f32; 5 * 5 * 2];
        let r = bicubic(&c, 5, 5, 2, 3, 8);
        assert!(r.iter().all(|v| (v - 0.7).abs() < 1e-6));
        let ramp: Vec<f32> = (0..4 * 4).map(|i| i as f32).collect();
        let same = bicubic(&ramp, 4, 4, 1, 4, 4);
        assert!(same.iter().zip(&ramp).all(|(a, b)| (a - b).abs() < 1e-5), "{same:?}");
    }

    #[test]
    fn bicubic_matches_torchs_taps() {
        // torch 2.15: interpolate(x, size=(3, 5), mode="bicubic", align_corners=False) of the [1, 1, 2, 4] table
        // [[0, 1, 2, 3], [3, 1, 4, 1]] (A = -0.75, edge taps clamped, the source coordinate not clamped at 0: row 0, col 0 is -0.337)
        let want = [-0.336_987_1_f32, 0.557235, 1.4050566, 2.2933736, 3.2554553, 1.5303752, 0.919_000_6, 2.046875, 2.9415002, 1.9392499, 3.3977373, 1.2807661,
                    2.6886935, 3.5896273, 0.623_045_6];
        let got = bicubic(&[0.0, 1.0, 2.0, 3.0, 3.0, 1.0, 4.0, 1.0], 2, 4, 1, 3, 5);
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 2e-6, "{got:?}");
        }
    }
}
