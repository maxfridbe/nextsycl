//! The upscalers in Rust and SYCL: Real-ESRGAN (xinntao: realesrgan-x4plus, realesrgan-x4plus-anime,
//! realesr-animevideov3 at 2x, 3x, 4x) and waifu2x (nihui's cunet, upconv_7 anime / photo, denoise -1 to 3, 1x to 32x),
//! their ncnn graphs and weights - the files realesrgan-ncnn-vulkan and waifu2x-ncnn-vulkan ship - run on the shared
//! convolutions (nsd: strided im2col + GEMM) and this crate's kernels (kernels/image/upscale). No Vulkan, no ncnn.
//!
//! ```text
//!   picture -> tiles of `tile` pixels, each read with its prepadding around it (Real-ESRGAN: 10, mirrored at the
//!   picture's edges; waifu2x: 7 / 18 / 28, edge pixels repeated, the far side rounded up for cunet's 2x2 strides)
//!   -> the graph -> the tile's part of the output (Real-ESRGAN's cut from the prepadding, waifu2x's from the corner)
//!   alpha: the tile's own bicubic (ncnn's Interp 3), as both programs do; waifu2x 4x and up: its 2x model again
//! ```
//!
//! Every choice is the two programs' own (realesrgan-ncnn-vulkan's and waifu2x-ncnn-vulkan's main.cpp, its .comp
//! shaders), so their output is the reference: `nextsycl image upscale-check` measures against it.

pub mod ffi;
pub mod ncnn;
pub mod net;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step,
    UpscaleInfo, UpscaleRequest, UpscaleVariant};

use crate::net::{Net, Pool, Shape};

pub const ARCH_ESRGAN: &str = "realesrgan";
pub const ARCH_WAIFU2X: &str = "waifu2x";

pub fn kind() -> ImageKind {
    ImageKind {
        archs: &[ARCH_ESRGAN, ARCH_WAIFU2X],
        name: "Real-ESRGAN and waifu2x upscalers (their ncnn graphs and weights, Rust + SYCL)",
        roles: &[],
        load,
        options: &[],
    }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(Upscaler::load(files, gpus, log)?))
}

const ESRGAN: &[UpscaleVariant] = &[
    UpscaleVariant { id: "realesrgan-x4plus", about: "photographs and general pictures (RRDB, 4x)", scales: &[4], noise: &[] },
    UpscaleVariant { id: "realesrgan-x4plus-anime", about: "illustration and anime (RRDB 6 blocks, 4x)", scales: &[4], noise: &[] },
    UpscaleVariant { id: "realesr-animevideov3", about: "video frames, fast (SRVGG compact; 2x, 3x, 4x)", scales: &[2, 3, 4], noise: &[] },
];

const WAIFU2X: &[UpscaleVariant] = &[
    UpscaleVariant { id: "models-cunet", about: "illustration (cunet); denoise -1 to 3", scales: &[1, 2, 4, 8, 16, 32], noise: &[-1, 0, 1, 2, 3] },
    UpscaleVariant { id: "models-upconv_7_anime_style_art_rgb", about: "anime-style art (upconv_7)", scales: &[2, 4, 8, 16, 32], noise: &[-1, 0, 1, 2, 3] },
    UpscaleVariant { id: "models-upconv_7_photo", about: "photographs (upconv_7)", scales: &[2, 4, 8, 16, 32], noise: &[-1, 0, 1, 2, 3] },
];

/// One pass of a model: its graph's files (the role names the card gives them), the factor, the prepadding, the border
struct Plan {
    /// the role of the .param (the .bin's is the same with .bin)
    stem: String,
    scale: usize,
    prepad: usize,
    /// 0: mirrored borders (Real-ESRGAN), 1: repeated (waifu2x)
    border: i32,
    /// how many times the model runs (waifu2x 4x: twice its 2x model)
    passes: usize,
    variant: String,
    noise: Option<i32>,
    tile: usize,
}

struct State {
    nets: HashMap<String, Arc<Net>>,
    pool: Pool,
}

pub struct Upscaler {
    arch: &'static str,
    gpu: Arc<Gpu>,
    nsd: Nsd,
    files: ModelFiles,
    state: Mutex<State>,
    load_secs: f64,
}

impl Upscaler {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<Upscaler> {
        let t0 = Instant::now();
        let gpu = gpus.first().cloned().ok_or_else(|| Error("an upscaler needs a GPU".into()))?;
        let arch = if files.keys().any(|k| k.starts_with("models-")) {
            ARCH_WAIFU2X
        } else if files.keys().any(|k| k.contains("realesr")) {
            ARCH_ESRGAN
        } else {
            return Err(Error(format!("no upscaler's files among {:?} (realesrgan-*.param or models-*/...)", files.keys().collect::<Vec<_>>())));
        };
        let nsd = Nsd::new(&gpu)?;
        ffi::api()?;
        let up = Upscaler { arch, gpu: gpu.clone(), nsd, files: files.clone(), state: Mutex::new(State { nets: HashMap::new(), pool: Pool::new(&gpu) }), load_secs: 0.0 };
        // the default variant's graph now (the others load when first asked: they are small)
        let default = if arch == ARCH_ESRGAN { "realesrgan-x4plus" } else { "models-cunet/noise1_scale2.0x_model" };
        if up.files.contains_key(&format!("{default}.param")) {
            up.net(default)?;
        }
        log(format!("{arch}: {} graph files on {}", up.files.len() / 2, gpu.name));
        Ok(Upscaler { load_secs: t0.elapsed().as_secs_f64(), ..up })
    }

    fn path(&self, role: &str) -> Result<PathBuf> {
        self.files.get(role).cloned().ok_or_else(|| Error(format!("{role}: not among this model's files (the card lists them)")))
    }

    fn net(&self, stem: &str) -> Result<Arc<Net>> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(n) = st.nets.get(stem) {
            return Ok(n.clone());
        }
        let param = std::fs::read_to_string(self.path(&format!("{stem}.param"))?).map_err(|e| Error(format!("{stem}.param: {e}")))?;
        let bin = std::fs::read(self.path(&format!("{stem}.bin"))?).map_err(|e| Error(format!("{stem}.bin: {e}")))?;
        let g = ncnn::parse_param(&param).map_err(|e| Error(format!("{stem}.param: {e}")))?;
        let n = Arc::new(Net::load(&self.gpu, &g, &bin, stem)?);
        st.nets.insert(stem.to_string(), n.clone());
        Ok(n)
    }

    /// What a request runs, as the programs choose it
    fn plan(&self, req: &UpscaleRequest) -> Result<Plan> {
        if self.arch == ARCH_ESRGAN {
            let scale = req.scale.unwrap_or(4) as usize;
            let variant = req.variant.clone().filter(|v| !v.is_empty())
                .unwrap_or_else(|| if scale == 2 || scale == 3 { "realesr-animevideov3".into() } else { "realesrgan-x4plus".into() });
            let v = ESRGAN.iter().find(|x| x.id == variant).ok_or_else(|| Error(format!("{variant}: Real-ESRGAN has {}", names(ESRGAN))))?;
            if !v.scales.contains(&(scale as u32)) {
                return Err(Error(format!("{variant} makes {:?}x, not {scale}x", v.scales)));
            }
            let stem = if variant == "realesr-animevideov3" { format!("{variant}-x{scale}") } else { variant.clone() };
            return Ok(Plan { stem, scale, prepad: 10, border: 0, passes: 1, variant, noise: None, tile: req.tile.unwrap_or(200).max(32) as usize });
        }
        let variant = req.variant.clone().filter(|v| !v.is_empty()).unwrap_or_else(|| "models-cunet".into());
        let v = WAIFU2X.iter().find(|x| x.id == variant).ok_or_else(|| Error(format!("{variant}: waifu2x has {}", names(WAIFU2X))))?;
        let scale = req.scale.unwrap_or(2);
        let noise = req.noise.unwrap_or(0);
        if !v.scales.contains(&scale) {
            return Err(Error(format!("{variant} makes {:?}x, not {scale}x", v.scales)));
        }
        if !v.noise.contains(&noise) {
            return Err(Error(format!("denoise {noise}: -1 (none) to 3")));
        }
        if scale == 1 && noise < 0 {
            return Err(Error("1x without denoise changes nothing: ask for a denoise level (0-3) or a scale".into()));
        }
        let cunet = variant == "models-cunet";
        // waifu2x-ncnn-vulkan's main.cpp: the model file and its prepadding
        let (stem, prepad) = if noise == -1 {
            ("scale2.0x_model".to_string(), if cunet { 18 } else { 7 })
        } else if scale == 1 {
            (format!("noise{noise}_model"), 28)
        } else {
            (format!("noise{noise}_scale2.0x_model"), if cunet { 18 } else { 7 })
        };
        let passes = if scale == 1 { 1 } else { scale.trailing_zeros() as usize };
        Ok(Plan { stem: format!("{variant}/{stem}"), scale: if scale == 1 { 1 } else { 2 }, prepad, border: 1, passes, variant, noise: Some(noise),
                  tile: req.tile.unwrap_or(400).max(32) as usize })
    }

    /// One pass over the picture on the card (u8 [h, w, ch]): the picture `scale` times bigger, on the card
    #[allow(clippy::too_many_arguments)]
    fn pass(&self, net: &Net, p: &Plan, src: &DevBuf, w: usize, h: usize, ch: usize, done: &mut u32, of: u32, t0: Instant,
            progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<DevBuf> {
        let k = ffi::api()?;
        let g = ffi::g(&self.gpu);
        let s = p.scale;
        let (ow, oh) = (w * s, h * s);
        let dst = DevBuf::new(&self.gpu, ow * oh * ch)?;
        let t = p.tile;
        let (xt, yt) = (w.div_ceil(t), h.div_ceil(t));
        let mut st = self.state.lock().unwrap_or_else(|q| q.into_inner());
        for yi in 0..yt {
            for xi in 0..xt {
                let tw = ((xi + 1) * t).min(w) - xi * t;
                let th = ((yi + 1) * t).min(h) - yi * t;
                // waifu2x's far side rounded up: its cunet halves the tile twice (scale 1) or once (scale 2)
                let (mut pr, mut pb) = (p.prepad, p.prepad);
                if p.border == 1 {
                    let m = if s == 1 { 4 } else { 2 };
                    pr += tw.div_ceil(m) * m - tw;
                    pb += th.div_ceil(m) * m - th;
                }
                let (iw, ih) = (tw + p.prepad + pr, th + p.prepad + pb);
                let x0 = (xi * t) as i64 - p.prepad as i64;
                let y0 = (yi * t) as i64 - p.prepad as i64;
                let input = st.pool.get(iw * ih * 3 * 2)?;
                // SAFETY: src holds h x w x ch bytes, input iw x ih x 3 halves.
                ffi::check(unsafe { (k.pre)(g, src.ptr() as *const u8, h as i64, w as i64, ch as i64, x0, y0, iw as i64, ih as i64, p.border, input.ptr()) },
                           "the tile")?;
                let (out, sh) = net.run(&self.gpu, &self.nsd, &mut st.pool, input, Shape { h: ih, w: iw, c: 3, vec: false })?;
                // where the tile's own pixels are in the output: Real-ESRGAN's past its prepadding, waifu2x's at the corner
                let (cx, cy) = if p.border == 0 { (p.prepad * s, p.prepad * s) } else { (0, 0) };
                if sh.c != 3 || sh.w < cx + tw * s || sh.h < cy + th * s {
                    return Err(Error(format!("{}: a {}x{}x{} output for a {iw}x{ih} tile (expected {}x{} from ({cx}, {cy}))", net.name, sh.w, sh.h, sh.c,
                                             tw * s, th * s)));
                }
                let alpha = if ch == 4 {
                    let a = st.pool.get(tw * th * 2)?;
                    // SAFETY: as above, a tw x th halves.
                    ffi::check(unsafe { (k.pre_alpha)(g, src.ptr() as *const u8, h as i64, w as i64, (xi * t) as i64, (yi * t) as i64, tw as i64, th as i64,
                                                      a.ptr()) }, "the tile's alpha")?;
                    if s == 1 {
                        Some(a)
                    } else {
                        let big = st.pool.get(tw * s * th * s * 2)?;
                        // SAFETY: as above.
                        ffi::check(unsafe { (k.resize)(g, a.ptr(), th as i64, tw as i64, 1, (th * s) as i64, (tw * s) as i64, 3, big.ptr()) }, "the alpha's bicubic")?;
                        st.pool.put(a);
                        Some(big)
                    }
                } else {
                    None
                };
                let ap = alpha.as_ref().map(|a| a.ptr() as *const _).unwrap_or(std::ptr::null());
                // SAFETY: out holds sh's halves, dst ow x oh x ch bytes.
                ffi::check(unsafe { (k.post)(g, out.ptr(), sh.w as i64, cx as i64, cy as i64, ap, dst.ptr() as *mut u8, ow as i64, ch as i64, (xi * t * s) as i64,
                                             (yi * t * s) as i64, (tw * s) as i64, (th * s) as i64) }, "the tile's pixels")?;
                st.pool.put(out);
                if let Some(a) = alpha {
                    st.pool.put(a);
                }
                *done += 1;
                drop(st);
                self.gpu.sync()?;
                progress(Step { picture: 0, at: *done, of, seconds: t0.elapsed().as_secs_f64() })?;
                st = self.state.lock().unwrap_or_else(|q| q.into_inner());
            }
        }
        Ok(dst)
    }
}

fn names(v: &[UpscaleVariant]) -> String {
    v.iter().map(|x| x.id).collect::<Vec<_>>().join(", ")
}

impl ImageEngine for Upscaler {
    fn arch(&self) -> &'static str {
        self.arch
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 0, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 0.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(&self, _req: &ImageRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error(format!("{} upscales pictures (image.upscale); it does not make them", self.arch)))
    }
    fn report(&self) -> Vec<String> {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        vec![format!("{} on {}: {} graph(s) loaded, {:.0} MB of buffers kept", self.arch, self.gpu.name, st.nets.len(), st.pool.held() as f64 / 1e6)]
    }
    fn upscale_variants(&self) -> Vec<UpscaleVariant> {
        if self.arch == ARCH_ESRGAN { ESRGAN.to_vec() } else { WAIFU2X.to_vec() }
    }
    fn upscale(&self, req: &UpscaleRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Picture, UpscaleInfo)> {
        let t0 = Instant::now();
        let p = self.plan(req)?;
        let net = self.net(&p.stem)?;
        let pic = &req.picture;
        let ch = pic.channels as usize;
        if !(ch == 3 || ch == 4) {
            return Err(Error(format!("a picture of {ch} channels: RGB or RGBA")));
        }
        let (mut w, mut h) = (pic.width as usize, pic.height as usize);
        if w == 0 || h == 0 {
            return Err(Error("an empty picture".into()));
        }
        let mut src = DevBuf::new(&self.gpu, pic.data.len())?;
        src.write(0, &pic.data)?;
        // the tiles of every pass, for the progress
        let mut of = 0u32;
        {
            let (mut a, mut b) = (w, h);
            for _ in 0..p.passes {
                of += (a.div_ceil(p.tile) * b.div_ceil(p.tile)) as u32;
                a *= p.scale;
                b *= p.scale;
            }
        }
        let mut done = 0u32;
        for _ in 0..p.passes {
            let out = self.pass(&net, &p, &src, w, h, ch, &mut done, of, t0, progress)?;
            src = out;
            w *= p.scale;
            h *= p.scale;
        }
        let mut data = vec![0u8; w * h * ch];
        src.read(0, &mut data)?;
        // the buffers stay for the next picture of this size; a big one should not keep a card's memory for good
        {
            let mut st = self.state.lock().unwrap_or_else(|q| q.into_inner());
            if st.pool.held() > (2usize << 30) {
                st.pool.clear();
            }
        }
        let total = p.scale.pow(p.passes as u32) as u32;
        Ok((Picture { width: w as u32, height: h as u32, channels: ch as u8, data },
            UpscaleInfo { variant: p.variant, scale: total, noise: p.noise, tile: p.tile as u32, tiles: done, seconds: t0.elapsed().as_secs_f64() }))
    }
}
