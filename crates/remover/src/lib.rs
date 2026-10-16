//! nextsycl-remover: the background remover the 3D engines share - BiRefNet (ZhengPeng7/BiRefNet: a Swin-L backbone
//! run at two scales, a decoder of deformable ASPP blocks fed the picture's tiles) in float32 on the GPU, through
//! the shared vision kernels (`kernels/diffusion/vision.h`) and the diffusion kernels' GEMMs and norms.
//!
//! A picture in, its object's mask (8-bit, the picture's size) or the picture cut out (RGBA) back; and `prepare`,
//! the object picture an image-to-3D model wants (TRELLIS.2's preprocessing): at most 1024 on its long side, cut out
//! when it has no alpha of its own, cropped to its alpha > 0.8 box, squared, premultiplied on black.

mod ffi;
mod net;
pub mod prep;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{Error, Gpu, Result};
use nextsycl_diffusion::Picture;
use nextsycl_gguf::safetensors::SafeTensors;

pub use net::Probe;
pub use prep::{has_alpha, prepare, prepare_margin};

/// The side BiRefNet sees
pub const SIDE: usize = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct Remover {
    net: net::BiRefNet,
    pub load_seconds: f64,
}

impl Remover {
    /// BiRefNet's weights: `model.safetensors`, or a directory holding it
    pub fn load(path: &Path, gpu: &Arc<Gpu>) -> Result<Remover> {
        let t0 = Instant::now();
        let file = if path.is_dir() { path.join("model.safetensors") } else { path.to_path_buf() };
        let st = SafeTensors::open(&file).map_err(|e| Error(e.0))?;
        if st.tensor("bb.layers.0.blocks.0.attn.qkv.weight").is_none() || st.tensor("decoder.conv_out1.0.weight").is_none() {
            return Err(Error(format!("{}: not a BiRefNet (Swin-L) file", file.display())));
        }
        let net = net::BiRefNet::load(&st, gpu)?;
        gpu.sync()?;
        Ok(Remover { net, load_seconds: t0.elapsed().as_secs_f64() })
    }

    /// The network's input for a picture: 1024 x 1024 (PIL's bilinear, as torchvision's Resize of a PIL picture), in
    /// [0, 1], ImageNet-normalized, channels-last
    pub fn input(pic: &Picture) -> Vec<f32> {
        let rgb = pic.rgb();
        let r = rgb.resize_bilinear(SIDE as u32, SIDE as u32);
        r.data.chunks_exact(3).flat_map(|p| (0..3).map(move |c| (p[c] as f32 / 255.0 - MEAN[c]) / STD[c])).collect()
    }

    /// The prediction's logits [1024 * 1024] for an `input`, and with `probe` the encoder's features
    pub fn logits(&self, input: &[f32], probe: bool) -> Result<(Vec<f32>, Option<Probe>)> {
        if input.len() != SIDE * SIDE * 3 {
            return Err(Error(format!("the remover's input: {} values, not 1024 x 1024 x 3", input.len())));
        }
        let x = net::Map { buf: nextsycl_core::DevBuf::from_f32(&self.net.gpu, input)?, h: SIDE, w: SIDE, c: 3 };
        self.net.forward(&x, probe)
    }

    /// The object's mask: 8 bits a pixel at the picture's size (the sigmoid as an 8-bit picture, truncated as
    /// torchvision's ToPILImage does, resized back with PIL's bicubic)
    pub fn mask(&self, pic: &Picture) -> Result<Picture> {
        let (l, _) = self.logits(&Self::input(pic), false)?;
        let m = Picture { width: SIDE as u32, height: SIDE as u32, channels: 1, data: l.iter().map(|v| (255.0f32 * (1.0 / (1.0 + (-v).exp()))) as u8).collect() };
        Ok(m.resize_bicubic(pic.width, pic.height))
    }

    /// The picture with its background cut out: RGBA, the mask as alpha
    pub fn cut_out(&self, pic: &Picture) -> Result<Picture> {
        let m = self.mask(pic)?;
        let rgb = pic.rgb();
        Ok(Picture { width: pic.width, height: pic.height, channels: 4,
                     data: rgb.data.chunks_exact(3).zip(&m.data).flat_map(|(p, a)| [p[0], p[1], p[2], *a]).collect() })
    }
}
