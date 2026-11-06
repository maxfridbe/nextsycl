//! The tiny conditional decoder (FlashVSR's `TCDecoder`: TAEHV's decoder, widened to [512, 256, 128, 128] and
//! conditioned on the low-quality frames): latents to frames, 4 a latent frame, with no VAE.
//!
//! ```text
//!   input  [h8, w8, 784]: the 4 cond frames 8 x 8-unshuffled (768) and the latent (16), then 3 tanh(x / 3)
//!   1 conv 784->512, ReLU, 3 conv 512->512, ReLU, 5-7 MemBlock 512, 9 TGrow x1 (1x1 512->512)
//!   10 up 2x, conv 512->256, 11-13 MemBlock 256, 15 TGrow x2 (1x1 256->512: two frames)
//!   16 up 2x, conv 256->128, 17-19 MemBlock 128, 21 TGrow x2 (1x1 128->256: two frames)
//!   22 up 2x, conv 128->128, ReLU, 24 conv 128->128, ReLU, 26 conv 128->3   -> [H, W, 3] in about [0, 1]
//!   MemBlock(x, past) = ReLU(conv(ReLU(conv(ReLU(conv([x, past])))) + x)), past = the block's input one frame before
//! ```
//!
//! Frames go through one at a time, in order (the reference's sequential mode), each block's past kept. A TGrow's 1x1
//! convolution is done before the nearest upsample in front of the next convolution (they commute). A MemBlock's first
//! convolution over [x, past] is two (its weights split by input), the second summed onto the first, so the two are
//! never copied side by side; its last convolution sums onto x in place (oneDNN's sum post-op), the ReLU after it fused.
//! The full-size end (the last upsample and three convolutions, 128 channels at the frame's size: 0.9 GB a buffer at
//! 2560 x 1408) goes a band of rows at a time (`BAND`), each band 3 rows longer on both sides inside the picture - each
//! convolution's border reaches a row further in - and only its own rows kept: the same frame, a fraction of the memory.
//! The stream's first 3 frames are dropped (the reference trims them).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::dit::{Act, Lin};
use crate::ffi::{self, check, g, half_bytes};
use crate::prof;

/// The first conv's input channels: 768 of condition, 16 of latent
const IN_CH: usize = 784;
/// Rows of the full-size end made at a time (and the 3 its three 3 x 3 convolutions reach into on each side)
const BAND: usize = 256;
const HALO: usize = 3;

/// The bands of a frame `h` rows high: (first row, rows) of each, its rows with the halo (first, rows)
pub fn bands(h: usize, band: usize) -> Vec<((usize, usize), (usize, usize))> {
    (0..h).step_by(band.max(1)).map(|y0| {
        let y1 = (y0 + band).min(h);
        let (a, b) = (y0.saturating_sub(HALO), (y1 + HALO).min(h));
        ((y0, y1 - y0), (a, b - a))
    }).collect()
}

/// A 2-D convolution: weights half [co, ci, k, k] (the checkpoint's layout), bias float32 [co] or none
struct Conv {
    /// its place, for the profile: the size it runs at
    what: &'static str,
    w: DevBuf,
    b: Option<DevBuf>,
    co: usize,
    ci: usize,
    k: usize,
}

impl Conv {
    /// The conv `name` (its input channels [c0, c0 + ci) of the checkpoint's: a MemBlock's first conv split in two)
    fn load(gpu: &Arc<Gpu>, st: &SafeTensors, name: &str, bias: bool, part: Option<(usize, usize)>) -> Result<Conv> {
        let t = st.need(&format!("{name}.weight")).map_err(|e| Error(e.0))?;
        if t.shape.len() != 4 || t.shape[2] != t.shape[3] {
            return Err(Error(format!("{name}.weight: {:?}, not a square 2-D convolution", t.shape)));
        }
        let (co, cin, k) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
        let w = st.f32(&format!("{name}.weight")).map_err(|e| Error(e.0))?;
        let (c0, ci) = part.unwrap_or((0, cin));
        if c0 + ci > cin {
            return Err(Error(format!("{name}: input channels {c0}..{} of {cin}", c0 + ci)));
        }
        let kk = k * k;
        let mut r = Vec::with_capacity(co * ci * kk);
        for o in 0..co {
            r.extend_from_slice(&w[(o * cin + c0) * kk..(o * cin + c0 + ci) * kk]);
        }
        let wd = DevBuf::new(gpu, r.len() * 2)?;
        wd.write(0, &half_bytes(&r))?;
        let b = if bias { Some(DevBuf::from_f32(gpu, &st.f32(&format!("{name}.bias")).map_err(|e| Error(e.0))?)?) } else { None };
        // the decoder's layer number says its size (decoder.N...: 1-7 at 1/8, 10-13 at 1/4, 16-19 at 1/2, 22-26 full)
        let layer: usize = name.split('.').nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        let what = match layer {
            0..=9 => "decoder: convolutions at 1/8",
            10..=15 => "decoder: convolutions at 1/4",
            16..=21 => "decoder: convolutions at 1/2",
            26 => "decoder: the last convolution (3 out)",
            _ => "decoder: convolutions at full size",
        };
        Ok(Conv { what, w: wd, b, co, ci, k })
    }

    fn bytes(&self) -> usize {
        self.co * self.ci * self.k * self.k * 2 + self.b.as_ref().map_or(0, |_| self.co * 4)
    }

    /// out [h, w, co] = act(conv(x [h, w, ci], pixels `ldx` apart) + bias (+ out's own values: accum)), zero padding
    #[allow(clippy::too_many_arguments)]
    fn run(&self, gpu: &Arc<Gpu>, x: *const std::ffi::c_void, ldx: usize, h: usize, w: usize, accum: bool, relu: bool, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        let px = h * w;
        let flops = 2.0 * (px * self.co * self.ci * self.k * self.k) as f64;
        prof::timed(gpu, self.what, flops, ((px * (self.ci + self.co * if accum { 2 } else { 1 })) * 2 + self.bytes()) as f64, || {
            let b = self.b.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
            // SAFETY: x holds h x w pixels of ldx halves, out h x w x co halves, the weights co x ci x k x k.
            check(unsafe { (k.conv)(g(gpu), x, 1, h as i64, w as i64, self.ci as i64, ldx as i64, self.w.ptr(), self.co as i64, self.k as i64,
                                    (self.k / 2) as i64, b, accum as i32, relu as i32, out.ptr()) }, "decoder conv")
        })
    }
}

/// A MemBlock: its first convolution in two (over x, over the past), the second and the third
struct Mem {
    c1x: Conv,
    c1p: Conv,
    c2: Conv,
    c3: Conv,
}

pub struct Tcd {
    gpu: Arc<Gpu>,
    c1: Conv,
    c3: Conv,
    mb: Vec<Mem>,
    tg9: Lin,
    c10: Conv,
    tg15: Lin,
    c16: Conv,
    /// TGrow 21's two frames as two products ([128, 128] each), a frame's channels made when its turn comes
    tg21: [Lin; 2],
    c22: Conv,
    c24: Conv,
    c26: Conv,
    pub bytes: usize,
    state: Option<State>,
}

struct State {
    h8: usize,
    w8: usize,
    /// each MemBlock's past (its input one frame before), and whether it has one yet
    past: Vec<DevBuf>,
    have: Vec<bool>,
    trim: usize,
    inp: DevBuf,
    /// a frame's working buffers, the largest each is asked to hold
    x: DevBuf,
    t1: DevBuf,
    t2: DevBuf,
    tg: DevBuf,
    /// TGrow 15's two frames [p4, 512], read by both halves of what follows
    tg2: DevBuf,
    up: DevBuf,
    /// a band's output [BAND + 2 HALO, W, 3], its own rows then copied into `out`
    ob: DevBuf,
    pub out: DevBuf,
}

/// A TGrow's 1x1 convolution as a linear layer: weight [co, ci, 1, 1] -> [co, ci], no bias
fn grow(gpu: &Arc<Gpu>, st: &SafeTensors, name: &str) -> Result<Lin> {
    let t = st.need(&format!("{name}.weight")).map_err(|e| Error(e.0))?;
    let (co, ci) = (t.shape[0] as usize, t.shape[1] as usize);
    Lin::from(gpu, &st.f32(&format!("{name}.weight")).map_err(|e| Error(e.0))?, None, co, ci)
}

impl Tcd {
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>) -> Result<Tcd> {
        let c = |n: &str, b: bool| Conv::load(gpu, st, &format!("decoder.{n}"), b, None);
        let mem = |i: usize, ch: usize| -> Result<Mem> {
            Ok(Mem { c1x: Conv::load(gpu, st, &format!("decoder.{i}.conv.0"), true, Some((0, ch)))?,
                     c1p: Conv::load(gpu, st, &format!("decoder.{i}.conv.0"), false, Some((ch, ch)))?,
                     c2: c(&format!("{i}.conv.2"), true)?, c3: c(&format!("{i}.conv.4"), true)? })
        };
        let mut mb = Vec::new();
        for (i, ch) in [(5, 512), (6, 512), (7, 512), (11, 256), (12, 256), (13, 256), (17, 128), (18, 128), (19, 128)] {
            mb.push(mem(i, ch)?);
        }
        let t = Tcd {
            gpu: gpu.clone(),
            c1: c("1", true)?,
            c3: c("3", false)?,
            mb,
            tg9: grow(gpu, st, "decoder.9.conv")?,
            c10: c("10", false)?,
            tg15: grow(gpu, st, "decoder.15.conv")?,
            c16: c("16", false)?,
            tg21: {
                let t = st.need("decoder.21.conv.weight").map_err(|e| Error(e.0))?;
                let (co, ci) = (t.shape[0] as usize, t.shape[1] as usize);
                if co != 256 || ci != 128 {
                    return Err(Error(format!("flashvsr decoder: TGrow 21 is {co} x {ci}, 256 x 128 expected")));
                }
                let w = st.f32("decoder.21.conv.weight").map_err(|e| Error(e.0))?;
                [Lin::from(gpu, &w[..128 * 128], None, 128, 128)?, Lin::from(gpu, &w[128 * 128..], None, 128, 128)?]
            },
            c22: c("22", false)?,
            c24: c("24", false)?,
            c26: c("26", true)?,
            bytes: 0,
            state: None,
        };
        if t.c1.ci != IN_CH || t.c26.co != 3 || t.tg15.n != 512 {
            return Err(Error("flashvsr decoder: not the TCDecoder of FlashVSR v1.1 (channels [512, 256, 128, 128], 784 in)".into()));
        }
        let mut bytes = [&t.c1, &t.c3, &t.c10, &t.c16, &t.c22, &t.c24, &t.c26].iter().map(|c| c.bytes()).sum::<usize>()
            + t.tg9.bytes() + t.tg15.bytes() + t.tg21[0].bytes() + t.tg21[1].bytes();
        for m in &t.mb {
            bytes += m.c1x.bytes() + m.c1p.bytes() + m.c2.bytes() + m.c3.bytes();
        }
        Ok(Tcd { bytes, ..t })
    }

    /// The memory a stream of [h8, w8] latents takes (bytes)
    pub fn need_bytes(h8: usize, w8: usize) -> usize {
        let p8 = h8 * w8;
        // pasts: 3 x (512 at 1/8, 256 at 1/4, 128 at 1/2); the working buffers: half the size's 128 channels (x, t1, t2),
        // its upsampled input of 256, or a band of the full size's
        let past = 3 * (p8 * 512 + 4 * p8 * 256 + 16 * p8 * 128) * 2;
        let (half, band, up) = Self::sizes(h8, w8);
        past + 3 * half.max(band) + up + 16 * p8 * 128 * 2 + 4 * p8 * 512 * 2 + p8 * IN_CH * 2 + 2 * 64 * p8 * 3 * 2 + (64 << 20)
    }

    /// (x / t1 / t2 at half the size, a band at full size, the upsampled input: the largest asked of it) bytes
    fn sizes(h8: usize, w8: usize) -> (usize, usize, usize) {
        let p2 = 16 * h8 * w8;
        let band = (BAND + 2 * HALO).min(8 * h8) * 8 * w8 * 128 * 2;
        (p2 * 128 * 2, band, (p2 * 256 * 2).max(band))
    }

    /// A new stream for latents of [h8, w8] (the pasts emptied, the first 3 frames to drop)
    /// The per-size state let go
    pub fn release(&mut self) {
        self.state = None;
    }

    pub fn reset(&mut self, h8: usize, w8: usize) -> Result<()> {
        if !self.state.as_ref().is_some_and(|s| s.h8 == h8 && s.w8 == w8) {
            self.state = None;
            let gpu = &self.gpu;
            let p8 = h8 * w8;
            let mut past = Vec::new();
            for (c, px) in [(512, p8), (256, 4 * p8), (128, 16 * p8)] {
                for _ in 0..3 {
                    past.push(DevBuf::new(gpu, px * c * 2)?);
                }
            }
            let (half, band, upb) = Self::sizes(h8, w8);
            let work = half.max(band);
            self.state = Some(State {
                h8,
                w8,
                past,
                have: vec![false; 9],
                trim: 3,
                inp: DevBuf::new(gpu, p8 * IN_CH * 2)?,
                x: DevBuf::new(gpu, half)?,
                t1: DevBuf::new(gpu, work)?,
                t2: DevBuf::new(gpu, work)?,
                // a TGrow's output: [p8, 512] or a frame's [p2, 128]
                tg: DevBuf::new(gpu, 16 * p8 * 128 * 2)?,
                tg2: DevBuf::new(gpu, 4 * p8 * 512 * 2)?,
                up: DevBuf::new(gpu, upb)?,
                ob: DevBuf::new(gpu, (BAND + 2 * HALO).min(8 * h8) * 8 * w8 * 3 * 2)?,
                out: DevBuf::new(gpu, 64 * p8 * 3 * 2)?,
            });
        }
        let s = self.state.as_mut().unwrap();
        s.have.iter_mut().for_each(|h| *h = false);
        s.trim = 3;
        Ok(())
    }

    /// MemBlock `i` over x [h w, c] in place (t1, t2 its scratch)
    #[allow(clippy::too_many_arguments)]
    fn memblock(&self, i: usize, past: &DevBuf, have: &mut bool, x: &DevBuf, c: usize, h: usize, w: usize, t1: &DevBuf, t2: &DevBuf) -> Result<()> {
        let m = &self.mb[i];
        let gpu = &self.gpu;
        if *have {
            m.c1x.run(gpu, x.ptr(), c, h, w, false, false, t1)?;
            m.c1p.run(gpu, past.ptr(), c, h, w, true, true, t1)?;
        } else {
            // no past: zeros (its half of the convolution adds nothing)
            m.c1x.run(gpu, x.ptr(), c, h, w, false, true, t1)?;
        }
        m.c2.run(gpu, t1.ptr(), c, h, w, false, true, t2)?;
        // the input is the next frame's past: kept before the block's output takes its place
        past.copy_within(0, x, 0, h * w * c * 2)?;
        *have = true;
        m.c3.run(gpu, t2.ptr(), c, h, w, true, true, x)
    }

    /// rows [r0, r0 + rows) of the nearest 2x of x [h, w, c] (pixels `ld` apart, channels from `c0`) into `out` [rows, 2w, c]
    #[allow(clippy::too_many_arguments)]
    fn up2(&self, x: *const std::ffi::c_void, h: usize, w: usize, c: usize, ld: usize, c0: usize, r0: usize, rows: usize, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        prof::timed(&self.gpu, "decoder: upsample, input, bytes", 0.0, (rows * 2 * w * c * 2 * 5 / 4) as f64, || {
            // SAFETY: x holds h x w pixels of ld halves, out rows x 2w x c halves.
            check(unsafe { (k.up2)(g(&self.gpu), x, h as i64, w as i64, c as i64, ld as i64, c0 as i64, r0 as i64, rows as i64, out.ptr()) }, "decoder upsample")
        })
    }

    /// One latent frame t of lat (float32 [16, f, h8, w8] on this card) with its 4 cond frames (half [4, H, W, 3]):
    /// each frame made (half [H, W, 3] in about [0, 1]) given to `emit` - none for the stream's first 3
    pub fn frame(&mut self, lat: &DevBuf, f: usize, t: usize, cond: *const std::ffi::c_void, emit: &mut dyn FnMut(&DevBuf) -> Result<()>) -> Result<()> {
        let mut s = self.state.take().ok_or_else(|| Error("flashvsr: the decoder was not reset".into()))?;
        let r = self.frame_in(&mut s, lat, f, t, cond, emit);
        self.state = Some(s);
        r
    }

    fn frame_in(&self, s: &mut State, lat: &DevBuf, f: usize, t: usize, cond: *const std::ffi::c_void, emit: &mut dyn FnMut(&DevBuf) -> Result<()>) -> Result<()> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let (h8, w8) = (s.h8, s.w8);
        let p8 = h8 * w8;
        prof::timed(gpu, "decoder: upsample, input, bytes", 0.0, (p8 * (IN_CH * 2 + 64 * 3 * 2 + 16 * 4)) as f64, || {
            // SAFETY: cond holds 4 frames of 8 h8 x 8 w8 x 3 halves, lat 16 x f x h8 x w8 floats, inp h8 x w8 x 784 halves.
            check(unsafe { (k.tcd_input)(g(gpu), cond, lat.fp(), f as i64, t as i64, h8 as i64, w8 as i64, IN_CH as i64, s.inp.ptr()) }, "decoder input")
        })?;
        let State { past, have, x, t1, t2, tg, tg2, up, ob, out, inp, trim, .. } = s;
        // 1/8 of the size, 512 channels
        self.c1.run(gpu, inp.ptr(), IN_CH, h8, w8, false, true, t1)?;
        self.c3.run(gpu, t1.ptr(), 512, h8, w8, false, true, x)?;
        for i in 0..3 {
            self.memblock(i, &past[i], &mut have[i], x, 512, h8, w8, t1, t2)?;
        }
        self.tg9.half(gpu, x.ptr(), 512, p8, tg.ptr(), 512, Act::None)?;
        // 1/4, 256 channels
        let (h4, w4) = (2 * h8, 2 * w8);
        self.up2(tg.ptr(), h8, w8, 512, 512, 0, 0, h4, up)?;
        self.c10.run(gpu, up.ptr(), 512, h4, w4, false, false, x)?;
        for i in 3..6 {
            self.memblock(i, &past[i], &mut have[i], x, 256, h4, w4, t1, t2)?;
        }
        // two frames: channels [0, 256) and [256, 512) of the 1x1's output
        self.tg15.half(gpu, x.ptr(), 256, h4 * w4, tg2.ptr(), 512, Act::None)?;
        let (h2, w2) = (2 * h4, 2 * w4);
        for sub in 0..2 {
            // 1/2, 128 channels
            self.up2(tg2.ptr(), h4, w4, 256, 512, sub * 256, 0, h2, up)?;
            self.c16.run(gpu, up.ptr(), 256, h2, w2, false, false, x)?;
            for i in 6..9 {
                self.memblock(i, &past[i], &mut have[i], x, 128, h2, w2, t1, t2)?;
            }
            // two frames again: channels [0, 128) and [128, 256) of TGrow 21, a product each
            let (h1, w1) = (2 * h2, 2 * w2);
            for tg21 in &self.tg21 {
                tg21.half(gpu, x.ptr(), 128, h2 * w2, tg.ptr(), 128, Act::None)?;
                // the full size a band of rows at a time: each band's rows and the halo its convolutions reach into
                for ((y0, n), (a, m)) in bands(h1, BAND) {
                    self.up2(tg.ptr(), h2, w2, 128, 128, 0, a, m, up)?;
                    self.c22.run(gpu, up.ptr(), 128, m, w1, false, true, t1)?;
                    self.c24.run(gpu, t1.ptr(), 128, m, w1, false, true, t2)?;
                    self.c26.run(gpu, t2.ptr(), 128, m, w1, false, false, ob)?;
                    out.copy_within(y0 * w1 * 3 * 2, ob, (y0 - a) * w1 * 3 * 2, n * w1 * 3 * 2)?;
                }
                if *trim > 0 {
                    *trim -= 1;
                } else {
                    emit(out)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bands_cover_the_frame_once_with_their_halos() {
        for (h, band) in [(1024, 256), (1408, 256), (768, 256), (100, 256), (130, 64)] {
            let b = bands(h, band);
            // their own rows: every row once, in order
            let mut next = 0;
            for &((y0, n), (a, m)) in &b {
                assert_eq!(y0, next);
                next = y0 + n;
                // the halo: 3 rows each side inside the frame, none outside it
                assert_eq!(a, y0.saturating_sub(HALO));
                assert_eq!(a + m, (y0 + n + HALO).min(h));
            }
            assert_eq!(next, h);
        }
        assert_eq!(bands(512, 256), vec![((0, 256), (0, 259)), ((256, 256), (253, 259))]);
    }
}
