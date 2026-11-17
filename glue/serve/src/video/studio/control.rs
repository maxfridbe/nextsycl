//! `video.control`: a clip turned into the control video H3's Fun ControlNet-Union steers by (video.generate's
//! `control_video`). The Union model was trained on canny, depth, HED, MLSD and pose frames (2.0 added scribble,
//! layout and gray); the ones made here need no model:
//!
//! - **canny**: edges, white on black - a 5 x 5 Gaussian (sigma 1), Sobel gradients, thinning along the gradient and
//!   two thresholds joined by hysteresis (OpenCV's Canny with its L2 gradient, so `low` / `high` mean what they mean
//!   to cv2.Canny: 100 / 200 is the ControlNet annotators' default);
//! - **gray**: the clip's brightness alone (BT.601 luma), for the 2.0 checkpoint's gray control (recolouring).
//!
//! - **depth** needs a model: a depth estimator (Depth Anything V2) on the image servers runs it as an image-queue
//!   job (the front door's `control_depth`, the image server's POST /v1/videos/depth) through `make_depth` here -
//!   each frame's relative inverse depth, kept until the clip ends, then the whole clip scaled by one range (the 0.5th
//!   to 99.5th percentile of every frame's values: `depth_range`), near bright. One range for the clip, not a frame's
//!   own min / max, so the grey of a wall does not pump as something comes close.
//!
//! The clip is read at 24 fps from `start_s`, scaled to cover the canvas and cut in the middle - what video.generate
//! does to a control video itself (`frames_for`: Fit::Cover) - so the control video is the clip's own size and length
//! and the thresholds act on the edges the denoiser will see. A clip shorter than `frames` holds its last frame, as
//! the engine would. The result is a file beside the clips (`ref_<unix>_<stem>-<control>.mp4`), which
//! video.generate's control_video takes by name and system.files.delete removes.
//!
//! (nextsycl: ported from nextcuda's glue/serve/src/video/studio/control.rs. ffmpeg / ffprobe through
//! `video::media::tool` - the host's when it handles H.264, else the video daemon's container's: Fedora's ffmpeg-free on
//! hardac1 cannot decode the clips. The image server's container has its own.)

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Stdio};
#[cfg(test)]
use std::process::Command;

use serde_json::{json, Value};

use crate::video::media::tool;

/// What `control` makes here, on the CPU (`video.control.<kind>` in system.version's features)
pub const KINDS: [&str; 2] = ["canny", "gray"];
/// What a model makes (an image server's job): depth, by a depth estimator
pub const MODEL_KINDS: [&str; 1] = ["depth"];

/// The canvas a control video is made at when none is asked: the source's aspect within the house size's pixels
/// (768 x 576), sides multiples of 32 within the model's 256 - 1344
pub const BUDGET: f64 = 768.0 * 576.0;
pub const SIDE: (u32, u32) = (256, 1344);

/// How a clip becomes a control video
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub control: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub start_s: f64,
    /// canny's thresholds on the gradient's size (0-255 pictures, Sobel)
    pub low: f32,
    pub high: f32,
    /// canny: stretch the clip's brightness to the full range first (`levels`)
    pub normalize: bool,
    /// the brightness that becomes 0 and 255 (the clip's 1st and 99th percentiles), when normalized
    pub levels: Option<(f32, f32)>,
}

/// A side on the 32 grid within the model's sides
fn side32(v: f64) -> u32 {
    (((v / 32.0).round() as u32) * 32).clamp(SIDE.0, SIDE.1)
}

/// The canvas for a source of `w` x `h` when the request names none: its aspect, about BUDGET pixels
pub fn default_canvas(w: u32, h: u32) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (768, 576);
    }
    let k = (BUDGET / (w as f64 * h as f64)).sqrt();
    (side32(w as f64 * k), side32(h as f64 * k))
}

/// The most frames (17k + 5, 22 - 362) a clip of `seconds` gives at 24 fps; None when it is shorter than 22 frames
pub fn frames_in(seconds: f64) -> Option<u32> {
    let n = (seconds * 24.0 + 1e-6).floor() as u32;
    (super::queue::FRAMES.0 as u32..=n.min(super::queue::FRAMES.1 as u32)).rev().find(|f| f % 17 == 5)
}

/// BT.601 luma of an RGB picture, 0-255
pub fn luma(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks_exact(3).map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect()
}

/// A 5 x 5 Gaussian (sigma 1), separable, edges repeated
fn blur(g: &[f32], w: usize, h: usize) -> Vec<f32> {
    const K: [f32; 5] = [0.054_488_685, 0.244_201_34, 0.402_619_93, 0.244_201_34, 0.054_488_685];
    let mut t = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (i, k) in K.iter().enumerate() {
                let xx = (x as isize + i as isize - 2).clamp(0, w as isize - 1) as usize;
                a += k * g[y * w + xx];
            }
            t[y * w + x] = a;
        }
    }
    let mut o = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (i, k) in K.iter().enumerate() {
                let yy = (y as isize + i as isize - 2).clamp(0, h as isize - 1) as usize;
                a += k * t[yy * w + x];
            }
            o[y * w + x] = a;
        }
    }
    o
}

/// Canny's edges of a brightness picture (0-255): 255 on an edge, 0 elsewhere
pub fn canny(g: &[f32], w: usize, h: usize, low: f32, high: f32) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w < 3 || h < 3 {
        return out;
    }
    let b = blur(g, w, h);
    let at = |x: usize, y: usize| b[y * w + x];
    // Sobel: the gradient's size and its direction in four bins (0: across x, 1: 45 deg, 2: across y, 3: 135 deg)
    let mut mag = vec![0f32; w * h];
    let mut dir = vec![0u8; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            mag[y * w + x] = (gx * gx + gy * gy).sqrt();
            // tan(22.5) = 0.4142, tan(67.5) = 2.4142
            let (ax, ay) = (gx.abs(), gy.abs());
            dir[y * w + x] = if ay <= ax * 0.414_213_56 {
                0
            } else if ay >= ax * 2.414_213_6 {
                2
            } else if (gx > 0.0) == (gy > 0.0) {
                1
            } else {
                3
            };
        }
    }
    // thinning: a pixel stays where it is the largest along its gradient; 2 strong, 1 weak, 0 none
    let mut keep = vec![0u8; w * h];
    let mut stack = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let m = mag[i];
            if m <= low {
                continue;
            }
            let (a, c) = match dir[i] {
                0 => (mag[i - 1], mag[i + 1]),
                2 => (mag[i - w], mag[i + w]),
                // y grows downwards: gx and gy of one sign run from top-left to bottom-right
                1 => (mag[i - w - 1], mag[i + w + 1]),
                _ => (mag[i - w + 1], mag[i + w - 1]),
            };
            // OpenCV's rule: strictly above one neighbour, at least the other (a plateau keeps one pixel)
            if m > a && m >= c {
                keep[i] = if m > high { 2 } else { 1 };
                if m > high {
                    stack.push(i);
                }
            }
        }
    }
    // hysteresis: weak pixels joined (8 ways) to a strong one are edges too
    while let Some(i) = stack.pop() {
        out[i] = 255;
        let (x, y) = (i % w, i / w);
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if keep[j] == 1 {
                    keep[j] = 2;
                    stack.push(j);
                }
            }
        }
    }
    out
}

/// One frame's control picture, RGB
pub fn frame(control: &str, rgb: &[u8], w: usize, h: usize, s: &Settings) -> Vec<u8> {
    let mut g = luma(rgb);
    if let (Some((lo, hi)), "canny") = (s.levels, control) {
        let k = 255.0 / (hi - lo).max(1.0);
        g.iter_mut().for_each(|v| *v = ((*v - lo) * k).clamp(0.0, 255.0));
    }
    let one: Vec<u8> = match control {
        "canny" => canny(&g, w, h, s.low, s.high),
        _ => g.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect(),
    };
    one.iter().flat_map(|v| [*v, *v, *v]).collect()
}

/// The request's `file` in the clips directory `dir` (as named, or with .mp4): its path and plain name
pub fn source(dir: &Path, p: &Value) -> Result<(std::path::PathBuf, String), super::RpcError> {
    use super::RpcError;
    let file = p["file"].as_str().unwrap_or("").trim();
    let n = crate::files::plain(file).ok_or_else(|| RpcError::param("file: a clip in the clips directory (video.upload, or a finished clip's name)"))?;
    let src = [n.to_string(), format!("{n}.mp4")].into_iter().map(|f| dir.join(f)).find(|f| f.is_file())
        .ok_or_else(|| RpcError::not_found(&format!("no clip {n} in the clips directory")))?;
    Ok((src, n.to_string()))
}

/// The control video's name beside its source `n`: ref_<unix>_<stem>-<control>.mp4 (the stem's plain characters, 40 at
/// most)
pub fn output_name(n: &str, control: &str, unix: u64) -> String {
    let stem: String = n.trim_end_matches(".mp4").trim_start_matches("ref_").chars().filter(|c| c.is_ascii_alphanumeric() || "_-".contains(*c)).take(40).collect();
    format!("ref_{unix}_{stem}-{control}.mp4")
}

/// The source's size and seconds (ffprobe); None when it has no video ffprobe reads
pub fn probe(src: &Path) -> Option<(u32, u32, f64)> {
    let o = tool("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration:format=duration", "-of", "json"])
        .arg(src)
        .output()
        .ok()?;
    let v: Value = serde_json::from_slice(&o.stdout).ok()?;
    let st = &v["streams"][0];
    let (w, h) = (st["width"].as_u64()? as u32, st["height"].as_u64()? as u32);
    let secs = |x: &Value| x.as_str().and_then(|t| t.parse::<f64>().ok()).or_else(|| x.as_f64());
    // a WebM from a browser's MediaRecorder carries no stream duration, only the container's
    let d = secs(&st["duration"]).or_else(|| secs(&v["format"]["duration"])).unwrap_or(0.0);
    Some((w, h, d))
}

/// The brightness at a share of a histogram's count (0-1)
fn percentile(hist: &[u64; 256], share: f64) -> f32 {
    let total: u64 = hist.iter().sum();
    let want = (total as f64 * share).ceil().max(1.0) as u64;
    let mut seen = 0;
    for (v, n) in hist.iter().enumerate() {
        seen += n;
        if seen >= want {
            return v as f32;
        }
    }
    255.0
}

/// The clip's brightness range (its 1st and 99th percentiles) from two small frames a second of the part used; None
/// when it is flat (nothing to stretch) or unreadable. A dim webcam take spans little of 0-255, and its edges fall
/// under fixed thresholds: canny reads it stretched.
pub fn levels(src: &Path, start_s: f64, seconds: f64) -> Option<(f32, f32)> {
    let o = tool("ffmpeg")
        .args(["-v", "error", "-nostdin", "-ss", &format!("{:.3}", start_s.max(0.0))])
        .arg("-i")
        .arg(src)
        .args(["-t", &format!("{:.3}", seconds.max(0.1)), "-an", "-vf", "fps=2,scale=160:-2", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .ok()?;
    let mut hist = [0u64; 256];
    o.stdout.iter().for_each(|v| hist[*v as usize] += 1);
    let (lo, hi) = (percentile(&hist, 0.01), percentile(&hist, 0.99));
    (o.stdout.len() > 64 && hi - lo >= 16.0).then_some((lo, hi))
}

/// The control video of `src` written to `dst` (an mp4): frames read, made and written a batch at a time on every
/// core, a short source's last one held to `frames`; (frames written, frames the source gave)
pub fn make(src: &Path, dst: &Path, s: &Settings) -> Result<(u32, u32), String> {
    let (w, h, n) = (s.width as usize, s.height as usize, s.frames as usize);
    let vf = format!("fps=24,scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h},setsar=1");
    let mut dec = tool("ffmpeg")
        .args(["-v", "error", "-nostdin"])
        .args(["-ss", &format!("{:.3}", s.start_s.max(0.0))])
        .arg("-i")
        .arg(src)
        .args(["-an", "-vf", &vf, "-frames:v", &n.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let part = dst.with_extension("mp4.part");
    let mut enc = tool("ffmpeg")
        .args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", "24", "-i", "-"])
        // near-lossless: a control video's thin edges are what it carries
        .args(["-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4"])
        .arg(&part)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let r = (|| -> Result<(u32, u32), String> {
        let mut input = dec.stdout.take().ok_or("ffmpeg's output")?;
        let mut output = enc.stdin.take().ok_or("ffmpeg's input")?;
        let size = w * h * 3;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
        let mut done = 0u32;
        let mut last: Vec<u8> = Vec::new();
        loop {
            let mut batch: Vec<Vec<u8>> = Vec::with_capacity(threads);
            while batch.len() < threads {
                let mut buf = vec![0u8; size];
                match input.read_exact(&mut buf) {
                    Ok(()) => batch.push(buf),
                    Err(_) => break,
                }
            }
            if batch.is_empty() {
                break;
            }
            let made: Vec<Vec<u8>> = std::thread::scope(|sc| {
                let hs: Vec<_> = batch.iter().map(|f| sc.spawn(move || frame(&s.control, f, w, h, s))).collect();
                hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
            });
            for f in made {
                output.write_all(&f).map_err(|e| format!("writing to ffmpeg: {e}"))?;
                done += 1;
                last = f;
            }
            if batch.len() < threads {
                break;
            }
        }
        // the source ended first: its last frame held, as video.generate would hold it
        let given = done;
        while done > 0 && (done as usize) < n {
            output.write_all(&last).map_err(|e| format!("writing to ffmpeg: {e}"))?;
            done += 1;
        }
        Ok((done, given))
    })();
    let given = r.as_ref().map_or(0, |x| x.1);
    finish(dec, enc, r.map(|x| x.0), &part, dst).map(|done| (done, given))
}

/// The decoder of a control video's frames: the source from `start_s` at 24 fps, scaled to cover the canvas and cut
/// in the middle, its last frame held to `frames` - RGB bytes on its output
pub fn decoder(src: &Path, s: &Settings) -> Result<Child, String> {
    let (w, h) = (s.width, s.height);
    let vf = format!("fps=24,scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h},tpad=stop_mode=clone:stop=-1,setsar=1");
    tool("ffmpeg")
        .args(["-v", "error", "-nostdin"])
        .args(["-ss", &format!("{:.3}", s.start_s.max(0.0))])
        .arg("-i")
        .arg(src)
        .args(["-an", "-vf", &vf, "-frames:v", &s.frames.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))
}

/// The encoder of a control video: raw frames of `pix_fmt` (rgb24, gray) on its input, H.264 at 24 fps into `part`
pub fn encoder(part: &Path, s: &Settings, pix_fmt: &str) -> Result<Child, String> {
    tool("ffmpeg")
        .args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", &format!("{}x{}", s.width, s.height), "-r", "24", "-i", "-"])
        // near-lossless: a control video's thin edges (and a depth map's smooth ramps) are what it carries
        .args(["-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4"])
        .arg(part)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))
}

/// Both ffmpegs waited for, the encoded part moved into place: the frames written, or why not (the part removed)
fn finish(mut dec: Child, mut enc: Child, r: Result<u32, String>, part: &Path, dst: &Path) -> Result<u32, String> {
    if r.is_err() {
        let _ = dec.kill();
    }
    let dec_err = {
        let mut e = String::new();
        if let Some(mut x) = dec.stderr.take() {
            let _ = x.read_to_string(&mut e);
        }
        e
    };
    let _ = dec.wait();
    drop(enc.stdin.take());
    let st = enc.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    let enc_err = {
        let mut e = String::new();
        if let Some(mut x) = enc.stderr.take() {
            let _ = x.read_to_string(&mut e);
        }
        e
    };
    let done = r.inspect_err(|_| {
        let _ = std::fs::remove_file(part);
    })?;
    if done == 0 {
        let _ = std::fs::remove_file(part);
        return Err(format!("no frames read from the clip{}", tail(&dec_err)));
    }
    if !st.success() {
        let _ = std::fs::remove_file(part);
        return Err(format!("encoding the control video failed{}", tail(&enc_err)));
    }
    std::fs::rename(part, dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    Ok(done)
}

/// One frame's depth kept until the clip's range is known: 16 bits against the frame's own range (finer than the
/// 8-bit video by far, half a float's memory)
pub struct Packed {
    lo: f32,
    step: f32,
    q: Vec<u16>,
}

impl Packed {
    pub fn of(d: &[f32]) -> Packed {
        let (lo, hi) = d.iter().filter(|v| v.is_finite()).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (0.0, 0.0) };
        let step = if hi > lo { (hi - lo) / 65535.0 } else { 1.0 };
        let q = d.iter().map(|&v| if v.is_finite() { ((v - lo) / step).round().clamp(0.0, 65535.0) as u16 } else { 0 }).collect();
        Packed { lo, step, q }
    }
    pub fn get(&self, i: usize) -> f32 {
        self.lo + self.q[i] as f32 * self.step
    }
    pub fn len(&self) -> usize {
        self.q.len()
    }
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }
}

/// Values of every frame the clip's range is read from (evenly spread over the frame)
const RANGE_SAMPLES: usize = 4096;
/// The range's ends: these fractions of the clip's values (a few stray pixels - a specular glint read as near - do not
/// stretch it)
pub const RANGE_LO: f64 = 0.005;
pub const RANGE_HI: f64 = 0.995;

/// The clip's depth range: the RANGE_LO and RANGE_HI quantiles of `samples` (every frame's share); (0, 1) for none
pub fn depth_range(samples: &mut [f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 1.0);
    }
    let n = samples.len();
    let at = |q: f64| ((n - 1) as f64 * q).round() as usize;
    let (i, j) = (at(RANGE_LO), at(RANGE_HI));
    let lo = *samples.select_nth_unstable_by(i, f32::total_cmp).1;
    let hi = *samples.select_nth_unstable_by(j, f32::total_cmp).1;
    (lo, hi)
}

/// A depth value as the control video's grey: the clip's range to 0-255, near (large inverse depth) bright
pub fn depth_byte(v: f32, lo: f32, hi: f32) -> u8 {
    if hi <= lo {
        return 0;
    }
    ((v - lo) / (hi - lo) * 255.0).round().clamp(0.0, 255.0) as u8
}

/// A frame's depth: RGB bytes at the canvas in, its relative inverse depth (a value a pixel) out
pub type DepthOf<'a> = dyn FnMut(&[u8]) -> Result<Vec<f32>, String> + 'a;

/// The depth control video of `src` written to `dst`: each frame (RGB bytes at the canvas) through `depth` (its
/// relative inverse depth at the canvas, a value a pixel), `progress` told the frames done (an error from either stops
/// it: a cancel); when the clip has ended, every frame scaled by the clip's range and encoded. The frames written and
/// the range
pub fn make_depth(src: &Path, dst: &Path, s: &Settings, depth: &mut DepthOf, progress: &mut dyn FnMut(u32) -> Result<(), String>)
                  -> Result<(u32, (f32, f32)), String> {
    let (w, h, n) = (s.width as usize, s.height as usize, s.frames as usize);
    let mut dec = decoder(src, s)?;
    let part = dst.with_extension("mp4.part");
    let mut range = (0.0, 1.0);
    let r = (|| -> Result<Vec<Packed>, String> {
        let mut input = dec.stdout.take().ok_or("ffmpeg's output")?;
        let mut frames: Vec<Packed> = Vec::with_capacity(n);
        let mut buf = vec![0u8; w * h * 3];
        while frames.len() < n && input.read_exact(&mut buf).is_ok() {
            let d = depth(&buf)?;
            if d.len() != w * h {
                return Err(format!("the depth model answered {} values for a {w} x {h} frame", d.len()));
            }
            frames.push(Packed::of(&d));
            progress(frames.len() as u32)?;
        }
        Ok(frames)
    })();
    // the encoder only now: nothing is written until the range is known
    let enc = encoder(&part, s, "gray");
    let (r, enc) = match (r, enc) {
        (Ok(frames), Ok(mut enc)) => {
            let wrote = (|| -> Result<u32, String> {
                let stride = (w * h / RANGE_SAMPLES).max(1);
                let mut samples: Vec<f32> = frames.iter().flat_map(|f| (0..f.len()).step_by(stride).map(|i| f.get(i))).collect();
                range = depth_range(&mut samples);
                let mut output = enc.stdin.take().ok_or("ffmpeg's input")?;
                let mut grey = vec![0u8; w * h];
                for f in &frames {
                    for (i, g) in grey.iter_mut().enumerate() {
                        *g = depth_byte(f.get(i), range.0, range.1);
                    }
                    output.write_all(&grey).map_err(|e| format!("writing to ffmpeg: {e}"))?;
                }
                Ok(frames.len() as u32)
            })();
            (wrote, enc)
        }
        (Err(e), Ok(enc)) => (Err(e), enc),
        (_, Err(e)) => {
            let _ = dec.kill();
            let _ = dec.wait();
            return Err(e);
        }
    };
    finish(dec, enc, r, &part, dst).map(|done| (done, range))
}

fn tail(e: &str) -> String {
    let t = e.trim();
    if t.is_empty() {
        String::new()
    } else {
        format!(": {}", t.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect::<String>())
    }
}

/// The request read against the source (`w` x `h`, `seconds` long): the settings, or what is wrong with it
pub fn settings(p: &Value, (w, h, seconds): (u32, u32, f64)) -> Result<Settings, super::RpcError> {
    use super::queue::num;
    use super::RpcError;
    let control = p["control"].as_str().unwrap_or("canny").trim().to_lowercase();
    if !KINDS.contains(&control.as_str()) && !MODEL_KINDS.contains(&control.as_str()) {
        return Err(RpcError::param(&format!("control {control:?}: this box makes {}, {} (pose needs an estimator it does not have)", KINDS.join(", "),
                                            MODEL_KINDS.join(", "))));
    }
    let start_s = num(p, "start_s", 0.0, 0.0, 86_400.0, false)?.as_f64().unwrap_or(0.0);
    let left = (seconds - start_s).max(0.0);
    let (dw, dh) = default_canvas(w, h);
    let side = |k: &str, d: u32| -> Result<u32, RpcError> {
        let v = num(p, k, d as f64, SIDE.0 as f64, SIDE.1 as f64, true)?.as_u64().unwrap_or(d as u64) as u32;
        if !v.is_multiple_of(32) {
            return Err(RpcError::param(&format!("{k} {v}: a multiple of 32")));
        }
        Ok(v)
    };
    let (width, height) = (side("width", dw)?, side("height", dh)?);
    let frames = match p.get("frames").filter(|v| !v.is_null()) {
        Some(_) => super::queue::check_frames(num(p, "frames", 0.0, 0.0, 1e6, true)?.as_u64().unwrap_or(0))? as u32,
        None => frames_in(left).ok_or_else(|| RpcError::param(&format!("the clip has {left:.2} s from start_s: a control video is 22 frames (0.92 s) at least")))?,
    };
    let low = num(p, "low", 100.0, 0.0, 1000.0, false)?.as_f64().unwrap_or(100.0) as f32;
    let high = num(p, "high", 200.0, 0.0, 1000.0, false)?.as_f64().unwrap_or(200.0) as f32;
    if high < low {
        return Err(RpcError::param(&format!("high {high} is below low {low}")));
    }
    let normalize = p.get("normalize").and_then(Value::as_bool).unwrap_or(true);
    Ok(Settings { control, width, height, frames, start_s, low, high, normalize, levels: None })
}

/// The answer's record of a control video made (`given`: the frames the source gave)
pub fn answer(name: &str, s: &Settings, given: u32, seconds: f64, took: f64) -> Value {
    json!({"name": name, "file": format!("/out/{name}"), "control": s.control, "width": s.width, "height": s.height, "frames": s.frames,
           "fps": 24, "seconds": s.frames as f64 / 24.0, "source_frames": given.min(s.frames), "source_seconds": seconds, "start_s": s.start_s,
           "low": s.low, "high": s.high, "levels": s.levels.map(|(a, b)| json!([a, b])), "took_s": (took * 100.0).round() / 100.0})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_has_edges_on_its_border_alone() {
        let (w, h) = (32, 32);
        let g: Vec<f32> = (0..w * h).map(|i| if (8..24).contains(&(i % w)) && (8..24).contains(&(i / w)) { 255.0 } else { 0.0 }).collect();
        let e = canny(&g, w, h, 100.0, 200.0);
        // the middle and the far background are not edges; the border is, all the way round
        assert_eq!(e[16 * w + 16], 0);
        assert_eq!(e[2 * w + 2], 0);
        for k in 10..22 {
            let row: Vec<u8> = (5..11).map(|x| e[k * w + x]).collect();
            assert!(row.contains(&255), "left edge at row {k}: {row:?}");
            let col: Vec<u8> = (5..11).map(|y| e[y * w + k]).collect();
            assert!(col.contains(&255), "top edge at column {k}: {col:?}");
        }
        // thin: one pixel across (the left border crossed at mid height)
        let across: usize = (0..16).filter(|x| e[16 * w + x] == 255).count();
        assert_eq!(across, 1, "{:?}", &e[16 * w..16 * w + 16]);
    }

    #[test]
    fn a_flat_picture_and_faint_noise_have_none() {
        let (w, h) = (24, 24);
        assert!(canny(&vec![128.0; w * h], w, h, 100.0, 200.0).iter().all(|v| *v == 0));
        let noise: Vec<f32> = (0..w * h).map(|i| 128.0 + ((i * 7919) % 13) as f32 - 6.0).collect();
        assert!(canny(&noise, w, h, 100.0, 200.0).iter().all(|v| *v == 0));
    }

    #[test]
    fn hysteresis_keeps_weak_edges_joined_to_strong_ones() {
        // a step whose height fades along the edge from strong (255) to weak (60, a gradient of about 155): the weak
        // end is kept through the strong one; the same weak step alone is dropped
        let (w, h) = (20, 40);
        let step = |y: usize| 255.0 - 195.0 * y as f32 / 39.0;
        let g: Vec<f32> = (0..w * h).map(|i| if i % w >= 10 { step(i / w) } else { 0.0 }).collect();
        let e = canny(&g, w, h, 100.0, 200.0);
        let row_has = |y: usize| (7..13).any(|x| e[y * w + x] == 255);
        assert!((2..38).all(row_has), "the edge all the way down");
        let weak: Vec<f32> = (0..w * h).map(|i| if i % w >= 10 { 60.0 } else { 0.0 }).collect();
        assert!(canny(&weak, w, h, 100.0, 200.0).iter().all(|v| *v == 0), "a weak edge alone");
    }

    #[test]
    fn a_dim_clip_is_stretched_before_its_edges_are_found() {
        // a square 30 levels brighter than its ground: under the thresholds as it is, an edge once stretched
        let (w, h) = (32, 32);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| {
            let v = if (8..24).contains(&(i % w)) && (8..24).contains(&(i / w)) { 50u8 } else { 20 };
            [v, v, v]
        }).collect();
        let mut s = Settings { control: "canny".into(), width: 32, height: 32, frames: 22, start_s: 0.0, low: 100.0, high: 200.0, normalize: true, levels: None };
        assert!(frame("canny", &rgb, w, h, &s).iter().all(|v| *v == 0));
        s.levels = Some((20.0, 50.0));
        assert!(frame("canny", &rgb, w, h, &s).contains(&255));
        let mut hist = [0u64; 256];
        hist[20] = 99;
        hist[200] = 1;
        assert_eq!((percentile(&hist, 0.01), percentile(&hist, 0.99), percentile(&hist, 1.0)), (20.0, 20.0, 200.0));
    }

    #[test]
    fn canvas_and_length_defaults() {
        assert_eq!(default_canvas(1280, 720), (896, 512));
        assert_eq!(default_canvas(640, 480), (768, 576));
        assert_eq!(default_canvas(720, 1280), (512, 896));
        assert_eq!(default_canvas(0, 0), (768, 576));
        assert_eq!(frames_in(5.0), Some(107), "120 frames: 107 is the most 17k + 5 under it");
        assert_eq!(frames_in(5.2), Some(124));
        assert_eq!(frames_in(30.0), Some(362), "at most 15.08 s");
        assert_eq!(frames_in(0.5), None);
        assert_eq!(frames_in(22.0 / 24.0), Some(22));
    }

    #[test]
    fn requests_are_read_against_the_source() {
        let src = (1280, 720, 6.0);
        let s = settings(&json!({}), src).unwrap();
        assert_eq!((s.control.as_str(), s.width, s.height, s.frames, s.low, s.high), ("canny", 896, 512, 141, 100.0, 200.0));
        let s = settings(&json!({"control": "Gray", "width": 768, "height": 576, "frames": 124, "start_s": 1}), src).unwrap();
        assert_eq!((s.control.as_str(), s.width, s.height, s.frames, s.start_s), ("gray", 768, 576, 124, 1.0));
        assert_eq!(settings(&json!({"start_s": 2}), src).unwrap().frames, 90, "4 s left");
        let err = |p: Value| settings(&p, src).unwrap_err().message;
        assert!(err(json!({"control": "pose"})).contains("canny, gray, depth"));
        assert_eq!(settings(&json!({"control": "depth"}), src).unwrap().control, "depth", "a model's: the front door routes it");
        assert!(err(json!({"width": 700})).contains("multiple of 32"));
        assert!(err(json!({"frames": 120})).contains("nearest: 107 or 124"));
        assert!(err(json!({"low": 150, "high": 100})).contains("below"));
        assert!(err(json!({"start_s": 5.5})).contains("22 frames"));
    }

    #[test]
    fn depth_is_kept_finely_and_scaled_by_the_clips_range() {
        // 16 bits against the frame's own range: well inside an 8-bit step of any range
        let d: Vec<f32> = (0..1000).map(|i| 0.37 + i as f32 * 0.0123).collect();
        let p = Packed::of(&d);
        // the range 12.3 in 65535 steps: within half a step (an 8-bit step of it is 0.048)
        assert!((0..1000).all(|i| (p.get(i) - d[i]).abs() < 2e-4), "packed");
        assert_eq!(p.len(), 1000);
        let flat = Packed::of(&[2.5; 4]);
        assert_eq!(flat.get(3), 2.5);
        // the range: quantiles, so a stray far value and a stray near one do not stretch it
        let mut v: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        v[10] = -50.0;
        v[20] = 80.0;
        let (lo, hi) = depth_range(&mut v);
        assert!((0.0..0.02).contains(&lo) && (0.98..1.0).contains(&hi), "{lo} {hi}");
        assert_eq!(depth_range(&mut []), (0.0, 1.0));
        // near (large) bright, clamped outside the range
        assert_eq!((depth_byte(0.0, 0.0, 1.0), depth_byte(1.0, 0.0, 1.0), depth_byte(0.5, 0.0, 1.0)), (0, 255, 128));
        assert_eq!((depth_byte(-3.0, 0.0, 1.0), depth_byte(7.0, 0.0, 1.0), depth_byte(1.0, 1.0, 1.0)), (0, 255, 0));
    }

    /// A depth video through ffmpeg (skipped where there is none), with a stand-in estimator: brightness as depth, a
    /// frame's grey the clip's range - a square brightening over the clip is not each frame's own white
    #[test]
    fn a_clip_becomes_its_depth_video() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("ns-depth-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.mp4");
        let ok = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=320x240:r=24:d=1", "-vf", "drawbox=x=100:y=80:w=80:h=80:color=white@1:t=fill,eq=brightness='t/2-0.4':eval=frame",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&src)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            return;
        }
        let (w, h, secs) = probe(&src).unwrap();
        let s = settings(&json!({"control": "depth", "width": 320, "height": 256, "frames": 22}), (w, h, secs)).unwrap();
        let dst = dir.join("out.mp4");
        let mut seen = 0;
        let (made, (lo, hi)) = make_depth(&src, &dst, &s, &mut |rgb| Ok(luma(rgb)), &mut |n| {
            seen = n;
            Ok(())
        }).unwrap();
        assert_eq!((made, seen), (22, 22));
        assert!(lo < hi, "{lo} {hi}");
        let (ow, oh, _) = probe(&dst).unwrap();
        assert_eq!((ow, oh), (320, 256));
        // a cancel from progress stops it and leaves no file
        let dst2 = dir.join("out2.mp4");
        let e = make_depth(&src, &dst2, &s, &mut |rgb| Ok(luma(rgb)), &mut |n| if n >= 3 { Err("cancelled".into()) } else { Ok(()) });
        assert_eq!(e.unwrap_err(), "cancelled");
        assert!(!dst2.exists() && !dst2.with_extension("mp4.part").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gray_is_luma() {
        let s = Settings { control: "gray".into(), width: 2, height: 1, frames: 22, start_s: 0.0, low: 100.0, high: 200.0, normalize: true, levels: Some((20.0, 60.0)) };
        assert_eq!(frame("gray", &[255, 0, 0, 0, 0, 255], 2, 1, &s), vec![76, 76, 76, 29, 29, 29], "gray is never stretched");
    }

    /// The whole path through ffmpeg (skipped where there is none): a moving square becomes its edges, as long as
    /// asked (the source's last frame held)
    #[test]
    fn a_clip_becomes_its_control_video() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("ns-control-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.mp4");
        let ok = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=24:d=1.5", "-f", "lavfi", "-i", "color=c=white:s=80x80:r=24:d=1.5",
                   "-filter_complex", "[0][1]overlay=x='40+t*60':y=80", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&src)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            return;
        }
        let (w, h, secs) = probe(&src).unwrap();
        assert_eq!((w, h), (320, 240));
        let s = settings(&json!({"control": "canny", "width": 320, "height": 256, "frames": 56}), (w, h, secs)).unwrap();
        let dst = dir.join("out.mp4");
        let made = make(&src, &dst, &s).unwrap();
        assert_eq!(made, (56, 36), "36 frames of clip and its last one held");
        assert_eq!(levels(&src, 0.0, 1.5), Some((0.0, 255.0)), "black and white");
        let (ow, oh, _) = probe(&dst).unwrap();
        assert_eq!((ow, oh), (320, 256));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
