//! Writing a clip: frames (and sound) -> .mp4, through ffmpeg (H.264 + AAC), the way the reference pipeline writes
//! its clips (reference/h3x.py `_write_video`): 24 fps, yuv420p, the sound cut or padded to exactly the video's
//! length and peak-normalized to -3 dBFS.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::{Ctx, Error, Result};

/// Sound for a clip: one buffer per channel.
pub struct Audio {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

/// Pictures for a clip: [3, count, height, width] in [0, 1] (planar, as the decoder writes them).
pub struct Frames<'a> {
    pub px: &'a [f32],
    pub count: usize,
    pub height: usize,
    pub width: usize,
    pub fps: u32,
}

pub fn write_mp4(path: &Path, v: &Frames, audio: Option<&Audio>, log: &mut dyn FnMut(String)) -> Result<()> {
    let (frames, f, h, w, fps) = (v.px, v.count, v.height, v.width, v.fps);
    let plane = h * w;
    if frames.len() != 3 * f * plane {
        return Err(Error(format!("write_mp4: {} values for 3 x {f} x {h} x {w}", frames.len())));
    }
    let tmp = path.with_extension("pcm.tmp");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-"]);
    if let Some(a) = audio {
        // the length lock: exactly F / fps seconds of sound
        let want = ((f as f64 / fps as f64) * a.sample_rate as f64).round() as usize;
        let peak = a.channels.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
        let gain = if peak > 1e-6 { 10f32.powf(-3.0 / 20.0) / peak } else { 1.0 };
        log(format!("audio  : peak {:.1} dBFS -> -3 dBFS (gain x{gain:.1})", 20.0 * peak.max(1e-12).log10()));
        let ch = a.channels.len();
        let mut pcm = Vec::with_capacity(want * ch * 2);
        for i in 0..want {
            for c in &a.channels {
                let v = c.get(i).copied().unwrap_or(0.0) * gain;
                pcm.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
            }
        }
        std::fs::write(&tmp, pcm).ctx("writing the sound for ffmpeg")?;
        cmd.args(["-f", "s16le", "-ar", &a.sample_rate.to_string(), "-ac", &ch.to_string(), "-i"]).arg(&tmp);
        cmd.args(["-c:a", "aac"]);
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]).arg(path);
    let mut child = cmd.stdin(Stdio::piped()).spawn().ctx("starting ffmpeg")?;
    {
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;
        let mut rgb = vec![0u8; plane * 3];
        for fi in 0..f {
            for c in 0..3 {
                let src = &frames[(c * f + fi) * plane..][..plane];
                for (i, v) in src.iter().enumerate() {
                    rgb[i * 3 + c] = (v.clamp(0.0, 1.0) * 255.0) as u8;
                }
            }
            stdin.write_all(&rgb).ctx("writing frames to ffmpeg")?;
        }
    }
    let st = child.wait()?;
    let _ = std::fs::remove_file(&tmp);
    if !st.success() {
        return Err(Error(format!("ffmpeg failed ({st})")));
    }
    Ok(())
}

/// How a picture is fitted to the canvas: stretched (the reference's keyframe anchors) or scaled to cover and
/// cropped in the middle (its guide clips).
#[derive(Clone, Copy)]
pub enum Fit {
    Stretch,
    Cover,
}

fn scale_filter(w: usize, h: usize, fit: Fit) -> String {
    match fit {
        Fit::Stretch => format!("scale={w}:{h}:flags=lanczos"),
        Fit::Cover => format!("scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=increase,crop={w}:{h}"),
    }
}

/// Every frame of a picture or video, fitted to w x h: [frames, h, w, 3] in [0, 1], and the frame count.
pub fn read_frames(path: &Path, w: usize, h: usize, fit: Fit) -> Result<(Vec<f32>, usize)> {
    read_frames_at(path, w, h, fit, None)
}

/// The same at a frame rate (a reference clip at the model's 24 fps), at most `max` frames from the start
pub fn read_frames_at(path: &Path, w: usize, h: usize, fit: Fit, fps: Option<(f64, usize)>) -> Result<(Vec<f32>, usize)> {
    let vf = match fps {
        Some((r, _)) => format!("fps={r},{}", scale_filter(w, h, fit)),
        None => scale_filter(w, h, fit),
    };
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-loglevel", "error", "-i"]).arg(path).args(["-vf", &vf]);
    if let Some((_, max)) = fps {
        cmd.args(["-frames:v", &max.to_string()]);
    }
    let out = cmd
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .ctx("starting ffmpeg")?;
    if !out.status.success() || out.stdout.is_empty() {
        return Err(Error(format!("{}: ffmpeg could not read frames ({})", path.display(), String::from_utf8_lossy(&out.stderr).trim())));
    }
    let frame = w * h * 3;
    let n = out.stdout.len() / frame;
    Ok((out.stdout[..n * frame].iter().map(|b| *b as f32 / 255.0).collect(), n))
}

/// A picture's or video's size in pixels (width, height)
pub fn size(path: &Path) -> Result<(usize, usize)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x"])
        .arg(path)
        .output()
        .ctx("starting ffprobe")?;
    let s = String::from_utf8_lossy(&out.stdout);
    let (w, h) = s.trim().lines().next().and_then(|l| l.split_once('x')).ok_or_else(|| Error(format!("{}: ffprobe found no picture ({})", path.display(),
                                                                                                    String::from_utf8_lossy(&out.stderr).trim())))?;
    match (w.trim().parse(), h.trim().parse()) {
        (Ok(w), Ok(h)) => Ok((w, h)),
        _ => Err(Error(format!("{}: a picture of size {}?", path.display(), s.trim()))),
    }
}

/// The last `n` frames of a picture or video (fewer if it has fewer): [frames, h, w, 3].
pub fn read_frames_tail(path: &Path, n: usize, w: usize, h: usize, fit: Fit) -> Result<(Vec<f32>, usize)> {
    let (all, count) = read_frames(path, w, h, fit)?;
    let keep = n.min(count);
    let frame = w * h * 3;
    Ok((all[(count - keep) * frame..].to_vec(), keep))
}

/// The last `seconds` of a file's sound (or all of it when 0) as stereo at `rate`: one buffer per channel.
pub fn read_audio_tail(path: &Path, seconds: f64, rate: u32) -> Result<Vec<Vec<f32>>> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-vn", "-ac", "2", "-ar", &rate.to_string(), "-f", "f32le", "-"])
        .output()
        .ctx("starting ffmpeg")?;
    if !out.status.success() || out.stdout.is_empty() {
        return Err(Error(format!("{}: ffmpeg could not read sound ({})", path.display(), String::from_utf8_lossy(&out.stderr).trim())));
    }
    let v: Vec<f32> = out.stdout.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let n = v.len() / 2;
    let keep = if seconds > 0.0 { ((seconds * rate as f64).round() as usize).min(n) } else { n };
    Ok((0..2).map(|c| (n - keep..n).map(|i| v[i * 2 + c]).collect()).collect())
}

/// A picture [h, w, 3] in [0, 1] as a lossless .png.
pub fn write_png(path: &Path, px: &[f32], w: usize, h: usize) -> Result<()> {
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-i", "-", "-frames:v", "1"])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .ctx("starting ffmpeg")?;
    {
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;
        let rgb: Vec<u8> = px.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).collect();
        stdin.write_all(&rgb).ctx("writing the picture to ffmpeg")?;
    }
    let st = child.wait()?;
    if !st.success() {
        return Err(Error(format!("ffmpeg failed writing {} ({st})", path.display())));
    }
    Ok(())
}

/// Mean luminance (Rec. 601) of pictures [.., 3] in [0, 1].
pub fn luma(px: &[f32]) -> f32 {
    let n = px.len() / 3;
    px.chunks_exact(3).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).sum::<f32>() / n.max(1) as f32
}
