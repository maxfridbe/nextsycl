//! The token sequence of one denoiser call: which rows are text, conditioning, audio and video; where each token sits
//! in (time, height, width) for the position rotation; and which timestep each kind of token runs at.
//!
//! The sequence is `[text | keyframe conditioning | audio | video]`. Text tokens count along the time axis from 0; the
//! target audio and video start where the text ends. Video positions come from an area-normalized grid (so the same
//! picture at another resolution lands on the same coordinates); audio rows sit at height 0 and at the two ends of
//! the width axis, one end per stereo channel.
//!
//! References ("ref2va": pictures, clips, sounds the prompt names) pack between the keyframes and the targets: they
//! take the time axis right after the text, and the targets start after them.

use crate::{Error, Result};

/// Latent frames cover 1, 4, 4, 4, 4, 1, 4 ... pixel frames (the first of every five is a single frame), and a pixel
/// frame is 5/3 of a position unit along the time axis.
const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const FRAME_RESCALE: f64 = 5.0 / 3.0;
/// Conditioning rows are presented as almost clean (video) or clean (audio).
pub const VISUAL_COND_TIMESTEP: f64 = 0.999;
pub const AUDIO_COND_TIMESTEP: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    /// Video rows of a keyframe (first / last frame conditioning).
    Cond,
    /// Audio rows of a keyframe.
    CondAudio,
    /// Audio rows of a reference (a voice to speak in: `<Audio j>` in the prompt, or a reference clip's sound).
    RefAudio,
    /// Video rows of a reference picture or clip (`<Picture i>`, `<Video k>`).
    RefImg,
    Audio,
    Video,
}

impl Kind {
    /// Which of a timestep's three table rows the kind uses: video-like 0, text 1, audio-like 2.
    pub fn modality(self) -> i32 {
        match self {
            Kind::Video | Kind::Cond | Kind::RefImg => 0,
            Kind::Text => 1,
            Kind::Audio | Kind::CondAudio | Kind::RefAudio => 2,
        }
    }
}

/// A keyframe to condition on: its position on the target timeline (in pixel frames) and how many latent frames of
/// video and of audio it brings.
#[derive(Clone, Copy, Debug)]
pub struct Keyframe {
    pub frame_index: f64,
    pub video_latent_t: Option<usize>,
    pub audio_latent_t: Option<usize>,
}

/// A reference the prompt refers to (ref2va), packed between the text and the targets. Pictures and clips also go
/// through the text encoder's vision tower; sounds never enter the text encoder (the prompt carries only their
/// `<Audio j>` label).
#[derive(Clone, Copy, Debug)]
pub enum RefBlock {
    /// `t` audio latent frames
    Audio { t: usize },
    /// a picture's latent, `h` x `w` latent pixels (its own size, not the clip's)
    Image { h: usize, w: usize },
    /// a clip's latent (`t` latent frames of `h` x `w`) and its sound's `audio_t` latent frames (0: none)
    Video { t: usize, h: usize, w: usize, audio_t: usize },
}

impl RefBlock {
    /// The stretch of the time axis it takes ahead of the targets.
    fn span(&self) -> f64 {
        match self {
            RefBlock::Audio { t } => *t as f64,
            RefBlock::Image { .. } => 1.0,
            RefBlock::Video { t, audio_t, .. } => (*audio_t as f64).max((0..*t).map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % 5]).sum()),
        }
    }
}

/// A frame's 2x2-patch rows on the area-normalized grid: (height, width) per row, and the width axis' ends
fn frame_grid(h: usize, w: usize) -> (Vec<(f64, f64)>, (f64, f64)) {
    let sqrt_area = ((h * w) as f64).sqrt();
    let (hs, ws) = (axis(h, sqrt_area), axis(w, sqrt_area));
    (hs.iter().flat_map(|a| ws.iter().map(move |b| (*a, *b))).collect(), (ws[0], ws[ws.len() - 1]))
}

/// `[start, stop)` rows of one kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub start: usize,
    pub stop: usize,
    pub kind: Kind,
}

pub struct Layout {
    pub segments: Vec<Segment>,
    /// [tokens, 3]: (time, height, width)
    pub positions: Vec<f64>,
    /// 2x2 latent patches per video frame
    pub frame_rows: usize,
}

/// `dim / 2` coordinates along one axis of a frame, centred, in units where a square frame spans 32.
fn axis(dim: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / 2;
    (0..n).map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0).collect()
}

/// Time coordinates of `n` latent frames starting at `origin`.
fn video_times(n: usize, origin: f64) -> Vec<f64> {
    let mut t = origin;
    (0..n)
        .map(|k| {
            let here = t;
            t += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5];
            here
        })
        .collect()
}

impl Layout {
    /// `latent_h`, `latent_w`: the video latent's size (two latent pixels per token along each); `audio_t`: audio
    /// latent frames (each gives two rows, one per stereo channel).
    pub fn new(text_len: usize, latent_t: usize, latent_h: usize, latent_w: usize, audio_t: usize, keyframes: &[Keyframe], refs: &[RefBlock]) -> Result<Layout> {
        if !latent_h.is_multiple_of(2) || !latent_w.is_multiple_of(2) || latent_h == 0 || latent_w == 0 {
            return Err(Error(format!("the video latent must have even height and width, got {latent_h} x {latent_w}")));
        }
        let (frame, (w_low, w_high)) = frame_grid(latent_h, latent_w);

        let mut segments = Vec::new();
        let mut positions = Vec::new();
        let mut row = 0;
        let mut push = |kind: Kind, n: usize, row: &mut usize| {
            segments.push(Segment { start: *row, stop: *row + n, kind });
            *row += n;
        };
        let video_on = |positions: &mut Vec<f64>, frame: &[(f64, f64)], vt: usize, origin: f64| {
            for t in video_times(vt, origin) {
                for (h, w) in frame {
                    positions.extend_from_slice(&[t, *h, *w]);
                }
            }
        };
        let video = |positions: &mut Vec<f64>, vt: usize, origin: f64| video_on(positions, &frame, vt, origin);
        // channel-major: every frame of the left channel, then every frame of the right
        let audio_on = |positions: &mut Vec<f64>, at: usize, origin: f64, (w_low, w_high): (f64, f64)| {
            for w in [w_low, w_high] {
                for i in 0..at {
                    positions.extend_from_slice(&[origin + i as f64, 0.0, w]);
                }
            }
        };
        let audio = |positions: &mut Vec<f64>, at: usize, origin: f64| audio_on(positions, at, origin, (w_low, w_high));

        for i in 0..text_len {
            positions.extend_from_slice(&[i as f64, 0.0, 0.0]);
        }
        push(Kind::Text, text_len, &mut row);
        // the references sit right after the text on the time axis; the targets start after them
        let cursor = text_len as f64 + refs.iter().map(|r| r.span()).sum::<f64>();
        for kf in keyframes {
            let at = cursor + FRAME_RESCALE * kf.frame_index;
            if let Some(vt) = kf.video_latent_t {
                video(&mut positions, vt, at);
                push(Kind::Cond, vt * frame.len(), &mut row);
            }
            if let Some(rt) = kf.audio_latent_t {
                audio(&mut positions, rt, at);
                push(Kind::CondAudio, rt * 2, &mut row);
            }
        }
        let mut rc = text_len as f64;
        for r in refs {
            match r {
                RefBlock::Audio { t } => {
                    if *t > 0 {
                        audio(&mut positions, *t, rc);
                        push(Kind::RefAudio, t * 2, &mut row);
                    }
                }
                RefBlock::Image { h, w } => {
                    let (f, _) = frame_grid(*h, *w);
                    for (a, b) in &f {
                        positions.extend_from_slice(&[rc, *a, *b]);
                    }
                    push(Kind::RefImg, f.len(), &mut row);
                }
                RefBlock::Video { t, h, w, audio_t } => {
                    // its sound's rows first, on its own width axis' ends, both from the same origin
                    let (f, ends) = frame_grid(*h, *w);
                    if *audio_t > 0 {
                        audio_on(&mut positions, *audio_t, rc, ends);
                        push(Kind::RefAudio, audio_t * 2, &mut row);
                    }
                    video_on(&mut positions, &f, *t, rc);
                    push(Kind::RefImg, t * f.len(), &mut row);
                }
            }
            rc += r.span();
        }
        audio(&mut positions, audio_t, cursor);
        push(Kind::Audio, audio_t * 2, &mut row);
        video(&mut positions, latent_t, cursor);
        push(Kind::Video, latent_t * frame.len(), &mut row);
        Ok(Layout { segments, positions, frame_rows: frame.len() })
    }

    pub fn tokens(&self) -> usize {
        self.segments.last().map_or(0, |s| s.stop)
    }

    pub fn segment(&self, kind: Kind) -> Option<Segment> {
        self.segments.iter().copied().find(|s| s.kind == kind)
    }
}

/// The audio stream runs on its own noise schedule: the same point of the base schedule, shifted differently.
pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

/// What one sampler step means for the tokens: the distinct timesteps in play, and each token's table row
/// (`timestep index * 3 + modality`).
pub struct Timesteps {
    /// distinct timesteps (1 - noise level), ascending
    pub values: Vec<f64>,
    /// [tokens]
    pub rows: Vec<i32>,
    /// index into `values` of the target video's and the target audio's timestep (the final layer uses them)
    pub video_index: usize,
    pub audio_index: usize,
}

impl Timesteps {
    /// `sigma_video`: the sampler's noise level for the video stream, in (0, 1].
    pub fn new(layout: &Layout, sigma_video: f64, shift_video: f64, shift_audio: f64) -> Timesteps {
        Timesteps::with_cond(layout, sigma_video, shift_video, shift_audio, VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP)
    }

    /// The same, with the timesteps conditioning rows are presented at (the keyframes' noise augmentation: 0.999
    /// and 1.0 unless a run asks for a less trusted anchor).
    pub fn with_cond(layout: &Layout, sigma_video: f64, shift_video: f64, shift_audio: f64, visual_cond: f64, audio_cond: f64) -> Timesteps {
        // the reference does this arithmetic in float32; the values are table keys, so it is followed exactly
        let sigma_v = (sigma_video as f32).max(1e-6);
        let base = sigma_v / (shift_video as f32 + sigma_v * (1.0 - shift_video as f32));
        let sigma_a = shift_audio as f32 * base / (1.0 + (shift_audio as f32 - 1.0) * base);
        let (t_v, t_a) = ((1.0 - sigma_v) as f64, (1.0 - sigma_a) as f64);
        let of = |kind: Kind| match kind {
            Kind::Text | Kind::Video => t_v,
            Kind::Audio => t_a,
            Kind::Cond | Kind::RefImg => t_v.max(visual_cond),
            Kind::CondAudio | Kind::RefAudio => t_a.max(audio_cond),
        };
        let mut values = vec![t_v, t_a];
        values.extend(layout.segments.iter().map(|s| of(s.kind)));
        values.sort_by(f64::total_cmp);
        values.dedup();
        let index = |t: f64| values.iter().position(|v| *v == t).expect("every timestep is in the list");
        let mut rows = vec![0i32; layout.tokens()];
        for s in &layout.segments {
            let r = index(of(s.kind)) as i32 * 3 + s.kind.modality();
            rows[s.start..s.stop].fill(r);
        }
        Timesteps { video_index: index(t_v), audio_index: index(t_a), values, rows }
    }

    /// The timestep embeddings [values, t_dim]: each timestep's point on the model's embedding curve, a table of
    /// `grid` rows over t in [0, 1], read with linear interpolation.
    pub fn embeddings(&self, table: &[f32], t_dim: usize) -> Vec<f32> {
        let grid = table.len() / t_dim;
        let mut out = Vec::with_capacity(self.values.len() * t_dim);
        for t in &self.values {
            let pos = (*t as f32).clamp(0.0, 1.0) * (grid - 1) as f32;
            let i0 = (pos.floor() as usize).min(grid - 2);
            let f = pos - i0 as f32;
            let (a, b) = (&table[i0 * t_dim..(i0 + 1) * t_dim], &table[(i0 + 1) * t_dim..(i0 + 2) * t_dim]);
            out.extend(a.iter().zip(b).map(|(a, b)| a + f * (b - a)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::Checkpoint;
    use std::path::Path;

    /// Positions and table rows of a real call of the reference pipeline (384x288, 2 s, first step), from
    /// reference/h3x.py's block dump.
    fn fixture() -> Checkpoint {
        Checkpoint::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/layout_384x288_2s.safetensors")).unwrap()
    }

    #[test]
    fn layout_matches_the_reference() {
        let fx = fixture();
        let m = |k: &str| fx.metadata[k].parse::<usize>().unwrap();
        let l = Layout::new(m("text_len"), m("latent_t"), m("latent_h"), m("latent_w"), m("audio_t"), &[], &[]).unwrap();
        assert_eq!(
            l.segments,
            vec![
                Segment { start: 0, stop: 137, kind: Kind::Text },
                Segment { start: 137, stop: 323, kind: Kind::Audio },
                Segment { start: 323, stop: 2159, kind: Kind::Video },
            ]
        );
        let want: Vec<f64> = fx.read("position_ids").unwrap().chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(l.positions.len(), want.len());
        let worst = l.positions.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-9, "positions differ by {worst}");
    }

    #[test]
    fn table_rows_match_the_reference() {
        let fx = fixture();
        let m = |k: &str| fx.metadata[k].parse::<usize>().unwrap();
        let l = Layout::new(m("text_len"), m("latent_t"), m("latent_h"), m("latent_w"), m("audio_t"), &[], &[]).unwrap();
        // the first step: noise level 1 for both streams, so one timestep (0) and three table rows
        let ts = Timesteps::new(&l, 1.0, 12.0, 3.0);
        assert_eq!(ts.values, vec![0.0]);
        let want: Vec<i32> = fx.read("mod_rows").unwrap().chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(ts.rows, want);
    }

    /// References and keyframes against ComfyUI's PackedLayout (tests/data/layout_refs.json: its segments and
    /// positions for pictures at their own sizes, a clip with and without sound, a sound, keyframes beside them)
    #[test]
    fn references_match_comfyui() {
        let raw = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/layout_refs.json")).unwrap();
        let cases: serde_json::Value = serde_json::from_str(&raw).unwrap();
        for c in cases.as_array().unwrap() {
            let u = |v: &serde_json::Value| v.as_u64().unwrap() as usize;
            let kfs: Vec<Keyframe> = c["keyframes"].as_array().unwrap().iter().map(|k| Keyframe {
                frame_index: k["index"].as_f64().unwrap(), video_latent_t: k["vt"].as_u64().map(|v| v as usize), audio_latent_t: k["rt"].as_u64().map(|v| v as usize),
            }).collect();
            let refs: Vec<RefBlock> = c["refs"].as_array().unwrap().iter().map(|r| match r["kind"].as_str().unwrap() {
                "audio" => RefBlock::Audio { t: u(&r["ref_audio_t"]) },
                "image" => RefBlock::Image { h: u(&r["latent_h"]), w: u(&r["latent_w"]) },
                _ => RefBlock::Video { t: u(&r["latent_t"]), h: u(&r["latent_h"]), w: u(&r["latent_w"]), audio_t: u(&r["ref_audio_t"]) },
            }).collect();
            let l = Layout::new(u(&c["text"]), u(&c["lt"]), u(&c["lh"]), u(&c["lw"]), u(&c["at"]), &kfs, &refs).unwrap();
            let kind = |k: &str| match k {
                "text" => Kind::Text, "cond" => Kind::Cond, "cond_audio" => Kind::CondAudio, "ref_img" => Kind::RefImg,
                "ref_audio" => Kind::RefAudio, "audio" => Kind::Audio, _ => Kind::Video,
            };
            let want: Vec<Segment> = c["segments"].as_array().unwrap().iter()
                .map(|s| Segment { start: u(&s[0]), stop: u(&s[1]), kind: kind(s[2].as_str().unwrap()) }).collect();
            assert_eq!(l.segments, want, "{}", c["name"]);
            let pos: Vec<f64> = c["positions"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            assert_eq!(l.positions.len(), pos.len(), "{}", c["name"]);
            let worst = l.positions.iter().zip(&pos).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-9, "{}: positions differ by {worst}", c["name"]);
        }
    }

    #[test]
    fn audio_schedule_and_curve() {
        // both ends of the schedule are fixed points of the shift
        assert!((time_shift_sigma(1.0, 12.0, 3.0) - 1.0).abs() < 1e-12);
        assert!(time_shift_sigma(0.0, 12.0, 3.0).abs() < 1e-12);
        // in between the audio stream is less noisy than the video stream
        assert!(time_shift_sigma(0.5, 12.0, 3.0) < 0.5);
        // the curve: a table of 5 rows, 2 wide; t = 0.375 is half-way between rows 1 and 2
        let table: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let l = Layout::new(1, 1, 2, 2, 1, &[], &[]).unwrap();
        let mut ts = Timesteps::new(&l, 1.0, 12.0, 3.0);
        ts.values = vec![0.375, 1.0];
        assert_eq!(ts.embeddings(&table, 2), vec![3.0, 4.0, 8.0, 9.0]);
    }
}
