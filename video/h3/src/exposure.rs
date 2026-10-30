//! A clip's exposure fixed after the decode, per request (`exposure_fix`): the video decoder darkens every frame by a
//! near-constant amount, and the frames just after a motion guide drift in brightness for a few frames (the denoiser
//! repaints them: a seed-dependent flash). The frames are lifted by `offset`, the guide's own frames are matched to
//! the guide's luma, and the frames after it ease from the guide's last luma to the clip's steady level.

/// How the correction is made (all of it from the request)
#[derive(Clone, Debug, PartialEq)]
pub struct Exposure {
    /// luma added to every frame, in 0..255 units (the decoder's darkening: about 2.3)
    pub offset: f32,
    /// frames after the anchor over which the target eases from the anchor's last luma to the clip's steady level
    pub ease: usize,
    /// the luma (0..1) each frame from `anchor_at` should have: the guide's frames, or one picture's
    pub anchor: Vec<f32>,
    pub anchor_at: usize,
}

/// Per-frame luma (0..1) before and after
#[derive(Debug, Default)]
pub struct Report {
    pub before: Vec<f32>,
    pub after: Vec<f32>,
    pub corrected: usize,
}

/// The gain on one frame is held to what a drift needs, as the picture exposure match does
const GAIN: (f32, f32) = (0.75, 1.35);

/// Luma (Rec. 601) of every frame of planar pixels [3, frames, plane] in [0, 1]
pub fn frame_luma(px: &[f32], frames: usize, plane: usize) -> Vec<f32> {
    (0..frames)
        .map(|f| {
            let c = |ch: usize| px[(ch * frames + f) * plane..][..plane].iter().sum::<f32>() / plane as f32;
            0.299 * c(0) + 0.587 * c(1) + 0.114 * c(2)
        })
        .collect()
}

fn median(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

impl Exposure {
    /// The luma each frame should have, none where the frame is left alone
    fn targets(&self, luma: &[f32]) -> Vec<Option<f32>> {
        let n = luma.len();
        let mut want = vec![None; n];
        let (a0, len) = (self.anchor_at.min(n), self.anchor.len());
        for (i, l) in self.anchor.iter().enumerate().take(n - a0) {
            want[a0 + i] = Some(*l);
        }
        let after = a0 + len;
        if len > 0 && self.ease > 0 && after < n {
            // the steady level: the median over the second after the ease
            let (s, e) = ((after + self.ease).min(n), (after + self.ease + 24).min(n));
            let base = if s < e { median(&luma[s..e]) } else { luma[n - 1] };
            let last = self.anchor[len - 1];
            for k in 0..self.ease.min(n - after) {
                want[after + k] = Some(last + (base - last) * (k + 1) as f32 / self.ease as f32);
            }
        }
        want
    }

    /// Correct planar pixels [3, frames, plane] in place
    pub fn apply(&self, px: &mut [f32], frames: usize, plane: usize) -> Report {
        let mut rep = Report { before: frame_luma(px, frames, plane), ..Report::default() };
        if self.offset != 0.0 {
            let o = self.offset / 255.0;
            px.iter_mut().for_each(|v| *v = (*v + o).clamp(0.0, 1.0));
        }
        let luma = frame_luma(px, frames, plane);
        for (f, want) in self.targets(&luma).into_iter().enumerate() {
            let Some(want) = want else { continue };
            if luma[f] < 1e-4 {
                continue;
            }
            let g = (want / luma[f]).clamp(GAIN.0, GAIN.1);
            if (g - 1.0).abs() > 1e-4 {
                rep.corrected += 1;
            }
            for ch in 0..3 {
                px[(ch * frames + f) * plane..][..plane].iter_mut().for_each(|v| *v = (*v * g).clamp(0.0, 1.0));
            }
        }
        rep.after = frame_luma(px, frames, plane);
        rep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// grey frames of the given luma, planar
    fn grey(levels: &[f32], plane: usize) -> Vec<f32> {
        let frames = levels.len();
        let mut px = vec![0.0; 3 * frames * plane];
        for ch in 0..3 {
            for (f, l) in levels.iter().enumerate() {
                px[(ch * frames + f) * plane..][..plane].fill(*l);
            }
        }
        px
    }

    #[test]
    fn a_flash_after_the_guide_is_eased_back_and_the_darkening_lifted() {
        // 60 frames at 0.30, the decoder's darkening (-0.009) and a +0.05 flash on frames 19..23
        let mut levels = vec![0.30 - 0.009; 60];
        levels[19..23].iter_mut().for_each(|l| *l += 0.05);
        let mut px = grey(&levels, 4);
        // the guide: 22 frames at 0.30
        let e = Exposure { offset: 2.3, ease: 24, anchor: vec![0.30; 22], anchor_at: 0 };
        let r = e.apply(&mut px, 60, 4);
        let flash = r.before[19..23].iter().cloned().fold(0.0, f32::max) - r.before[10];
        assert!(flash > 0.04);
        // the guide's span matches the guide, the flash is gone, the steady level is only lifted
        assert!(r.after[..22].iter().all(|l| (l - 0.30).abs() < 0.004), "{:?}", &r.after[..22]);
        let spread = r.after[..60].iter().cloned().fold(0.0, f32::max) - r.after[..60].iter().cloned().fold(1.0, f32::min);
        assert!(spread < 0.02, "{spread}");
        assert!((r.after[50] - (0.291 + 2.3 / 255.0)).abs() < 1e-3);
        assert!(r.corrected > 0);
    }

    #[test]
    fn nothing_to_do_changes_nothing() {
        let mut px = grey(&[0.4; 30], 4);
        let want = px.clone();
        let r = Exposure { offset: 0.0, ease: 24, anchor: Vec::new(), anchor_at: 0 }.apply(&mut px, 30, 4);
        assert_eq!(px, want);
        assert_eq!(r.corrected, 0);
    }
}
