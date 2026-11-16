//! One conversation, frame by frame (haru/session.py's DuplexSession): `start` runs the prompt (the character card as
//! a ChatML system turn, then the voice reference as audio on the model's stream) and makes frame 0; `step` takes one
//! user frame (3,200 samples at 24 kHz) and makes the next. Frame t of the model comes from the hidden state at the
//! position before: its text token from the tied output embedding (DRY-penalized, sampled), its audio latent from
//! the MeanFlow head, decoded at once by the streaming codec. When user frame t arrives, the position
//! text[t] + model_audio[t] + user_audio[t] runs and gives frame t + 1 - the model stays one frame ahead.
//!
//! ```text
//!   position:  [0, P)          [P, S)                 [S, ...)
//!   text:      system card     PAD ... PAD EOP        the live text stream
//!   model:     -               reference latents      the model's audio
//!   user:      -               -                      the user's audio
//! ```

use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::codec::Stream;
use crate::lm::{State, ROWS};
use crate::sample::{self, Detok, Dry, DryState, Rng};
use crate::Model;

/// A session's settings (haru/session.py's DuplexConfig)
#[derive(Clone, Debug)]
pub struct Settings {
    /// audio guidance (1 - 2)
    pub guidance: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub dry: bool,
    /// added to the EOP logit: > 0 makes the model start speaking sooner
    pub eop_bias: f32,
    /// the model opens the conversation
    pub speak_first: bool,
    pub seed: u64,
    /// the codec's latent noise drawn (the release's); false: the means (a check)
    pub latent_noise: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { guidance: 1.5, temperature: 0.7, top_k: 25, top_p: 1.0, dry: true, eop_bias: 0.0, speak_first: false, seed: 0, latent_noise: true }
    }
}

/// One frame of the model: its 3,200 samples, its text token and the text it adds, and the time it took
#[derive(Clone, Debug)]
pub struct Frame {
    pub audio: Vec<f32>,
    pub token: u32,
    pub text: String,
    pub index: usize,
    pub seconds: f64,
}

/// The prompt's three streams (haru/prompt.py's build_prompt): latents raw (the codec's scale)
pub struct Prompt {
    pub tokens: Vec<u32>,
    /// (first row, rows of latent values) of each audio stream
    pub model: (usize, Vec<f32>),
    pub user: (usize, Vec<f32>),
}

pub struct Session {
    pub state: State,
    enc: Stream,
    dec: Stream,
    pub cfg: Settings,
    rng: Rng,
    token: u32,
    /// the model's last latent (raw)
    latent: Vec<f32>,
    force_text: usize,
    dry: Option<DryState>,
    detok: Detok,
    pub index: usize,
    pub prompt_len: usize,
}

/// Mono samples at 24 kHz: silence trimmed (below -45 dBFS in 20 ms windows, 0.2 s kept around the voice), the last
/// `max_s` seconds (haru/audio.py's prepare_reference)
pub fn prepare_reference(x: &[f32], max_s: f32) -> Vec<f32> {
    let win = 24000 / 50;
    let n = x.len() / win;
    let voiced: Vec<usize> = (0..n).filter(|i| {
        let w = &x[i * win..(i + 1) * win];
        let rms = (w.iter().map(|v| v * v).sum::<f32>() / win as f32 + 1e-12).sqrt();
        20.0 * rms.log10() > -45.0
    }).collect();
    let (Some(first), Some(last)) = (voiced.first(), voiced.last()) else { return if n == 0 { x.to_vec() } else { Vec::new() } };
    let margin = (0.2 * 24000.0) as usize;
    let a = (first * win).saturating_sub(margin);
    let b = ((last + 1) * win + margin).min(x.len());
    let v = &x[a..b];
    let keep = (max_s * 24000.0) as usize;
    v[v.len().saturating_sub(keep)..].to_vec()
}

impl Model {
    /// The codec encoder's latent means of whole frames of 24 kHz audio, a frame at a time through `s` (from its
    /// current state)
    pub fn encode_frames(&self, s: &Stream, pcm: &[f32]) -> Result<Vec<f32>> {
        let f = self.cfg.frame;
        if !pcm.len().is_multiple_of(f) {
            return Err(Error(format!("{} samples: whole frames of {f}", pcm.len())));
        }
        let mut out = Vec::with_capacity(pcm.len() / f * self.cfg.latent);
        let (buf, at) = self.enc.input(s);
        for chunk in pcm.chunks_exact(f) {
            buf.write(at * 4, &crate::weights::bytes(chunk))?;
            let (o, rows) = self.enc.push(&self.ops, s, None, f)?;
            if rows != 1 {
                return Err(Error(format!("the encoder made {rows} latents of a frame")));
            }
            out.extend(o.to_f32()?.into_iter().take(self.cfg.latent));
        }
        self.ops.mark(|| "codec encode".into())?;
        Ok(out)
    }

    /// One latent (raw) through the decoder stream `s`: its frame of audio, clamped to [-1, 1]
    pub fn decode_frame(&self, s: &Stream, latent: &[f32]) -> Result<Vec<f32>> {
        let (buf, at) = self.dec.input(s);
        buf.write(at * 4, &crate::weights::bytes(latent))?;
        let (o, rows) = self.dec.push(&self.ops, s, None, 1)?;
        let mut v = o.to_f32()?;
        v.truncate(rows);
        v.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
        self.ops.mark(|| "codec decode".into())?;
        Ok(v)
    }

    /// Codec latents -> the model's space: (raw + bias) * scale
    pub fn scale(&self, raw: &[f32]) -> Vec<f32> {
        raw.iter().map(|x| (x + self.latent_bias) * self.latent_scale).collect()
    }

    /// A latent drawn around a mean (haru's VibeVoiceCodec.sample: one noise scale a frame, std latent_std / 0.8)
    pub fn draw_latent(&self, mean: &[f32], rng: &mut Rng) -> Vec<f32> {
        let s = rng.normal() * self.cfg.latent_std / 0.8;
        mean.iter().map(|m| m + s * rng.normal()).collect()
    }

    /// The prompt: the card's ChatML ids, then the reference's frames (PAD ... EOP on the text stream, its latents on
    /// the model's), then (speak first) PAD, EOP over the speaker's room tone and the user's silence
    pub fn prompt(&self, card: &str, reference: Option<&[f32]>, speak_first: bool, rng: Option<&mut Rng>) -> Result<Prompt> {
        let c = &self.cfg;
        let ids: Vec<u32> = if card.is_empty() { Vec::new() } else { self.tok.encode(&format!("<|im_start|>system\n{card}<|im_end|>\n")) };
        let mut model = Vec::new();
        let mut room = self.silence.clone();
        let mut n_ref = 0;
        if let Some(r) = reference.filter(|r| !r.is_empty()) {
            let frames = r.len().div_ceil(c.frame);
            let mut pcm = r.to_vec();
            pcm.resize(frames * c.frame, 0.0);
            let s = self.enc.stream(&self.ops, c.frame)?;
            let means = self.encode_frames(&s, &pcm)?;
            // the quietest whole frame's latent: the speaker's room tone
            let whole = r.len() / c.frame;
            if whole > 0 {
                let q = (0..whole).min_by(|a, b| {
                    let e = |i: &usize| r[i * c.frame..(i + 1) * c.frame].iter().map(|x| x * x).sum::<f32>();
                    e(a).total_cmp(&e(b))
                }).unwrap_or(0);
                room = means[q * c.latent..(q + 1) * c.latent].to_vec();
            }
            match rng {
                Some(rng) => {
                    for m in means.chunks_exact(c.latent) {
                        model.extend(self.draw_latent(m, rng));
                    }
                    if whole > 0 {
                        let q = (0..whole).min_by(|a, b| {
                            let e = |i: &usize| r[i * c.frame..(i + 1) * c.frame].iter().map(|x| x * x).sum::<f32>();
                            e(a).total_cmp(&e(b))
                        }).unwrap_or(0);
                        room = model[q * c.latent..(q + 1) * c.latent].to_vec();
                    }
                }
                None => model = means,
            }
            n_ref = frames;
        }
        let mut grid = if speak_first { vec![c.pad, c.eop] } else { Vec::new() };
        if ids.is_empty() && n_ref == 0 && grid.is_empty() {
            grid = vec![c.pad];
        }
        let mut tokens = ids.clone();
        if n_ref > 0 {
            tokens.extend(std::iter::repeat_n(c.pad, n_ref - 1));
            tokens.push(c.eop);
        }
        tokens.extend(&grid);
        let mut user = Vec::new();
        for _ in 0..grid.len() {
            model.extend(&room);
            user.extend(&self.silence);
        }
        Ok(Prompt { model: (ids.len(), model), user: (tokens.len() - grid.len(), user), tokens })
    }

    /// The prompt's rows through the language model, ROWS at a time; the last row's hidden state in `lm.hidden`
    pub fn prefill(&self, st: &mut State, p: &Prompt) -> Result<()> {
        let c = &self.cfg;
        let n = p.tokens.len();
        let mut s = 0;
        while s < n {
            let e = (s + ROWS).min(n);
            let span = |(r0, v): &(usize, Vec<f32>)| -> Option<(usize, Vec<f32>)> {
                let rows = v.len() / c.latent;
                let (a, b) = ((*r0).max(s), (r0 + rows).min(e));
                (a < b).then(|| (a - s, self.scale(&v[(a - r0) * c.latent..(b - r0) * c.latent])))
            };
            let (m, u) = (span(&p.model), span(&p.user));
            self.lm.embed_rows(&self.ops, &p.tokens[s..e], m.as_ref().map(|(a, v)| (*a, v.as_slice())), u.as_ref().map(|(a, v)| (*a, v.as_slice())))?;
            self.lm.forward(&self.ops, st, e - s, false)?;
            s = e;
        }
        Ok(())
    }

    /// A new conversation's session (its state of `ctx` positions, its codec streams)
    pub fn session(&self, cfg: Settings, ctx: usize) -> Result<Session> {
        let c = &self.cfg;
        let silent = vec![c.pad, c.eop];
        let mut banned = silent.clone();
        banned.extend(c.eos);
        Ok(Session {
            state: self.lm.state(&self.ops, ctx)?,
            enc: self.enc.stream(&self.ops, c.frame)?,
            dec: self.dec.stream(&self.ops, 1)?,
            rng: Rng::new(cfg.seed),
            dry: cfg.dry.then(|| DryState::new(Dry::default(), banned)),
            cfg,
            token: c.pad,
            latent: vec![0.0; c.latent],
            force_text: 0,
            detok: Detok::new(silent),
            index: 0,
            prompt_len: 0,
        })
    }

    /// Run the prompt (card + voice reference: mono 24 kHz) and make frame 0
    pub fn start(&self, s: &mut Session, card: &str, reference: Option<&[f32]>) -> Result<Frame> {
        let t0 = Instant::now();
        self.lm.reset(&s.state)?;
        s.state.pos = 0;
        self.enc.reset(&s.enc)?;
        self.dec.reset(&s.dec)?;
        let p = self.prompt(card, reference, s.cfg.speak_first, s.cfg.latent_noise.then_some(&mut s.rng))?;
        if p.tokens.len() + 16 > s.state.ctx {
            return Err(Error(format!("a prompt of {} positions in a context of {} (a shorter voice reference)", p.tokens.len(), s.state.ctx)));
        }
        self.prefill(&mut s.state, &p)?;
        s.force_text = if s.cfg.speak_first { 1 } else { 0 };
        s.prompt_len = p.tokens.len();
        s.index = 0;
        self.emit(s, t0)
    }

    /// One user frame (frame samples at 24 kHz) in; the model's next frame out
    pub fn step(&self, s: &mut Session, user: &[f32]) -> Result<Frame> {
        let t0 = Instant::now();
        if user.len() != self.cfg.frame {
            return Err(Error(format!("a user frame is {} samples, not {}", self.cfg.frame, user.len())));
        }
        let mean = self.encode_frames(&s.enc, user)?;
        let ul = if s.cfg.latent_noise { self.draw_latent(&mean, &mut s.rng) } else { mean };
        self.advance(s, s.token, &s.latent.clone(), &ul)?;
        self.emit(s, t0)
    }

    /// The position of (token, the model's raw latent, the user's raw latent) through the model
    pub fn advance(&self, s: &mut Session, token: u32, model: &[f32], user: &[f32]) -> Result<()> {
        let (m, u) = (self.scale(model), self.scale(user));
        self.lm.embed_rows(&self.ops, &[token], Some((0, &m)), Some((0, &u)))?;
        self.ops.mark(|| "embed".into())?;
        self.lm.forward(&self.ops, &mut s.state, 1, false)
    }

    /// The next frame from the hidden state in `lm.hidden`: its token (sampled), its latent (the head), its audio
    pub fn emit(&self, s: &mut Session, t0: Instant) -> Result<Frame> {
        let c = &self.cfg;
        let mut logits = self.lm.logits(&self.ops, 0)?;
        if let Some(d) = &s.dry {
            d.apply(&mut logits);
        }
        if s.cfg.eop_bias != 0.0 {
            logits[c.eop as usize] += s.cfg.eop_bias;
        }
        if s.force_text > 0 {
            logits[c.pad as usize] = f32::NEG_INFINITY;
            logits[c.eop as usize] = f32::NEG_INFINITY;
            if let Some(e) = c.eos {
                logits[e as usize] = f32::NEG_INFINITY;
            }
            s.force_text -= 1;
        }
        let token = sample::draw(&logits, s.cfg.temperature, s.cfg.top_k, s.cfg.top_p, &mut s.rng);
        let noise = s.rng.normals(c.latent);
        let z = self.head.sample(&self.ops, self.lm.hidden.fp(), &noise, s.cfg.guidance)?;
        let raw: Vec<f32> = z.iter().map(|x| x / self.latent_scale - self.latent_bias).collect();
        let audio = self.decode_frame(&s.dec, &raw)?;
        s.token = token;
        s.latent = raw;
        if let Some(d) = &mut s.dry {
            d.update(token);
        }
        let text = s.detok.push(&self.tok, token);
        let f = Frame { audio, token, text, index: s.index, seconds: t0.elapsed().as_secs_f64() };
        s.index += 1;
        Ok(f)
    }

    /// The model's next frame held silent (a cancel): PAD on its text stream, its room tone's latent on its audio -
    /// the position still runs, so it keeps listening; its own audio out is silence
    pub fn mute(&self, s: &mut Session) {
        s.token = self.cfg.pad;
        s.latent = self.silence.clone();
    }
}
