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
//!
//! The prompt is most of a call's start (a voice of 39 s: 293 frames encoded, ~560 positions), so it is made fast
//! three ways (`StartPath`; each can be turned off for a check against the old path): the reference encoded
//! `ENC_FRAMES` frames a pass instead of one, the positions run `WIDE` a pass on the matrix engines instead of 16 on
//! the small-batch product, and the result kept per voice (`cache.rs`: a repeat call restores it instead).

use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::cache::{self, Entry, PromptKey};
use crate::codec::Stream;
use crate::lm::{State, ROWS, WIDE};
use crate::sample::{self, Detok, Dry, DryState, Rng};
use crate::{Model, ENC_FRAMES};

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
    /// the seed of the voice reference's latent noise, apart from `seed` (which then seeds the frames alone): a
    /// session that names no seed draws the reference's noise from a fixed one, so its prompt is the same call after
    /// call and the voice's cache serves it. None: one stream from `seed` for both (the release's; a seeded session)
    pub voice_seed: Option<u64>,
    /// the codec's latent noise drawn (the release's); false: the means (a check)
    pub latent_noise: bool,
}

/// How a session's prompt runs: the fast paths, each of which a check can turn off (all off: the path before them)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StartPath {
    /// the reference through the encoder ENC_FRAMES frames a pass (else one, as a live frame)
    pub batch_encode: bool,
    /// the positions WIDE a pass on the matrix engines (else ROWS on the small-batch product)
    pub wide: bool,
    /// the voice's cache: a prompt made once is restored after (and the reference's encoding kept)
    pub cache: bool,
}

impl Default for StartPath {
    fn default() -> StartPath {
        StartPath { batch_encode: true, wide: true, cache: true }
    }
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { guidance: 1.5, temperature: 0.7, top_k: 25, top_p: 1.0, dry: true, eop_bias: 0.0, speak_first: false, seed: 0, voice_seed: None,
                   latent_noise: true }
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
    /// the logits' argmax before any sampling (a check's)
    pub argmax: u32,
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
    /// the prompt this session's state holds, not yet moved on from (`prime`): `start` then only makes frame 0
    primed: Option<PromptKey>,
    /// the prompt's last hidden state (normed): the model's `hidden` is shared by every session, so frame 0 reads
    /// this copy
    hidden: Vec<f32>,
}

impl Session {
    /// The model's last frame's raw latent
    pub fn latent(&self) -> &[f32] {
        &self.latent
    }
}

/// Where a start's time went (seconds; Model::start_timed)
#[derive(Clone, Debug, Default)]
pub struct StartTimes {
    /// the reference through the codec's encoder (0 on a cache hit)
    pub encode: f64,
    /// the prompt's positions through the language model (or, on a hit, the state restored)
    pub prefill: f64,
    /// a hit: the cache's state put back
    pub restore: f64,
    /// a miss: the state copied into the cache
    pub store: f64,
    /// frame 0: the logits, the sampling, the head and the decoder
    pub frame0: f64,
    pub total: f64,
    pub hit: bool,
    pub positions: usize,
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
    /// The codec encoder's latent means of a voice reference (whole frames of 24 kHz audio, from a fresh stream):
    /// ENC_FRAMES frames a pass through the model's wide stream (StartPath::batch_encode), else a frame at a time
    pub fn encode_reference(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let f = self.cfg.frame;
        if !self.path.batch_encode {
            let s = self.enc.stream(&self.ops, f)?;
            return self.encode_frames(&s, pcm);
        }
        if !pcm.len().is_multiple_of(f) {
            return Err(Error(format!("{} samples: whole frames of {f}", pcm.len())));
        }
        let s = &self.enc_wide;
        self.enc.reset(s)?;
        let mut out = Vec::with_capacity(pcm.len() / f * self.cfg.latent);
        let (buf, at) = self.enc.input(s);
        for chunk in pcm.chunks(ENC_FRAMES * f) {
            buf.write(at * 4, &crate::weights::bytes(chunk))?;
            let (o, rows) = self.enc.push(&self.ops, s, None, chunk.len())?;
            if rows != chunk.len() / f {
                return Err(Error(format!("the encoder made {rows} latents of {} frames", chunk.len() / f)));
            }
            let mut v = vec![0u8; rows * self.cfg.latent * 4];
            o.read(0, &mut v)?;
            out.extend(v.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        self.ops.mark(|| "codec encode (reference)".into())?;
        Ok(out)
    }

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
            let fp = cache::fingerprint(r);
            let means = match self.path.cache.then(|| self.cache.means(fp)).flatten() {
                Some(m) => m,
                None => {
                    let mut pcm = r.to_vec();
                    pcm.resize(frames * c.frame, 0.0);
                    let m = self.encode_reference(&pcm)?;
                    if self.path.cache {
                        self.cache.keep_means(fp, &m);
                    }
                    m
                }
            };
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

    /// The prompt's rows through the language model, WIDE at a time (StartPath::wide; else ROWS); the last row's
    /// hidden state in `lm.hidden`
    pub fn prefill(&self, st: &mut State, p: &Prompt) -> Result<()> {
        self.prefill_rows(st, p, if self.path.wide { WIDE } else { ROWS })
    }

    /// The prompt's rows through the language model, `rows` a pass
    pub fn prefill_rows(&self, st: &mut State, p: &Prompt, rows: usize) -> Result<()> {
        let c = &self.cfg;
        let n = p.tokens.len();
        let mut s = 0;
        while s < n {
            let e = (s + rows).min(n);
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
            primed: None,
            hidden: Vec::new(),
        })
    }

    /// Run the prompt (card + voice reference: mono 24 kHz) and make frame 0
    pub fn start(&self, s: &mut Session, card: &str, reference: Option<&[f32]>) -> Result<Frame> {
        Ok(self.start_timed(s, card, reference)?.0)
    }

    /// `start`, and where its time went
    pub fn start_timed(&self, s: &mut Session, card: &str, reference: Option<&[f32]>) -> Result<(Frame, StartTimes)> {
        let t0 = Instant::now();
        let mut t = self.prime(s, card, reference)?;
        s.primed = None;
        s.force_text = if s.cfg.speak_first { 1 } else { 0 };
        s.index = 0;
        self.lm.hidden.write(0, &crate::weights::bytes(&s.hidden))?;
        let t1 = Instant::now();
        let f = self.emit(s, t0)?;
        t.frame0 = t1.elapsed().as_secs_f64();
        t.total = t0.elapsed().as_secs_f64();
        Ok((f, t))
    }

    /// The prompt into the session's state, ready for `start` (which then only makes frame 0): from the voice's
    /// cache when it holds this prompt, else run (and kept there). A session already primed with it does nothing.
    /// The session's random numbers start over from its seed (a seeded session gives the same call each time).
    pub fn prime(&self, s: &mut Session, card: &str, reference: Option<&[f32]>) -> Result<StartTimes> {
        let mut t = StartTimes::default();
        let key = PromptKey { voice: reference.filter(|r| !r.is_empty()).map(cache::fingerprint), card: card.to_string(), speak_first: s.cfg.speak_first,
                              noise: s.cfg.latent_noise, seed: s.cfg.voice_seed.unwrap_or(s.cfg.seed), path: (self.path.batch_encode, self.path.wide) };
        if s.primed.as_ref() == Some(&key) {
            t.hit = true;
            t.positions = s.prompt_len;
            return Ok(t);
        }
        s.primed = None;
        self.enc.reset(&s.enc)?;
        self.dec.reset(&s.dec)?;
        let hit = if self.path.cache { self.cache.get(&key) } else { None };
        let rng = match hit {
            Some(e) => {
                let t1 = Instant::now();
                if e.snapshot.pos + 16 > s.state.ctx {
                    return Err(Error(format!("a prompt of {} positions in a context of {} (a shorter voice reference)", e.snapshot.pos, s.state.ctx)));
                }
                s.state.restore(&e.snapshot)?;
                s.hidden = e.hidden.clone();
                s.prompt_len = e.snapshot.pos;
                t.restore = t1.elapsed().as_secs_f64();
                t.hit = true;
                e.rng.clone()
            }
            None => {
                self.lm.reset(&s.state)?;
                s.state.pos = 0;
                let mut rng = Rng::new(key.seed);
                let t1 = Instant::now();
                let p = self.prompt(card, reference, s.cfg.speak_first, s.cfg.latent_noise.then_some(&mut rng))?;
                self.ops.gpu.sync()?;
                t.encode = t1.elapsed().as_secs_f64();
                if p.tokens.len() + 16 > s.state.ctx {
                    return Err(Error(format!("a prompt of {} positions in a context of {} (a shorter voice reference)", p.tokens.len(), s.state.ctx)));
                }
                let t2 = Instant::now();
                self.prefill(&mut s.state, &p)?;
                s.hidden = self.lm.hidden.to_f32()?[..self.cfg.hidden].to_vec();
                t.prefill = t2.elapsed().as_secs_f64();
                s.prompt_len = p.tokens.len();
                if self.path.cache {
                    let t3 = Instant::now();
                    let e = Entry { snapshot: s.state.snapshot(self.cfg.kv_heads, self.cfg.head_dim)?, hidden: s.hidden.clone(), rng: rng.clone() };
                    self.cache.put(key.clone(), e);
                    t.store = t3.elapsed().as_secs_f64();
                }
                rng
            }
        };
        // the frames' random numbers: the prompt's stream on (one seed), or their own seed (a voice seed apart)
        s.rng = if s.cfg.voice_seed.is_some() { Rng::new(s.cfg.seed) } else { rng };
        t.positions = s.prompt_len;
        s.primed = Some(key);
        Ok(t)
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
        s.primed = None;
        let (m, u) = (self.scale(model), self.scale(user));
        self.lm.embed_rows(&self.ops, &[token], Some((0, &m)), Some((0, &u)))?;
        self.ops.mark(|| "embed".into())?;
        self.lm.forward(&self.ops, &mut s.state, 1, false)
    }

    /// The next frame from the hidden state in `lm.hidden`: its token (sampled), its latent (the head), its audio
    pub fn emit(&self, s: &mut Session, t0: Instant) -> Result<Frame> {
        let c = &self.cfg;
        let mut logits = self.lm.logits(&self.ops, 0)?;
        let argmax = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
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
        let f = Frame { audio, token, text, index: s.index, seconds: t0.elapsed().as_secs_f64(), argmax };
        s.index += 1;
        Ok(f)
    }

    /// The model's last frame taken as `token` and the raw latent `latent` instead of its own (a check's teacher
    /// forcing: the next position reads them)
    pub fn force(&self, s: &mut Session, token: u32, latent: &[f32]) {
        s.token = token;
        s.latent = latent.to_vec();
    }

    /// The model's next frame held silent (a cancel): PAD on its text stream, its room tone's latent on its audio -
    /// the position still runs, so it keeps listening; its own audio out is silence
    pub fn mute(&self, s: &mut Session) {
        s.token = self.cfg.pad;
        s.latent = self.silence.clone();
    }
}
