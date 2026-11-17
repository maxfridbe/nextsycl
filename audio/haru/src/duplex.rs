//! A live conversation (the audio contract's `Conversation`) in lockstep, as the release's web demo runs one: the
//! client's audio is cut into 133 ms frames (3,200 samples at 24 kHz) as it comes, and every frame is answered by one
//! frame of the model's own - its audio (silence included) and the text it adds - within one long-running response.
//! There are no turns: the model hears and speaks at once, decides itself when to talk, back-channel or stop when cut
//! in on. The prompt (the character card, the voice) runs when the first audio arrives (or a response.create); a
//! session.update that changes the card, the voice, the language or the sampling starts the conversation over. A
//! voice is needed (a saved voice's recording): without one the model stays silent, so nothing starts and the client
//! gets an error saying so.
//!
//! - **Frames**: the worker takes each whole frame in order; the model is one frame ahead (frame 0 comes from the
//!   prompt), so frame k + 1 answers the user's frame k. A client that sends faster than real time gets the answers
//!   as fast as they are made; one that stops sending stops the model with it.
//! - **Speech events**: input_audio_buffer.speech_started / speech_stopped as an energy gate hears the user (a client's
//!   display; the model does not need them): a frame above max(0.01, 3 x the noise floor) starts, 4 quiet frames end.
//! - **Cancel**: response.cancel ends the response (cancelled) and silences the model: its text held at PAD and its
//!   own audio stream at the room tone while it keeps listening; nothing goes out until a response.create.
//! - **The prompt early**: once a session names its voice, the worker runs the prompt into the conversation's state
//!   (`Model::prime`) while the client is still getting ready, so the first audio needs only frame 0; a repeat voice
//!   comes from the voices' cache (cache.rs). A session without a seed draws the reference's noise from a fixed one
//!   (Settings::voice_seed), so its prompt is the same each call.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nextsycl_audio::{Conversation, Error, RealtimeEvent, RealtimeSession, Result};

use crate::resample;
use crate::session::{prepare_reference, Session, Settings};
use crate::{card, settings, Inner, RATE, VOICE_SEED};

/// What makes the prompt: when it changes, the conversation starts over
#[derive(Clone, PartialEq)]
struct Key {
    card: String,
    voice: Option<(usize, u64)>,
    guidance: u32,
    temperature: u32,
    top_k: usize,
    eop_bias: u32,
    speak_first: bool,
    dry: bool,
}

struct St {
    card: String,
    voice: Option<Arc<Vec<f32>>>,
    settings: Settings,
    key: Key,
    /// the key the running session was started with
    running: Option<Key>,
    /// the key the session's state was primed with (the prompt run ahead of the first audio)
    primed: Option<Key>,
    want_start: bool,
    /// the client's rate and its resampler to 24 kHz
    rs: Option<resample::Stream>,
    input: VecDeque<f32>,
    /// user frames taken, and the ms of audio they hold
    frames_in: u64,
    // the response going out
    muted: bool,
    response: Option<String>,
    resp_n: u64,
    text: String,
    sent: u64,
    // the energy gate
    noise: f32,
    speaking: bool,
    quiet: u32,
    full: bool,
    /// the client was told it needs a voice (once a setting)
    told: bool,
}

struct Shared {
    eng: Arc<Inner>,
    st: Mutex<St>,
    wake: Condvar,
    events: Mutex<VecDeque<RealtimeEvent>>,
    ev: Condvar,
    closed: AtomicBool,
}

pub struct Duplex {
    s: Arc<Shared>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn hash(v: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v.iter().step_by(97) {
        h = (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// The card's text, the voice (24 kHz), the settings and their key
type Conf = (String, Option<Arc<Vec<f32>>>, Settings, Key);

/// The settings, card and voice a session's options make
fn conf(eng: &Inner, c: &RealtimeSession) -> Result<Conf> {
    let lang = card::language(c.language.as_deref());
    let cardv = match c.card.as_deref() {
        Some(j) => Some(serde_json::from_str::<serde_json::Value>(j).map_err(|e| Error(format!("nextsycl.card: {e}")))?),
        None => None,
    };
    let text = card::system_prompt(cardv.as_ref(), c.instructions.as_deref(), &lang)?;
    let voice = c.reference.as_ref().map(|r| Arc::new(prepare_reference(&resample::resample(&r.samples, r.rate, RATE), 180.0)));
    let mut s = settings(&c.extra, None, &eng.defaults)?;
    if let Some(t) = c.temperature {
        s.temperature = t.clamp(0.0, 2.0);
    }
    s.seed = if c.seed != 0 { c.seed } else { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64) };
    s.voice_seed = (c.seed == 0).then_some(VOICE_SEED);
    let key = Key { card: text.clone(), voice: voice.as_ref().map(|v| (v.len(), hash(v))), guidance: s.guidance.to_bits(), temperature: s.temperature.to_bits(),
                    top_k: s.top_k, eop_bias: s.eop_bias.to_bits(), speak_first: s.speak_first, dry: s.dry };
    Ok((text, voice, s, key))
}

impl Duplex {
    pub fn start(eng: Arc<Inner>, c: &RealtimeSession) -> Result<Duplex> {
        let (card, voice, settings, key) = conf(&eng, c)?;
        let st = St { card, voice, settings, key, running: None, primed: None, want_start: false, rs: None, input: VecDeque::new(), frames_in: 0, muted: false, response: None,
                      resp_n: 0, text: String::new(), sent: 0, noise: 0.003, speaking: false, quiet: 0, full: false,
                      told: false };
        let s = Arc::new(Shared { eng, st: Mutex::new(st), wake: Condvar::new(), events: Mutex::new(VecDeque::new()), ev: Condvar::new(),
                                  closed: AtomicBool::new(false) });
        let w = s.clone();
        std::thread::Builder::new().name("haru-duplex".into()).spawn(move || run(w)).map_err(|e| Error(format!("a conversation's thread: {e}")))?;
        Ok(Duplex { s })
    }
}

impl Drop for Duplex {
    fn drop(&mut self) {
        self.close();
    }
}

impl Shared {
    fn emit(&self, e: RealtimeEvent) {
        lock(&self.events).push_back(e);
        self.ev.notify_all();
    }

    /// A new response (unless one runs or the model is silenced)
    fn begin(&self, st: &mut St) {
        if st.response.is_none() && !st.muted {
            st.resp_n += 1;
            let id = format!("resp_haru{:04}", st.resp_n);
            st.text.clear();
            st.sent = 0;
            st.response = Some(id.clone());
            self.emit(RealtimeEvent::ResponseStarted { id });
        }
    }

    fn finish(&self, st: &mut St, status: &str) {
        if let Some(id) = st.response.take() {
            self.emit(RealtimeEvent::ResponseDone { id, status: status.into(), text: std::mem::take(&mut st.text), audio_ms: st.sent * 1000 / RATE as u64 });
        }
    }
}

impl Conversation for Duplex {
    fn update(&self, c: &RealtimeSession) -> Result<()> {
        let (card, voice, settings, key) = conf(&self.s.eng, c)?;
        let mut st = lock(&self.s.st);
        st.card = card;
        st.voice = voice;
        st.settings = settings;
        st.key = key;
        st.told = false;
        drop(st);
        self.s.wake.notify_all();
        Ok(())
    }

    fn append(&self, samples: &[f32], rate: u32) -> Result<()> {
        let mut st = lock(&self.s.st);
        if st.rs.as_ref().is_none_or(|r| r.rate() != rate) {
            st.rs = Some(resample::Stream::new(rate, RATE));
        }
        let x = st.rs.as_mut().map(|r| r.push(samples)).unwrap_or_default();
        st.input.extend(x);
        st.want_start = true;
        drop(st);
        self.s.wake.notify_all();
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        Ok(())
    }

    fn clear(&self) {
        lock(&self.s.st).input.clear();
    }

    fn respond(&self) -> Result<()> {
        let mut st = lock(&self.s.st);
        st.muted = false;
        st.want_start = true;
        if st.running.is_some() {
            self.s.begin(&mut st);
        }
        drop(st);
        self.s.wake.notify_all();
        Ok(())
    }

    fn cancel(&self) {
        let mut st = lock(&self.s.st);
        st.muted = true;
        self.s.finish(&mut st, "cancelled");
    }

    fn truncate(&self, _response: &str, _audio_ms: u64) {}

    fn tool_output(&self, _call_id: &str, _output: &str) -> Result<()> {
        Err(Error("haru calls no functions".into()))
    }

    fn add_text(&self, _role: &str, _text: &str) -> Result<()> {
        Err(Error("haru hears audio only: a turn given as text is not taken (speak it)".into()))
    }

    fn next(&self, wait: Duration) -> Option<RealtimeEvent> {
        let mut q = lock(&self.s.events);
        if q.is_empty() {
            q = self.s.ev.wait_timeout(q, wait).map(|(g, _)| g).unwrap_or_else(|e| e.into_inner().0);
        }
        q.pop_front()
    }

    fn close(&self) {
        self.s.closed.store(true, Ordering::SeqCst);
        self.s.wake.notify_all();
    }
}

/// What the worker does next
enum Work {
    Start(String, Option<Arc<Vec<f32>>>, Settings, Key),
    /// the prompt ahead of the start (a voice named, no audio yet)
    Prime(String, Option<Arc<Vec<f32>>>, Settings, Key),
    Frame(Vec<f32>),
}

/// The worker: the prompt when asked for (or again when the settings change), then each whole frame in order
fn run(s: Arc<Shared>) {
    let mut ses: Option<Session> = None;
    let (mut n, mut sum, mut worst, mut since) = (0u64, 0f64, 0f64, Instant::now());
    let fs = lock(&s.eng.model).cfg.frame;
    loop {
        let work = {
            let mut st = lock(&s.st);
            loop {
                if s.closed.load(Ordering::SeqCst) {
                    s.finish(&mut st, "completed");
                    return;
                }
                // the release speaks only in a voice it is given (its server refuses a session without one): without
                // one it stays silent, so the client is told instead
                if st.want_start && st.voice.is_none() && !st.told {
                    st.told = true;
                    s.emit(RealtimeEvent::Error { message: "haru speaks in a voice it is given: name a saved voice (session.update {voice: NAME}: 3 s to 3 \
                                                             minutes of one speaker)".into() });
                }
                if st.want_start && st.voice.is_some() && st.running.as_ref() != Some(&st.key) && !st.full {
                    let k = st.key.clone();
                    break Work::Start(st.card.clone(), st.voice.clone(), st.settings.clone(), k);
                }
                if st.running.is_some() && st.input.len() >= fs && !st.full {
                    let f: Vec<f32> = st.input.drain(..fs).collect();
                    break Work::Frame(f);
                }
                if !st.want_start && st.voice.is_some() && st.running.as_ref() != Some(&st.key) && st.primed.as_ref() != Some(&st.key) && !st.full {
                    let k = st.key.clone();
                    st.primed = Some(k.clone());
                    break Work::Prime(st.card.clone(), st.voice.clone(), st.settings.clone(), k);
                }
                st = s.wake.wait_timeout(st, Duration::from_millis(200)).map(|(g, _)| g).unwrap_or_else(|e| e.into_inner().0);
            }
        };
        match work {
            Work::Prime(cardt, voice, set, _key) => {
                let r = (|| -> Result<crate::session::StartTimes> {
                    let m = lock(&s.eng.model);
                    if ses.is_none() {
                        ses = Some(m.session(set.clone(), s.eng.ctx)?);
                    }
                    let se = ses.as_mut().ok_or_else(|| Error("no session".into()))?;
                    se.cfg = set.clone();
                    m.prime(se, &cardt, voice.as_deref().map(|v| v.as_slice()))
                })();
                match r {
                    Ok(t) => eprintln!("haru: the prompt ahead: {} positions in {:.2} s ({})", t.positions, t.encode + t.prefill + t.restore + t.store,
                                       if t.hit { "the voice's cache" } else { "encoded and run" }),
                    // the start runs it again and reports what fails
                    Err(e) => eprintln!("haru: the prompt ahead: {}", e.0),
                }
            }
            Work::Start(cardt, voice, set, key) => {
                let t0 = Instant::now();
                let r = (|| -> Result<(crate::session::Frame, crate::session::StartTimes)> {
                    let m = lock(&s.eng.model);
                    if ses.is_none() {
                        ses = Some(m.session(set.clone(), s.eng.ctx)?);
                    }
                    let se = ses.as_mut().ok_or_else(|| Error("no session".into()))?;
                    se.cfg = set.clone();
                    m.start_timed(se, &cardt, voice.as_deref().map(|v| v.as_slice()))
                })();
                let mut st = lock(&s.st);
                s.finish(&mut st, "completed");
                match r {
                    Ok((f, t)) => {
                        eprintln!("haru: a conversation started: prompt of {} positions in {:.2} s ({}; encode {:.0} ms, prefill {:.0} ms, restore {:.0} ms, \
                                   store {:.0} ms, frame 0 {:.0} ms)", t.positions, t0.elapsed().as_secs_f64(),
                                  if t.encode + t.prefill + t.restore == 0.0 { "primed ahead" } else if t.hit { "the voice's cache" } else { "cold" },
                                  1e3 * t.encode, 1e3 * t.prefill, 1e3 * t.restore, 1e3 * t.store, 1e3 * t.frame0);
                        st.running = Some(key);
                        s.begin(&mut st);
                        out(&s, &mut st, f, &mut ses);
                    }
                    Err(e) => {
                        st.want_start = false;
                        st.running = None;
                        s.emit(RealtimeEvent::Error { message: format!("haru: the prompt: {}", e.0) });
                    }
                }
            }
            Work::Frame(pcm) => {
                gate(&s, &pcm);
                let r = (|| -> Result<crate::session::Frame> {
                    let m = lock(&s.eng.model);
                    let se = ses.as_mut().ok_or_else(|| Error("no session".into()))?;
                    let muted = lock(&s.st).muted;
                    let f = m.step(se, &pcm)?;
                    if muted {
                        m.mute(se);
                    }
                    Ok(f)
                })();
                let mut st = lock(&s.st);
                st.frames_in += 1;
                match r {
                    Ok(f) => {
                        n += 1;
                        sum += f.seconds;
                        worst = worst.max(f.seconds);
                        if since.elapsed().as_secs_f64() >= 30.0 {
                            eprintln!("haru: {n} frames, {:.1} ms a frame (worst {:.1} ms; real time is 133 ms)", 1e3 * sum / n as f64, 1e3 * worst);
                            (n, sum, worst, since) = (0, 0.0, 0.0, Instant::now());
                        }
                        out(&s, &mut st, f, &mut ses);
                    }
                    Err(e) => {
                        st.full = true;
                        s.finish(&mut st, "failed");
                        s.emit(RealtimeEvent::Error { message: format!("haru: {} (the conversation stops; start a new one)", e.0) });
                    }
                }
            }
        }
    }
}

/// A frame of the model out: its text and audio within the response (none while silenced)
fn out(s: &Shared, st: &mut St, f: crate::session::Frame, _ses: &mut Option<Session>) {
    let Some(id) = st.response.clone() else { return };
    if !f.text.is_empty() {
        st.text.push_str(&f.text);
        s.emit(RealtimeEvent::TextDelta { id: id.clone(), delta: f.text });
    }
    st.sent += f.audio.len() as u64;
    s.emit(RealtimeEvent::AudioDelta { id, samples: f.audio });
}

/// The energy gate over a user frame: speech_started / speech_stopped for the client's display
fn gate(s: &Shared, pcm: &[f32]) {
    let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len().max(1) as f32).sqrt();
    let mut st = lock(&s.st);
    let at_ms = st.frames_in * 1000 * pcm.len() as u64 / RATE as u64;
    let thr = (3.0 * st.noise).max(0.01);
    if rms > thr {
        st.quiet = 0;
        if !st.speaking {
            st.speaking = true;
            s.emit(RealtimeEvent::SpeechStarted { audio_ms: at_ms });
        }
    } else {
        st.noise = 0.95 * st.noise + 0.05 * rms;
        if st.speaking {
            st.quiet += 1;
            if st.quiet >= 4 {
                st.speaking = false;
                s.emit(RealtimeEvent::SpeechStopped { audio_ms: at_ms.saturating_sub(3 * 1000 * pcm.len() as u64 / RATE as u64) });
            }
        }
    }
}
