//! A live conversation (the audio contract's `Conversation`), as the release's duplex.py runs one in vLLM-Omni:
//!
//! - **Listening**: the client's audio in 20 ms frames against an adaptive noise floor (threshold max(0.012, 3.5 x
//!   noise)); 100 ms of voice starts the user's turn (with the 0.6 s before it), its speech kept.
//! - **Turn-taking**: after the user pauses, at 0.2 / 0.35 / 0.5 / 0.8 / 1.3 / 2.0 s, the brain is asked the turn
//!   question over the turn's speech (what it said last as context); P(finished) over the four option letters against
//!   that pause's threshold (0.95 ... 0.25) answers; at 2 s of silence it answers anyway. Speech again voids the checks.
//! - **Replying**: system + history (each earlier turn's speech, and Sophon's own reply as the audio it spoke - or its
//!   text when cut off or calling a tool) + this turn's speech; the brain writes, the talker speaks as it writes; text
//!   and audio go out as they come.
//! - **Barge-in**: the user starting to speak while a reply is made or played stops it; the history keeps what was
//!   heard (the played share of its text, "……（被用户打断）" after it). The user speaking again within 2 s of a reply
//!   means it cut in too early: that reply is dropped and the two stretches of speech become one turn.
//! - **Tools**: a reply's <tool_call>s go out as function calls; their results (tool_output) and a response.create (or
//!   a second's wait) bring the spoken answer.
//!
//! Three threads: the listener (frames, decisions), the brain's work (a check or a reply's text) and the talker's.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nextsycl_audio::{Conversation, Error, RealtimeEvent, RealtimeSession, Result};

use crate::brain::Rows;
use crate::sample::{self, Rng};
use crate::text::{self, Msg};
use crate::{ear, resample, speak, Inner, Mind};

const AUDIO_ID: u32 = 151669;
const STOP: [u32; 2] = [151645, 151643];
/// a key's tag for an audio position (clip << 20 | row)
const AUDIO_KEY: u64 = 1 << 62;
const MERGE_WINDOW: f64 = 2.0;
const HIST_TURNS: usize = 8;
const HIST_SEC: f64 = 120.0;
/// talker.py / proc.py: text tokens in hand before the talker starts
const LEAD: usize = 3;

#[derive(Clone)]
struct Clip {
    id: u64,
    /// 16 kHz
    s16: Arc<Vec<f32>>,
}

#[derive(Clone, Default)]
struct Reply {
    id: String,
    /// its text (tags kept), cut short when the user cut in
    text: String,
    /// what it spoke (16 kHz), when it was spoken whole and called no tool
    audio: Option<Clip>,
    /// (call id, name, arguments)
    calls: Vec<(String, String, String)>,
}

#[derive(Clone, Default)]
struct Turn {
    user: Option<Clip>,
    user_text: Option<String>,
    replies: Vec<Reply>,
    /// the results of replies[i]'s calls, (call id, output)
    results: Vec<Vec<(String, String)>>,
}

/// The reply being made
struct Active {
    stop: Arc<AtomicBool>,
    /// when its first audio went out, and how much has gone
    first: Option<Instant>,
    sent_s: f64,
}

/// A finished reply still playing on the client (its audio went out faster than real time)
struct Playing {
    turn: usize,
    reply: usize,
    first: Instant,
    dur: f64,
}

struct State {
    cfg: RealtimeSession,
    lang: String,
    xvec: Option<Vec<f32>>,
    tools: Vec<serde_json::Value>,
    /// the input rate and the samples not yet framed
    rate: u32,
    rem: Vec<f32>,
    input_ms: u64,
    // the listener (duplex.py's SophonBuffer)
    noise: f32,
    voiced_ms: u32,
    speech_ms: u32,
    silence_ms: u32,
    sil_at_ms: u64,
    utt: Vec<f32>,
    pre: VecDeque<f32>,
    next_check: usize,
    check_inflight: bool,
    pending: bool,
    gen: u64,
    /// the user's previous turn and when it was answered (premature-reply recovery)
    prev_user: Option<(Vec<f32>, Instant, String)>,
    // the conversation
    turns: Vec<Turn>,
    active: Option<Active>,
    playing: Option<Playing>,
    last_said: String,
    /// a reply was asked for (commit, response.create, a tool result)
    want_reply: bool,
    followup_at: Option<Instant>,
    n: u64,
}

struct Shared {
    eng: Arc<Inner>,
    st: Mutex<State>,
    wake: Condvar,
    events: Mutex<VecDeque<RealtimeEvent>>,
    ev: Condvar,
    closed: AtomicBool,
    embs: Mutex<HashMap<u64, Arc<Vec<f32>>>>,
}

static CLIPS: AtomicU64 = AtomicU64::new(1);

fn clip(s16: Vec<f32>) -> Clip {
    Clip { id: CLIPS.fetch_add(1, Ordering::Relaxed), s16: Arc::new(s16) }
}

pub struct Duplex {
    s: Arc<Shared>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Duplex {
    pub fn start(eng: Arc<Inner>, cfg: &RealtimeSession) -> Result<Duplex> {
        let st = State {
            cfg: RealtimeSession::default(),
            lang: "en".into(),
            xvec: None,
            tools: Vec::new(),
            rate: ear::RATE,
            rem: Vec::new(),
            input_ms: 0,
            noise: 0.003,
            voiced_ms: 0,
            speech_ms: 0,
            silence_ms: 0,
            sil_at_ms: 0,
            utt: Vec::new(),
            pre: VecDeque::new(),
            next_check: 0,
            check_inflight: false,
            pending: false,
            gen: 0,
            prev_user: None,
            turns: Vec::new(),
            active: None,
            playing: None,
            last_said: String::new(),
            want_reply: false,
            followup_at: None,
            n: 0,
        };
        let s = Arc::new(Shared {
            eng,
            st: Mutex::new(st),
            wake: Condvar::new(),
            events: Mutex::new(VecDeque::new()),
            ev: Condvar::new(),
            closed: AtomicBool::new(false),
            embs: Mutex::new(HashMap::new()),
        });
        let d = Duplex { s: s.clone() };
        d.update(cfg)?;
        std::thread::Builder::new().name("sophon-listen".into()).spawn(move || listen(s)).map_err(|e| Error(e.to_string()))?;
        Ok(d)
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

    fn id(&self, st: &mut State, what: &str) -> String {
        st.n += 1;
        format!("{what}_{:06}", st.n)
    }
}

impl Conversation for Duplex {
    fn update(&self, c: &RealtimeSession) -> Result<()> {
        let eng = &self.s.eng;
        let lang = c.language.as_deref().filter(|l| !l.is_empty()).unwrap_or("en").to_lowercase();
        let lang = lang.chars().take(2).collect::<String>();
        let xvec = match &c.reference {
            Some(r) => Some(lock(&eng.talk).as_ref().ok_or_else(|| Error("loaded without its talker".into()))?.clone_voice(&r.samples, r.rate)?),
            None => Some(eng.voice(c.voice.as_deref(), &lang)?),
        };
        let tools: Vec<serde_json::Value> = c.tools.iter().filter_map(|t| serde_json::from_str(t).ok()).collect();
        let tools = text::normalize_tools(&tools);
        let mut st = lock(&self.s.st);
        st.cfg = c.clone();
        st.lang = lang;
        st.xvec = xvec;
        st.tools = tools;
        Ok(())
    }

    fn append(&self, samples: &[f32], rate: u32) -> Result<()> {
        if rate == 0 {
            return Err(Error("audio at rate 0".into()));
        }
        let mut st = lock(&self.s.st);
        if st.rem.is_empty() && st.utt.is_empty() && st.pre.is_empty() {
            st.rate = rate;
        }
        let x = if rate == st.rate { samples.to_vec() } else { resample::resample(samples, rate, st.rate) };
        st.rem.extend_from_slice(&x);
        drop(st);
        self.s.wake.notify_all();
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        let mut st = lock(&self.s.st);
        st.want_reply = true;
        drop(st);
        self.s.wake.notify_all();
        Ok(())
    }

    fn clear(&self) {
        let mut st = lock(&self.s.st);
        st.utt.clear();
        st.rem.clear();
        st.speech_ms = 0;
        st.silence_ms = 0;
    }

    fn respond(&self) -> Result<()> {
        self.commit()
    }

    fn cancel(&self) {
        let st = lock(&self.s.st);
        if let Some(a) = &st.active {
            a.stop.store(true, Ordering::SeqCst);
        }
    }

    fn truncate(&self, response: &str, audio_ms: u64) {
        let mut st = lock(&self.s.st);
        let st = &mut *st;
        for t in st.turns.iter_mut().rev() {
            if let Some(r) = t.replies.iter_mut().find(|r| r.id == response) {
                cut_reply(r, audio_ms as f64 / 1000.0);
                st.last_said = r.text.trim_end_matches(text::CUT).to_string();
                break;
            }
        }
        st.playing = None;
    }

    fn tool_output(&self, call_id: &str, output: &str) -> Result<()> {
        let mut guard = lock(&self.s.st);
        let st = &mut *guard;
        let t = st.turns.iter_mut().rev().find(|t| t.replies.iter().any(|r| r.calls.iter().any(|c| c.0 == call_id)))
            .ok_or_else(|| Error(format!("call {call_id}: no such call in this conversation")))?;
        let i = t.replies.iter().position(|r| r.calls.iter().any(|c| c.0 == call_id)).unwrap_or(0);
        while t.results.len() <= i {
            t.results.push(Vec::new());
        }
        t.results[i].push((call_id.to_string(), output.to_string()));
        // answered when the client asks (response.create), or after a second
        st.followup_at = Some(Instant::now() + Duration::from_secs(1));
        drop(guard);
        self.s.wake.notify_all();
        Ok(())
    }

    fn add_text(&self, role: &str, t: &str) -> Result<()> {
        let mut st = lock(&self.s.st);
        match role {
            "user" => st.turns.push(Turn { user_text: Some(t.to_string()), ..Default::default() }),
            "assistant" => {
                let id = format!("item_text_{}", st.turns.len());
                match st.turns.last_mut() {
                    Some(last) if last.replies.is_empty() => last.replies.push(Reply { id, text: t.to_string(), ..Default::default() }),
                    _ => st.cfg.history.push(("assistant".into(), t.to_string())),
                }
            }
            r => return Err(Error(format!("role {r:?}: user or assistant"))),
        }
        Ok(())
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
        if let Some(a) = &lock(&self.s.st).active {
            a.stop.store(true, Ordering::SeqCst);
        }
        self.s.wake.notify_all();
        self.s.ev.notify_all();
    }
}

/// A reply cut at `played` seconds of its audio: its text's share, the cut marker after it, the audio kept when
/// over half a second was heard
fn cut_reply(r: &mut Reply, played: f64) {
    let dur = r.audio.as_ref().map(|a| a.s16.len() as f64 / ear::RATE as f64).unwrap_or(0.0);
    if dur <= 0.0 || played >= dur - 0.3 {
        return;
    }
    let chars: Vec<char> = r.text.chars().collect();
    let n = (chars.len() as f64 * (played / dur).clamp(0.0, 1.0)) as usize;
    let mut said: String = chars[..n].iter().collect();
    // back to a word boundary (never mid-word in English)
    if said.chars().last().is_some_and(|c| c.is_ascii_alphanumeric()) {
        if let Some(k) = said.rfind(' ') {
            said.truncate(k + 1);
        }
    }
    r.text = format!("{said}{}", text::CUT);
    r.audio = r.audio.take().filter(|_| played > 0.5).map(|a| clip(a.s16[..((played * ear::RATE as f64) as usize).min(a.s16.len())].to_vec()));
}

/// The listener: frames in, decisions out (duplex.py's prepare_append, as a loop)
fn listen(s: Arc<Shared>) {
    let mut last = Instant::now();
    while !s.closed.load(Ordering::SeqCst) {
        let mut st = lock(&s.st);
        // nothing to frame and nothing asked: wait (a tool follow-up's time is checked every 20 ms)
        if st.rem.is_empty() && !st.want_reply {
            st = s.wake.wait_timeout(st, Duration::from_millis(20)).map(|(g, _)| g).unwrap_or_else(|e| e.into_inner().0);
        }
        let frame = (st.rate / 50).max(1) as usize;
        let checks = s.eng.turn_checks.clone();
        let thrs = s.eng.turn_thresholds.clone();
        let fallback = st.cfg.silence_s.unwrap_or(2.0);
        while st.rem.len() >= frame {
            let f: Vec<f32> = st.rem.drain(..frame).collect();
            st.input_ms += 20;
            let r = (f.iter().map(|x| x * x).sum::<f32>() / frame as f32).sqrt();
            let thr = (st.noise * 3.5).max(0.012);
            if st.speech_ms == 0 {
                st.pre.extend(f.iter().copied());
                let keep = (st.rate as f32 * 0.6) as usize;
                while st.pre.len() > keep {
                    st.pre.pop_front();
                }
                if r > thr {
                    st.voiced_ms += 20;
                    if st.voiced_ms >= 100 {
                        onset(&s, &mut st);
                    }
                } else {
                    st.voiced_ms = 0;
                    st.noise = 0.98 * st.noise + 0.02 * r.min(0.05);
                }
            } else {
                st.utt.extend_from_slice(&f);
                if r > thr {
                    if st.silence_ms > 0 {
                        st.gen += 1;
                        st.check_inflight = false;
                        st.pending = false;
                    }
                    st.speech_ms += 20;
                    st.silence_ms = 0;
                    st.next_check = 0;
                } else {
                    if st.silence_ms == 0 {
                        st.sil_at_ms = st.input_ms - 20;
                    }
                    st.silence_ms += 20;
                }
            }
        }
        // the turn: checks at the pauses, a reply when finished (or at the fallback)
        let free = st.active.is_none();
        if st.speech_ms >= 120 && st.cfg.turn_detection {
            if !st.check_inflight && !st.pending && st.next_check < checks.len() && st.silence_ms as f32 >= checks[st.next_check] * 1000.0 {
                let thr = thrs[st.next_check.min(thrs.len().saturating_sub(1))];
                st.next_check += 1;
                st.check_inflight = true;
                start_check(&s, &st, thr);
            }
            if st.silence_ms as f32 >= fallback * 1000.0 && !st.pending {
                st.pending = true;
            }
        }
        if free && (st.pending || (st.want_reply && !st.utt.is_empty())) && st.speech_ms > 0 {
            st.pending = false;
            st.want_reply = false;
            let audio_ms = st.sil_at_ms;
            s.emit(RealtimeEvent::SpeechStopped { audio_ms });
            let utt = std::mem::take(&mut st.utt);
            let rate = st.rate;
            let u16k = resample::resample(&utt, rate, ear::RATE);
            st.prev_user = Some((u16k.clone(), Instant::now(), String::new()));
            reset_turn(&mut st);
            let c = clip(u16k);
            let item = s.id(&mut st, "item");
            s.emit(RealtimeEvent::Committed { item });
            st.turns.push(Turn { user: Some(c), ..Default::default() });
            start_reply(&s, &mut st);
        } else if free && (st.want_reply || st.followup_at.is_some_and(|t| Instant::now() >= t)) && st.speech_ms == 0 {
            // a reply asked for with no new speech: a text turn, or a tool result to answer from
            st.want_reply = false;
            st.followup_at = None;
            let ready = st.turns.last().is_some_and(|t| (t.replies.is_empty() && t.user_text.is_some()) || (!t.replies.is_empty()
                && t.results.len() == t.replies.len() && t.replies.last().is_some_and(|r| !r.calls.is_empty())));
            if ready {
                start_reply(&s, &mut st);
            }
        }
        drop(st);
        if last.elapsed() > Duration::from_secs(60) {
            last = Instant::now();
            prune_embeddings(&s);
        }
    }
}

/// The user started speaking: their turn begins (barge-in on a reply, a merge with a turn answered too early)
fn onset(s: &Arc<Shared>, st: &mut State) {
    let audio_ms = st.input_ms.saturating_sub(st.voiced_ms as u64);
    s.emit(RealtimeEvent::SpeechStarted { audio_ms });
    // a reply still being made: stopped; its history keeps what was heard (start_reply's end does it)
    if let Some(a) = &st.active {
        a.stop.store(true, Ordering::SeqCst);
    }
    // a reply already made but still playing: cut where the user came in
    if let Some(p) = st.playing.take() {
        let played = (p.first.elapsed().as_secs_f64() - 0.25).max(0.0);
        if played < p.dur - 0.3 {
            if let Some(r) = st.turns.get_mut(p.turn).and_then(|t| t.replies.get_mut(p.reply)) {
                cut_reply(r, played);
                st.last_said = r.text.trim_end_matches(text::CUT).to_string();
            }
        }
    }
    // the user resumed right after a reply: it came too early - dropped, the two stretches made one turn
    let mut merged = Vec::new();
    if let Some((u, at, _)) = st.prev_user.take() {
        if at.elapsed().as_secs_f64() < MERGE_WINDOW {
            if let Some(t) = st.turns.last() {
                if t.replies.iter().all(|r| r.calls.is_empty()) && t.user.is_some() {
                    st.turns.pop();
                    st.last_said = st.turns.last().and_then(|t| t.replies.last()).map(|r| r.text.clone()).unwrap_or_default();
                    merged = resample::resample(&u, ear::RATE, st.rate);
                }
            }
        }
    }
    let pre: Vec<f32> = st.pre.drain(..).collect();
    st.utt = merged;
    st.utt.extend(pre);
    st.speech_ms = st.voiced_ms;
    st.silence_ms = 0;
    st.next_check = 0;
}

fn reset_turn(st: &mut State) {
    st.gen += 1;
    st.speech_ms = 0;
    st.silence_ms = 0;
    st.voiced_ms = 0;
    st.next_check = 0;
    st.check_inflight = false;
    st.pending = false;
}

/// The rows a clip makes (the ear's, computed once a clip)
fn embedding(s: &Shared, m: &Mind, c: &Clip) -> Result<Arc<Vec<f32>>> {
    if let Some(e) = lock(&s.embs).get(&c.id) {
        return Ok(e.clone());
    }
    let (_, rows) = m.ear.hear(&c.s16)?;
    let e = Arc::new(rows);
    lock(&s.embs).insert(c.id, e.clone());
    Ok(e)
}

fn prune_embeddings(s: &Shared) {
    let st = lock(&s.st);
    let live: Vec<u64> = st.turns.iter().flat_map(|t| t.user.iter().map(|c| c.id).chain(t.replies.iter().filter_map(|r| r.audio.as_ref().map(|a| a.id))))
        .collect();
    drop(st);
    lock(&s.embs).retain(|k, _| live.contains(k));
}

/// A prompt's token ids, its positions' keys, and its audio spans (first position, rows)
type Expanded = (Vec<u32>, Vec<u64>, Vec<(usize, Arc<Vec<f32>>)>);

/// A prompt's token ids, keys and audio spans from rendered text with one `<|audio|>` a clip, in order
fn expand(eng: &Inner, rendered: &str, clips: &[(Clip, Arc<Vec<f32>>)], hidden: usize) -> Result<Expanded> {
    let raw = eng.tok.encode(rendered);
    let mut ids = Vec::with_capacity(raw.len());
    let mut keys = Vec::with_capacity(raw.len());
    let mut spans = Vec::new();
    let mut k = 0;
    for t in raw {
        if t == AUDIO_ID {
            let (c, e) = clips.get(k).ok_or_else(|| Error("more <|audio|> in the prompt than clips".into()))?;
            k += 1;
            let n = e.len() / hidden;
            spans.push((ids.len(), e.clone()));
            for i in 0..n {
                ids.push(AUDIO_ID);
                keys.push(AUDIO_KEY | (c.id << 20) | i as u64);
            }
        } else {
            ids.push(t);
            keys.push(t as u64);
        }
    }
    if k != clips.len() {
        return Err(Error(format!("{} clips for {k} <|audio|> in the prompt", clips.len())));
    }
    Ok((ids, keys, spans))
}

/// The letters' probabilities: [A, B, C, D] (the larger of "X" and " X")
pub fn letter_probs(eng: &Inner, logits: &[f32]) -> [f32; 4] {
    let mut z = [f32::NEG_INFINITY; 4];
    for (i, l) in ["A", "B", "C", "D"].iter().enumerate() {
        for v in [l.to_string(), format!(" {l}")] {
            if let Some(&t) = eng.tok.encode(&v).first() {
                if (t as usize) < logits.len() && eng.tok.encode(&v).len() == 1 {
                    z[i] = z[i].max(logits[t as usize]);
                }
            }
        }
    }
    let mx = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p = z.map(|x| (x - mx).exp());
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= sum);
    p
}

/// The turn question over the turn's speech so far, on the brain's own thread
fn start_check(s: &Arc<Shared>, st: &State, thr: f32) {
    let utt = resample::resample(&st.utt, st.rate, ear::RATE);
    let (said, lang, gen) = (st.last_said.clone(), st.lang.clone(), st.gen);
    let s = s.clone();
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let r = (|| -> Result<(f32, f32)> {
            let mut m = lock(&s.eng.mind);
            if lock(&s.st).gen != gen {
                return Err(Error("stale".into()));
            }
            let (q, fin, back) = text::turn_question(&lang, &said);
            let rendered = text::render(&[Msg::new("user", q)], &[]);
            let c = clip(utt);
            let (_, e) = m.ear.hear(&c.s16)?;
            let e = Arc::new(e);
            let h = m.brain.hidden;
            let (ids, keys, spans) = expand(&s.eng, &rendered, &[(c, e)], h)?;
            let sp: Vec<(usize, &[f32])> = spans.iter().map(|(p, e)| (*p, e.as_slice())).collect();
            let mind = &mut *m;
            let (logits, _, _) = mind.brain.prompt(&mut mind.check, &keys, &Rows { ids: &ids, spans: &sp }, false)?;
            let p = letter_probs(&s.eng, &logits);
            Ok((p[fin], p[back]))
        })();
        let mut st = lock(&s.st);
        if st.gen != gen {
            return;
        }
        st.check_inflight = false;
        match r {
            Ok((p, _pb)) => {
                if p >= thr {
                    st.pending = true;
                }
                if std::env::var("NS_SOPHON_LOG").is_ok_and(|v| v == "1") {
                    eprintln!("sophon: turn check P(finished) {p:.2} (threshold {thr:.2}) in {:.2} s", t0.elapsed().as_secs_f64());
                }
            }
            Err(e) if e.0 != "stale" => s.emit(RealtimeEvent::Error { message: format!("the turn check failed: {e}") }),
            Err(_) => {}
        }
        drop(st);
        s.wake.notify_all();
    });
}

/// What the brain hands the talker
struct Feed {
    ids: Vec<u32>,
    done: bool,
    stopped: bool,
}

/// A reply to the conversation as it stands (the last turn's speech or text, or its tool results)
fn start_reply(s: &Arc<Shared>, st: &mut State) {
    let id = s.id(st, "resp");
    let stop = Arc::new(AtomicBool::new(false));
    st.active = Some(Active { stop: stop.clone(), first: None, sent_s: 0.0 });
    st.playing = None;
    s.emit(RealtimeEvent::ResponseStarted { id: id.clone() });
    // the history within its bounds (duplex.py: 8 turns, 120 s of the user's speech)
    while st.turns.len() > HIST_TURNS
        || (st.turns.len() > 1 && st.turns.iter().filter_map(|t| t.user.as_ref()).map(|c| c.s16.len()).sum::<usize>() as f64 / ear::RATE as f64 > HIST_SEC) {
        st.turns.remove(0);
    }
    let turns = st.turns.clone();
    let cfg = st.cfg.clone();
    let (lang, tools, xvec) = (st.lang.clone(), st.tools.clone(), st.xvec.clone());
    let s = s.clone();
    std::thread::spawn(move || {
        let r = reply(&s, &id, &turns, &cfg, &lang, &tools, xvec, &stop);
        let mut st = lock(&s.st);
        let (status, rep, audio_s, first) = match r {
            Ok(x) => x,
            Err(e) => {
                s.emit(RealtimeEvent::Error { message: e.0.clone() });
                ("failed", Reply { id: id.clone(), ..Default::default() }, 0.0, None)
            }
        };
        let text_all = rep.text.clone();
        let calls = rep.calls.clone();
        // into the history: the last turn's next reply
        let ti = st.turns.len().saturating_sub(1);
        let mut ri = 0;
        if let Some(t) = st.turns.last_mut() {
            if status != "failed" && !(status == "cancelled" && rep.text.is_empty()) {
                t.replies.push(rep);
                ri = t.replies.len() - 1;
            }
        }
        if status == "completed" {
            if let Some(f) = first {
                st.playing = Some(Playing { turn: ti, reply: ri, first: f, dur: audio_s });
            }
            st.last_said = text::plain(&text_all);
        }
        // a reply that called a tool is never merged away (duplex.py's _announce); otherwise the merge window runs from
        // when the reply was asked for
        if !calls.is_empty() {
            st.prev_user = None;
        }
        st.active = None;
        drop(st);
        for (call_id, name, arguments) in calls {
            s.emit(RealtimeEvent::ToolCall { id: id.clone(), call_id, name, arguments });
        }
        s.emit(RealtimeEvent::ResponseDone { id, status: status.to_string(), text: text_all, audio_ms: (audio_s * 1000.0) as u64 });
        s.wake.notify_all();
    });
}

type ReplyResult = (&'static str, Reply, f64, Option<Instant>);

/// The reply: the brain writes (this thread), the talker speaks (its own); returns its status, the reply as the history
/// keeps it, its audio's seconds and when its first audio went out
#[allow(clippy::too_many_arguments)]
fn reply(s: &Arc<Shared>, id: &str, turns: &[Turn], cfg: &RealtimeSession, lang: &str, tools: &[serde_json::Value], xvec: Option<Vec<f32>>, stop: &Arc<AtomicBool>)
         -> Result<ReplyResult> {
    let eng = &s.eng;
    let xvec = xvec.ok_or_else(|| Error("no voice".into()))?;
    // the messages: system, the session's text history, the turns (audio where it was spoken whole)
    let mut sys = cfg.instructions.clone().filter(|i| !i.trim().is_empty()).unwrap_or_else(|| text::system(lang).to_string());
    if !tools.is_empty() {
        sys += "\n\n";
        sys += text::system_tools(lang);
    }
    let mut msgs = vec![Msg::new("system", sys)];
    for (r, t) in &cfg.history {
        if (r == "user" || r == "assistant") && !t.trim().is_empty() {
            msgs.push(Msg::new(if r == "user" { "user" } else { "assistant" }, t.clone()));
        }
    }
    let mut clips: Vec<Clip> = Vec::new();
    for t in turns {
        match (&t.user, &t.user_text) {
            (Some(c), _) => {
                msgs.push(Msg::new("user", text::AUDIO));
                clips.push(c.clone());
            }
            (None, Some(x)) => msgs.push(Msg::new("user", x.clone())),
            _ => {}
        }
        for (i, r) in t.replies.iter().enumerate() {
            match (&r.audio, r.calls.is_empty()) {
                (Some(a), true) => {
                    msgs.push(Msg::new("assistant", text::AUDIO));
                    clips.push(a.clone());
                }
                _ => msgs.push(Msg { role: "assistant", text: r.text.trim().to_string(), calls: r.calls.iter().map(|c| (c.1.clone(), c.2.clone())).collect() }),
            }
            for (_, out) in t.results.get(i).into_iter().flatten() {
                msgs.push(Msg::new("tool", out.clone()));
            }
        }
    }
    let rendered = text::render(&msgs, tools);
    let draw = eng.brain_draw(&cfg.extra, cfg.temperature);
    let tdraw = eng.talker_draw(&cfg.extra);
    let max_tokens = cfg.max_tokens.map(|m| m as usize).unwrap_or(if tools.is_empty() { 120 } else { 240 });
    let feed = Arc::new((Mutex::new(Feed { ids: Vec::new(), done: false, stopped: false }), Condvar::new()));
    // the talker, on its card
    let talker = {
        let (s, feed, stop, lang, id) = (s.clone(), feed.clone(), stop.clone(), lang.to_string(), id.to_string());
        let seed = cfg.seed;
        std::thread::spawn(move || -> Result<(Vec<f32>, Option<Instant>)> {
            let guard = lock(&s.eng.talk);
            let talk = guard.as_ref().ok_or_else(|| Error("loaded without its talker".into()))?;
            let mut all = Vec::new();
            let mut first = None;
            let mut next = |want: usize| -> speak::Feed {
                let (m, cv) = &*feed;
                let mut f = lock(m);
                // the first chunk once LEAD tokens are in hand
                let need = if want <= 1 { LEAD } else { want };
                while f.ids.len() < need && !f.done && !f.stopped {
                    f = cv.wait_timeout(f, Duration::from_millis(50)).map(|(g, _)| g).unwrap_or_else(|e| e.into_inner().0);
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                }
                speak::Feed { ids: f.ids.clone(), done: f.done || f.stopped }
            };
            let mut out = |w: &[f32]| -> Result<()> {
                if first.is_none() {
                    first = Some(Instant::now());
                    if let Some(a) = lock(&s.st).active.as_mut() {
                        a.first = first;
                    }
                }
                all.extend_from_slice(w);
                if let Some(a) = lock(&s.st).active.as_mut() {
                    a.sent_s = all.len() as f64 / speak::RATE as f64;
                }
                s.emit(RealtimeEvent::AudioDelta { id: id.clone(), samples: w.to_vec() });
                Ok(())
            };
            talk.speak(&xvec, &lang, tdraw, seed, &mut next, &mut out, &|| stop.load(Ordering::SeqCst))?;
            Ok((all, first))
        })
    };
    let set_feed = |ids: Vec<u32>, done: bool| {
        let (m, cv) = &*feed;
        let mut f = lock(m);
        if ids.len() > f.ids.len() || done {
            f.ids = ids;
        }
        f.done |= done;
        cv.notify_all();
    };
    // the brain writes
    let written = (|| -> Result<(String, bool)> {
        let mut m = lock(&eng.mind);
        let mut embs = Vec::new();
        for c in &clips {
            embs.push((c.clone(), embedding(s, &m, c)?));
        }
        let h = m.brain.hidden;
        let (ids, keys, spans) = expand(eng, &rendered, &embs, h)?;
        let sp: Vec<(usize, &[f32])> = spans.iter().map(|(p, e)| (*p, e.as_slice())).collect();
        let mind = &mut *m;
        let tp = Instant::now();
        let (mut logits, _, computed) = mind.brain.prompt(&mut mind.reply, &keys, &Rows { ids: &ids, spans: &sp }, false)?;
        let pp = tp.elapsed().as_secs_f64();
        let mut rng = Rng::new(cfg.seed ^ 0x5eed);
        let mut out: Vec<u32> = Vec::new();
        let mut shown = String::new();
        let tg = Instant::now();
        let mut stopped = false;
        for _ in 0..max_tokens {
            if stop.load(Ordering::SeqCst) {
                stopped = true;
                break;
            }
            let t = sample::pick(&logits, draw, &mut rng) as u32;
            if STOP.contains(&t) {
                break;
            }
            out.push(t);
            let raw = eng.tok.decode(&out);
            // a token that ends mid-character waits for the next
            if !raw.ends_with('\u{FFFD}') {
                let vis = text::cut_tool(&raw, false);
                if vis.len() > shown.len() && vis.starts_with(&shown) {
                    s.emit(RealtimeEvent::TextDelta { id: id.to_string(), delta: vis[shown.len()..].to_string() });
                    shown = vis;
                }
                let sp_ids = eng.speakable_ids(&raw, false);
                let stable = sp_ids.len().saturating_sub(1);
                set_feed(sp_ids[..stable].to_vec(), false);
            }
            logits = mind.brain.step(&mut mind.reply, t as u64, t)?;
        }
        let raw = eng.tok.decode(&out);
        set_feed(eng.speakable_ids(&raw, true), true);
        let vis = text::cut_tool(&raw, true);
        if vis.len() > shown.len() && vis.starts_with(&shown) {
            s.emit(RealtimeEvent::TextDelta { id: id.to_string(), delta: vis[shown.len()..].to_string() });
        }
        if std::env::var("NS_SOPHON_LOG").is_ok_and(|v| v == "1") {
            eprintln!("sophon: reply {id}: prompt {} positions ({computed} computed) pp {:.0} tok/s; {} tokens tg {:.1} tok/s: {raw:?}", ids.len(),
                      computed as f64 / pp.max(1e-6), out.len(), out.len() as f64 / tg.elapsed().as_secs_f64().max(1e-6));
        }
        Ok((raw, stopped))
    })();
    let (raw, stopped) = match written {
        Ok(x) => x,
        Err(e) => {
            stop.store(true, Ordering::SeqCst);
            let (m, cv) = &*feed;
            lock(m).stopped = true;
            cv.notify_all();
            let _ = talker.join();
            return Err(e);
        }
    };
    let (audio, first) = talker.join().map_err(|_| Error("the talker's thread failed".into()))??;
    let stopped = stopped || stop.load(Ordering::SeqCst);
    let calls: Vec<(String, String, String)> = if stopped {
        Vec::new()
    } else {
        text::tool_calls(&raw).into_iter().enumerate().map(|(i, (n, a))| (format!("call_{}_{i}", id.trim_start_matches("resp_")), n, a)).collect()
    };
    let spoken_text = text::cut_tool(&raw, true).trim().to_string();
    let dur = audio.len() as f64 / speak::RATE as f64;
    let mut rep = Reply { id: id.to_string(), text: spoken_text, audio: None, calls };
    if dur > 0.1 {
        rep.audio = Some(clip(resample::resample(&audio, speak::RATE, ear::RATE)));
    }
    if stopped {
        // cut by the user: what was heard of it (the client's truncate may set it more exactly)
        let played = first.map(|f| (f.elapsed().as_secs_f64() - 0.25).max(0.0)).unwrap_or(0.0);
        cut_reply(&mut rep, played.min(dur));
        if !rep.text.ends_with(text::CUT) {
            rep.text = format!("{}{}", rep.text, text::CUT);
            rep.audio = None;
        }
        return Ok(("cancelled", rep, dur, first));
    }
    Ok(("completed", rep, dur, first))
}
