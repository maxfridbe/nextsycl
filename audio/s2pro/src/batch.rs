//! Several requests in the same steps (docs/proposals/speech-batching.md, as Breeze TTS 2's batch.rs): one thread
//! owns the GPU and advances every request in hand by a frame at a time - the slow model's products over all their
//! rows at once, each row its own cache and position (`Lm::step`), the fast model over all of them together (its
//! positions are a frame's 0..10 for everyone). A request's turns (its text's speaker batches) go one after another
//! as alone: at a turn's end its next turn's prompt is read in before its next frame. A row's numbers are the ones it
//! gets alone (the product's rows are independent; the attention runs a sequence at a time), so a request's codes
//! are the same in a batch as on its own.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{Audio, AudioEngine, AudioRequest, Defaults, EngineOption, Error, Result, Speech, Step};
use nextsycl_core::DevBuf;

use crate::lm::{Cache, Work};
use crate::ops::fp;
use crate::prompt::{self, Row, Spec};
use crate::sample::{self, Draw, Rng};
use crate::{bytes, codec, S2Pro, FPS, MAX_FRAMES, MAX_SEQ, PREFILL};

/// the rows a step takes by default (a request is one)
pub const ROWS: usize = 8;

enum Msg {
    Step(Step),
    Done(Result<Audio>),
}

struct Job {
    req: AudioRequest,
    tx: Sender<Msg>,
    stop: Arc<AtomicBool>,
    t0: Instant,
}

/// A request checked: its spec, draws, frames at most
type Prepared = (Spec, Draw, usize);

/// A request in the batch: its turns and where it is in them, its cache and last hidden state, its draws
struct Live {
    job: Job,
    d: Draw,
    rng: Rng,
    max: usize,
    system: Vec<Row>,
    users: Vec<Vec<Row>>,
    /// the turn being made, where it began, the position next, whether a turn made audio before
    turn: usize,
    start: usize,
    pos: usize,
    open: bool,
    made: usize,
    window: Vec<usize>,
    out: Vec<Vec<i32>>,
    c: Cache,
    last: DevBuf,
}

/// The engine served: the model and the thread that runs its batch
pub struct Batched {
    m: Arc<S2Pro>,
    tx: Mutex<Sender<Job>>,
    rows: usize,
}

impl Batched {
    /// The model with its batch thread (`rows`: the requests a step takes, 1..8; 1 runs them one at a time)
    pub fn new(m: S2Pro, rows: usize) -> Result<Batched> {
        let rows = rows.clamp(1, ROWS);
        let m = Arc::new(m);
        let (tx, rx) = channel();
        let w = m.clone();
        std::thread::Builder::new().name("s2pro-batch".into()).spawn(move || run(&w, rx, rows))
            .map_err(|e| Error(format!("the batch thread: {e}")))?;
        Ok(Batched { m, tx: Mutex::new(tx), rows })
    }
}

impl AudioEngine for Batched {
    fn arch(&self) -> &'static str {
        self.m.arch()
    }
    fn defaults(&self) -> Defaults {
        self.m.defaults()
    }
    fn speech(&self) -> Option<Speech> {
        self.m.speech()
    }
    fn options(&self) -> &'static [EngineOption] {
        self.m.options()
    }
    fn load_seconds(&self) -> f64 {
        self.m.load_seconds()
    }
    fn batch(&self) -> usize {
        self.rows
    }
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let (tx, rx) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let job = Job { req: req.clone(), tx, stop: stop.clone(), t0: Instant::now() };
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).send(job).map_err(|_| Error("the batch thread has ended".into()))?;
        let mut stopped: Option<Error> = None;
        loop {
            match rx.recv() {
                Ok(Msg::Step(st)) => {
                    if stopped.is_none() {
                        if let Err(e) = progress(st) {
                            stop.store(true, Ordering::Relaxed);
                            stopped = Some(e);
                        }
                    }
                }
                Ok(Msg::Done(r)) => return match stopped {
                    Some(e) => Err(e),
                    None => r,
                },
                Err(_) => return Err(Error("the batch thread has ended".into())),
            }
        }
    }
    fn report(&self) -> Vec<String> {
        let mut r = self.m.report();
        r.push(format!("requests together: up to {} a frame", self.rows));
        r
    }
}

/// The batch thread: takes requests in as rows free up (in order), advances those in hand a frame at a time,
/// answers each at its end
fn run(m: &S2Pro, rx: Receiver<Job>, rows: usize) {
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut live: Vec<Live> = Vec::new();
    let mut bufs: Option<Bufs> = None;
    loop {
        if live.is_empty() && waiting.is_empty() {
            match rx.recv() {
                Ok(j) => waiting.push_back(j),
                Err(_) => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            waiting.push_back(j);
        }
        while live.len() < rows {
            let Some(job) = waiting.pop_front() else { break };
            if job.stop.load(Ordering::Relaxed) {
                let _ = job.tx.send(Msg::Done(Err(Error("cancelled".into()))));
                continue;
            }
            match prepare(m, &job.req).and_then(|p| join(m, &job, p)) {
                Ok(parts) => live.push(Live { job, ..parts }),
                Err(e) => {
                    let _ = job.tx.send(Msg::Done(Err(e)));
                }
            }
        }
        if live.is_empty() {
            continue;
        }
        if bufs.is_none() {
            match Bufs::new(m, rows) {
                Ok(b) => bufs = Some(b),
                Err(e) => {
                    for l in live.drain(..) {
                        let _ = l.job.tx.send(Msg::Done(Err(Error(e.0.clone()))));
                    }
                    continue;
                }
            }
        }
        if let Err(e) = frame(m, bufs.as_ref().expect("made above"), &mut live) {
            for l in live.drain(..) {
                let _ = l.job.tx.send(Msg::Done(Err(Error(e.0.clone()))));
            }
        }
    }
}

fn prepare(m: &S2Pro, req: &AudioRequest) -> Result<Prepared> {
    let s = m.spec(req)?;
    let d = m.drawing(req)?;
    let seconds = req.seconds.unwrap_or(m.defaults().seconds);
    if seconds <= 0.0 {
        return Err(Error(format!("{seconds} s: a length above zero")));
    }
    Ok((s, d, ((seconds * FPS) as usize).clamp(1, MAX_FRAMES)))
}

/// A request's prompt (its system turn and first user turn) read in: its place in the batch (the job is the
/// caller's to put in)
fn join(m: &S2Pro, job: &Job, (s, d, max): Prepared) -> Result<Live> {
    let _ = job.tx.send(Msg::Step(Step { phase: "prompt", at: 0, of: 1, seconds: job.t0.elapsed().as_secs_f64() }));
    let system = prompt::system(&m.tok, &m.ids, s.reference.as_ref());
    let users: Vec<Vec<Row>> = s.batches().iter().map(|b| prompt::user(&m.tok, &m.ids, b)).collect();
    let positions = system.len() + users.iter().map(|u| u.len() + 2).sum::<usize>() + max + 1;
    if positions > MAX_SEQ {
        return Err(Error(format!("the prompt and {max} frames are {positions} positions; the model reads {MAX_SEQ}")));
    }
    let c = m.slow.cache(&m.ops, 1, PREFILL, positions)?;
    let last = DevBuf::f32(&m.ops.gpu, m.slow.hidden)?;
    // the job is a placeholder here: `run` puts the request's own in
    let mut l = Live { job: Job { req: AudioRequest::default(), tx: job.tx.clone(), stop: job.stop.clone(), t0: job.t0 }, d, rng: Rng::new(job.req.seed), max,
                       system, users, turn: 0, start: 0, pos: 0, open: false, made: 0, window: Vec::new(), out: Vec::new(), c, last };
    if !begin_turn(m, &mut l)? {
        return Err(Error("the text makes no turn to speak".into()));
    }
    Ok(l)
}

/// The request's turn `l.turn` read in (with the system turn first, and the end of the one before when it spoke);
/// false: no turn left, or no frames left
fn begin_turn(m: &S2Pro, l: &mut Live) -> Result<bool> {
    if l.turn >= l.users.len() || l.out.len() >= l.max {
        return Ok(false);
    }
    l.start = l.pos;
    let mut rows = if l.turn == 0 { l.system.clone() } else { Vec::new() };
    if l.open {
        rows.extend(prompt::end(&m.ids));
    }
    rows.extend(l.users[l.turn].iter().cloned());
    let x = m.embed_rows(&rows)?;
    m.prefill(&l.c, &x, l.pos, &l.last)?;
    l.pos += rows.len();
    l.made = 0;
    l.window.clear();
    let _ = l.job.tx.send(Msg::Step(Step { phase: "prompt", at: l.turn as u32 + 1, of: l.users.len() as u32, seconds: l.job.t0.elapsed().as_secs_f64() }));
    Ok(true)
}

/// A turn ended (its end drawn, or the frames all made): back to before it if it spoke nothing (as upstream), then
/// the next turn; false: the request is done
fn end_turn(m: &S2Pro, l: &mut Live) -> Result<bool> {
    if l.made == 0 {
        l.pos = l.start + if l.turn == 0 { l.system.len() } else { 0 };
    } else {
        l.open = true;
    }
    l.turn += 1;
    begin_turn(m, l)
}

/// The step's buffers, for `rows` requests
struct Bufs {
    work: Work,
    /// the fast model's cache: a sequence a request, a frame's positions
    fc: Cache,
    lasts: DevBuf,
    logits: DevBuf,
}

impl Bufs {
    fn new(m: &S2Pro, rows: usize) -> Result<Bufs> {
        let (ops, h) = (&m.ops, m.slow.hidden);
        Ok(Bufs {
            work: m.slow.work(ops, rows)?,
            fc: m.fast.cache(ops, rows, 2 * rows, m.groups)?,
            lasts: DevBuf::f32(&ops.gpu, rows * h)?,
            logits: DevBuf::f32(&ops.gpu, rows * m.head.n.max(m.codes))?,
        })
    }
}

/// The first `n` rows of `w` floats each
fn read(m: &S2Pro, logits: &DevBuf, n: usize, w: usize) -> Result<Vec<Vec<f32>>> {
    m.ops.gpu.sync()?;
    let mut v = vec![0u8; n * w * 4];
    logits.read(0, &mut v)?;
    let all: Vec<f32> = v.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
    Ok((0..n).map(|b| all[b * w..(b + 1) * w].to_vec()).collect())
}

/// One frame of every request in hand; a request whose turn ends goes on to its next turn (read in now, its next
/// frame in the next step) or leaves, answered
fn frame(m: &S2Pro, b: &Bufs, live: &mut Vec<Live>) -> Result<()> {
    let (ops, nsd, h) = (&m.ops, &m.nsd, m.slow.hidden);
    let (n0, end) = (m.head.n, m.codes);
    // cancelled ones leave, and the ones at their most frames
    let mut i = 0;
    while i < live.len() {
        if live[i].job.stop.load(Ordering::Relaxed) {
            let l = live.remove(i);
            let _ = l.job.tx.send(Msg::Done(Err(Error("cancelled".into()))));
            continue;
        }
        // at its most frames: no turn makes more
        if live[i].out.len() >= live[i].max {
            finish(m, live.remove(i));
            continue;
        }
        i += 1;
    }
    if live.is_empty() {
        return Ok(());
    }
    let gather = |live: &[Live]| -> Result<usize> {
        for (r, l) in live.iter().enumerate() {
            b.lasts.copy_within(r * h * 4, &l.last, 0, h * 4)?;
        }
        Ok(live.len())
    };
    let r = gather(live)?;
    m.head.gemv(ops, b.lasts.fp(), r, b.logits.fp(), n0, false)?;
    let ls = read(m, &b.logits, r, n0)?;
    let mut c0s = Vec::new();
    let mut ended = Vec::new();
    for (li, l) in live.iter_mut().enumerate() {
        let c0 = sample::pick_ras(&ls[li], l.d, &l.window, |i| i < end, &mut l.rng);
        if c0 >= end {
            ended.push(li);
        } else if l.made > 0 {
            l.window.push(c0);
            if l.window.len() > sample::WINDOW {
                l.window.remove(0);
            }
        }
        c0s.push(c0);
    }
    // the turns that drew their end: the next turn read in (its frames from the next step), or the request done
    for &li in ended.iter().rev() {
        c0s.remove(li);
        let mut l = live.remove(li);
        if end_turn(m, &mut l)? {
            live.push(l);
            // read in now, it takes no part in this frame: its row stays out until the next step
            c0s.push(usize::MAX);
        } else {
            finish(m, l);
        }
    }
    // the requests making a frame now, first in `live` (the ones that just began a turn are at its end)
    let go = c0s.iter().take_while(|c| **c != usize::MAX).count();
    if go == 0 {
        return Ok(());
    }
    let r = if ended.is_empty() { r } else { gather(&live[..go])? };
    // the fast model: [hidden, E(c0)] at positions 0, 1 a request; then a code a position
    for (s, c0) in c0s[..go].iter().enumerate() {
        b.fc.x.copy_within(2 * s * h * 4, &b.lasts, s * h * 4, h * 4)?;
        ops.embed(&m.fast_emb, h, &[*c0 as i32], fp(&b.fc.x, (2 * s + 1) * h), h, 1)?;
    }
    m.fast.pass_b(ops, nsd, &b.fc, r, 2, 0, &m.cs)?;
    let mut frames: Vec<Vec<i32>> = c0s[..go].iter().map(|c| vec![*c as i32]).collect();
    for cb in 1..m.groups {
        if cb == 1 {
            for s in 0..r {
                m.fast_out.gemv(ops, fp(&b.fc.out, (2 * s + 1) * h), 1, fp(&b.logits, s * m.codes), m.codes, false)?;
            }
        } else {
            m.fast_out.gemv(ops, b.fc.out.fp(), r, b.logits.fp(), m.codes, false)?;
        }
        let ls = read(m, &b.logits, r, m.codes)?;
        for (s, (l, f)) in live[..go].iter_mut().zip(frames.iter_mut()).enumerate() {
            let code = sample::pick(&ls[s], l.d, &mut l.rng) as i32;
            f.push(code);
            if cb + 1 < m.groups {
                ops.embed(&m.fast_emb, h, &[code.clamp(0, m.codes as i32 - 1)], fp(&b.fc.x, s * h), h, 1)?;
            }
        }
        if cb + 1 < m.groups {
            m.fast.pass_b(ops, nsd, &b.fc, r, 1, cb + 1, &m.cs)?;
        }
    }
    // the frames fed back: the slow model a step over them all
    for (s, f) in frames.iter().enumerate() {
        let x = m.embed_rows(&[Row { token: m.ids.semantic0 + f[0] as u32, codes: Some(f.clone()) }])?;
        b.work.x.write(s * h * 4, &bytes(&x))?;
    }
    let seqs: Vec<(&Cache, usize)> = live[..go].iter().map(|l| (&l.c, l.pos)).collect();
    m.slow.step(ops, nsd, &b.work, &seqs, &m.cs)?;
    for (s, (l, f)) in live[..go].iter_mut().zip(frames).enumerate() {
        l.last.copy_within(0, &b.work.out, s * h * 4, h * 4)?;
        l.pos += 1;
        l.made += 1;
        l.out.push(f);
        let _ = l.job.tx.send(Msg::Step(Step { phase: "tokens", at: l.out.len() as u32, of: l.max as u32, seconds: l.job.t0.elapsed().as_secs_f64() }));
    }
    Ok(())
}

/// A request at its end: its frames through the codec, the sound answered
fn finish(m: &S2Pro, l: Live) {
    let t = &l.job;
    let r = (|| -> Result<Audio> {
        if l.out.is_empty() {
            return Err(Error("the model ended the speech before its first frame".into()));
        }
        let _ = t.tx.send(Msg::Step(Step { phase: "decode", at: 0, of: 1, seconds: t.t0.elapsed().as_secs_f64() }));
        m.ops.gpu.sync()?;
        let samples = m.codec.decode(&m.nsd, &l.out)?;
        let _ = t.tx.send(Msg::Step(Step { phase: "decode", at: 1, of: 1, seconds: t.t0.elapsed().as_secs_f64() }));
        Ok(Audio { rate: codec::RATE, channels: 1, samples })
    })();
    let _ = t.tx.send(Msg::Done(r));
}
