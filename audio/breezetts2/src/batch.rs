//! Several requests in the same steps (docs/proposals/speech-batching.md): one thread owns the GPU and advances every
//! request in hand by a frame at a time - the backbone's products over all their rows at once, each row its own
//! cache and position (`Lm::step`), the depth decoder over all of them together (its positions are a frame's 0..16
//! for everyone) - so a frame of four requests reads the weights about as often as a frame of one. A frame is
//! bound by reading the weights, not by the arithmetic: the product takes 8 rows at about the cost of 1.
//!
//! A request joins at the next frame after its prompt pass and leaves at its end (continuous batching); each keeps
//! its own caches, position, draws (seed, temperature, top-k, the penalty's codes) and guidance. A row's numbers are
//! the ones it gets alone (the product's rows are independent; the attention runs a sequence at a time), so a
//! request's codes and samples are the same in a batch as on its own.
//!
//! The callers (the server's request threads) only hand requests in and read what comes back: progress, then the
//! sound or the error; a cancel is a flag the thread reads between frames.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{Audio, AudioEngine, AudioRequest, Defaults, EngineOption, Error, Result, Speech, Step};
use nextsycl_core::DevBuf;

use crate::lm::{Cache, Work};
use crate::ops::fp;
use crate::prompt::Spec;
use crate::sample::{self, Rng};
use crate::{BreezeTts2, Drawing, FPS, MAX_FRAMES, MAX_SEQ};

/// the rows a step takes by default: a guided request (with an instruction) is two, a plain one one
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
type Prepared = (Spec, Drawing, usize);
/// A request's branches: each its cache, its prompt's length, its last hidden state
type Seqs = Vec<(Cache, usize, DevBuf)>;

/// A request in the batch
struct Live {
    job: Job,
    s: Spec,
    d: Drawing,
    rng: Rng,
    max: usize,
    /// the frames' first codes so far (the penalty's) and the frames
    seen: Vec<i32>,
    codes: Vec<Vec<i32>>,
    /// a branch each: its cache, its prompt's length, its last hidden state [hidden]
    seqs: Vec<(Cache, usize, DevBuf)>,
}

/// The engine served: the model and the thread that runs its batch
pub struct Batched {
    m: Arc<BreezeTts2>,
    tx: Mutex<Sender<Job>>,
    rows: usize,
}

impl Batched {
    /// The model with its batch thread (`rows`: the rows a step takes, 1..8; 1 runs the requests one at a time)
    pub fn new(m: BreezeTts2, rows: usize) -> Result<Batched> {
        let rows = rows.clamp(1, ROWS);
        let m = Arc::new(m);
        let (tx, rx) = channel();
        let w = m.clone();
        std::thread::Builder::new().name("breeze-batch".into()).spawn(move || run(&w, rx, rows))
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
        r.push(format!("requests together: up to {} rows a frame (a guided request two)", self.rows));
        r
    }
}

/// The batch thread: takes requests in as rows free up (in the order they came), advances those in hand a frame at
/// a time, answers each at its end
fn run(m: &BreezeTts2, rx: Receiver<Job>, rows: usize) {
    let mut waiting: VecDeque<(Job, Option<Prepared>)> = VecDeque::new();
    let mut live: Vec<Live> = Vec::new();
    let mut bufs: Option<Bufs> = None;
    loop {
        // nothing in hand: wait for a request (the senders gone: the engine is dropped)
        if live.is_empty() && waiting.is_empty() {
            match rx.recv() {
                Ok(j) => waiting.push_back((j, None)),
                Err(_) => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            waiting.push_back((j, None));
        }
        // the ones that fit join, in order
        let mut used: usize = live.iter().map(|l| l.seqs.len()).sum();
        while let Some((job, prep)) = waiting.pop_front() {
            if job.stop.load(Ordering::Relaxed) {
                let _ = job.tx.send(Msg::Done(Err(Error("cancelled".into()))));
                continue;
            }
            let prep = match prep {
                Some(p) => p,
                None => match prepare(m, &job.req) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = job.tx.send(Msg::Done(Err(e)));
                        continue;
                    }
                },
            };
            let n = prep.0.branches().len();
            if used + n > rows {
                waiting.push_front((job, Some(prep)));
                break;
            }
            match join(m, job, prep) {
                Ok(l) => {
                    used += n;
                    live.push(l);
                }
                Err(b) => {
                    let (job, e) = *b;
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
        let b = bufs.as_ref().expect("made above");
        if let Err(e) = frame(m, b, &mut live) {
            // the device failed (or a bug): every request in hand ends with it
            for l in live.drain(..) {
                let _ = l.job.tx.send(Msg::Done(Err(Error(e.0.clone()))));
            }
        }
    }
}

/// What a request asks for, checked (a clone's recording encoded): its spec, draws, frames at most
fn prepare(m: &BreezeTts2, req: &AudioRequest) -> Result<Prepared> {
    let s = m.spec(req)?;
    let d = m.drawing(req)?;
    let seconds = req.seconds.unwrap_or(m.defaults().seconds);
    if seconds <= 0.0 {
        return Err(Error(format!("{seconds} s: a length above zero")));
    }
    Ok((s, d, ((seconds * FPS) as usize).clamp(1, MAX_FRAMES)))
}

/// A request's prompt through the backbone, a branch at a time: it joins the batch
fn join(m: &BreezeTts2, job: Job, (s, d, max): Prepared) -> std::result::Result<Live, Box<(Job, Error)>> {
    let _ = job.tx.send(Msg::Step(Step { phase: "prompt", at: 0, of: 1, seconds: job.t0.elapsed().as_secs_f64() }));
    let r = (|| -> Result<(Seqs, usize)> {
        let (ops, h) = (&m.ops, m.bb.hidden);
        let mut seqs = Vec::new();
        let mut longest = 0;
        for segs in s.branches() {
            let rows = m.prompt_rows(&segs, None)?;
            let l = rows.len() / h;
            longest = longest.max(l);
            if l + 1 >= MAX_SEQ {
                return Err(Error(format!("the prompt is {l} positions; the model reads {MAX_SEQ} with the speech")));
            }
            let c = m.bb.cache(ops, 1, l, (l + max + 2).min(MAX_SEQ))?;
            c.x.write(0, &crate::bytes(&rows))?;
            m.bb.pass(ops, &m.nsd, &c, l, 0)?;
            let last = DevBuf::f32(&ops.gpu, h)?;
            last.copy_within(0, &c.out, (l - 1) * h * 4, h * 4)?;
            seqs.push((c, l, last));
        }
        Ok((seqs, max.min(MAX_SEQ - 1 - longest)))
    })();
    match r {
        Ok((seqs, max)) => {
            let _ = job.tx.send(Msg::Step(Step { phase: "prompt", at: 1, of: 1, seconds: job.t0.elapsed().as_secs_f64() }));
            let rng = Rng::new(job.req.seed);
            Ok(Live { job, s, d, rng, max, seen: Vec::new(), codes: Vec::new(), seqs })
        }
        Err(e) => Err(Box::new((job, e))),
    }
}

/// The step's buffers, for `rows` rows
struct Bufs {
    work: Work,
    /// the depth decoder's cache: a sequence a row, a frame's 16 positions
    dc: Cache,
    lasts: DevBuf,
    e: DevBuf,
    logits: DevBuf,
    tmp: DevBuf,
}

impl Bufs {
    fn new(m: &BreezeTts2, rows: usize) -> Result<Bufs> {
        let (ops, h) = (&m.ops, m.bb.hidden);
        Ok(Bufs {
            work: m.bb.work(ops, rows)?,
            dc: m.dd.cache(ops, rows, 2 * rows, m.groups)?,
            lasts: DevBuf::f32(&ops.gpu, rows * h)?,
            e: DevBuf::f32(&ops.gpu, rows * h)?,
            logits: DevBuf::f32(&ops.gpu, rows * (m.vocab + 1))?,
            tmp: DevBuf::f32(&ops.gpu, h)?,
        })
    }
}

/// Each request's logits rows [n][w] of the step's first n * w floats
fn read(m: &BreezeTts2, logits: &DevBuf, n: usize, w: usize) -> Result<Vec<Vec<f32>>> {
    m.ops.gpu.sync()?;
    let all = logits.to_f32()?;
    Ok((0..n).map(|b| all[b * w..(b + 1) * w].to_vec()).collect())
}

/// The guided logits of the request whose rows start at `at` (its second the plain branch's, if guided)
fn guided(ls: &[Vec<f32>], at: usize, l: &Live) -> Vec<f32> {
    sample::guided(&ls[at], (l.seqs.len() > 1).then(|| ls[at + 1].as_slice()), l.s.cfg)
}

/// One frame of every request in hand: their first codes from the backbone's head, the depth decoder's fifteen, the
/// frame fed back; the ones that end leave (answered), cancelled ones too
fn frame(m: &BreezeTts2, b: &Bufs, live: &mut Vec<Live>) -> Result<()> {
    let (ops, nsd, h, dh) = (&m.ops, &m.nsd, m.bb.hidden, m.dd.hidden);
    let (vocab, eos) = (m.vocab, m.vocab as i32);
    // cancelled, or at their most frames: they leave now
    let mut i = 0;
    while i < live.len() {
        if live[i].job.stop.load(Ordering::Relaxed) {
            let l = live.remove(i);
            let _ = l.job.tx.send(Msg::Done(Err(Error("cancelled".into()))));
        } else if live[i].codes.len() >= live[i].max {
            finish(m, live.remove(i));
        } else {
            i += 1;
        }
    }
    if live.is_empty() {
        return Ok(());
    }
    // the rows' last hidden states together, the backbone's head over them
    let gather = |live: &[Live]| -> Result<usize> {
        let mut r = 0;
        for l in live {
            for (_, _, last) in &l.seqs {
                b.lasts.copy_within(r * h * 4, last, 0, h * 4)?;
                r += 1;
            }
        }
        Ok(r)
    };
    let r = gather(live)?;
    ops.gemv(b.lasts.fp(), r, &m.head0, b.logits.fp(), m.head0.n, false)?;
    let ls = read(m, &b.logits, r, vocab + 1)?;
    let mut c0s = Vec::new();
    let (mut at, mut ended) = (0, Vec::new());
    for (li, l) in live.iter_mut().enumerate() {
        let mut l0 = guided(&ls, at, l);
        at += l.seqs.len();
        sample::repetition_penalty(&mut l0, &l.seen, l.d.penalty);
        sample::suppress(&mut l0, m.codes, vocab);
        let c0 = sample::pick(&l0, l.d.backbone, &mut l.rng) as i32;
        if c0 == eos {
            ended.push(li);
        }
        c0s.push(c0);
    }
    // the ones that drew the end leave before the depth decoder
    for &li in ended.iter().rev() {
        c0s.remove(li);
        finish(m, live.remove(li));
    }
    if live.is_empty() {
        return Ok(());
    }
    let r = if ended.is_empty() { r } else { gather(live)? };
    // the depth decoder: [projection(hidden), projection(E(c0))] a row, at positions 0, 1
    let mut at = 0;
    for (l, c0) in live.iter().zip(&c0s) {
        ops.embed(&m.table, h, &[*c0], fp(&b.e, at * h), h, l.seqs.len())?;
        at += l.seqs.len();
    }
    ops.gemv(b.lasts.fp(), r, &m.proj, b.dc.x.fp(), 2 * dh, false)?;
    ops.gemv(b.e.fp(), r, &m.proj, fp(&b.dc.x, dh), 2 * dh, false)?;
    m.dd.pass_b(ops, nsd, &b.dc, r, 2, 0)?;
    let mut frames: Vec<Vec<i32>> = c0s.iter().map(|c| vec![*c]).collect();
    for g in 1..m.groups {
        let hm = &m.heads[g - 1];
        if g == 1 {
            // the first step's rows are 2 a sequence (its second the one read)
            for s in 0..r {
                ops.gemv(fp(&b.dc.out, (2 * s + 1) * dh), 1, hm, fp(&b.logits, s * vocab), hm.n, false)?;
            }
        } else {
            ops.gemv(b.dc.out.fp(), r, hm, b.logits.fp(), hm.n, false)?;
        }
        let ls = read(m, &b.logits, r, vocab)?;
        let mut at = 0;
        for (l, f) in live.iter_mut().zip(frames.iter_mut()) {
            let mut lg = guided(&ls, at, l);
            sample::suppress(&mut lg, m.codes, vocab);
            let code = sample::pick(&lg, l.d.depth, &mut l.rng) as i32;
            if g + 1 < m.groups {
                ops.embed(&m.table, h, &[code + (g * vocab) as i32], fp(&b.e, at * h), h, l.seqs.len())?;
            }
            at += l.seqs.len();
            f.push(code);
        }
        if g + 1 < m.groups {
            ops.gemv(b.e.fp(), r, &m.proj, b.dc.x.fp(), dh, false)?;
            m.dd.pass_b(ops, nsd, &b.dc, r, 1, g + 1)?;
        }
    }
    // the frames fed back: each branch's row the frame's summed embeddings, the backbone a step over them all
    let mut at = 0;
    for (l, f) in live.iter().zip(&frames) {
        m.sum_rows(&m.frame_idx(f), fp(&b.work.x, at * h), &b.tmp)?;
        for j in 1..l.seqs.len() {
            b.work.x.copy_within((at + j) * h * 4, &b.work.x, at * h * 4, h * 4)?;
        }
        at += l.seqs.len();
    }
    let seqs: Vec<(&Cache, usize)> = live.iter().flat_map(|l| l.seqs.iter().map(|(c, p, _)| (c, p + l.codes.len()))).collect();
    m.bb.step(ops, nsd, &b.work, &seqs)?;
    let mut at = 0;
    for (l, f) in live.iter_mut().zip(frames) {
        for (_, _, last) in &l.seqs {
            last.copy_within(0, &b.work.out, at * h * 4, h * 4)?;
            at += 1;
        }
        l.seen.push(f[0]);
        l.codes.push(f);
        let _ = l.job.tx.send(Msg::Step(Step { phase: "tokens", at: l.codes.len() as u32, of: l.max as u32, seconds: l.job.t0.elapsed().as_secs_f64() }));
    }
    Ok(())
}

/// A request at its end: its frames through the codec, the sound answered
fn finish(m: &BreezeTts2, l: Live) {
    let t = &l.job;
    let r = (|| -> Result<Audio> {
        if l.codes.is_empty() {
            return Err(Error("the model ended the speech before its first frame".into()));
        }
        let _ = t.tx.send(Msg::Step(Step { phase: "decode", at: 0, of: 1, seconds: t.t0.elapsed().as_secs_f64() }));
        m.ops.gpu.sync()?;
        let samples = m.codec.decode(&m.nsd, &l.codes)?;
        let _ = t.tx.send(Msg::Step(Step { phase: "decode", at: 1, of: 1, seconds: t.t0.elapsed().as_secs_f64() }));
        Ok(Audio { rate: nextsycl_codec12hz::RATE, channels: 1, samples })
    })();
    let _ = t.tx.send(Msg::Done(r));
}
