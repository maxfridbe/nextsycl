//! The frame on the device, through the engine's silo (`kernels/audio/qwen3tts/silo`, q3s.h): a frame's every step -
//! the first code's logits and draw, the predictor's fifteen passes and draws, the talker's input and its step - in
//! fused kernels with the draws on the GPU, recorded once a request as a SYCL command graph and replayed a frame at
//! a time. The host only reads each frame's sixteen codes back, a few frames behind the GPU (`LAG`), so the GPU never
//! waits for it. Without the silo (or `NS_SILO=0`) the engine runs `Talker::frame` (talker.rs: the shared kernels,
//! the draws on the host).
//!
//! What changes between frames lives in device memory (`State`: the frame's index and talker position, the
//! generator's state, the codes; `Draws`: the request's sampling), so one recording serves every frame.
//!
//! The predictor's input rows are the projection of an embedding row (`proj(E_g(c)) + b`), read from a table made at
//! load (`Talker::ptab`: every codebook's rows through the projection once) instead of a product a code.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, HostBuf};

use crate::ffi::{self, Silo};
use crate::ops::{fp, nul, Mat, Ops};
use crate::sample::Draw;
use crate::talker::{Cache, Drawing, Layer, Lm, Session, Talker};

const HEAD: usize = 128;
const EPS: f32 = 1e-6;
const ACC: i32 = 1;
const NORM: i32 = 2;
const SWIGLU: i32 = 4;
/// frames the GPU runs ahead of the host's reads
const LAG: usize = 3;

/// q3s.h's ns_q3s_draws
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Draws {
    sample: [i32; 2],
    temperature: [f32; 2],
    top_k: [i32; 2],
    penalty: f32,
    suppress_from: i32,
    eos: i32,
    min_frames: i32,
    forced: i32,
    n_text: i32,
}

/// q3s.h's ns_q3s_state
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct State {
    frame: i32,
    pos: i32,
    rng: [u32; 2],
    codes: [i32; 16],
    misses: [i32; 2],
}

fn raw<T: Copy>(v: &T) -> Vec<u8> {
    // SAFETY: a plain #[repr(C)] value of integers and floats, read as its bytes.
    unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), std::mem::size_of::<T>()) }.to_vec()
}

/// The silo, when the engine's shapes are its (None: the shared path)
pub fn silo(t: &Talker) -> Result<Option<&'static Silo>> {
    let Some(k) = ffi::silo()? else { return Ok(None) };
    let ok = |lm: &Lm| {
        // SAFETY: arithmetic only.
        unsafe { (k.supported)(HEAD as i64, (lm.heads / lm.kv) as i64, lm.hidden as i64) == 1 && (k.supported)(HEAD as i64, 1, lm.ffn as i64) == 1 }
    };
    Ok((ok(&t.lm) && ok(&t.pred) && t.groups == 16 && t.lm.hidden <= 2048 && t.pred.heads / t.pred.kv <= 4).then_some(k))
}

/// The silo's calls on device pointers (checked)
pub struct Dev<'a> {
    pub k: &'static Silo,
    pub ops: &'a Ops,
}

impl Dev<'_> {
    fn g(&self) -> *mut c_void {
        self.ops.gpu.raw()
    }

    pub fn gemv(&self, x: *const f32, m: usize, w: &Mat, bias: *const c_void, out: *mut f32, ldo: usize, flags: i32, norm: Option<&DevBuf>) -> Result<()> {
        let (wt, sc) = match &w.scale {
            None => (0, std::ptr::null()),
            Some(s) => (1, s.fp() as *const f32),
        };
        let nw = norm.map_or(std::ptr::null(), |n| n.fp() as *const f32);
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gemv)(self.g(), x, m as i64, w.k as i64, w.w.ptr(), wt, sc, w.n as i64, bias.cast(), out, ldo as i64, flags, nw, EPS) },
                   "q3s gemv")
    }

    /// `n` rows of c.x through every layer (positions from p0, or the talker's one row at the device's `pos`)
    fn pass(&self, lm: &Lm, c: &Cache, n: usize, p0: usize, pos: Option<(*const i32, usize, &DevBuf, &DevBuf)>) -> Result<()> {
        let (h, w) = (lm.hidden, (lm.heads + 2 * lm.kv) * HEAD);
        let g = self.g();
        for (li, l) in lm.layers.iter().enumerate() {
            self.gemv(c.x.fp(), n, &l.qkv, nul(), c.qkv.fp(), w, NORM, Some(&l.in_norm))?;
            match pos {
                Some((p, splits, part, counters)) => {
                    // SAFETY: the qkv row holds heads + 2 kv heads of 128; the caches [kv, t, 128] half; part holds the
                    // splits' scratch, counters kv ints.
                    ffi::check(unsafe { (self.k.attn)(g, c.qkv.fp(), lm.heads as i64, lm.kv as i64, l.q_norm.fp(), l.k_norm.fp(), EPS, lm.inv_freq.fp(), p,
                                                      c.k[li].ptr(), c.v[li].ptr(), c.t as i64, splits as i64, c.att.fp(), part.fp(), counters.ptr().cast()) },
                               "q3s attention")?;
                }
                None => {
                    // SAFETY: n rows of qkv, w floats apart; the caches [kv, t, 128] half; att [n, heads x 128].
                    ffi::check(unsafe { (self.k.attn_small)(g, c.qkv.fp(), w as i64, n as i64, p0 as i64, lm.heads as i64, lm.kv as i64, l.q_norm.fp(),
                                                            l.k_norm.fp(), EPS, lm.inv_freq.fp(), c.k[li].ptr(), c.v[li].ptr(), c.t as i64, c.att.fp()) },
                               "q3s predictor attention")?;
                }
            }
            self.gemv(c.att.fp(), n, &l.o, nul(), c.x.fp(), h, ACC, None)?;
            self.gemv(c.x.fp(), n, &l.gu, nul(), c.act.fp(), lm.ffn, NORM | SWIGLU, Some(&l.post_norm))?;
            self.gemv(c.act.fp(), n, &l.down, nul(), c.x.fp(), h, ACC, None)?;
        }
        Ok(())
    }
}

/// A request's device side: its draws, state, the codes it seen, the text rows to feed, a forced run's codes
pub struct Run {
    draws: DevBuf,
    state: DevBuf,
    seen: DevBuf,
    force: DevBuf,
    feed: DevBuf,
    part: DevBuf,
    counters: DevBuf,
    splits: usize,
}

impl Talker {
    /// The predictor's input rows of every code: proj(E_g(c)) + b for the talker's codebook and the predictor's first
    /// fourteen (float32 [vocab + 14 x pvocab, predictor hidden])
    pub fn project_tables(&self, ops: &Ops, nsd: &nextsycl_diffusion::kernels::Nsd) -> Result<DevBuf> {
        let rows = self.vocab + (self.groups - 2) * self.pvocab;
        let ph = self.pred.hidden;
        let out = DevBuf::f32(&ops.gpu, rows * ph)?;
        nsd.linear(self.table.ptr(), nextsycl_diffusion::kernels::Dt::F16, rows, self.lm.hidden, self.proj.w.ptr(), ph, self.proj_b.ptr(), out.ptr(),
                   nextsycl_diffusion::kernels::Dt::F32)?;
        ops.gpu.sync()?;
        Ok(out)
    }

    /// One frame's work on the device (recorded into the graph): reads x row 0 (the talker's last hidden state,
    /// unnormed), leaves the next one there
    fn body(&self, d: &Dev, s: &Session, r: &Run) -> Result<()> {
        let k = d.k;
        let g = d.ops.gpu.raw();
        let ptab = self.ptab.as_ref().ok_or_else(|| Error("no projected tables".into()))?;
        let (h, ph) = (self.lm.hidden, self.pred.hidden);
        let st = r.state.ptr();
        // the first code from the talker's state, then the predictor's first input rows: proj(h), proj(E0(c0))
        d.gemv(s.lm.x.fp(), 1, &self.head, nul(), s.logits.fp(), self.head.n, NORM, Some(&self.lm.norm))?;
        // SAFETY: logits [vocab]; the draws / state / seen / force buffers sized for them; ptab rows of ph.
        ffi::check(unsafe { (k.draw)(g, s.logits.fp(), self.head.n as i64, 0, r.draws.ptr(), st, r.seen.ptr().cast(), r.force.ptr().cast(), ptab.fp(), 0,
                                     ph as i64, fp(&s.pred.x, ph)) }, "q3s draw")?;
        d.gemv(s.lm.x.fp(), 1, &self.proj, self.proj_b.ptr(), s.pred.x.fp(), ph, NORM, Some(&self.lm.norm))?;
        d.pass(&self.pred, &s.pred, 2, 0, None)?;
        for gi in 1..self.groups {
            let m = &self.heads[gi - 1];
            let row = if gi == 1 { 1 } else { 0 };
            d.gemv(fp(&s.pred.x, row * ph), 1, m, nul(), s.logits.fp(), m.n, NORM, Some(&self.pred.norm))?;
            let next = gi + 1 < self.groups;
            let (base, dst) = if next { (self.index(gi, 0) as i64, s.pred.x.fp()) } else { (0, std::ptr::null_mut()) };
            // SAFETY: as above.
            ffi::check(unsafe { (k.draw)(g, s.logits.fp(), m.n as i64, gi as i32, r.draws.ptr(), st, r.seen.ptr().cast(), r.force.ptr().cast(), ptab.fp(),
                                         base, ph as i64, dst) }, "q3s draw")?;
            if next {
                d.pass(&self.pred, &s.pred, 1, gi + 1, None)?;
            }
        }
        // the frame fed back: its codes' embeddings and the text row, through the talker at the device's position
        // SAFETY: the table [rows, h] half; feed [n_text + 1, h]; x row 0 [h].
        ffi::check(unsafe { (k.frame_in)(g, self.table.ptr(), h as i64, self.vocab as i64, self.pvocab as i64, st, r.draws.ptr(), r.feed.fp(), s.lm.x.fp()) },
                   "q3s frame input")?;
        let pos = (st.cast::<u8>().wrapping_add(4) as *const i32, r.splits, &r.part, &r.counters);
        d.pass(&self.lm, &s.lm, 1, 0, Some(pos))?;
        // SAFETY: the state.
        ffi::check(unsafe { (k.advance)(g, st) }, "q3s advance")
    }

    /// The frames on the device: `l` prompt rows already through the talker (its last row's x at row l - 1), `feed`
    /// the text rows fed a frame each then the pad row. `force`: a check's codes to follow. Calls `got` with each
    /// frame's codes as the host reads them; stops at the end of speech or `max`. Returns a forced run's misses.
    #[allow(clippy::too_many_arguments)]
    pub fn frames_device(&self, k: &'static Silo, ops: &Ops, s: &Session, l: usize, feed: &[f32], d: &Drawing, max: usize, seed: u64,
                         force: Option<&[Vec<i32>]>, got: &mut dyn FnMut(Vec<i32>) -> Result<()>) -> Result<[usize; 2]> {
        let gpu = &ops.gpu;
        let h = self.lm.hidden;
        let dev = Dev { k, ops };
        let n_text = feed.len() / h - 1;
        // the session's talker cache splits its keys this many ways (each a share of the keys so far)
        let splits = s.lm.t.div_ceil(32).clamp(1, 32);
        // SAFETY: arithmetic only.
        let part_n = unsafe { (k.attn_scratch)(self.lm.heads as i64, splits as i64) }.max(1) as usize;
        let draw = |x: Draw| (x.sample as i32, x.temperature, x.top_k as i32);
        let (ts, tt, tk) = draw(d.talker);
        let (ps, pt, pk) = draw(d.predictor);
        if (d.talker.sample && d.talker.top_p < 1.0) || (d.predictor.sample && d.predictor.top_p < 1.0) {
            return Err(Error("top-p below 1 is drawn on the host only".into()));
        }
        let draws = Draws {
            sample: [ts, ps], temperature: [tt, pt], top_k: [tk, pk], penalty: d.repetition_penalty, suppress_from: d.suppress_from as i32, eos: d.eos,
            min_frames: d.min_frames as i32, forced: force.is_some() as i32, n_text: n_text as i32,
        };
        // the generator seeded as the host's (sample.rs: splitmix64 from seed ^ a constant)
        let r0 = seed ^ 0x7177_3374_7473_0000;
        let state = State { frame: 0, pos: l as i32, rng: [r0 as u32, (r0 >> 32) as u32], ..Default::default() };
        let fc: Vec<i32> = match force {
            Some(f) => (0..max).flat_map(|i| f.get(i).cloned().unwrap_or_else(|| vec![d.eos; 16])).collect(),
            None => vec![0; 16],
        };
        let run = Run {
            draws: DevBuf::new(gpu, std::mem::size_of::<Draws>())?,
            state: DevBuf::new(gpu, std::mem::size_of::<State>())?,
            seen: DevBuf::new(gpu, self.vocab.max(self.pvocab))?,
            force: DevBuf::new(gpu, fc.len() * 4)?,
            feed: DevBuf::from_f32(gpu, feed)?,
            part: DevBuf::f32(gpu, part_n)?,
            counters: DevBuf::new(gpu, self.lm.kv * 4)?,
            splits,
        };
        run.draws.write(0, &raw(&draws))?;
        run.state.write(0, &raw(&state))?;
        run.seen.fill(0)?;
        run.counters.fill(0)?;
        run.force.write(0, &fc.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        // the prompt's last row is the first frame's state
        if l > 1 {
            s.lm.x.copy_within(0, &s.lm.x, (l - 1) * h * 4, h * 4)?;
        }
        let t_rec = Instant::now();
        let eager = std::env::var("NS_Q3T_EAGER").is_ok_and(|v| v == "1");
        let mut exec: *mut c_void = std::ptr::null_mut();
        if !eager {
            let mut graph = std::ptr::null_mut();
            // SAFETY: the GPU's handle; an out-pointer to a local.
            ffi::check(unsafe { (k.graph_begin)(gpu.raw(), &mut graph) }, "q3s graph")?;
            let rec = self.body(&dev, s, &run);
            // SAFETY: the graph begun above (ended even when the recording failed).
            let end = ffi::check(unsafe { (k.graph_end)(gpu.raw(), graph, &mut exec) }, "q3s graph");
            rec?;
            end?;
        }
        let rec_s = t_rec.elapsed().as_secs_f64();
        struct Exec(&'static Silo, *mut c_void);
        impl Drop for Exec {
            fn drop(&mut self) {
                if !self.1.is_null() {
                    // SAFETY: an executable graph made by graph_end; the queue is synchronised before it goes.
                    unsafe { (self.0.graph_free)(self.1) };
                }
            }
        }
        let exec = Exec(k, exec);
        // each frame's state copied back to pinned memory, a ring of slots
        const RING: usize = 8;
        let ss = std::mem::size_of::<State>();
        let mut host = HostBuf::new(gpu, RING * ss)?;
        let hp = host.as_mut_slice().as_mut_ptr();
        let api = nextsycl_core::api().map_err(|e| Error(e.0))?;
        let mut tickets = [0i64; RING];
        let mut misses = [0usize; 2];
        let t0 = Instant::now();
        let mut launched = 0;
        let mut done = 0;
        let mut ended = false;
        let result = (|| -> Result<()> {
            while done < max && !ended {
                // queue frames up to LAG ahead
                while launched < max && launched < done + LAG {
                    if exec.1.is_null() {
                        self.body(&dev, s, &run)?;
                    } else {
                        // SAFETY: the executable graph of this session.
                        ffi::check(unsafe { (k.graph_launch)(gpu.raw(), exec.1) }, "q3s graph launch")?;
                    }
                    let slot = launched % RING;
                    // SAFETY: the slot is within the pinned ring; the state is ss bytes.
                    ffi::check(unsafe { (k.read_async)(gpu.raw(), hp.add(slot * ss).cast(), run.state.ptr(), ss) }, "q3s read")?;
                    // SAFETY: the GPU's handle; an out-pointer.
                    ffi::check(unsafe { (api.mark)(gpu.raw(), &mut tickets[slot]) }, "q3s mark")?;
                    launched += 1;
                }
                let slot = done % RING;
                // SAFETY: a ticket marked above.
                ffi::check(unsafe { (api.await_)(gpu.raw(), tickets[slot]) }, "q3s wait")?;
                // SAFETY: the slot's bytes, written by the copy the ticket waited for.
                let st: State = unsafe { std::ptr::read_volatile(hp.add(slot * ss).cast::<State>()) };
                misses = [st.misses[0] as usize, st.misses[1] as usize];
                done += 1;
                if st.codes[0] == d.eos {
                    ended = true;
                } else {
                    got(st.codes.to_vec())?;
                }
            }
            Ok(())
        })();
        // the frames queued past the end finish before their buffers go
        gpu.sync()?;
        drop(host);
        result?;
        if crate::talker::profiling() {
            let n = done.max(1) as f64;
            eprintln!("[q3t profile] device frames: graph recorded in {:.1} ms; {done} frames ({launched} run) in {:.1} ms, {:.3} ms a frame",
                      rec_s * 1e3, t0.elapsed().as_secs_f64() * 1e3, t0.elapsed().as_secs_f64() * 1e3 / n);
        }
        Ok(misses)
    }
}

/// `check --stages kernels`: each kind of the frame's products timed on this GPU, cycling through the layers' own
/// matrices (so the cache holds what a frame's would): the silo's in a graph of `N` calls, and the shared kernel's
/// launched one by one, against the bytes read
pub fn bench(t: &Talker, ops: &Ops, log: &mut dyn FnMut(String)) -> Result<()> {
    let k = silo(t)?.ok_or_else(|| Error("the silo is not built (or not for this checkpoint)".into()))?;
    let gpu = &ops.gpu;
    let d = Dev { k, ops };
    let x = DevBuf::f32(gpu, 8 * 8192)?;
    x.fill(0)?;
    let out = DevBuf::f32(gpu, 8 * 16384)?;
    type Case<'a> = (&'a str, Vec<(&'a Mat, Option<&'a DevBuf>)>, i32, usize);
    let tl = |f: fn(&Layer) -> (&Mat, Option<&DevBuf>)| t.lm.layers.iter().map(f).collect::<Vec<_>>();
    let pl = |f: fn(&Layer) -> (&Mat, Option<&DevBuf>)| t.pred.layers.iter().map(f).collect::<Vec<_>>();
    let cases: Vec<Case> = vec![
        ("talker qkv (norm)", tl(|l| (&l.qkv, Some(&l.in_norm))), NORM, 1),
        ("talker o (+=)", tl(|l| (&l.o, None)), ACC, 1),
        ("talker gate/up (norm, swiglu)", tl(|l| (&l.gu, Some(&l.post_norm))), NORM | SWIGLU, 1),
        ("talker down (+=)", tl(|l| (&l.down, None)), ACC, 1),
        ("talker head (norm)", vec![(&t.head, Some(&t.lm.norm))], NORM, 1),
        ("predictor qkv (norm)", pl(|l| (&l.qkv, Some(&l.in_norm))), NORM, 1),
        ("predictor qkv (norm) x2", pl(|l| (&l.qkv, Some(&l.in_norm))), NORM, 2),
        ("predictor o (+=)", pl(|l| (&l.o, None)), ACC, 1),
        ("predictor gate/up (norm, swiglu)", pl(|l| (&l.gu, Some(&l.post_norm))), NORM | SWIGLU, 1),
        ("predictor down (+=)", pl(|l| (&l.down, None)), ACC, 1),
        ("predictor heads (norm)", t.heads.iter().map(|h| (h, Some(&t.pred.norm))).collect(), NORM, 1),
    ];
    const N: usize = 280;
    for (name, ws, flags, m) in cases {
        let call = |i: usize| {
            let (w, norm) = ws[i % ws.len()];
            d.gemv(x.fp(), m, w, nul(), out.fp(), if flags & SWIGLU != 0 { w.n / 2 } else { w.n }, flags, norm)
        };
        // a graph of N calls, replayed twice (the second timed)
        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();
        // SAFETY: the GPU's handle, out-pointers to locals.
        ffi::check(unsafe { (k.graph_begin)(gpu.raw(), &mut graph) }, "graph")?;
        let rec = (0..N).try_for_each(call);
        // SAFETY: the graph begun above.
        ffi::check(unsafe { (k.graph_end)(gpu.raw(), graph, &mut exec) }, "graph")?;
        rec?;
        let mut graph_us = 0.0;
        for _ in 0..2 {
            let t0 = Instant::now();
            // SAFETY: the graph just made.
            ffi::check(unsafe { (k.graph_launch)(gpu.raw(), exec) }, "graph launch")?;
            gpu.sync()?;
            graph_us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;
        }
        // SAFETY: made above, the queue idle.
        unsafe { (k.graph_free)(exec) };
        let t0 = Instant::now();
        for i in 0..N {
            let (w, _) = ws[i % ws.len()];
            ops.gemv(x.fp(), m, w.k, w, nul(), out.fp(), w.n, false)?;
        }
        gpu.sync()?;
        let shared_us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;
        let gb = ws[0].0.bytes() as f64 / 1e9;
        log(format!("{name:<34} {:>6.1} MB  silo {graph_us:>7.1} us ({:>4.0} GB/s)  shared {shared_us:>7.1} us ({:>4.0} GB/s)", gb * 1e3,
                    gb / (graph_us * 1e-6), gb / (shared_us * 1e-6)));
    }
    Ok(())
}
