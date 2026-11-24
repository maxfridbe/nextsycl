//! MovieRoll's three passes over the DiT (its fine-tune of HunyuanVideo-1.5: the same blocks, q / k / v fused in the
//! file): the inference regime of `MovieRollDiffusionTransformer` (github Tencent-Hunyuan/MovieRoll 0b5f73e2,
//! movieroll_transformer.py `forward` with `training=False`; SGLang-MovieRoll's runtime/models/dits/movieroll.py spells
//! the same three modes out):
//!
//! ```text
//!   denoise    a window of noisy chunks, each chunk at its own timestep (the image stream modulated frame by frame),
//!              the text stream at the window's first chunk's timestep: image queries over [history K, V | window |
//!              text], text queries over [window | text] - the text never sees the history
//!   encode     one clean chunk at t = 0, no text: its queries over [history | chunk]; its K, V appended to the cache
//!              (a sink-recent block first trims its cache back to the sink)
//!   recompute  the cache rebuilt from chosen clean chunks at t = 0, no text, block-causal (chunk i over chunks 0..i):
//!              the full-history blocks over every chunk, the sink-recent blocks over the first and the last of them
//!              as the full-history blocks left them
//! ```
//!
//! HieraKV: blocks 0..`full` keep the whole history; the others the sink (the first chunk) and the most recent one.
//! RoPE positions are compact: a block's cached chunks sit at frames 0.. in cache order, the window right after them
//! (`RoPEResampler._resample_inference`), so a block group's window rows turn by its own cache's length. The cache
//! keeps K after its RoPE (a chunk's position does not change while it is cached: appended at the end, or rebuilt by a
//! recompute that sets every position) and V, half, per block; upstream keeps K before its RoPE in bfloat16.

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::dit::{bias, fptr, hptr, w32, Attn, Batch, Branch, Dit, Steps, Text, DIM, EPS, FFN, HEAD, HEADS, IN, LATENT};
use crate::ffi::T;
use crate::w::Post;

/// The latent frames the RoPE grid holds (`max_num_latent_frames`): a block's cache and the window after it stay below
pub const MAX_FRAMES: usize = 120;

/// One block's cache: K (after RoPE) and V, half [rows, 2048] each - or int8 with a float32 scale a (row, head) -, and
/// its chunks' latent frame counts in order
pub struct KvBlock {
    k: DevBuf,
    v: DevBuf,
    /// int8: the scales [rows, 16] of k and v
    ks: Option<(DevBuf, DevBuf)>,
    cap_frames: usize,
    frames: Vec<usize>,
}

/// The clean history's KV cache of one generation
pub struct Kv {
    blocks: Vec<KvBlock>,
    /// tokens a latent frame (h x w)
    pub tpf: usize,
    /// the full-history blocks (0..full); the rest are sink-recent
    pub full: usize,
}

impl Kv {
    /// The bytes a cache of these sizes takes
    pub fn need(blocks: usize, full: usize, tpf: usize, full_frames: usize, sr_frames: usize, int8: bool) -> usize {
        let row = if int8 { 2 * (DIM + HEADS * 4) } else { 2 * DIM * 2 };
        (full * full_frames + (blocks - full) * sr_frames) * tpf * row
    }

    /// `full` full-history blocks of `full_frames` latent frames at most, the rest `sr_frames` (the sink and one
    /// recent); `int8`: K and V in int8 with a scale a (row, head)
    pub fn new(gpu: &std::sync::Arc<Gpu>, blocks: usize, full: usize, tpf: usize, full_frames: usize, sr_frames: usize, int8: bool) -> Result<Kv> {
        if full > blocks || full_frames >= MAX_FRAMES || sr_frames >= MAX_FRAMES {
            return Err(Error(format!("kv cache: {full} of {blocks} blocks full-history, {full_frames} / {sr_frames} frames (the grid holds {MAX_FRAMES})")));
        }
        let mut v = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let cap_frames = if b < full { full_frames } else { sr_frames };
            let rows = cap_frames * tpf;
            let bytes = rows * DIM * if int8 { 1 } else { 2 };
            let ks = if int8 { Some((DevBuf::f32(gpu, rows * HEADS)?, DevBuf::f32(gpu, rows * HEADS)?)) } else { None };
            v.push(KvBlock { k: DevBuf::new(gpu, bytes)?, v: DevBuf::new(gpu, bytes)?, ks, cap_frames, frames: Vec::new() });
        }
        Ok(Kv { blocks: v, tpf, full })
    }

    /// The bytes it holds on the card
    pub fn bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.k.len + b.v.len + b.ks.as_ref().map_or(0, |(a, c)| a.len + c.len)).sum()
    }

    /// Whether it keeps int8
    pub fn int8(&self) -> bool {
        self.blocks.first().is_some_and(|b| b.ks.is_some())
    }

    /// Latent frames block `b` holds
    pub fn frames(&self, b: usize) -> usize {
        self.blocks[b].frames.iter().sum()
    }

    /// Chunks cached (read off block 0, as upstream's `should_recompute`)
    pub fn chunks(&self) -> usize {
        self.blocks.first().map_or(0, |b| b.frames.len())
    }

    /// Each block's chunk frame counts (for checks and logs)
    pub fn layout(&self, b: usize) -> &[usize] {
        &self.blocks[b].frames
    }

    pub fn clear(&mut self) {
        for b in &mut self.blocks {
            b.frames.clear();
        }
    }

    /// A sink-recent block before a chunk is encoded: back to its sink (the first chunk)
    fn trim_to_sink(&mut self, b: usize) {
        self.blocks[b].frames.truncate(1);
    }
}

/// What one pass does
pub enum Pass<'a> {
    /// a window's velocity: the text and its step row (the window's first chunk's timestep)
    Denoise { text: &'a Text, tsi: usize },
    /// one clean chunk into the cache (t = 0: `spans` at the zero step row)
    Encode,
    /// the cache rebuilt from the chunks of `spans` (t = 0)
    Recompute,
}

/// The working buffers of the passes, sized once for the largest window, history and text
pub struct RollBuf {
    rows: usize,
    x: DevBuf,
    h: DevBuf,
    big: DevBuf,
    att: DevBuf,
    mlp: DevBuf,
    inp: DevBuf,
    rx: DevBuf,
    tmp: DevBuf,
    /// [history | window | text] keys and values of one block's attention, half
    sk: DevBuf,
    sv: DevBuf,
    kv_rows: usize,
    tq_max: usize,
}

impl RollBuf {
    /// Room for passes of `pass_frames` latent frames (a window, a recompute's chunks) and `text` text rows (`tq` of
    /// them Qwen's), attending over `kv_frames` (the most cached + a window)
    pub fn new(dit: &Dit, gpu: &std::sync::Arc<Gpu>, tpf: usize, pass_frames: usize, kv_frames: usize, text: usize, tq: usize) -> Result<RollBuf> {
        let n = pass_frames * tpf;
        let rows = n + text;
        let kv_rows = kv_frames.max(pass_frames) * tpf + text;
        let silo = dit.k.silo().is_some();
        Ok(RollBuf {
            rows,
            x: DevBuf::f32(gpu, rows * DIM)?,
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            big: DevBuf::new(gpu, rows * 3 * DIM * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            mlp: DevBuf::new(gpu, rows * FFN * 2)?,
            inp: DevBuf::f32(gpu, n * IN)?,
            rx: DevBuf::f32(gpu, tq.max(1) * (4 * DIM + FFN + DIM))?,
            tmp: DevBuf::f32(gpu, if silo { 1 } else { rows * FFN })?,
            sk: DevBuf::new(gpu, kv_rows * DIM * 2)?,
            sv: DevBuf::new(gpu, kv_rows * DIM * 2)?,
            kv_rows,
            tq_max: tq,
        })
    }

    pub fn bytes(&self) -> usize {
        [&self.x, &self.h, &self.big, &self.att, &self.mlp, &self.inp, &self.rx, &self.tmp, &self.sk, &self.sv].iter().map(|b| b.len).sum()
    }
}

/// RoPE (cos, sin) rows of `f` latent frames of h x w from frame `f0` of the grid
pub fn rope_rows(f0: usize, f: usize, h: usize, w: usize) -> Vec<f32> {
    let mut cs = crate::dit::rope_table(f0 + f, h, w, 0);
    cs.drain(..f0 * h * w * HEAD);
    cs
}

impl Dit {
    /// One pass (see the module): `lat` float32 [n, 32] (frames, rows, columns; channels last), `spans` its chunks as
    /// (latent frames, step row of `st`) in order, `grid` (h, w). Denoise returns the velocity float32 [n, 32].
    #[allow(clippy::too_many_arguments)]
    pub fn roll_pass(&self, nsd: &Nsd, st: &Steps, rb: &RollBuf, kv: &mut Kv, lat: &DevBuf, spans: &[(usize, usize)], hw: (usize, usize), pass: Pass,
                     attn: Attn) -> Result<Option<DevBuf>> {
        let gpu = &nsd.gpu;
        let k = &self.k;
        let tpf = hw.0 * hw.1;
        if tpf != kv.tpf {
            return Err(Error(format!("roll: {} tokens a frame, the cache {}", tpf, kv.tpf)));
        }
        let frames: usize = spans.iter().map(|s| s.0).sum();
        let n = frames * tpf;
        if spans.iter().any(|s| s.0 == 0 || s.1 >= st.n) || n == 0 || lat.len < n * LATENT * 4 {
            return Err(Error(format!("roll: chunks {spans:?} against {} step rows and {} latent values", st.n, lat.len / 4)));
        }
        let t = match &pass {
            Pass::Denoise { text, tsi } => {
                if *tsi >= st.n || text.tq > rb.tq_max {
                    return Err(Error(format!("roll: text step row {tsi} of {}, {} text rows for {}", st.n, text.tq, rb.tq_max)));
                }
                text.tokens()
            }
            _ => 0,
        };
        if n + t > rb.rows {
            return Err(Error(format!("roll: {n} + {t} rows in buffers of {}", rb.rows)));
        }
        if matches!(pass, Pass::Recompute) {
            kv.clear();
        }
        // the hidden states: img_in over [latents | zeros (the condition copy, unused at inference) | 0 (the mask)]
        rb.inp.fill(0)?;
        rb.inp.copy_2d(0, IN * 4, lat, 0, LATENT * 4, LATENT * 4, n)?;
        k.linear32(rb.inp.ptr(), n, IN, w32(&self.img_in)?, DIM, bias(&self.img_in), rb.x.ptr())?;
        let mut text_pass = None;
        if let Pass::Denoise { text, tsi } = &pass {
            // the token refiner at the text's timestep, into rows n.. (the refine of a one-branch batch over views)
            let view = |b: &DevBuf| b.view(0, b.len);
            let bt = Batch {
                branches: vec![Branch { text, n, t, r0: 0, cs: rb.tmp.view(0, 4)? }],
                rows: n + t,
                grid: (frames, hw.0, hw.1),
                x: view(&rb.x)?,
                h: view(&rb.h)?,
                big: view(&rb.big)?,
                att: view(&rb.att)?,
                mlp: view(&rb.mlp)?,
                inp: view(&rb.inp)?,
                rx: view(&rb.rx)?,
                tmp: view(&rb.tmp)?,
            };
            self.refine(nsd, st, *tsi, &bt, &bt.branches[0], &mut |_, _, _, _| Ok(()))?;
            text_pass = Some(*tsi);
        }
        // the image rows of each chunk and their step rows
        let mut spans_rows: Vec<(usize, usize, usize)> = Vec::with_capacity(spans.len());
        let mut r = 0;
        for (f, s) in spans {
            spans_rows.push((r, f * tpf, *s));
            r += f * tpf;
        }
        let sc = &self.scratch;
        let mut n = n;
        // RoPE tables by (first frame, frames): the full-history group's and the sink-recent group's
        let mut tables: Vec<((usize, usize), DevBuf)> = Vec::new();
        let mut table = |f0: usize, f: usize| -> Result<*const std::ffi::c_void> {
            if f0 + f > MAX_FRAMES {
                return Err(Error(format!("roll: frames {f0}..{} past the RoPE grid's {MAX_FRAMES}", f0 + f)));
            }
            if let Some((_, b)) = tables.iter().find(|(key, _)| *key == (f0, f)) {
                return Ok(b.ptr().cast_const());
            }
            let b = DevBuf::from_f32(gpu, &rope_rows(f0, f, hw.0, hw.1))?;
            let p = b.ptr().cast_const();
            tables.push(((f0, f), b));
            Ok(p)
        };
        for (bi, d) in self.doubles.iter().enumerate() {
            let sr = bi >= kv.full;
            if matches!(pass, Pass::Recompute) && bi == kv.full && kv.full > 0 {
                // the sink-recent blocks rebuild from the first and the last chunk as the full-history blocks left them
                let (r0, rows0, s0) = spans_rows[0];
                let (rl, rowsl, sl) = *spans_rows.last().unwrap_or(&spans_rows[0]);
                if spans_rows.len() > 1 {
                    if rl != rows0 {
                        rb.x.copy_within(rows0 * DIM * 4, &rb.x, rl * DIM * 4, rowsl * DIM * 4)?;
                    }
                    spans_rows = vec![(r0, rows0, s0), (rows0, rowsl, sl)];
                }
                n = spans_rows.iter().map(|s| s.1).sum();
            }
            if matches!(pass, Pass::Encode) && sr {
                kv.trim_to_sink(bi);
            }
            let hrows = if matches!(pass, Pass::Recompute) { 0 } else { kv.frames(bi) * tpf };
            let f0 = hrows / tpf;
            let cs = table(f0, n / tpf)?;
            let (mi, mt) = (&st.dbl[bi].0, &st.dbl[bi].1);
            let md = |m: &DevBuf, s: usize, j: usize| -> *const f32 { fptr(m, s * 6 * DIM + j * DIM) };
            // modulate, project, norm and turn
            for (r0, rows, s) in &spans_rows {
                k.ln_mod(fptr(&rb.x, r0 * DIM), *rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(mi, *s, 1), md(mi, *s, 0), hptr(&rb.h, r0 * DIM), T::F16)?;
            }
            d.img.qkv.mm(nsd, k, sc, &rb.tmp, hptr(&rb.h, 0), DIM, n, hptr(&rb.big, 0), 3 * DIM, Post::Half)?;
            nsd.rms_rope(hptr(&rb.big, 0), Dt::F16, n, HEADS, HEAD, 3 * DIM, d.img.q_norm.ptr(), EPS, cs, HEAD)?;
            nsd.rms_rope(hptr(&rb.big, DIM), Dt::F16, n, HEADS, HEAD, 3 * DIM, d.img.k_norm.ptr(), EPS, cs, HEAD)?;
            if let Some(tsi) = text_pass {
                k.ln_mod(fptr(&rb.x, n * DIM), t, DIM, EPS, std::ptr::null(), std::ptr::null(), md(mt, tsi, 1), md(mt, tsi, 0), hptr(&rb.h, n * DIM), T::F16)?;
                d.txt.qkv.mm(nsd, k, sc, &rb.tmp, hptr(&rb.h, n * DIM), DIM, t, hptr(&rb.big, n * 3 * DIM), 3 * DIM, Post::Half)?;
                nsd.rms_rope(hptr(&rb.big, n * 3 * DIM), Dt::F16, t, HEADS, HEAD, 3 * DIM, d.txt.q_norm.ptr(), EPS, cs, 0)?;
                nsd.rms_rope(hptr(&rb.big, n * 3 * DIM + DIM), Dt::F16, t, HEADS, HEAD, 3 * DIM, d.txt.k_norm.ptr(), EPS, cs, 0)?;
            }
            // the keys and values [history | window | text]
            let total = hrows + n + t;
            if total > rb.kv_rows {
                return Err(Error(format!("roll: block {bi} attends over {total} rows, the buffers hold {}", rb.kv_rows)));
            }
            let blk = &kv.blocks[bi];
            if hrows > 0 {
                match &blk.ks {
                    Some((ks, vs)) => {
                        k.kv_dq8(blk.k.ptr(), ks.ptr(), hrows, HEADS, rb.sk.ptr(), DIM)?;
                        k.kv_dq8(blk.v.ptr(), vs.ptr(), hrows, HEADS, rb.sv.ptr(), DIM)?;
                    }
                    None => {
                        rb.sk.copy_within(0, &blk.k, 0, hrows * DIM * 2)?;
                        rb.sv.copy_within(0, &blk.v, 0, hrows * DIM * 2)?;
                    }
                }
            }
            let rows_in = n + t;
            rb.sk.copy_2d(hrows * DIM * 2, DIM * 2, &rb.big, DIM * 2, 3 * DIM * 2, DIM * 2, rows_in)?;
            rb.sv.copy_2d(hrows * DIM * 2, DIM * 2, &rb.big, 2 * DIM * 2, 3 * DIM * 2, DIM * 2, rows_in)?;
            let q = |row: usize| hptr(&rb.big, row * 3 * DIM).cast_const();
            let kr = |row: usize| hptr(&rb.sk, row * DIM).cast_const();
            let vr = |row: usize| hptr(&rb.sv, row * DIM).cast_const();
            let out = |row: usize| hptr(&rb.att, row * DIM);
            let att = |qrow: usize, sq: usize, krow: usize, skv: usize, orow: usize| -> Result<()> {
                match attn {
                    Attn::Half(tile) if k.has_attention2() => k.attention2(q(qrow), 3 * DIM, sq, kr(krow), vr(krow), DIM, skv, HEADS, HEAD, out(orow), DIM, false, tile),
                    Attn::Sage if k.has_attention2() => k.attention2(q(qrow), 3 * DIM, sq, kr(krow), vr(krow), DIM, skv, HEADS, HEAD, out(orow), DIM, true, 0),
                    _ => nsd.attention_qk(q(qrow), sq, 3 * DIM, kr(krow), vr(krow), skv, DIM, Dt::F16, HEADS, HEAD, out(orow), Dt::F16),
                }
            };
            match &pass {
                Pass::Denoise { .. } => {
                    att(0, n, 0, total, 0)?;
                    att(n, t, hrows, n + t, n)?;
                }
                Pass::Encode => att(0, n, 0, hrows + n, 0)?,
                Pass::Recompute => {
                    // block-causal: chunk i over chunks 0..=i
                    for (r0, rows, _) in &spans_rows {
                        att(*r0, *rows, 0, r0 + rows, *r0)?;
                    }
                }
            }
            // into the cache: the chunk appended, or the rebuilt chunks
            match &pass {
                Pass::Encode | Pass::Recompute => {
                    let at = if matches!(pass, Pass::Encode) { hrows } else { 0 };
                    let new_frames: Vec<usize> = spans_rows.iter().map(|s| s.1 / tpf).collect();
                    let blk = &mut kv.blocks[bi];
                    let have = if matches!(pass, Pass::Encode) { blk.frames.iter().sum::<usize>() } else { 0 };
                    if have + n / tpf > blk.cap_frames {
                        return Err(Error(format!("roll: block {bi}'s cache holds {} latent frames, {} asked (raise the cache or recompute sooner)",
                                                 blk.cap_frames, have + n / tpf)));
                    }
                    match &blk.ks {
                        Some((ks, vs)) => {
                            k.kv_q8(hptr(&rb.sk, hrows * DIM), n, HEADS, DIM, blk.k.ptr().cast::<i8>().wrapping_add(at * DIM).cast(),
                                    fptr(ks, at * HEADS).cast())?;
                            k.kv_q8(hptr(&rb.sv, hrows * DIM), n, HEADS, DIM, blk.v.ptr().cast::<i8>().wrapping_add(at * DIM).cast(),
                                    fptr(vs, at * HEADS).cast())?;
                        }
                        None => {
                            blk.k.copy_within(at * DIM * 2, &rb.sk, hrows * DIM * 2, n * DIM * 2)?;
                            blk.v.copy_within(at * DIM * 2, &rb.sv, hrows * DIM * 2, n * DIM * 2)?;
                        }
                    }
                    if matches!(pass, Pass::Encode) {
                        blk.frames.push(n / tpf);
                    } else {
                        blk.frames = new_frames;
                    }
                }
                Pass::Denoise { .. } => {}
            }
            // the rest of the block: projection and gated residual, norm, MLP
            for (r0, rows, s) in &spans_rows {
                d.img.proj.mm(nsd, k, sc, &rb.tmp, hptr(&rb.att, r0 * DIM), DIM, *rows, fptr(&rb.x, r0 * DIM).cast(), DIM, Post::Gate(md(mi, *s, 2)))?;
                k.ln_mod(fptr(&rb.x, r0 * DIM), *rows, DIM, EPS, std::ptr::null(), std::ptr::null(), md(mi, *s, 4), md(mi, *s, 3), hptr(&rb.h, r0 * DIM), T::F16)?;
                d.img.fc1.mm(nsd, k, sc, &rb.tmp, hptr(&rb.h, r0 * DIM), DIM, *rows, hptr(&rb.mlp, r0 * FFN), FFN, Post::Gelu)?;
                d.img.fc2.mm(nsd, k, sc, &rb.tmp, hptr(&rb.mlp, r0 * FFN), FFN, *rows, fptr(&rb.x, r0 * DIM).cast(), DIM, Post::Gate(md(mi, *s, 5)))?;
            }
            if let Some(tsi) = text_pass {
                let r0 = n;
                d.txt.proj.mm(nsd, k, sc, &rb.tmp, hptr(&rb.att, r0 * DIM), DIM, t, fptr(&rb.x, r0 * DIM).cast(), DIM, Post::Gate(md(mt, tsi, 2)))?;
                k.ln_mod(fptr(&rb.x, r0 * DIM), t, DIM, EPS, std::ptr::null(), std::ptr::null(), md(mt, tsi, 4), md(mt, tsi, 3), hptr(&rb.h, r0 * DIM), T::F16)?;
                d.txt.fc1.mm(nsd, k, sc, &rb.tmp, hptr(&rb.h, r0 * DIM), DIM, t, hptr(&rb.mlp, r0 * FFN), FFN, Post::Gelu)?;
                d.txt.fc2.mm(nsd, k, sc, &rb.tmp, hptr(&rb.mlp, r0 * FFN), FFN, t, fptr(&rb.x, r0 * DIM).cast(), DIM, Post::Gate(md(mt, tsi, 5)))?;
            }
        }
        if !matches!(pass, Pass::Denoise { .. }) {
            nsd.wait()?;
            return Ok(None);
        }
        // the final layer, frame by frame
        let o = DevBuf::f32(gpu, n * DIM)?;
        for (r0, rows, s) in &spans_rows {
            k.ln_mod(fptr(&rb.x, r0 * DIM), *rows, DIM, EPS, std::ptr::null(), std::ptr::null(), fptr(&st.fin, s * 2 * DIM + DIM), fptr(&st.fin, s * 2 * DIM),
                     fptr(&o, r0 * DIM).cast(), T::F32)?;
        }
        let v = DevBuf::f32(gpu, n * LATENT)?;
        k.linear32(o.ptr(), n, DIM, w32(&self.final_lin)?, LATENT, bias(&self.final_lin), v.ptr())?;
        nsd.wait()?;
        Ok(Some(v))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rope_rows_start_at_their_frame() {
        let (h, w, head) = (2, 3, super::HEAD);
        let all = crate::dit::rope_table(4, h, w, 0);
        let part = super::rope_rows(2, 2, h, w);
        assert_eq!(part.len(), 2 * h * w * head);
        assert_eq!(&part[..], &all[2 * h * w * head..]);
    }
}
