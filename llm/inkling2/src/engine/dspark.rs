//! Inkling 2 Mini's DSpark drafter (its `mtp.safetensors`, Unsloth's `dspark-Inkling-2-Mini-*.gguf`, arch `dflash`
//! with a Markov head; the reference is SGLang's models/inkling2_mtp.py, models/dspark.py, models/dflash.py and
//! kernels/ops/speculative/inkling2_mtp.py). Every position the target passes over leaves four features: the depth
//! attention's running stream going into layer applications 5, 23 and 41 and the head's normed input, each rounded to
//! BF16. A drafter layer's context is their mix (`dspark_fuse` softmaxed, a row a layer) RMS-normed (`dflare_norm`),
//! its k (normed, NEOX-rotated) and v into the layer's cache. A draft is one pass of [the last token (the target's
//! normed embedding), MASK x n] at s.pos.. through three layers (q / k norms, NEOX rotation at theta 1e6, attention
//! over the context and the whole block - not causal; layers 0 and 2 over a 1,024 window - a SwiGLU), the drafter's
//! norm, the target's head (base logits); then the Markov head a position at a time: an RNN over [state, w1[previous
//! token], the hidden row, f_rnn] (f_rnn: the last committed position's features mixed by the fuse's last row, normed
//! by `dspark_norm`), its latent folded through the head (the head times `dspark_hidden_out`, computed at load) as a
//! bias on the base logits; the draft the argmax of softcap(base + bias).

use super::*;

/// The features the target leaves a position: the streams into these layer applications, then the head's input
pub const FEATS: usize = 4;

pub struct DLayer {
    attn_norm: DevBuf,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: DevBuf,
    k_norm: DevBuf,
    k_ctx: Mat,
    v_ctx: Mat,
    k_ctx_norm: DevBuf,
    ctx_norm: DevBuf,
    ffn_norm: DevBuf,
    gate: Mat,
    up: Mat,
    down: Mat,
    /// its attention's window (0: everything)
    win: usize,
}

pub struct DSpark {
    layers: Vec<DLayer>,
    /// softmax(dspark_fuse) rows: a layer's context mix each, the last f_rnn's
    alpha: Vec<DevBuf>,
    norm: DevBuf,
    frnn_norm: DevBuf,
    /// the mask embedding a row, MAXV rows
    mask_rows: DevBuf,
    w1: DevBuf,
    joint: Mat,
    joint_b: DevBuf,
    /// head x dspark_hidden_out [vocab_pad, rank] BF16, head x its bias [vocab_pad]
    fold: Mat,
    fold_b: DevBuf,
    /// the target's application indices whose input streams are features 0..3
    pub taps: Vec<usize>,
    rank: usize,
    heads: usize,
    heads_kv: usize,
    ff: usize,
    theta: f32,
    eps: f32,
    pub block: usize,
    pub bytes: u64,
}

/// A session's drafter state: each layer's cache (keys, values, slots), f_rnn of the last pass's rows (all of them for
/// decode passes, the last otherwise)
pub struct DsState {
    caches: Vec<(DevBuf, DevBuf, usize)>,
    frnn: DevBuf,
    frnn_all: bool,
}

impl DSpark {
    /// The bytes a session's caches take
    pub fn session_bytes(g: &Geometry, max_ctx: usize, chunk: usize, block: usize) -> usize {
        let row = g.heads_kv * g.head * 2 * 2;
        [g.window, 0, g.window].iter().map(|&w| row * cap(w, max_ctx, chunk, block)).sum::<usize>() + MAXV * g.n_embd * 4
    }
}

fn cap(win: usize, max_ctx: usize, chunk: usize, block: usize) -> usize {
    if win > 0 { (win + chunk + block + MAXV).min(max_ctx + block) } else { max_ctx + block }
}

/// NS_INK2_DRAFT: the drafter's GGUF; else the dspark-*.gguf beside the model (its smallest)
pub fn draft_path(f: &Gguf) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NS_INK2_DRAFT") {
        return Some(p.into());
    }
    let dir = f.paths.first()?.parent()?;
    let mut c: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("dspark-") && n.ends_with(".gguf")))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.len(), p))).collect();
    c.sort();
    c.into_iter().next().map(|(_, p)| p)
}

/// NS_INK2_DRAFT_PMIN: a draft (and those after it) dropped when the drafter gives it less than this probability
pub fn draft_pmin() -> f32 {
    std::env::var("NS_INK2_DRAFT_PMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.7)
}

/// NS_INK2_DRAFTS: tokens drafted a step (at most the block's less one; 0: none)
pub fn draft_count() -> Option<usize> {
    std::env::var("NS_INK2_DRAFTS").ok().and_then(|v| v.parse().ok())
}

/// The drafter from its GGUF (the silo needed: its kernels), the fold through the target's head computed here
pub fn load(d: &Gguf, g: &Geometry, gpu: &Arc<Gpu>, k: &ffi::Api, s: &'static ffi::Silo, output: &Mat, arena: &Arena) -> Result<DSpark> {
    let loaded = std::cell::Cell::new(0u64);
    let read = |name: &str| -> Result<Mat> { mat(d, gpu, Some(s), name, &loaded) };
    let meta_u = |key: &str| d.meta(&format!("dflash.{key}")).and_then(|v| v.as_u64());
    let block = meta_u("block_size").unwrap_or(8) as usize;
    let nl = meta_u("block_count").ok_or_else(|| e("the drafter: no dflash.block_count"))? as usize;
    let heads = meta_u("attention.head_count").unwrap_or(24) as usize;
    let heads_kv = meta_u("attention.head_count_kv").unwrap_or(8) as usize;
    let ff = meta_u("feed_forward_length").unwrap_or(9216) as usize;
    let win = meta_u("attention.sliding_window").unwrap_or(1024) as usize;
    let theta = d.meta("dflash.rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(1e6) as f32;
    let eps = d.meta("dflash.attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let pattern: Vec<bool> = d.meta("dflash.attention.sliding_window_pattern").and_then(|v| v.as_array())
        .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) != 0).collect()).unwrap_or_else(|| vec![true, false, true]);
    let taps: Vec<usize> = d.meta("dflash.target_layers").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect()).unwrap_or_default();
    if heads_kv != g.heads_kv || heads != g.heads || taps.len() != FEATS || *taps.last().unwrap_or(&0) != g.order().len() {
        return Err(e(format!("the drafter: {heads} / {heads_kv} heads, target layers {taps:?} - not this model's")));
    }
    if block > MAXV {
        return Err(e(format!("the drafter's block of {block}: at most {MAXV}")));
    }
    let mut layers = Vec::with_capacity(nl);
    for l in 0..nl {
        let n = |x: &str| format!("blk.{l}.{x}.weight");
        layers.push(DLayer {
            attn_norm: read(&n("attn_norm"))?.buf,
            q: read(&n("attn_q"))?,
            k: read(&n("attn_k"))?,
            v: read(&n("attn_v"))?,
            o: read(&n("attn_output"))?,
            q_norm: read(&n("attn_q_norm"))?.buf,
            k_norm: read(&n("attn_k_norm"))?.buf,
            k_ctx: read(&n("attn_k_ctx"))?,
            v_ctx: read(&n("attn_v_ctx"))?,
            k_ctx_norm: read(&n("attn_k_ctx_norm"))?.buf,
            ctx_norm: read(&n("dflare_norm"))?.buf,
            ffn_norm: read(&n("ffn_norm"))?.buf,
            gate: read(&n("ffn_gate"))?,
            up: read(&n("ffn_up"))?,
            down: read(&n("ffn_down"))?,
            win: if pattern.get(l).copied().unwrap_or(false) { win } else { 0 },
        });
    }
    for dl in &layers {
        for m in [&dl.q, &dl.k, &dl.v, &dl.o, &dl.k_ctx, &dl.v_ctx, &dl.gate, &dl.up, &dl.down] {
            if !m.packed {
                return Err(e(format!("the drafter: a {} projection (the silo takes Q8_0 / NVFP4)", m.ty.name())));
            }
        }
    }
    // the fuse's rows softmaxed (on the host: 4 x 4)
    let fuse = d.tensor("dspark_fuse.weight").ok_or_else(|| e("the drafter: no dspark_fuse.weight"))?;
    let raw = d.read(fuse).map_err(|x| e(x.0))?;
    let fw: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if fw.len() != (nl + 1) * FEATS {
        return Err(e(format!("the drafter: dspark_fuse of {} values, {} expected", fw.len(), (nl + 1) * FEATS)));
    }
    let alpha = fw.chunks_exact(FEATS).map(|r| {
        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = r.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = ex.iter().sum();
        DevBuf::from_f32(gpu, &ex.iter().map(|v| v / sum).collect::<Vec<f32>>())
    }).collect::<Result<Vec<_>>>()?;
    let mask = d.tensor("mask_embd.weight").ok_or_else(|| e("the drafter: no mask_embd.weight"))?;
    let mraw = d.read(mask).map_err(|x| e(x.0))?;
    let mask_rows = DevBuf::new(gpu, MAXV * mraw.len())?;
    for r in 0..MAXV {
        mask_rows.write(r * mraw.len(), &mraw)?;
    }
    let w1t = d.tensor("markov_w1.weight").ok_or_else(|| e("the drafter: no markov_w1.weight"))?;
    if w1t.ty != GType::BF16 {
        return Err(e("the drafter: markov_w1 not BF16"));
    }
    let rank = w1t.shape[w1t.shape.len() - 1] as usize;
    let w1raw = d.read(w1t).map_err(|x| e(x.0))?;
    let w1 = DevBuf::new(gpu, w1raw.len())?;
    w1.write(0, &w1raw)?;
    let joint = read("joint_proj.weight")?;
    let joint_b = read("joint_proj.bias")?.buf;
    if joint.rows != 3 * rank || joint.cols != 2 * rank + 2 * g.n_embd || joint.ty != GType::BF16 {
        return Err(e(format!("the drafter: joint_proj {}x{} {}", joint.rows, joint.cols, joint.ty.name())));
    }
    // the fold: head [vocab_pad, C] . hidden_out [C, rank] (and its bias [C]) - the head's rows in chunks expanded to half,
    // against hidden_out^T and the bias as an extra row in half; the products in float32, kept in BF16 (the bias float32)
    let ho = d.tensor("dspark_hidden_out.weight").ok_or_else(|| e("the drafter: no dspark_hidden_out.weight"))?;
    let hb = d.tensor("dspark_hidden_out.bias").ok_or_else(|| e("the drafter: no dspark_hidden_out.bias"))?;
    let c = g.n_embd;
    if ho.ty != GType::BF16 || ho.shape[0] as usize != c || ho.shape[1] as usize != rank {
        return Err(e("the drafter: dspark_hidden_out not BF16 [C, rank]"));
    }
    let hor = d.read(ho).map_err(|x| e(x.0))?;
    let hbr = d.read(hb).map_err(|x| e(x.0))?;
    let bf = |b: &[u8], i: usize| f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16);
    let mut wt = vec![0f32; (rank + 1) * c];
    for cc in 0..c {
        for j in 0..rank {
            wt[j * c + cc] = bf(&hor, cc * rank + j);
        }
        wt[rank * c + cc] = f32::from_le_bytes([hbr[4 * cc], hbr[4 * cc + 1], hbr[4 * cc + 2], hbr[4 * cc + 3]]);
    }
    let wt32 = DevBuf::from_f32(gpu, &wt)?;
    let wt_dev = DevBuf::new(gpu, wt.len() * 2)?;
    // SAFETY: (rank + 1) x c floats -> half.
    unsafe { ffi::check((k.to_f16)(gpu.raw(), wt32.fp(), wt_dev.ptr().cast(), wt.len() as i64), "fold half")? };
    let vp = output.rows;
    let fold_buf = DevBuf::new(gpu, vp * rank * 2)?;
    let fold_b = DevBuf::new(gpu, vp * 4)?;
    let mark = arena.mark();
    const CH: usize = 4096;
    let xh = arena.bytes(CH * c * 2)?;
    let y = arena.bytes(CH * (rank + 1) * 4)?;
    let yw = arena.bytes(CH * rank * 4)?;
    let raw_g = gpu.raw();
    let mut r0 = 0;
    while r0 < vp {
        let rn = CH.min(vp - r0);
        // SAFETY: rows r0.. of the repacked head -> half [rn, c]; y [rn, rank + 1] = xh . wt^T; the columns split into
        // the fold's BF16 rows and the bias.
        unsafe {
            if output.packed {
                ffi::check((s.unpack_f16)(raw_g, output.ty.code() as c_int, output.buf.ptr(), vp as i64, r0 as i64, rn as i64, c as i64, xh.ptr().cast()),
                           "fold unpack")?;
            } else {
                // a head in its stored blocks (UD-Q8_K_XL's BF16): the chunk's rows from their offset
                let row = output.ty.bytes(c as u64).ok_or_else(|| e("the head's row size"))? as usize;
                let src = output.buf.view(r0 * row, rn * row)?;
                ffi::check((k.dequant_f16)(raw_g, output.ty.code() as c_int, src.ptr(), (rn * c) as i64, xh.ptr().cast()), "fold expand")?;
            }
            ffi::check((k.gemm_f16)(raw_g, rn as i64, (rank + 1) as i64, c as i64, xh.ptr().cast(), c as i64, wt_dev.ptr().cast(), y.fp(), (rank + 1) as i64, 0),
                       "fold gemm")?;
        }
        gpu.sync()?;
        let mut yh = vec![0u8; rn * (rank + 1) * 4];
        y.read(0, &mut yh)?;
        let mut wrows = vec![0u8; rn * rank * 4];
        let mut brow = vec![0u8; rn * 4];
        for r in 0..rn {
            wrows[r * rank * 4..(r + 1) * rank * 4].copy_from_slice(&yh[r * (rank + 1) * 4..(r * (rank + 1) + rank) * 4]);
            brow[r * 4..r * 4 + 4].copy_from_slice(&yh[(r * (rank + 1) + rank) * 4..(r * (rank + 1) + rank + 1) * 4]);
        }
        yw.write(0, &wrows)?;
        fold_b.write(r0 * 4, &brow)?;
        // SAFETY: rn x rank floats -> BF16 into the fold's rows.
        unsafe { ffi::check((s.draft.to_bf16)(raw_g, yw.fp(), fold_buf.view(r0 * rank * 2, rn * rank * 2)?.ptr().cast(), (rn * rank) as i64), "fold bf16")? };
        gpu.sync()?;
        r0 += rn;
    }
    arena.rewind(mark);
    let fold = Mat { buf: fold_buf, ty: GType::BF16, rows: vp, cols: rank, packed: false };
    let bytes = loaded.get() + w1raw.len() as u64 + (vp * rank * 2 + vp * 4) as u64;
    Ok(DSpark { layers, alpha, norm: read("output_norm.weight")?.buf, frnn_norm: read("dspark_norm.weight")?.buf, mask_rows, w1, joint, joint_b, fold, fold_b,
                taps: taps[..FEATS - 1].to_vec(), rank, heads, heads_kv, ff, theta, eps, block, bytes })
}

impl DSpark {
    /// The buffers a checkpoint keeps for the drafter: each layer's keys and values, then f_rnn
    pub fn save_bufs(&self) -> usize {
        2 * self.layers.len() + 1
    }
}

impl Inkling2<'_> {
    /// One session's drafter state into another's (positions the drafter still reads up to `pos`; f_rnn's rows)
    pub(super) fn ds_copy(&self, dst: &mut DsState, src: &DsState, pos: usize, _last_rows: usize) -> Result<()> {
        let Some(d) = &self.ds else { return Ok(()) };
        for (dl, ((dk, dv, dc), (sk, sv, sc))) in d.layers.iter().zip(dst.caches.iter().zip(&src.caches)) {
            let from = Self::kept(dl.win, pos);
            self.copy_ring(dk, *dc, sk, *sc, d.heads_kv, from, pos)?;
            self.copy_ring(dv, *dc, sv, *sc, d.heads_kv, from, pos)?;
        }
        dst.frnn.copy_within(0, &src.frnn, 0, src.frnn.len)?;
        dst.frnn_all = src.frnn_all;
        Ok(())
    }

    /// The drafter's part of a checkpoint: each layer's keys and values it still reads at `pos`, f_rnn of the last
    /// committed row
    pub(super) fn ds_save(&self, st: &DsState, pos: usize, last_rows: usize) -> Result<Vec<Vec<u8>>> {
        let Some(d) = &self.ds else { return Ok(Vec::new()) };
        let row = self.g.head * 2;
        let mut bufs = Vec::new();
        for (dl, (kc, vc, cp)) in d.layers.iter().zip(&st.caches) {
            let from = Self::kept(dl.win, pos);
            let n = pos - from;
            for c in [kc, vc] {
                let mut b = vec![0u8; d.heads_kv * n * row];
                for h in 0..d.heads_kv {
                    let mut p = from;
                    while p < pos {
                        let si = p % cp;
                        let run = (pos - p).min(cp - si);
                        let at = (h * n + p - from) * row;
                        c.read((h * cp + si) * row, &mut b[at..at + run * row])?;
                        p += run;
                    }
                }
                bufs.push(b);
            }
        }
        let c = self.g.n_embd;
        let r = if st.frnn_all { last_rows.saturating_sub(1) } else { 0 };
        let mut fb = vec![0u8; c * 4];
        st.frnn.read(r * c * 4, &mut fb)?;
        bufs.push(fb);
        Ok(bufs)
    }

    pub(super) fn ds_restore(&self, st: &mut DsState, pos: usize, bufs: &[&Vec<u8>]) -> Result<()> {
        let Some(d) = &self.ds else { return Ok(()) };
        let row = self.g.head * 2;
        let mut it = bufs.iter();
        for (dl, (kc, vc, cp)) in d.layers.iter().zip(&st.caches) {
            let from = Self::kept(dl.win, pos);
            let n = pos - from;
            for c in [kc, vc] {
                let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                if b.len() != d.heads_kv * n * row {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..d.heads_kv {
                    let mut p = from;
                    while p < pos {
                        let si = p % cp;
                        let run = (pos - p).min(cp - si);
                        let at = (h * n + p - from) * row;
                        c.write((h * cp + si) * row, &b[at..at + run * row])?;
                        p += run;
                    }
                }
            }
        }
        let fb = it.next().ok_or_else(|| e("a short checkpoint"))?;
        if fb.len() != self.g.n_embd * 4 {
            return Err(e("a checkpoint of another shape"));
        }
        st.frnn.write(0, fb)?;
        st.frnn_all = false;
        Ok(())
    }

    /// A session's drafter state
    pub(super) fn ds_state(&self, max_ctx: usize) -> Result<Option<DsState>> {
        let Some(d) = &self.ds else { return Ok(None) };
        let g = &self.g;
        let row = d.heads_kv * g.head * 2;
        let caches = d.layers.iter().map(|dl| {
            let cp = cap(dl.win, max_ctx, self.chunk, d.block);
            Ok((DevBuf::new(&self.gpu, row * cp)?, DevBuf::new(&self.gpu, row * cp)?, cp))
        }).collect::<Result<Vec<_>>>()?;
        Ok(Some(DsState { caches, frnn: DevBuf::new(&self.gpu, MAXV * g.n_embd * 4)?, frnn_all: false }))
    }

    /// After a pass of t rows at p0 (its features in self.feats): each drafter layer's context k / v into its cache, and
    /// f_rnn of the rows (decode passes: each; else the last)
    pub(super) fn ds_inject(&self, s: &mut Session, t: usize, p0: usize) -> Result<()> {
        let (Some(d), Some(st), Some(feats), Some(sl)) = (&self.ds, s.ds.as_mut(), &self.feats, self.silo) else { return Ok(()) };
        let (g, dk) = (&self.g, &sl.draft);
        let c = g.n_embd;
        let kd = d.heads_kv * g.head;
        let mark = self.arena.mark();
        for (i, (dl, (kc, vc, cp))) in d.layers.iter().zip(&st.caches).enumerate() {
            let ctx = self.f32(t * c)?;
            let xq = match self.dec(t) {
                // SAFETY: arithmetic only.
                Some(s) => Some(self.ab(unsafe { (s.xq_bytes)(t as i64, c as i64) })?),
                None => None,
            };
            // SAFETY: feats [t, 4, c], the layer's mix and norm -> ctx [t, c] (and its quants).
            self.op("draft ctx norm", || unsafe {
                (dk.feat_norm)(self.raw(), feats.fp(), FEATS as i64, d.alpha[i].fp(), dl.ctx_norm.fp(), d.eps, ctx.fp(), xq.as_ref().map_or(std::ptr::null_mut(), |q| q.ptr()),
                               t as i64, c as i64)
            })?;
            let kx = self.mmq(&dl.k_ctx, &ctx, xq.as_ref(), t)?;
            let vx = self.mmq(&dl.v_ctx, &ctx, xq.as_ref(), t)?;
            // SAFETY: kx [t, kd] normed and rotated in place; then k, v into the cache at p0..
            unsafe {
                self.op("draft k norm rope", || (dk.qk_norm_rope)(self.raw(), kx.fp(), t as i64, d.heads_kv as i64, dl.k_ctx_norm.fp(), d.eps, d.theta, p0 as i64))?;
                self.op("draft kv store", || (self.k.kv_store)(self.raw(), kx.fp(), vx.fp(), t as i64, d.heads_kv as i64, g.head as i64, *cp as i64, p0 as i64,
                                                               kc.ptr(), vc.ptr()))?;
            }
            let _ = kd;
            self.arena.rewind(mark);
        }
        // f_rnn: the rows' features by the fuse's last row, normed
        let (from, n) = if t <= MAXV { (0, t) } else { (t - 1, 1) };
        let fv = feats.view(from * FEATS * c * 4, n * FEATS * c * 4)?;
        // SAFETY: the rows' features -> f_rnn [n, c].
        self.op("draft f_rnn", || unsafe {
            (dk.feat_norm)(self.raw(), fv.fp(), FEATS as i64, d.alpha[d.layers.len()].fp(), d.frnn_norm.fp(), d.eps, st.frnn.fp(), std::ptr::null_mut(), n as i64, c as i64)
        })?;
        st.frnn_all = t <= MAXV;
        Ok(())
    }

    /// Up to `nd` drafts after `tail` (committed at s.pos, not fed), `last` the last pass's row the target committed
    /// last (its f_rnn): one pass of the drafter over [tail, MASK x nd], then the Markov head's nd steps
    pub(super) fn ds_draft(&self, s: &Session, tail: u32, last: usize, nd: usize) -> Result<Vec<u32>> {
        let (Some(d), Some(st), Some(sl)) = (&self.ds, s.ds.as_ref(), self.silo) else { return Ok(Vec::new()) };
        let nd = nd.min(d.block - 1).min(s.max_ctx.saturating_sub(s.pos + 1));
        if nd == 0 {
            return Ok(Vec::new());
        }
        let (g, dk) = (&self.g, &sl.draft);
        let (c, rows, p0) = (g.n_embd, nd + 1, s.pos);
        let (qd, kd) = (d.heads * g.head, d.heads_kv * g.head);
        self.arena.reset();
        // the block: the tail's normed embedding, then the mask's rows
        let x = self.f32(rows * c)?;
        let emb = self.embed(&[tail])?;
        let x0 = self.rms(&emb, &self.tok_norm, 1)?;
        x.copy_within(0, &x0, 0, c * 4)?;
        x.copy_within(c * 4, &d.mask_rows, 0, nd * c * 4)?;
        let xq_of = |w: &DevBuf| -> Result<DevBuf> {
            // SAFETY: arithmetic only; then x [rows, c] normed by w into its quants.
            let q = self.ab(unsafe { (sl.xq_bytes)(rows as i64, c as i64) })?;
            self.op("draft rms xq", || unsafe { (sl.rms_xq)(self.raw(), x.fp(), w.fp(), std::ptr::null_mut(), q.ptr(), rows as i64, c as i64, d.eps) })?;
            Ok(q)
        };
        for (dl, (kc, vc, cp)) in d.layers.iter().zip(&st.caches) {
            let mark = self.arena.mark();
            let hq = xq_of(&dl.attn_norm)?;
            let [q, kx, vx]: [DevBuf; 3] = self.mm_many(&[(&dl.q, &x, Some(&hq)), (&dl.k, &x, Some(&hq)), (&dl.v, &x, Some(&hq))], rows)?
                .try_into().map_err(|_| e("three products"))?;
            // SAFETY: arithmetic only.
            let ns = unsafe { (sl.fused.attn_splits)(rows as i64, p0 as i64, dl.win as i64) } as usize;
            let part = self.f32(rows * d.heads * ns * (g.head + 2))?;
            let att = self.f32(rows * qd)?;
            // SAFETY: arithmetic only.
            let aq = self.ab(unsafe { (sl.xq_bytes)(rows as i64, qd as i64) })?;
            // SAFETY: the rows and caches as silo.h names them.
            unsafe {
                self.op("draft q norm rope", || (dk.qk_norm_rope)(self.raw(), q.fp(), rows as i64, d.heads as i64, dl.q_norm.fp(), d.eps, d.theta, p0 as i64))?;
                self.op("draft k norm rope", || (dk.qk_norm_rope)(self.raw(), kx.fp(), rows as i64, d.heads_kv as i64, dl.k_norm.fp(), d.eps, d.theta, p0 as i64))?;
                self.op("draft kv store", || (self.k.kv_store)(self.raw(), kx.fp(), vx.fp(), rows as i64, d.heads_kv as i64, g.head as i64, *cp as i64, p0 as i64,
                                                               kc.ptr(), vc.ptr()))?;
                self.op("draft attention", || (dk.attn_part_block)(self.raw(), q.fp(), kc.ptr(), vc.ptr(), part.fp(), rows as i64, d.heads as i64, d.heads_kv as i64,
                                                                    *cp as i64, p0 as i64, dl.win as i64, 1.0 / (g.head as f32).sqrt()))?;
                self.op("draft merge", || (sl.fused.merge_gate_xq)(self.raw(), part.fp(), std::ptr::null(), std::ptr::null(), att.fp(), aq.ptr(), rows as i64,
                                                                    d.heads as i64, ns as i64))?;
            }
            let _ = kd;
            let o = self.mmq(&dl.o, &att, Some(&aq), rows)?;
            // SAFETY: rows x c floats each.
            self.op("draft add", || unsafe { (self.k.add)(self.raw(), x.fp(), o.fp(), x.fp(), (rows * c) as i64) })?;
            let fq = xq_of(&dl.ffn_norm)?;
            let [fg, fu]: [DevBuf; 2] = self.mm_many(&[(&dl.gate, &x, Some(&fq)), (&dl.up, &x, Some(&fq))], rows)?.try_into().map_err(|_| e("two products"))?;
            // SAFETY: arithmetic only.
            let xd = self.ab(unsafe { (sl.xq_bytes)(rows as i64, d.ff as i64) })?;
            // SAFETY: gate and up [rows, ff] -> silu(gate) up quantized.
            self.op("draft swiglu xq", || unsafe { (dk.swiglu_xq)(self.raw(), fg.fp(), fu.fp(), xd.ptr(), rows as i64, d.ff as i64) })?;
            let dn = self.mmq(&dl.down, &fg, Some(&xd), rows)?;
            // SAFETY: rows x c floats each.
            self.op("draft add", || unsafe { (self.k.add)(self.raw(), x.fp(), dn.fp(), x.fp(), (rows * c) as i64) })?;
            self.arena.rewind(mark);
        }
        // the block's rows after the first: the drafter's norm, the target's head (base logits)
        let after = x.view(c * 4, nd * c * 4)?;
        let (hid, hq) = self.rms_q(&after, &d.norm, nd)?;
        let vp = self.output.rows;
        let base = self.mmq(&self.output, &hid, hq.as_ref(), nd)?;
        // the Markov head, a position at a time: the token before it (the tail first), the hidden row, f_rnn
        let toks = self.ab((nd + 1) * 4)?;
        toks.write_async(0, &tail.to_le_bytes())?;
        let state = self.f32(d.rank)?;
        state.fill(0)?;
        let latent = self.f32(d.rank)?;
        let inp = self.f32(2 * d.rank + 2 * c)?;
        let proj = self.f32(3 * d.rank)?;
        let bias = self.f32(vp)?;
        let probs = self.f32(nd)?;
        let fr = st.frnn.view(if st.frnn_all { last * c * 4 } else { 0 }, c * 4)?;
        for j in 0..nd {
            let tj = toks.view(j * 4, 4)?;
            let hj = hid.view(j * c * 4, c * 4)?;
            // SAFETY: the buffers as silo.h names them.
            unsafe {
                self.op("draft rnn in", || (dk.rnn_inputs)(self.raw(), state.fp(), tj.ptr().cast(), d.w1.ptr().cast(), d.rank as i64, hj.fp(), fr.fp(), c as i64, inp.fp()))?;
            }
            self.matmul(&d.joint, &inp, 1, &proj)?;
            // SAFETY: as above.
            unsafe { self.op("draft rnn gates", || (dk.rnn_gates)(self.raw(), proj.fp(), d.joint_b.fp(), state.fp(), latent.fp(), d.rank as i64))? };
            self.matmul(&d.fold, &latent, 1, &bias)?;
            let next = toks.view((j + 1) * 4, 4)?;
            let bj = base.view(j * vp * 4, vp * 4)?;
            // SAFETY: as above.
            unsafe {
                let pj = probs.view(j * 4, 4)?;
                self.op("draft pick", || (dk.bias_argmax)(self.raw(), bj.fp(), bias.fp(), d.fold_b.fp(), g.vocab as i64, g.softcap, std::ptr::null_mut(),
                                                          next.ptr().cast(), pj.fp()))?
            };
        }
        let mut b = vec![0u8; (nd + 1) * 4];
        toks.read(0, &mut b)?;
        let mut pb = vec![0u8; nd * 4];
        probs.read(0, &mut pb)?;
        // the drafts up to the first the drafter is not sure enough of (NS_INK2_DRAFT_PMIN: its probability)
        let pmin = draft_pmin();
        let sure = pb.chunks_exact(4).take_while(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) >= pmin).count();
        Ok(b.chunks_exact(4).skip(1).take(sure).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}
