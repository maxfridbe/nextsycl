//! The text encoder: Qwen3-VL 32B's language model cut to its first 50 layers (ComfyUI `text_encoders/minimax.py`),
//! from the llama.cpp-quantized file the reference uses (Q4_K / Q6_K weights). The conditioning is the hidden state
//! after layer 50 - no final norm - for every prompt token: [tokens, 5120].
//!
//! One layer, as the reference computes it (in IEEE half):
//!
//! ```text
//!   h = rmsnorm(x)                          q = h Wq [64 heads x 128], k = h Wk, v = h Wv [8 heads x 128]
//!   q, k: per-head rmsnorm + rotary positions (theta 5e6, all 128 features)
//!   x += attention(q, k, v) Wo              causal; 8 query heads share each key/value head
//!   h = rmsnorm(x)
//!   x += (silu(h Wgate) * (h Wup)) Wdown
//! ```
//!
//! 16.5 GB of weights for a step that runs once per prompt, beside a denoiser that holds 18 GB of the card: the
//! layers are streamed instead - read by a thread one layer ahead into pinned host memory, sent to the card by DMA on
//! the copy queue while the layer before computes, expanded to half precision there (one matrix at a time), used,
//! and overwritten by the layer after next.
//!
//! The pinned memory is the whole file's matrices when the host has the room (`H3_TE_PIN`, default on above
//! 16.5 GB + 8 GiB free): kept by the process between clips, so after the first clip no layer is read again. Without
//! the room, two pinned layer-sized slots. Either way the result is the same bytes as reading the file each time.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::device::{Buf, Device, Pinned, Tensor};
use crate::dtype::{bf16_to_f32, f32_to_f16, DType};
use crate::gguf::{GEntry, GType, Gguf};
use crate::ops::{self, Rows};
use crate::{Ctx, Error, Result};

const EPS: f32 = 1e-6;
const THETA: f32 = 5_000_000.0;
const HEAD_DIM: usize = 128;
const MATS: [&str; 7] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];
const NORMS: [&str; 4] = ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"];

pub struct TextEncoder {
    file: Gguf,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    /// where the last `encode` read its layers from (for the job's log)
    pub source: Mutex<String>,
}

/// The layers' matrices in pinned host memory, kept by the process between encodes: for one file (path, size, mtime)
/// and one device context.
struct Pin {
    key: (PathBuf, u64, Option<SystemTime>),
    host: Pinned,
    /// which layers' bytes are in `host` (a layer is marked once its upload has landed)
    filled: Vec<bool>,
    /// held (an exclusive lock) for as long as the copy lives: one pinned copy per host, whatever the workers
    _lock: File,
}

/// One worker at a time keeps the pinned copy: the lock file (`H3_TE_PIN_LOCK`, default /tmp/h3-te-pin.lock - the
/// daemon's workers share the container's /tmp). 2026-10-05: two workers starting their encodes together each saw
/// the room free and pinned 14.4 GiB, and the host ran out of memory (the kernel's OOM killer took the user's
/// session processes). Free RAM alone is checked too late to be a guard between processes.
fn pin_lock() -> Option<File> {
    let path = std::env::var("H3_TE_PIN_LOCK").unwrap_or_else(|_| "/tmp/h3-te-pin.lock".into());
    let f = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(&path).ok()?;
    f.try_lock().ok().map(|_| f)
}
static PIN: Mutex<Option<Pin>> = Mutex::new(None);

/// The host's free memory (MemAvailable), bytes; 0 when it cannot tell.
fn mem_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

/// Reads byte ranges of a file into the given buffers with `threads` readers in 32 MiB pieces.
fn read_into(path: &Path, dsts: Vec<(u64, &mut [u8])>, threads: usize) -> Result<()> {
    const PIECE: usize = 32 << 20;
    // hand each reader disjoint pieces: split every buffer into its pieces up front
    let mut pieces: Vec<(&mut [u8], u64)> = Vec::new();
    for (off, buf) in dsts {
        let mut rest: &mut [u8] = buf;
        let mut fo = off;
        while !rest.is_empty() {
            let k = PIECE.min(rest.len());
            let (head, tail) = rest.split_at_mut(k);
            pieces.push((head, fo));
            fo += k as u64;
            rest = tail;
        }
    }
    let n = pieces.len();
    let per = n.div_ceil(threads.max(1));
    let mut failed = None;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut it = pieces.into_iter();
        loop {
            let chunk: Vec<_> = it.by_ref().take(per.max(1)).collect();
            if chunk.is_empty() {
                break;
            }
            handles.push(s.spawn(move || -> Result<()> {
                let f = File::open(path)?;
                for (dst, fo) in chunk {
                    f.read_exact_at(dst, fo).ctx("reading the text encoder")?;
                }
                Ok(())
            }));
        }
        for h in handles {
            if let Err(e) = h.join().unwrap_or_else(|_| Err(Error("a reader panicked".into()))) {
                failed.get_or_insert(e);
            }
        }
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Reads byte ranges of a file with `threads` readers.
fn read_ranges(path: &Path, ranges: &[(u64, usize)], threads: usize) -> Result<Vec<Vec<u8>>> {
    let mut bufs: Vec<Vec<u8>> = ranges.iter().map(|(_, n)| vec![0u8; *n]).collect();
    read_into(path, ranges.iter().map(|r| r.0).zip(bufs.iter_mut().map(|b| b.as_mut_slice())).collect(), threads)?;
    Ok(bufs)
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f32_to_f16(*f).to_le_bytes()).collect()
}

fn f32_tensor(dev: &Arc<Device>, v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, &[v.len()], &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

impl TextEncoder {
    pub fn open(path: &Path) -> Result<TextEncoder> {
        let file = Gguf::open(path)?;
        let layers = file.entries.keys().filter_map(|k| k.strip_prefix("model.layers.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        let q = file.get("model.layers.0.self_attn.q_proj.weight")?.shape.clone(); // [heads * 128, hidden]
        let k = file.get("model.layers.0.self_attn.k_proj.weight")?.shape.clone();
        let ffn = file.get("model.layers.0.mlp.gate_proj.weight")?.shape[0];
        Ok(TextEncoder { layers, hidden: q[1], heads: q[0] / HEAD_DIM, kv_heads: k[0] / HEAD_DIM, ffn, file, source: Mutex::new(String::new()) })
    }

    fn layer_mats(&self, i: usize) -> Result<Vec<GEntry>> {
        MATS.iter().map(|m| self.file.get(&format!("model.layers.{i}.{m}.weight")).cloned()).collect()
    }

    /// A layer's four norms, as float32 (small: read from the file every time).
    fn read_norms(&self, i: usize) -> Result<Vec<Vec<f32>>> {
        let norms: Vec<GEntry> = NORMS.iter().map(|m| self.file.get(&format!("model.layers.{i}.{m}.weight")).cloned()).collect::<Result<_>>()?;
        let ranges: Vec<(u64, usize)> = norms.iter().map(|e| (e.offset, e.bytes)).collect();
        let nb = read_ranges(&self.file.path, &ranges, 1)?;
        nb.iter()
            .zip(&norms)
            .map(|(b, e)| match e.ty {
                GType::BF16 => Ok(b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
                GType::F32 => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                t => Err(Error(format!("{}: a norm of type {t:?}", e.name))),
            })
            .collect::<Result<Vec<Vec<f32>>>>()
    }

    /// The conditioning for `tokens`: [tokens, hidden] float32. `tick(layer)` before each layer. `vision`: the vision
    /// blocks among the tokens (their rows replace the placeholders' embeddings, their deepstack features are added
    /// after the first layers, and every token takes its M-RoPE position).
    /// `embeds`: rows [n, hidden] that replace the input of the tokens from their index (`embedding:NAME`).
    pub fn encode(&self, dev: &Arc<Device>, tokens: &[u32], vision: &[crate::vision::Seen], embeds: &[(usize, Vec<f32>)], threads: usize,
                  tick: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let l = tokens.len();
        let (c, hq, hkv, ffn) = (self.hidden, self.heads, self.kv_heads, self.ffn);
        let (wq, wkv) = (hq * HEAD_DIM, hkv * HEAD_DIM);
        // embeddings: the tokens' rows, read straight from the file
        let emb = self.file.get("model.embed_tokens.weight")?;
        if emb.ty != GType::BF16 || emb.shape[1] != c {
            return Err(Error("the token embedding is expected as bfloat16 [vocab, hidden]".into()));
        }
        let f = File::open(&self.file.path)?;
        let mut x0 = Vec::with_capacity(l * c);
        let mut row = vec![0u8; c * 2];
        for t in tokens {
            if *t as usize >= emb.shape[0] {
                return Err(Error(format!("token id {t} is outside the vocabulary")));
            }
            f.read_exact_at(&mut row, emb.offset + (*t as u64) * (c as u64 * 2)).ctx("reading the token embedding")?;
            x0.extend(row.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))));
        }
        for v in vision {
            let n = v.grid.0 * v.grid.1;
            if v.tokens.len() != n * c || v.at + n > l {
                return Err(Error(format!("a vision block of {} values for {n} tokens of {c} at {} (of {l})", v.tokens.len(), v.at)));
            }
            x0[v.at * c..(v.at + n) * c].copy_from_slice(&v.tokens);
        }
        for (at, rows) in embeds {
            if rows.len() % c != 0 || at * c + rows.len() > l * c {
                return Err(Error(format!("an embedding of {} values at token {at} (of {l}, {c} wide)", rows.len())));
            }
            x0[at * c..at * c + rows.len()].copy_from_slice(rows);
        }
        // deepstack: per depth, the blocks' features at their rows (zero elsewhere), added after that layer
        let depths = vision.iter().map(|v| v.deep.len()).max().unwrap_or(0);
        let deep: Vec<Tensor> = (0..depths)
            .map(|k| {
                let mut d = vec![0f32; l * c];
                for v in vision.iter().filter(|v| v.deep.len() > k) {
                    d[v.at * c..v.at * c + v.deep[k].len()].copy_from_slice(&v.deep[k]);
                }
                Tensor::from_bytes(dev, DType::F16, &[l, c], &f16_bytes(&d))
            })
            .collect::<Result<_>>()?;
        let dt = DType::F16;
        let x = Tensor::from_bytes(dev, dt, &[l, c], &f16_bytes(&x0))?;
        // rotary table: position p, pair j -> angle p / theta^(2j / 128); with vision blocks, Qwen3-VL's M-RoPE
        let cs: Vec<f32> = if vision.is_empty() {
            let inv: Vec<f32> = (0..HEAD_DIM / 2).map(|j| 1.0 / THETA.powf((2 * j) as f32 / HEAD_DIM as f32)).collect();
            (0..l).flat_map(|p| inv.iter().flat_map(move |f| [(p as f32 * f).cos(), (p as f32 * f).sin()])).collect()
        } else {
            crate::vision::rope_table(&crate::vision::positions(l, vision), HEAD_DIM, THETA)
        };
        let cs = Tensor::from_bytes(dev, DType::F32, &[l, HEAD_DIM / 2, 2], &cs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;

        let h = Tensor::new(dev, dt, &[l, c])?;
        let q = Tensor::new(dev, dt, &[l, wq])?;
        let k = Tensor::new(dev, dt, &[l, wkv])?;
        let v = Tensor::new(dev, dt, &[l, wkv])?;
        let att = Tensor::new(dev, dt, &[l, wq])?;
        let o = Tensor::new(dev, dt, &[l, c])?;
        let gu = Tensor::new(dev, dt, &[l, 2 * ffn])?;
        let act = Tensor::new(dev, dt, &[l, ffn])?;
        // one layer's largest matrix (gate and up together), expanded to half precision
        let w = Tensor::new(dev, dt, &[2 * ffn * c])?;

        let d = dev.clone();
        let lin = |input: &Tensor, n: usize, kk: usize, out: &Tensor| -> Result<()> {
            // SAFETY: device buffers of this device; input [l, kk], w holds at least [n, kk], out [l, n].
            let rc = unsafe { (d.api.linear)(d.ctx, input.buf.ptr(), dt.kernel_code()?, l as i64, kk as i64, w.buf.ptr(), n as i64, std::ptr::null(), out.buf.ptr(), dt.kernel_code()?) };
            d.check(rc)
        };
        let expand = |blob: &Buf, off: usize, e: &GEntry, at: usize| -> Result<()> {
            let code = e.ty.quant_code().ok_or_else(|| Error(format!("{}: type {:?} is not a k-quant", e.name, e.ty)))?;
            // SAFETY: blob holds e.bytes of k-quant blocks at off; w has room for `at + elements` half values.
            let rc = unsafe {
                (d.api.dequant)(d.ctx, blob.ptr().cast::<u8>().add(off).cast(), code, e.elements() as i64, w.buf.ptr().cast::<u8>().add(at * 2).cast(), dt.kernel_code()?)
            };
            d.check(rc)
        };

        // each layer's seven matrices, back to back: where each starts in the layer, the layer's bytes, where the
        // layer starts in the whole file's pinned copy
        let mats: Vec<Vec<GEntry>> = (0..self.layers).map(|i| self.layer_mats(i)).collect::<Result<_>>()?;
        let offs: Vec<Vec<usize>> = mats.iter().map(|m| m.iter().scan(0, |o, e| { let a = *o; *o += e.bytes; Some(a) }).collect()).collect();
        let lbytes: Vec<usize> = mats.iter().map(|m| m.iter().map(|e| e.bytes).sum()).collect();
        let lstart: Vec<usize> = lbytes.iter().scan(0, |o, b| { let a = *o; *o += b; Some(a) }).collect();
        let (total, max_l) = (lbytes.iter().sum::<usize>(), lbytes.iter().copied().max().unwrap_or(0));

        // the pinned source: the whole file's matrices, kept between encodes, or two layer slots
        let meta = std::fs::metadata(&self.file.path)?;
        let key = (self.file.path.clone(), meta.len(), meta.modified().ok());
        let mut pin = PIN.lock().unwrap_or_else(|e| e.into_inner());
        let kept = pin.as_ref().is_some_and(|p| p.key == key && p.host.device().same(dev) && p.host.len() >= total);
        if !kept {
            *pin = None; // another file or context: its memory goes back first
            let want = std::env::var("H3_TE_PIN").map_or(true, |v| v != "0");
            // the lock first (another worker may be allocating its copy right now), then the room: 12 GiB left over
            let lock = if want { pin_lock() } else { None };
            if let Some(lock) = lock {
                if mem_available() >= total as u64 + (12u64 << 30) {
                    match dev.alloc_pinned(total) {
                        Ok(host) => *pin = Some(Pin { key, host, filled: vec![false; self.layers], _lock: lock }),
                        Err(e) => eprintln!("te: no pinned copy of the text encoder ({e}); two pinned layer slots instead"),
                    }
                }
            }
        }
        let staging = if pin.is_none() { Some(dev.alloc_pinned(2 * max_l)?) } else { None };
        let filled: Vec<bool> = pin.as_ref().map_or_else(|| vec![false; self.layers], |p| p.filled.clone());
        let cached = filled.iter().filter(|f| **f).count();
        *self.source.lock().unwrap_or_else(|e| e.into_inner()) = match (&staging, cached) {
            (Some(_), _) => "the file, through two pinned layer slots (another worker keeps the pinned copy, or no room)".into(),
            (None, n) if n == self.layers => "pinned host memory (kept from an earlier clip)".into(),
            (None, 0) => format!("the file, into pinned host memory kept for the next clips ({:.1} GiB)", total as f64 / 1073741824.0),
            (None, n) => format!("pinned host memory ({n} layers) and the file"),
        };
        let use_staging = staging.is_some();
        let host: &Pinned = match (&staging, pin.as_ref()) {
            (Some(st), _) => st,
            (None, Some(p)) => &p.host,
            (None, None) => unreachable!("staging is made when there is no pinned copy"),
        };
        let region = |i: usize| if use_staging { ((i % 2) * max_l, lbytes[i]) } else { (lstart[i], lbytes[i]) };
        // the layers whose upload landed (their pinned bytes are complete): marked kept after the encode
        let landed = std::cell::Cell::new(0usize);
        // two device buffers: layer i computes from one while layer i + 1 lands in the other
        let blobs = [dev.alloc(max_l.max(1))?, dev.alloc(max_l.max(1))?];

        // the reader thread, one layer ahead: the layer's matrices into its pinned region (unless kept), its norms
        let (tx, rx) = mpsc::sync_channel::<Result<Vec<Vec<f32>>>>(1);
        // a staging slot comes back once its upload has landed (layer i's slot serves layer i + 2)
        let (free_tx, free_rx) = mpsc::sync_channel::<()>(self.layers.max(1));
        let out = std::thread::scope(|s| -> Result<Vec<f32>> {
            // owned here, so a return (an error, a cancel) drops them before the scope waits for the reader, which
            // then stops instead of blocking on a full channel
            let (rx, free_tx) = (rx, free_tx);
            let (mats, filled) = (&mats, &filled);
            s.spawn(move || {
                for i in 0..self.layers {
                    if use_staging && i >= 2 && free_rx.recv().is_err() {
                        return; // the consumer stopped
                    }
                    let r = (|| -> Result<Vec<Vec<f32>>> {
                        if !filled[i] {
                            let (off, n) = region(i);
                            // SAFETY: this region is written by this thread alone: a staging slot comes back only after
                            // its upload landed, and a kept layer's region is written once, before its first upload.
                            let mut rest: &mut [u8] = unsafe { host.region(off, n) };
                            let mut dsts = Vec::new();
                            for e in &mats[i] {
                                let (head, tail) = rest.split_at_mut(e.bytes);
                                dsts.push((e.offset, head));
                                rest = tail;
                            }
                            read_into(&self.file.path, dsts, threads)?;
                        }
                        self.read_norms(i)
                    })();
                    if tx.send(r).is_err() {
                        return;
                    }
                }
            });
            let upload = |i: usize| -> Result<()> {
                let (off, n) = region(i);
                blobs[i % 2].upload(0, host, off, n)
            };
            let mut norms = rx.recv().map_err(|_| Error("the text encoder's reader stopped".into()))??;
            upload(0)?;
            for i in 0..self.layers {
                tick(i)?;
                dev.upload_wait()?; // layer i's matrices are on the card
                landed.set(i + 1);
                if use_staging {
                    let _ = free_tx.send(()); // its slot may take layer i + 2
                }
                let hl = std::mem::take(&mut norms);
                if i + 1 < self.layers {
                    norms = rx.recv().map_err(|_| Error("the text encoder's reader stopped".into()))??;
                    upload(i + 1)?; // lands while this layer computes
                }
                let blob = &blobs[i % 2];
                let [ln1, ln2, qn, kn] = [0, 1, 2, 3].map(|j| f32_tensor(dev, &hl[j]));
                let (ln1, ln2, qn, kn) = (ln1?, ln2?, qn?, kn?);
                let e = |j: usize| (&mats[i][j], offs[i][j]);
                // attention half
                ops::rms_norm_mod(&x, &ln1, EPS, None, &h)?;
                expand(blob, e(0).1, e(0).0, 0)?;
                lin(&h, wq, c, &q)?;
                expand(blob, e(1).1, e(1).0, 0)?;
                lin(&h, wkv, c, &k)?;
                expand(blob, e(2).1, e(2).0, 0)?;
                lin(&h, wkv, c, &v)?;
                ops::rms_rope(Rows { t: &q, offset: 0, stride: wq, tokens: l, heads: hq, dim: HEAD_DIM }, &qn, EPS, &cs, HEAD_DIM)?;
                ops::rms_rope(Rows { t: &k, offset: 0, stride: wkv, tokens: l, heads: hkv, dim: HEAD_DIM }, &kn, EPS, &cs, HEAD_DIM)?;
                // SAFETY: q [l, wq], k and v [l, wkv], att [l, wq], all half on this device.
                let rc = unsafe {
                    (d.api.attention_causal)(d.ctx, q.buf.ptr(), k.buf.ptr(), v.buf.ptr(), dt.kernel_code()?, l as i64, hq as i64, hkv as i64, HEAD_DIM as i64,
                                             wq as i64, wkv as i64, att.buf.ptr())
                };
                d.check(rc)?;
                expand(blob, e(3).1, e(3).0, 0)?;
                lin(&att, c, wq, &o)?;
                ops::add(&x, &o)?;
                // MLP half: gate and up expanded one after the other, multiplied at once ([gate | up] per row)
                ops::rms_norm_mod(&x, &ln2, EPS, None, &h)?;
                expand(blob, e(4).1, e(4).0, 0)?;
                expand(blob, e(5).1, e(5).0, ffn * c)?;
                lin(&h, 2 * ffn, c, &gu)?;
                ops::swiglu(&gu, &act)?;
                expand(blob, e(6).1, e(6).0, 0)?;
                lin(&act, c, ffn, &o)?;
                ops::add(&x, &o)?;
                if let Some(d) = deep.get(i) {
                    ops::add(&x, d)?;
                }
                dev.wait()?; // this layer's buffer takes layer i + 2
            }
            drop(rx);
            x.to_f32()
        });
        if let Some(p) = pin.as_mut() {
            for f in p.filled.iter_mut().take(landed.get()) {
                *f = true;
            }
        }
        out
    }
}
