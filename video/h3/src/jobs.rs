//! What the engine can be asked to do, written once for both ways of asking: the one-shot commands (`h3d bench-blocks
//! ...`, which load, run and exit) and the daemon (`h3d daemon`, which keeps the model loaded between jobs).
//!
//! A job reports through a log callback and looks at a cancel flag between blocks - never inside one: a GPU process
//! stopped in the middle of a kernel can leave the xe driver stuck.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::device::{Device, Tensor};
use crate::dtype::{f32_to_bf16, DType};
use crate::rng::Rng;
use crate::safetensors::Checkpoint;
use crate::denoiser::{self, Conditions, Denoiser, KeyframeIn, Outer, RefIn, Schedule, Shape, TextRefiner};
use crate::vision::{Frames, Piece};
use crate::{dit, reference, Error, Result};
use serde_json::{json, Value};

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// The loaded engine: the GPU, the checkpoint, the denoiser's blocks and the small parts around them on the GPU.
pub struct Engine {
    pub dev: Arc<Device>,
    pub ck: Checkpoint,
    pub model: dit::Blocks,
    pub outer: Outer,
    pub threads: usize,
}

impl Engine {
    /// Opens the GPU and loads the first `count` blocks (all of them when `None`).
    pub fn load(path: &Path, count: Option<usize>, threads: usize, log: &mut dyn FnMut(String)) -> Result<Engine> {
        Engine::load_on(Device::open()?, path, count, threads, log)
    }

    /// The same, on a GPU already opened.
    pub fn load_on(dev: Arc<Device>, path: &Path, count: Option<usize>, threads: usize, log: &mut dyn FnMut(String)) -> Result<Engine> {
        let ck = Checkpoint::open(path)?;
        log(format!("device : {}", dev.name()));
        let model = dit::Blocks::load(&dev, &ck, count, threads)?;
        log(format!("blocks : {} loaded, {:.2} GiB in {:.1} s", model.blocks.len(), gib(model.load_bytes), model.load_seconds));
        let outer = Outer::load(&dev, &ck)?;
        denoiser::check_fits(&model.cfg, &outer)?;
        Ok(Engine { dev, ck, model, outer, threads })
    }

    fn inv_freq(&self) -> Result<Vec<f32>> {
        crate::dtype::bytes_to_f32(&self.ck.read("rope.inv_freq")?, self.ck.get("rope.inv_freq")?.dtype)
    }
}

/// How a running job talks back.
pub struct Ctl<'a> {
    pub log: &'a mut dyn FnMut(String),
    pub cancel: &'a AtomicBool,
    /// how far it is: (done, total), e.g. blocks
    pub progress: Option<&'a mut dyn FnMut(usize, usize)>,
}

impl Ctl<'_> {
    fn step(&mut self, done: usize, total: usize) {
        if let Some(p) = self.progress.as_mut() {
            p(done, total)
        }
    }

    fn say(&mut self, s: String) {
        (self.log)(s)
    }
    /// Between blocks: stop here if the job was cancelled. The GPU queue is asynchronous - without a wait, a job
    /// would have queued every block (48 s of work at 47k tokens) long before it looked at the flag - so a block
    /// boundary means: what was queued has run.
    fn check_at(&self, dev: &Device) -> Result<()> {
        dev.wait()?;
        self.check()
    }

    fn check(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(Error("cancelled".into()))
        } else {
            Ok(())
        }
    }
}

/// Random activations as bfloat16: the bytes for the device and the values they stand for.
pub fn activations(rng: &mut Rng, n: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(n * 2);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        let b = f32_to_bf16(rng.normal() * 0.7);
        bytes.extend_from_slice(&b.to_le_bytes());
        vals.push(crate::dtype::bf16_to_f32(b));
    }
    (bytes, vals)
}

/// Columns [from, from + width) of every `stride`-wide row.
fn columns(v: &[f32], stride: usize, from: usize, width: usize) -> Vec<f32> {
    v.chunks_exact(stride).flat_map(|r| r[from..from + width].iter().copied()).collect()
}

/// The denoiser's blocks against a dump of the reference pipeline (reference/h3x.py, H3X_DUMP_BLOCK): the host-side
/// tables, every stage of block 0, then the stream after later blocks. Fails when the result drifts.
pub fn check_block(e: &Engine, dump_path: &Path, ctl: &mut Ctl) -> Result<Value> {
    let dump = Checkpoint::open(dump_path)?;
    let mut cfg = e.model.cfg;
    for (key, eps) in [("norm_eps", &mut cfg.norm_eps), ("qk_eps", &mut cfg.qk_eps)] {
        if let Some(v) = dump.metadata.get(key) {
            *eps = v.parse().map_err(|_| Error(format!("the dump's {key} is not a number: {v}")))?;
        }
    }
    if cfg.norm_eps != e.model.cfg.norm_eps || cfg.qk_eps != e.model.cfg.qk_eps {
        return Err(Error("the dump was made with other norm constants than the loaded model".into()));
    }
    let f32s = |name: &str| -> Result<Vec<f32>> { crate::dtype::bytes_to_f32(&dump.read(name)?, dump.get(name)?.dtype) };
    fn report(ctl: &mut Ctl, what: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
        let (rel, cos) = reference::compare(got, want);
        ctl.say(format!("  {what:34} rel err {rel:.2e}  cosine {cos:.6}"));
        (rel, cos)
    }

    let rows: Vec<i32> = dump.read("mod_rows")?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let positions: Vec<f64> = dump.read("position_ids")?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let inv_freq = e.inv_freq()?;
    let t_emb = f32s("t_emb")?;
    let tokens = rows.len();
    ctl.say(format!("dump   : {tokens} tokens, {} timestep rows", t_emb.len() / cfg.t_dim));

    ctl.say("host side, against the reference:".into());
    let tables = e.model.blocks[0].tables(&cfg, &t_emb);
    let mut worst = 0f64;
    for (t, name) in tables.iter().zip(["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]) {
        worst = worst.max(reference::compare(t, &f32s(&format!("b0.{name}"))?).0);
    }
    ctl.say(format!("  {:34} rel err {worst:.2e}  (worst of six)", "block 0's tables"));
    // the dump's rotation matrices are [S, pairs, 2, 2] = (cos, -sin, sin, cos)
    let cs = dit::rotations(&positions, &inv_freq);
    let want_cs: Vec<f32> = f32s("rope_table")?.chunks_exact(4).flat_map(|m| [m[0], m[2]]).collect();
    report(ctl, "position rotations (cos, sin)", &cs, &want_cs);

    let dev = &e.dev;
    let step = dit::Step::new(dev, &e.model, &rows, &positions, &inv_freq, &t_emb)?;
    let scratch = dit::Scratch::new(dev, &cfg, tokens, DType::BF16)?;
    let x_in = dump.read("x_in")?;
    let x = Tensor::from_bytes(dev, DType::BF16, &[tokens, cfg.hidden], &x_in)?;

    ctl.say("block 0, stage by stage (each stage runs on this engine's own previous stage):".into());
    let w = cfg.heads * cfg.head_dim;
    let mut worst_cos = 1f64;
    {
        let mut tap = |name: &str, t: &Tensor| -> Result<()> {
            let got = t.to_f32()?;
            let c = match name {
                "qkv_rotated" => {
                    let (_, cq) = report(ctl, "q after norm + rotation", &columns(&got, 3 * w, 0, w), &f32s("b0.q_rope")?);
                    let (_, ck) = report(ctl, "k after norm + rotation", &columns(&got, 3 * w, w, w), &f32s("b0.k_rope")?);
                    cq.min(ck)
                }
                _ => report(ctl, name, &got, &f32s(&format!("b0.{name}"))?).1,
            };
            worst_cos = worst_cos.min(c);
            Ok(())
        };
        e.model.block(0, &x, &step, &scratch, None, Some(&mut tap))?;
    }

    ctl.say("the stream after later blocks:".into());
    let last = e.model.blocks.len() - 1;
    let mut after = Vec::new();
    for i in 1..=last {
        ctl.check_at(dev)?;
        ctl.step(i, 2 * (last + 1));
        e.model.block(i, &x, &step, &scratch, None, None)?;
        let key = if dump.metadata.get("last_block").is_some_and(|l| *l == i.to_string()) { "out.last".to_string() } else { format!("out.{i}") };
        if dump.entries.contains_key(&key) {
            let c = report(ctl, &format!("after block {i}"), &x.to_f32()?, &f32s(&key)?).1;
            after.push(json!({"block": i, "cosine": c}));
            worst_cos = worst_cos.min(c);
        }
    }

    // speed: the whole stack again, one wait at the end
    x.buf.write(0, &x_in)?;
    dev.wait()?;
    let t0 = Instant::now();
    for i in 0..=last {
        ctl.check_at(dev)?;
        ctl.step(last + 1 + i, 2 * (last + 1));
        e.model.block(i, &x, &step, &scratch, None, None)?;
    }
    dev.wait()?;
    let s = t0.elapsed().as_secs_f64();
    ctl.say(format!("speed  : {} blocks on {tokens} tokens in {:.2} s ({:.1} ms per block)", last + 1, s, s * 1e3 / (last + 1) as f64));
    if worst_cos > 0.99 {
        Ok(json!({"tokens": tokens, "worst_cosine": worst_cos, "after": after, "seconds": s}))
    } else {
        Err(Error(format!("the engine's result drifts from the reference (worst cosine {worst_cos:.4})")))
    }
}

/// The block stack on made-up tokens at a chosen sequence length: what one denoiser step costs, and where.
/// `blocks`: run only the first that many (all the loaded ones when `None`).
pub fn bench_blocks(e: &Engine, tokens: usize, blocks: Option<usize>, ctl: &mut Ctl) -> Result<Value> {
    let cfg = e.model.cfg;
    let dev = &e.dev;
    let mut rng = Rng::new(0);
    let rows: Vec<i32> = (0..tokens).map(|i| if i < tokens / 20 { 1 } else { 0 }).collect();
    let positions: Vec<f64> = (0..tokens * 3).map(|_| (rng.uniform() * 64.0) as f64).collect();
    let t_emb: Vec<f32> = (0..cfg.t_dim).map(|_| rng.normal()).collect();
    let step = dit::Step::new(dev, &e.model, &rows, &positions, &e.inv_freq()?, &t_emb)?;
    let scratch = dit::Scratch::new(dev, &cfg, tokens, DType::BF16)?;
    // made-up input: a million random values repeated (drawing all 250 million at 47k tokens took the slow host
    // half a minute, during which a cancel could not get through)
    let (pattern, _) = activations(&mut rng, 1 << 20);
    let xb: Vec<u8> = pattern.iter().copied().cycle().take(tokens * cfg.hidden * 2).collect();
    ctl.check()?;
    let x = Tensor::from_bytes(dev, DType::BF16, &[tokens, cfg.hidden], &xb)?;
    let n = blocks.unwrap_or(e.model.blocks.len()).min(e.model.blocks.len());

    // warm-up: the first call of each kernel shape compiles it
    e.model.block(0, &x, &step, &scratch, None, None)?;
    dev.wait()?;
    x.buf.write(0, &xb)?;
    let t0 = Instant::now();
    for i in 0..n {
        ctl.check_at(dev)?;
        ctl.step(i, 2 * n);
        e.model.block(i, &x, &step, &scratch, None, None)?;
    }
    dev.wait()?;
    let total = t0.elapsed().as_secs_f64();
    ctl.say(format!(
        "speed  : {n} blocks on {tokens} tokens in {total:.2} s ({:.1} ms per block), {:.1} of {:.1} GiB in use",
        total * 1e3 / n as f64,
        gib(dev.mem_used()),
        gib(dev.mem_cap())
    ));

    // where it goes: the same again, waiting for the device after every stage
    x.buf.write(0, &xb)?;
    let mut stages: Vec<(String, f64)> = Vec::new();
    let mut mark = Instant::now();
    {
        let mut tap = |name: &str, t: &Tensor| -> Result<()> {
            t.buf.device().wait()?;
            let dt = mark.elapsed().as_secs_f64();
            match stages.iter_mut().find(|(n, _)| n == name) {
                Some(s) => s.1 += dt,
                None => stages.push((name.to_string(), dt)),
            }
            mark = Instant::now();
            Ok(())
        };
        for i in 0..n {
            ctl.check_at(dev)?;
            if let Some(p) = ctl.progress.as_mut() {
                p(n + i, 2 * n);
            }
            e.model.block(i, &x, &step, &scratch, None, Some(&mut tap))?;
        }
    }
    let sum: f64 = stages.iter().map(|s| s.1).sum();
    let label = |n: &str| match n {
        "h1" | "h2" => "norm + scale/shift",
        "qkv" => "linear: q, k, v",
        "qkv_rotated" => "per-head norm + rotation (q, k)",
        "att" => "attention",
        "attn_out" => "linear: attention out",
        "x1" | "x2" => "gated add",
        "fc1" => "linear: MLP in",
        "mlp" => "gated activation + linear: MLP out",
        _ => "other",
    };
    let mut merged: Vec<(&str, f64)> = Vec::new();
    for (n, s) in &stages {
        match merged.iter_mut().find(|(l, _)| *l == label(n)) {
            Some(m) => m.1 += s,
            None => merged.push((label(n), *s)),
        }
    }
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    ctl.say("stages (a wait after each, so the sum is a little above the run without):".into());
    let mut out = serde_json::Map::new();
    for (l, s) in merged {
        ctl.say(format!("  {l:36} {:7.1} ms per block  {:4.1}%", s * 1e3 / n as f64, s / sum * 100.0));
        out.insert(l.to_string(), json!(s * 1e3 / n as f64));
    }
    Ok(json!({"tokens": tokens, "blocks": n, "seconds": total, "ms_per_block": total * 1e3 / n as f64,
              "gib_in_use": gib(dev.mem_used()), "stages_ms_per_block": out}))
}

/// A whole sampling run against a run dump of the reference pipeline (reference/h3x.py, H3X_DUMP_RUN): the same
/// text conditioning, starting noise and schedule; every step's denoised estimate compared with the reference's,
/// then the finished latents. `out`: where to write the latents (for the reference's decoders).
pub fn denoise(e: &Engine, dump_path: &Path, out: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    let dump = Checkpoint::open(dump_path)?;
    let f32s = |name: &str| -> Result<Vec<f32>> { crate::dtype::bytes_to_f32(&dump.read(name)?, dump.get(name)?.dtype) };
    let meta = |k: &str, d: f32| -> f32 { dump.metadata.get(k).and_then(|v| v.parse().ok()).unwrap_or(d) };
    let schedule = Schedule { shift_video: meta("shift", 12.0), shift_audio: meta("audio_shift", 3.0), audio_scale: meta("audio_scale", 4.0) };
    if e.model.blocks.len() != e.model.cfg.blocks {
        return Err(Error("denoise needs every block loaded".into()));
    }
    let vs = &dump.get("noise.video")?.shape; // [1, C, T, H, W]
    let as_ = &dump.get("noise.audio")?.shape; // [1, C, 2, T]
    let shape = Shape { t: vs[2], h: vs[3], w: vs[4], audio_t: as_[3] };
    let cs = &dump.get("context")?.shape; // [1, L, text_dim]
    let (l, text_dim) = (cs[1], cs[2]);
    let tags: Option<Vec<i32>> = match dump.entries.contains_key("token_tags") {
        true => Some(dump.read("token_tags")?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        false => None,
    };
    let sigmas = f32s("sigmas")?;
    let own = denoiser::sigmas(sigmas.len() - 1, schedule.shift_video);
    let worst_sigma = own.iter().zip(&sigmas).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    ctl.say(format!("dump   : {l} text tokens, video latent {}x{}x{}, audio {} frames, {} steps (schedule differs by {worst_sigma:.1e})",
                    shape.t, shape.h, shape.w, shape.audio_t, sigmas.len() - 1));

    // the text: projection + refiner, loaded for this and dropped
    let t0 = Instant::now();
    let text = {
        let refiner = TextRefiner::load(&e.dev, &e.ck, e.threads)?;
        let ctx = Tensor::from_bytes(&e.dev, DType::BF16, &[l, text_dim], &denoiser::bf16_bytes(&f32s("context")?))?;
        let t = refiner.run(&e.model.cfg, &ctx, None)?;
        e.dev.wait()?;
        t
    };
    ctl.say(format!("text   : refined in {:.1} s (refiner loaded and freed)", t0.elapsed().as_secs_f64()));
    ctl.check()?;

    // the starting noise: drawn here as PyTorch draws it (same seed), checked against the dump's
    let seed: u64 = dump.metadata.get("seed").and_then(|v| v.parse().ok()).unwrap_or(0);
    let (want_v, want_a) = (f32s("noise.video")?, f32s("noise.audio")?);
    let (noise_v, noise_a) = crate::noise::clip_noise(seed, want_v.len(), want_a.len());
    let worst_noise = noise_v.iter().zip(&want_v).chain(noise_a.iter().zip(&want_a)).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    ctl.say(format!("noise  : seed {seed}, drawn by the engine; differs from the reference's by {worst_noise:.1e} at most"));
    if worst_noise > 1e-4 {
        return Err(Error("the engine's starting noise is not the reference's".into()));
    }

    // keyframes, as the reference's denoiser received them
    let mut cond = Conditions { seed, ..Conditions::default() };
    if let Some(v) = dump.metadata.get("visual_cond_noise_aug").and_then(|v| v.parse().ok()) {
        cond.visual_aug = v;
    }
    if let Some(v) = dump.metadata.get("audio_cond_noise_aug").and_then(|v| v.parse().ok()) {
        cond.audio_aug = v;
    }
    if let Some(list) = dump.metadata.get("keyframe_list") {
        let list: Value = serde_json::from_str(list).map_err(|e| Error(format!("keyframe_list: {e}")))?;
        for (i, k) in list.as_array().ok_or("keyframe_list is not a list")?.iter().enumerate() {
            let index = k["index"].as_u64().ok_or("a keyframe without an index")? as usize;
            let video = match k.get("video").is_some() {
                true => Some((f32s(&format!("kf.{i}.video"))?, dump.get(&format!("kf.{i}.video"))?.shape[2])),
                false => None,
            };
            let audio = match k.get("audio").is_some() {
                true => Some((f32s(&format!("kf.{i}.audio"))?, dump.get(&format!("kf.{i}.audio"))?.shape[3])),
                false => None,
            };
            ctl.say(format!("keyframe {i}: frame {index}{}{}", video.as_ref().map_or(String::new(), |v| format!(", video {} latent frames", v.1)),
                            audio.as_ref().map_or(String::new(), |a| format!(", audio {} latent frames", a.1))));
            cond.keyframes.push(KeyframeIn { frame_index: index, video, audio });
        }
    }
    if let Some(list) = dump.metadata.get("ref_list") {
        let list: Value = serde_json::from_str(list).map_err(|e| Error(format!("ref_list: {e}")))?;
        for (i, r) in list.as_array().ok_or("ref_list is not a list")?.iter().enumerate() {
            let audio = |i: usize| -> Result<Option<(Vec<f32>, usize)>> {
                let k = format!("ref.{i}.audio");
                if !dump.entries.contains_key(&k) {
                    return Ok(None);
                }
                Ok(Some((f32s(&k)?, dump.get(&k)?.shape[3])))
            };
            let video = |i: usize| -> Result<(Vec<f32>, [usize; 3])> {
                let k = format!("ref.{i}.video");
                let sh = &dump.get(&k)?.shape; // [1, 24, t, h, w]
                Ok((f32s(&k)?, [sh[2], sh[3], sh[4]]))
            };
            let r = match r["kind"].as_str() {
                Some("audio") => {
                    let (latent, t) = audio(i)?.ok_or("an audio reference without its latent")?;
                    RefIn::Audio { latent, t }
                }
                Some("image") => {
                    let (latent, [_, h, w]) = video(i)?;
                    RefIn::Image { latent, h, w }
                }
                Some("video" | "video_audio") => {
                    let (latent, [t, h, w]) = video(i)?;
                    RefIn::Video { latent, t, h, w, audio: audio(i)? }
                }
                other => return Err(Error(format!("reference {i}: a {other:?} reference?"))),
            };
            ctl.say(format!("ref {i}  : {}", match &r {
                RefIn::Audio { t, .. } => format!("audio, {t} latent frames"),
                RefIn::Image { h, w, .. } => format!("picture, {w}x{h} latent"),
                RefIn::Video { t, h, w, audio, .. } => format!("clip, {t} x {w}x{h} latent{}", audio.as_ref().map_or(String::new(), |a| format!(" + {} audio frames", a.1))),
            }));
            cond.refs.push(r);
        }
    }
    let mut d = Denoiser::new(&e.model, &e.outer, text, tags, shape, schedule, &cond)?;
    let steps = sigmas.len() - 1;
    let nblocks = e.model.blocks.len();
    ctl.say(format!("tokens : {}", d.tokens()));
    let mut per_step = Vec::new();
    let mut worst = 1f64;
    let mut t_step = Instant::now();
    let (v, a) = {
        let dev = e.dev.clone();
        let cancel = ctl.cancel;
        let mut progress = ctl.progress.take();
        let mut lines: Vec<String> = Vec::new();
        let r = denoiser::sample(
            &mut d,
            &noise_v,
            &noise_a,
            &sigmas,
            &mut |i, dv, da| {
                let secs = t_step.elapsed().as_secs_f64();
                t_step = Instant::now();
                let key = format!("x0.{:02}", i + 1);
                let (mut cv, mut ca) = (f64::NAN, f64::NAN);
                if dump.entries.contains_key(&format!("{key}.video")) {
                    cv = reference::compare(dv, &f32s(&format!("{key}.video"))?).1;
                    ca = reference::compare(da, &f32s(&format!("{key}.audio"))?).1;
                    worst = worst.min(cv).min(ca);
                }
                lines.push(format!("  step {}/{steps}  sigma {:.4}  {secs:6.2} s   cosine to the reference: video {cv:.5}  audio {ca:.5}", i + 1, sigmas[i]));
                per_step.push(json!({"step": i + 1, "seconds": secs, "cosine_video": cv, "cosine_audio": ca}));
                Ok(())
            },
            &mut |i, b| {
                if b == 0 || b == nblocks / 2 {
                    dev.wait()?;
                    if cancel.load(Ordering::Relaxed) {
                        return Err(Error("cancelled".into()));
                    }
                }
                if let Some(p) = progress.as_mut() {
                    p(i * nblocks + b, steps * nblocks);
                }
                Ok(())
            },
        );
        ctl.progress = progress;
        for s in lines {
            ctl.say(s);
        }
        r?
    };
    let cv = reference::compare(&v, &f32s("samples.video")?);
    let ca = reference::compare(&a, &f32s("samples.audio")?);
    ctl.say(format!("latents: video rel err {:.2e} cosine {:.5}; audio rel err {:.2e} cosine {:.5}", cv.0, cv.1, ca.0, ca.1));
    if let Some(p) = out {
        let mut t = BTreeMap::new();
        t.insert("samples.video".to_string(), (dump.get("samples.video")?.shape.clone(), v));
        t.insert("samples.audio".to_string(), (dump.get("samples.audio")?.shape.clone(), a));
        crate::safetensors::write_f32(p, &t, &dump.metadata)?;
        ctl.say(format!("written: {}", p.display()));
    }
    let secs = t_all.elapsed().as_secs_f64();
    ctl.say(format!("total  : {secs:.1} s"));
    Ok(json!({"tokens": d.tokens(), "steps": per_step, "cosine_video": cv.1, "cosine_audio": ca.1, "worst_step_cosine": worst, "seconds": secs}))
}

/// The decoders' checkpoints: video, and audio (optional: a silent clip without it).
pub struct Vaes<'a> {
    pub video: &'a Path,
    pub audio: Option<&'a Path>,
    /// the latent upscaler and its factor (the video latents are upscaled before the video decoder)
    pub upscale: Option<(&'a Path, f32)>,
    /// instead: the pixel upscaler (an ESRGAN-type network, esrgan.rs) and the factor - the clip is decoded at its
    /// sampled size and the frames enlarged to the size the latent upscaler would give
    pub pixel: Option<(&'a Path, f32)>,
}

/// Latents -> a clip: the video decoder (and the audio decoder) on the latents in `latents` (a `.safetensors` with
/// `samples.video` / `samples.audio`, as `denoise` writes it, or `latents.*`), written to `out` (.mp4). `check`: a
/// decode dump of the reference (h3x.py H3X_DUMP_DECODE) to compare the frames and the sound with.
pub fn decode(dev: &Arc<Device>, threads: usize, latents: &Path, vaes: &Vaes, out: Option<&Path>, check: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    let lat = Checkpoint::open(latents)?;
    let key = ["samples.video", "latents.video"].into_iter().find(|k| lat.entries.contains_key(*k)).ok_or("the latents file has no samples.video")?;
    let shape = lat.get(key)?.shape.clone(); // [1, 24, T, H, W]
    let video = crate::dtype::bytes_to_f32(&lat.read(key)?, lat.get(key)?.dtype)?;
    let audio = match ["samples.audio", "latents.audio"].into_iter().find(|k| lat.entries.contains_key(*k)) {
        Some(k) => Some((crate::dtype::bytes_to_f32(&lat.read(k)?, lat.get(k)?.dtype)?, lat.get(k)?.shape[3])), // [1, 32, 2, T]
        None => None,
    };
    decode_latents(dev, threads, Latents { video, t: shape[2], h: shape[3], w: shape[4], audio }, vaes, out, check, ctl)
}

/// A clip's finished latents: video [24, t, h, w], and audio [32, 2, T] with its T.
pub struct Latents {
    pub video: Vec<f32>,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub audio: Option<(Vec<f32>, usize)>,
}

/// Latents in memory -> frames and sound (and an .mp4); see `decode`.
pub fn decode_latents(dev: &Arc<Device>, threads: usize, lat: Latents, vaes: &Vaes, out: Option<&Path>, check: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    decode_latents_chain(dev, threads, lat, vaes, out, check, None, ctl)
}

/// The same; `chain`: also write the next clip's anchors beside it - `<base>.last.png` (the last frame, lossless)
/// and `<base>.lastaud.safetensors` (the last 2 s of sound at the model's own level).
#[allow(clippy::too_many_arguments)]
pub fn decode_latents_chain(dev: &Arc<Device>, threads: usize, lat: Latents, vaes: &Vaes, out: Option<&Path>, check: Option<&Path>, chain: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    let vae = vaes.video;
    let t_all = Instant::now();
    let (t, mut h, mut w) = (lat.t, lat.h, lat.w);
    let mut z = lat.video;
    let mut report = json!({});
    let latent_up = if vaes.pixel.is_some() { None } else { vaes.upscale };
    if let Some((up, s)) = latent_up {
        let t0 = Instant::now();
        let u = crate::upscale::Upscaler::load(dev, &Checkpoint::open(up)?)?;
        let cancel = ctl.cancel;
        let (z2, ho, wo) = u.upscale(&z, t, h, w, s, &mut || {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error("cancelled".into()));
            }
            Ok(())
        })?;
        let secs = t0.elapsed().as_secs_f64();
        ctl.say(format!("upscale: latents {w}x{h} -> {wo}x{ho} (x{s}) in {secs:.1} s"));
        report["upscale_seconds"] = json!(secs);
        if let Some(c) = check {
            let d = Checkpoint::open(c)?;
            let want = crate::dtype::bytes_to_f32(&d.read("latents.video")?, d.get("latents.video")?.dtype)?;
            if want.len() == z2.len() {
                let (rel, cos) = reference::compare(&z2, &want);
                ctl.say(format!("check  : against the reference's upscaled latents: rel err {rel:.2e}, cosine {cos:.6}"));
                report["upscale_cosine"] = json!(cos);
            } else {
                ctl.say(format!("check  : the reference's latents have {} values ({:?}), ours {}", want.len(), d.get("latents.video")?.shape, z2.len()));
            }
        }
        drop(u);
        (z, h, w) = (z2, ho, wo);
    }
    let ck = Checkpoint::open(vae)?;
    let dec = crate::vae::VideoDecoder::load(dev, &ck, threads)?;
    ctl.say(format!("vae    : {} loaded, {:.2} GiB in {:.1} s", vae.display(), gib(dec.load_bytes), dec.load_seconds));
    ctl.say(format!("latents: video {t}x{h}x{w}"));
    let t0 = Instant::now();
    let mut tiles = 0usize;
    let cancel = ctl.cancel;
    let (px, frames) = dec.decode(&z, t, h, w, &mut || {
        tiles += 1;
        dev.wait()?;
        if cancel.load(Ordering::Relaxed) {
            return Err(Error("cancelled".into()));
        }
        Ok(())
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let (mut fh, mut fw) = (h * 16, w * 16);
    ctl.say(format!("decoded: {frames} frames of {fw}x{fh} in {secs:.1} s ({tiles} batches of tiles)"));
    let mut px = px;
    if let Some((pp, s)) = vaes.pixel {
        // the size the latent upscaler would have given: its latent size, in pixels
        let (ho, wo) = (((h as f32 * s).round() as usize) * 16, ((w as f32 * s).round() as usize) * 16);
        let t0 = Instant::now();
        let u = crate::esrgan::PixelUpscaler::load(dev, &Checkpoint::open(pp)?)?;
        px = u.upscale(&px, frames, fh, fw, ho, wo, &mut || ctl.check())?;
        let secs = t0.elapsed().as_secs_f64();
        ctl.say(format!("pixels : {fw}x{fh} -> {wo}x{ho} (x{s}, network x{} then area) by {} in {secs:.1} s", u.scale, pp.display()));
        report["pixel_upscale_seconds"] = json!(secs);
        (fh, fw) = (ho, wo);
    }
    for (k, v) in [("frames", json!(frames)), ("width", json!(fw)), ("height", json!(fh)), ("seconds", json!(secs)), ("tiles", json!(tiles))] {
        report[k] = v;
    }
    if let Some(c) = check {
        let d = Checkpoint::open(c)?;
        let want = crate::dtype::bytes_to_f32(&d.read("images")?, d.get("images")?.dtype)?; // [.., F, H, W, 3]
        if want.len() != px.len() {
            return Err(Error(format!("the reference decoded {} values ({:?}), this engine {}", want.len(), d.get("images")?.shape, px.len())));
        }
        let plane = fh * fw;
        // ours planar [3, F, H, W] -> the reference's [F, H, W, 3]
        let mut ours = vec![0f32; px.len()];
        for c in 0..3 {
            for fi in 0..frames {
                for i in 0..plane {
                    ours[(fi * plane + i) * 3 + c] = px[(c * frames + fi) * plane + i];
                }
            }
        }
        let (rel, cos) = reference::compare(&ours, &want);
        let mse = ours.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / ours.len() as f64;
        let psnr = 10.0 * (1.0 / mse.max(1e-20)).log10();
        ctl.say(format!("check  : against the reference's frames: rel err {rel:.2e}, cosine {cos:.6}, PSNR {psnr:.1} dB"));
        report["psnr"] = json!(psnr);
        report["cosine"] = json!(cos);
    }
    // the sound
    let mut sound = None;
    if let Some(ap) = vaes.audio {
        let (za, at) = lat.audio.as_ref().ok_or("the latents have no audio")?;
        let (za, at) = (za, *at);
        let t0 = Instant::now();
        let ad = crate::audio::AudioDecoder::load(dev, &Checkpoint::open(ap)?)?;
        let ch = ad.decode(za, at, &mut || ctl.check())?;
        let secs = t0.elapsed().as_secs_f64();
        ctl.say(format!("audio  : {at} latent frames -> {} samples x 2 in {secs:.1} s", ch[0].len()));
        report["audio_seconds"] = json!(secs);
        if let Some(c) = check {
            let d = Checkpoint::open(c)?;
            if d.entries.contains_key("waveform") {
                let want = crate::dtype::bytes_to_f32(&d.read("waveform")?, d.get("waveform")?.dtype)?; // [1, L, 2]
                let ours: Vec<f32> = (0..ch[0].len()).flat_map(|i| [ch[0][i], ch[1][i]]).collect();
                if want.len() == ours.len() {
                    let (rel, cos) = reference::compare(&ours, &want);
                    ctl.say(format!("check  : against the reference's sound: rel err {rel:.2e}, cosine {cos:.6}"));
                    report["audio_cosine"] = json!(cos);
                } else {
                    ctl.say(format!("check  : the reference's sound has {} values, ours {}", want.len(), ours.len()));
                }
            }
        }
        sound = Some(crate::media::Audio { channels: ch, sample_rate: crate::audio::SAMPLE_RATE });
    }
    if let Some(base) = chain {
        let plane = fh * fw;
        let last: Vec<f32> = (0..plane).flat_map(|i| (0..3).map(move |c| (c, i))).map(|(c, i)| px[(c * frames + frames - 1) * plane + i]).collect();
        let png = base.with_extension("last.png");
        crate::media::write_png(&png, &last, fw, fh)?;
        let mut written = vec![png.display().to_string()];
        if let Some(snd) = &sound {
            let keep = (2 * snd.sample_rate as usize).min(snd.channels[0].len());
            let n = snd.channels[0].len();
            let wav: Vec<f32> = snd.channels.iter().flat_map(|c| c[n - keep..].to_vec()).collect();
            let p = base.with_extension("lastaud.safetensors");
            crate::safetensors::write_f32(&p, &BTreeMap::from([("waveform".to_string(), (vec![1, 2, keep], wav))]),
                                            &BTreeMap::from([("sr".to_string(), snd.sample_rate.to_string())]))?;
            written.push(p.display().to_string());
        }
        ctl.say(format!("chain  : {}", written.join(", ")));
    }
    if let Some(o) = out {
        crate::media::write_mp4(o, &crate::media::Frames { px: &px, count: frames, height: fh, width: fw, fps: 24 }, sound.as_ref(), ctl.log)?;
        ctl.say(format!("written: {}", o.display()));
    }
    ctl.say(format!("total  : {:.1} s", t_all.elapsed().as_secs_f64()));
    Ok(report)
}

/// The encoders against a reference encoder dump (h3x.py encdump): the video encoder on a picture and on a clip's
/// last frames, the audio encoder on a stretch of sound - the same inputs, the latents compared.
pub fn check_encoders(dev: &Arc<Device>, vae: &Path, audio_vae: &Path, dump_path: &Path, ctl: &mut Ctl) -> Result<Value> {
    let d = Checkpoint::open(dump_path)?;
    let f = |k: &str| -> Result<Vec<f32>> { crate::dtype::bytes_to_f32(&d.read(k)?, d.get(k)?.dtype) };
    let mut report = serde_json::Map::new();
    let venc = crate::venc::VideoEncoder::load(dev, &Checkpoint::open(vae)?)?;
    for which in ["image", "clip"] {
        let key = format!("{which}.pixels");
        if !d.entries.contains_key(&key) {
            continue;
        }
        let sh = d.get(&key)?.shape.clone(); // [F, H, W, 3]
        let t0 = Instant::now();
        let (z, t) = venc.encode(&f(&key)?, sh[0], sh[1], sh[2], &mut || ctl.check())?;
        let secs = t0.elapsed().as_secs_f64();
        let want = f(&format!("{which}.latent"))?;
        let ws = d.get(&format!("{which}.latent"))?.shape.clone();
        if want.len() != z.len() {
            ctl.say(format!("{which:6}: {} frames -> {t} latent frames, the reference {:?}: sizes differ", sh[0], ws));
            continue;
        }
        let (rel, cos) = reference::compare(&z, &want);
        ctl.say(format!("{which:6}: {} frames {}x{} -> {t} latent frames in {secs:.2} s; against the reference: rel err {rel:.2e}, cosine {cos:.6}", sh[0], sh[2], sh[1]));
        report.insert(which.into(), json!({"cosine": cos, "seconds": secs}));
    }
    drop(venc);
    if d.entries.contains_key("audio.waveform") {
        let aenc = crate::audio::AudioEncoder::load(dev, &Checkpoint::open(audio_vae)?)?;
        let sh = d.get("audio.waveform")?.shape.clone(); // [1, C, L]
        let w = f("audio.waveform")?;
        let chans: Vec<Vec<f32>> = (0..2).map(|c| w[(c.min(sh[1] - 1)) * sh[2]..][..sh[2]].to_vec()).collect();
        let t0 = Instant::now();
        let (z, t) = aenc.encode(&chans)?;
        let secs = t0.elapsed().as_secs_f64();
        let want = f("audio.latent")?;
        if want.len() == z.len() {
            let (rel, cos) = reference::compare(&z, &want);
            ctl.say(format!("audio : {} samples -> {t} latent frames in {secs:.2} s; against the reference: rel err {rel:.2e}, cosine {cos:.6}", sh[2]));
            report.insert("audio".into(), json!({"cosine": cos, "seconds": secs}));
        } else {
            ctl.say(format!("audio : {t} latent frames, the reference {:?}: sizes differ", d.get("audio.latent")?.shape));
        }
    }
    Ok(Value::Object(report))
}

/// The text encoder's files: its GGUF weights, the tokenizer's directory, and its vision tower (pictures and clips in
/// the prompt need it).
pub struct TeFiles<'a> {
    pub te: &'a Path,
    pub tokenizer: &'a Path,
    pub visual: Option<&'a Path>,
    /// where `embedding:NAME` finds its rows (the effect embeddings)
    pub embeddings: Option<&'a Path>,
}

/// A prompt -> the denoiser's text conditioning: the tokenizer and the text encoder, streamed layer by layer.
/// `check`: a run dump whose `context` is the reference's conditioning for the same prompt. `out`: a `.safetensors`
/// with `context` [1, L, 5120] and `token_tags`.
pub fn encode(dev: &Arc<Device>, threads: usize, prompt: &str, files: &TeFiles, out: Option<&Path>, check: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    encode_with(dev, threads, &[], prompt, files, out, check, ctl)
}

/// The same with pictures ahead of the prompt (`<Picture i>: ` and each picture at its own size, sides rounded to 32),
/// as the image-to-video presentation shows its keyframes
#[allow(clippy::too_many_arguments)]
pub fn encode_with(dev: &Arc<Device>, threads: usize, pictures: &[&Path], prompt: &str, files: &TeFiles, out: Option<&Path>, check: Option<&Path>,
                   ctl: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    let mut pieces = Vec::new();
    for (i, p) in pictures.iter().enumerate() {
        let (pw, ph) = crate::media::size(p)?;
        let (w, h) = (round32(pw as f64), round32(ph as f64));
        let (px, _) = crate::media::read_frames_tail(p, 1, w, h, crate::media::Fit::Stretch)?;
        pieces.push(Piece::Text(format!("<Picture {}>: ", i + 1)));
        pieces.push(Piece::Vision(Frames { frames: vec![px], w, h }));
    }
    pieces.push(Piece::Text(prompt.to_string()));
    let (ctx, presented, hidden, secs) = encode_pieces(dev, threads, &pieces, files, ctl)?;
    let ids = presented.ids.clone();
    let mut report = json!({"tokens": ids.len(), "seconds": secs});
    if let Some(c) = check {
        let d = Checkpoint::open(c)?;
        let want = crate::dtype::bytes_to_f32(&d.read("context")?, d.get("context")?.dtype)?;
        if want.len() == ctx.len() {
            let (rel, cos) = reference::compare(&ctx, &want);
            ctl.say(format!("check  : against the reference's conditioning: rel err {rel:.2e}, cosine {cos:.6}"));
            // per token, the worst
            let worst = ctx.chunks(hidden).zip(want.chunks(hidden)).map(|(a, b)| reference::compare(a, b).1).fold(1f64, f64::min);
            ctl.say(format!("check  : the worst token's cosine {worst:.6}"));
            report["cosine"] = json!(cos);
            report["worst_token_cosine"] = json!(worst);
        } else {
            ctl.say(format!("check  : the reference's conditioning has {} values ({:?}), ours {}", want.len(), d.get("context")?.shape, ctx.len()));
        }
    }
    if let Some(o) = out {
        let mut t = BTreeMap::new();
        t.insert("context".to_string(), (vec![1, ids.len(), hidden], ctx));
        t.insert("token_tags".to_string(), (vec![ids.len()], presented.tags.iter().map(|t| *t as f32).collect()));
        crate::safetensors::write_f32(o, &t, &BTreeMap::from([("prompt".to_string(), prompt.to_string())]))?;
        ctl.say(format!("written: {}", o.display()));
    }
    ctl.say(format!("total  : {:.1} s", t_all.elapsed().as_secs_f64()));
    Ok(report)
}

/// The tokenizer and the streamed text encoder: (conditioning [L, hidden], token ids, hidden, seconds).
pub fn encode_prompt(dev: &Arc<Device>, threads: usize, prompt: &str, files: &TeFiles, ctl: &mut Ctl) -> Result<(Vec<f32>, Vec<u32>, usize, f64)> {
    encode_presentation(dev, threads, &[], prompt, files, ctl)
}

/// The same with labels ahead of the prompt (`<Audio 1>: ` per reference audio): the reference tokenizes every
/// piece of the presentation on its own and joins the ids.
pub fn encode_presentation(dev: &Arc<Device>, threads: usize, labels: &[String], prompt: &str, files: &TeFiles, ctl: &mut Ctl) -> Result<(Vec<f32>, Vec<u32>, usize, f64)> {
    let mut pieces: Vec<Piece> = labels.iter().map(|l| Piece::Text(l.clone())).collect();
    pieces.push(Piece::Text(prompt.to_string()));
    let (ctx, p, hidden, secs) = encode_pieces(dev, threads, &pieces, files, ctl)?;
    Ok((ctx, p.ids, hidden, secs))
}

/// The presentation's pieces - text, and pictures through the vision tower - to the conditioning: (conditioning
/// [L, hidden], what was presented: ids, modality tags, vision blocks; hidden, seconds)
pub fn encode_pieces(dev: &Arc<Device>, threads: usize, pieces: &[Piece], files: &TeFiles, ctl: &mut Ctl) -> Result<(Vec<f32>, crate::vision::Presented, usize, f64)> {
    let (te, tokenizer) = (files.te, files.tokenizer);
    let tok = crate::tokenizer::Tokenizer::load(tokenizer)?;
    let presented = crate::vision::present(pieces, &tok, files.visual, files.embeddings, dev.index, &mut |l| ctl.say(l))?;
    let ids = &presented.ids;
    ctl.say(format!("prompt : {} tokens{}", ids.len(), if presented.has_vision() {
        format!(" ({} in {} vision blocks)", presented.seen.iter().map(|s| s.grid.0 * s.grid.1 + 2).sum::<usize>(), presented.seen.len())
    } else {
        String::new()
    }));
    let enc = crate::te::TextEncoder::open(te)?;
    ctl.say(format!("te     : {} layers, hidden {}, {} query / {} key-value heads ({})", enc.layers, enc.hidden, enc.heads, enc.kv_heads, te.display()));
    let t0 = Instant::now();
    let cancel = ctl.cancel;
    let layers = enc.layers;
    let mut progress = ctl.progress.take();
    let r = enc.encode(dev, ids, &presented.seen, &presented.embeds, threads, &mut |i| {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error("cancelled".into()));
        }
        if let Some(p) = progress.as_mut() {
            p(i, layers);
        }
        Ok(())
    });
    ctl.progress = progress;
    let ctx = r?;
    let secs = t0.elapsed().as_secs_f64();
    ctl.say(format!("encoded: {} x {} in {secs:.1} s (layers from {})", ids.len(), enc.hidden, enc.source.lock().map(|s| s.clone()).unwrap_or_default()));
    Ok((ctx, presented, enc.hidden, secs))
}

/// What a `generate` job makes: the prompt, the canvas, the length, the sampling, the files it reads.
pub struct ClipSpec<'a> {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub seconds: f64,
    pub seed: u64,
    pub steps: usize,
    /// the sampler and the schedule (Euler on H3's own shifted schedule by default; the others are ComfyUI's)
    pub sampler: nextsycl_diffusion::Sampler,
    pub schedule: nextsycl_diffusion::Schedule,
    pub te: TeFiles<'a>,
    pub vaes: Vaes<'a>,
    /// LoRAs and their strengths (they stack)
    pub lora: Vec<(&'a Path, f32)>,
    pub inputs: ClipInputs<'a>,
}

/// What a clip is anchored to, as the reference's options name them (reference/h3x.py gen --help).
#[derive(Default)]
pub struct ClipInputs<'a> {
    /// a previous clip's `.lastlat.safetensors`: its last latent frame pinned at frame 0 (no decode, no codec)
    pub first_latent: Option<&'a Path>,
    /// a picture (or a video's last frame) pinned at frame 0 / at the last frame
    pub first_frame: Option<&'a Path>,
    pub last_frame: Option<&'a Path>,
    /// a picture whose exposure the keyframe pictures are matched to (a chain's drift fix)
    pub frame_ref: Option<&'a Path>,
    /// motion guides: the last N frames of a clip anchored as one moving keyframe at a frame (negative: from the end)
    pub guides: Vec<(&'a Path, usize, i64)>,
    /// the last seconds of a sound (`.lastaud.safetensors` or any media file) pinned at frame 0
    pub first_audio: Option<(&'a Path, f64)>,
    /// how much to trust the video keyframes (0.999 by default; lower = trust a degraded anchor less)
    pub cond_noise_aug: Option<f32>,
    /// the flow shifts (12 and 3 by default)
    pub shift: Option<(f32, f32)>,
    /// reference audio, a voice to speak in: `<Audio 1>`, `<Audio 2>` ... in the prompt (up to 3)
    pub ref_audio: Vec<&'a Path>,
    /// reference pictures (identity, a look): `<Picture 1>` ... in the prompt (up to 9) - with references the clip is
    /// ref2va (the ref2va denoiser is the one trained for it)
    pub ref_images: Vec<&'a Path>,
    /// reference pictures at the reference pipeline's 2048-pixel short side instead of the clip's area (closer
    /// likeness, several times the tokens)
    pub ref_image_max: bool,
    /// reference clips (motion, a performance): `<Video 1>` ... (up to 3), 24 fps, their own sound as `<Audio j>`
    /// unless `ref_video_sound` is off
    pub ref_videos: Vec<&'a Path>,
    pub ref_video_sound: bool,
    /// the keyframe pictures also shown to the text encoder as `<Picture i>` (as ComfyUI's image-to-video node does;
    /// off: the latents alone, as before)
    pub te_pictures: bool,
    /// a masked run: a previous clip's `.latents.safetensors`; the parts not regenerated are kept from it (a source
    /// shorter than the clip is extended)
    pub source: Option<&'a Path>,
    /// the stretch to regenerate, in seconds [from, to)
    pub regen: Option<(f64, f64)>,
    /// and only inside this box, in pixels (x0, y0, x1, y1)
    pub regen_box: Option<(usize, usize, usize, usize)>,
}

/// The masks of a masked run from the source latents and the stretch / box to regenerate.
fn clip_inpaint(inp: &ClipInputs, shape: Shape) -> Result<Option<denoiser::Inpaint>> {
    let Some(src) = inp.source else { return Ok(None) };
    let ck = Checkpoint::open(src)?;
    let (ev, ea) = (ck.get("samples.video")?.clone(), ck.get("samples.audio")?.clone());
    let (st, sh_, sw) = (ev.shape[2], ev.shape[3], ev.shape[4]);
    let sat = ea.shape[3];
    if sh_ != shape.h || sw != shape.w {
        return Err(Error(format!("{}: a {}x{} latent for a {}x{} clip", src.display(), sw, sh_, shape.w, shape.h)));
    }
    let v = crate::dtype::bytes_to_f32(&ck.read("samples.video")?, ev.dtype)?;
    let a = crate::dtype::bytes_to_f32(&ck.read("samples.audio")?, ea.dtype)?;
    let (t, n, at) = (shape.t, shape.h * shape.w, shape.audio_t);
    // the source fitted to the clip (cut, or padded with zeros where the clip extends it)
    let mut source_v = vec![0f32; 24 * t * n];
    for c in 0..24 {
        for ti in 0..t.min(st) {
            source_v[(c * t + ti) * n..][..n].copy_from_slice(&v[(c * st + ti) * n..][..n]);
        }
    }
    let mut source_a = vec![0f32; 32 * 2 * at];
    for c in 0..64 {
        let k = at.min(sat);
        source_a[c * at..c * at + k].copy_from_slice(&a[c * sat..c * sat + k]);
    }
    // 1 = generate: where the source ends, and the stretch / box asked for
    let mut mask_v: Vec<f32> = (0..t * n).map(|i| if i / n >= st { 1.0 } else { 0.0 }).collect();
    let mut mask_a: Vec<f32> = (0..at).map(|i| if i >= sat { 1.0 } else { 0.0 }).collect();
    if let Some((from, to)) = inp.regen {
        let (f0, f1) = (from * 24.0, to * 24.0);
        let (bx0, by0, bx1, by1) = inp.regen_box.map(|(x0, y0, x1, y1)| (x0 / 16, y0 / 16, x1.div_ceil(16), y1.div_ceil(16))).unwrap_or((0, 0, shape.w, shape.h));
        for ti in 0..t {
            // the pixel frames latent frame ti covers: frame 0 alone, then four each
            let (p0, p1) = if ti == 0 { (0.0, 1.0) } else { ((1 + 4 * (ti - 1)) as f64, (1 + 4 * ti) as f64) };
            if p1 <= f0 || p0 >= f1 {
                continue;
            }
            for y in by0..by1.min(shape.h) {
                for x in bx0..bx1.min(shape.w) {
                    mask_v[ti * n + y * shape.w + x] = 1.0;
                }
            }
        }
        for (j, m) in mask_a.iter_mut().enumerate() {
            let (s0, s1) = (j as f64 / 40.0, (j + 1) as f64 / 40.0);
            if s1 > from && s0 < to {
                *m = 1.0;
            }
        }
    }
    if mask_v.iter().all(|m| *m == 0.0) && mask_a.iter().all(|m| *m == 0.0) {
        return Err(Error("a source and nothing to regenerate: give regen (seconds) or a longer clip than the source".into()));
    }
    Ok(Some(denoiser::Inpaint { source_v, source_a, mask_v, mask_a }))
}

/// The video keyframes and audio keyframe of a clip, encoded (the encoders loaded only when a picture or sound
/// needs them).
fn clip_keyframes(dev: &Arc<Device>, inp: &ClipInputs, vaes: &Vaes, (w, h, frames): (usize, usize, usize), ctl: &mut Ctl) -> Result<(Vec<KeyframeIn>, Vec<Frames>)> {
    let mut kfs: Vec<KeyframeIn> = Vec::new();
    let pictures = std::cell::RefCell::new(Vec::new());
    let needs_venc = inp.first_frame.is_some() || inp.last_frame.is_some() || !inp.guides.is_empty();
    let venc = match needs_venc {
        true => Some(crate::venc::VideoEncoder::load(dev, &Checkpoint::open(vaes.video)?)?),
        false => None,
    };
    let reference_luma = match inp.frame_ref {
        Some(p) => Some(crate::media::luma(&crate::media::read_frames_tail(p, 1, w, h, crate::media::Fit::Stretch)?.0)),
        None => None,
    };
    let picture = |path: &Path, index: usize, what: &str, ctl: &mut Ctl| -> Result<KeyframeIn> {
        let (mut px, _) = crate::media::read_frames_tail(path, 1, w, h, crate::media::Fit::Stretch)?;
        if let Some(rl) = reference_luma {
            let l = crate::media::luma(&px);
            if l > 1e-4 {
                // correct the drift, never re-grade the shot
                let g = (rl / l).clamp(0.75, 1.35);
                px.iter_mut().for_each(|v| *v = (*v * g).clamp(0.0, 1.0));
                ctl.say(format!("exposure: {what} Y {:.1} -> {:.1}, gain {g:.3}", l * 255.0, rl * 255.0));
            }
        }
        let (z, vt) = venc.as_ref().expect("encoder loaded").encode(&px, 1, h, w, &mut || Ok(()))?;
        ctl.say(format!("keyframe: {what} {} at frame {index}", path.display()));
        pictures.borrow_mut().push(Frames { frames: vec![px], w, h });
        Ok(KeyframeIn { frame_index: index, video: Some((z, vt)), audio: None })
    };
    if let Some(p) = inp.first_latent {
        let ck = Checkpoint::open(p)?;
        let e = ck.get("latent")?; // [1, 24, 1, H, W]
        if e.shape[3] != h / 16 || e.shape[4] != w / 16 {
            return Err(Error(format!("{}: a {:?} latent for a {w}x{h} clip", p.display(), e.shape)));
        }
        let z = crate::dtype::bytes_to_f32(&ck.read("latent")?, e.dtype)?;
        ctl.say(format!("keyframe: last latent of {} at frame 0", p.display()));
        kfs.push(KeyframeIn { frame_index: 0, video: Some((z, e.shape[2])), audio: None });
    } else if let Some(p) = inp.first_frame {
        kfs.push(picture(p, 0, "first frame", ctl)?);
    }
    if let Some(p) = inp.last_frame {
        kfs.push(picture(p, frames - 1, "last frame", ctl)?);
    }
    for (p, n, idx) in &inp.guides {
        let (px, mut got) = crate::media::read_frames_tail(p, *n, w, h, crate::media::Fit::Cover)?;
        // guide clips run 5, 22, 39 ... frames (17k + 5); fewer than 5: the first picture alone
        let fsz = w * h * 3;
        let mut start = 0;
        if got < 5 {
            got = 1;
        } else {
            while got % 17 != 5 {
                got -= 1;
                start += 1;
            }
        }
        let index = if *idx >= 0 { *idx as usize } else { (frames as i64 + idx).max(0) as usize };
        if index + got > frames {
            return Err(Error(format!("a {got}-frame guide at frame {idx} does not fit in the clip's {frames} frames")));
        }
        let (z, vt) = venc.as_ref().expect("encoder loaded").encode(&px[start * fsz..(start + got) * fsz], got, h, w, &mut || ctl.check())?;
        ctl.say(format!("keyframe: motion guide, last {got} frames of {} ({vt} latent frames) at frame {index}", p.display()));
        kfs.push(KeyframeIn { frame_index: index, video: Some((z, vt)), audio: None });
    }
    drop(venc);
    if let Some((p, secs)) = inp.first_audio {
        // the audio encoder needs ~0.6 s at least; the reference anchors at least 1 s
        let secs = secs.max(1.0);
        let rate = crate::audio::SAMPLE_RATE;
        let chans: Vec<Vec<f32>> = if p.extension().is_some_and(|e| e == "safetensors") {
            let ck = Checkpoint::open(p)?;
            let e = ck.get("waveform")?; // [1, C, L]
            let w = crate::dtype::bytes_to_f32(&ck.read("waveform")?, e.dtype)?;
            let (c, l) = (e.shape[1], e.shape[2]);
            let keep = ((secs * rate as f64) as usize).min(l);
            (0..2).map(|ch| w[ch.min(c - 1) * l + l - keep..ch.min(c - 1) * l + l].to_vec()).collect()
        } else {
            crate::media::read_audio_tail(p, secs, rate)?
        };
        let aenc = crate::audio::AudioEncoder::load(dev, &Checkpoint::open(vaes.audio.ok_or("an audio keyframe needs the audio decoder's checkpoint (audio_vae)")?)?)?;
        let (z, rt) = aenc.encode(&chans)?;
        ctl.say(format!("keyframe: last {secs:.2} s of the sound of {} ({rt} latent frames) at frame 0", p.display()));
        // shares the frame-0 entry with a picture anchor, as the reference does
        match kfs.iter_mut().find(|k| k.frame_index == 0 && k.audio.is_none()) {
            Some(k) => k.audio = Some((z, rt)),
            None => kfs.push(KeyframeIn { frame_index: 0, video: None, audio: Some((z, rt)) }),
        }
    }
    Ok((kfs, pictures.into_inner()))
}

const CANVAS_MULTIPLE: usize = 32;

fn round32(x: f64) -> usize {
    ((x / CANVAS_MULTIPLE as f64).round() as usize * CANVAS_MULTIPLE).max(CANVAS_MULTIPLE)
}

/// A reference clip's canvas: a 768 short side within 768 x 1344 pixels, each side to a multiple of 32 (ComfyUI's
/// adapt_canvas); a smaller clip keeps its own size
fn ref_clip_canvas(vw: usize, vh: usize) -> (usize, usize) {
    let ratio = vw as f64 / vh as f64;
    let (mut w, mut h) = if ratio >= 1.0 { (768.0 * ratio, 768.0) } else { (768.0, 768.0 / ratio) };
    let max = 768.0 * 1344.0;
    if w * h > max {
        let s = (max / (w * h)).sqrt();
        (w, h) = (w * s, h * s);
    }
    let (cw, ch) = (round32(w), round32(h));
    if vw * vh < cw * ch {
        (round32(vw as f64), round32(vh as f64))
    } else {
        (cw, ch)
    }
}

/// The references (ref2va, ComfyUI's MiniMaxH3ReferenceToVideo): their pieces of the presentation - pictures first,
/// then clips (a clip's sound labelled `<Audio j>` right before its `<Video k>`), then sounds - and their latents in
/// the same order
fn clip_refs(dev: &Arc<Device>, inp: &ClipInputs, vaes: &Vaes, (w, h, frames): (usize, usize, usize), ctl: &mut Ctl) -> Result<(Vec<Piece>, Vec<RefIn>)> {
    let (mut pieces, mut refs) = (Vec::new(), Vec::new());
    if inp.ref_images.is_empty() && inp.ref_videos.is_empty() && inp.ref_audio.is_empty() {
        return Ok((pieces, refs));
    }
    let venc = match inp.ref_images.is_empty() && inp.ref_videos.is_empty() {
        true => None,
        false => Some(crate::venc::VideoEncoder::load(dev, &Checkpoint::open(vaes.video)?)?),
    };
    let aenc = std::cell::OnceCell::new();
    let audio_enc = || -> Result<&crate::audio::AudioEncoder> {
        if aenc.get().is_none() {
            let ck = Checkpoint::open(vaes.audio.ok_or("reference sound needs the audio decoder's checkpoint (audio_vae)")?)?;
            let _ = aenc.set(crate::audio::AudioEncoder::load(dev, &ck)?);
        }
        Ok(aenc.get().expect("set above"))
    };
    let rate = crate::audio::SAMPLE_RATE;
    let (mut n_audio, mut n_video) = (0, 0);
    for (i, p) in inp.ref_images.iter().enumerate() {
        let (pw, ph) = crate::media::size(p)?;
        let scale = if inp.ref_image_max { (2048.0 / pw.min(ph) as f64).min(1.0) } else { ((w * h) as f64 / (pw * ph) as f64).sqrt().min(1.0) };
        let (tw, th) = (round32(pw as f64 * scale), round32(ph as f64 * scale));
        let (px, _) = crate::media::read_frames_tail(p, 1, tw, th, crate::media::Fit::Stretch)?;
        let (z, _) = venc.as_ref().expect("loaded for pictures").encode(&px, 1, th, tw, &mut || ctl.check())?;
        ctl.say(format!("ref    : <Picture {}> = {} ({tw}x{th})", i + 1, p.display()));
        pieces.push(Piece::Text(format!("<Picture {}>: ", i + 1)));
        pieces.push(Piece::Vision(Frames { frames: vec![px], w: tw, h: th }));
        refs.push(RefIn::Image { latent: z, h: th / 16, w: tw / 16 });
    }
    for p in &inp.ref_videos {
        let (vw, vh) = crate::media::size(p)?;
        let (cw, ch) = ref_clip_canvas(vw, vh);
        let (px, got) = crate::media::read_frames_at(p, cw, ch, crate::media::Fit::Stretch, Some((24.0, frames)))?;
        if got < 5 {
            return Err(Error(format!("{}: a reference clip needs 5 frames at least (~0.2 s at 24 fps)", p.display())));
        }
        let mut n = got;
        while n % 17 != 5 {
            n -= 1;
        }
        let fsz = cw * ch * 3;
        let sound = if inp.ref_video_sound { crate::media::read_audio_tail(p, 0.0, rate).ok().filter(|c| !c[0].is_empty()) } else { None };
        let audio = match sound {
            Some(chans) => {
                n_audio += 1;
                pieces.push(Piece::Text(format!("<Audio {n_audio}>: ")));
                Some(audio_enc()?.encode(&chans)?)
            }
            None => None,
        };
        n_video += 1;
        pieces.push(Piece::Text(format!("<Video {n_video}>: ")));
        // the encoder sees it at 2 fps, two frames a block (an odd last one repeated), each block labelled with its
        // middle time
        let picks: Vec<usize> = (0..n).step_by(12).collect();
        for (j, pair) in picks.chunks(2).enumerate() {
            let (a, b) = (pair[0], *pair.last().expect("non-empty"));
            let t = if pair.len() == 2 { (2 * j) as f64 / 2.0 + 0.25 } else { (2 * j) as f64 / 2.0 };
            pieces.push(Piece::Text(format!("<{t:.1} seconds>")));
            pieces.push(Piece::Vision(Frames { frames: vec![px[a * fsz..(a + 1) * fsz].to_vec(), px[b * fsz..(b + 1) * fsz].to_vec()], w: cw, h: ch }));
        }
        let (z, vt) = venc.as_ref().expect("loaded for clips").encode(&px[..n * fsz], n, ch, cw, &mut || ctl.check())?;
        ctl.say(format!("ref    : <Video {n_video}> = {} ({cw}x{ch}, {n} frames{})", p.display(),
                        audio.as_ref().map_or(String::new(), |a| format!(", its sound as <Audio {n_audio}>: {} latent frames", a.1))));
        refs.push(RefIn::Video { latent: z, t: vt, h: ch / 16, w: cw / 16, audio });
    }
    for p in &inp.ref_audio {
        let chans = crate::media::read_audio_tail(p, 0.0, rate)?;
        let (z, t) = audio_enc()?.encode(&chans)?;
        n_audio += 1;
        ctl.say(format!("ref    : <Audio {n_audio}> = {} ({:.1} s, {t} latent frames)", p.display(), chans[0].len() as f64 / rate as f64));
        pieces.push(Piece::Text(format!("<Audio {n_audio}>: ")));
        refs.push(RefIn::Audio { latent: z, t });
    }
    Ok((pieces, refs))
}

/// The model's frame grid: frames at 24 fps snapped up to 17k + 5; (frames, video latent frames, audio latent frames).
pub fn temporal_shape(seconds: f64) -> (usize, usize, usize) {
    let mut n = ((seconds * 24.0).round() as usize).max(5);
    while n % 17 != 5 {
        n += 1;
    }
    let lt = if n <= 5 { 2 } else { (n - 5) / 17 * 5 + 2 };
    (n, lt, (n as f64 / 24.0 * 40.0).round() as usize)
}

/// A whole clip in the engine: the prompt through the text encoder, the sampler on the loaded denoiser, the
/// (upscaler and) decoders, an .mp4.
pub fn generate(e: &Engine, c: &ClipSpec, out: &Path, ctl: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    if !c.width.is_multiple_of(32) || !c.height.is_multiple_of(32) {
        return Err(Error(format!("the canvas must be a multiple of 32 on both sides, got {}x{}", c.width, c.height)));
    }
    let (frames, lt, at) = temporal_shape(c.seconds);
    let shape = Shape { t: lt, h: c.height / 16, w: c.width / 16, audio_t: at };
    ctl.say(format!("clip   : {}x{}, {frames} frames ({:.2} s), {} steps, seed {}", c.width, c.height, frames as f64 / 24.0, c.steps, c.seed));
    // the keyframes and the references first (their pictures are part of the prompt's presentation)
    let mut cond = Conditions { seed: c.seed, ..Conditions::default() };
    if let Some(a) = c.inputs.cond_noise_aug {
        cond.visual_aug = a;
    }
    let (keyframes, kf_pictures) = clip_keyframes(&e.dev, &c.inputs, &c.vaes, (c.width, c.height, frames), ctl)?;
    cond.keyframes = keyframes;
    let (mut pieces, refs) = clip_refs(&e.dev, &c.inputs, &c.vaes, (c.width, c.height, frames), ctl)?;
    cond.refs = refs;
    if pieces.is_empty() && c.inputs.te_pictures && !kf_pictures.is_empty() {
        if c.te.visual.is_some() {
            for (i, f) in kf_pictures.into_iter().enumerate() {
                pieces.push(Piece::Text(format!("<Picture {}>: ", i + 1)));
                pieces.push(Piece::Vision(f));
            }
        } else {
            ctl.say("vision : no vision tower file (te_visual): the keyframes' pictures reach the denoiser only".to_string());
        }
    }
    pieces.push(Piece::Text(c.prompt.clone()));
    let (ctx, presented, hidden, te_secs) = encode_pieces(&e.dev, e.threads, &pieces, &c.te, ctl)?;
    let ids = &presented.ids;
    ctl.check()?;
    let mut lora: Option<crate::lora::LoraSet> = None;
    for (p, strength) in &c.lora {
        let l = crate::lora::LoraSet::load(&e.dev, &Checkpoint::open(p)?, *strength, e.model.blocks.len(), 2)?;
        if l.max_rank > crate::dit::MAX_LORA_RANK {
            return Err(Error(format!("LoRA rank {} is over {}", l.max_rank, crate::dit::MAX_LORA_RANK)));
        }
        ctl.say(format!("lora   : {} @ {strength} ({} layers, rank {})", p.display(), l.layers, l.max_rank));
        match lora.as_mut() {
            Some(set) => set.stack(l),
            None => lora = Some(l),
        }
    }
    let t0 = Instant::now();
    let text = {
        let refiner = TextRefiner::load(&e.dev, &e.ck, e.threads)?;
        let t = Tensor::from_bytes(&e.dev, DType::BF16, &[ids.len(), hidden], &denoiser::bf16_bytes(&ctx))?;
        let r = refiner.run(&e.model.cfg, &t, lora.as_ref().map(|l| l.refiner.as_slice()))?;
        e.dev.wait()?;
        r
    };
    ctl.say(format!("text   : refined in {:.1} s", t0.elapsed().as_secs_f64()));
    let mut schedule = Schedule::default();
    if let Some((sv, sa)) = c.inputs.shift {
        schedule.shift_video = sv;
        schedule.shift_audio = sa;
    }
    let inpaint = clip_inpaint(&c.inputs, shape)?;
    // vision blocks run on the video tables (tag 0), the rest on the text ones
    let tags = presented.has_vision().then(|| presented.tags.clone());
    let mut d = Denoiser::new(&e.model, &e.outer, text, tags, shape, schedule, &cond)?;
    d.lora = lora.as_ref();
    if let Some(p) = &inpaint {
        let gen_v = p.mask_v.iter().filter(|m| **m > 0.0).count() as f64 / p.mask_v.len() as f64;
        let gen_a = p.mask_a.iter().filter(|m| **m > 0.0).count() as f64 / p.mask_a.len() as f64;
        ctl.say(format!("masked : regenerating {:.0}% of the video, {:.0}% of the sound; the rest kept from the source", gen_v * 100.0, gen_a * 100.0));
        d.masks = Some(denoiser::RowMasks::new(p, shape));
    }
    let n_v = 24 * shape.t * shape.h * shape.w;
    let (noise_v, noise_a) = crate::noise::clip_noise(c.seed, n_v, 32 * 2 * shape.audio_t);
    // H3's own schedule (the video shift), or one of ComfyUI's at that shift
    let sigmas = if c.schedule == nextsycl_diffusion::Schedule::Shift {
        denoiser::sigmas(c.steps, schedule.shift_video)
    } else {
        nextsycl_diffusion::samplers::sigmas_for(c.sampler, c.schedule, c.steps as u32, schedule.shift_video)
    };
    if c.sampler != nextsycl_diffusion::Sampler::Euler || c.schedule != nextsycl_diffusion::Schedule::Shift {
        ctl.say(format!("sampler: {} on the {} schedule ({} levels)", c.sampler.name(), c.schedule.name(), sigmas.len()));
    }
    ctl.say(format!("tokens : {} in the denoiser", d.tokens()));
    let nblocks = e.model.blocks.len();
    let steps = c.steps;
    let t0 = Instant::now();
    let (v, a) = {
        let dev = e.dev.clone();
        let cancel = ctl.cancel;
        let mut progress = ctl.progress.take();
        let mut times = Vec::new();
        let mut t_step = Instant::now();
        let r = denoiser::sample_with(&mut d, c.sampler, &noise_v, &noise_a, &sigmas, inpaint.as_ref(), c.seed, &mut |_, _, _| {
            times.push(t_step.elapsed().as_secs_f64());
            t_step = Instant::now();
            Ok(())
        }, &mut |i, b| {
            if b == 0 || b == nblocks / 2 {
                dev.wait()?;
                if cancel.load(Ordering::Relaxed) {
                    return Err(Error("cancelled".into()));
                }
            }
            if let Some(p) = progress.as_mut() {
                p(i * nblocks + b, steps * nblocks);
            }
            Ok(())
        });
        ctl.progress = progress;
        let (v, a) = r?;
        ctl.say(format!("sampled: {steps} steps in {:.1} s ({})", t0.elapsed().as_secs_f64(), times.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>().join(" ")));
        (v, a)
    };
    let sample_secs = t0.elapsed().as_secs_f64();
    drop(d);
    // the next clip's latent anchor: the last latent frame, before any upscaling
    {
        let n = shape.h * shape.w;
        let last: Vec<f32> = (0..24).flat_map(|ch| v[(ch * shape.t + shape.t - 1) * n..][..n].to_vec()).collect();
        let p = out.with_extension("lastlat.safetensors");
        crate::safetensors::write_f32(&p, &BTreeMap::from([("latent".to_string(), (vec![1, 24, 1, shape.h, shape.w], last))]), &BTreeMap::new())?;
        // and the whole clip's latents: a later masked run regenerates part of it or extends it
        let p = out.with_extension("latents.safetensors");
        crate::safetensors::write_f32(&p, &BTreeMap::from([
            ("samples.video".to_string(), (vec![1, 24, shape.t, shape.h, shape.w], v.clone())),
            ("samples.audio".to_string(), (vec![1, 32, 2, shape.audio_t], a.clone())),
        ]), &BTreeMap::new())?;
    }
    let mut rep = decode_latents_chain(&e.dev, e.threads, Latents { video: v, t: shape.t, h: shape.h, w: shape.w, audio: Some((a, shape.audio_t)) }, &c.vaes, Some(out), None, Some(out), ctl)?;
    let total = t_all.elapsed().as_secs_f64();
    rep["text_encoder_seconds"] = json!(te_secs);
    rep["sample_seconds"] = json!(sample_secs);
    rep["total_seconds"] = json!(total);
    ctl.say(format!("clip   : {} in {total:.1} s", out.display()));
    Ok(rep)
}

/// One job, as the daemon receives it: `{"kind": "bench-blocks", "tokens": 47173}` and so on.
pub fn run(e: &Engine, spec: &Value, ctl: &mut Ctl) -> Result<Value> {
    let kind = spec.get("kind").and_then(|k| k.as_str()).ok_or("a job needs a \"kind\"")?;
    let num = |k: &str| spec.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    match kind {
        "bench-blocks" => bench_blocks(e, num("tokens").unwrap_or(16500), num("blocks"), ctl),
        "check-block" => {
            let dump = spec.get("dump").and_then(|d| d.as_str()).ok_or("check-block needs \"dump\": a path the engine can read")?;
            check_block(e, Path::new(dump), ctl)
        }
        "denoise" => {
            let dump = spec.get("dump").and_then(|d| d.as_str()).ok_or("denoise needs \"dump\": a run dump the engine can read")?;
            let out = spec.get("out").and_then(|d| d.as_str()).map(Path::new);
            denoise(e, Path::new(dump), out, ctl)
        }
        "decode" => {
            let s = |k: &str| spec.get(k).and_then(|d| d.as_str()).map(Path::new);
            let latents = s("latents").ok_or("decode needs \"latents\": a latents file the engine can read")?;
            let video = s("vae").ok_or("decode needs \"vae\": the video decoder's checkpoint")?;
            let upscale = s("upscaler").map(|p| (p, spec.get("upscale").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32));
            let factor = spec.get("upscale").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
            let pixel = s("pixel_upscaler").map(|p| (p, factor));
            decode(&e.dev, e.threads, latents, &Vaes { video, audio: s("audio_vae"), upscale, pixel }, s("out"), s("check"), ctl)
        }
        "generate" => {
            let s = |k: &str| spec.get(k).and_then(|d| d.as_str());
            let n = |k: &str| spec.get(k).and_then(|d| d.as_f64());
            let prompt = match (s("prompt"), s("prompt_file")) {
                (Some(p), _) => p.to_string(),
                (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?.trim().to_string(),
                _ => return Err(Error("generate needs \"prompt\" or \"prompt_file\"".into())),
            };
            let out = s("out").ok_or("generate needs \"out\": where to write the .mp4")?;
            let upscale = n("upscale").filter(|u| *u > 1.0);
            let c = ClipSpec {
                prompt,
                width: n("width").unwrap_or(384.0) as usize,
                height: n("height").unwrap_or(288.0) as usize,
                seconds: n("seconds").unwrap_or(2.0),
                seed: n("seed").unwrap_or(0.0) as u64,
                steps: n("steps").unwrap_or(8.0) as usize,
                sampler: match s("sampler") {
                    Some(x) => nextsycl_diffusion::Sampler::parse(x).ok_or_else(|| Error(format!("sampler {x}? ({})", nextsycl_diffusion::samplers::NAMES.join(", "))))?,
                    None => nextsycl_diffusion::Sampler::Euler,
                },
                schedule: match s("schedule") {
                    Some(x) => nextsycl_diffusion::Schedule::parse(x).ok_or_else(|| Error(format!("schedule {x}?")))?,
                    None => nextsycl_diffusion::Schedule::Shift,
                },
                te: TeFiles {
                    te: Path::new(s("te").unwrap_or("/models/teacher/qwen3vl_32b_minimax_h3-Q4_K_M.gguf")),
                    tokenizer: Path::new(s("tokenizer").unwrap_or("/app/tokenizer")),
                    visual: s("te_visual").map(Path::new).filter(|p| p.exists()),
                    embeddings: s("embeddings").map(Path::new).filter(|p| p.is_dir()),
                },
                vaes: Vaes {
                    video: Path::new(s("vae").unwrap_or("/models/Comfy-Org-MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors")),
                    audio: Some(Path::new(s("audio_vae").unwrap_or("/models/Comfy-Org-MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors"))),
                    upscale: upscale.map(|u| (Path::new(s("upscaler").unwrap_or("/models/upscaler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")), u as f32)),
                    // "pixel_upscaler": an ESRGAN-type network's checkpoint - the frames enlarged instead of the latents
                    // (missing weights: a note in the log and the latent upscaler instead of a failed clip)
                    pixel: upscale.and_then(|u| s("pixel_upscaler").map(|p| (Path::new(p), u as f32))).filter(|(p, _)| {
                        let there = p.exists();
                        if !there {
                            ctl.say(format!("pixels : {} is not there - the latent upscaler instead", p.display()));
                        }
                        there
                    }),
                },
                lora: Vec::new(),
                inputs: ClipInputs {
                    first_latent: s("first_latent").map(Path::new),
                    first_frame: s("first_frame").map(Path::new),
                    last_frame: s("last_frame").map(Path::new),
                    frame_ref: s("first_frame_ref").map(Path::new),
                    guides: match s("guide_clip") {
                        Some(g) => {
                            let mut it = g.split(':');
                            let p = it.next().unwrap_or(g);
                            let n = it.next().and_then(|v| v.parse().ok()).unwrap_or(22);
                            let i = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                            vec![(Path::new(p), n, i)]
                        }
                        None => Vec::new(),
                    },
                    first_audio: s("first_audio").map(|p| (Path::new(p), n("first_audio_s").unwrap_or(1.0))),
                    cond_noise_aug: n("cond_noise_aug").map(|v| v as f32),
                    shift: match (n("shift_video"), n("shift_audio")) {
                        (None, None) => None,
                        (v, a) => Some((v.unwrap_or(12.0) as f32, a.unwrap_or(3.0) as f32)),
                    },
                    ref_audio: s("ref_audio").map(|r| r.split(',').map(Path::new).take(3).collect()).unwrap_or_default(),
                    ref_images: s("ref_images").map(|r| r.split(',').filter(|x| !x.is_empty()).map(Path::new).take(9).collect()).unwrap_or_default(),
                    ref_image_max: s("ref_image_size") == Some("max"),
                    ref_videos: s("ref_videos").map(|r| r.split(',').filter(|x| !x.is_empty()).map(Path::new).take(3).collect()).unwrap_or_default(),
                    ref_video_sound: spec.get("ref_video_sound").and_then(Value::as_bool).unwrap_or(true),
                    te_pictures: spec.get("te_pictures").and_then(Value::as_bool).unwrap_or(true),
                    source: s("source").map(Path::new),
                    regen: s("regen").and_then(|r| {
                        let (a, b) = r.split_once('-')?;
                        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
                    }),
                    regen_box: s("regen_box").and_then(|r| {
                        let v: Vec<usize> = r.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                        (v.len() == 4).then(|| (v[0], v[1], v[2], v[3]))
                    }),
                },
            };
            // "PATH[:STRENGTH][,PATH[:STRENGTH]...]"
            let lora: Vec<(&Path, f32)> = s("lora")
                .map(|l| {
                    l.split(',')
                        .filter(|x| !x.trim().is_empty())
                        .map(|x| {
                            let x = x.trim();
                            let (p, st) = x.rsplit_once(':').filter(|(_, st)| st.parse::<f32>().is_ok()).unwrap_or((x, "1"));
                            (Path::new(p), st.parse::<f32>().unwrap_or(1.0))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let c = ClipSpec { lora, ..c };
            generate(e, &c, Path::new(out), ctl)
        }
        "encode" => {
            let s = |k: &str| spec.get(k).and_then(|d| d.as_str());
            let prompt = match (s("prompt"), s("prompt_file")) {
                (Some(p), _) => p.to_string(),
                (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?.trim().to_string(),
                _ => return Err(Error("encode needs \"prompt\" or \"prompt_file\"".into())),
            };
            let te = s("te").ok_or("encode needs \"te\": the text encoder's GGUF file")?;
            let tokenizer = s("tokenizer").unwrap_or("/app/tokenizer");
            let pictures: Vec<&Path> = s("pictures").map(|r| r.split(',').filter(|x| !x.is_empty()).map(Path::new).collect()).unwrap_or_default();
            let files = TeFiles { te: Path::new(te), tokenizer: Path::new(tokenizer), visual: s("te_visual").map(Path::new), embeddings: s("embeddings").map(Path::new) };
            encode_with(&e.dev, e.threads, &pictures, &prompt, &files, s("out").map(Path::new), s("check").map(Path::new), ctl)
        }
        other => Err(Error(format!("unknown job kind {other:?} (known: bench-blocks, check-block, denoise, decode, encode, generate)"))),
    }
}
