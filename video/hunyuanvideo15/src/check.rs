//! Stage by stage against the reference (reference/hunyuanvideo15/ref.py: the official modules on the same files,
//! float32 on the CPU): each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   te     tokens.json             -> te_layer0.npy, embeds.npy (neg_*: the negative prompt's)
//!   byt5   byt5.json               -> byt5.npy
//!   dit    embeds + byt5 + noise   -> noise (the seed's draws), vec0, txt0 (the token refiner), txt_all0 (the text rows),
//!                                     dbl0_img / dbl0_txt (double block 0), dbl_img (after the 54), v0 (the velocity)
//!   steps                          -> latents_N.npy every step (Euler over sigmas.npy)
//!   i2v    siglip_pixels, cond_pixels -> vision.npy (SigLIP's tokens), cond_latent.npy (the VAE encoder's)
//!   vae    latents (latents.npy or the last step's) -> video.npy (RGB in [-1, 1])
//! A dit.json with "i2v" runs the I2V DiT on the reference's cond_latent and vision tokens.
//!
//! `bench` (no reference needed): the card's plain product rates (half, int8) and the attention at the 5 s clip's
//! length (spec "tokens", default 49,290 + 256), as TFLOPS.
//! ```
//!
//! The job: `nextsycl video job check --dir DIR --stages te,byt5,dit,steps,vae --here` (with the hunyuanvideo15 model).

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_hunyuan_text::{byt5, te};
use nextsycl_video::{Error, JobCtl, Result};
use serde_json::{json, Value};

use crate::{dit, sigmas, HunyuanVideo15};

/// A little-endian float32 .npy: (shape, values)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if !head.contains("'<f4'") || head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: {head} (float32, C order expected)", path.display())));
    }
    let shape: Vec<usize> = head
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let v = b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((shape, v))
}

/// A float32 .npy of `v` with `shape`
pub fn write_npy(path: &Path, shape: &[usize], v: &[f32]) -> Result<()> {
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let mut head = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}{}), }}", if shape.len() == 1 { "," } else { "" });
    while !(10 + head.len() + 1).is_multiple_of(64) {
        head.push(' ');
    }
    head.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((head.len() as u16).to_le_bytes());
    b.extend(head.as_bytes());
    for x in v {
        b.extend(x.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
}

/// How far `got` is from `want`: (relative RMS error, max abs error, cosine)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
        mx = mx.max((g - w).abs());
    }
    ((e2 / w2.max(1e-30)).sqrt(), mx, dot / (g2.sqrt() * w2.sqrt()).max(1e-30))
}

struct Report<'a> {
    log: &'a mut dyn FnMut(String),
    rows: Vec<Value>,
    worst: f64,
}

impl Report<'_> {
    fn line(&mut self, what: &str, got: &[f32], want: &[f32]) {
        if got.len() != want.len() {
            (self.log)(format!("{what:<16} {} values here, {} in the reference", got.len(), want.len()));
            self.worst = f64::INFINITY;
            self.rows.push(json!({"stage": what, "error": "sizes differ"}));
            return;
        }
        let (rel, mx, cos) = compare(got, want);
        (self.log)(format!("{what:<16} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.6}  ({} values)", want.len()));
        self.worst = self.worst.max(rel);
        self.rows.push(json!({"stage": what, "rel_rms": rel, "max_abs": mx, "cosine": cos, "values": want.len()}));
    }
}

fn json_file(path: &Path) -> Result<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?)
        .map_err(|e| Error(format!("{}: {e}", path.display())))
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default()
}

/// The check job: {"dir": DIR, "stages": "te,byt5,dit,steps,vae"}
fn bytemuck_u16(v: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding; the byte view covers the same memory.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 2) }
}

/// The yardsticks: plain products and the attention, timed
fn bench(m: &HunyuanVideo15, spec: &Value, log: &mut dyn FnMut(String)) -> Result<()> {
    let iters = spec.get("iters").and_then(Value::as_u64).unwrap_or(10).clamp(1, 100) as usize;
    for int8 in if spec.get("attention_only").and_then(Value::as_bool).unwrap_or(false) { vec![] } else { vec![false, true] } {
        for (mm, n, k) in [(8192, 8192, 8192), (16384, 16384, 8192), (49290, 6144, 2048), (49290, 8192, 2048), (49290, 2048, 8192)] {
            let ms = m.k.bench_gemm(int8, mm, n, k, iters)?;
            log(format!("[bench gemm {} {mm} x {n} x {k}: {ms:.3} ms, {:.1} T{}]", if int8 { "int8" } else { "half" },
                        2.0 * (mm * n * k) as f64 / ms / 1e9, if int8 { "OPS" } else { "FLOPS" }));
        }
    }
    // the attention on values like the DiT's (q and k unit-RMS after their norm, a spread of scales; v as it comes):
    // each kernel timed, and its result against the shared kernel's (relative RMS, cosine)
    for s in spec.get("tokens").and_then(Value::as_str).unwrap_or("4096,49546").split(',').filter_map(|x| x.trim().parse::<usize>().ok()) {
        let s = s.clamp(64, 200_000);
        let (h, d) = (dit::HEADS, dit::HEAD);
        let qkv = DevBuf::new(&m.nsd.gpu, s * 3 * h * d * 2)?;
        let out = DevBuf::new(&m.nsd.gpu, s * h * d * 2)?;
        let mut st = 0x2545_f491_4f6c_dd1du64;
        let mut host = vec![0u16; s * 3 * h * d];
        for (i, x) in host.iter_mut().enumerate() {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            // roughly normal (a sum of four uniforms), scaled: q and k x 2.5 (sharp attention, as after the norm's gain), v x 1
            let u: f32 = (0..4).map(|j| ((st >> (j * 16)) & 0xffff) as f32 / 65535.0 - 0.5).sum::<f32>() * 1.7;
            let col = i % (3 * h * d);
            *x = crate::w::f32_to_f16(if col < 2 * h * d { u * 2.5 } else { u });
        }
        qkv.write(0, bytemuck_u16(&host))?;
        drop(host);
        let p = qkv.ptr() as *const u16;
        let read = |b: &DevBuf| -> Result<Vec<f32>> {
            let mut raw = vec![0u8; s * h * d * 2];
            b.read(0, &mut raw)?;
            Ok(raw.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
        };
        let flops = 4.0 * (s * s * d * h) as f64;
        let time = |name: String, f: &dyn Fn() -> Result<()>, log: &mut dyn FnMut(String)| -> Result<()> {
            f()?;
            m.nsd.wait()?;
            let t = std::time::Instant::now();
            let n = iters.min(5);
            for _ in 0..n {
                f()?;
            }
            m.nsd.wait()?;
            let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            log(format!("[bench attention {name} {s} tokens: {ms:.2} ms, {:.1} TFLOPS (4 S^2 D H)]", flops / ms / 1e9));
            Ok(())
        };
        let shared = || m.nsd.attention_qk(p.cast(), s, 3 * h * d, p.wrapping_add(h * d).cast(), p.wrapping_add(2 * h * d).cast(), s, 3 * h * d,
                                           nextsycl_diffusion::kernels::Dt::F16, h, d, out.ptr().cast(), nextsycl_diffusion::kernels::Dt::F16);
        time("shared".into(), &shared, log)?;
        let want = read(&out)?;
        for tile in 0..m.k.attention_tiles() {
            let f = || m.k.attention(qkv.ptr(), s, h, d, 3 * h * d, out.ptr(), h * d, false, tile);
            time(format!("half tile {tile}"), &f, log)?;
            let (rel, mx, cos) = compare(&read(&out)?, &want);
            log(format!("[bench attention half tile {tile} vs shared: rel {rel:.3e} max {mx:.3e} cos {cos:.8}]"));
        }
        if m.k.attention_tiles() > 0 {
            let f = || m.k.attention(qkv.ptr(), s, h, d, 3 * h * d, out.ptr(), h * d, true, 0);
            time("sage".into(), &f, log)?;
            let (rel, mx, cos) = compare(&read(&out)?, &want);
            log(format!("[bench attention sage vs shared: rel {rel:.3e} max {mx:.3e} cos {cos:.8}]"));
        }
    }
    Ok(())
}

pub fn job(m: &HunyuanVideo15, spec: &Value, ctl: &mut JobCtl) -> Result<Value> {
    let dir = Path::new(spec.get("dir").and_then(Value::as_str).ok_or_else(|| Error("check: no dir".into()))?).to_path_buf();
    let stages: Vec<String> = spec.get("stages").and_then(Value::as_str).unwrap_or("te,byt5,dit,vae").split(',').map(|s| s.trim().to_string()).collect();
    let has = |s: &str| stages.iter().any(|x| x == s);
    let mut r = Report { log: &mut *ctl.log, rows: Vec::new(), worst: 0.0 };
    let nsd = &m.nsd;
    if has("bench") {
        bench(m, spec, &mut *r.log)?;
    }
    if has("te") {
        let toks = json_file(&dir.join("tokens.json"))?;
        let crop = toks["crop"].as_u64().unwrap_or(0) as usize;
        (r.log)(format!("crop: {} here, {crop} in the reference", m.te.template.crop));
        if crop != m.te.template.crop {
            r.worst = f64::INFINITY;
        }
        for (p, key) in [("", "ids"), ("neg_", "neg_ids")] {
            let want = ids(&toks[key]);
            if want.is_empty() {
                continue;
            }
            let prompt = toks[if p.is_empty() { "prompt" } else { "negative" }].as_str().unwrap_or("");
            let mine = m.te.ids(prompt);
            (r.log)(format!("{p}tokens: {} here, {} in the reference{}", mine.len(), want.len(), if mine == want { " - the same" } else { " - DIFFERENT" }));
            if mine != want {
                r.worst = f64::INFINITY;
            }
            let l0 = m.te.run(nsd, &want, 1)?.to_f32()?;
            r.line(&format!("{p}te layer 0"), &l0, &read_npy(&dir.join(format!("{p}te_layer0.npy")))?.1);
            let all = m.te.run(nsd, &want, te::LAYERS_USED)?.to_f32()?;
            r.line(&format!("{p}te embeds"), &all[crop * te::DIM..], &read_npy(&dir.join(format!("{p}embeds.npy")))?.1);
        }
    }
    if has("byt5") {
        let b = json_file(&dir.join("byt5.json"))?;
        let text = b["text"].as_str().map(str::to_string);
        if let Some(p) = json_file(&dir.join("tokens.json")).ok().and_then(|t| t["prompt"].as_str().map(str::to_string)) {
            let mine = byt5::quoted(&p, byt5::Quotes::InOrder);
            (r.log)(format!("glyph text: {mine:?} here, {text:?} in the reference"));
            if mine != text {
                r.worst = f64::INFINITY;
            }
        }
        if let Some(t) = text {
            let want = ids(&b["ids"]);
            if byt5::ids(&t, 256) != want {
                (r.log)("byt5 ids differ".into());
                r.worst = f64::INFINITY;
            }
            let got = m.byt5.encode(nsd, &want)?;
            r.line("byt5", &got, &read_npy(&dir.join("byt5.npy"))?.1);
        }
    }
    if has("i2v") {
        let (_, sp) = read_npy(&dir.join("siglip_pixels.npy"))?;
        let v = m.vision.as_ref().ok_or_else(|| Error("check i2v: the model has no vision file".into()))?;
        let t0 = std::time::Instant::now();
        let tok = v.encode(nsd, &sp)?;
        (r.log)(format!("siglip: {:.2} s", t0.elapsed().as_secs_f64()));
        r.line("vision", &tok, &read_npy(&dir.join("vision.npy"))?.1);
        let (shape, cp) = read_npy(&dir.join("cond_pixels.npy"))?;
        let t1 = std::time::Instant::now();
        let lat = m.vae.encode(nsd, &cp, shape[0], shape[1])?;
        (r.log)(format!("vae encode: {:.2} s", t1.elapsed().as_secs_f64()));
        r.line("cond latent", &lat, &read_npy(&dir.join("cond_latent.npy"))?.1);
    }
    if has("dit") || has("steps") {
        let d = json_file(&dir.join("dit.json"))?;
        let lat = d["latent"].as_array().ok_or_else(|| Error("dit.json: no latent".into()))?;
        let grid = (lat[0].as_u64().unwrap_or(1) as usize, lat[1].as_u64().unwrap_or(1) as usize, lat[2].as_u64().unwrap_or(1) as usize);
        let steps = d["steps"].as_u64().unwrap_or(1) as usize;
        let shift = d["shift"].as_f64().unwrap_or(5.0) as f32;
        let cfg = d["cfg"].as_f64().map(|x| x as f32);
        let seed = d["seed"].as_u64().unwrap_or(0);
        let n = grid.0 * grid.1 * grid.2;
        let (_, noise) = read_npy(&dir.join("noise.npy"))?;
        r.line("noise (seed)", &crate::noise(seed, grid.0, grid.1, grid.2), &noise);
        let sig = sigmas(steps, shift);
        r.line("sigmas", &sig, &read_npy(&dir.join("sigmas.npy"))?.1);
        let i2v = d["i2v"].as_bool().unwrap_or(false);
        let vision = if i2v { Some(read_npy(&dir.join("vision.npy"))?.1) } else { None };
        let text = |p: &str| -> Result<dit::Text> {
            let (_, e) = read_npy(&dir.join(format!("{p}embeds.npy")))?;
            let b5 = dir.join(format!("{p}byt5.npy"));
            let b = if b5.exists() { Some(read_npy(&b5)?.1) } else { None };
            m.dit.text(nsd, &e, b.as_deref(), vision.as_deref())
        };
        let pos = text("")?;
        let neg = match cfg {
            Some(_) => Some(text("neg_")?),
            None => None,
        };
        let w = dit::IN - dit::LATENT;
        let mut c = vec![0f32; n * w];
        if i2v {
            // the first frame's latent and the mask on frame 0's rows
            let (_, cl) = read_npy(&dir.join("cond_latent.npy"))?;
            for (r, row) in cl.chunks_exact(dit::LATENT).enumerate() {
                c[r * w..r * w + dit::LATENT].copy_from_slice(row);
                c[r * w + dit::LATENT] = 1.0;
            }
        }
        let cond = DevBuf::from_f32(&nsd.gpu, &c)?;
        let ts: Vec<(f32, Option<f32>)> =
            (0..steps).map(|i| (sig[i] * 1000.0, m.dit.meanflow.then(|| if i + 1 < steps { sig[i + 1] * 1000.0 } else { 0.0 }))).collect();
        let st = m.dit.steps(nsd, &ts)?;
        r.line("vec0", &st.vec.to_f32()?[..dit::DIM], &read_npy(&dir.join("vec0.npy"))?.1);
        let mut texts = vec![&pos];
        if let Some(n) = &neg {
            texts.push(n);
        }
        let bt = m.dit.batch(nsd, &texts, grid)?;
        let mut lat = DevBuf::from_f32(&nsd.gpu, &noise)?;
        let t0 = std::time::Instant::now();
        let last = if has("steps") { steps } else { 1 };
        for i in 0..last {
            let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
            let lats: Vec<&DevBuf> = texts.iter().map(|_| &lat).collect();
            let v = m.dit.forward(nsd, &st, i, &bt, &lats, &cond, &mut |name, buf, r0, rows| {
                if i == 0 {
                    let all = buf.to_f32()?;
                    taps.push((name.to_string(), all[r0 * dit::DIM..(r0 + rows) * dit::DIM].to_vec()));
                }
                Ok(())
            })?;
            for (name, got) in &taps {
                r.line(name, got, &read_npy(&dir.join(format!("{name}.npy")))?.1);
            }
            let mut v0 = v[0].to_f32()?;
            if let (Some(scale), Some(vn)) = (cfg, v.get(1)) {
                let vn = vn.to_f32()?;
                v0 = v0.iter().zip(&vn).map(|(c, u)| u + scale * (c - u)).collect();
            }
            if i == 0 {
                r.line("v0", &v0, &read_npy(&dir.join("v0.npy"))?.1);
            }
            let mut l = lat.to_f32()?;
            for (a, b) in l.iter_mut().zip(&v0) {
                *a += (sig[i + 1] - sig[i]) * b;
            }
            lat = DevBuf::from_f32(&nsd.gpu, &l)?;
            let p = dir.join(format!("latents_{i}.npy"));
            if has("steps") && p.exists() {
                r.line(&format!("latents {i}"), &l, &read_npy(&p)?.1);
            }
            if has("steps") && i + 1 == steps {
                write_npy(&dir.join("port_latents.npy"), &[n, dit::LATENT], &l)?;
            }
        }
        (r.log)(format!("dit: {last} pass(es) in {:.1} s", t0.elapsed().as_secs_f64()));
    }
    if has("vae") {
        let src = spec.get("latents").and_then(Value::as_str).map(Path::new).map(Path::to_path_buf);
        let (grid, lat) = match src {
            Some(p) => {
                let f = spec.get("frames").and_then(Value::as_u64).unwrap_or(9) as usize;
                let h = spec.get("height").and_then(Value::as_u64).unwrap_or(128) as usize;
                let w = spec.get("width").and_then(Value::as_u64).unwrap_or(224) as usize;
                (((f - 1) / 4 + 1, h / 16, w / 16), read_npy(&p)?.1)
            }
            None => {
                let d = json_file(&dir.join("dit.json"))?;
                let lat = d["latent"].as_array().ok_or_else(|| Error("dit.json: no latent".into()))?;
                let steps = d["steps"].as_u64().unwrap_or(1);
                ((lat[0].as_u64().unwrap_or(1) as usize, lat[1].as_u64().unwrap_or(1) as usize, lat[2].as_u64().unwrap_or(1) as usize),
                 read_npy(&dir.join(format!("latents_{}.npy", steps - 1)))?.1)
            }
        };
        let t0 = std::time::Instant::now();
        let log = &mut *r.log;
        let (v, to, h, w) = m.vae.decode(nsd, &lat, grid.0, grid.1, grid.2, &mut |s| log(format!("decode: {s}")))?;
        (r.log)(format!("vae: {to} frames {w}x{h} in {:.1} s", t0.elapsed().as_secs_f64()));
        r.line("video", &v, &read_npy(&dir.join("video.npy"))?.1);
    }
    if has("stream") {
        // the decoder as a stream (MovieRoll's chunked decode): the latents in chunks - 1 frame, then `chunk` at a time
        // - against the whole video's decode (the reference's, untiled: small sizes)
        let d = json_file(&dir.join("dit.json"))?;
        let lat = d["latent"].as_array().ok_or_else(|| Error("dit.json: no latent".into()))?;
        let (f, h, w) = (lat[0].as_u64().unwrap_or(1) as usize, lat[1].as_u64().unwrap_or(1) as usize, lat[2].as_u64().unwrap_or(1) as usize);
        let steps = d["steps"].as_u64().unwrap_or(1);
        let (_, z) = read_npy(&dir.join(format!("latents_{}.npy", steps - 1)))?;
        let chunk = spec.get("chunk").and_then(Value::as_f64).unwrap_or(1.0).max(1.0) as usize;
        let plane = h * w * crate::vae::Z;
        m.vae.stream_begin()?;
        let mut out = Vec::new();
        let mut at = 0;
        let t0 = std::time::Instant::now();
        while at < f {
            let n = if at == 0 { 1 } else { chunk.min(f - at) };
            let (v, to) = m.vae.stream_chunk(nsd, &z[at * plane..(at + n) * plane], n, h, w)?;
            (r.log)(format!("stream: latent frames {at}..{} -> {to} frames", at + n));
            out.extend(v);
            at += n;
        }
        m.vae.stream_end()?;
        (r.log)(format!("stream: {f} latent frames in {:.1} s", t0.elapsed().as_secs_f64()));
        r.line(&format!("stream (1 + {chunk}s)"), &out, &read_npy(&dir.join("video.npy"))?.1);
    }
    (r.log)(format!("worst relative error {:.2e}", r.worst));
    Ok(json!({"worst_rel_rms": if r.worst.is_finite() { json!(r.worst) } else { json!("inf") }, "stages": r.rows}))
}
