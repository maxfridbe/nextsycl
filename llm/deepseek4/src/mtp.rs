//! DeepSeek-V4-Flash's draft block (MTP), which unsloth's GGUFs leave out: from the original checkpoint's last
//! safetensors shard (`mtp.0.*`) into a sidecar GGUF beside the model, in this engine's names (block 43). The experts'
//! FP4 is GGUF's MXFP4 (one E8M0 scale a 32 values; only the nibbles' order changes: a byte's low half value 2k and
//! high 2k + 1 there, value j and j + 16 here): lossless. The FP8 matrices (E4M3, an E8M0 scale a 128 x 128 block)
//! become Q8_0; the norms BF16 -> F32; the mHC parameters as they are. `prepare` runs it once after a pull (the
//! servers' model directory is read-only to them).

use std::path::Path;

use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_gguf::writer::Writer;
use nextsycl_gguf::{GType, Gguf, Value};

/// The draft block's sidecar, beside the model's first file
pub const SIDECAR: &str = "deepseek-v4-flash-mtp.gguf";
/// The original checkpoint's shard holding it, as the model card puts it beside the model
pub const SOURCE: &str = "DeepSeek-V4-Flash-00046-of-00046.safetensors";
/// the draft block's layer
const L: &str = "blk.43.";

fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let (e, m) = (((b >> 3) & 0xf) as i32, (b & 7) as f32);
    if e == 0 { s * (m / 8.0) * 2f32.powi(-6) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

fn e8m0(b: u8) -> f32 {
    2f32.powi(b as i32 - 127)
}

/// IEEE half from a float, rounding to nearest even (numpy's astype(float16))
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 0x1f {
        return sign | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rem > half || (rem == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e16 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

fn f16_value(h: u16) -> f32 {
    nextsycl_gguf::safetensors::f16_to_f32(h)
}

fn read(st: &SafeTensors, name: &str) -> Result<(Vec<u8>, Vec<u64>, String), String> {
    let t = st.need(name).map_err(|e| e.0)?;
    Ok((st.read(t).map_err(|e| e.0)?, t.shape.clone(), t.dtype.clone()))
}

/// An FP8 matrix with its block scales, as Q8_0 blocks (a row's 32 values: an fp16 amax / 127, rounded to even)
fn q8_0(st: &SafeTensors, name: &str) -> Result<(Vec<u8>, Vec<u64>), String> {
    let (w, shape, dt) = read(st, &format!("{name}.weight"))?;
    let (s, sshape, sdt) = read(st, &format!("{name}.scale"))?;
    if dt != "F8_E4M3" || sdt != "F8_E8M0" || shape.len() != 2 || sshape.len() != 2 {
        return Err(format!("{name}: {dt} with {sdt} scales, not an FP8 block matrix"));
    }
    let (r, c, sc) = (shape[0] as usize, shape[1] as usize, sshape[1] as usize);
    let lut: Vec<f32> = (0..=255u8).map(e4m3).collect();
    let mut out = Vec::with_capacity(r * c / 32 * 34);
    let mut row = vec![0f32; c];
    for i in 0..r {
        for (j, v) in row.iter_mut().enumerate() {
            *v = lut[w[i * c + j] as usize] * e8m0(s[(i / 128) * sc + j / 128]);
        }
        for b in row.chunks_exact(32) {
            let amax = b.iter().fold(0f32, |a, x| a.max(x.abs()));
            let d = f16_bits(amax / 127.0);
            let df = f16_value(d);
            out.extend(d.to_le_bytes());
            for x in b {
                let q = if df > 0.0 { (x / df).round_ties_even().clamp(-127.0, 127.0) } else { 0.0 };
                out.push(q as i8 as u8);
            }
        }
    }
    Ok((out, shape))
}

/// Packed FP4 [rows, cols / 2] with E8M0 scales [rows, cols / 32] as MXFP4 blocks
fn mxfp4(st: &SafeTensors, name: &str, out: &mut Vec<u8>) -> Result<Vec<u64>, String> {
    let (w, shape, dt) = read(st, &format!("{name}.weight"))?;
    let (s, _, sdt) = read(st, &format!("{name}.scale"))?;
    if dt != "I8" || sdt != "F8_E8M0" {
        return Err(format!("{name}: {dt} with {sdt} scales, not packed FP4"));
    }
    let (r, cb) = (shape[0] as usize, shape[1] as usize);
    let c = cb * 2;
    for i in 0..r {
        let wr = &w[i * cb..(i + 1) * cb];
        for b in 0..c / 32 {
            out.push(s[i * (c / 32) + b]);
            // value v of the block: byte (32 b + v) / 2's low (even v) or high (odd v) nibble
            let code = |v: usize| {
                let x = wr[(32 * b + v) / 2];
                if v.is_multiple_of(2) { x & 0xf } else { x >> 4 }
            };
            for j in 0..16 {
                out.push(code(j) | (code(j + 16) << 4));
            }
        }
    }
    Ok(vec![r as u64, c as u64])
}

/// `mtp.0.*` of the shard `src` into the sidecar `out` (written beside it as .part, then renamed)
pub fn convert(src: &Path, out: &Path, log: &mut dyn FnMut(String)) -> Result<(), String> {
    let st = SafeTensors::open(src).map_err(|e| e.0)?;
    let n_expert = (0..).take_while(|e| st.tensor(&format!("mtp.0.ffn.experts.{e}.w1.weight")).is_some()).count();
    if n_expert == 0 {
        return Err(format!("{}: no draft block (mtp.0.*) in it", src.display()));
    }
    let q8 = [("attn.wq_a", "attn_q_a"), ("attn.wq_b", "attn_q_b"), ("attn.wkv", "attn_kv"), ("attn.wo_a", "attn_output_a"), ("attn.wo_b", "attn_output_b"),
              ("ffn.shared_experts.w1", "ffn_gate_shexp"), ("ffn.shared_experts.w3", "ffn_up_shexp"), ("ffn.shared_experts.w2", "ffn_down_shexp"),
              ("e_proj", "nextn.e_proj"), ("h_proj", "nextn.h_proj")];
    let norms = [("attn.q_norm", "attn_q_a_norm"), ("attn.kv_norm", "attn_kv_a_norm"), ("attn_norm", "attn_norm"), ("ffn_norm", "ffn_norm"),
                 ("enorm", "nextn.enorm"), ("hnorm", "nextn.hnorm"), ("norm", "nextn.norm")];
    let f32s = [("attn.attn_sink", "attn_sinks.weight"), ("ffn.gate.bias", "exp_probs_b.bias"), ("hc_attn_base", "hc_attn_base.weight"),
                ("hc_attn_fn", "hc_attn_fn.weight"), ("hc_attn_scale", "hc_attn_scale.weight"), ("hc_ffn_base", "hc_ffn_base.weight"),
                ("hc_ffn_fn", "hc_ffn_fn.weight"), ("hc_ffn_scale", "hc_ffn_scale.weight"), ("hc_head_base", "nextn.hc_head_base.weight"),
                ("hc_head_fn", "nextn.hc_head_fn.weight"), ("hc_head_scale", "nextn.hc_head_scale.weight")];
    let experts = [("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"), ("w2", "ffn_down_exps")];
    // the directory first: every tensor's type and shape
    let shape_of = |name: &str| st.tensor(name).map(|t| t.shape.clone()).ok_or_else(|| format!("{name} is missing"));
    let mut dir: Vec<(String, GType, Vec<u64>)> = Vec::new();
    for (s, d) in q8 {
        dir.push((format!("{L}{d}.weight"), GType::Q8_0, shape_of(&format!("mtp.0.{s}.weight"))?));
    }
    for (s, d) in norms {
        dir.push((format!("{L}{d}.weight"), GType::F32, shape_of(&format!("mtp.0.{s}.weight"))?));
    }
    dir.push((format!("{L}ffn_gate_inp.weight"), GType::F32, shape_of("mtp.0.ffn.gate.weight")?));
    for (s, d) in f32s {
        dir.push((format!("{L}{d}"), GType::F32, shape_of(&format!("mtp.0.{s}"))?));
    }
    for (s, d) in experts {
        let sh = shape_of(&format!("mtp.0.ffn.experts.0.{s}.weight"))?;
        dir.push((format!("{L}{d}.weight"), GType::MXFP4, vec![n_expert as u64, sh[0], sh[1] * 2]));
    }
    let part = out.with_extension("gguf.part");
    let meta = [("general.architecture", Value::Str("deepseek4".into())), ("general.name", Value::Str("DeepSeek-V4-Flash MTP block (layer 43)".into()))];
    let mut w = Writer::create(&part, &meta, &dir).map_err(|e| e.0)?;
    for (s, _) in q8 {
        w.data(&q8_0(&st, &format!("mtp.0.{s}"))?.0).map_err(|e| e.0)?;
    }
    for (s, _) in norms {
        let (b, _, dt) = read(&st, &format!("mtp.0.{s}.weight"))?;
        if dt != "BF16" {
            return Err(format!("mtp.0.{s}: {dt}, not BF16"));
        }
        let v: Vec<u8> = b.chunks_exact(2).flat_map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect();
        w.data(&v).map_err(|e| e.0)?;
    }
    let (b, _, _) = read(&st, "mtp.0.ffn.gate.weight")?;
    let v: Vec<u8> = b.chunks_exact(2).flat_map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect();
    w.data(&v).map_err(|e| e.0)?;
    for (s, _) in f32s {
        let (b, _, dt) = read(&st, &format!("mtp.0.{s}"))?;
        if dt != "F32" {
            return Err(format!("mtp.0.{s}: {dt}, not F32"));
        }
        w.data(&b).map_err(|e| e.0)?;
    }
    for (s, d) in experts {
        let mut buf = Vec::new();
        for e in 0..n_expert {
            buf.clear();
            mxfp4(&st, &format!("mtp.0.ffn.experts.{e}.{s}"), &mut buf)?;
            w.data(&buf).map_err(|e| e.0)?;
        }
        log(format!("{d}: {n_expert} experts as MXFP4"));
    }
    w.finish().map_err(|e| e.0)?;
    std::fs::rename(&part, out).map_err(|e| format!("{}: {e}", out.display()))
}

/// After a pull (`nextsycl models pull`): the sidecar made from the shard beside the model, once
pub fn prepare(f: &Gguf) -> Result<(), String> {
    let Some(dir) = f.paths.first().and_then(|p| p.parent()) else { return Ok(()) };
    let (src, out) = (dir.join(SOURCE), dir.join(SIDECAR));
    if out.is_file() || !src.is_file() {
        return Ok(());
    }
    println!("  the draft block (MTP): {} -> {}", src.display(), out.display());
    convert(&src, &out, &mut |l| println!("  {l}"))
}
