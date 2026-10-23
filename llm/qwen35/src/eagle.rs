//! The EAGLE-4 draft for Qwen3.8-27B (`LlamaForCausalLMEagle4`, a folder of config.json and model.safetensors): one
//! Llama layer over the target's features. A pass's rows feed it the target's residual stream after three of its
//! layers (`eagle_aux_hidden_state_layer_ids`: layer outputs), each RMS-normed, through `fc` (3H -> H): the feature of
//! a position. Row i of the draft reads token i (the target's embedding) with the feature of position i - 1:
//!
//! ```text
//!   x = [input_layernorm(embed(token)) | hidden_norm(feature)]      (2H)
//!   r = feature + o(attention(rope(q x), rope(k x), v x))           (40 heads of 128, a sliding window of 4096)
//!   s = r + down(silu(gate n) * up n),  n = post_attention_layernorm(r)
//!   out = norm(s): the head reads it, and the next draft step takes it as its feature (norm_output)
//! ```
//!
//! The weights come as bfloat16; they are kept as Q8_0 (made on the host at load), the products' columns split where
//! the input is a concatenation (q / k / v: the embedding's half and the feature's; fc: its three aux states). A draft
//! token comes from the low-rank head as trained: rank-128 scores over the vocabulary (`efficient_lm_head` .
//! `efficient_lm_head_down_proj` . out), their top 128 scored again with the full head's rows, the best of those -
//! 35 MB read a draft step, not the full head's 1.35 GB (NS_Q35_EAGLE_HEAD=full: the full head).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_gguf::GType;
use nextsycl_gguf::safetensors::{SafeTensors, StTensor};
use nextsycl_llm::{Error, Result};

use crate::engine::Mat;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

pub struct Eagle {
    /// the target layers whose outputs it reads
    pub ids: [usize; 3],
    pub fc_norm: [DevBuf; 3],
    /// fc's columns for each aux state
    pub fc: [Mat; 3],
    pub in_norm: DevBuf,
    pub hidden_norm: DevBuf,
    /// q / k / v: the columns for the normed embedding, then for the normed feature
    pub q: [Mat; 2],
    pub k: [Mat; 2],
    pub v: [Mat; 2],
    pub o: Mat,
    pub post_norm: DevBuf,
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
    pub norm: DevBuf,
    pub head: Mat,
    /// the low-rank head: [rank, H] (Q8_0), and [vocab, rank] as int8 rows [vocab][128] with their scales apart
    /// ([vocab][4] half: blocks of 32) - aligned loads; its candidates a step
    pub lr_down: Mat,
    pub lr_q: DevBuf,
    pub lr_d: DevBuf,
    pub cands: usize,
    /// NS_Q35_EAGLE_HEAD=full: the full head
    pub full_head: bool,
    pub heads: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub theta: f32,
    pub eps: f32,
    /// keys a query sees (itself the last); 0: all
    pub window: usize,
    /// positions its cache holds at most (NS_Q35_EAGLE_CTX): past them it does not draft
    pub ctx: usize,
    pub bytes: u64,
}

/// NS_Q35_EAGLE: the draft's folder (config.json, model.safetensors), or `auto` for an eagle4/ folder beside the model's
/// file; unset: none. Opt-in: on the B70 it drafts no better than MTP (2026-10-23, Q4, 512 greedy tokens, 3 drafts: code
/// 57.6 / 55.8 tok/s, prose 43.1 / 42.8, a 14K prompt read 960 / 1,111 tok/s), and it has no drafts past its cache
/// and no batch steps
pub fn path(model: &Path) -> Option<PathBuf> {
    match std::env::var("NS_Q35_EAGLE").as_deref() {
        Ok("auto") => {
            let d = model.parent()?.join("eagle4");
            d.join("model.safetensors").is_file().then_some(d)
        }
        Ok(v) if !v.is_empty() && v != "0" => Some(v.into()),
        _ => None,
    }
}

/// NS_Q35_EAGLE_CTX: the draft cache's positions (default 32768; a position takes 20 KiB)
fn ctx_setting() -> usize {
    std::env::var("NS_Q35_EAGLE_CTX").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= 256).unwrap_or(32768)
}

/// Its weights' bytes on the GPU (Q8_0, the norms float32), from the file's header
pub fn weight_bytes(dir: &Path) -> Result<u64> {
    let st = SafeTensors::open(&dir.join("model.safetensors")).map_err(|x| e(x.0))?;
    Ok(st.tensors.iter().filter(|t| !t.name.starts_with("embed_tokens"))
        .map(|t| if t.shape.len() == 1 { t.elements() * 4 } else { t.elements() / 32 * 34 }).sum())
}

impl Eagle {
    /// Loads the draft onto `gpu`; `n_embd` / `layers` / `vocab`: the target's (checked against the config)
    pub fn load(dir: &Path, gpu: &Arc<Gpu>, n_embd: usize, layers: usize, vocab: usize) -> Result<Eagle> {
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(|x| e(format!("{}: {x}", dir.display())))?)
            .map_err(|x| e(format!("{}/config.json: {x}", dir.display())))?;
        let num = |k: &str| cfg[k].as_u64().map(|v| v as usize).ok_or_else(|| e(format!("eagle config: no {k}")));
        let arch = cfg["architectures"][0].as_str().unwrap_or("");
        if arch != "LlamaForCausalLMEagle4" {
            return Err(e(format!("eagle: {arch}, not LlamaForCausalLMEagle4")));
        }
        let h = num("hidden_size")?;
        if h != n_embd || num("target_hidden_size").unwrap_or(h) != n_embd || num("num_hidden_layers")? != 1 {
            return Err(e(format!("eagle: hidden {h} over a target of {n_embd}, or more than one layer")));
        }
        if cfg["fc_norm"].as_bool() != Some(true) || cfg["norm_output"].as_bool() != Some(true) {
            return Err(e("eagle: a draft without fc_norm and norm_output"));
        }
        let ids: Vec<usize> = cfg["eagle_config"]["eagle_aux_hidden_state_layer_ids"].as_array().into_iter().flatten()
            .filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        if ids.len() != 3 || ids.iter().any(|l| *l >= layers) {
            return Err(e(format!("eagle: aux layers {ids:?} (three of the target's {layers})")));
        }
        let (heads, head_dim) = (num("num_attention_heads")?, num("head_dim")?);
        if num("num_key_value_heads")? != heads || heads * head_dim != h || head_dim != 128 {
            return Err(e("eagle: an attention other than one key head a query head of 128"));
        }
        let use_window = cfg["use_sliding_window"].as_bool().unwrap_or(false)
            || cfg["layer_types"][0].as_str() == Some("sliding_attention");
        let window = if use_window { num("sliding_window").unwrap_or(0) } else { 0 };
        let theta = cfg["rope_parameters"]["rope_theta"].as_f64().or_else(|| cfg["rope_theta"].as_f64()).unwrap_or(1e4) as f32;
        let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let n_ff = num("intermediate_size")?;
        let st = SafeTensors::open(&dir.join("model.safetensors")).map_err(|x| e(x.0))?;
        let mut bytes = 0u64;
        let mut vecf = |name: &str| -> Result<DevBuf> {
            let v = st.f32(name).map_err(|x| e(x.0))?;
            let b = DevBuf::f32(gpu, v.len())?;
            b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            bytes += b.len as u64;
            Ok(b)
        };
        let fc_norm = [vecf("fc_norm.0.weight")?, vecf("fc_norm.1.weight")?, vecf("fc_norm.2.weight")?];
        let in_norm = vecf("layers.0.input_layernorm.weight")?;
        let hidden_norm = vecf("layers.0.hidden_norm.weight")?;
        let post_norm = vecf("layers.0.post_attention_layernorm.weight")?;
        let norm = vecf("norm.weight")?;
        let mut mat = |name: &str, c0: usize, nc: Option<usize>| -> Result<Mat> {
            let m = q8_mat(&st, st.need(name).map_err(|x| e(x.0))?, c0, nc, gpu)?;
            bytes += m.buf.len as u64;
            Ok(m)
        };
        let fc = [mat("fc.weight", 0, Some(h))?, mat("fc.weight", h, Some(h))?, mat("fc.weight", 2 * h, Some(h))?];
        let a = "layers.0.self_attn.";
        let q = [mat(&format!("{a}q_proj.weight"), 0, Some(h))?, mat(&format!("{a}q_proj.weight"), h, Some(h))?];
        let k = [mat(&format!("{a}k_proj.weight"), 0, Some(h))?, mat(&format!("{a}k_proj.weight"), h, Some(h))?];
        let v = [mat(&format!("{a}v_proj.weight"), 0, Some(h))?, mat(&format!("{a}v_proj.weight"), h, Some(h))?];
        let o = mat(&format!("{a}o_proj.weight"), 0, None)?;
        let gate = mat("layers.0.mlp.gate_proj.weight", 0, None)?;
        let up = mat("layers.0.mlp.up_proj.weight", 0, None)?;
        let down = mat("layers.0.mlp.down_proj.weight", 0, None)?;
        let head = mat("lm_head.weight", 0, None)?;
        let lr_down = mat("efficient_lm_head_down_proj.weight", 0, None)?;
        let lt = st.need("efficient_lm_head.weight").map_err(|x| e(x.0))?;
        let cands = cfg["efficient_lm_head_config"]["num_candidates"].as_u64().unwrap_or(128) as usize;
        if lt.shape != [head.rows as u64, 128] || lr_down.rows != 128 || lr_down.cols != h || cands == 0 || cands > head.rows {
            return Err(e("eagle: a low-rank head of other shapes (rank 128)"));
        }
        let (lr_q, lr_d) = i8_rows(&st, lt, gpu)?;
        bytes += (lr_q.len + lr_d.len) as u64;
        if head.rows != vocab {
            return Err(e(format!("eagle: a head of {} rows, a target vocabulary of {vocab}", head.rows)));
        }
        gpu.sync()?;
        Ok(Eagle { ids: [ids[0], ids[1], ids[2]], fc_norm, fc, in_norm, hidden_norm, q, k, v, o, post_norm, gate, up, down, norm, head, lr_down, lr_q, lr_d, cands,
                   full_head: std::env::var("NS_Q35_EAGLE_HEAD").is_ok_and(|v| v == "full"), heads, head_dim,
                   n_ff, theta, eps, window, ctx: ctx_setting(), bytes })
    }
}

/// A bfloat16 matrix [rows, 128] as int8 rows and their blocks' half scales apart ([rows][4]): Q8_0's values, aligned
fn i8_rows(st: &SafeTensors, t: &StTensor, gpu: &Arc<Gpu>) -> Result<(DevBuf, DevBuf)> {
    let rows = t.shape[0] as usize;
    let raw = st.read(t).map_err(|x| e(x.0))?;
    let mut q = vec![0u8; rows * 128];
    let mut d = vec![0u8; rows * 8];
    for (r, row) in raw.chunks_exact(256).enumerate() {
        for b in 0..4 {
            let x: Vec<f32> = row[b * 64..(b + 1) * 64].chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
            let s = x.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            let id = if s > 0.0 { 1.0 / s } else { 0.0 };
            d[r * 8 + b * 2..r * 8 + b * 2 + 2].copy_from_slice(&crate::vision::f32_to_f16(s).to_le_bytes());
            for (j, v) in x.iter().enumerate() {
                q[r * 128 + b * 32 + j] = (v * id).round().clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    }
    let (qb, db) = (DevBuf::new(gpu, q.len())?, DevBuf::new(gpu, d.len())?);
    qb.write(0, &q)?;
    db.write(0, &d)?;
    Ok((qb, db))
}

/// A bfloat16 matrix [rows, cols] (its columns [c0, c0 + nc)) as Q8_0 blocks on the GPU: quantized on the host, a
/// band of rows at a time over the host's threads
fn q8_mat(st: &SafeTensors, t: &StTensor, c0: usize, nc: Option<usize>, gpu: &Arc<Gpu>) -> Result<Mat> {
    if t.dtype != "BF16" || t.shape.len() != 2 {
        return Err(e(format!("{}: {} {:?}, not a bfloat16 matrix", t.name, t.dtype, t.shape)));
    }
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
    let nc = nc.unwrap_or(cols - c0);
    if c0 + nc > cols || !nc.is_multiple_of(32) {
        return Err(e(format!("{}: columns {c0}..{} of {cols}", t.name, c0 + nc)));
    }
    let rb = nc / 32 * 34;
    let buf = DevBuf::new(gpu, rows * rb)?;
    let band = (16 << 20) / (cols * 2).max(1);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let mut raw = vec![0u8; band * cols * 2];
    let mut out = vec![0u8; band * rb];
    let mut r0 = 0;
    while r0 < rows {
        let n = band.min(rows - r0);
        st.read_into(t, (r0 * cols * 2) as u64, &mut raw[..n * cols * 2]).map_err(|x| e(x.0))?;
        let per = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (oc, ic) in out[..n * rb].chunks_mut(per * rb).zip(raw[..n * cols * 2].chunks(per * cols * 2)) {
                s.spawn(move || {
                    for (o, i) in oc.chunks_exact_mut(rb).zip(ic.chunks_exact(cols * 2)) {
                        let row = &i[c0 * 2..(c0 + nc) * 2];
                        for (ob, ib) in o.chunks_exact_mut(34).zip(row.chunks_exact(64)) {
                            let x: Vec<f32> = ib.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
                            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                            let d = amax / 127.0;
                            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                            ob[..2].copy_from_slice(&crate::vision::f32_to_f16(d).to_le_bytes());
                            for (j, v) in x.iter().enumerate() {
                                ob[2 + j] = (v * id).round().clamp(-127.0, 127.0) as i8 as u8;
                            }
                        }
                    }
                });
            }
        });
        buf.write(r0 * rb, &out[..n * rb])?;
        r0 += n;
    }
    Ok(Mat { buf, ty: GType::Q8_0, rows, cols: nc })
}
