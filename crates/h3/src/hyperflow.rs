//! HyperFlow's two-time conditioning (Video-Rebirth's 8-step adapter, `minimax_h3_hyperflow_8step_v1.0`), as
//! Open-H3-Regenerate-2K-IC runs it (infer/ref2va_ic/hyperflow.py): the timestep embedding of a step is
//!
//! ```text
//!   e(t, r) = e_t(t) + gate * (e_r(r) - e_t(t))       e_t: the time embedder with the adapter's LoRA on it
//!                                                      e_r: a copy of it with its own LoRA (the "endpoint")
//! ```
//!
//! with r = 1 - the next sigma for the timestep the generated rows run at, r = t for every other one. The int8 checkpoint
//! we run has no time embedder: ComfyUI's pruned form keeps `silu(e(t))` as 8 coordinates on a curve (`adaln_t_table`)
//! and every AdaLN projection on those 8; a two-time embedding is off that curve (3% of the modulation lost in
//! projecting it back; reference/regen2k/timecurve.py), so the tables of a run are made from the full projections:
//! the time embedder and the blocks' and the final layer's AdaLN weights taken from ComfyUI's bf16 Ref2VA checkpoint
//! (only those tensors: a card file with a `tensors` filter). Once per run (`prepare`): every (t, r) the sampler will
//! ask for, each block's modulation for each, read through the 26 GB of projections once (host, threads); the net then
//! takes its tables from there (`tables_at`). LMS Sharp's AdaLN branches (on the curve at t) are added to the same
//! tables.

use std::collections::BTreeMap;

use crate::dit::host;
use crate::dtype::{bf16_to_f32, DType};
use crate::safetensors::Checkpoint;
use crate::{Ctx, Error, Result};

/// The time embedder's input: sinusoids of t (Diffusers' Timesteps(256, flip_sin_to_cos=True, shift 0))
const FREQ_DIM: usize = 256;

/// One MLP: proj_in [hidden, 256] -> SiLU -> proj_out [t_dim, hidden], float32
struct Embedder {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    hidden: usize,
    out: usize,
}

impl Embedder {
    fn run(&self, t: f32) -> Vec<f32> {
        let half = FREQ_DIM / 2;
        let mut x = vec![0f32; FREQ_DIM];
        for i in 0..half {
            let f = (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32;
            let a = t * f;
            // flip_sin_to_cos: [cos | sin]
            x[i] = a.cos();
            x[half + i] = a.sin();
        }
        let h: Vec<f32> = (0..self.hidden).map(|o| {
            let v = self.b1[o] + self.w1[o * FREQ_DIM..(o + 1) * FREQ_DIM].iter().zip(&x).map(|(a, b)| a * b).sum::<f32>();
            v / (1.0 + (-v).exp())
        }).collect();
        (0..self.out).map(|o| self.b2[o] + self.w2[o * self.hidden..(o + 1) * self.hidden].iter().zip(&h).map(|(a, b)| a * b).sum::<f32>()).collect()
    }
}

/// `w += scale * B A` for a LoRA pair of the adapter file (A [r, in], B [out, r], float32)
fn merge(w: &mut [f32], cols: usize, lora: &Checkpoint, key: &str, scale: f32) -> Result<()> {
    let (ka, kb) = (format!("{key}.lora_A.weight"), format!("{key}.lora_B.weight"));
    let (a, b) = (host(lora, &ka)?, host(lora, &kb)?);
    let r = lora.get(&ka)?.shape[0];
    let rows = w.len() / cols;
    if a.len() != r * cols || b.len() != rows * r {
        return Err(Error(format!("{key}: a LoRA of {:?} / {:?} for a [{rows}, {cols}] matrix", lora.get(&ka)?.shape, lora.get(&kb)?.shape)));
    }
    for o in 0..rows {
        for k in 0..r {
            let s = b[o * r + k] * scale;
            if s != 0.0 {
                w[o * cols..(o + 1) * cols].iter_mut().zip(&a[k * cols..(k + 1) * cols]).for_each(|(x, y)| *x += s * y);
            }
        }
    }
    Ok(())
}

/// The adapter's time conditioning with the full AdaLN projections it needs
pub struct TwoTime {
    base: Embedder,
    endpoint: Embedder,
    gate: f32,
    /// ComfyUI's bf16 Ref2VA tensors: time_embedder.*, blocks.N.adaln_proj.linear.*, final_layer.adaln_proj.linear.*
    full: Checkpoint,
    pub blocks: usize,
    /// the noise levels of the adapter's 8-step grid (shifted), for checks and the samplers that run part of it
    pub grid: Vec<f32>,
}

/// One net call's tables: every block's six (as `Block::tables`), the final layer's shift and scale
pub type Tables = (Vec<[Vec<f32>; 6]>, Vec<f32>, Vec<f32>);

/// One run's tables: per (t, r) point each block's modulation [rows] and the final layer's [2 * hidden]
pub struct Prepared {
    points: Vec<(f64, f64)>,
    /// [block][point * rows + o]
    blocks: Vec<Vec<f32>>,
    /// [point * 2 * hidden + o]
    last: Vec<f32>,
    rows: usize,
    last_rows: usize,
    /// the run's noise levels (the endpoint of a step is the next one)
    pub sigmas: Vec<f32>,
}

impl TwoTime {
    /// `full`: the time embedder and AdaLN tensors of the bf16 checkpoint; `adapter`: HyperFlow's file (its time LoRAs;
    /// the block LoRAs are `LoraSet`'s)
    pub fn load(full: &std::path::Path, adapter: &Checkpoint) -> Result<TwoTime> {
        let full = Checkpoint::open(full)?;
        if adapter.metadata.get("hyperflow").map(String::as_str) != Some("true") {
            return Err(Error(format!("{}: not a HyperFlow adapter (no hyperflow metadata)", adapter.path.display())));
        }
        let num = |k: &str| -> Result<f32> {
            adapter.metadata.get(k).and_then(|v| v.parse().ok()).ok_or_else(|| Error(format!("{}: no {k} in its metadata", adapter.path.display())))
        };
        let (gate, alpha, rank, shift) = (num("hyperflow_gate")?, num("lora_alpha")?, num("lora_rank")?, num("hyperflow_video_shift")?);
        let scale = alpha / rank;
        let load = |lora: &str| -> Result<Embedder> {
            let (w1, b1, w2, b2) = (host(&full, "time_embedder.proj_in.weight")?, host(&full, "time_embedder.proj_in.bias")?,
                                    host(&full, "time_embedder.proj_out.weight")?, host(&full, "time_embedder.proj_out.bias")?);
            let (hidden, out) = (b1.len(), b2.len());
            let mut e = Embedder { w1, b1, w2, b2, hidden, out };
            if e.w1.len() != hidden * FREQ_DIM || e.w2.len() != out * hidden {
                return Err(Error(format!("{}: a time embedder of {} / {} weights", full.path.display(), e.w1.len(), e.w2.len())));
            }
            merge(&mut e.w1, FREQ_DIM, adapter, &format!("transformer.{lora}.linear_1"), scale)?;
            merge(&mut e.w2, hidden, adapter, &format!("transformer.{lora}.linear_2"), scale)?;
            Ok(e)
        };
        let (base, endpoint) = (load("time_embedder")?, load("endpoint_time_embedder")?);
        let blocks = full.entries.keys().filter_map(|k| k.strip_prefix("blocks.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        let raw: Vec<f32> = adapter.metadata.get("hyperflow_sigmas").and_then(|s| serde_json::from_str(s).ok()).unwrap_or_default();
        let grid = raw.iter().map(|s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
        Ok(TwoTime { base, endpoint, gate, full, blocks, grid })
    }

    /// silu(e(t, r)) [t_dim]: what the AdaLN projections read
    pub fn embedding(&self, t: f64, r: f64) -> Vec<f32> {
        let (et, er) = (self.base.run(t as f32), self.endpoint.run(r as f32));
        et.iter().zip(&er).map(|(a, b)| {
            let v = a + self.gate * (b - a);
            v / (1.0 + (-v).exp())
        }).collect()
    }

    /// The tables for every point: `W s + b` with each projection read once; `lms` per block [rows, curve dims] (LMS
    /// Sharp's branches, on `curve` at t: [grid, curve dims], `adaln_t_table`)
    pub fn prepare(&self, points: &[(f64, f64)], sigmas: &[f32], lms: Option<&[Option<Vec<f32>>]>, curve: &[f32], curve_dim: usize, threads: usize)
                   -> Result<Prepared> {
        let s: Vec<Vec<f32>> = points.iter().map(|(t, r)| self.embedding(*t, *r)).collect();
        let td = s.first().map_or(0, Vec::len);
        let curve_at = |t: f64| -> Vec<f32> {
            let grid = curve.len() / curve_dim.max(1);
            let pos = (t as f32).clamp(0.0, 1.0) * (grid - 1) as f32;
            let i0 = (pos.floor() as usize).min(grid - 2);
            let f = pos - i0 as f32;
            (0..curve_dim).map(|c| curve[i0 * curve_dim + c] + f * (curve[(i0 + 1) * curve_dim + c] - curve[i0 * curve_dim + c])).collect()
        };
        let cs: Vec<Vec<f32>> = points.iter().map(|(t, _)| curve_at(*t)).collect();
        let project = |name: &str, extra: Option<&[f32]>| -> Result<(Vec<f32>, usize)> {
            let we = self.full.get(&format!("{name}.weight"))?.clone();
            if we.dtype != DType::BF16 || we.shape.len() != 2 || we.shape[1] != td {
                return Err(Error(format!("{name}: {:?} {:?}, bfloat16 [rows, {td}] expected", we.dtype, we.shape)));
            }
            let rows = we.shape[0];
            let w = self.full.read(&format!("{name}.weight")).ctx(name)?;
            let b = host(&self.full, &format!("{name}.bias"))?;
            let mut out = vec![0f32; points.len() * rows];
            let per = rows.div_ceil(threads.max(1)).max(1);
            // the table [points, rows] split by rows: each thread its rows of every point
            let parts: Vec<(usize, usize, Vec<f32>)> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..rows).step_by(per).map(|r0| {
                    let (w, b, s, cs) = (&w, &b, &s, &cs);
                    let r1 = (r0 + per).min(rows);
                    sc.spawn(move || {
                        let mut part = vec![0f32; points.len() * (r1 - r0)];
                        let mut row = vec![0f32; td];
                        for o in r0..r1 {
                            for (k, v) in row.iter_mut().enumerate() {
                                let i = 2 * (o * td + k);
                                *v = bf16_to_f32(u16::from_le_bytes([w[i], w[i + 1]]));
                            }
                            for p in 0..s.len() {
                                let mut acc = b[o] + row.iter().zip(&s[p]).map(|(a, c)| a * c).sum::<f32>();
                                if let Some(d) = extra {
                                    let cd = cs[p].len();
                                    acc += d[o * cd..(o + 1) * cd].iter().zip(&cs[p]).map(|(a, c)| a * c).sum::<f32>();
                                }
                                part[p * (r1 - r0) + o - r0] = acc;
                            }
                        }
                        (r0, r1, part)
                    })
                }).collect();
                handles.into_iter().filter_map(|h| h.join().ok()).collect()
            });
            if parts.iter().map(|p| p.1 - p.0).sum::<usize>() != rows {
                return Err(Error(format!("{name}: a projection thread failed")));
            }
            for (r0, r1, part) in parts {
                for p in 0..points.len() {
                    out[p * rows + r0..p * rows + r1].copy_from_slice(&part[p * (r1 - r0)..(p + 1) * (r1 - r0)]);
                }
            }
            Ok((out, rows))
        };
        let mut blocks = Vec::with_capacity(self.blocks);
        let mut rows = 0;
        for i in 0..self.blocks {
            let extra = lms.and_then(|l| l.get(i)).and_then(|x| x.as_deref());
            let (t, r) = project(&format!("blocks.{i}.adaln_proj.linear"), extra)?;
            rows = r;
            blocks.push(t);
        }
        let (last, last_rows) = project("final_layer.adaln_proj.linear", None)?;
        Ok(Prepared { points: points.to_vec(), blocks, last, rows, last_rows, sigmas: sigmas.to_vec() })
    }
}

impl Prepared {
    /// The (t, r) points prepared
    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }

    /// Block `block`'s modulation at point `p` [rows] (None: the final layer's [2 * hidden])
    pub fn row(&self, block: Option<usize>, p: usize) -> &[f32] {
        match block {
            Some(b) => &self.blocks[b][p * self.rows..(p + 1) * self.rows],
            None => &self.last[p * self.last_rows..(p + 1) * self.last_rows],
        }
    }

    fn point(&self, t: f64, r: f64) -> Result<usize> {
        self.points.iter().position(|p| p.0 == t && p.1 == r).ok_or_else(|| Error(format!("hyperflow: no tables prepared for t {t}, r {r}")))
    }

    /// The endpoint of the timestep `t` at noise level `sigma`: the next level's for the generated rows' timestep
    /// (`t_gen`), t itself for every other
    pub fn endpoint(&self, sigma: f32, t: f64, t_gen: f64) -> Result<f64> {
        if t != t_gen {
            return Ok(t);
        }
        let i = self.sigmas.iter().position(|s| *s == sigma).ok_or_else(|| Error(format!("hyperflow: sigma {sigma} is not in the run's levels")))?;
        Ok(1.0 - *self.sigmas.get(i + 1).unwrap_or(&0.0) as f64)
    }

    /// Every block's six tables and the final layer's shift and scale for the timesteps `values` at `sigma`, laid out as
    /// `Block::tables` and `Outer::final_tables` lay theirs out
    pub fn tables_at(&self, values: &[f64], t_gen: f64, sigma: f32, hidden: usize, modalities: usize) -> Result<Tables> {
        let idx: Vec<usize> = values.iter().map(|t| self.point(*t, self.endpoint(sigma, *t, t_gen)?)).collect::<Result<_>>()?;
        if self.rows != 6 * hidden * modalities || self.last_rows != 2 * hidden {
            return Err(Error(format!("hyperflow: tables of {} / {} rows for hidden {hidden}", self.rows, self.last_rows)));
        }
        let c = hidden;
        let blocks = self.blocks.iter().map(|m| {
            let mut out: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0f32; values.len() * modalities * c]);
            for (vi, p) in idx.iter().enumerate() {
                let src = &m[p * self.rows..(p + 1) * self.rows];
                for mi in 0..modalities {
                    for (ci, table) in out.iter_mut().enumerate() {
                        let o0 = mi * 6 * c + ci * c;
                        table[(vi * modalities + mi) * c..(vi * modalities + mi + 1) * c].copy_from_slice(&src[o0..o0 + c]);
                    }
                }
            }
            out
        }).collect();
        let (mut shift, mut scale) = (Vec::with_capacity(values.len() * c), Vec::with_capacity(values.len() * c));
        for p in &idx {
            let src = &self.last[p * self.last_rows..(p + 1) * self.last_rows];
            shift.extend_from_slice(&src[..c]);
            scale.extend_from_slice(&src[c..2 * c]);
        }
        Ok((blocks, shift, scale))
    }
}

/// The distinct (t, r) points of a run: `values_at(sigma)` gives the timesteps one net call at that level uses and the
/// generated rows' one
pub fn points(sigmas: &[f32], values_at: &mut dyn FnMut(f32) -> (Vec<f64>, f64)) -> Vec<(f64, f64)> {
    let mut seen = BTreeMap::new();
    for (i, s) in sigmas.iter().enumerate().take(sigmas.len().saturating_sub(1)) {
        let (values, t_gen) = values_at(*s);
        for t in values {
            let r = if t == t_gen { 1.0 - sigmas[i + 1] as f64 } else { t };
            seen.insert((t.to_bits(), r.to_bits()), (t, r));
        }
    }
    seen.into_values().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_run_asks_for_each_step_and_its_endpoint() {
        let sig = [0.834942f32, 0.696852, 0.468753, 0.0];
        let p = super::points(&sig, &mut |s| (vec![1.0 - s as f64, 0.999, 1.0], 1.0 - s as f64));
        assert_eq!(p.len(), 5);
        assert!(p.contains(&(1.0 - 0.834942f32 as f64, 1.0 - 0.696852f32 as f64)));
        assert!(p.contains(&(1.0 - 0.468753f32 as f64, 1.0)));
        assert!(p.contains(&(0.999, 0.999)));
    }

    /// The sinusoids: cos first (flip_sin_to_cos), frequencies exp(-ln(10000) i / 128)
    #[test]
    fn the_time_embedding_input_is_diffusers_sinusoids() {
        let e = super::Embedder { w1: (0..256).map(|i| if i == 0 || i == 128 + 1 { 1.0 } else { 0.0 }).collect(), b1: vec![0.0], w2: vec![1.0], b2: vec![0.0],
                                  hidden: 1, out: 1 };
        // hidden = silu(cos(t * 1) + sin(t * f1)), f1 = 10000^(-1/128)
        let t = 0.5f32;
        let f1 = (-(10000f64.ln()) / 128.0).exp() as f32;
        let v = (t).cos() + (t * f1).sin();
        assert!((e.run(t)[0] - v / (1.0 + (-v).exp())).abs() < 1e-6);
    }
}
