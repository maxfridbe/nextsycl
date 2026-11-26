//! pi-H3 (Lakonik/pi-H3): MiniMax H3 in 2 or 4 steps, distilled with pi-Flow v2, as adapters on the FL2VA denoiser we
//! serve (the pruned base's revision checked: reference/pi-h3/base_check.py). An adapter holds, under ComfyUI's names
//! (the `*_int8_convrot.safetensors` files) or Diffusers' (the adapter folders):
//!
//! - rank-64 LoRAs on every block's attention output and both MLP matrices (`attn.out_proj`, `mlp.fc1`, `mlp.fc2`) -
//!   46 blocks in the DWD adapters, all 50 in the VAE ones; run beside the int8 matrices as our LoRAs are;
//! - every block's AdaLN projection on the timestep curve, in place of the base's;
//! - in the DWD adapters, blocks 46-49 tuned whole: their out_proj / fc1 / fc2 (int8 ConvRot in the ComfyUI file) and
//!   AdaLN bias in place of the base's (q, k, v stay the base's);
//! - the final layer whole (its norm, AdaLN projection and both output linears);
//! - the DWD head (`dwd_norm_out`, `dwd_pixel_refiner`: dwd.rs), which makes the frames at the last step.
//!
//! Merged at load: the engine is the adapted model (a card names the adapter file as its `adapter`). Sampling is
//! LakonLab's FlowEulerODEScheduler (`schedule`): Euler, no CFG, from the noise itself, on raw times
//! `linspace(max_raw_t, max_raw_t * s / (n - 1 + s), n)` (s = final_step_size_scale 0.6; max_raw_t 0.93 for 2 steps,
//! 0.96 for 4) through the video shift 12 - the audio through 3, as H3's sampler derives it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::{host, small, Weight};
use crate::dtype::DType;
use crate::dwd::DwdHead;
use crate::jobs::Engine;
use crate::lora::LoraSet;
use crate::ops::{Int8Linear, Linear};
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

/// The raw-time start of the 2- and 4-step adapters (LakonLab's examples, ComfyUI-piFlow's adapter node)
pub fn max_raw_t(steps: usize) -> Option<f32> {
    match steps {
        2 => Some(0.93),
        4 => Some(0.96),
        _ => None,
    }
}

/// The last step's size against the others
pub const FINAL_STEP_SIZE_SCALE: f32 = 0.6;

/// FlowEulerODEScheduler's sigmas for `steps` steps through `shift` (float32 as it computes them: numpy's linspace in
/// float64 rounded to float32, the shift in float32), then 0
pub fn schedule(steps: usize, max_raw_t: f32, final_step_size_scale: f32, shift: f32) -> Vec<f32> {
    let (a, s) = (max_raw_t as f64, final_step_size_scale as f64);
    let end = a * s / (steps as f64 - 1.0 + s);
    let mut out: Vec<f32> = (0..steps)
        .map(|i| if steps == 1 { a } else { a + (end - a) * i as f64 / (steps - 1) as f64 })
        .map(|t| crate::denoiser::shifted(t as f32, shift))
        .collect();
    out.push(0.0);
    out
}

/// The step count an adapter was distilled for, from its file name (`pi_h3_2step_...`, `pi_h3_4step_...`)
pub fn steps_in_name(p: &Path) -> Option<usize> {
    let n = p.to_string_lossy();
    [2, 4].into_iter().find(|s| n.contains(&format!("{s}step")))
}

/// A loaded adapter: what stays with the engine (the LoRAs, the DWD head), and what it changed
pub struct Adapter {
    pub path: PathBuf,
    pub steps: usize,
    pub lora: LoraSet,
    /// the DWD head, when the adapter has one and the H3 video silo is here
    pub dwd: Option<DwdHead>,
    /// the adapter has a DWD head (whether or not it could be loaded)
    pub has_dwd: bool,
    pub diffusers: bool,
    /// blocks whose matrices it replaced
    pub replaced: Vec<usize>,
}

impl Adapter {
    /// One line for the log and the job reports
    pub fn describe(&self) -> String {
        format!("{} ({} steps, {} layout; {} LoRA layers rank {}, AdaLN projections of every block, blocks {:?} tuned whole, the final layer{})",
                self.path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default(), self.steps,
                if self.diffusers { "Diffusers" } else { "ComfyUI" }, self.lora.layers, self.lora.max_rank, self.replaced,
                match (&self.dwd, self.has_dwd) {
                    (Some(d), _) => format!(", the DWD head ({} Sol layers, {}, GPU {})", d.layers(), d.form, d.gpu()),
                    (None, true) => ", a DWD head (not loaded: no H3 video silo - the VAE decodes)".to_string(),
                    _ => String::new(),
                })
    }
}

/// An adapter file's tensor names under ComfyUI's: the Diffusers layout's block and final-layer names mapped (fc1's
/// halves are put in ComfyUI's order where they are read)
struct Names<'a> {
    ck: &'a Checkpoint,
    diffusers: bool,
}

impl Names<'_> {
    /// The file's key for a ComfyUI name, if the file has it
    fn key(&self, comfy: &str) -> Option<String> {
        let k = match self.diffusers {
            false => comfy.to_string(),
            true => {
                let mut k = comfy.to_string();
                if let Some(rest) = k.strip_prefix("blocks.") {
                    k = format!("transformer_blocks.{rest}");
                    for (a, b) in [(".attn.out_proj.", ".attn.to_out.0."), (".mlp.fc1.", ".ff.net.0.proj."), (".mlp.fc2.", ".ff.net.2.")] {
                        k = k.replace(a, b);
                    }
                }
                for (a, b) in [("final_layer.adaln_proj.linear.", "norm_out.linear."), ("final_layer.norm.", "norm_out.norm."), ("final_layer.video_out.", "proj_out."),
                               ("final_layer.audio_out.", "audio_proj_out.")] {
                    if let Some(rest) = k.strip_prefix(a) {
                        k = format!("{b}{rest}");
                    }
                }
                k
            }
        };
        self.ck.entries.contains_key(&k).then_some(k)
    }

    fn host(&self, comfy: &str) -> Result<Vec<f32>> {
        let k = self.key(comfy).ok_or_else(|| Error(format!("{}: no {comfy}", self.ck.path.display())))?;
        host(self.ck, &k)
    }
}

fn f32_tensor(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

/// Loads the adapter at `path` onto the engine's model: every change made, the LoRAs and the DWD head kept in the
/// returned `Adapter` (the engine holds it). `steps`: the step count it was distilled for (None: from the file's name).
pub fn apply(e: &mut Engine, path: &Path, steps: Option<usize>, log: &mut dyn FnMut(String)) -> Result<Adapter> {
    let ck = Checkpoint::open(path)?;
    let diffusers = ck.entries.keys().any(|k| k.starts_with("transformer_blocks."));
    let names = Names { ck: &ck, diffusers };
    let steps = steps.or_else(|| steps_in_name(path)).ok_or_else(|| Error(format!("{}: how many steps? (H3_PI_STEPS 2 or 4)", path.display())))?;
    if max_raw_t(steps).is_none() {
        return Err(Error(format!("pi-H3 adapters run 2 or 4 steps, not {steps}")));
    }
    if e.model.resident() != e.model.blocks.len() {
        return Err(Error("a pi-H3 adapter needs every block on the card (no resident-blocks)".into()));
    }
    let cfg = e.model.cfg;
    let n = e.model.blocks.len();
    // the AdaLN projections of every block (the bias where the adapter has one)
    let mut adaln = 0;
    for i in 0..n {
        let p = format!("blocks.{i}.adaln_proj.linear");
        if names.key(&format!("{p}.weight")).is_none() {
            continue;
        }
        let w = names.host(&format!("{p}.weight"))?;
        let b = match names.key(&format!("{p}.bias")) {
            Some(_) => Some(names.host(&format!("{p}.bias"))?),
            None => None,
        };
        e.model.blocks[i].set_adaln(w, b).map_err(|x| Error(format!("block {i}: {x}")))?;
        adaln += 1;
    }
    if adaln != n {
        return Err(Error(format!("{}: AdaLN projections for {adaln} of the {n} blocks", path.display())));
    }
    // blocks tuned whole: their int8 ConvRot matrices in place of the base's
    let mut replaced = Vec::new();
    for i in 0..n {
        for (j, l) in crate::lora::LINEARS.iter().enumerate() {
            let name = format!("blocks.{i}.{l}");
            let Some(wk) = names.key(&format!("{name}.weight")) else { continue };
            if diffusers {
                return Err(Error(format!("{}: {wk} is a whole matrix in float - give the adapter's int8 ConvRot file (ComfyUI layout)", path.display())));
            }
            let q = ck.quant(&name)?.ok_or_else(|| Error(format!("{name}: a whole matrix without its int8 record")))?;
            let (we, se) = (ck.get(&wk)?, ck.get(&format!("{name}.weight_scale"))?);
            if we.dtype != DType::I8 || se.dtype != DType::F32 {
                return Err(Error(format!("{name}: int8 weights with float32 scales expected")));
            }
            let lin = Int8Linear { weight: Tensor::from_bytes(&e.dev, DType::I8, &we.shape, &ck.read(&wk)?)?,
                                   scale: Tensor::from_bytes(&e.dev, DType::F32, &[we.shape[0]], &ck.read(&format!("{name}.weight_scale"))?)?, bias: None,
                                   group: q.convrot.then_some(q.group) };
            e.model.blocks[i].set_linear(j, Weight::I8(lin)).map_err(|x| Error(format!("{name}: {x}")))?;
            if replaced.last() != Some(&i) {
                replaced.push(i);
            }
        }
    }
    // the final layer whole
    {
        let dev = &e.dev;
        let lin = |name: &str| -> Result<Linear> {
            let k = names.key(&format!("{name}.weight")).ok_or_else(|| Error(format!("{}: no {name}", path.display())))?;
            Ok(Linear { weight: f32_tensor(dev, &ck.get(&k)?.shape, &host(&ck, &k)?)?, bias: Some(f32_tensor(dev, &[ck.get(&k)?.shape[0]], &names.host(&format!("{name}.bias"))?)?) })
        };
        let nk = names.key("final_layer.norm.weight").ok_or_else(|| Error(format!("{}: no final layer", path.display())))?;
        let norm = small(dev, &ck, &nk)?;
        e.outer.set_final(norm, names.host("final_layer.adaln_proj.linear.weight")?, names.host("final_layer.adaln_proj.linear.bias")?, lin("final_layer.video_out")?,
                          lin("final_layer.audio_out")?)?;
    }
    // the LoRAs, on the card for the engine's life
    let lora = LoraSet::load(&e.dev, &ck, 1.0, n, 2)?;
    if lora.max_rank > crate::dit::MAX_LORA_RANK {
        return Err(Error(format!("LoRA rank {} is over {}", lora.max_rank, crate::dit::MAX_LORA_RANK)));
    }
    let has_dwd = DwdHead::in_file(&ck);
    let dwd = match has_dwd {
        true => {
            if diffusers {
                log("pi-h3  : the adapter's Diffusers file holds its DWD head in float16 - loaded as bfloat16".to_string());
            }
            DwdHead::load(e.dev.index, &ck, e.threads)?
        }
        false => None,
    };
    if let Some(d) = &dwd {
        if d.hidden != cfg.hidden || d.t_dim != cfg.t_dim {
            return Err(Error(format!("{}: a DWD head for a {}-wide model on a {}-point curve, this one is {} on {}", path.display(), d.hidden, d.t_dim, cfg.hidden, cfg.t_dim)));
        }
    }
    e.dev.wait()?;
    Ok(Adapter { path: path.to_path_buf(), steps, lora, dwd, has_dwd, diffusers, replaced })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LakonLab's FlowEulerODEScheduler(num_train_timesteps=1, shift, max_raw_t, final_step_size_scale=0.6), set_timesteps
    /// (2 / 4): the sigmas it gives (run in the reference venv)
    #[test]
    fn the_schedule_is_lakonlabs() {
        let two = schedule(2, 0.93, 0.6, 12.0);
        let want2 = [0.993_766_665_458_679_2_f64, 0.865_339_875_221_252_4, 0.0];
        let four = schedule(4, 0.96, 0.6, 12.0);
        let want4 = [0.996_539_771_556_854_2_f64, 0.964_451_253_414_154, 0.899_297_356_605_529_8, 0.695_652_186_870_575, 0.0];
        for (g, w) in two.iter().zip(want2).chain(four.iter().zip(want4)) {
            assert!((*g as f64 - w).abs() < 1e-7, "{two:?} {four:?}");
        }
        assert_eq!(two.len(), 3);
        assert_eq!(four.len(), 5);
        // the audio's (shift 3) is what H3's sampler derives from the video's: the same raw times
        let audio = schedule(2, 0.93, 0.6, 3.0);
        for (v, a) in two.iter().zip(&audio).take(2) {
            let derived = crate::layout::time_shift_sigma(*v as f64, 12.0, 3.0) as f32;
            assert!((derived - a).abs() < 1e-6, "{derived} {a}");
        }
        assert!((audio[0] as f64 - 0.975_524_425_506_591_8).abs() < 1e-7 && (audio[1] as f64 - 0.616_347_551_345_825_2).abs() < 1e-7, "{audio:?}");
    }

    #[test]
    fn steps_come_from_the_file_name() {
        assert_eq!(steps_in_name(Path::new("/m/pi_h3_2step_dwd_adapter_int8_convrot.safetensors")), Some(2));
        assert_eq!(steps_in_name(Path::new("/m/pi_h3_4step_adapter/diffusion_pytorch_model.safetensors")), Some(4));
        assert_eq!(steps_in_name(Path::new("/m/other.safetensors")), None);
    }
}
