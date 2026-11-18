//! MiniMax H3 on Intel Arc (SYCL): text, pictures, a voice and other clips to video with sound - H3's engine (the
//! sycl-h3 studio's `h3-core`, its jobs and its mp4 writer) moved in whole, so its numbers are H3's. The engine
//! itself lives in the foundation (`crates/h3`, nextsycl-h3: the image engine of P-H3-Image runs the same denoiser);
//! this crate is H3 behind the video contract.
//!
//! The contract (`nextsycl_video::VideoEngine`): `generate` for a clip from a request, and H3's jobs by kind
//! (`run_job`: generate, encode, decode, denoise, check-block, bench-blocks) for the daemon and `nextsycl video job`.

pub use nextsycl_h3::*;

pub mod silo;

// ---- the engine behind the video contract -----------------------------------------------------------------------

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use nextsycl_video::{At, Clip, Defaults, EngineOption, Gpu, JobCtl, LoadOptions, ModelFiles, Progress, Sampler, Schedule, VideoEngine,
                     VideoKind, VideoRequest};
use serde_json::{json, Value};

pub const ARCH: &str = "minimax-h3";

/// The files it takes, by role (the denoiser at load; the others go into each job, where a job does not name its own)
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "tokenizer", "vae", "audio-vae", "vision", "controlnet", "ic-lora", "hyperflow", "adaln", "sharp-lora",
                             "silence", "fast-vae", "mpeg-maps", "region-head"];

/// A role and the job key it fills
const ROLE_KEYS: &[(&str, &str)] = &[("text-encoder", "te"), ("tokenizer", "tokenizer"), ("vae", "vae"), ("audio-vae", "audio_vae"),
                                     ("upscaler", "upscaler"), ("pixel-upscaler", "pixel_upscaler"), ("vision", "te_visual"), ("controlnet", "controlnet"),
                                     // Open-H3-Regenerate-2K-IC's parts (the regen2k job)
                                     ("ic-lora", "ic_lora"), ("hyperflow", "hyperflow"), ("adaln", "adaln"), ("sharp-lora", "sharp_lora"), ("silence", "silence"),
                                     ("fast-vae", "regen_fast_vae"),
                                     // MPEG-Diffusion's (the generate job's two-track sampler, crates/h3/src/mpeg.rs)
                                     ("mpeg-maps", "mpeg_maps"), ("region-head", "region_head")];

/// MPEG-Diffusion's settings a model card may set (H3_MPEG=1 turns it on): the variable and the generate job's key
const MPEG_SETTINGS: &[(&str, &str)] = &[("H3_MPEG_SWITCH", "mpeg_switch"), ("H3_MPEG_HR_STEPS", "mpeg_hr_steps"), ("H3_MPEG_ALPHA", "mpeg_alpha"),
                                         ("H3_MPEG_FLOOR", "mpeg_floor"), ("H3_MPEG_FEATHER", "mpeg_feather")];

/// The options it takes (`--opt-NAME`): at load the variables they set; per request the generate job's own keys
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "threads", env: "H3_THREADS", value: "N", help: "threads reading the checkpoint at load (default 8)", at: At::Load },
    EngineOption { name: "resident-blocks", env: "H3_RESIDENT_BLOCKS", value: "N", help: "keep only the first N denoiser blocks on the card and stream the rest from the int8 file as they run (the card for the activations: Open-H3-Regenerate-2K's 2560x1440 sequences; default all)", at: At::Load },
    EngineOption { name: "te-pin", env: "H3_TE_PIN", value: "0|1", help: "keep the text encoder in pinned host memory between clips (default 1)", at: At::Load },
    EngineOption { name: "regen-vae", env: "H3_REGEN_VAE", value: "fast|standard", help: "regen2k's decoder: Open-H3-Regenerate-2K's fast decoder (the card's fast-vae file; the H3 video silo's kernels) or the H3 video decoder (default standard)", at: At::Load },
    EngineOption { name: "vae-attn-per-tile", env: "H3_VAE_ATTN_PER_TILE", value: "0|1", help: "the video decoder's attention a tile at a time", at: At::Load },
    EngineOption { name: "attn", env: "NSD_ATTN", value: "sage|onednn", help: "long sequences' attention: SageAttention (int8 q, k) or oneDNN's", at: At::Load },
    EngineOption { name: "sage-min-s", env: "NSD_SAGE_MIN_S", value: "TOKENS", help: "the shortest sequence Sage takes (default 8192)", at: At::Load },
    EngineOption { name: "attn-rows", env: "NSD_ATTN_ROWS", value: "N", help: "query rows a fused attention call takes", at: At::Load },
    EngineOption { name: "mem-fraction", env: "NSD_MEM_FRACTION", value: "F", help: "the share of the card the kernels may hold (default 0.94)", at: At::Load },
    EngineOption { name: "profile", env: "NSD_PROFILE", value: "", help: "the kernels' and decoders' times by part, on stderr", at: At::Load },
    EngineOption { name: "mpeg", env: "H3_MPEG", value: "0|1", help: "MPEG-Diffusion: every clip from text alone made on two tracks - half resolution, then full-resolution tokens on a region the model picks (needs the mpeg-maps and region-head files; the minimax-h3-mpeg card)", at: At::Load },
    EngineOption { name: "mpeg-switch", env: "H3_MPEG_SWITCH", value: "N", help: "with mpeg: half-resolution steps before the region joins (default 16, of 20)", at: At::Load },
    EngineOption { name: "mpeg-hr-steps", env: "H3_MPEG_HR_STEPS", value: "N", help: "with mpeg: two-track steps after the switch, linear in sigma (default 6; 0: the schedule's rest)", at: At::Load },
    EngineOption { name: "mpeg-alpha", env: "H3_MPEG_ALPHA", value: "X", help: "with mpeg: the region's threshold m + alpha (p - m) (default 0.5; higher: a smaller region)", at: At::Load },
    EngineOption { name: "mpeg-floor", env: "H3_MPEG_FLOOR", value: "X", help: "with mpeg: a clip whose 99th-percentile score is under this stays half-resolution (default none)", at: At::Load },
    EngineOption { name: "mpeg-feather", env: "H3_MPEG_FEATHER", value: "N", help: "with mpeg: the paste's ramp inside the region's edge, full-res latent pixels (default 4)", at: At::Load },
    EngineOption { name: "first_frame", env: "", value: "PNG", help: "the clip starts on this picture (a clip: its last frame, a still - no motion; guide_clip carries motion)", at: At::Request },
    EngineOption { name: "last_frame", env: "", value: "PNG", help: "the clip ends on this picture", at: At::Request },
    EngineOption { name: "first_frame_ref", env: "", value: "PNG", help: "exposure match: the first and last frame pictures rescaled to this picture's luminance (a chain's drift; clip 1's frame)", at: At::Request },
    EngineOption { name: "first_latent", env: "", value: "FILE", help: "continue from a clip's last latent (.lastlat) - jumps lighter or darker with false texture: chain a PNG", at: At::Request },
    EngineOption { name: "guide_clip", env: "", value: "MP4[:N[:I]][,...]", help: "a clip's last N frames (5, 22, 39: 17k + 5) as a moving keyframe at frame I - motion across a cut (22 frames + a 0.9167 s first_audio); several comma-separated; a PNG (or N under 5) is one frame, the pose at frame I, no motion", at: At::Request },
    EngineOption { name: "first_audio", env: "", value: "FILE", help: "continue a clip's sound (.lastaud)", at: At::Request },
    EngineOption { name: "soundtrack", env: "", value: "FILE", help: "a driving soundtrack (a song): the clip's length of it from soundtrack_start is pinned under the whole clip as an audio keyframe, the picture is made to it, and the song is laid under the clip", at: At::Request },
    EngineOption { name: "soundtrack_start", env: "", value: "S", help: "seconds into the soundtrack at the clip's first frame (default 0)", at: At::Request },
    EngineOption { name: "first_audio_s", env: "", value: "S", help: "seconds of it (default 1, at least 0.75; a guide's length with guide_clip) - the join must overlap exactly this", at: At::Request },
    EngineOption { name: "ref_audio", env: "", value: "WAV[,WAV...]", help: "a voice to hold (up to three)", at: At::Request },
    EngineOption { name: "source", env: "", value: "MP4", help: "a clip to regenerate part of", at: At::Request },
    EngineOption { name: "regen", env: "", value: "FROM-TO", help: "the seconds of the source to regenerate", at: At::Request },
    EngineOption { name: "regen_box", env: "", value: "X0,Y0,X1,Y1", help: "the region of the source to regenerate (pixels: left, top, right, bottom)", at: At::Request },
    EngineOption { name: "cond_noise_aug", env: "", value: "X", help: "DO NOT USE: noise added to the keyframes, which are never denoised - it decodes as coloured confetti (seen at 0.03)", at: At::Request },
    EngineOption { name: "exposure_fix", env: "", value: "BOOL", help: "after the decode: lift the frames by exposure_offset, match a motion guide's frames to the guide's luma and ease the frames after it to the clip's steady level (a flash after the cut); off by default", at: At::Request },
    EngineOption { name: "exposure_offset", env: "", value: "LUMA", help: "with exposure_fix: luma (0-255) added to every frame, the decoder's darkening (default 2.3)", at: At::Request },
    EngineOption { name: "exposure_ease", env: "", value: "FRAMES", help: "with exposure_fix: frames after the guide over which the luma eases to the steady level (default 24)", at: At::Request },
    EngineOption { name: "exposure_target", env: "", value: "LUMA", help: "with exposure_fix and no guide_clip: the luma (0-255) of the clip's first frame (default: the first_frame picture's)", at: At::Request },
    EngineOption { name: "knots", env: "", value: "T,T,..", help: "a distilled model's own noise levels, as timesteps of 1000 from 1000 down to 0 (Pika H3 Flex: 1000,750,500,250,100,40,0 with the lcm sampler); one step between two", at: At::Request },
    EngineOption { name: "refine_from", env: "", value: "FILE", help: "a refine pass (with the 2K refiner LoRA): a draft's .latents.safetensors, doubled to this clip's canvas, its sound kept; a draft over 3 s goes in overlapping windows", at: At::Request },
    EngineOption { name: "refine_upscaler", env: "", value: "FILE", help: "with refine_from: the latent upscaler", at: At::Request },
    EngineOption { name: "refine_sigmas", env: "", value: "S,S,..", help: "with refine_from: the noise levels (default 0.909375,0.857143,0.666667,0)", at: At::Request },
    EngineOption { name: "shift_video", env: "", value: "X", help: "the video schedule's shift (default 12)", at: At::Request },
    EngineOption { name: "shift_audio", env: "", value: "X", help: "the audio schedule's shift (default 3)", at: At::Request },
    EngineOption { name: "pixel_upscaler", env: "", value: "FILE", help: "an ESRGAN network to enlarge the frames (instead of the latents)", at: At::Request },
    EngineOption { name: "regen2k", env: "", value: "MP4", help: "Open-H3-Regenerate-2K-IC: this 768p draft made again at 2560x1440 (the regen2k job; the model's ic-lora, hyperflow, adaln and silence files; its own sound kept)", at: At::Request },
    EngineOption { name: "regen_sampler", env: "", value: "up3|up4|hf8", help: "with regen2k: 3 or 4 steps from the upscaled draft, or 8 from noise (default up3)", at: At::Request },
    EngineOption { name: "regen_resolution", env: "", value: "2k|1080p", help: "with regen2k: 2560x1440 (default) or 1920x1080", at: At::Request },
    EngineOption { name: "sharp", env: "", value: "X", help: "with regen2k: LMS Sharp's strength (default 1; 0 off)", at: At::Request },
    EngineOption { name: "regen_vae", env: "", value: "fast|standard", help: "with regen2k: the decoder for this clip (default the model's H3_REGEN_VAE)", at: At::Request },
];

/// This engine's registry entry
pub fn kind() -> VideoKind {
    VideoKind { archs: &[ARCH], name: "MiniMax H3 (50-block DiT, Qwen3-VL 32B text encoder, video + audio VAEs, upscalers)", roles: ROLES, load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> nextsycl_video::Result<Box<dyn VideoEngine>> {
    let denoiser = files.get("denoiser").ok_or_else(|| nextsycl_video::Error(format!("{ARCH}: no denoiser file")))?;
    let gpu = gpus.first().map_or(0, |g| g.index);
    let threads = o.setting("H3_THREADS").and_then(|v| v.parse().ok()).unwrap_or(8);
    let t0 = std::time::Instant::now();
    let mpeg = o.setting("H3_MPEG").is_some_and(|v| v == "1");
    let mpeg_silo = if mpeg { silo::mpeg().map_err(|e| nextsycl_video::Error(e.0))? } else { None };
    // its own context on the GPU (H3's device), as each worker process had; with MPEG-Diffusion's silo, a context on the
    // core's queue of the card, so the silo's kernels are ordered with the engine's
    let dev = match (mpeg_silo, gpus.first()) {
        // SAFETY: the core's GPU (its queue's owner) is kept in the engine beside the device.
        (Some(_), Some(g)) => unsafe { device::Device::on_queue(g.queue(), g.index) }.map_err(ve)?,
        _ => device::Device::open_index(gpu).map_err(ve)?,
    };
    let resident = o.setting("H3_RESIDENT_BLOCKS").and_then(|v| v.parse().ok());
    let mut e = jobs::Engine::load_resident(dev, denoiser, None, resident, threads, log).map_err(ve)?;
    if mpeg {
        for role in ["mpeg-maps", "region-head"] {
            if files.get(role).is_none() {
                return Err(nextsycl_video::Error(format!("{ARCH}: H3_MPEG=1 needs the {role} file (the minimax-h3-mpeg card)")));
            }
        }
        match (mpeg_silo, gpus.first()) {
            (Some(s), Some(g)) => {
                e.model.gather = Some(silo::gather(g, s));
                log(format!("{ARCH}: MPEG-Diffusion on, the {} silo's gather", silo::SILO));
            }
            _ => log(format!("{ARCH}: MPEG-Diffusion on, no {} silo: the two-track steps copy a row at a time", silo::SILO)),
        }
    }
    let mpeg = mpeg.then(|| MPEG_SETTINGS.iter().filter_map(|(var, key)| Some((key.to_string(), o.setting(var)?))).collect());
    // regen2k's decoder (the card's H3_REGEN_VAE): the fast one needs its file and this engine's silo
    let regen_vae = o.setting("H3_REGEN_VAE").filter(|v| !v.is_empty());
    if files.contains_key("fast-vae") || regen_vae.as_deref() == Some("fast") {
        match silo::install() {
            Ok(Some(p)) => log(format!("silo   : {} (the fast decoder's kernels)", p.display())),
            Ok(None) => log("silo   : none - regen2k decodes with the H3 video decoder".to_string()),
            Err(e) => log(format!("silo   : {e} - regen2k decodes with the H3 video decoder")),
        }
    }
    Ok(Box::new(H3 { e, files: files.clone(), load_s: t0.elapsed().as_secs_f64(), mpeg, regen_vae, _gpu: gpus.first().cloned() }))
}

fn ve(e: Error) -> nextsycl_video::Error {
    nextsycl_video::Error(e.0)
}

pub struct H3 {
    pub e: jobs::Engine,
    files: ModelFiles,
    load_s: f64,
    /// MPEG-Diffusion on (H3_MPEG=1): the card's settings as the generate job's keys
    mpeg: Option<Vec<(String, String)>>,
    /// the core's GPU, whose queue the device may share (kept for its life)
    _gpu: Option<Arc<Gpu>>,
    /// regen2k's decoder when a request does not say (H3_REGEN_VAE: fast | standard)
    regen_vae: Option<String>,
}

impl H3 {
    /// A job's spec with the model's files for the keys it leaves out
    fn with_files(&self, spec: &Value) -> Value {
        let mut s = spec.clone();
        if let Some(m) = s.as_object_mut() {
            for (role, key) in ROLE_KEYS {
                if let (None, Some(p)) = (m.get(*key), self.files.get(*role)) {
                    // the tokenizer is a directory (vocab.json, merges.txt, tokenizer_config.json): a file's
                    let p = if *role == "tokenizer" && p.is_file() { p.parent().map(Path::to_path_buf).unwrap_or_else(|| p.clone()) } else { p.clone() };
                    m.insert((*key).into(), json!(p));
                }
            }
            // MPEG-Diffusion: the generate job's two-track sampler, unless the request says "mpeg": false
            if let (Some(set), Some("generate")) = (&self.mpeg, m.get("kind").and_then(Value::as_str)) {
                if !m.contains_key("mpeg") {
                    m.insert("mpeg".into(), json!(true));
                }
                for (k, v) in set {
                    if !m.contains_key(k) {
                        m.insert(k.clone(), json!(v));
                    }
                }
            }
            if let (None, Some(v)) = (m.get("regen_vae"), &self.regen_vae) {
                m.insert("regen_vae".into(), json!(v));
            }
            // the effect embeddings (roles embedding-NAME): `embedding:NAME` in a prompt finds them in their directory
            if !m.contains_key("embeddings") {
                if let Some(d) = self.files.iter().find(|(r, _)| r.starts_with("embedding")).and_then(|(_, p)| p.parent()) {
                    m.insert("embeddings".into(), json!(d));
                }
            }
        }
        s
    }
}

impl VideoEngine for H3 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 384, height: 288, seconds: 2.0, fps: 24.0, steps: 8, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift,
                   shift: 12.0, upscale: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn jobs(&self) -> &'static [&'static str] {
        &["generate", "encode", "decode", "denoise", "check-block", "bench-blocks", "regen2k"]
    }
    fn run_job(&self, spec: &Value, ctl: &mut JobCtl) -> nextsycl_video::Result<Value> {
        let spec = self.with_files(spec);
        let mut c = jobs::Ctl { log: &mut *ctl.log, cancel: ctl.cancel, progress: ctl.progress.as_mut().map(|p| &mut **p as &mut dyn FnMut(usize, usize)) };
        jobs::run(&self.e, &spec, &mut c).map_err(ve)
    }
    fn device_stats(&self) -> Option<(u64, u64, Option<u64>)> {
        Some((self.e.dev.mem_used(), self.e.dev.mem_cap(), self.e.dev.mem_free()))
    }
    /// A clip from a request: H3's generate job (its own keys from the request's options)
    fn generate(&self, req: &VideoRequest, progress: &mut dyn FnMut(Progress), cancel: &dyn Fn() -> bool) -> nextsycl_video::Result<Clip> {
        let d = self.defaults();
        let mut spec = json!({
            "kind": "generate", "prompt": req.prompt, "out": req.out,
            "width": req.width.unwrap_or(d.width), "height": req.height.unwrap_or(d.height),
            "seconds": req.seconds.unwrap_or(d.seconds), "steps": req.steps.unwrap_or(d.steps), "seed": req.seed,
        });
        if let Some(u) = req.upscale.filter(|u| *u > 1.0) {
            spec["upscale"] = json!(u);
        }
        if let Some(x) = req.sampler {
            spec["sampler"] = json!(x.name());
        }
        if let Some(x) = req.schedule {
            spec["schedule"] = json!(x.name());
        }
        if !req.loras.is_empty() {
            spec["lora"] = json!(req.loras.iter().map(|l| format!("{}:{}", l.path.display(), l.scale)).collect::<Vec<_>>().join(","));
        }
        for (k, v) in &req.extra {
            spec[k] = v.parse::<f64>().map_or_else(|_| json!(v), |n| json!(n));
        }
        // a draft to regenerate at 2K: the regen2k job, the request's prompt as the draft's caption
        if let Some(draft) = req.extra.get("regen2k") {
            spec = json!({"kind": "regen2k", "draft": draft, "caption": req.prompt, "out": req.out, "seed": req.seed,
                          "sampler": req.extra.get("regen_sampler").map_or("up3", String::as_str),
                          "resolution": req.extra.get("regen_resolution").map_or("2k", String::as_str)});
            if let Some(x) = req.extra.get("sharp").and_then(|v| v.parse::<f64>().ok()) {
                spec["sharp"] = json!(x);
            }
            if let Some(x) = req.extra.get("regen_vae") {
                spec["regen_vae"] = json!(x);
            }
        }
        let flag = AtomicBool::new(false);
        let t0 = std::time::Instant::now();
        let mut log = |l: String| eprintln!("{l}");
        let mut steps = |done: usize, total: usize| {
            if cancel() {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            progress(Progress { stage: "denoise", at: done as u32, of: total as u32, seconds: t0.elapsed().as_secs_f64() });
        };
        let v = self.run_job(&spec, &mut JobCtl { log: &mut log, cancel: &flag, progress: Some(&mut steps) })?;
        let (frames, _, _) = jobs::temporal_shape(req.seconds.unwrap_or(d.seconds) as f64);
        let out = v.get("out").and_then(Value::as_str).map(PathBuf::from).unwrap_or_else(|| req.out.clone());
        let n = |k: &str, dflt: u32| v.get(k).and_then(Value::as_u64).map_or(dflt, |x| x as u32);
        Ok(Clip { path: out, width: n("width", req.width.unwrap_or(d.width)), height: n("height", req.height.unwrap_or(d.height)),
                  frames: n("frames", frames as u32), fps: 24.0, audio: Path::new(&self.files.get("audio-vae").cloned().unwrap_or_default()).exists() })
    }
}
