//! MiniMax H3 on Intel Arc (SYCL): text, pictures, a voice and other clips to video with sound - H3's engine (the
//! sycl-h3 studio's `h3-core`, its jobs and its mp4 writer) moved in whole, so its numbers are H3's. The arithmetic
//! runs in the diffusion kernels (`kernels/diffusion`, H3's under the `nsd_` prefix) of the video kind's library;
//! this crate owns everything around it: the device and its memory, reading checkpoints, the model graph, the
//! sampler, the encoders and decoders, the jobs.
//!
//! The contract (`nextsycl_video::VideoEngine`): `generate` for a clip from a request, and H3's jobs by kind
//! (`run_job`: generate, encode, decode, denoise, check-block, bench-blocks) for the daemon and `nextsycl video job`.

pub mod audio;
pub mod control;
pub mod denoiser;
pub mod device;
pub mod dit;
pub mod dtype;
pub mod esrgan;
pub mod jobs;
pub mod media;
pub mod gguf;
pub mod layout;
pub mod noise;
pub mod load;
pub mod lora;
pub mod ops;
pub mod reference;
pub mod rng;
pub mod safetensors;
pub mod sys;
pub mod te;
pub mod tokenizer;
pub mod upscale;
pub mod venc;
pub mod vae;
pub mod vision;

pub type Result<T> = std::result::Result<T, Error>;

/// One error type for the engine: a message, with the context added on the way up.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(e.to_string())
    }
}
impl From<String> for Error {
    fn from(e: String) -> Self {
        Error(e)
    }
}
impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error(e.to_string())
    }
}

/// Adds what was being done to an error: `read(..).ctx("reading the header")?`.
pub trait Ctx<T> {
    fn ctx(self, what: impl std::fmt::Display) -> Result<T>;
}
impl<T, E: std::fmt::Display> Ctx<T> for std::result::Result<T, E> {
    fn ctx(self, what: impl std::fmt::Display) -> Result<T> {
        self.map_err(|e| Error(format!("{what}: {e}")))
    }
}

// ---- the engine behind the video contract -----------------------------------------------------------------------

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use nextsycl_video::{At, Clip, Defaults, EngineOption, Gpu, JobCtl, LoadOptions, ModelFiles, Progress, Sampler, Schedule, VideoEngine,
                     VideoKind, VideoRequest};
use serde_json::{json, Value};

pub const ARCH: &str = "minimax-h3";

/// The files it takes, by role (the denoiser at load; the others go into each job, where a job does not name its own)
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "tokenizer", "vae", "audio-vae", "vision", "controlnet"];

/// A role and the job key it fills
const ROLE_KEYS: &[(&str, &str)] = &[("text-encoder", "te"), ("tokenizer", "tokenizer"), ("vae", "vae"), ("audio-vae", "audio_vae"),
                                     ("upscaler", "upscaler"), ("pixel-upscaler", "pixel_upscaler"), ("vision", "te_visual"), ("controlnet", "controlnet")];

/// The options it takes (`--opt-NAME`): at load the variables they set; per request the generate job's own keys
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "threads", env: "H3_THREADS", value: "N", help: "threads reading the checkpoint at load (default 8)", at: At::Load },
    EngineOption { name: "te-pin", env: "H3_TE_PIN", value: "0|1", help: "keep the text encoder in pinned host memory between clips (default 1)", at: At::Load },
    EngineOption { name: "vae-attn-per-tile", env: "H3_VAE_ATTN_PER_TILE", value: "0|1", help: "the video decoder's attention a tile at a time", at: At::Load },
    EngineOption { name: "attn", env: "NSD_ATTN", value: "sage|onednn", help: "long sequences' attention: SageAttention (int8 q, k) or oneDNN's", at: At::Load },
    EngineOption { name: "sage-min-s", env: "NSD_SAGE_MIN_S", value: "TOKENS", help: "the shortest sequence Sage takes (default 8192)", at: At::Load },
    EngineOption { name: "attn-rows", env: "NSD_ATTN_ROWS", value: "N", help: "query rows a fused attention call takes", at: At::Load },
    EngineOption { name: "mem-fraction", env: "NSD_MEM_FRACTION", value: "F", help: "the share of the card the kernels may hold (default 0.94)", at: At::Load },
    EngineOption { name: "profile", env: "NSD_PROFILE", value: "", help: "the kernels' and decoders' times by part, on stderr", at: At::Load },
    EngineOption { name: "first_frame", env: "", value: "PNG", help: "the clip starts on this picture", at: At::Request },
    EngineOption { name: "last_frame", env: "", value: "PNG", help: "the clip ends on this picture", at: At::Request },
    EngineOption { name: "first_frame_ref", env: "", value: "PNG", help: "a picture the clip refers to (not a frame of it)", at: At::Request },
    EngineOption { name: "first_latent", env: "", value: "FILE", help: "continue from a clip's last latent (.lastlat)", at: At::Request },
    EngineOption { name: "guide_clip", env: "", value: "MP4[:N[:I]]", help: "a clip's motion as a guide: N frames from frame I", at: At::Request },
    EngineOption { name: "first_audio", env: "", value: "FILE", help: "continue a clip's sound (.lastaud)", at: At::Request },
    EngineOption { name: "first_audio_s", env: "", value: "S", help: "seconds of it (default 1)", at: At::Request },
    EngineOption { name: "ref_audio", env: "", value: "WAV[,WAV...]", help: "a voice to hold (up to three)", at: At::Request },
    EngineOption { name: "source", env: "", value: "MP4", help: "a clip to regenerate part of", at: At::Request },
    EngineOption { name: "regen", env: "", value: "FROM-TO", help: "the seconds of the source to regenerate", at: At::Request },
    EngineOption { name: "regen_box", env: "", value: "X,Y,W,H", help: "the region of the source to regenerate", at: At::Request },
    EngineOption { name: "cond_noise_aug", env: "", value: "X", help: "noise added to the conditioning frames", at: At::Request },
    EngineOption { name: "shift_video", env: "", value: "X", help: "the video schedule's shift (default 12)", at: At::Request },
    EngineOption { name: "shift_audio", env: "", value: "X", help: "the audio schedule's shift (default 3)", at: At::Request },
    EngineOption { name: "pixel_upscaler", env: "", value: "FILE", help: "an ESRGAN network to enlarge the frames (instead of the latents)", at: At::Request },
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
    // its own context on the GPU (H3's device), as each worker process had
    let dev = device::Device::open_index(gpu).map_err(ve)?;
    let e = jobs::Engine::load_on(dev, denoiser, None, threads, log).map_err(ve)?;
    Ok(Box::new(H3 { e, files: files.clone(), load_s: t0.elapsed().as_secs_f64() }))
}

fn ve(e: Error) -> nextsycl_video::Error {
    nextsycl_video::Error(e.0)
}

pub struct H3 {
    pub e: jobs::Engine,
    files: ModelFiles,
    load_s: f64,
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
        &["generate", "encode", "decode", "denoise", "check-block", "bench-blocks"]
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
