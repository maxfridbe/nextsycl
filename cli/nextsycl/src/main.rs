//! `nextsycl`: the command line - the server as a service (start / stop / status / ps / cache / chat / logs, sycl-h3's
//! scheme: a container, a control socket, NS_* settings), and the tools that run in this process.
//!
//!     nextsycl info <model.gguf>      the architecture, its geometry, every tensor checked by role, bytes by group
//!     nextsycl gpus                   each GPU in its own context: memory, copies and their rates, GPU to GPU, and
//!                                     that device memory costs no host RAM
//!     nextsycl serve <model.gguf> [--gpu 0,1] [--host 0.0.0.0] [--port 8085] [--name ID] [--ctx 8192] [--effort low]
//!                                     the OpenAI-compatible server (serve.rs)
//!     nextsycl check <model.gguf> <dump dir> [--gpu N]
//!                                     the forward pass on a reference dump's prompt (reference/llama-dump), every
//!                                     step compared: cosine and relative error per tensor, then the next token

mod bench;
mod client;
mod container;
mod drives;
mod audio;
mod model3d;
mod host;
mod image;
mod video;
mod service;
mod served;

use nextsycl_llm::sampling::{dist, sample, Rng};
use nextsycl_models::{config, registry as models};
use nextsycl_serve::{cache, http, serve};

use std::path::Path;
use std::process::ExitCode;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{Engine, EngineKind, LoadOptions};

/// yy.mmdd.### from git (version.sh), handed in by build.sh; "dev" for a build without it
pub const VERSION: &str = match option_env!("NS_VERSION") {
    Some(v) => v,
    None => "dev",
};

const USAGE: &str = "nextsycl - language, image, video, audio and 3D models on Intel Arc GPUs (Rust + SYCL)

every kind:
  nextsycl models [list] [--json]           the registry (NS_REGISTRY): every model this machine serves, its kind
  nextsycl models add <id> <file.gguf> [--title T] [--gpu 0[,1] | all] [--ctx N[,M...]] [--set NAME=VALUE]...
                      [--no-tools] [--no-tasks] [--disabled]
  nextsycl models search [TEXT] [--kind llm|image|video|audio|model3d|lora]     the catalog of supported models
  nextsycl models pull <id>... [--dir DIR] [--from DIR]... [--verify] [--again]
                                its files downloaded (resumed when run again; each checked by SHA-256) or linked
                                (the same file already here; --from DIR: copies elsewhere on the machine), registered
  nextsycl models pull <id> <url | hf:org/repo/path/file.gguf> [--dir DIR] [add's options]
                                a file outside the catalog (every shard of a split file); then added
  nextsycl models remove <id> [--files]     --files: its files deleted too
  nextsycl models enable <id> | disable <id>
  nextsycl models move <id> --to <drive | mount> [--yes]   its files to another drive: copied, read back (SHA-256),
                                the registry repointed, the old files removed (those another model names stay);
                                the front door's job when it is up; --cancel stops the move under way
  nextsycl drives [--json]      the drives the models are on: device, PCIe link, mount, size, free, models, read speed
  nextsycl drives measure <drive>   its read speed: 4 GiB of model files past the page cache (O_DIRECT), kept
  nextsycl gpus                 each GPU in its own context: memory, copy rates, GPU to GPU, host RAM unaffected
  nextsycl switch [--port 8001] [--upstream 127.0.0.1:8085] [--studio URL] [--alias OLD=NEW]...
                                the chat front end's model switcher: every registered chat model on one endpoint,
                                a swap (through the studio's llm.mode) when a request names another
  nextsycl gpustat [--out /run/gpustat.json] [--interval 3] [--pci ADDR] [--once]
                                GPU telemetry (VRAM from the DRM clients, busy %, power, temperatures, the PCIe link)
                                into a JSON file, as root
  nextsycl maintenance [on|off|status] [--message TEXT] [--eta MINUTES] [--unload]
                                the front door's maintenance mode: callers told, new work refused, queues waiting;
                                on prints the bypass key (header X-Maintenance-Bypass) for testing meanwhile
  nextsycl version

language models (nextsycl llm ...; the names without `llm` still work):
  nextsycl llm start <id>       a registered model as a service (a container; the model stays loaded on the GPUs);
                                flags beside it win: [--gpu N ...] [--port N] [--host H] [--ctx N | N,M,...] [--name ID]
                                [--effort low|medium|high|max] [--prompt-cache-mib N] [--no-mtp]
  nextsycl llm serve <id>       the same in the foreground, its log here (Ctrl-C ends it)
  nextsycl llm stop             gracefully: the request running finishes, then the server ends
  nextsycl llm status [--no-stream]   live, like docker stats: the model, its GPUs, the request running, the cache
  nextsycl llm ps [-a]          the request running and the last ones
  nextsycl llm inspect <id>     one request (an ID from ps) as JSON: settings, messages, timings, previews
  nextsycl llm chat <text> [--effort E] [--max N] [--temp T]     one request, streamed
  nextsycl llm cache [ls | clear]     the prompt cache's checkpoints
  nextsycl llm logs [--no-follow]     the server's log, followed
  nextsycl llm bench [--sizes 20,2185,8000,40000] [--new 256] [--parallel 1,2,4] [--out DIR]
  nextsycl llm bench --needle [--sizes 32768,131072,250000] [--depths 10,50,90] [--out DIR]
  nextsycl llm engines          the architectures this build serves
  nextsycl llm selftest [--gpu N]    the llm kernel library, an engine's own symbol, a kernel on the GPU

  in this process (inside the image: the kernels need the oneAPI runtime):
  nextsycl llm serve <model.gguf> [--gpu 0,1 | all] [--host H] [--port N] [--name ID] [--ctx N | N,M,...] [--effort E]
                 [--socket PATH] [--expert-gib G] [--mirror-gib G] [--no-mtp] [--prompt-cache-mib N (4096; 0 = off)]
                 [--cors ORIGINS] [--keep-requests N (100)] [--parallel N (2)] [--max-tokens N]
                 [--cache-dir DIR [--cache-disk-gib G (32)] [--cache-ttl-hours H (24)]]
                                the server itself (what start runs in its container)
  nextsycl llm generate <model.gguf> --prompt TEXT | --prompt-file PATH | --ids FILE [--ignore-eos] [--ctx N]
                 [--effort low|medium|high|max] [--max N] [--temp T] [--top-p P] [--seed S] [--gpu N[,M]] [--no-mtp]
                 [--image PATH ...]  pictures before the prompt (a vision model's)
                 [--logits-out PATH]  the prompt's last logits as float32 (a check against a reference)
                 [--score FILE [--score-max N (256)] [--dump OUT | --ref OUT]]  FILE's tokens teacher-forced after the
                                prompt: their perplexity; --dump writes the logits, --ref compares with a dump (KL, top-1)
  nextsycl llm info <model.gguf>      the architecture and geometry, every tensor checked by role, bytes by group
  nextsycl llm tokenize <model.gguf> <text>
  nextsycl llm check <model.gguf> <dump dir> [--gpu N[,M]]    the forward pass against a reference dump
  nextsycl llm spec-check <model.gguf> --prompt TEXT [--n N] [--rows R (2..5)] [--gpu 0,1] [--prefill | --prompt-decode]
                                verify passes against one-token decode, logits compared
  nextsycl llm kernels <model.gguf> [--gpu N]   each weight type's decode kernel against the exact path
  nextsycl llm prepare <model.gguf>             the files its engine makes beside it (deepseek4: the MTP sidecar)

images (nextsycl image ...):
  nextsycl image engines        the architectures this build serves (gen, edit, start, serve come with the first)
  nextsycl image selftest [--gpu N]  the image kernel library, an engine's own symbol, a kernel on the GPU

video (nextsycl video ...):
  nextsycl video engines        the architectures this build serves (H3's commands move here)
  nextsycl video selftest [--gpu N]  the video kernel library, an engine's own symbol, a kernel on the GPU

audio (nextsycl audio ...):
  nextsycl audio gen \"<description>\" --lyrics TEXT | --lyrics-file FILE [--seconds N] [--seed N] [--out FILE]
                                a song (MiniMax Music 3), in this process (inside the image)
  nextsycl audio serve [MODEL] [--wfe] [--port 8087] [--host H] [--out DIR]
                                the speech API (/v1/audio/speech: input = lyrics, instructions = description) in a
                                container (nextsycl-audio); --wfe: the web page at /
  nextsycl audio check <dump dir> [--stages ar,dit,voc,chunks]    the engine against a reference's dumps
  nextsycl audio engines        the architectures this build serves
  nextsycl audio selftest [--gpu N]  the audio kernel library, an engine's own symbol, a kernel on the GPU

3D (nextsycl model3d ... or nextsycl 3d ...):
  nextsycl model3d gen PICTURE [PICTURE:SIDE]... [--model ID] [--texture] [--seed N] [--faces N] [--out FILE.glb]
                                a mesh from a picture or several views (TRELLIS.2, Hunyuan3D, Pixal3D), in this process
  nextsycl model3d rig MESH.glb [--model ID] [--prompt \"the motion\"]   a skeleton and skin weights, or motion
  nextsycl model3d serve [MODEL] [--wfe] [--port 8088] [--out DIR]    the 3D API (/v1/model3d/generate, /rig) in a container
  nextsycl model3d tool decimate|remesh|uv|convert IN OUT [...]       the CPU mesh tools (no GPU)
  nextsycl model3d engines | selftest [--gpu N]

settings (environment, or NAME=value lines in nextsycl.conf beside the repository or ~/.config/nextsycl.conf):
  NS_MODELS        host directory with the model files, seen as /models                  (required for start)
  NS_MODEL         the model as seen in the container (default /models/glm53-iq2/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf)
  NS_GPUS          GPUs, e.g. \"0 1\" (default all)
  NS_HOST, NS_PORT the OpenAI API and /api/chat (default 127.0.0.1, 8085; 0.0.0.0 = the network, no password)
  NS_CORS          web pages that may call the API from a browser, beyond the loopback ones, e.g.
                   \"http://studio:8095\" (comma-separated; * = any)
  NS_CTX           tokens of context per session (default 65536), or one a session: 262144,32768 (one long, one short)
  NS_NAME          the model id clients see (glm-5.3-flash-uncensored)
  NS_EFFORT        default reasoning effort (low)        NS_NO_MTP=1   decode without the draft block
  NS_KV            the latent cache's form: q8 (default, 544 bytes a token and layer) or f16 (1,024)
  NS_PROMPT_CACHE_MIB   host memory for the prompt cache's checkpoints (default 4096; 0 = off)
  NS_CACHE_DIR, NS_CACHE_DISK_GIB, NS_CACHE_TTL_HOURS   checkpoints pushed out of memory (and those in it at a stop)
                   on disk (default ~/.cache/nextsycl/prompts, 32 GiB - 0 = none, 24 h unused before removal)
  NS_KEEP_REQUESTS ended requests the server keeps for ps and inspect (default 100)
  NS_PARALLEL      requests decoded together, each with a session of its own (default 2; 1 = one at a time)
  NS_MAX_TOKENS    the tokens a request without max_tokens may make (default: to the end of the context)
  NS_SOCKET_DIR    where the control socket lives (default $XDG_RUNTIME_DIR/nextsycl)
  NS_IMAGE, NS_CONTAINER_ENGINE   the image with the oneAPI runtime (localhost/nextsycl-build: ./build.sh image) and podman / docker
  NS_MOUNTS        more read-only directories for the container, host:inside[,host:inside] (another model folder, ...)
  NS_QW_*          the qwen4exp engine's settings, passed to the server (docs/engines.md): NS_QW_MTP (the draft
                   layer), NS_QW_EXPERT_PROFILE, NS_QW_CVEC / _LAYERS / _MODE / _DIR (a control vector), NS_QW_CHUNK";

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The llm engines this program has, one a model architecture (llm/<arch>)
pub(crate) fn engines() -> Vec<EngineKind> {
    vec![nextsycl_llm_glm5next::kind(), nextsycl_llm_qwen4exp::kind(), nextsycl_llm_qwen35::kind(), nextsycl_llm_deepseek2::kind(), nextsycl_llm_deepseek4::kind(), nextsycl_llm_deepseek41::kind(), nextsycl_llm_museglimmer::kind(), nextsycl_llm_nemotronh::kind(), nextsycl_llm_example::kind()]
}

/// The engine for `f`'s architecture, loaded on `gpus`
fn load_engine<'g>(f: &'g Gguf, gpus: &[std::sync::Arc<nextsycl_core::Gpu>], o: LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>, String> {
    let kinds = engines();
    let k = nextsycl_llm::kind_for(&kinds, f)?;
    // the --opt-NAMEs: the variables the engine and its kernels read, set before it loads (no other thread reads the
    // environment yet: the server's start after the load)
    for (var, v) in opt_env(&llm_options(k), nextsycl_core::At::Load, k.name)? {
        log(format!("option {var}={v}"));
        std::env::set_var(var, v);
    }
    (k.load)(f, gpus, &o, log).map_err(|e| e.0)
}

/// The chat template of the engine serving `f`
fn chat_of(f: &Gguf) -> Result<nextsycl_llm::ChatFn, String> {
    Ok(nextsycl_llm::kind_for(&engines(), f)?.chat)
}

fn info(path: &Path) -> Result<(), String> {
    let f = Gguf::open(path).map_err(|e| e.0)?;
    println!("file     : {} ({} file{}, {:.2} GiB of tensors, {} tensors)", path.display(), f.paths.len(), if f.paths.len() > 1 { "s" } else { "" },
             gib(f.total_bytes()), f.tensors.len());
    let kinds = engines();
    let k = nextsycl_llm::kind_for(&kinds, &f)?;
    println!("engine   : {}", k.name);
    print!("{}", (k.info)(&f)?);
    Ok(())
}

/// MemAvailable, bytes.
fn host_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

fn gpus() -> Result<(), String> {
    use nextsycl_core::{DevBuf, Gpu, HostBuf};
    use std::time::Instant;
    let e = |x: nextsycl_core::Error| x.0;
    let list = nextsycl_core::gpus().map_err(e)?;
    if list.is_empty() {
        return Err("no Level Zero GPU".into());
    }
    let mut open = Vec::new();
    for (i, name) in &list {
        let g = Gpu::open(*i).map_err(e)?;
        let (total, free) = g.memory().map_err(e)?;
        let units = nextsycl_core::gpu_units(*i).map_or(String::new(), |(u, m)| format!(", {u} compute units at {m} MHz"));
        println!("GPU {i}: {name}, {:.1} GiB{}{units}", gib(total), free.map_or(String::new(), |f| format!(" ({:.1} GiB free)", gib(f))));
        open.push(g);
    }
    // copies: 256 MiB through pinned memory, each way, checked
    const N: usize = 256 << 20;
    for g in &open {
        let d = DevBuf::new(g, N).map_err(e)?;
        let mut h = HostBuf::new(g, N).map_err(e)?;
        for (i, b) in h.as_mut_slice().iter_mut().enumerate() {
            *b = (i * 7 + g.index) as u8;
        }
        let t = Instant::now();
        d.write(0, h.as_slice()).map_err(e)?;
        g.sync().map_err(e)?;
        let up = N as f64 / t.elapsed().as_secs_f64() / 1e9;
        let mut back = HostBuf::new(g, N).map_err(e)?;
        let t = Instant::now();
        d.read(0, back.as_mut_slice()).map_err(e)?;
        let down = N as f64 / t.elapsed().as_secs_f64() / 1e9;
        let ok = back.as_slice() == h.as_slice();
        println!("GPU {}: 256 MiB to the GPU {up:.1} GB/s, back {down:.1} GB/s, {}", g.index, if ok { "identical" } else { "DIFFERENT" });
        if !ok {
            return Err(format!("GPU {}: the round trip changed the data", g.index));
        }
    }
    // GPU to GPU, through host memory
    if open.len() > 1 {
        let (a, b) = (&open[0], &open[1]);
        let (src, dst) = (DevBuf::new(a, N).map_err(e)?, DevBuf::new(b, N).map_err(e)?);
        let mut h = HostBuf::new(a, N).map_err(e)?;
        for (i, x) in h.as_mut_slice().iter_mut().enumerate() {
            *x = (i * 13) as u8;
        }
        src.write(0, h.as_slice()).map_err(e)?;
        a.sync().map_err(e)?;
        let mut staging = vec![0u8; N];
        let t = Instant::now();
        dst.copy_from_peer(&src, &mut staging).map_err(e)?;
        let rate = N as f64 / t.elapsed().as_secs_f64() / 1e9;
        let mut back = vec![0u8; N];
        dst.read(0, &mut back).map_err(e)?;
        let ok = back == h.as_slice();
        println!("GPU 0 -> GPU 1: 256 MiB through host memory {rate:.1} GB/s, {}", if ok { "identical" } else { "DIFFERENT" });
        if !ok {
            return Err("the GPU-to-GPU copy changed the data".into());
        }
    }
    // device memory must not cost host memory (a context spanning both GPUs made xe mirror it); 1 GiB a card,
    // small enough beside a loaded model (a card past its memory spills, and on this box that can livelock)
    let before = host_available();
    let held: Vec<DevBuf> = open.iter().map(|g| {
        let b = DevBuf::new(g, 1 << 30)?;
        b.fill(1)?;
        g.sync()?;
        Ok(b)
    }).collect::<Result<_, nextsycl_core::Error>>().map_err(e)?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    let after = host_available();
    let used = before.saturating_sub(after);
    println!("{} GiB of device memory (1 a card) took {:.2} GiB of host memory{}", held.len(), gib(used),
             if used < 1 << 29 { " - no mirror" } else { " - MIRRORED" });
    if used >= 1 << 29 {
        return Err("device memory is being mirrored into host RAM".into());
    }
    Ok(())
}

/// `rows-check <model.gguf> --prompt-file P --text T [--at N] [--rows R] [--gpu 0,1]`: the prompt and the text's first N
/// tokens read, then the next R tokens as one pass of R rows and, from a copy of the session, as R passes of one row;
/// every tapped tensor compared row by row (cosine), in the pass's order - where a multi-row pass first differs from decode
fn rows_check(args: &[String]) -> Result<(), String> {
    use std::collections::BTreeMap;
    let e = |x: nextsycl_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("rows-check <model.gguf> --prompt-file P --text T")?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = nextsycl_tok::Tokenizer::from_gguf(&f)?;
    let pf = opt("--prompt-file").ok_or("--prompt-file P")?;
    let prompt = std::fs::read_to_string(&pf).map_err(|x| format!("{pf}: {x}"))?;
    let tf = opt("--text").ok_or("--text T")?;
    let text = tok.encode(&std::fs::read_to_string(&tf).map_err(|x| format!("{tf}: {x}"))?);
    let at: usize = opt("--at").and_then(|v| v.parse().ok()).unwrap_or(200);
    let rows: usize = opt("--rows").and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut ids = tok.encode(&chat_of(&f)?(&[nextsycl_tok::Message { role: "user", content: &prompt, ..Default::default() }], nextsycl_tok::Effort::Low, &[]));
    ids.extend_from_slice(&text[..at.min(text.len())]);
    let next = text.get(at..at + rows).ok_or("--text: too short for --at and --rows")?.to_vec();
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "0".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let mut log = |l: String| eprintln!("[{l}]");
    let eng = load_engine(&f, &gs, LoadOptions { expert_bytes: None, mirror_bytes: None, draft: false, kv: (ids.len() + rows + 8, 2) }, &mut log)?;
    let mut a = eng.session(ids.len() + rows + 8).map_err(e)?;
    let mut none = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
    eng.feed(&mut a, &ids, &mut none).map_err(e)?;
    let mut b = eng.session(ids.len() + rows + 8).map_err(e)?;
    eng.copy_session(&mut b, &a).map_err(e)?;
    // the pass of R rows: each tensor once, R rows; the R passes of one row: each tensor R times, a row each
    let mut order: Vec<String> = Vec::new();
    let mut one: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    {
        let mut tap = |n: &str, d: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> {
            if !one.contains_key(n) {
                order.push(n.to_string());
            }
            one.insert(n.to_string(), d.to_f32()?);
            Ok(())
        };
        eng.forward_rows(&mut a, &next, rows, &mut tap).map_err(e)?;
    }
    let mut many: BTreeMap<String, Vec<Vec<f32>>> = BTreeMap::new();
    for t in &next {
        let mut tap = |n: &str, d: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> {
            many.entry(n.to_string()).or_default().push(d.to_f32()?);
            Ok(())
        };
        eng.forward_rows(&mut b, &[*t], 1, &mut tap).map_err(e)?;
    }
    let cos = |x: &[f32], y: &[f32]| {
        let (mut d, mut nx, mut ny) = (0f64, 0f64, 0f64);
        for (p, q) in x.iter().zip(y) {
            d += *p as f64 * *q as f64;
            nx += (*p as f64).powi(2);
            ny += (*q as f64).powi(2);
        }
        if nx == 0.0 && ny == 0.0 { 1.0 } else { d / (nx.sqrt() * ny.sqrt()).max(1e-30) }
    };
    println!("{} prompt tokens, rows {at}..{} of the text: {rows} rows a pass against {rows} passes of one", ids.len(), at + rows);
    let mut shown = 0;
    for n in &order {
        let (Some(x), Some(ys)) = (one.get(n), many.get(n)) else { continue };
        if ys.len() != rows || ys.iter().any(|y| y.len() * rows != x.len()) {
            continue;
        }
        let w = ys[0].len();
        let per: Vec<f64> = (0..rows).map(|r| cos(&x[r * w..(r + 1) * w], &ys[r])).collect();
        let worst = per.iter().copied().fold(1.0, f64::min);
        if worst < 0.9999 || shown < 3 {
            println!("{n:<28} worst {worst:.6}  rows {}", per.iter().map(|c| format!("{c:.5}")).collect::<Vec<_>>().join(" "));
            shown += 1;
            if shown >= 40 {
                break;
            }
        }
    }
    Ok(())
}

/// (cosine, relative L2 error, max |a - b|)
fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut ab, mut aa, mut bb, mut dd, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        dd += (x - y) * (x - y);
        mx = mx.max((x - y).abs());
    }
    (ab / (aa.sqrt() * bb.sqrt()).max(1e-30), (dd / bb.max(1e-30)).sqrt(), mx)
}

fn check(model: &Path, dump: &Path, gpus: &[usize]) -> Result<(), String> {
    use std::collections::BTreeMap;
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let read = |n: &str| std::fs::read_to_string(dump.join(n)).map_err(|e| format!("{}: {e}", dump.join(n).display()));
    let tokens: Vec<u32> = read("tokens.txt")?.split_whitespace().map(|t| t.parse().map_err(|_| format!("token {t}"))).collect::<Result<_, _>>()?;
    let want_next: Option<u32> = read("next.txt").ok().and_then(|s| s.trim().parse().ok());
    // name -> element count (the last line of a name wins: its file holds the last write)
    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    for line in read("index.tsv")?.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() >= 6 {
            let n: usize = c[2..6].iter().map(|v| v.parse::<usize>().unwrap_or(1)).product();
            index.insert(c[0].to_string(), n);
        }
    }
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    for g in &gs {
        println!("gpu      : {} ({})", g.name, g.index);
    }
    let mut log = |l: String| println!("load     : {l}");
    // one GPU: no host mirror (bring-up); several: the mirror as generate has it (a long prompt touches every expert)
    let mirror = if gs.len() == 1 { Some(0) } else { None };
    let eng = load_engine(&f, &gs, LoadOptions { expert_bytes: None, mirror_bytes: mirror, draft: false, kv: (tokens.len() + 8, 2) }, &mut log)?;
    println!("load     : {:.2} GiB in {:.1} s", gib(eng.load_bytes()), eng.load_seconds());
    println!("prompt   : {} tokens {:?}", tokens.len(), tokens);
    println!("{:<26} {:>10} {:>10} {:>10}", "tensor", "cosine", "rel err", "max diff");
    let mut worst: (f64, String) = (1.0, String::new());
    let t0 = std::time::Instant::now();
    // a prompt longer than a chunk: all but the last chunk first, then the last one compared - its rows against the
    // reference's last rows (the dump holds the whole prompt when llama.cpp ran it as one ubatch)
    let total = tokens.len();
    let last = total - (total - 1) % eng.prefill_chunk() - 1;
    let tc = total - last;
    let mut tap = |name: &str, b: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> {
        let Some(&n) = index.get(name) else { return Ok(()) };
        let mine = b.to_f32()?;
        // NS_CHECK_SAVE=DIR: this engine's tensors too, as <name>.f32 (for a closer look than a cosine)
        if let Ok(dir) = std::env::var("NS_CHECK_SAVE") {
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.f32")), mine.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        }
        let path = dump.join(format!("{name}.f32"));
        let raw = std::fs::read(&path).map_err(|x| nextsycl_core::Error(format!("{}: {x}", path.display())))?;
        let theirs: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if theirs.len() != n {
            println!("{name:<26} the reference file holds {} values, its index {n}", theirs.len());
            return Ok(());
        }
        // per-token tensors: the reference has `total` rows (or one: the head), mine the last chunk's
        let (mine, theirs): (&[f32], &[f32]) = if n % total == 0 && total > tc && mine.len() >= n / total * tc {
            let row = n / total;
            (&mine[..row * tc], &theirs[row * (total - tc)..])
        } else if mine.len() >= n {
            (&mine[..n], &theirs[..])
        } else {
            println!("{name:<26} sizes differ: mine {}, the reference {n}", mine.len());
            return Ok(());
        };
        let (cos, rel, mx) = compare(mine, theirs);
        println!("{name:<26} {cos:>10.6} {rel:>10.2e} {mx:>10.3e}");
        if cos < worst.0 {
            worst = (cos, name.to_string());
        }
        Ok(())
    };
    let mut sess = eng.session(tokens.len().max(16)).map_err(e)?;
    if last > 0 {
        let mut quiet = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
        eng.feed(&mut sess, &tokens[..last], &mut quiet).map_err(e)?;
        println!("prefix   : {last} tokens fed, comparing the last {tc} (positions {last}-{})", total - 1);
    }
    let logits = eng.forward(&mut sess, &tokens[last..], &mut tap).map_err(e)?;
    let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    let vocab = f.meta("tokenizer.ggml.tokens").and_then(|v| v.as_array());
    let word = |id: u32| vocab.and_then(|v| v.get(id as usize)).and_then(|v| v.as_str()).unwrap_or("?").replace('\u{120}', " ").to_string();
    println!("forward  : {:.1} s", t0.elapsed().as_secs_f64());
    // the same prompt incrementally: all but the last token, then the last alone (decode's path)
    if tokens.len() > 1 && tokens.len() <= eng.prefill_chunk() {
        let mut none = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
        let mut s2 = eng.session(tokens.len()).map_err(e)?;
        eng.forward(&mut s2, &tokens[..tokens.len() - 1], &mut none).map_err(e)?;
        let inc = eng.forward(&mut s2, &tokens[tokens.len() - 1..], &mut none).map_err(e)?;
        let (cos, rel, _) = compare(&inc, &logits);
        println!("decode   : the last token alone after the rest: logits cosine {cos:.7}, rel err {rel:.2e} against the whole prompt at once");
    }
    println!("worst    : {} (cosine {:.6})", worst.1, worst.0);
    println!("next     : {best} {:?}{}", word(best), want_next.map_or(String::new(), |w| format!(", the reference {w} {:?}{}", word(w),
                                                         if w == best { " - the same" } else { " - DIFFERENT" })));
    // how close the call is: the top logits, and where the reference's token stands
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let top: Vec<String> = order.iter().take(5).map(|&i| format!("{:?} {:.3}", word(i as u32), logits[i])).collect();
    println!("top      : {}{}", top.join(", "), want_next.map_or(String::new(), |w| format!("; the reference's token {:.3}", logits[w as usize])));
    Ok(())
}

/// `nextsycl tokenize <model.gguf> <text>`: the ids and their pieces (special tokens parsed).
fn tokenize(model: &Path, text: &str) -> Result<(), String> {
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let t = nextsycl_tok::Tokenizer::from_gguf(&f)?;
    let ids = t.encode(text);
    println!("{} tokens: {:?}", ids.len(), ids);
    for id in &ids {
        println!("  {id:>7} {:?}", t.decode(&[*id]));
    }
    let back = t.decode(&ids);
    println!("round trip: {}", if back == text { "identical" } else { "DIFFERENT" });
    Ok(())
}

/// A sampler for `Engine::step` at a temperature and top-p: speculative sampling of the drafts (nextsycl-llm's Sampler)
pub(crate) struct TempSampler<'a> {
    pub temp: f32,
    pub top_p: f32,
    pub rng: &'a mut Rng,
    /// draws on the GPU (Sampler::device) from this seed
    pub seed: u64,
}

impl nextsycl_llm::Sampler for TempSampler<'_> {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        sample(logits, self.temp, self.top_p, self.rng)
    }
    fn dist(&mut self, logits: &[f32]) -> Option<Vec<(u32, f32)>> {
        dist(logits, self.temp, self.top_p)
    }
    fn uniform(&mut self) -> f32 {
        self.rng.next_f32()
    }
    fn greedy(&self) -> bool {
        self.temp <= 0.0
    }
    fn device(&self) -> Option<nextsycl_llm::DeviceSampling> {
        (self.temp > 0.0).then_some(nextsycl_llm::DeviceSampling { temperature: self.temp, top_p: self.top_p, seed: self.seed })
    }
}

/// `nextsycl generate <model.gguf> --prompt TEXT [--effort low|medium|high|max] [--max N] [--temp T] [--top-p P] [--gpu N]`
fn generate(args: &[String]) -> Result<(), String> {
    use std::io::Write;
    let e = |x: nextsycl_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("generate <model.gguf> --prompt ...")?;
    // --ids FILE: the prompt as token ids (comma-separated, the chat template already in them - Strata's bench files)
    let ids_file = opt("--ids");
    // --ignore-eos: the full --max tokens whatever is generated (a bench's fixed output length)
    let ignore_eos = args.iter().any(|a| a == "--ignore-eos");
    let prompt = match (opt("--prompt-file"), &ids_file) {
        (_, Some(_)) => String::new(),
        (Some(f), None) => std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?,
        (None, None) => opt("--prompt").ok_or("--prompt TEXT, --prompt-file PATH or --ids FILE")?,
    };
    let effort = nextsycl_tok::Effort::parse(&opt("--effort").unwrap_or_else(|| "low".into())).ok_or("--effort low|medium|high|max")?;
    let max: usize = opt("--max").and_then(|v| v.parse().ok()).unwrap_or(256);
    let temp: f32 = opt("--temp").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let top_p: f32 = opt("--top-p").and_then(|v| v.parse().ok()).unwrap_or(0.95);
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "0".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}: a GPU number"))).collect::<Result<_, _>>()?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = nextsycl_tok::Tokenizer::from_gguf(&f)?;
    let ids = match &ids_file {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?
            .split([',', ' ', '\n']).filter(|v| !v.trim().is_empty())
            .map(|v| v.trim().parse::<u32>().map_err(|_| format!("{p}: {v:?} is not a token id")))
            .collect::<Result<Vec<u32>, String>>()?,
        None => tok.encode(&chat_of(&f)?(&[nextsycl_tok::Message { role: "user", content: &prompt, ..Default::default() }], effort, &[])),
    };
    // --ids-only (with --score and --ids-out): write the ids and stop, no GPU
    if args.iter().any(|a| a == "--ids-only") {
        let (sf, p) = (opt("--score").ok_or("--ids-only: --score FILE")?, opt("--ids-out").ok_or("--ids-only: --ids-out PATH")?);
        let mut sc = tok.encode(&std::fs::read_to_string(&sf).map_err(|x| format!("{sf}: {x}"))?);
        sc.truncate(opt("--score-max").and_then(|v| v.parse().ok()).unwrap_or(256));
        let line = |v: &[u32]| v.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
        std::fs::write(&p, format!("{}\n{}\n", line(&ids), line(&sc))).map_err(|x| format!("{p}: {x}"))?;
        eprintln!("[{} prompt ids and {} text ids in {p}]", ids.len(), sc.len());
        return Ok(());
    }
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let expert_gib: Option<f64> = opt("--expert-gib").and_then(|v| v.parse().ok());
    let mirror_gib: Option<f64> = opt("--mirror-gib").and_then(|v| v.parse().ok());
    let mut log = |l: String| eprintln!("[{l}]");
    let mtp = !args.iter().any(|a| a == "--no-mtp");
    // --ctx N: the session's context (default the prompt and --max: the server's sessions are larger)
    // --image PATH (repeatable): pictures before the prompt's text, for a vision model (each up to 1,024 positions)
    let images: Vec<String> = args.windows(2).filter(|w| w[0] == "--image").map(|w| w[1].clone()).collect();
    // --score FILE: that text's tokens teacher-forced after the prompt (--score-max N, 256): their perplexity; --dump
    // OUT writes every row's logits, --ref OUT compares with such a dump (another run's: a cache form against another)
    let score_max: usize = opt("--score-max").and_then(|v| v.parse().ok()).unwrap_or(256);
    let room = images.len() * 1024 + if opt("--score").is_some() { score_max } else { 0 };
    let ctx = opt("--ctx").and_then(|v| v.parse().ok()).unwrap_or(ids.len() + room + max + 1).max(ids.len() + max + 1);
    let eng = load_engine(&f, &gs, LoadOptions { expert_bytes: expert_gib.map(|x| (x * (1u64 << 30) as f64) as usize),
                                                 mirror_bytes: mirror_gib.map(|x| (x * (1u64 << 30) as f64) as usize), draft: mtp,
                                                 kv: (ctx, 1) }, &mut log)?;
    let ids = if images.is_empty() {
        ids
    } else {
        let mk = eng.image_marker().ok_or("--image: this model reads no images")?.to_string();
        let content = format!("{}{prompt}", mk.repeat(images.len()));
        let raw = tok.encode(&chat_of(&f)?(&[nextsycl_tok::Message { role: "user", content: &content, ..Default::default() }], effort, &[]));
        let mid = match tok.encode(&mk).as_slice() {
            [one] => *one,
            _ => return Err(format!("the image marker {mk} is not one token")),
        };
        let mut out = Vec::new();
        let mut k = 0;
        for t in raw {
            if t == mid && k < images.len() {
                let bytes = std::fs::read(&images[k]).map_err(|x| format!("{}: {x}", images[k]))?;
                out.extend(eng.image_tokens(&bytes).map_err(e)?);
                k += 1;
            } else {
                out.push(t);
            }
        }
        if out.len() + max + 1 > ctx {
            return Err(format!("the prompt with its pictures is {} positions; --ctx {ctx}", out.len()));
        }
        out
    };
    eprintln!("[{} on {}, {} prompt tokens, loaded in {:.1} s]", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"),
              gs.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(" + "), ids.len(), eng.load_seconds());
    let mut sess = eng.session(ctx).map_err(e)?;
    let mut none = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
    let t0 = std::time::Instant::now();
    let logits = eng.feed(&mut sess, &ids, &mut none).map_err(e)?;
    let prefill = t0.elapsed().as_secs_f64();
    // --logits-out PATH: the prompt's last position's logits as float32 (a check against a reference's)
    if let Some(p) = opt("--logits-out") {
        std::fs::write(&p, logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|x| format!("{p}: {x}"))?;
    }
    if let Some(sf) = opt("--score") {
        let text = std::fs::read_to_string(&sf).map_err(|x| format!("{sf}: {x}"))?;
        let mut sc = tok.encode(&text);
        sc.truncate(score_max);
        // --ids-out PATH: the prompt's ids and the text's, a line each (another engine's run of the same tokens)
        if let Some(p) = opt("--ids-out") {
            let line = |v: &[u32]| v.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
            std::fs::write(&p, format!("{}\n{}\n", line(&ids), line(&sc))).map_err(|x| format!("{p}: {x}"))?;
        }
        if sc.is_empty() || ids.len() + sc.len() + 1 > ctx {
            return Err(format!("--score {sf}: {} tokens after a prompt of {} in a context of {ctx}", sc.len(), ids.len()));
        }
        // row i: the distribution that predicts sc[i] (row 0 the prompt's last logits)
        let mut rows = vec![logits];
        let mut at = 0;
        while at + 1 < sc.len() {
            // 8 rows a pass: the head reads every row (decode-width products; a wider pass would expand it to half);
            // NS_SCORE_ROWS=1: one row a pass, as decode runs (up to 64: above 8 the prompt path)
            let end = (at + std::env::var("NS_SCORE_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8).clamp(1, 64)).min(sc.len() - 1);
            rows.extend(eng.forward_rows(&mut sess, &sc[at..end], end - at, &mut none).map_err(e)?);
            at = end;
        }
        let lsm = |l: &[f32]| -> Vec<f64> {
            let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
            l.iter().map(|&v| v as f64 - lse).collect()
        };
        let m = rows.len() as f64;
        let nll: f64 = rows.iter().zip(&sc).map(|(r, &y)| -lsm(r)[y as usize]).sum::<f64>() / m;
        // how often the likeliest token is the text's (a greedy run's own text: 100% from the same numerics), and where
        // it first is not
        let argmax = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
        let same: Vec<bool> = rows.iter().zip(&sc).map(|(r, &y)| argmax(r) == y as usize).collect();
        let first = same.iter().position(|s| !s).map_or("none".to_string(), |i| i.to_string());
        eprintln!("[score: {} tokens, perplexity {:.4}, the likeliest token the text's at {:.2}% (first not at row {first})]", rows.len(), nll.exp(),
                  100.0 * same.iter().filter(|s| **s).count() as f64 / m);
        if let Some(out) = opt("--dump") {
            let b: Vec<u8> = rows.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&out, b).map_err(|x| format!("{out}: {x}"))?;
        }
        if let Some(rf) = opt("--ref") {
            let raw = std::fs::read(&rf).map_err(|x| format!("{rf}: {x}"))?;
            let nv = rows[0].len();
            // --ref-from N: the reference holds rows N.. only (as many as its size says)
            let from: usize = opt("--ref-from").and_then(|v| v.parse().ok()).unwrap_or(0);
            let nref = raw.len() / (nv * 4);
            if raw.len() % (nv * 4) != 0 || from + nref > rows.len() || (from == 0 && nref != rows.len()) {
                return Err(format!("--ref {rf}: {} bytes, {} rows of {nv} logits expected from row {from}", raw.len(), rows.len() - from));
            }
            let m = nref as f64;
            let (mut kl, mut kmax, mut top1, mut rnll) = (0.0f64, 0.0f64, 0usize, 0.0f64);
            let argmax = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
            let nll: f64 = rows[from..from + nref].iter().zip(&sc[from..]).map(|(r, &y)| -lsm(r)[y as usize]).sum::<f64>() / m;
            for (j, r) in rows[from..from + nref].iter().enumerate() {
                let i = from + j;
                let refr: Vec<f32> = raw[j * nv * 4..(j + 1) * nv * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let (p, q) = (lsm(&refr), lsm(r));
                // KL(reference || this run) of the row
                let k: f64 = p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum();
                kl += k;
                kmax = kmax.max(k);
                top1 += (argmax(&refr) == argmax(r)) as usize;
                rnll -= p[sc[i] as usize];
            }
            eprintln!("[against {rf}, rows {from}..{}: KL mean {:.5} max {:.4}, top-1 the same {:.1}%, perplexity {:.4} (the reference's {:.4})]", from + nref, kl / m, kmax,
                      100.0 * top1 as f64 / m, nll.exp(), (rnll / m).exp());
        }
        return Ok(());
    }
    let seed: u64 = opt("--seed").and_then(|v| v.parse().ok()).unwrap_or(1);
    // the host's draws from the seed too (they were the same for every seed)
    let mut rng = Rng((0x9E3779B97F4A7C15u64 ^ seed.wrapping_mul(0xBF58476D1CE4E5B9)) | 1);
    let mut draw = TempSampler { temp, top_p, rng: &mut rng, seed };
    let mut dec = eng.decoder(logits, mtp);
    dec.set_context(&ids);
    let mut pending: Vec<u8> = Vec::new();
    let mut out = std::io::stdout();
    print!("<think>");
    let t1 = std::time::Instant::now();
    let mut n = 0;
    'gen: while n < max {
        for next in eng.step(&mut sess, &mut dec, &mut draw, &mut none).map_err(e)? {
            if (!ignore_eos && tok.stop.contains(&next)) || n >= max {
                break 'gen;
            }
            n += 1;
            pending.extend(tok.decode_bytes(&[next]));
            // print what is valid UTF-8 so far; keep a cut character for the next token
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(err) => err.valid_up_to(),
            };
            print!("{}", String::from_utf8_lossy(&pending[..valid]));
            pending.drain(..valid);
            let _ = out.flush();
        }
    }
    println!("{}", String::from_utf8_lossy(&pending));
    let dt = t1.elapsed().as_secs_f64();
    for line in eng.report(n) {
        eprintln!("{line}");
    }
    let x = eng.expert_stats();
    let (drafted, accepted) = dec.drafts();
    eprintln!("[prompt {} tokens in {prefill:.1} s ({:.1} tok/s); {n} generated in {dt:.1} s ({:.2} tok/s); drafts {accepted} of {drafted} accepted; experts: {} VRAM hits, {} read from host memory by prompt passes, {} swapped in ({} of them from host memory), {} prefetched ({} of them asked for)]",
              ids.len(), ids.len() as f64 / prefill, n as f64 / dt.max(1e-9), x.vram_hits, x.read_direct, x.swapped_in, x.from_host, x.prefetched, x.prefetch_used);
    let (ng_drafted, ng_accepted) = dec.lookup_drafts();
    if ng_drafted > 0 {
        eprintln!("[prompt lookup: {ng_accepted} of {ng_drafted} drafts accepted]");
    }
    // NS_EXPERT_PROFILE=FILE: the experts this run asked for blended into the file, which the next load fills VRAM from
    if let Ok(p) = std::env::var("NS_EXPERT_PROFILE") {
        match eng.save_profile(Path::new(&p)) {
            Ok(k) => eprintln!("[expert profile: {k} experts written to {p}]"),
            Err(x) => eprintln!("[expert profile {p}: {x}]"),
        }
    }
    Ok(())
}

/// `nextsycl kernels <model.gguf> [--gpu N]`: each stored weight type of the file through the decode kernels against
/// the exact path (expand to float32, multiply), on a real matrix of that type; and the decode kernel's rate.
fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let kinds = engines();
    let k = nextsycl_llm::kind_for(&kinds, &f)?;
    (k.kernels.ok_or(format!("{} has no kernel test", k.name))?)(model, gpu)
}

/// `nextsycl spec-check`: greedy decode one token at a time (the reference), then the same tokens as 2-row verify
/// passes each rolled back to its first row; every row's logits against the reference's at that position.
/// `nextsycl batch-check <model.gguf> [--prompts "a|b|c"] [--n N] [--gpu 1,0]`: several conversations decoded
/// together (`forward_batch`) against each decoded alone - the same greedy tokens, the logits' largest difference -
/// and the speed of both.
fn batch_check(args: &[String]) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("batch-check <model.gguf> ...")?;
    let prompts: Vec<String> = opt("--prompts")
        .unwrap_or_else(|| "Write a haiku about rivers.|Explain TCP slow start.|List five prime numbers and why.|What is a monad?".into())
        .split('|').map(String::from).collect();
    let n: usize = opt("--n").and_then(|v| v.parse().ok()).unwrap_or(32);
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "1,0".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}"))).collect::<Result<_, _>>()?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = nextsycl_tok::Tokenizer::from_gguf(&f)?;
    let chat = chat_of(&f)?;
    let ids: Vec<Vec<u32>> = prompts.iter()
        .map(|p| tok.encode(&chat(&[nextsycl_tok::Message { role: "user", content: p, ..Default::default() }], nextsycl_tok::Effort::Low, &[])))
        .collect();
    let ctx = ids.iter().map(|v| v.len()).max().unwrap_or(0) + n + 8;
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let mut log = |l: String| eprintln!("[{l}]");
    let mtp = args.iter().any(|a| a == "--mtp"); // the draft block loaded (it is not used here; its experts take store slots)
    let eng = load_engine(&f, &gs, LoadOptions { expert_bytes: None, mirror_bytes: None, draft: mtp, kv: (ctx, prompts.len() + 1) }, &mut log)?;
    let mut none = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i) as u32;
    // alone: each conversation, one token a pass
    let mut solo_toks: Vec<Vec<u32>> = Vec::new();
    let mut solo_logits: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut solo_s = 0.0;
    // --no-solo: the batch alone (its profile; the tokens are then taken greedily from the batch itself)
    let solo = !args.iter().any(|a| a == "--no-solo");
    for p in ids.iter().filter(|_| solo) {
        let mut s = eng.session(ctx).map_err(e)?;
        let mut l = eng.feed(&mut s, p, &mut none).map_err(e)?;
        let (mut ts, mut ls) = (Vec::new(), Vec::new());
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let t = argmax(&l);
            ts.push(t);
            l = eng.forward(&mut s, &[t], &mut none).map_err(e)?;
            ls.push(l.clone());
        }
        solo_s += t0.elapsed().as_secs_f64();
        solo_toks.push(ts);
        solo_logits.push(ls);
    }
    // together
    let mut sess: Vec<nextsycl_llm::Session> = Vec::new();
    let mut last: Vec<Vec<f32>> = Vec::new();
    for p in &ids {
        let mut s = eng.session(ctx).map_err(e)?;
        last.push(eng.feed(&mut s, p, &mut none).map_err(e)?);
        sess.push(s);
    }
    let mut worst = 0f32;
    let mut differ = 0;
    let t0 = std::time::Instant::now();
    for step in 0..n {
        let toks: Vec<u32> = last.iter().map(|l| argmax(l)).collect();
        if solo {
            for (b, t) in toks.iter().enumerate() {
                if *t != solo_toks[b][step] {
                    differ += 1;
                }
            }
        }
        // each session takes the token its own run took, so the comparison stays aligned
        let feed: Vec<u32> = if solo { (0..ids.len()).map(|b| solo_toks[b][step]).collect() } else { toks.clone() };
        let mut refs: Vec<&mut nextsycl_llm::Session> = sess.iter_mut().collect();
        last = eng.forward_batch(&mut refs, &feed, &mut none).map_err(e)?;
        for (b, l) in last.iter().enumerate().filter(|_| solo) {
            let r = &solo_logits[b][step];
            worst = worst.max(l.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
        }
    }
    let batch_s = t0.elapsed().as_secs_f64();
    let b = ids.len();
    println!("{b} conversations, {n} tokens each: alone {:.2} tok/s (each, {:.1} s in all); together {:.2} tok/s in all ({:.2} each)",
             n as f64 * b as f64 / solo_s, solo_s, n as f64 * b as f64 / batch_s, n as f64 / batch_s);
    println!("greedy tokens different in {differ} of {} steps; the logits' largest difference {worst:.3e}", n * b);
    let x = eng.expert_stats();
    println!("experts (both runs): {} VRAM hits, {} swapped in, {} prefetched ({} asked for)", x.vram_hits, x.swapped_in, x.prefetched, x.prefetch_used);
    for line in eng.report(n * b) {
        println!("{line}");
    }
    Ok(())
}

fn spec_check(args: &[String]) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("spec-check <model.gguf> --prompt ...")?;
    let prompt = match opt("--prompt-file") {
        Some(f) => std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?,
        None => opt("--prompt").ok_or("--prompt TEXT or --prompt-file PATH")?,
    };
    let n: usize = opt("--n").and_then(|v| v.parse().ok()).unwrap_or(32);
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "0,1".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}"))).collect::<Result<_, _>>()?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = nextsycl_tok::Tokenizer::from_gguf(&f)?;
    let ids = tok.encode(&chat_of(&f)?(&[nextsycl_tok::Message { role: "user", content: &prompt, ..Default::default() }], nextsycl_tok::Effort::Low, &[]));
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let mut log = |l: String| eprintln!("[{l}]");
    let eng = load_engine(&f, &gs, LoadOptions { expert_bytes: None, mirror_bytes: None, draft: true, kv: (ids.len() + n + 4, 2) }, &mut log)?;
    let mut none = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    // the margin between the best two logits
    let margin = |v: &[f32]| {
        let mut s: Vec<f32> = v.to_vec();
        s.sort_by(|a, b| b.total_cmp(a));
        s[0] - s[1]
    };
    if args.iter().any(|a| a == "--prompt-decode") {
        // the prompt pass's last logits against the prompt fed a token at a time (the decode path)
        let mut a = eng.session(ids.len() + 4).map_err(e)?;
        let got = eng.feed(&mut a, &ids, &mut none).map_err(e)?;
        let mut b = eng.session(ids.len() + 4).map_err(e)?;
        let mut want = Vec::new();
        for t in &ids {
            want = eng.forward(&mut b, &[*t], &mut none).map_err(e)?;
        }
        let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let d = got.iter().zip(want.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let top = |v: &[f32]| {
            let mut i: Vec<usize> = (0..v.len()).collect();
            i.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
            i[..5].iter().map(|&k| format!("{k}:{:.3}", v[k])).collect::<Vec<_>>().join(" ")
        };
        println!("{} prompt tokens: prompt pass against a token at a time: cosine {:.6}, max |diff| {d:.4}, top1 {}", ids.len(), dot / (na * nb).max(1e-30),
                 if argmax(&got) == argmax(&want) { "same" } else { "FLIP" });
        println!("  prompt pass:    {}\n  token at a time: {}", top(&got), top(&want));
        return Ok(());
    }
    let mut a = eng.session(ids.len() + n + 4).map_err(e)?;
    let mut la = eng.feed(&mut a, &ids, &mut none).map_err(e)?;
    let mut toks = Vec::new();
    let mut refs = Vec::new(); // refs[i]: the logits after toks[i]
    for _ in 0..n {
        let t = argmax(&la) as u32;
        toks.push(t);
        la = eng.forward(&mut a, &[t], &mut none).map_err(e)?;
        refs.push(la.clone());
    }
    if args.iter().any(|a| a == "--prefill") {
        // the one-token passes' logits against a prompt pass over the same tokens, at a quarter, half, three quarters
        // and all of them: the decode path's caches (a window, compressed keys, indexer keys) against the prompt path's
        println!("{:>6} {:>10} {:>12} {:>6}", "tokens", "cosine", "max |diff|", "top1");
        for m in [n / 4, n / 2, 3 * n / 4, n].into_iter().filter(|m| *m > 0) {
            let mut c = eng.session(ids.len() + n + 4).map_err(e)?;
            let all: Vec<u32> = ids.iter().chain(&toks[..m]).copied().collect();
            let got = eng.feed(&mut c, &all, &mut none).map_err(e)?;
            let want = &refs[m - 1];
            let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
            let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let d = got.iter().zip(want.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            println!("{:>6} {:>10.6} {:>12.5} {:>6}", all.len(), dot / (na * nb).max(1e-30), d, if argmax(&got) == argmax(want) { "same" } else { "FLIP" });
        }
        return Ok(());
    }
    let mut b = eng.session(ids.len() + n + 4).map_err(e)?;
    eng.feed(&mut b, &ids, &mut none).map_err(e)?;
    if args.iter().any(|a| a == "--layers") {
        // a verify pass's row against a one-token pass from the same state, step by step; --at N: N one-token passes
        // first
        let at: usize = opt("--at").and_then(|v| v.parse().ok()).unwrap_or(0);
        for &t in &toks[..at] {
            eng.forward(&mut b, &[t], &mut none).map_err(e)?;
        }
        let toks = &toks[at..];
        let mut c = eng.session(ids.len() + n + 4).map_err(e)?;
        eng.copy_session(&mut c, &b).map_err(e)?;
        let mut one: Vec<(String, Vec<f32>)> = Vec::new();
        let mut keep = |name: &str, x: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> {
            one.push((name.to_string(), x.to_f32()?));
            Ok(())
        };
        // --rows R --row k: row k of an R-row pass against a one-token pass at its position (k one-token passes first)
        let width: usize = opt("--rows").and_then(|v| v.parse().ok()).unwrap_or(2).clamp(2, eng.max_verify());
        let row: usize = opt("--row").and_then(|v| v.parse().ok()).unwrap_or(0).min(width - 1);
        for &t in &toks[..row] {
            eng.forward(&mut c, &[t], &mut none).map_err(e)?;
        }
        eng.forward(&mut c, &[toks[row]], &mut keep).map_err(e)?;
        let mut two: Vec<(String, Vec<f32>)> = Vec::new();
        let mut keep2 = |name: &str, x: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> {
            two.push((name.to_string(), x.to_f32()?));
            Ok(())
        };
        let mut d = eng.session(ids.len() + n + 4).map_err(e)?;
        eng.copy_session(&mut d, &b).map_err(e)?;
        eng.forward_rows(&mut d, &toks[..width], 1, &mut keep2).map_err(e)?;
        // matched by name (a pass of several rows taps some steps it splits by row only once, or not at all)
        let by_name: std::collections::HashMap<&str, &Vec<f32>> = two.iter().map(|(k, v)| (k.as_str(), v)).collect();
        for (na, va) in &one {
            let Some(vb) = by_name.get(na.as_str()) else { continue };
            // row `row` of the R-row tensor (rows are outermost)
            let w = va.len();
            if na.starts_with("result") || vb.len() < w * width {
                continue;
            }
            let rb = &vb[row * w..(row + 1) * w];
            let dd = va.iter().zip(rb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let mx = va.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-20);
            println!("{na:<24} {:.3e}", dd / mx);
        }
        return Ok(());
    }
    println!("{:>4} {:>4} {:>12} {:>10} {:>6} {:>8}", "pos", "row", "max |diff|", "max |ref|", "top1", "margin");
    let (mut worst, mut flips) = (0f32, 0);
    // --rows R: verify passes of R tokens (2 = one draft; up to 5 with NS_NGRAM=4's snapshots)
    let asked: usize = opt("--rows").and_then(|v| v.parse().ok()).unwrap_or(2);
    let width = asked.clamp(2, eng.max_verify().max(2));
    if width < asked {
        eprintln!("[--rows {asked}: this engine verifies {width} rows at most as configured (NS_NGRAM=4 or NS_DRAFTS=2 widen it)]");
    }
    for i in 0..n + 1 - width {
        let rows = eng.forward_rows(&mut b, &toks[i..i + width], width, &mut none).map_err(e)?;
        for (r, (got, want)) in rows.iter().zip(&refs[i..i + width]).enumerate() {
            let d = got.iter().zip(want.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let mx = want.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let same = argmax(got) == argmax(want);
            worst = worst.max(d / mx);
            flips += usize::from(!same);
            if i < 4 || !same || i % 8 == 0 || d > 0.0 {
                println!("{:>4} {:>4} {:>12.5} {:>10.3} {:>6} {:>8.4}", i, r, d, mx, if same { "same" } else { "FLIP" }, margin(want));
            }
        }
        eng.rollback(&mut b, 1).map_err(e)?;
    }
    println!("worst max|diff| / max|ref| {worst:.2e}; top-1 differs in {flips} of {} rows ({width}-row passes)", width * (n + 1 - width));
    Ok(())
}

/// `nextsycl serve`: load, then answer on HTTP (serve.rs).
fn serve_cmd(args: &[String]) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("serve <model.gguf> ...")?;
    let g = opt("--gpu").unwrap_or_else(|| "0".into());
    let gpus: Vec<usize> = if g == "all" {
        // the one that computes most last: it takes the head and the draft block (measured on a B65 + B70: the
        // B70 last reads prompts 6% faster; decode the same)
        nextsycl_core::gpus_weakest_first().map_err(e)?
    } else {
        g.split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}"))).collect::<Result<_, _>>()?
    };
    let addr = format!("{}:{}", opt("--host").unwrap_or_else(|| "127.0.0.1".into()), opt("--port").unwrap_or_else(|| "8085".into()));
    // --ctx N (every session N tokens) or N,M,... (a session each: one long, others short - a 256K session's attention
    // cache is 3.4 GiB of VRAM)
    let ctx_list: Vec<usize> = opt("--ctx").unwrap_or_else(|| "8192".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    if ctx_list.is_empty() || ctx_list.iter().any(|c| *c < 256) {
        return Err("--ctx N or N,M,... (tokens, at least 256 each)".into());
    }
    let effort = nextsycl_tok::Effort::parse(&opt("--effort").unwrap_or_else(|| "low".into())).ok_or("--effort low|medium|high|max")?;
    let gib_opt = |k: &str| opt(k).and_then(|v| v.parse::<f64>().ok()).map(|x| (x * (1u64 << 30) as f64) as usize);
    // the model's file lives as long as the server
    let f: &'static Gguf = Box::leak(Box::new(Gguf::open(Path::new(model)).map_err(|e| e.0)?));
    let name = opt("--name").unwrap_or_else(|| Path::new(model).file_stem().map_or("model".into(), |s| s.to_string_lossy().to_lowercase()));
    let tok = nextsycl_tok::Tokenizer::from_gguf(f)?;
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let mut log = |l: String| eprintln!("[{l}]");
    let mtp = !args.iter().any(|a| a == "--no-mtp");
    // the prompt cache lives in host memory: what it may take comes out of the expert mirror's default share
    let cache_mib: usize = opt("--prompt-cache-mib").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let cache = cache_mib << 20;
    let mirror = gib_opt("--mirror-gib").or_else(|| {
        let avail = std::fs::read_to_string("/proc/meminfo").ok()?
            .lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<usize>().ok()? * 1024;
        Some(avail.saturating_sub((10 << 30) + cache))
    });
    // the sessions decoding together (their attention caches come out of the expert store)
    let parallel: usize = opt("--parallel").and_then(|v| v.parse().ok()).unwrap_or(2).clamp(1, 8);
    let slot_ctx: Vec<usize> = if ctx_list.len() > 1 { ctx_list.clone() } else { vec![ctx_list[0]; parallel] };
    if slot_ctx.len() > 8 {
        return Err("at most 8 sessions".into());
    }
    // the attention reserve is per token: the sessions' tokens in all (a pool row each besides)
    let kv = (slot_ctx.iter().sum::<usize>() + 4 * slot_ctx.len(), 1);
    let eng = load_engine(f, &gs, LoadOptions { expert_bytes: gib_opt("--expert-gib"), mirror_bytes: mirror, draft: mtp, kv }, &mut log)?;
    eprintln!("[{} loaded on {} in {:.1} s]", name, gs.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(" + "), eng.load_seconds());
    let cors: Vec<String> = opt("--cors").unwrap_or_default().split([',', ' ']).filter(|o| !o.is_empty()).map(String::from).collect();
    let keep: usize = opt("--keep-requests").and_then(|v| v.parse().ok()).unwrap_or(100);
    // what a request without max_tokens may make: --max-tokens N (0 or none: to the end of the context)
    let default_max = opt("--max-tokens").and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0);
    // the prompt cache's disk tier: --cache-dir DIR [--cache-disk-gib G (32)]
    let mut pc = cache::PromptCache::new(cache);
    if let Some(dir) = opt("--cache-dir") {
        let g: f64 = opt("--cache-disk-gib").and_then(|v| v.parse().ok()).unwrap_or(32.0);
        let hours: f64 = opt("--cache-ttl-hours").and_then(|v| v.parse().ok()).unwrap_or(24.0);
        // what a checkpoint must match to fit these sessions: the model file, the cache's form, the draft block
        let size = std::fs::metadata(model).map_or(0, |m| m.len());
        let fp = format!("{model} {size} {}", eng.cache_fingerprint());
        pc = pc.with_disk(std::path::PathBuf::from(dir), (g * (1u64 << 30) as f64) as usize, &fp,
                          std::time::Duration::from_secs_f64(hours.max(0.0) * 3600.0))?;
    }
    let srv = std::sync::Arc::new(serve::Server::new(eng, tok, chat_of(f)?, name, slot_ctx, effort, pc, cors, keep, default_max)?);
    srv.run(&addr, opt("--socket").map(std::path::PathBuf::from))
}

/// `nextsycl llm <command>`: the language-model commands (the old top-level names still work)
fn llm(cfg: &nextsycl_models::Config, args: &[String]) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("start") => service::start(cfg, rest),
        Some("stop") => service::stop(cfg),
        Some("logs") => service::logs(cfg, rest),
        Some("status") => client::status(rest),
        Some("ps") => client::ps(rest),
        Some("inspect") => client::inspect(rest),
        Some("bench") => bench::bench(rest),
        Some("cache") => client::cache(rest),
        Some("chat") => client::chat(rest),
        // a registered model in the foreground (its container, its log here), or a file in this process (inside the
        // image: what `start` runs)
        Some("serve") => match rest.first() {
            Some(m) if Path::new(m).exists() => serve_cmd(args),
            Some(_) => service::serve_foreground(cfg, rest),
            None => Err("nextsycl llm serve <model>".into()),
        },
        Some("engines") => {
            list_engines(engines().iter().map(|k| (k.archs.join(", "), k.name, llm_options(k))));
            Ok(())
        }
        Some("selftest") => selftest("llm", rest, nextsycl_llm_example::selftest),
        Some("info") if args.len() == 2 => info(Path::new(&args[1])),
        Some("tokenize") if args.len() == 3 => tokenize(Path::new(&args[1]), &args[2]),
        Some("generate") => generate(args),
        Some("spec-check") => spec_check(args),
        Some("batch-check") => batch_check(args),
        // the files an engine makes beside a model (as after a pull: deepseek4's draft block from its shard)
        Some("prepare") if args.len() == 2 => {
            let f = Gguf::open(Path::new(&args[1])).map_err(|e| e.0)?;
            match nextsycl_llm::kind_for(&engines(), &f)?.prepare {
                Some(p) => p(&f),
                None => Ok(()),
            }
        }
        Some("kernels") if args.len() >= 2 => {
            let gpu = args.iter().position(|a| a == "--gpu").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0);
            kernels(Path::new(&args[1]), gpu)
        }
        Some("check") if args.len() >= 3 => {
            let gpus: Vec<usize> = args.iter().position(|a| a == "--gpu").and_then(|i| args.get(i + 1)).map_or(vec![0], |v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            check(Path::new(&args[1]), Path::new(&args[2]), &gpus)
        }
        _ => Err(USAGE.into()),
    }
}


/// `nextsycl <kind> selftest [--gpu N]`: the kind's kernel library opens, an engine's own symbol binds, a kernel runs
/// on the GPU and its result comes back right (the kind's template engine's kernel)
fn selftest(kind: &'static str, args: &[String], run: fn(&std::sync::Arc<nextsycl_core::Gpu>) -> nextsycl_core::Result<Vec<f32>>) -> Result<(), String> {
    nextsycl_core::use_kind(kind);
    let i: usize = args.iter().position(|a| a == "--gpu").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0);
    let gpu = nextsycl_core::Gpu::open(i).map_err(|e| e.0)?;
    let y = run(&gpu).map_err(|e| e.0)?;
    println!("{kind}: libnextsycl-{kind}.so on {} (GPU {i}): [1, 2, 3, 4] x 2 = {y:?} - ok", gpu.name);
    Ok(())
}

/// The --opt-NAME [VALUE]s given on this command line
pub(crate) static ENGINE_OPTS: std::sync::OnceLock<nextsycl_core::options::Given> = std::sync::OnceLock::new();

pub(crate) fn engine_opts() -> &'static nextsycl_core::options::Given {
    ENGINE_OPTS.get_or_init(Default::default)
}

/// The variables the --opt-NAMEs set for an engine that takes `declared` (an error lists its options)
pub(crate) fn opt_env(declared: &[nextsycl_core::EngineOption], at: nextsycl_core::At, engine: &str) -> Result<Vec<(String, String)>, String> {
    nextsycl_core::options::resolve(engine_opts(), declared, at, engine)
}

/// The engines and, under each, the options it takes (`--opt-NAME`)
fn list_engines(rows: impl Iterator<Item = (String, &'static str, Vec<nextsycl_core::EngineOption>)>) {
    println!("{:<24} ENGINE", "ARCHITECTURE");
    for (a, n, opts) in rows {
        println!("{a:<24} {n}");
        if !opts.is_empty() {
            println!("{}", nextsycl_core::options::usage(&opts));
        }
    }
    println!("\n--opt-NAME VALUE on any command (or NAME the variable it sets) reaches the engine; `options` in an API request too");
}

/// An llm engine's options and the runtime's common ones
fn llm_options(k: &EngineKind) -> Vec<nextsycl_core::EngineOption> {
    k.options.iter().chain(nextsycl_llm::COMMON_OPTIONS).copied().collect()
}


extern "C" {
    fn signal(sig: i32, handler: usize) -> usize;
}

fn main() -> ExitCode {
    // a closed pipe (`nextsycl models list | head`) ends the program quietly, as other tools do, not with a panic:
    // Rust ignores SIGPIPE by default
    // SAFETY: SIGPIPE (13) back to its default action (0), before any thread starts.
    unsafe { signal(13, 0) };
    let args: Vec<String> = std::env::args().skip(1).collect();
    // every --opt-NAME [VALUE]: the engine's own options, checked and forwarded where it loads (nextsycl_core::options)
    let _ = ENGINE_OPTS.set(nextsycl_core::options::given(&args));
    let cfg = config::Config::load();
    // a command that runs the kernels here, on a host without oneAPI: the same command inside the image
    if let Some(r) = container::rerun_in_image(&cfg, &args) {
        return match r {
            Ok(code) => ExitCode::from(code.clamp(0, 255) as u8),
            Err(e) => {
                eprintln!("nextsycl: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let _ = client::SERVER.set(http::Target::Unix(cfg.socket()));
    let rest = args.get(1..).unwrap_or(&[]);
    let r = match args.first().map(String::as_str) {
        Some("llm") => llm(&cfg, &args[1..]),
        Some("image") => image::cmd(&cfg, &args[1..], |r| selftest("image", r, nextsycl_image_example::selftest)),
        Some("video") => video::cmd(&cfg, &args[1..], |r| selftest("video", r, nextsycl_video_example::selftest)),
        Some("audio") => audio::cmd(&cfg, &args[1..], |r| selftest("audio", r, nextsycl_audio_example::selftest)),
        Some("model3d" | "3d") => model3d::cmd(&cfg, &args[1..], |r| selftest("model3d", r, nextsycl_model3d_example::selftest)),
        // the llm commands' names from before the kinds (nextsycl start = nextsycl llm start, ...)
        Some("start") => service::start(&cfg, rest),
        Some("models") if rest.first().is_some_and(|a| a == "move") => drives::move_cmd(&cfg, rest),
        Some("drives") => drives::cmd(&cfg, rest),
        Some("models") => models::cmd(&cfg, rest, &|g: &Gguf| {
            let all = engines();
            let k = nextsycl_llm::kind_for(&all, g)?;
            if let Some(p) = k.prepare {
                p(g)?;
            }
            Ok(k.name.to_string())
        }),
        Some("stop") => service::stop(&cfg),
        Some("logs") => service::logs(&cfg, rest),
        Some("status") => client::status(rest),
        Some("ps") => client::ps(rest),
        Some("inspect") => client::inspect(rest),
        Some("bench") => bench::bench(rest),
        Some("cache") => client::cache(rest),
        Some("chat") => client::chat(rest),
        Some("version") => {
            println!("nextsycl {VERSION}");
            Ok(())
        }
        Some("help") | Some("--help") | Some("-h") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("info") if args.len() == 2 => info(Path::new(&args[1])),
        Some("gpus") => gpus(),
        Some("switch") => host::switch(rest),
        Some("gpustat") => host::gpustat(rest),
        Some("maintenance") => host::maintenance(&cfg, rest),
        Some("tokenize") if args.len() == 3 => tokenize(Path::new(&args[1]), &args[2]),
        Some("generate") => generate(&args),
        // `serve FILE` (the llm server's old name), else the box at a glance
        Some("serve") if args.get(1).is_some_and(|a| Path::new(a).exists()) => serve_cmd(&args),
        Some("serve") => host::home(&cfg, rest),
        Some("spec-check") => spec_check(&args),
        Some("batch-check") => batch_check(&args),
        Some("rows-check") => rows_check(&args),
        Some("kernels") if args.len() >= 2 => {
            let gpu = args.iter().position(|a| a == "--gpu").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0);
            kernels(Path::new(&args[1]), gpu)
        }
        Some("check") if args.len() >= 3 => {
            let gpus: Vec<usize> = args.iter().position(|a| a == "--gpu").and_then(|i| args.get(i + 1)).map_or(vec![0], |v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            check(Path::new(&args[1]), Path::new(&args[2]), &gpus)
        }
        _ => Err(USAGE.into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nextsycl: {e}");
            ExitCode::FAILURE
        }
    }
}
