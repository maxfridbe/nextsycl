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
mod image;
mod service;

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

const USAGE: &str = "nextsycl - language, image and video models on Intel Arc GPUs (Rust + SYCL)

every kind:
  nextsycl models [list] [--json]           the registry (NS_REGISTRY): every model this machine serves, its kind
  nextsycl models add <id> <file.gguf> [--title T] [--gpu 0[,1] | all] [--ctx N[,M...]] [--set NAME=VALUE]...
                      [--no-tools] [--no-tasks] [--disabled]
  nextsycl models search [TEXT] [--kind llm|image|video|lora]     the catalog of supported models
  nextsycl models pull <id>... [--dir DIR] [--from DIR]... [--verify] [--again]
                                its files downloaded (resumed when run again; each checked by SHA-256) or linked
                                (the same file already here; --from DIR: copies elsewhere on the machine), registered
  nextsycl models pull <id> <url | hf:org/repo/path/file.gguf> [--dir DIR] [add's options]
                                a file outside the catalog (every shard of a split file); then added
  nextsycl models remove <id> [--files]     --files: its files deleted too
  nextsycl models enable <id> | disable <id>
  nextsycl gpus                 each GPU in its own context: memory, copy rates, GPU to GPU, host RAM unaffected
  nextsycl version

language models (nextsycl llm ...; the names without `llm` still work):
  nextsycl llm start <id>       a registered model as a service (a container; the model stays loaded on the GPUs);
                                flags beside it win: [--gpu N ...] [--port N] [--host H] [--ctx N | N,M,...] [--name ID]
                                [--effort low|high|max] [--prompt-cache-mib N] [--no-mtp]
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
                 [--effort low|high|max] [--max N] [--temp T] [--top-p P] [--seed S] [--gpu N[,M]] [--no-mtp]
  nextsycl llm info <model.gguf>      the architecture and geometry, every tensor checked by role, bytes by group
  nextsycl llm tokenize <model.gguf> <text>
  nextsycl llm check <model.gguf> <dump dir> [--gpu N[,M]]    the forward pass against a reference dump
  nextsycl llm spec-check <model.gguf> --prompt TEXT [--n N] [--rows R (2..5)] [--gpu 0,1]
                                verify passes against one-token decode, logits compared
  nextsycl llm kernels <model.gguf> [--gpu N]   each weight type's decode kernel against the exact path

images (nextsycl image ...):
  nextsycl image engines        the architectures this build serves (gen, edit, start, serve come with the first)
  nextsycl image selftest [--gpu N]  the image kernel library, an engine's own symbol, a kernel on the GPU

video (nextsycl video ...):
  nextsycl video engines        the architectures this build serves (H3's commands move here)
  nextsycl video selftest [--gpu N]  the video kernel library, an engine's own symbol, a kernel on the GPU

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
  NS_IMAGE, NS_CONTAINER_ENGINE   the image with the oneAPI runtime (localhost/h3-build) and podman / docker
  NS_MOUNTS        more read-only directories for the container, host:inside[,host:inside] (another model folder, ...)
  NS_QW_*          the qwen4exp engine's settings, passed to the server (docs/engines.md): NS_QW_MTP (the draft
                   layer), NS_QW_EXPERT_PROFILE, NS_QW_CVEC / _LAYERS / _MODE / _DIR (a control vector), NS_QW_CHUNK";

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The llm engines this program has, one a model architecture (llm/<arch>)
pub(crate) fn engines() -> Vec<EngineKind> {
    vec![nextsycl_llm_glm5next::kind(), nextsycl_llm_qwen4exp::kind(), nextsycl_llm_example::kind()]
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

/// `nextsycl generate <model.gguf> --prompt TEXT [--effort low|high|max] [--max N] [--temp T] [--top-p P] [--gpu N]`
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
    let effort = nextsycl_tok::Effort::parse(&opt("--effort").unwrap_or_else(|| "low".into())).ok_or("--effort low|high|max")?;
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
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let expert_gib: Option<f64> = opt("--expert-gib").and_then(|v| v.parse().ok());
    let mirror_gib: Option<f64> = opt("--mirror-gib").and_then(|v| v.parse().ok());
    let mut log = |l: String| eprintln!("[{l}]");
    let mtp = !args.iter().any(|a| a == "--no-mtp");
    // --ctx N: the session's context (default the prompt and --max: the server's sessions are larger)
    let ctx = opt("--ctx").and_then(|v| v.parse().ok()).unwrap_or(ids.len() + max + 1).max(ids.len() + max + 1);
    let eng = load_engine(&f, &gs, LoadOptions { expert_bytes: expert_gib.map(|x| (x * (1u64 << 30) as f64) as usize),
                                                 mirror_bytes: mirror_gib.map(|x| (x * (1u64 << 30) as f64) as usize), draft: mtp,
                                                 kv: (ctx, 1) }, &mut log)?;
    eprintln!("[{} on {}, {} prompt tokens, loaded in {:.1} s]", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"),
              gs.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(" + "), ids.len(), eng.load_seconds());
    let mut sess = eng.session(ctx).map_err(e)?;
    let mut none = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
    let t0 = std::time::Instant::now();
    let logits = eng.feed(&mut sess, &ids, &mut none).map_err(e)?;
    let prefill = t0.elapsed().as_secs_f64();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let seed: u64 = opt("--seed").and_then(|v| v.parse().ok()).unwrap_or(1);
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
    let effort = nextsycl_tok::Effort::parse(&opt("--effort").unwrap_or_else(|| "low".into())).ok_or("--effort low|high|max")?;
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

/// The video engines this program has (video/<arch>)
fn video_engines() -> Vec<nextsycl_video::VideoKind> {
    vec![nextsycl_video_example::kind()]
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

/// `nextsycl video <command>`: the video commands - for now the engines; H3's commands move here next
fn video(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("engines") => {
            list_engines(video_engines().iter().map(|k| (k.archs.join(", "), k.name, k.options.to_vec())));
            Ok(())
        }
        Some("selftest") => selftest("video", &args[1..], nextsycl_video_example::selftest),
        _ => Err("nextsycl video engines | selftest   (gen, serve, start, ... come with the H3 engine: CONTRIBUTING.md)".into()),
    }
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
    let _ = client::SERVER.set(http::Target::Unix(cfg.socket()));
    let rest = args.get(1..).unwrap_or(&[]);
    let r = match args.first().map(String::as_str) {
        Some("llm") => llm(&cfg, &args[1..]),
        Some("image") => image::cmd(&cfg, &args[1..], |r| selftest("image", r, nextsycl_image_example::selftest)),
        Some("video") => video(&args[1..]),
        // the llm commands' names from before the kinds (nextsycl start = nextsycl llm start, ...)
        Some("start") => service::start(&cfg, rest),
        Some("models") => models::cmd(&cfg, rest, &|g: &Gguf| nextsycl_llm::kind_for(&engines(), g).map(|k| k.name.to_string())),
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
        Some("tokenize") if args.len() == 3 => tokenize(Path::new(&args[1]), &args[2]),
        Some("generate") => generate(&args),
        Some("serve") => serve_cmd(&args),
        Some("spec-check") => spec_check(&args),
        Some("batch-check") => batch_check(&args),
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
