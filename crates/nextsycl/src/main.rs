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
mod cache;
mod client;
mod config;
mod container;
mod http;
mod serve;
mod service;
mod telemetry;

use std::path::Path;
use std::process::ExitCode;

use ns_gguf::Gguf;
use ns_model::glm5next::{Group, Model};

/// yy.mmdd.### from git (version.sh), handed in by build.sh; "dev" for a build without it
pub const VERSION: &str = match option_env!("NS_VERSION") {
    Some(v) => v,
    None => "dev",
};

const USAGE: &str = "nextsycl - GLM-5.3-Flash on Intel Arc GPUs (Rust + SYCL)

the server (a container; the model stays loaded on the GPUs):
  nextsycl start [--gpu N ...] [--model PATH] [--port N] [--host H] [--ctx N] [--name ID] [--effort low|high|max]
                 [--prompt-cache-mib N] [--no-mtp]
                                the OpenAI API on NS_HOST:NS_PORT (default 127.0.0.1:8085), control on a Unix socket
  nextsycl stop                 gracefully: the request running finishes, then the server ends

the server (over its socket):
  nextsycl status [--no-stream] live, like docker stats: the model, its GPUs, the request running, the prompt cache
  nextsycl ps [-a]              the request running and the last ones
  nextsycl inspect <id>         one request (an ID from ps) as JSON: settings, messages, timings, previews
  nextsycl bench [--sizes 20,2185,8000,40000] [--new 256] [--parallel 1,2,4] [--out DIR]
                                benchy v1 against the running server, and several requests at once (matrix.md)
  nextsycl cache [ls | clear]   the prompt cache's checkpoints
  nextsycl chat <text> [--effort E] [--max N] [--temp T]
                                one request, streamed
  nextsycl logs [--no-follow]   the server's log, followed
  nextsycl version

in this process (inside the image: the kernels need the oneAPI runtime):
  nextsycl serve <model.gguf> [--gpu 0,1 | all] [--host H] [--port N] [--name ID] [--ctx N] [--effort E] [--socket PATH]
                 [--expert-gib G] [--mirror-gib G] [--no-mtp] [--prompt-cache-mib N (4096; 0 = off)] [--cors ORIGINS]
                 [--keep-requests N (100)] [--parallel N (2: requests decoded together)]
                 [--max-tokens N (a request without max_tokens: N; default the rest of the context)]
                                the server in the foreground (what start runs)
  nextsycl generate <model.gguf> --prompt TEXT | --prompt-file PATH [--effort low|high|max] [--max N] [--temp T] [--top-p P] [--gpu N[,M]]
                    [--expert-gib G] [--mirror-gib G] [--no-mtp]
  nextsycl info <model.gguf>    the architecture and geometry, every tensor checked by role, bytes by group
  nextsycl gpus                 each GPU in its own context: memory, copy rates, GPU to GPU, host RAM unaffected
  nextsycl tokenize <model.gguf> <text>
  nextsycl check <model.gguf> <dump dir> [--gpu N[,M]]
                                the forward pass on a reference dump's prompt, every step compared, the next token
  nextsycl spec-check <model.gguf> --prompt TEXT [--n N] [--gpu 0,1]
                                verify passes (2 rows, then a rollback to 1) against one-token decode, logits compared
  nextsycl kernels <model.gguf> [--gpu N]   each weight type's decode kernel against the exact path

settings (environment, or NAME=value lines in nextsycl.conf beside the repository or ~/.config/nextsycl.conf):
  NS_MODELS        host directory with the model files, seen as /models                  (required for start)
  NS_MODEL         the model as seen in the container (default /models/glm53-iq2/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf)
  NS_GPUS          GPUs, e.g. \"0 1\" (default all)
  NS_HOST, NS_PORT the OpenAI API and /api/chat (default 127.0.0.1, 8085; 0.0.0.0 = the network, no password)
  NS_CORS          web pages that may call the API from a browser, beyond the loopback ones, e.g.
                   \"http://studio:8095\" (comma-separated; * = any)
  NS_CTX           tokens of context (default 65536)      NS_NAME   the model id clients see (glm-5.3-flash-uncensored)
  NS_EFFORT        default reasoning effort (low)        NS_NO_MTP=1   decode without the draft block
  NS_PROMPT_CACHE_MIB   host memory for the prompt cache's checkpoints (default 4096; 0 = off)
  NS_KEEP_REQUESTS ended requests the server keeps for ps and inspect (default 100)
  NS_PARALLEL      requests decoded together, each with a session of its own (default 2; 1 = one at a time)
  NS_MAX_TOKENS    the tokens a request without max_tokens may make (default: to the end of the context)
  NS_SOCKET_DIR    where the control socket lives (default $XDG_RUNTIME_DIR/nextsycl)
  NS_IMAGE, NS_CONTAINER_ENGINE   the image with the oneAPI runtime (localhost/h3-build) and podman / docker";

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn info(path: &Path) -> Result<(), String> {
    let f = Gguf::open(path).map_err(|e| e.0)?;
    println!("file     : {} ({} file{}, {:.2} GiB of tensors, {} tensors)", path.display(), f.paths.len(), if f.paths.len() > 1 { "s" } else { "" },
             gib(f.total_bytes()), f.tensors.len());
    let arch = ns_model::architecture(&f).map_err(|e| e.0)?;
    println!("arch     : {arch} ({})", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"));
    let m = Model::open(&f).map_err(|e| e.0)?;
    let g = &m.g;
    println!("scheme   : {:?} names", m.scheme);
    println!("layers   : {} ({} dense, {} MoE; MLA at {:?}){}", g.n_layer, g.n_dense, g.n_layer - g.n_dense,
             (0..g.n_layer).filter(|l| g.is_mla(*l)).collect::<Vec<_>>(), if g.n_mtp > 0 { format!(" + {} MTP block", g.n_mtp) } else { " - no MTP block".into() });
    println!("width    : hidden {}, vocab {}, dense FFN {}", g.n_embd, g.n_vocab, g.ffn_dense);
    println!("experts  : {} of {} per token + {} shared, FFN {}, scale {} {}, SwiGLU limit {}", g.n_expert_used, g.n_expert, g.n_expert_shared, g.ffn_expert,
             g.expert_scale, if g.expert_norm { "(normalized)" } else { "" }, g.swiglu_limit);
    println!("MLA      : {} heads of {}, q LoRA {}, kv LoRA {}; indexer {} heads of {}, top {}, pool {}", g.n_head, g.head_dim, g.q_lora, g.kv_lora,
             g.idx_heads, g.idx_dim, g.idx_top_k, g.idx_pool);
    println!("KDA      : {} heads of {}, conv {}, gate rank {}, gate floor {}", g.kda_heads, g.kda_dim, g.kda_conv, g.kda_rank, g.kda_gate_low);
    println!("hc       : {} streams, {} Sinkhorn iterations, eps {:e}; rms eps {:e}", g.hc, g.hc_iters, g.hc_eps, g.rms_eps);

    let errs = m.check();
    if errs.is_empty() {
        println!("check    : every tensor present with the shape the runtime needs, none left over");
    } else {
        println!("check    : {} problem{}", errs.len(), if errs.len() > 1 { "s" } else { "" });
        for e in errs.iter().take(40) {
            println!("  {e}");
        }
    }

    println!("bytes    :");
    let by = m.bytes();
    let mut groups: std::collections::BTreeMap<Group, u64> = std::collections::BTreeMap::new();
    for ((grp, ty), (b, n)) in &by {
        *groups.entry(*grp).or_default() += b;
        println!("  {:<10} {:<8} {:>8.2} GiB  {:>4} tensors", format!("{grp:?}"), ty.name(), gib(*b), n);
    }
    for (grp, b) in &groups {
        println!("  {:<19} {:>8.2} GiB", format!("{grp:?}"), gib(*b));
    }
    let e = (g.n_dense..g.n_layer).map(|l| m.expert_bytes(l)).collect::<Vec<_>>();
    let (lo, hi) = (e.iter().min().copied().unwrap_or(0), e.iter().max().copied().unwrap_or(0));
    println!("expert   : {:.1}-{:.1} MiB each (gate + up + down), {:.2}-{:.2} GiB per layer; a token reads {:.2} GiB of routed experts",
             lo as f64 / 1048576.0, hi as f64 / 1048576.0, gib(lo * g.n_expert), gib(hi * g.n_expert),
             gib(e.iter().sum::<u64>() * g.n_expert_used));
    if !errs.is_empty() {
        return Err(format!("{} layout problem(s)", errs.len()));
    }
    Ok(())
}

/// MemAvailable, bytes.
fn host_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

fn gpus() -> Result<(), String> {
    use ns_core::{DevBuf, Gpu, HostBuf};
    use std::time::Instant;
    let e = |x: ns_core::Error| x.0;
    let list = ns_core::gpus().map_err(e)?;
    if list.is_empty() {
        return Err("no Level Zero GPU".into());
    }
    let mut open = Vec::new();
    for (i, name) in &list {
        let g = Gpu::open(*i).map_err(e)?;
        let (total, free) = g.memory().map_err(e)?;
        let units = ns_core::gpu_units(*i).map_or(String::new(), |(u, m)| format!(", {u} compute units at {m} MHz"));
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
    }).collect::<Result<_, ns_core::Error>>().map_err(e)?;
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
    let e = |x: ns_core::Error| x.0;
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
    let gs: Vec<std::sync::Arc<ns_core::Gpu>> = gpus.iter().map(|i| ns_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    for g in &gs {
        println!("gpu      : {} ({})", g.name, g.index);
    }
    let mut log = |l: String| println!("load     : {l}");
    // one GPU: no host mirror (bring-up); several: the mirror as generate has it (a long prompt touches every expert)
    let mirror = if gs.len() == 1 { Some(0) } else { None };
    let glm = ns_engine::glm5next::Glm::load(&f, &gs, None, mirror, false, (tokens.len() + 8, 2), &mut log).map_err(e)?;
    println!("load     : {:.2} GiB in {:.1} s", gib(glm.load_bytes), glm.load_seconds);
    println!("prompt   : {} tokens {:?}", tokens.len(), tokens);
    println!("{:<26} {:>10} {:>10} {:>10}", "tensor", "cosine", "rel err", "max diff");
    let mut worst: (f64, String) = (1.0, String::new());
    let t0 = std::time::Instant::now();
    // a prompt longer than a chunk: all but the last chunk first, then the last one compared - its rows against the
    // reference's last rows (the dump holds the whole prompt when llama.cpp ran it as one ubatch)
    let total = tokens.len();
    let last = total - (total - 1) % ns_engine::glm5next::prefill_chunk() - 1;
    let tc = total - last;
    let mut tap = |name: &str, b: &ns_core::DevBuf| -> ns_core::Result<()> {
        let Some(&n) = index.get(name) else { return Ok(()) };
        let mine = b.to_f32()?;
        let path = dump.join(format!("{name}.f32"));
        let raw = std::fs::read(&path).map_err(|x| ns_core::Error(format!("{}: {x}", path.display())))?;
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
    let mut sess = glm.session(tokens.len().max(16)).map_err(e)?;
    if last > 0 {
        let mut quiet = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
        glm.feed(&mut sess, &tokens[..last], &mut quiet).map_err(e)?;
        println!("prefix   : {last} tokens fed, comparing the last {tc} (positions {last}-{})", total - 1);
    }
    let logits = glm.forward(&mut sess, &tokens[last..], &mut tap).map_err(e)?;
    let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    let vocab = f.meta("tokenizer.ggml.tokens").and_then(|v| v.as_array());
    let word = |id: u32| vocab.and_then(|v| v.get(id as usize)).and_then(|v| v.as_str()).unwrap_or("?").replace('\u{120}', " ").to_string();
    println!("forward  : {:.1} s", t0.elapsed().as_secs_f64());
    // the same prompt incrementally: all but the last token, then the last alone (decode's path)
    if tokens.len() > 1 && tokens.len() <= ns_engine::glm5next::prefill_chunk() {
        let mut none = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
        let mut s2 = glm.session(tokens.len()).map_err(e)?;
        glm.forward(&mut s2, &tokens[..tokens.len() - 1], &mut none).map_err(e)?;
        let inc = glm.forward(&mut s2, &tokens[tokens.len() - 1..], &mut none).map_err(e)?;
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
    let t = ns_tok::Tokenizer::from_gguf(&f)?;
    let ids = t.encode(text);
    println!("{} tokens: {:?}", ids.len(), ids);
    for id in &ids {
        println!("  {id:>7} {:?}", t.decode(&[*id]));
    }
    let back = t.decode(&ids);
    println!("round trip: {}", if back == text { "identical" } else { "DIFFERENT" });
    Ok(())
}

/// A tiny deterministic generator for sampling (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Greedy at temperature 0; else softmax(logits / temp) restricted to the smallest set of top tokens holding top_p.
fn sample(logits: &[f32], temp: f32, top_p: f32, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let mx = logits[idx[0]];
    let mut p: Vec<(usize, f32)> = idx.iter().take(256).map(|&i| (i, ((logits[i] - mx) / temp).exp())).collect();
    let sum: f32 = p.iter().map(|x| x.1).sum();
    let mut acc = 0.0;
    let mut keep = p.len();
    for (n, x) in p.iter_mut().enumerate() {
        x.1 /= sum;
        acc += x.1;
        if acc >= top_p {
            keep = n + 1;
            break;
        }
    }
    p.truncate(keep);
    let total: f32 = p.iter().map(|x| x.1).sum();
    let mut r = rng.next_f32() * total;
    for (i, w) in &p {
        r -= w;
        if r <= 0.0 {
            return *i as u32;
        }
    }
    p.last().map_or(0, |x| x.0 as u32)
}

/// `nextsycl generate <model.gguf> --prompt TEXT [--effort low|high|max] [--max N] [--temp T] [--top-p P] [--gpu N]`
fn generate(args: &[String]) -> Result<(), String> {
    use std::io::Write;
    let e = |x: ns_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("generate <model.gguf> --prompt ...")?;
    let prompt = match opt("--prompt-file") {
        Some(f) => std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?,
        None => opt("--prompt").ok_or("--prompt TEXT or --prompt-file PATH")?,
    };
    let effort = ns_tok::Effort::parse(&opt("--effort").unwrap_or_else(|| "low".into())).ok_or("--effort low|high|max")?;
    let max: usize = opt("--max").and_then(|v| v.parse().ok()).unwrap_or(256);
    let temp: f32 = opt("--temp").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let top_p: f32 = opt("--top-p").and_then(|v| v.parse().ok()).unwrap_or(0.95);
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "0".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}: a GPU number"))).collect::<Result<_, _>>()?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = ns_tok::Tokenizer::from_gguf(&f)?;
    let text = ns_tok::glm_chat(&[ns_tok::Message { role: "user", content: &prompt, reasoning: None }], effort);
    let ids = tok.encode(&text);
    let gs: Vec<std::sync::Arc<ns_core::Gpu>> = gpus.iter().map(|i| ns_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let expert_gib: Option<f64> = opt("--expert-gib").and_then(|v| v.parse().ok());
    let mirror_gib: Option<f64> = opt("--mirror-gib").and_then(|v| v.parse().ok());
    let mut log = |l: String| eprintln!("[{l}]");
    let mtp = !args.iter().any(|a| a == "--no-mtp");
    let glm = ns_engine::glm5next::Glm::load(&f, &gs, expert_gib.map(|x| (x * (1u64 << 30) as f64) as usize),
                                                 mirror_gib.map(|x| (x * (1u64 << 30) as f64) as usize), mtp, (ids.len() + max + 1, 1), &mut log).map_err(e)?;
    eprintln!("[{} on {}, {} prompt tokens, loaded in {:.1} s]", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"),
              gs.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(" + "), ids.len(), glm.load_seconds);
    let mut sess = glm.session(ids.len() + max + 1).map_err(e)?;
    let mut none = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
    let t0 = std::time::Instant::now();
    let logits = glm.feed(&mut sess, &ids, &mut none).map_err(e)?;
    let prefill = t0.elapsed().as_secs_f64();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut draw = |l: &[f32]| sample(l, temp, top_p, &mut rng);
    let mut dec = glm.decoder(logits, mtp);
    let mut pending: Vec<u8> = Vec::new();
    let mut out = std::io::stdout();
    print!("<think>");
    let t1 = std::time::Instant::now();
    let mut n = 0;
    'gen: while n < max {
        for next in glm.step(&mut sess, &mut dec, &mut draw, &mut none).map_err(e)? {
            if tok.stop.contains(&next) || n >= max {
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
    for (name, secs, calls) in glm.profile() {
        eprintln!("[profile {name:<34} {secs:>7.2} s  {calls:>6} calls  {:>8.2} ms/token]", secs * 1000.0 / (n + 1) as f64);
    }
    eprintln!("[host waited {:.2} s for the routers' logits]",
              ns_engine::glm5next::ROUTER_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9);
    for (i, (peak, spills)) in glm.arena_peaks().iter().enumerate() {
        eprintln!("[arena {i}: peak {:.2} GiB, {spills} request(s) past it]", *peak as f64 / (1u64 << 30) as f64);
    }
    let (hits, misses, mirrored, direct, pf, pf_used) = glm.expert_stats();
    eprintln!("[prompt {} tokens in {prefill:.1} s ({:.1} tok/s); {n} generated in {dt:.1} s ({:.2} tok/s); drafts {} of {} accepted; experts: {hits} VRAM hits, {direct} read from host memory by prompt passes, {misses} swapped in ({mirrored} of them from host memory), {pf} prefetched ({pf_used} of them asked for)]",
              ids.len(), ids.len() as f64 / prefill, n as f64 / dt.max(1e-9), dec.accepted, dec.drafted);
    Ok(())
}

/// `nextsycl kernels <model.gguf> [--gpu N]`: each stored weight type of the file through the decode kernels against
/// the exact path (expand to float32, multiply), on a real matrix of that type; and the decode kernel's rate.
fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    use ns_core::{DevBuf, Ops};
    use ns_model::glm5next::Role;
    let e = |x: ns_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let m = Model::open(&f).map_err(|e| e.0)?;
    let g = ns_core::Gpu::open(gpu).map_err(e)?;
    let o = Ops { gpu: g.clone() };
    println!("gpu: {}", g.name);
    // one matrix per (type, role kind): routed experts' expert 0, else the whole matrix
    let mut picks: Vec<(String, ns_gguf::GType, usize, usize, Vec<u8>)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for l in 0..m.g.n_layer {
        for r in m.roles(l) {
            let Some(t) = m.tensor(l, r) else { continue };
            if t.shape.len() < 2 || t.ty == ns_gguf::GType::F32 || !seen.insert((t.ty, matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown))) {
                continue;
            }
            let (rows, cols) = (t.shape[t.shape.len() - 2] as usize, t.shape[t.shape.len() - 1] as usize);
            let bytes = t.ty.bytes((rows * cols) as u64).unwrap_or(0) as usize;
            let mut b = vec![0u8; bytes];
            f.read_into(t, 0, &mut b).map_err(|e| e.0)?;
            picks.push((t.name.clone(), t.ty, rows, cols, b));
        }
    }
    let mut rng = Rng(12345);
    println!("{:<34} {:<8} {:>12} {:>10} {:>10} {:>11}", "matrix", "type", "rows x cols", "1 col err", "4 col err", "1 col rate");
    for (name, ty, rows, cols, bytes) in picks {
        let w = DevBuf::new(&g, bytes.len()).map_err(e)?;
        w.write(0, &bytes).map_err(e)?;
        let wf = DevBuf::f32(&g, rows * cols).map_err(e)?;
        o.dequant(ty.code(), &w, 0, bytes.len(), rows * cols, &wf).map_err(e)?;
        if !o.mmvq_supported(ty.code()) {
            println!("{name:<34} {:<8} {:>12} no decode kernel", ty.name(), format!("{rows}x{cols}"));
            continue;
        }
        let mut errs = Vec::new();
        let mut rate = 0.0;
        for nc in [1usize, 4] {
            let xs: Vec<f32> = (0..nc * cols).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            let x = DevBuf::from_f32(&g, &xs).map_err(e)?;
            let yr = DevBuf::f32(&g, nc * rows).map_err(e)?;
            o.gemm(nc, rows, cols, &x, &wf, &yr, false).map_err(e)?;
            let q = DevBuf::new(&g, o.q8_1_bytes(cols, nc)).map_err(e)?;
            o.quantize_q8_1((&x, 0), &q, cols, nc).map_err(e)?;
            let y = DevBuf::f32(&g, nc * rows).map_err(e)?;
            o.mmvq(ty.code(), (&w, 0), bytes.len(), &q, (&y, 0), cols, rows, nc).map_err(e)?;
            let (_, rel, _) = compare(&y.to_f32().map_err(e)?, &yr.to_f32().map_err(e)?);
            errs.push(rel);
            if nc == 1 {
                g.sync().map_err(e)?;
                let n = 50;
                let t0 = std::time::Instant::now();
                for _ in 0..n {
                    o.mmvq(ty.code(), (&w, 0), bytes.len(), &q, (&y, 0), cols, rows, 1).map_err(e)?;
                }
                g.sync().map_err(e)?;
                rate = bytes.len() as f64 * n as f64 / t0.elapsed().as_secs_f64() / 1e9;
            }
        }
        println!("{name:<34} {:<8} {:>12} {:>10.2e} {:>10.2e} {:>8.0} GB/s", ty.name(), format!("{rows}x{cols}"), errs[0], errs[1], rate);
    }
    Ok(())
}

/// `nextsycl spec-check`: greedy decode one token at a time (the reference), then the same tokens as 2-row verify
/// passes each rolled back to its first row; every row's logits against the reference's at that position.
/// `nextsycl batch-check <model.gguf> [--prompts "a|b|c"] [--n N] [--gpu 1,0]`: several conversations decoded
/// together (`forward_batch`) against each decoded alone - the same greedy tokens, the logits' largest difference -
/// and the speed of both.
fn batch_check(args: &[String]) -> Result<(), String> {
    let e = |x: ns_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("batch-check <model.gguf> ...")?;
    let prompts: Vec<String> = opt("--prompts")
        .unwrap_or_else(|| "Write a haiku about rivers.|Explain TCP slow start.|List five prime numbers and why.|What is a monad?".into())
        .split('|').map(String::from).collect();
    let n: usize = opt("--n").and_then(|v| v.parse().ok()).unwrap_or(32);
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "1,0".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}"))).collect::<Result<_, _>>()?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = ns_tok::Tokenizer::from_gguf(&f)?;
    let ids: Vec<Vec<u32>> = prompts.iter()
        .map(|p| tok.encode(&ns_tok::glm_chat(&[ns_tok::Message { role: "user", content: p, reasoning: None }], ns_tok::Effort::Low)))
        .collect();
    let ctx = ids.iter().map(|v| v.len()).max().unwrap_or(0) + n + 8;
    let gs: Vec<std::sync::Arc<ns_core::Gpu>> = gpus.iter().map(|i| ns_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let mut log = |l: String| eprintln!("[{l}]");
    let mtp = args.iter().any(|a| a == "--mtp"); // the draft block loaded (it is not used here; its experts take store slots)
    let glm = ns_engine::glm5next::Glm::load(&f, &gs, None, None, mtp, (ctx, prompts.len() + 1), &mut log).map_err(e)?;
    let mut none = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i) as u32;
    // alone: each conversation, one token a pass
    let mut solo_toks: Vec<Vec<u32>> = Vec::new();
    let mut solo_logits: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut solo_s = 0.0;
    for p in &ids {
        let mut s = glm.session(ctx).map_err(e)?;
        let mut l = glm.feed(&mut s, p, &mut none).map_err(e)?;
        let (mut ts, mut ls) = (Vec::new(), Vec::new());
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let t = argmax(&l);
            ts.push(t);
            l = glm.forward(&mut s, &[t], &mut none).map_err(e)?;
            ls.push(l.clone());
        }
        solo_s += t0.elapsed().as_secs_f64();
        solo_toks.push(ts);
        solo_logits.push(ls);
    }
    // together
    let mut sess: Vec<ns_engine::glm5next::Session> = Vec::new();
    let mut last: Vec<Vec<f32>> = Vec::new();
    for p in &ids {
        let mut s = glm.session(ctx).map_err(e)?;
        last.push(glm.feed(&mut s, p, &mut none).map_err(e)?);
        sess.push(s);
    }
    let mut worst = 0f32;
    let mut differ = 0;
    let t0 = std::time::Instant::now();
    for step in 0..n {
        let toks: Vec<u32> = last.iter().map(|l| argmax(l)).collect();
        for (b, t) in toks.iter().enumerate() {
            if *t != solo_toks[b][step] {
                differ += 1;
            }
        }
        // each session takes the token its own run took, so the comparison stays aligned
        let feed: Vec<u32> = (0..ids.len()).map(|b| solo_toks[b][step]).collect();
        let mut refs: Vec<&mut ns_engine::glm5next::Session> = sess.iter_mut().collect();
        last = glm.forward_batch(&mut refs, &feed, &mut none).map_err(e)?;
        for (b, l) in last.iter().enumerate() {
            let r = &solo_logits[b][step];
            worst = worst.max(l.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
        }
    }
    let batch_s = t0.elapsed().as_secs_f64();
    let b = ids.len();
    println!("{b} conversations, {n} tokens each: alone {:.2} tok/s (each, {:.1} s in all); together {:.2} tok/s in all ({:.2} each)",
             n as f64 * b as f64 / solo_s, solo_s, n as f64 * b as f64 / batch_s, n as f64 / batch_s);
    println!("greedy tokens different in {differ} of {} steps; the logits' largest difference {worst:.3e}", n * b);
    let (hits, misses, _, _, pf, pf_used) = glm.expert_stats();
    println!("experts (both runs): {hits} VRAM hits, {misses} swapped in, {pf} prefetched ({pf_used} asked for)");
    for (name, secs, calls) in glm.profile() {
        println!("[profile {name:<40} {secs:7.2} s {calls:>8} calls]");
    }
    Ok(())
}

fn spec_check(args: &[String]) -> Result<(), String> {
    let e = |x: ns_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("spec-check <model.gguf> --prompt ...")?;
    let prompt = match opt("--prompt-file") {
        Some(f) => std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?,
        None => opt("--prompt").ok_or("--prompt TEXT or --prompt-file PATH")?,
    };
    let n: usize = opt("--n").and_then(|v| v.parse().ok()).unwrap_or(32);
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "0,1".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}"))).collect::<Result<_, _>>()?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = ns_tok::Tokenizer::from_gguf(&f)?;
    let ids = tok.encode(&ns_tok::glm_chat(&[ns_tok::Message { role: "user", content: &prompt, reasoning: None }], ns_tok::Effort::Low));
    let gs: Vec<std::sync::Arc<ns_core::Gpu>> = gpus.iter().map(|i| ns_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let mut log = |l: String| eprintln!("[{l}]");
    let glm = ns_engine::glm5next::Glm::load(&f, &gs, None, None, true, (ids.len() + n + 4, 2), &mut log).map_err(e)?;
    let mut none = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    // the margin between the best two logits
    let margin = |v: &[f32]| {
        let mut s: Vec<f32> = v.to_vec();
        s.sort_by(|a, b| b.total_cmp(a));
        s[0] - s[1]
    };
    let mut a = glm.session(ids.len() + n + 4).map_err(e)?;
    let mut la = glm.feed(&mut a, &ids, &mut none).map_err(e)?;
    let mut toks = Vec::new();
    let mut refs = Vec::new(); // refs[i]: the logits after toks[i]
    for _ in 0..n {
        let t = argmax(&la) as u32;
        toks.push(t);
        la = glm.forward(&mut a, &[t], &mut none).map_err(e)?;
        refs.push(la.clone());
    }
    let mut b = glm.session(ids.len() + n + 4).map_err(e)?;
    glm.feed(&mut b, &ids, &mut none).map_err(e)?;
    if args.iter().any(|a| a == "--layers") {
        // the first verify pass's row 0 against a one-token pass from the same state, step by step
        let mut c = glm.session(ids.len() + n + 4).map_err(e)?;
        glm.copy_session(&mut c, &b).map_err(e)?;
        let mut one: Vec<(String, Vec<f32>)> = Vec::new();
        let mut keep = |name: &str, x: &ns_core::DevBuf| -> ns_core::Result<()> {
            one.push((name.to_string(), x.to_f32()?));
            Ok(())
        };
        glm.forward(&mut c, &[toks[0]], &mut keep).map_err(e)?;
        let mut two: Vec<(String, Vec<f32>)> = Vec::new();
        let mut keep2 = |name: &str, x: &ns_core::DevBuf| -> ns_core::Result<()> {
            two.push((name.to_string(), x.to_f32()?));
            Ok(())
        };
        let mut d = glm.session(ids.len() + n + 4).map_err(e)?;
        glm.copy_session(&mut d, &b).map_err(e)?;
        glm.forward_rows(&mut d, &[toks[0], toks[1]], 1, &mut keep2).map_err(e)?;
        for ((na, va), (_, vb)) in one.iter().zip(&two) {
            // row 0 of the 2-row tensor: its first len/1 values (rows are outermost)
            let w = va.len();
            if na.starts_with("result") || vb.len() < w {
                continue;
            }
            let rb = &vb[..w];
            let dd = va.iter().zip(rb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let mx = va.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-20);
            println!("{na:<24} {:.3e}", dd / mx);
        }
        return Ok(());
    }
    println!("{:>4} {:>4} {:>12} {:>10} {:>6} {:>8}", "pos", "row", "max |diff|", "max |ref|", "top1", "margin");
    let (mut worst, mut flips) = (0f32, 0);
    for i in 0..n - 1 {
        let rows = glm.forward_rows(&mut b, &[toks[i], toks[i + 1]], 2, &mut none).map_err(e)?;
        for (r, (got, want)) in rows.iter().zip([&refs[i], &refs[i + 1]]).enumerate() {
            let d = got.iter().zip(want.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let mx = want.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let same = argmax(got) == argmax(want);
            worst = worst.max(d / mx);
            flips += usize::from(!same);
            if i < 4 || !same || i % 8 == 0 {
                println!("{:>4} {:>4} {:>12.5} {:>10.3} {:>6} {:>8.4}", i, r, d, mx, if same { "same" } else { "FLIP" }, margin(want));
            }
        }
        glm.rollback(&mut b, 1).map_err(e)?;
    }
    println!("worst max|diff| / max|ref| {worst:.2e}; top-1 differs in {flips} of {} rows", 2 * (n - 1));
    Ok(())
}

/// `nextsycl serve`: load, then answer on HTTP (serve.rs).
fn serve_cmd(args: &[String]) -> Result<(), String> {
    let e = |x: ns_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("serve <model.gguf> ...")?;
    let g = opt("--gpu").unwrap_or_else(|| "0".into());
    let gpus: Vec<usize> = if g == "all" {
        // the one that computes most last: it takes the head and the draft block (measured on a B65 + B70: the
        // B70 last reads prompts 6% faster; decode the same)
        ns_core::gpus_weakest_first().map_err(e)?
    } else {
        g.split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}"))).collect::<Result<_, _>>()?
    };
    let addr = format!("{}:{}", opt("--host").unwrap_or_else(|| "127.0.0.1".into()), opt("--port").unwrap_or_else(|| "8085".into()));
    let ctx: usize = opt("--ctx").and_then(|v| v.parse().ok()).unwrap_or(8192);
    let effort = ns_tok::Effort::parse(&opt("--effort").unwrap_or_else(|| "low".into())).ok_or("--effort low|high|max")?;
    let gib_opt = |k: &str| opt(k).and_then(|v| v.parse::<f64>().ok()).map(|x| (x * (1u64 << 30) as f64) as usize);
    // the model's file lives as long as the server
    let f: &'static Gguf = Box::leak(Box::new(Gguf::open(Path::new(model)).map_err(|e| e.0)?));
    let name = opt("--name").unwrap_or_else(|| Path::new(model).file_stem().map_or("model".into(), |s| s.to_string_lossy().to_lowercase()));
    let tok = ns_tok::Tokenizer::from_gguf(f)?;
    let gs: Vec<std::sync::Arc<ns_core::Gpu>> = gpus.iter().map(|i| ns_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
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
    let glm = ns_engine::glm5next::Glm::load(f, &gs, gib_opt("--expert-gib"), mirror, mtp, (ctx, parallel), &mut log).map_err(e)?;
    eprintln!("[{} loaded on {} in {:.1} s]", name, gs.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(" + "), glm.load_seconds);
    let cors: Vec<String> = opt("--cors").unwrap_or_default().split([',', ' ']).filter(|o| !o.is_empty()).map(String::from).collect();
    let keep: usize = opt("--keep-requests").and_then(|v| v.parse().ok()).unwrap_or(100);
    // what a request without max_tokens may make: --max-tokens N (0 or none: to the end of the context)
    let default_max = opt("--max-tokens").and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0);
    let srv = std::sync::Arc::new(serve::Server::new(glm, tok, name, ctx, effort, cache, cors, keep, parallel, default_max)?);
    srv.run(&addr, opt("--socket").map(std::path::PathBuf::from))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = config::Config::load();
    let _ = client::SERVER.set(http::Target::Unix(cfg.socket()));
    let rest = args.get(1..).unwrap_or(&[]);
    let r = match args.first().map(String::as_str) {
        Some("start") => service::start(&cfg, rest),
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
