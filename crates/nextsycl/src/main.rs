//! `nextsycl`: the command line.
//!
//!     nextsycl info <model.gguf>      the architecture, its geometry, every tensor checked by role, bytes by group
//!     nextsycl gpus                   each GPU in its own context: memory, copies and their rates, GPU to GPU, and
//!                                     that device memory costs no host RAM
//!     nextsycl check <model.gguf> <dump dir> [--gpu N]
//!                                     the forward pass on a reference dump's prompt (reference/llama-dump), every
//!                                     step compared: cosine and relative error per tensor, then the next token

use std::path::Path;
use std::process::ExitCode;

use ns_gguf::Gguf;
use ns_model::glm5next::{Group, Model};

const USAGE: &str = "usage:
  nextsycl info <model.gguf>    the architecture and geometry, every tensor checked by role, bytes by group
  nextsycl gpus                 each GPU in its own context: memory, copy rates, GPU to GPU, host RAM unaffected
  nextsycl check <model.gguf> <dump dir> [--gpu N]
                                the forward pass on a reference dump's prompt, every step compared, the next token
  nextsycl tokenize <model.gguf> <text>
  nextsycl generate <model.gguf> --prompt TEXT [--effort low|high|max] [--max N] [--temp T] [--top-p P] [--gpu N]";

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
        println!("GPU {i}: {name}, {:.1} GiB{}", gib(total), free.map_or(String::new(), |f| format!(" ({:.1} GiB free)", gib(f))));
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

fn check(model: &Path, dump: &Path, gpu: usize) -> Result<(), String> {
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
    let g = ns_core::Gpu::open(gpu).map_err(e)?;
    println!("gpu      : {} ({})", g.name, g.index);
    let mut log = |l: String| println!("load     : {l}");
    let glm = ns_engine::glm5next::Glm::load(&f, &g, &mut log).map_err(e)?;
    println!("load     : {:.2} GiB in {:.1} s", gib(glm.load_bytes), glm.load_seconds);
    println!("prompt   : {} tokens {:?}", tokens.len(), tokens);
    println!("{:<26} {:>10} {:>10} {:>10}", "tensor", "cosine", "rel err", "max diff");
    let mut worst: (f64, String) = (1.0, String::new());
    let t0 = std::time::Instant::now();
    let mut tap = |name: &str, b: &ns_core::DevBuf| -> ns_core::Result<()> {
        let Some(&n) = index.get(name) else { return Ok(()) };
        let mine = b.to_f32()?;
        let path = dump.join(format!("{name}.f32"));
        let raw = std::fs::read(&path).map_err(|x| ns_core::Error(format!("{}: {x}", path.display())))?;
        let theirs: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if theirs.len() != n || mine.len() < n {
            println!("{name:<26} sizes differ: mine {}, the reference {} ({n} in its index)", mine.len(), theirs.len());
            return Ok(());
        }
        let (cos, rel, mx) = compare(&mine[..n], &theirs);
        println!("{name:<26} {cos:>10.6} {rel:>10.2e} {mx:>10.3e}");
        if cos < worst.0 {
            worst = (cos, name.to_string());
        }
        Ok(())
    };
    let mut sess = glm.session(tokens.len().max(16)).map_err(e)?;
    let logits = glm.forward(&mut sess, &tokens, &mut tap).map_err(e)?;
    let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    let vocab = f.meta("tokenizer.ggml.tokens").and_then(|v| v.as_array());
    let word = |id: u32| vocab.and_then(|v| v.get(id as usize)).and_then(|v| v.as_str()).unwrap_or("?").replace('\u{120}', " ").to_string();
    println!("forward  : {:.1} s", t0.elapsed().as_secs_f64());
    // the same prompt incrementally: all but the last token, then the last alone (decode's path)
    if tokens.len() > 1 {
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
    let prompt = opt("--prompt").ok_or("--prompt TEXT")?;
    let effort = ns_tok::Effort::parse(&opt("--effort").unwrap_or_else(|| "low".into())).ok_or("--effort low|high|max")?;
    let max: usize = opt("--max").and_then(|v| v.parse().ok()).unwrap_or(256);
    let temp: f32 = opt("--temp").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let top_p: f32 = opt("--top-p").and_then(|v| v.parse().ok()).unwrap_or(0.95);
    let gpu: usize = opt("--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let tok = ns_tok::Tokenizer::from_gguf(&f)?;
    let text = ns_tok::glm_chat(&[ns_tok::Message { role: "user", content: &prompt, reasoning: None }], effort);
    let ids = tok.encode(&text);
    let g = ns_core::Gpu::open(gpu).map_err(e)?;
    let mut log = |_: String| {};
    let glm = ns_engine::glm5next::Glm::load(&f, &g, &mut log).map_err(e)?;
    eprintln!("[{} on {}, {} prompt tokens, loaded in {:.1} s]", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"), g.name, ids.len(), glm.load_seconds);
    let mut sess = glm.session(ids.len() + max + 1).map_err(e)?;
    let mut none = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
    let t0 = std::time::Instant::now();
    let mut logits = glm.forward(&mut sess, &ids, &mut none).map_err(e)?;
    let prefill = t0.elapsed().as_secs_f64();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut pending: Vec<u8> = Vec::new();
    let mut out = std::io::stdout();
    print!("<think>");
    let t1 = std::time::Instant::now();
    let mut n = 0;
    for _ in 0..max {
        let next = sample(&logits, temp, top_p, &mut rng);
        if tok.stop.contains(&next) {
            break;
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
        logits = glm.forward(&mut sess, &[next], &mut none).map_err(e)?;
    }
    println!("{}", String::from_utf8_lossy(&pending));
    let dt = t1.elapsed().as_secs_f64();
    eprintln!("[prompt {} tokens in {prefill:.1} s ({:.1} tok/s); {n} generated in {dt:.1} s ({:.2} tok/s)]", ids.len(), ids.len() as f64 / prefill,
              n as f64 / dt.max(1e-9));
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("info") if args.len() == 2 => info(Path::new(&args[1])),
        Some("gpus") => gpus(),
        Some("tokenize") if args.len() == 3 => tokenize(Path::new(&args[1]), &args[2]),
        Some("generate") => generate(&args),
        Some("check") if args.len() >= 3 => {
            let gpu = args.iter().position(|a| a == "--gpu").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0);
            check(Path::new(&args[1]), Path::new(&args[2]), gpu)
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
