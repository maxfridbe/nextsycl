//! `nextsycl bench`: benchy v1 (Strata's standard bench, its prompts as text) against the running server, and how it
//! does with several requests at once.
//!
//!     nextsycl bench [--sizes 20,2185,8000,40000] [--new 256] [--parallel 1,2,4] [--out DIR]
//!
//! - **Sizes:** benchy v1's prompts (bench/v1): the 20-token Fibonacci request and the 2,185-token document to
//!   summarize (token counts of Strata's tokenizer; GLM's differ a little - the table gives the counts it read); the
//!   longer sizes repeat the document to length. Each answer is 256 greedy tokens.
//! - **Each run:** the prompt cache cleared and a run number put first, so nothing is reused; streamed, for the time
//!   to the first token; the rest from the server's record of the request (`inspect`): the prompt's read speed,
//!   decode speed, drafts accepted, energy and power.
//! - **At once:** N clients send the short prompt together (each its own run number); the table gives the tokens of
//!   all of them over the wall time, and each client's wait. The server runs one request at a time today, so this is
//!   the baseline batching would raise.
//!
//! Out: `matrix.md` (printed too) and `matrix.jsonl` in --out (default benchy-results/v1-<date>).
//!
//!     nextsycl bench --needle [--sizes 32768,131072,250000] [--depths 10,50,90] [--out DIR]
//!
//! - **Needle:** a passphrase sentence placed at each depth (% of the document) of filler of each size (the document
//!   repeated), asked for at the end; found = the answer (or its thinking) carries it. The read time of each prompt
//!   from the server's record. Sizes past the server's context are skipped. Out: `needle.md`, `needle.jsonl`.

use std::io::{BufRead, Write};
use std::time::Instant;

use serde_json::{json, Value};

use crate::client::{get, post, SERVER};
use crate::http;

/// a client's request: its seconds, to the first token, the server's record
type Done = (f64, Option<f64>, Value);

const SHORT: &str = include_str!("../../../bench/v1/short.txt");
const LONG: &str = include_str!("../../../bench/v1/long.txt");

/// One request through the socket, streamed: (seconds to the first token, the server's record of it)
fn run(prompt: &str, new: u64) -> Result<(Option<f64>, Value), String> {
    let body = json!({"messages": [{"role": "user", "content": prompt}], "max_tokens": new, "temperature": 0, "stream": true});
    let t0 = Instant::now();
    let target = SERVER.get().ok_or("no server socket")?;
    let (status, r) = http::send(target, "POST", "/v1/chat/completions", Some(&body))?;
    if status != 200 {
        return Err(format!("the server answered {status}"));
    }
    let mut first = None;
    for line in r.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" {
            break;
        }
        if first.is_none() {
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                let d = &v["choices"][0]["delta"];
                if d["content"].is_string() || d["reasoning_content"].is_string() {
                    first = Some(t0.elapsed().as_secs_f64());
                }
            }
        }
    }
    // the request just ended is the newest the server remembers
    let rec = get("/server/requests")?["done"][0].clone();
    Ok((first, rec))
}

/// A prompt of about `n` tokens: the short one, the document, or the document repeated to length (`cpt`: characters
/// a token, from the document's own count)
fn prompt(n: usize, cpt: f64) -> String {
    if n <= 64 {
        return SHORT.to_string();
    }
    let (head, doc) = LONG.split_once("\n\n").unwrap_or(("", LONG));
    let want = (n as f64 * cpt) as usize;
    if want <= LONG.len() + 200 {
        return LONG.to_string();
    }
    let mut s = format!("{head}\n\n");
    while s.len() < want {
        for line in doc.lines() {
            if s.len() >= want {
                break;
            }
            s.push_str(line);
            s.push('\n');
        }
    }
    s
}

fn f(v: &Value, spec: usize) -> String {
    v.as_f64().map_or("-".into(), |x| format!("{x:.spec$}"))
}

pub fn bench(raw: &[String]) -> Result<(), String> {
    if raw.iter().any(|a| a == "--needle") {
        return needle(raw);
    }
    let opt = |k: &str| raw.iter().position(|a| a == k).and_then(|i| raw.get(i + 1)).cloned();
    let list = |k: &str, d: &str| -> Vec<usize> { opt(k).unwrap_or_else(|| d.into()).split(',').filter_map(|x| x.trim().parse().ok()).collect() };
    let sizes = list("--sizes", "20,2185,8000,40000");
    let parallel = list("--parallel", "1,2,4");
    let new: u64 = opt("--new").and_then(|v| v.parse().ok()).unwrap_or(256);
    let st = get("/server/status")?;
    let ctx = st["context"].as_u64().unwrap_or(0) as usize;
    let date = std::process::Command::new("date").arg("+%Y-%m-%d").output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let out = std::path::PathBuf::from(opt("--out").unwrap_or_else(|| format!("benchy-results/v1-{date}")));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut jl = std::fs::File::create(out.join("matrix.jsonl")).map_err(|e| e.to_string())?;
    let mut tag = 0u32;
    let mut fresh = |p: &str| -> String {
        tag += 1;
        format!("(benchy run {tag})\n\n{p}")
    };

    // the document's characters a token, from one read of it (the cache cleared before)
    let _ = post("/server/cache/clear", None)?;
    eprintln!("[bench: the document once, for its token count]");
    let (_, rec) = run(&fresh(LONG), 1)?;
    let cpt = LONG.len() as f64 / rec["prompt_tokens"].as_f64().unwrap_or(2185.0).max(1.0);

    let mut rows = Vec::new();
    for &n in &sizes {
        if n + new as usize + 64 > ctx {
            rows.push(format!("| {n} | - | - | - | - | - | - | - | skipped: the context is {ctx} |"));
            continue;
        }
        let _ = post("/server/cache/clear", None)?;
        let p = fresh(&prompt(n, cpt));
        eprintln!("[bench: about {n} tokens]");
        let (ttft, r) = run(&p, new)?;
        let pp = r["prompt_tokens"].as_f64().zip(r["read_seconds"].as_f64()).map(|(t, s)| t / s.max(1e-9));
        let acc = r["drafts"].as_array().and_then(|d| Some(d.first()?.as_f64()? / d.get(1)?.as_f64()?.max(1.0)));
        writeln!(jl, "{}", json!({"kind": "size", "target": n, "ttft_s": ttft, "pp_tok_s": pp, "record": r})).map_err(|e| e.to_string())?;
        rows.push(format!("| {} | {} | {} | {} | {} | {} | {} | {} | {} |", t(&r["prompt_tokens"]), pp.map_or("-".into(), |x| format!("{x:.1}")),
                          ttft.map_or("-".into(), |x| format!("{x:.2}")), f(&r["tok_s"], 2), acc.map_or("-".into(), |a| format!("{:.0}%", a * 100.0)),
                          t(&r["generated"]), f(&r["energy_j"], 0), f(&r["avg_watts"], 0), r["finish"].as_str().unwrap_or("-")));
    }

    let mut prow = Vec::new();
    for &k in &parallel {
        let _ = post("/server/cache/clear", None)?;
        let ps: Vec<String> = (0..k).map(|_| fresh(SHORT)).collect();
        eprintln!("[bench: {k} at once]");
        let t0 = Instant::now();
        let res: Vec<Result<Done, String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = ps.iter().map(|p| sc.spawn(move || {
                let t = Instant::now();
                run(p, new).map(|(ttft, r)| (t.elapsed().as_secs_f64(), ttft, r))
            })).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a client panicked".into()))).collect()
        });
        let wall = t0.elapsed().as_secs_f64();
        let ok: Vec<&Done> = res.iter().filter_map(|r| r.as_ref().ok()).collect();
        // each client's own record is not the newest when others ended after it: count its tokens from the ones
        // the server holds for this round
        let done = get("/server/requests")?["done"].as_array().cloned().unwrap_or_default();
        let round: Vec<&Value> = done.iter().take(k).collect();
        let tokens: f64 = round.iter().filter_map(|r| r["generated"].as_f64()).sum();
        let energy: f64 = round.iter().filter_map(|r| r["energy_j"].as_f64()).sum();
        let lat: Vec<f64> = ok.iter().map(|x| x.0).collect();
        let first: Vec<f64> = ok.iter().filter_map(|x| x.1).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
        writeln!(jl, "{}", json!({"kind": "parallel", "clients": k, "wall_s": wall, "tokens": tokens, "latencies_s": lat, "ttft_s": first}))
            .map_err(|e| e.to_string())?;
        prow.push(format!("| {k} | {}/{k} | {tokens:.0} | {wall:.1} | {:.2} | {:.1} / {:.1} | {:.1} / {:.1} | {:.0} |", ok.len(), tokens / wall.max(1e-9),
                          mean(&lat), max(&lat), mean(&first), max(&first), energy / tokens.max(1.0)));
    }

    let gpus: Vec<String> = st["gpus"].as_array().cloned().unwrap_or_default().iter()
        .map(|g| format!("{} ({})", g["name"].as_str().unwrap_or("?").replace("Intel(R) ", "").replace("(TM)", ""), g["pci"].as_str().unwrap_or("?")))
        .collect();
    let md = format!("**benchy v1, nextsycl** ({date}) - {} on {}, context {}, MTP {}\n\n\
| Input tokens | PP (tok/s) | TTFT (s) | TG (tok/s) | Drafts accepted | Output tokens | Energy (J) | Avg power (W) | Finish |\n\
|---:|---:|---:|---:|---:|---:|---:|---:|---|\n{}\n\n\
Several at once (the short prompt, {new} tokens each, sent together):\n\n\
| Clients | Completed | Tokens | Wall (s) | Tokens/s, all | Latency mean / max (s) | First token mean / max (s) | J a token |\n\
|---:|---:|---:|---:|---:|---:|---:|---:|\n{}\n\n\
Caveats: the running server, warm (its weights loaded, the page cache as it is); the prompt cache cleared and a run \
number first, so no prompt is reused; greedy, {new} new tokens; TG with MTP's drafts; energy is both cards' (idle \
included). Input tokens are GLM's count of benchy v1's prompts (made with Strata's tokenizer).\n",
                     st["model"].as_str().unwrap_or("?"), gpus.join(" + "), ctx, if st["mtp"].as_bool() == Some(true) { "on" } else { "off" },
                     rows.join("\n"), prow.join("\n"));
    std::fs::write(out.join("matrix.md"), &md).map_err(|e| e.to_string())?;
    print!("{md}");
    eprintln!("[written to {}]", out.display());
    let _ = std::io::stdout().flush();
    Ok(())
}

fn t(v: &Value) -> String {
    match v {
        Value::Null => "-".into(),
        o => o.to_string(),
    }
}

/// One request through the socket, streamed: what it answered (thinking and content)
fn ask(prompt: &str, new: u64) -> Result<String, String> {
    let body = json!({"messages": [{"role": "user", "content": prompt}], "max_tokens": new, "temperature": 0, "stream": true});
    let target = SERVER.get().ok_or("no server socket")?;
    let (status, r) = http::send(target, "POST", "/v1/chat/completions", Some(&body))?;
    if status != 200 {
        return Err(format!("the server answered {status}"));
    }
    let mut out = String::new();
    for line in r.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" {
            break;
        }
        if let Ok(v) = serde_json::from_str::<Value>(data) {
            let d = &v["choices"][0]["delta"];
            for k in ["reasoning_content", "content"] {
                if let Some(t) = d[k].as_str() {
                    out.push_str(t);
                }
            }
        }
    }
    Ok(out)
}

/// `bench --needle`: can the model find a fact anywhere in a long context
fn needle(raw: &[String]) -> Result<(), String> {
    let opt = |k: &str| raw.iter().position(|a| a == k).and_then(|i| raw.get(i + 1)).cloned();
    let list = |k: &str, d: &str| -> Vec<usize> { opt(k).unwrap_or_else(|| d.into()).split(',').filter_map(|x| x.trim().parse().ok()).collect() };
    let sizes = list("--sizes", "32768,131072,250000");
    let depths = list("--depths", "10,50,90");
    let st = get("/server/status")?;
    let ctx = st["context"].as_u64().unwrap_or(0) as usize;
    let date = std::process::Command::new("date").arg("+%Y-%m-%d").output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let out = std::path::PathBuf::from(opt("--out").unwrap_or_else(|| format!("benchy-results/needle-{date}")));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut jl = std::fs::File::create(out.join("needle.jsonl")).map_err(|e| e.to_string())?;
    // the document's characters a token, from one read of it
    eprintln!("[needle: the document once, for its token count]");
    let (_, rec) = run(&format!("(needle calibration)\n\n{LONG}"), 1)?;
    let cpt = LONG.len() as f64 / rec["prompt_tokens"].as_f64().unwrap_or(2185.0).max(1.0);
    let doc: Vec<&str> = LONG.split_once("\n\n").map_or(LONG, |(_, d)| d).lines().collect();
    const WORDS: [&str; 8] = ["amber", "falcon", "copper", "willow", "harbor", "quartz", "meadow", "cinder"];
    let mut rows = Vec::new();
    let mut trial = 0usize;
    for &n in &sizes {
        if n + 512 > ctx {
            rows.push(format!("| {n} | - | - | - | skipped: the context is {ctx} |"));
            continue;
        }
        // the filler: the document's lines, repeated to about n tokens (less the question's few)
        let want = ((n - 200) as f64 * cpt) as usize;
        let mut lines: Vec<&str> = Vec::new();
        let mut len = 0;
        while len < want {
            for l in &doc {
                if len >= want {
                    break;
                }
                lines.push(l);
                len += l.len() + 1;
            }
        }
        for &dp in &depths {
            trial += 1;
            let pass = format!("{}-{}-{}", WORDS[trial % 8], WORDS[(trial * 3 + 1) % 8], 1000 + 37 * trial);
            let at = lines.len() * dp.min(100) / 100;
            let mut text = String::with_capacity(len + 256);
            for (i, l) in lines.iter().enumerate() {
                if i == at {
                    text.push_str(&format!("The secret passphrase for the vault is {pass}. Remember it.\n"));
                }
                text.push_str(l);
                text.push('\n');
            }
            let prompt = format!("Below is a long document. Somewhere in it is a passphrase.\n\n{text}\nWhat is the secret passphrase for the vault? \
                                  Answer with the passphrase only.");
            eprintln!("[needle: about {n} tokens, at {dp}%]");
            let answer = ask(&prompt, 96)?;
            let rec = get("/server/requests")?["done"][0].clone();
            let found = answer.contains(&pass);
            writeln!(jl, "{}", json!({"target": n, "depth": dp, "passphrase": pass, "found": found, "answer": answer, "record": rec}))
                .map_err(|e| e.to_string())?;
            let tail: String = answer.chars().rev().take(60).collect::<Vec<_>>().into_iter().rev().collect();
            rows.push(format!("| {} | {dp}% | {} | {} | {} |", t(&rec["prompt_tokens"]), if found { "yes" } else { "**no**" }, f(&rec["read_seconds"], 1),
                              tail.replace(['\n', '|'], " ")));
        }
    }
    let md = format!("**needle, nextsycl** ({date}) - {} , context {ctx}\n\n| Prompt tokens | Depth | Found | Read (s) | Answer (end) |\n\
                      |---:|---:|---|---:|---|\n{}\n", st["model"].as_str().unwrap_or("?"), rows.join("\n"));
    std::fs::write(out.join("needle.md"), &md).map_err(|e| e.to_string())?;
    print!("{md}");
    eprintln!("[written to {}]", out.display());
    Ok(())
}
