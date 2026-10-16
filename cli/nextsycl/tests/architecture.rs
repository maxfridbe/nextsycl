//! The directives (CONTRIBUTING.md's rules), checked on every `cargo test`, each by the tests named here:
//!
//! 1. Every language model caches prompts: `every_llm_template_keeps_a_cacheable_prefix` (each engine's chat template
//!    has a user-turn marker the server checkpoints at, and a later turn's prompt begins with the earlier one's up to
//!    its last user turn) and `every_llm_engine_saves_checkpoints` (each served engine's save / restore / read).
//! 2. SYCL is the only GPU runtime: `sycl_is_the_only_gpu_runtime` (no crate that brings another anywhere in
//!    Cargo.lock) and `own_kernels_include_no_other_runtime` (kernels/ns, kernels/<kind>/<arch>); `./build.sh test`
//!    checks the libraries link against no other runtime.
//! 3. Third-party crates bring no other GPU runtime: `sycl_is_the_only_gpu_runtime`.
//! 4. The architecture: `layers_depend_only_downward` (the foundation on the foundation, a contract on the
//!    foundation, an engine on the foundation and its own kind's contract, the glue never on an engine).
//! 5. Exact before fast: a review rule (the pull request's parity check and measurements), not a test.
//! 6. Tuning in silos: `silos_are_their_engines_own` (a silo is opened by its engine, included by nothing outside
//!    its arch, and gates its ops with `_supported` checks).
//!
//! And every environment variable the code reads has a row in docs/environment.md:
//! `every_environment_variable_is_documented`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Every workspace crate: its directory (relative to the root) and its package name
fn crates() -> BTreeMap<String, String> {
    let root = root();
    let ws = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let members = ws.split("members = [").nth(1).unwrap().split(']').next().unwrap();
    let mut out = BTreeMap::new();
    for dir in members.split(',').map(|m| m.trim().trim_matches('"')).filter(|m| !m.is_empty()) {
        let toml = std::fs::read_to_string(root.join(dir).join("Cargo.toml")).unwrap();
        let name = toml.lines().find_map(|l| l.strip_prefix("name = \"")).unwrap().trim_end_matches('"').to_string();
        out.insert(name, dir.to_string());
    }
    out
}

/// A crate's dependencies' names (the [dependencies] table)
fn deps(dir: &str) -> Vec<String> {
    let toml = std::fs::read_to_string(root().join(dir).join("Cargo.toml")).unwrap();
    let mut on = false;
    let mut out = Vec::new();
    for l in toml.lines() {
        let l = l.trim();
        if l.starts_with('[') {
            on = l == "[dependencies]";
            continue;
        }
        if on && !l.is_empty() && !l.starts_with('#') {
            out.push(l.split(['=', '.', ' ']).next().unwrap().to_string());
        }
    }
    out
}

/// The layer of a crate directory: ("foundation" | "contract" | "engine" | "glue" | "program", its kind)
fn layer(dir: &str) -> (&'static str, String) {
    let mut parts = dir.split('/');
    let (top, sub) = (parts.next().unwrap(), parts.next().unwrap_or(""));
    match top {
        "crates" => ("foundation", String::new()),
        "glue" => ("glue", String::new()),
        "cli" => ("program", String::new()),
        kind if sub == "contract" => ("contract", kind.to_string()),
        kind => ("engine", kind.to_string()),
    }
}

#[test]
fn layers_depend_only_downward() {
    let crates = crates();
    let mut bad = Vec::new();
    for (name, dir) in &crates {
        let (l, kind) = layer(dir);
        for d in deps(dir) {
            let Some(ddir) = crates.get(&d) else { continue }; // a third-party crate (rule 3 checks those)
            let (dl, dkind) = layer(ddir);
            let ok = match l {
                "foundation" => dl == "foundation",
                "contract" => dl == "foundation",
                "engine" => dl == "foundation" || (dl == "contract" && dkind == kind),
                "glue" => dl == "foundation" || dl == "contract",
                _ => true,
            };
            if !ok {
                bad.push(format!("{name} ({l}{}) depends on {d} ({dl}{})", if kind.is_empty() { String::new() } else { format!(" {kind}") },
                                 if dkind.is_empty() { String::new() } else { format!(" {dkind}") }));
            }
        }
    }
    assert!(bad.is_empty(), "layering broken (CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

#[test]
fn sycl_is_the_only_gpu_runtime() {
    const OTHERS: &[&str] = &["wgpu", "vulkano", "ash", "cudarc", "cust", "cuda-sys", "opencl3", "ocl", "cl-sys", "metal", "candle-core", "tch",
                              "torch-sys", "ort", "onnxruntime", "burn", "burn-wgpu", "burn-cuda", "hip-sys", "rocm", "vulkanalia", "naga"];
    let lock = std::fs::read_to_string(root().join("Cargo.lock")).unwrap();
    let found: Vec<&str> = lock.lines().filter_map(|l| l.strip_prefix("name = \"")).map(|n| n.trim_end_matches('"'))
        .filter(|n| OTHERS.contains(n)).collect();
    assert!(found.is_empty(), "crates that bring another GPU runtime (only SYCL is allowed: CONTRIBUTING.md): {found:?}");
}

#[test]
fn own_kernels_include_no_other_runtime() {
    const OTHERS: &[&str] = &["cuda", "vulkan", "CL/", "hip/", "Metal/", "level_zero/"];
    let k = root().join("kernels");
    let mut dirs = vec![k.join("ns")];
    for kind in ["llm", "image", "video", "audio", "model3d"] {
        if let Ok(rd) = std::fs::read_dir(k.join(kind)) {
            dirs.extend(rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
        }
    }
    let mut bad = Vec::new();
    for d in dirs {
        for f in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = f.path();
            if !matches!(p.extension().and_then(|e| e.to_str()), Some("cpp" | "hpp" | "h")) {
                continue;
            }
            for (n, line) in std::fs::read_to_string(&p).unwrap_or_default().lines().enumerate() {
                let l = line.trim_start();
                if let Some(inc) = l.strip_prefix("#include") {
                    let inc = inc.trim().trim_start_matches(['<', '"']);
                    if OTHERS.iter().any(|o| inc.starts_with(o)) {
                        bad.push(format!("{}:{}: {}", p.display(), n + 1, line.trim()));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "another runtime's headers in the project's kernels (SYCL only: CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

/// Every environment variable the code reads, each with the first place it is read (file:line): `env::var("X")`,
/// `getenv("X")`, `option_env!("X")`, the settings file's `cfg.get("X")` / `cfg.or("X", ..)` / `with_prefix(..)`,
/// an engine's `setting("X")` and its option table's `env: "X"`, the settings page's `st("X", ..)`, the helpers that
/// wrap those (`port`, `peak`, `env_on`, `env_flag`, `lanes_env`, `cfg, "X"`, `("--flag", "X", ..)`), and the build
/// scripts' `${X:-default}`. Over *.rs, *.cpp, *.hpp, *.h and *.sh, kernels/strata included; target/ (build output),
/// dist/ (the built program) and node_modules/ are not sources. reference/ is ours (the parity tools) and is scanned.
fn env_reads() -> BTreeMap<String, String> {
    const CALLS: &[&str] = &["env::var(", "env::var_os(", "getenv(", "option_env!(", ".get(", ".or(", ".setting(", "with_prefix(",
                             "env_on(", "env_flag(", "lanes_env(", "port(", "peak(", "st(", "env:", "cfg,"];
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let name_at = |s: &str| -> String { s.chars().take_while(|&c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_').collect() };
    let root = root();
    let mut out = BTreeMap::new();
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            let file = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if !matches!(file.as_str(), "target" | "dist" | "node_modules") && !file.starts_with('.') {
                    stack.push(p);
                }
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext, "rs" | "cpp" | "hpp" | "h" | "sh") {
                continue;
            }
            let rel = p.strip_prefix(&root).unwrap_or(&p).display().to_string();
            for (n, line) in std::fs::read_to_string(&p).unwrap_or_default().lines().enumerate() {
                let mut found = Vec::new();
                if ext == "sh" {
                    for (i, _) in line.match_indices("${") {
                        let name = name_at(&line[i + 2..]);
                        // not ${1:-all}: an argument
                        if name.starts_with(|c: char| c.is_ascii_uppercase()) && line[i + 2 + name.len()..].starts_with([':', '-', '+', '=']) {
                            found.push(name);
                        }
                    }
                } else {
                    for (i, _) in line.match_indices('"') {
                        let name = name_at(&line[i + 1..]);
                        if name.len() < 2 || !name.starts_with(|c: char| c.is_ascii_uppercase()) || !line[i + 1 + name.len()..].starts_with('"') {
                            continue;
                        }
                        let before = line[..i].trim_end();
                        let call = CALLS.iter().any(|&c| {
                            before.strip_suffix(c).is_some_and(|b| c.starts_with(['.', ':']) || !b.ends_with(ident))
                        });
                        // ("--flag", "X", default): a flag and the setting it falls back to
                        let flag = before.strip_suffix(',').map(str::trim_end).and_then(|b| b.strip_suffix('"'))
                            .and_then(|b| b.rfind('"').map(|q| &b[q + 1..])).is_some_and(|f| f.starts_with("--"));
                        if call || flag {
                            found.push(name);
                        }
                    }
                }
                for name in found {
                    out.entry(name).or_insert_with(|| format!("{rel}:{}", n + 1));
                }
            }
        }
    }
    out
}

#[test]
fn every_environment_variable_is_documented() {
    let doc = std::fs::read_to_string(root().join("docs/environment.md")).unwrap();
    // the doc's words: a variable is documented when its name is one (NS_QW_ matches the pattern row `NS_QW_*`)
    let words: std::collections::BTreeSet<&str> = doc.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|w| !w.is_empty()).collect();
    let reads = env_reads();
    assert!(reads.len() > 100, "the scan found only {} variables: is the pattern list broken?", reads.len());
    let missing: Vec<String> = reads.iter().filter(|(name, _)| !words.contains(name.as_str())).map(|(name, at)| format!("{name} ({at})")).collect();
    assert!(missing.is_empty(), "environment variables read but not in docs/environment.md (a row each: CONTRIBUTING.md):\n  {}", missing.join("\n  "));
}

/// Every language engine's registry entry (the program's own list: `engines()` in main.rs)
fn llm_kinds() -> Vec<nextsycl_llm::EngineKind> {
    vec![nextsycl_llm_glm5next::kind(), nextsycl_llm_qwen4exp::kind(), nextsycl_llm_qwen35::kind(), nextsycl_llm_deepseek2::kind(),
         nextsycl_llm_deepseek4::kind(), nextsycl_llm_museglimmer::kind(), nextsycl_llm_nemotronh::kind(), nextsycl_llm_example::kind()]
}

/// The engines under llm/ (each but the contract)
fn llm_arch_dirs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(root().join("llm")).unwrap().flatten().filter(|e| e.path().join("Cargo.toml").exists())
        .map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n != "contract").collect();
    v.sort();
    v
}

#[test]
fn every_llm_template_keeps_a_cacheable_prefix() {
    use nextsycl_tok::{Effort, Message, USER_TURNS};
    let main = std::fs::read_to_string(root().join("cli/nextsycl/src/main.rs")).unwrap();
    for arch in llm_arch_dirs() {
        let krate = format!("nextsycl_llm_{arch}::kind()");
        assert!(main.contains(&krate), "llm/{arch} is not in main.rs's engines()");
        assert!(std::fs::read_to_string(root().join(file!()).canonicalize().unwrap_or_else(|_| root().join("cli/nextsycl/tests/architecture.rs")))
                    .unwrap().contains(&krate), "llm/{arch} is not in this test's llm_kinds()");
    }
    let tool = r#"{"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}"#.to_string();
    let m = |role, content| Message { role, content, ..Default::default() };
    let answer = |content| Message { role: "assistant", content, reasoning: Some("Let me think about it."), ..Default::default() };
    let sys = m("system", "You are a careful assistant. Keep answers short.");
    let turns: [Vec<Message>; 3] = [
        vec![m("system", sys.content), m("user", "What is a nebula?")],
        vec![m("system", sys.content), m("user", "What is a nebula?"), answer("A cloud of gas and dust."), m("user", "And a pulsar?")],
        vec![m("system", sys.content), m("user", "What is a nebula?"), answer("A cloud of gas and dust."), m("user", "And a pulsar?"),
             answer("A spinning neutron star."), m("user", "Thanks.")],
    ];
    // where the server checkpoints a prompt for the next turn: the start of its last user turn
    let last_user = |p: &str| USER_TURNS.iter().filter_map(|u| p.rfind(u)).max();
    let mut bad = Vec::new();
    for k in llm_kinds() {
        for effort in [Effort::Low, Effort::High, Effort::Max] {
            for tools in [vec![], vec![tool.clone()]] {
                let p: Vec<String> = turns.iter().map(|t| (k.chat)(t, effort, &tools)).collect();
                let tag = format!("{} ({effort:?}{})", k.name, if tools.is_empty() { "" } else { ", tools" });
                let mut prev = 0;
                for i in 0..2 {
                    let Some(at) = last_user(&p[i]) else {
                        bad.push(format!("{tag}: turn {} has no user-turn marker the server knows (nextsycl_tok::USER_TURNS)", i + 1));
                        break;
                    };
                    if !p[i + 1].starts_with(&p[i][..at]) {
                        let same = p[i].bytes().zip(p[i + 1].bytes()).take_while(|(a, b)| a == b).count();
                        bad.push(format!("{tag}: turn {} does not begin with turn {}'s prompt up to its last user turn (they part at byte {same} of {at}: {:?} / {:?})",
                                         i + 2, i + 1, &p[i][same..(same + 40).min(p[i].len())], &p[i + 1][same..(same + 40).min(p[i + 1].len())]));
                    }
                    if at <= prev || !p[i][..at].contains(sys.content) {
                        bad.push(format!("{tag}: turn {}'s checkpoint ({at}) does not move past the system prompt and the turns before", i + 1));
                    }
                    prev = at;
                }
            }
        }
    }
    assert!(bad.is_empty(), "prompt caching broken (rule 1, CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

#[test]
fn every_llm_engine_saves_checkpoints() {
    let mut bad = Vec::new();
    for arch in llm_arch_dirs().into_iter().filter(|a| a != "example") {
        let lib = std::fs::read_to_string(root().join(format!("llm/{arch}/src/lib.rs"))).unwrap();
        for f in ["fn cache_fingerprint(", "fn save(", "fn restore(", "fn read_checkpoint("] {
            let Some(at) = lib.find(f) else {
                bad.push(format!("llm/{arch}: no {f}..)"));
                continue;
            };
            let body = &lib[at..];
            let body = &body[..body[1..].find("\n    fn ").map_or(body.len(), |e| e + 1)];
            // a refusal: the body is an error outright (an error under a condition - a verify pass open - is fine)
            let first = body.split_once('{').map_or("", |b| b.1).trim_start();
            if ["Err(", "Ok(Err", "unimplemented!", "todo!", "panic!"].iter().any(|r| first.starts_with(r)) || body.contains("not supported") {
                bad.push(format!("llm/{arch}: {f}..) refuses: {}", body.lines().take(4).collect::<Vec<_>>().join(" ")));
            }
        }
    }
    assert!(bad.is_empty(), "engines without prompt-cache checkpoints (rule 1, CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

#[test]
fn silos_are_their_engines_own() {
    let root = root();
    let mut bad = Vec::new();
    let mut silos = Vec::new();
    for kind in ["llm", "image", "video", "audio", "model3d"] {
        for e in std::fs::read_dir(root.join("kernels").join(kind)).into_iter().flatten().flatten() {
            if e.path().join("silo").is_dir() {
                silos.push((kind, e.file_name().to_string_lossy().to_string()));
            }
        }
    }
    assert!(!silos.is_empty(), "no silos found: is the scan broken?");
    // every C / C++ file of the kernels: (its path relative to the root, its text)
    let mut files = Vec::new();
    let mut stack = vec![root.join("kernels")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if matches!(p.extension().and_then(|x| x.to_str()), Some("cpp" | "hpp" | "h")) {
                files.push((p.strip_prefix(&root).unwrap().display().to_string(), std::fs::read_to_string(&p).unwrap_or_default()));
            }
        }
    }
    for (kind, arch) in &silos {
        // its engine opens it (nextsycl_core::silo), so it runs without it on the shared kernels
        let src = root.join(kind).join(arch).join("src");
        let opens = std::fs::read_dir(&src).into_iter().flatten().flatten()
            .any(|f| std::fs::read_to_string(f.path()).unwrap_or_default().contains("nextsycl_core::silo("));
        if !opens {
            bad.push(format!("{kind}/{arch}: kernels/{kind}/{arch}/silo exists but {} never opens it (nextsycl_core::silo)", src.display()));
        }
        // nothing outside its arch includes it
        let own = format!("kernels/{kind}/{arch}/");
        for (path, text) in &files {
            if !path.starts_with(&own) && text.lines().any(|l| l.trim_start().starts_with("#include") && l.contains(&format!("{arch}/silo/"))) {
                bad.push(format!("{path} includes {arch}'s silo (a silo is its engine's own)"));
            }
        }
        // its header gates its ops
        let gated = files.iter().any(|(p, t)| p.starts_with(&format!("{own}silo/")) && p.ends_with(".h") && t.contains("_supported("));
        if !gated {
            bad.push(format!("kernels/{kind}/{arch}/silo: no `_supported` check in its header"));
        }
    }
    assert!(bad.is_empty(), "silos broken (rule 6, CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}
