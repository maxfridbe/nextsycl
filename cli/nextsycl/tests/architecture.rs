//! The architecture's rules, checked on every `cargo test` (CONTRIBUTING.md explains them):
//!
//! 1. Layers: the foundation (crates/) depends only on the foundation; a kind's contract (<kind>/contract) on the
//!    foundation; an engine (<kind>/<arch>) on the foundation and its own kind's contract - never on another kind,
//!    the glue or the program; the glue (glue/) on the foundation and the contracts, never on an engine.
//! 2. One GPU runtime: SYCL. No crate that brings another (Vulkan, CUDA, OpenCL, Metal, wgpu, ONNX Runtime,
//!    PyTorch, candle, burn...) anywhere in Cargo.lock.
//! 3. The project's own kernels (kernels/ns, kernels/<kind>/<arch>) include no other runtime's headers.
//! 4. Every environment variable the code reads has a row in docs/environment.md.

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
            let Some(ddir) = crates.get(&d) else { continue }; // a third-party crate (rule 2 checks those)
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
    for kind in ["llm", "image", "video", "audio"] {
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
