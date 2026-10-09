//! The architecture's rules, checked on every `cargo test` (CONTRIBUTING.md explains them):
//!
//! 1. Layers: the foundation (crates/) depends only on the foundation; a kind's contract (<kind>/contract) on the
//!    foundation; an engine (<kind>/<arch>) on the foundation and its own kind's contract - never on another kind,
//!    the glue or the program; the glue (glue/) on the foundation and the contracts, never on an engine.
//! 2. One GPU runtime: SYCL. No crate that brings another (Vulkan, CUDA, OpenCL, Metal, wgpu, ONNX Runtime,
//!    PyTorch, candle, burn...) anywhere in Cargo.lock.
//! 3. The project's own kernels (kernels/ns, kernels/<kind>/<arch>) include no other runtime's headers.

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
    for kind in ["llm", "image", "video"] {
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
