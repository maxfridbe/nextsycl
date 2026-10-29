//! Embeds the built web front ends (dist/wfe, or NS_WFE_DIST) into the binary: `wfe::FILES`, every file as
//! ("app/path", bytes), sorted by path. Without a built dist/wfe the set is empty and the servers read the directory
//! (`wfe.rs`); build the front ends first (./build.sh wfe) for a single-file binary.
use std::fmt::Write;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, rel: &str, out: &mut Vec<(String, PathBuf)>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let (path, r) = (e.path(), if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") });
        if path.is_dir() {
            walk(&path, &r, out);
        } else {
            out.push((r, path));
        }
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    println!("cargo:rerun-if-env-changed=NS_WFE_DIST");
    let dist = std::env::var("NS_WFE_DIST").map(PathBuf::from).unwrap_or_else(|_| manifest.join("../../dist/wfe"));
    println!("cargo:rerun-if-changed={}", dist.display());
    let mut files = Vec::new();
    // an app is built when it has its page
    if dist.join("home/index.html").is_file() {
        walk(&dist, "", &mut files);
    }
    files.sort();
    let mut src = String::from("pub static FILES: &[(&str, &[u8])] = &[\n");
    for (rel, path) in &files {
        let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        writeln!(src, "    ({rel:?}, include_bytes!({:?})),", abs.display().to_string()).expect("string write");
    }
    src.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it")).join("wfe_embedded.rs");
    std::fs::write(out, src).expect("write wfe_embedded.rs");
}
