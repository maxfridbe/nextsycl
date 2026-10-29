//! Embeds the kernel libraries (dist/libnextsycl-<kind>.so, libnextsycl-flash.so, libdnnl.so.3; or NS_LIB_DIST) into the
//! binary: `LIBS`, each as (file name, bytes). The kernels are built before the Rust (./build.sh does); without them the
//! set is empty and the libraries are opened from files (lib.rs). Silos stay files: they open on demand.
use std::fmt::Write;
use std::path::PathBuf;

const LIBS: &[&str] = &["libnextsycl-llm.so", "libnextsycl-image.so", "libnextsycl-video.so", "libnextsycl-audio.so", "libnextsycl-model3d.so",
                        "libnextsycl-flash.so", "libdnnl.so.3"];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    println!("cargo:rerun-if-env-changed=NS_LIB_DIST");
    let dist = std::env::var("NS_LIB_DIST").map(PathBuf::from).unwrap_or_else(|_| manifest.join("../../dist"));
    let mut src = String::from("pub static LIBS: &[(&str, &[u8])] = &[\n");
    for name in LIBS {
        let p = dist.join(name);
        println!("cargo:rerun-if-changed={}", p.display());
        // a real file (libnextsycl.so is a link to the llm library: not listed)
        if p.is_file() {
            let abs = std::fs::canonicalize(&p).unwrap_or(p);
            writeln!(src, "    ({name:?}, include_bytes!({:?})),", abs.display().to_string()).expect("string write");
        }
    }
    src.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it")).join("libs_embedded.rs");
    std::fs::write(out, src).expect("write libs_embedded.rs");
}
