//! The web front ends, inside the binary: `build.rs` embeds dist/wfe (every app's modules, pages and styles, the shared
//! vendor/ and static/) when it is built before the binary (./build.sh builds it first), so a server needs no files
//! beside it. The embedded copy is read first. A directory is read when the binary has none embedded (a build made
//! before ./build.sh wfe) or when NS_WFE_FROM_DISK=1 asks for it (editing a page without a rebuild).

use std::borrow::Cow;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/wfe_embedded.rs"));

/// Whether the front ends are in this binary
pub fn embedded() -> bool {
    !FILES.is_empty()
}

fn from_disk() -> bool {
    std::env::var("NS_WFE_FROM_DISK").is_ok_and(|v| v == "1")
}

fn embedded_file(rel: &str) -> Option<&'static [u8]> {
    FILES.binary_search_by(|(p, _)| (*p).cmp(rel)).ok().map(|i| FILES[i].1)
}

/// A file of the front ends by its path under dist/wfe ("home/index.html", "vendor/snabbdom/..."): the embedded copy,
/// else `root`'s (the directory)
pub fn read(root: &Path, rel: &str) -> Option<Cow<'static, [u8]>> {
    if !from_disk() {
        if let Some(b) = embedded_file(rel) {
            return Some(Cow::Borrowed(b));
        }
    }
    std::fs::read(root.join(rel)).ok().map(Cow::Owned)
}

/// `read` as text
pub fn text(root: &Path, rel: &str) -> Option<String> {
    read(root, rel).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// Whether a file of the front ends is there (embedded or in `root`)
pub fn exists(root: &Path, rel: &str) -> bool {
    (!from_disk() && embedded_file(rel).is_some()) || root.join(rel).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_set_is_sorted_and_found_by_path() {
        assert!(FILES.windows(2).all(|w| w[0].0 < w[1].0), "binary_search needs the paths sorted");
        if embedded() {
            assert!(embedded_file("home/index.html").is_some());
            assert!(exists(Path::new("/nonexistent"), "home/index.html"));
            assert!(read(Path::new("/nonexistent"), "home/index.html").is_some_and(|b| !b.is_empty()));
        }
        assert!(embedded_file("../etc/passwd").is_none());
        assert!(!exists(Path::new("/nonexistent"), "no/such/file"));
    }
}
