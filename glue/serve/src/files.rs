//! The files the servers make, as the API reports them: size, SHA-256 (to check a download against), modified time.

use std::io::Read;
use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// A file's name, bytes, sha256 (hex) and mtime (unix seconds)
pub fn stat(p: &Path) -> Result<Value, String> {
    let meta = std::fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if !meta.is_file() {
        return Err(format!("{}: not a file", p.display()));
    }
    let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let sha: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0.0, |d| d.as_secs_f64());
    Ok(json!({"name": p.file_name().map(|n| n.to_string_lossy().into_owned()), "bytes": meta.len(), "sha256": sha, "mtime": mtime.round()}))
}

/// A plain file name (no directory, not hidden): what a delete or stat may name
pub fn plain(name: &str) -> Option<&str> {
    (!name.is_empty() && !name.contains('/') && !name.contains('\\') && !name.starts_with('.') && name != "..").then_some(name)
}
