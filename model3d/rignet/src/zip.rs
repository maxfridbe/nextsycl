//! RigNet's weights as they are published: one ZIP (trained_models.zip, from the authors' Google Drive) holding a
//! legacy PyTorch checkpoint for each network, deflated. The engine reads the four it needs straight out of it -
//! inflated into a temporary folder for the load, removed after - so the downloaded file is the only one there is.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use nextsycl_model3d::{Error, Result};

fn u16le(b: &[u8], i: usize) -> u64 {
    u16::from_le_bytes([b[i], b[i + 1]]) as u64
}
fn u32le(b: &[u8], i: usize) -> u64 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as u64
}

fn read_at(f: &File, at: u64, n: usize, what: &str) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    f.read_exact_at(&mut v, at).map_err(|e| Error(format!("{what}: {e}")))?;
    Ok(v)
}

/// The entries of a ZIP (no ZIP64: RigNet's is 196 MB): (name, local header offset, compressed size, method)
fn entries(f: &File, what: &str) -> Result<Vec<(String, u64, u64, u64)>> {
    let len = f.metadata().map_err(|e| Error(format!("{what}: {e}")))?.len();
    let tail_len = len.min(65_557) as usize;
    let tail = read_at(f, len - tail_len as u64, tail_len, what)?;
    let eocd = (0..tail.len().saturating_sub(21)).rev().find(|&i| u32le(&tail, i) == 0x0605_4b50)
        .ok_or_else(|| Error(format!("{what}: not a ZIP file (RigNet's trained_models.zip)")))?;
    let (n, cd_len, cd_at) = (u16le(&tail, eocd + 10), u32le(&tail, eocd + 12), u32le(&tail, eocd + 16));
    let cd = read_at(f, cd_at, cd_len as usize, what)?;
    let mut out = Vec::new();
    let mut i = 0usize;
    for _ in 0..n {
        if i + 46 > cd.len() || u32le(&cd, i) != 0x0201_4b50 {
            return Err(Error(format!("{what}: a broken ZIP directory")));
        }
        let (method, csize) = (u16le(&cd, i + 10), u32le(&cd, i + 20));
        let (nl, el, cl) = (u16le(&cd, i + 28) as usize, u16le(&cd, i + 30) as usize, u16le(&cd, i + 32) as usize);
        let off = u32le(&cd, i + 42);
        out.push((String::from_utf8_lossy(&cd[i + 46..i + 46 + nl]).into_owned(), off, csize, method));
        i += 46 + nl + el + cl;
    }
    Ok(out)
}

/// The entry of the ZIP whose name ends with `suffix`, uncompressed
pub fn read(zip: &Path, suffix: &str) -> Result<Vec<u8>> {
    let what = zip.display().to_string();
    let f = File::open(zip).map_err(|e| Error(format!("{what}: {e}")))?;
    let (name, off, csize, method) = entries(&f, &what)?.into_iter().find(|e| e.0.ends_with(suffix))
        .ok_or_else(|| Error(format!("{what}: no {suffix} inside")))?;
    let lh = read_at(&f, off, 30, &what)?;
    let data = off + 30 + u16le(&lh, 26) + u16le(&lh, 28);
    let raw = read_at(&f, data, csize as usize, &what)?;
    match method {
        0 => Ok(raw),
        8 => miniz_oxide::inflate::decompress_to_vec(&raw).map_err(|e| Error(format!("{what}: {name}: {e:?}"))),
        m => Err(Error(format!("{what}: {name}: compression method {m}"))),
    }
}

/// A temporary folder removed when dropped
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new() -> Result<Scratch> {
        let d = std::env::temp_dir().join(format!("nextsycl-rignet-{}-{}", std::process::id(),
                                                  std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |t| t.as_nanos())));
        std::fs::create_dir_all(&d).map_err(|e| Error(format!("{}: {e}", d.display())))?;
        Ok(Scratch(d))
    }

    /// `suffix`'s entry of the ZIP written into the folder as `name`
    pub fn extract(&self, zip: &Path, suffix: &str, name: &str) -> Result<PathBuf> {
        let p = self.0.join(name);
        std::fs::write(&p, read(zip, suffix)?).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        Ok(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_entry_reads_back() {
        // one stored entry "a/b.txt" = "hello"
        let name = b"a/b.txt";
        let body = b"hello";
        let mut z = Vec::new();
        z.extend(0x0403_4b50u32.to_le_bytes());
        z.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend([0u8; 4]);
        z.extend((body.len() as u32).to_le_bytes());
        z.extend((body.len() as u32).to_le_bytes());
        z.extend((name.len() as u16).to_le_bytes());
        z.extend(0u16.to_le_bytes());
        z.extend(name);
        z.extend(body);
        let cd = z.len() as u32;
        z.extend(0x0201_4b50u32.to_le_bytes());
        z.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend([0u8; 4]);
        z.extend((body.len() as u32).to_le_bytes());
        z.extend((body.len() as u32).to_le_bytes());
        z.extend((name.len() as u16).to_le_bytes());
        z.extend([0u8; 12]);
        z.extend(0u32.to_le_bytes());
        z.extend(name);
        let cdl = z.len() as u32 - cd;
        z.extend(0x0605_4b50u32.to_le_bytes());
        z.extend([0, 0, 0, 0, 1, 0, 1, 0]);
        z.extend(cdl.to_le_bytes());
        z.extend(cd.to_le_bytes());
        z.extend([0, 0]);
        let s = Scratch::new().unwrap();
        let p = s.0.join("t.zip");
        std::fs::write(&p, &z).unwrap();
        assert_eq!(read(&p, "b.txt").unwrap(), b"hello");
        assert!(read(&p, "c.txt").is_err());
    }
}
