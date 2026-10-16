//! Float32 .npy files: what the reference script and the stage checks hand each other

use std::path::Path;

use nextsycl_model3d::{Error, Result};

/// A little-endian float32 .npy: (shape, values)
pub fn read(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if !head.contains("'<f4'") || head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: {head} (float32, C order expected)", path.display())));
    }
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    Ok((shape, b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
}

/// `v` as a float32 .npy of `shape`
pub fn write(path: &Path, shape: &[usize], v: &[f32]) -> Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let tuple = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut head = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {tuple}, }}");
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((head.len() as u16).to_le_bytes());
    b.extend(head.as_bytes());
    b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
}

/// (relative RMS error, max abs error)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut e2, mut w2, mut mx) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        mx = mx.max((g - w).abs());
    }
    ((e2 / w2.max(1e-30)).sqrt(), mx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_round_trips() {
        let p = std::env::temp_dir().join(format!("ns-rignet-npy-{}.npy", std::process::id()));
        write(&p, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (s, v) = read(&p).unwrap();
        assert_eq!(s, vec![2, 3]);
        assert_eq!(v[5], 6.0);
        write(&p, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(read(&p).unwrap().0, vec![4]);
        std::fs::remove_file(&p).ok();
    }
}
