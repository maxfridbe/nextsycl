//! Wavefront OBJ: `v` (with optional colours), `vt`, `vn` and `f` (polygons fanned into triangles, negative indices
//! counted from the end). Each distinct (v, vt, vn) corner becomes a vertex. Materials and groups are ignored.

use std::collections::HashMap;

use crate::Mesh;

pub fn read(text: &str) -> Result<Mesh, String> {
    let (mut v, mut c, mut vt, mut vn) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut m = Mesh::default();
    let mut corners: HashMap<(usize, Option<usize>, Option<usize>), u32> = HashMap::new();
    let (mut any_uv, mut any_n, mut all_uv, mut all_n) = (false, false, true, true);
    let num = |s: &str, line: usize| s.parse::<f32>().map_err(|_| format!("line {line}: {s:?} is not a number"));
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let x: Vec<f32> = it.map(|s| num(s, ln + 1)).collect::<Result<_, _>>()?;
                if x.len() < 3 {
                    return Err(format!("line {}: a vertex needs 3 coordinates", ln + 1));
                }
                v.push([x[0], x[1], x[2]]);
                c.push((x.len() >= 6).then(|| [x[3], x[4], x[5], 1.0]));
            }
            Some("vt") => {
                let x: Vec<f32> = it.map(|s| num(s, ln + 1)).collect::<Result<_, _>>()?;
                // OBJ's v grows up; glTF's down
                vt.push([*x.first().unwrap_or(&0.0), 1.0 - *x.get(1).unwrap_or(&0.0)]);
            }
            Some("vn") => {
                let x: Vec<f32> = it.map(|s| num(s, ln + 1)).collect::<Result<_, _>>()?;
                if x.len() < 3 {
                    return Err(format!("line {}: a normal needs 3 coordinates", ln + 1));
                }
                vn.push([x[0], x[1], x[2]]);
            }
            Some("f") => {
                let mut poly = Vec::new();
                for corner in it {
                    let mut p = corner.split('/');
                    let idx = |s: Option<&str>, len: usize| -> Result<Option<usize>, String> {
                        match s.filter(|s| !s.is_empty()) {
                            None => Ok(None),
                            Some(s) => {
                                let i: i64 = s.parse().map_err(|_| format!("line {}: index {s:?}", ln + 1))?;
                                let k = if i < 0 { len as i64 + i } else { i - 1 };
                                if k < 0 || k as usize >= len {
                                    return Err(format!("line {}: index {i} out of range", ln + 1));
                                }
                                Ok(Some(k as usize))
                            }
                        }
                    };
                    let vi = idx(p.next(), v.len())?.ok_or_else(|| format!("line {}: a face corner without a vertex", ln + 1))?;
                    let ti = idx(p.next(), vt.len())?;
                    let ni = idx(p.next(), vn.len())?;
                    any_uv |= ti.is_some();
                    any_n |= ni.is_some();
                    all_uv &= ti.is_some();
                    all_n &= ni.is_some();
                    let key = (vi, ti, ni);
                    let id = *corners.entry(key).or_insert_with(|| {
                        m.positions.push(v[vi]);
                        m.uvs.push(ti.map_or([0.0, 0.0], |t| vt[t]));
                        m.normals.push(ni.map_or([0.0, 0.0, 0.0], |n| vn[n]));
                        m.colors.push(c[vi].unwrap_or([1.0; 4]));
                        (m.positions.len() - 1) as u32
                    });
                    poly.push(id);
                }
                for k in 1..poly.len().saturating_sub(1) {
                    m.indices.extend([poly[0], poly[k], poly[k + 1]]);
                }
            }
            _ => {}
        }
    }
    if m.indices.is_empty() {
        return Err("no faces in the file".into());
    }
    if !(any_uv && all_uv) {
        m.uvs.clear();
    }
    if !(any_n && all_n) {
        m.normals.clear();
    }
    if c.iter().all(|x| x.is_none()) {
        m.colors.clear();
    }
    m.validate()?;
    Ok(m)
}

/// The mesh as OBJ text (positions, UVs, normals; one corner index set a vertex)
pub fn write(m: &Mesh) -> String {
    let mut s = String::from("# nextsycl\n");
    for (i, p) in m.positions.iter().enumerate() {
        match m.colors.get(i) {
            Some(c) => s += &format!("v {} {} {} {} {} {}\n", p[0], p[1], p[2], c[0], c[1], c[2]),
            None => s += &format!("v {} {} {}\n", p[0], p[1], p[2]),
        }
    }
    for t in &m.uvs {
        s += &format!("vt {} {}\n", t[0], 1.0 - t[1]);
    }
    for n in &m.normals {
        s += &format!("vn {} {} {}\n", n[0], n[1], n[2]);
    }
    let (u, n) = (!m.uvs.is_empty(), !m.normals.is_empty());
    for t in m.indices.chunks_exact(3) {
        s += "f";
        for &i in t {
            let i = i + 1;
            s += &match (u, n) {
                (true, true) => format!(" {i}/{i}/{i}"),
                (true, false) => format!(" {i}/{i}"),
                (false, true) => format!(" {i}//{i}"),
                (false, false) => format!(" {i}"),
            };
        }
        s += "\n";
    }
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_quad_fans_and_round_trips() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n";
        let m = super::read(text).unwrap();
        assert_eq!(m.triangles(), 2);
        assert_eq!(m.uvs[0], [0.0, 1.0]);
        let back = super::read(&super::write(&m)).unwrap();
        assert_eq!(back.positions, m.positions);
        assert_eq!(back.uvs, m.uvs);
        assert_eq!(back.indices, m.indices);
        assert!(super::read("f -1 -2 -3\n").is_err());
    }
}
