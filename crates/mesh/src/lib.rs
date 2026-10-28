//! Meshes, as every 3D engine makes and takes them: triangles with their normals, UVs and vertex colours, a PBR
//! material (base colour, metallic-roughness and normal textures), a skeleton with skin weights, and a morph
//! animation (the vertices' positions frame by frame). Read from and written to GLB (glTF 2.0 binary, `glb`) and read
//! from OBJ (`obj`). The CPU tools a 3D pipeline runs around its models live here too (`tools`): weld, decimate,
//! remesh, re-UV.

pub mod glb;
pub mod obj;
pub mod tools;

/// A picture a material samples: RGBA8, rows top first
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Texture {
    pub fn new(width: u32, height: u32) -> Texture {
        Texture { width, height, rgba: vec![0; width as usize * height as usize * 4] }
    }

    /// As PNG bytes
    pub fn png(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, self.width, self.height);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().map_err(|e| format!("png: {e}"))?;
            w.write_image_data(&self.rgba).map_err(|e| format!("png: {e}"))?;
        }
        Ok(out)
    }

    /// From PNG or JPEG bytes
    pub fn decode(bytes: &[u8]) -> Result<Texture, String> {
        let img = image::load_from_memory(bytes).map_err(|e| format!("a texture: {e}"))?.to_rgba8();
        Ok(Texture { width: img.width(), height: img.height(), rgba: img.into_raw() })
    }

    /// The texel nearest to (u, v) (v down, as glTF), wrapped, as [0, 1] floats
    pub fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        if self.width == 0 || self.height == 0 {
            return [1.0; 4];
        }
        let x = ((u.rem_euclid(1.0) * self.width as f32) as u32).min(self.width - 1);
        let y = ((v.rem_euclid(1.0) * self.height as f32) as u32).min(self.height - 1);
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [0, 1, 2, 3].map(|c| self.rgba[i + c] as f32 / 255.0)
    }
}

/// A PBR material (glTF's metallic-roughness model)
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub base_color: Option<Texture>,
    /// glTF's packing: roughness in G, metallic in B
    pub metallic_roughness: Option<Texture>,
    pub normal: Option<Texture>,
    pub base_color_factor: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for Material {
    fn default() -> Material {
        Material { base_color: None, metallic_roughness: None, normal: None, base_color_factor: [1.0; 4], metallic: 0.0, roughness: 1.0 }
    }
}

/// A bone: its name, its parent (an index into the skeleton, before it), and where its head is (model space)
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub name: String,
    pub parent: Option<usize>,
    pub head: [f32; 3],
}

/// A skeleton and how each vertex follows it: up to four (joint, weight) pairs a vertex, the weights summing to 1
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skin {
    pub joints: Vec<Joint>,
    pub weights: Vec<[(u16, f32); 4]>,
    /// the joints in motion (glTF skeletal animation), if any
    pub motion: Option<SkelMotion>,
}

/// A skeleton in motion: every frame's rotation of every joint (a node's local rotation, as x y z w; the joints'
/// rest rotations are identity) and the root joint's translation, `fps` frames a second
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkelMotion {
    pub fps: f32,
    /// [frame][joint]
    pub rotations: Vec<Vec<[f32; 4]>>,
    /// the root joint's node translation a frame (its rest translation plus the motion's travel)
    pub root: Vec<[f32; 3]>,
    /// which joint `root` moves
    pub root_joint: usize,
}

/// An animation of the vertices: every frame's positions (each as long as the mesh's), `fps` frames a second
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Morph {
    pub fps: f32,
    pub frames: Vec<Vec<[f32; 3]>>,
}

/// A triangle mesh. `normals`, `uvs` and `colors` are empty or one a vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 4]>,
    /// three a triangle, counter-clockwise seen from outside
    pub indices: Vec<u32>,
    pub material: Option<Material>,
    pub skin: Option<Skin>,
    pub animation: Option<Morph>,
}

pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}
pub fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = length(a);
    if l > 0.0 { scale(a, 1.0 / l) } else { [0.0, 0.0, 1.0] }
}

impl Mesh {
    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// The smallest box holding every vertex: (min, max); a mesh without vertices: zeros
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        if self.positions.is_empty() { ([0.0; 3], [0.0; 3]) } else { (lo, hi) }
    }

    /// Area-weighted vertex normals from the triangles
    pub fn compute_normals(&mut self) {
        let mut n = vec![[0f32; 3]; self.positions.len()];
        for t in self.indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let f = cross(sub(self.positions[b], self.positions[a]), sub(self.positions[c], self.positions[a]));
            for v in [a, b, c] {
                n[v] = add(n[v], f);
            }
        }
        self.normals = n.into_iter().map(normalize).collect();
    }

    /// Checks what a writer and the tools rely on: indices in range, attributes one a vertex, the skin's and the
    /// animation's sizes
    pub fn validate(&self) -> Result<(), String> {
        let n = self.positions.len();
        if !self.indices.len().is_multiple_of(3) {
            return Err(format!("{} indices: not whole triangles", self.indices.len()));
        }
        if let Some(i) = self.indices.iter().find(|i| **i as usize >= n) {
            return Err(format!("index {i} past the {n} vertices"));
        }
        for (name, len) in [("normals", self.normals.len()), ("uvs", self.uvs.len()), ("colors", self.colors.len())] {
            if len != 0 && len != n {
                return Err(format!("{len} {name} for {n} vertices"));
            }
        }
        if let Some(s) = &self.skin {
            if s.weights.len() != n {
                return Err(format!("{} skin weights for {n} vertices", s.weights.len()));
            }
            if s.joints.len() > u16::MAX as usize {
                return Err("more joints than 16 bits hold".into());
            }
            for (i, j) in s.joints.iter().enumerate() {
                if j.parent.is_some_and(|p| p >= i) {
                    return Err(format!("joint {i} ({}): its parent must come before it", j.name));
                }
            }
            if let Some(w) = s.weights.iter().flatten().find(|(j, _)| *j as usize >= s.joints.len()) {
                return Err(format!("a skin weight names joint {} of {}", w.0, s.joints.len()));
            }
        }
        if let Some(a) = &self.animation {
            if let Some(f) = a.frames.iter().find(|f| f.len() != n) {
                return Err(format!("an animation frame of {} positions for {n} vertices", f.len()));
            }
            if a.fps.is_nan() || a.fps <= 0.0 {
                return Err("an animation needs fps > 0".into());
            }
        }
        Ok(())
    }

    /// Moved and scaled to fit [-1, 1] on its longest side, centred (what the generators make and the riggers take)
    pub fn normalize_to_unit(&mut self) {
        let (lo, hi) = self.bounds();
        let c = scale(add(lo, hi), 0.5);
        let half = (0..3).map(|i| hi[i] - lo[i]).fold(0f32, f32::max) * 0.5;
        let s = if half > 0.0 { 1.0 / half } else { 1.0 };
        let fit = |p: [f32; 3]| scale(sub(p, c), s);
        for p in &mut self.positions {
            *p = fit(*p);
        }
        if let Some(sk) = &mut self.skin {
            for j in &mut sk.joints {
                j.head = fit(j.head);
            }
        }
        if let Some(a) = &mut self.animation {
            for p in a.frames.iter_mut().flatten() {
                *p = fit(*p);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn quad() -> Mesh {
        Mesh {
            positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        }
    }

    #[test]
    fn normals_face_the_winding() {
        let mut m = quad();
        m.compute_normals();
        assert!(m.normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn unit_fit_and_validation() {
        let mut m = quad();
        m.normalize_to_unit();
        let (lo, hi) = m.bounds();
        assert_eq!((lo[0], hi[0], lo[1], hi[1]), (-1.0, 1.0, -1.0, 1.0));
        assert!(m.validate().is_ok());
        m.indices.push(9);
        assert!(m.validate().is_err());
    }

    #[test]
    fn a_texture_round_trips_through_png() {
        let mut t = Texture::new(2, 1);
        t.rgba.copy_from_slice(&[255, 0, 0, 255, 0, 0, 255, 128]);
        let back = Texture::decode(&t.png().unwrap()).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.sample(0.75, 0.5), [0.0, 0.0, 1.0, 128.0 / 255.0]);
    }
}
