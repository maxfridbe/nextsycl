//! GLB: glTF 2.0's binary form - a JSON chunk describing the scene and a BIN chunk holding the arrays and pictures.
//! `write` makes one mesh node (its material's textures as PNG; a skin as a joint node tree with inverse bind
//! matrices, JOINTS_0 / WEIGHTS_0; a morph animation as one target a frame and a weights track that steps through
//! them). `read` takes every triangle primitive of the default scene, its node transforms applied, into one mesh,
//! with the first textured material's pictures.

use serde_json::{json, Value};

use crate::{Joint, Material, Mesh, Morph, SkelMotion, Skin, Texture};

const GLB: u32 = 0x4654_6C67; // "glTF"
const JSON: u32 = 0x4E4F_534A;
const BIN: u32 = 0x004E_4942;

const FLOAT: u32 = 5126;
const U8: u32 = 5121;
const U16: u32 = 5123;
const U32: u32 = 5125;

/// The BIN chunk being built and the views and accessors into it
#[derive(Default)]
struct Out {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Out {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let mut v = json!({"buffer": 0, "byteOffset": self.bin.len(), "byteLength": bytes.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.bin.extend_from_slice(bytes);
        self.views.push(v);
        self.views.len() - 1
    }

    fn accessor(&mut self, bytes: &[u8], component: u32, count: usize, kind: &str, target: Option<u32>, extra: Value) -> usize {
        let view = self.view(bytes, target);
        let mut a = json!({"bufferView": view, "componentType": component, "count": count, "type": kind});
        if let (Some(o), Some(e)) = (a.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }

    fn floats<const N: usize>(&mut self, v: &[[f32; N]], kind: &str, bounds: bool) -> usize {
        let bytes: Vec<u8> = v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
        let extra = if bounds && !v.is_empty() {
            let lo: Vec<f32> = (0..N).map(|c| v.iter().map(|p| p[c]).fold(f32::INFINITY, f32::min)).collect();
            let hi: Vec<f32> = (0..N).map(|c| v.iter().map(|p| p[c]).fold(f32::NEG_INFINITY, f32::max)).collect();
            json!({"min": lo, "max": hi})
        } else {
            json!({})
        };
        self.accessor(&bytes, FLOAT, v.len(), kind, Some(34962), extra)
    }
}

/// A column-major 4x4 translation
fn translation(t: [f32; 3]) -> [f32; 16] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, t[0], t[1], t[2], 1.0]
}

/// The mesh as GLB bytes
pub fn write(m: &Mesh) -> Result<Vec<u8>, String> {
    m.validate()?;
    let mut o = Out::default();
    let mut attrs = json!({"POSITION": o.floats(&m.positions, "VEC3", true)});
    if !m.normals.is_empty() {
        attrs["NORMAL"] = json!(o.floats(&m.normals, "VEC3", false));
    }
    if !m.uvs.is_empty() {
        attrs["TEXCOORD_0"] = json!(o.floats(&m.uvs, "VEC2", false));
    }
    if !m.colors.is_empty() {
        attrs["COLOR_0"] = json!(o.floats(&m.colors, "VEC4", false));
    }
    let ib: Vec<u8> = m.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let indices = o.accessor(&ib, U32, m.indices.len(), "SCALAR", Some(34963), json!({}));
    let mut prim = json!({"attributes": attrs, "indices": indices, "mode": 4});

    let (mut images, mut textures, mut materials) = (Vec::new(), Vec::new(), Vec::new());
    if let Some(mat) = &m.material {
        let mut tex = |t: &Option<Texture>, o: &mut Out| -> Result<Option<usize>, String> {
            let Some(t) = t else { return Ok(None) };
            let view = o.view(&t.png()?, None);
            images.push(json!({"bufferView": view, "mimeType": "image/png"}));
            textures.push(json!({"sampler": 0, "source": images.len() - 1}));
            Ok(Some(textures.len() - 1))
        };
        let mut pbr = json!({"baseColorFactor": mat.base_color_factor, "metallicFactor": mat.metallic, "roughnessFactor": mat.roughness});
        if let Some(i) = tex(&mat.base_color, &mut o)? {
            pbr["baseColorTexture"] = json!({"index": i});
        }
        if let Some(i) = tex(&mat.metallic_roughness, &mut o)? {
            pbr["metallicRoughnessTexture"] = json!({"index": i});
        }
        let mut material = json!({"name": "material", "pbrMetallicRoughness": pbr, "doubleSided": false});
        if let Some(i) = tex(&mat.normal, &mut o)? {
            material["normalTexture"] = json!({"index": i});
        }
        materials.push(material);
        prim["material"] = json!(0);
    }

    // node 0: the mesh; then the joints (node 1 + j)
    let mut mesh_node = json!({"name": "mesh", "mesh": 0});
    let mut nodes = vec![];
    let mut scene_nodes = vec![json!(0)];
    let mut skins = Vec::new();
    if let Some(s) = &m.skin {
        let j: Vec<[u16; 4]> = s.weights.iter().map(|w| w.map(|x| x.0)).collect();
        let jb: Vec<u8> = j.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
        prim["attributes"]["JOINTS_0"] = json!(o.accessor(&jb, U16, j.len(), "VEC4", Some(34962), json!({})));
        let w: Vec<[f32; 4]> = s.weights.iter().map(|w| {
            let sum: f32 = w.iter().map(|x| x.1).sum();
            w.map(|x| if sum > 0.0 { x.1 / sum } else { 0.0 })
        }).collect();
        prim["attributes"]["WEIGHTS_0"] = json!(o.floats(&w, "VEC4", false));
        let ibm: Vec<[f32; 16]> = s.joints.iter().map(|j| translation(crate::scale(j.head, -1.0))).collect();
        let ib: Vec<u8> = ibm.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
        let inverse = o.accessor(&ib, FLOAT, ibm.len(), "MAT4", None, json!({}));
        let mut joint_nodes: Vec<Value> = s.joints.iter().map(|j| {
            let rel = match j.parent {
                Some(p) => crate::sub(j.head, s.joints[p].head),
                None => j.head,
            };
            json!({"name": j.name, "translation": rel})
        }).collect();
        for (i, j) in s.joints.iter().enumerate() {
            match j.parent {
                Some(p) => {
                    let kids = joint_nodes[p]["children"].as_array().cloned().unwrap_or_default();
                    let mut kids = kids;
                    kids.push(json!(1 + i));
                    joint_nodes[p]["children"] = json!(kids);
                }
                None => scene_nodes.push(json!(1 + i)),
            }
        }
        let roots: Vec<usize> = s.joints.iter().enumerate().filter(|(_, j)| j.parent.is_none()).map(|(i, _)| 1 + i).collect();
        let mut skin = json!({"joints": (1..=s.joints.len()).collect::<Vec<_>>(), "inverseBindMatrices": inverse});
        if roots.len() == 1 {
            skin["skeleton"] = json!(roots[0]);
        }
        skins.push(skin);
        mesh_node["skin"] = json!(0);
        nodes = joint_nodes;
    }

    let mut animations = Vec::new();
    let mut weights0 = None;
    if let Some(a) = &m.animation {
        if !a.frames.is_empty() {
            // one target a frame (its offset from the rest pose), and a track that weighs frame k fully at its time
            let mut targets = Vec::new();
            for f in &a.frames {
                let d: Vec<[f32; 3]> = f.iter().zip(&m.positions).map(|(p, b)| crate::sub(*p, *b)).collect();
                targets.push(json!({"POSITION": o.floats(&d, "VEC3", true)}));
            }
            prim["targets"] = json!(targets);
            let n = a.frames.len();
            let times: Vec<[f32; 1]> = (0..n).map(|k| [k as f32 / a.fps]).collect();
            let input = {
                let b: Vec<u8> = times.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
                o.accessor(&b, FLOAT, n, "SCALAR", None, json!({"min": [0.0], "max": [(n - 1) as f32 / a.fps]}))
            };
            let w: Vec<f32> = (0..n).flat_map(|k| (0..n).map(move |t| if t == k { 1.0 } else { 0.0 })).collect();
            let output = {
                let b: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                o.accessor(&b, FLOAT, n * n, "SCALAR", None, json!({}))
            };
            animations.push(json!({"name": "motion", "samplers": [{"input": input, "output": output, "interpolation": "LINEAR"}],
                                   "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]}));
            weights0 = Some(vec![0.0f32; n]);
        }
    }

    if let Some((s, mo)) = m.skin.as_ref().and_then(|s| s.motion.as_ref().map(|mo| (s, mo))) {
        let n = mo.rotations.len();
        if n > 0 && mo.rotations.iter().all(|f| f.len() == s.joints.len()) && mo.fps > 0.0 {
            // skeletal animation: one rotation track a joint and a translation track for the root, one shared clock
            let times: Vec<f32> = (0..n).map(|k| k as f32 / mo.fps).collect();
            let tb: Vec<u8> = times.iter().flat_map(|x| x.to_le_bytes()).collect();
            let input = o.accessor(&tb, FLOAT, n, "SCALAR", None, json!({"min": [0.0], "max": [times[n - 1]]}));
            let (mut samplers, mut channels) = (Vec::new(), Vec::new());
            for j in 0..s.joints.len() {
                let q: Vec<u8> = (0..n).flat_map(|k| mo.rotations[k][j]).flat_map(|x| x.to_le_bytes()).collect();
                let out = o.accessor(&q, FLOAT, n, "VEC4", None, json!({}));
                samplers.push(json!({"input": input, "output": out, "interpolation": "LINEAR"}));
                channels.push(json!({"sampler": samplers.len() - 1, "target": {"node": 1 + j, "path": "rotation"}}));
            }
            if mo.root.len() == n && mo.root_joint < s.joints.len() {
                let t: Vec<u8> = mo.root.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
                let out = o.accessor(&t, FLOAT, n, "VEC3", None, json!({}));
                samplers.push(json!({"input": input, "output": out, "interpolation": "LINEAR"}));
                channels.push(json!({"sampler": samplers.len() - 1, "target": {"node": 1 + mo.root_joint, "path": "translation"}}));
            }
            animations.push(json!({"name": "skeleton", "samplers": samplers, "channels": channels}));
        }
    }

    let mut mesh = json!({"name": "mesh", "primitives": [prim]});
    if let Some(w) = weights0 {
        mesh["weights"] = json!(w);
    }
    let mut all_nodes = vec![mesh_node];
    all_nodes.extend(nodes);
    while !o.bin.len().is_multiple_of(4) {
        o.bin.push(0);
    }
    let mut doc = json!({
        "asset": {"version": "2.0", "generator": "nextsycl"},
        "scene": 0, "scenes": [{"nodes": scene_nodes}], "nodes": all_nodes, "meshes": [mesh],
        "buffers": [{"byteLength": o.bin.len()}], "bufferViews": o.views, "accessors": o.accessors,
    });
    if !materials.is_empty() {
        doc["materials"] = json!(materials);
        doc["textures"] = json!(textures);
        doc["images"] = json!(images);
        doc["samplers"] = json!([{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]);
    }
    if !skins.is_empty() {
        doc["skins"] = json!(skins);
    }
    if !animations.is_empty() {
        doc["animations"] = json!(animations);
    }
    let mut js = serde_json::to_vec(&doc).map_err(|e| e.to_string())?;
    while !js.len().is_multiple_of(4) {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + o.bin.len();
    let mut out = Vec::with_capacity(total);
    for w in [GLB, 2, total as u32, js.len() as u32, JSON] {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&js);
    out.extend_from_slice(&(o.bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&BIN.to_le_bytes());
    out.extend_from_slice(&o.bin);
    Ok(out)
}

/// A GLB file's JSON and BIN chunks
pub fn chunks(b: &[u8]) -> Result<(Value, &[u8]), String> {
    let u32at = |i: usize| -> Result<u32, String> {
        b.get(i..i + 4).map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]])).ok_or_else(|| "a GLB cut short".to_string())
    };
    if b.len() < 20 || u32at(0)? != GLB {
        return Err("not a GLB file (glTF binary)".into());
    }
    if u32at(4)? != 2 {
        return Err(format!("glTF version {}: 2 expected", u32at(4)?));
    }
    let jl = u32at(12)? as usize;
    if u32at(16)? != JSON {
        return Err("a GLB whose first chunk is not JSON".into());
    }
    let js = b.get(20..20 + jl).ok_or("a GLB cut short")?;
    let doc: Value = serde_json::from_slice(js).map_err(|e| format!("its JSON: {e}"))?;
    let at = 20 + jl;
    let bin: &[u8] = if at + 8 <= b.len() && u32at(at + 4)? == BIN {
        let bl = u32at(at)? as usize;
        b.get(at + 8..at + 8 + bl).ok_or("a GLB cut short")?
    } else {
        &[]
    };
    Ok((doc, bin))
}

struct In<'a> {
    doc: &'a Value,
    bin: &'a [u8],
}

impl In<'_> {
    fn view(&self, i: usize) -> Result<&[u8], String> {
        let v = &self.doc["bufferViews"][i];
        if v["buffer"].as_u64().unwrap_or(0) != 0 {
            return Err("a buffer other than the GLB's own".into());
        }
        let off = v["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = v["byteLength"].as_u64().ok_or("a bufferView without byteLength")? as usize;
        self.bin.get(off..off + len).ok_or_else(|| "a bufferView past the BIN chunk".into())
    }

    /// An accessor's elements as floats (normalized integers scaled to [0, 1]), `n` components each
    fn floats(&self, i: usize) -> Result<(Vec<f32>, usize), String> {
        let a = &self.doc["accessors"][i];
        if a.get("sparse").is_some() {
            return Err("sparse accessors are not read".into());
        }
        let n = match a["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            Some("MAT4") => 16,
            t => return Err(format!("accessor type {t:?}")),
        };
        let count = a["count"].as_u64().unwrap_or(0) as usize;
        let ct = a["componentType"].as_u64().unwrap_or(0) as u32;
        let norm = a["normalized"] == true;
        let size = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return Err(format!("component type {ct}")),
        };
        let Some(view) = a["bufferView"].as_u64() else { return Ok((vec![0.0; count * n], n)) };
        let data = self.view(view as usize)?;
        let off = a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = self.doc["bufferViews"][view as usize]["byteStride"].as_u64().map_or(size * n, |s| s as usize);
        let mut out = Vec::with_capacity(count * n);
        for e in 0..count {
            for c in 0..n {
                let p = off + e * stride + c * size;
                let x = data.get(p..p + size).ok_or("an accessor past its bufferView")?;
                out.push(match ct {
                    5126 => f32::from_le_bytes([x[0], x[1], x[2], x[3]]),
                    5125 => u32::from_le_bytes([x[0], x[1], x[2], x[3]]) as f32,
                    5123 => { let v = u16::from_le_bytes([x[0], x[1]]) as f32; if norm { v / 65535.0 } else { v } }
                    5122 => { let v = i16::from_le_bytes([x[0], x[1]]) as f32; if norm { (v / 32767.0).max(-1.0) } else { v } }
                    5121 => { let v = x[0] as f32; if norm { v / 255.0 } else { v } }
                    _ => { let v = x[0] as i8 as f32; if norm { (v / 127.0).max(-1.0) } else { v } }
                });
            }
        }
        Ok((out, n))
    }

    /// Integer indices (exact past 2^24, unlike floats)
    fn indices(&self, i: usize) -> Result<Vec<u32>, String> {
        let a = &self.doc["accessors"][i];
        let count = a["count"].as_u64().unwrap_or(0) as usize;
        let ct = a["componentType"].as_u64().unwrap_or(0) as u32;
        let size = match ct { U8 => 1, U16 => 2, U32 => 4, _ => return Err(format!("index component type {ct}")) };
        let view = a["bufferView"].as_u64().ok_or("indices without a bufferView")? as usize;
        let data = self.view(view)?;
        let off = a["byteOffset"].as_u64().unwrap_or(0) as usize;
        (0..count).map(|e| {
            let p = off + e * size;
            let x = data.get(p..p + size).ok_or("indices past their bufferView")?;
            Ok(match size { 1 => x[0] as u32, 2 => u16::from_le_bytes([x[0], x[1]]) as u32, _ => u32::from_le_bytes([x[0], x[1], x[2], x[3]]) })
        }).collect()
    }

    fn texture(&self, t: &Value) -> Result<Option<Texture>, String> {
        let Some(ti) = t["index"].as_u64() else { return Ok(None) };
        let Some(src) = self.doc["textures"][ti as usize]["source"].as_u64() else { return Ok(None) };
        let img = &self.doc["images"][src as usize];
        match img["bufferView"].as_u64() {
            Some(v) => Texture::decode(self.view(v as usize)?).map(Some),
            None => Ok(None), // an external picture: not here
        }
    }
}

/// A node's local matrix (column-major): `matrix`, else T * R * S
fn local(n: &Value) -> [f32; 16] {
    if let Some(m) = n["matrix"].as_array().filter(|m| m.len() == 16) {
        let mut out = [0f32; 16];
        for (o, v) in out.iter_mut().zip(m) {
            *o = v.as_f64().unwrap_or(0.0) as f32;
        }
        return out;
    }
    let get = |k: &str, d: &[f32]| -> Vec<f32> {
        n[k].as_array().map_or(d.to_vec(), |a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
    };
    let t = get("translation", &[0.0, 0.0, 0.0]);
    let r = get("rotation", &[0.0, 0.0, 0.0, 1.0]);
    let s = get("scale", &[1.0, 1.0, 1.0]);
    let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
    let rot = [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
        2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
        2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y),
    ];
    let mut m = [0f32; 16];
    for c in 0..3 {
        for r_ in 0..3 {
            m[c * 4 + r_] = rot[c * 3 + r_] * s[c];
        }
    }
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m[15] = 1.0;
    m
}

fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut o = [0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    o
}

fn apply(m: &[f32; 16], p: [f32; 3], w: f32) -> [f32; 3] {
    [0, 1, 2].map(|r| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] * w)
}

/// A GLB's triangles as one mesh (every mesh node of the default scene, transformed), with the first textured
/// material's pictures and the first skin when the file has a single skinned primitive
pub fn read(bytes: &[u8]) -> Result<Mesh, String> {
    let (doc, bin) = chunks(bytes)?;
    let inp = In { doc: &doc, bin };
    let scene = doc["scene"].as_u64().unwrap_or(0) as usize;
    let roots: Vec<usize> = match doc["scenes"][scene]["nodes"].as_array() {
        Some(a) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect(),
        None => (0..doc["nodes"].as_array().map_or(0, |a| a.len())).collect(),
    };
    let ident = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut stack: Vec<(usize, [f32; 16], usize)> = roots.into_iter().map(|r| (r, ident, 0)).collect();
    let mut m = Mesh::default();
    let mut has = (true, true, true); // normals, uvs, colors in every primitive so far
    let mut prims = 0;
    let (mut skin_of, mut skin_broken) = (None::<u64>, false);
    while let Some((ni, parent, depth)) = stack.pop() {
        if depth > 64 {
            return Err("a node tree deeper than 64 (a cycle?)".into());
        }
        let node = &doc["nodes"][ni];
        let world = mul(&parent, &local(node));
        if let Some(kids) = node["children"].as_array() {
            stack.extend(kids.iter().filter_map(|k| k.as_u64()).map(|k| (k as usize, world, depth + 1)));
        }
        let Some(mi) = node["mesh"].as_u64() else { continue };
        // a skinned mesh's own node transform does not apply (glTF 2.0, skins): its vertices are in the bind space the
        // inverse bind matrices map from - the joints' heads read below are there too
        let world = if node["skin"].as_u64().is_some() { ident } else { world };
        for p in doc["meshes"][mi as usize]["primitives"].as_array().into_iter().flatten() {
            if p["mode"].as_u64().unwrap_or(4) != 4 {
                continue; // points and lines
            }
            let a = &p["attributes"];
            let Some(pos) = a["POSITION"].as_u64() else { continue };
            let (pv, _) = inp.floats(pos as usize)?;
            let base = m.positions.len() as u32;
            let n = pv.len() / 3;
            m.positions.extend(pv.chunks_exact(3).map(|c| apply(&world, [c[0], c[1], c[2]], 1.0)));
            let attr = |k: &str| -> Result<Option<(Vec<f32>, usize)>, String> { a[k].as_u64().map(|i| inp.floats(i as usize)).transpose() };
            match attr("NORMAL")? {
                Some((v, _)) if has.0 => m.normals.extend(v.chunks_exact(3).map(|c| crate::normalize(apply(&world, [c[0], c[1], c[2]], 0.0)))),
                _ => has.0 = false,
            }
            match attr("TEXCOORD_0")? {
                Some((v, _)) if has.1 => m.uvs.extend(v.chunks_exact(2).map(|c| [c[0], c[1]])),
                _ => has.1 = false,
            }
            match attr("COLOR_0")? {
                Some((v, k)) if has.2 => m.colors.extend(v.chunks_exact(k).map(|c| [c[0], c[1], c[2], if k == 4 { c[3] } else { 1.0 }])),
                _ => has.2 = false,
            }
            match p["indices"].as_u64() {
                Some(i) => m.indices.extend(inp.indices(i as usize)?.into_iter().map(|x| x + base)),
                None => m.indices.extend((0..n as u32).map(|x| x + base)),
            }
            if m.material.is_none() {
                if let Some(mat) = p["material"].as_u64().map(|i| &doc["materials"][i as usize]) {
                    let pbr = &mat["pbrMetallicRoughness"];
                    let f = |v: &Value, d: f32| v.as_f64().map_or(d, |x| x as f32);
                    let mut bc = [1.0f32; 4];
                    if let Some(a) = pbr["baseColorFactor"].as_array() {
                        for (o, v) in bc.iter_mut().zip(a) {
                            *o = f(v, 1.0);
                        }
                    }
                    let material = Material {
                        base_color: inp.texture(&pbr["baseColorTexture"])?,
                        metallic_roughness: inp.texture(&pbr["metallicRoughnessTexture"])?,
                        normal: inp.texture(&mat["normalTexture"])?,
                        base_color_factor: bc,
                        metallic: f(&pbr["metallicFactor"], 1.0),
                        roughness: f(&pbr["roughnessFactor"], 1.0),
                    };
                    m.material = Some(material);
                }
            }
            // one skin over every primitive (the same skin node-to-node): the weights are appended in order
            let skinned = match (node["skin"].as_u64(), a["JOINTS_0"].as_u64(), a["WEIGHTS_0"].as_u64()) {
                (Some(si), Some(j), Some(w)) if prims == 0 || skin_of == Some(si) => Some((si, read_skin(&inp, si as usize, j as usize, w as usize)?)),
                _ => None,
            };
            match skinned {
                Some((si, sk)) => match m.skin.as_mut() {
                    Some(have) if prims > 0 => have.weights.extend(sk.weights),
                    _ => {
                        skin_of = Some(si);
                        m.skin = Some(sk);
                    }
                },
                None => skin_broken = true,
            }
            prims += 1;
        }
    }
    if prims == 0 {
        return Err("no triangles in the file".into());
    }
    let nv = m.positions.len();
    if skin_broken {
        m.skin = None;
    }
    if !has.0 || m.normals.len() != nv {
        m.normals.clear();
    }
    if !has.1 || m.uvs.len() != nv {
        m.uvs.clear();
    }
    if !has.2 || m.colors.len() != nv {
        m.colors.clear();
    }
    if m.skin.as_ref().is_some_and(|s| s.weights.len() != nv) {
        m.skin = None;
    }
    m.validate()?;
    Ok(m)
}

/// A skin: its joints' heads from the inverse bind matrices (else the nodes' rest transforms), parents from the
/// node tree, and the per-vertex weights
fn read_skin(inp: &In, si: usize, j: usize, w: usize) -> Result<Skin, String> {
    let s = &inp.doc["skins"][si];
    let nodes: Vec<usize> = s["joints"].as_array().ok_or("a skin without joints")?.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
    let ibm = match s["inverseBindMatrices"].as_u64() {
        Some(i) => Some(inp.floats(i as usize)?.0),
        None => None,
    };
    let mut joints = Vec::new();
    for (k, &n) in nodes.iter().enumerate() {
        let head = match &ibm {
            Some(m) if m.len() >= (k + 1) * 16 => {
                // the joint's world position: the inverse of its inverse bind matrix applied to the origin
                let x: [f32; 16] = m[k * 16..k * 16 + 16].try_into().unwrap();
                invert_affine(&x).map_or([0.0; 3], |inv| [inv[12], inv[13], inv[14]])
            }
            _ => [0.0; 3],
        };
        let parent = nodes.iter().position(|&p| inp.doc["nodes"][p]["children"].as_array().is_some_and(|c| c.iter().any(|c| c.as_u64() == Some(n as u64))));
        joints.push(Joint { name: inp.doc["nodes"][n]["name"].as_str().unwrap_or("joint").to_string(), parent, head });
    }
    // parents must come first: reorder when the file lists a child ahead of its parent
    let order = topo(&joints);
    let remap: Vec<usize> = { let mut r = vec![0; order.len()]; for (new, &old) in order.iter().enumerate() { r[old] = new; } r };
    let joints: Vec<Joint> = order.iter().map(|&i| { let mut j = joints[i].clone(); j.parent = j.parent.map(|p| remap[p]); j }).collect();
    let (jv, _) = inp.floats(j)?;
    let (wv, _) = inp.floats(w)?;
    let weights = jv.chunks_exact(4).zip(wv.chunks_exact(4))
        .map(|(a, b)| [0, 1, 2, 3].map(|c| ((remap.get(a[c] as usize).copied().unwrap_or(0)) as u16, b[c])))
        .collect();
    Ok(Skin { joints, weights, motion: None })
}

fn topo(j: &[Joint]) -> Vec<usize> {
    let mut out = Vec::with_capacity(j.len());
    let mut done = vec![false; j.len()];
    fn visit(i: usize, j: &[Joint], done: &mut Vec<bool>, out: &mut Vec<usize>, depth: usize) {
        if done[i] || depth > j.len() {
            return;
        }
        if let Some(p) = j[i].parent {
            visit(p, j, done, out, depth + 1);
        }
        if !done[i] {
            done[i] = true;
            out.push(i);
        }
    }
    for i in 0..j.len() {
        visit(i, j, &mut done, &mut out, 0);
    }
    out
}

/// The inverse of an affine column-major 4x4
fn invert_affine(m: &[f32; 16]) -> Option<[f32; 16]> {
    let a = [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = |r: usize, c: usize| -> f32 {
        let (r1, r2, c1, c2) = ((r + 1) % 3, (r + 2) % 3, (c + 1) % 3, (c + 2) % 3);
        (a[c1][r1] * a[c2][r2] - a[c1][r2] * a[c2][r1]) / det
    };
    let mut o = [0f32; 16];
    for r in 0..3 {
        for c in 0..3 {
            o[c * 4 + r] = inv(r, c);
        }
    }
    let t = [m[12], m[13], m[14]];
    for r in 0..3 {
        o[12 + r] = -(o[r] * t[0] + o[4 + r] * t[1] + o[8 + r] * t[2]);
    }
    o[15] = 1.0;
    Some(o)
}

/// A morph animation read back (the targets of the first primitive as frames): for tests and tools
pub fn read_morph(bytes: &[u8]) -> Result<Option<Morph>, String> {
    let (doc, bin) = chunks(bytes)?;
    let inp = In { doc: &doc, bin };
    let p = &doc["meshes"][0]["primitives"][0];
    let Some(targets) = p["targets"].as_array() else { return Ok(None) };
    let (base, _) = inp.floats(p["attributes"]["POSITION"].as_u64().ok_or("no POSITION")? as usize)?;
    let mut frames = Vec::new();
    for t in targets {
        let (d, _) = inp.floats(t["POSITION"].as_u64().ok_or("a target without POSITION")? as usize)?;
        frames.push(base.chunks_exact(3).zip(d.chunks_exact(3)).map(|(b, d)| [b[0] + d[0], b[1] + d[1], b[2] + d[2]]).collect());
    }
    let times = doc["animations"][0]["samplers"][0]["input"].as_u64().map(|i| inp.floats(i as usize)).transpose()?;
    let fps = times.and_then(|(t, _)| (t.len() > 1 && t[1] > 0.0).then(|| 1.0 / t[1])).unwrap_or(24.0);
    Ok(Some(Morph { fps, frames }))
}

/// A skeletal animation read back (the "skeleton" animation's rotation tracks by joint, the translation track of the
/// root): for tests and tools. Joints are in the skin's own order.
pub fn read_motion(bytes: &[u8]) -> Result<Option<SkelMotion>, String> {
    let (doc, bin) = chunks(bytes)?;
    let inp = In { doc: &doc, bin };
    let Some(skin) = doc["skins"][0]["joints"].as_array() else { return Ok(None) };
    let nodes: Vec<usize> = skin.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
    let Some(anim) = doc["animations"].as_array().and_then(|a| a.iter().find(|a| a["name"] == "skeleton")) else { return Ok(None) };
    let mut mo = SkelMotion::default();
    let mut rot: Vec<Option<Vec<[f32; 4]>>> = vec![None; nodes.len()];
    for c in anim["channels"].as_array().map_or(&[][..], |c| c.as_slice()) {
        let node = c["target"]["node"].as_u64().ok_or("a channel without a node")? as usize;
        let sampler = &anim["samplers"][c["sampler"].as_u64().ok_or("a channel without a sampler")? as usize];
        let (times, _) = inp.floats(sampler["input"].as_u64().ok_or("no input")? as usize)?;
        let (out, k) = inp.floats(sampler["output"].as_u64().ok_or("no output")? as usize)?;
        if times.len() > 1 && times[1] > 0.0 {
            mo.fps = 1.0 / times[1];
        }
        let Some(j) = nodes.iter().position(|&n| n == node) else { continue };
        match c["target"]["path"].as_str() {
            Some("rotation") => rot[j] = Some(out.chunks_exact(k).map(|q| [q[0], q[1], q[2], q[3]]).collect()),
            Some("translation") => {
                mo.root = out.chunks_exact(k).map(|t| [t[0], t[1], t[2]]).collect();
                mo.root_joint = j;
            }
            _ => {}
        }
    }
    let n = rot.iter().flatten().map(Vec::len).next().unwrap_or(0);
    mo.rotations = (0..n).map(|f| rot.iter().map(|r| r.as_ref().map_or([0.0, 0.0, 0.0, 1.0], |r| r[f])).collect()).collect();
    Ok(Some(mo))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri() -> Mesh {
        Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            ..Default::default()
        }
    }

    #[test]
    fn a_textured_mesh_round_trips() {
        let mut m = tri();
        let mut t = Texture::new(2, 2);
        t.rgba.iter_mut().enumerate().for_each(|(i, x)| *x = (i * 13) as u8);
        m.material = Some(Material { base_color: Some(t.clone()), metallic: 0.25, roughness: 0.5, ..Default::default() });
        let b = write(&m).unwrap();
        assert_eq!(&b[..4], b"glTF");
        assert_eq!(b.len() % 4, 0);
        let back = read(&b).unwrap();
        assert_eq!(back.positions, m.positions);
        assert_eq!(back.uvs, m.uvs);
        assert_eq!(back.indices, m.indices);
        let mat = back.material.unwrap();
        assert_eq!(mat.base_color, Some(t));
        assert_eq!((mat.metallic, mat.roughness), (0.25, 0.5));
    }

    #[test]
    fn a_skin_and_a_morph_round_trip() {
        let mut m = tri();
        m.skin = Some(Skin {
            joints: vec![Joint { name: "root".into(), parent: None, head: [0.0, 0.0, 0.0] },
                         Joint { name: "tip".into(), parent: Some(0), head: [0.0, 1.0, 0.0] }],
            weights: vec![[(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)], [(0, 0.5), (1, 0.5), (0, 0.0), (0, 0.0)], [(1, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]],
            motion: None,
        });
        let lifted: Vec<[f32; 3]> = m.positions.iter().map(|p| [p[0], p[1], p[2] + 1.0]).collect();
        m.animation = Some(Morph { fps: 12.0, frames: vec![m.positions.clone(), lifted.clone()] });
        let b = write(&m).unwrap();
        let back = read(&b).unwrap();
        let s = back.skin.unwrap();
        assert_eq!(s.joints.len(), 2);
        assert_eq!(s.joints[1].parent, Some(0));
        assert!((s.joints[1].head[1] - 1.0).abs() < 1e-6);
        assert_eq!(s.weights[1][1], (1, 0.5));
        let morph = read_morph(&b).unwrap().unwrap();
        assert_eq!(morph.frames[1], lifted);
        assert!((morph.fps - 12.0).abs() < 1e-4);
    }

    #[test]
    fn a_skeletal_motion_round_trips() {
        let mut m = tri();
        let q = |a: f32| [0.0, (a / 2.0).sin(), 0.0, (a / 2.0).cos()];
        m.skin = Some(Skin {
            joints: vec![Joint { name: "root".into(), parent: None, head: [0.0, 0.0, 0.0] },
                         Joint { name: "tip".into(), parent: Some(0), head: [0.0, 1.0, 0.0] }],
            weights: vec![[(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)], [(0, 0.5), (1, 0.5), (0, 0.0), (0, 0.0)], [(1, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]],
            motion: Some(SkelMotion { fps: 30.0, rotations: vec![vec![q(0.0), q(0.5)], vec![q(0.3), q(1.0)], vec![q(0.6), q(1.5)]],
                                      root: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]], root_joint: 0 }),
        });
        let b = write(&m).unwrap();
        let back = read_motion(&b).unwrap().unwrap();
        assert_eq!(back.rotations.len(), 3);
        assert_eq!(back.rotations[2][1], q(1.5));
        assert_eq!(back.root[1], [0.0, 0.0, 0.5]);
        assert!((back.fps - 30.0).abs() < 1e-3);
        assert!(read_morph(&b).unwrap().is_none());
    }

    #[test]
    fn node_transforms_apply() {
        let m = tri();
        let b = write(&m).unwrap();
        let (mut doc, bin) = chunks(&b).unwrap();
        doc["nodes"][0]["translation"] = json!([0.0, 0.0, 5.0]);
        let mut js = serde_json::to_vec(&doc).unwrap();
        while !js.len().is_multiple_of(4) {
            js.push(b' ');
        }
        let mut out = Vec::new();
        for w in [GLB, 2, (12 + 8 + js.len() + 8 + bin.len()) as u32, js.len() as u32, JSON] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&js);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&BIN.to_le_bytes());
        out.extend_from_slice(bin);
        assert!(read(&out).unwrap().positions.iter().all(|p| p[2] == 5.0));
    }

    #[test]
    fn a_skinned_mesh_ignores_its_own_node_transform() {
        let mut m = tri();
        m.skin = Some(Skin {
            joints: vec![Joint { name: "root".into(), parent: None, head: [0.0, 0.0, 0.0] }],
            weights: vec![[(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]; 3],
            motion: None,
        });
        let b = write(&m).unwrap();
        let (mut doc, bin) = chunks(&b).unwrap();
        let ni = doc["nodes"].as_array().unwrap().iter().position(|n| n["mesh"].is_u64() && n["skin"].is_u64()).expect("a skinned mesh node");
        doc["nodes"][ni]["translation"] = json!([0.0, 0.0, 5.0]);
        let mut js = serde_json::to_vec(&doc).unwrap();
        while !js.len().is_multiple_of(4) {
            js.push(b' ');
        }
        let mut out = Vec::new();
        for w in [GLB, 2, (12 + 8 + js.len() + 8 + bin.len()) as u32, js.len() as u32, JSON] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&js);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&BIN.to_le_bytes());
        out.extend_from_slice(bin);
        assert_eq!(read(&out).unwrap().positions, m.positions);
    }

    #[test]
    fn junk_is_refused() {
        assert!(read(b"not a glb at all, really").is_err());
    }
}
