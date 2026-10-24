//! The contract between the 3D server / command line and the 3D engines. An engine is one model architecture's whole
//! pipeline with its own kernels (`kernels/model3d/<arch>`) and memory plan, and does one or both of:
//!
//! - **generate**: a mesh from one or more pictures of an object (TRELLIS.2, Hunyuan3D, Pixal3D) - its shape, and
//!   with `texture` its PBR textures;
//! - **rig**: a mesh given a skeleton and skin weights (UniRig, RigNet), or set in motion by a description
//!   (AnimateAnyMesh: a morph animation).
//!
//! Meshes are `nextsycl_mesh::Mesh` (GLB in and out); the CPU tools (decimate, remesh, re-UV, bake) are there too.
//! The program drives an engine only through `Model3dEngine`; it is chosen by the architecture its registry entry
//! names (`Model3dKind`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

pub use nextsycl_core::options::Given;
pub use nextsycl_core::{At, EngineOption, Error, Gpu, Result};
pub use nextsycl_mesh::{self as mesh, Mesh, Texture};

/// One picture of the object: RGBA (a transparent background is taken as the object's outline; an opaque one is
/// cut out by the engine's background remover unless the request says not to), and where it was seen from
#[derive(Clone, Debug, Default)]
pub struct View {
    pub image: Texture,
    /// the side it shows: "front" (the default for one picture), "back", "left", "right", "top", "bottom"; or an
    /// azimuth in degrees ("90")
    pub side: Option<String>,
}

/// One mesh to make. `None` fields take the engine's defaults.
#[derive(Clone, Debug, Default)]
pub struct MeshRequest {
    /// one picture, or several of the same object from different sides (an engine takes up to `Abilities::views`)
    pub views: Vec<View>,
    /// a mesh to paint (an engine that paints: `Abilities::texture`) instead of a shape made from the pictures: the
    /// pictures are what it is painted like
    pub mesh: Option<Mesh>,
    /// textures too (base colour, metallic-roughness), when the engine paints them; false: the shape alone
    pub texture: bool,
    pub seed: u64,
    /// denoising steps of each stage
    pub steps: Option<u32>,
    /// guidance scale
    pub cfg: Option<f32>,
    /// the shape's grid or octree resolution (detail against time and memory)
    pub resolution: Option<u32>,
    /// at most this many triangles in the result (decimated to it)
    pub faces: Option<u32>,
    /// the texture's side in texels
    pub texture_size: Option<u32>,
    /// cut the background out of opaque pictures (default true)
    pub remove_background: Option<bool>,
    /// the engine's own options for this request, by name (`--opt-NAME`, an API request's `options`)
    pub extra: Given,
}

/// What to do to a mesh
#[derive(Clone, Debug, Default)]
pub struct RigRequest {
    pub mesh: Mesh,
    pub seed: u64,
    /// an animating engine: the motion, in words ("a dragon flapping its wings")
    pub prompt: Option<String>,
    /// an animating engine: frames to make, and their rate
    pub frames: Option<u32>,
    pub fps: Option<f32>,
    pub steps: Option<u32>,
    pub cfg: Option<f32>,
    pub extra: Given,
}

/// What an engine does
#[derive(Clone, Copy, Debug, Default)]
pub struct Abilities {
    /// meshes from pictures (`generate`)
    pub generate: bool,
    /// textures on the meshes it makes
    pub texture: bool,
    /// the most pictures `generate` takes at once (1: a single view)
    pub views: u32,
    /// a skeleton and skin weights (`rig`)
    pub rig: bool,
    /// motion from a description (`rig` with a prompt: a morph animation)
    pub animate: bool,
}

/// The engine's own defaults, shown by `engines` and used for the request's `None` fields
#[derive(Clone, Copy, Debug, Default)]
pub struct Defaults {
    pub steps: u32,
    pub cfg: f32,
    pub resolution: u32,
    pub faces: u32,
    pub texture_size: u32,
    /// an animating engine: frames and their rate
    pub frames: u32,
    pub fps: f32,
}

/// Progress, reported as the engine goes: `phase` is the engine's ("image", "structure", "shape", "texture",
/// "decode", "mesh", "bake", "skeleton", "skin", "motion" ...), `at` of `of` its units (steps, chunks)
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub phase: &'static str,
    pub at: u32,
    pub of: u32,
    /// seconds since the request started
    pub seconds: f64,
}

/// A model's files by role (the engine says which it needs)
pub type ModelFiles = BTreeMap<String, PathBuf>;

/// How an engine is loaded
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// settings by name (a registry entry's `env`, the --opt-NAMEs taken at load): what the process environment would
    /// hold when the engine is served; an engine reads a setting here first, then from the environment
    pub settings: BTreeMap<String, String>,
}

impl LoadOptions {
    /// A setting: from `settings`, else the environment
    pub fn setting(&self, name: &str) -> Option<String> {
        self.settings.get(name).cloned().or_else(|| std::env::var(name).ok())
    }
}

/// The runtime one 3D architecture brings
pub trait Model3dEngine: Send + Sync {
    /// its architecture (the registry entry's "arch")
    fn arch(&self) -> &'static str;
    fn abilities(&self) -> Abilities;
    fn defaults(&self) -> Defaults;
    /// the options it takes beyond the request's fields (its kind's `options`)
    fn options(&self) -> &'static [EngineOption] {
        &[]
    }
    /// seconds the load took
    fn load_seconds(&self) -> f64;
    /// a mesh from the request's pictures, reporting as it goes; an error from `progress` stops it (a cancel)
    fn generate(&self, _req: &MeshRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        Err(Error(format!("{} does not make meshes from pictures (it rigs them: model3d.rig)", self.arch())))
    }
    /// the request's mesh rigged (a skeleton and skin weights) or animated (a morph animation)
    fn rig(&self, _req: &RigRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        Err(Error(format!("{} does not rig meshes (it makes them: model3d.generate)", self.arch())))
    }
    /// lines for `status`: its GPUs, memory, what is loaded where
    fn report(&self) -> Vec<String> {
        Vec::new()
    }
}

/// How an engine loads: its files, its GPUs, the options, a log
pub type LoadFn = fn(&ModelFiles, &[Arc<Gpu>], &LoadOptions, &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>>;

/// An engine's registry entry
pub struct Model3dKind {
    /// the architectures it serves (a registry entry's "arch")
    pub archs: &'static [&'static str],
    /// what it is
    pub name: &'static str,
    /// the file roles it needs (a role ending in `*`: one or more, `<role>-1`, `<role>-2` ... - a sharded file; one
    /// ending in `?`: optional)
    pub roles: &'static [&'static str],
    pub load: LoadFn,
    /// the options it takes (`--opt-NAME`: nextsycl_core::options) - at load as the variables they name (in
    /// `LoadOptions::settings` too), per request in `MeshRequest::extra` / `RigRequest::extra`
    pub options: &'static [EngineOption],
}

/// The entry of `kinds` serving `arch`
pub fn kind_for<'k>(kinds: &'k [Model3dKind], arch: &str) -> std::result::Result<&'k Model3dKind, String> {
    kinds.iter().find(|k| k.archs.contains(&arch)).ok_or_else(|| {
        let known: Vec<&str> = kinds.iter().flat_map(|k| k.archs.iter().copied()).collect();
        format!("3D architecture {arch:?} has no engine here (known: {})", known.join(", "))
    })
}

/// The files of a sharded role, in order: `role` itself, or `role-1`, `role-2` ...
pub fn shards(files: &ModelFiles, role: &str) -> Vec<PathBuf> {
    if let Some(p) = files.get(role) {
        return vec![p.clone()];
    }
    (1..).map_while(|i| files.get(&format!("{role}-{i}")).cloned()).collect()
}

/// The roles `files` lacks (optional ones, ending in `?`, never missing; a sharded one, ending in `*`, needs `-1`)
pub fn missing(files: &ModelFiles, roles: &[&str]) -> Vec<String> {
    roles.iter().filter(|r| !r.ends_with('?')).filter_map(|r| {
        let base = r.trim_end_matches('*');
        let has = if r.ends_with('*') { !shards(files, base).is_empty() } else { files.contains_key(base) };
        (!has).then(|| base.to_string())
    }).collect()
}

/// A picture people send: PNG, JPEG or WebP bytes (or a GLB / OBJ for a mesh: `read_mesh`)
pub fn read_picture(bytes: &[u8]) -> std::result::Result<Texture, String> {
    Texture::decode(bytes)
}

/// A mesh people send: GLB (glTF binary) or OBJ text, by its first bytes
pub fn read_mesh(bytes: &[u8]) -> std::result::Result<Mesh, String> {
    if bytes.starts_with(b"glTF") {
        nextsycl_mesh::glb::read(bytes)
    } else {
        let text = std::str::from_utf8(bytes).map_err(|_| "not a GLB nor an OBJ (text) file".to_string())?;
        nextsycl_mesh::obj::read(text)
    }
}

/// The finishing steps every generator's mesh takes (CPU): decimated to `faces`, normals computed
pub fn finish(mut m: Mesh, faces: Option<u32>) -> Mesh {
    if let Some(f) = faces.filter(|f| (*f as usize) < m.triangles()) {
        let mat = m.material.take();
        m = nextsycl_mesh::tools::decimate(&m, f as usize);
        m.material = mat;
    }
    if m.normals.len() != m.positions.len() {
        m.compute_normals();
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_roles_are_named() {
        let mut f = ModelFiles::new();
        f.insert("dit".into(), "a".into());
        f.insert("vae-1".into(), "b".into());
        assert_eq!(missing(&f, &["dit", "vae*", "image_encoder", "remover?"]), vec!["image_encoder".to_string()]);
    }

    #[test]
    fn meshes_are_read_by_their_bytes() {
        let m = read_mesh(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        assert_eq!(m.triangles(), 1);
        let glb = nextsycl_mesh::glb::write(&m).unwrap();
        assert_eq!(read_mesh(&glb).unwrap().positions, m.positions);
        assert!(read_mesh(&[0xff, 0xfe, 0x00]).is_err());
    }
}
