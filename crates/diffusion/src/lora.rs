//! LoRA files: each target matrix's low-rank update dW = scale * B . A, read from the forms the tools write -
//!
//! ```text
//!   PEFT / diffusers   <module>.lora_A[.default].weight [r, k], <module>.lora_B[.default].weight [n, r]
//!   diffusers (old)    <module>.lora.down.weight, <module>.lora.up.weight
//!   kohya / ComfyUI    <module>.lora_down.weight, <module>.lora_up.weight, <module>.alpha (scale alpha / r)
//! ```
//!
//! with the prefixes they put before a module's name (`diffusion_model.`, `transformer.`, `base_model.model.`,
//! `lora_unet_`) taken off. Modules are matched by `key`: the name with '.' and '_' alike, as kohya writes them.

use std::collections::BTreeMap;
use std::path::Path;

use nextsycl_gguf::safetensors::SafeTensors;

/// A module's tensors by name: A (down), B (up), alpha
type Pair = (Option<String>, Option<String>, Option<String>);

/// One matrix's update
#[derive(Clone, Debug)]
pub struct Delta {
    /// the module, as the file names it (prefixes off)
    pub target: String,
    /// rank
    pub r: usize,
    /// rows (outputs) and columns (inputs) of the matrix
    pub n: usize,
    pub k: usize,
    /// A [r, k]
    pub a: Vec<f32>,
    /// B [n, r], times the scale (alpha / r if the file gives alpha, times the use's strength)
    pub b: Vec<f32>,
}

/// A module name as a match key: '.' and '_' alike, no prefix
pub fn key(name: &str) -> String {
    let mut n = name;
    for p in ["base_model.model.", "diffusion_model.", "transformer.", "model.", "lora_unet_", "lora_transformer_"] {
        n = n.strip_prefix(p).unwrap_or(n);
    }
    n.replace('.', "_")
}

/// The updates in `path`, B scaled by `strength` (and by alpha / r where the file gives alpha)
pub fn read(path: &Path, strength: f32) -> Result<Vec<Delta>, String> {
    let st = SafeTensors::open(path).map_err(|e| e.0)?;
    // module -> its A, B and alpha tensors' names
    let mut parts: BTreeMap<String, Pair> = BTreeMap::new();
    for t in &st.tensors {
        let name = t.name.as_str();
        let forms: [(&str, usize); 8] = [
            (".lora_A.default.weight", 0), (".lora_B.default.weight", 1), (".lora_A.weight", 0), (".lora_B.weight", 1),
            (".lora_down.weight", 0), (".lora_up.weight", 1), (".lora.down.weight", 0), (".lora.up.weight", 1),
        ];
        if let Some((module, which)) = forms.iter().find_map(|(suf, w)| name.strip_suffix(suf).map(|m| (m, *w))) {
            let e = parts.entry(module.to_string()).or_default();
            if which == 0 { e.0 = Some(name.to_string()) } else { e.1 = Some(name.to_string()) }
        } else if let Some(module) = name.strip_suffix(".alpha") {
            parts.entry(module.to_string()).or_default().2 = Some(name.to_string());
        } else {
            return Err(format!("{}: {name} is not a LoRA tensor this reader knows (A / B, down / up, alpha)", path.display()));
        }
    }
    let mut out = Vec::new();
    for (module, (a, b, alpha)) in parts {
        let (Some(a), Some(b)) = (a, b) else {
            return Err(format!("{}: {module} has only half of its pair", path.display()));
        };
        let at = st.need(&a).map_err(|e| e.0)?;
        let bt = st.need(&b).map_err(|e| e.0)?;
        if at.shape.len() != 2 || bt.shape.len() != 2 || at.shape[0] != bt.shape[1] {
            return Err(format!("{}: {module}: A {:?} and B {:?} do not pair", path.display(), at.shape, bt.shape));
        }
        let (r, k, n) = (at.shape[0] as usize, at.shape[1] as usize, bt.shape[0] as usize);
        let scale = match &alpha {
            Some(al) => st.f32(al).map_err(|e| e.0)?.first().copied().unwrap_or(r as f32) / r as f32,
            None => 1.0,
        } * strength;
        let mut bv = st.f32(&b).map_err(|e| e.0)?;
        bv.iter_mut().for_each(|x| *x *= scale);
        let target = {
            let mut m = module.as_str();
            for p in ["base_model.model.", "diffusion_model.", "transformer.", "lora_unet_", "lora_transformer_"] {
                m = m.strip_prefix(p).unwrap_or(m);
            }
            m.to_string()
        };
        out.push(Delta { target, r, n, k, a: st.f32(&a).map_err(|e| e.0)?, b: bv });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::key;

    #[test]
    fn module_names_match_across_forms() {
        assert_eq!(key("transformer_blocks.3.attn.to_q"), key("lora_unet_transformer_blocks_3_attn_to_q"));
        assert_eq!(key("diffusion_model.transformer_blocks.3.attn.to_out.0"), key("transformer_blocks_3_attn_to_out_0"));
    }
}
