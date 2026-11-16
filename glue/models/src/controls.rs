//! A picture-making image model's request rules (`controls`, schema 1): what one shared image control on the platform
//! may send it - the steps, guidance, sizes, edits ... it makes WELL, not all its engine accepts. Written in its model
//! card (model-cards/README.md, "Image controls"); the front door copies it into the model's row of system.models and
//! system.state (its registry entry's, else its card's). The same block as nextcuda's (docs/models.md, "Image controls
//! (schema 1)"): a reader treats a higher `version` as rules it does not know.

use serde_json::Value;

/// The schema this program writes and checks
pub const VERSION: u64 = 1;

/// Whether a card is a picture-making image model (text to image, edits): an image card with a step count in its
/// defaults. Describers, OCR, upscalers, background removers and vectorizers take no steps and have no controls
pub fn makes_pictures(card: &Value) -> bool {
    card["kind"] == "image" && card["defaults"]["steps"].is_u64()
}

fn int(v: &Value, what: &str) -> Result<u64, String> {
    v.as_u64().ok_or_else(|| format!("controls.{what}: a whole number"))
}

fn flag(c: &Value, k: &str) -> Result<bool, String> {
    c[k].as_bool().ok_or_else(|| format!("controls.{k}: true or false"))
}

/// A {min, max, default} range: min <= default <= max, whole numbers when `whole`
fn range(v: &Value, what: &str, whole: bool) -> Result<(f64, f64, f64), String> {
    let o = v.as_object().ok_or_else(|| format!("controls.{what}: {{min, max, default}}"))?;
    let get = |k: &str| -> Result<f64, String> {
        let x = o.get(k).ok_or_else(|| format!("controls.{what}.{k}: missing"))?;
        if whole {
            x.as_u64().map(|n| n as f64).ok_or_else(|| format!("controls.{what}.{k}: a whole number"))
        } else {
            x.as_f64().ok_or_else(|| format!("controls.{what}.{k}: a number"))
        }
    };
    let (min, max, default) = (get("min")?, get("max")?, get("default")?);
    if !(min <= default && default <= max) {
        return Err(format!("controls.{what}: min {min} <= default {default} <= max {max}"));
    }
    if let Some(k) = o.keys().find(|k| !["min", "max", "default"].contains(&k.as_str())) {
        return Err(format!("controls.{what}.{k}: not a field (min, max, default)"));
    }
    Ok((min, max, default))
}

/// "WxH" as (w, h)
pub fn size_of(s: &str) -> Option<(u64, u64)> {
    let (w, h) = s.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// The block checked: its shape, every range ordered, the default size within its own rules, version 1. The fields
/// nextcuda's schema 1 has and no others
pub fn check(c: &Value) -> Result<(), String> {
    let o = c.as_object().ok_or("controls: an object")?;
    const FIELDS: &[&str] = &["version", "steps", "cfg", "negative", "size", "seed", "n", "loras", "edit", "transparent", "sampler", "shift", "options", "note"];
    if let Some(k) = o.keys().find(|k| !FIELDS.contains(&k.as_str())) {
        return Err(format!("controls.{k}: not a field of schema {VERSION}"));
    }
    if c["version"].as_u64() != Some(VERSION) {
        return Err(format!("controls.version: {VERSION}"));
    }
    let (smin, _, _) = range(&c["steps"], "steps", true)?;
    if smin < 1.0 {
        return Err("controls.steps.min: 1 at least".into());
    }
    if !c["cfg"].is_null() {
        let (min, _, _) = range(&c["cfg"], "cfg", false)?;
        if min < 0.0 {
            return Err("controls.cfg.min: 0 at least".into());
        }
    }
    flag(c, "negative")?;
    // a negative prompt steers only with guidance
    if c["negative"] == true && c["cfg"].is_null() {
        return Err("controls.negative: true needs cfg (a negative prompt is read only with guidance)".into());
    }
    let sz = &c["size"];
    let sz_o = sz.as_object().ok_or("controls.size: an object")?;
    const SIZE: &[&str] = &["min_side", "max_side", "multiple", "max_pixels", "default", "sizes", "presets"];
    if let Some(k) = sz_o.keys().find(|k| !SIZE.contains(&k.as_str())) {
        return Err(format!("controls.size.{k}: not a field"));
    }
    let (lo, hi, mult) = (int(&sz["min_side"], "size.min_side")?, int(&sz["max_side"], "size.max_side")?, int(&sz["multiple"], "size.multiple")?);
    if mult == 0 || lo == 0 || lo > hi || !lo.is_multiple_of(mult) || !hi.is_multiple_of(mult) {
        return Err(format!("controls.size: min_side {lo} <= max_side {hi}, both multiples of {mult}"));
    }
    let max_px = match &sz["max_pixels"] {
        Value::Null => None,
        v => Some(int(v, "size.max_pixels")?),
    };
    let fits = |w: u64, h: u64| (lo..=hi).contains(&w) && (lo..=hi).contains(&h) && w.is_multiple_of(mult) && h.is_multiple_of(mult) && max_px.is_none_or(|m| w * h <= m);
    let (dw, dh) = sz["default"].as_str().and_then(size_of).ok_or("controls.size.default: \"WxH\"")?;
    if !fits(dw, dh) {
        return Err(format!("controls.size.default {dw}x{dh}: outside its own rules"));
    }
    if let Some(list) = sz.get("sizes") {
        let list = list.as_array().ok_or("controls.size.sizes: a list of \"WxH\"")?;
        for x in list {
            let (w, h) = x.as_str().and_then(size_of).ok_or("controls.size.sizes: a list of \"WxH\"")?;
            if !fits(w, h) {
                return Err(format!("controls.size.sizes {w}x{h}: outside its own rules"));
            }
        }
        if !list.is_empty() && !list.iter().any(|x| x == &sz["default"]) {
            return Err("controls.size.default: one of sizes".into());
        }
    }
    if let Some(p) = sz.get("presets") {
        let ratios = p["ratios"].as_array().filter(|a| !a.is_empty()).ok_or("controls.size.presets.ratios: a list of \"W:H\"")?;
        if ratios.iter().any(|r| r.as_str().and_then(|r| r.split_once(':')).is_none_or(|(a, b)| a.parse::<u32>().is_err() || b.parse::<u32>().is_err())) {
            return Err("controls.size.presets.ratios: \"W:H\" each".into());
        }
        let pixels = p["pixels"].as_array().filter(|a| !a.is_empty()).ok_or("controls.size.presets.pixels: a list of pixel budgets")?;
        if pixels.iter().any(|x| x.as_u64().is_none_or(|n| n == 0)) {
            return Err("controls.size.presets.pixels: whole numbers".into());
        }
    }
    flag(c, "seed")?;
    if !(1..=64).contains(&int(&c["n"], "n")?) {
        return Err("controls.n: 1 to 64 pictures a request".into());
    }
    flag(c, "loras")?;
    flag(c, "transparent")?;
    if !c["edit"].is_null() {
        let e = c["edit"].as_object().ok_or("controls.edit: an object or null")?;
        const EDIT: &[&str] = &["images", "mask", "strength", "reference", "strength_with_reference"];
        if let Some(k) = e.keys().find(|k| !EDIT.contains(&k.as_str())) {
            return Err(format!("controls.edit.{k}: not a field"));
        }
        if int(&c["edit"]["images"], "edit.images")? < 1 {
            return Err("controls.edit.images: 1 at least".into());
        }
        let ed = &c["edit"];
        for k in ["mask", "strength", "reference", "strength_with_reference"] {
            ed[k].as_bool().ok_or_else(|| format!("controls.edit.{k}: true or false"))?;
        }
        if ed["strength"] == false && ed["reference"] == false {
            return Err("controls.edit: neither strength nor reference (null: text to image only)".into());
        }
        if ed["strength_with_reference"] == true && (ed["strength"] == false || ed["reference"] == false) {
            return Err("controls.edit.strength_with_reference: needs strength and reference".into());
        }
    }
    if let Some(s) = o.get("sampler") {
        let vals = s["values"].as_array().filter(|a| !a.is_empty() && a.iter().all(Value::is_string)).ok_or("controls.sampler.values: a list of names")?;
        if !vals.contains(&s["default"]) {
            return Err("controls.sampler.default: one of values".into());
        }
    }
    if let Some(s) = o.get("shift") {
        let (min, _, _) = range(s, "shift", false)?;
        if min <= 0.0 {
            return Err("controls.shift.min: above 0".into());
        }
    }
    if let Some(opts) = o.get("options") {
        for (i, x) in opts.as_array().ok_or("controls.options: a list")?.iter().enumerate() {
            if x["name"].as_str().is_none_or(str::is_empty) {
                return Err(format!("controls.options[{i}].name: required"));
            }
            match x["type"].as_str() {
                Some("number") => {
                    if x.get("min").is_some() || x.get("max").is_some() {
                        let (lo, hi) = (x["min"].as_f64().ok_or("options min: a number")?, x["max"].as_f64().ok_or("options max: a number")?);
                        if x["default"].as_f64().is_some_and(|d| !(lo..=hi).contains(&d)) || lo > hi {
                            return Err(format!("controls.options[{i}]: min <= default <= max"));
                        }
                    }
                }
                Some("bool") => {
                    if !x["default"].is_null() && !x["default"].is_boolean() {
                        return Err(format!("controls.options[{i}].default: true or false"));
                    }
                }
                Some("text") => {}
                _ => return Err(format!("controls.options[{i}].type: number, bool or text")),
            }
            if x["help"].as_str().is_none_or(str::is_empty) {
                return Err(format!("controls.options[{i}].help: required"));
            }
        }
    }
    if o.get("note").is_some_and(|n| !n.is_string()) {
        return Err("controls.note: text".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good() -> Value {
        json!({"version": 1, "steps": {"min": 6, "max": 6, "default": 6}, "cfg": null, "negative": false,
               "size": {"min_side": 256, "max_side": 4096, "multiple": 16, "max_pixels": 4194304, "default": "1024x1024",
                        "presets": {"ratios": ["1:1", "16:9"], "pixels": [1048576]}},
               "seed": true, "n": 8, "loras": true, "transparent": true,
               "edit": {"images": 8, "mask": false, "strength": true, "reference": true, "strength_with_reference": false},
               "sampler": {"values": ["euler"], "default": "euler"}, "shift": {"min": 0.1, "max": 20, "default": 3},
               "options": [{"name": "eta", "type": "number", "min": 0, "max": 1, "default": 0, "help": "x"}], "note": "a sentence"})
    }

    #[test]
    fn a_whole_block_passes_and_each_fault_is_named() {
        check(&good()).unwrap();
        let bad = |f: &dyn Fn(&mut Value), want: &str| {
            let mut c = good();
            f(&mut c);
            let e = check(&c).unwrap_err();
            assert!(e.contains(want), "{e} (wanted {want})");
        };
        bad(&|c| c["version"] = json!(2), "version");
        bad(&|c| c["steps"]["default"] = json!(9), "steps");
        bad(&|c| c["steps"]["min"] = json!(0), "steps.min");
        bad(&|c| c["cfg"] = json!({"min": 5, "max": 1, "default": 3}), "cfg");
        bad(&|c| c["negative"] = json!(true), "negative");
        bad(&|c| c["size"]["default"] = json!("1000x1000"), "size.default");
        bad(&|c| c["size"]["default"] = json!("4096x4096"), "size.default");
        bad(&|c| c["size"]["min_side"] = json!(100), "size");
        bad(&|c| c["size"]["sizes"] = json!(["512x512"]), "one of sizes");
        bad(&|c| c["n"] = json!(0), "n");
        bad(&|c| {
            c["edit"]["strength_with_reference"] = json!(true);
            c["edit"]["reference"] = json!(false);
        }, "strength_with_reference");
        bad(&|c| c["edit"]["images"] = json!(0), "edit.images");
        bad(&|c| c["sampler"]["default"] = json!("lcm"), "sampler");
        bad(&|c| c["shift"]["min"] = json!(0), "shift");
        bad(&|c| c["options"][0]["type"] = json!("float"), "options");
        bad(&|c| c["colour"] = json!(1), "colour");
        let mut c = good();
        c["edit"] = Value::Null;
        c["cfg"] = json!({"min": 1, "max": 30, "default": 4});
        c["negative"] = json!(true);
        check(&c).unwrap();
    }
}
