//! The platform's pieces agree with each other, checked on every `cargo test` with no GPU and no model:
//!
//! 1. Every model card names an architecture some engine of its kind loads (`every_card_has_an_engine`) - a card that
//!    `nextsycl models pull` fetches and nothing can run is a bug at the card.
//! 2. Every engine of a kind is reachable from at least one card, or is the kind's example
//!    (`every_engine_has_a_card`) - an engine nobody can pull is dead weight.
//! 3. A card's `does` (model3d) names abilities the front door routes (`cards_say_only_what_the_door_routes`).
//! 4. No two engines of a kind claim one architecture (`no_architecture_is_claimed_twice`).
//! 5. A picture-making card's request rules (`controls`) promise no more than its engine takes: sizes within the
//!    engine's own limits and on its grid, no edits from a text-to-image engine, a fixed step count where the engine
//!    runs only its own (`picture_cards_promise_no_more_than_their_engines_take`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn cards() -> Vec<Value> {
    nextsycl_models::catalog::Catalog::from_dir(&root().join("model-cards")).expect("the model cards read").models
}

/// `nextsycl serve <id>` hands the engine the registry's `ctx` as text (`as_str`), so a card that writes a bare number
/// (`ctx: 32768`) starts the engine with an empty `--ctx` and it exits at once: write it as text (`'32768'`, `131072,32768`)
#[test]
fn every_llm_card_writes_ctx_as_text() {
    let mut seen = 0;
    for c in cards().into_iter().filter(|c| c["kind"] == "llm") {
        if let Some(v) = c.get("ctx") {
            seen += 1;
            assert!(v.is_string(), "{}: ctx must be text ('32768' or '131072,32768'), not {v}", c["catalog"]);
        }
    }
    assert!(seen > 10, "the llm cards with a ctx were not found ({seen})");
}

/// Each engine as (kind, the architectures it loads) - the program's own lists: `engines()` in image.rs, audio.rs, model3d.rs, main.rs
fn engines() -> Vec<(&'static str, Vec<&'static str>)> {
    let mut v: Vec<(&'static str, Vec<&'static str>)> = Vec::new();
    for k in [nextsycl_llm_glm5next::kind(), nextsycl_llm_qwen4exp::kind(), nextsycl_llm_qwen35::kind(), nextsycl_llm_deepseek2::kind(), nextsycl_llm_deepseek4::kind(),
              nextsycl_llm_deepseek41::kind(), nextsycl_llm_museglimmer::kind(), nextsycl_llm_nemotronh::kind(), nextsycl_llm_minimaxm3::kind(), nextsycl_llm_ricursive::kind(), nextsycl_llm_example::kind()] {
        v.push(("llm", k.archs.to_vec()));
    }
    for k in [nextsycl_image_qwenimage21::kind(), nextsycl_image_unlimitedocr::kind(), nextsycl_image_paddleocrvl::kind(), nextsycl_image_olmocr2::kind(),
              nextsycl_image_hunyuanimage21::kind(), nextsycl_image_upscale::kind(), nextsycl_image_starvector::kind(),
              nextsycl_image_mingimage::kind(), nextsycl_image_mossimage::kind(), nextsycl_image_seedvr2::kind(), nextsycl_image_flashvsr::kind(), nextsycl_image_rembg::kind(), nextsycl_image_ph3image::kind(), nextsycl_image_h3image::kind(), nextsycl_image_zimage::kind(), nextsycl_image_vldescribe::kind(), nextsycl_image_example::kind()] {
        v.push(("image", k.archs.to_vec()));
    }
    for k in [nextsycl_audio_minimaxmusic3::kind(), nextsycl_audio_qwen3tts::kind(), nextsycl_audio_voxcpm2::kind(), nextsycl_audio_breezetts2::kind(),
              nextsycl_audio_vibevoice::kind(), nextsycl_audio_s2pro::kind(), nextsycl_audio_voxtraltts::kind(), nextsycl_audio_whisper::kind(),
              nextsycl_audio_acestep::kind(), nextsycl_audio_yue2::kind(), nextsycl_audio_sophon::kind(), nextsycl_audio_example::kind()] {
        v.push(("audio", k.archs.to_vec()));
    }
    for k in [nextsycl_model3d_trellis2::kind(), nextsycl_model3d_trellis2::kind_pixal(), nextsycl_model3d_hunyuan3d::kind(), nextsycl_model3d_unirig::kind(), nextsycl_model3d_skintokens::kind(),
              nextsycl_model3d_rignet::kind(), nextsycl_model3d_animateanymesh::kind(), nextsycl_model3d_unimate::kind(), nextsycl_model3d_budget3d::kind(),
              nextsycl_model3d_example::kind()] {
        v.push(("model3d", k.archs.to_vec()));
    }
    v
}

/// kind -> every architecture its engines load
fn archs() -> std::collections::BTreeMap<&'static str, Vec<&'static str>> {
    let mut m: std::collections::BTreeMap<&'static str, Vec<&'static str>> = std::collections::BTreeMap::new();
    for (kind, a) in engines() {
        m.entry(kind).or_default().extend(a);
    }
    m
}

/// Cards of a kind served by something other than a contract engine (the video daemon, the LoRAs): not checked here
const OTHER_KINDS: &[&str] = &["video", "lora"];

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or("")
}

#[test]
fn every_card_has_an_engine() {
    let archs = archs();
    let mut missing = Vec::new();
    for c in cards() {
        let kind = s(&c, "kind");
        if OTHER_KINDS.contains(&kind) {
            continue;
        }
        let Some(known) = archs.get(kind) else {
            missing.push(format!("{}: kind {kind} has no engine list in this test", s(&c, "id")));
            continue;
        };
        if !known.contains(&s(&c, "arch")) {
            missing.push(format!("{}: no {kind} engine loads architecture {:?} (they load {known:?})", s(&c, "id"), s(&c, "arch")));
        }
    }
    assert!(missing.is_empty(), "cards nothing can run:\n  {}", missing.join("\n  "));
}

#[test]
fn every_engine_has_a_card() {
    let carded: BTreeSet<(String, String)> = cards().iter().map(|c| (s(c, "kind").to_string(), s(c, "arch").to_string())).collect();
    let mut orphans = Vec::new();
    for (kind, list) in engines() {
        // the example engines are the template for a new one, not a model
        if list.iter().any(|a| a.starts_with("example") || *a == "echo") {
            continue;
        }
        // an engine is reachable when a card names any architecture it loads (it may load several names)
        if !list.iter().any(|a| carded.contains(&(kind.to_string(), a.to_string()))) {
            orphans.push(format!("{kind}/{}", list.join("|")));
        }
    }
    assert!(orphans.is_empty(), "engines no card can pull (add a card, or drop the engine): {orphans:?}");
}

#[test]
fn no_architecture_is_claimed_twice() {
    for (kind, list) in archs() {
        let mut seen = BTreeSet::new();
        for a in &list {
            assert!(seen.insert(*a), "{kind}: two engines load architecture {a:?}");
        }
    }
}

#[test]
fn cards_say_only_what_the_door_routes() {
    // model3d: what the front door routes by (api.rs: generate, rig; paint for a given mesh) and advertises (life.rs)
    const DOES: &[&str] = &["generate", "texture", "paint", "views", "rig", "animate", "splat"];
    for c in cards().iter().filter(|c| s(c, "kind") == "model3d") {
        let does: Vec<&str> = c["does"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        assert!(!does.is_empty(), "{}: a 3D card says what it does", s(c, "id"));
        for d in &does {
            assert!(DOES.contains(d), "{}: does {d:?} - the door routes {DOES:?}", s(c, "id"));
        }
        // splats are what a generating model makes instead of a mesh
        if does.contains(&"splat") {
            assert!(does.contains(&"generate") && !does.contains(&"texture"), "{}: splats come from generate, untextured", s(c, "id"));
        }
        // painting a mesh needs the paint model: a card that paints also textures
        if does.contains(&"paint") {
            assert!(does.contains(&"texture"), "{}: paints a mesh but does not texture", s(c, "id"));
        }
    }
}

/// A picture-making card's `controls` (model-cards/README.md, Image controls) within what its engine takes: the
/// platform refuses requests outside them, so a rule wider than the engine would pass requests the engine refuses
#[test]
fn picture_cards_promise_no_more_than_their_engines_take() {
    use nextsycl_image_h3image as h3i;
    use nextsycl_image_ph3image as ph3;
    use nextsycl_image_zimage as zi;
    let mut seen = 0;
    for c in cards().iter().filter(|c| nextsycl_models::controls::makes_pictures(c)) {
        let (id, k) = (s(c, "id").to_string(), &c["controls"]);
        let sz = &k["size"];
        let num = |v: &Value| v.as_u64().unwrap_or(0);
        let (lo, hi, mult, px) = (num(&sz["min_side"]), num(&sz["max_side"]), num(&sz["multiple"]), sz["max_pixels"].as_u64());
        // every engine refuses a side under 64, and takes multiples of 16 (Qwen-Image, Z-Image, MOSS) or 32 (the rest)
        assert!(lo >= 64 && mult > 0 && mult.is_multiple_of(16), "{id}: sides");
        match s(c, "arch") {
            "z-image" => {
                assert!(lo as usize >= zi::MIN_SIDE && hi as usize <= zi::MAX_SIDE && px.is_some_and(|p| p as usize <= zi::MAX_PIXELS), "{id}: z-image's limits");
                assert!(k["edit"].is_null() && k["cfg"].is_null() && k["loras"] == false && k["transparent"] == false, "{id}: z-image: text to image, no guidance, LoRAs or alpha");
            }
            "minimax-h3-image" => {
                assert!(lo >= h3i::MIN_SIDE as u64 && hi <= h3i::MAX_SIDE as u64 && px.is_some_and(|p| p <= h3i::MAX_PIXELS) && mult.is_multiple_of(h3i::GRID as u64), "{id}: the engine's limits");
                assert!(k["cfg"].is_null() && k["negative"] == false && num(&k["edit"]["images"]) as usize <= h3i::MAX_PICTURES, "{id}: no guidance, its pictures");
            }
            "p-h3-image" => {
                assert!(lo >= 256 && px.is_some_and(|p| p <= ph3::MAX_PIXELS) && mult.is_multiple_of(ph3::GRID as u64), "{id}: the engine's limits");
                assert!(k["cfg"].is_null() && k["negative"] == false && num(&k["edit"]["images"]) as usize <= ph3::MAX_PICTURES, "{id}: no guidance, its pictures");
                // it runs its grid's 8 steps (another grid: options.sigmas) and reads an edit's pictures at strength 1
                assert!(k["steps"]["min"] == 8 && k["steps"]["max"] == 8 && k["edit"]["strength"] == false, "{id}: 8 fixed steps, no img2img");
            }
            "hunyuanimage-2.1" | "ming-image" | "moss-image" => {
                assert!(k["edit"].is_null() && hi <= 4096, "{id}: text to image only, 4096 a side at most");
            }
            "qwen-image-2.1" => {
                // a few-step distill runs its own sigmas: a fixed count
                if let Some(sig) = c["env"]["NS_QI_SIGMAS"].as_str() {
                    let n = sig.split(',').count() as u64;
                    assert!(num(&k["steps"]["min"]) == n && num(&k["steps"]["max"]) == n, "{id}: its own {n} sigmas");
                }
            }
            a => panic!("{id}: a picture-making architecture {a} this test does not know: add its engine's limits here"),
        }
        // the front door takes 1 to 8 pictures a request, and 8 pictures an edit
        assert!(num(&k["n"]) <= 8 && k["edit"]["images"].as_u64().is_none_or(|n| n <= 8), "{id}: the front door's limits");
        seen += 1;
    }
    assert!(seen >= 20, "the picture-making cards were not found ({seen})");
}
