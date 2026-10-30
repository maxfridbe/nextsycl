//! The front door end to end, with no GPU and no model: the real `Home` listening on a port of its own over a temporary
//! registry (an in-memory `Store`), called as a client calls it - /rpc/<method> with a bearer token, the OpenAI routes,
//! /openapi.yml. What it checks is the platform's contract: the envelope, the token, maintenance, the methods and
//! features a client can test for, the cards' models that are not downloaded, how 3D work is routed (a given mesh goes
//! to a model that paints), and the specification against the methods. Nothing here starts a model server.
//!
//! `cargo test -p nextsycl-serve --test front_door`; tests/run.sh runs it with the rest.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nextsycl_serve::home::registry::Store;
use nextsycl_serve::home::{Home, Options, Ports};
use serde_json::{json, Value};

const TOKEN: &str = "ns-test-token";

/// A registry in memory, its files in a directory of the test's own
struct Mem {
    dir: PathBuf,
    models: Mutex<Vec<Value>>,
    cards: Vec<Value>,
}

impl Store for Mem {
    fn models(&self) -> Result<Vec<Value>, String> {
        Ok(self.models.lock().unwrap().clone())
    }
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String> {
        let mut all = self.models.lock().unwrap();
        let m = all.iter_mut().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id}"))?;
        f(m);
        Ok(m.clone())
    }
    fn registry_file(&self) -> PathBuf {
        self.dir.join("models.json")
    }
    fn get(&self, name: &str) -> Option<String> {
        (name == "NS_API_TOKEN").then(|| TOKEN.to_string())
    }
    fn in_env(&self, _name: &str) -> bool {
        false
    }
    fn file(&self) -> PathBuf {
        self.dir.join("nextsycl.conf")
    }
    fn guide(&self, _entry: &Value) -> Option<String> {
        None
    }
    fn cards(&self) -> Vec<Value> {
        self.cards.clone()
    }
    fn card_files(&self, id: &str) -> Vec<(String, u64)> {
        self.cards.iter().find(|c| c["id"] == id).map(|c| vec![(format!("{id}/model.gguf"), c["bytes"].as_u64().unwrap_or(1 << 30))]).unwrap_or_default()
    }
}

fn entry(id: &str, kind: &str, does: &[&str], enabled: bool) -> Value {
    json!({"id": id, "kind": kind, "enabled": enabled, "does": does, "file": format!("/nonexistent/{id}.gguf"), "gpus": "0", "title": id})
}

/// A front door on a free port: the registry's entries and the cards' models, and its address
fn door(tag: &str, models: Vec<Value>, cards: Vec<Value>) -> String {
    let dir = std::env::temp_dir().join(format!("nextsycl-front-door-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let ports = Ports { llm: 0, switch: 0, image: 0, audio: 0, model3d: 0, image2: 0, audio2: 0, model3d2: 0, video: 0, chat_ui: 0 };
    let cfg = Box::new(Mem { dir: dir.clone(), models: Mutex::new(models), cards });
    let home = Home::new(Options { wfe: dir.join("wfe"), exe: PathBuf::from("/nonexistent/nextsycl"), ports, serve_host: "127.0.0.1".into(), door: "127.0.0.1:1".into(), cfg, idle_minutes: 0 })
        .expect("the front door starts");
    let addr = format!("127.0.0.1:{port}");
    let a = addr.clone();
    std::thread::spawn(move || {
        let _ = Arc::clone(&home).run(&a);
    });
    for _ in 0..100 {
        if TcpStream::connect(&addr).is_ok() {
            return addr;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("the front door did not listen on {addr}");
}

/// One request: (HTTP status, body)
fn http(addr: &str, method: &str, path: &str, token: Option<&str>, body: Option<&Value>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let b = body.map(Value::to_string).unwrap_or_default();
    let auth = token.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).unwrap();
    let mut raw = Vec::new();
    let _ = s.read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw).into_owned();
    let status = text.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let body = text.split_once("\r\n\r\n").map_or("", |x| x.1).to_string();
    (status, body)
}

/// /rpc/<method> with the token: the whole envelope
fn rpc(addr: &str, method: &str, body: Value) -> Value {
    let (st, text) = http(addr, "POST", &format!("/rpc/{method}"), Some(TOKEN), Some(&body));
    assert_eq!(st, 200, "{method}: /rpc answers HTTP 200 whatever the result; got {st}: {text}");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{method}: not JSON ({e}): {text}"))
}

fn ok(addr: &str, method: &str, body: Value) -> Value {
    let v = rpc(addr, method, body);
    assert_eq!(v["ok"], true, "{method} failed: {v}");
    v["result"].clone()
}

fn err(addr: &str, method: &str, body: Value) -> Value {
    let v = rpc(addr, method, body);
    assert_eq!(v["ok"], false, "{method} should have been refused: {v}");
    v["error"].clone()
}

#[test]
fn an_answer_is_an_envelope_and_a_call_needs_the_token() {
    let a = door("envelope", vec![entry("chat", "llm", &[], true)], vec![]);
    // no token, a wrong one: refused 401 in the envelope
    for t in [None, Some("wrong")] {
        let (st, text) = http(&a, "POST", "/rpc/system.ping", t, Some(&json!({})));
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!((st, v["ok"].clone(), v["error"]["status"].clone()), (200, json!(false), json!(401)), "{text}");
    }
    // an unknown method: 404 in the envelope, naming the method
    let e = err(&a, "system.nonsense", json!({}));
    assert_eq!(e["status"], 404, "{e}");
    // an OpenAI route's error is OpenAI's
    let (st, text) = http(&a, "POST", "/v1/chat/completions", None, Some(&json!({"model": "chat", "messages": []})));
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!((st, v["error"]["type"].clone()), (401, json!("invalid_api_key")), "{text}");
}

#[test]
fn ping_says_the_box_is_here_and_takes_work() {
    let a = door("ping", vec![entry("chat", "llm", &[], true)], vec![]);
    let p = ok(&a, "system.ping", json!({}));
    assert_eq!(p["available"], true);
    assert_eq!(p["maintenance"], false);
    assert!(p["reason"].is_null() && p["retry_after_s"].is_null(), "{p}");
    assert!(p["api"].as_u64().is_some_and(|v| v >= 3), "{p}");
    assert!(p["at"].as_f64().is_some_and(|t| t > 1.0e9), "{p}");
}

#[test]
fn version_lists_the_features_and_only_the_methods_that_work() {
    let a = door("version", vec![entry("chat", "llm", &[], true), entry("pic", "image", &[], true)], vec![]);
    let v = ok(&a, "system.version", json!({}));
    let features: Vec<&str> = v["features"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    for f in ["system.ping", "system.busy", "system.version", "jobs.image", "reentrant_id"] {
        assert!(features.contains(&f), "feature {f} missing from {features:?}");
    }
    let methods: Vec<&str> = v["methods"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    // the kinds enabled only: chat and image are, audio and model3d are not
    assert!(methods.contains(&"llm.chat") && methods.contains(&"image.generate"), "{methods:?}");
    assert!(!methods.contains(&"audio.speak") && !methods.contains(&"model3d.generate"), "a kind that is not enabled lists no methods: {methods:?}");
    assert!(methods.contains(&"system.ping") && methods.contains(&"system.models.pull"), "{methods:?}");
}

#[test]
fn maintenance_refuses_work_and_answers_the_rest() {
    let a = door("maintenance", vec![entry("pic", "image", &[], true)], vec![]);
    ok(&a, "system.maintenance", json!({"on": true, "message": "swapping a card", "eta_s": 600}));
    // ping, state and busy still answer; ping says why and when to ask again
    let p = ok(&a, "system.ping", json!({}));
    assert_eq!(p["maintenance"], true, "{p}");
    assert_eq!(p["reason"], "swapping a card", "{p}");
    assert!(p["retry_after_s"].as_u64().is_some_and(|s| s > 0), "{p}");
    ok(&a, "system.state", json!({}));
    ok(&a, "system.busy", json!({}));
    // work is refused 503 maintenance in the envelope, with the notice
    let e = err(&a, "image.generate", json!({"prompt": "a lighthouse"}));
    assert_eq!((e["status"].clone(), e["code"].clone()), (json!(503), json!("maintenance")), "{e}");
    assert!(e["retry_after_s"].as_u64().is_some(), "{e}");
    // an OpenAI client gets a 503 with Retry-After's worth in the body
    let (st, text) = http(&a, "POST", "/v1/images/generations", Some(TOKEN), Some(&json!({"prompt": "x"})));
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!((st, v["error"]["code"].clone()), (503, json!("maintenance")), "{text}");
    // every way to start work is refused, the newer ones too: upscaling, vectorizing, a recording
    for method in ["image.upscale", "image.vectorize", "video.upscale", "audio.transcribe"] {
        let e = err(&a, method, json!({"image": "AAAA"}));
        assert_eq!(e["code"], "maintenance", "{method} starts work: {e}");
    }
    for path in ["/v1/images/upscale", "/v1/images/vectorize", "/v1/videos/upscale", "/v1/audio/transcriptions"] {
        let (st, text) = http(&a, "POST", path, Some(TOKEN), Some(&json!({"image": "AAAA"})));
        assert_eq!(st, 503, "{path} starts work: {text}");
    }
    // off: the notice is cleared
    ok(&a, "system.maintenance", json!({"on": false}));
    let p = ok(&a, "system.ping", json!({}));
    assert_eq!(p["maintenance"], false, "{p}");
}

fn card(id: &str, kind: &str, does: &[&str], gib: u64) -> Value {
    json!({"id": id, "kind": kind, "arch": id, "title": id, "about": format!("{id} for a test"), "does": does, "gpus": "0",
           "bytes": gib << 30, "vram_gib": gib})
}

#[test]
fn a_model_a_card_offers_and_is_not_downloaded_says_so() {
    let a = door("pull", vec![entry("chat", "llm", &[], true)], vec![card("ghost-image", "image", &[], 8)]);
    // listed as not downloaded, with its size and where it would go
    let all = ok(&a, "system.models", json!({}));
    let ghost = all.as_array().into_iter().flatten().chain(all["models"].as_array().into_iter().flatten()).find(|m| m["id"] == "ghost-image");
    assert!(ghost.is_some_and(|g| g["downloaded"] == false), "system.models should list ghost-image as not downloaded: {all}");
    // asked for by name: 409 model_not_downloaded with what a client needs to fetch it
    let e = err(&a, "image.generate", json!({"model": "ghost-image", "prompt": "x"}));
    assert_eq!((e["status"].clone(), e["code"].clone()), (json!(409), json!("model_not_downloaded")), "{e}");
    assert_eq!(e["model"], "ghost-image", "{e}");
    assert_eq!(e["pull"], "system.models.pull", "{e}");
    assert_eq!(e["bytes"], 8u64 << 30, "{e}");
    // and on the OpenAI routes, the same code in OpenAI's shape
    let (st, text) = http(&a, "POST", "/v1/images/generations", Some(TOKEN), Some(&json!({"model": "ghost-image", "prompt": "x"})));
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!((st, v["error"]["code"].clone()), (409, json!("model_not_downloaded")), "{text}");
    // a model no card names is plain not found, not a download
    let e = err(&a, "system.models.pull", json!({"model": "no-such-model"}));
    assert_eq!(e["status"], 404, "{e}");
    // no download under way to cancel
    let e = err(&a, "system.models.pull.cancel", json!({"model": "ghost-image"}));
    assert_eq!(e["status"], 409, "{e}");
}

#[test]
fn a_mesh_given_to_3d_goes_to_a_model_that_paints() {
    // TRELLIS-like: makes and textures its own; no paint. Hunyuan-like: also paints a mesh it is given.
    let trellis = entry("trellis", "model3d", &["generate", "texture"], true);
    let a = door("paint-none", vec![trellis.clone()], vec![]);
    // no mesh: generate has a model
    let e = err(&a, "model3d.generate", json!({"model": "nope"}));
    assert_ne!(e["status"], 409, "a named model that is not there is not a routing refusal: {e}");
    // a mesh given: nothing that paints is enabled, so it is refused rather than sent to a model that would ignore it
    let e = err(&a, "model3d.generate", json!({"mesh": "AAAA", "image": "AAAA"}));
    assert_eq!(e["status"], 409, "{e}");
    assert!(e["message"].as_str().is_some_and(|m| m.contains("does paint")), "{e}");
    // naming the one that does not paint refuses too
    let e = err(&a, "model3d.generate", json!({"model": "trellis", "mesh": "AAAA", "image": "AAAA"}));
    assert_eq!(e["status"], 400, "{e}");
    assert!(e["message"].as_str().is_some_and(|m| m.contains("does not paint")), "{e}");
}

#[test]
fn what_a_3d_model_advertises_follows_its_card() {
    let models = vec![entry("trellis", "model3d", &["generate", "texture"], true), entry("hy-paint", "model3d", &["texture", "paint"], true),
                      entry("hy-full", "model3d", &["generate", "texture", "paint", "views"], true), entry("rigger", "model3d", &["rig"], true)];
    let a = door("caps", models, vec![]);
    let all = ok(&a, "system.models", json!({}));
    let list: Vec<Value> = all.as_array().cloned().or_else(|| all["models"].as_array().cloned()).unwrap_or_default();
    let caps = |id: &str| -> (Vec<String>, Vec<String>) {
        let m = list.iter().find(|m| m["id"] == id).unwrap_or_else(|| panic!("{id} not listed in {all}"));
        let s = |k: &str| m[k].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>();
        (s("capabilities"), s("methods"))
    };
    let (c, m) = caps("trellis");
    assert!(c.contains(&"texture".into()) && !c.contains(&"paint".into()) && m.contains(&"model3d.generate".into()), "{c:?} {m:?}");
    // a paint-only model advertises generate (it takes a mesh) and paint, and no mesh-making
    let (c, m) = caps("hy-paint");
    assert!(c.contains(&"paint".into()) && c.contains(&"texture".into()) && !c.contains(&"image_to_3d".into()), "{c:?}");
    assert!(m.contains(&"model3d.generate".into()), "{m:?}");
    let (c, _) = caps("hy-full");
    assert!(c.contains(&"paint".into()) && c.contains(&"image_to_3d".into()) && c.contains(&"multi_view".into()), "{c:?}");
    let (c, m) = caps("rigger");
    assert!(c.contains(&"rig".into()) && !m.contains(&"model3d.generate".into()), "{c:?} {m:?}");
}

#[test]
fn a_recording_goes_to_a_recogniser_and_to_nothing_else() {
    let mut voice = entry("voxcpm2", "audio", &[], true);
    voice["arch"] = json!("voxcpm2");
    voice["speech"] = json!(["clone"]);
    // only a speech model is enabled: transcribing has nobody to go to (409, never retry as is)
    let a = door("stt-none", vec![voice.clone()], vec![]);
    let e = err(&a, "audio.transcribe", json!({"audio": "AAAA"}));
    assert_eq!(e["status"], 409, "{e}");
    assert!(e["message"].as_str().is_some_and(|m| m.contains("no recogniser")), "{e}");
    // naming the speech model for a recording is refused: it makes sound, it does not listen
    let e = err(&a, "audio.transcribe", json!({"audio": "AAAA", "model": "voxcpm2"}));
    assert_eq!(e["status"], 400, "{e}");
    assert!(e["message"].as_str().is_some_and(|m| m.contains("makes sound")), "{e}");
    // with a recogniser enabled it advertises transcribe, and the speech model does not
    let mut whisper = entry("whisper-large-v3-turbo", "audio", &[], true);
    whisper["arch"] = json!("whisper");
    let a = door("stt", vec![voice, whisper], vec![]);
    let all = ok(&a, "system.models", json!({}));
    let list: Vec<Value> = all.as_array().cloned().or_else(|| all["models"].as_array().cloned()).unwrap_or_default();
    let methods = |id: &str| -> Vec<String> {
        list.iter().find(|m| m["id"] == id).unwrap_or_else(|| panic!("{id} not listed in {all}"))["methods"].as_array().into_iter().flatten()
            .filter_map(Value::as_str).map(str::to_string).collect()
    };
    assert_eq!(methods("whisper-large-v3-turbo"), vec!["audio.transcribe".to_string()]);
    assert!(methods("voxcpm2").contains(&"audio.speak".to_string()) && !methods("voxcpm2").contains(&"audio.transcribe".to_string()));
    let v = ok(&a, "system.version", json!({}));
    assert!(v["methods"].as_array().unwrap().iter().any(|m| m == "audio.transcribe"));
    assert!(v["features"].as_array().unwrap().iter().any(|m| m == "audio.transcribe"));
    // a recording is work: maintenance refuses it on /rpc and on the OpenAI route
    ok(&a, "system.maintenance", json!({"on": true, "message": "x", "eta_s": 60}));
    let e = err(&a, "audio.transcribe", json!({"audio": "AAAA"}));
    assert_eq!(e["code"], "maintenance", "{e}");
    let (st, text) = http(&a, "POST", "/v1/audio/transcriptions", Some(TOKEN), Some(&json!({"audio": "AAAA"})));
    assert_eq!(st, 503, "{text}");
}

#[test]
fn a_song_from_a_recording_is_checked_before_it_is_queued() {
    let mut ace = entry("acestep-1.5", "audio", &[], true);
    ace["arch"] = json!("acestep");
    let mut voice = entry("voxcpm2", "audio", &[], true);
    voice["arch"] = json!("voxcpm2");
    voice["speech"] = json!(["clone"]);
    let a = door("song", vec![ace, voice], vec![]);
    // a task that works on a recording needs the recording
    let e = err(&a, "audio.generate", json!({"model": "acestep-1.5", "instructions": "calm", "task": "cover"}));
    assert_eq!(e["status"], 400, "{e}");
    assert!(e["message"].as_str().is_some_and(|m| m.contains("the recording")), "{e}");
    // an unknown task is refused with the ones there are (the option is an enum: the type check names them)
    let e = err(&a, "audio.generate", json!({"model": "acestep-1.5", "instructions": "calm", "task": "remix"}));
    assert_eq!(e["status"], 400, "{e}");
    assert!(e["message"].as_str().is_some_and(|m| ["remix", "text2music", "cover", "repaint", "edit", "lego"].iter().all(|w| m.contains(w))), "{e}");
    // an edit says what the recording sings now
    let e = err(&a, "audio.generate", json!({"model": "acestep-1.5", "instructions": "calm", "task": "edit", "audio": "AAAA"}));
    assert!(e["message"].as_str().is_some_and(|m| m.contains("source_lyrics")), "{e}");
    // a strength is a fraction
    let e = err(&a, "audio.generate", json!({"model": "acestep-1.5", "instructions": "calm", "task": "cover", "audio": "AAAA", "cover_strength": 3}));
    assert_eq!(e["status"], 400, "{e}");
    // a model that cannot take a recording is refused, and named
    let e = err(&a, "audio.generate", json!({"model": "voxcpm2", "instructions": "calm", "task": "cover", "audio": "AAAA"}));
    assert!(e["message"].as_str().is_some_and(|m| m.contains("makes no song from a recording")), "{e}");
}

#[test]
fn state_and_busy_answer_with_nothing_loaded() {
    let a = door("state", vec![entry("chat", "llm", &[], true), entry("pic", "image", &[], false)], vec![card("ghost", "audio", &[], 4)]);
    let st = ok(&a, "system.state", json!({}));
    for k in ["host", "cards", "models", "services", "runs", "maintenance", "available", "pulls", "auto_pull"] {
        assert!(st.get(k).is_some(), "system.state lacks {k}: {st}");
    }
    assert!(st["available"].as_array().is_some_and(|v| v.iter().any(|m| m["id"] == "ghost")), "the card's model is offered: {st}");
    assert_eq!(st["pulls"], json!([]));
    let b = ok(&a, "system.busy", json!({}));
    assert_eq!(b["busy"], false, "{b}");
}

#[test]
fn the_specification_describes_every_method_the_box_offers() {
    let a = door("spec", vec![entry("chat", "llm", &[], true), entry("pic", "image", &[], true), entry("m3d", "model3d", &["generate", "paint"], true)], vec![]);
    // the YAML and the JSON describe the same box
    let (st, yaml) = http(&a, "GET", "/openapi.yml", Some(TOKEN), None);
    assert_eq!(st, 200);
    assert!(yaml.lines().any(|l| l.starts_with("openapi:")) && yaml.lines().any(|l| l.starts_with("paths:")), "a YAML OpenAPI document");
    let (st, text) = http(&a, "GET", "/openapi.json", Some(TOKEN), None);
    assert_eq!(st, 200);
    let spec: Value = serde_json::from_str(&text).expect("openapi.json is JSON");
    assert!(spec["openapi"].as_str().is_some_and(|v| v.starts_with("3.")), "{}", spec["openapi"]);
    let paths = spec["paths"].as_object().expect("paths");
    let v = ok(&a, "system.version", json!({}));
    for m in v["methods"].as_array().unwrap().iter().filter_map(Value::as_str) {
        // each callable method is a path of the specification (/rpc/<method>)
        assert!(paths.contains_key(&format!("/rpc/{m}")), "{m} is callable but not in /openapi.json");
    }
    assert!(paths.contains_key("/rpc/system.ping"));
    assert!(text.contains("model_not_downloaded"), "the not-downloaded error is documented");
    // a kind not enabled is not described
    assert!(!paths.contains_key("/rpc/audio.speak"), "audio is not enabled here");
}

#[test]
fn a_busy_door_keeps_answering_its_other_callers() {
    // many callers at once: the door's per-connection threads answer ping while others ask for state
    let a = Arc::new(door("concurrent", vec![entry("chat", "llm", &[], true)], vec![]));
    let hs: Vec<_> = (0..12).map(|i| {
        let a = a.clone();
        std::thread::spawn(move || {
            let m = if i % 2 == 0 { "system.ping" } else { "system.busy" };
            assert_eq!(rpc(&a, m, json!({}))["ok"], true);
        })
    }).collect();
    for h in hs {
        h.join().unwrap();
    }
}
