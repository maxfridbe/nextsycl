//! `nextsycl models refresh`: an installed entry brought up to its model card from the command line, with a throwaway
//! registry and cards (NS_REGISTRY, NS_MODEL_CARDS): the first write wants --adopt, a dry run writes nothing, a write
//! backs the registry up first, and a registry from another cards directory (a build copy's) is refused.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

const CARD: &str = "id: pic\nkind: image\narch: qimg\ntitle: Pic v2\ngpus: '0'\nvram_gib: 21.0\nenv:\n  NS_A: '{model}'\nfiles:\n  model:\n    key: pic-m\n    \
                    url: https://example.com/m.gguf\n    size: 1\n    sha256: 0000000000000000000000000000000000000000000000000000000000000000\n    path: pic/m.gguf\n";

fn run(dir: &Path, args: &[&str]) -> (bool, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_nextsycl")).arg("models").args(args)
        .env("HOME", dir.join("home")).env("NS_REGISTRY", dir.join("models.json")).env("NS_MODEL_CARDS", dir.join("cards"))
        .env_remove("NS_STUDIO_MODES").output().unwrap();
    (o.status.success(), format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
}

fn backups(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("models.json.bak-")).collect();
    v.sort();
    v
}

#[test]
fn refresh_brings_an_entry_up_to_its_card_and_backs_the_registry_up() {
    let dir = std::env::temp_dir().join(format!("ns-models-refresh-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cards/image")).unwrap();
    std::fs::create_dir_all(dir.join("home")).unwrap();
    std::fs::write(dir.join("cards/image/pic.yml"), CARD).unwrap();
    let entry = json!({"id": "pic", "kind": "image", "arch": "qimg", "title": "Pic", "file": "/m/pic/m.gguf", "files": {"model": "/m/pic/m.gguf"},
                       "gpus": "0,1", "env": {"NS_A": "/m/pic/m.gguf", "NS_OWN": "y"}, "enabled": false, "pinned": true, "catalog": "pic", "vram_gib": 18.0});
    let reg = dir.join("models.json");
    let original = serde_json::to_string_pretty(&json!({"models": [entry, {"id": "hand", "file": "/m/h.gguf", "gpus": "all", "env": {}, "enabled": true}]})).unwrap();
    std::fs::write(&reg, &original).unwrap();

    // no record yet: refused without --adopt, nothing written
    let (ok, out) = run(&dir, &["refresh", "pic"]);
    assert!(!ok && out.contains("--adopt"), "{out}");
    // a dry run shows the changes and writes nothing
    let (ok, out) = run(&dir, &["refresh", "--all", "--dry-run"]);
    assert!(ok && out.contains("title: \"Pic\" -> \"Pic v2\"") && out.contains("vram_gib: 18.0 -> 21.0") && out.contains("gpus: kept \"0,1\"")
            && out.contains("hand: no model card") && out.contains("nothing written"), "{out}");
    assert_eq!(std::fs::read_to_string(&reg).unwrap(), original);
    assert!(backups(&dir).is_empty());

    // adopted: written, the original kept beside it
    let (ok, out) = run(&dir, &["refresh", "pic", "--adopt"]);
    assert!(ok && out.contains("backed up"), "{out}");
    let b = backups(&dir);
    assert_eq!(b.len(), 1);
    assert_eq!(std::fs::read_to_string(&b[0]).unwrap(), original);
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&reg).unwrap()).unwrap();
    let pic = &v["models"][0];
    assert_eq!((pic["title"].clone(), pic["vram_gib"].clone(), pic["gpus"].clone(), pic["enabled"].clone(), pic["pinned"].clone(), pic["env"]["NS_OWN"].clone()),
               (json!("Pic v2"), json!(21.0), json!("0,1"), json!(false), json!(true), json!("y")), "{pic}");
    assert_eq!(v["models"][1]["id"], "hand", "an entry with no card stays");

    // again: nothing to write, no second backup
    let (ok, out) = run(&dir, &["refresh", "--all"]);
    assert!(ok && out.contains("pic: up to date") && out.contains("nothing to write"), "{out}");
    assert_eq!(backups(&dir).len(), 1);

    // the card changes: --reset takes the card's cards
    std::fs::write(dir.join("cards/image/pic.yml"), CARD.replace("vram_gib: 21.0", "vram_gib: 22.5")).unwrap();
    let (ok, out) = run(&dir, &["refresh", "pic", "--reset", "gpus"]);
    assert!(ok && out.contains("vram_gib: 21.0 -> 22.5") && out.contains("gpus: \"0,1\" -> \"0\""), "{out}");

    // a registry another install's cards wrote (a build copy pointed at production's): refused
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&reg).unwrap()).unwrap();
    v["models"][0]["from_card"]["cards"] = json!("/production/model-cards");
    std::fs::write(&reg, v.to_string()).unwrap();
    let (ok, out) = run(&dir, &["refresh", "--all"]);
    assert!(!ok && out.contains("/production/model-cards") && out.contains("NS_REGISTRY"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}
