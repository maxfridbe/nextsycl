//! `nextsycl models refresh`: an installed model's registry entry brought up to its model card, its files untouched.
//!
//! ```text
//!   nextsycl models refresh <id>... | --all [--dry-run] [--keep FIELD[,FIELD]] [--reset FIELD[,FIELD]] [--adopt]
//! ```
//!
//! The fields an entry takes from its card (`kind`, `arch`, `title`, `gpus`, `env` and `catalog::CARD_KEYS`; the licence
//! fields and `controls` / `annotations` only where the entry has its own) are made again from the card as `pull` makes
//! them, with the entry's own files (where they are now: a drive move repoints them) and its made-locally parts. What
//! is not from the card stays: `file`, `files`, `enabled`, `pinned`, `vouch_mb`, any other key, and every setting the
//! card does not name.
//!
//! What the user changed on a card's field is kept too. `pull` and `refresh` record the card's values on the entry
//! (`from_card`): a field that differs from them was changed on the entry, and stays (`--reset FIELD` takes the card's).
//! An entry with no record yet (pulled before records) keeps its `gpus` (the front door's enable and card picker set
//! them) and its settings the card does not name; its other card fields become the card's. `--keep FIELD` keeps any
//! field (`env.NAME` one setting, `env` all).
//!
//! Each changed model is printed field by field; nothing is written with `--dry-run`. Before writing, the registry is
//! copied to `<registry>.bak-<YYYYmmdd-HHMMSS>`.
//!
//! A build copy must never write the production registry (its cards may be older or newer). The record names the
//! cards directory it came from: a registry whose entries came from another cards directory is refused, and so is
//! the first refresh of a registry no pull or refresh has recorded yet - `--adopt` takes it over, once, on purpose.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::catalog::{self, Catalog, CARD_KEYS};
use crate::config::Config;

/// Where an entry keeps what its card said when it was last pulled or refreshed: {cards, values}
pub const RECORD: &str = "from_card";

/// Card fields the front door sets on an entry (enable with cards, the card picker): with no record to tell, the
/// entry's value stays
const USER_SET: &[&str] = &["gpus"];

/// Card fields the front door reads from the card when an entry lacks them: brought up to the card only where an
/// entry has its own
const LIVE: &[&str] = &["license", "commercial", "license_note", "controls", "annotations", "annotations_note", "max_seconds"];

/// The cards directory as an entry's record names it
pub fn cards_tag(cfg: &Config) -> String {
    let d = catalog::cards_dir(cfg);
    std::fs::canonicalize(&d).unwrap_or(d).to_string_lossy().into_owned()
}

/// The card-derived fields of an entry `pull` made
fn card_part(fresh: &Value) -> Map<String, Value> {
    ["kind", "arch", "title", "gpus", "env"].iter().chain(CARD_KEYS).filter_map(|k| Some((k.to_string(), fresh.get(*k)?.clone()))).collect()
}

/// A pulled entry with its card's values recorded
pub fn stamped(mut e: Value, cards: &str) -> Value {
    e[RECORD] = json!({"cards": cards, "values": card_part(&e)});
    e
}

/// One field's change: `kept` when the entry's value stays (set on the entry, or --keep)
#[derive(Debug, PartialEq)]
pub struct Change {
    pub field: String,
    pub was: Option<Value>,
    pub card: Option<Value>,
    pub kept: bool,
}

impl std::fmt::Display for Change {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let v = |x: &Option<Value>| x.as_ref().map_or("(none)".to_string(), Value::to_string);
        match (&self.was, &self.card, self.kept) {
            (_, _, true) => write!(f, "{}: kept {} (set on the entry; the card says {}; --reset {} takes the card's)", self.field, v(&self.was), v(&self.card), self.field),
            (None, Some(n), _) => write!(f, "{}: {n} (new)", self.field),
            (Some(o), None, _) => write!(f, "{}: removed (was {o})", self.field),
            _ => write!(f, "{}: {} -> {}", self.field, v(&self.was), v(&self.card)),
        }
    }
}

/// What --keep / --reset name: FIELD, env, env.NAME
fn named(list: &[String], field: &str) -> bool {
    list.iter().any(|l| l == field || (l == "env" && field.starts_with("env.")))
}

/// `cur` brought up to `card`: the new entry, each field that differs, and notes (files its card names that it lacks)
pub fn refresh_entry(cat: &Catalog, card: &Value, cur: &Value, cards: &str, keep: &[String], reset: &[String]) -> (Value, Vec<Change>, Vec<String>) {
    let id = cur["id"].as_str().unwrap_or("");
    let mut notes = Vec::new();
    // its own files by role, where they are now; an entry from before roles: its one file
    let have: BTreeMap<String, PathBuf> = cur["files"].as_object().into_iter().flatten().filter_map(|(r, p)| Some((r.clone(), PathBuf::from(p.as_str()?)))).collect();
    let roles = cat.files_of(card);
    let mut paths = BTreeMap::new();
    for (role, _, _) in &roles {
        let p = have.get(*role).cloned().or_else(|| (have.is_empty() && roles.len() == 1).then(|| PathBuf::from(cur["file"].as_str().unwrap_or(""))).filter(|p| !p.as_os_str().is_empty()));
        match p {
            Some(p) => {
                paths.insert(role.to_string(), p);
            }
            None => notes.push(format!("its card names a file it does not have ({role}): nextsycl models pull {id} --again fetches it")),
        }
    }
    // its made-locally parts: what its settings name for them now
    let mut local = BTreeMap::new();
    for (role, _) in card["local"].as_object().into_iter().flatten() {
        let token = format!("{{{role}}}");
        let setting = card["env"].as_object().and_then(|e| e.iter().find(|(_, v)| v.as_str() == Some(token.as_str()))).map(|(k, _)| k.clone());
        if let Some(v) = setting.and_then(|k| cur["env"][&k].as_str().map(str::to_string)) {
            local.insert(role.clone(), v);
        }
    }
    let mut new = card_part(&catalog::entry(card, &paths, &local));
    for k in LIVE {
        if cur.get(*k).is_some() {
            if let Some(v) = card.get(*k) {
                new.insert(k.to_string(), v.clone());
            }
        }
    }
    let rec = cur[RECORD]["values"].as_object();
    let mut out = cur.clone();
    let mut changes = Vec::new();
    // one field (or setting): (the entry's, the card's, the record's); the value it ends with, None to drop it
    let mut decide = |field: String, c: Option<&Value>, n: Option<&Value>, r: Option<Option<&Value>>, legacy_user: bool| -> Option<Option<Value>> {
        if c == n {
            return None;
        }
        let user = match r {
            Some(r) => c != r,
            None => legacy_user,
        };
        let kept = named(keep, &field) || (user && !named(reset, &field));
        changes.push(Change { field, was: c.cloned(), card: n.cloned(), kept });
        (!kept).then(|| n.cloned())
    };
    let fields: Vec<String> = ["kind", "arch", "title", "gpus"].iter().chain(CARD_KEYS).map(|k| k.to_string())
        .chain(LIVE.iter().filter(|k| cur.get(**k).is_some()).map(|k| k.to_string())).collect();
    for k in fields {
        let (c, n) = (cur.get(&k), new.get(&k));
        let legacy_user = c.is_some() && (USER_SET.contains(&k.as_str()) || (LIVE.contains(&k.as_str()) && n.is_none()));
        if let Some(to) = decide(k.clone(), c, n, rec.map(|r| r.get(&k)), legacy_user) {
            match (to, out.as_object_mut()) {
                (Some(v), Some(o)) => {
                    o.insert(k, v);
                }
                (None, Some(o)) => {
                    o.remove(&k);
                }
                _ => {}
            }
        }
    }
    // the settings one by one: those the card does not name are the entry's own
    let empty = Map::new();
    let cur_env = cur["env"].as_object().unwrap_or(&empty);
    let new_env = new.get("env").and_then(Value::as_object).unwrap_or(&empty);
    let rec_env = rec.map(|r| r.get("env").and_then(Value::as_object).unwrap_or(&empty));
    let mut env = cur_env.clone();
    let names: Vec<String> = cur_env.keys().chain(new_env.keys()).chain(rec_env.into_iter().flat_map(|e| e.keys())).fold(Vec::new(), |mut v, k| {
        if !v.contains(k) {
            v.push(k.clone());
        }
        v
    });
    for k in names {
        let (c, n) = (cur_env.get(&k), new_env.get(&k));
        if let Some(to) = decide(format!("env.{k}"), c, n, rec_env.map(|r| r.get(&k)), n.is_none()) {
            match to {
                Some(v) => {
                    env.insert(k, v);
                }
                None => {
                    env.remove(&k);
                }
            }
        }
    }
    if cur.get("env").is_some() || !env.is_empty() {
        out["env"] = Value::Object(env);
    }
    out[RECORD] = json!({"cards": cards, "values": new});
    (out, changes, notes)
}

/// `<registry>.bak-<YYYYmmdd-HHMMSS>` (the local time; -2, -3 ... when that name is taken)
fn backup(path: &Path) -> Result<PathBuf, String> {
    let t = std::process::Command::new("date").arg("+%Y%m%d-%H%M%S").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|t| !t.is_empty())
        .unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0).to_string());
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "models.json".into());
    let mut to = path.with_file_name(format!("{name}.bak-{t}"));
    let mut n = 2;
    while to.exists() {
        to = path.with_file_name(format!("{name}.bak-{t}-{n}"));
        n += 1;
    }
    std::fs::copy(path, &to).map_err(|e| format!("{}: the backup failed, nothing written: {e}", to.display()))?;
    Ok(to)
}

/// Whose registry this is: Err when its entries came from another cards directory, or none has a record yet (the
/// first refresh: --adopt)
fn owner_check(path: &Path, models: &[Value], cards: &str) -> Result<(), String> {
    let mut other: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ours = false;
    for m in models {
        match m[RECORD]["cards"].as_str() {
            Some(c) if c == cards => ours = true,
            Some(c) => other.entry(c.to_string()).or_default().push(m["id"].as_str().unwrap_or("?").to_string()),
            None => {}
        }
    }
    if let Some((c, ids)) = other.iter().next() {
        let more = if ids.len() > 3 { format!(" and {} more", ids.len() - 3) } else { String::new() };
        return Err(format!("{}: {} came from the model cards at {c}, and this program reads {cards}. A build copy must not write \
                            another install's registry: give it its own (NS_REGISTRY in its nextsycl.conf). --adopt takes the \
                            registry over for these cards", path.display(), ids.iter().take(3).cloned().collect::<Vec<_>>().join(", ") + &more));
    }
    if !ours {
        return Err(format!("{}: no entry records which model cards it came from yet. Check that this is this install's \
                            registry (a build copy must not write production's: NS_REGISTRY), then --adopt it for the cards \
                            at {cards} (once; --dry-run shows what changes)", path.display()));
    }
    Ok(())
}

/// `nextsycl models refresh <id>... | --all [--dry-run] [--keep F,..] [--reset F,..] [--adopt]`
pub fn cmd(cfg: &Config, path: &Path, models: &mut [Value], ids: &[String], raw: &[String]) -> Result<bool, String> {
    let flag = |k: &str| raw.iter().any(|a| a == k);
    let list = |k: &str| -> Vec<String> {
        raw.iter().enumerate().filter(|(_, a)| *a == k).filter_map(|(i, _)| raw.get(i + 1)).flat_map(|v| v.split(',')).map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect()
    };
    let (all, dry, adopt) = (flag("--all"), flag("--dry-run"), flag("--adopt"));
    if ids.is_empty() == !all {
        return Err("nextsycl models refresh <id>... | --all [--dry-run] [--keep FIELD,..] [--reset FIELD,..] [--adopt]".into());
    }
    let (keep, reset) = (list("--keep"), list("--reset"));
    let cat = Catalog::load(cfg)?;
    let cards = cards_tag(cfg);
    if let Err(e) = owner_check(path, models, &cards) {
        if !adopt && !dry {
            return Err(e);
        }
        if !adopt {
            println!("note (a dry run goes on): {e}");
        }
    }
    let at: Vec<usize> = if all {
        (0..models.len()).collect()
    } else {
        ids.iter().map(|id| models.iter().position(|m| m["id"] == id.as_str()).ok_or_else(|| format!("no model {id} (nextsycl models list)"))).collect::<Result<_, _>>()?
    };
    let mut changed = 0;
    for i in at {
        let cur = &models[i];
        let id = cur["id"].as_str().unwrap_or("").to_string();
        let card_id = cur["catalog"].as_str().filter(|c| !c.is_empty()).unwrap_or(&id).to_string();
        let Some(card) = cat.model(&card_id) else {
            println!("{id}: no model card ({card_id}): left as it is");
            continue;
        };
        let (new, changes, notes) = refresh_entry(&cat, card, cur, &cards, &keep, &reset);
        let applied = changes.iter().filter(|c| !c.kept).count();
        if new == *cur {
            println!("{id}: up to date");
        } else {
            changed += 1;
            if applied == 0 {
                println!("{id} (card {card_id}): no field from the card changes; the card's values recorded");
            } else {
                println!("{id} (card {card_id}): {applied} field(s) from the card");
            }
        }
        for c in &changes {
            println!("  {c}");
        }
        for n in &notes {
            println!("  note: {n}");
        }
        models[i] = new;
    }
    if changed == 0 {
        println!("nothing to write");
        return Ok(false);
    }
    if dry {
        println!("dry run: {changed} entr{} would change; nothing written ({})", if changed == 1 { "y" } else { "ies" }, path.display());
        return Ok(false);
    }
    let b = backup(path)?;
    println!("backed up {} to {}", path.display(), b.display());
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat(dir: &Path) -> Catalog {
        std::fs::create_dir_all(dir.join("image")).unwrap();
        let card = "id: pic\nkind: image\narch: qimg\ntitle: Pic v2\ngpus: '0'\nvram_gib: 21.0\nlicense: apache-2.0\nenv:\n  NS_A: '{model}'\n  NS_B: '2'\n  NS_NEW: x\n\
                    files:\n  model:\n    key: pic-m\n    url: https://example.com/m.gguf\n    size: 1\n    sha256: 0000000000000000000000000000000000000000000000000000000000000000\n    path: pic/m.gguf\n";
        std::fs::write(dir.join("image/pic.yml"), card).unwrap();
        Catalog::from_dir(dir).unwrap()
    }

    #[test]
    fn an_entry_takes_its_cards_fields_and_keeps_its_own() {
        let d = std::env::temp_dir().join(format!("nextsycl-refresh-{}", std::process::id()));
        let c = cat(&d);
        let card = c.model("pic").unwrap();
        // an entry pulled before records: moved to another drive, enabled on both cards, pinned, a setting of its own
        let cur = json!({"id": "pic", "kind": "image", "arch": "qimg", "title": "Pic", "file": "/moved/pic/m.gguf", "files": {"model": "/moved/pic/m.gguf"},
                         "gpus": "0,1", "env": {"NS_A": "/old/pic/m.gguf", "NS_B": "1", "NS_OWN": "y", "NS_SILO": "0"}, "enabled": true, "pinned": true,
                         "catalog": "pic", "vram_gib": 18.0, "streams": true, "vouch_mb": {"0": 100}});
        let (new, changes, notes) = refresh_entry(&c, card, &cur, "/cards", &[], &[]);
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!((new["title"].clone(), new["vram_gib"].clone(), new.get("streams")), (json!("Pic v2"), json!(21.0), None), "{new}");
        // its own: files, enabled, pinned, its picked cards, its setting, the front door's figures
        assert_eq!((new["file"].clone(), new["enabled"].clone(), new["pinned"].clone(), new["gpus"].clone(), new["vouch_mb"].clone()),
                   (json!("/moved/pic/m.gguf"), json!(true), json!(true), json!("0,1"), json!({"0": 100})));
        // the card's settings with its files where they are now; one it does not name stays
        assert_eq!(new["env"], json!({"NS_A": "/moved/pic/m.gguf", "NS_B": "2", "NS_OWN": "y", "NS_SILO": "0", "NS_NEW": "x"}));
        assert!(changes.iter().any(|x| x.field == "gpus" && x.kept) && changes.iter().any(|x| x.field == "env.NS_OWN" && x.kept));
        assert!(changes.iter().any(|x| x.field == "streams" && !x.kept && x.card.is_none()));
        assert_eq!(new[RECORD]["cards"], "/cards");
        // the licence: read from the card while the entry has none
        assert!(new.get("license").is_none());

        // refreshed again: nothing changes
        let (again, ch, _) = refresh_entry(&c, card, &new, "/cards", &[], &[]);
        assert_eq!(again, new);
        assert!(ch.iter().all(|x| x.kept), "{ch:?}");
        // with a record: a field the user changed after it stays, one they did not follows the card; --reset takes the card's
        let mut user = new.clone();
        user["title"] = json!("My pic");
        user["env"]["NS_B"] = json!("5");
        let (u, _, _) = refresh_entry(&c, card, &user, "/cards", &[], &[]);
        assert_eq!((u["title"].clone(), u["env"]["NS_B"].clone()), (json!("My pic"), json!("5")));
        let (u, _, _) = refresh_entry(&c, card, &user, "/cards", &[], &["title".into(), "env".into()]);
        assert_eq!((u["title"].clone(), u["env"]["NS_B"].clone(), u["gpus"].clone()), (json!("Pic v2"), json!("2"), json!("0,1")));
        // the user moved it back to the card's gpus: from then on it follows the card
        let mut back = new.clone();
        back["gpus"] = json!("0");
        let (b, _, _) = refresh_entry(&c, card, &back, "/cards", &[], &[]);
        assert_eq!(b["gpus"], "0");
        // --keep holds a field the card changed
        let (k, ch, _) = refresh_entry(&c, card, &cur, "/cards", &["vram_gib".into()], &[]);
        assert_eq!(k["vram_gib"], 18.0);
        assert!(ch.iter().any(|x| x.field == "vram_gib" && x.kept));
        // a licence the entry carries follows the card
        let mut lic = cur.clone();
        lic["license"] = json!("other");
        assert_eq!(refresh_entry(&c, card, &lic, "/cards", &[], &[]).0["license"], "apache-2.0");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_registry_from_other_cards_is_refused() {
        let p = Path::new("/r/models.json");
        let ours = json!({"id": "a", RECORD: {"cards": "/mine"}});
        let theirs = json!({"id": "b", RECORD: {"cards": "/prod/model-cards"}});
        let bare = json!({"id": "c"});
        assert!(owner_check(p, &[ours.clone(), bare.clone()], "/mine").is_ok());
        let e = owner_check(p, &[ours, theirs, bare.clone()], "/mine").unwrap_err();
        assert!(e.contains("/prod/model-cards") && e.contains("NS_REGISTRY") && e.contains("--adopt"), "{e}");
        let e = owner_check(p, &[bare], "/mine").unwrap_err();
        assert!(e.contains("--adopt"), "{e}");
    }

    #[test]
    fn a_change_says_what_it_did() {
        let c = |was: Option<Value>, card: Option<Value>, kept| Change { field: "f".into(), was, card, kept }.to_string();
        assert_eq!(c(Some(json!(1)), Some(json!(2)), false), "f: 1 -> 2");
        assert_eq!(c(None, Some(json!(2)), false), "f: 2 (new)");
        assert_eq!(c(Some(json!(1)), None, false), "f: removed (was 1)");
        assert!(c(Some(json!("0,1")), Some(json!("0")), true).starts_with("f: kept \"0,1\" (set on the entry"));
    }
}
