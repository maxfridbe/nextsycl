//! A model's hold: the front door's promise to leave it loaded for a while after a request. `rsvp_s` on a request and
//! X-Keep-Up-S / `keep_up_s` on a chat request are the same hold. The door writes it to `holds.json` beside the
//! registry, so a restart keeps it and the switcher (another process, maybe another user) reads it before it swaps or
//! unloads a chat model. One difference is kept from before: a hold by `rsvp_s` also stops another chat model taking
//! the one chat server, while a keep yields to one - Open WebUI sends its keep on every chat, and picking another model
//! there has always worked.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// `NS_MAX_RSVP_S`'s default: a day
pub const MAX_RSVP_S: u64 = 86400;

/// Until when (unix seconds) a model is kept, and of that until when by rsvp_s
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hold {
    pub keep_until: f64,
    pub rsvp_until: f64,
}

impl Hold {
    /// Seconds it is held for yet, by either
    pub fn left(&self, now: f64) -> f64 {
        (self.keep_until.max(self.rsvp_until) - now).max(0.0)
    }

    /// Seconds rsvp_s holds it for yet
    pub fn reserved(&self, now: f64) -> f64 {
        (self.rsvp_until - now).max(0.0)
    }
}

pub type Holds = BTreeMap<String, Hold>;

pub fn file_beside(registry: &Path) -> PathBuf {
    registry.with_file_name("holds.json")
}

/// A setting's NS_MAX_RSVP_S (unset or not a number: a day)
pub fn max_rsvp_s(setting: Option<String>) -> u64 {
    setting.and_then(|v| v.trim().parse().ok()).unwrap_or(MAX_RSVP_S)
}

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// The holds not yet over
pub fn read(file: &Path) -> Holds {
    let v: Value = std::fs::read(file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
    let t = now();
    v.as_object().into_iter().flatten().filter_map(|(id, h)| {
        let h = Hold { keep_until: h["keep_until"].as_f64().unwrap_or(0.0), rsvp_until: h["rsvp_until"].as_f64().unwrap_or(0.0) };
        (h.left(t) > 0.0).then(|| (id.clone(), h))
    }).collect()
}

/// The holds saved, readable by everyone (the switcher may run as another user)
pub fn write(file: &Path, holds: &Holds) -> Result<(), String> {
    let v: serde_json::Map<String, Value> = holds.iter().map(|(id, h)| (id.clone(), json!({"keep_until": h.keep_until, "rsvp_until": h.rsvp_until}))).collect();
    let tmp = file.with_extension("json.tmp");
    std::fs::write(&tmp, Value::Object(v).to_string()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644));
    }
    std::fs::rename(&tmp, file).map_err(|e| format!("{}: {e}", file.display()))
}

/// `rsvp_s` granted at `now`: clamped to `max`, never shortening what is held, 0 releasing the model altogether.
/// The seconds now held by rsvp_s.
pub fn grant_rsvp(holds: &mut Holds, id: &str, secs: f64, max: u64, now: f64) -> u64 {
    if secs <= 0.0 {
        holds.remove(id);
        return 0;
    }
    let h = holds.entry(id.to_string()).or_default();
    h.rsvp_until = h.rsvp_until.max(now + secs.min(max as f64));
    h.reserved(now).round() as u64
}

/// A keep (X-Keep-Up-S, keep_up_s) at `now`: the same clamp and never shortening; 0 ends the keep, not a reservation
pub fn grant_keep(holds: &mut Holds, id: &str, secs: u64, max: u64, now: f64) {
    if secs == 0 {
        if let Some(h) = holds.get_mut(id) {
            h.keep_until = 0.0;
            if h.left(now) == 0.0 {
                holds.remove(id);
            }
        }
        return;
    }
    let h = holds.entry(id.to_string()).or_default();
    h.keep_until = h.keep_until.max(now + secs.min(max) as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsvp_is_clamped_never_shortened_and_zero_releases() {
        let mut h = Holds::new();
        assert_eq!(grant_rsvp(&mut h, "m", 500.0, 86400, 1000.0), 500);
        assert_eq!(grant_rsvp(&mut h, "m", 100.0, 86400, 1000.0), 500, "a smaller value does not shorten it");
        assert_eq!(grant_rsvp(&mut h, "n", 900.0, 600, 1000.0), 600, "clamped to the most allowed");
        assert_eq!(grant_rsvp(&mut h, "m", 0.0, 600, 1000.0), 0);
        assert!(!h.contains_key("m"), "0 releases");
    }

    #[test]
    fn a_keep_is_the_same_hold_and_cannot_release_a_reservation() {
        let mut h = Holds::new();
        grant_keep(&mut h, "m", 300, 86400, 1000.0);
        assert_eq!(h["m"].left(1000.0), 300.0);
        assert_eq!(h["m"].reserved(1000.0), 0.0, "a keep yields to another chat model");
        grant_keep(&mut h, "m", 60, 86400, 1000.0);
        assert_eq!(h["m"].left(1000.0), 300.0, "not shortened");
        grant_rsvp(&mut h, "m", 50.0, 86400, 1000.0);
        grant_keep(&mut h, "m", 0, 86400, 1000.0);
        assert_eq!((h["m"].left(1000.0), h["m"].reserved(1000.0)), (50.0, 50.0), "the header's 0 ends only the keep");
        h.clear();
        grant_keep(&mut h, "m", 5, 86400, 1000.0);
        grant_keep(&mut h, "m", 0, 86400, 1000.0);
        assert!(h.is_empty());
    }

    #[test]
    fn holds_survive_in_their_file_and_the_finished_are_left_out() {
        let dir = std::env::temp_dir().join(format!("nextsycl-holds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = file_beside(&dir.join("models.json"));
        let mut h = Holds::new();
        grant_rsvp(&mut h, "alive", 600.0, MAX_RSVP_S, now());
        grant_rsvp(&mut h, "over", 5.0, MAX_RSVP_S, now() - 100.0);
        write(&f, &h).unwrap();
        let back = read(&f);
        assert!(back.contains_key("alive") && !back.contains_key("over"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_setting_reads_or_is_a_day() {
        assert_eq!(max_rsvp_s(None), 86400);
        assert_eq!(max_rsvp_s(Some("60".into())), 60);
        assert_eq!(max_rsvp_s(Some("soon".into())), 86400);
    }
}
