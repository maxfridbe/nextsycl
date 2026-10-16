//! Maintenance mode: the box set aside for development. The front door (`home`) turns it on and off
//! (`system.maintenance`, the page, `nextsycl maintenance`) and keeps the notice in `maintenance.json` beside the
//! registry - readable by every service, so the switcher (Open WebUI's endpoint, another user) tells chats too. While
//! it is on, what would start work (chat, pictures, sound, clips, loading or enabling a model) is refused with the
//! error code `maintenance` (503, retryable) carrying the notice; status, listings, cancels and guides answer as
//! always; queued jobs wait; running work finishes. A request with `X-Maintenance-Bypass: <key>` (the key
//! `system.maintenance` answers when it turns the mode on) goes through - a developer testing the box meanwhile.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// The notice's file beside the registry
pub fn file_beside(registry: &Path) -> PathBuf {
    registry.with_file_name("maintenance.json")
}

/// The notice when the mode is on: {on, message, since, until (unix seconds or null), retry_after_s}
pub fn read(file: &Path) -> Option<Value> {
    let v: Value = serde_json::from_slice(&std::fs::read(file).ok()?).ok()?;
    (v["on"] == true).then(|| with_retry(v))
}

/// The seconds until the given end (none given: 15 minutes, for a client's retry)
fn with_retry(mut v: Value) -> Value {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
    let left = v["until"].as_f64().map_or(900.0, |u| (u - now).max(60.0));
    v["retry_after_s"] = json!(left.round() as u64);
    v
}

/// Write the notice (on) or clear it (none); the file stays readable by everyone (the switcher runs as another user)
pub fn write(file: &Path, notice: Option<&Value>) -> Result<(), String> {
    let v = notice.cloned().unwrap_or(json!({"on": false}));
    let tmp = file.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&v).unwrap_or_default()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644));
    }
    std::fs::rename(&tmp, file).map_err(|e| format!("{}: {e}", file.display()))
}

/// The sentence a refused caller reads
pub fn message(n: &Value) -> String {
    let m = n["message"].as_str().filter(|m| !m.trim().is_empty()).unwrap_or("the box is set aside for development");
    let back = n["until"].as_f64().map(|u| {
        let left = (u - std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())).max(0.0);
        format!(" - back in about {} min", (left / 60.0).ceil() as u64)
    }).unwrap_or_default();
    format!("nextsycl is in maintenance: {m}{back}. Try again later (retry_after_s {}).", n["retry_after_s"].as_u64().unwrap_or(900))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_off_and_the_sentence() {
        let dir = std::env::temp_dir().join(format!("ns-maint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = file_beside(&dir.join("models.json"));
        assert!(read(&f).is_none());
        write(&f, Some(&json!({"on": true, "message": "kernel work", "since": 0.0, "until": null}))).unwrap();
        let n = read(&f).unwrap();
        assert_eq!(n["retry_after_s"], 900);
        assert!(message(&n).contains("kernel work"));
        write(&f, None).unwrap();
        assert!(read(&f).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
