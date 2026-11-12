//! The image page's "Describe image" panel (`api/describe` under /open/<id>/, wfe/image): the page's server has no
//! token and no chat model, so this door answers for it with the methods it already has - `image.describers` (GET),
//! `image.describe` as a job (POST: the page's fields, `wait: false`, a label of the page's own), `image.status` (GET
//! `api/describe/status?label=`) and `image.cancel` (POST `api/describe/cancel`, only a label the page made: a
//! cancel by anything else would stop work that is not the page's). The contract is docs/api/image-describe.md.

use std::sync::Arc;

use serde_json::{json, Value};

use super::{now, Home};
use crate::http::{self, Conn};

/// The labels the page makes: a cancel from it names one of these, nothing else
pub(super) const LABEL: &str = "page-describe-";

/// The fields the page may send to image.describe; the rest (queue, priority, rsvp ...) are an API caller's
const FIELDS: [&str; 8] = ["image", "target", "describer", "detail", "max_words", "include_negative", "language", "instructions"];

/// The request image.describe is given for the page's fields: only the ones the page may set, empty ones left out, a job
/// (`wait: false`) with a label of the page's (made when it has none)
pub(super) fn describe_body(b: &Value, stamp: u64) -> Result<Value, String> {
    let mut out = json!({"wait": false});
    for k in FIELDS {
        let v = &b[k];
        if v.is_null() || v.as_str().is_some_and(|x| x.trim().is_empty()) {
            continue;
        }
        out[k] = v.clone();
    }
    if out["image"].is_null() {
        return Err("image: a picture is needed".into());
    }
    out["label"] = match b["label"].as_str() {
        Some(l) if l.starts_with(LABEL) && l.len() <= 80 => json!(l),
        Some(l) if !l.is_empty() => return Err(format!("label {l:?}: the page's labels start with {LABEL}")),
        _ => json!(format!("{LABEL}{stamp}")),
    };
    Ok(out)
}

impl Home {
    /// `api/describe[/status|/cancel]` for an image model's page
    pub(super) fn open_describe(self: &Arc<Self>, s: &mut Conn, req: &http::Request, sub: &str) {
        let reply = |s: &mut Conn, method: &str, r: Result<Value, (u16, String)>| match r {
            Ok(v) => http::respond(s, 200, &json!({"ok": true, "result": v})),
            Err((c, m)) => http::respond(s, c, &json!({"ok": false, "error": super::api::rpc_error(method, c, None, &m)})),
        };
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(json!({}));
        let label = req.target.split_once("label=").map_or_else(|| body["label"].as_str().unwrap_or("").to_string(), |(_, l)| l.split('&').next().unwrap_or("").to_string());
        match (req.method.as_str(), sub) {
            ("GET", "api/describe") => reply(s, "image.describers", self.rpc("image.describers", &json!({}))),
            ("GET", "api/describe/status") if label.starts_with(LABEL) => reply(s, "image.status", self.rpc("image.status", &json!({"label": label}))),
            ("POST", "api/describe/cancel") if label.starts_with(LABEL) => reply(s, "image.cancel", self.rpc("image.cancel", &json!({"label": label}))),
            ("GET" | "POST", "api/describe/status" | "api/describe/cancel") => {
                reply(s, "image.status", Err((400, format!("label: one the page made (it starts with {LABEL})"))))
            }
            ("POST", "api/describe") => {
                if let Some(n) = self.maintenance() {
                    return http::respond(s, 503, &json!({"ok": false, "error": super::api::rpc_error("image.describe", 503, Some("maintenance"), &crate::maintenance::message(&n))}));
                }
                let r = describe_body(&body, now() as u64).map_err(|m| (400, m)).and_then(|b| self.rpc("image.describe", &b));
                reply(s, "image.describe", r)
            }
            _ => http::respond(s, 404, &json!({"ok": false, "error": {"message": format!("no route {} /{sub}", req.method)}})),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_sends_a_job_with_its_own_label_and_only_its_fields() {
        let b = json!({"image": "data:image/png;base64,AA==", "target": "z-image-turbo", "describer": "", "detail": "long", "priority": 9,
                       "queue": false, "include_negative": true, "language": " "});
        let r = describe_body(&b, 7).unwrap();
        assert_eq!(r["wait"], false);
        assert_eq!(r["label"], "page-describe-7");
        assert_eq!((r["target"].as_str(), r["detail"].as_str(), r["include_negative"].as_bool()), (Some("z-image-turbo"), Some("long"), Some(true)));
        for gone in ["describer", "language", "priority", "queue"] {
            assert!(r.get(gone).is_none(), "{gone}");
        }
    }

    #[test]
    fn a_picture_is_needed_and_a_foreign_label_is_refused() {
        assert!(describe_body(&json!({"target": "h3"}), 1).is_err());
        let e = describe_body(&json!({"image": "x", "label": "film-reel"}), 1).unwrap_err();
        assert!(e.contains(LABEL), "{e}");
        assert_eq!(describe_body(&json!({"image": "x", "label": "page-describe-42"}), 1).unwrap()["label"], "page-describe-42");
    }
}
