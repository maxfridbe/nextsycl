//! What the OpenAPI document says beyond the method table: the answers' shapes (`result` of the envelope) and the
//! errors each method can give, as JSON Schema text read once per document.

/// A queue item: a video.generate body as the studio checked it (defaults filled in, files reduced to their names)
const ITEM: &str = r#"{"type": "object", "description": "the request as checked: every video.generate key, defaults filled in (absent ones null)",
  "properties": {"prompt": {"type": "string"}, "label": {"type": "string"}, "project": {"type": "string"}, "batch": {"type": "string"},
    "seconds": {"type": "number"}, "steps": {"type": "integer"}, "seed": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"},
    "engine": {"type": "string"}, "priority": {"type": "integer"}, "te": {"type": "string"}, "chain_mode": {"type": "string"},
    "first_frame": {"type": "string", "nullable": true}, "last_frame": {"type": "string", "nullable": true}, "camera": {"type": "string", "nullable": true}},
  "additionalProperties": true}"#;

/// The running or last clip's record: the item, the anchors resolved to files, the engine's job
const JOB: &str = r#"{"type": "object", "description": "the queue item (every video.generate key) plus what starting it settled; anchors (first_frame ...) now name the files they resolved to",
  "properties": {"name": {"type": "string", "description": "the clip: h3_YYYYMMDD_HHMMSS; its file <name>.mp4 (GET /v1/video/files/<name>.mp4) once done"},
    "id": {"type": "string", "description": "the engine's job id"}, "started": {"type": "number", "description": "unix seconds"},
    "label": {"type": "string"}, "project": {"type": "string"}, "seconds": {"type": "number"}, "engine": {"type": "string"},
    "energy_wh": {"type": "number", "nullable": true}},
  "additionalProperties": true}"#;

const QUEUE_ITEM: &str = r#"{"type": "object", "properties": {"label": {"type": "string"}, "seconds": {"type": "number"}, "steps": {"type": "integer"},
  "engine": {"type": "string"}, "size": {"type": "string", "description": "WxH"}, "chain": {"type": "boolean", "description": "starts on a first_frame"},
  "project": {"type": "string"}, "batch": {"type": "string"}, "held": {"type": "boolean", "description": "paused (queue, project or batch): not run until resumed"}}}"#;

const PROJECT: &str = r#"{"type": "object", "properties": {"project": {"type": "string"}, "clips": {"type": "integer"}, "video_s": {"type": "number"},
  "paused": {"type": "boolean"}, "first_index": {"type": "integer", "description": "its first clip's place in the queue (0-based)"},
  "batches": {"type": "array", "items": {"type": "object", "properties": {"batch": {"type": "string"}, "clips": {"type": "integer"}, "video_s": {"type": "number"}, "paused": {"type": "boolean"}}}}}}"#;

/// video.status (and video.wait)
fn status() -> String {
    format!(r#"{{"type": "object", "required": ["idle", "queue", "queue_items", "version"],
  "description": "the clip running, or the last one (idle: true then; no job at all after a restart with nothing made)",
  "properties": {{"idle": {{"type": "boolean", "description": "no clip is being made (one may still wait in the queue: queue_items)"}},
    "version": {{"type": "string", "description": "changes whenever anything here does (video.wait)"}},
    "job": {JOB}, "stage": {{"type": "string", "description": "starting, text encoder, sampling N/M, video decoded, done, failed ..."}},
    "pct": {{"type": "integer"}}, "eta": {{"type": "integer", "description": "seconds left (estimate)"}}, "elapsed": {{"type": "integer"}},
    "done": {{"type": "boolean"}}, "error": {{"type": "string", "nullable": true, "description": "failed or cancelled, with the engine's last lines"}},
    "steps_done": {{"type": "integer"}}, "steps_total": {{"type": "integer"}}, "frames": {{"type": "integer"}},
    "times": {{"type": "object", "additionalProperties": {{"type": "number"}}, "description": "seconds by stage"}},
    "file": {{"type": "string", "nullable": true, "description": "done: the clip's URL (/v1/video/files/<name>.mp4)"}},
    "engine_log": {{"type": "array", "items": {{"type": "string"}}}},
    "queue": {{"type": "array", "items": {{"type": "string"}}, "description": "the queue's labels in run order"}},
    "queue_items": {{"type": "array", "items": {QUEUE_ITEM}}}, "queue_len": {{"type": "integer"}},
    "paused": {{"type": "boolean"}}, "paused_projects": {{"type": "array", "items": {{"type": "string"}}}}, "paused_batches": {{"type": "array", "items": {{"type": "string"}}}},
    "projects": {{"type": "array", "items": {PROJECT}}}, "all_projects": {{"type": "array", "items": {{"type": "string"}}}}}},
  "additionalProperties": true}}"#)
}

/// The answer of a method, as JSON Schema text (none: not described)
pub fn result(method: &str) -> Option<String> {
    Some(match method {
        "system.busy" => r#"{"type": "object", "required": ["busy"], "properties": {
  "busy": {"type": "boolean", "description": "anything below is true or non-empty"},
  "reasons": {"type": "array", "items": {"type": "string"}, "description": "what is working, in words"},
  "loading": {"type": "array", "items": {"type": "string"}, "description": "models loading"},
  "working": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "kind": {"type": "string"}}}, "description": "picture and sound models answering a request"},
  "llm": {"type": "object", "properties": {"active": {"type": "integer", "description": "chat requests being answered"}}},
  "video": {"type": "object", "nullable": true, "description": "null: no video model enabled", "properties": {
    "running": {"type": "boolean"}, "label": {"type": "string", "nullable": true}, "name": {"type": "string", "nullable": true},
    "stage": {"type": "string", "nullable": true}, "pct": {"type": "integer", "nullable": true}, "eta_s": {"type": "integer", "nullable": true},
    "queued": {"type": "integer"}, "runnable": {"type": "integer", "description": "queued and not held"}, "held": {"type": "integer"},
    "accepts": {"type": "boolean", "description": "video.generate without queue: true starts at once"}}}}}"#.into(),
        "video.generate" => format!(r#"{{"oneOf": [
  {{"type": "object", "description": "queued", "properties": {{"queued": {{"type": "boolean", "enum": [true]}}, "position": {{"type": "integer", "description": "1: made next"}},
    "label": {{"type": "string"}}, "project": {{"type": "string"}}, "batch": {{"type": "string"}}, "held": {{"type": "boolean"}}}}}},
  {{"type": "object", "description": "started", "properties": {{"queued": {{"type": "boolean", "enum": [false]}}, "job": {JOB}}}}}]}}"#),
        "video.status" => status(),
        "video.wait" => status().replacen(r#""required": ["idle","#, r#""required": ["changed", "idle","#, 1)
            .replacen(r#""properties": {"idle""#, r#""properties": {"changed": {"type": "boolean", "description": "false: the timeout passed"}, "idle""#, 1),
        "video.cancel" => r#"{"type": "object", "properties": {"stopped": {"type": "boolean", "description": "the running clip was stopped (at its next step)"},
  "removed": {"type": "integer", "description": "queued clips dropped"}, "label": {"type": "string"}}}"#.into(),
        "video.queue.list" => format!(r#"{{"type": "object", "properties": {{"queue": {{"type": "array", "items": {ITEM}}}}}}}"#),
        "video.queue.move" => r#"{"type": "object", "properties": {"label": {"type": "string"}, "position": {"type": "integer"}, "queue": {"type": "array", "items": {"type": "string"}, "description": "the labels in run order"}}}"#.into(),
        "video.queue.remove" => r#"{"type": "object", "properties": {"removed": {"type": "integer"}, "label": {"type": "string"}}}"#.into(),
        "video.queue.clear" => r#"{"type": "object", "properties": {"cleared": {"type": "integer"}}}"#.into(),
        "video.queue.pause" | "video.queue.resume" => r#"{"type": "object", "properties": {"paused": {"type": "string", "description": "pause: queue | project <p> | batch <b>"},
  "resumed": {"type": "string", "description": "resume: the same"}, "queued": {"type": "integer"}, "held": {"type": "integer"}, "paused_all": {"type": "boolean"},
  "paused_projects": {"type": "array", "items": {"type": "string"}}, "paused_batches": {"type": "array", "items": {"type": "string"}}}}"#.into(),
        "video.projects.list" => format!(r#"{{"type": "object", "properties": {{"projects": {{"type": "array", "items": {PROJECT}}}, "paused_all": {{"type": "boolean"}}}}}}"#),
        "video.timeline" => r#"{"type": "object", "properties": {"clips": {"type": "array", "description": "by n; for each n the newest finished clip, else the running / queued one", "items": {"type": "object", "properties": {
  "n": {"type": "integer", "description": "the NN of the label (0: no number)"}, "name": {"type": "string", "nullable": true, "description": "the clip (done or running)"},
  "state": {"type": "string", "enum": ["done", "running", "held", "queued"]}, "seconds": {"type": "number"}, "camera": {"type": "string", "nullable": true},
  "label": {"type": "string"}, "dialogue": {"type": "string", "description": "the spoken lines"},
  "file": {"type": "string", "nullable": true, "description": "done: the clip's URL"}, "thumb": {"type": "string", "nullable": true, "description": "done: a 240-pixel still's URL (webp)"}}}}}}"#.into(),
        "video.jobs.list" => r#"{"type": "object", "properties": {"jobs": {"type": "array", "description": "every mp4 in the clips directory, newest first", "items": {"type": "object", "properties": {
  "file": {"type": "string", "description": "<name>.mp4 (GET /v1/video/files/<file>)"}, "mtime": {"type": "integer"},
  "size": {"description": "with a record: WxH (string); without: bytes (integer)"}, "size2": {"type": "integer", "description": "bytes (with a record)"},
  "label": {"type": "string"}, "project": {"type": "string"}, "seconds": {"type": "number"}, "steps": {"type": "integer"}, "engine": {"type": "string"},
  "total": {"type": "number", "description": "seconds it took"}, "energy_wh": {"type": "number", "nullable": true}, "camera": {"type": "string", "nullable": true},
  "speech_pct": {"type": "number", "nullable": true}, "te": {"type": "string"}}}}}}"#.into(),
        "video.jobs.get" => format!(r#"{{"type": "object", "description": "a finished clip's record (<name>.json beside the mp4)", "properties": {{
  "job": {JOB}, "times": {{"type": "object", "additionalProperties": {{"type": "number"}}, "description": "seconds by stage; TOTAL the whole"}},
  "frames": {{"type": "integer"}}, "wall": {{"type": "number", "description": "seconds from start to done"}}, "energy_wh": {{"type": "number", "nullable": true}}}}}}"#),
        "video.upload" => r#"{"type": "object", "properties": {"name": {"type": "string", "description": "the file's name in the clips directory (ref_<unix>_<name>): what first_frame, ref_images ... take"}}}"#.into(),
        "video.engines.list" => r#"{"type": "object", "properties": {"engines": {"type": "array", "items": {"type": "object", "properties": {
  "quant": {"type": "string", "description": "the name video.generate's engine takes"}, "file": {"type": "string"}, "path": {"type": "string"},
  "gib": {"type": "number"}, "ready": {"type": "boolean"}, "default": {"type": "boolean"}}}}}}"#.into(),
        "video.canvases.list" => r#"{"type": "object", "properties": {"canvases": {"type": "array", "items": {"type": "object", "properties": {
  "width": {"type": "integer"}, "height": {"type": "integer"}, "aspect": {"type": "string"}, "note": {"type": "string"}, "mpx": {"type": "number"},
  "cost": {"type": "number", "description": "time relative to 640 x 480"}, "default": {"type": "boolean"}}}}}}"#.into(),
        "video.films.list" => r#"{"type": "object", "properties": {"films": {"type": "array", "items": {"type": "object", "properties": {
  "file": {"type": "string"}, "mb": {"type": "number"}, "seconds": {"type": "number", "nullable": true}, "mtime": {"type": "number"}}}}}}"#.into(),
        "video.templates.list" => r#"{"type": "object", "properties": {"templates": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "text": {"type": "string"}}}}}}"#.into(),
        "video.summary" => r#"{"type": "object", "properties": {"rate_wall_per_video_s": {"type": "number"}, "rate_wh_per_video_s": {"type": "number"},
  "running": {"type": "boolean"}, "current_eta_s": {"type": "number"}, "queued": {"type": "integer"}, "queued_video_s": {"type": "number"},
  "remaining_wall_s": {"type": "number"}, "eta_ts": {"type": "number", "nullable": true}, "remaining_kwh": {"type": "number"}}, "additionalProperties": true}"#.into(),
        _ => return None,
    })
}

/// The errors a method gives beyond every call's (401, 400 for a bad body, 409 when its kind has no model enabled):
/// (status, code, when)
pub fn errors(method: &str) -> &'static [(u16, &'static str, &'static str)] {
    match method {
        "video.generate" => &[(400, "invalid_params", "a value outside its limits (the message names it), an unknown engine or sampler, more references than allowed, references with a ControlNet"),
                              (404, "not_found", "a file named (first_frame, ref_images, control_video ...) is not in the clips directory"),
                              (409, "busy", "a clip is being made or a runnable one waits, and queue is not true")],
        "video.cancel" => &[(404, "not_found", "label: no running or queued clip with it"), (409, "not_running", "no label and no clip is being made")],
        "video.queue.move" | "video.queue.remove" => &[(404, "not_found", "no queued clip with that label")],
        "video.jobs.get" => &[(404, "not_found", "no record of that clip")],
        "video.upload" => &[(400, "invalid_params", "not a picture, clip or sound (png jpg jpeg webp mp4 mov webm mkv wav mp3 flac m4a ogg), or empty")],
        "system.enable" | "system.disable" | "system.load" | "system.unload" | "system.gpus" => &[(404, "not_found", "no such model")],
        _ => &[],
    }
}

/// What every answer's error code means
pub const CODES: &str = "Error codes (error.code, with the HTTP status): invalid_request 400 (the body is not a JSON object), invalid_params 400, \
unauthorized 401, not_found 404 (also an unknown method), conflict 409 (a model disabled, none of the kind enabled, a name taken), \
busy 409 (video.generate while a clip is made, without queue: true), not_running 409 (video.cancel with nothing running), \
unavailable 502/503 (the kind's server does not answer), timeout 504, internal 500.";

/// The label convention the video methods share
pub const LABELS: &str = "Video labels: \"<series> NN/MM: text\" (\"Harbour 03/12: the lamp\"). NN orders video.timeline (the edit order); \
the series (the text before NN) is where the anchors prev / prev_cam / first / first_cam look, and the project when the request names none. \
video.cancel, video.queue.move and video.queue.remove name clips by label.";

#[cfg(test)]
mod tests {
    use super::super::api::METHODS;

    #[test]
    fn every_schema_parses() {
        for m in METHODS {
            let op = format!("{}.{}", m.kind, m.name);
            for (f, ty, _, _) in m.fields {
                if ty.starts_with('{') {
                    serde_json::from_str::<serde_json::Value>(ty).unwrap_or_else(|e| panic!("{op} {f}: {e}"));
                }
            }
            if let Some(r) = super::result(&op) {
                serde_json::from_str::<serde_json::Value>(&r).unwrap_or_else(|e| panic!("{op} result: {e}"));
            }
            serde_json::from_str::<serde_json::Value>(m.example).unwrap_or_else(|e| panic!("{op} example: {e}"));
        }
        let w: serde_json::Value = serde_json::from_str(&super::result("video.wait").unwrap()).unwrap();
        assert!(w["properties"]["changed"].is_object() && w["required"][0] == "changed");
    }
}
