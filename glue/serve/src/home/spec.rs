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
    "label": {"type": "string"}, "project": {"type": "string"}, "seconds": {"type": "number"}, "frames": {"type": "integer"}, "engine": {"type": "string"},
    "reentrant_id": {"type": "string", "nullable": true},
    "energy_wh": {"type": "number", "nullable": true, "description": "while it runs: so far (updated with each status)"}},
  "additionalProperties": true}"#;

const QUEUE_ITEM: &str = r#"{"type": "object", "properties": {"label": {"type": "string"}, "seconds": {"type": "number"}, "steps": {"type": "integer"},
  "engine": {"type": "string"}, "size": {"type": "string", "description": "WxH"}, "chain": {"type": "boolean", "description": "starts on a first_frame"},
  "project": {"type": "string"}, "batch": {"type": "string"}, "held": {"type": "boolean", "description": "paused (queue, project or batch): not run until resumed"}}}"#;

const PROJECT: &str = r#"{"type": "object", "properties": {"project": {"type": "string"}, "clips": {"type": "integer"}, "video_s": {"type": "number"},
  "paused": {"type": "boolean"}, "first_index": {"type": "integer", "description": "its first clip's place in the queue (0-based)"},
  "batches": {"type": "array", "items": {"type": "object", "properties": {"batch": {"type": "string"}, "clips": {"type": "integer"}, "video_s": {"type": "number"}, "paused": {"type": "boolean"}}}}}}"#;

/// image.generate / image.edit (and /v1/images/*): OpenAI's images answer, with ours
const PICTURES: &str = r#"{"type": "object", "properties": {"created": {"type": "integer"},
  "data": {"type": "array", "items": {"type": "object", "properties": {"url": {"type": "string", "description": "response_format url (the default here): GET it with the token"},
    "b64_json": {"type": "string", "description": "response_format b64_json: the PNG"}, "revised_prompt": {"type": "string"}, "seed": {"type": "integer"},
    "width": {"type": "integer"}, "height": {"type": "integer"}}}},
  "nextsycl": {"type": "object", "properties": {"seconds": {"type": "number"}, "steps": {"type": "integer"}, "seed": {"type": "integer", "description": "the first picture's; the n-th is seed + n"},
    "loras": {"type": "array", "items": {"type": "string"}}, "wh": {"type": "number", "nullable": true, "description": "the card's energy, watt-hours"}}}}}"#;

/// A sound made (audio.history's rows; audio.generate's data[])
const SOUND_ITEM: &str = r#"{"type": "object", "properties": {"url": {"type": "string", "description": "the WAV: GET it with the token"},
    "seconds": {"type": "number", "description": "its length"}, "prompt": {"type": "string"}, "lyrics": {"type": "string"},
    "model": {"type": "string"}, "seed": {"type": "integer"}, "steps": {"type": "integer"}, "cfg": {"type": "number"}, "took": {"type": "number", "description": "seconds it took"},
    "wh": {"type": "number", "nullable": true}, "created": {"type": "integer"},
    "voice": {"type": "string", "nullable": true, "description": "speech"}, "language": {"type": "string", "nullable": true}, "instructions": {"type": "string", "nullable": true},
    "cloned": {"type": "boolean", "description": "speech: spoken in a recording's voice"}}}"#;

/// audio.generate / audio.speak (and /v1/audio/speech with response_format url)
fn sound() -> String {
    format!(r#"{{"type": "object", "properties": {{"created": {{"type": "integer"}}, "data": {{"type": "array", "items": {SOUND_ITEM}}},
  "nextsycl": {{"type": "object", "properties": {{"took": {{"type": "number"}}, "seconds": {{"type": "number"}}, "seed": {{"type": "integer"}}, "steps": {{"type": "integer"}},
    "cfg": {{"type": "number"}}, "wh": {{"type": "number", "nullable": true}}}}}}}}}}"#)
}

/// An image or audio job (jobs.rs): <kind>.status's answer, and generate's with wait: false
const JOB_VIEW: &str = r#"{"type": "object", "required": ["id", "label", "state"], "properties": {
  "id": {"type": "string"}, "kind": {"type": "string", "enum": ["image", "audio"]}, "method": {"type": "string"},
  "label": {"type": "string", "description": "the request's label, else the id"}, "reentrant_id": {"type": "string", "nullable": true},
  "state": {"type": "string", "enum": ["queued", "running", "done", "failed", "cancelled"], "description": "queued: waiting for its server (or its model loading); running: being made (progress)"},
  "created": {"type": "number"}, "started": {"type": "number", "nullable": true}, "ended": {"type": "number", "nullable": true}, "seconds": {"type": "number", "description": "ended: started to ended"},
  "progress": {"type": "object", "description": "running: the server's progress (step at / of, picture, phase, seconds)", "additionalProperties": true},
  "result": {"type": "object", "description": "done: the answer generate gives with wait (data[].url ...)", "additionalProperties": true},
  "error": {"type": "object", "properties": {"status": {"type": "integer"}, "code": {"type": "string"}, "message": {"type": "string"}}},
  "restarted": {"type": "boolean", "description": "the front door restarted while it was unfinished: run again from the start"},
  "reentered": {"type": "boolean", "description": "generate: its reentrant_id named this job, nothing new was started"}}}"#;

/// <kind>.status without a label
const JOB_LIST: &str = r#"{"type": "object", "properties": {"running": {"type": "array", "items": {"type": "object"}}, "queued": {"type": "array", "items": {"type": "object"}},
  "recent": {"type": "array", "items": {"type": "object"}, "description": "the last 20 ended, newest first"}}, "description": "each item as <kind>.status {label} answers"}"#;

/// A chat request as llm.requests lists it
const CHAT_REQUEST: &str = r#"{"type": "object", "properties": {"id": {}, "model": {"type": "string"}, "state": {"type": "string"}, "via": {"type": "string"},
  "started": {"type": "number"}, "ended": {"type": "number", "nullable": true}, "finish": {"type": "string", "nullable": true},
  "prompt_tokens": {"type": "integer"}, "reused": {"type": "integer", "description": "prompt tokens taken from the cache"}, "read": {"type": "integer"},
  "read_tok_s": {"type": "number"}, "read_seconds": {"type": "number"}, "generated": {"type": "integer"}, "max_tokens": {"type": "integer"},
  "tok_s": {"type": "number"}, "generate_seconds": {"type": "number"}, "drafts": {}, "energy_j": {"type": "number"}, "avg_watts": {"type": "number"}}}"#;

const VOICE: &str = r#"{"type": "object", "properties": {"name": {"type": "string"}, "text": {"type": "string", "nullable": true, "description": "what the sample says"},
  "description": {"type": "string", "nullable": true}, "language": {"type": "string", "nullable": true}, "kind": {"type": "string", "enum": ["clone", "design"]},
  "seconds": {"type": "number", "description": "the sample's length (cut to 15 s)"}, "rate": {"type": "integer"}, "created": {"type": "integer"}, "from": {"type": "string"},
  "sample": {"type": "string", "description": "voices.list: the sample's URL (GET with the token)"}}}"#;

const MODEL_STATE: &str = r#"{"type": "object", "properties": {"id": {"type": "string"}, "kind": {"type": "string", "enum": ["llm", "image", "audio", "video"]},
  "title": {"type": "string"}, "arch": {"type": "string"}, "enabled": {"type": "boolean"}, "gpus": {"type": "array", "items": {"type": "integer"}},
  "state": {"type": "string", "enum": ["disabled", "unloaded", "loading", "loaded", "busy", "ready"], "description": "busy: an image or audio model answering; ready: a video model the daemon has, not loaded"},
  "vouch_mb": {"type": "object", "additionalProperties": {"type": "integer"}, "description": "VRAM it is counted for, by card"},
  "measured": {"type": "boolean"}, "headroom_mb": {"type": "object", "additionalProperties": {"type": "integer"}}, "headroom_given": {"type": "boolean"},
  "page": {"type": "integer", "description": "the port of its own page"}}}"#;

/// video.status (and video.wait)
fn status() -> String {
    format!(r#"{{"type": "object", "required": ["idle", "job", "queue", "queue_items", "version"],
  "description": "the clip running, or the last one (idle: true then; no job at all after a restart with nothing made)",
  "properties": {{"idle": {{"type": "boolean", "description": "no clip is being made (one may still wait in the queue: queue_items)"}},
    "version": {{"type": "string", "description": "changes whenever anything here does (video.wait)"}},
    "job": {{"allOf": [{JOB}], "nullable": true, "description": "null: nothing made since the studio started"}}, "stage": {{"type": "string", "description": "starting, text encoder, sampling N/M, video decoded, done, failed ..."}},
    "pct": {{"type": "integer"}}, "eta": {{"type": "integer", "description": "seconds left (estimate)"}}, "elapsed": {{"type": "integer"}},
    "done": {{"type": "boolean"}}, "error": {{"type": "string", "nullable": true, "description": "failed or cancelled, with the engine's last lines"}},
    "steps_done": {{"type": "integer"}}, "steps_total": {{"type": "integer"}}, "frames": {{"type": "integer"}},
    "times": {{"type": "object", "additionalProperties": {{"type": "number"}}, "description": "seconds by stage"}},
    "file": {{"type": "string", "nullable": true, "description": "done: the clip's URL (/v1/video/files/<name>.mp4)"}},
    "preview": {{"type": "string", "nullable": true, "description": "while it samples: a picture of the shot so far (/v1/video/files/<name>.preview.png?t=<when>, PNG, 4x the latent grid: 320x176 for 1280x704) - the denoised estimate's middle frame after each step, its colours from a fitted latent-to-RGB map (close, not the decoder's). The URL changes with each step; null when idle or before the first step"}},
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
  "jobs": {"type": "array", "description": "image and audio jobs not ended", "items": {"type": "object", "properties": {"id": {"type": "string"}, "kind": {"type": "string"}, "label": {"type": "string"}, "state": {"type": "string"}}}},
  "video": {"type": "object", "nullable": true, "description": "null: no video model enabled", "properties": {
    "running": {"type": "boolean"}, "label": {"type": "string", "nullable": true}, "name": {"type": "string", "nullable": true},
    "stage": {"type": "string", "nullable": true}, "pct": {"type": "integer", "nullable": true}, "eta_s": {"type": "integer", "nullable": true}, "elapsed_s": {"type": "integer", "nullable": true},
    "gpu": {"type": "integer", "nullable": true, "description": "the card the clip is made on"}, "energy_wh": {"type": "number", "nullable": true, "description": "the clip's energy so far"},
    "queued": {"type": "integer"}, "runnable": {"type": "integer", "description": "queued and not held"}, "held": {"type": "integer"},
    "accepts": {"type": "boolean", "description": "video.generate without queue: true starts at once"},
    "slots": {"type": "integer", "description": "clips made at once: 1 (the studio makes one clip at a time, on whichever card is free)"}}},
  "slots": {"type": "array", "description": "every place work runs: image and audio servers (two each: one per card), the video daemon's cards, the chat server", "items": {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["image", "audio", "video", "llm"]}, "slot": {}, "port": {"type": "integer"}, "gpu": {"type": "integer", "nullable": true, "description": "its card (null: not running)"},
    "gpus": {"type": "array", "items": {"type": "integer"}, "description": "llm: every card it spans"}, "model": {"type": "string", "nullable": true},
    "up": {"type": "boolean", "description": "a server or engine holds a model there"}, "loading": {"type": "boolean"}, "busy": {"type": "boolean"}}}},
  "cards": {"type": "array", "description": "each GPU and the work on it", "items": {"type": "object", "properties": {
    "gpu": {"type": "integer"}, "name": {"type": "string"}, "vram_total_mb": {"type": "integer"}, "vram_used_mb": {"type": "integer"}, "busy_pct": {"type": "number"}, "power_w": {"type": "number"},
    "models": {"type": "array", "items": {"type": "string"}, "description": "registered models whose processes hold memory there"}, "busy": {"type": "boolean"},
    "work": {"type": "array", "items": {"type": "object", "properties": {"kind": {"type": "string"}, "model": {"type": "string", "nullable": true}}}}}}},
  "parallel": {"type": "object", "description": "how many image / audio jobs run at the same time. Jobs of one model run side by side up to models[id]: a copy of the model a card it is enabled on, at most servers. The first copy takes the first job; a second job finds it busy and starts a second copy on the other card when that card has the room (else it waits for the first). So a long book can submit two audio.speak at once (wait: false) and keep both busy. Jobs of two different models run side by side when each has a server", "properties": {
    "image": {"type": "object", "properties": {"servers": {"type": "integer", "description": "the kind's servers (2)"}, "models": {"type": "object", "additionalProperties": {"type": "integer"}, "description": "enabled model id: its jobs that run at once"}}}, "audio": {"type": "object", "properties": {"servers": {"type": "integer", "description": "the kind's servers (2)"}, "models": {"type": "object", "additionalProperties": {"type": "integer"}, "description": "enabled model id: its jobs that run at once"}}}}}}}"#.into(),
        "system.version" => r#"{"type": "object", "properties": {"version": {"type": "string", "description": "the release (yy.mmdd.###)"}, "api": {"type": "integer", "description": "raised when an answer changes incompatibly"},
  "kinds": {"type": "array", "items": {"type": "string"}, "description": "kinds with an enabled model"}, "features": {"type": "array", "items": {"type": "string"}, "description": "what this build does (names stay; new ones are added)"},
  "methods": {"type": "array", "items": {"type": "string"}, "description": "the methods callable now"}}}"#.into(),
        "system.files.stat" => r#"{"type": "object", "properties": {"name": {"type": "string"}, "bytes": {"type": "integer"}, "sha256": {"type": "string", "description": "hex, of the whole file"},
  "mtime": {"type": "number"}, "url": {"type": "string", "description": "where to GET it (with the token)"}}}"#.into(),
        "system.files.delete" => r#"{"type": "object", "properties": {"deleted": {"type": "array", "items": {"type": "string"}, "description": "the files removed (a clip: its mp4, record, last frame, latents, sound tail)"}}}"#.into(),
        "image.canvases.list" => r#"{"type": "object", "properties": {"rule": {"type": "string"}, "canvases": {"type": "array", "items": {"type": "object", "properties": {
  "size": {"type": "string", "description": "WxH, as image.generate's size takes it"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "aspect": {"type": "string"},
  "mpx": {"type": "number"}, "note": {"type": "string"}, "default": {"type": "boolean"},
  "seconds": {"type": "number", "nullable": true, "description": "measured: the mean time of a picture of this size made here (the pictures kept), per step"},
  "made": {"type": "integer", "description": "pictures of this size it is measured on"}}}}}}"#.into(),
        m if m.ends_with(".describePromptGuideMD") => r#"{"type": "object", "properties": {"model": {"type": "string"}, "arch": {"type": "string"},
  "kind": {"type": "string"}, "markdown": {"type": "string", "description": "the model card's prompt guide: the form the model was trained on, with rules and an example"}}}"#.into(),
        "system.state" => format!(r#"{{"type": "object", "description": "the box, as the page shows it", "properties": {{"ts": {{"type": "number"}},
  "host": {{"type": "object", "description": "memory (meminfo)", "additionalProperties": true}},
  "cards": {{"type": "array", "items": {{"type": "object", "description": "a GPU: memory, the processes on it with their models", "additionalProperties": true}}}},
  "models": {{"type": "array", "items": {MODEL_STATE}}}, "services": {{"type": "object", "additionalProperties": true}},
  "runs": {{"type": "array", "items": {{"type": "object", "properties": {{"id": {{"type": "integer"}}, "label": {{"type": "string"}}, "started": {{"type": "number"}}, "rc": {{"type": "integer", "nullable": true}}, "out": {{"type": "string"}}}}}}}},
  "idle_minutes": {{"type": "integer"}}, "ports": {{"type": "object", "additionalProperties": {{"type": "integer"}}}}}}}}"#),
        "system.models" => format!(r#"{{"type": "object", "properties": {{"models": {{"type": "array", "items": {MODEL_STATE}}}}}}}"#),
        "system.enable" => r#"{"type": "object", "properties": {"model": {"type": "string"}, "title": {"type": "string"}, "enabled": {"type": "boolean", "enum": [true]}}}"#.into(),
        "system.disable" => r#"{"type": "object", "properties": {"model": {"type": "string"}, "enabled": {"type": "boolean", "enum": [false]}}}"#.into(),
        "system.gpus" => r#"{"type": "object", "properties": {"model": {"type": "string"}, "gpus": {"type": "array", "items": {"type": "integer"}}, "changed": {"type": "boolean"},
  "was": {"type": "array", "items": {"type": "integer"}}}}"#.into(),
        "system.load" => r#"{"type": "object", "properties": {"model": {"type": "string"}, "loaded": {"type": "boolean", "description": "wait: true"}, "loading": {"type": "boolean", "description": "without wait: it loads in the background (system.busy / system.models)"}}}"#.into(),
        "system.unload" => r#"{"type": "object", "properties": {"model": {"type": "string"}, "loaded": {"type": "boolean", "enum": [false]}}}"#.into(),
        "system.config" => r#"{"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"},
  "settings": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "group": {"type": "string"}, "value": {}, "default": {}, "what": {"type": "string"}, "readers": {"type": "array", "items": {"type": "string"}}}, "additionalProperties": true}},
  "other": {"type": "array", "items": {"type": "object", "additionalProperties": true}, "description": "lines of the file the program does not read"}}, "additionalProperties": true}"#.into(),
        "system.configSet" => r#"{"type": "object", "properties": {"changed": {"type": "array", "items": {"type": "string"}}, "restart": {"type": "array", "items": {"type": "string"}, "description": "what system.restart must restart to see it"}, "path": {"type": "string"}}, "additionalProperties": true}"#.into(),
        "system.restart" => r#"{"type": "object", "properties": {"restarting": {"type": "string"}}}"#.into(),
        "llm.chat" => r#"{"type": "object", "description": "OpenAI's chat.completion", "properties": {"id": {"type": "string"}, "object": {"type": "string"}, "created": {"type": "integer"}, "model": {"type": "string"},
  "choices": {"type": "array", "items": {"type": "object", "properties": {"index": {"type": "integer"}, "finish_reason": {"type": "string"},
    "message": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string", "nullable": true}, "reasoning_content": {"type": "string", "nullable": true}, "tool_calls": {"type": "array", "items": {"type": "object"}}}}}}},
  "usage": {"type": "object", "properties": {"prompt_tokens": {"type": "integer"}, "completion_tokens": {"type": "integer"}, "total_tokens": {"type": "integer"}}}}, "additionalProperties": true}"#.into(),
        "llm.requests" => format!(r#"{{"type": "object", "properties": {{"active": {{"type": "array", "items": {CHAT_REQUEST}}}, "done": {{"type": "array", "description": "the last 10", "items": {CHAT_REQUEST}}}}}}}"#),
        "llm.models" => r#"{"type": "object", "properties": {"models": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "gpus": {"type": "array", "items": {"type": "integer"}}, "loaded": {"type": "boolean"}}}}}}"#.into(),
        "image.generate" | "image.edit" => format!(r#"{{"oneOf": [{{"allOf": [{PICTURES}, {{"type": "object", "properties": {{"job": {JOB_VIEW}}}}}], "description": "wait (default): the pictures, and the job"}},
  {{"allOf": [{JOB_VIEW}], "description": "wait: false - the job at once (image.status {{label}} for the rest)"}}]}}"#),
        "image.status" | "audio.status" => format!(r#"{{"oneOf": [{JOB_VIEW}, {JOB_LIST}]}}"#),
        "image.cancel" | "audio.cancel" => r#"{"type": "object", "properties": {"cancelled": {"description": "label: how many jobs; none: whether anything was stopped"}, "stopped": {"type": "integer", "description": "requests stopped while running"},
  "label": {"type": "string"}, "jobs": {"type": "array", "items": {"type": "string"}}}}"#.into(),
        "image.ocr" => r#"{"type": "object", "properties": {"created": {"type": "integer"}, "model": {"type": "string"}, "text": {"type": "string", "description": "the reading: Markdown, tables as HTML, LaTeX; pages after a <PAGE> mark each"},
  "prompt": {"type": "string", "description": "the instruction used"}, "pages": {"type": "integer"}, "tokens": {"type": "integer", "description": "tokens written"}, "prompt_tokens": {"type": "integer"},
  "prompt_seconds": {"type": "number"}, "read_tok_s": {"type": "number"}, "write_seconds": {"type": "number"}, "tok_s": {"type": "number"}, "seconds": {"type": "number"}, "wh": {"type": "number", "nullable": true}}}"#.into(),
        "image.info" | "audio.info" => r#"{"type": "object", "properties": {"loaded": {"type": "boolean"},
  "info": {"type": "object", "description": "loaded: the server's own - model, arch, defaults, samplers, schedules, options, loras; image: edits, reads (an OCR model), prompts (its instructions), max_pictures 8, max_pages 24; audio: lyrics, speech (voices, languages, design, clone ...)", "additionalProperties": true},
  "model": {"type": "string", "description": "not loaded"}, "title": {"type": "string"}, "defaults": {"type": "object", "additionalProperties": true}, "note": {"type": "string"}}}"#.into(),
        "image.progress" => r#"{"type": "object", "properties": {"busy": {"type": "boolean"}, "waiting": {"type": "integer", "description": "requests queued behind it"},
  "loaded": {"type": "boolean", "description": "false: no server running (then busy is false)"}, "prompt": {"type": "string"}, "loading": {"type": "boolean"},
  "picture": {"type": "integer"}, "pictures": {"type": "integer"}, "at": {"type": "integer", "description": "step"}, "of": {"type": "integer"}, "seconds": {"type": "number"},
  "reading": {"type": "boolean", "description": "an OCR reading: text so far in text, tokens"}, "pages": {"type": "integer"}, "text": {"type": "string"}, "tokens": {"type": "integer"}}}"#.into(),
        "audio.progress" => r#"{"type": "object", "properties": {"busy": {"type": "boolean"}, "waiting": {"type": "integer"}, "loaded": {"type": "boolean"}, "prompt": {"type": "string"},
  "phase": {"type": "string"}, "at": {"type": "integer"}, "of": {"type": "integer"}, "seconds": {"type": "number"}, "want": {"type": "number", "description": "seconds asked for"}}}"#.into(),
        "image.history" => r#"{"type": "object", "properties": {"data": {"type": "array", "description": "newest first (200 at most; OCR readings 50)", "items": {"type": "object", "properties": {
  "url": {"type": "string", "description": "a server-relative /v1/images/files/<file> (the same route here)"}, "kind": {"type": "string", "description": "ocr for a reading"}, "prompt": {"type": "string"}, "seed": {"type": "integer"}, "steps": {"type": "integer"},
  "width": {"type": "integer"}, "height": {"type": "integer"}, "loras": {"type": "array", "items": {"type": "string"}}, "seconds": {"type": "number"}, "wh": {"type": "number", "nullable": true},
  "text": {"type": "string"}, "pages": {"type": "integer"}, "created": {"type": "integer"}}}}}}"#.into(),
        "audio.history" => format!(r#"{{"type": "object", "properties": {{"data": {{"type": "array", "description": "newest first (200 at most); url server-relative (/v1/audio/files/<file>, the same route here)", "items": {SOUND_ITEM}}}}}}}"#),
        "audio.generate" | "audio.speak" => format!(r#"{{"oneOf": [{{"allOf": [{}, {{"type": "object", "properties": {{"job": {JOB_VIEW}}}}}], "description": "wait (default): the sound, and the job"}},
  {{"allOf": [{JOB_VIEW}], "description": "wait: false - the job at once (audio.status {{label}} for the rest)"}}]}}"#, sound()),
        "audio.voices.list" => format!(r#"{{"type": "object", "properties": {{"voices": {{"type": "array", "items": {VOICE}}}}}}}"#),
        "audio.voices.add" | "audio.voices.design" => format!(r#"{{"type": "object", "properties": {{"voice": {VOICE}, "sample": {{"type": "string", "description": "the sample's URL"}},
  "seed": {{"type": "integer", "description": "design: the seed it was read with"}}}}}}"#),
        "audio.voices.remove" => r#"{"type": "object", "properties": {"removed": {"type": "string"}}}"#.into(),
        "video.generate" => {
            let common = r#""frames": {"type": "integer", "description": "the clip's frames as made (17k + 5)"}, "seconds": {"type": "number", "description": "frames / 24"},
    "ignored": {"type": "array", "items": {"type": "string"}, "description": "keys given that the box does nothing with (negative_prompt, cfg, unknown chain keys)"},
    "reentered": {"type": "boolean", "description": "its reentrant_id named a clip already queued, running or made: that one is answered, nothing new started"},
    "state": {"type": "string", "enum": ["queued", "held", "running", "done"], "description": "reentered: where that clip is"}"#;
            format!(r#"{{"oneOf": [
  {{"type": "object", "description": "queued", "properties": {{"queued": {{"type": "boolean", "enum": [true]}}, "position": {{"type": "integer", "description": "1: made next"}},
    "label": {{"type": "string"}}, "project": {{"type": "string"}}, "batch": {{"type": "string"}}, "held": {{"type": "boolean"}}, {common}}}}},
  {{"type": "object", "description": "started (or reentered: running, or done with name and file)", "properties": {{"queued": {{"type": "boolean", "enum": [false]}}, "job": {JOB},
    "name": {{"type": "string", "description": "reentered, done: the clip"}}, "file": {{"type": "string", "description": "reentered, done: its URL"}}, {common}}}}}]}}"#)
        }
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
  "size": {"type": "integer", "description": "bytes"}, "canvas": {"type": "string", "nullable": true, "description": "WxH; null for an mp4 without a record"},
  "label": {"type": "string"}, "project": {"type": "string"}, "seconds": {"type": "number"}, "steps": {"type": "integer"}, "engine": {"type": "string"},
  "total": {"type": "number", "description": "seconds it took"}, "energy_wh": {"type": "number", "nullable": true}, "camera": {"type": "string", "nullable": true},
  "speech_pct": {"type": "number", "nullable": true}, "te": {"type": "string"}}}}}}"#.into(),
        "video.jobs.get" => format!(r#"{{"type": "object", "description": "a finished clip's record (<name>.json beside the mp4)", "properties": {{
  "job": {JOB}, "times": {{"type": "object", "additionalProperties": {{"type": "number"}}, "description": "seconds by stage; TOTAL the whole"}},
  "frames": {{"type": "integer"}}, "wall": {{"type": "number", "description": "seconds from start to done"}}, "energy_wh": {{"type": "number", "nullable": true}}}}}}"#),
        "video.upload" => r#"{"type": "object", "properties": {"name": {"type": "string", "description": "the file's name in the clips directory (ref_<unix>_<name>): what first_frame, ref_images ... take"}}}"#.into(),
        "video.engines.list" => r#"{"type": "object", "properties": {"engines": {"type": "array", "items": {"type": "object", "properties": {
  "quant": {"type": "string", "description": "the name video.generate's engine takes"}, "file": {"type": "string"}, "path": {"type": "string"},
  "gib": {"type": "number"}, "ready": {"type": "boolean"}, "default": {"type": "boolean"},
  "inputs": {"type": "object", "description": "the video.generate keys it reads and the ones it does not", "properties": {"takes": {"type": "array", "items": {"type": "string"}},
    "not": {"type": "array", "items": {"type": "string"}}, "chosen": {"type": "string", "description": "when it is used"}}}}}}}}"#.into(),
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
        "video.generate" => &[(400, "invalid_params", "a value outside its limits (the message names it), frames not 17k + 5 (the message names the nearest), an unknown engine or sampler, more references than allowed, references with a ControlNet, a bad chain"),
                              (404, "not_found", "a file named (first_frame, ref_images, control_video ...) is not in the clips directory"),
                              (409, "busy", "a clip is being made or a runnable one waits, and queue is not true")],
        "video.cancel" => &[(404, "not_found", "label: no running or queued clip with it"), (409, "not_running", "no label and no clip is being made")],
        "video.queue.move" | "video.queue.remove" => &[(404, "not_found", "no queued clip with that label")],
        "video.jobs.get" => &[(404, "not_found", "no record of that clip")],
        "video.upload" => &[(400, "invalid_params", "not a picture, clip or sound (png jpg jpeg webp mp4 mov webm mkv wav mp3 flac m4a ogg), or empty")],
        "system.enable" | "system.disable" | "system.load" | "system.unload" | "system.gpus" => &[(404, "not_found", "no such model")],
        "image.generate" | "image.edit" | "image.ocr" | "audio.generate" | "audio.speak" | "audio.voices.design" =>
            &[(400, "invalid_params", "the server refused a value (a size, a sampler, too many pictures or pages, an unknown voice ...)"),
              (404, "not_found", "no such model"), (409, "conflict", "the model is disabled, or none that can do it is enabled"),
              (499, "cancelled", "stopped by <kind>.cancel"), (502, "unavailable", "the kind's server did not start or answer"),
              (504, "timeout", "wait: still running after an hour (the job goes on: <kind>.status {label})")],
        "llm.chat" => &[(400, "invalid_params", "the chat server refused the request"), (502, "unavailable", "the chat server does not answer")],
        "audio.voices.add" => &[(400, "invalid_params", "a bad name, a recording shorter than 3 s or not decodable")],
        "audio.voices.remove" => &[(404, "not_found", "no saved voice of that name")],
        "image.status" | "audio.status" => &[(404, "not_found", "no job by that label or reentrant_id")],
        "image.cancel" | "audio.cancel" => &[(404, "not_found", "label: no queued or running job with it")],
        "system.files.stat" => &[(400, "invalid_params", "kind is not image, audio or video, or name is not a plain file name"), (404, "not_found", "no such file")],
        "system.files.delete" => &[(400, "invalid_params", "kind, or a name that is not a clip, upload, picture or sound"), (404, "not_found", "no such file"),
                                   (409, "busy", "video: the clip is being made (video.cancel first)")],
        _ => &[],
    }
}

/// What every answer's error code means
pub const CODES: &str = "Error codes (error.code, with the HTTP status): invalid_request 400 (the body is not a JSON object), invalid_params 400, \
unauthorized 401, not_found 404 (also an unknown method), conflict 409 (a model disabled, none of the kind enabled, a name taken), \
busy 409 (video.generate while a clip is made, without queue: true), not_running 409 (video.cancel with nothing running), \
cancelled 499 (image.cancel / audio.cancel stopped it), unavailable 502/503 (the kind's server does not answer), timeout 504, internal 500.";

/// The label convention the video methods share
pub const LABELS: &str = "Video labels: \"<series> NN/MM: text\" (\"Harbour 03/12: the lamp\"). NN orders video.timeline (the edit order); \
the series (the text before NN) is where the anchors prev / prev_cam / first / first_cam look, and the project when the request names none. \
video.cancel, video.queue.move and video.queue.remove name clips by label.\n\n\
Jobs: image.generate, image.edit, audio.generate and audio.speak run as jobs here - the call can drop and the job goes on; \
label one (else its id is its label), read it with <kind>.status {label}, stop it with <kind>.cancel {label}; wait: false answers at once. \
reentrant_id (every generate, video's too): a submit naming the id of a job queued, running or done gets that job back, nothing new is started \
(a failed or cancelled one is made again).";

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
            let r = super::result(&op).unwrap_or_else(|| panic!("{op}: no answer described"));
            serde_json::from_str::<serde_json::Value>(&r).unwrap_or_else(|e| panic!("{op} result: {e}"));
            serde_json::from_str::<serde_json::Value>(m.example).unwrap_or_else(|e| panic!("{op} example: {e}"));
        }
        let w: serde_json::Value = serde_json::from_str(&super::result("video.wait").unwrap()).unwrap();
        assert!(w["properties"]["changed"].is_object() && w["required"][0] == "changed");
    }
}
