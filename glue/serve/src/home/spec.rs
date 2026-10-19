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
  "run": {"type": "object", "description": "how it was made (the /v1 routes call this nextsycl; /rpc names it run)", "properties": {"seconds": {"type": "number"}, "steps": {"type": "integer"}, "seed": {"type": "integer", "description": "the first picture's; the n-th is seed + n"},
    "loras": {"type": "array", "items": {"type": "string"}}, "wh": {"type": "number", "nullable": true, "description": "the card's energy, watt-hours"}}}}}"#;

/// A sound made (audio.history's rows; audio.generate's data[])
const SOUND_ITEM: &str = r#"{"type": "object", "properties": {"url": {"type": "string", "description": "the WAV: GET it with the token"},
    "seconds": {"type": "number", "description": "its length"}, "prompt": {"type": "string"}, "lyrics": {"type": "string"},
    "model": {"type": "string"}, "seed": {"type": "integer"}, "steps": {"type": "integer"}, "cfg": {"type": "number"}, "took": {"type": "number", "description": "seconds it took"},
    "wh": {"type": "number", "nullable": true}, "created": {"type": "integer"},
    "voice": {"type": "string", "nullable": true, "description": "speech"}, "language": {"type": "string", "nullable": true}, "instructions": {"type": "string", "nullable": true},
    "cloned": {"type": "boolean", "description": "speech: spoken in a recording's voice"}}}"#;

/// One mesh made (model3d.generate / model3d.rig; model3d.history's items)
const MESH_ITEM: &str = r#"{"type": "object", "properties": {
    "url": {"type": "string", "description": "the GLB (glTF binary): GET it with the token"}, "picture": {"type": "string", "description": "generate: the picture it was made from (PNG)"},
    "model": {"type": "string"}, "arch": {"type": "string"}, "seed": {"type": "integer"}, "steps": {"type": "integer"}, "cfg": {"type": "number"}, "resolution": {"type": "integer"},
    "faces": {"type": "integer", "description": "the triangle budget asked"}, "triangles": {"type": "integer", "description": "what it has"}, "texture": {"type": "boolean"}, "textured": {"type": "boolean"},
    "rigged": {"type": "integer", "nullable": true, "description": "rig: its joints"}, "animated": {"type": "integer", "nullable": true, "description": "animate: its frames"},
    "prompt": {"type": "string", "nullable": true}, "views": {"type": "array", "items": {"type": "string", "nullable": true}}, "bytes": {"type": "integer"},
    "took": {"type": "number"}, "wh": {"type": "number", "nullable": true}, "created": {"type": "integer"}}}"#;

/// model3d.generate / model3d.rig (and /v1/model3d/... with response_format url)
fn meshes() -> String {
    format!(r#"{{"type": "object", "properties": {{"created": {{"type": "integer"}}, "data": {{"type": "array", "items": {MESH_ITEM}}},
  "run": {{"type": "object", "description": "how it was made (the /v1 routes call this nextsycl)", "properties": {{"took": {{"type": "number"}}, "seed": {{"type": "integer"}}, "wh": {{"type": "number", "nullable": true}}, "triangles": {{"type": "integer"}}}}}}}}}}"#)
}

/// audio.generate / audio.speak (and /v1/audio/speech with response_format url)
fn sound() -> String {
    format!(r#"{{"type": "object", "properties": {{"created": {{"type": "integer"}}, "data": {{"type": "array", "items": {SOUND_ITEM}}},
  "run": {{"type": "object", "description": "how it was made (the /v1 routes call this nextsycl; /rpc names it run)", "properties": {{"took": {{"type": "number"}}, "seconds": {{"type": "number"}}, "seed": {{"type": "integer"}}, "steps": {{"type": "integer"}},
    "cfg": {{"type": "number"}}, "wh": {{"type": "number", "nullable": true}}}}}}}}}}"#)
}

/// An image, audio or 3D job (jobs.rs): <kind>.status's answer, and generate's with wait: false
const JOB_VIEW: &str = r#"{"type": "object", "required": ["id", "label", "state"], "properties": {
  "id": {"type": "string"}, "kind": {"type": "string", "enum": ["image", "audio", "model3d"]}, "method": {"type": "string"},
  "label": {"type": "string", "description": "the request's label, else the id"}, "reentrant_id": {"type": "string", "nullable": true},
  "state": {"type": "string", "enum": ["queued", "running", "done", "failed", "cancelled"], "description": "queued: not being made yet (stage says why); running: being made (progress)"},
  "stage": {"type": "string", "enum": ["queued", "held", "starting", "waiting", "loading", "running", "done", "failed", "cancelled"], "description": "where it is: queued (in the kind's queue behind others), held (paused: queue.resume), starting (being handed to its server), waiting (for room on a card, a server to finish another model's or another job's work), loading (its model), running, then how it ended"},
  "waiting_for": {"type": "string", "nullable": true, "description": "not running yet: what it waits for, in words - the jobs ahead, the model loading on GPU n, the room on a card (free and needed, and who holds the rest), the job its server finishes first"},
  "position": {"type": "integer", "description": "queued: its place in the queue (1: next; held ones count)"}, "held": {"type": "boolean", "description": "queued: held by itself, its project, batch or the queue"},
  "priority": {"type": "integer"}, "project": {"type": "string", "nullable": true}, "batch": {"type": "string", "nullable": true},
  "created": {"type": "number"}, "started": {"type": "number", "nullable": true}, "ended": {"type": "number", "nullable": true}, "seconds": {"type": "number", "description": "ended: started to ended"},
  "progress": {"type": "object", "description": "running: the server's progress (step at / of, picture, phase, seconds)", "additionalProperties": true},
  "result": {"type": "object", "description": "done: the answer generate gives with wait (data[].url ...)", "additionalProperties": true},
  "error": {"type": "object", "description": "failed or cancelled: the /rpc error object (code, message, retryable, status, method, field)", "properties": {"status": {"type": "integer"}, "code": {"type": "string"}, "message": {"type": "string"}, "retryable": {"type": "boolean", "description": "the same call may succeed later unchanged (no_room, unavailable, timeout); false: change the request (invalid_params, not_found, conflict, cancelled)"},
    "method": {"type": "string", "nullable": true}, "field": {"type": "string", "nullable": true}}},
  "restarted": {"type": "boolean", "description": "the front door restarted while it was unfinished: run again from the start"},
  "reentered": {"type": "boolean", "description": "generate: its reentrant_id named this job, nothing new was started"}}}"#;

/// <kind>.jobs
const JOB_LIST: &str = r#"{"type": "object", "properties": {"running": {"type": "array", "items": {"type": "object"}}, "queued": {"type": "array", "items": {"type": "object"}, "description": "in run order"},
  "recent": {"type": "array", "items": {"type": "object"}, "description": "the last ended (limit, 20 by default), newest first"},
  "at_once": {"type": "integer", "description": "jobs of the kind made at once (its servers: one a card)"},
  "paused": {"type": "object", "description": "what queue.pause holds: all, projects, batches", "properties": {"all": {"type": "boolean"}, "projects": {"type": "array", "items": {"type": "string"}}, "batches": {"type": "array", "items": {"type": "string"}}}}},
  "description": "each item as <kind>.status {label} answers"}"#;

/// A chat request as llm.requests lists it
const CHAT_REQUEST: &str = r#"{"type": "object", "properties": {"id": {}, "model": {"type": "string"}, "state": {"type": "string"}, "via": {"type": "string"},
  "started": {"type": "number"}, "ended": {"type": "number", "nullable": true}, "finish": {"type": "string", "nullable": true},
  "prompt_tokens": {"type": "integer"}, "reused": {"type": "integer", "description": "prompt tokens taken from the cache"}, "read": {"type": "integer"},
  "read_tok_s": {"type": "number"}, "read_seconds": {"type": "number"}, "generated": {"type": "integer"}, "max_tokens": {"type": "integer"},
  "tok_s": {"type": "number"}, "generate_seconds": {"type": "number"}, "drafts": {}, "energy_j": {"type": "number"}, "avg_watts": {"type": "number"}}}"#;

const VOICE: &str = r#"{"type": "object", "properties": {"name": {"type": "string"}, "text": {"type": "string", "nullable": true, "description": "what the sample says"},
  "description": {"type": "string", "nullable": true}, "language": {"type": "string", "nullable": true}, "kind": {"type": "string", "enum": ["clone", "design"]},
  "seconds": {"type": "number", "description": "the sample's length (cut to 15 s)"}, "rate": {"type": "integer"}, "created": {"type": "integer"}, "from": {"type": "string"},
  "sample": {"type": "string", "description": "voices.list: the sample's URL (GET with the token)"},
  "owner": {"type": "string", "nullable": true, "description": "who it belongs to (voices.add / voices.design owner)"},
  "ready": {"type": "boolean", "description": "voices.list: it can be spoken now - its sample reads and an enabled model clones (models)"},
  "models": {"type": "array", "items": {"type": "string"}, "description": "voices.list: the enabled models that speak it"},
  "problem": {"type": "string", "description": "voices.list, not ready: why"}}}"#;

const MODEL_STATE: &str = r#"{"type": "object", "properties": {"id": {"type": "string"}, "kind": {"type": "string", "enum": ["llm", "image", "audio", "video"]},
  "title": {"type": "string"}, "arch": {"type": "string"}, "enabled": {"type": "boolean"}, "gpus": {"type": "array", "items": {"type": "integer"}},
  "state": {"type": "string", "enum": ["disabled", "unloaded", "loading", "loaded", "busy", "ready"], "description": "busy: an image or audio model answering; ready: a video model the daemon has, not loaded"},
  "vouch_mb": {"type": "object", "additionalProperties": {"type": "integer"}, "description": "VRAM it is counted for, by card"},
  "spans": {"type": "boolean", "description": "a chat model spread over every card (its model card: gpus all): its cards are not a choice"},
  "measured": {"type": "boolean"}, "headroom_mb": {"type": "object", "additionalProperties": {"type": "integer"}}, "headroom_given": {"type": "boolean"},
  "page": {"type": "integer", "description": "the port of its own page"},
  "capabilities": {"type": "array", "items": {"type": "string"}, "description": "what it does, in neutral words: chat, stream, tools; generate, edit, compose, img2img, loras, few_step; ocr, read_pages, pdf; song, lyrics, instrumental; speech, builtin_voices, clone, saved_voices, enrolled_voices, continue_from, voice_design, voice_direction, vocal_events; text_to_video, image_to_video, sound, chain, first_last_frame, voice_reference, controlnet, inpaint, reference_images, reference_videos"},
  "methods": {"type": "array", "items": {"type": "string"}, "description": "the /rpc methods that take it as model"}}}"#;

/// video.status (and video.wait)
fn status() -> String {
    format!(r#"{{"type": "object", "required": ["idle", "job", "queue", "queue_items", "version"],
  "description": "the clip running, or the last one (idle: true then; no job at all after a restart with nothing made)",
  "properties": {{"idle": {{"type": "boolean", "description": "no clip is being made (one may still wait in the queue: queue_items)"}},
    "version": {{"type": "string", "description": "changes whenever anything here does (video.wait)"}},
    "job": {{"allOf": [{JOB}], "nullable": true, "description": "null: nothing made since the studio started"}}, "stage": {{"type": "string", "description": "starting, text encoder, sampling N/M, video decoded, done, failed ..."}},
    "pct": {{"type": "integer"}}, "eta": {{"type": "integer", "description": "seconds left (estimate)"}}, "elapsed": {{"type": "integer"}},
    "done": {{"type": "boolean"}}, "error": {{"type": "string", "nullable": true, "description": "failed or cancelled, with the engine's last lines (text, as before; failure has it as data)"}},
    "failure": {{"type": "object", "nullable": true, "description": "failed or cancelled: what kind, and whether the same request again can help", "properties": {{
      "code": {{"type": "string", "enum": ["out_of_memory", "engine_lost", "invalid_input", "engine_error", "cancelled"]}},
      "retryable": {{"type": "boolean", "description": "true only for engine_lost (the engine process ended or stopped answering); out_of_memory needs fewer frames, a smaller canvas or engine"}},
      "message": {{"type": "string", "description": "the engine's error"}}, "hint": {{"type": "string"}}}}}},
    "gpu": {{"type": "integer", "nullable": true, "description": "the card the clip runs on (null when idle; while it waits for a card, the one it asked for or null)"}},
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
/// A move of a model's files (system.move, system.drives' move)
const MOVE: &str = r#"{"type": "object", "properties": {"model": {"type": "string"}, "to": {"type": "string", "description": "the drive"}, "from": {"type": "array", "items": {"type": "string"}},
  "folder": {"type": "string"}, "state": {"type": "string", "enum": ["copying", "registry", "removing", "done", "failed", "cancelled"]}, "running": {"type": "boolean"},
  "files": {"type": "integer"}, "file_index": {"type": "integer"}, "file": {"type": "string", "nullable": true}, "file_to": {"type": "string", "nullable": true},
  "bytes": {"type": "integer", "description": "the files moved"}, "need": {"type": "integer", "description": "bytes written there (a file there already, the same bytes, is used)"},
  "done": {"type": "integer"}, "work": {"type": "integer", "description": "bytes read and written in all: each copied byte twice (copied, read back)"}, "pct": {"type": "number"},
  "rate_mb_s": {"type": "number"}, "eta_s": {"type": "number", "nullable": true}, "started": {"type": "number"}, "seconds": {"type": "number"},
  "result": {"type": "object", "nullable": true, "description": "done: renamed (old path: new), removed (the old files), kept (old files another model names, with those models)", "additionalProperties": true},
  "error": {"type": "string", "nullable": true}, "warning": {"type": "string", "nullable": true, "description": "a model that streams from the disk, to a slower drive"}}}"#;

pub fn result(method: &str) -> Option<String> {
    Some(match method {
        "system.busy" => r#"{"type": "object", "required": ["busy"], "properties": {
  "busy": {"type": "boolean", "description": "anything below is true or non-empty"},
  "reasons": {"type": "array", "items": {"type": "string"}, "description": "what is working, in words"},
  "loading": {"type": "array", "items": {"type": "string"}, "description": "models loading"},
  "working": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "kind": {"type": "string"}}}, "description": "picture and sound models answering a request"},
  "llm": {"type": "object", "properties": {"active": {"type": "integer", "description": "chat requests being answered"}}},
  "jobs": {"type": "array", "description": "image and audio jobs not ended (held ones too; they are no reason for busy)", "items": {"type": "object", "properties": {"id": {"type": "string"}, "kind": {"type": "string"}, "label": {"type": "string"}, "state": {"type": "string"}, "held": {"type": "boolean"}}}},
  "video": {"type": "object", "nullable": true, "description": "null: no video model enabled", "properties": {
    "running": {"type": "boolean"}, "label": {"type": "string", "nullable": true}, "name": {"type": "string", "nullable": true},
    "stage": {"type": "string", "nullable": true}, "pct": {"type": "integer", "nullable": true}, "eta_s": {"type": "integer", "nullable": true}, "elapsed_s": {"type": "integer", "nullable": true},
    "gpu": {"type": "integer", "nullable": true, "description": "the card the clip is made on (null when nothing runs, or while the clip waits for a card the engine has not given it yet)"}, "energy_wh": {"type": "number", "nullable": true, "description": "the clip's energy so far"},
    "queued": {"type": "integer"}, "runnable": {"type": "integer", "description": "queued and not held"}, "held": {"type": "integer"},
    "accepts": {"type": "boolean", "description": "video.generate without queue: true starts at once"},
    "slots": {"type": "integer", "description": "clips made at once: 1. The studio makes one clip at a time; the daemon has a worker on each card (cards, and the video rows of the slots list) and the clip goes to whichever is free, so two video rows never mean two clips"},
    "cards": {"type": "integer", "description": "cards the video daemon can make a clip on"}}},
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
        "system.drives" => format!(r#"{{"type": "object", "properties": {{
  "drives": {{"type": "array", "items": {{"type": "object", "properties": {{
    "id": {{"type": "string", "description": "the device's name (nvme1n1, nvme0n1p3): what drive takes"}}, "disk": {{"type": "string"}}, "model": {{"type": "string", "description": "the disk's model (sysfs)"}},
    "link": {{"type": "string", "nullable": true, "description": "an NVMe's PCIe link now: PCIe 4.0 x4"}}, "link_gb_s": {{"type": "number", "nullable": true}},
    "device": {{"type": "string"}}, "fstype": {{"type": "string"}}, "mount": {{"type": "string"}}, "folder": {{"type": "string", "description": "where models moved here go (a models folder set on it, else <mount>/nextsycl-models)"}},
    "size": {{"type": "integer"}}, "free": {{"type": "integer"}}, "margin": {{"type": "integer", "description": "bytes a move leaves free (1%, at least 4 GiB)"}},
    "models_bytes": {{"type": "integer", "description": "the registered models' files on it (a file once)"}}, "models": {{"type": "array", "items": {{"type": "string"}}}},
    "measured": {{"type": "object", "nullable": true, "properties": {{"mb_s": {{"type": "number", "description": "MB/s read past the page cache"}}, "bytes": {{"type": "integer"}}, "seconds": {{"type": "number"}}, "at": {{"type": "number", "description": "unix seconds"}}, "file": {{"type": "string"}}}}}},
    "measuring": {{"type": "boolean"}}, "rate_gb_s": {{"type": "number", "nullable": true, "description": "as measured, else its link's"}}}}}}}},
  "models": {{"type": "object", "description": "model id: its files' place", "additionalProperties": {{"type": "object", "properties": {{
    "bytes": {{"type": "integer"}}, "files": {{"type": "integer"}}, "missing": {{"type": "integer"}}, "drives": {{"type": "object", "additionalProperties": {{"type": "integer"}}, "description": "drive id: its bytes there"}},
    "streams": {{"type": "boolean", "description": "reads its weights from the disk while it serves (its model card's streams): its drive's speed is its speed"}},
    "moving": {{"type": "boolean"}}, "move": {{"type": "object", "description": "each drive it could move to: {{need: bytes to write}}, with why when it cannot now", "additionalProperties": {{"type": "object", "properties": {{"need": {{"type": "integer"}}, "why": {{"type": "string"}}}}}}}}}}}}}},
  "move": {{"nullable": true, "allOf": [{MOVE}]}}, "measuring": {{"type": "string", "nullable": true}}, "folders": {{"type": "array", "items": {{"type": "string"}}, "description": "the models folders set (NS_MODELS, NS_MODEL_DIRS)"}}}}}}"#),
        "system.drive.measure" => r#"{"type": "object", "properties": {"drive": {"type": "string"}, "measuring": {"type": "boolean"}, "reads_up_to": {"type": "integer"}}}"#.into(),
        "system.move" => format!(r#"{{"type": "object", "properties": {{"move": {MOVE}}}}}"#),
        "system.move.cancel" => r#"{"type": "object", "properties": {"model": {"type": "string"}, "cancelling": {"type": "boolean"}}}"#.into(),
        "system.maintenance" => r#"{"type": "object", "required": ["on"], "properties": {
  "on": {"type": "boolean"},
  "maintenance": {"type": "object", "nullable": true, "description": "the notice while on", "properties": {"message": {"type": "string"}, "notice": {"type": "string", "description": "the sentence refused callers read"}, "since": {"type": "number"}, "until": {"type": "number", "nullable": true}, "retry_after_s": {"type": "integer"}}},
  "bypass": {"type": "string", "description": "turning it on: the X-Maintenance-Bypass key that lets a request through"},
  "unloading": {"type": "array", "items": {"type": "string"}, "description": "models being unloaded (unload: true)"},
  "draining": {"type": "array", "items": {"type": "string"}, "description": "work still running, as system.busy's reasons"}}}"#.into(),
        "llm.chat" => r#"{"type": "object", "description": "OpenAI's chat.completion", "properties": {"id": {"type": "string"}, "object": {"type": "string"}, "created": {"type": "integer"}, "model": {"type": "string"},
  "choices": {"type": "array", "items": {"type": "object", "properties": {"index": {"type": "integer"}, "finish_reason": {"type": "string"},
    "message": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string", "nullable": true}, "reasoning_content": {"type": "string", "nullable": true}, "tool_calls": {"type": "array", "items": {"type": "object"}}}}}}},
  "usage": {"type": "object", "properties": {"prompt_tokens": {"type": "integer"}, "completion_tokens": {"type": "integer"}, "total_tokens": {"type": "integer"}}}}, "additionalProperties": true}"#.into(),
        "llm.requests" => format!(r#"{{"type": "object", "properties": {{"active": {{"type": "array", "items": {CHAT_REQUEST}}}, "done": {{"type": "array", "description": "the last 10", "items": {CHAT_REQUEST}}}}}}}"#),
        "llm.models" => r#"{"type": "object", "properties": {"models": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "gpus": {"type": "array", "items": {"type": "integer"}}, "loaded": {"type": "boolean"}}}}}}"#.into(),
        "image.generate" | "image.edit" => format!(r#"{{"oneOf": [{{"allOf": [{PICTURES}, {{"type": "object", "properties": {{"job": {JOB_VIEW}}}}}], "description": "wait (default): the pictures, and the job"}},
  {{"allOf": [{JOB_VIEW}], "description": "wait: false - the job at once (image.status {{label}} for the rest)"}}]}}"#),
        "model3d.generate" | "model3d.rig" => format!(r#"{{"oneOf": [{{"allOf": [{}, {{"type": "object", "properties": {{"job": {JOB_VIEW}}}}}], "description": "wait (default): the mesh, and the job"}},
  {{"allOf": [{JOB_VIEW}], "description": "wait: false - the job at once (model3d.status {{label}} for the rest)"}}]}}"#, meshes()),
        "model3d.tool" => format!(r#"{{"type": "object", "properties": {{"created": {{"type": "integer"}}, "data": {{"type": "array", "items": {MESH_ITEM}}}}}}}"#),
        "model3d.history" => format!(r#"{{"type": "object", "properties": {{"data": {{"type": "array", "description": "newest first (200 at most); url server-relative (/v1/model3d/files/<file>, the same route here)", "items": {MESH_ITEM}}}}}}}"#),
        "model3d.progress" => r#"{"type": "object", "properties": {"busy": {"type": "boolean"}, "waiting": {"type": "integer"}, "loaded": {"type": "boolean"}, "prompt": {"type": "string", "description": "what it works on: N picture(s), or the motion"},
  "phase": {"type": "string", "description": "image, structure, shape, texture, decode, mesh, bake, skeleton, skin, motion"}, "at": {"type": "integer"}, "of": {"type": "integer"}, "seconds": {"type": "number"}}}"#.into(),
        "image.status" | "audio.status" | "model3d.status" => JOB_VIEW.into(),
        "image.jobs" | "audio.jobs" | "model3d.jobs" => JOB_LIST.into(),
        "image.queue.list" | "audio.queue.list" | "model3d.queue.list" => format!(r#"{{"type": "object", "properties": {{"queue": {{"type": "array", "items": {JOB_VIEW}, "description": "in run order"}},
  "paused": {{"type": "object", "additionalProperties": true}}}}}}"#),
        "image.queue.move" | "audio.queue.move" | "model3d.queue.move" => r#"{"type": "object", "properties": {"label": {"type": "string"}, "position": {"type": "integer"}, "queue": {"type": "array", "items": {"type": "string"}, "description": "the waiting labels in run order"}}}"#.into(),
        "image.queue.pause" | "image.queue.resume" | "audio.queue.pause" | "audio.queue.resume" | "model3d.queue.pause" | "model3d.queue.resume" => r#"{"type": "object", "properties": {"paused": {"type": "string", "description": "pause: queue | project <p> | batch <b> | N job(s)"},
  "resumed": {"type": "string", "description": "resume: the same"}, "queued": {"type": "integer"}, "held": {"type": "integer"}, "paused_state": {"type": "object", "additionalProperties": true}}}"#.into(),
        "image.upload" | "audio.upload" | "model3d.upload" => r#"{"type": "object", "properties": {"name": {"type": "string", "description": "up_<unix>_<name> (audio: .wav)"}, "ref": {"type": "string", "description": "upload:<name>: what image / images / ref_audio / sample take in place of base64"},
  "bytes": {"type": "integer"}, "sha256": {"type": "string"}, "mtime": {"type": "number"}}}"#.into(),
        "system.cards" => r#"{"type": "object", "properties": {"cards": {"type": "array", "items": {"type": "object", "properties": {
  "gpu": {"type": "integer"}, "name": {"type": "string"}, "bus": {"type": "string", "description": "its PCI address"},
  "vram_total_mb": {"type": "integer"}, "vram_used_mb": {"type": "integer"}, "vram_free_mb": {"type": "integer"},
  "busy_pct": {"type": "number"}, "power_w": {"type": "number", "nullable": true}, "power_cap_w": {"type": "number", "nullable": true}, "temp_c": {"type": "number", "nullable": true}, "temp_vram_c": {"type": "number", "nullable": true},
  "models": {"type": "array", "items": {"type": "string"}, "description": "registered models whose processes hold memory there"},
  "processes": {"type": "array", "items": {"type": "object", "properties": {"pid": {"type": "integer"}, "program": {"type": "string"}, "kind": {"type": "string", "nullable": true}, "model": {"type": "string", "nullable": true}, "vram_mb": {"type": "integer"}}}},
  "enabled_models": {"type": "array", "items": {"type": "string"}, "description": "the enabled models counted on it"}, "vouched_mb": {"type": "integer", "description": "what they are counted for"},
  "overbooked": {"type": "boolean", "description": "they are counted for more than it holds (the first to load holds it)"}}}}}}"#.into(),
        "image.cancel" | "audio.cancel" | "model3d.cancel" => r#"{"type": "object", "properties": {"cancelled": {"description": "label: how many jobs; none: whether anything was stopped"}, "stopped": {"type": "integer", "description": "requests stopped while running"},
  "label": {"type": "string"}, "reentrant_id": {"type": "string", "nullable": true}, "jobs": {"type": "array", "items": {"type": "string"}},
  "already_ended": {"type": "boolean", "description": "the job named had ended before the cancel (nothing stopped; state says how it ended) - a cancel is safe to repeat"},
  "state": {"type": "string", "description": "already_ended: done, failed or cancelled"}}}"#.into(),
        "image.ocr" => r#"{"type": "object", "properties": {"created": {"type": "integer"}, "model": {"type": "string"}, "text": {"type": "string", "description": "the reading: Markdown, tables as HTML, LaTeX; pages after a <PAGE> mark each"},
  "prompt": {"type": "string", "description": "the instruction used"}, "pages": {"type": "integer"}, "tokens": {"type": "integer", "description": "tokens written"}, "prompt_tokens": {"type": "integer"},
  "prompt_seconds": {"type": "number"}, "read_tok_s": {"type": "number"}, "write_seconds": {"type": "number"}, "tok_s": {"type": "number"}, "seconds": {"type": "number"}, "wh": {"type": "number", "nullable": true}}}"#.into(),
        "image.info" | "audio.info" | "model3d.info" => r#"{"type": "object", "properties": {"loaded": {"type": "boolean"},
  "info": {"type": "object", "description": "loaded: the server's own - model, arch, defaults, samplers, schedules, options, loras; image: edits, reads (an OCR model), prompts (its instructions), max_pictures 8, max_pages 24; audio: lyrics, speech (voices, languages, design, clone ...)", "additionalProperties": true},
  "model": {"type": "string"}, "title": {"type": "string", "description": "not loaded"}, "defaults": {"type": "object", "additionalProperties": true, "description": "not loaded: the registry's"}, "note": {"type": "string"},
  "steps": {"type": "object", "description": "image, loaded or not: the steps a request may send", "properties": {"default": {"type": "integer"}, "locked": {"type": "boolean", "description": "true: a few-step model (turbo) runs only allowed"},
    "allowed": {"type": "array", "nullable": true, "items": {"type": "integer"}, "description": "locked: the only count (6 for turbo); null: any from min to max"}, "min": {"type": "integer"}, "max": {"type": "integer"}, "note": {"type": "string"}}},
  "max_pictures": {"type": "integer", "description": "image.edit: pictures a request (8)"}, "max_picture_pixels": {"type": "integer", "description": "each picture sent: at most this many pixels (40 MP; it is resized to about 1 MP anyway, so 2048 px on the long side is plenty)"},
  "max_request_bytes": {"type": "integer", "description": "the whole request body (256 MiB: about 190 MiB of pictures as base64)"}, "size_multiple": {"type": "integer", "description": "size: both sides a multiple of this"}}}"#.into(),
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
        "audio.voices.enroll" => r#"{"type": "object", "description": "wait: false - the job (as audio.status answers it, voice: the saved voice); else when it ends:", "properties": {
  "voice": {"type": "string"}, "model": {"type": "string", "description": "the model it is enrolled for"}, "arch": {"type": "string"},
  "gpu": {"type": "integer", "description": "the card the fit ran on"}, "steps": {"type": "integer"}, "seconds": {"type": "number"},
  "enrolled": {"type": "array", "items": {"type": "string"}, "description": "the architectures the voice is enrolled for now"},
  "already": {"type": "boolean", "description": "it was enrolled before (replace: true fits it again)"},
  "report": {"type": "object", "description": "the fit's report: its likeness and loss"}}}"#.into(),
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
    "not": {"type": "array", "items": {"type": "string"}, "description": "refused with this engine (a 400)"}, "chosen": {"type": "string", "description": "when it is used"},
    "ignored": {"type": "array", "items": {"type": "string"}, "description": "accepted but read by nothing (negative_prompt, cfg: H3 runs one pass a step at cfg 1; chain.pin_renorm): echoed back in video.generate's ignored"}}}}}}}}"#.into(),
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
        "system.load" => &[(404, "not_found", "no such model"), (503, "no_room", "its cards are held by other models now (retryable)")],
        "system.enable" | "system.disable" | "system.unload" | "system.gpus" => &[(404, "not_found", "no such model"), (409, "conflict", "enable: the model's files are moving (system.move)")],
        "system.drive.measure" => &[(404, "not_found", "no such drive (system.drives)"), (409, "busy", "a model loads from it, a move uses it, or another drive is being measured")],
        "system.move" => &[(400, "invalid_params", "its files are on that drive already, missing, or one would land on another file"), (404, "not_found", "no such model or drive"),
                           (409, "conflict", "the model is loaded or loading (stop it first), a process has its files open, or the drive lacks the room"),
                           (409, "busy", "another move runs, or a drive is being measured")],
        "system.move.cancel" => &[(409, "conflict", "no move is under way")],
        "image.generate" | "image.edit" | "image.ocr" | "audio.generate" | "audio.speak" | "audio.voices.design" | "audio.voices.enroll" | "model3d.generate" | "model3d.rig" =>
            &[(400, "invalid_params", "a value refused, when the job is submitted (wait false too): steps a few-step model does not run, a size, a sampler, a picture that does not decode, a voice or language the speech model does not have, a priority outside -9..9 ..."),
              (404, "not_found", "no such model; a continue_from or upload:<name> that is not here (error.field names which)"), (409, "conflict", "the model is disabled, or none that can do it is enabled (never retry as is)"),
              (409, "busy", "queue: false and the job would not start at once (retryable)"),
              (499, "cancelled", "stopped by <kind>.cancel"), (502, "unavailable", "the kind's server did not start or answer"),
              (503, "no_room", "the model's cards have no room for it (other work that is not idle holds them; idle models there are stopped for it at once). A job waits for the room up to 30 minutes first - stage waiting, waiting_for says who holds the card - then fails with this; image.ocr and voices.design wait 90 s. Retryable"),
              (504, "timeout", "wait: still running after an hour (the job goes on: <kind>.status {label})")],
        "llm.chat" => &[(400, "invalid_params", "the chat server refused the request"), (502, "unavailable", "the chat server does not answer"),
                        (503, "no_room", "the model is not loaded and its cards have no room for it (work that is not idle holds them; idle picture, sound and video work is stopped for it at once): retryable")],
        "audio.voices.add" => &[(400, "invalid_params", "a bad name, a recording shorter than 3 s or not decodable"), (409, "conflict", "replace over a voice another owner has")],
        "audio.voices.remove" => &[(404, "not_found", "no saved voice of that name"), (409, "conflict", "the voice has an owner and the request names another")],
        "image.status" | "audio.status" | "model3d.status" => &[(400, "invalid_params", "neither label nor reentrant_id (<kind>.jobs lists them)"), (404, "not_found", "no job by that label or reentrant_id")],
        "image.queue.move" | "audio.queue.move" | "image.queue.pause" | "image.queue.resume" | "audio.queue.pause" | "audio.queue.resume"
        | "model3d.queue.move" | "model3d.queue.pause" | "model3d.queue.resume" => &[(404, "not_found", "label / reentrant_id: no waiting job by it")],
        "model3d.tool" => &[(400, "invalid_params", "an op it does not know, faces missing for decimate, or a mesh that does not read"), (404, "not_found", "file: no GLB of that name here")],
        "image.upload" | "audio.upload" | "model3d.upload" => &[(400, "invalid_params", "a name of another type, data not base64, empty, or not a picture / recording / mesh that decodes (a picture over 40 MP)")],
        "image.cancel" | "audio.cancel" | "model3d.cancel" => &[(404, "not_found", "no job with that label or reentrant_id at all (the message names which); an ended one answers 200 already_ended")],
        "system.files.stat" => &[(400, "invalid_params", "kind is not image, audio, video or model3d, or name is not a plain file name"), (404, "not_found", "no such file")],
        "system.files.delete" => &[(400, "invalid_params", "kind, or a name that is not a clip, upload, picture or sound"), (404, "not_found", "no such file"),
                                   (409, "busy", "video: the clip is being made (video.cancel first)")],
        _ => &[],
    }
}

/// The error every method that starts work may give besides its own (maint.rs's starts_work)
pub const MAINTENANCE: (u16, &str, &str) = (503, "maintenance", "maintenance mode is on (system.maintenance): refused with the notice (error.maintenance) and error.retry_after_s; retryable");

/// What every answer's error code means
pub const CODES: &str = "Every /rpc answer is the envelope: {\"ok\": true, \"result\": ...} or {\"ok\": false, \"error\": {code, message, retryable, status, method, field}}, \
always with HTTP 200 - errors too (401, an unknown method, a body too large to read). Read ok and error.code: error.status is the code's number by convention, \
in the body so another transport carries the same envelope. Only the file routes (/v1/.../files/..., the OpenAI routes) answer with HTTP statuses. \
A job's failure (image/audio status, a failed submit replayed) has the same error object. \
Error codes (error.code, error.status by convention): invalid_request 400 (not POST, a query string, the body is not a JSON object), request_too_large 400 (over 256 MiB), invalid_params 400, \
unauthorized 401, not_found 404 (also an unknown method), conflict 409 (a model disabled, none of the kind enabled, a name taken), \
busy 409 (video.generate while a clip is made, or an image / audio job that would wait, without queue: true), not_running 409 (video.cancel with nothing running), \
cancelled 499 (image.cancel / audio.cancel stopped it), unavailable 502 (the kind's server does not answer), \
no_room 503 (the model's cards are held by work that is not idle; idle models are stopped for it first), \
maintenance 503 (system.maintenance is on: what starts work - chat, generation, edits, speech, system.load / enable - is refused; error.maintenance has the notice, \
error.retry_after_s when to try again; status, listings and cancels answer as always), timeout 504, internal 500. \
Every error has retryable: true for no_room, maintenance, unavailable, timeout and video's busy (the same call may work later), false otherwise \
(change the request: a bad value, a disabled or missing model, a cancel).";

/// The label convention the video methods share
pub const LABELS: &str = "Video labels: \"<series> NN/MM: text\" (\"Harbour 03/12: the lamp\"). NN orders video.timeline (the edit order); \
the series (the text before NN) is where the anchors prev / prev_cam / first / first_cam look, and the project when the request names none. \
video.cancel, video.queue.move and video.queue.remove name clips by label.\n\n\
Jobs: image.generate, image.edit, audio.generate and audio.speak run as jobs here - the call can drop and the job goes on; \
label one (else its id is its label), read it with <kind>.status {label} (stage and waiting_for say what it waits for), list them with <kind>.jobs, \
stop it with <kind>.cancel {label}; wait: false answers at once. A kind makes as many jobs at once as it has servers (one a card); the rest wait \
in a queue as video's: priority, held, project and batch on submit, <kind>.queue.list / move / pause / resume. \
Pictures and recordings used again (an init picture, edit references, a voice sample) can be sent once with <kind>.upload and named as \
\"upload:<name>\" after. \
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
