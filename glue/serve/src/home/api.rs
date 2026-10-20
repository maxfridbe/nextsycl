//! The API on :8000.
//!
//! - `POST /rpc/<kind>.<method>`: H3's rules - POST only, one fixed route a method, every variable in the JSON body (no
//!   path variables, no query strings); the answer `{"ok": true, "result": ...}` or `{"ok": false, "error": {"code",
//!   "message", "retryable", "status", "method", "field"}}` - the body says it all: every /rpc answer is HTTP 200,
//!   errors too (error.status carries the number by convention), so another transport can carry the same envelope;
//!   only the file routes (/v1/.../files/...) use HTTP statuses. Kinds: `system` (the box: models, enabling, loading), `llm`, `image`, `audio`,
//!   `video` (the studio's methods, `video.generate` ...). Every kind but system has `describePromptGuideMD {model}`.
//! - The OpenAI-compatible routes (`/v1/models`, `/v1/chat/completions`, `/v1/images/generations`, `/v1/images/edits`,
//!   `/v1/audio/speech` and the files they link) - passed to the switcher or the kind's server, which is started for
//!   the model the request names first.
//! - Both want `Authorization: Bearer <token>`. `GET /openapi.yml` (open) describes the methods of the kinds that have
//!   an enabled model.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use super::life::s;
use super::{now, Home};
use crate::http::{self, call_for, Conn, Request};
use super::registry;
use super::spec;

/// One method: its kind and name, what it does, its body's fields (name, JSON type, description, required) and an
/// example body
pub struct Method {
    pub kind: &'static str,
    pub name: &'static str,
    pub summary: &'static str,
    pub fields: &'static [(&'static str, &'static str, &'static str, bool)],
    pub example: &'static str,
}

const MODEL: (&str, &str, &str, bool) = ("model", "string", "the model's registry id (system.models)", true);
const MODEL_OPT: (&str, &str, &str, bool) = ("model", "string", "the model's id; none: the loaded one, else the first enabled", false);

/// A job's place in its kind's queue (image and audio, as video.generate's)
const QUEUE: (&str, &str, &str, bool) = ("queue", r#"{"type": "boolean", "default": true}"#, "false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn", false);
const PRIORITY: (&str, &str, &str, bool) = ("priority", r#"{"type": "integer", "minimum": -9, "maximum": 9, "default": 0}"#, "a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)", false);
/// An image request's own LoRAs
const LORAS: (&str, &str, &str, bool) = ("loras", r#"{"type": "array", "items": {"oneOf": [{"type": "string", "description": "name:scale"}, {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}, "scale": {"type": "number", "default": 1}}}]}}"#,
    "LoRAs for this request (a registered LoRA's id - system.models kind lora - or a file), each with its strength: Qwen-Image 2.1 runs them beside the loaded weights for this request only (no reload; each file read once and kept, a few-step LoRA's sigmas applied); other models reload with them merged", false);
const HELD: (&str, &str, &str, bool) = ("held", r#"{"type": "boolean", "default": false}"#, "queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch", false);
const PROJECT: (&str, &str, &str, bool) = ("project", r#"{"type": "string", "maxLength": 60}"#, "a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}", false);
const BATCH: (&str, &str, &str, bool) = ("batch", r#"{"type": "string", "maxLength": 60}"#, "a group inside the project", false);
const JOB_ID: [(&str, &str, &str, bool); 2] = [("label", "string", "the job's label (looked up first)", false),
                                               ("reentrant_id", "string", "or the reentrant_id it was submitted with (used when label is absent or empty)", false)];

pub const METHODS: &[Method] = &[
    Method { kind: "system", name: "state", summary: "The box: the GPUs (memory by process and vouched for), every model's state, the front ends", fields: &[], example: "{}" },
    Method { kind: "system", name: "busy", summary: "Is anything working: a clip made or waiting to run, a picture, sound or chat answered, a model loading. video.accepts says whether video.generate starts a clip at once (else 409 busy without queue: true)",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "version", summary: "The release, the API's version, the kinds enabled, the features this build has (names a client can test for) and the methods callable now",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "files.stat", summary: "A file an answer linked: its bytes and SHA-256 (to check a download), mtime, URL",
             fields: &[("kind", r#"{"type": "string", "enum": ["image", "audio", "video", "model3d"]}"#, "", true), ("name", "string", "the file's name (the last part of its URL)", true)],
             example: r#"{"kind": "video", "name": "h3_20261011_220723.mp4"}"# },
    Method { kind: "system", name: "files.delete", summary: "Delete a file: a picture, a song or speech, a clip (with its record, last frame, latents and sound tail; not while it is made) or an upload (ref_...)",
             fields: &[("kind", r#"{"type": "string", "enum": ["image", "audio", "video", "model3d"]}"#, "", true), ("name", "string", "the file's name; a clip: h3_... with or without .mp4", true)],
             example: r#"{"kind": "video", "name": "h3_20261011_220723"}"# },
    Method { kind: "system", name: "models", summary: "Every registered model: kind, enabled, its GPUs, its state (disabled, unloaded, loading, loaded, busy, ready), what it can do (capabilities, methods); downloaded: true. Then every model a card offers that is not downloaded: downloaded: false, state not_downloaded, enabled false, bytes, the drive and folder it would go to, fits, its download under way (pull) - system.models.pull fetches one",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "models.pull", summary: "Download a model a card offers (system.models: downloaded false), in the background: its files fetched (or linked when the machine has them; a partial file resumed), each SHA-256 checked, then the model registered and enabled. Into the first models folder with the room (NS_MODELS, then NS_MODEL_DIRS, then each drive's); one download at a time a folder (a second waits, queued). Answers at once with the download (system.models.pulls follows it)",
             fields: &[("model", "string", "the card's id (system.models, downloaded false)", true)], example: r#"{"model": "qwen-image-2.1-q8"}"# },
    Method { kind: "system", name: "models.pulls", summary: "The downloads under way and the last ones, newest first: model, state (queued, downloading, verifying, done, failed, cancelled), bytes_done, bytes_total, pct, speed (bytes a second), eta_s, drive, folder, error",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "models.pull.cancel", summary: "Stop a model's download: curl stopped, what it fetched kept (a pull again resumes it), nothing registered",
             fields: &[("model", "string", "the model being downloaded", true)], example: r#"{"model": "qwen-image-2.1-q8"}"# },
    Method { kind: "system", name: "cards", summary: "The GPUs, one neutral list: name, memory (total, used, free, and what the enabled models are counted for), load, power, temperature, the models and processes on each",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "enable", summary: "Offer a model on GPU(s): it loads on its first request and unloads when idle; chat models join Open WebUI's list",
             fields: &[MODEL, ("gpus", "integer[]", "the cards (default: as registered); several for a model that spans cards", false)],
             example: r#"{"model": "qwen-image-2.1-q8", "gpus": [0]}"# },
    Method { kind: "system", name: "gpus", summary: "Change the cards a model may use (enabled or not); more than fit is allowed (the card shows overbooked). A loaded model moved off its card unloads: its next request loads it on the new cards",
             fields: &[MODEL, ("gpus", "integer[]", "the cards: one or more (an image or audio model loads on the first of them with room)", true)],
             example: r#"{"model": "qwen3.8-27b-q4", "gpus": [0, 1]}"# },
    Method { kind: "system", name: "disable", summary: "Withdraw a model: unloaded (after the requests it is answering), off Open WebUI's list and this spec",
             fields: &[MODEL, ("force", "boolean", "a video model while a clip renders: stop the clip at its next step", false)], example: r#"{"model": "qwen-image-2.1-q8"}"# },
    Method { kind: "system", name: "load", summary: "Load an enabled model now (otherwise its first request does)",
             fields: &[MODEL, ("wait", "boolean", "answer when it has loaded (default: at once, loading in the background)", false)],
             example: r#"{"model": "minimax-music3", "wait": true}"# },
    Method { kind: "system", name: "unload", summary: "Unload a model now (it stays enabled)", fields: &[MODEL], example: r#"{"model": "minimax-music3"}"# },
    Method { kind: "system", name: "config", summary: "The settings file (nextsycl.conf): its text, every setting the program reads (value, default, what reads it), the file's others",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "configSet", summary: "Change settings (the file copied first): set {NAME: value, NAME: null to remove} or text (the whole file); answers what changed and what must restart",
             fields: &[("set", "object", "{NAME: value | null}", false), ("text", "string", "the whole file instead", false)],
             example: r#"{"set": {"NS_IDLE_MINUTES": "15"}}"# },
    Method { kind: "system", name: "restart", summary: "Restart what reads the settings: serve, switch, studio, video, image, audio or llm",
             fields: &[("service", "string", "serve | switch | studio | video | image | audio | llm", true)], example: r#"{"service": "switch"}"# },
    Method { kind: "system", name: "maintenance", summary: "Maintenance mode: the box set aside for development. On, what would start work (llm.chat, image/audio/video generation, edits, speech, system.load / enable, the OpenAI POST routes) is refused with error code maintenance (retryable, retry_after_s, the notice in error.maintenance; /v1: HTTP 503 + Retry-After); status, listings, cancels and guides answer as always; queued image and audio jobs wait, the video queue pauses, running work finishes; system.busy / version / state carry the notice, and Open WebUI's chats are answered with it. The answer to turning it on carries a bypass key: a request with the header X-Maintenance-Bypass: <key> goes through (a developer testing meanwhile). Without on: the current state",
             fields: &[("on", "boolean", "true: on, false: off; absent: just the state", false),
                       ("message", "string", "what callers are told (kernel work on the B70, say)", false),
                       ("eta_s", "number", "about how long, in seconds: retry_after_s and the notice's end", false),
                       ("unload", "boolean", "true: unload the loaded models not answering now (frees the cards)", false)],
             example: r#"{"on": true, "message": "kernel work", "eta_s": 3600, "unload": true}"# },
    Method { kind: "system", name: "drives", summary: "The drives the models are on: device, PCIe link, mount, size, free space, the models' bytes, the read speed as measured (and when); each model's bytes by drive, whether it streams its weights from the disk while it serves, and for each other drive what a move needs or why it cannot move now; the move under way (or the last)",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "drive.measure", summary: "Measure a drive's read speed in the background (up to 4 GiB of its largest model files read with O_DIRECT, past the page cache; nothing written), kept with its date (system.drives). Refused while a model loads from it or a move uses it",
             fields: &[("drive", "string", "the drive's id (nvme1n1) or its mount point (system.drives)", true)], example: r#"{"drive": "nvme1n1"}"# },
    Method { kind: "system", name: "move", summary: "Move a model's files to another drive, as a job (its progress in system.drives' move): each file copied and read back (size, SHA-256), then the registry repointed, then the old files removed - those another model names stay. Refused while the model is loaded or loading (stop it first), another move runs, or the drive lacks the room (plus 1%, at least 4 GiB); nothing loads the model while it moves",
             fields: &[MODEL, ("drive", "string", "the drive's id (nvme1n1) or its mount point (system.drives)", true)], example: r#"{"model": "olmocr-2-7b-q8", "drive": "nvme1n1"}"# },
    Method { kind: "system", name: "move.cancel", summary: "Stop the move under way: what it copied is removed and the registry stays as it was (once the registry is repointed it finishes)",
             fields: &[], example: "{}" },

    Method { kind: "llm", name: "chat", summary: "A chat completion (OpenAI's body; stream is ignored - use /v1/chat/completions to stream)",
             fields: &[("model", "string", "a chat model's id", true), ("messages", "object[]", "[{role, content}]", true),
                       ("max_tokens", "integer", "", false), ("temperature", "number", "", false), ("tools", "object[]", "OpenAI tool definitions", false),
                       ("keep_up_s", r#"{"type": "number", "minimum": 0, "maximum": 86400}"#, "keep the model loaded this long after the answer: picture, sound and video work does not stop it for room meanwhile (as /v1/chat/completions' X-Keep-Up-S header, which Open WebUI sends: 300). Absent: nothing kept - once idle the model may be stopped for other work", false)],
             example: r#"{"model": "qwen3.8-flash-next-iq2_xs", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 200}"# },
    Method { kind: "llm", name: "requests", summary: "The chat requests running and the last ones that ended, with their speeds as the server reports them live: prompt tokens (reused from the cache), read seconds, tokens generated, tok/s",
             fields: &[], example: "{}" },
    Method { kind: "llm", name: "models", summary: "The enabled chat models (what Open WebUI lists) and which is loaded", fields: &[], example: "{}" },
    Method { kind: "llm", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "qwen3.8-flash-next-iq2_xs"}"# },

    Method { kind: "image", name: "generate", summary: "Pictures from a prompt, or from an init picture (image + strength: img2img); the model loads first if it must. Checked when submitted (a bad value is a 400 even with wait: false)",
             fields: &[MODEL_OPT, ("prompt", "string", "", true), ("size", "string", "WxH, e.g. 1024x1024, both multiples of 16 (default: the model's, or the init picture's aspect)", false), ("n", "integer", "pictures", false),
                       ("image", "string", "an init picture (base64, a data: URL, or upload:<name> from image.upload; at most 40 MP): a plain img2img - the denoise starts from it at strength and the prompt guides it (it is no reference: for an edit that reads the picture use image.edit). Needs strength below 1", false),
                       ("strength", r#"{"type": "number", "minimum": 0.05, "maximum": 1}"#, "with image: how far from it (ComfyUI's denoise: the last steps of a round(steps / strength)-step schedule); 0.3-0.6 is the usual img2img range", false),
                       ("steps", "integer", "image.info's steps: a few-step model (turbo) runs only its own count (6)", false), ("seed", "integer", "", false), ("sampler", "string", "ComfyUI's names (euler, euler_ancestral, dpmpp_2m ...)", false),
                       ("schedule", "string", "", false), ("cfg", "number", "true guidance (needs negative_prompt)", false), ("negative_prompt", "string", "", false),
                       LORAS, ("response_format", "string", "url (default) or b64_json", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false),
                       QUEUE, PRIORITY, HELD, PROJECT, BATCH],
             example: r#"{"model": "qwen-image-2.1-q8", "prompt": "a lighthouse at dusk, oil painting", "size": "1024x768"}"# },
    Method { kind: "image", name: "edit", summary: "Edit or compose pictures: the first is changed, the others are references the prompt names. With reference: false and strength below 1 it is a plain img2img (as image.generate's image). Checked when submitted (a bad value is a 400 even with wait: false)",
             fields: &[MODEL_OPT, ("prompt", "string", "the instructions", true), ("images", "string[]", "up to 8, base64, data: URLs or upload:<name> (image.upload) - PNG, JPEG, WebP; each at most 40 MP, the request at most 256 MiB; each is resized to about 1 MP, so 2048 px on the long side is plenty", true),
                       ("reference", r#"{"type": "boolean", "default": true}"#, "true (default): the pictures condition the model as an edit's references (its vision tower reads them) as well as being where strength starts; false: a plain img2img - the denoise starts from the first picture and the prompt alone guides it, comparable with other tools' img2img (needs strength below 1)", false),
                       ("size", "string", "default: the last picture's aspect", false), ("steps", "integer", "", false), ("seed", "integer", "", false),
                       ("strength", r#"{"type": "number", "minimum": 0.05, "maximum": 1, "default": 1}"#, "how far from the first picture: 1 (default) makes a new picture guided by it; below 1 starts from the first picture, encoded and noised to the level strength reaches: the run is the last steps of a round(steps / strength)-step schedule, exactly ComfyUI's KSampler denoise (a few-step model such as turbo starts at strength itself, then takes its own levels below it). With reference true the picture also conditions the model, which holds it closer than a plain img2img at the same number: measured on Qwen-Image 2.1, 0.9 keeps the framing, face, hair and clothes and still makes the change asked; 0.75 and below give back nearly the same picture. With reference false it behaves as other tools' img2img. The first picture is resized to the output's size", false),
                       LORAS, ("response_format", "string", "url (default) or b64_json", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false),
                       QUEUE, PRIORITY, HELD, PROJECT, BATCH],
             example: r#"{"prompt": "make it night, keep the lighthouse", "images": ["data:image/png;base64,iVBORw0..."]}"# },
    Method { kind: "image", name: "ocr", summary: "Read pages (an OCR model: Unlimited-OCR): their text as Markdown, tables as HTML, boxes with <|grounding|>",
             fields: &[("model", "string", "an OCR model's id (default: the first enabled)", false), ("images", "string[]", "the pages, base64, data: URLs or upload:<name> (image.upload), in order: pictures, or PDFs (split into their pages, 150 dpi)", true),
                       ("prompt", "string", "\"document parsing.\" (default), \"Free OCR.\", \"<|grounding|>Convert the document to markdown.\", \"Parse the figure.\"; several pages: each page under it, or (each_page false) the model's \"Multi page parsing.\"; a <PAGE> mark before each page", false),
                       ("tiles", "boolean", "one page: read its 640-pixel tiles too (default true)", false),
                       ("figures", "boolean", "read the text inside the pictures the model marks (a second pass each; default true)", false),
                       ("columns", "boolean", "find blocks set side by side (a gutter between) and read each on its own (default true)", false),
                       ("each_page", "boolean", "several pages: each read on its own with its tiles (default true); false: the model's one-pass multi-page reading", false), ("max_tokens", "integer", "default 8192", false),
                       ("temperature", "number", "0 (default): greedy", false), ("ngram", "integer", "no n-gram of this many tokens repeated (35; 0: off)", false)],
             example: r#"{"images": ["data:image/png;base64,iVBORw0..."], "prompt": "document parsing."}"# },
    Method { kind: "image", name: "info", summary: "The model's defaults, samplers, schedules, options and LoRAs (loaded: from the server)", fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "image", name: "status", summary: "One job (image.generate / image.edit) by its label or reentrant_id (one of them is needed: image.jobs lists them): state, stage and what it waits for, its place in the queue, progress, its pictures when done",
             fields: &JOB_ID, example: r#"{"label": "poster-7"}"# },
    Method { kind: "image", name: "jobs", summary: "The image jobs: running, waiting (in run order) and the last ended, each as image.status answers it",
             fields: &[("limit", r#"{"type": "integer", "minimum": 0, "maximum": 200, "default": 20}"#, "ended jobs listed", false), ("project", "string", "only this project's", false)], example: r#"{"limit": 5}"# },
    Method { kind: "image", name: "queue.list", summary: "The waiting image jobs in run order (held ones too: held, waiting_for)", fields: &[], example: "{}" },
    Method { kind: "image", name: "queue.move", summary: "Move a waiting job to a place in the queue (1 = made next)",
             fields: &[JOB_ID[0], JOB_ID[1], ("position", r#"{"type": "integer", "minimum": 1, "default": 1}"#, "", false),
                       ("priority", r#"{"type": "integer", "minimum": -9, "maximum": 9}"#, "give it this priority too (where later jobs go)", false)], example: r#"{"label": "poster-9", "position": 1}"# },
    Method { kind: "image", name: "queue.pause", summary: "Hold a waiting job (label or reentrant_id), a project or batch (also the ones submitted later), or (none) the whole image queue; running jobs go on",
             fields: &[JOB_ID[0], JOB_ID[1], ("project", "string", "", false), ("batch", "string", "", false)], example: r#"{"project": "Harbour"}"# },
    Method { kind: "image", name: "queue.resume", summary: "Release what queue.pause held (none: everything)",
             fields: &[JOB_ID[0], JOB_ID[1], ("project", "string", "", false), ("batch", "string", "", false)], example: "{}" },
    Method { kind: "image", name: "upload", summary: "Keep a picture here once (PNG, JPEG, WebP; at most 40 MP): image.generate's image, image.edit's and image.ocr's images then name it as \"upload:<name>\" instead of sending it again. Kept until system.files.delete {kind: image, name}",
             fields: &[("name", "string", "the file's name: png jpg jpeg webp; kept as up_<unix seconds>_<name> (letters, digits, . _ - kept, the rest _, 60 characters)", true),
                       ("data", "string", "the picture, base64 (a data: URL too)", true)],
             example: r#"{"name": "hero.png", "data": "iVBORw0KGgo..."}"# },
    Method { kind: "image", name: "cancel", summary: "Stop a job by its label or reentrant_id (running: between steps; waiting: at once; ended: ok with already_ended); neither: whatever the image servers are making",
             fields: &JOB_ID, example: r#"{"label": "poster-7"}"# },
    Method { kind: "image", name: "canvases.list", summary: "Sizes for image.generate: the model's tested ones with their aspect, and the time measured here per size",
             fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "image", name: "progress", summary: "The request running: step of steps, how many wait", fields: &[], example: "{}" },
    Method { kind: "image", name: "history", summary: "The pictures made, newest first, with their settings", fields: &[], example: "{}" },
    Method { kind: "image", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "qwen-image-2.1-q8"}"# },

    Method { kind: "audio", name: "generate", summary: "A song from a description and lyrics (the model loads first if it must); the answer links the WAV",
             fields: &[MODEL_OPT, ("instructions", "string", "the music: genre, mood, tempo, instruments, voice", true),
                       ("input", "string", "the lyrics, [verse] / [chorus] tags on lines of their own (empty: instrumental)", false),
                       ("seconds", "number", "", false), ("seed", "integer", "", false), ("steps", "integer", "", false), ("cfg", "number", "", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false),
                       QUEUE, PRIORITY, HELD, PROJECT, BATCH],
             example: r#"{"model": "minimax-music3", "instructions": "warm acoustic folk, 90 BPM, a soft male voice", "input": "[verse]\nRiver runs slow tonight\n[chorus]\nCarry me home", "seconds": 60}"# },
    Method { kind: "audio", name: "speak", summary: "Speech from text (a speech model: Qwen3-TTS, VoxCPM2, Breeze TTS 2, VibeVoice - a script of up to four speakers); the answer links the WAV. Checked when submitted (voice, language, continue_from, ref_audio: a 400 even with wait: false). Two run at once (one per card: system.busy.parallel), so a book can keep two chunks in flight with wait: false - but a chunk with continue_from needs the chunk before it finished",
             fields: &[MODEL_OPT, ("input", "string", "the text to say (Breeze TTS 2: vocal events inline - English (laugh) (sigh) (cough) (clears throat), Chinese [笑] [叹气] [咳嗽] [清嗓子]; VibeVoice: a script, \"Speaker 1: ...\\nSpeaker 2: ...\", up to four speakers)", true),
                       ("voice", "string", "a built-in voice (qwen3-tts-custom: serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan; audio.info lists them) or a saved one (audio.voices.list: spoken by a model that clones). Checked when submitted, the model loaded or not: an unknown name is a 400 at once (the message lists the voices)", false),
                       ("language", "string", "english, chinese, japanese ... or auto (default). Checked when submitted against the model's list (Qwen3-TTS: chinese, english, german, italian, portuguese, spanish, japanese, korean, french, russian); VoxCPM2 has no list and takes any", false),
                       ("instructions", "string", "how to say it (CustomVoice), or the voice to make up (VoiceDesign, Breeze TTS 2: write it in the language of the text); with a cloned voice (ref_audio or a saved voice) the direction of its tone, emotion, pace (Breeze TTS 2: voice direction - picked for you when model is not given)", false),
                       ("ref_audio", "string", "a recording of the voice to clone: WAV as base64 or a data: URL, or upload:<name> (audio.upload, any format it takes) - the Base model", false),
                       ("ref_text", "string", "what the recording says: the closer, in-context clone (Breeze TTS 2: required, the exact transcript)", false),
                       ("voices", "string[]", "a multi-speaker script's saved voices (audio.voices.list), one a speaker in order: Speaker 1's first (VibeVoice: up to four; picked for you when model is not given). Not with voice / ref_audio", false),
                       ("ref_audios", "string[]", "a multi-speaker script's recordings, one a speaker in order (WAV as base64, a data: URL, or upload:<name>); not with voices / voice / ref_audio", false),
                       ("continue_from", r#"{"type": "string", "maxLength": 200}"#, "carry on from speech made here (a book in chunks): the file name (or url) of the chunk before; its last sentences (up to 15 s, cut at a pause) are the in-context reference, after the saved voice's sample when voice names one - pace and voice go on, anchored to the voice. Needs a model that clones (picked for you when model is not given); a built-in voice name is dropped", false),
                       ("seconds", "number", "the most to make", false), ("seed", "integer", "", false),
                       ("cfg", "number", "guidance with instructions (Breeze TTS 2: 4 by default for design and direction; 1 turns it off)", false),
                       ("options", "object", "the engine's own: greedy, temperature, top-k, streaming", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false),
                       QUEUE, PRIORITY, HELD, PROJECT, BATCH],
             example: r#"{"model": "qwen3-tts-custom", "input": "Good morning. The bread is still warm.", "voice": "ryan", "language": "english", "instructions": "cheerful, a little hurried"}"# },
    Method { kind: "audio", name: "transcribe", summary: "Speech to text (a recogniser: Whisper large-v3-turbo, 99 languages): the text, its segments timed by the model's timestamps and (words, the default) each word timed by its cross-attention. A job like audio.speak (wait, reentrant_id, status, cancel); long recordings go in 30 s windows, so an hour takes minutes",
             fields: &[MODEL_OPT, ("audio", "string", "the recording: WAV, MP3, FLAC, Ogg or M4A as base64, a data: URL, or upload:<name> (audio.upload); any rate, mixed to mono", true),
                       ("language", "string", "a code or a name (en, english, de, german ...); none or auto: detected on the first 30 s", false),
                       ("words", r#"{"type": "boolean", "default": true}"#, "each word's start and end (and the model's probability for it)", false),
                       ("translate", "boolean", "into English instead of the language spoken", false),
                       ("prompt", "string", "text that leads the first window: names, spelling, style (Whisper's initial_prompt)", false),
                       ("options", "object", "the engine's own: condition (each window prompted with the text before it), no-speech (the silence threshold), fallback (decode an unsure window again, warmer)", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false),
                       QUEUE, PRIORITY, HELD, PROJECT, BATCH],
             example: r#"{"audio": "upload:up_1792400000_take2.wav", "language": "english", "words": true}"# },
    Method { kind: "audio", name: "voices.list", summary: "The saved voices (cloned or designed): name, owner, what the sample says, its description, length, whether it can be spoken now (ready) and by which enabled models; sample: its WAV",
             fields: &[("owner", "string", "only this owner's voices", false)], example: r#"{"owner": "film-app"}"# },
    Method { kind: "audio", name: "voices.add", summary: "Save a voice cloned from a recording (WAV, MP3, FLAC, Ogg, M4A; 3-15 s of one speaker is best)",
             fields: &[("name", "string", "lowercase letters, digits, - and _", true), ("sample", "string", "the recording, base64, a data: URL or upload:<name> (audio.upload)", true),
                       ("text", "string", "what the recording says (a closer clone)", false), ("description", "string", "", false),
                       ("language", "string", "", false), ("replace", "boolean", "over a voice of that name", false),
                       ("owner", r#"{"type": "string", "maxLength": 60}"#, "who it belongs to (an app or a person): voices.list filters by it, and replace / voices.remove of an owned voice must name it", false)],
             example: r#"{"name": "grandpa", "sample": "data:audio/mpeg;base64,SUQzBAAAAA...", "text": "Well, back in my day we walked to school."}"# },
    Method { kind: "audio", name: "voices.design", summary: "Make up a voice from a description (the voice-design model reads a sample) and save it",
             fields: &[("name", "string", "", true), ("instructions", "string", "the voice: age, gender, timbre, pace, emotion, accent", true),
                       ("text", "string", "what the sample says (default: a ten-second line in the language)", false), ("language", "string", "", false),
                       ("seed", "integer", "another seed, another speaker of that kind", false), ("model", "string", "a voice-design model", false),
                       ("replace", "boolean", "", false), ("owner", r#"{"type": "string", "maxLength": 60}"#, "who it belongs to (as voices.add)", false)],
             example: r#"{"name": "narrator", "instructions": "A deep, warm male narrator in his fifties, unhurried, a slight rasp.", "language": "english"}"# },
    Method { kind: "audio", name: "voices.enroll", summary: "Enroll a saved voice for a speech model without an encoder (Voxtral TTS; experimental: the voice fitted through its decoder, a minute or two on a card) - a job (audio.status / audio.jobs / audio.cancel by its label; progress: the fit's step). It waits for a card with the fit's memory (~8 GiB) and 1.5 GiB more free, as a load does; once done the voice is offered to that model (audio.voices.list: models) and audio.speak speaks it there as voice",
             fields: &[("name", "string", "the saved voice (audio.voices.list)", true), ("model", "string", "the model to enroll it for (default: the registered one that speaks enrolled voices)", false),
                       ("gpu", "integer", "this card only (default: the model's cards, the first with room)", false),
                       ("steps", r#"{"type": "integer", "minimum": 50, "maximum": 20000, "default": 2000}"#, "the fit's steps", false),
                       ("replace", "boolean", "fit it again when it is enrolled already", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for audio.status and audio.cancel (default: its id)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends; false: answer at once with the job", false), QUEUE, PRIORITY],
             example: r#"{"name": "grandpa", "wait": false, "label": "enroll-grandpa"}"# },
    Method { kind: "audio", name: "voices.remove", summary: "Delete a saved voice (an owned one: owner must match)", fields: &[("name", "string", "", true), ("owner", "string", "the voice's owner, when it has one", false)], example: r#"{"name": "narrator"}"# },
    Method { kind: "audio", name: "info", summary: "The model's defaults, limits and options (a speech model: its voices and languages)", fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "audio", name: "progress", summary: "The request running: its phase and how far", fields: &[], example: "{}" },
    Method { kind: "audio", name: "history", summary: "The songs and speech made, newest first", fields: &[], example: "{}" },
    Method { kind: "audio", name: "status", summary: "One job (audio.generate / audio.speak) by its label or reentrant_id (one of them is needed: audio.jobs lists them): state, stage and what it waits for, its place in the queue, progress, its sound when done",
             fields: &JOB_ID, example: r#"{"label": "theme-3"}"# },
    Method { kind: "audio", name: "jobs", summary: "The audio jobs: running, waiting (in run order) and the last ended, each as audio.status answers it",
             fields: &[("limit", r#"{"type": "integer", "minimum": 0, "maximum": 200, "default": 20}"#, "ended jobs listed", false), ("project", "string", "only this project's", false)], example: r#"{"limit": 5}"# },
    Method { kind: "audio", name: "queue.list", summary: "The waiting audio jobs in run order (held ones too: held, waiting_for)", fields: &[], example: "{}" },
    Method { kind: "audio", name: "queue.move", summary: "Move a waiting job to a place in the queue (1 = made next)",
             fields: &[JOB_ID[0], JOB_ID[1], ("position", r#"{"type": "integer", "minimum": 1, "default": 1}"#, "", false),
                       ("priority", r#"{"type": "integer", "minimum": -9, "maximum": 9}"#, "give it this priority too (where later jobs go)", false)], example: r#"{"label": "chapter-4", "position": 1}"# },
    Method { kind: "audio", name: "queue.pause", summary: "Hold a waiting job (label or reentrant_id), a project or batch (also the ones submitted later), or (none) the whole audio queue; running jobs go on",
             fields: &[JOB_ID[0], JOB_ID[1], ("project", "string", "", false), ("batch", "string", "", false)], example: r#"{"project": "Harbour"}"# },
    Method { kind: "audio", name: "queue.resume", summary: "Release what queue.pause held (none: everything)",
             fields: &[JOB_ID[0], JOB_ID[1], ("project", "string", "", false), ("batch", "string", "", false)], example: "{}" },
    Method { kind: "audio", name: "upload", summary: "Keep a recording here once (WAV, MP3, FLAC, Ogg, M4A; kept as WAV): audio.speak's ref_audio and audio.voices.add's sample then name it as \"upload:<name>\" instead of sending it again. Kept until system.files.delete {kind: audio, name}",
             fields: &[("name", "string", "the file's name; kept as up_<unix seconds>_<name>.wav (letters, digits, . _ - kept, the rest _, 60 characters)", true),
                       ("data", "string", "the recording, base64 (a data: URL too)", true)],
             example: r#"{"name": "grandpa.mp3", "data": "SUQzBAAAAA..."}"# },
    Method { kind: "audio", name: "cancel", summary: "Stop a job by its label or reentrant_id (running: between steps or frames; waiting: at once; ended: ok with already_ended); neither: whatever the audio servers are making",
             fields: &JOB_ID, example: r#"{"label": "theme-3"}"# },
    Method { kind: "audio", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "minimax-music3"}"# },

    Method { kind: "model3d", name: "generate", summary: "A mesh from a picture of an object, or several views of it (TRELLIS.2, Hunyuan3D, Pixal3D; the model loads first if it must): its shape and, with texture, PBR textures; the answer links the GLB. A job like image.generate (wait, reentrant_id, status, cancel)",
             fields: &[MODEL_OPT, ("image", "string", "the picture: PNG, JPEG or WebP as base64, a data: URL or upload:<name> (model3d.upload); a transparent background is the object's outline, an opaque one is cut out (remove_background)", false),
                       ("images", r#"{"type": "array", "maxItems": 8, "items": {"oneOf": [{"type": "string"}, {"type": "object", "properties": {"image": {"type": "string"}, "side": {"type": "string", "description": "front, back, left, right, top, bottom, or an azimuth in degrees"}}}]}}"#, "several views of the object (a multi-view model: model3d.info's abilities.views); each a picture or {image, side}", false),
                       ("texture", "boolean", "PBR textures too (base colour, metallic-roughness); default: what the model does (model3d.info)", false),
                       ("seed", "integer", "", false), ("steps", "integer", "denoising steps of each stage", false), ("cfg", "number", "guidance", false),
                       ("resolution", "integer", "the shape's grid / octree resolution: detail against time and memory", false),
                       ("faces", "integer", "at most this many triangles (decimated to it)", false), ("texture_size", "integer", "the texture's side in texels", false),
                       ("remove_background", r#"{"type": "boolean", "default": true}"#, "cut the background out of an opaque picture", false),
                       ("options", "object", "the engine's own (model3d.info lists them)", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false),
                       QUEUE, PRIORITY, HELD, PROJECT, BATCH],
             example: r#"{"model": "trellis2", "image": "data:image/png;base64,iVBORw0...", "texture": true, "seed": 42, "wait": false, "reentrant_id": "chair-1"}"# },
    Method { kind: "model3d", name: "rig", summary: "A mesh given a skeleton and skin weights (UniRig, RigNet), or set in motion by words (AnimateAnyMesh: prompt); the answer links the rigged or animated GLB. A job like model3d.generate",
             fields: &[MODEL_OPT, ("mesh", "string", "the mesh: GLB or OBJ as base64, a data: URL or upload:<name> (model3d.upload)", false),
                       ("file", "string", "or a GLB made here (model3d.generate's file name or url)", false),
                       ("prompt", "string", "an animating model: the motion, in words (\"a dragon flapping its wings\")", false),
                       ("frames", "integer", "an animating model: frames to make", false), ("fps", "number", "", false),
                       ("seed", "integer", "", false), ("steps", "integer", "", false), ("cfg", "number", "", false), ("options", "object", "the engine's own", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit (as model3d.generate)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends; false: answer at once with the job", false),
                       QUEUE, PRIORITY, HELD, PROJECT, BATCH],
             example: r#"{"model": "unirig", "file": "1792132541-trellis2-42.glb"}"# },
    Method { kind: "model3d", name: "tool", summary: "The CPU mesh tools, here (no GPU, no model): decimate (to a triangle budget), remesh (even triangles), uv (a fresh atlas, the texture or colours baked across), convert; the result is saved beside the meshes made (model3d.history) and linked",
             fields: &[("op", r#"{"type": "string", "enum": ["decimate", "remesh", "uv", "convert"]}"#, "", true),
                       ("mesh", "string", "the mesh: GLB or OBJ as base64, a data: URL or upload:<name>", false), ("file", "string", "or a GLB made here", false),
                       ("faces", "integer", "decimate: at most this many triangles", false), ("edge", "number", "remesh: the edge length (default: the mesh's mean)", false),
                       ("passes", "integer", "remesh: passes (5)", false), ("size", "integer", "uv: the texture's side (2048)", false)],
             example: r#"{"op": "decimate", "file": "1792132541-trellis2-42.glb", "faces": 20000}"# },
    Method { kind: "model3d", name: "info", summary: "The model's abilities (generate, texture, views, rig, animate), defaults and options (loaded: from the server)", fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "model3d", name: "progress", summary: "The request running: its phase (image, structure, shape, texture, decode, mesh, bake, skeleton, skin, motion) and how far", fields: &[], example: "{}" },
    Method { kind: "model3d", name: "history", summary: "The meshes made, newest first, with their settings and pictures", fields: &[], example: "{}" },
    Method { kind: "model3d", name: "status", summary: "One job (model3d.generate / model3d.rig) by its label or reentrant_id: state, stage and what it waits for, its place in the queue, progress, its GLB when done",
             fields: &JOB_ID, example: r#"{"label": "chair-1"}"# },
    Method { kind: "model3d", name: "jobs", summary: "The 3D jobs: running, waiting (in run order) and the last ended, each as model3d.status answers it",
             fields: &[("limit", r#"{"type": "integer", "minimum": 0, "maximum": 200, "default": 20}"#, "ended jobs listed", false), ("project", "string", "only this project's", false)], example: r#"{"limit": 5}"# },
    Method { kind: "model3d", name: "queue.list", summary: "The waiting 3D jobs in run order (held ones too: held, waiting_for)", fields: &[], example: "{}" },
    Method { kind: "model3d", name: "queue.move", summary: "Move a waiting job to a place in the queue (1 = made next)",
             fields: &[JOB_ID[0], JOB_ID[1], ("position", r#"{"type": "integer", "minimum": 1, "default": 1}"#, "", false),
                       ("priority", r#"{"type": "integer", "minimum": -9, "maximum": 9}"#, "give it this priority too (where later jobs go)", false)], example: r#"{"label": "chair-2", "position": 1}"# },
    Method { kind: "model3d", name: "queue.pause", summary: "Hold a waiting job (label or reentrant_id), a project or batch (also the ones submitted later), or (none) the whole 3D queue; running jobs go on",
             fields: &[JOB_ID[0], JOB_ID[1], ("project", "string", "", false), ("batch", "string", "", false)], example: r#"{"project": "Props"}"# },
    Method { kind: "model3d", name: "queue.resume", summary: "Release what queue.pause held (none: everything)",
             fields: &[JOB_ID[0], JOB_ID[1], ("project", "string", "", false), ("batch", "string", "", false)], example: "{}" },
    Method { kind: "model3d", name: "upload", summary: "Keep a picture (PNG, JPEG, WebP) or a mesh (GLB, OBJ) here once: model3d.generate's image / images and model3d.rig's mesh then name it as \"upload:<name>\". Kept until system.files.delete {kind: model3d, name}",
             fields: &[("name", "string", "the file's name: png jpg jpeg webp glb obj; kept as up_<unix seconds>_<name>", true), ("data", "string", "base64 (a data: URL too)", true)],
             example: r#"{"name": "chair.png", "data": "iVBORw0KGgo..."}"# },
    Method { kind: "model3d", name: "cancel", summary: "Stop a job by its label or reentrant_id (running: between steps; waiting: at once; ended: ok with already_ended); neither: whatever the 3D servers are making",
             fields: &JOB_ID, example: r#"{"label": "chair-1"}"# },
    Method { kind: "model3d", name: "describePromptGuideMD", summary: "How to get good meshes from the model, as Markdown", fields: &[MODEL], example: r#"{"model": "trellis2"}"# },

    Method { kind: "video", name: "generate", summary: "Make a clip: it starts at once when the studio is idle (queued: false, the job), else it is queued with queue: true (queued: true, its position) or refused with 409 busy. Anchors (first_frame, last_frame, exposure_ref, first_audio, guide_clip, source) take prev, prev_cam, first, first_cam - clips of the label's series, see label - or a file in the clips directory (video.upload, or a finished clip's name)",
             fields: &[("prompt", "string", "read as written - tokenized whole, no template, no sections parsed. H3 was trained on three fields in a fixed order: integrated_multimodal_description (shots, actions, camera, dialogue), overall_soundscape, non_diegetic_music; video.describePromptGuideMD has the form and its rules", true),
                       ("seconds", r#"{"type": "number", "minimum": 1, "maximum": 15.1, "default": 10}"#, "length; snapped up to the model's frame grid (17k + 5 frames at 24 fps: 4 s -> 107 frames); the model is trained on 5.2-15.1 s. The answer's frames and seconds say what is made", false),
                       ("frames", r#"{"type": "integer", "minimum": 22, "maximum": 362}"#, "the length exactly, instead of seconds: 17k + 5 (22, 39, 56 ... 107, 124 ... 362); another count is refused, the message naming the nearest two", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a clip of this id queued, running or made is answered (reentered: true) instead of a new one; a failed one is made again", false),
                       ("chain", r#"{"type": "object", "properties": {"from": {"type": "string", "default": "prev", "description": "anchor or file: the clip carried on from"},
                         "context_frames": {"type": "integer", "description": "0 or 17k + 5 (22 = 0.92 s): the last N frames of from as one moving keyframe at frame 0 (guide_clip) - the clip replays them, so trim N/2 each side at the join"},
                         "sound": {"type": "boolean", "default": true, "description": "with context_frames: the sound's last N/24 s too (first_audio, first_audio_s)"},
                         "still": {"type": "boolean", "description": "from's last frame as the first frame (first_frame); default: when context_frames is 0"},
                         "mode": {"type": "string", "enum": ["video", "png", "latent"], "description": "still: chain_mode"},
                         "color_reanchor": {"description": "true (= from), an anchor or a picture: the carried pictures scaled to its brightness (exposure_ref) - gain 0.75-1.35 on luminance, against drift"}}}"#,
                        "how the clip carries on from another, as one object (it sets guide_clip, first_audio, first_audio_s, first_frame, chain_mode, exposure_ref; those given flat win). There is no renormalisation step here: pin_renorm and other keys are listed back in ignored", false),
                       ("negative_prompt", "string", "ignored: H3 runs at cfg 1 (one pass a step, no unconditional branch), so a negative prompt has nothing to steer - say what is there instead (negations render). Accepted and listed in the answer's ignored", false),
                       ("steps", r#"{"type": "integer", "minimum": 1, "maximum": 40, "default": 10}"#, "denoising steps; below 8 only with a turbo LoRA", false),
                       ("seed", r#"{"type": "integer", "minimum": -2147483648, "maximum": 2147483648, "default": 0}"#, "", false),
                       ("width", r#"{"type": "integer", "minimum": 256, "maximum": 1344, "multipleOf": 32, "default": 768}"#, "a multiple of 32; width x height at most 1032192 pixels (768 x 1344); video.canvases.list has the tested ones", false),
                       ("height", r#"{"type": "integer", "minimum": 256, "maximum": 1344, "multipleOf": 32, "default": 576}"#, "as width", false),
                       ("engine", r#"{"type": "string", "default": "INT8"}"#, "the denoiser, by video.engines.list's quant (Q8_0 is read as INT8); REF2VA is chosen by itself when ref_images or ref_videos are given", false),
                       ("label", r#"{"type": "string", "maxLength": 80}"#, "\"<series> NN/MM: text\", e.g. \"Harbour 03/12: the lamp\": NN is the clip's place in video.timeline (the edit order), the text before it the series the anchors prev / first look in and the project when none is given. Letters, digits, _ and  -.,:'()!?/ only (others are dropped); labels name clips for video.cancel, video.queue.move and video.queue.remove", false),
                       ("project", r#"{"type": "string", "maxLength": 60}"#, "default: the label's series", false),
                       ("batch", r#"{"type": "string", "maxLength": 60}"#, "a group inside the project, held or cleared together", false),
                       ("queue", r#"{"type": "boolean", "default": false}"#, "queue the clip when the studio is busy (false: 409 busy)", false),
                       ("priority", r#"{"type": "integer", "minimum": -9, "maximum": 9, "default": 0}"#, "a queued clip goes ahead of every clip of a lower priority (same priority: first come first made)", false),
                       ("te", r#"{"type": "string", "enum": ["teacher", "student"], "default": "teacher"}"#, "the text encoder", false),
                       ("chain_mode", r#"{"type": "string", "enum": ["video", "png", "latent", "none"], "default": "video"}"#, "how a first_frame clip is carried over: its mp4's last frame, its lossless .last.png, its last latent, or not at all", false),
                       ("first_frame", "string", "anchor or file: the clip starts on this picture / clip's last frame", false),
                       ("last_frame", "string", "anchor or file: the clip ends on this picture (prev: the newest clip anywhere)", false),
                       ("exposure_ref", "string", "anchor or file: a picture whose exposure and colour the clip keeps", false),
                       ("first_audio", "string", "anchor or file: the sound the clip continues from", false),
                       ("first_audio_s", r#"{"type": "number"}"#, "seconds of first_audio to continue from", false),
                       ("guide_clip", "string", "a motion guide: \"<anchor or file>[:frames[:at]]\" - the last frames of a clip as one moving keyframe", false),
                       ("camera", "string", "a camera name: prev_cam / first_cam anchor on the series' clips of the same camera", false),
                       ("upscale", r#"{"type": "number", "minimum": 1}"#, "upscale at decode (1.5: the canvas 1.5 times); above 1 only", false),
                       ("upscaler", r#"{"type": "string", "enum": ["esrgan-general", "esrgan-anime", "latent"], "default": "esrgan-general"}"#, "with upscale", false),
                       ("loras", r#"{"type": "array", "items": {"type": "string", "maxLength": 200}}"#, "\"path:strength\"", false),
                       ("ref_images", r#"{"type": "array", "maxItems": 9, "items": {"type": "string"}}"#, "Ref2VA reference pictures (files in the clips directory), <Image 1> ... in the prompt", false),
                       ("ref_videos", r#"{"type": "array", "maxItems": 3, "items": {"type": "string"}}"#, "Ref2VA reference clips", false),
                       ("ref_audios", r#"{"type": "array", "maxItems": 3, "items": {"type": "string"}}"#, "voices to speak in: <Audio 1> ... in the prompt", false),
                       ("ref_image_size", r#"{"type": "string", "enum": ["match", "max"]}"#, "reference pictures resized to the canvas, or kept as large as fits", false),
                       ("ref_video_sound", "boolean", "the reference clips' sound too", false),
                       ("te_pictures", "boolean", "the text encoder sees the reference pictures", false),
                       ("control_video", "string", "a ControlNet control clip (file); not with Ref2VA", false),
                       ("control_mask", "string", "a mask clip or picture (file): regenerate inside it", false),
                       ("control_source", "string", "the clip the mask applies to (file)", false),
                       ("control_strength", r#"{"type": "number", "minimum": 0, "maximum": 10, "default": 1}"#, "", false),
                       ("control_start", r#"{"type": "number", "minimum": 0, "maximum": 1, "default": 0}"#, "the part of the steps the control steers", false),
                       ("control_end", r#"{"type": "number", "minimum": 0, "maximum": 1, "default": 1}"#, "", false),
                       ("source", "string", "anchor or file: a masked run - this clip's latents are kept outside the regenerated part", false),
                       ("regen", "string", "\"from-to\" seconds to regenerate in source; a source shorter than the clip is extended", false),
                       ("regen_box", "string", "\"x0,y0,x1,y1\" pixels: regenerate only inside", false),
                       ("shift_video", "number", "flow shift of the picture (12)", false), ("shift_audio", "number", "flow shift of the sound (3)", false),
                       ("cond_noise_aug", "number", "noise on the first frame (harmful here: shows as confetti)", false),
                       ("sampler", "string", "ComfyUI's name (none: the engine's own Euler)", false), ("schedule", "string", "ComfyUI's name", false)],
             example: r#"{"prompt": "integrated_multimodal_description: [Shot 1] Live-action, a red fox trots through deep snow at dawn.\n\noverall_soundscape: soft wind, crunching snow.\n\nnon_diegetic_music: None.", "label": "Fox 01/03: snow", "seconds": 5, "steps": 8, "queue": true}"# },
    Method { kind: "video", name: "status", summary: "The clip running or last: stage, percent, ETA, timings, a preview picture of the shot while it samples (preview), the queue, version", fields: &[], example: "{}" },
    Method { kind: "video", name: "wait", summary: "Long poll: answers when the status's version changes, or at the timeout",
             fields: &[("version", "string", "the last version seen", false), ("timeout_s", r#"{"type": "number", "minimum": 0, "maximum": 60, "default": 30}"#, "the longest it waits (above 60: refused)", false)], example: r#"{"version": "", "timeout_s": 30}"# },
    Method { kind: "video", name: "cancel", summary: "Stop the running clip at its next step; with a label, that clip only: stopped if it runs, dropped from the queue if it waits",
             fields: &[("label", "string", "a clip's label (none: the running clip)", false)], example: r#"{"label": "Fox 02/03: the den"}"# },
    Method { kind: "video", name: "queue.list", summary: "The queue in the order it will run (each item as video.generate checked it)", fields: &[], example: "{}" },
    Method { kind: "video", name: "queue.move", summary: "Move a queued clip (the first with that label) to a place in the queue: 1 = made next",
             fields: &[("label", "string", "", true), ("position", r#"{"type": "integer", "minimum": 1, "default": 1}"#, "", false),
                       ("priority", r#"{"type": "integer", "minimum": -9, "maximum": 9}"#, "give it this priority too (where later clips go)", false)],
             example: r#"{"label": "Fox 03/03: night", "position": 1}"# },
    Method { kind: "video", name: "queue.remove", summary: "Drop the queued clips with that label (a running one: video.cancel)", fields: &[("label", "string", "", true)],
             example: r#"{"label": "Fox 03/03: night"}"# },
    Method { kind: "video", name: "queue.pause", summary: "Hold a project or batch (none: everything)", fields: &[("project", "string", "", false), ("batch", "string", "", false)], example: r#"{"project": "Harbour"}"# },
    Method { kind: "video", name: "queue.resume", summary: "Release a project or batch (none: everything)", fields: &[("project", "string", "", false), ("batch", "string", "", false)], example: "{}" },
    Method { kind: "video", name: "queue.clear", summary: "Drop the queue, or a project's / batch's part of it", fields: &[("project", "string", "", false), ("batch", "string", "", false)], example: "{}" },
    Method { kind: "video", name: "jobs.list", summary: "Finished clips (GET /v1/video/files/<file> fetches one)", fields: &[("limit", "integer", "", false)], example: r#"{"limit": 20}"# },
    Method { kind: "video", name: "jobs.get", summary: "A clip's record: settings, per-stage seconds, energy", fields: &[("name", "string", "h3_YYYYMMDD_HHMMSS", true)], example: r#"{"name": "h3_20261009_113734"}"# },
    Method { kind: "video", name: "timeline", summary: "A project's clips in edit order: done (with its file's name), running, held, queued",
             fields: &[("project", "string", "", true)], example: r#"{"project": "Harbour"}"# },
    Method { kind: "video", name: "projects.list", summary: "The projects and batches", fields: &[], example: "{}" },
    Method { kind: "video", name: "films.list", summary: "Joined films", fields: &[], example: "{}" },
    Method { kind: "video", name: "templates.list", summary: "Prompt templates", fields: &[], example: "{}" },
    Method { kind: "video", name: "engines.list", summary: "The denoisers the daemon has, and the video.generate keys each takes: INT8 / Q6_K / Q4_K_M (text and image to video: anchors, guides, ControlNet, masked runs) and REF2VA (reference pictures and clips; no ControlNet or masked runs). ref_audios, anchors and guides work with every one", fields: &[], example: "{}" },
    Method { kind: "video", name: "canvases.list", summary: "The canvases and their measured cost", fields: &[], example: "{}" },
    Method { kind: "video", name: "summary", summary: "Measured GPU time per second of video, energy, queue ETA", fields: &[], example: "{}" },
    Method { kind: "video", name: "upload", summary: "Put a reference, control video, mask or keyframe beside the clips (its name is what the other keys take). The request is at most 256 MiB, so the file at most about 190 MiB as base64. Uploads are kept: nothing deletes them (they sit in the clips directory with the clips)",
             fields: &[("name", "string", "the file's name: png jpg jpeg webp mp4 mov webm mkv wav mp3 flac m4a ogg; saved as ref_<unix seconds>_<name> (letters, digits, . _ - kept, the rest _, 60 characters)", true),
                       ("data", "string", "the file, base64 (a data: URL too)", true)],
             example: r#"{"name": "fox.png", "data": "iVBORw0KGgo..."}"# },
    Method { kind: "video", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "minimax-h3"}"# },
];

/// system.version's api: raised when an answer changes in a way an old client misreads
const API_VERSION: u32 = 3;

/// What this build does, by names a client can test for (system.version); names are added, never reused
const FEATURES: &[&str] = &[
    "system.busy", "system.busy.cards", "system.busy.slots", "system.version", "system.files.stat", "system.files.delete",
    "jobs.image", "jobs.audio", "jobs.cancel.label", "reentrant_id",
    "video.frames", "video.chain", "video.priority", "video.cancel.label", "video.queue.move", "video.queue.remove", "video.engines.inputs",
    "video.negative_prompt.ignored", "image.canvases", "image.cancel", "spec.open",
    "image.edit.strength", "audio.speak.continue_from", "video.preview", "system.busy.parallel",
    "errors.retryable", "errors.no_room", "jobs.precheck", "image.info.steps", "image.generate.init_image", "image.edit.reference",
    "video.failure", "video.gpu", "video.engines.ignored", "audio.speak.precheck.voice", "jobs.cancel.reentrant_id", "rpc.errors.body", "audio.transcribe",
    "rpc.http200", "result.run", "models.capabilities", "system.cards", "jobs.list", "jobs.status.one", "jobs.queue", "jobs.priority", "jobs.held",
    "jobs.stage", "uploads", "voices.owner", "voices.ready", "room.free_idle", "room.wait", "llm.no_room", "llm.keep_up_s",
    "system.maintenance", "errors.maintenance", "jobs.model3d", "model3d.generate", "model3d.rig", "model3d.upload.mesh",
    "system.drives", "system.move", "errors.cards_taken", "audio.voices.enroll", "drives.sidecars", "image.loras.request",
];

/// Qwen-Image's sizes: its training buckets (multiples of 16) and the guide's defaults
const IMAGE_CANVASES: &[(u32, u32, &str, &str)] = &[
    (1024, 1024, "1:1", "the default"), (1328, 1328, "1:1", "the model's own square, the largest trained"),
    (1344, 768, "16:9", ""), (768, 1344, "9:16", ""), (1664, 928, "16:9", "the model's own wide"), (928, 1664, "9:16", "the model's own tall"),
    (1152, 864, "4:3", ""), (864, 1152, "3:4", ""), (1472, 1104, "4:3", "the model's own 4:3 (1472 x 1140 there, 1104 here: multiples of 16)"),
    (1104, 1472, "3:4", ""), (1248, 832, "3:2", ""), (832, 1248, "2:3", ""), (1584, 1056, "3:2", "the model's own 3:2"), (1056, 1584, "2:3", ""),
    (512, 512, "1:1", "a quick draft"),
];

pub(crate) fn code_of(status: u16) -> &'static str {
    match status {
        400 => "invalid_params",
        401 => "unauthorized",
        404 => "not_found",
        499 => "cancelled",
        409 => "conflict",
        502 => "unavailable",
        503 => "no_room",
        504 => "timeout",
        _ => "internal",
    }
}

/// Whether the same call may succeed later unchanged: the box was busy or a server did not answer (not a bad value,
/// a missing or disabled model, a cancel)
pub(crate) fn retryable(status: u16, code: &str) -> bool {
    matches!(status, 502..=504) || code == "busy" || code == "maintenance" || code == "cards_taken"
}

/// An OpenAI route's error: {"error": {message, type}}; cards taken by a GPU run (crate::gpulock) as maintenance's,
/// with its code and Retry-After
fn openai_error(s: &mut Conn, st: u16, m: &str) {
    if m.starts_with("cards taken:") {
        let body = json!({"error": {"message": m, "type": "cards_taken", "code": "cards_taken"}});
        let extra = format!("Retry-After: {}\r\n", crate::gpulock::RETRY_AFTER_S);
        return http::respond_bytes(s, st, "application/json", body.to_string().as_bytes(), &extra);
    }
    // a model not downloaded says so (code model_not_downloaded, its size, the call)
    if let Some((f, words)) = super::pulls::untag(m) {
        let mut e = json!({"message": words, "type": "model_not_downloaded", "code": "model_not_downloaded"});
        if let (Some(o), Some(f)) = (e.as_object_mut(), f.as_object()) {
            o.extend(f.clone());
        }
        return http::respond(s, st, &json!({"error": e}));
    }
    http::respond(s, st, &json!({"error": {"message": m, "type": if st == 401 { "invalid_api_key" } else { "invalid_request_error" }}}));
}

/// An error's body: code, message, whether to retry, and the status as a number (the same as the HTTP one, so a
/// client - or another transport - needs only the body)
pub(crate) fn error_of(status: u16, m: &str) -> Value {
    let code = code_of(status);
    json!({"code": code, "message": m, "retryable": retryable(status, code), "status": status})
}

/// An /rpc error in full: error_of with the method, the request key at fault when the message starts with one
/// ("steps 8: ...", "images[0]: ...", "voice bogus: ..."), and a code of its own where the status's is too broad
pub(crate) fn rpc_error(method: &str, status: u16, code: Option<&str>, m: &str) -> Value {
    // a model a card offers that is not downloaded (pulls.rs): its own code and what fetches it
    if let Some((fields, words)) = super::pulls::untag(m) {
        let mut e = rpc_error(method, status, Some("model_not_downloaded"), &words);
        if let (Some(o), Some(f)) = (e.as_object_mut(), fields.as_object()) {
            o.extend(f.clone());
        }
        return e;
    }
    let mut e = error_of(status, m);
    // a job refused with queue: false says "busy: ..." (as video's busy)
    let code = code.or_else(|| (status == 409 && m.starts_with("busy:")).then_some("busy"));
    // a load refused while a GPU run holds the cards (crate::gpulock) says "cards taken: ..."
    let code = code.or_else(|| m.starts_with("cards taken:").then_some("cards_taken"));
    if let Some(c) = code {
        e["code"] = json!(c);
        e["retryable"] = json!(retryable(status, c));
    }
    if e["code"] == "cards_taken" {
        e["retry_after_s"] = json!(crate::gpulock::RETRY_AFTER_S);
    }
    e["method"] = if method.is_empty() { Value::Null } else { json!(method) };
    e["field"] = field_of(method, m).map_or(Value::Null, |f| json!(f));
    e
}

/// The request key a message names first, when it is one of the method's fields
fn field_of(method: &str, m: &str) -> Option<String> {
    let (kind, name) = method.split_once('.')?;
    let me = METHODS.iter().find(|x| x.kind == kind && x.name == name)?;
    let first: String = m.trim_start_matches('"').chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    let mentioned = |f: &str| first == f || m.contains(&format!("needs \"{f}\""));
    me.fields.iter().map(|f| f.0).chain(["strength", "image", "reference", "continue_from"]).find(|f| mentioned(f)).map(str::to_string)
}

type Fail = (u16, String);

/// A call to a kind's server whose failure keeps that server's status (a 400 stays a 400); 502 when it does not answer
fn relay(t: &crate::http::Target, method: &str, path: &str, body: Option<&Value>, secs: u64) -> Result<Value, Fail> {
    match http::call_status(t, method, path, body, secs) {
        Ok((200, v)) => Ok(v),
        Ok((st, v)) => Err((st, v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("request failed").to_string())),
        Err(e) => Err((502, e)),
    }
}

impl Home {
    fn authorized(&self, req: &Request) -> bool {
        http::header(&req.head, "authorization").is_some_and(|h| h.trim().strip_prefix("Bearer ").is_some_and(|t| t.trim() == self.token))
    }

    pub(crate) fn api(self: &Arc<Self>, mut s: Conn, req: Request) {
        let rpc = req.path.starts_with("/rpc/");
        let method = req.path.strip_prefix("/rpc/").unwrap_or("").to_string();
        let fail_as = |s: &mut Conn, st: u16, code: Option<&str>, m: String| {
            if rpc {
                // HTTP 200: the envelope is the answer, error.status keeps the number
                http::respond(s, 200, &json!({"ok": false, "error": rpc_error(&method, st, code, &m)}));
            } else {
                openai_error(s, st, &m);
            }
        };
        let fail = |s: &mut Conn, st: u16, m: String| {
            if rpc {
                fail_as(s, st, None, m);
            } else {
                openai_error(s, st, &m);
            }
        };
        if !self.authorized(&req) {
            return fail(&mut s, 401, "Authorization: Bearer <token> (the token is on the page's API tab)".into());
        }
        // maintenance: what would start work refused with the notice (maint.rs)
        if self.maintenance_refused(&mut s, &req) {
            return;
        }
        if rpc {
            if req.method != "POST" || req.target.contains('?') {
                return fail_as(&mut s, 400, Some("invalid_request"), "POST only, every variable in the JSON body (no query strings)".into());
            }
            let body: Value = if req.body.is_empty() { json!({}) } else {
                match serde_json::from_slice(&req.body) {
                    Ok(v @ Value::Object(_)) => v,
                    _ => return fail_as(&mut s, 400, Some("invalid_request"), "the body must be a JSON object".into()),
                }
            };
            let name = &req.path["/rpc/".len()..];
            if let Some((_, v)) = self.studio_rpc(name, &body, req.host.as_deref().unwrap_or("localhost:8000")) {
                return http::respond(&mut s, 200, &v);
            }
            match self.rpc(name, &body) {
                Ok(v) => {
                    // the files an answer links, as the caller reaches them (through here: /v1/.../files/...)
                    let host = req.host.clone().unwrap_or_else(|| "localhost:8000".into());
                    let v: Value = serde_json::from_str(&v.to_string().replace("http://nextsycl/", &format!("http://{host}/"))).unwrap_or(v);
                    http::respond(&mut s, 200, &json!({"ok": true, "result": v}))
                }
                Err((st, m)) => fail(&mut s, st, m),
            }
            return;
        }
        if let Err((st, m)) = self.openai(&mut s, &req) {
            fail(&mut s, st, m);
        }
    }

    /// The kinds with an enabled model (system always)
    pub(super) fn kinds_on(&self, all: &[Value]) -> Vec<&'static str> {
        let mut v = vec!["system"];
        for k in ["llm", "image", "audio", "video", "model3d"] {
            if all.iter().any(|m| registry::kind_of(m) == k && m["enabled"] != false) {
                v.push(k);
            }
        }
        v
    }

    /// A kind's model for a request: the one it names (enabled, of the kind), else the loaded one, else the first
    /// enabled
    pub(super) fn pick(&self, kind: &str, body: &Value, all: &[Value]) -> Result<String, Fail> {
        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == kind && m["enabled"] != false).collect();
        if let Some(id) = body["model"].as_str().filter(|x| !x.is_empty()) {
            return match all.iter().find(|m| m["id"] == id) {
                Some(m) if registry::kind_of(m) != kind => Err((400, format!("{id} is a {} model, not {kind}", registry::kind_of(m)))),
                Some(m) if m["enabled"] == false => Err((409, format!("{id} is disabled (system.enable)"))),
                Some(_) => Ok(id.to_string()),
                // a card's model not here yet: refused (409 model_not_downloaded), or with NS_AUTO_PULL fetched first -
                // a job's request id lets it say so while it waits
                None if self.not_downloaded(id).is_some() => self.need_downloaded(id, body["request_id"].as_str(), true).map(|_| id.to_string()),
                None => Err((404, format!("no model {id} (system.models)"))),
            };
        }
        let loaded = self.loaded_of(kind);
        loaded.into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
            .ok_or_else(|| (409, format!("no {kind} model is enabled (system.enable)")))
    }

    /// The saved voices (beside the audio server's output: NS_AUDIO_OUT)
    pub(crate) fn voices(&self) -> crate::voices::Library {
        let out = self.o.cfg.get("NS_AUDIO_OUT").unwrap_or_else(|| format!("{}/.local/share/nextsycl/audio", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
        crate::voices::Library::of(std::path::Path::new(&out))
    }

    /// The recogniser for audio.transcribe: the one named (it must listen), else the loaded or first enabled one
    pub(super) fn recogniser(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        let listens = |m: &Value| s(m, "arch") == "whisper";
        if b["model"].as_str().is_some_and(|x| !x.is_empty()) {
            let id = self.pick("audio", b, all)?;
            if all.iter().any(|m| m["id"] == id.as_str() && !listens(m)) {
                return Err((400, format!("{id} makes sound; audio.transcribe needs a recogniser (whisper-large-v3-turbo)")));
            }
            return Ok(id);
        }
        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && listens(m)).collect();
        self.loaded_of("audio").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
            .ok_or_else(|| (409, "no recogniser is enabled (system.enable whisper-large-v3-turbo)".to_string()))
    }

    /// The speech model for a request that names none (or names one not registered): one that clones for a saved
    /// voice or a recording, the built-in voices' for a voice, the voice-design one for instructions alone; the
    /// loaded one first
    pub(super) fn speech_model(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        // one named that is here, or a card's not downloaded yet (pick refuses or fetches it)
        if b["model"].as_str().is_some_and(|m| all.iter().any(|x| x["id"] == m) || self.not_downloaded(m).is_some()) {
            return self.pick("audio", b, all);
        }
        let voice = s(b, "voice");
        let cloned = !s(b, "ref_audio").is_empty() || !s(b, "continue_from").is_empty() || (!voice.is_empty() && self.voices().get(&voice).is_some());
        // a saved voice alone (no recording, no carry-on): a model it is enrolled for speaks it too
        let saved_only = !voice.is_empty() && s(b, "ref_audio").is_empty() && s(b, "continue_from").is_empty() && self.voices().get(&voice).is_some();
        let lib = self.voices();
        let speakers = ["voices", "ref_audios"].iter().any(|k| b[*k].as_array().is_some_and(|a| !a.is_empty()));
        // a clone with an instruction: a model that directs a cloned voice (Breeze TTS 2)
        let need = if speakers {
            "speakers"
        } else if cloned && !s(b, "instructions").is_empty() {
            "direction"
        } else if cloned {
            "clone"
        } else if !voice.is_empty() {
            "voices"
        } else if !s(b, "instructions").is_empty() {
            "design"
        } else {
            ""
        };
        // a saved voice without its transcript: not a model that clones only with one
        let untold = !voice.is_empty() && self.voices().get(&voice).is_some_and(|v| v["text"].as_str().is_none_or(|t| t.trim().is_empty()));
        let can: Vec<String> = all.iter()
            .filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false)
            .filter(|m| m["speech"].as_array().is_some_and(|a| need.is_empty() || a.iter().any(|x| x == need)))
            .filter(|m| !(untold && needs_text(all, &m["id"])))
            // a built-in voice: a model that has it (or lists none)
            .filter(|m| need != "voices" || speaks(m, &voice))
            .map(|m| s(m, "id")).collect();
        // a saved voice no enabled model clones: a model it is enrolled for (experimental; models that clone come first)
        let can = if can.is_empty() && need == "clone" && saved_only {
            all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && enrolled_for(&lib, &voice, m)).map(|m| s(m, "id")).collect()
        } else {
            can
        };
        let loaded = self.loaded_of("audio");
        loaded.into_iter().find(|l| can.contains(l)).or_else(|| can.first().cloned()).ok_or_else(|| {
            (409, match need {
                "clone" => "no enabled speech model clones a voice (system.enable qwen3-tts-base)".to_string(),
                "speakers" => "no enabled speech model speaks a script of several voices (system.enable vibevoice-1.5b)".to_string(),
                "direction" => "no enabled speech model directs a cloned voice by instructions (system.enable breeze-tts-2)".to_string(),
                "design" => "no enabled speech model designs a voice (system.enable qwen3-tts-design)".to_string(),
                "voices" => "no enabled speech model has built-in voices (system.enable qwen3-tts-custom)".to_string(),
                _ => "no speech model is enabled (system.enable qwen3-tts-custom ...)".to_string(),
            })
        })
    }

    /// audio.voices.design: the voice-design model reads a sample in the voice described, which is saved
    fn design_voice(self: &Arc<Self>, b: &Value, all: &[Value]) -> Result<Value, Fail> {
        let name = s(b, "name");
        crate::voices::check_name(&name).map_err(|e| (400, e))?;
        let lib = self.voices();
        if lib.get(&name).is_some() && b["replace"] != true {
            return Err((409, format!("a voice {name:?} exists already (replace: true)")));
        }
        self.owned(&name, b)?;
        let lang = b["language"].as_str().filter(|l| !l.is_empty());
        let text = b["text"].as_str().filter(|t| !t.trim().is_empty()).map(str::to_string)
            .or_else(|| crate::voices::design_text(lang).map(str::to_string))
            .ok_or_else(|| (400, format!("text: what the sample says (no default line in {})", lang.unwrap_or("that language"))))?;
        let id = self.speech_model(&json!({"model": b["model"], "instructions": s(b, "instructions")}), all)?;
        let t = self.ensure(&id)?;
        let mut q = json!({"model": id, "input": text, "instructions": s(b, "instructions"), "response_format": "url"});
        for k in ["language", "seed", "options"] {
            if b.get(k).is_some_and(|v| !v.is_null()) {
                q[k] = b[k].clone();
            }
        }
        let r = relay(&t, "POST", "/v1/audio/speech", Some(&q), 600)?;
        let url = r["data"][0]["url"].as_str().ok_or_else(|| (502, format!("the speech server answered {r}")))?;
        let file = lib.dir.parent().map(|d| d.join(url.rsplit('/').next().unwrap_or(""))).unwrap_or_default();
        let bytes = std::fs::read(&file).map_err(|e| (500, format!("{}: {e}", file.display())))?;
        let seed = r["nextsycl"]["seed"].clone();
        let mut v = lib.add(&name, &bytes, Some(&text), Some(&s(b, "instructions")), lang, "design", &format!("designed by {id}, seed {seed}"), b["replace"] == true)
            .map_err(|e| (400, e))?;
        if let Some(owner) = b["owner"].as_str().filter(|o| !o.trim().is_empty()) {
            v = lib.tag(&name, "owner", json!(owner.trim())).map_err(|e| (500, e))?;
        }
        Ok(json!({"voice": v, "sample": format!("http://nextsycl/v1/audio/voices/{name}.wav"), "seed": seed}))
    }

    /// The method a name calls, its body's required fields there, its kind enabled
    fn checked(&self, name: &str, b: &Value, all: &[Value]) -> Result<&'static Method, Fail> {
        let Some(m) = METHODS.iter().find(|m| format!("{}.{}", m.kind, m.name) == name) else {
            return Err((404, format!("no method {name} (GET /openapi.yml)")));
        };
        for (f, _, _, req) in m.fields {
            if *req && b.get(*f).is_none_or(Value::is_null) {
                return Err((400, format!("{name} needs \"{f}\"")));
            }
        }
        // a model named that a card offers and is not downloaded: refused now (409 model_not_downloaded), or with
        // NS_AUTO_PULL its download started and the request let through (its kind is on once it lands; pick waits)
        if m.kind != "system" {
            if let Some(id) = b["model"].as_str().filter(|id| !all.iter().any(|x| x["id"] == *id) && self.not_downloaded(id).is_some()) {
                self.need_downloaded(id, None, false)?;
                return Ok(m);
            }
        }
        if !self.kinds_on(all).contains(&m.kind) && !self.pulling(m.kind) {
            return Err((409, format!("no {} model is enabled (system.enable)", m.kind)));
        }
        Ok(m)
    }

    /// A studio method (video.*, but upload and the guide): the studio's answer as it gave it - its status and error
    /// code (busy, not_running ...) - with the files it names as URLs through here
    fn studio_rpc(&self, name: &str, b: &Value, host: &str) -> Option<(u16, Value)> {
        let method = name.strip_prefix("video.").filter(|m| !matches!(*m, "upload" | "describePromptGuideMD"))?;
        let all = self.models();
        let envelope = |st: u16, m: String| (st, json!({"ok": false, "error": rpc_error(name, st, None, &m)}));
        if let Err((st, m)) = self.checked(name, b, &all) {
            return Some(envelope(st, m));
        }
        let (st, mut v) = match http::call_status(&self.target("video"), "POST", &format!("/rpc/{method}"), Some(b), 120) {
            Ok(r) => r,
            Err(e) => return Some(envelope(502, format!("the video studio does not answer: {e}"))),
        };
        if v["ok"] == false {
            // one spelling for both servers
            if v["error"]["code"] == "invalid_param" {
                v["error"]["code"] = json!("invalid_params");
            }
            // the studio's own code kept; the rest as every /rpc error has it
            let code = v["error"]["code"].as_str().unwrap_or("").to_string();
            let m = v["error"]["message"].as_str().unwrap_or("").to_string();
            let mut e = rpc_error(name, st, Some(&code).filter(|c| !c.is_empty()).map(String::as_str), &m);
            if let Some(o) = v["error"].as_object() {
                for (k, x) in o.iter().filter(|(k, _)| !["code", "message"].contains(&k.as_str())) {
                    e[k.as_str()] = x.clone();
                }
            }
            v["error"] = e;
            return Some((st, v));
        }
        let url = |kind: &str, n: &str| format!("http://{host}/v1/video/{kind}/{n}");
        let r = &mut v["result"];
        for k in ["file", "preview"] {
            if let Some(f) = r[k].as_str().and_then(|f| f.strip_prefix("/out/")).map(str::to_string) {
                r[k] = json!(url("files", &f));
            }
        }
        for c in r.get_mut("clips").and_then(Value::as_array_mut).into_iter().flatten() {
            if c["state"] == "done" {
                if let Some(n) = c["name"].as_str().map(str::to_string) {
                    c["file"] = json!(url("files", &format!("{n}.mp4")));
                    c["thumb"] = json!(url("thumbs", &format!("{n}.webp")));
                }
            }
        }
        Some((st, v))
    }

    /// Whether anything is working: models loading or answering, chat requests, image and audio jobs, the clip being
    /// made and the queue; where work runs (slots) and each card's share of it
    pub(super) fn busy(&self, all: &[Value]) -> Value {
        let cards = self.named_cards(all);
        let states = self.model_states(all, &cards);
        let with = |st: &str| states.iter().filter(|m| m["state"] == st).collect::<Vec<_>>();
        let loading: Vec<Value> = with("loading").iter().map(|m| m["id"].clone()).collect();
        let working: Vec<Value> = with("busy").iter().map(|m| json!({"id": m["id"], "kind": m["kind"]})).collect();
        let mut reasons: Vec<String> = loading.iter().map(|m| format!("{} loading", m.as_str().unwrap_or(""))).collect();
        reasons.extend(working.iter().map(|m| format!("{} answering", m["id"].as_str().unwrap_or(""))));
        let active = if self.kinds_on(all).contains(&"llm") {
            let dir = self.o.cfg.get("NS_SOCKET_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into())).join("nextsycl")
            });
            call_for(&crate::http::Target::Unix(dir.join("nextsycl.sock")), "GET", "/server/requests", None, 3).ok()
                .and_then(|v| v["active"].as_array().map(|a| a.len())).unwrap_or(0)
        } else {
            0
        };
        if active > 0 {
            reasons.push(format!("{active} chat request(s)"));
        }
        let jobs = self.jobs_open();
        // held jobs wait for a release, not for the box: they are listed, not reasons
        let live = jobs.iter().filter(|j| j["held"] != true).count();
        if live > 0 {
            reasons.push(format!("{live} image/audio/3D job(s)"));
        }
        let mut slots = self.work_slots(all, &cards);
        let video = if self.kinds_on(all).contains(&"video") {
            let st = call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 10).ok().map(|v| v["result"].clone()).unwrap_or(Value::Null);
            let items = st["queue_items"].as_array().cloned().unwrap_or_default();
            let held = items.iter().filter(|i| i["held"] == true).count();
            let running = !st.is_null() && st["idle"] != true;
            let runnable = items.len() - held;
            if running {
                reasons.push(format!("clip {} {}", st["job"]["label"].as_str().unwrap_or(""), st["stage"].as_str().unwrap_or("")));
            }
            if runnable > 0 {
                reasons.push(format!("{runnable} clip(s) queued"));
            }
            // the studio's job as the engine reports it, else the card the daemon shows busy
            let gpu = Some(st["gpu"].clone()).filter(|g| running && !g.is_null())
                .or_else(|| slots.iter().find(|x| x["kind"] == "video" && x["busy"] == true).map(|x| x["gpu"].clone())).unwrap_or(Value::Null);
            let on = |k: &str| if running { st[k].clone() } else { Value::Null };
            json!({"running": running, "label": if running { st["job"]["label"].clone() } else { Value::Null }, "name": if running { st["job"]["name"].clone() } else { Value::Null },
                   "stage": on("stage"), "pct": on("pct"), "eta_s": on("eta"), "elapsed_s": on("elapsed"), "gpu": gpu,
                   "energy_wh": if running { st["job"]["energy_wh"].clone() } else { Value::Null },
                   "queued": items.len(), "runnable": runnable, "held": held,
                   "accepts": !st.is_null() && !running && runnable == 0, "reachable": !st.is_null(),
                   "slots": 1, "cards": slots.iter().filter(|x| x["kind"] == "video").count()})
        } else {
            Value::Null
        };
        // a chat server answering marks its slot busy
        for x in slots.iter_mut().filter(|x| x["kind"] == "llm") {
            x["busy"] = json!(active > 0);
        }
        let per_card: Vec<Value> = cards.as_array().into_iter().flatten().map(|c| {
            let g = c["index"].clone();
            let here: Vec<&Value> = slots.iter().filter(|x| x["gpu"] == g || x["gpus"].as_array().is_some_and(|a| a.contains(&g))).collect();
            let models: Vec<Value> = c["procs"].as_array().into_iter().flatten().filter_map(|p| p["model_id"].as_str().map(|m| json!(m))).collect();
            json!({"gpu": g, "name": c["name"], "vram_total_mb": c["vram_total_mb"], "vram_used_mb": c["vram_used_mb"], "busy_pct": c["busy_pct"],
                   "power_w": c["power_w"], "models": models, "busy": here.iter().any(|x| x["busy"] == true),
                   "work": here.iter().filter(|x| x["busy"] == true).map(|x| json!({"kind": x["kind"], "model": x["model"]})).collect::<Vec<_>>()})
        }).collect();
        let maintenance = self.maintenance().map(|n| super::maint::public(&n));
        json!({"busy": !reasons.is_empty(), "reasons": reasons, "loading": loading, "working": working, "llm": {"active": active},
               "maintenance": maintenance, "cards_taken": self.cards_taken(),
               "jobs": jobs, "video": video, "cards": per_card, "slots": slots, "parallel": self.parallel(all)})
    }

    /// How many requests of each image and audio model run at once: a copy a card it is enabled on, as many as the
    /// kind has servers (a second copy starts when the first has work and its card has the room then)
    fn parallel(&self, all: &[Value]) -> Value {
        let mut out = serde_json::Map::new();
        for kind in super::jobs::KINDS {
            let servers = self.slots(kind).len();
            let models: serde_json::Map<String, Value> = all.iter().filter(|m| registry::kind_of(m) == kind && m["enabled"] != false)
                .map(|m| (s(m, "id"), json!(self.gpus_of(m).len().min(servers)))).collect();
            out.insert(kind.into(), json!({"servers": servers, "models": models}));
        }
        Value::Object(out)
    }

    fn rpc(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        // pictures and recordings sent before (upload:<name>) read in
        let resolved = self.uploads_in(name, b)?;
        let b = resolved.as_ref().unwrap_or(b);
        // pictures, songs and speech run as jobs (jobs.rs), which call work
        if super::jobs::METHODS.contains(&name) {
            let all = self.models();
            self.checked(name, b, &all)?;
            // what the server would refuse, refused now (wait: false would answer OK, then fail)
            self.precheck(name, b, &all)?;
            return self.submit(name, b);
        }
        self.work(name, b)
    }

    /// A method's work (a job's too)
    pub(super) fn work(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        let all = self.models();
        let m = self.checked(name, b, &all)?;
        let model = || s(b, "model");
        let t = self.target(m.kind);
        match (m.kind, m.name) {
            ("system", "state") => Ok(self.state()),
            ("system", "busy") => Ok(self.busy(&all)),
            ("system", "version") => {
                let on = self.kinds_on(&all);
                let methods: Vec<String> = METHODS.iter().filter(|m| on.contains(&m.kind)).map(|m| format!("{}.{}", m.kind, m.name)).collect();
                Ok(json!({"version": crate::VERSION, "api": API_VERSION, "kinds": on, "features": FEATURES, "methods": methods,
                          "maintenance": self.maintenance().map(|n| super::maint::public(&n))}))
            }
            ("system", "files.stat" | "files.delete") => self.file_op(m.name, b),
            ("system", "models") => {
                let cards = self.named_cards(&all);
                // the registered ones (downloaded), then the cards' that are not downloaded (added: no field renamed)
                let mut models = self.model_states(&all, &cards);
                for m in models.iter_mut() {
                    m["downloaded"] = json!(true);
                }
                for mut nd in self.not_downloaded_all() {
                    nd["state"] = json!("not_downloaded");
                    nd["enabled"] = json!(false);
                    models.push(nd);
                }
                Ok(json!({"models": models}))
            }
            ("system", "models.pull") => self.pull_start(&model()).map(|p| json!({"pull": p.view()})),
            ("system", "models.pulls") => Ok(json!({"pulls": self.pulls_view(), "auto_pull": self.auto_pull()})),
            ("system", "models.pull.cancel") => self.pull_cancel(&model()),
            ("system", "cards") => Ok(self.cards_view(&all)),
            ("system", "enable") => self.enable(&model(), b.get("gpus")).map(|t| json!({"model": model(), "title": t, "enabled": true})),
            ("system", "gpus") => self.set_gpus(&model(), b.get("gpus")),
            ("system", "disable") => self.disable(&model(), b["force"] == true).map(|_| json!({"model": model(), "enabled": false})),
            ("system", "load") if b["wait"] == true => self.load(&model()).map(|_| json!({"model": model(), "loaded": true})),
            ("system", "load") => self.load_async(&model()).map(|_| json!({"model": model(), "loading": true})),
            ("system", "unload") => self.unload(&model()).map(|_| json!({"model": model(), "loaded": false})),
            ("system", "config") => Ok(self.config()),
            ("system", "configSet") => self.config_set(b),
            ("system", "restart") => self.restart(&s(b, "service")),
            ("system", "maintenance") => self.maintenance_set(b),
            ("system", "drives") => {
                let cards = self.named_cards(&all);
                Ok(self.drives_view(&all, &self.model_states(&all, &cards)))
            }
            ("system", "drive.measure") => self.drive_measure(&s(b, "drive")),
            ("system", "move") => self.model_move(&model(), &s(b, "drive")),
            ("system", "move.cancel") => self.move_cancel(),
            (k, "describePromptGuideMD") => {
                let e = all.iter().find(|x| x["id"] == model().as_str()).ok_or_else(|| (404, format!("no model {}", model())))?;
                if registry::kind_of(e) != k {
                    return Err((400, format!("{} is a {} model", model(), registry::kind_of(e))));
                }
                // the model card's guide (model-cards/), else its architecture's or its kind's
                let md = self.o.cfg.guide(e).ok_or_else(|| (404, format!("no prompt guide for {} (its model card's `guide`)", model())))?;
                Ok(json!({"model": model(), "arch": e["arch"], "kind": k, "markdown": md}))
            }
            ("llm", "chat") => {
                let t = self.chat_target(b["model"].as_str())?;
                let mut body = b.clone();
                body["stream"] = json!(false);
                let keep = body.as_object_mut().and_then(|o| o.remove("keep_up_s"));
                let r = relay(&t, "POST", "/v1/chat/completions", Some(&body), 1800);
                // kept from the end of the answer (the model that answered)
                if let Some(k) = keep.filter(|k| !k.is_null()) {
                    let secs = k.as_f64().filter(|x| *x >= 0.0).map_or(crate::switch::KEEP_UP_S, |x| (x as u64).min(86400));
                    if let Some(id) = self.chat_loaded(&self.models()) {
                        self.keep_chat(&id, secs);
                    }
                }
                r
            }
            ("llm", "requests") => {
                // the chat server's control socket (its /server routes are not on its port)
                let dir = self.o.cfg.get("NS_SOCKET_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
                    std::path::PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into())).join("nextsycl")
                });
                let v = call_for(&crate::http::Target::Unix(dir.join("nextsycl.sock")), "GET", "/server/requests", None, 5)
                    .map_err(|e| (502, format!("the chat server does not answer: {e}")))?;
                let keep = ["id", "model", "state", "via", "started", "ended", "finish", "prompt_tokens", "reused", "source", "read", "read_tok_s", "read_seconds",
                            "generated", "max_tokens", "tok_s", "generate_seconds", "drafts", "energy_j", "avg_watts"];
                let slim = |r: &Value| Value::from(keep.iter().filter_map(|k| r.get(*k).map(|x| (k.to_string(), x.clone()))).collect::<serde_json::Map<_, _>>());
                let list = |k: &str| v[k].as_array().into_iter().flatten().map(slim).collect::<Vec<_>>();
                Ok(json!({"active": list("active"), "done": list("done").into_iter().take(10).collect::<Vec<_>>()}))
            }
            ("llm", "models") => {
                let loaded = self.chat_loaded(&all);
                let list: Vec<Value> = all.iter().filter(|m| registry::kind_of(m) == "llm" && m["enabled"] != false)
                    .map(|m| json!({"id": m["id"], "title": m["title"], "gpus": m["gpus"], "loaded": loaded.as_deref() == m["id"].as_str()})).collect();
                Ok(json!({"models": list}))
            }
            ("image", "ocr") => {
                // a reading model: the one named, else the loaded or first enabled one whose architecture reads
                let reads = |m: &Value| s(m, "arch").contains("ocr");
                let id = match b["model"].as_str().filter(|x| !x.is_empty()) {
                    Some(_) => self.pick("image", b, &all)?,
                    None => {
                        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "image" && m["enabled"] != false && reads(m)).collect();
                        self.loaded_of("image").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
                            .ok_or_else(|| (409, "no OCR model is enabled (system.enable unlimited-ocr-q8)".to_string()))?
                    }
                };
                if !all.iter().any(|m| m["id"] == id.as_str() && reads(m)) {
                    return Err((400, format!("{id} makes pictures; image.ocr needs a model that reads them")));
                }
                let t = self.ensure(&id)?;
                let mut body = b.clone();
                body["model"] = json!(id);
                relay(&t, "POST", "/v1/ocr", Some(&body), 3600)
            }
            ("image", "generate") | ("image", "edit") => {
                let id = self.pick("image", b, &all)?;
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                if body.get("response_format").is_none() {
                    body["response_format"] = json!("url");
                }
                let path = if m.name == "edit" { "/v1/images/edits" } else { "/v1/images/generations" };
                relay(&t, "POST", path, Some(&body), 3600)
            }
            ("audio", "transcribe") => {
                let id = self.recogniser(b, &all)?;
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                relay(&t, "POST", "/v1/audio/transcriptions", Some(&body), 7200)
            }
            ("audio", "generate" | "speak") => {
                let mut b = b.clone();
                if m.name == "speak" {
                    b["model"] = json!(self.speech_model(&b, &all)?);
                } else if b["model"].as_str().is_none_or(|x| x.is_empty()) {
                    // a song: the loaded or first enabled audio model that makes songs (not a speech model, not a recogniser)
                    let songs: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && !m["speech"].is_array()
                        && s(m, "arch") != "whisper").collect();
                    if let Some(id) = self.loaded_of("audio").into_iter().find(|l| songs.iter().any(|m| m["id"] == l.as_str())).or_else(|| songs.first().map(|m| s(m, "id"))) {
                        b["model"] = json!(id);
                    }
                }
                let b = &b;
                let id = self.pick("audio", b, &all)?;
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                body["response_format"] = json!("url");
                relay(&t, "POST", "/v1/audio/speech", Some(&body), 3600)
            }
            ("model3d", "tool") => self.mesh_tool(b),
            ("model3d", "generate" | "rig") => {
                // a model that does it: the one named, else the loaded or first enabled one whose card says so (rig: a
                // rigger or an animator)
                let want = if m.name == "rig" { "rig" } else { "generate" };
                let able = |m: &Value| m["does"].as_array().is_none_or(|a| a.iter().any(|x| x == want || (want == "rig" && x == "animate")));
                let id = match b["model"].as_str().filter(|x| !x.is_empty()) {
                    Some(_) => self.pick("model3d", b, &all)?,
                    None => {
                        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "model3d" && m["enabled"] != false && able(m)).collect();
                        self.loaded_of("model3d").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
                            .ok_or_else(|| (409, format!("no enabled 3D model does {want} (system.models: capabilities)")))?
                    }
                };
                if !all.iter().any(|m| m["id"] == id.as_str() && able(m)) {
                    return Err((400, format!("{id} does not {want} (model3d.{})", if want == "rig" { "generate" } else { "rig" })));
                }
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                body["response_format"] = json!("url");
                relay(&t, "POST", &format!("/v1/model3d/{}", m.name), Some(&body), 7200)
            }
            ("image" | "audio" | "model3d", "info") => {
                let id = self.pick(m.kind, b, &all)?;
                let e = all.iter().find(|x| x["id"] == id.as_str()).cloned().unwrap_or(Value::Null);
                // what a request may send, known without the model loaded
                let limits = if m.kind == "image" {
                    json!({"steps": super::precheck::steps_of(&e), "max_pictures": 8, "max_picture_pixels": crate::image::MAX_PICTURE_PIXELS,
                           "max_request_bytes": 256u64 << 20, "size_multiple": 16})
                } else {
                    json!({"max_request_bytes": 256u64 << 20})
                };
                let mut v = match call_for(&t, "GET", "/api/info", None, 5) {
                    Ok(i) if s(&i, "model") == id => json!({"loaded": true, "model": id, "info": i}),
                    _ => json!({"loaded": false, "model": id, "title": e["title"], "defaults": e["defaults"], "voices": e["voices"], "languages": e["languages"],
                                "note": "not loaded: the server's own details (info) come once it is (system.load); steps and the limits are known now"}),
                };
                if let (Some(o), Some(l)) = (v.as_object_mut(), limits.as_object()) {
                    o.extend(l.clone());
                }
                Ok(v)
            }
            ("image" | "audio" | "model3d", "progress") => Ok(call_for(&t, "GET", "/api/progress", None, 5).unwrap_or(json!({"busy": false, "loaded": false}))),
            ("image" | "audio" | "model3d", "history") => Ok(call_for(&t, "GET", "/api/history", None, 10).unwrap_or(json!({"data": []}))),
            ("image" | "audio" | "model3d", "status") => self.job_status(m.kind, b),
            ("image" | "audio" | "model3d", "jobs") => self.job_list(m.kind, b),
            ("image" | "audio" | "model3d", "queue.list") => Ok(self.queue_list(m.kind)),
            ("image" | "audio" | "model3d", "queue.move") => self.queue_move(m.kind, b),
            ("image" | "audio" | "model3d", "queue.pause") => self.queue_hold(m.kind, b, true),
            ("image" | "audio" | "model3d", "queue.resume") => self.queue_hold(m.kind, b, false),
            ("image" | "audio" | "model3d", "upload") => self.upload(m.kind, b),
            ("image", "canvases.list") => self.image_canvases(b, &all),
            ("image" | "audio" | "model3d", "cancel") => self.job_cancel(m.kind, b),
            ("audio", "voices.list") => {
                // the enabled models that speak a saved voice: the ones that clone
                let speakers: Vec<Value> = all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false)
                    .filter(|m| m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "clone"))).map(|m| m["id"].clone()).collect();
                let lib = self.voices();
                let owner = s(b, "owner");
                let list: Vec<Value> = lib.list().into_iter().filter(|v| owner.is_empty() || v["owner"] == owner.as_str()).map(|mut v| {
                    let name = s(&v, "name");
                    v["sample"] = json!(format!("http://nextsycl/v1/audio/voices/{name}.wav"));
                    // ready: its sample is there and reads, and an enabled model can speak it now
                    let reads = lib.sample_path(&name).is_some_and(|f| nextsycl_audio::wav_info(&f).is_ok_and(|(_, secs)| secs > 0.0));
                    // a voice that does not say what its sample says: not the models that clone only with the transcript
                    let said = v["text"].as_str().is_some_and(|t| !t.trim().is_empty());
                    let mut speakers: Vec<Value> = speakers.iter().filter(|id| said || !needs_text(&all, id)).cloned().collect();
                    // and the models without an encoder it is enrolled for (experimental: Voxtral TTS)
                    speakers.extend(all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && enrolled_for(&lib, &name, m))
                        .map(|m| m["id"].clone()));
                    v["ready"] = json!(reads && !speakers.is_empty());
                    v["models"] = json!(speakers);
                    if !reads {
                        v["problem"] = json!("its sample is missing or does not read: add it again");
                    } else if speakers.is_empty() {
                        v["problem"] = json!("no enabled speech model clones a voice (system.enable qwen3-tts-base or voxcpm2)");
                    }
                    v
                }).collect();
                Ok(json!({"voices": list}))
            }
            ("audio", "voices.remove") => {
                self.owned(&s(b, "name"), b)?;
                self.voices().remove(&s(b, "name")).map(|_| json!({"removed": s(b, "name")})).map_err(|e| (404, e))
            }
            ("audio", "voices.add") => {
                let bytes = http::unbase64(&s(b, "sample")).map_err(|e| (400, format!("sample: {e}")))?;
                let o = |k: &str| b[k].as_str().filter(|x| !x.trim().is_empty());
                if b["replace"] == true {
                    self.owned(&s(b, "name"), b)?;
                }
                let mut v = self.voices().add(&s(b, "name"), &bytes, o("text"), o("description"), o("language"), "clone", "a recording", b["replace"] == true)
                    .map_err(|e| (400, e))?;
                if let Some(owner) = o("owner") {
                    v = self.voices().tag(&s(b, "name"), "owner", json!(owner)).map_err(|e| (500, e))?;
                }
                Ok(json!({"voice": v, "sample": format!("http://nextsycl/v1/audio/voices/{}.wav", s(b, "name"))}))
            }
            ("audio", "voices.design") => self.design_voice(b, &all),
            ("audio", "voices.enroll") => self.enroll_voice(b, &all),
            ("video", "upload") => {
                let data = http::unbase64(&s(b, "data")).map_err(|e| (400, format!("data: {e}")))?;
                let r = self.studio_upload(&s(b, "name"), &data)?;
                Ok(r)
            }
            ("video", method) => {
                let v = call_for(&t, "POST", &format!("/rpc/{method}"), Some(b), 120).map_err(|e| (502, e))?;
                Ok(v.get("result").cloned().unwrap_or(v))
            }
            _ => Err((404, format!("no method {name}"))),
        }
    }

    /// `model3d.tool`: a CPU mesh tool run here, its result saved beside the meshes the 3D servers make
    fn mesh_tool(&self, b: &Value) -> Result<Value, Fail> {
        use nextsycl_model3d::mesh::tools;
        let dir = self.out_dir("model3d");
        let bytes = match (b["mesh"].as_str().filter(|x| !x.is_empty()), b["file"].as_str().filter(|x| !x.is_empty())) {
            (Some(m), _) => http::unbase64(m).map_err(|e| (400, format!("mesh: {e}")))?,
            (None, Some(f)) => {
                let f = f.rsplit('/').next().unwrap_or(f);
                let plain = crate::files::plain(f).filter(|x| x.ends_with(".glb")).ok_or_else(|| (400, format!("file {f}: a GLB's name")))?;
                std::fs::read(dir.join(plain)).map_err(|_| (404, format!("file {f}: no GLB of that name here")))?
            }
            _ => return Err((400, "mesh (GLB or OBJ, base64 / a data: URL / upload:<name>) or file (a GLB made here): required".into())),
        };
        let m = nextsycl_model3d::read_mesh(&bytes).map_err(|e| (400, format!("mesh: {e}")))?;
        let op = s(b, "op");
        let before = m.triangles();
        let t0 = std::time::Instant::now();
        let num = |k: &str| b[k].as_f64();
        let out = match op.as_str() {
            "decimate" => {
                let faces = num("faces").ok_or((400, "faces: required by decimate".to_string()))? as usize;
                let mat = m.material.clone();
                let mut d = tools::decimate(&m, faces.max(4));
                d.material = mat;
                d.compute_normals();
                d
            }
            "remesh" => {
                let mut r = tools::remesh(&m, num("edge").map(|x| x as f32), num("passes").map_or(5, |x| x.clamp(1.0, 20.0) as usize));
                r.compute_normals();
                r
            }
            "uv" => {
                let size = num("size").map_or(2048, |x| x.clamp(64.0, 8192.0) as u32);
                let mut u = tools::unwrap(&m, size, 4);
                if m.material.as_ref().is_some_and(|x| x.base_color.is_some()) || !m.colors.is_empty() {
                    u.material = Some(tools::bake(&u, &m, size).map_err(|e| (400, e))?);
                }
                if u.normals.is_empty() {
                    u.compute_normals();
                }
                u
            }
            "convert" => m,
            other => return Err((400, format!("op {other:?}: decimate, remesh, uv or convert"))),
        };
        let glb = nextsycl_model3d::mesh::glb::write(&out).map_err(|e| (500, e))?;
        std::fs::create_dir_all(&dir).map_err(|e| (500, format!("{}: {e}", dir.display())))?;
        let created = now() as u64;
        let name = format!("{created}-tool-{op}-{}.glb", self.next.lock().map(|mut n| { *n += 1; *n }).unwrap_or(0));
        let path = dir.join(&name);
        std::fs::write(&path, &glb).map_err(|e| (500, format!("{name}: {e}")))?;
        let took = t0.elapsed().as_secs_f64();
        let meta = json!({"model": "tool", "arch": format!("tool:{op}"), "prompt": op, "triangles": out.triangles(), "before": before, "took": took, "created": created,
                          "textured": out.material.as_ref().is_some_and(|x| x.base_color.is_some()), "rigged": out.skin.as_ref().map(|s| s.joints.len()),
                          "animated": out.animation.as_ref().map(|a| a.frames.len())});
        let _ = std::fs::write(path.with_extension("json"), serde_json::to_vec_pretty(&meta).unwrap_or_default());
        let mut e = crate::model3d::entry(&path).unwrap_or(meta);
        e["url"] = json!(format!("http://nextsycl/v1/model3d/files/{name}"));
        Ok(json!({"created": created, "data": [e]}))
    }

    /// A kind's output directory (the servers' NS_IMAGE_OUT / NS_AUDIO_OUT / NS_MODEL3D_OUT)
    fn out_dir(&self, kind: &str) -> std::path::PathBuf {
        let (key, def) = match kind {
            "image" => ("NS_IMAGE_OUT", "images"),
            "model3d" => ("NS_MODEL3D_OUT", "model3d"),
            _ => ("NS_AUDIO_OUT", "audio"),
        };
        std::path::PathBuf::from(self.o.cfg.get(key).unwrap_or_else(|| format!("{}/.local/share/nextsycl/{def}", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))))
    }

    /// `<kind>.upload {name, data}`: a picture (checked, kept as sent) or a recording (any format voices take, kept as
    /// WAV) under <output>/uploads, named up_<unix>_<name>
    fn upload(&self, kind: &str, b: &Value) -> Result<Value, Fail> {
        let data = http::unbase64(&s(b, "data")).map_err(|e| (400, format!("data: {e}")))?;
        if data.is_empty() {
            return Err((400, "data: empty".into()));
        }
        let raw = s(b, "name");
        let clean: String = raw.rsplit('/').next().unwrap_or("").chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).take(60).collect();
        let ext = clean.rsplit_once('.').map(|(_, e)| e.to_lowercase()).unwrap_or_default();
        let (stem, bytes) = if kind == "image" {
            if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp") {
                return Err((400, format!("name {raw:?}: a picture's name (png, jpg, jpeg, webp)")));
            }
            crate::image::picture(&data, "data").map_err(|e| (400, e))?;
            (clean.clone(), data)
        } else if kind == "model3d" {
            match ext.as_str() {
                "png" | "jpg" | "jpeg" | "webp" => {
                    nextsycl_model3d::read_picture(&data).map_err(|e| (400, format!("data: {e}")))?;
                }
                "glb" | "obj" => {
                    nextsycl_model3d::read_mesh(&data).map_err(|e| (400, format!("data: {e}")))?;
                }
                _ => return Err((400, format!("name {raw:?}: a picture's or a mesh's name (png, jpg, jpeg, webp, glb, obj)"))),
            }
            (clean.clone(), data)
        } else {
            if !matches!(ext.as_str(), "wav" | "mp3" | "flac" | "ogg" | "m4a") {
                return Err((400, format!("name {raw:?}: a recording's name (wav, mp3, flac, ogg, m4a)")));
            }
            let (samples, rate) = nextsycl_audio::decode_audio(&data).map_err(|e| (400, format!("data: {e}")))?;
            let a = nextsycl_audio::Audio { rate, channels: 1, samples };
            (format!("{}.wav", clean.rsplit_once('.').map_or(clean.as_str(), |x| x.0)), a.wav(&[]))
        };
        let dir = self.out_dir(kind).join("uploads");
        std::fs::create_dir_all(&dir).map_err(|e| (500, format!("{}: {e}", dir.display())))?;
        let name = format!("up_{}_{stem}", now() as u64);
        std::fs::write(dir.join(&name), &bytes).map_err(|e| (500, format!("{name}: {e}")))?;
        let mut r = crate::files::stat(&dir.join(&name)).map_err(|e| (500, e))?;
        r["name"] = json!(name);
        r["ref"] = json!(format!("upload:{name}"));
        Ok(r)
    }

    /// The fields of a request that may name an upload, by method: (field, is a list)
    fn upload_fields(name: &str) -> (&'static str, &'static [(&'static str, bool)]) {
        match name {
            "image.generate" => ("image", &[("image", false), ("init_image", false)]),
            "image.edit" => ("image", &[("images", true), ("image", false)]),
            "image.ocr" => ("image", &[("images", true)]),
            "audio.speak" => ("audio", &[("ref_audio", false), ("ref_audios", true)]),
            "audio.transcribe" => ("audio", &[("audio", false)]),
            "audio.voices.add" => ("audio", &[("sample", false)]),
            "model3d.generate" => ("model3d", &[("image", false), ("images", true)]),
            "model3d.rig" | "model3d.tool" => ("model3d", &[("mesh", false)]),
            _ => ("", &[]),
        }
    }

    /// The request with every "upload:<name>" read in as base64 (none named: None); an upload not here is 404
    fn uploads_in(&self, method: &str, b: &Value) -> Result<Option<Value>, Fail> {
        let (kind, fields) = Self::upload_fields(method);
        // a picture of a list may be {image, side} (model3d.generate's views): its image named
        let named = |v: &Value| v.as_str().or_else(|| v["image"].as_str()).and_then(|x| x.strip_prefix("upload:")).map(str::to_string);
        if !fields.iter().any(|(f, list)| if *list { b[*f].as_array().is_some_and(|a| a.iter().any(|v| named(v).is_some())) } else { named(&b[*f]).is_some() }) {
            return Ok(None);
        }
        let dir = self.out_dir(kind).join("uploads");
        let read = |what: String, n: &str| -> Result<Value, Fail> {
            let plain = crate::files::plain(n).filter(|x| x.starts_with("up_")).ok_or_else(|| (400, format!("{what}: upload:{n} is not an upload's name ({kind}.upload answers it)")))?;
            let bytes = std::fs::read(dir.join(plain)).map_err(|_| (404, format!("{what}: no upload {plain} here ({kind}.upload it, or it was deleted)")))?;
            Ok(json!(http::base64(&bytes)))
        };
        let mut out = b.clone();
        for (f, list) in fields {
            if *list {
                if let Some(a) = out[*f].as_array_mut() {
                    for (i, v) in a.iter_mut().enumerate() {
                        if let Some(n) = named(v) {
                            if v.is_object() {
                                v["image"] = read(format!("{f}[{i}].image"), &n)?;
                            } else {
                                *v = read(format!("{f}[{i}]"), &n)?;
                            }
                        }
                    }
                }
            } else if let Some(n) = named(&out[*f]) {
                out[*f] = read(f.to_string(), &n)?;
            }
        }
        Ok(Some(out))
    }

    /// A saved voice that has an owner may be replaced or removed only by a request naming that owner
    fn owned(&self, name: &str, b: &Value) -> Result<(), Fail> {
        let Some(v) = self.voices().get(name) else { return Ok(()) };
        match v["owner"].as_str() {
            Some(o) if o != s(b, "owner").trim() => Err((409, format!("owner: voice {name} belongs to {o:?} - name that owner to replace or remove it"))),
            _ => Ok(()),
        }
    }

    /// system.cards: each GPU in one neutral shape - what any engine on any box can report
    fn cards_view(&self, all: &[Value]) -> Value {
        let mut cards = self.named_cards(all);
        let models = self.model_states(all, &cards);
        self.vouch(&mut cards, &models);
        let list: Vec<Value> = cards.as_array().into_iter().flatten().map(|c| {
            let total = c["vram_total_mb"].as_u64().unwrap_or(0);
            let used = c["vram_used_mb"].as_u64().unwrap_or(0);
            let procs: Vec<Value> = c["procs"].as_array().into_iter().flatten().map(|p| json!({"pid": p["pid"], "program": p["program"], "kind": p["kind"],
                "model": p["model_id"].as_str().or(p["model"].as_str()), "vram_mb": p["vram_mb"]})).collect();
            let models: Vec<Value> = c["procs"].as_array().into_iter().flatten().filter_map(|p| p["model_id"].as_str().map(|m| json!(m))).collect();
            json!({"gpu": c["index"], "name": c["name"], "bus": c["pci"], "vram_total_mb": total, "vram_used_mb": used, "vram_free_mb": total.saturating_sub(used),
                   "busy_pct": c["busy_pct"], "power_w": c["power_w"], "power_cap_w": c["power_cap_w"], "temp_c": c["temp_pkg"], "temp_vram_c": c["temp_vram"],
                   "models": models, "processes": procs, "vouched_mb": c["vouched_mb"], "overbooked": c["overbooked"],
                   "enabled_models": c["vouched"].as_array().into_iter().flatten().map(|v| v["model"].clone()).collect::<Vec<_>>()})
        }).collect();
        json!({"cards": list})
    }

    /// system.files.stat / files.delete: a picture or sound in its kind's output directory, or through the studio
    /// a clip or upload
    fn file_op(&self, op: &str, b: &Value) -> Result<Value, Fail> {
        let kind = s(b, "kind");
        let name = crate::files::plain(&s(b, "name")).map(str::to_string).ok_or_else(|| (400, "name: a plain file name (the last part of its URL)".to_string()))?;
        if kind == "video" {
            let (st, v) = http::call_status(&self.target("video"), "POST", &format!("/rpc/{op}"), Some(&json!({"name": name})), 120)
                .map_err(|e| (502, format!("the video studio does not answer: {e}")))?;
            if v["ok"] != true {
                return Err((st, s(&v["error"], "message")));
            }
            let mut r = v["result"].clone();
            if op == "files.stat" {
                r["url"] = json!(format!("http://nextsycl/v1/video/files/{name}"));
            }
            return Ok(r);
        }
        let sub = match kind.as_str() {
            "image" => "images",
            "audio" => "audio",
            "model3d" => "model3d",
            _ => return Err((400, "kind: image, audio, video or model3d".into())),
        };
        // an upload (up_...) is kept under uploads/
        let upload = name.starts_with("up_");
        let dir = if upload { self.out_dir(&kind).join("uploads") } else { self.out_dir(&kind) };
        let path = dir.join(&name);
        // what the servers make, not the saved voices or anything else kept there
        let ext = name.rsplit('.').next().unwrap_or("").to_lowercase();
        let ok = match kind.as_str() {
            "image" => matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp"),
            // a mesh and the picture and settings beside it; an upload (a picture or a mesh)
            "model3d" => matches!(ext.as_str(), "glb" | "obj" | "png" | "jpg" | "jpeg" | "webp" | "json"),
            _ => ext == "wav",
        };
        if !ok || !path.is_file() {
            return Err((404, format!("no {kind} file {name}")));
        }
        if op == "files.stat" {
            let mut r = crate::files::stat(&path).map_err(|e| (404, e))?;
            r["url"] = if upload { Value::Null } else { json!(format!("http://nextsycl/v1/{sub}/files/{name}")) };
            if upload {
                r["ref"] = json!(format!("upload:{name}"));
            }
            return Ok(r);
        }
        std::fs::remove_file(&path).map_err(|e| (500, format!("{name}: {e}")))?;
        Ok(json!({"deleted": [name]}))
    }

    /// image.canvases.list: the model's tested sizes, with the time measured on the pictures kept
    fn image_canvases(&self, b: &Value, all: &[Value]) -> Result<Value, Fail> {
        let id = self.pick("image", b, all)?;
        let hist = call_for(&self.file_target("image"), "GET", "/api/history", None, 10).unwrap_or(json!({"data": []}));
        let def = all.iter().find(|m| m["id"] == id.as_str()).and_then(|m| m["defaults"]["size"].as_str().map(str::to_string)).unwrap_or_else(|| "1024x1024".into());
        let list: Vec<Value> = IMAGE_CANVASES.iter().map(|(w, h, a, note)| {
            // seconds per step of the pictures of this size (several to a request: its time over them)
            let made: Vec<f64> = hist["data"].as_array().into_iter().flatten()
                .filter(|e| e["width"] == *w && e["height"] == *h && e["kind"] != "ocr")
                .filter_map(|e| Some(e["seconds"].as_f64()? / e["steps"].as_f64().filter(|s| *s > 0.0)?)).collect();
            let size = format!("{w}x{h}");
            json!({"size": size, "width": w, "height": h, "aspect": a, "note": note, "mpx": ((w * h) as f64 / 1e4).round() / 100.0, "default": size == def,
                   "seconds": if made.is_empty() { Value::Null } else { json!((made.iter().sum::<f64>() / made.len() as f64 * 100.0).round() / 100.0) },
                   "made": made.len()})
        }).collect();
        Ok(json!({"model": id, "rule": "any width and height that are multiples of 16; the model was trained up to about 1.8 megapixels (1328 x 1328) - larger works, slower and less sure",
                  "canvases": list}))
    }

    /// A file into the studio's output (its upload route takes the raw bytes)
    fn studio_upload(&self, name: &str, data: &[u8]) -> Result<Value, Fail> {
        use std::io::{Read, Write};
        let mut c = Conn::connect(&self.target("video")).map_err(|e| (502, format!("the studio does not answer: {e}")))?;
        let q: String = name.chars().map(|ch| if ch.is_ascii_alphanumeric() || "._-".contains(ch) { ch.to_string() } else { format!("%{:02X}", ch as u32 & 0xff) }).collect();
        write!(c, "POST /api/upload?name={q} HTTP/1.1\r\nHost: studio\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len())
            .and_then(|_| c.write_all(data)).map_err(|e| (502, e.to_string()))?;
        let mut out = Vec::new();
        c.read_to_end(&mut out).map_err(|e| (502, e.to_string()))?;
        let text = String::from_utf8_lossy(&out);
        let v: Value = text.split_once("\r\n\r\n").and_then(|(_, b)| serde_json::from_str(b).ok()).unwrap_or(Value::Null);
        if v["ok"] == true { Ok(json!({"name": v["name"]})) } else { Err((400, v["error"].as_str().unwrap_or("the upload failed").to_string())) }
    }

    /// The OpenAI-compatible routes: to the switcher, or to the kind's server for the model the request names
    fn openai(self: &Arc<Self>, s: &mut Conn, req: &Request) -> Result<(), Fail> {
        let all = self.models();
        let on = self.kinds_on(&all);
        let need = |k: &str| if on.contains(&k) { Ok(()) } else { Err((409u16, format!("no {k} model is enabled"))) };
        let pass = |s: &mut Conn, t: &crate::http::Target| http::forward(s, t, req).map_err(|e| (502, e));
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/models") => {
                let mut data: Vec<Value> = Vec::new();
                for m in all.iter().filter(|m| m["enabled"] != false && matches!(registry::kind_of(m), "llm" | "image" | "audio")) {
                    data.push(json!({"id": m["id"], "object": "model", "owned_by": "nextsycl", "created": 0, "kind": registry::kind_of(m), "name": m["title"]}));
                }
                http::respond(s, 200, &json!({"object": "list", "data": data}));
                Ok(())
            }
            ("POST", "/v1/chat/completions") => {
                let model = serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string));
                // a card's model not downloaded is said as such (chat_target refuses or fetches it), not "none enabled"
                if !model.as_deref().is_some_and(|m| self.not_downloaded(m).is_some()) {
                    need("llm")?;
                }
                let t = self.chat_target(model.as_deref())?;
                let r = pass(s, &t);
                if let Some(secs) = crate::switch::keep_up_s(&req.head) {
                    if let Some(id) = self.chat_loaded(&self.models()) {
                        self.keep_chat(&id, secs);
                    }
                }
                r
            }
            ("POST", p @ ("/v1/images/generations" | "/v1/images/edits" | "/v1/audio/speech")) => {
                let kind = if p.starts_with("/v1/images") { "image" } else { "audio" };
                // the model the body names (a JSON field, or a multipart part)
                let ctype = http::header(&req.head, "content-type").unwrap_or_default();
                let named = if let Some(b) = ctype.split("boundary=").nth(1) {
                    http::multipart(&req.body, b.trim_matches('"')).ok()
                        .and_then(|ps| ps.into_iter().find(|p| p.name == "model").map(|p| String::from_utf8_lossy(&p.data).trim().to_string()))
                } else {
                    serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string))
                };
                // a card's model not downloaded: refused (409 model_not_downloaded), or with NS_AUTO_PULL fetched first
                let fetched = named.as_deref().filter(|n| self.not_downloaded(n).is_some()).map(str::to_string);
                if let Some(n) = &fetched {
                    self.need_downloaded(n, None, true)?;
                }
                let all = if fetched.is_some() { self.models() } else { all.clone() };
                if fetched.is_none() {
                    need(kind)?;
                }
                // OpenAI's own names (dall-e-3, tts-1 ...) mean "the kind's model"
                let named = named.filter(|n| all.iter().any(|m| m["id"] == n.as_str()));
                // speech with a voice or a recording and no model of ours named: the speech model for it
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                let has = |k: &str| body[k].as_str().is_some_and(|v| !v.is_empty());
                let id = if kind == "audio" && named.is_none() && (has("voice") || has("ref_audio")) {
                    self.speech_model(&body, &all)?
                } else {
                    self.pick(kind, &json!({"model": named}), &all)?
                };
                let t = self.ensure(&id)?;
                pass(s, &t)
            }
            ("POST", "/v1/audio/transcriptions") => {
                need("audio")?;
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                // OpenAI's whisper-1 (or no name): the recogniser here
                let named = body["model"].as_str().filter(|n| all.iter().any(|m| m["id"] == *n));
                let id = self.recogniser(&json!({"model": named}), &all)?;
                let t = self.ensure(&id)?;
                pass(s, &t)
            }
            ("GET", p) if p.starts_with("/v1/images/files/") => pass(s, &self.file_target("image")),
            ("GET", p) if p.starts_with("/v1/audio/files/") => pass(s, &self.file_target("audio")),
            ("GET", p) if p.starts_with("/v1/model3d/files/") => {
                // from the directory (a tool's result needs no 3D server running)
                let name = p["/v1/model3d/files/".len()..].split('?').next().unwrap_or("");
                let kind = match name.rsplit('.').next() {
                    Some("glb") => "model/gltf-binary",
                    Some("png") => "image/png",
                    Some("json") => "application/json",
                    _ => return Err((404, format!("no file {name}"))),
                };
                let plain = crate::files::plain(name).ok_or_else(|| (404, format!("no file {name}")))?;
                let b = std::fs::read(self.out_dir("model3d").join(plain)).map_err(|_| (404, format!("no file {name}")))?;
                http::respond_bytes(s, 200, kind, &b, "Cache-Control: max-age=31536000, immutable\r\n");
                Ok(())
            }
            ("POST", p @ ("/v1/model3d/generate" | "/v1/model3d/rig")) => {
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                // a card's 3D model not downloaded: refused, or with NS_AUTO_PULL fetched first
                let fetched = body["model"].as_str().filter(|n| self.not_downloaded(n).is_some()).map(str::to_string);
                if let Some(n) = &fetched {
                    self.need_downloaded(n, None, true)?;
                } else {
                    need("model3d")?;
                }
                let all = if fetched.is_some() { self.models() } else { all.clone() };
                let named = body["model"].as_str().filter(|n| all.iter().any(|m| m["id"] == *n)).map(str::to_string);
                let want = if p.ends_with("/rig") { "rig" } else { "generate" };
                let id = match named {
                    Some(n) => self.pick("model3d", &json!({"model": n}), &all)?,
                    None => all.iter().filter(|m| registry::kind_of(m) == "model3d" && m["enabled"] != false)
                        .find(|m| m["does"].as_array().is_none_or(|a| a.iter().any(|x| x == want || (want == "rig" && x == "animate"))))
                        .map(|m| m["id"].as_str().unwrap_or("").to_string()).ok_or_else(|| (409u16, format!("no enabled 3D model does {want}")))?,
                };
                let t = self.ensure(&id)?;
                pass(s, &t)
            }
            ("GET", p) if p.starts_with("/v1/video/files/") => {
                // a clip in the studio's output (its /out/ route; a Range header passed on for seeking)
                need("video")?;
                let name = p["/v1/video/files/".len()..].split('?').next().unwrap_or("");
                let still = name.ends_with(".preview.png") || name.ends_with(".last.png");
                if name.is_empty() || name.contains('/') || !(name.ends_with(".mp4") || still) {
                    return Err((404, format!("no clip {name}")));
                }
                let head = String::from_utf8_lossy(&req.head).into_owned();
                let range = head.split("\r\n").find(|l| l.split(':').next().is_some_and(|k| k.trim().eq_ignore_ascii_case("range")));
                let up = format!("GET /out/{name} HTTP/1.1\r\nHost: studio\r\n{}Connection: close\r\n\r\n", range.map(|r| format!("{r}\r\n")).unwrap_or_default());
                let req = Request { method: "GET".into(), path: format!("/out/{name}"), target: format!("/out/{name}"), head: up.into_bytes(),
                                    origin: None, host: None, body: Vec::new() };
                http::forward(s, &self.target("video"), &req).map_err(|e| (502, e))
            }
            ("GET", p) if p.starts_with("/v1/video/thumbs/") => {
                need("video")?;
                let name = &p["/v1/video/thumbs/".len()..];
                if name.is_empty() || name.contains('/') || !name.ends_with(".webp") {
                    return Err((404, format!("no still {name}")));
                }
                let req = Request { method: "GET".into(), path: format!("/thumb/{name}"), target: format!("/thumb/{name}"),
                                    head: format!("GET /thumb/{name} HTTP/1.1\r\nHost: studio\r\nConnection: close\r\n\r\n").into_bytes(),
                                    origin: None, host: None, body: Vec::new() };
                http::forward(s, &self.target("video"), &req).map_err(|e| (502, e))
            }
            ("GET", p) if p.starts_with("/v1/audio/voices/") => {
                let name = p["/v1/audio/voices/".len()..].trim_end_matches(".wav");
                match self.voices().sample_path(name).and_then(|f| std::fs::read(f).ok()) {
                    Some(b) => {
                        http::respond_bytes(s, 200, "audio/wav", &b, "Cache-Control: no-cache\r\n");
                        Ok(())
                    }
                    None => Err((404, format!("no saved voice {name}"))),
                }
            }
            (m, p) => Err((404, format!("no route {m} {p} (GET /openapi.yml)"))),
        }
    }

    /// The OpenAPI document of what the enabled kinds offer. Open (GET /openapi.yml needs no token), so it carries
    /// none: `<token>` in the examples; the page's own copy (`token`) has it filled in
    pub(crate) fn openapi(&self, req: &Request, token: bool) -> Value {
        let tok = if token { self.token.as_str() } else { "<token>" };
        let all = self.models();
        let on = self.kinds_on(&all);
        let host = req.host.clone().unwrap_or_else(|| "localhost:8000".into());
        let ids = |k: &str| -> Vec<Value> { all.iter().filter(|m| registry::kind_of(m) == k && m["enabled"] != false).map(|m| m["id"].clone()).collect() };
        let envelope = json!({"type": "object", "properties": {
            "ok": {"type": "boolean"}, "result": {"description": "the method's answer"},
            "error": {"type": "object", "required": ["code", "message", "retryable", "status"], "description": "ok false: everything about the failure (the HTTP status is 200 all the same)", "properties": {
                "code": {"type": "string", "description": "the convention to branch on (the list in the description above): invalid_request, invalid_params, unauthorized, not_found, conflict, busy, not_running, model_not_downloaded, no_room, cancelled, unavailable, timeout, request_too_large, internal"},
                "message": {"type": "string", "description": "what was wrong, in words: the value, the limit or the nearest legal one, what to do"},
                "retryable": {"type": "boolean", "description": "true: the same call may work later unchanged (no_room, unavailable, timeout, busy); false: change the request"},
                "status": {"type": "integer", "description": "the code's number by convention (400, 404, 409, 499, 502, 503 ...): in the body, not the HTTP status, so another transport carries it"},
                "method": {"type": "string", "nullable": true, "description": "the method called"},
                "field": {"type": "string", "nullable": true, "description": "the request key at fault, when the message names one (steps, voice, images, size ...)"}}}}});
        let engines: Vec<String> = if on.contains(&"video") {
            call_for(&self.target("video"), "POST", "/rpc/engines.list", Some(&json!({})), 3).ok()
                .and_then(|v| v["result"]["engines"].as_array().map(|a| a.iter().filter_map(|e| e["quant"].as_str().map(str::to_string)).collect()))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut paths = Map::new();
        for m in METHODS.iter().filter(|m| on.contains(&m.kind)) {
            let mut props = Map::new();
            let mut required = Vec::new();
            for (f, ty, d, req) in m.fields {
                let mut p = match *ty {
                    t if t.starts_with('{') => serde_json::from_str(t).unwrap_or_else(|e| panic!("{}.{} {f}: {e}", m.kind, m.name)),
                    t if t.ends_with("[]") => json!({"type": "array", "items": {"type": &t[..t.len() - 2]}}),
                    t => json!({"type": t}),
                };
                if !d.is_empty() {
                    p["description"] = json!(d);
                }
                if *f == "model" && m.kind != "system" {
                    p["enum"] = Value::Array(ids(m.kind));
                }
                // the names the studio takes now: the daemon's denoisers, ComfyUI's samplers and schedules
                if m.kind == "video" && m.name == "generate" {
                    match *f {
                        "engine" if !engines.is_empty() => p["enum"] = json!(engines),
                        "sampler" => p["enum"] = json!(nextsycl_diffusion::samplers::NAMES),
                        "schedule" => p["enum"] = json!(nextsycl_diffusion::Schedule::ALL.iter().map(|s| s.name()).collect::<Vec<_>>()),
                        _ => {}
                    }
                }
                props.insert(f.to_string(), p);
                if *req {
                    required.push(json!(f));
                }
            }
            let mut schema = json!({"type": "object", "properties": props});
            if !required.is_empty() {
                schema["required"] = Value::Array(required);
            }
            let example: Value = serde_json::from_str(m.example).unwrap_or(json!({}));
            let route = format!("/rpc/{}.{}", m.kind, m.name);
            let op = format!("{}.{}", m.kind, m.name);
            let mut ok = envelope.clone();
            if let Some(r) = spec::result(&op) {
                ok["properties"]["result"] = serde_json::from_str(&r).unwrap_or_else(|e| panic!("{op} result: {e}"));
            }
            // every answer is HTTP 200 with the envelope; the errors it can carry are listed (x-errors, and in words)
            let mut errors = vec![json!({"status": 400, "code": "invalid_request", "when": "not POST, a query string, the body not a JSON object"}),
                                  json!({"status": 400, "code": "invalid_params", "when": "a value refused (error.message and error.field say which)"}),
                                  json!({"status": 401, "code": "unauthorized", "when": "no or a wrong token"})];
            let mut add = |st: u16, code: &str, when: String| match errors.iter_mut().find(|e| e["code"] == code && e["status"] == st) {
                // one entry a code: the method's own case first
                Some(e) => e["when"] = json!(format!("{when}; {}", e["when"].as_str().unwrap_or(""))),
                None => errors.push(json!({"status": st, "code": code, "when": when})),
            };
            for (st, code, when) in spec::errors(&op) {
                add(*st, code, when.to_string());
            }
            if m.kind != "system" {
                add(409, "conflict", format!("no {} model is enabled", m.kind));
            }
            if super::maint::starts_work(&op) {
                add(spec::MAINTENANCE.0, spec::MAINTENANCE.1, spec::MAINTENANCE.2.to_string());
            }
            let words: Vec<String> = errors.iter().map(|e| format!("{} {}: {}", e["code"].as_str().unwrap_or(""), e["status"], e["when"].as_str().unwrap_or(""))).collect();
            let responses = json!({"200": {"description": format!("{{\"ok\": true, \"result\": ...}}, or {{\"ok\": false, \"error\": ...}} - errors are HTTP 200 too. Its errors: {}", words.join("; ")),
                                           "content": {"application/json": {"schema": ok}}}});
            let curl = format!("curl -s http://{host}{route} -H 'Authorization: Bearer {}' -H 'Content-Type: application/json' -d '{}'",
                               tok, example.to_string().replace('\'', "'\\''"));
            paths.insert(route, json!({"post": {
                "tags": [m.kind], "operationId": format!("{}.{}", m.kind, m.name), "summary": m.summary,
                "requestBody": {"required": true, "content": {"application/json": {"schema": schema, "example": example}}},
                "responses": responses,
                "x-errors": errors,
                "x-curl": curl,
            }}));
        }
        let oa = |summary: &str, body: Value, answer: &str| {
            let schema: Value = spec::result(answer).and_then(|r| serde_json::from_str(&r).ok()).unwrap_or(json!({"type": "object"}));
            json!({"tags": ["openai"], "summary": summary, "requestBody": {"required": true, "content": {"application/json": {"example": body}}},
                   "responses": {"200": {"description": format!("OpenAI's shape, no envelope (as {answer}'s result)"), "content": {"application/json": {"schema": schema}}},
                                 "4XX": {"description": "OpenAI's error: {\"error\": {\"message\", \"type\"}}"}}})
        };
        // a file an answer links: the bearer token like every route here; byte ranges where the kind's server serves them
        let file = |summary: &str, name: &str, mime: &str, ranges: bool| {
            let mut params = vec![json!({"name": "name", "in": "path", "required": true, "schema": {"type": "string"}, "description": name})];
            let mut responses = json!({"200": {"description": format!("the file ({mime})"), "content": {mime: {"schema": {"type": "string", "format": "binary"}}}},
                                       "401": {"description": "unauthorized: the bearer token is needed here too"},
                                       "404": {"description": "no such file"}});
            if ranges {
                params.push(json!({"name": "Range", "in": "header", "required": false, "schema": {"type": "string"},
                                   "description": "bytes=a-b, bytes=a- or bytes=-n (one span): answered 206 with Content-Range; Accept-Ranges: bytes"}));
                responses["206"] = json!({"description": "the span asked for", "content": {mime: {"schema": {"type": "string", "format": "binary"}}}});
            }
            json!({"get": {"tags": ["files"], "summary": format!("{summary}{}", if ranges { "; byte ranges (Range) supported" } else { "; no byte ranges (always the whole file)" }),
                           "security": [{"bearer": []}], "parameters": params, "responses": responses}})
        };
        paths.insert("/v1/models".into(), json!({"get": {"tags": ["openai"], "summary": "The enabled chat, image and audio models", "responses": {"200": {"description": "OpenAI's list"}}}}));
        if on.contains(&"llm") {
            paths.insert("/v1/chat/completions".into(), json!({"post": oa("Chat (streams with stream: true; tools)",
                json!({"model": ids("llm").first(), "messages": [{"role": "user", "content": "Hello"}], "stream": false}), "llm.chat")}));
        }
        if on.contains(&"image") {
            paths.insert("/v1/images/generations".into(), json!({"post": oa("Pictures (and ours: steps, seed, sampler, schedule, cfg, negative_prompt, loras)",
                json!({"model": ids("image").first(), "prompt": "a lighthouse at dusk", "size": "1024x1024"}), "image.generate")}));
            paths.insert("/v1/images/edits".into(), json!({"post": oa("Edits: multipart (image, image[] ...) or JSON images",
                json!({"prompt": "make it night", "images": ["data:image/png;base64,..."]}), "image.edit")}));
            paths.insert("/v1/images/files/{name}".into(), file("A picture an answer links (data[].url of image.generate / image.edit)", "<file>.png", "image/png", false));
        }
        if on.contains(&"audio") {
            paths.insert("/v1/audio/speech".into(), json!({"post": oa("A song (input: the lyrics, instructions: the music) or speech (input: the text; voice, \
                language, instructions, ref_audio, ref_text); response_format wav, pcm or url; stream: true (or stream_format audio) sends \
                the sound as it is made (wav or pcm), stream_format sse as speech.audio.delta events",
                json!({"model": ids("audio").first(), "instructions": "warm folk", "input": "[verse]\nRiver runs slow", "response_format": "url"}), "audio.generate")}));
            paths.insert("/v1/audio/files/{name}".into(), file("A song or speech an answer links (data[].url of audio.generate / audio.speak)", "<file>.wav", "audio/wav", true));
            paths.insert("/v1/audio/voices/{name}".into(), file("A saved voice's sample (audio.voices.list's sample)", "<voice>.wav", "audio/wav", false));
        }
        if on.contains(&"model3d") {
            paths.insert("/v1/model3d/generate".into(), json!({"post": oa("A mesh from a picture (image) or several views (images: [{image, side}]); texture, seed, steps, faces; response_format glb or url",
                json!({"model": ids("model3d").first(), "image": "data:image/png;base64,iVBORw0...", "texture": true, "response_format": "url"}), "model3d.generate")}));
            paths.insert("/v1/model3d/rig".into(), json!({"post": oa("A mesh rigged (skeleton and skin weights) or animated (prompt: the motion); mesh as base64 GLB / OBJ, or file: a GLB made here",
                json!({"model": ids("model3d").first(), "file": "1792132541-trellis2-42.glb", "response_format": "url"}), "model3d.rig")}));
            paths.insert("/v1/model3d/files/{name}".into(), file("A mesh an answer links (data[].url of model3d.generate / model3d.rig), or its picture (.png)", "<file>.glb", "model/gltf-binary", true));
        }
        if on.contains(&"video") {
            paths.insert("/v1/video/files/{name}".into(), file("A finished clip: <clip>.mp4 (video.jobs.list's file, video.status's and video.timeline's file URLs); also <clip>.preview.png (video.status's preview while it samples) and <clip>.last.png (a clip's last frame), image/png", "<clip>.mp4, e.g. h3_20261011_220723.mp4", "video/mp4", true));
            paths.insert("/v1/video/thumbs/{name}".into(), file("A finished clip's 240-pixel still, 5 s in (video.timeline's thumb)", "<clip>.webp", "image/webp", false));
        }
        json!({
            "openapi": "3.0.3",
            "info": {"title": "nextsycl", "version": crate::VERSION, "description": format!(
                "The box's API (nextsycl serve). Rules: POST /rpc/<kind>.<method>, every variable in the JSON body - no path variables, no query \
                 strings; answers {{\"ok\": true, \"result\": ...}} or {{\"ok\": false, \"error\": {{\"code\", \"message\", ...}}}}, always HTTP 200. The \
                 OpenAI-compatible /v1 routes and the files are the exception (HTTP statuses). Send Authorization: Bearer <token> with every /rpc and /v1 call (the token is on the page's API tab; \
                 this document needs none). Kinds here: {}. Generated {:.0}.\n\n{}\n\n{}",
                on.join(", "), now(), spec::CODES, spec::LABELS)},
            "servers": [{"url": format!("http://{host}")}],
            "security": [{"bearer": []}],
            "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer", "description": "the token on the page's API tab (NS_API_TOKEN, or serve.json)"}}},
            "paths": paths,
        })
    }
}

/// Whether a speech model clones only with what the recording says (its card's `speech` lists `transcript`)
pub(super) fn needs_text(all: &[Value], id: &Value) -> bool {
    all.iter().find(|m| &m["id"] == id).is_some_and(|m| m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "transcript")))
}

/// Whether a model speaks the saved voice `voice` by its enrolment (its card's `speech` lists `enrolled`, and the voice
/// has a fit for its architecture)
pub(super) fn enrolled_for(lib: &crate::voices::Library, voice: &str, m: &Value) -> bool {
    m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "enrolled")) && m["arch"].as_str().is_some_and(|arch| lib.enrolled_path(voice, arch).is_some())
}

/// Whether a speech model speaks the built-in voice `voice`: it lists it (any case), or lists none (its config or card
/// names no voices: any name is its engine's to check)
pub(super) fn speaks(m: &Value, voice: &str) -> bool {
    let (own, _) = super::precheck::speech_lists(m);
    own.is_empty() || own.contains(&voice.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::{code_of, enrolled_for, rpc_error, speaks};

    #[test]
    fn an_enrolled_voice_goes_to_its_model() {
        let dir = std::env::temp_dir().join(format!("ns-enrolled-{}", std::process::id()));
        let lib = crate::voices::Library::of(&dir);
        let vox = serde_json::json!({"id": "voxtral-4b-tts", "arch": "voxtral-tts", "speech": ["voices", "enrolled"]});
        assert!(!enrolled_for(&lib, "anna", &vox));
        let f = lib.enrolled_file("anna", "voxtral-tts").unwrap();
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(&f, b"x").unwrap();
        assert!(enrolled_for(&lib, "anna", &vox));
        // a model whose card does not speak enrolled voices, or of another architecture
        assert!(!enrolled_for(&lib, "anna", &serde_json::json!({"arch": "voxtral-tts", "speech": ["voices"]})));
        assert!(!enrolled_for(&lib, "anna", &serde_json::json!({"arch": "qwen3-tts", "speech": ["enrolled"]})));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_built_in_voice_goes_to_a_model_that_has_it() {
        let vox = serde_json::json!({"id": "voxtral-4b-tts", "voices": ["neutral_female", "fr_male"]});
        assert!(speaks(&vox, "Neutral_Female"));
        assert!(!speaks(&vox, "serena"));
        // no list: the engine decides
        assert!(speaks(&serde_json::json!({"id": "x"}), "serena"));
    }

    #[test]
    fn an_rpc_error_says_everything_in_its_body() {
        let e = rpc_error("image.generate", 400, None, "steps 8: qwen-image-2.1-turbo-q8 runs its own 6 sigmas");
        assert_eq!((e["code"].as_str(), e["status"].as_u64(), e["retryable"].as_bool()), (Some("invalid_params"), Some(400), Some(false)));
        assert_eq!((e["method"].as_str(), e["field"].as_str()), (Some("image.generate"), Some("steps")));
        // a missing field, named in quotes
        assert_eq!(rpc_error("image.generate", 400, None, "image.generate needs \"prompt\"")["field"], "prompt");
        // a key that is not the method's is no field
        assert!(rpc_error("image.generate", 400, None, "qwen-image is busy")["field"].is_null());
        // no method (an unreadable request): null, not ""
        assert!(rpc_error("", 400, Some("request_too_large"), "over 256 MiB")["method"].is_null());
    }

    #[test]
    fn retry_follows_the_code_not_the_status_alone() {
        assert_eq!(rpc_error("system.load", 503, None, "its cards are busy")["code"], "no_room");
        assert_eq!(rpc_error("system.load", 503, None, "x")["retryable"], true);
        assert_eq!(rpc_error("video.generate", 409, Some("busy"), "a clip is being made")["retryable"], true);
        assert_eq!(rpc_error("video.cancel", 409, Some("not_running"), "no job is running")["retryable"], false);
        assert_eq!(rpc_error("image.enable", 409, None, "disabled")["retryable"], false);
        assert_eq!(rpc_error("audio.speak", 499, None, "cancelled")["code"], "cancelled");
        let e = rpc_error("image.generate", 503, None, "cards taken: a GPU run holds /tmp/ds2run.lock (pid 7)");
        assert_eq!((e["code"].as_str(), e["retryable"].as_bool(), e["retry_after_s"].as_u64()), (Some("cards_taken"), Some(true), Some(120)));
        for (st, c) in [(400, "invalid_params"), (401, "unauthorized"), (404, "not_found"), (502, "unavailable"), (504, "timeout"), (500, "internal")] {
            assert_eq!(code_of(st), c);
        }
    }
}
