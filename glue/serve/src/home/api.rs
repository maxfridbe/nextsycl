//! The API on :8000.
//!
//! - `POST /rpc/<kind>.<method>`: H3's rules - POST only, one fixed route a method, every variable in the JSON body (no
//!   path variables, no query strings); the answer `{"ok": true, "result": ...}` or `{"ok": false, "error": {"code",
//!   "message"}}` with the HTTP status. Kinds: `system` (the box: models, enabling, loading), `llm`, `image`, `audio`,
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

pub const METHODS: &[Method] = &[
    Method { kind: "system", name: "state", summary: "The box: the GPUs (memory by process and vouched for), every model's state, the front ends", fields: &[], example: "{}" },
    Method { kind: "system", name: "busy", summary: "Is anything working: a clip made or waiting to run, a picture, sound or chat answered, a model loading. video.accepts says whether video.generate starts a clip at once (else 409 busy without queue: true)",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "version", summary: "The release, the API's version, the kinds enabled, the features this build has (names a client can test for) and the methods callable now",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "files.stat", summary: "A file an answer linked: its bytes and SHA-256 (to check a download), mtime, URL",
             fields: &[("kind", r#"{"type": "string", "enum": ["image", "audio", "video"]}"#, "", true), ("name", "string", "the file's name (the last part of its URL)", true)],
             example: r#"{"kind": "video", "name": "h3_20261011_220723.mp4"}"# },
    Method { kind: "system", name: "files.delete", summary: "Delete a file: a picture, a song or speech, a clip (with its record, last frame, latents and sound tail; not while it is made) or an upload (ref_...)",
             fields: &[("kind", r#"{"type": "string", "enum": ["image", "audio", "video"]}"#, "", true), ("name", "string", "the file's name; a clip: h3_... with or without .mp4", true)],
             example: r#"{"kind": "video", "name": "h3_20261011_220723"}"# },
    Method { kind: "system", name: "models", summary: "Every registered model: kind, enabled, its GPUs, its state (disabled, unloaded, loading, loaded, busy, ready)", fields: &[], example: "{}" },
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

    Method { kind: "llm", name: "chat", summary: "A chat completion (OpenAI's body; stream is ignored - use /v1/chat/completions to stream)",
             fields: &[("model", "string", "a chat model's id", true), ("messages", "object[]", "[{role, content}]", true),
                       ("max_tokens", "integer", "", false), ("temperature", "number", "", false), ("tools", "object[]", "OpenAI tool definitions", false)],
             example: r#"{"model": "qwen3.8-flash-next-iq2_xs", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 200}"# },
    Method { kind: "llm", name: "requests", summary: "The chat requests running and the last ones that ended, with their speeds as the server reports them live: prompt tokens (reused from the cache), read seconds, tokens generated, tok/s",
             fields: &[], example: "{}" },
    Method { kind: "llm", name: "models", summary: "The enabled chat models (what Open WebUI lists) and which is loaded", fields: &[], example: "{}" },
    Method { kind: "llm", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "qwen3.8-flash-next-iq2_xs"}"# },

    Method { kind: "image", name: "generate", summary: "Pictures from a prompt, or from an init picture (image + strength: img2img); the model loads first if it must. Checked when submitted (a bad value is a 400 even with wait: false)",
             fields: &[MODEL_OPT, ("prompt", "string", "", true), ("size", "string", "WxH, e.g. 1024x1024, both multiples of 16 (default: the model's, or the init picture's aspect)", false), ("n", "integer", "pictures", false),
                       ("image", "string", "an init picture (base64 or a data: URL; at most 40 MP): a plain img2img - the denoise starts from it at strength and the prompt guides it (it is no reference: for an edit that reads the picture use image.edit). Needs strength below 1", false),
                       ("strength", r#"{"type": "number", "minimum": 0.05, "maximum": 1}"#, "with image: how far from it (ComfyUI's denoise: the last steps of a round(steps / strength)-step schedule); 0.3-0.6 is the usual img2img range", false),
                       ("steps", "integer", "image.info's steps: a few-step model (turbo) runs only its own count (6)", false), ("seed", "integer", "", false), ("sampler", "string", "ComfyUI's names (euler, euler_ancestral, dpmpp_2m ...)", false),
                       ("schedule", "string", "", false), ("cfg", "number", "true guidance (needs negative_prompt)", false), ("negative_prompt", "string", "", false),
                       ("loras", "string[]", "name:scale", false), ("response_format", "string", "url (default) or b64_json", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false)],
             example: r#"{"model": "qwen-image-2.1-q8", "prompt": "a lighthouse at dusk, oil painting", "size": "1024x768"}"# },
    Method { kind: "image", name: "edit", summary: "Edit or compose pictures: the first is changed, the others are references the prompt names. With reference: false and strength below 1 it is a plain img2img (as image.generate's image). Checked when submitted (a bad value is a 400 even with wait: false)",
             fields: &[MODEL_OPT, ("prompt", "string", "the instructions", true), ("images", "string[]", "up to 8, base64 or data: URLs (PNG, JPEG, WebP); each at most 40 MP, the request at most 256 MiB; each is resized to about 1 MP, so 2048 px on the long side is plenty", true),
                       ("reference", r#"{"type": "boolean", "default": true}"#, "true (default): the pictures condition the model as an edit's references (its vision tower reads them) as well as being where strength starts; false: a plain img2img - the denoise starts from the first picture and the prompt alone guides it, comparable with other tools' img2img (needs strength below 1)", false),
                       ("size", "string", "default: the last picture's aspect", false), ("steps", "integer", "", false), ("seed", "integer", "", false),
                       ("strength", r#"{"type": "number", "minimum": 0.05, "maximum": 1, "default": 1}"#, "how far from the first picture: 1 (default) makes a new picture guided by it; below 1 starts from the first picture, encoded and noised to the level strength reaches: the run is the last steps of a round(steps / strength)-step schedule, exactly ComfyUI's KSampler denoise (a few-step model such as turbo starts at strength itself, then takes its own levels below it). With reference true the picture also conditions the model, which holds it closer than a plain img2img at the same number: measured on Qwen-Image 2.1, 0.9 keeps the framing, face, hair and clothes and still makes the change asked; 0.75 and below give back nearly the same picture. With reference false it behaves as other tools' img2img. The first picture is resized to the output's size", false),
                       ("response_format", "string", "url (default) or b64_json", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false)],
             example: r#"{"prompt": "make it night, keep the lighthouse", "images": ["data:image/png;base64,iVBORw0..."]}"# },
    Method { kind: "image", name: "ocr", summary: "Read pages (an OCR model: Unlimited-OCR): their text as Markdown, tables as HTML, boxes with <|grounding|>",
             fields: &[("model", "string", "an OCR model's id (default: the first enabled)", false), ("images", "string[]", "the pages, base64 or data: URLs, in order: pictures, or PDFs (split into their pages, 150 dpi)", true),
                       ("prompt", "string", "\"document parsing.\" (default), \"Free OCR.\", \"<|grounding|>Convert the document to markdown.\", \"Parse the figure.\"; several pages: each page under it, or (each_page false) the model's \"Multi page parsing.\"; a <PAGE> mark before each page", false),
                       ("tiles", "boolean", "one page: read its 640-pixel tiles too (default true)", false),
                       ("figures", "boolean", "read the text inside the pictures the model marks (a second pass each; default true)", false),
                       ("columns", "boolean", "find blocks set side by side (a gutter between) and read each on its own (default true)", false),
                       ("each_page", "boolean", "several pages: each read on its own with its tiles (default true); false: the model's one-pass multi-page reading", false), ("max_tokens", "integer", "default 8192", false),
                       ("temperature", "number", "0 (default): greedy", false), ("ngram", "integer", "no n-gram of this many tokens repeated (35; 0: off)", false)],
             example: r#"{"images": ["data:image/png;base64,iVBORw0..."], "prompt": "document parsing."}"# },
    Method { kind: "image", name: "info", summary: "The model's defaults, samplers, schedules, options and LoRAs (loaded: from the server)", fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "image", name: "status", summary: "A job (image.generate / image.edit) by its label or reentrant_id: state, progress, its pictures when done; none: the running, queued and last ended",
             fields: &[("label", "string", "", false), ("reentrant_id", "string", "", false)], example: r#"{"label": "poster-7"}"# },
    Method { kind: "image", name: "cancel", summary: "Stop a job by its label (running: between steps; waiting: before it starts); none: whatever the image servers are making",
             fields: &[("label", "string", "", false), ("reentrant_id", "string", "", false)], example: r#"{"label": "poster-7"}"# },
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
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false)],
             example: r#"{"model": "minimax-music3", "instructions": "warm acoustic folk, 90 BPM, a soft male voice", "input": "[verse]\nRiver runs slow tonight\n[chorus]\nCarry me home", "seconds": 60}"# },
    Method { kind: "audio", name: "speak", summary: "Speech from text (a speech model: Qwen3-TTS, VoxCPM2); the answer links the WAV",
             fields: &[MODEL_OPT, ("input", "string", "the text to say", true),
                       ("voice", "string", "a built-in voice (audio.info lists them) or a saved one (audio.voices.list: spoken by a model that clones)", false),
                       ("language", "string", "english, chinese, japanese ... or auto (default)", false),
                       ("instructions", "string", "how to say it (CustomVoice), or the voice to make up (VoiceDesign)", false),
                       ("ref_audio", "string", "a recording of the voice to clone, WAV as base64 or a data: URL (the Base model)", false),
                       ("ref_text", "string", "what the recording says: the closer, in-context clone", false),
                       ("continue_from", r#"{"type": "string", "maxLength": 200}"#, "carry on from speech made here (a book in chunks): the file name (or url) of the chunk before; its last sentences (up to 15 s, cut at a pause) are the in-context reference, after the saved voice's sample when voice names one - pace and voice go on, anchored to the voice. Needs a model that clones (picked for you when model is not given); a built-in voice name is dropped", false),
                       ("seconds", "number", "the most to make", false), ("seed", "integer", "", false),
                       ("options", "object", "the engine's own: greedy, temperature, top-k, streaming", false),
                       ("label", r#"{"type": "string", "maxLength": 120}"#, "names the job for <kind>.status and <kind>.cancel (default: its id)", false),
                       ("reentrant_id", r#"{"type": "string", "maxLength": 200}"#, "idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)", false),
                       ("wait", r#"{"type": "boolean", "default": true}"#, "true: answer when it ends (the job goes on if the call drops); false: answer at once with the job", false)],
             example: r#"{"model": "qwen3-tts-custom", "input": "Good morning. The bread is still warm.", "voice": "ryan", "language": "english", "instructions": "cheerful, a little hurried"}"# },
    Method { kind: "audio", name: "voices.list", summary: "The saved voices (cloned or designed): name, what the sample says, its description, length; sample: its WAV",
             fields: &[], example: "{}" },
    Method { kind: "audio", name: "voices.add", summary: "Save a voice cloned from a recording (WAV, MP3, FLAC, Ogg, M4A; 3-15 s of one speaker is best)",
             fields: &[("name", "string", "lowercase letters, digits, - and _", true), ("sample", "string", "the recording, base64 or a data: URL", true),
                       ("text", "string", "what the recording says (a closer clone)", false), ("description", "string", "", false),
                       ("language", "string", "", false), ("replace", "boolean", "over a voice of that name", false)],
             example: r#"{"name": "grandpa", "sample": "data:audio/mpeg;base64,SUQzBAAAAA...", "text": "Well, back in my day we walked to school."}"# },
    Method { kind: "audio", name: "voices.design", summary: "Make up a voice from a description (the voice-design model reads a sample) and save it",
             fields: &[("name", "string", "", true), ("instructions", "string", "the voice: age, gender, timbre, pace, emotion, accent", true),
                       ("text", "string", "what the sample says (default: a ten-second line in the language)", false), ("language", "string", "", false),
                       ("seed", "integer", "another seed, another speaker of that kind", false), ("model", "string", "a voice-design model", false),
                       ("replace", "boolean", "", false)],
             example: r#"{"name": "narrator", "instructions": "A deep, warm male narrator in his fifties, unhurried, a slight rasp.", "language": "english"}"# },
    Method { kind: "audio", name: "voices.remove", summary: "Delete a saved voice", fields: &[("name", "string", "", true)], example: r#"{"name": "narrator"}"# },
    Method { kind: "audio", name: "info", summary: "The model's defaults, limits and options (a speech model: its voices and languages)", fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "audio", name: "progress", summary: "The request running: its phase and how far", fields: &[], example: "{}" },
    Method { kind: "audio", name: "history", summary: "The songs and speech made, newest first", fields: &[], example: "{}" },
    Method { kind: "audio", name: "status", summary: "A job (audio.generate / audio.speak) by its label or reentrant_id: state, progress, its sound when done; none: the running, queued and last ended",
             fields: &[("label", "string", "", false), ("reentrant_id", "string", "", false)], example: r#"{"label": "theme-3"}"# },
    Method { kind: "audio", name: "cancel", summary: "Stop a job by its label (running: between steps or frames; waiting: before it starts); none: whatever the audio servers are making",
             fields: &[("label", "string", "", false), ("reentrant_id", "string", "", false)], example: r#"{"label": "theme-3"}"# },
    Method { kind: "audio", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "minimax-music3"}"# },

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
const API_VERSION: u32 = 2;

/// What this build does, by names a client can test for (system.version); names are added, never reused
const FEATURES: &[&str] = &[
    "system.busy", "system.busy.cards", "system.busy.slots", "system.version", "system.files.stat", "system.files.delete",
    "jobs.image", "jobs.audio", "jobs.cancel.label", "reentrant_id",
    "video.frames", "video.chain", "video.priority", "video.cancel.label", "video.queue.move", "video.queue.remove", "video.engines.inputs",
    "video.negative_prompt.ignored", "image.canvases", "image.cancel", "spec.open",
    "image.edit.strength", "audio.speak.continue_from", "video.preview", "system.busy.parallel",
    "errors.retryable", "errors.no_room", "jobs.precheck", "image.info.steps", "image.generate.init_image", "image.edit.reference",
    "video.failure", "video.gpu", "video.engines.ignored",
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
    matches!(status, 502..=504) || code == "busy"
}

/// An error's body: code, message, whether to retry
pub(crate) fn error_of(status: u16, m: &str) -> Value {
    let code = code_of(status);
    json!({"code": code, "message": m, "retryable": retryable(status, code)})
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
        let fail = |s: &mut Conn, st: u16, m: String| {
            if rpc {
                http::respond(s, st, &json!({"ok": false, "error": error_of(st, &m)}));
            } else {
                http::respond(s, st, &json!({"error": {"message": m, "type": if st == 401 { "invalid_api_key" } else { "invalid_request_error" }}}));
            }
        };
        if !self.authorized(&req) {
            return fail(&mut s, 401, "Authorization: Bearer <token> (the token is on the page's API tab)".into());
        }
        if rpc {
            if req.method != "POST" || req.target.contains('?') {
                return fail(&mut s, 400, "POST only, every variable in the JSON body (no query strings)".into());
            }
            let body: Value = if req.body.is_empty() { json!({}) } else {
                match serde_json::from_slice(&req.body) {
                    Ok(v @ Value::Object(_)) => v,
                    _ => return fail(&mut s, 400, "the body must be a JSON object".into()),
                }
            };
            let name = &req.path["/rpc/".len()..];
            if let Some((st, v)) = self.studio_rpc(name, &body, req.host.as_deref().unwrap_or("localhost:8000")) {
                return http::respond(&mut s, st, &v);
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
    fn kinds_on(&self, all: &[Value]) -> Vec<&'static str> {
        let mut v = vec!["system"];
        for k in ["llm", "image", "audio", "video"] {
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

    /// The speech model for a request that names none (or names one not registered): one that clones for a saved
    /// voice or a recording, the built-in voices' for a voice, the voice-design one for instructions alone; the
    /// loaded one first
    pub(super) fn speech_model(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        if b["model"].as_str().is_some_and(|m| all.iter().any(|x| x["id"] == m)) {
            return self.pick("audio", b, all);
        }
        let voice = s(b, "voice");
        let need = if !s(b, "ref_audio").is_empty() || !s(b, "continue_from").is_empty() || (!voice.is_empty() && self.voices().get(&voice).is_some()) {
            "clone"
        } else if !voice.is_empty() {
            "voices"
        } else if !s(b, "instructions").is_empty() {
            "design"
        } else {
            ""
        };
        let can: Vec<String> = all.iter()
            .filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false)
            .filter(|m| m["speech"].as_array().is_some_and(|a| need.is_empty() || a.iter().any(|x| x == need)))
            .map(|m| s(m, "id")).collect();
        let loaded = self.loaded_of("audio");
        loaded.into_iter().find(|l| can.contains(l)).or_else(|| can.first().cloned()).ok_or_else(|| {
            (409, match need {
                "clone" => "no enabled speech model clones a voice (system.enable qwen3-tts-base)".to_string(),
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
        let v = lib.add(&name, &bytes, Some(&text), Some(&s(b, "instructions")), lang, "design", &format!("designed by {id}, seed {seed}"), b["replace"] == true)
            .map_err(|e| (400, e))?;
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
        if !self.kinds_on(all).contains(&m.kind) {
            return Err((409, format!("no {} model is enabled (system.enable)", m.kind)));
        }
        Ok(m)
    }

    /// A studio method (video.*, but upload and the guide): the studio's answer as it gave it - its status and error
    /// code (busy, not_running ...) - with the files it names as URLs through here
    fn studio_rpc(&self, name: &str, b: &Value, host: &str) -> Option<(u16, Value)> {
        let method = name.strip_prefix("video.").filter(|m| !matches!(*m, "upload" | "describePromptGuideMD"))?;
        let all = self.models();
        let envelope = |st: u16, m: String| (st, json!({"ok": false, "error": error_of(st, &m)}));
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
            let code = v["error"]["code"].as_str().unwrap_or("").to_string();
            v["error"]["retryable"] = json!(retryable(st, &code));
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
    fn busy(&self, all: &[Value]) -> Value {
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
        if !jobs.is_empty() {
            reasons.push(format!("{} image/audio job(s)", jobs.len()));
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
        json!({"busy": !reasons.is_empty(), "reasons": reasons, "loading": loading, "working": working, "llm": {"active": active},
               "jobs": jobs, "video": video, "cards": per_card, "slots": slots, "parallel": self.parallel(all)})
    }

    /// How many requests of each image and audio model run at once: a copy a card it is enabled on, as many as the
    /// kind has servers (a second copy starts when the first has work and its card has the room then)
    fn parallel(&self, all: &[Value]) -> Value {
        let mut out = serde_json::Map::new();
        for kind in ["image", "audio"] {
            let servers = self.slots(kind).len();
            let models: serde_json::Map<String, Value> = all.iter().filter(|m| registry::kind_of(m) == kind && m["enabled"] != false)
                .map(|m| (s(m, "id"), json!(self.gpus_of(m).len().min(servers)))).collect();
            out.insert(kind.into(), json!({"servers": servers, "models": models}));
        }
        Value::Object(out)
    }

    fn rpc(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
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
                Ok(json!({"version": crate::VERSION, "api": API_VERSION, "kinds": on, "features": FEATURES, "methods": methods}))
            }
            ("system", "files.stat" | "files.delete") => self.file_op(m.name, b),
            ("system", "models") => {
                let cards = self.named_cards(&all);
                Ok(json!({"models": self.model_states(&all, &cards)}))
            }
            ("system", "enable") => self.enable(&model(), b.get("gpus")).map(|t| json!({"model": model(), "title": t, "enabled": true})),
            ("system", "gpus") => self.set_gpus(&model(), b.get("gpus")),
            ("system", "disable") => self.disable(&model(), b["force"] == true).map(|_| json!({"model": model(), "enabled": false})),
            ("system", "load") if b["wait"] == true => self.load(&model()).map(|_| json!({"model": model(), "loaded": true})),
            ("system", "load") => self.load_async(&model()).map(|_| json!({"model": model(), "loading": true})),
            ("system", "unload") => self.unload(&model()).map(|_| json!({"model": model(), "loaded": false})),
            ("system", "config") => Ok(self.config()),
            ("system", "configSet") => self.config_set(b),
            ("system", "restart") => self.restart(&s(b, "service")),
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
                relay(&t, "POST", "/v1/chat/completions", Some(&body), 1800)
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
                let t = self.ensure(&id)?;
                let mut body = b.clone();
                body["model"] = json!(id);
                if body.get("response_format").is_none() {
                    body["response_format"] = json!("url");
                }
                let path = if m.name == "edit" { "/v1/images/edits" } else { "/v1/images/generations" };
                relay(&t, "POST", path, Some(&body), 3600)
            }
            ("audio", "generate" | "speak") => {
                let mut b = b.clone();
                if m.name == "speak" {
                    b["model"] = json!(self.speech_model(&b, &all)?);
                }
                let b = &b;
                let id = self.pick("audio", b, &all)?;
                let t = self.ensure(&id)?;
                let mut body = b.clone();
                body["model"] = json!(id);
                body["response_format"] = json!("url");
                relay(&t, "POST", "/v1/audio/speech", Some(&body), 3600)
            }
            ("image" | "audio", "info") => {
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
                    _ => json!({"loaded": false, "model": id, "title": e["title"], "defaults": e["defaults"],
                                "note": "not loaded: the server's own details (info) come once it is (system.load); steps and the limits are known now"}),
                };
                if let (Some(o), Some(l)) = (v.as_object_mut(), limits.as_object()) {
                    o.extend(l.clone());
                }
                Ok(v)
            }
            ("image" | "audio", "progress") => Ok(call_for(&t, "GET", "/api/progress", None, 5).unwrap_or(json!({"busy": false, "loaded": false}))),
            ("image" | "audio", "history") => Ok(call_for(&t, "GET", "/api/history", None, 10).unwrap_or(json!({"data": []}))),
            ("image" | "audio", "status") => self.job_status(m.kind, b),
            ("image", "canvases.list") => self.image_canvases(b, &all),
            ("image" | "audio", "cancel") => self.job_cancel(m.kind, b),
            ("audio", "voices.list") => {
                let list: Vec<Value> = self.voices().list().into_iter().map(|mut v| {
                    v["sample"] = json!(format!("http://nextsycl/v1/audio/voices/{}.wav", s(&v, "name")));
                    v
                }).collect();
                Ok(json!({"voices": list}))
            }
            ("audio", "voices.remove") => self.voices().remove(&s(b, "name")).map(|_| json!({"removed": s(b, "name")})).map_err(|e| (404, e)),
            ("audio", "voices.add") => {
                let bytes = http::unbase64(&s(b, "sample")).map_err(|e| (400, format!("sample: {e}")))?;
                let o = |k: &str| b[k].as_str().filter(|x| !x.trim().is_empty());
                let v = self.voices().add(&s(b, "name"), &bytes, o("text"), o("description"), o("language"), "clone", "a recording", b["replace"] == true)
                    .map_err(|e| (400, e))?;
                Ok(json!({"voice": v, "sample": format!("http://nextsycl/v1/audio/voices/{}.wav", s(b, "name"))}))
            }
            ("audio", "voices.design") => self.design_voice(b, &all),
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
        let (key, sub, def) = match kind.as_str() {
            "image" => ("NS_IMAGE_OUT", "images", "images"),
            "audio" => ("NS_AUDIO_OUT", "audio", "audio"),
            _ => return Err((400, "kind: image, audio or video".into())),
        };
        let dir = self.o.cfg.get(key).unwrap_or_else(|| format!("{}/.local/share/nextsycl/{def}", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
        let path = std::path::Path::new(&dir).join(&name);
        // what the servers make, not the saved voices or anything else kept there
        let ext = name.rsplit('.').next().unwrap_or("").to_lowercase();
        let ok = if kind == "image" { matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp") } else { ext == "wav" };
        if !ok || !path.is_file() {
            return Err((404, format!("no {kind} file {name}")));
        }
        if op == "files.stat" {
            let mut r = crate::files::stat(&path).map_err(|e| (404, e))?;
            r["url"] = json!(format!("http://nextsycl/v1/{sub}/files/{name}"));
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
                need("llm")?;
                let model = serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string));
                let t = self.chat_target(model.as_deref())?;
                pass(s, &t)
            }
            ("POST", p @ ("/v1/images/generations" | "/v1/images/edits" | "/v1/audio/speech")) => {
                let kind = if p.starts_with("/v1/images") { "image" } else { "audio" };
                need(kind)?;
                // the model the body names (a JSON field, or a multipart part)
                let ctype = http::header(&req.head, "content-type").unwrap_or_default();
                let named = if let Some(b) = ctype.split("boundary=").nth(1) {
                    http::multipart(&req.body, b.trim_matches('"')).ok()
                        .and_then(|ps| ps.into_iter().find(|p| p.name == "model").map(|p| String::from_utf8_lossy(&p.data).trim().to_string()))
                } else {
                    serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string))
                };
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
            ("GET", p) if p.starts_with("/v1/images/files/") => pass(s, &self.file_target("image")),
            ("GET", p) if p.starts_with("/v1/audio/files/") => pass(s, &self.file_target("audio")),
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
            "error": {"type": "object", "properties": {"code": {"type": "string"}, "message": {"type": "string"}, "retryable": {"type": "boolean", "description": "true: the same call may work later (no_room, unavailable, timeout, busy)"}}}}});
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
            let mut responses = json!({"200": {"description": "{\"ok\": true, \"result\": ...}", "content": {"application/json": {"schema": ok}}},
                                       "401": {"description": "unauthorized: no or a wrong token"},
                                       "400": {"description": "invalid_request / invalid_params: the body or a value (error.message says which)"}});
            for (st, code, when) in spec::errors(&op) {
                let k = st.to_string();
                let d = format!("{code}: {when}");
                let d = match responses[&k]["description"].as_str() { Some(o) if *st != 400 => format!("{o}; {d}"), _ => d };
                responses[&k] = json!({"description": d, "content": {"application/json": {"schema": envelope.clone()}}});
            }
            if m.kind != "system" {
                let d = format!("conflict: no {} model is enabled", m.kind);
                let d = match responses["409"]["description"].as_str() { Some(o) => format!("{o}; {d}"), None => d };
                responses["409"] = json!({"description": d});
            }
            let curl = format!("curl -s http://{host}{route} -H 'Authorization: Bearer {}' -H 'Content-Type: application/json' -d '{}'",
                               tok, example.to_string().replace('\'', "'\\''"));
            paths.insert(route, json!({"post": {
                "tags": [m.kind], "operationId": format!("{}.{}", m.kind, m.name), "summary": m.summary,
                "requestBody": {"required": true, "content": {"application/json": {"schema": schema, "example": example}}},
                "responses": responses,
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
                language, instructions, ref_audio, ref_text); response_format wav or url",
                json!({"model": ids("audio").first(), "instructions": "warm folk", "input": "[verse]\nRiver runs slow", "response_format": "url"}), "audio.generate")}));
            paths.insert("/v1/audio/files/{name}".into(), file("A song or speech an answer links (data[].url of audio.generate / audio.speak)", "<file>.wav", "audio/wav", true));
            paths.insert("/v1/audio/voices/{name}".into(), file("A saved voice's sample (audio.voices.list's sample)", "<voice>.wav", "audio/wav", false));
        }
        if on.contains(&"video") {
            paths.insert("/v1/video/files/{name}".into(), file("A finished clip: <clip>.mp4 (video.jobs.list's file, video.status's and video.timeline's file URLs); also <clip>.preview.png (video.status's preview while it samples) and <clip>.last.png (a clip's last frame), image/png", "<clip>.mp4, e.g. h3_20261011_220723.mp4", "video/mp4", true));
            paths.insert("/v1/video/thumbs/{name}".into(), file("A finished clip's 240-pixel still, 5 s in (video.timeline's thumb)", "<clip>.webp", "image/webp", false));
        }
        json!({
            "openapi": "3.0.3",
            "info": {"title": "nextsycl", "version": crate::VERSION, "description": format!(
                "The box's API (nextsycl serve). Rules: POST /rpc/<kind>.<method>, every variable in the JSON body - no path variables, no query \
                 strings; answers {{\"ok\": true, \"result\": ...}} or {{\"ok\": false, \"error\": {{\"code\", \"message\"}}}}. The OpenAI-compatible \
                 /v1 routes are the exception. Send Authorization: Bearer <token> with every /rpc and /v1 call (the token is on the page's API tab; \
                 this document needs none). Kinds here: {}. Generated {:.0}.\n\n{}\n\n{}",
                on.join(", "), now(), spec::CODES, spec::LABELS)},
            "servers": [{"url": format!("http://{host}")}],
            "security": [{"bearer": []}],
            "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer", "description": "the token on the page's API tab (NS_API_TOKEN, or serve.json)"}}},
            "paths": paths,
        })
    }
}
