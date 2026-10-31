//! The front door's RPC methods (`POST /rpc/<kind>.<method>`), each its #[rpc] handler (rpcgen, vendored in
//! vendor/rpcgen from vibe-creative-platform). The method table, the OpenAPI document's operations and request
//! schemas (api.rs `openapi`, which adds the box's own model ids and engines to them), system.version's list and
//! the wfe pages' typed client (`nextsycl gen-api`: wfe/<page>/src/gen/nextsycl.ts) are all made from what is
//! declared here - a method's name, its doc comment (the summary), its request struct (fields, their limits
//! with #[schema], their descriptions) and its example. Nothing about a method is written a second time: a
//! method missing here is no method, two of one name or a request type without #[derive(Rpc)] do not compile,
//! and a wfe call with a wrong field does not type-check.
//!
//! Each request struct passes on what else a caller sent (`rest`), so a request reaches the work as it came, its
//! known fields now checked for their types. The work is still `Home::rpc_inner`'s (api.rs); a handler's body
//! takes it over as each method moves here. The answers are `Untyped` until their structs are written (spec.rs
//! still describes them).

#![allow(clippy::large_enum_variant)]

use std::sync::{Arc, OnceLock};

use rpcgen::{rpc, Rpc, Untyped};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Home;

type Fail = (u16, String);

/// A method's failure: the status and the words the envelope carries (a body that does not fit is a 400).
pub struct RpcFail(pub Fail);

impl From<rpcgen::BadPayload> for RpcFail {
    fn from(b: rpcgen::BadPayload) -> Self {
        RpcFail((400, b.0))
    }
}

impl From<Fail> for RpcFail {
    fn from(f: Fail) -> Self {
        RpcFail(f)
    }
}

type Answer = Result<Untyped, RpcFail>;

impl Home {
    /// A method's work as api.rs does it, given the request as it was checked
    fn legacy<T: Serialize>(self: &Arc<Self>, name: &str, req: T) -> Answer {
        let b = serde_json::to_value(req).map_err(|e| RpcFail((500, format!("{name}: the request did not encode: {e}"))))?;
        self.rpc_inner(name, &b).map(Untyped).map_err(RpcFail)
    }
}

/// A method as the rest of the front door reads it: its kind and name (`image` and `generate`), what it says it
/// does, its body's fields and an example body - all from its #[rpc] declaration.
pub struct Method {
    pub kind: &'static str,
    pub name: &'static str,
    pub summary: &'static str,
    pub fields: Vec<rpcgen::Field>,
    pub example: Value,
    pub info: rpcgen::MethodInfo,
    pub handler: fn(&Arc<Home>, Value) -> Result<Value, RpcFail>,
    pub check: fn(&Value) -> Result<(), rpcgen::BadPayload>,
}

impl Method {
    /// `kind.name`
    pub fn full(&self) -> &'static str {
        self.info.name
    }
}

/// Every method, in the order declared (the document's).
pub fn methods() -> &'static [Method] {
    static M: OnceLock<Vec<Method>> = OnceLock::new();
    M.get_or_init(|| {
        rpc_methods()
            .into_iter()
            .map(|m| {
                let (kind, name) = m.info.name.split_once('.').expect("a dotted name");
                Method {
                    kind,
                    name,
                    summary: m.info.summary,
                    fields: m.info.request_fields(),
                    example: m.info.example_body(),
                    info: m.info.clone(),
                    handler: m.handler,
                    check: m.check,
                }
            })
            .collect()
    })
}

/// The method of a name, in any case (`IMAGE.Generate` is image.generate).
pub fn find(name: &str) -> Option<&'static Method> {
    methods().iter().find(|m| m.full().eq_ignore_ascii_case(name))
}

/// The table's own descriptions, for the document and the wfe client.
pub fn infos() -> Vec<rpcgen::MethodInfo> {
    methods().iter().map(|m| m.info.clone()).collect()
}

// generated once from the published request schemas (2026-10-26); maintained by hand from here

// ---- the options: a field of a fixed set of values is an enum (its word on the wire, as before) ----

/// A file's kind (where it lives): image | audio | video | model3d
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum FileKind {
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "audio")]
    Audio,
    #[serde(rename = "video")]
    Video,
    #[serde(rename = "model3d")]
    Model3d,
}

/// How a picture comes back: a URL, or the bytes as base64 (OpenAI's names): url | b64_json
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum ResponseFormat {
    #[serde(rename = "url")]
    Url,
    #[serde(rename = "b64_json")]
    B64Json,
}

/// A repaint's balance: balanced | conservative | aggressive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum RepaintMode {
    #[serde(rename = "balanced")]
    Balanced,
    #[serde(rename = "conservative")]
    Conservative,
    #[serde(rename = "aggressive")]
    Aggressive,
}

/// ACE-Step 1.5's task: text2music | cover | repaint | edit | lego
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum AudioTask {
    #[serde(rename = "text2music")]
    Text2music,
    #[serde(rename = "cover")]
    Cover,
    #[serde(rename = "repaint")]
    Repaint,
    #[serde(rename = "edit")]
    Edit,
    #[serde(rename = "lego")]
    Lego,
}

/// A mesh tool's operation: decimate | remesh | uv | convert
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum MeshOp {
    #[serde(rename = "decimate")]
    Decimate,
    #[serde(rename = "remesh")]
    Remesh,
    #[serde(rename = "uv")]
    Uv,
    #[serde(rename = "convert")]
    Convert,
}

/// How a clip carries on from the last: video | png | latent | none
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum ChainMode {
    #[serde(rename = "video")]
    Video,
    #[serde(rename = "png")]
    Png,
    #[serde(rename = "latent")]
    Latent,
    #[serde(rename = "none")]
    None,
}

/// A reference picture's size: match | max
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum RefImageSize {
    #[serde(rename = "match")]
    Match,
    #[serde(rename = "max")]
    Max,
}

/// The text encoder: teacher | student
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum TextEncoder {
    #[serde(rename = "teacher")]
    Teacher,
    #[serde(rename = "student")]
    Student,
}

/// The upscaler: esrgan-general | esrgan-anime | latent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Rpc)]
pub enum Upscaler {
    #[serde(rename = "esrgan-general")]
    EsrganGeneral,
    #[serde(rename = "esrgan-anime")]
    EsrganAnime,
    #[serde(rename = "latent")]
    Latent,
}

/// What system.state takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemStateReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.busy takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemBusyReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.ping takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemPingReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.version takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemVersionReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.files.stat takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemFilesStatReq {
    pub kind: FileKind,
    /// the file's name (the last part of its URL)
    pub name: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.files.delete takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemFilesDeleteReq {
    pub kind: FileKind,
    /// the file's name; a clip: h3_... with or without .mp4
    pub name: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.models takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemModelsReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.models.pull takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemModelsPullReq {
    /// the card's id (system.models, downloaded false)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.models.pulls takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemModelsPullsReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.models.pull.cancel takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemModelsPullCancelReq {
    /// the model being downloaded
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.cards takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemCardsReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.enable takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemEnableReq {
    /// the cards (default: as registered); several for a model that spans cards
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_vec_int")]
    pub gpus: Option<Vec<i64>>,
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.gpus takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemGpusReq {
    /// the cards: one or more (an image or audio model loads on the first of them with room)
    #[serde(deserialize_with = "rpcgen::de::vec_int")]
    pub gpus: Vec<i64>,
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.disable takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemDisableReq {
    /// a video model while a clip renders: stop the clip at its next step
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.load takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemLoadReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// answer when it has loaded (default: at once, loading in the background)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.pin takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemPinReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.unpin takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemUnpinReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.eta takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemEtaReq {
    /// the method the request would call (image.generate, llm.chat ...)
    pub method: String,
    /// the request's parameters as the method takes them (only the size-driving ones matter)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// the model (default: the one the request would pick)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// the longest you would wait for it to start: the answer says whether it would be accepted (would_accept)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_willing_to_wait_s: Option<f64>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.unload takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemUnloadReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.config takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemConfigReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.configSet takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemConfigSetReq {
    /// {NAME: value | null}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set: Option<serde_json::Map<String, serde_json::Value>>,
    /// the whole file instead
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.restart takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemRestartReq {
    /// serve | switch | studio | video | image | audio | llm
    pub service: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.maintenance takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemMaintenanceReq {
    /// about how long, in seconds: retry_after_s and the notice's end
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_s: Option<f64>,
    /// what callers are told (kernel work on the B70, say)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// true: on, false: off; absent: just the state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on: Option<bool>,
    /// true: unload the loaded models not answering now (frees the cards)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unload: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.drives takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemDrivesReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.drive.measure takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemDriveMeasureReq {
    /// the drive's id (nvme1n1) or its mount point (system.drives)
    pub drive: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.move takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemMoveReq {
    /// the drive's id (nvme1n1) or its mount point (system.drives)
    pub drive: String,
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What system.move.cancel takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct SystemMoveCancelReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What llm.chat takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct LlmChatReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// keep the model loaded this long after the answer: picture, sound and video work does not stop it for room meanwhile (as /v1/chat/completions' X-Keep-Up-S header, which Open WebUI sends: 300). Absent: nothing kept - once idle the model may be stopped for other work
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 86400)]
    pub keep_up_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub max_tokens: Option<i64>,
    /// [{role, content}]
    pub messages: Vec<serde_json::Map<String, serde_json::Value>>,
    /// a chat model's id
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// OpenAI tool definitions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Map<String, serde_json::Value>>>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What llm.requests takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct LlmRequestsReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What llm.models takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct LlmModelsReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What llm.describePromptGuideMD takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct LlmDescribePromptGuideMdReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.generate takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageGenerateReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// true guidance (needs negative_prompt)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// an init picture (base64, a data: URL, or upload:<name> from image.upload; at most 40 MP): a plain img2img - the denoise starts from it at strength and the prompt guides it (it is no reference: for an edit that reads the picture use image.edit). Needs strength below 1
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// LoRAs for this request (a registered LoRA's id - system.models kind lora - or a file), each with its strength: Qwen-Image 2.1 runs them beside the loaded weights for this request only (no reload; each file read once and kept, a few-step LoRA's sigmas applied); other models reload with them merged
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(json = r#"{"items": {"oneOf": [{"type": "string", "description": "name:scale"}, {"properties": {"name": {"type": "string"}, "scale": {"default": 1, "type": "number"}}, "required": ["name"], "type": "object"}]}}"#)]
    pub loras: Option<Vec<serde_json::Value>>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// pictures
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub n: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    pub prompt: String,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// url (default) or b64_json
    #[serde(rename = "response_format", default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// ComfyUI's names (euler, euler_ancestral, dpmpp_2m ...)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    /// WxH, e.g. 1024x1024, both multiples of 16 (default: the model's, or the init picture's aspect)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// image.info's steps: a few-step model (turbo) runs only its own count (6)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub steps: Option<i64>,
    /// with image: how far from it (ComfyUI's denoise: the last steps of a round(steps / strength)-step schedule); 0.3-0.6 is the usual img2img range
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0.05, max = 1)]
    pub strength: Option<f64>,
    /// true: answer when it ends (the job goes on if the call drops); false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.edit takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageEditReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// up to 8, base64, data: URLs or upload:<name> (image.upload) - PNG, JPEG, WebP; each at most 40 MP, the request at most 256 MiB; each is resized to about 1 MP, so 2048 px on the long side is plenty
    pub images: Vec<String>,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// LoRAs for this request (a registered LoRA's id - system.models kind lora - or a file), each with its strength: Qwen-Image 2.1 runs them beside the loaded weights for this request only (no reload; each file read once and kept, a few-step LoRA's sigmas applied); other models reload with them merged
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(json = r#"{"items": {"oneOf": [{"type": "string", "description": "name:scale"}, {"properties": {"name": {"type": "string"}, "scale": {"default": 1, "type": "number"}}, "required": ["name"], "type": "object"}]}}"#)]
    pub loras: Option<Vec<serde_json::Value>>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// the instructions
    pub prompt: String,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// true (default): the pictures condition the model as an edit's references (its vision tower reads them) as well as being where strength starts; false: a plain img2img - the denoise starts from the first picture and the prompt alone guides it, comparable with other tools' img2img (needs strength below 1)
    #[serde(rename = "reference", default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub reference: Option<bool>,
    /// url (default) or b64_json
    #[serde(rename = "response_format", default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    /// default: the last picture's aspect
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub steps: Option<i64>,
    /// how far from the first picture: 1 (default) makes a new picture guided by it; below 1 starts from the first picture, encoded and noised to the level strength reaches: the run is the last steps of a round(steps / strength)-step schedule, exactly ComfyUI's KSampler denoise (a few-step model such as turbo starts at strength itself, then takes its own levels below it). With reference true the picture also conditions the model, which holds it closer than a plain img2img at the same number: measured on Qwen-Image 2.1, 0.9 keeps the framing, face, hair and clothes and still makes the change asked; 0.75 and below give back nearly the same picture. With reference false it behaves as other tools' img2img. The first picture is resized to the output's size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0.05, max = 1, default = 1)]
    pub strength: Option<f64>,
    /// true: answer when it ends (the job goes on if the call drops); false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.upscale takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageUpscaleReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// the picture: PNG, JPEG or WebP as base64, a data: URL or upload:<name> (image.upload)
    pub image: String,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// waifu2x's denoise level (-1: none)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -1, max = 3)]
    pub noise: Option<i64>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit (as image.generate)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    #[serde(rename = "response_format", default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// the factor: Real-ESRGAN 2, 3 (animevideov3) or 4; waifu2x 1 (denoise only), 2, 4 ... 32 (its 2x model again each time)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub scale: Option<i64>,
    /// the tile's side in input pixels (Real-ESRGAN 200, waifu2x 400 by default: as the ncnn programs)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 32)]
    pub tile: Option<i64>,
    /// the upscaler's model (image.info's upscale list): realesrgan-x4plus (default at 4x), realesr-animevideov3 (default at 2x / 3x), ..., models-cunet (waifu2x's default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    /// true: answer when it ends; false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.vectorize takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageVectorizeReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 1, max = 8, default = 2)]
    pub beams: Option<i64>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// the picture: PNG, JPEG or WebP as base64, a data: URL or upload:<name> (image.upload)
    pub image: String,
    /// names the job for image.status and image.cancel
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// the sequence at most as the model counts it (the picture's 257 rows and the prompt first: 5000 leaves 4741 tokens of SVG)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 260, max = 8192, default = 5000)]
    pub max_length: Option<i64>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit (as image.generate)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    #[serde(rename = "repetition_penalty", default, skip_serializing_if = "Option::is_none")]
    #[schema(default = 1.0)]
    pub repetition_penalty: Option<f64>,
    /// false: the likeliest tokens (beam search, or greedy with beams 1)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub sample: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = 1.0)]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = 0.9)]
    pub top_p: Option<f64>,
    /// true: answer when it ends; false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.upscale takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoUpscaleReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// the H.264 encode's quality (lower: better, bigger)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 0, max = 40, default = 16)]
    pub crf: Option<i64>,
    /// or a clip made here (a video.generate clip's name, h3_..., with or without .mp4)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// names the job for image.status and image.cancel
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// as image.upscale
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub noise: Option<i64>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit (as image.generate)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// url (default): the server keeps the MP4 and links it
    #[serde(rename = "response_format", default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// as image.upscale
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub scale: Option<i64>,
    /// as image.upscale
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub tile: Option<i64>,
    /// as image.upscale (realesr-animevideov3: trained on video frames)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    /// the clip: MP4 / WebM / MOV ... as base64 or a data: URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<String>,
    /// true: answer when it ends; false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.ocr takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageOcrReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// find blocks set side by side (a gutter between) and read each on its own (default true)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub columns: Option<bool>,
    /// several pages: each read on its own with its tiles (default true); false: the model's one-pass multi-page reading
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub each_page: Option<bool>,
    /// read the text inside the pictures the model marks (a second pass each; default true)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub figures: Option<bool>,
    /// the pages, base64, data: URLs or upload:<name> (image.upload), in order: pictures, or PDFs (split into their pages, 150 dpi)
    pub images: Vec<String>,
    /// default 8192
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub max_tokens: Option<i64>,
    /// an OCR model's id (default: the first enabled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// no n-gram of this many tokens repeated (35; 0: off)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub ngram: Option<i64>,
    /// "document parsing." (default), "Free OCR.", "<|grounding|>Convert the document to markdown.", "Parse the figure."; several pages: each page under it, or (each_page false) the model's "Multi page parsing."; a <PAGE> mark before each page
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// 0 (default): greedy
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// one page: read its 640-pixel tiles too (default true)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.info takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageInfoReq {
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.status takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageStatusReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.jobs takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageJobsReq {
    /// ended jobs listed
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 0, max = 200, default = 20)]
    pub limit: Option<i64>,
    /// only this project's
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.queue.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageQueueListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.queue.move takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageQueueMoveReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 1, default = 1)]
    pub position: Option<i64>,
    /// give it this priority too (where later jobs go)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9)]
    pub priority: Option<i64>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.queue.pause takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageQueuePauseReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.queue.resume takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageQueueResumeReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.upload takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageUploadReq {
    /// the picture, base64 (a data: URL too)
    pub data: String,
    /// the file's name: png jpg jpeg webp; kept as up_<unix seconds>_<name> (letters, digits, . _ - kept, the rest _, 60 characters)
    pub name: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.cancel takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageCancelReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.canvases.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageCanvasesListReq {
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.progress takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageProgressReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.history takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageHistoryReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What image.describePromptGuideMD takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct ImageDescribePromptGuideMdReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.generate takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioGenerateReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// acestep-1.5, a task on a recording: the recording (WAV, MP3, FLAC, Ogg, M4A as base64 or a data: URL, or upload:<name> from audio.upload); the song is as long as it (repaint: to repaint_end)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// acestep-1.5: the tempo (else the planner's, or none said)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub bpm: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    /// cover: how close the start is to the recording, 0 (from noise) to 1 (0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover_noise: Option<f64>,
    /// cover / repaint: the share of the steps conditioned on the recording, 0 to 1 (0.5)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover_strength: Option<f64>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// the lyrics, [verse] / [chorus] tags on lines of their own (empty: instrumental)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// the music: genre, mood, tempo, instruments, voice (with audio it may be left out: the recording's sound is kept)
    pub instructions: String,
    /// acestep-1.5: key and scale, "C major", "F# minor"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// acestep-1.5: the lyrics' language as a code (en, zh, ja, es ...; default unknown)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// acestep-1.5 text2music: the 5 Hz planner LM writes the metas, the caption and the song's audio codes first (true, as the platform's use_lm)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner: Option<bool>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// repaint: seconds; -1 or none: the recording's end; past it: the song extended to it
    #[serde(rename = "repaint_end", default, skip_serializing_if = "Option::is_none")]
    pub repaint_end: Option<f64>,
    /// repaint: how much of the recording is kept at the span's edges (balanced)
    #[serde(rename = "repaint_mode", default, skip_serializing_if = "Option::is_none")]
    pub repaint_mode: Option<RepaintMode>,
    /// repaint: seconds into the recording (0)
    #[serde(rename = "repaint_start", default, skip_serializing_if = "Option::is_none")]
    pub repaint_start: Option<f64>,
    /// repaint, balanced: 0 keeps the most, 1 the least (0.5)
    #[serde(rename = "repaint_strength", default, skip_serializing_if = "Option::is_none")]
    pub repaint_strength: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    /// edit: the words the recording sings now
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_lyrics: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub steps: Option<i64>,
    /// acestep-1.5: text2music (default: from the words alone); with audio - cover (the recording performed anew: input its words, instructions its sound), repaint (repaint_start to repaint_end made again; an end past the recording extends it), edit (new words sung over the recording: source_lyrics the words it sings now), lego (a vocal track over an instrumental: acestep-1.5-base)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<AudioTask>,
    /// acestep-1.5: beats a bar: "4", "3", "6"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_signature: Option<String>,
    /// true: answer when it ends (the job goes on if the call drops); false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.speak takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioSpeakReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// guidance with instructions (Breeze TTS 2: 4 by default for design and direction; 1 turns it off)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    /// carry on from speech made here (a book in chunks): the file name (or url) of the chunk before; its last sentences (up to 15 s, cut at a pause) are the in-context reference, after the saved voice's sample when voice names one - pace and voice go on, anchored to the voice. Needs a model that clones (picked for you when model is not given); a built-in voice name is dropped
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub continue_from: Option<String>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// the text to say (Breeze TTS 2: vocal events inline - English (laugh) (sigh) (cough) (clears throat), Chinese [笑] [叹气] [咳嗽] [清嗓子]; VibeVoice: a script, "Speaker 1: ...\nSpeaker 2: ...", up to four speakers)
    pub input: String,
    /// how to say it (CustomVoice), or the voice to make up (VoiceDesign, Breeze TTS 2: write it in the language of the text); with a cloned voice (ref_audio or a saved voice) the direction of its tone, emotion, pace (Breeze TTS 2: voice direction - picked for you when model is not given)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// english, chinese, japanese ... or auto (default). Checked when submitted against the model's list (Qwen3-TTS: chinese, english, german, italian, portuguese, spanish, japanese, korean, french, russian); VoxCPM2 has no list and takes any
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// the engine's own: greedy, temperature, top-k, streaming
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// a recording of the voice to clone: WAV as base64 or a data: URL, or upload:<name> (audio.upload, any format it takes) - the Base model
    #[serde(rename = "ref_audio", default, skip_serializing_if = "Option::is_none")]
    pub ref_audio: Option<String>,
    /// a multi-speaker script's recordings, one a speaker in order (WAV as base64, a data: URL, or upload:<name>); not with voices / voice / ref_audio
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_audios: Option<Vec<String>>,
    /// what the recording says: the closer, in-context clone (Breeze TTS 2: required, the exact transcript)
    #[serde(rename = "ref_text", default, skip_serializing_if = "Option::is_none")]
    pub ref_text: Option<String>,
    /// the most to make
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    /// a built-in voice (qwen3-tts-custom: serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan; audio.info lists them) or a saved one (audio.voices.list: spoken by a model that clones). Checked when submitted, the model loaded or not: an unknown name is a 400 at once (the message lists the voices)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// a multi-speaker script's saved voices (audio.voices.list), one a speaker in order: Speaker 1's first (VibeVoice: up to four; picked for you when model is not given). Not with voice / ref_audio
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voices: Option<Vec<String>>,
    /// true: answer when it ends (the job goes on if the call drops); false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.transcribe takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioTranscribeReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// the recording: WAV, MP3, FLAC, Ogg or M4A as base64, a data: URL, or upload:<name> (audio.upload); any rate, mixed to mono
    pub audio: String,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// a code or a name (en, english, de, german ...); none or auto: detected on the first 30 s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// the engine's own: condition (each window prompted with the text before it), no-speech (the silence threshold), fallback (decode an unsure window again, warmer)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// text that leads the first window: names, spelling, style (Whisper's initial_prompt)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// into English instead of the language spoken
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translate: Option<bool>,
    /// true: answer when it ends (the job goes on if the call drops); false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// each word's start and end (and the model's probability for it)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub words: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.voices.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioVoicesListReq {
    /// only this owner's voices
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.voices.add takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioVoicesAddReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// lowercase letters, digits, - and _
    pub name: String,
    /// who it belongs to (an app or a person): voices.list filters by it, and replace / voices.remove of an owned voice must name it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub owner: Option<String>,
    /// over a voice of that name
    #[serde(rename = "replace", default, skip_serializing_if = "Option::is_none")]
    pub replace: Option<bool>,
    /// the recording, base64, a data: URL or upload:<name> (audio.upload)
    pub sample: String,
    /// what the recording says (a closer clone)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.voices.design takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioVoicesDesignReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// the voice: age, gender, timbre, pace, emotion, accent
    pub instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// a voice-design model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub name: String,
    /// who it belongs to (as voices.add)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub owner: Option<String>,
    #[serde(rename = "replace", default, skip_serializing_if = "Option::is_none")]
    pub replace: Option<bool>,
    /// another seed, another speaker of that kind
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    /// what the sample says (default: a ten-second line in the language)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.voices.enroll takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioVoicesEnrollReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// this card only (default: the model's cards, the first with room)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub gpu: Option<i64>,
    /// names the job for audio.status and audio.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// the model to enroll it for (default: the registered one that speaks enrolled voices)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// the saved voice (audio.voices.list)
    pub name: String,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// fit it again when it is enrolled already
    #[serde(rename = "replace", default, skip_serializing_if = "Option::is_none")]
    pub replace: Option<bool>,
    /// the fit's steps
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 50, max = 20000, default = 2000)]
    pub steps: Option<i64>,
    /// true: answer when it ends; false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.voices.remove takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioVoicesRemoveReq {
    pub name: String,
    /// the voice's owner, when it has one
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.info takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioInfoReq {
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.progress takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioProgressReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.history takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioHistoryReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.status takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioStatusReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.jobs takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioJobsReq {
    /// ended jobs listed
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 0, max = 200, default = 20)]
    pub limit: Option<i64>,
    /// only this project's
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.queue.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioQueueListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.queue.move takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioQueueMoveReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 1, default = 1)]
    pub position: Option<i64>,
    /// give it this priority too (where later jobs go)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9)]
    pub priority: Option<i64>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.queue.pause takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioQueuePauseReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.queue.resume takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioQueueResumeReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.upload takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioUploadReq {
    /// the recording, base64 (a data: URL too)
    pub data: String,
    /// the file's name; kept as up_<unix seconds>_<name>.wav (letters, digits, . _ - kept, the rest _, 60 characters)
    pub name: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.cancel takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioCancelReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What audio.describePromptGuideMD takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct AudioDescribePromptGuideMdReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.generate takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dGenerateReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// guidance
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    /// at most this many triangles (decimated to it)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub faces: Option<i64>,
    /// or a GLB made here to paint (its file name or url), as mesh
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// the picture: PNG, JPEG or WebP as base64, a data: URL or upload:<name> (model3d.upload); a transparent background is the object's outline, an opaque one is cut out (remove_background)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// several views of the object (a multi-view model: model3d.info's abilities.views); each a picture or {image, side}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_items = 8, json = r#"{"items": {"oneOf": [{"type": "string"}, {"properties": {"image": {"type": "string"}, "side": {"description": "front, back, left, right, top, bottom, or an azimuth in degrees", "type": "string"}}, "type": "object"}]}}"#)]
    pub images: Option<Vec<serde_json::Value>>,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// a mesh to paint instead of making one (a model whose card says paint: Hunyuan3D 2.1, hunyuan3d-2.1-paint): a GLB or OBJ as base64 or a data: URL; image says how it is painted
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<String>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// the engine's own (model3d.info lists them)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit: a job of this id queued, running or done is answered instead of a new one (failed / cancelled: made again)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// cut the background out of an opaque picture
    #[serde(rename = "remove_background", default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub remove_background: Option<bool>,
    /// the shape's grid / octree resolution: detail against time and memory
    #[serde(rename = "resolution", default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub resolution: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    /// denoising steps of each stage
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub steps: Option<i64>,
    /// PBR textures too (base colour, metallic-roughness); default: what the model does (model3d.info)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<bool>,
    /// the texture's side in texels
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub texture_size: Option<i64>,
    /// true: answer when it ends (the job goes on if the call drops); false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.rig takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dRigReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    /// or a GLB made here (model3d.generate's file name or url)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    /// an animating model: frames to make
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub frames: Option<i64>,
    /// queue it held: it waits until <kind>.queue.resume names it (its label), its project or batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub held: Option<bool>,
    /// names the job for <kind>.status and <kind>.cancel (default: its id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 120)]
    pub label: Option<String>,
    /// the mesh: GLB or OBJ as base64, a data: URL or upload:<name> (model3d.upload)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<String>,
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// the engine's own
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
    /// a waiting job goes ahead of every waiting job of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// a group of jobs held or released together (<kind>.queue.pause / resume), listed with <kind>.jobs {project}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// an animating model: the motion, in words ("a dragon flapping its wings")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// false: refused at once (409 busy, retryable) when the job would not start now - a server free and no runnable job waiting before it; true (default): it waits its turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub queue: Option<bool>,
    /// idempotent submit (as model3d.generate)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub steps: Option<i64>,
    /// true: answer when it ends; false: answer at once with the job
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = true)]
    pub wait: Option<bool>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.tool takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dToolReq {
    /// remesh: the edge length (default: the mesh's mean)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge: Option<f64>,
    /// decimate: at most this many triangles
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub faces: Option<i64>,
    /// or a GLB made here
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// the mesh: GLB or OBJ as base64, a data: URL or upload:<name>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<String>,
    pub op: MeshOp,
    /// remesh: passes (5)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub passes: Option<i64>,
    /// uv: the texture's side (2048)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub size: Option<i64>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.info takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dInfoReq {
    /// the model's id; none: the loaded one, else the first enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.progress takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dProgressReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.history takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dHistoryReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.status takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dStatusReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.jobs takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dJobsReq {
    /// ended jobs listed
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 0, max = 200, default = 20)]
    pub limit: Option<i64>,
    /// only this project's
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.queue.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dQueueListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.queue.move takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dQueueMoveReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 1, default = 1)]
    pub position: Option<i64>,
    /// give it this priority too (where later jobs go)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9)]
    pub priority: Option<i64>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.queue.pause takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dQueuePauseReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.queue.resume takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dQueueResumeReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.upload takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dUploadReq {
    /// base64 (a data: URL too)
    pub data: String,
    /// the file's name: png jpg jpeg webp glb obj; kept as up_<unix seconds>_<name>
    pub name: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.cancel takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dCancelReq {
    /// the job's label (looked up first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// or the reentrant_id it was submitted with (used when label is absent or empty)
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What model3d.describePromptGuideMD takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct Model3dDescribePromptGuideMdReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.generate takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoGenerateReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// a group inside the project, held or cleared together
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub batch: Option<String>,
    /// a camera name: prev_cam / first_cam anchor on the series' clips of the same camera
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    /// how the clip carries on from another, as one object (it sets guide_clip, first_audio, first_audio_s, first_frame, chain_mode, exposure_ref; those given flat win). There is no renormalisation step here: pin_renorm and other keys are listed back in ignored
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(json = r#"{"properties": {"color_reanchor": {"description": "true (= from), an anchor or a picture: the carried pictures scaled to its brightness (exposure_ref) - gain 0.75-1.35 on luminance, against drift"}, "context_frames": {"description": "0 or 17k + 5 (22 = 0.92 s): the last N frames of from as one moving keyframe at frame 0 (guide_clip) - the clip replays them, so trim N/2 each side at the join", "type": "integer"}, "from": {"default": "prev", "description": "anchor or file: the clip carried on from", "type": "string"}, "mode": {"description": "still: chain_mode", "enum": ["video", "png", "latent"], "type": "string"}, "sound": {"default": true, "description": "with context_frames: the sound's last N/24 s too (first_audio, first_audio_s)", "type": "boolean"}, "still": {"description": "from's last frame as the first frame (first_frame); default: when context_frames is 0", "type": "boolean"}}}"#)]
    pub chain: Option<serde_json::Map<String, serde_json::Value>>,
    /// how a first_frame clip is carried over: its mp4's last frame, its lossless .last.png, its last latent, or not at all
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = "video")]
    pub chain_mode: Option<ChainMode>,
    /// the clip's place in its project's timeline (1 and up). Say it: without it the timeline reads the label's NN/MM, else the #N of reentrant_id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 1, max = 100000)]
    pub clip: Option<i64>,
    /// noise on the first frame (harmful here: shows as confetti)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cond_noise_aug: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 1, default = 1)]
    pub control_end: Option<f64>,
    /// a mask clip or picture (file): regenerate inside it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_mask: Option<String>,
    /// the clip the mask applies to (file)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_source: Option<String>,
    /// the part of the steps the control steers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 1, default = 0)]
    pub control_start: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 10, default = 1)]
    pub control_strength: Option<f64>,
    /// a ControlNet control clip (file); not with Ref2VA
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_video: Option<String>,
    /// the denoiser, by video.engines.list's quant (Q8_0 is read as INT8); REF2VA is chosen by itself when ref_images or ref_videos are given
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = "INT8")]
    pub engine: Option<String>,
    /// anchor or file: a picture whose exposure and colour the clip keeps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_ref: Option<String>,
    /// anchor or file: the sound the clip continues from
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_audio: Option<String>,
    /// seconds of first_audio to continue from
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_audio_s: Option<f64>,
    /// file: a driving soundtrack (a song, video.upload): the clip's length of it from soundtrack_start pinned under the whole clip as an audio keyframe - the picture is made to it (a singer's mouth to the singing) and the clip carries the file's own sound. Not with first_audio; a chain carries the picture only. For a whole song: a chain of clips, each from where its first frame is in the song (the previous start + (frames - context_frames) / 24), joined, the song laid under the join
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soundtrack: Option<String>,
    /// seconds into the soundtrack at the clip's first frame (default 0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, default = 0)]
    pub soundtrack_start: Option<f64>,
    /// true: fix the clip's exposure after the decode - every frame lifted by exposure_offset (the decoder darkens), a guide_clip's frames matched to the guide's own luma, and the frames after it eased over exposure_ease frames from the guide's last luma to the clip's steady level (the flash that follows a cut). Off by default; a request without it is unchanged. The report carries luma_before and luma_after per frame under exposure
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_fix: Option<bool>,
    /// with exposure_fix: luma (0-255) added to every frame (default 2.3)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 20, default = 2.3)]
    pub exposure_offset: Option<f64>,
    /// with exposure_fix: frames after the guide over which the luma eases to the steady level (default 24; 0: only the guide's own frames are matched)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 240, default = 24)]
    pub exposure_ease: Option<i64>,
    /// with exposure_fix and no guide_clip: the luma (0-255) the clip's first frame should have (default: the first_frame picture's)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 255)]
    pub exposure_target: Option<f64>,
    /// anchor or file: the clip starts on this picture / clip's last frame
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame: Option<String>,
    /// the length exactly, instead of seconds: 17k + 5 (22, 39, 56 ... 107, 124 ... 362); another count is refused, the message naming the nearest two
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 22, max = 362)]
    pub frames: Option<i64>,
    /// a motion guide: "<anchor or file>[:frames[:at]]" - the last frames of a clip as one moving keyframe
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guide_clip: Option<String>,
    /// as width
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 256, max = 1344, default = 576, json = r#"{"multipleOf": 32}"#)]
    pub height: Option<i64>,
    /// "<series> NN/MM: text", e.g. "Harbour 03/12: the lamp": NN is the clip's place in video.timeline (the edit order), the text before it the series the anchors prev / first look in and the project when none is given. Letters, digits, _ and  -.,:'()!?/ only (others are dropped); labels name clips for video.cancel, video.queue.move and video.queue.remove
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 80)]
    pub label: Option<String>,
    /// anchor or file: the clip ends on this picture (prev: the newest clip anywhere)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame: Option<String>,
    /// "path:strength"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(json = r#"{"items": {"maxLength": 200, "type": "string"}}"#)]
    pub loras: Option<Vec<String>>,
    /// ignored: H3 runs at cfg 1 (one pass a step, no unconditional branch), so a negative prompt has nothing to steer - say what is there instead (negations render). Accepted and listed in the answer's ignored
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// a queued clip goes ahead of every clip of a lower priority (same priority: first come first made)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9, default = 0)]
    pub priority: Option<i64>,
    /// default: the label's series
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 60)]
    pub project: Option<String>,
    /// read as written - tokenized whole, no template, no sections parsed. H3 was trained on three fields in a fixed order: integrated_multimodal_description (shots, actions, camera, dialogue), overall_soundscape, non_diegetic_music; video.describePromptGuideMD has the form and its rules
    pub prompt: String,
    /// queue the clip when the studio is busy (false: 409 busy)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = false)]
    pub queue: Option<bool>,
    /// idempotent submit: a clip of this id queued, running or made is answered (reentered: true) instead of a new one; a failed one is made again
    #[serde(rename = "reentrant_id", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_length = 200)]
    pub reentrant_id: Option<String>,
    /// voices to speak in: <Audio 1> ... in the prompt
    #[serde(rename = "ref_audios", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_items = 3)]
    pub ref_audios: Option<Vec<String>>,
    /// reference pictures resized to the canvas, or kept as large as fits
    #[serde(rename = "ref_image_size", default, skip_serializing_if = "Option::is_none")]
    pub ref_image_size: Option<RefImageSize>,
    /// Ref2VA reference pictures (files in the clips directory), <Image 1> ... in the prompt
    #[serde(rename = "ref_images", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_items = 9)]
    pub ref_images: Option<Vec<String>>,
    /// the reference clips' sound too
    #[serde(rename = "ref_video_sound", default, skip_serializing_if = "Option::is_none")]
    pub ref_video_sound: Option<bool>,
    /// Ref2VA reference clips
    #[serde(rename = "ref_videos", default, skip_serializing_if = "Option::is_none")]
    #[schema(max_items = 3)]
    pub ref_videos: Option<Vec<String>>,
    /// "from-to" seconds to regenerate in source; a source shorter than the clip is extended
    #[serde(rename = "regen", default, skip_serializing_if = "Option::is_none")]
    pub regen: Option<String>,
    /// "x0,y0,x1,y1" pixels: regenerate only inside
    #[serde(rename = "regen_box", default, skip_serializing_if = "Option::is_none")]
    pub regen_box: Option<String>,
    /// ComfyUI's name (none: the engine's own Euler)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    /// ComfyUI's name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
    /// length; snapped up to the model's frame grid (17k + 5 frames at 24 fps: 4 s -> 107 frames); the model is trained on 5.2-15.1 s. The answer's frames and seconds say what is made
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 1, max = 15.1, default = 10)]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -2147483648, max = 2147483648, default = 0)]
    pub seed: Option<i64>,
    /// flow shift of the sound (3)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift_audio: Option<f64>,
    /// flow shift of the picture (12)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift_video: Option<f64>,
    /// anchor or file: a masked run - this clip's latents are kept outside the regenerated part
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// denoising steps; below 8 only with a turbo LoRA
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 1, max = 40, default = 10)]
    pub steps: Option<i64>,
    /// the text encoder
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = "teacher")]
    pub te: Option<TextEncoder>,
    /// the text encoder sees the reference pictures
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub te_pictures: Option<bool>,
    /// upscale at decode (1.5: the canvas 1.5 times); above 1 only
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 1)]
    pub upscale: Option<f64>,
    /// with upscale
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = "esrgan-general")]
    pub upscaler: Option<Upscaler>,
    /// a multiple of 32; width x height at most 1032192 pixels (768 x 1344); video.canvases.list has the tested ones
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 256, max = 1344, default = 768, json = r#"{"multipleOf": 32}"#)]
    pub width: Option<i64>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.status takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoStatusReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.wait takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoWaitReq {
    /// the longest it waits (above 60: refused)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 60, default = 30)]
    pub timeout_s: Option<f64>,
    /// the last version seen
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.cancel takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoCancelReq {
    /// a clip's label (none: the running clip)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.queue.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoQueueListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.queue.move takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoQueueMoveReq {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = 1, default = 1)]
    pub position: Option<i64>,
    /// give it this priority too (where later clips go)
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    #[schema(min = -9, max = 9)]
    pub priority: Option<i64>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.queue.remove takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoQueueRemoveReq {
    pub label: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.queue.pause takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoQueuePauseReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.queue.resume takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoQueueResumeReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.queue.clear takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoQueueClearReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.jobs.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoJobsListReq {
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub limit: Option<i64>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.jobs.get takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoJobsGetReq {
    /// h3_YYYYMMDD_HHMMSS
    pub name: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.timeline takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoTimelineReq {
    pub project: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.projects.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoProjectsListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.films.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoFilmsListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.templates.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoTemplatesListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.engines.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoEnginesListReq {
    /// hold the model this long after the request ends: nothing stops it for room meanwhile. Clamped to NS_MAX_RSVP_S (a day), never shortened by a smaller value, 0 releases; the answer says rsvp_granted_s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub rsvp_s: Option<f64>,
    /// the longest this request may wait to START. A known wait over it is declined at once (wait_too_long, retryable, nothing queued or stopped); an unknown one goes ahead and is withdrawn at the limit; a model already loaded with a free server always starts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0)]
    pub max_willing_to_wait_s: Option<f64>,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.canvases.list takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoCanvasesListReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.summary takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoSummaryReq {
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.upload takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoUploadReq {
    /// the file, base64 (a data: URL too)
    pub data: String,
    /// the file's name: png jpg jpeg webp mp4 mov webm mkv wav mp3 flac m4a ogg; saved as ref_<unix seconds>_<name> (letters, digits, . _ - kept, the rest _, 60 characters)
    pub name: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// What video.describePromptGuideMD takes.
#[derive(Deserialize, Serialize, Rpc)]
#[allow(non_snake_case)]
pub struct VideoDescribePromptGuideMdReq {
    /// the model's registry id (system.models)
    pub model: String,
    /// what else the caller sent: passed on as it came
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// The box: the GPUs (memory by process and vouched for), every model's state, the front ends
#[rpc(name = "system.state", tags = "system", example = r#"{}"#)]
fn system_state(h: &Arc<Home>, req: SystemStateReq) -> Answer {
    h.legacy("system.state", req)
}

/// Is anything working: a clip made or waiting to run, a picture, sound or chat answered, a model loading. video.accepts says whether video.generate starts a clip at once (else 409 busy without queue: true)
#[rpc(name = "system.busy", tags = "system", example = r#"{}"#)]
fn system_busy(h: &Arc<Home>, req: SystemBusyReq) -> Answer {
    h.legacy("system.busy", req)
}

/// Is the control plane here, and does it take work: `available` (it answered), `maintenance` and, when on, `reason` (the notice's words) and `retry_after_s`. Nothing is loaded or started: the cheap call a client makes before sending work and after an error
#[rpc(name = "system.ping", tags = "system", example = r#"{}"#)]
fn system_ping(h: &Arc<Home>, req: SystemPingReq) -> Answer {
    h.legacy("system.ping", req)
}

/// The release, the API's version, the kinds enabled, the features this build has (names a client can test for) and the methods callable now
#[rpc(name = "system.version", tags = "system", example = r#"{}"#)]
fn system_version(h: &Arc<Home>, req: SystemVersionReq) -> Answer {
    h.legacy("system.version", req)
}

/// A file an answer linked: its bytes and SHA-256 (to check a download), mtime, URL
#[rpc(name = "system.files.stat", tags = "system", example = r#"{"kind": "video", "name": "h3_20261011_220723.mp4"}"#)]
fn system_files_stat(h: &Arc<Home>, req: SystemFilesStatReq) -> Answer {
    h.legacy("system.files.stat", req)
}

/// Delete a file: a picture, a song or speech, a clip (with its record, last frame, latents and sound tail; not while it is made) or an upload (ref_...)
#[rpc(name = "system.files.delete", tags = "system", example = r#"{"kind": "video", "name": "h3_20261011_220723"}"#)]
fn system_files_delete(h: &Arc<Home>, req: SystemFilesDeleteReq) -> Answer {
    h.legacy("system.files.delete", req)
}

/// Every registered model: kind, enabled, its GPUs, its state (disabled, unloaded, loading, loaded, busy, ready), what it can do (capabilities, methods); downloaded: true. Then every model a card offers that is not downloaded: downloaded: false, state not_downloaded, enabled false, bytes, the drive and folder it would go to, fits, its download under way (pull) - system.models.pull fetches one
#[rpc(name = "system.models", tags = "system", example = r#"{}"#)]
fn system_models(h: &Arc<Home>, req: SystemModelsReq) -> Answer {
    h.legacy("system.models", req)
}

/// Download a model a card offers (system.models: downloaded false), in the background: its files fetched (or linked when the machine has them; a partial file resumed), each SHA-256 checked, then the model registered and enabled. Into the first models folder with the room (NS_MODELS, then NS_MODEL_DIRS, then each drive's); one download at a time a folder (a second waits, queued). Answers at once with the download (system.models.pulls follows it)
#[rpc(name = "system.models.pull", tags = "system", example = r#"{"model": "qwen-image-2.1-q8"}"#)]
fn system_models_pull(h: &Arc<Home>, req: SystemModelsPullReq) -> Answer {
    h.legacy("system.models.pull", req)
}

/// The downloads under way and the last ones, newest first: model, state (queued, downloading, verifying, done, failed, cancelled), bytes_done, bytes_total, pct, speed (bytes a second), eta_s, drive, folder, error
#[rpc(name = "system.models.pulls", tags = "system", example = r#"{}"#)]
fn system_models_pulls(h: &Arc<Home>, req: SystemModelsPullsReq) -> Answer {
    h.legacy("system.models.pulls", req)
}

/// Stop a model's download: curl stopped, what it fetched kept (a pull again resumes it), nothing registered
#[rpc(name = "system.models.pull.cancel", tags = "system", example = r#"{"model": "qwen-image-2.1-q8"}"#)]
fn system_models_pull_cancel(h: &Arc<Home>, req: SystemModelsPullCancelReq) -> Answer {
    h.legacy("system.models.pull.cancel", req)
}

/// The GPUs, one neutral list: name, memory (total, used, free, and what the enabled models are counted for), load, power, temperature, the models and processes on each
#[rpc(name = "system.cards", tags = "system", example = r#"{}"#)]
fn system_cards(h: &Arc<Home>, req: SystemCardsReq) -> Answer {
    h.legacy("system.cards", req)
}

/// Offer a model on GPU(s): it loads on its first request and unloads when idle; chat models join Open WebUI's list
#[rpc(name = "system.enable", tags = "system", example = r#"{"gpus": [0], "model": "qwen-image-2.1-q8"}"#)]
fn system_enable(h: &Arc<Home>, req: SystemEnableReq) -> Answer {
    h.legacy("system.enable", req)
}

/// Change the cards a model may use (enabled or not); more than fit is allowed (the card shows overbooked). A loaded model moved off its card unloads: its next request loads it on the new cards
#[rpc(name = "system.gpus", tags = "system", example = r#"{"gpus": [0, 1], "model": "qwen3.8-27b-q4"}"#)]
fn system_gpus(h: &Arc<Home>, req: SystemGpusReq) -> Answer {
    h.legacy("system.gpus", req)
}

/// Withdraw a model: unloaded (after the requests it is answering), off Open WebUI's list and this spec
#[rpc(name = "system.disable", tags = "system", example = r#"{"model": "qwen-image-2.1-q8"}"#)]
fn system_disable(h: &Arc<Home>, req: SystemDisableReq) -> Answer {
    h.legacy("system.disable", req)
}

/// Load an enabled model now (otherwise its first request does)
#[rpc(name = "system.load", tags = "system", example = r#"{"model": "minimax-music3", "wait": true}"#)]
fn system_load(h: &Arc<Home>, req: SystemLoadReq) -> Answer {
    h.legacy("system.load", req)
}

/// Unload a model now (it stays enabled)
#[rpc(name = "system.unload", tags = "system", example = r#"{"model": "minimax-music3"}"#)]
fn system_unload(h: &Arc<Home>, req: SystemUnloadReq) -> Answer {
    h.legacy("system.unload", req)
}

/// Pin an enabled model: it loads now (in the background) and stays: no idle unload, never stopped for another model's room (that one answers 503 no_room naming the pin). Kept across a restart of the front door; disabling the model unpins it
#[rpc(name = "system.pin", tags = "system", example = r#"{"model": "qwen-image-2.1-q8"}"#)]
fn system_pin(h: &Arc<Home>, req: SystemPinReq) -> Answer {
    h.legacy("system.pin", req)
}

/// Unpin a model: only the flag is cleared; it stays loaded until the idle rules take it
#[rpc(name = "system.unpin", tags = "system", example = r#"{"model": "qwen-image-2.1-q8"}"#)]
fn system_unpin(h: &Arc<Home>, req: SystemUnpinReq) -> Answer {
    h.legacy("system.unpin", req)
}

/// The ETA a request would get now - {wait_s, run_s, total_s, samples, basis (shape | model | none), spread} from the last 100 requests of its shape - and whether max_willing_to_wait_s would accept it; nothing is started, queued or loaded, so a caller can ask first and choose another machine
#[rpc(name = "system.eta", tags = "system", example = r#"{"method": "image.generate", "params": {"width": 1024, "height": 1024, "steps": 28}, "max_willing_to_wait_s": 120}"#)]
fn system_eta(h: &Arc<Home>, req: SystemEtaReq) -> Answer {
    h.legacy("system.eta", req)
}

/// The settings file (nextsycl.conf): its text, every setting the program reads (value, default, what reads it), the file's others
#[rpc(name = "system.config", tags = "system", example = r#"{}"#)]
fn system_config(h: &Arc<Home>, req: SystemConfigReq) -> Answer {
    h.legacy("system.config", req)
}

/// Change settings (the file copied first): set {NAME: value, NAME: null to remove} or text (the whole file); answers what changed and what must restart
#[rpc(name = "system.configSet", tags = "system", example = r#"{"set": {"NS_IDLE_MINUTES": "15"}}"#)]
fn system_config_set(h: &Arc<Home>, req: SystemConfigSetReq) -> Answer {
    h.legacy("system.configSet", req)
}

/// Restart what reads the settings: serve, switch, studio, video, image, audio or llm
#[rpc(name = "system.restart", tags = "system", example = r#"{"service": "switch"}"#)]
fn system_restart(h: &Arc<Home>, req: SystemRestartReq) -> Answer {
    h.legacy("system.restart", req)
}

/// Maintenance mode: the box set aside for development. On, what would start work (llm.chat, image/audio/video generation, edits, speech, system.load / enable, the OpenAI POST routes) is refused with error code maintenance (retryable, retry_after_s, the notice in error.maintenance; /v1: HTTP 503 + Retry-After); status, listings, cancels and guides answer as always; queued image and audio jobs wait, the video queue pauses, running work finishes; system.busy / version / state carry the notice, and Open WebUI's chats are answered with it. The answer to turning it on carries a bypass key: a request with the header X-Maintenance-Bypass: <key> goes through (a developer testing meanwhile). Without on: the current state
#[rpc(name = "system.maintenance", tags = "system", example = r#"{"eta_s": 3600, "message": "kernel work", "on": true, "unload": true}"#)]
fn system_maintenance(h: &Arc<Home>, req: SystemMaintenanceReq) -> Answer {
    h.legacy("system.maintenance", req)
}

/// The drives the models are on: device, PCIe link, mount, size, free space, the models' bytes, the read speed as measured (and when); each model's bytes by drive, whether it streams its weights from the disk while it serves, and for each other drive what a move needs or why it cannot move now; the move under way (or the last)
#[rpc(name = "system.drives", tags = "system", example = r#"{}"#)]
fn system_drives(h: &Arc<Home>, req: SystemDrivesReq) -> Answer {
    h.legacy("system.drives", req)
}

/// Measure a drive's read speed in the background (up to 4 GiB of its largest model files read with O_DIRECT, past the page cache; nothing written), kept with its date (system.drives). Refused while a model loads from it or a move uses it
#[rpc(name = "system.drive.measure", tags = "system", example = r#"{"drive": "nvme1n1"}"#)]
fn system_drive_measure(h: &Arc<Home>, req: SystemDriveMeasureReq) -> Answer {
    h.legacy("system.drive.measure", req)
}

/// Move a model's files to another drive, as a job (its progress in system.drives' move): each file copied and read back (size, SHA-256), then the registry repointed, then the old files removed - those another model names stay. Refused while the model is loaded or loading (stop it first), another move runs, or the drive lacks the room (plus 1%, at least 4 GiB); nothing loads the model while it moves
#[rpc(name = "system.move", tags = "system", example = r#"{"drive": "nvme1n1", "model": "olmocr-2-7b-q8"}"#)]
fn system_move(h: &Arc<Home>, req: SystemMoveReq) -> Answer {
    h.legacy("system.move", req)
}

/// Stop the move under way: what it copied is removed and the registry stays as it was (once the registry is repointed it finishes)
#[rpc(name = "system.move.cancel", tags = "system", example = r#"{}"#)]
fn system_move_cancel(h: &Arc<Home>, req: SystemMoveCancelReq) -> Answer {
    h.legacy("system.move.cancel", req)
}

/// A chat completion (OpenAI's body; stream is ignored - use /v1/chat/completions to stream)
#[rpc(name = "llm.chat", tags = "llm", example = r#"{"max_tokens": 200, "messages": [{"content": "Say hello in five words.", "role": "user"}], "model": "qwen3.8-flash-next-iq2_xs"}"#)]
fn llm_chat(h: &Arc<Home>, req: LlmChatReq) -> Answer {
    h.legacy("llm.chat", req)
}

/// The chat requests running and the last ones that ended, with their speeds as the server reports them live: prompt tokens (reused from the cache), read seconds, tokens generated, tok/s
#[rpc(name = "llm.requests", tags = "llm", example = r#"{}"#)]
fn llm_requests(h: &Arc<Home>, req: LlmRequestsReq) -> Answer {
    h.legacy("llm.requests", req)
}

/// The enabled chat models (what Open WebUI lists) and which is loaded
#[rpc(name = "llm.models", tags = "llm", example = r#"{}"#)]
fn llm_models(h: &Arc<Home>, req: LlmModelsReq) -> Answer {
    h.legacy("llm.models", req)
}

/// How to prompt the model, as Markdown
#[rpc(name = "llm.describePromptGuideMD", tags = "llm", example = r#"{"model": "qwen3.8-flash-next-iq2_xs"}"#)]
fn llm_describe_prompt_guide_md(h: &Arc<Home>, req: LlmDescribePromptGuideMdReq) -> Answer {
    h.legacy("llm.describePromptGuideMD", req)
}

/// Pictures from a prompt, or from an init picture (image + strength: img2img); the model loads first if it must. Checked when submitted (a bad value is a 400 even with wait: false)
#[rpc(name = "image.generate", tags = "image", example = r#"{"model": "qwen-image-2.1-q8", "prompt": "a lighthouse at dusk, oil painting", "size": "1024x768"}"#)]
fn image_generate(h: &Arc<Home>, req: ImageGenerateReq) -> Answer {
    h.legacy("image.generate", req)
}

/// Edit or compose pictures: the first is changed, the others are references the prompt names. With reference: false and strength below 1 it is a plain img2img (as image.generate's image). Checked when submitted (a bad value is a 400 even with wait: false)
#[rpc(name = "image.edit", tags = "image", example = r#"{"images": ["data:image/png;base64,iVBORw0..."], "prompt": "make it night, keep the lighthouse"}"#)]
fn image_edit(h: &Arc<Home>, req: ImageEditReq) -> Answer {
    h.legacy("image.edit", req)
}

/// A picture made bigger (an upscaler: Real-ESRGAN - realesrgan-x4plus, realesrgan-x4plus-anime, realesr-animevideov3 2x/3x/4x - or waifu2x - cunet, upconv_7 anime / photo, denoise -1 to 3, 1x to 32x), in overlapped tiles; alpha kept. A job like image.generate (wait, reentrant_id, status, cancel)
#[rpc(name = "image.upscale", tags = "image", example = r#"{"image": "upload:up_1792400000_still.png", "model": "real-esrgan", "scale": 4, "variant": "realesrgan-x4plus"}"#)]
fn image_upscale(h: &Arc<Home>, req: ImageUpscaleReq) -> Answer {
    h.legacy("image.upscale", req)
}

/// A picture written as SVG code (StarVector-1B im2svg: a CLIP encoder and StarCoder-1B; logos, icons, simple illustrations). As the model's own generate_im2svg: the picture flattened on white, 2 beams sampled (top_p 0.9) until </svg> or max_length; the SVG may stop short there (closed: false - the caller closes or cuts it). A job like image.generate (wait, reentrant_id, status, cancel)
#[rpc(name = "image.vectorize", tags = "image", example = r#"{"image": "upload:up_1792400000_logo.png", "max_length": 5000, "model": "starvector-1b-im2svg"}"#)]
fn image_vectorize(h: &Arc<Home>, req: ImageVectorizeReq) -> Answer {
    h.legacy("image.vectorize", req)
}

/// A clip made bigger frame by frame by an upscaler (as image.upscale: the variant, scale, noise, tile), its sound copied as it was; an MP4 back. A job in the image queue (image.status, image.cancel)
#[rpc(name = "video.upscale", tags = "video", example = r#"{"file": "h3_20261019_121939", "model": "real-esrgan", "scale": 2, "variant": "realesr-animevideov3"}"#)]
fn video_upscale(h: &Arc<Home>, req: VideoUpscaleReq) -> Answer {
    h.legacy("video.upscale", req)
}

/// Read pages (an OCR model: Unlimited-OCR): their text as Markdown, tables as HTML, boxes with <|grounding|>
#[rpc(name = "image.ocr", tags = "image", example = r#"{"images": ["data:image/png;base64,iVBORw0..."], "prompt": "document parsing."}"#)]
fn image_ocr(h: &Arc<Home>, req: ImageOcrReq) -> Answer {
    h.legacy("image.ocr", req)
}

/// The model's defaults, samplers, schedules, options and LoRAs (loaded: from the server)
#[rpc(name = "image.info", tags = "image", example = r#"{}"#)]
fn image_info(h: &Arc<Home>, req: ImageInfoReq) -> Answer {
    h.legacy("image.info", req)
}

/// One job (image.generate / image.edit) by its label or reentrant_id (one of them is needed: image.jobs lists them): state, stage and what it waits for, its place in the queue, progress, its pictures when done
#[rpc(name = "image.status", tags = "image", example = r#"{"label": "poster-7"}"#)]
fn image_status(h: &Arc<Home>, req: ImageStatusReq) -> Answer {
    h.legacy("image.status", req)
}

/// The image jobs: running, waiting (in run order) and the last ended, each as image.status answers it
#[rpc(name = "image.jobs", tags = "image", example = r#"{"limit": 5}"#)]
fn image_jobs(h: &Arc<Home>, req: ImageJobsReq) -> Answer {
    h.legacy("image.jobs", req)
}

/// The waiting image jobs in run order (held ones too: held, waiting_for)
#[rpc(name = "image.queue.list", tags = "image", example = r#"{}"#)]
fn image_queue_list(h: &Arc<Home>, req: ImageQueueListReq) -> Answer {
    h.legacy("image.queue.list", req)
}

/// Move a waiting job to a place in the queue (1 = made next)
#[rpc(name = "image.queue.move", tags = "image", example = r#"{"label": "poster-9", "position": 1}"#)]
fn image_queue_move(h: &Arc<Home>, req: ImageQueueMoveReq) -> Answer {
    h.legacy("image.queue.move", req)
}

/// Hold a waiting job (label or reentrant_id), a project or batch (also the ones submitted later), or (none) the whole image queue; running jobs go on
#[rpc(name = "image.queue.pause", tags = "image", example = r#"{"project": "Harbour"}"#)]
fn image_queue_pause(h: &Arc<Home>, req: ImageQueuePauseReq) -> Answer {
    h.legacy("image.queue.pause", req)
}

/// Release what queue.pause held (none: everything)
#[rpc(name = "image.queue.resume", tags = "image", example = r#"{}"#)]
fn image_queue_resume(h: &Arc<Home>, req: ImageQueueResumeReq) -> Answer {
    h.legacy("image.queue.resume", req)
}

/// Keep a picture here once (PNG, JPEG, WebP; at most 40 MP): image.generate's image, image.edit's and image.ocr's images then name it as "upload:<name>" instead of sending it again. Kept until system.files.delete {kind: image, name}
#[rpc(name = "image.upload", tags = "image", example = r#"{"data": "iVBORw0KGgo...", "name": "hero.png"}"#)]
fn image_upload(h: &Arc<Home>, req: ImageUploadReq) -> Answer {
    h.legacy("image.upload", req)
}

/// Stop a job by its label or reentrant_id (running: between steps; waiting: at once; ended: ok with already_ended); neither: whatever the image servers are making
#[rpc(name = "image.cancel", tags = "image", example = r#"{"label": "poster-7"}"#)]
fn image_cancel(h: &Arc<Home>, req: ImageCancelReq) -> Answer {
    h.legacy("image.cancel", req)
}

/// Sizes for image.generate: the model's tested ones with their aspect, and the time measured here per size
#[rpc(name = "image.canvases.list", tags = "image", example = r#"{}"#)]
fn image_canvases_list(h: &Arc<Home>, req: ImageCanvasesListReq) -> Answer {
    h.legacy("image.canvases.list", req)
}

/// The request running: step of steps, how many wait
#[rpc(name = "image.progress", tags = "image", example = r#"{}"#)]
fn image_progress(h: &Arc<Home>, req: ImageProgressReq) -> Answer {
    h.legacy("image.progress", req)
}

/// The pictures made, newest first, with their settings
#[rpc(name = "image.history", tags = "image", example = r#"{}"#)]
fn image_history(h: &Arc<Home>, req: ImageHistoryReq) -> Answer {
    h.legacy("image.history", req)
}

/// How to prompt the model, as Markdown
#[rpc(name = "image.describePromptGuideMD", tags = "image", example = r#"{"model": "qwen-image-2.1-q8"}"#)]
fn image_describe_prompt_guide_md(h: &Arc<Home>, req: ImageDescribePromptGuideMdReq) -> Answer {
    h.legacy("image.describePromptGuideMD", req)
}

/// A song from a description and lyrics, or from a recording (acestep-1.5: a cover, a repaint or an extension, new words over it); the model loads first if it must; the answer links the WAV
#[rpc(name = "audio.generate", tags = "audio", example = r#"{"input": "[verse]\nRiver runs slow tonight\n[chorus]\nCarry me home", "instructions": "warm acoustic folk, 90 BPM, a soft male voice", "model": "minimax-music3", "seconds": 60}"#)]
fn audio_generate(h: &Arc<Home>, req: AudioGenerateReq) -> Answer {
    h.legacy("audio.generate", req)
}

/// Speech from text (a speech model: Qwen3-TTS, VoxCPM2, Breeze TTS 2, VibeVoice - a script of up to four speakers); the answer links the WAV. Checked when submitted (voice, language, continue_from, ref_audio: a 400 even with wait: false). Two run at once (one per card: system.busy.parallel), so a book can keep two chunks in flight with wait: false - but a chunk with continue_from needs the chunk before it finished
#[rpc(name = "audio.speak", tags = "audio", example = r#"{"input": "Good morning. The bread is still warm.", "instructions": "cheerful, a little hurried", "language": "english", "model": "qwen3-tts-custom", "voice": "ryan"}"#)]
fn audio_speak(h: &Arc<Home>, req: AudioSpeakReq) -> Answer {
    h.legacy("audio.speak", req)
}

/// Speech to text (a recogniser: Whisper large-v3-turbo, 99 languages): the text, its segments timed by the model's timestamps and (words, the default) each word timed by its cross-attention. A job like audio.speak (wait, reentrant_id, status, cancel); long recordings go in 30 s windows, so an hour takes minutes
#[rpc(name = "audio.transcribe", tags = "audio", example = r#"{"audio": "upload:up_1792400000_take2.wav", "language": "english", "words": true}"#)]
fn audio_transcribe(h: &Arc<Home>, req: AudioTranscribeReq) -> Answer {
    h.legacy("audio.transcribe", req)
}

/// The saved voices (cloned or designed): name, owner, what the sample says, its description, length, whether it can be spoken now (ready) and by which enabled models; sample: its WAV
#[rpc(name = "audio.voices.list", tags = "audio", example = r#"{"owner": "film-app"}"#)]
fn audio_voices_list(h: &Arc<Home>, req: AudioVoicesListReq) -> Answer {
    h.legacy("audio.voices.list", req)
}

/// Save a voice cloned from a recording (WAV, MP3, FLAC, Ogg, M4A; 3-15 s of one speaker is best)
#[rpc(name = "audio.voices.add", tags = "audio", example = r#"{"name": "grandpa", "sample": "data:audio/mpeg;base64,SUQzBAAAAA...", "text": "Well, back in my day we walked to school."}"#)]
fn audio_voices_add(h: &Arc<Home>, req: AudioVoicesAddReq) -> Answer {
    h.legacy("audio.voices.add", req)
}

/// Make up a voice from a description (the voice-design model reads a sample) and save it
#[rpc(name = "audio.voices.design", tags = "audio", example = r#"{"instructions": "A deep, warm male narrator in his fifties, unhurried, a slight rasp.", "language": "english", "name": "narrator"}"#)]
fn audio_voices_design(h: &Arc<Home>, req: AudioVoicesDesignReq) -> Answer {
    h.legacy("audio.voices.design", req)
}

/// Enroll a saved voice for a speech model without an encoder (Voxtral TTS; experimental: the voice fitted through its decoder, a minute or two on a card) - a job (audio.status / audio.jobs / audio.cancel by its label; progress: the fit's step). It waits for a card with the fit's memory (~8 GiB) and 1.5 GiB more free, as a load does; once done the voice is offered to that model (audio.voices.list: models) and audio.speak speaks it there as voice
#[rpc(name = "audio.voices.enroll", tags = "audio", example = r#"{"label": "enroll-grandpa", "name": "grandpa", "wait": false}"#)]
fn audio_voices_enroll(h: &Arc<Home>, req: AudioVoicesEnrollReq) -> Answer {
    h.legacy("audio.voices.enroll", req)
}

/// Delete a saved voice (an owned one: owner must match)
#[rpc(name = "audio.voices.remove", tags = "audio", example = r#"{"name": "narrator"}"#)]
fn audio_voices_remove(h: &Arc<Home>, req: AudioVoicesRemoveReq) -> Answer {
    h.legacy("audio.voices.remove", req)
}

/// The model's defaults, limits and options (a speech model: its voices and languages)
#[rpc(name = "audio.info", tags = "audio", example = r#"{}"#)]
fn audio_info(h: &Arc<Home>, req: AudioInfoReq) -> Answer {
    h.legacy("audio.info", req)
}

/// The request running: its phase and how far
#[rpc(name = "audio.progress", tags = "audio", example = r#"{}"#)]
fn audio_progress(h: &Arc<Home>, req: AudioProgressReq) -> Answer {
    h.legacy("audio.progress", req)
}

/// The songs and speech made, newest first
#[rpc(name = "audio.history", tags = "audio", example = r#"{}"#)]
fn audio_history(h: &Arc<Home>, req: AudioHistoryReq) -> Answer {
    h.legacy("audio.history", req)
}

/// One job (audio.generate / audio.speak) by its label or reentrant_id (one of them is needed: audio.jobs lists them): state, stage and what it waits for, its place in the queue, progress, its sound when done
#[rpc(name = "audio.status", tags = "audio", example = r#"{"label": "theme-3"}"#)]
fn audio_status(h: &Arc<Home>, req: AudioStatusReq) -> Answer {
    h.legacy("audio.status", req)
}

/// The audio jobs: running, waiting (in run order) and the last ended, each as audio.status answers it
#[rpc(name = "audio.jobs", tags = "audio", example = r#"{"limit": 5}"#)]
fn audio_jobs(h: &Arc<Home>, req: AudioJobsReq) -> Answer {
    h.legacy("audio.jobs", req)
}

/// The waiting audio jobs in run order (held ones too: held, waiting_for)
#[rpc(name = "audio.queue.list", tags = "audio", example = r#"{}"#)]
fn audio_queue_list(h: &Arc<Home>, req: AudioQueueListReq) -> Answer {
    h.legacy("audio.queue.list", req)
}

/// Move a waiting job to a place in the queue (1 = made next)
#[rpc(name = "audio.queue.move", tags = "audio", example = r#"{"label": "chapter-4", "position": 1}"#)]
fn audio_queue_move(h: &Arc<Home>, req: AudioQueueMoveReq) -> Answer {
    h.legacy("audio.queue.move", req)
}

/// Hold a waiting job (label or reentrant_id), a project or batch (also the ones submitted later), or (none) the whole audio queue; running jobs go on
#[rpc(name = "audio.queue.pause", tags = "audio", example = r#"{"project": "Harbour"}"#)]
fn audio_queue_pause(h: &Arc<Home>, req: AudioQueuePauseReq) -> Answer {
    h.legacy("audio.queue.pause", req)
}

/// Release what queue.pause held (none: everything)
#[rpc(name = "audio.queue.resume", tags = "audio", example = r#"{}"#)]
fn audio_queue_resume(h: &Arc<Home>, req: AudioQueueResumeReq) -> Answer {
    h.legacy("audio.queue.resume", req)
}

/// Keep a recording here once (WAV, MP3, FLAC, Ogg, M4A; kept as WAV): audio.speak's ref_audio and audio.voices.add's sample then name it as "upload:<name>" instead of sending it again. Kept until system.files.delete {kind: audio, name}
#[rpc(name = "audio.upload", tags = "audio", example = r#"{"data": "SUQzBAAAAA...", "name": "grandpa.mp3"}"#)]
fn audio_upload(h: &Arc<Home>, req: AudioUploadReq) -> Answer {
    h.legacy("audio.upload", req)
}

/// Stop a job by its label or reentrant_id (running: between steps or frames; waiting: at once; ended: ok with already_ended); neither: whatever the audio servers are making
#[rpc(name = "audio.cancel", tags = "audio", example = r#"{"label": "theme-3"}"#)]
fn audio_cancel(h: &Arc<Home>, req: AudioCancelReq) -> Answer {
    h.legacy("audio.cancel", req)
}

/// How to prompt the model, as Markdown
#[rpc(name = "audio.describePromptGuideMD", tags = "audio", example = r#"{"model": "minimax-music3"}"#)]
fn audio_describe_prompt_guide_md(h: &Arc<Home>, req: AudioDescribePromptGuideMdReq) -> Answer {
    h.legacy("audio.describePromptGuideMD", req)
}

/// A mesh from a picture of an object, or several views of it (TRELLIS.2, Hunyuan3D, Pixal3D; the model loads first if it must): its shape and, with texture, PBR textures - or, given a mesh (or file), that mesh painted like the picture; the answer links the GLB. A job like image.generate (wait, reentrant_id, status, cancel)
#[rpc(name = "model3d.generate", tags = "model3d", example = r#"{"image": "data:image/png;base64,iVBORw0...", "model": "trellis2", "reentrant_id": "chair-1", "seed": 42, "texture": true, "wait": false}"#)]
fn model3d_generate(h: &Arc<Home>, req: Model3dGenerateReq) -> Answer {
    h.legacy("model3d.generate", req)
}

/// A mesh given a skeleton and skin weights (UniRig, RigNet), or set in motion by words (AnimateAnyMesh: prompt); the answer links the rigged or animated GLB. A job like model3d.generate
#[rpc(name = "model3d.rig", tags = "model3d", example = r#"{"file": "1792132541-trellis2-42.glb", "model": "unirig"}"#)]
fn model3d_rig(h: &Arc<Home>, req: Model3dRigReq) -> Answer {
    h.legacy("model3d.rig", req)
}

/// The CPU mesh tools, here (no GPU, no model): decimate (to a triangle budget), remesh (even triangles), uv (a fresh atlas, the texture or colours baked across), convert; the result is saved beside the meshes made (model3d.history) and linked
#[rpc(name = "model3d.tool", tags = "model3d", example = r#"{"faces": 20000, "file": "1792132541-trellis2-42.glb", "op": "decimate"}"#)]
fn model3d_tool(h: &Arc<Home>, req: Model3dToolReq) -> Answer {
    h.legacy("model3d.tool", req)
}

/// The model's abilities (generate, texture, views, rig, animate), defaults and options (loaded: from the server)
#[rpc(name = "model3d.info", tags = "model3d", example = r#"{}"#)]
fn model3d_info(h: &Arc<Home>, req: Model3dInfoReq) -> Answer {
    h.legacy("model3d.info", req)
}

/// The request running: its phase (image, structure, shape, texture, decode, mesh, bake, skeleton, skin, motion) and how far
#[rpc(name = "model3d.progress", tags = "model3d", example = r#"{}"#)]
fn model3d_progress(h: &Arc<Home>, req: Model3dProgressReq) -> Answer {
    h.legacy("model3d.progress", req)
}

/// The meshes made, newest first, with their settings and pictures
#[rpc(name = "model3d.history", tags = "model3d", example = r#"{}"#)]
fn model3d_history(h: &Arc<Home>, req: Model3dHistoryReq) -> Answer {
    h.legacy("model3d.history", req)
}

/// One job (model3d.generate / model3d.rig) by its label or reentrant_id: state, stage and what it waits for, its place in the queue, progress, its GLB when done
#[rpc(name = "model3d.status", tags = "model3d", example = r#"{"label": "chair-1"}"#)]
fn model3d_status(h: &Arc<Home>, req: Model3dStatusReq) -> Answer {
    h.legacy("model3d.status", req)
}

/// The 3D jobs: running, waiting (in run order) and the last ended, each as model3d.status answers it
#[rpc(name = "model3d.jobs", tags = "model3d", example = r#"{"limit": 5}"#)]
fn model3d_jobs(h: &Arc<Home>, req: Model3dJobsReq) -> Answer {
    h.legacy("model3d.jobs", req)
}

/// The waiting 3D jobs in run order (held ones too: held, waiting_for)
#[rpc(name = "model3d.queue.list", tags = "model3d", example = r#"{}"#)]
fn model3d_queue_list(h: &Arc<Home>, req: Model3dQueueListReq) -> Answer {
    h.legacy("model3d.queue.list", req)
}

/// Move a waiting job to a place in the queue (1 = made next)
#[rpc(name = "model3d.queue.move", tags = "model3d", example = r#"{"label": "chair-2", "position": 1}"#)]
fn model3d_queue_move(h: &Arc<Home>, req: Model3dQueueMoveReq) -> Answer {
    h.legacy("model3d.queue.move", req)
}

/// Hold a waiting job (label or reentrant_id), a project or batch (also the ones submitted later), or (none) the whole 3D queue; running jobs go on
#[rpc(name = "model3d.queue.pause", tags = "model3d", example = r#"{"project": "Props"}"#)]
fn model3d_queue_pause(h: &Arc<Home>, req: Model3dQueuePauseReq) -> Answer {
    h.legacy("model3d.queue.pause", req)
}

/// Release what queue.pause held (none: everything)
#[rpc(name = "model3d.queue.resume", tags = "model3d", example = r#"{}"#)]
fn model3d_queue_resume(h: &Arc<Home>, req: Model3dQueueResumeReq) -> Answer {
    h.legacy("model3d.queue.resume", req)
}

/// Keep a picture (PNG, JPEG, WebP) or a mesh (GLB, OBJ) here once: model3d.generate's image / images and model3d.rig's mesh then name it as "upload:<name>". Kept until system.files.delete {kind: model3d, name}
#[rpc(name = "model3d.upload", tags = "model3d", example = r#"{"data": "iVBORw0KGgo...", "name": "chair.png"}"#)]
fn model3d_upload(h: &Arc<Home>, req: Model3dUploadReq) -> Answer {
    h.legacy("model3d.upload", req)
}

/// Stop a job by its label or reentrant_id (running: between steps; waiting: at once; ended: ok with already_ended); neither: whatever the 3D servers are making
#[rpc(name = "model3d.cancel", tags = "model3d", example = r#"{"label": "chair-1"}"#)]
fn model3d_cancel(h: &Arc<Home>, req: Model3dCancelReq) -> Answer {
    h.legacy("model3d.cancel", req)
}

/// How to get good meshes from the model, as Markdown
#[rpc(name = "model3d.describePromptGuideMD", tags = "model3d", example = r#"{"model": "trellis2"}"#)]
fn model3d_describe_prompt_guide_md(h: &Arc<Home>, req: Model3dDescribePromptGuideMdReq) -> Answer {
    h.legacy("model3d.describePromptGuideMD", req)
}

/// Make a clip: it starts at once when the studio is idle (queued: false, the job), else it is queued with queue: true (queued: true, its position) or refused with 409 busy. Anchors (first_frame, last_frame, exposure_ref, first_audio, guide_clip, source) take prev, prev_cam, first, first_cam - clips of the label's series, see label - or a file in the clips directory (video.upload, or a finished clip's name)
#[rpc(name = "video.generate", tags = "video", example = r#"{"label": "Fox 01/03: snow", "prompt": "integrated_multimodal_description: [Shot 1] Live-action, a red fox trots through deep snow at dawn.\n\noverall_soundscape: soft wind, crunching snow.\n\nnon_diegetic_music: None.", "queue": true, "seconds": 5, "steps": 8}"#)]
fn video_generate(h: &Arc<Home>, req: VideoGenerateReq) -> Answer {
    h.legacy("video.generate", req)
}

/// The clip running or last: stage, percent, ETA, timings, a preview picture of the shot while it samples (preview), the queue, version
#[rpc(name = "video.status", tags = "video", example = r#"{}"#)]
fn video_status(h: &Arc<Home>, req: VideoStatusReq) -> Answer {
    h.legacy("video.status", req)
}

/// Long poll: answers when the status's version changes, or at the timeout
#[rpc(name = "video.wait", tags = "video", example = r#"{"timeout_s": 30, "version": ""}"#)]
fn video_wait(h: &Arc<Home>, req: VideoWaitReq) -> Answer {
    h.legacy("video.wait", req)
}

/// Stop the running clip at its next step; with a label, that clip only: stopped if it runs, dropped from the queue if it waits
#[rpc(name = "video.cancel", tags = "video", example = r#"{"label": "Fox 02/03: the den"}"#)]
fn video_cancel(h: &Arc<Home>, req: VideoCancelReq) -> Answer {
    h.legacy("video.cancel", req)
}

/// The queue in the order it will run (each item as video.generate checked it)
#[rpc(name = "video.queue.list", tags = "video", example = r#"{}"#)]
fn video_queue_list(h: &Arc<Home>, req: VideoQueueListReq) -> Answer {
    h.legacy("video.queue.list", req)
}

/// Move a queued clip (the first with that label) to a place in the queue: 1 = made next
#[rpc(name = "video.queue.move", tags = "video", example = r#"{"label": "Fox 03/03: night", "position": 1}"#)]
fn video_queue_move(h: &Arc<Home>, req: VideoQueueMoveReq) -> Answer {
    h.legacy("video.queue.move", req)
}

/// Drop the queued clips with that label (a running one: video.cancel)
#[rpc(name = "video.queue.remove", tags = "video", example = r#"{"label": "Fox 03/03: night"}"#)]
fn video_queue_remove(h: &Arc<Home>, req: VideoQueueRemoveReq) -> Answer {
    h.legacy("video.queue.remove", req)
}

/// Hold a project or batch (none: everything)
#[rpc(name = "video.queue.pause", tags = "video", example = r#"{"project": "Harbour"}"#)]
fn video_queue_pause(h: &Arc<Home>, req: VideoQueuePauseReq) -> Answer {
    h.legacy("video.queue.pause", req)
}

/// Release a project or batch (none: everything)
#[rpc(name = "video.queue.resume", tags = "video", example = r#"{}"#)]
fn video_queue_resume(h: &Arc<Home>, req: VideoQueueResumeReq) -> Answer {
    h.legacy("video.queue.resume", req)
}

/// Drop the queue, or a project's / batch's part of it
#[rpc(name = "video.queue.clear", tags = "video", example = r#"{}"#)]
fn video_queue_clear(h: &Arc<Home>, req: VideoQueueClearReq) -> Answer {
    h.legacy("video.queue.clear", req)
}

/// Finished clips (GET /v1/video/files/<file> fetches one)
#[rpc(name = "video.jobs.list", tags = "video", example = r#"{"limit": 20}"#)]
fn video_jobs_list(h: &Arc<Home>, req: VideoJobsListReq) -> Answer {
    h.legacy("video.jobs.list", req)
}

/// A clip's record: settings, per-stage seconds, energy
#[rpc(name = "video.jobs.get", tags = "video", example = r#"{"name": "h3_20261009_113734"}"#)]
fn video_jobs_get(h: &Arc<Home>, req: VideoJobsGetReq) -> Answer {
    h.legacy("video.jobs.get", req)
}

/// A project's clips in edit order: done (with its file's name), running, held, queued
#[rpc(name = "video.timeline", tags = "video", example = r#"{"project": "Harbour"}"#)]
fn video_timeline(h: &Arc<Home>, req: VideoTimelineReq) -> Answer {
    h.legacy("video.timeline", req)
}

/// The projects and batches
#[rpc(name = "video.projects.list", tags = "video", example = r#"{}"#)]
fn video_projects_list(h: &Arc<Home>, req: VideoProjectsListReq) -> Answer {
    h.legacy("video.projects.list", req)
}

/// Joined films
#[rpc(name = "video.films.list", tags = "video", example = r#"{}"#)]
fn video_films_list(h: &Arc<Home>, req: VideoFilmsListReq) -> Answer {
    h.legacy("video.films.list", req)
}

/// Prompt templates
#[rpc(name = "video.templates.list", tags = "video", example = r#"{}"#)]
fn video_templates_list(h: &Arc<Home>, req: VideoTemplatesListReq) -> Answer {
    h.legacy("video.templates.list", req)
}

/// The denoisers the daemon has, and the video.generate keys each takes: INT8 / Q6_K / Q4_K_M (text and image to video: anchors, guides, ControlNet, masked runs) and REF2VA (reference pictures and clips; no ControlNet or masked runs). ref_audios, anchors and guides work with every one
#[rpc(name = "video.engines.list", tags = "video", example = r#"{}"#)]
fn video_engines_list(h: &Arc<Home>, req: VideoEnginesListReq) -> Answer {
    h.legacy("video.engines.list", req)
}

/// The canvases and their measured cost
#[rpc(name = "video.canvases.list", tags = "video", example = r#"{}"#)]
fn video_canvases_list(h: &Arc<Home>, req: VideoCanvasesListReq) -> Answer {
    h.legacy("video.canvases.list", req)
}

/// Measured GPU time per second of video, energy, queue ETA
#[rpc(name = "video.summary", tags = "video", example = r#"{}"#)]
fn video_summary(h: &Arc<Home>, req: VideoSummaryReq) -> Answer {
    h.legacy("video.summary", req)
}

/// Put a reference, control video, mask or keyframe beside the clips (its name is what the other keys take). The request is at most 256 MiB, so the file at most about 190 MiB as base64. Uploads are kept: nothing deletes them (they sit in the clips directory with the clips)
#[rpc(name = "video.upload", tags = "video", example = r#"{"data": "iVBORw0KGgo...", "name": "fox.png"}"#)]
fn video_upload(h: &Arc<Home>, req: VideoUploadReq) -> Answer {
    h.legacy("video.upload", req)
}

/// How to prompt the model, as Markdown
#[rpc(name = "video.describePromptGuideMD", tags = "video", example = r#"{"model": "minimax-h3"}"#)]
fn video_describe_prompt_guide_md(h: &Arc<Home>, req: VideoDescribePromptGuideMdReq) -> Answer {
    h.legacy("video.describePromptGuideMD", req)
}

rpcgen::rpc_methods![sync Arc<Home>, RpcFail;
    system_state, system_busy, system_ping, system_version, system_files_stat, system_files_delete,
    system_models, system_models_pull, system_models_pulls, system_models_pull_cancel, system_cards, system_enable,
    system_gpus, system_disable, system_load, system_unload, system_pin, system_unpin, system_eta, system_config, system_config_set,
    system_restart, system_maintenance, system_drives, system_drive_measure, system_move, system_move_cancel,
    llm_chat, llm_requests, llm_models, llm_describe_prompt_guide_md, image_generate, image_edit,
    image_upscale, image_vectorize, video_upscale, image_ocr, image_info, image_status,
    image_jobs, image_queue_list, image_queue_move, image_queue_pause, image_queue_resume, image_upload,
    image_cancel, image_canvases_list, image_progress, image_history, image_describe_prompt_guide_md, audio_generate,
    audio_speak, audio_transcribe, audio_voices_list, audio_voices_add, audio_voices_design, audio_voices_enroll, audio_voices_remove,
    audio_info, audio_progress, audio_history, audio_status, audio_jobs, audio_queue_list,
    audio_queue_move, audio_queue_pause, audio_queue_resume, audio_upload, audio_cancel, audio_describe_prompt_guide_md,
    model3d_generate, model3d_rig, model3d_tool, model3d_info, model3d_progress, model3d_history,
    model3d_status, model3d_jobs, model3d_queue_list, model3d_queue_move, model3d_queue_pause, model3d_queue_resume,
    model3d_upload, model3d_cancel, model3d_describe_prompt_guide_md, video_generate, video_status, video_wait,
    video_cancel, video_queue_list, video_queue_move, video_queue_remove, video_queue_pause, video_queue_resume,
    video_queue_clear, video_jobs_list, video_jobs_get, video_timeline, video_projects_list, video_films_list,
    video_templates_list, video_engines_list, video_canvases_list, video_summary, video_upload, video_describe_prompt_guide_md,
];

#[cfg(test)]
mod tests {
    /// Every name a request refers to (an option's enum) is a named type of the table: the document's components
    /// carry them, so its $refs resolve.
    #[test]
    fn every_reference_is_described() {
        let comps = rpcgen::components(&super::infos()).unwrap();
        for m in super::methods() {
            for f in &m.fields {
                let s = serde_json::to_string(&rpcgen::field_schema(f)).unwrap();
                for r in s.split("#/components/schemas/").skip(1) {
                    let name = r.split('"').next().unwrap();
                    assert!(comps.contains_key(name), "{} {}: {name} is not described", m.full(), f.name);
                }
            }
        }
        assert!(comps.contains_key("ChainMode") && comps.contains_key("FileKind"));
    }
}

