//! rpcgen: an RPC method is its handler (`#[rpc]`), and every description of it is made from that handler and its
//! types - the route table, the OpenAPI document, an API page and a web UI's typed TypeScript client. Nothing
//! about a method is written a second time by hand, so the document, the router and the UI cannot disagree: a type
//! that is not an `Rpc` type, a handler of the wrong shape, two methods of one name or names of two styles in one
//! table do not compile, and a UI call with a wrong field does not type-check (`tsc`).
//!
//! It lives in vibe-creative-platform and is vendored into nextcuda (vendor/rpcgen, copied by its
//! tooling/sync_rpcgen.sh with the copy's hash recorded; a test in each repo fails when a copy was edited by hand).
//! The platform uses the `axum` feature (async handlers answering a Response); nextcuda, without it, sync handlers
//! in `rpc_methods![sync State, Error; ...]`.
//!
//! Names: one table uses one style - `GetJob` (PascalCase, verb first: the platform) or `system.ping` (dotted
//! lower case: nextcuda) - and matching them in any case is the transport's (both match without case).
//!
//! Before this (2026-10-26) the platform kept a method table, a hand-written OpenAPI YAML and untyped UI calls:
//! the Model Maker's document listed 15 of its 52 methods and five paths it did not serve.
//!
//! A method, and its table:
//!
//! ```
//! use rpcgen::{rpc, rpc_methods, Rpc};
//! type App = std::sync::Arc<()>;
//! type Error = (axum::http::StatusCode, String);
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct ThingRef { pub id: String }
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Thing { pub id: String, pub tags: Vec<String> }
//! #[derive(serde::Deserialize, Rpc)]
//! #[serde(rename_all = "lowercase")]
//! pub enum Kind { Image, Audio }
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct Make {
//!     /// how far from the picture
//!     #[schema(min = 0.05, max = 1, default = 0.6)]
//!     pub strength: Option<f64>,
//!     pub kind: Kind,
//! }
//! /// Make a thing
//! #[rpc(name = "CreateThing")]
//! async fn create_thing(_app: App, _req: Make) -> Result<Thing, Error> { Ok(Thing { id: "1".into(), tags: vec![] }) }
//! rpc_methods![App; get_thing, create_thing];
//! let infos: Vec<_> = rpc_methods().into_iter().map(|m| m.info).collect();
//! let ts = rpcgen::typescript("t", &infos).unwrap();
//! assert!(ts.contains("export type Kind = \"image\" | \"audio\";") && ts.contains("kind: Kind;") && ts.contains("strength?: number | null;"), "{ts}");
//! rpcgen::parity(&infos).unwrap();
//! let doc = rpcgen::openapi(serde_json::json!({"title": "t", "version": "1"}), "/", &infos).unwrap();
//! assert!(doc.contains("maximum: 1") && doc.contains("- image"), "{doc}");
//! /// Get a thing by its id
//! #[rpc(name = "GetThing", tags = "things")]
//! async fn get_thing(_app: App, req: ThingRef) -> Result<Thing, Error> {
//!     Ok(Thing { id: req.id, tags: vec![] })
//! }
//! let m = &rpc_methods()[0].info;
//! assert_eq!((m.name, m.summary), ("GetThing", "Get a thing by its id"));
//! assert!(rpcgen::typescript("t", &[m.clone()]).unwrap().contains("GetThing: (req: ThingRef): Promise<Thing> => post(\"GetThing\", req)"));
//! ```
//!
//! What does not compile - two methods of one name in a table:
//!
//! ```compile_fail,E0080
//! use rpcgen::{rpc, rpc_methods, Rpc};
//! type App = std::sync::Arc<()>;
//! type Error = (axum::http::StatusCode, String);
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct ThingRef { pub id: String }
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Thing { pub id: String, pub tags: Vec<String> }
//! /// Get a thing
//! #[rpc(name = "GetThing")]
//! async fn a(_app: App, req: ThingRef) -> Result<Thing, Error> { Ok(Thing { id: req.id, tags: vec![] }) }
//! /// Get it again
//! #[rpc(name = "GetThing")]
//! async fn b(_app: App, req: ThingRef) -> Result<Thing, Error> { Ok(Thing { id: req.id, tags: vec![] }) }
//! rpc_methods![App; a, b];
//! ```
//!
//! a name that does not say what it does:
//!
//! ```compile_fail
//! use rpcgen::{rpc, rpc_methods, Rpc};
//! type App = std::sync::Arc<()>;
//! type Error = (axum::http::StatusCode, String);
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct ThingRef { pub id: String }
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Thing { pub id: String, pub tags: Vec<String> }
//! /// The thing
//! #[rpc(name = "Thing")]
//! async fn a(_app: App, req: ThingRef) -> Result<Thing, Error> { Ok(Thing { id: req.id, tags: vec![] }) }
//! ```
//!
//! A sync table (nextcuda's), with and without a request: it compiles and answers.
//!
//! ```
//! use rpcgen::{rpc, BadPayload, Rpc};
//! #[derive(Debug)]
//! pub struct Fail(String);
//! impl From<BadPayload> for Fail { fn from(b: BadPayload) -> Self { Fail(b.0) } }
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct Ping { pub n: i64 }
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Pong { pub n: i64 }
//! /// Ping it
//! #[rpc(name = "system.ping")]
//! fn ping(_s: &(), req: Ping) -> Result<Pong, Fail> { Ok(Pong { n: req.n }) }
//! /// Its state
//! #[rpc(name = "system.state")]
//! fn state(_s: &()) -> Result<Pong, Fail> { Ok(Pong { n: 0 }) }
//! rpcgen::rpc_methods![sync (), Fail; ping, state];
//! let m = rpc_methods();
//! assert_eq!((m[0].handler)(&(), serde_json::json!({"n": 3})).unwrap(), serde_json::json!({"n": 3}));
//! ```
//!
//! A download's media types are media types:
//!
//! ```compile_fail
//! use rpcgen::{rpc, Raw, Rpc};
//! type App = std::sync::Arc<()>;
//! type Error = (axum::http::StatusCode, String);
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct ThingRef { pub id: String }
//! /// Download a thing
//! #[rpc(name = "DownloadThing", produces = "wav")]
//! async fn a(_app: App, _req: ThingRef) -> Result<Raw, Error> { unimplemented!() }
//! ```
//!
//! a request (or response) type that is not an Rpc type:
//!
//! ```compile_fail,E0277
//! use rpcgen::{rpc, rpc_methods, Rpc};
//! type App = std::sync::Arc<()>;
//! type Error = (axum::http::StatusCode, String);
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct ThingRef { pub id: String }
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Thing { pub id: String, pub tags: Vec<String> }
//! #[derive(serde::Deserialize)]
//! pub struct Plain { pub id: String }
//! /// Get a thing
//! #[rpc(name = "GetThing")]
//! async fn a(_app: App, req: Plain) -> Result<Thing, Error> { Ok(Thing { id: req.id, tags: vec![] }) }
//! rpc_methods![App; a];
//! ```
//!
//! a field of a fixed set of values written as a string (it is an enum):
//!
//! ```compile_fail
//! use rpcgen::Rpc;
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct Make {
//!     #[schema(one_of = "image, audio")]
//!     pub kind: String,
//! }
//! ```
//!
//! and a method with nothing said of it:
//!
//! ```compile_fail
//! use rpcgen::{rpc, rpc_methods, Rpc};
//! type App = std::sync::Arc<()>;
//! type Error = (axum::http::StatusCode, String);
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct ThingRef { pub id: String }
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Thing { pub id: String, pub tags: Vec<String> }
//! #[rpc(name = "GetThing")]
//! async fn a(_app: App, req: ThingRef) -> Result<Thing, Error> { Ok(Thing { id: req.id, tags: vec![] }) }
//! ```

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Map, Value};

pub use rpcgen_macros::{rpc, rpc_methods, Rpc};
#[doc(hidden)]
pub use serde_json;

#[cfg(feature = "axum")]
pub use web::*;

/// The async methods of an axum server (the platform): a handler answers a Response.
#[cfg(feature = "axum")]
mod web {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use std::future::Future;
    use std::pin::Pin;

    /// What a handler's future answers.
    pub type Reply = Pin<Box<dyn Future<Output = Response> + Send>>;

    /// A request's headers, for a handler that takes them as its third parameter.
    pub use axum::http::HeaderMap;

    /// One method of a table: what it is, and its handler (the request decoded from JSON, the answer encoded).
    pub struct Method<S> {
        pub info: MethodInfo,
        pub handler: fn(S, Value, HeaderMap) -> Reply,
        /// whether a body fits the request type, without running the method
        pub check: fn(&Value) -> Result<(), BadPayload>,
    }

    impl<S> Clone for Method<S> {
        fn clone(&self) -> Self {
            Method { info: self.info.clone(), handler: self.handler, check: self.check }
        }
    }

    /// How an answer becomes the HTTP response: JSON for a typed value, the bytes for a download.
    pub trait RpcResponse: RpcType {
        fn into_reply(self) -> Response;
    }

    impl<T: Serialize + RpcType> RpcResponse for T {
        fn into_reply(self) -> Response {
            axum::Json(self).into_response()
        }
    }

    impl RpcResponse for Raw {
        fn into_reply(self) -> Response {
            let mut r = (StatusCode::OK, self.bytes).into_response();
            if let Ok(v) = self.content_type.parse() {
                r.headers_mut().insert(axum::http::header::CONTENT_TYPE, v);
            }
            if let Some(n) = self.filename {
                let safe: String = n.chars().map(|c| if c.is_ascii_graphic() && c != '"' || c == ' ' { c } else { '_' }).collect();
                if let Ok(v) = format!("attachment; filename=\"{safe}\"").parse() {
                    r.headers_mut().insert(axum::http::header::CONTENT_DISPOSITION, v);
                }
            }
            r
        }
    }

    /// The routes of a table: `POST /api/<Method>` each, its body (none is `{}`) decoded into the method's request.
    /// A name is matched as written here; the public door matches it in any case and passes it on as written.
    pub fn router<S: Clone + Send + Sync + 'static>(methods: Vec<Method<S>>) -> axum::Router<S> {
        let mut r = axum::Router::new();
        for m in methods {
            let handler = m.handler;
            r = r.route(
                &format!("/api/{}", m.info.name),
                axum::routing::post(move |axum::extract::State(s): axum::extract::State<S>, headers: HeaderMap, body: axum::body::Bytes| async move {
                    // no body, or only whitespace, is {} (a method whose fields are all optional); else it is JSON
                    let text = std::str::from_utf8(&body).unwrap_or("").trim();
                    if text.is_empty() || text == "null" {
                        return handler(s, Value::Null, headers).await;
                    }
                    match serde_json::from_str::<Value>(text) {
                        Ok(v) => handler(s, v, headers).await,
                        Err(e) => bad_payload(format!("the body is not JSON: {e}")),
                    }
                }),
            );
        }
        r
    }

    /// An answer passed on as the handler made it (a maker's handler answering a Response): untyped, and counted with
    /// the Untyped ones (`untyped_responses`) until the method answers a type of its own.
    pub struct Passthrough(pub Response);

    impl RpcType for Passthrough {
        fn shape(_: &mut Defs) -> Shape {
            Shape::Untyped
        }
    }

    impl RpcResponse for Passthrough {
        fn into_reply(self) -> Response {
            self.0
        }
    }

    /// An answer the handler made as JSON itself (a maker's handler answering a Response), described as `T`: the
    /// document and the client show T, and `Shaped::check` holds each answer to it (`conforms`), so the description
    /// cannot drift from what is sent. A debug build (the tests, `tooling/integration.sh`) answers a mismatch as an
    /// error, `answer_shape`, naming where; a release build sends the answer and says on stderr where it differs
    /// (the place only: a value could be a secret). Errors, other statuses and non-JSON bodies pass unchecked.
    pub struct Shaped<T>(pub Response, std::marker::PhantomData<fn() -> T>);

    impl<T: RpcType> RpcType for Shaped<T> {
        fn shape(defs: &mut Defs) -> Shape {
            T::shape(defs)
        }
    }

    impl<T: RpcType> RpcResponse for Shaped<T> {
        fn into_reply(self) -> Response {
            self.0
        }
    }

    impl<T: DeserializeOwned + Serialize> Shaped<T> {
        /// The answer, checked against T.
        pub async fn check(r: Response) -> Self {
            let json = r
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with("application/json"));
            if r.status() != StatusCode::OK || !json {
                return Shaped(r, std::marker::PhantomData);
            }
            let (parts, body) = r.into_parts();
            let bytes = match axum::body::to_bytes(body, usize::MAX).await {
                Ok(b) => b,
                Err(e) => {
                    let body = error_body(&format!("the answer could not be read: {e}"), "answer_shape", true);
                    return Shaped((StatusCode::INTERNAL_SERVER_ERROR, axum::Json(body)).into_response(), std::marker::PhantomData);
                }
            };
            let problem = match serde_json::from_slice::<Value>(&bytes) {
                Ok(v) => conforms::<T>(&v).err(),
                Err(_) => Some(Mismatch { at: String::new(), what: "the answer is not JSON".into() }),
            };
            if let Some(r) = problem.and_then(Self::mismatch) {
                return r;
            }
            Shaped(Response::from_parts(parts, axum::body::Body::from(bytes)), std::marker::PhantomData)
        }

        /// An answer the handler made as a JSON value, checked against T and sent as JSON.
        pub fn value(v: Value) -> Self {
            if let Some(r) = conforms::<T>(&v).err().and_then(Self::mismatch) {
                return r;
            }
            Shaped(axum::Json(v).into_response(), std::marker::PhantomData)
        }

        /// A debug build's error for a mismatch; a release build's line on stderr (None: send the answer).
        fn mismatch(m: Mismatch) -> Option<Self> {
            let ty = std::any::type_name::<T>().rsplit("::").next().unwrap_or("");
            if cfg!(debug_assertions) {
                let body = error_body(&format!("the answer is not a {ty}: {m}"), "answer_shape", false);
                return Some(Shaped((StatusCode::INTERNAL_SERVER_ERROR, axum::Json(body)).into_response(), std::marker::PhantomData));
            }
            eprintln!("rpcgen: an answer is not the {ty} its method declares, at `{}`", if m.at.is_empty() { "." } else { &m.at });
            None
        }
    }

    /// What a handler answers: a `Result` of an Rpc response and an error that is an HTTP response. `#[rpc]` names
    /// the answer's type through this, so the handler may spell its Result with an alias.
    pub trait RpcResult {
        type Ok: RpcResponse;
        fn into_reply(self) -> Response;
    }

    impl<T: RpcResponse, E: IntoResponse> RpcResult for Result<T, E> {
        type Ok = T;
        fn into_reply(self) -> Response {
            match self {
                Ok(v) => v.into_reply(),
                Err(e) => e.into_response(),
            }
        }
    }

    /// The error every method answers for a payload that does not fit its request type.
    pub fn bad_payload(msg: impl std::fmt::Display) -> Response {
        (StatusCode::BAD_REQUEST, axum::Json(error_body(&format!("bad payload: {msg}"), "bad_payload", false))).into_response()
    }

    /// A request from the JSON body (none is `{}`).
    pub fn decode<T: DeserializeOwned>(body: Value) -> Result<T, Response> {
        let body = if body.is_null() { json!({}) } else { body };
        serde_json::from_value(body).map_err(bad_payload)
    }
}

/// An error's machine-readable outcome for an HTTP status that came without one: what a client branches on (never
/// the status itself - docs/RPC_RULES.md, "Outcomes are in the payload; HTTP status codes only help").
pub fn code_for(status: u16) -> &'static str {
    match status {
        400 | 422 => "bad_request",
        401 | 403 => "unauthorized",
        404 => "not_found",
        405 => "bad_request",
        409 => "conflict",
        413 => "too_large",
        415 => "bad_payload",
        423 => "busy",
        429 => "busy",
        502 | 503 | 504 => "unavailable",
        s if s >= 500 => "internal",
        _ => "error",
    }
}

/// Whether the same request may work later, for a status that came without a say.
pub fn retryable_for(status: u16) -> bool {
    matches!(status, 409 | 423 | 429 | 502 | 503 | 504)
}

/// Every error's body: `{"error": words, "code": machine-readable, "retryable": bool}`. The status is set to match
/// (it helps a log and a proxy), but a client reads these.
pub fn error_body(message: &str, code: &str, retryable: bool) -> Value {
    json!({"error": message, "code": code, "retryable": retryable})
}

/// A method's error: what went wrong, its code, whether to try again, and the HTTP status that matches it.
#[derive(Debug, Clone)]
pub struct RpcError {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
    pub retryable: bool,
}

impl RpcError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        RpcError { status, code: code_for(status), message: message.into(), retryable: retryable_for(status) }
    }
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = code;
        self
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(404, message)
    }
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(400, message)
    }
    pub fn body(&self) -> Value {
        error_body(&self.message, self.code, self.retryable)
    }
}

impl From<BadPayload> for RpcError {
    fn from(b: BadPayload) -> Self {
        RpcError::new(400, b.to_string()).with_code("bad_payload")
    }
}

#[cfg(feature = "axum")]
impl axum::response::IntoResponse for RpcError {
    fn into_response(self) -> axum::response::Response {
        let status = axum::http::StatusCode::from_u16(self.status).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        (status, axum::Json(self.body())).into_response()
    }
}

/// Lenient decoders for `#[serde(deserialize_with = ...)]`: JSON has numbers, not integers, and a caller's arithmetic
/// sends 512.0 for 512 - an integer field takes a whole number written either way (and nothing else).
pub mod de {
    use serde::{Deserialize, Deserializer};

    fn whole<E: serde::de::Error>(v: serde_json::Value) -> Result<i64, E> {
        match &v {
            serde_json::Value::Number(n) => n
                .as_i64()
                .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15).map(|f| f as i64))
                .ok_or_else(|| E::custom(format!("{n} is not a whole number"))),
            other => Err(E::custom(format!("expected a whole number, got {other}"))),
        }
    }

    /// an i64 from 5 or 5.0
    pub fn int<'de, D: Deserializer<'de>>(d: D) -> Result<i64, D::Error> {
        whole(serde_json::Value::deserialize(d)?)
    }

    /// an Option<i64> from null, 5 or 5.0
    pub fn opt_int<'de, D: Deserializer<'de>>(d: D) -> Result<Option<i64>, D::Error> {
        match serde_json::Value::deserialize(d)? {
            serde_json::Value::Null => Ok(None),
            v => whole(v).map(Some),
        }
    }

    /// a Vec<i64> of whole numbers written either way
    pub fn vec_int<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<i64>, D::Error> {
        Vec::<serde_json::Value>::deserialize(d)?.into_iter().map(whole).collect()
    }

    /// an Option<Vec<i64>>
    pub fn opt_vec_int<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<i64>>, D::Error> {
        match serde_json::Value::deserialize(d)? {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::Array(a) => a.into_iter().map(whole).collect::<Result<Vec<_>, _>>().map(Some),
            other => Err(serde::de::Error::custom(format!("expected a list of whole numbers, got {other}"))),
        }
    }
}

/// A payload that does not fit a sync method's request type: the table's error type is made from it.
#[derive(Debug, Clone)]
pub struct BadPayload(pub String);

impl std::fmt::Display for BadPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bad payload: {}", self.0)
    }
}

/// One method of a sync table (nextcuda): what it is, and its handler.
pub struct SyncMethod<S, E> {
    pub info: MethodInfo,
    pub handler: fn(&S, Value) -> Result<Value, E>,
    /// whether a body fits the request type, without running the method (a request passed on elsewhere)
    pub check: fn(&Value) -> Result<(), BadPayload>,
}

impl<S, E> Clone for SyncMethod<S, E> {
    fn clone(&self) -> Self {
        SyncMethod { info: self.info.clone(), handler: self.handler, check: self.check }
    }
}

/// Whether a body fits `T` (none is `{}`), its words when not.
pub fn check<T: DeserializeOwned>(body: &Value) -> Result<(), BadPayload> {
    let body = if body.is_null() { json!({}) } else { body.clone() };
    serde_json::from_value::<T>(body).map(|_| ()).map_err(|e| BadPayload(e.to_string()))
}

/// What a sync handler answers: a `Result` of a serializable Rpc type and an error a bad payload converts into.
pub trait SyncResult {
    type Ok: RpcType;
    type Err: From<BadPayload>;
    fn into_value(self) -> Result<Value, Self::Err>;
}

impl<T: Serialize + RpcType, E: From<BadPayload>> SyncResult for Result<T, E> {
    type Ok = T;
    type Err = E;
    fn into_value(self) -> Result<Value, E> {
        let v = self?;
        serde_json::to_value(v).map_err(|e| E::from(BadPayload(format!("the answer did not encode: {e}"))))
    }
}

/// A sync handler's answer for a handler that answers `R`.
pub type SyncReply<R> = Result<Value, <R as SyncResult>::Err>;

/// A sync method's request from its JSON body (none is `{}`).
pub fn decode_sync<T: DeserializeOwned, E: From<BadPayload>>(body: Value) -> Result<T, E> {
    let body = if body.is_null() { json!({}) } else { body };
    serde_json::from_value(body).map_err(|e| E::from(BadPayload(e.to_string())))
}

/// A name's style: 1 PascalCase (`GetJob`), 2 dotted lower case (`system.ping`), 0 neither.
pub const fn style_of(name: &str) -> u8 {
    let b = name.as_bytes();
    if b.is_empty() {
        return 0;
    }
    if b[0].is_ascii_uppercase() {
        let mut i = 0;
        while i < b.len() {
            if !b[i].is_ascii_alphanumeric() {
                return 0;
            }
            i += 1;
        }
        return 1;
    }
    // dotted: each part starts lower case (system.ping, system.configSet)
    let (mut i, mut dots, mut last_dot) = (0, 0, true);
    while i < b.len() {
        let c = b[i];
        if c == b'.' {
            if last_dot {
                return 0;
            }
            dots += 1;
            last_dot = true;
        } else if last_dot && !c.is_ascii_lowercase() {
            return 0;
        } else if c.is_ascii_alphanumeric() || c == b'_' {
            last_dot = false;
        } else {
            return 0;
        }
        i += 1;
    }
    if dots > 0 && !last_dot { 2 } else { 0 }
}

/// Whether every name is of one style: `rpc_methods!` asserts it at compile time.
pub const fn one_style(names: &[&str]) -> bool {
    if names.is_empty() {
        return true;
    }
    let first = style_of(names[0]);
    let mut i = 0;
    while i < names.len() {
        if style_of(names[i]) != first || first == 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// A type's shape on the wire.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    String,
    Integer,
    Number,
    Bool,
    /// any JSON (a `Value` field: its shape is the data's own)
    Json,
    /// a response not typed yet (`Untyped`): the client gets `any`; `untyped_responses` counts these
    Untyped,
    /// nothing in particular: a method that takes no parameters
    Empty,
    /// raw bytes with a content type (a download)
    Blob,
    Array(Box<Shape>),
    Map(Box<Shape>),
    /// `Option<T>`: null, or T
    Nullable(Box<Shape>),
    /// a named type (#[derive(Rpc)]), in `Defs`
    Ref(String),
}

/// A field of an object type.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub doc: String,
    pub shape: Shape,
    /// may be left out (Option, a serde default)
    pub optional: bool,
    /// the JSON Schema keywords `#[schema(...)]` adds (minimum, default, maxLength, ...), as an object
    pub extra: Value,
    /// `#[schema(one_of = "...")]`: the only values a string takes (a union in TypeScript, an enum in the schema)
    pub one_of: Vec<String>,
}

/// An object type's fields, as `#[derive(Rpc)]` lists them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Object {
    pub fields: Vec<Field>,
    /// a flattened map (`#[serde(flatten)] rest: Map<..>`): the object has other keys too, of this shape
    pub rest: Option<Shape>,
}

impl Object {
    pub fn field<T: RpcType>(&mut self, defs: &mut Defs, name: &str, doc: &str, has_default: bool) {
        self.field_with::<T>(defs, name, doc, has_default, "", &[]);
    }

    /// A field with what `#[schema(...)]` says of it: `extra` is JSON Schema keywords as JSON text.
    pub fn field_with<T: RpcType>(&mut self, defs: &mut Defs, name: &str, doc: &str, has_default: bool, extra: &str, one_of: &[&str]) {
        let shape = T::shape(defs);
        let extra = if extra.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(extra).unwrap_or_else(|e| panic!("#[schema] on {name}: not JSON ({e}): {extra}"))
        };
        // a field's type is its Rust type and a fixed set of values is an enum: said in json they would be in the
        // client alone (the document's type is the shape's), so they are refused here
        for k in ["type", "enum"] {
            if extra.get(k).is_some() {
                panic!("#[schema(json)] on {name}: `{k}` is the field's Rust type (a set of values is a #[derive(Rpc)] enum)");
            }
        }
        self.fields.push(Field {
            name: name.into(),
            doc: doc.into(),
            shape,
            optional: has_default || T::optional(),
            extra,
            one_of: one_of.iter().map(|s| s.to_string()).collect(),
        });
    }

    /// a `#[serde(flatten)]` field: its fields are this object's
    pub fn flatten(&mut self, other: Object) {
        self.fields.extend(other.fields);
        if other.rest.is_some() {
            self.rest = other.rest;
        }
    }
}

/// A named type.
#[derive(Debug, Clone, PartialEq)]
pub enum Named {
    Object { doc: String, object: Object },
    Union { doc: String, values: Vec<String> },
    /// being described (a type that refers to itself)
    Pending,
}

/// The named types the methods of a document use.
#[derive(Debug, Clone, Default)]
pub struct Defs {
    pub types: BTreeMap<String, Named>,
}

impl Defs {
    /// Whether `name` is to be described now (false when it is, or is being, already).
    pub fn begin(&mut self, name: &str) -> bool {
        if self.types.contains_key(name) {
            return false;
        }
        self.types.insert(name.into(), Named::Pending);
        true
    }

    pub fn named(&mut self, name: &str, doc: &str, object: Object) {
        self.types.insert(name.into(), Named::Object { doc: doc.into(), object });
    }

    pub fn named_union(&mut self, name: &str, doc: &str, values: &[&str]) {
        self.types.insert(name.into(), Named::Union { doc: doc.into(), values: values.iter().map(|v| v.to_string()).collect() });
    }

    /// Another method's types: two different types under one name is a mistake a document would hide
    fn merge(&mut self, other: &Defs) -> Result<(), String> {
        for (k, v) in &other.types {
            match self.types.get(k) {
                Some(have) if have != v => return Err(format!("two different types are both called {k}: rename one")),
                Some(_) => {}
                None => {
                    self.types.insert(k.clone(), v.clone());
                }
            }
        }
        Ok(())
    }
}

/// A type a method takes or answers. `#[derive(Rpc)]` writes it for a maker's structs and unit enums; it is
/// implemented here for what serde writes plainly. A request or response type without it does not compile.
pub trait RpcType {
    fn shape(defs: &mut Defs) -> Shape;
    /// an object type's fields (for `#[serde(flatten)]`)
    fn object(defs: &mut Defs) -> Object {
        let _ = defs;
        Object::default()
    }
    /// may be left out of an object
    fn optional() -> bool {
        false
    }
}

macro_rules! plain {
    ($shape:expr => $($t:ty),*) => {$(
        impl RpcType for $t {
            fn shape(_: &mut Defs) -> Shape { $shape }
        }
    )*};
}
plain!(Shape::String => String, str, &str, std::path::PathBuf, char);
plain!(Shape::Integer => i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
plain!(Shape::Number => f32, f64);
plain!(Shape::Bool => bool);
plain!(Shape::Json => Value);

/// A JSON object of anything: an object (its keys the caller's)
impl RpcType for Map<String, Value> {
    fn shape(_: &mut Defs) -> Shape {
        Shape::Map(Box::new(Shape::Json))
    }
    fn object(_: &mut Defs) -> Object {
        Object { fields: Vec::new(), rest: Some(Shape::Json) }
    }
}

impl<T: RpcType> RpcType for Option<T> {
    fn shape(defs: &mut Defs) -> Shape {
        Shape::Nullable(Box::new(T::shape(defs)))
    }
    fn optional() -> bool {
        true
    }
}

impl<T: RpcType> RpcType for Vec<T> {
    fn shape(defs: &mut Defs) -> Shape {
        Shape::Array(Box::new(T::shape(defs)))
    }
}

impl<T: RpcType> RpcType for Box<T> {
    fn shape(defs: &mut Defs) -> Shape {
        T::shape(defs)
    }
    fn object(defs: &mut Defs) -> Object {
        T::object(defs)
    }
}

impl<K, T: RpcType> RpcType for BTreeMap<K, T> {
    fn shape(defs: &mut Defs) -> Shape {
        Shape::Map(Box::new(T::shape(defs)))
    }
    fn object(defs: &mut Defs) -> Object {
        Object { fields: Vec::new(), rest: Some(T::shape(defs)) }
    }
}

impl<K, T: RpcType, H> RpcType for std::collections::HashMap<K, T, H> {
    fn shape(defs: &mut Defs) -> Shape {
        Shape::Map(Box::new(T::shape(defs)))
    }
    fn object(defs: &mut Defs) -> Object {
        Object { fields: Vec::new(), rest: Some(T::shape(defs)) }
    }
}

/// A field of a change that may be left out (keep what is there), null (clear it) or a value (set it): a token,
/// a name. `#[serde(default, skip_serializing_if = "Patch::is_keep")]` on the field.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Patch<T> {
    #[default]
    Keep,
    Clear,
    Set(T),
}

impl<T> Patch<T> {
    pub fn is_keep(&self) -> bool {
        matches!(self, Patch::Keep)
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Patch<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match Option::<T>::deserialize(d)? {
            None => Patch::Clear,
            Some(v) => Patch::Set(v),
        })
    }
}

impl<T: Serialize> Serialize for Patch<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Patch::Set(v) => v.serialize(s),
            _ => s.serialize_none(),
        }
    }
}

impl<T: RpcType> RpcType for Patch<T> {
    fn shape(defs: &mut Defs) -> Shape {
        Shape::Nullable(Box::new(T::shape(defs)))
    }
    fn optional() -> bool {
        true
    }
}

/// A method that takes no parameters: any body (or none) is fine.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoParams;

impl<'de> serde::Deserialize<'de> for NoParams {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        serde::de::IgnoredAny::deserialize(d).map(|_| NoParams)
    }
}

impl RpcType for NoParams {
    fn shape(_: &mut Defs) -> Shape {
        Shape::Empty
    }
}

/// Where an answer differs from its type, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    /// the place, as `jobs[3].stage` (empty: the whole answer)
    pub at: String,
    pub what: String,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.at.is_empty() {
            write!(f, "{}", self.what)
        } else {
            write!(f, "{}: {}", self.at, self.what)
        }
    }
}

/// Whether a JSON value is a `T`: it decodes as one, and nothing in it is lost when that T is encoded again (a
/// field the type does not name; a null is not lost, it is what an absent Option says).
pub fn conforms<T: DeserializeOwned + Serialize>(v: &Value) -> Result<(), Mismatch> {
    let t: T = serde_path_to_error::deserialize(v).map_err(|e| {
        let at = e.path().to_string();
        Mismatch { at: if at == "." { String::new() } else { at }, what: e.into_inner().to_string() }
    })?;
    let back = serde_json::to_value(&t).map_err(|e| Mismatch { at: String::new(), what: format!("it does not encode again: {e}") })?;
    let mut lost = Vec::new();
    lost_fields(v, &back, String::new(), &mut lost);
    match lost.first() {
        None => Ok(()),
        Some(first) => Err(Mismatch {
            at: first.clone(),
            what: format!("not a field of the type{}", if lost.len() > 1 { format!(" (nor {})", lost[1..].iter().take(9).cloned().collect::<Vec<_>>().join(", ")) } else { String::new() }),
        }),
    }
}

fn lost_fields(a: &Value, b: &Value, at: String, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, va) in x {
                let here = if at.is_empty() { k.clone() } else { format!("{at}.{k}") };
                match y.get(k) {
                    None if !va.is_null() => out.push(here),
                    None => {}
                    Some(vb) => lost_fields(va, vb, here, out),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            for (i, (va, vb)) in x.iter().zip(y).enumerate() {
                lost_fields(va, vb, format!("{at}[{i}]"), out);
            }
        }
        _ => {}
    }
}

/// An answer not typed yet: what the method answered before it had a type. The client sees `any`, and the
/// count of these (`untyped_responses`) is held by a test that only lets it fall.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Untyped(pub Value);

impl RpcType for Untyped {
    fn shape(_: &mut Defs) -> Shape {
        Shape::Untyped
    }
}

/// An answer that is a file: its bytes with their Content-Type, as `Raw` sends them. For `Shaped<Blob>`, a handler
/// that makes the download itself (the document shows octet-stream, the client gets a Blob).
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Blob;

impl RpcType for Blob {
    fn shape(_: &mut Defs) -> Shape {
        Shape::Blob
    }
}

/// `{"ok": true}`: an action with nothing else to say.
#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
pub struct Done {
    pub ok: bool,
}

impl Done {
    pub fn new() -> Self {
        Done { ok: true }
    }
}

impl RpcType for Done {
    fn shape(defs: &mut Defs) -> Shape {
        if defs.begin("Done") {
            let mut o = Object::default();
            o.field::<bool>(defs, "ok", "true", false);
            defs.named("Done", "the action was done", o);
        }
        Shape::Ref("Done".into())
    }
}

/// A download: raw bytes, their content type and (when it is a file to keep) its name.
pub struct Raw {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub filename: Option<String>,
}

impl RpcType for Raw {
    fn shape(_: &mut Defs) -> Shape {
        Shape::Blob
    }
}

/// Whether every name is different: `rpc_methods!` asserts it at compile time.
pub const fn distinct(names: &[&str]) -> bool {
    let mut i = 0;
    while i < names.len() {
        let mut j = i + 1;
        while j < names.len() {
            if same(names[i].as_bytes(), names[j].as_bytes()) {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

const fn same(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// What a method is: everything the documents and the client say of it.
#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub name: &'static str,
    pub summary: &'static str,
    pub description: &'static str,
    pub tags: Vec<&'static str>,
    pub request: Shape,
    pub response: Shape,
    pub defs: Defs,
    /// `#[rpc(example = r#"{...}"#)]`: the body a caller sends, as JSON text
    pub example: Option<&'static str>,
    /// `#[rpc(examples = r#"{"name": {...}}"#)]`: several, by name, as JSON text
    pub examples: Option<&'static str>,
    /// `#[rpc(produces = "audio/wav, audio/mpeg")]`: a download's media types (none: application/octet-stream)
    pub produces: Vec<&'static str>,
}

impl MethodInfo {
    /// Whether it takes no parameters: nothing, or an object with no named fields (only a pass-through)
    pub fn takes_nothing(&self) -> bool {
        match &self.request {
            Shape::Empty => true,
            Shape::Ref(n) => matches!(self.defs.types.get(n), Some(Named::Object { object, .. }) if object.fields.is_empty()),
            _ => false,
        }
    }

    pub fn new<Req: RpcType, Resp: RpcType>(name: &'static str, summary: &'static str, description: &'static str, tags: &[&'static str]) -> Self {
        let mut defs = Defs::default();
        let request = Req::shape(&mut defs);
        let response = Resp::shape(&mut defs);
        MethodInfo { name, summary, description, tags: tags.to_vec(), request, response, defs, example: None, examples: None, produces: Vec::new() }
    }

    pub fn with_example(mut self, example: Option<&'static str>) -> Self {
        self.example = example;
        self
    }

    pub fn with_examples(mut self, examples: Option<&'static str>) -> Self {
        self.examples = examples;
        self
    }

    pub fn with_produces(mut self, produces: &[&'static str]) -> Self {
        self.produces = produces.to_vec();
        self
    }

    /// The named examples (`#[rpc(examples)]`), name -> body.
    pub fn named_examples(&self) -> Map<String, Value> {
        self.examples.and_then(|e| serde_json::from_str::<Value>(e).ok()).and_then(|v| v.as_object().cloned()).unwrap_or_default()
    }

    /// The request's fields, when it is an object type (for the catalogue's example body).
    pub fn request_fields(&self) -> Vec<Field> {
        match &self.request {
            Shape::Ref(n) => match self.defs.types.get(n) {
                Some(Named::Object { object, .. }) => object.fields.clone(),
                _ => Vec::new(),
            },
            _ => Vec::new(),
        }
    }

    /// An example body: the method's own (`#[rpc(example)]`), else every field the request has, with a placeholder
    /// of its type (`name?` for an optional one).
    pub fn example_body(&self) -> Value {
        if let Some(e) = self.example.and_then(|e| serde_json::from_str::<Value>(e).ok()) {
            return e;
        }
        if let Some((_, e)) = self.named_examples().into_iter().next() {
            return e;
        }
        let mut o = Map::new();
        for f in self.request_fields() {
            let v = match &f.shape {
                Shape::Integer | Shape::Number => json!(0),
                Shape::Bool => json!(false),
                Shape::Array(_) => json!([]),
                Shape::Map(_) | Shape::Ref(_) | Shape::Json => json!({}),
                _ => json!("…"),
            };
            o.insert(if f.optional { format!("{}?", f.name) } else { f.name.clone() }, v);
        }
        Value::Object(o)
    }
}

/// How many of these methods answer `Untyped` (the ratchet: a test holds each table's count and only lets it fall).
pub fn untyped_responses(methods: &[MethodInfo]) -> Vec<&'static str> {
    methods.iter().filter(|m| m.response == Shape::Untyped).map(|m| m.name).collect()
}

fn all_defs(methods: &[MethodInfo]) -> Result<Defs, String> {
    let mut d = Defs::default();
    for m in methods {
        d.merge(&m.defs).map_err(|e| format!("{}: {e}", m.name))?;
    }
    Ok(d)
}

// ---- OpenAPI ------------------------------------------------------------------------------------------------

fn schema(s: &Shape) -> Value {
    match s {
        Shape::String => json!({"type": "string"}),
        Shape::Integer => json!({"type": "integer"}),
        Shape::Number => json!({"type": "number"}),
        Shape::Bool => json!({"type": "boolean"}),
        Shape::Json | Shape::Untyped => json!({}),
        Shape::Empty => json!({"type": "object"}),
        Shape::Blob => json!({"type": "string", "format": "binary"}),
        Shape::Array(t) => json!({"type": "array", "items": schema(t)}),
        Shape::Map(t) if **t == Shape::Json => json!({"type": "object"}),
        Shape::Map(t) => json!({"type": "object", "additionalProperties": schema(t)}),
        Shape::Nullable(t) => json!({"anyOf": [schema(t), {"type": "null"}]}),
        Shape::Ref(n) => json!({"$ref": format!("#/components/schemas/{n}")}),
    }
}

/// A field's schema: its type's, with `#[schema]`'s keywords and its values
pub fn field_schema(f: &Field) -> Value {
    let mut s = schema(&f.shape);
    if !f.one_of.is_empty() {
        let target = if let Some(Value::Array(any)) = s.get_mut("anyOf") { &mut any[0] } else { &mut s };
        target["enum"] = json!(f.one_of);
    }
    if let (Some(o), Value::Object(extra)) = (s.as_object_mut(), &f.extra) {
        for (k, v) in extra {
            o.insert(k.clone(), v.clone());
        }
    }
    s
}

/// A shape's JSON Schema (its named types under #/components/schemas).
pub fn shape_schema(s: &Shape) -> Value {
    schema(s)
}

/// The named types of these methods as JSON Schema, for a project that writes its own document around them.
pub fn components(methods: &[MethodInfo]) -> Result<Map<String, Value>, String> {
    Ok(all_defs(methods)?.types.iter().map(|(k, v)| (k.clone(), named_schema(v))).collect())
}

fn named_schema(n: &Named) -> Value {
    match n {
        Named::Object { doc, object } => {
            let mut props = Map::new();
            let mut required = Vec::new();
            for f in &object.fields {
                let mut s = field_schema(f);
                if !f.doc.is_empty() {
                    s["description"] = json!(f.doc);
                }
                props.insert(f.name.clone(), s);
                if !f.optional {
                    required.push(json!(f.name));
                }
            }
            let mut v = json!({"type": "object", "properties": props});
            if !required.is_empty() {
                v["required"] = Value::Array(required);
            }
            if let Some(rest) = &object.rest {
                v["additionalProperties"] = if *rest == Shape::Json { json!(true) } else { schema(rest) };
            }
            if !doc.is_empty() {
                v["description"] = json!(doc);
            }
            v
        }
        Named::Union { doc, values } => {
            let mut v = json!({"type": "string", "enum": values});
            if !doc.is_empty() {
                v["description"] = json!(doc);
            }
            v
        }
        Named::Pending => json!({}),
    }
}

/// The OpenAPI document of a table: its head as given - `info` alone (title, version, description: the project's
/// own prose), or `{"info": ..., "tags": [...], "paths": {...}, "components": {"schemas": {...}}}` (a description
/// per tag; paths that are not methods of the table, such as an OpenAI-compatible API that keeps OpenAI's shape,
/// with the schemas they use) - `servers` as given, a path per method (`/api/<Method>`, POST), the named types as
/// components. A method and a head path of one path is a mistake.
pub fn openapi(head: Value, server: &str, methods: &[MethodInfo]) -> Result<String, String> {
    let (info, tags, extra_paths, extra_schemas) = match head.get("info") {
        Some(i) => (
            i.clone(),
            head.get("tags").cloned(),
            head.get("paths").and_then(|p| p.as_object()).cloned().unwrap_or_default(),
            head.pointer("/components/schemas").and_then(|p| p.as_object()).cloned().unwrap_or_default(),
        ),
        None => (head, None, Map::new(), Map::new()),
    };
    let defs = all_defs(methods)?;
    let mut paths = Map::new();
    for m in methods {
        let mut op = json!({"operationId": m.name, "summary": m.summary, "tags": m.tags});
        if m.description.len() > m.summary.len() {
            op["description"] = json!(m.description);
        }
        if !m.takes_nothing() {
            // the method's own examples only (a placeholder body is no example; the TypeScript has none either)
            let mut media = json!({"schema": schema(&m.request)});
            let all = method_examples(m);
            if let Some((_, e)) = all.iter().find(|(n, _)| n.is_empty()) {
                media["example"] = e.clone();
            }
            let named: Map<String, Value> = all.into_iter().filter(|(n, _)| !n.is_empty()).map(|(k, v)| (k, json!({"value": v}))).collect();
            if !named.is_empty() {
                media["examples"] = Value::Object(named);
            }
            op["requestBody"] = json!({"required": true, "content": {"application/json": media}});
        }
        if !m.produces.is_empty() && m.response != Shape::Blob {
            return Err(format!("{}: produces is a download's media types, and it answers JSON", m.name));
        }
        let ok = match &m.response {
            Shape::Blob => {
                let types = if m.produces.is_empty() { vec!["application/octet-stream"] } else { m.produces.clone() };
                let content: Map<String, Value> = types.iter().map(|t| (t.to_string(), json!({"schema": schema(&m.response)}))).collect();
                json!({"description": "the bytes, with their Content-Type", "content": content})
            }
            s => json!({"description": "ok", "content": {"application/json": {"schema": schema(s)}}}),
        };
        op["responses"] = json!({"200": ok, "400": {"description": "a payload that does not fit: {error}"}, "404": {"description": "not there: {error}"}});
        paths.insert(format!("/api/{}", m.name), json!({"post": op}));
    }
    for (k, v) in extra_paths {
        if paths.insert(k.clone(), v).is_some() {
            return Err(format!("{k} is both a method and a path of the document's head"));
        }
    }
    let mut schemas: Map<String, Value> = defs.types.iter().map(|(k, v)| (k.clone(), named_schema(v))).collect();
    for (k, v) in extra_schemas {
        if schemas.contains_key(&k) {
            return Err(format!("{k} is both a method's type and a schema of the document's head"));
        }
        schemas.insert(k, v);
    }
    // in a document's own order (a JSON map would sort `components` first and `openapi` last)
    let y = |v: Value| serde_yaml::to_value(v).map_err(|e| e.to_string());
    let mut doc = serde_yaml::Mapping::new();
    doc.insert("openapi".into(), "3.1.0".into());
    doc.insert("info".into(), y(info)?);
    doc.insert("servers".into(), y(json!([{"url": server}]))?);
    if let Some(t) = tags {
        doc.insert("tags".into(), y(t)?);
    }
    doc.insert("paths".into(), y(Value::Object(paths))?);
    doc.insert("components".into(), y(json!({"schemas": schemas}))?);
    serde_yaml::to_string(&doc).map_err(|e| e.to_string())
}

// ---- TypeScript -------------------------------------------------------------------------------------------------

fn ts(s: &Shape) -> String {
    match s {
        Shape::String | Shape::Blob => "string".into(),
        Shape::Integer | Shape::Number => "number".into(),
        Shape::Bool => "boolean".into(),
        Shape::Json => "unknown".into(),
        Shape::Untyped => "any".into(),
        Shape::Empty => "Record<string, never>".into(),
        Shape::Array(t) => {
            let inner = ts(t);
            if inner.contains(' ') { format!("({inner})[]") } else { format!("{inner}[]") }
        }
        Shape::Map(t) => format!("Record<string, {}>", ts(t)),
        Shape::Nullable(t) => format!("{} | null", ts(t)),
        Shape::Ref(n) => n.clone(),
    }
}

// ---- facts: what both documents say of a field and a method, from one place ----------------------------------

/// A field's `#[schema]` facts as JSDoc / JSON Schema pairs, in a fixed order: (tag, JSON value). Both generators
/// write exactly these (the OpenAPI schema's keywords, the TypeScript JSDoc tags), and `parity` checks it.
pub fn field_facts(f: &Field) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    if let Value::Object(extra) = &f.extra {
        let order = ["minimum", "maximum", "default", "minLength", "maxLength", "maxItems", "pattern", "format", "example"];
        for k in order {
            if let Some(v) = extra.get(k) {
                out.push((k.to_string(), v.clone()));
            }
        }
        for (k, v) in extra {
            if !order.contains(&k.as_str()) {
                out.push((k.clone(), v.clone()));
            }
        }
    }
    out
}

/// A method's examples: (name, body) - the single one is named "" . A method that takes nothing has none: its
/// body is ignored, so an example of it shows nothing (the document has no request body to put one in).
pub fn method_examples(m: &MethodInfo) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    if m.takes_nothing() {
        return out;
    }
    if let Some(e) = m.example.and_then(|e| serde_json::from_str::<Value>(e).ok()) {
        out.push((String::new(), e));
    }
    for (k, v) in m.named_examples() {
        out.push((k, v));
    }
    out
}

/// A field's TypeScript: its one_of values as a union of literals (kept nullable and in arrays as the shape is)
fn field_ts(f: &Field) -> String {
    if f.one_of.is_empty() {
        return ts(&f.shape);
    }
    let union = f.one_of.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>().join(" | ");
    match &f.shape {
        Shape::String => union,
        Shape::Nullable(inner) if **inner == Shape::String => format!("{union} | null"),
        Shape::Array(inner) if **inner == Shape::String => format!("({union})[]"),
        other => ts(other),
    }
}

fn ts_doc(out: &mut String, indent: &str, doc: &str) {
    ts_jsdoc(out, indent, doc, &[]);
}

/// A JSDoc block: the text (its paragraphs kept), then `@tag value` lines.
fn ts_jsdoc(out: &mut String, indent: &str, doc: &str, tags: &[(String, String)]) {
    let clean = |t: &str| t.replace("*/", "* /");
    if doc.is_empty() && tags.is_empty() {
        return;
    }
    if tags.is_empty() && !doc.contains('\n') {
        out.push_str(&format!("{indent}/** {} */\n", clean(doc)));
        return;
    }
    out.push_str(&format!("{indent}/**\n"));
    for line in doc.split('\n') {
        out.push_str(&format!("{indent} *{}{}\n", if line.is_empty() { "" } else { " " }, clean(line)));
    }
    for (t, v) in tags {
        out.push_str(&format!("{indent} * @{t} {}\n", clean(v)));
    }
    out.push_str(&format!("{indent} */\n"));
}

/// A field's JSDoc tags: its `#[schema]` facts, each `@<keyword> <JSON>`
fn field_tags(f: &Field) -> Vec<(String, String)> {
    field_facts(f).into_iter().map(|(k, v)| (k, v.to_string())).collect()
}

/// A method's JSDoc: its summary and description, `@tags`, `@example [name] <JSON>`
fn method_doc(m: &MethodInfo) -> (String, Vec<(String, String)>) {
    let mut text = m.summary.to_string();
    if m.description.len() > m.summary.len() {
        text = m.description.to_string();
    }
    let mut tags = Vec::new();
    if !m.tags.is_empty() {
        tags.push(("tags".to_string(), m.tags.join(", ")));
    }
    if !m.produces.is_empty() {
        tags.push(("produces".to_string(), m.produces.join(", ")));
    }
    for (name, body) in method_examples(m) {
        tags.push(("example".to_string(), if name.is_empty() { body.to_string() } else { format!("{name} {body}") }));
    }
    (text, tags)
}

fn ident_ok(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A method name as a TypeScript key (dotted names are quoted).
fn ts_key(name: &str) -> String {
    if ident_ok(name) { name.to_string() } else { format!("{name:?}") }
}

/// The typed client of a table, as a TypeScript module: an interface or a union per named type; `Methods`, every
/// method by name with what it takes and answers (for a project's own `rpc(name, body)` helper, which then takes
/// only real names and their request types); `NoParamMethods`; and `client(post, raw)` with one function per method.
/// `post(name, body)` is the project's own JSON call (its URL scheme and error handling); `raw` answers a
/// download as a Blob. A UI call with a wrong name or field fails `tsc`.
pub fn typescript(title: &str, methods: &[MethodInfo]) -> Result<String, String> {
    let defs = all_defs(methods)?;
    let mut out = String::new();
    out.push_str(&format!(
        "// {title}: the RPC methods' types and a typed client.\n// Generated by rpcgen from the #[rpc] handlers: do not edit; regenerate.\n/* eslint-disable */\n\n"
    ));
    for (name, n) in &defs.types {
        match n {
            Named::Object { doc, object } => {
                ts_doc(&mut out, "", doc);
                out.push_str(&format!("export interface {name} {{\n"));
                for f in &object.fields {
                    ts_jsdoc(&mut out, "  ", &f.doc, &field_tags(f));
                    out.push_str(&format!("  {}{}: {};\n", ts_key(&f.name), if f.optional { "?" } else { "" }, field_ts(f)));
                }
                if object.rest.is_some() {
                    // its other keys (a flattened map): `unknown`, as an index signature must admit every field's type
                    out.push_str("  /** other keys */\n  [key: string]: unknown;\n");
                }
                out.push_str("}\n\n");
            }
            Named::Union { doc, values } => {
                ts_doc(&mut out, "", doc);
                let v: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
                out.push_str(&format!("export type {name} = {};\n\n", if v.is_empty() { "never".into() } else { v.join(" | ") }));
            }
            Named::Pending => {}
        }
    }
    // every method by name: its request and its answer, for a helper that takes the name (`rpc("GetJob", req)`)
    out.push_str("/** Every method: what it takes and what it answers. */\nexport interface Methods {\n");
    for m in methods {
        let resp = if m.response == Shape::Blob { "Blob".to_string() } else { ts(&m.response) };
        let (text, tags) = method_doc(m);
        ts_jsdoc(&mut out, "  ", &text, &tags);
        out.push_str(&format!("  {}: {{ req: {}; resp: {resp} }};\n", ts_key(m.name), ts(&m.request)));
    }
    out.push_str("}\n\n/** The methods that take no parameters (their body may be left out). */\nexport type NoParamMethods = ");
    let none: Vec<String> = methods.iter().filter(|m| m.takes_nothing()).map(|m| format!("{:?}", m.name)).collect();
    out.push_str(&format!("{};\n\n", if none.is_empty() { "never".into() } else { none.join(" | ") }));
    out.push_str("/** Every error's body, whatever the HTTP status: branch on `code`, never on the status (docs/RPC_RULES.md). */\nexport interface RpcError {\n  /** what went wrong, in words */\n  error: string;\n  /** machine-readable: not_found, bad_payload, bad_request, conflict, busy, unavailable, unauthorized, internal, or the method's own */\n  code: string;\n  /** the same request may work later */\n  retryable: boolean;\n}\n\n");
    out.push_str("/** The project's JSON call of a method by its name. */\nexport type Post = (method: string, body: unknown) => Promise<any>;\n");
    out.push_str("/** A call that answers raw bytes (a download). */\nexport type PostRaw = (method: string, body: unknown) => Promise<Blob>;\n\n");
    out.push_str("export function client(post: Post, raw?: PostRaw) {\n  const bytes = (m: string, b: unknown) => (raw ? raw(m, b) : Promise.reject(new Error(\"no raw call given\")));\n  void bytes;\n  return {\n");
    for m in methods {
        let (text, tags) = method_doc(m);
        ts_jsdoc(&mut out, "    ", &text, &tags);
        let resp = ts(&m.response);
        let (params, body) = if m.takes_nothing() { (format!("req: {} = {{}}", ts(&m.request)), "req".to_string()) } else { (format!("req: {}", ts(&m.request)), "req".to_string()) };
        let call = if m.response == Shape::Blob { format!("Promise<Blob> => bytes({:?}, {body})", m.name) } else { format!("Promise<{resp}> => post({:?}, {body})", m.name) };
        out.push_str(&format!("    {}: ({params}): {call},\n", ts_key(m.name)));
    }
    out.push_str("  };\n}\n");
    Ok(out)
}

// ---- parity: the two documents say the same ------------------------------------------------------------------

/// What one document says, as lines (`GetJob: text: ...`, `Thing.id: doc: ...`, `Thing.id: @minimum 0`,
/// `Thing.id: optional`, `Kind: values: a | b`), sorted: two documents agree when their lines are equal.
type Facts = std::collections::BTreeSet<String>;

fn facts_of_openapi(doc: &str) -> Result<Facts, String> {
    let v: Value = serde_yaml::from_str::<Value>(doc).map_err(|e| format!("the OpenAPI document does not parse: {e}"))?;
    let mut f = Facts::new();
    for (path, ops) in v["paths"].as_object().into_iter().flatten() {
        let Some(name) = path.strip_prefix("/api/") else { continue };
        let op = &ops["post"];
        let text = op["description"].as_str().or_else(|| op["summary"].as_str()).unwrap_or("");
        f.insert(format!("{name}: text: {text}"));
        for t in op["tags"].as_array().into_iter().flatten() {
            f.insert(format!("{name}: tag: {}", t.as_str().unwrap_or("")));
        }
        for ct in op["responses"]["200"]["content"].as_object().into_iter().flatten().map(|(k, _)| k) {
            if ct != "application/json" && ct != "application/octet-stream" {
                f.insert(format!("{name}: produces: {ct}"));
            }
        }
        let media = &op["requestBody"]["content"]["application/json"];
        if let Some(e) = media.get("example") {
            f.insert(format!("{name}: example: {e}"));
        }
        for (k, e) in media["examples"].as_object().into_iter().flatten() {
            f.insert(format!("{name}: example {k}: {}", e["value"]));
        }
    }
    let known = ["minimum", "maximum", "default", "minLength", "maxLength", "maxItems", "pattern", "format", "example"];
    for (tname, t) in v["components"]["schemas"].as_object().into_iter().flatten() {
        if let Some(d) = t["description"].as_str() {
            f.insert(format!("{tname}: doc: {d}"));
        }
        if let Some(e) = t["enum"].as_array() {
            f.insert(format!("{tname}: values: {}", e.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join(" | ")));
        }
        if t.get("properties").is_some() && t.get("additionalProperties").is_some() {
            f.insert(format!("{tname}: other keys"));
        }
        let req: Vec<&str> = t["required"].as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect();
        for (fname, p) in t["properties"].as_object().into_iter().flatten() {
            let at = format!("{tname}.{fname}");
            if let Some(d) = p["description"].as_str() {
                f.insert(format!("{at}: doc: {d}"));
            }
            if !req.contains(&fname.as_str()) {
                f.insert(format!("{at}: optional"));
            }
            for (k, val) in p.as_object().into_iter().flatten() {
                // the shape writes `items` as a bare type or $ref and never inline `properties`; anything more is the
                // field's own #[schema(json)], which the TypeScript carries as a tag
                let own = match k.as_str() {
                    "items" => val.as_object().is_some_and(|o| o.keys().any(|x| x != "type" && x != "$ref")),
                    "properties" => true,
                    _ => false,
                };
                if own || known.contains(&k.as_str()) || !["type", "description", "items", "anyOf", "$ref", "enum", "additionalProperties", "properties", "required"].contains(&k.as_str()) {
                    f.insert(format!("{at}: @{k} {val}"));
                }
            }
        }
    }
    Ok(f)
}

/// The JSDoc block just before line `i`: its text (paragraphs kept) and its `@tag value` lines.
fn jsdoc_before(lines: &[&str], i: usize) -> (String, Vec<String>) {
    let mut j = i;
    if j == 0 || !lines[j - 1].trim_end().ends_with("*/") {
        return (String::new(), Vec::new());
    }
    let end = j - 1;
    while j > 0 && !lines[j - 1].trim_start().starts_with("/**") {
        j -= 1;
    }
    let start = j - 1;
    let block: Vec<&str> = lines[start..=end].to_vec();
    if block.len() == 1 {
        let t = block[0].trim().trim_start_matches("/**").trim_end_matches("*/").trim().to_string();
        return (t, Vec::new());
    }
    let (mut text, mut tags) = (Vec::new(), Vec::new());
    for l in &block[1..block.len() - 1] {
        let l = l.trim_start().strip_prefix('*').unwrap_or(l).strip_prefix(' ').unwrap_or_else(|| l.trim_start().strip_prefix('*').unwrap_or(""));
        if let Some(t) = l.strip_prefix('@') {
            tags.push(t.to_string());
        } else {
            text.push(l.to_string());
        }
    }
    (text.join("\n"), tags)
}

fn facts_of_typescript(ts: &str) -> Facts {
    let lines: Vec<&str> = ts.lines().collect();
    let mut f = Facts::new();
    let mut i = 0;
    let mut in_client = false;
    while i < lines.len() {
        let l = lines[i];
        if let Some(rest) = l.strip_prefix("export interface ") {
            let tname = rest.trim_end_matches(" {").trim();
            if tname == "Methods" || tname == "RpcError" {
                i += 1;
                continue;
            }
            let (d, _) = jsdoc_before(&lines, i);
            if !d.is_empty() {
                f.insert(format!("{tname}: doc: {d}"));
            }
            i += 1;
            while i < lines.len() && lines[i] != "}" {
                let p = lines[i].trim();
                if p.starts_with("[key: string]") {
                    f.insert(format!("{tname}: other keys"));
                } else if !p.starts_with("/**") && !p.starts_with('*') && p.contains(':') {
                    let key = p.split(':').next().unwrap_or("");
                    let (fname, opt) = match key.strip_suffix('?') {
                        Some(k) => (k.trim_matches('"').to_string(), true),
                        None => (key.trim_matches('"').to_string(), false),
                    };
                    let at = format!("{tname}.{fname}");
                    let (d, tags) = jsdoc_before(&lines, i);
                    if !d.is_empty() {
                        f.insert(format!("{at}: doc: {d}"));
                    }
                    if opt {
                        f.insert(format!("{at}: optional"));
                    }
                    for t in tags {
                        f.insert(format!("{at}: @{t}"));
                    }
                }
                i += 1;
            }
        } else if let Some(rest) = l.strip_prefix("export type ").filter(|r| r.contains(" = \"") || r.contains(" = never")) {
            let (tname, vals) = rest.split_once(" = ").unwrap();
            if tname == "NoParamMethods" {
                i += 1;
                continue;
            }
            let (d, _) = jsdoc_before(&lines, i);
            if !d.is_empty() {
                f.insert(format!("{tname}: doc: {d}"));
            }
            let vals: Vec<String> = vals.trim_end_matches(';').split(" | ").map(|v| v.trim_matches('"').to_string()).collect();
            f.insert(format!("{tname}: values: {}", vals.join(" | ")));
        } else if l.starts_with("export function client(") {
            in_client = true;
        } else if in_client && l.starts_with("    ") && !l.trim_start().starts_with(['/', '*']) && l.contains(": (") {
            let name = l.trim().split(": (").next().unwrap_or("").trim_matches('"').to_string();
            let (d, tags) = jsdoc_before(&lines, i);
            f.insert(format!("{name}: text: {d}"));
            for t in tags {
                if let Some(v) = t.strip_prefix("tags ") {
                    for tag in v.split(", ") {
                        f.insert(format!("{name}: tag: {tag}"));
                    }
                } else if let Some(v) = t.strip_prefix("produces ") {
                    for ct in v.split(", ") {
                        f.insert(format!("{name}: produces: {ct}"));
                    }
                } else if let Some(v) = t.strip_prefix("example ") {
                    match v.split_once(' ').filter(|(n, _)| !n.starts_with(['{', '['])) {
                        Some((n, body)) => f.insert(format!("{name}: example {n}: {body}")),
                        None => f.insert(format!("{name}: example: {v}")),
                    };
                }
            }
        }
        i += 1;
    }
    f
}

/// Whether the OpenAPI document and the TypeScript made from these methods say the same: every description,
/// tag, example, optional field and `#[schema]` fact in one is in the other. Each project calls it in its
/// staleness test, so a generator change that drops something from one of them fails `cargo test`.
pub fn parity(methods: &[MethodInfo]) -> Result<(), String> {
    let doc = openapi(json!({"title": "parity", "version": "0"}), "/", methods)?;
    let ts = typescript("parity", methods)?;
    let a = facts_of_openapi(&doc)?;
    let b = facts_of_typescript(&ts);
    let only_doc: Vec<&String> = a.difference(&b).collect();
    let only_ts: Vec<&String> = b.difference(&a).collect();
    if only_doc.is_empty() && only_ts.is_empty() {
        return Ok(());
    }
    let mut msg = String::from("the OpenAPI document and the TypeScript differ:\n");
    for x in only_doc.iter().take(40) {
        msg.push_str(&format!("  only in the document: {x}\n"));
    }
    for x in only_ts.iter().take(40) {
        msg.push_str(&format!("  only in the TypeScript: {x}\n"));
    }
    Err(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize, Serialize)]
    struct Card {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
        stages: Vec<Stage>,
    }

    #[derive(serde::Deserialize, Serialize)]
    struct Stage {
        name: String,
    }

    /// An answer conforms to its type when it decodes and nothing in it falls outside the type; a mismatch names
    /// its place.
    #[test]
    fn conforms_names_where_an_answer_differs() {
        assert_eq!(conforms::<Card>(&json!({"id": "a", "note": null, "stages": [{"name": "lyrics"}]})), Ok(()));
        let e = conforms::<Card>(&json!({"id": "a", "stages": [{"name": "lyrics"}, {"name": 3}]})).unwrap_err();
        assert_eq!(e.at, "stages[1].name");
        let e = conforms::<Card>(&json!({"stages": []})).unwrap_err();
        assert!(e.what.contains("missing field `id`"), "{e}");
        let e = conforms::<Card>(&json!({"id": "a", "stages": [{"name": "x", "secs": 2}], "extra": 1})).unwrap_err();
        assert_eq!(e.at, "extra");
        assert!(e.what.contains("nor stages[0].secs"), "{e}");
    }

    /// The checker sees a fact one document drops: a limit, a field's words, an example, a tag.
    #[test]
    fn parity_finds_what_one_side_dropped() {
        let mut defs = Defs::default();
        let mut o = Object::default();
        o.field_with::<f64>(&mut defs, "strength", "how far from the picture", true, r#"{"minimum": 0.05, "default": 0.6}"#, &[]);
        o.field::<String>(&mut defs, "id", "the job", false);
        defs.named("Make", "what Make takes", o);
        let m = MethodInfo {
            name: "CreateThing", summary: "Make a thing", description: "Make a thing", tags: vec!["things"],
            request: Shape::Ref("Make".into()), response: Shape::Json, defs, example: Some(r#"{"id": "1"}"#), examples: None, produces: Vec::new(),
        };
        let doc = openapi(json!({"title": "t", "version": "0"}), "/", std::slice::from_ref(&m)).unwrap();
        let ts = typescript("t", std::slice::from_ref(&m)).unwrap();
        let a = facts_of_openapi(&doc).unwrap();
        assert_eq!(a, facts_of_typescript(&ts), "the two agree to begin with");
        for (drop, what) in [("   * @minimum 0.05\n", "@minimum"), ("   * how far from the picture\n", "doc"),
                             ("     * @example {\"id\":\"1\"}\n", "example"), ("     * @tags things\n", "tag")] {
            assert!(ts.contains(drop), "{drop:?} is in the TypeScript");
            let cut = facts_of_typescript(&ts.replace(drop, ""));
            assert!(a.difference(&cut).any(|x| x.contains(what)), "dropping {what} is seen");
        }
        let cut = facts_of_openapi(&doc.replace("minimum: 0.05", "")).unwrap();
        assert!(facts_of_typescript(&ts).difference(&cut).any(|x| x.contains("@minimum")), "and from the document too");
    }

    #[test]
    fn a_name_has_one_style_and_a_table_one() {
        assert_eq!((style_of("GetJob"), style_of("system.ping"), style_of("video.jobs.get")), (1, 2, 2));
        assert_eq!((style_of("getJob"), style_of("system."), style_of(".x"), style_of("a..b"), style_of("Get-Job")), (0, 0, 0, 0, 0));
        assert_eq!((style_of("system.configSet"), style_of("llm.describePromptGuideMD"), style_of("system.Ping")), (2, 2, 0));
        assert!(one_style(&["GetJob", "ListJobs"]) && one_style(&["system.ping", "video.status"]));
        assert!(!one_style(&["GetJob", "system.ping"]));
    }

    #[test]
    fn an_integer_is_a_whole_number_written_either_way() {
        #[derive(serde::Deserialize)]
        struct R {
            #[serde(deserialize_with = "de::int")]
            n: i64,
            #[serde(default, deserialize_with = "de::opt_int")]
            m: Option<i64>,
        }
        let r: R = serde_json::from_value(json!({"n": 512.0, "m": 3})).unwrap();
        assert_eq!((r.n, r.m), (512, Some(3)));
        assert!(serde_json::from_value::<R>(json!({"n": 1.5})).is_err());
        assert!(serde_json::from_value::<R>(json!({"n": "512"})).is_err());
        let r: R = serde_json::from_value(json!({"n": 1, "m": null})).unwrap();
        assert_eq!(r.m, None);
    }

    #[test]
    fn names_are_checked_distinct() {
        assert!(distinct(&["GetJob", "ListJobs", "GetJobLog"]));
        assert!(!distinct(&["GetJob", "ListJobs", "GetJob"]));
    }

    #[cfg(feature = "axum")]
    #[test]
    fn a_payload_that_does_not_fit_is_a_json_400() {
        #[derive(serde::Deserialize, Debug)]
        struct Req {
            #[allow(dead_code)]
            id: String,
        }
        let e = decode::<Req>(json!({"nope": 1})).unwrap_err();
        assert_eq!(e.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(decode::<NoParams>(Value::Null).is_ok());
        assert!(decode::<NoParams>(json!({"anything": [1]})).is_ok());
    }
}
