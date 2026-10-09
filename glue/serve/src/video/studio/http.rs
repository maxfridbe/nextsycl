//! H3's HTTP calls (its `h3-http` API: results with an error type, `respond_bytes` without extra headers) over the
//! glue's `http` - so the studio's code reads as H3's did and later H3 changes merge in as they are.

use std::io::Write;

use serde_json::Value;

pub use crate::http::{content_type, Request, Target};

/// An error: its message
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error(e)
    }
}

impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error(e.into())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn respond_bytes(w: impl Write, status: u16, content_type: &str, body: &[u8]) -> Result<()> {
    crate::http::respond_bytes(w, status, content_type, body, "Cache-Control: no-cache\r\n");
    Ok(())
}

/// JSON, pretty (as H3's front end and tools read it)
pub fn respond(w: impl Write, status: u16, body: &Value) -> Result<()> {
    let text = serde_json::to_vec_pretty(body).map_err(|e| Error(e.to_string()))?;
    respond_bytes(w, status, "application/json", &text)
}

pub fn forward(client: impl Write, upstream: &Target, req: &Request) -> Result<()> {
    crate::http::forward(client, upstream, req).map_err(Error)
}

pub fn call(t: &Target, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
    crate::http::call(t, method, path, body).map_err(Error)
}

pub fn read_request(stream: impl std::io::Read) -> Result<Request> {
    crate::http::read_request(stream).map_err(Error)
}

pub fn split_url(url: &str) -> Result<(String, String)> {
    crate::http::split_url(url).map_err(Error)
}
