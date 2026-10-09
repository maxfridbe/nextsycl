//! Just enough HTTP/1.1, one request per connection, over TCP or a Unix socket - the server (`serve` answers the
//! OpenAI API on TCP and its control routes on a Unix socket too) and the host command line that calls it (the
//! sycl-h3 transport, as H3's `h3-http`). JSON bodies; standard library and serde_json only.

use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::Value;

/// Where a server is: a Unix socket (the control sockets), or `host:port`
#[derive(Clone, Debug)]
pub enum Target {
    Unix(PathBuf),
    Tcp(String),
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Target::Unix(p) => write!(f, "{}", p.display()),
            Target::Tcp(a) => f.write_str(a),
        }
    }
}

/// A connection of either kind.
pub enum Conn {
    Tcp(TcpStream),
    Unix(UnixStream),
}

impl Conn {
    pub fn connect(t: &Target) -> std::io::Result<Conn> {
        Ok(match t {
            Target::Unix(p) => Conn::Unix(UnixStream::connect(p)?),
            Target::Tcp(a) => Conn::Tcp(TcpStream::connect(a)?),
        })
    }
    pub fn set_read_timeout(&self, d: Option<Duration>) -> std::io::Result<()> {
        match self {
            Conn::Tcp(s) => s.set_read_timeout(d),
            Conn::Unix(s) => s.set_read_timeout(d),
        }
    }
    pub fn try_clone(&self) -> std::io::Result<Conn> {
        Ok(match self {
            Conn::Tcp(s) => Conn::Tcp(s.try_clone()?),
            Conn::Unix(s) => Conn::Unix(s.try_clone()?),
        })
    }
    /// "socket" for the control socket, "tcp" for the API port
    pub fn kind(&self) -> &'static str {
        match self {
            Conn::Tcp(_) => "tcp",
            Conn::Unix(_) => "socket",
        }
    }
}

impl Read for Conn {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.read(b),
            Conn::Unix(s) => s.read(b),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.write(b),
            Conn::Unix(s) => s.write(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Tcp(s) => s.flush(),
            Conn::Unix(s) => s.flush(),
        }
    }
}

pub struct Request {
    pub method: String,
    /// the path without the query
    pub path: String,
    /// the path with the query, as received
    pub target: String,
    /// the request line and headers exactly as received (for passing the request on: `forward`)
    pub head: Vec<u8>,
    /// the Origin header (a browser's request)
    pub origin: Option<String>,
    /// the Host header (for absolute links in an answer)
    pub host: Option<String>,
    pub body: Vec<u8>,
}

/// Reads one request from a connection.
pub fn read_request(stream: impl Read) -> Result<Request, String> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request")?.to_string();
    let target = parts.next().ok_or("no path in the request")?.to_string();
    let path = target.split('?').next().unwrap_or("/").to_string();
    let mut head = line.clone().into_bytes();
    let mut length = 0usize;
    let mut origin = None;
    let mut host = None;
    loop {
        let mut h = String::new();
        let n = r.read_line(&mut h).map_err(|e| e.to_string())?;
        head.extend_from_slice(h.as_bytes());
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                length = v.trim().parse().map_err(|_| "bad Content-Length")?;
            } else if k.trim().eq_ignore_ascii_case("origin") {
                origin = Some(v.trim().to_string());
            } else if k.trim().eq_ignore_ascii_case("host") {
                host = Some(v.trim().to_string());
            }
        }
    }
    if length > 64 << 20 {
        return Err("request body over 64 MiB".into());
    }
    let mut body = vec![0u8; length];
    r.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, target, head, origin, host, body })
}

impl Request {
    /// The body as JSON (null when empty)
    pub fn json(&self) -> Result<Value, String> {
        if self.body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&self.body).map_err(|e| format!("the body is not JSON: {e}"))
    }
}

/// Passes a request on to `upstream` unchanged and copies the answer back as it arrives (long polls and file downloads
/// work)
pub fn forward(mut client: impl Write, upstream: &Target, req: &Request) -> Result<(), String> {
    let mut up = Conn::connect(upstream).map_err(|e| format!("{upstream} does not answer ({e})"))?;
    up.write_all(&req.head).and_then(|_| up.write_all(&req.body)).and_then(|_| up.flush()).map_err(|e| e.to_string())?;
    std::io::copy(&mut up, &mut client).map_err(|e| e.to_string())?;
    Ok(())
}

/// `http://host:port/path` -> (`host:port`, `/path`); plain http only
pub fn split_url(url: &str) -> Result<(String, String), String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("{url}: only http:// URLs"))?;
    let (host, path) = rest.split_once('/').map_or((rest, "/".to_string()), |(h, p)| (h, format!("/{p}")));
    let host = if host.contains(':') { host.to_string() } else { format!("{host}:80") };
    Ok((host, path))
}

/// A front-end file's content type, by its extension
pub fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "webp" => "image/webp",
        "jpg" | "jpeg" => "image/jpeg",
        "mp4" => "video/mp4",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    }
}

/// An answer of raw bytes (a picture, a page)
pub fn respond_bytes(mut w: impl Write, status: u16, content_type: &str, body: &[u8], headers: &str) {
    let _ = write!(w, "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{headers}Connection: close\r\n\r\n",
                   reason(status), body.len());
    let _ = w.write_all(body);
    let _ = w.flush();
}

/// Standard base64 (with padding)
pub fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

pub fn respond(w: impl Write, status: u16, body: &Value) {
    respond_with(w, status, body, "")
}

/// `respond` with more header lines (each ending in \r\n), e.g. CORS
pub fn respond_with(mut w: impl Write, status: u16, body: &Value, headers: &str) {
    let b = body.to_string();
    let _ = write!(w, "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{headers}Connection: close\r\n\r\n{b}",
                   reason(status), b.len());
    let _ = w.flush();
}

/// Sends one request; the connection with the answer's status line and headers read, for streaming the body.
pub fn send(t: &Target, method: &str, path: &str, body: Option<&Value>) -> Result<(u16, BufReader<Conn>), String> {
    let mut s = Conn::connect(t).map_err(|e| format!("nothing answers at {t} ({e})"))?;
    let text = body.map(|b| serde_json::to_vec(b).unwrap()).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: nextsycl\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", text.len())
        .and_then(|_| s.write_all(&text))
        .map_err(|e| e.to_string())?;
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| e.to_string())?;
    let status: u16 = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or("a malformed status line")?;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).map_err(|e| e.to_string())? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    Ok((status, r))
}

/// One request; the JSON answer, or an error carrying the answer's message.
pub fn call(t: &Target, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
    let (status, mut r) = send(t, method, path, body)?;
    r.get_ref().set_read_timeout(Some(Duration::from_secs(60))).map_err(|e| e.to_string())?;
    let mut all = String::new();
    r.read_to_string(&mut all).map_err(|e| e.to_string())?;
    let v: Value = if all.trim().is_empty() { Value::Null } else { serde_json::from_str(&all).map_err(|e| format!("the answer is not JSON: {e}"))? };
    if status == 200 {
        Ok(v)
    } else {
        Err(v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("request failed").to_string())
    }
}
