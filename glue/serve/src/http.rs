//! Just enough HTTP/1.1, one request per connection, over TCP or a Unix socket - the server (`serve` answers the
//! OpenAI API on TCP and its control routes on a Unix socket too) and the host command line that calls it (the
//! sycl-h3 transport, as H3's `h3-http`). JSON bodies; standard library and serde_json only.

use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
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

/// A connection of any kind.
pub enum Conn {
    Tcp(TcpStream),
    Unix(UnixStream),
    /// An HTTPS connection (crate::tls): the plaintext side of its TLS session, and the TCP connection's own peer and
    /// local addresses as the kernel gave them when it was accepted
    Tls(TlsConn),
}

/// The server's side of an HTTPS connection: the decrypted bytes over one end of a socket pair (crate::tls's pump
/// threads hold the other end and the TLS session), and who the TCP peer is. Nothing the client sends can change `peer`
/// or `local`: they are the accepted socket's addresses, fixed before its first byte was read. The socket pair has no
/// name and no address: nothing else can connect to it.
pub struct TlsConn {
    pub(crate) s: UnixStream,
    pub(crate) peer: SocketAddr,
    pub(crate) local: SocketAddr,
}

impl TlsConn {
    /// The TCP peer's address (the client's)
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer
    }
    /// The address the TCP connection came in on (this machine's)
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }
}

/// A TCP peer is on this machine: a loopback address, or this machine's own address the connection came in on
fn same_box_addrs(peer: IpAddr, local: IpAddr) -> bool {
    peer.is_loopback() || peer == local
}

impl Conn {
    /// The peer is on this machine: loopback, or this machine's own address the connection came in on. Over HTTPS the
    /// TCP connection's addresses decide the same way, never anything sent inside the session (no header is read)
    pub fn same_box(&self) -> bool {
        match self {
            Conn::Unix(_) => true,
            Conn::Tcp(c) => match (c.peer_addr(), c.local_addr()) {
                (Ok(p), Ok(l)) => same_box_addrs(p.ip(), l.ip()),
                _ => false,
            },
            Conn::Tls(t) => same_box_addrs(t.peer.ip(), t.local.ip()),
        }
    }

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
            Conn::Tls(t) => t.s.set_read_timeout(d),
        }
    }
    pub fn try_clone(&self) -> std::io::Result<Conn> {
        Ok(match self {
            Conn::Tcp(s) => Conn::Tcp(s.try_clone()?),
            Conn::Unix(s) => Conn::Unix(s.try_clone()?),
            Conn::Tls(t) => Conn::Tls(TlsConn { s: t.s.try_clone()?, peer: t.peer, local: t.local }),
        })
    }
    /// An HTTPS connection
    pub fn is_tls(&self) -> bool {
        matches!(self, Conn::Tls(_))
    }
    /// "socket" for the control socket, "tcp" for the API port (plain or HTTPS)
    pub fn kind(&self) -> &'static str {
        match self {
            Conn::Tcp(_) | Conn::Tls(_) => "tcp",
            Conn::Unix(_) => "socket",
        }
    }
}

impl Read for Conn {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        (&*self).read(b)
    }
}

impl Write for Conn {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        (&*self).write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        (&*self).flush()
    }
}

/// As `&TcpStream`: a shared connection read and written (the studio's handlers take `&Conn`)
impl Read for &Conn {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        match *self {
            Conn::Tcp(s) => (&*s).read(b),
            Conn::Unix(s) => (&*s).read(b),
            Conn::Tls(t) => (&t.s).read(b),
        }
    }
}

impl Write for &Conn {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match *self {
            Conn::Tcp(s) => (&*s).write(b),
            Conn::Unix(s) => (&*s).write(b),
            Conn::Tls(t) => (&t.s).write(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match *self {
            Conn::Tcp(s) => (&*s).flush(),
            Conn::Unix(s) => (&*s).flush(),
            Conn::Tls(t) => (&t.s).flush(),
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
    /// "http", or "https" when the server marks a request that came in over TLS (absolute links in an answer)
    pub scheme: &'static str,
    pub body: Vec<u8>,
}

/// Reads one request from a connection.
pub fn read_request(stream: impl Read) -> Result<Request, String> {
    read_request_at(stream).map_err(|e| e.1)
}

/// read_request, its error with the path the request named ("" when it did not get that far): an API answers in the
/// shape of the route asked for
pub fn read_request_at(stream: impl Read) -> Result<Request, (String, String)> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| (String::new(), e.to_string()))?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or((String::new(), "empty request".to_string()))?.to_string();
    let target = parts.next().ok_or((String::new(), "no path in the request".to_string()))?.to_string();
    let path = target.split('?').next().unwrap_or("/").to_string();
    let fail = |e: String| (path.clone(), e);
    let mut head = line.clone().into_bytes();
    let mut length = 0usize;
    let mut origin = None;
    let mut host = None;
    loop {
        let mut h = String::new();
        let n = r.read_line(&mut h).map_err(|e| fail(e.to_string()))?;
        head.extend_from_slice(h.as_bytes());
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                length = v.trim().parse().map_err(|_| fail("bad Content-Length".into()))?;
            } else if k.trim().eq_ignore_ascii_case("origin") {
                origin = Some(v.trim().to_string());
            } else if k.trim().eq_ignore_ascii_case("host") {
                host = Some(v.trim().to_string());
            }
        }
    }
    if length > 256 << 20 {
        return Err(fail("request body over 256 MiB".into()));
    }
    let mut body = vec![0u8; length];
    r.read_exact(&mut body).map_err(|e| fail(e.to_string()))?;
    Ok(Request { method, path, target, head, origin, host, scheme: "http", body })
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

/// `forward` to another path: the request line names `target` (a path and its query) instead of the one received, the
/// client's Connection header gives way to `Connection: close`, every other header and the body pass unchanged, and
/// the answer is copied back as it arrives (streamed answers included) until upstream closes
pub fn forward_to(mut client: impl Write, upstream: &Target, req: &Request, target: &str) -> Result<(), String> {
    let head = String::from_utf8_lossy(&req.head);
    let mut lines = head.split("\r\n").filter(|l| !l.is_empty());
    let first = lines.next().unwrap_or("");
    let version = first.rsplit(' ').next().filter(|v| v.starts_with("HTTP/")).unwrap_or("HTTP/1.1");
    let mut out = format!("{} {target} {version}\r\n", req.method);
    for l in lines {
        if l.split(':').next().unwrap_or("").trim().eq_ignore_ascii_case("connection") {
            continue;
        }
        out.push_str(l);
        out.push_str("\r\n");
    }
    out.push_str("Connection: close\r\n\r\n");
    let mut up = Conn::connect(upstream).map_err(|e| format!("{upstream} does not answer ({e})"))?;
    up.write_all(out.as_bytes()).and_then(|_| up.write_all(&req.body)).and_then(|_| up.flush()).map_err(|e| e.to_string())?;
    std::io::copy(&mut up, &mut client).map_err(|e| e.to_string())?;
    Ok(())
}

/// `forward` with another body: the request's head with its Content-Length set to `body`'s and the connection to
/// close (the answer is then copied back as it arrives - streamed answers included - until upstream closes)
pub fn forward_body(mut client: impl Write, upstream: &Target, req: &Request, body: &[u8]) -> Result<(), String> {
    let head = String::from_utf8_lossy(&req.head);
    let mut lines = head.split("\r\n").filter(|l| !l.is_empty());
    let mut out = format!("{}\r\n", lines.next().unwrap_or(""));
    for l in lines {
        let k = l.split(':').next().unwrap_or("").trim();
        if ["content-length", "connection", "transfer-encoding"].iter().any(|h| k.eq_ignore_ascii_case(h)) {
            continue;
        }
        out.push_str(l);
        out.push_str("\r\n");
    }
    out.push_str(&format!("Content-Length: {}\r\nConnection: close\r\n\r\n", body.len()));
    let mut up = Conn::connect(upstream).map_err(|e| format!("{upstream} does not answer ({e})"))?;
    up.write_all(out.as_bytes()).and_then(|_| up.write_all(body)).and_then(|_| up.flush()).map_err(|e| e.to_string())?;
    std::io::copy(&mut up, &mut client).map_err(|e| e.to_string())?;
    Ok(())
}

/// Whether the request asks to become a WebSocket (`Upgrade: websocket`)
pub fn is_upgrade(head: &[u8]) -> bool {
    header(head, "upgrade").is_some_and(|u| u.eq_ignore_ascii_case("websocket"))
}

/// An upgrade (a WebSocket) passed on to `upstream` as `target` (a path and its query; the head's other lines unchanged,
/// `Connection: Upgrade` kept), then both directions copied until either side closes - the front door between a
/// client and a model server's /v1/realtime
pub fn splice(client: &Conn, upstream: &Target, req: &Request, target: &str) -> Result<(), String> {
    let head = String::from_utf8_lossy(&req.head);
    let mut lines = head.split("\r\n").filter(|l| !l.is_empty());
    let first = lines.next().unwrap_or("");
    let version = first.rsplit(' ').next().filter(|v| v.starts_with("HTTP/")).unwrap_or("HTTP/1.1");
    let mut out = format!("{} {target} {version}\r\n", req.method);
    for l in lines {
        out.push_str(l);
        out.push_str("\r\n");
    }
    out.push_str("\r\n");
    let mut up = Conn::connect(upstream).map_err(|e| format!("{upstream} does not answer ({e})"))?;
    client.nodelay();
    up.nodelay();
    up.write_all(out.as_bytes()).and_then(|_| up.flush()).map_err(|e| e.to_string())?;
    let (mut c_in, mut c_out) = (client.try_clone().map_err(|e| e.to_string())?, client.try_clone().map_err(|e| e.to_string())?);
    let (mut u_in, mut u_out) = (up.try_clone().map_err(|e| e.to_string())?, up);
    let to_up = std::thread::spawn(move || {
        let _ = std::io::copy(&mut c_in, &mut u_out);
        u_out.shutdown();
    });
    let _ = std::io::copy(&mut u_in, &mut c_out);
    c_out.shutdown();
    let _ = to_up.join();
    Ok(())
}

impl Conn {
    /// Small writes go out at once (TCP_NODELAY): a WebSocket's frames are small and each waits for the other side's,
    /// so Nagle's delay is paid on every one (nextcuda 24d27fa: a lockstep frame's turnaround 60.5 -> 17.5 ms). A Unix
    /// socket has no such delay; an HTTPS connection's TCP side is set when it is accepted (crate::tls)
    pub fn nodelay(&self) {
        if let Conn::Tcp(s) = self {
            let _ = s.set_nodelay(true);
        }
    }

    /// Both directions closed (a splice's end: the other copy's read returns)
    pub fn shutdown(&self) {
        let _ = match self {
            Conn::Tcp(s) => s.shutdown(std::net::Shutdown::Both),
            Conn::Unix(s) => s.shutdown(std::net::Shutdown::Both),
            // the pump sees the end: close_notify, then the TCP connection closed (crate::tls)
            Conn::Tls(t) => t.s.shutdown(std::net::Shutdown::Both),
        };
    }
}

/// `call_for` that keeps the answer whatever its status: (status, body)
pub fn call_status(t: &Target, method: &str, path: &str, body: Option<&Value>, secs: u64) -> Result<(u16, Value), String> {
    let (status, mut r) = send(t, method, path, body)?;
    r.get_ref().set_read_timeout(Some(Duration::from_secs(secs))).map_err(|e| e.to_string())?;
    let mut all = String::new();
    r.read_to_string(&mut all).map_err(|e| e.to_string())?;
    let v: Value = if all.trim().is_empty() { Value::Null } else { serde_json::from_str(&all).map_err(|e| format!("the answer is not JSON: {e}"))? };
    Ok((status, v))
}

/// `call` with a read timeout of `secs`
pub fn call_for(t: &Target, method: &str, path: &str, body: Option<&Value>, secs: u64) -> Result<Value, String> {
    let (status, v) = call_status(t, method, path, body, secs)?;
    if status == 200 {
        Ok(v)
    } else {
        Err(v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("request failed").to_string())
    }
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
        "wav" => "audio/wav",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

/// The status line's reason phrase for a status (an unlisted one gets its class's name, never a wrong specific one)
pub fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        101 => "Switching Protocols",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        206 => "Partial Content",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        411 => "Length Required",
        412 => "Precondition Failed",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        416 => "Range Not Satisfiable",
        422 => "Unprocessable Content",
        423 => "Locked",
        425 => "Too Early",
        426 => "Upgrade Required",
        428 => "Precondition Required",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        499 => "Client Closed Request",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        507 => "Insufficient Storage",
        _ => match status / 100 {
            1 => "Informational",
            2 => "Success",
            3 => "Redirection",
            4 => "Client Error",
            _ => "Server Error",
        },
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

/// Standard base64 (padding and whitespace optional; a data: URL's prefix is skipped)
pub fn unbase64(s: &str) -> Result<Vec<u8>, String> {
    let s = s.split_once(";base64,").map_or(s, |(_, b)| b);
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c).ok_or_else(|| format!("not base64 (a {:?})", c as char))?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// A request header's value (the head as received)
pub fn header(head: &[u8], name: &str) -> Option<String> {
    String::from_utf8_lossy(head).lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_string())
    })
}

/// One part of a multipart/form-data body: its field name, file name (a file's), bytes
pub struct Part {
    pub name: String,
    pub filename: Option<String>,
    pub data: Vec<u8>,
}

/// The parts of a multipart/form-data body with this boundary
pub fn multipart(body: &[u8], boundary: &str) -> Result<Vec<Part>, String> {
    let delim = format!("--{boundary}").into_bytes();
    let find = |hay: &[u8], needle: &[u8], from: usize| -> Option<usize> {
        hay.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|p| p + from)
    };
    let mut parts = Vec::new();
    let mut at = find(body, &delim, 0).ok_or("a multipart body without its boundary")? + delim.len();
    loop {
        if body.get(at..at + 2) == Some(b"--") {
            break;
        }
        // skip the line break after the boundary
        at += if body.get(at..at + 2) == Some(b"\r\n") { 2 } else { 0 };
        let head_end = find(body, b"\r\n\r\n", at).ok_or("a part without its headers' end")?;
        let head = String::from_utf8_lossy(&body[at..head_end]).to_string();
        let next = find(body, &delim, head_end + 4).ok_or("a part without the closing boundary")?;
        let mut data = &body[head_end + 4..next];
        if data.ends_with(b"\r\n") {
            data = &data[..data.len() - 2];
        }
        let disp = head.lines().find(|l| l.to_ascii_lowercase().starts_with("content-disposition")).unwrap_or("");
        let attr = |k: &str| -> Option<String> {
            disp.split(';').map(str::trim).find_map(|a| a.strip_prefix(&format!("{k}=")).map(|v| v.trim_matches('"').to_string()))
        };
        parts.push(Part { name: attr("name").unwrap_or_default(), filename: attr("filename"), data: data.to_vec() });
        at = next + delim.len();
    }
    Ok(parts)
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
    send_with(t, method, path, body, "")
}

/// `send` with more header lines (each ending in \r\n)
pub fn send_with(t: &Target, method: &str, path: &str, body: Option<&Value>, headers: &str) -> Result<(u16, BufReader<Conn>), String> {
    let mut s = Conn::connect(t).map_err(|e| format!("nothing answers at {t} ({e})"))?;
    let text = body.map(|b| serde_json::to_vec(b).unwrap()).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: nextsycl\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", text.len())
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

/// A method of the front door's API (`POST /rpc/<name>` with its bearer token): its result, or the error's message
pub fn rpc(t: &Target, token: &str, name: &str, body: &Value, secs: u64) -> Result<Value, String> {
    let (_, mut r) = send_with(t, "POST", &format!("/rpc/{name}"), Some(body), &format!("Authorization: Bearer {token}\r\n"))?;
    r.get_ref().set_read_timeout(Some(Duration::from_secs(secs))).map_err(|e| e.to_string())?;
    let mut all = String::new();
    r.read_to_string(&mut all).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&all).map_err(|e| format!("the answer is not JSON: {e}"))?;
    if v["ok"] == true {
        Ok(v["result"].clone())
    } else {
        Err(v["error"]["message"].as_str().unwrap_or("the call failed").to_string())
    }
}

/// One request; the JSON answer, or an error carrying the answer's message.
pub fn call(t: &Target, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
    call_for(t, method, path, body, 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_lines_carry_their_own_reason_phrase() {
        // 2026-11-16: a 401 went out as "401 Internal Server Error"
        for (status, phrase) in [(200, "OK"), (206, "Partial Content"), (302, "Found"), (400, "Bad Request"), (401, "Unauthorized"),
                                 (403, "Forbidden"), (404, "Not Found"), (405, "Method Not Allowed"), (409, "Conflict"),
                                 (413, "Content Too Large"), (429, "Too Many Requests"), (499, "Client Closed Request"),
                                 (500, "Internal Server Error"), (502, "Bad Gateway"), (503, "Service Unavailable"), (504, "Gateway Timeout")] {
            let mut out = Vec::new();
            respond(&mut out, status, &serde_json::json!({}));
            let line = String::from_utf8(out).unwrap().lines().next().unwrap().to_string();
            assert_eq!(line, format!("HTTP/1.1 {status} {phrase}"));
            let mut out = Vec::new();
            respond_bytes(&mut out, status, "text/plain", b"x", "");
            assert!(String::from_utf8(out).unwrap().starts_with(&format!("HTTP/1.1 {status} {phrase}\r\n")));
        }
        // an unlisted status never claims another's name
        assert_eq!(reason(418), "Client Error");
        assert_eq!(reason(599), "Server Error");
        assert_ne!(reason(401), reason(500));
    }

    #[test]
    fn base64_round_trips_and_multipart_splits() {
        let data: Vec<u8> = (0..=255u8).collect();
        assert_eq!(unbase64(&base64(&data)).unwrap(), data);
        assert_eq!(unbase64("data:image/png;base64,aGk=").unwrap(), b"hi");
        let body = b"--XX\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nmake it red\r\n--XX\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n--XX--\r\n";
        let p = multipart(body, "XX").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].name.as_str(), p[0].data.as_slice()), ("prompt", b"make it red".as_slice()));
        assert_eq!((p[1].name.as_str(), p[1].filename.as_deref(), p[1].data.as_slice()), ("image[]", Some("a.png"), b"\x89PNG".as_slice()));
    }

    #[test]
    fn forward_to_names_the_new_path_and_passes_the_rest() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let up = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let got = read_request(&mut c).unwrap();
            c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok").unwrap();
            got
        });
        let raw = b"POST /open/m/v1/x?a=1 HTTP/1.1\r\nHost: box:8000\r\nConnection: keep-alive\r\nContent-Length: 5\r\n\r\nhello";
        let req = read_request(&raw[..]).unwrap();
        let mut answer = Vec::new();
        forward_to(&mut answer, &Target::Tcp(addr), &req, "/v1/x?a=1").unwrap();
        let got = up.join().unwrap();
        assert_eq!((got.method.as_str(), got.target.as_str()), ("POST", "/v1/x?a=1"));
        assert_eq!(got.host.as_deref(), Some("box:8000"));
        assert_eq!(got.body, b"hello");
        let head = String::from_utf8_lossy(&got.head).to_ascii_lowercase();
        assert!(head.contains("connection: close") && !head.contains("keep-alive"), "{head}");
        assert!(answer.ends_with(b"\r\n\r\nok"), "{}", String::from_utf8_lossy(&answer));
    }

    /// An upgrade passes on with its Upgrade / Connection headers, then bytes flow both ways until upstream closes
    #[test]
    fn a_splice_copies_both_ways() {
        let up_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let up_addr = up_l.local_addr().unwrap().to_string();
        let upstream = std::thread::spawn(move || {
            let (mut c, _) = up_l.accept().unwrap();
            let got = read_request(&mut c).unwrap();
            c.write_all(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n").unwrap();
            // echo what the client sends, upper-cased, then close
            let mut b = [0u8; 5];
            c.read_exact(&mut b).unwrap();
            c.write_all(&b.to_ascii_uppercase()).unwrap();
            got
        });
        // the door: a listener whose one connection is spliced to upstream
        let door_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let door_addr = door_l.local_addr().unwrap();
        let door = std::thread::spawn(move || {
            let (c, _) = door_l.accept().unwrap();
            let mut conn = Conn::Tcp(c);
            let req = read_request(&mut conn).unwrap();
            assert!(is_upgrade(&req.head));
            splice(&conn, &Target::Tcp(up_addr), &req, "/v1/realtime?model=m").unwrap();
        });
        let mut c = std::net::TcpStream::connect(door_addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        c.write_all(b"GET /open/m/v1/realtime?model=m HTTP/1.1\r\nHost: box\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n").unwrap();
        // a WebSocket client sends nothing before the 101 (RFC 6455): the head the door read had no bytes after it
        let mut head = Vec::new();
        let mut b = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            c.read_exact(&mut b).unwrap();
            head.push(b[0]);
        }
        assert!(head.starts_with(b"HTTP/1.1 101"), "{}", String::from_utf8_lossy(&head));
        c.write_all(b"hello").unwrap();
        let mut all = Vec::new();
        c.read_to_end(&mut all).unwrap();
        assert_eq!(all, b"HELLO");
        let got = upstream.join().unwrap();
        assert_eq!(got.target, "/v1/realtime?model=m");
        assert!(is_upgrade(&got.head) && header(&got.head, "connection").as_deref() == Some("Upgrade"));
        door.join().unwrap();
    }
}
