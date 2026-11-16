//! HTTPS beside the plain HTTP ports: the front door (`serve`, 8443 beside 8000) and the video studio (its port + 1)
//! answer the same routes over TLS, so a browser on another machine gets a secure context (the microphone).
//!
//! - **The certificate** is self-signed, made here (rcgen) and kept in a `tls/` directory (`cert.pem`, `key.pem` mode
//!   0600): made once, then reused. It is made again only when missing, unreadable, expired or within 30 days of it, or
//!   when it no longer names this host: `localhost`, `127.0.0.1`, `::1`, the hostname, its FQDN, the tailnet name, and
//!   every address on the host's interfaces (IPv6 link-local ones aside: they come and go with containers). Ten years.
//!   A new one names what the last one did as well (`names.txt`), so a name away at one start is no new certificate
//!   when it is back.
//!   `GET /cert.pem` on both ports hands it out, to trust once.
//! - **A connection**: the TLS handshake in the connection's own thread, then two pump threads move bytes between the
//!   TLS session (rustls, ring provider) and one end of an anonymous Unix socket pair; the server reads and writes the
//!   other end as `Conn::Tls`, which carries the TCP connection's real peer and local addresses as accepted. So
//!   `Conn::same_box` judges an HTTPS client exactly as a plain one - by the kernel's addresses, never by anything the
//!   client sends - and the socket pair, having no name or address, is no way in for anyone else. The plaintext side
//!   is an ordinary socket: `try_clone`, read timeouts, shutdown, streamed answers and WebSocket splices work as on TCP.

use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::{ServerConfig, ServerConnection};

use crate::http::{Conn, TlsConn};

/// The longest a client may take over its handshake
const HANDSHAKE_S: u64 = 20;
/// A kept certificate this close to its end is made again
const RENEW_BEFORE_S: u64 = 30 * 86400;

/// The names a certificate must cover: DNS names and IP addresses
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Names {
    pub dns: Vec<String>,
    pub ips: Vec<IpAddr>,
}

impl Names {
    /// This host's: localhost and the loopback addresses, the hostname, its FQDN, the tailnet name (when tailscale says
    /// one quickly), and the addresses on its interfaces
    pub fn of_host() -> Names {
        let mut n = Names { dns: vec!["localhost".into()], ips: vec![IpAddr::from([127, 0, 0, 1]), IpAddr::V6(Ipv6Addr::LOCALHOST)] };
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim().to_string();
        n.add_dns(&host);
        if let Some(f) = output_within("hostname", &["-f"], 2) {
            n.add_dns(f.trim());
        }
        if let Some(t) = output_within("tailscale", &["status", "--json", "--peers=false"], 2)
            .and_then(|o| serde_json::from_str::<serde_json::Value>(&o).ok())
        {
            n.add_dns(t["Self"]["DNSName"].as_str().unwrap_or(""));
        }
        for ip in interface_ips() {
            n.add_ip(ip);
        }
        n
    }

    /// A DNS name (lower case, without a trailing dot), once; names a certificate cannot hold are left out
    pub fn add_dns(&mut self, name: &str) {
        let name = name.trim().trim_end_matches('.').to_ascii_lowercase();
        if name.is_empty() || name.parse::<IpAddr>().is_ok() || self.dns.contains(&name) || ServerName::try_from(name.as_str()).is_err() {
            return;
        }
        self.dns.push(name);
    }

    pub fn add_ip(&mut self, ip: IpAddr) {
        if !self.ips.contains(&ip) {
            self.ips.push(ip);
        }
    }

    /// Every name as the certificate's subject alternative names take them
    fn all(&self) -> Vec<String> {
        self.dns.iter().cloned().chain(self.ips.iter().map(IpAddr::to_string)).collect()
    }
}

/// The addresses on this host's interfaces: IPv4 from the kernel's local routes, IPv6 from if_inet6 (link-local left out)
fn interface_ips() -> Vec<IpAddr> {
    let mut out = Vec::new();
    // /proc/net/fib_trie: an address line ("|-- 192.168.1.5"), then "/32 host LOCAL" when it is one of this host's
    let trie = std::fs::read_to_string("/proc/net/fib_trie").unwrap_or_default();
    let mut last: Option<IpAddr> = None;
    for l in trie.lines() {
        let t = l.trim();
        if let Some(a) = t.strip_prefix("|-- ").or_else(|| t.strip_prefix("+-- ")) {
            last = a.split('/').next().and_then(|a| a.parse().ok());
        } else if t.starts_with("/32 host LOCAL") {
            if let Some(ip) = last.filter(|ip| !out.contains(ip)) {
                out.push(ip);
            }
        }
    }
    // /proc/net/if_inet6: 32 hex digits, the interface index, prefix length, scope, flags, name
    for l in std::fs::read_to_string("/proc/net/if_inet6").unwrap_or_default().lines() {
        let Some(hex) = l.split_whitespace().next().filter(|h| h.len() == 32) else { continue };
        let Ok(v) = u128::from_str_radix(hex, 16) else { continue };
        let ip = Ipv6Addr::from(v);
        if (ip.segments()[0] & 0xffc0) == 0xfe80 {
            continue;
        }
        if !out.contains(&IpAddr::V6(ip)) {
            out.push(IpAddr::V6(ip));
        }
    }
    out
}

/// A command's standard output, when it succeeds within `secs` (otherwise killed: None)
fn output_within(cmd: &str, args: &[&str], secs: u64) -> Option<String> {
    let mut c = std::process::Command::new(cmd).args(args).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null()).spawn().ok()?;
    let mut out = c.stdout.take()?;
    // read beside the wait: a large answer would otherwise fill the pipe and never exit
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let t0 = Instant::now();
    loop {
        match c.try_wait() {
            Ok(Some(st)) => {
                let s = reader.join().ok()?;
                return st.success().then_some(s);
            }
            Ok(None) if t0.elapsed() < Duration::from_secs(secs) => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = c.kill();
                let _ = c.wait();
                return None;
            }
        }
    }
}

/// A server's TLS: its certificate and the rustls configuration made from it
pub struct Tls {
    config: Arc<ServerConfig>,
    cert_pem: String,
    dir: PathBuf,
    /// made by this start (not reused)
    pub made: bool,
}

impl Tls {
    /// The certificate kept in `dir` (cert.pem, key.pem) for this host's names, made there when it has to be
    pub fn open(dir: &Path) -> Result<Tls, String> {
        Self::open_for(dir, &Names::of_host())
    }

    /// `open` for these names
    pub fn open_for(dir: &Path, names: &Names) -> Result<Tls, String> {
        match load(dir, names) {
            Ok(t) => Ok(t),
            Err(why) => {
                eprintln!("{}: {why}: making a new self-signed certificate", dir.display());
                // the names the last one held as well (names.txt beside it): a name or address that is away at this
                // start (a VPN down) and back at the next does not make one certificate after another
                let mut all = names.clone();
                for l in std::fs::read_to_string(dir.join("names.txt")).unwrap_or_default().lines().map(str::trim) {
                    match l.parse::<IpAddr>() {
                        Ok(ip) => all.add_ip(ip),
                        Err(_) => all.add_dns(l),
                    }
                }
                make(dir, &all)?;
                let mut t = load(dir, names).map_err(|e| format!("{}: the certificate just made does not load: {e}", dir.display()))?;
                t.made = true;
                Ok(t)
            }
        }
    }

    /// The certificate, PEM (GET /cert.pem)
    pub fn cert_pem(&self) -> &str {
        &self.cert_pem
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// An accepted TCP connection's TLS handshake, then the connection as the server reads it: `Conn::Tls`, with the
    /// socket's own peer and local addresses
    pub fn accept(&self, tcp: TcpStream) -> io::Result<Conn> {
        let (peer, local) = (tcp.peer_addr()?, tcp.local_addr()?);
        // records out at once (a WebSocket's small frames: crate::http::Conn::nodelay)
        let _ = tcp.set_nodelay(true);
        self.accept_as(tcp, peer, local)
    }

    /// `accept` with the addresses given (the tests: a peer from elsewhere without a second machine)
    pub(crate) fn accept_as(&self, tcp: TcpStream, peer: SocketAddr, local: SocketAddr) -> io::Result<Conn> {
        let mut conn = ServerConnection::new(self.config.clone()).map_err(io::Error::other)?;
        // the pumps drain what they encrypt as they go: no limit needed on the session's own buffer
        conn.set_buffer_limit(None);
        tcp.set_read_timeout(Some(Duration::from_secs(HANDSHAKE_S)))?;
        tcp.set_write_timeout(Some(Duration::from_secs(HANDSHAKE_S)))?;
        let mut t = &tcp;
        while conn.is_handshaking() {
            conn.complete_io(&mut t)?;
        }
        tcp.set_read_timeout(None)?;
        tcp.set_write_timeout(None)?;
        let (app, pump) = UnixStream::pair()?;
        let shared = Arc::new(Session { conn: Mutex::new(conn), out: Mutex::new(tcp.try_clone()?) });
        let (sh, tcp_in, pump_out) = (shared.clone(), tcp.try_clone()?, pump.try_clone()?);
        std::thread::spawn(move || inbound(&sh, tcp_in, pump_out));
        std::thread::spawn(move || outbound(&shared, pump, tcp));
        Ok(Conn::Tls(TlsConn { s: app, peer, local }))
    }
}

/// One TLS session shared by its two pumps: the session itself, then the TCP connection's writing side (taken before
/// the session is let go, so records reach the wire in the order they were made)
struct Session {
    conn: Mutex<ServerConnection>,
    out: Mutex<TcpStream>,
}

impl Session {
    /// What the session has to send, sent: made under its lock, written under the writer's
    fn send(&self, mut c: std::sync::MutexGuard<'_, ServerConnection>) -> io::Result<()> {
        let mut bytes = Vec::new();
        while c.wants_write() {
            c.write_tls(&mut bytes)?;
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        drop(c);
        out.write_all(&bytes)?;
        out.flush()
    }
}

/// The client's bytes: read from TCP, decrypted, written to the server's side; the end (EOF, close_notify) passed on
/// as the end of what the server reads
fn inbound(sh: &Session, mut tcp: TcpStream, mut app: UnixStream) {
    let mut buf = vec![0u8; 64 << 10];
    let (mut at, mut end) = (0usize, 0usize);
    let r: io::Result<()> = (|| loop {
        let mut plain = Vec::new();
        let mut closed = false;
        {
            let mut c = sh.conn.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                let mut fed = 0;
                if at < end {
                    fed = c.read_tls(&mut &buf[at..end])?;
                    at += fed;
                }
                let st = match c.process_new_packets() {
                    Ok(st) => st,
                    Err(e) => {
                        // the alert that says why, then the end
                        let _ = sh.send(c);
                        return Err(io::Error::new(io::ErrorKind::InvalidData, e));
                    }
                };
                let k = st.plaintext_bytes_to_read();
                if k > 0 {
                    let o = plain.len();
                    plain.resize(o + k, 0);
                    c.reader().read_exact(&mut plain[o..])?;
                }
                closed |= st.peer_has_closed();
                if at >= end || closed {
                    break;
                }
                if fed == 0 && k == 0 {
                    return Err(io::Error::other("the TLS session takes no more of what was read"));
                }
            }
            sh.send(c)?;
        }
        if !plain.is_empty() && app.write_all(&plain).is_err() {
            // the server's side has gone: the other pump ends the connection
            return Ok(());
        }
        if closed {
            return Ok(());
        }
        let n = tcp.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        (at, end) = (0, n);
    })();
    // what the server reads ends here (its answer can still go out); on an error the whole connection
    let _ = app.shutdown(Shutdown::Write);
    if r.is_err() {
        let _ = app.shutdown(Shutdown::Both);
        let _ = tcp.shutdown(Shutdown::Both);
    }
}

/// The server's bytes: read from its side, encrypted, sent; when it closes (or shuts down), close_notify and the TCP
/// connection closed
fn outbound(sh: &Session, mut app: UnixStream, tcp: TcpStream) {
    let mut buf = vec![0u8; 64 << 10];
    loop {
        let n = match app.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let mut c = sh.conn.lock().unwrap_or_else(|e| e.into_inner());
        if c.writer().write_all(&buf[..n]).is_err() || sh.send(c).is_err() {
            // the client has gone: the server's next write fails as it would on TCP
            let _ = app.shutdown(Shutdown::Both);
            let _ = tcp.shutdown(Shutdown::Both);
            return;
        }
    }
    let mut c = sh.conn.lock().unwrap_or_else(|e| e.into_inner());
    c.send_close_notify();
    let _ = sh.send(c);
    let _ = tcp.shutdown(Shutdown::Both);
}

/// The kept certificate and key, when they load, are still good for 30 days and name every one of `names`
fn load(dir: &Path, names: &Names) -> Result<Tls, String> {
    let (cf, kf) = (dir.join("cert.pem"), dir.join("key.pem"));
    let cert_pem = std::fs::read_to_string(&cf).map_err(|e| format!("no certificate ({}: {e})", cf.display()))?;
    let cert = CertificateDer::from_pem_slice(cert_pem.as_bytes()).map_err(|e| format!("an unreadable certificate ({e:?})"))?;
    let key = PrivateKeyDer::from_pem_file(&kf).map_err(|e| format!("an unreadable key ({}: {e:?})", kf.display()))?;
    let ee = webpki::EndEntityCert::try_from(&cert).map_err(|e| format!("an unreadable certificate ({e})"))?;
    // valid now and in 30 days: checked as its own trust anchor (it signed itself)
    let anchor = webpki::anchor_from_trusted_cert(&cert).map_err(|e| format!("an unreadable certificate ({e})"))?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    for t in [now, now + Duration::from_secs(RENEW_BEFORE_S)] {
        ee.verify_for_usage(webpki::ALL_VERIFICATION_ALGS, std::slice::from_ref(&anchor), &[], UnixTime::since_unix_epoch(t),
                            webpki::KeyUsage::server_auth(), None, None)
            .map_err(|e| format!("the certificate is no longer valid ({e:?})"))?;
    }
    for n in &names.dns {
        let sn = ServerName::try_from(n.as_str()).map_err(|e| format!("{n}: {e}"))?;
        ee.verify_is_valid_for_subject_name(&sn).map_err(|_| format!("the certificate does not name {n}"))?;
    }
    for ip in &names.ips {
        ee.verify_is_valid_for_subject_name(&ServerName::IpAddress((*ip).into())).map_err(|_| format!("the certificate does not name {ip}"))?;
    }
    // the key kept private
    if let Ok(m) = std::fs::metadata(&kf) {
        if m.permissions().mode() & 0o077 != 0 {
            let _ = std::fs::set_permissions(&kf, std::fs::Permissions::from_mode(0o600));
        }
    }
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)
        .map_err(|e| format!("the certificate and key do not go together ({e})"))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Tls { config: Arc::new(config), cert_pem, dir: dir.to_path_buf(), made: false })
}

/// A new self-signed certificate for `names`, ten years from now, and its key (0600) in `dir`
fn make(dir: &Path, names: &Names) -> Result<(), String> {
    let fail = |e: rcgen::Error| format!("cannot make a certificate: {e}");
    let mut params = rcgen::CertificateParams::new(names.all()).map_err(fail)?;
    let host = names.dns.iter().find(|n| *n != "localhost").cloned().unwrap_or_else(|| "localhost".into());
    params.distinguished_name = rcgen::DistinguishedName::new();
    params.distinguished_name.push(rcgen::DnType::CommonName, format!("nextsycl {host}"));
    params.distinguished_name.push(rcgen::DnType::OrganizationName, "nextsycl (self-signed)");
    let year = 1970 + (SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() / 31_556_952) as i32;
    params.not_before = rcgen::date_time_ymd(year - 1, 1, 1);
    params.not_after = rcgen::date_time_ymd(year + 10, 1, 1);
    params.key_usages = vec![rcgen::KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ServerAuth];
    let key = rcgen::KeyPair::generate().map_err(fail)?;
    let cert = params.self_signed(&key).map_err(fail)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    // each written beside, then moved in: a start that dies half way leaves the old pair or none, never half of each
    let write = |name: &str, text: &str, mode: u32| -> Result<(), String> {
        let (tmp, to) = (dir.join(format!(".{name}.tmp")), dir.join(name));
        let _ = std::fs::remove_file(&tmp);
        let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(mode).open(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        f.write_all(text.as_bytes()).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(mode)).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &to).map_err(|e| format!("{}: {e}", to.display()))
    };
    write("key.pem", &key.serialize_pem(), 0o600)?;
    write("cert.pem", &cert.pem(), 0o644)?;
    // what it names, for the next one (a list only: the certificate itself is what is checked)
    write("names.txt", &(names.all().join("\n") + "\n"), 0o644)?;
    Ok(())
}

/// HTTPS on `addr`: each connection's handshake in a thread of its own, then `handle` with it (the plain port's handler).
/// The certificate from (or made in) `dir`. Returns the address bound and the TLS (its certificate for /cert.pem); the
/// accepting goes on in a thread
pub fn listen(addr: &str, dir: &Path, handle: impl Fn(Conn) + Send + Sync + 'static) -> Result<(SocketAddr, Arc<Tls>), String> {
    let tls = Arc::new(Tls::open(dir)?);
    let bound = listen_with(addr, tls.clone(), handle)?;
    Ok((bound, tls))
}

/// `listen` with a TLS already opened
pub fn listen_with(addr: &str, tls: Arc<Tls>, handle: impl Fn(Conn) + Send + Sync + 'static) -> Result<SocketAddr, String> {
    let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
    let bound = l.local_addr().map_err(|e| e.to_string())?;
    let handle = Arc::new(handle);
    std::thread::spawn(move || {
        for c in l.incoming().flatten() {
            let (tls, handle) = (tls.clone(), handle.clone());
            std::thread::spawn(move || {
                // a failed handshake (a client that does not trust the certificate, a port scan) ends quietly
                if let Ok(conn) = tls.accept(c) {
                    handle(conn);
                }
            });
        }
    });
    Ok(bound)
}

/// The HTTPS address for a plain `host:port` listen address and an HTTPS port (`[::]:8090`, 8091 -> `[::]:8091`)
pub fn beside(listen: &str, port: u16) -> String {
    match listen.rsplit_once(':') {
        Some((h, _)) => format!("{h}:{port}"),
        None => format!("{listen}:{port}"),
    }
}

/// `GET /cert.pem`: the certificate to trust (404 when HTTPS is off)
pub fn respond_cert(w: impl Write, tls: Option<&Tls>) {
    match tls {
        Some(t) => crate::http::respond_bytes(w, 200, "application/x-pem-file", t.cert_pem().as_bytes(),
                                              "Content-Disposition: inline; filename=\"nextsycl-cert.pem\"\r\nCache-Control: no-cache\r\n"),
        None => crate::http::respond(w, 404, &serde_json::json!({"error": "HTTPS is off here: no certificate"})),
    }
}

/// A TLS client that trusts `cert_pem` only (the tests: this module's and the front door's)
#[doc(hidden)]
pub fn test_client(cert_pem: &str, addr: SocketAddr, name: &str) -> io::Result<rustls::StreamOwned<rustls::ClientConnection, TcpStream>> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add(CertificateDer::from_pem_slice(cert_pem.as_bytes()).map_err(|e| io::Error::other(format!("{e:?}")))?).map_err(io::Error::other)?;
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .map_err(io::Error::other)?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let sn = ServerName::try_from(name.to_string()).map_err(io::Error::other)?;
    let conn = rustls::ClientConnection::new(Arc::new(config), sn).map_err(io::Error::other)?;
    let tcp = TcpStream::connect(addr)?;
    tcp.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut s = rustls::StreamOwned::new(conn, tcp);
    while s.conn.is_handshaking() {
        s.conn.complete_io(&mut s.sock)?;
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{self, Target};
    use std::io::BufRead;
    use std::sync::mpsc;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nextsycl-tls-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn names() -> Names {
        let mut n = Names { dns: vec!["localhost".into()], ips: vec![IpAddr::from([127, 0, 0, 1]), IpAddr::V6(Ipv6Addr::LOCALHOST)] };
        n.add_dns("box.example.");
        n.add_ip(IpAddr::from([192, 0, 2, 10]));
        n
    }

    /// The whole answer of one request over TLS (the server closes)
    fn get(cert: &str, addr: SocketAddr, req: &str) -> String {
        let mut s = test_client(cert, addr, "localhost").unwrap();
        s.write_all(req.as_bytes()).unwrap();
        s.flush().unwrap();
        let mut all = Vec::new();
        // close_notify ends it cleanly; a bare TCP end would be an error here
        s.read_to_end(&mut all).unwrap();
        String::from_utf8_lossy(&all).into_owned()
    }

    #[test]
    fn the_certificate_names_the_host_and_lasts() {
        let dir = tmp("names");
        let n = names();
        let t = Tls::open_for(&dir, &n).unwrap();
        assert!(t.made);
        let der = CertificateDer::from_pem_slice(t.cert_pem().as_bytes()).unwrap();
        let ee = webpki::EndEntityCert::try_from(&der).unwrap();
        for d in ["localhost", "box.example"] {
            assert!(ee.verify_is_valid_for_subject_name(&ServerName::try_from(d).unwrap()).is_ok(), "{d}");
        }
        for ip in ["127.0.0.1", "::1", "192.0.2.10"] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(ee.verify_is_valid_for_subject_name(&ServerName::IpAddress(ip.into())).is_ok(), "{ip}");
        }
        assert!(ee.verify_is_valid_for_subject_name(&ServerName::try_from("other.example").unwrap()).is_err());
        assert!(ee.verify_is_valid_for_subject_name(&ServerName::IpAddress(IpAddr::from([192, 0, 2, 11]).into())).is_err());
        // good for years: still valid nine years on
        let anchor = webpki::anchor_from_trusted_cert(&der).unwrap();
        let later = SystemTime::now().duration_since(UNIX_EPOCH).unwrap() + Duration::from_secs(9 * 365 * 86400);
        ee.verify_for_usage(webpki::ALL_VERIFICATION_ALGS, &[anchor], &[], UnixTime::since_unix_epoch(later), webpki::KeyUsage::server_auth(), None, None)
            .unwrap();
        // this host's own names: localhost, the loopback addresses, its hostname
        let h = Names::of_host();
        assert!(h.dns.contains(&"localhost".to_string()) && h.ips.contains(&IpAddr::from([127, 0, 0, 1])) && h.ips.contains(&IpAddr::V6(Ipv6Addr::LOCALHOST)), "{h:?}");
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim().to_ascii_lowercase();
        if ServerName::try_from(host.as_str()).is_ok() && !host.is_empty() {
            assert!(h.dns.contains(&host), "{h:?}");
        }
        assert!(!h.ips.iter().any(|ip| matches!(ip, IpAddr::V6(v) if v.segments()[0] & 0xffc0 == 0xfe80)), "link-local left out: {h:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_certificate_is_kept_and_made_again_only_when_it_must() {
        let dir = tmp("keep");
        let n = names();
        let first = Tls::open_for(&dir, &n).unwrap();
        assert!(first.made);
        assert_eq!(std::fs::metadata(dir.join("key.pem")).unwrap().permissions().mode() & 0o777, 0o600);
        // a second start: the same certificate
        let again = Tls::open_for(&dir, &n).unwrap();
        assert!(!again.made);
        assert_eq!(first.cert_pem(), again.cert_pem());
        // fewer names than it holds: still kept
        let fewer = Names { dns: vec!["localhost".into()], ips: vec![IpAddr::from([127, 0, 0, 1])] };
        assert!(!Tls::open_for(&dir, &fewer).unwrap().made);
        // a key left readable to others is made private again
        std::fs::set_permissions(dir.join("key.pem"), std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(!Tls::open_for(&dir, &n).unwrap().made);
        assert_eq!(std::fs::metadata(dir.join("key.pem")).unwrap().permissions().mode() & 0o777, 0o600);
        // the host has a new address: made again, naming it
        let mut more = n.clone();
        more.add_ip(IpAddr::from([198, 51, 100, 4]));
        let renewed = Tls::open_for(&dir, &more).unwrap();
        assert!(renewed.made);
        assert_ne!(renewed.cert_pem(), first.cert_pem());
        assert!(!Tls::open_for(&dir, &more).unwrap().made);
        // a new name likewise
        let mut named = more.clone();
        named.add_dns("box.tail1234.ts.net");
        assert!(Tls::open_for(&dir, &named).unwrap().made);
        // an address gone and another new: made again, keeping the old ones too (back next start: no new certificate)
        let mut moved = Names { dns: vec!["localhost".into()], ips: vec![IpAddr::from([127, 0, 0, 1])] };
        moved.add_ip(IpAddr::from([203, 0, 113, 9]));
        assert!(Tls::open_for(&dir, &moved).unwrap().made);
        assert!(!Tls::open_for(&dir, &named).unwrap().made, "the names before are still held");
        assert!(!Tls::open_for(&dir, &moved).unwrap().made);
        // unreadable: made again
        std::fs::write(dir.join("cert.pem"), "not a certificate").unwrap();
        assert!(Tls::open_for(&dir, &named).unwrap().made);
        std::fs::remove_file(dir.join("key.pem")).unwrap();
        assert!(Tls::open_for(&dir, &named).unwrap().made);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A plain GET, a streamed answer (each part arrives as it is written), /cert.pem and a WebSocket splice, all
    /// over HTTPS to the handler a plain port would have; the peer as it really is (loopback here: this box)
    #[test]
    fn https_answers_as_the_plain_port_does() {
        let dir = tmp("trip");
        let tls = Arc::new(Tls::open_for(&dir, &names()).unwrap());
        // an upstream WebSocket server for the splice: the 101, then what it reads upper-cased back
        let up_l = TcpListener::bind("127.0.0.1:0").unwrap();
        let up_addr = up_l.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (mut c, _) = up_l.accept().unwrap();
            let got = http::read_request(&mut c).unwrap();
            assert_eq!(got.target, "/v1/realtime?model=m");
            c.write_all(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n").unwrap();
            let mut b = [0u8; 5];
            c.read_exact(&mut b).unwrap();
            c.write_all(&b.to_ascii_uppercase()).unwrap();
        });
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        let t2 = tls.clone();
        let addr = listen_with("127.0.0.1:0", tls.clone(), move |mut c| {
            let req = http::read_request(&mut c).unwrap();
            match req.path.as_str() {
                "/plain" => {
                    let same = c.same_box();
                    http::respond(&mut c, 200, &serde_json::json!({"same_box": same, "body": String::from_utf8_lossy(&req.body)}))
                }
                "/cert.pem" => respond_cert(&mut c, Some(&t2)),
                "/stream" => {
                    // the first part, then the second only once the client has the first
                    c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: one\n\n").unwrap();
                    c.flush().unwrap();
                    go_rx.lock().unwrap().recv_timeout(Duration::from_secs(10)).expect("the client saw the first part before the answer ended");
                    c.write_all(b"data: two\n\n").unwrap();
                }
                "/open/m/v1/realtime" => {
                    assert!(http::is_upgrade(&req.head));
                    http::splice(&c, &Target::Tcp(up_addr.clone()), &req, "/v1/realtime?model=m").unwrap();
                }
                _ => http::respond(&mut c, 404, &serde_json::json!({})),
            }
        }).unwrap();

        let a = get(tls.cert_pem(), addr, "POST /plain HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\nhello");
        assert!(a.starts_with("HTTP/1.1 200"), "{a}");
        assert!(a.contains(r#""same_box":true"#) && a.contains(r#""body":"hello""#), "{a}");

        let a = get(tls.cert_pem(), addr, "GET /cert.pem HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(a.contains("Content-Type: application/x-pem-file") && a.ends_with(tls.cert_pem()), "{a}");

        // streamed: the first part is read while the server still holds the rest back
        let mut s = test_client(tls.cert_pem(), addr, "127.0.0.1").unwrap();
        s.write_all(b"GET /stream HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        s.flush().unwrap();
        let mut r = std::io::BufReader::new(s);
        let mut seen = String::new();
        while !seen.contains("data: one") {
            let mut l = String::new();
            assert!(r.read_line(&mut l).unwrap() > 0, "ended before the first part: {seen}");
            seen.push_str(&l);
        }
        go_tx.send(()).unwrap();
        let mut rest = String::new();
        r.read_to_string(&mut rest).unwrap();
        assert!(rest.contains("data: two"), "{rest}");

        // a WebSocket upgrade spliced on, then bytes both ways
        let mut s = test_client(tls.cert_pem(), addr, "localhost").unwrap();
        s.write_all(b"GET /open/m/v1/realtime HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n").unwrap();
        s.flush().unwrap();
        let mut head = Vec::new();
        let mut b = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            s.read_exact(&mut b).unwrap();
            head.push(b[0]);
        }
        assert!(head.starts_with(b"HTTP/1.1 101"), "{}", String::from_utf8_lossy(&head));
        s.write_all(b"hello").unwrap();
        s.flush().unwrap();
        let mut all = Vec::new();
        s.read_to_end(&mut all).unwrap();
        assert_eq!(all, b"HELLO");

        // a client that does not trust this certificate gets no connection
        let other = Tls::open_for(&tmp("other"), &names()).unwrap();
        assert!(test_client(other.cert_pem(), addr, "localhost").and_then(|mut s| {
            s.write_all(b"GET /plain HTTP/1.1\r\n\r\n")?;
            s.flush()?;
            let mut v = Vec::new();
            s.read_to_end(&mut v).map(|_| v)
        }).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(tmp("other"));
    }

    /// The security of `same_box` over HTTPS: the TCP peer's address decides, as the kernel gave it. A client from
    /// another machine is not this box, whatever headers or prefixes it sends inside the session; and bytes sent
    /// before the handshake (a PROXY-protocol line) make no connection at all.
    #[test]
    fn a_peer_from_elsewhere_is_not_this_box_whatever_it_sends() {
        let dir = tmp("peer");
        let tls = Arc::new(Tls::open_for(&dir, &names()).unwrap());
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let remote: SocketAddr = "203.0.113.7:51000".parse().unwrap();
        let here: SocketAddr = "192.0.2.10:8443".parse().unwrap();
        let forged = [
            "GET /chat/held HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Forwarded-For: 127.0.0.1\r\nX-Real-IP: 127.0.0.1\r\nForwarded: for=127.0.0.1;by=127.0.0.1\r\n\r\n",
            "PROXY TCP4 127.0.0.1 127.0.0.1 40000 8443\r\nGET /chat/held HTTP/1.1\r\nHost: localhost\r\n\r\n",
            "GET /chat/held HTTP/1.1\r\nHost: 192.0.2.10:8443\r\nX-Forwarded-For: 192.0.2.10\r\nX-Original-Peer: 192.0.2.10\r\n\r\n",
        ];
        let server = {
            let tls = tls.clone();
            std::thread::spawn(move || {
                let mut verdicts = Vec::new();
                for _ in 0..forged.len() {
                    let (tcp, _) = l.accept().unwrap();
                    let mut c = tls.accept_as(tcp, remote, here).unwrap();
                    let req = http::read_request(&mut c);
                    // as the front door gates /chat/held and /video/blocked
                    let (same, clone_same) = (c.same_box(), c.try_clone().unwrap().same_box());
                    let st = if same { 200 } else { 404 };
                    http::respond(&mut c, st, &serde_json::json!({}));
                    verdicts.push((same, clone_same, req.is_ok()));
                }
                // bytes before the handshake: no HTTPS connection comes of them
                let (tcp, _) = l.accept().unwrap();
                let pre = tls.accept_as(tcp, remote, here);
                (verdicts, pre.is_err())
            })
        };
        for f in forged {
            let a = get(tls.cert_pem(), addr, f);
            assert!(a.starts_with("HTTP/1.1 404"), "{a}");
        }
        let mut raw = TcpStream::connect(addr).unwrap();
        raw.write_all(b"PROXY TCP4 127.0.0.1 127.0.0.1 40000 8443\r\n").unwrap();
        let mut sink = Vec::new();
        raw.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let _ = raw.read_to_end(&mut sink);
        let (verdicts, pre_refused) = server.join().unwrap();
        assert!(verdicts.iter().all(|(same, clone_same, _)| !same && !clone_same), "{verdicts:?}");
        assert!(pre_refused);

        // the rule itself, on the addresses only: loopback and this machine's own address are this box, others not
        let conn = |peer: &str, local: &str| {
            let (a, _b) = UnixStream::pair().unwrap();
            Conn::Tls(TlsConn { s: a, peer: peer.parse().unwrap(), local: local.parse().unwrap() })
        };
        assert!(conn("127.0.0.1:5", "127.0.0.1:8443").same_box());
        assert!(conn("[::1]:5", "[::1]:8443").same_box());
        assert!(conn("192.0.2.10:5", "192.0.2.10:8443").same_box());
        assert!(!conn("192.0.2.11:5", "192.0.2.10:8443").same_box());
        assert!(!conn("203.0.113.7:5", "127.0.0.1:8443").same_box());
        assert!(!conn("[2001:db8::1]:5", "[2001:db8::2]:8443").same_box());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_https_address_is_beside_the_plain_one() {
        assert_eq!(beside("0.0.0.0:8090", 8091), "0.0.0.0:8091");
        assert_eq!(beside("[::]:8000", 8443), "[::]:8443");
        assert_eq!(beside("127.0.0.1:8095", 8096), "127.0.0.1:8096");
    }
}
