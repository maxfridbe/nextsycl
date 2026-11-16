//! Live conversations: `GET /v1/realtime` upgraded to a WebSocket speaking OpenAI Realtime's events, over a full-duplex
//! speech engine (`nextsycl_audio::AudioEngine::converse`; docs/api/realtime.md). The audio server answers it for its
//! model; the front door passes the upgrade on to the server of the model asked for (`?model=`) and copies both ways.
//!
//! ```text
//!   client                                          server
//!   session.update {session: {instructions, voice,  session.created / session.updated {session: ..., nextsycl: {...}}
//!     input_audio_format, output_audio_format,      input_audio_buffer.speech_started / speech_stopped / committed
//!     turn_detection, tools, temperature, ...}}     conversation.item.created (the user's turn; an answer's message)
//!   input_audio_buffer.append {audio: base64}       response.created, response.output_item.added, content_part.added
//!   input_audio_buffer.commit / clear               response.audio_transcript.delta {delta}  (its text as written)
//!   response.create / response.cancel               response.audio.delta {delta: base64}      (its sound as spoken)
//!   conversation.item.create {item: message |       response.function_call_arguments.done {call_id, name, arguments}
//!     function_call_output}                         response.audio.done, response.audio_transcript.done, ...,
//!   conversation.item.truncate {item_id,            response.done {response: {id, status, output}}
//!     audio_end_ms}                                 error {error: {type, code, message}}
//! ```
//!
//! Audio: pcm16 (16-bit little-endian mono, 24 kHz unless the session says otherwise), g711_ulaw / g711_alaw (8 kHz),
//! or pcm_f32le (Sophon's own server's); in at any rate (`sample_rate_hz` / the GA shape's `rate`), resampled for the
//! model. The GA shape (session.type "realtime", audio.input / audio.output, response.output_audio.delta ...) is
//! spoken back to a client that uses it; beta names otherwise. Extensions under `nextsycl` in session.update: language,
//! saved voices by name (the voice library's recording, cloned), the engine's options, a lockstep model's character
//! card; session.created tells the model's rates, voices and whether it runs in lockstep (`frame_ms`: Haru-4B's one
//! frame in, one out - every input frame answered by one audio delta of its own within one long-running response).

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nextsycl_audio::{AudioEngine, Conversation, RealtimeEvent, RealtimeSession, Reference};
use serde_json::{json, Value};

use crate::http::{self, Conn, Request};

/// The largest message taken (a few seconds of audio in base64 is well under it)
const MAX_MESSAGE: usize = 32 << 20;

// ---- WebSocket (RFC 6455)

/// SHA-1, for the handshake's accept key only
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut m = data.to_vec();
    let bits = (data.len() as u64) * 8;
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bits.to_be_bytes());
    for block in m.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// The handshake's Sec-WebSocket-Accept for a client's key
pub fn accept_key(key: &str) -> String {
    http::base64(&sha1(format!("{}258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key.trim()).as_bytes()))
}

/// A message: text or binary (pings answered, a close ends it: None)
enum Msg {
    Text(String),
    Binary(Vec<u8>),
}

/// The writing side of a WebSocket (frames from the server are not masked)
#[derive(Clone)]
pub struct WsOut(Arc<Mutex<Conn>>);

impl WsOut {
    fn frame(&self, op: u8, payload: &[u8]) -> std::io::Result<()> {
        let mut h = vec![0x80 | op];
        match payload.len() {
            n if n < 126 => h.push(n as u8),
            n if n < 65536 => {
                h.push(126);
                h.extend_from_slice(&(n as u16).to_be_bytes());
            }
            n => {
                h.push(127);
                h.extend_from_slice(&(n as u64).to_be_bytes());
            }
        }
        let mut c = self.0.lock().unwrap_or_else(|e| e.into_inner());
        c.write_all(&h)?;
        c.write_all(payload)?;
        c.flush()
    }

    pub fn text(&self, v: &Value) -> std::io::Result<()> {
        self.frame(0x1, v.to_string().as_bytes())
    }

    fn close(&self) {
        let _ = self.frame(0x8, &1000u16.to_be_bytes());
    }
}

/// The next message (fragments joined, pings answered); None: closed
fn read_msg(r: &mut impl Read, out: &WsOut) -> std::io::Result<Option<Msg>> {
    let mut data = Vec::new();
    let mut kind = 0u8;
    loop {
        let mut h = [0u8; 2];
        r.read_exact(&mut h)?;
        let (fin, op, masked) = (h[0] & 0x80 != 0, h[0] & 0x0f, h[1] & 0x80 != 0);
        let mut len = (h[1] & 0x7f) as u64;
        if len == 126 {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            len = u16::from_be_bytes(b) as u64;
        } else if len == 127 {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            len = u64::from_be_bytes(b);
        }
        if len as usize + data.len() > MAX_MESSAGE {
            return Err(std::io::Error::other("a message over 32 MiB"));
        }
        let mut mask = [0u8; 4];
        if masked {
            r.read_exact(&mut mask)?;
        }
        let mut p = vec![0u8; len as usize];
        r.read_exact(&mut p)?;
        if masked {
            for (i, b) in p.iter_mut().enumerate() {
                *b ^= mask[i % 4];
            }
        }
        match op {
            0x8 => {
                out.close();
                return Ok(None);
            }
            0x9 => {
                out.frame(0xA, &p)?;
                continue;
            }
            0xA => continue,
            0x0 => data.extend_from_slice(&p),
            0x1 | 0x2 => {
                kind = op;
                data = p;
            }
            _ => return Err(std::io::Error::other(format!("WebSocket opcode {op}"))),
        }
        if fin {
            return Ok(Some(if kind == 0x2 { Msg::Binary(data) } else { Msg::Text(String::from_utf8_lossy(&data).into_owned()) }));
        }
    }
}

/// The 101 answer: the accept key, and "realtime" when the client offered sub-protocols (a browser's: realtime,
/// openai-insecure-api-key.<key>, openai-beta.realtime-v1)
fn upgrade(c: &mut Conn, req: &Request) -> std::io::Result<()> {
    let key = http::header(&req.head, "sec-websocket-key").unwrap_or_default();
    let proto = http::header(&req.head, "sec-websocket-protocol")
        .map(|p| if p.split(',').any(|x| x.trim() == "realtime") { "Sec-WebSocket-Protocol: realtime\r\n".to_string() } else { String::new() })
        .unwrap_or_default();
    write!(c, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n{proto}\r\n", accept_key(&key))?;
    c.flush()
}

// ---- audio formats

/// How the session's audio is carried
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fmt {
    Pcm16(u32),
    F32(u32),
    Ulaw,
    Alaw,
}

impl Fmt {
    fn rate(self) -> u32 {
        match self {
            Fmt::Pcm16(r) | Fmt::F32(r) => r,
            _ => 8000,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Fmt::Pcm16(_) => "pcm16",
            Fmt::F32(_) => "pcm_f32le",
            Fmt::Ulaw => "g711_ulaw",
            Fmt::Alaw => "g711_alaw",
        }
    }
    /// A format named (beta: "pcm16"; GA: {"type": "audio/pcm", "rate": 24000}), at `rate` when it has one
    fn parse(v: &Value, rate: Option<u32>) -> Option<Fmt> {
        let name = v.as_str().or_else(|| v["type"].as_str())?;
        let rate = v["rate"].as_u64().map(|r| r as u32).or(rate).unwrap_or(24000);
        Some(match name {
            "pcm16" | "audio/pcm" | "pcm" => Fmt::Pcm16(rate),
            "pcm_f32le" | "f32" => Fmt::F32(rate),
            "g711_ulaw" | "audio/pcmu" => Fmt::Ulaw,
            "g711_alaw" | "audio/pcma" => Fmt::Alaw,
            _ => return None,
        })
    }
    fn decode(self, b: &[u8]) -> Vec<f32> {
        match self {
            Fmt::Pcm16(_) => b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
            Fmt::F32(_) => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Fmt::Ulaw => b.iter().map(|x| ulaw(*x)).collect(),
            Fmt::Alaw => b.iter().map(|x| alaw(*x)).collect(),
        }
    }
    fn encode(self, s: &[f32]) -> Vec<u8> {
        match self {
            Fmt::Pcm16(_) => nextsycl_audio::pcm16(s),
            Fmt::F32(_) => s.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Fmt::Ulaw => s.iter().map(|x| to_ulaw(*x)).collect(),
            Fmt::Alaw => s.iter().map(|x| to_alaw(*x)).collect(),
        }
    }
}

fn ulaw(u: u8) -> f32 {
    let u = !u;
    let t = (((u & 0x0f) as i32) << 3) + 0x84;
    let t = t << ((u & 0x70) >> 4);
    (if u & 0x80 != 0 { 0x84 - t } else { t - 0x84 }) as f32 / 32768.0
}

fn to_ulaw(x: f32) -> u8 {
    let s = (x.clamp(-1.0, 1.0) * 32767.0) as i32;
    let sign = if s < 0 { 0x80 } else { 0 };
    let mut m = s.abs().min(32635) + 0x84;
    let mut exp = 7;
    while exp > 0 && m & (0x4000 >> (7 - exp)) == 0 {
        exp -= 1;
    }
    m >>= exp + 3;
    !(sign | (exp << 4) as u8 | (m & 0x0f) as u8)
}

fn alaw(a: u8) -> f32 {
    let a = a ^ 0x55;
    let mut t = ((a & 0x0f) as i32) << 4;
    let seg = (a & 0x70) >> 4;
    match seg {
        0 => t += 8,
        1 => t += 0x108,
        _ => {
            t += 0x108;
            t <<= seg - 1;
        }
    }
    (if a & 0x80 != 0 { t } else { -t }) as f32 / 32768.0
}

fn to_alaw(x: f32) -> u8 {
    let s = (x.clamp(-1.0, 1.0) * 32767.0) as i32;
    let (mask, v) = if s >= 0 { (0xD5u8, s) } else { (0x55u8, -s - 1) };
    let v = v.min(32767);
    let seg = (0..8).find(|i| v <= (0xFF << i) | ((1 << i) - 1) && v < (0x100 << i)).unwrap_or(7);
    let aval = if seg < 2 { ((v >> 4) & 0x0f) as u8 } else { ((v >> (seg + 3)) & 0x0f) as u8 };
    ((seg as u8) << 4 | aval) ^ mask
}

// ---- the session

/// What a client asked for: the conversation's settings and how its audio is carried
struct Session {
    conf: RealtimeSession,
    input: Fmt,
    output: Fmt,
    ga: bool,
    modalities: Vec<String>,
    voice: Option<String>,
    raw: Value,
}

/// The server's events with their names in the client's flavour (beta / GA)
struct Names {
    ga: bool,
}

impl Names {
    fn audio_delta(&self) -> &'static str {
        if self.ga { "response.output_audio.delta" } else { "response.audio.delta" }
    }
    fn audio_done(&self) -> &'static str {
        if self.ga { "response.output_audio.done" } else { "response.audio.done" }
    }
    fn transcript_delta(&self, audio: bool) -> &'static str {
        match (self.ga, audio) {
            (true, true) => "response.output_audio_transcript.delta",
            (false, true) => "response.audio_transcript.delta",
            (true, false) => "response.output_text.delta",
            (false, false) => "response.text.delta",
        }
    }
    fn transcript_done(&self, audio: bool) -> &'static str {
        match (self.ga, audio) {
            (true, true) => "response.output_audio_transcript.done",
            (false, true) => "response.audio_transcript.done",
            (true, false) => "response.output_text.done",
            (false, false) => "response.text.done",
        }
    }
    fn item_created(&self) -> &'static str {
        if self.ga { "conversation.item.added" } else { "conversation.item.created" }
    }
}

/// The voice library's recording of a saved voice (for a model that clones)
pub type VoiceLookup<'a> = &'a dyn Fn(&str) -> Option<(Vec<f32>, u32)>;

/// Reads a session.update's `session` into the settings (beta and GA shapes, Sophon's extra_body, ours under nextsycl)
fn apply(s: &mut Session, v: &Value, voices: VoiceLookup, built_in: &[String]) -> Result<(), String> {
    let ga = v["type"] == "realtime" || v["audio"].is_object();
    s.ga |= ga;
    if let Some(i) = v["instructions"].as_str() {
        s.conf.instructions = Some(i.to_string()).filter(|i| !i.trim().is_empty());
    }
    let eb = &v["extra_body"];
    let ns = &v["nextsycl"];
    let rate = |x: &Value| x.as_u64().map(|r| r as u32);
    let in_rate = rate(&ns["input_sample_rate"]).or_else(|| rate(&v["input_sample_rate_hz"])).or_else(|| rate(&eb["sample_rate_hz"]));
    // a rate alone keeps the format and changes its rate
    let at = |f: Fmt, r: Option<u32>| match (f, r) {
        (Fmt::Pcm16(_), Some(r)) => Fmt::Pcm16(r),
        (Fmt::F32(_), Some(r)) => Fmt::F32(r),
        (f, _) => f,
    };
    s.input = Fmt::parse(&v["input_audio_format"], in_rate).or_else(|| Fmt::parse(&v["audio"]["input"]["format"], in_rate)).unwrap_or(at(s.input, in_rate));
    let out_rate = rate(&ns["output_sample_rate"]);
    s.output = Fmt::parse(&v["output_audio_format"], out_rate).or_else(|| Fmt::parse(&v["audio"]["output"]["format"], out_rate)).unwrap_or(at(s.output, out_rate));
    let td = if v.get("turn_detection").is_some() { Some(&v["turn_detection"]) } else if v["audio"]["input"].get("turn_detection").is_some() {
        Some(&v["audio"]["input"]["turn_detection"])
    } else {
        None
    };
    if let Some(td) = td {
        s.conf.turn_detection = !td.is_null() && td["type"].as_str() != Some("none");
        if let Some(ms) = td["silence_duration_ms"].as_u64() {
            // OpenAI's "silence ends the turn": here the most a pause may last before the model answers anyway
            s.conf.silence_s = Some((ms as f32 / 1000.0).max(0.3));
        }
    }
    if let Some(f) = eb["turn_fallback_s"].as_f64() {
        s.conf.silence_s = Some(f as f32);
    }
    if let Some(m) = v["modalities"].as_array().or_else(|| v["output_modalities"].as_array()) {
        s.modalities = m.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
    }
    if let Some(t) = v["temperature"].as_f64() {
        s.conf.temperature = Some(t as f32);
    }
    let mt = if v.get("max_response_output_tokens").is_some() { &v["max_response_output_tokens"] } else { &v["max_output_tokens"] };
    if let Some(n) = mt.as_u64() {
        s.conf.max_tokens = Some(n as u32);
    } else if mt.as_str() == Some("inf") {
        // OpenAI's "no limit": up to what the model's context has room for (None is the model's own default)
        s.conf.max_tokens = Some(u32::MAX);
    }
    let tools = if v["tools"].is_array() { &v["tools"] } else { &eb["tools"] };
    if let Some(t) = tools.as_array() {
        s.conf.tools = t.iter().map(|x| x.to_string()).collect();
    }
    if v["tool_choice"].as_str() == Some("none") {
        s.conf.tools.clear();
    }
    if let Some(l) = ns["language"].as_str().or_else(|| eb["language"].as_str()).or_else(|| v["language"].as_str()) {
        s.conf.language = Some(l.to_string());
    }
    if let Some(h) = eb["history"].as_array().or_else(|| ns["history"].as_array()) {
        s.conf.history = h.iter().filter_map(|m| Some((m["role"].as_str()?.to_string(), m["text"].as_str().or_else(|| m["content"].as_str())?.to_string())))
            .collect();
    }
    if let Some(o) = ns["options"].as_object() {
        for (k, x) in o {
            s.conf.extra.insert(k.clone(), x.as_str().map_or_else(|| x.to_string(), str::to_string));
        }
    }
    if let Some(seed) = ns["seed"].as_u64() {
        s.conf.seed = seed;
    }
    // a lockstep model's character card (Haru-4B): its fields as an object, or the card's text; null: the model's own
    if let Some(c) = ns.get("card") {
        s.conf.card = match c {
            Value::Null => None,
            Value::Object(_) | Value::String(_) => Some(c.to_string()),
            _ => return Err("nextsycl.card: an object of the card's fields (persona, name, gender, backchannel, other_name, relationship, situation) or \
                             its text".into()),
        };
    }
    let voice = v["voice"].as_str().or_else(|| v["audio"]["output"]["voice"].as_str()).or_else(|| ns["voice"].as_str());
    if let Some(name) = voice {
        let low = name.to_lowercase();
        if name.trim().is_empty() {
            // no voice named: the model's own (its default voice, or - a model without built-in ones - none cloned)
            s.conf.voice = built_in.first().cloned();
            s.conf.reference = None;
        } else if built_in.contains(&low) {
            s.conf.voice = Some(low);
            s.conf.reference = None;
        } else if let Some((samples, rate)) = voices(name) {
            s.conf.reference = Some(Reference { samples, rate, text: None });
            s.conf.voice = None;
        } else {
            return Err(format!("voice {name:?}: one of {} or a saved voice", built_in.join(", ")));
        }
        s.voice = if name.trim().is_empty() { built_in.first().cloned() } else { Some(name.to_string()) };
    }
    s.raw = v.clone();
    Ok(())
}

/// The session as session.created / session.updated tell it
fn describe(s: &Session, model: &str, rt: &nextsycl_audio::Realtime) -> Value {
    let td = if s.conf.turn_detection { json!({"type": "server_vad", "silence_duration_ms": (s.conf.silence_s.unwrap_or(2.0) * 1000.0) as u64}) } else { Value::Null };
    let mut v = if s.ga {
        json!({"type": "realtime", "object": "realtime.session", "model": model, "output_modalities": s.modalities,
               "audio": {"input": {"format": {"type": "audio/pcm", "rate": s.input.rate()}, "turn_detection": td},
                         "output": {"format": {"type": "audio/pcm", "rate": s.output.rate()}, "voice": s.voice}}})
    } else {
        json!({"object": "realtime.session", "model": model, "modalities": s.modalities, "voice": s.voice,
               "input_audio_format": s.input.name(), "output_audio_format": s.output.name(), "turn_detection": td})
    };
    v["instructions"] = json!(s.conf.instructions);
    v["tools"] = json!(s.conf.tools.iter().filter_map(|t| serde_json::from_str::<Value>(t).ok()).collect::<Vec<_>>());
    v["temperature"] = json!(s.conf.temperature);
    v["nextsycl"] = json!({"input_rate": rt.input_rate, "output_rate": rt.output_rate, "frame_ms": rt.frame_ms, "lockstep": rt.frame_ms > 0,
                           "voices": rt.voices, "clone": rt.clone, "languages": rt.languages, "tools": rt.tools,
                           "input_sample_rate": s.input.rate(), "output_sample_rate": s.output.rate(), "language": s.conf.language,
                           "card": s.conf.card.as_deref().and_then(|c| serde_json::from_str::<Value>(c).ok())});
    v
}

/// A live conversation on `c` (the request was its upgrade) with `engine`, until either side closes. `busy` is set
/// while it runs (the server's progress shows it; an idle server does not end under it)
pub fn serve(mut c: Conn, req: &Request, engine: &dyn AudioEngine, model: &str, voices: VoiceLookup) -> Result<(), String> {
    let rt = engine.realtime().ok_or_else(|| format!("{model} does not hold live conversations (/v1/realtime needs a full-duplex speech model)"))?;
    upgrade(&mut c, req).map_err(|e| e.to_string())?;
    let out = WsOut(Arc::new(Mutex::new(c.try_clone().map_err(|e| e.to_string())?)));
    let ga = http::header(&req.head, "openai-beta").is_none() && req.target.contains("flavor=ga");
    let mut s = Session { conf: RealtimeSession { turn_detection: true, ..Default::default() }, input: Fmt::Pcm16(24000), output: Fmt::Pcm16(24000), ga,
                          modalities: vec!["audio".into(), "text".into()], voice: rt.voices.first().cloned(), raw: Value::Null };
    s.conf.voice = s.voice.clone();
    let conv: Arc<dyn Conversation> = match engine.converse(&s.conf) {
        Ok(cv) => Arc::from(cv),
        Err(e) => {
            let _ = out.text(&json!({"type": "error", "error": {"type": "server_error", "code": "no_conversation", "message": e.0}}));
            out.close();
            return Err(e.0);
        }
    };
    let ev_n = std::sync::atomic::AtomicU64::new(0);
    let eid = || format!("event_{:06}", ev_n.fetch_add(1, Ordering::Relaxed));
    let _ = out.text(&json!({"type": "session.created", "event_id": eid(), "session": describe(&s, model, &rt)}));
    let names = Arc::new(Mutex::new(Names { ga: s.ga }));
    let output = Arc::new(Mutex::new((s.output, s.modalities.iter().any(|m| m == "audio"))));
    let done = Arc::new(AtomicBool::new(false));
    // the server's events, as the conversation makes them
    let writer = {
        let (conv, out, names, output, done, rate) = (conv.clone(), out.clone(), names.clone(), output.clone(), done.clone(), rt.output_rate);
        std::thread::spawn(move || {
            let mut n = 0u64;
            let mut eid = || {
                n += 1;
                format!("event_s{n:06}")
            };
            let mut prev_item: Option<String> = None;
            let mut user_item = String::new();
            let mut text = String::new();
            while !done.load(Ordering::SeqCst) {
                let Some(e) = conv.next(Duration::from_millis(100)) else { continue };
                let nm = names.lock().unwrap_or_else(|x| x.into_inner());
                let (fmt, audio) = *output.lock().unwrap_or_else(|x| x.into_inner());
                let msgs: Vec<Value> = match e {
                    RealtimeEvent::SpeechStarted { audio_ms } => {
                        user_item = format!("item_u{audio_ms}");
                        vec![json!({"type": "input_audio_buffer.speech_started", "event_id": eid(), "audio_start_ms": audio_ms, "item_id": user_item})]
                    }
                    RealtimeEvent::SpeechStopped { audio_ms } => vec![json!({"type": "input_audio_buffer.speech_stopped", "event_id": eid(),
                                                                              "audio_end_ms": audio_ms, "item_id": user_item})],
                    RealtimeEvent::Committed { item } => {
                        let id = if user_item.is_empty() { item } else { std::mem::take(&mut user_item) };
                        let v = vec![json!({"type": "input_audio_buffer.committed", "event_id": eid(), "previous_item_id": prev_item, "item_id": id}),
                                     json!({"type": nm.item_created(), "event_id": eid(), "previous_item_id": prev_item,
                                            "item": {"id": id, "object": "realtime.item", "type": "message", "status": "completed", "role": "user",
                                                     "content": [{"type": "input_audio", "transcript": null}]}})];
                        prev_item = Some(id);
                        v
                    }
                    RealtimeEvent::ResponseStarted { id } => {
                        text.clear();
                        let item = format!("msg_{id}");
                        let part = if audio { json!({"type": "audio", "transcript": ""}) } else { json!({"type": "text", "text": ""}) };
                        vec![json!({"type": "response.created", "event_id": eid(), "response": {"id": id, "object": "realtime.response", "status": "in_progress",
                                                                                                "output": []}}),
                             json!({"type": "response.output_item.added", "event_id": eid(), "response_id": id, "output_index": 0,
                                    "item": {"id": item, "object": "realtime.item", "type": "message", "status": "in_progress", "role": "assistant", "content": []}}),
                             json!({"type": nm.item_created(), "event_id": eid(), "previous_item_id": prev_item,
                                    "item": {"id": item, "object": "realtime.item", "type": "message", "status": "in_progress", "role": "assistant", "content": []}}),
                             json!({"type": "response.content_part.added", "event_id": eid(), "response_id": id, "item_id": item, "output_index": 0,
                                    "content_index": 0, "part": part})]
                    }
                    RealtimeEvent::TextDelta { id, delta } => {
                        text.push_str(&delta);
                        vec![json!({"type": nm.transcript_delta(audio), "event_id": eid(), "response_id": id, "item_id": format!("msg_{id}"), "output_index": 0,
                                    "content_index": 0, "delta": delta})]
                    }
                    RealtimeEvent::AudioDelta { id, samples } => {
                        if !audio {
                            continue;
                        }
                        let s = if fmt.rate() == rate { samples } else { crate::realtime::resample(&samples, rate, fmt.rate()) };
                        vec![json!({"type": nm.audio_delta(), "event_id": eid(), "response_id": id, "item_id": format!("msg_{id}"), "output_index": 0,
                                    "content_index": 0, "delta": http::base64(&fmt.encode(&s))})]
                    }
                    RealtimeEvent::ToolCall { id, call_id, name, arguments } => {
                        let item = format!("fc_{call_id}");
                        let it = json!({"id": item, "object": "realtime.item", "type": "function_call", "status": "completed", "name": name, "call_id": call_id,
                                        "arguments": arguments});
                        vec![json!({"type": "response.output_item.added", "event_id": eid(), "response_id": id, "output_index": 1, "item": it}),
                             json!({"type": "response.function_call_arguments.delta", "event_id": eid(), "response_id": id, "item_id": item, "output_index": 1,
                                    "call_id": call_id, "delta": arguments}),
                             json!({"type": "response.function_call_arguments.done", "event_id": eid(), "response_id": id, "item_id": item, "output_index": 1,
                                    "call_id": call_id, "name": name, "arguments": arguments}),
                             json!({"type": "response.output_item.done", "event_id": eid(), "response_id": id, "output_index": 1, "item": it})]
                    }
                    RealtimeEvent::ResponseDone { id, status, text: whole, audio_ms } => {
                        let item = format!("msg_{id}");
                        let t = if whole.is_empty() { text.clone() } else { whole };
                        let part = if audio { json!({"type": "audio", "transcript": t}) } else { json!({"type": "text", "text": t}) };
                        let msg = json!({"id": item, "object": "realtime.item", "type": "message", "status": if status == "completed" { "completed" } else { "incomplete" },
                                         "role": "assistant", "content": [part]});
                        prev_item = Some(item.clone());
                        let mut v = Vec::new();
                        if audio {
                            v.push(json!({"type": nm.audio_done(), "event_id": eid(), "response_id": id, "item_id": item, "output_index": 0, "content_index": 0}));
                        }
                        let tk = if audio { "transcript" } else { "text" };
                        let mut td = json!({"type": nm.transcript_done(audio), "event_id": eid(), "response_id": id, "item_id": item, "output_index": 0,
                                            "content_index": 0});
                        td[tk] = json!(t);
                        v.push(td);
                        v.push(json!({"type": "response.content_part.done", "event_id": eid(), "response_id": id, "item_id": item, "output_index": 0,
                                      "content_index": 0, "part": part}));
                        v.push(json!({"type": "response.output_item.done", "event_id": eid(), "response_id": id, "output_index": 0, "item": msg}));
                        let st = match status.as_str() {
                            "completed" => "completed",
                            "cancelled" => "cancelled",
                            _ => "failed",
                        };
                        v.push(json!({"type": "response.done", "event_id": eid(), "response": {"id": id, "object": "realtime.response", "status": st,
                                                                                            "output": [msg], "nextsycl": {"audio_ms": audio_ms}}}));
                        v
                    }
                    RealtimeEvent::Error { message } => vec![json!({"type": "error", "event_id": eid(), "error": {"type": "server_error", "message": message}})],
                };
                drop(nm);
                for m in msgs {
                    if out.text(&m).is_err() {
                        done.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
        })
    };
    // the client's events
    let built_in = rt.voices.clone();
    let mut r = c;
    let result = loop {
        let m = match read_msg(&mut r, &out) {
            Ok(Some(m)) => m,
            Ok(None) => break Ok(()),
            Err(e) => break Err(e.to_string()),
        };
        let v: Value = match m {
            Msg::Text(t) => match serde_json::from_str(&t) {
                Ok(v) => v,
                Err(e) => {
                    let _ = out.text(&json!({"type": "error", "event_id": eid(), "error": {"type": "invalid_request_error", "code": "invalid_json", "message": e.to_string()}}));
                    continue;
                }
            },
            // a binary message: the user's audio in the session's format (an extension: no base64)
            Msg::Binary(b) => {
                let f = s.input;
                let _ = conv.append(&f.decode(&b), f.rate());
                continue;
            }
        };
        let ty = v["type"].as_str().unwrap_or("");
        let client_id = v["event_id"].clone();
        let fail = |code: &str, m: String| {
            let _ = out.text(&json!({"type": "error", "event_id": eid(), "error": {"type": "invalid_request_error", "code": code, "message": m, "event_id": client_id}}));
        };
        match ty {
            "session.update" => {
                let lookup = |n: &str| voices(n);
                match apply(&mut s, &v["session"], &lookup, &built_in) {
                    Ok(()) => {
                        if let Err(e) = conv.update(&s.conf) {
                            fail("session_update_failed", e.0);
                            continue;
                        }
                        names.lock().unwrap_or_else(|x| x.into_inner()).ga = s.ga;
                        *output.lock().unwrap_or_else(|x| x.into_inner()) = (s.output, s.modalities.iter().any(|m| m == "audio"));
                        let _ = out.text(&json!({"type": "session.updated", "event_id": eid(), "session": describe(&s, model, &rt)}));
                    }
                    Err(e) => fail("invalid_session", e),
                }
            }
            "input_audio_buffer.append" => {
                let f = match (v["format"].as_str(), v["sample_rate_hz"].as_u64()) {
                    (Some(n), r) => Fmt::parse(&json!(n), r.map(|r| r as u32)).unwrap_or(s.input),
                    (None, _) => s.input,
                };
                match http::unbase64(v["audio"].as_str().unwrap_or("")) {
                    Ok(b) => {
                        if let Err(e) = conv.append(&f.decode(&b), f.rate()) {
                            fail("append_failed", e.0);
                        }
                    }
                    Err(e) => fail("invalid_audio", e),
                }
            }
            "input_audio_buffer.commit" => {
                if let Err(e) = conv.commit() {
                    fail("commit_failed", e.0);
                }
            }
            "input_audio_buffer.clear" => {
                conv.clear();
                let _ = out.text(&json!({"type": "input_audio_buffer.cleared", "event_id": eid()}));
            }
            "response.create" => {
                if let Err(e) = conv.respond() {
                    fail("response_failed", e.0);
                }
            }
            "response.cancel" => conv.cancel(),
            "conversation.item.create" => {
                let it = &v["item"];
                match it["type"].as_str() {
                    Some("function_call_output") => {
                        let (call, o) = (it["call_id"].as_str().unwrap_or(""), it["output"].as_str().map_or_else(|| it["output"].to_string(), str::to_string));
                        match conv.tool_output(call, &o) {
                            Ok(()) => {
                                let nm = names.lock().unwrap_or_else(|x| x.into_inner());
                                let id = it["id"].as_str().map_or_else(|| format!("fco_{call}"), str::to_string);
                                let _ = out.text(&json!({"type": nm.item_created(), "event_id": eid(), "item": {"id": id, "object": "realtime.item",
                                                         "type": "function_call_output", "call_id": call, "output": o}}));
                            }
                            Err(e) => fail("invalid_call", e.0),
                        }
                    }
                    Some("message") | None => {
                        let role = it["role"].as_str().unwrap_or("user");
                        let t: String = it["content"].as_array().into_iter().flatten()
                            .filter_map(|p| p["text"].as_str().or_else(|| p["transcript"].as_str())).collect::<Vec<_>>().join(" ");
                        match conv.add_text(role, &t) {
                            Ok(()) => {
                                let nm = names.lock().unwrap_or_else(|x| x.into_inner());
                                let id = it["id"].as_str().map_or_else(eid, str::to_string);
                                let _ = out.text(&json!({"type": nm.item_created(), "event_id": eid(), "item": {"id": id, "object": "realtime.item",
                                                         "type": "message", "role": role, "content": it["content"]}}));
                            }
                            Err(e) => fail("invalid_item", e.0),
                        }
                    }
                    Some(t) => fail("unsupported_item", format!("item type {t:?}: message or function_call_output")),
                }
            }
            "conversation.item.truncate" => {
                let item = v["item_id"].as_str().unwrap_or("");
                let ms = v["audio_end_ms"].as_u64().unwrap_or(0);
                conv.truncate(item.trim_start_matches("msg_"), ms);
                let _ = out.text(&json!({"type": "conversation.item.truncated", "event_id": eid(), "item_id": item, "content_index": v["content_index"],
                                         "audio_end_ms": ms}));
            }
            "conversation.item.delete" | "conversation.item.retrieve" | "output_audio_buffer.clear" => {}
            "session.close" => break Ok(()),
            t => fail("unknown_event", format!("event type {t:?} is not one this server takes")),
        }
    };
    done.store(true, Ordering::SeqCst);
    conv.close();
    let _ = writer.join();
    out.close();
    result
}

/// Linear-interpolated resampling of an answer's audio to the client's rate (the model's own is 24 kHz; a client at
/// another takes this)
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64) as usize;
    (0..n).map(|i| {
        let t = i as f64 * from as f64 / to as f64;
        let j = t.floor() as usize;
        let f = (t - j as f64) as f32;
        let a = x[j.min(x.len() - 1)];
        let b = x[(j + 1).min(x.len() - 1)];
        a + (b - a) * f
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_handshake_key_is_rfc_6455s() {
        assert_eq!(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
        assert_eq!(http::base64(&sha1(b"abc")), http::base64(&[0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50, 0xc2, 0x6c, 0x9c,
                                                                0xd0, 0xd8, 0x9d]));
    }

    #[test]
    fn g711_round_trips() {
        for x in [-0.9f32, -0.3, -0.01, 0.0, 0.02, 0.25, 0.8] {
            assert!((ulaw(to_ulaw(x)) - x).abs() < 0.03, "ulaw {x}: {}", ulaw(to_ulaw(x)));
            assert!((alaw(to_alaw(x)) - x).abs() < 0.03, "alaw {x}: {}", alaw(to_alaw(x)));
        }
    }

    use std::collections::VecDeque;
    use std::net::{TcpListener, TcpStream};

    use nextsycl_audio::{Audio, AudioRequest, Defaults, Realtime, Result as AResult, Step};

    /// A conversation that answers every commit with "hi" and 0.1 s of a tone, and tells what it was given
    struct MockConv {
        q: Mutex<VecDeque<RealtimeEvent>>,
        heard: Mutex<(usize, u32)>,
    }

    impl Conversation for MockConv {
        fn update(&self, _s: &RealtimeSession) -> AResult<()> {
            Ok(())
        }
        fn append(&self, samples: &[f32], rate: u32) -> AResult<()> {
            let mut h = self.heard.lock().unwrap();
            h.0 += samples.len();
            h.1 = rate;
            Ok(())
        }
        fn commit(&self) -> AResult<()> {
            let mut q = self.q.lock().unwrap();
            q.push_back(RealtimeEvent::Committed { item: "item_1".into() });
            q.push_back(RealtimeEvent::ResponseStarted { id: "resp_1".into() });
            q.push_back(RealtimeEvent::TextDelta { id: "resp_1".into(), delta: "hi".into() });
            q.push_back(RealtimeEvent::AudioDelta { id: "resp_1".into(), samples: (0..2400).map(|i| (i as f32 * 0.05).sin() * 0.5).collect() });
            q.push_back(RealtimeEvent::ResponseDone { id: "resp_1".into(), status: "completed".into(), text: "hi".into(), audio_ms: 100 });
            Ok(())
        }
        fn clear(&self) {}
        fn respond(&self) -> AResult<()> {
            self.commit()
        }
        fn cancel(&self) {}
        fn truncate(&self, _r: &str, _ms: u64) {}
        fn tool_output(&self, _c: &str, _o: &str) -> AResult<()> {
            Ok(())
        }
        fn add_text(&self, _r: &str, _t: &str) -> AResult<()> {
            Ok(())
        }
        fn next(&self, wait: Duration) -> Option<RealtimeEvent> {
            let e = self.q.lock().unwrap().pop_front();
            if e.is_none() {
                std::thread::sleep(wait.min(Duration::from_millis(10)));
            }
            e
        }
        fn close(&self) {}
    }

    struct MockEngine(Arc<MockConv>);

    impl AudioEngine for MockEngine {
        fn arch(&self) -> &'static str {
            "mock"
        }
        fn defaults(&self) -> Defaults {
            Defaults { seconds: 1.0, max_seconds: 1.0, steps: 0, cfg: 1.0, rate: 24000 }
        }
        fn load_seconds(&self) -> f64 {
            0.0
        }
        fn generate(&self, _r: &AudioRequest, _p: &mut dyn FnMut(Step) -> AResult<()>) -> AResult<Audio> {
            Ok(Audio::default())
        }
        fn realtime(&self) -> Option<Realtime> {
            Some(Realtime { input_rate: 16000, output_rate: 24000, voices: vec!["chelsie".into()], ..Default::default() })
        }
        fn converse(&self, _s: &RealtimeSession) -> AResult<Box<dyn Conversation>> {
            struct Fwd(Arc<MockConv>);
            impl Conversation for Fwd {
                fn update(&self, s: &RealtimeSession) -> AResult<()> { self.0.update(s) }
                fn append(&self, x: &[f32], r: u32) -> AResult<()> { self.0.append(x, r) }
                fn commit(&self) -> AResult<()> { self.0.commit() }
                fn clear(&self) { self.0.clear() }
                fn respond(&self) -> AResult<()> { self.0.respond() }
                fn cancel(&self) { self.0.cancel() }
                fn truncate(&self, r: &str, ms: u64) { self.0.truncate(r, ms) }
                fn tool_output(&self, c: &str, o: &str) -> AResult<()> { self.0.tool_output(c, o) }
                fn add_text(&self, r: &str, t: &str) -> AResult<()> { self.0.add_text(r, t) }
                fn next(&self, w: Duration) -> Option<RealtimeEvent> { self.0.next(w) }
                fn close(&self) { self.0.close() }
            }
            Ok(Box::new(Fwd(self.0.clone())))
        }
    }

    fn send_masked(s: &mut TcpStream, text: &str) {
        let d = text.as_bytes();
        let mut h = vec![0x81u8];
        if d.len() < 126 {
            h.push(0x80 | d.len() as u8);
        } else {
            h.push(0x80 | 126);
            h.extend_from_slice(&(d.len() as u16).to_be_bytes());
        }
        let mask = [7u8, 1, 9, 3];
        h.extend_from_slice(&mask);
        h.extend(d.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        s.write_all(&h).unwrap();
    }

    fn read_text(s: &mut TcpStream) -> Option<Value> {
        let mut h = [0u8; 2];
        s.read_exact(&mut h).ok()?;
        let mut n = (h[1] & 0x7f) as usize;
        if n == 126 {
            let mut b = [0u8; 2];
            s.read_exact(&mut b).ok()?;
            n = u16::from_be_bytes(b) as usize;
        } else if n == 127 {
            let mut b = [0u8; 8];
            s.read_exact(&mut b).ok()?;
            n = u64::from_be_bytes(b) as usize;
        }
        let mut p = vec![0u8; n];
        s.read_exact(&mut p).ok()?;
        if h[0] & 0x0f == 0x8 {
            return None;
        }
        serde_json::from_slice(&p).ok()
    }

    /// The socket end to end on a mock model: the handshake, OpenAI's session and response events in order, the audio
    /// in at the client's rate and out as 16-bit PCM
    #[test]
    fn a_conversation_speaks_openai_realtime() {
        let conv = Arc::new(MockConv { q: Mutex::new(VecDeque::new()), heard: Mutex::new((0, 0)) });
        let engine = MockEngine(conv.clone());
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let srv = std::thread::spawn(move || {
            let (c, _) = l.accept().unwrap();
            let mut conn = Conn::Tcp(c);
            let req = http::read_request(&mut conn).unwrap();
            assert!(http::is_upgrade(&req.head));
            serve(conn, &req, &engine, "mock", &|_| None).unwrap();
        });
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        write!(s, "GET /v1/realtime HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n").unwrap();
        let mut head = Vec::new();
        let mut b = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            s.read_exact(&mut b).unwrap();
            head.push(b[0]);
        }
        let head = String::from_utf8_lossy(&head);
        assert!(head.starts_with("HTTP/1.1 101") && head.contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="), "{head}");
        let first = read_text(&mut s).unwrap();
        assert_eq!(first["type"], "session.created");
        assert_eq!(first["session"]["nextsycl"]["output_rate"], 24000);
        send_masked(&mut s, &json!({"type": "session.update", "session": {"voice": "chelsie", "nextsycl": {"input_sample_rate": 16000}}}).to_string());
        let pcm: Vec<u8> = (0..1600i16).flat_map(|i| (i * 7).to_le_bytes()).collect();
        send_masked(&mut s, &json!({"type": "input_audio_buffer.append", "audio": http::base64(&pcm)}).to_string());
        send_masked(&mut s, &json!({"type": "input_audio_buffer.commit"}).to_string());
        let mut types = Vec::new();
        let mut audio_bytes = 0;
        while let Some(v) = read_text(&mut s) {
            let t = v["type"].as_str().unwrap_or("").to_string();
            if t == "response.audio.delta" {
                audio_bytes += http::unbase64(v["delta"].as_str().unwrap()).unwrap().len();
            }
            if t == "response.done" {
                assert_eq!(v["response"]["status"], "completed");
                types.push(t);
                break;
            }
            types.push(t);
        }
        assert_eq!(*conv.heard.lock().unwrap(), (1600, 16000), "the audio in, at the client's rate");
        assert_eq!(audio_bytes, 4800, "0.1 s at 24 kHz, 16-bit");
        for want in ["session.updated", "input_audio_buffer.committed", "conversation.item.created", "response.created", "response.output_item.added",
                     "response.content_part.added", "response.audio_transcript.delta", "response.audio.delta", "response.audio.done",
                     "response.audio_transcript.done", "response.content_part.done", "response.output_item.done", "response.done"] {
            assert!(types.iter().any(|t| t == want), "no {want} in {types:?}");
        }
        assert!(types.iter().position(|t| t == "response.created") < types.iter().position(|t| t == "response.audio.delta"));
        send_masked(&mut s, &json!({"type": "session.close"}).to_string());
        srv.join().unwrap();
    }

    #[test]
    fn sessions_read_both_shapes() {
        let none = |_: &str| None;
        let voices = vec!["chelsie".to_string(), "brand".to_string()];
        let mut s = Session { conf: RealtimeSession { turn_detection: true, ..Default::default() }, input: Fmt::Pcm16(24000), output: Fmt::Pcm16(24000), ga: false,
                              modalities: vec![], voice: None, raw: Value::Null };
        apply(&mut s, &json!({"voice": "brand", "input_audio_format": "pcm_f32le", "extra_body": {"language": "zh", "sample_rate_hz": 16000},
                              "turn_detection": null, "tools": [{"type": "function", "name": "w"}]}), &none, &voices).unwrap();
        assert_eq!((s.conf.voice.as_deref(), s.input, s.conf.language.as_deref(), s.conf.turn_detection, s.conf.tools.len(), s.ga),
                   (Some("brand"), Fmt::F32(16000), Some("zh"), false, 1, false));
        apply(&mut s, &json!({"type": "realtime", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": {"type": "server_vad"}},
                                                             "output": {"voice": "chelsie"}}}), &none, &voices).unwrap();
        assert_eq!((s.conf.voice.as_deref(), s.input, s.conf.turn_detection, s.ga), (Some("chelsie"), Fmt::Pcm16(24000), true, true));
        assert!(apply(&mut s, &json!({"voice": "nobody"}), &none, &voices).is_err());
        let saved = |n: &str| (n == "alice").then(|| (vec![0.0; 16000], 16000));
        apply(&mut s, &json!({"voice": "alice"}), &saved, &voices).unwrap();
        assert!(s.conf.reference.is_some() && s.conf.voice.is_none());
        // an empty voice: the model's own (its first built-in one; none for a model without)
        apply(&mut s, &json!({"voice": ""}), &saved, &voices).unwrap();
        assert!(s.conf.reference.is_none() && s.conf.voice.as_deref() == Some("chelsie"));
        apply(&mut s, &json!({"voice": ""}), &saved, &[]).unwrap();
        assert!(s.conf.reference.is_none() && s.conf.voice.is_none());
        // a lockstep model's card: an object kept as JSON, null back to the model's own, anything else refused
        apply(&mut s, &json!({"nextsycl": {"card": {"name": "Kai", "persona": "A pirate."}}}), &none, &voices).unwrap();
        let c: Value = serde_json::from_str(s.conf.card.as_deref().unwrap()).unwrap();
        assert_eq!(c["name"], "Kai");
        apply(&mut s, &json!({"nextsycl": {"card": null}}), &none, &voices).unwrap();
        assert!(s.conf.card.is_none());
        assert!(apply(&mut s, &json!({"nextsycl": {"card": 3}}), &none, &voices).is_err());
    }

    /// A lockstep model as Haru-4B is one: every whole frame in answered by a frame out within one long response
    struct LockConv {
        q: Mutex<VecDeque<RealtimeEvent>>,
        buf: Mutex<(usize, bool)>,
        card: Mutex<Option<String>>,
    }

    impl Conversation for LockConv {
        fn update(&self, s: &RealtimeSession) -> AResult<()> {
            *self.card.lock().unwrap() = s.card.clone();
            Ok(())
        }
        fn append(&self, samples: &[f32], _rate: u32) -> AResult<()> {
            let mut b = self.buf.lock().unwrap();
            let mut q = self.q.lock().unwrap();
            if !b.1 {
                b.1 = true;
                q.push_back(RealtimeEvent::ResponseStarted { id: "resp_l".into() });
            }
            b.0 += samples.len();
            while b.0 >= 3200 {
                b.0 -= 3200;
                q.push_back(RealtimeEvent::TextDelta { id: "resp_l".into(), delta: "mm ".into() });
                q.push_back(RealtimeEvent::AudioDelta { id: "resp_l".into(), samples: vec![0.25; 3200] });
            }
            Ok(())
        }
        fn commit(&self) -> AResult<()> {
            Ok(())
        }
        fn clear(&self) {}
        fn respond(&self) -> AResult<()> {
            Ok(())
        }
        fn cancel(&self) {}
        fn truncate(&self, _r: &str, _ms: u64) {}
        fn tool_output(&self, _c: &str, _o: &str) -> AResult<()> {
            Ok(())
        }
        fn add_text(&self, _r: &str, _t: &str) -> AResult<()> {
            Ok(())
        }
        fn next(&self, wait: Duration) -> Option<RealtimeEvent> {
            let e = self.q.lock().unwrap().pop_front();
            if e.is_none() {
                std::thread::sleep(wait.min(Duration::from_millis(10)));
            }
            e
        }
        fn close(&self) {}
    }

    struct LockEngine(Arc<LockConv>);

    impl AudioEngine for LockEngine {
        fn arch(&self) -> &'static str {
            "lock"
        }
        fn defaults(&self) -> Defaults {
            Defaults { seconds: 1.0, max_seconds: 1.0, steps: 0, cfg: 1.0, rate: 24000 }
        }
        fn load_seconds(&self) -> f64 {
            0.0
        }
        fn generate(&self, _r: &AudioRequest, _p: &mut dyn FnMut(Step) -> AResult<()>) -> AResult<Audio> {
            Ok(Audio::default())
        }
        fn realtime(&self) -> Option<Realtime> {
            Some(Realtime { input_rate: 24000, output_rate: 24000, frame_ms: 133, clone: true, ..Default::default() })
        }
        fn converse(&self, _s: &RealtimeSession) -> AResult<Box<dyn Conversation>> {
            struct Fwd(Arc<LockConv>);
            impl Conversation for Fwd {
                fn update(&self, s: &RealtimeSession) -> AResult<()> { self.0.update(s) }
                fn append(&self, x: &[f32], r: u32) -> AResult<()> { self.0.append(x, r) }
                fn commit(&self) -> AResult<()> { self.0.commit() }
                fn clear(&self) { self.0.clear() }
                fn respond(&self) -> AResult<()> { self.0.respond() }
                fn cancel(&self) { self.0.cancel() }
                fn truncate(&self, r: &str, ms: u64) { self.0.truncate(r, ms) }
                fn tool_output(&self, c: &str, o: &str) -> AResult<()> { self.0.tool_output(c, o) }
                fn add_text(&self, r: &str, t: &str) -> AResult<()> { self.0.add_text(r, t) }
                fn next(&self, w: Duration) -> Option<RealtimeEvent> { self.0.next(w) }
                fn close(&self) { self.0.close() }
            }
            Ok(Box::new(Fwd(self.0.clone())))
        }
    }

    /// The lockstep extension end to end: session.created says lockstep and the frame, the card reaches the model, and
    /// three frames of audio in (in two appends of any size) come back as one response's three audio deltas of a frame
    /// each - no turn events, no response.done while the conversation runs
    #[test]
    fn a_lockstep_model_answers_every_frame() {
        let conv = Arc::new(LockConv { q: Mutex::new(VecDeque::new()), buf: Mutex::new((0, false)), card: Mutex::new(None) });
        let engine = LockEngine(conv.clone());
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let srv = std::thread::spawn(move || {
            let (c, _) = l.accept().unwrap();
            let mut conn = Conn::Tcp(c);
            let req = http::read_request(&mut conn).unwrap();
            serve(conn, &req, &engine, "lock", &|_| None).unwrap();
        });
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        write!(s, "GET /v1/realtime HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n").unwrap();
        let mut head = Vec::new();
        let mut b = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            s.read_exact(&mut b).unwrap();
            head.push(b[0]);
        }
        let first = read_text(&mut s).unwrap();
        assert_eq!((first["session"]["nextsycl"]["lockstep"].as_bool(), first["session"]["nextsycl"]["frame_ms"].as_u64()), (Some(true), Some(133)));
        send_masked(&mut s, &json!({"type": "session.update", "session": {"nextsycl": {"card": {"name": "Kai"}}}}).to_string());
        assert_eq!(read_text(&mut s).unwrap()["type"], "session.updated");
        assert!(conv.card.lock().unwrap().as_deref().is_some_and(|c| c.contains("Kai")));
        for n in [4000i16, 5600] {
            let pcm: Vec<u8> = (0..n).flat_map(|i| (i % 300).to_le_bytes()).collect();
            send_masked(&mut s, &json!({"type": "input_audio_buffer.append", "audio": http::base64(&pcm)}).to_string());
        }
        let (mut types, mut deltas) = (Vec::new(), Vec::new());
        while deltas.len() < 3 {
            let v = read_text(&mut s).unwrap();
            let t = v["type"].as_str().unwrap_or("").to_string();
            if t == "response.audio.delta" {
                deltas.push(http::unbase64(v["delta"].as_str().unwrap()).unwrap().len());
            }
            types.push(t);
        }
        assert_eq!(deltas, vec![6400, 6400, 6400], "a frame of 16-bit audio for each frame in");
        assert_eq!(types.iter().filter(|t| *t == "response.created").count(), 1, "{types:?}");
        assert!(!types.iter().any(|t| t == "response.done" || t == "input_audio_buffer.committed"), "{types:?}");
        send_masked(&mut s, &json!({"type": "session.close"}).to_string());
        srv.join().unwrap();
    }
}
