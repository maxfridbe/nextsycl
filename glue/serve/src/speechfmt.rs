//! The sound formats of `/v1/audio/speech` (OpenAI's `response_format`: mp3, opus, aac, flac, wav, pcm) and its `speed`.
//! The engines make 16-bit PCM; anything else - another format, another rate (OpenAI's pcm is 24 kHz), another speed -
//! goes through one ffmpeg child (`-f s16le` in, the format out) that is fed as the sound is made and read as it
//! encodes, so a streamed answer starts at the first sentence whatever its format.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};

/// OpenAI's pcm: 16-bit little-endian mono at this rate
pub const PCM_RATE: u32 = 24_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fmt {
    Wav,
    Pcm,
    Mp3,
    Opus,
    Aac,
    Flac,
}

impl Fmt {
    pub fn parse(name: &str) -> Option<Fmt> {
        Some(match name {
            "wav" => Fmt::Wav,
            "pcm" => Fmt::Pcm,
            "mp3" => Fmt::Mp3,
            "opus" => Fmt::Opus,
            "aac" => Fmt::Aac,
            "flac" => Fmt::Flac,
            _ => return None,
        })
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Fmt::Wav => "audio/wav",
            Fmt::Pcm => "audio/pcm",
            Fmt::Mp3 => "audio/mpeg",
            Fmt::Opus => "audio/ogg",
            Fmt::Aac => "audio/aac",
            Fmt::Flac => "audio/flac",
        }
    }

    /// ffmpeg's output options for it
    fn out_args(self) -> &'static [&'static str] {
        match self {
            Fmt::Wav => &["-f", "wav"],
            Fmt::Pcm => &["-f", "s16le"],
            Fmt::Mp3 => &["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
            Fmt::Opus => &["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
            Fmt::Aac => &["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
            Fmt::Flac => &["-c:a", "flac", "-f", "flac"],
        }
    }
}

/// ffmpeg's atempo takes 0.5 to 2 a stage: `speed` (0.25 - 4, OpenAI's range) as a chain of them
pub fn atempo(speed: f64) -> String {
    let (mut s, mut stages) = (speed, Vec::new());
    while s > 2.0 {
        stages.push(2.0);
        s /= 2.0;
    }
    while s < 0.5 {
        stages.push(0.5);
        s /= 0.5;
    }
    stages.push(s);
    stages.iter().map(|x| format!("atempo={x:.4}")).collect::<Vec<_>>().join(",")
}

/// `speed` from a request: 1 when absent, an error outside OpenAI's 0.25 - 4
pub fn speed_of(b: &serde_json::Value) -> Result<f64, String> {
    match &b["speed"] {
        serde_json::Value::Null => Ok(1.0),
        v => match v.as_f64() {
            Some(s) if (0.25..=4.0).contains(&s) => Ok(s),
            _ => Err("speed: a number from 0.25 to 4".into()),
        },
    }
}

/// Whether ffmpeg is on this machine's path
pub fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// An ffmpeg child between the engine's PCM and the client: `feed` what the engine makes, `ready` what has come out
pub struct Pipe {
    child: Child,
    stdin: Option<ChildStdin>,
    out: Receiver<Vec<u8>>,
}

impl Pipe {
    /// `rate` Hz mono 16-bit in; `fmt` out at `to` Hz (none: the input's), `speed` times as fast
    pub fn start(rate: u32, fmt: Fmt, to: Option<u32>, speed: f64) -> Result<Pipe, String> {
        let mut c = Command::new("ffmpeg");
        c.args(["-v", "error", "-f", "s16le", "-ar", &rate.to_string(), "-ac", "1", "-i", "-"]);
        if (speed - 1.0).abs() > 1e-6 {
            c.args(["-af", &atempo(speed)]);
        }
        if let Some(to) = to.filter(|t| *t != rate) {
            c.args(["-ar", &to.to_string()]);
        }
        c.args(fmt.out_args()).arg("-");
        let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()
            .map_err(|e| format!("{} needs ffmpeg on the server's path ({e})", if fmt == Fmt::Pcm { "speed or a rate" } else { "this format" }))?;
        let stdin = child.stdin.take();
        let mut stdout = child.stdout.take().ok_or("ffmpeg's output")?;
        let (tx, out) = channel();
        // read as it encodes: the pipe never fills while the engine is still writing
        std::thread::spawn(move || {
            let mut buf = vec![0u8; 16 << 10];
            while let Ok(n) = stdout.read(&mut buf) {
                if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                    break;
                }
            }
        });
        Ok(Pipe { child, stdin, out })
    }

    pub fn feed(&mut self, pcm: &[u8]) -> std::io::Result<()> {
        match self.stdin.as_mut() {
            Some(s) => s.write_all(pcm),
            None => Err(std::io::ErrorKind::BrokenPipe.into()),
        }
    }

    /// What has been encoded so far (no waiting)
    pub fn ready(&mut self) -> Vec<Vec<u8>> {
        self.out.try_iter().collect()
    }

    /// The input ended: everything left, waiting for ffmpeg to finish
    pub fn finish(mut self) -> Vec<Vec<u8>> {
        drop(self.stdin.take());
        let rest: Vec<Vec<u8>> = self.out.iter().collect();
        let _ = self.child.wait();
        rest
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        // a client that left: ffmpeg is not left running
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A chunk of a chunked answer (`Transfer-Encoding: chunked`)
pub fn chunk(w: &mut impl Write, b: &[u8]) -> std::io::Result<()> {
    if b.is_empty() {
        return Ok(());
    }
    write!(w, "{:x}\r\n", b.len())?;
    w.write_all(b)?;
    w.write_all(b"\r\n")
}

/// The last chunk
pub fn chunk_end(w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(b"0\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atempo_stays_in_ffmpegs_range() {
        assert_eq!(atempo(1.5), "atempo=1.5000");
        assert_eq!(atempo(4.0), "atempo=2.0000,atempo=2.0000");
        assert_eq!(atempo(0.25), "atempo=0.5000,atempo=0.5000");
        assert_eq!(atempo(3.0), "atempo=2.0000,atempo=1.5000");
    }

    #[test]
    fn speed_is_openais_range() {
        let ok = |s: serde_json::Value| speed_of(&serde_json::json!({"speed": s}));
        assert_eq!(speed_of(&serde_json::json!({})), Ok(1.0));
        assert_eq!(ok(serde_json::json!(0.25)), Ok(0.25));
        assert!(ok(serde_json::json!(4.5)).is_err());
        assert!(ok(serde_json::json!("fast")).is_err());
    }

    #[test]
    fn formats_have_names_and_types() {
        for (n, t) in [("mp3", "audio/mpeg"), ("opus", "audio/ogg"), ("aac", "audio/aac"), ("flac", "audio/flac"), ("pcm", "audio/pcm"), ("wav", "audio/wav")] {
            assert_eq!(Fmt::parse(n).map(Fmt::content_type), Some(t));
        }
        assert!(Fmt::parse("ogg").is_none());
    }

    #[test]
    fn chunks_are_framed() {
        let mut v = Vec::new();
        chunk(&mut v, b"hello").unwrap();
        chunk(&mut v, b"").unwrap();
        chunk_end(&mut v).unwrap();
        assert_eq!(v, b"5\r\nhello\r\n0\r\n\r\n");
    }

    #[test]
    fn a_pipe_resamples_and_encodes() {
        if !have_ffmpeg() {
            return;
        }
        // one second of 48 kHz silence -> 24 kHz pcm: half the bytes
        let mut p = Pipe::start(48_000, Fmt::Pcm, Some(PCM_RATE), 1.0).unwrap();
        p.feed(&vec![0u8; 48_000 * 2]).unwrap();
        let n: usize = p.finish().iter().map(Vec::len).sum();
        assert!((n as i64 - 48_000).abs() < 400, "{n}");
        // twice as fast: half the length again
        let mut p = Pipe::start(24_000, Fmt::Pcm, None, 2.0).unwrap();
        p.feed(&vec![0u8; 24_000 * 2]).unwrap();
        let n: usize = p.finish().iter().map(Vec::len).sum();
        assert!((n as i64 - 24_000).abs() < 2000, "{n}");
        let mut p = Pipe::start(24_000, Fmt::Mp3, None, 1.0).unwrap();
        p.feed(&vec![0u8; 24_000 * 2]).unwrap();
        let out: Vec<u8> = p.finish().concat();
        assert!(out.len() > 100 && (out.starts_with(b"ID3") || out[0] == 0xFF), "{:?}", &out[..4.min(out.len())]);
    }
}
