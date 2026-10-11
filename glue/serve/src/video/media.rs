//! ffmpeg for what runs on the host (the studio's thumbnails, `video join`): the host's when it reads and writes
//! H.264, else the video daemon's container's, which sees the clips at the same paths. Fedora's ffmpeg-free cannot
//! (its openh264 wrapper fails to start, and it has no libx264), and the clips are H.264 - every thumbnail failed.

use std::process::Command;
use std::sync::OnceLock;

/// The video daemon's container (`nextsycl video start`)
pub const DAEMON: &str = "nextsycl-video";

/// Whether this host's ffmpeg decodes H.264 and encodes it with libx264
pub fn host_h264() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let list = |what: &str| Command::new("ffmpeg").args(["-hide_banner", what]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        list("-decoders").contains(" h264 ") && list("-encoders").contains("libx264")
    })
}

/// `ffmpeg` (or `ffprobe`) that handles the clips: the host's, else run in the daemon's container
pub fn tool(name: &str) -> Command {
    if host_h264() {
        return Command::new(name);
    }
    let engine = std::env::var("NS_CONTAINER_ENGINE").unwrap_or_else(|_| "podman".into());
    let mut c = Command::new(engine);
    c.args(["exec", "-i", DAEMON, name]);
    c
}
