//! The GPU run lock: GPU work this box's services do not manage - a test run, a benchmark, a build - holds a lock
//! file (`flock`, NS_GPU_LOCK, default /tmp/ds2run.lock) while it lasts, and while it is held the cards are taken:
//! the front door and the switcher load no model (requests are refused with the retryable code `cards_taken`, jobs
//! wait in their queues, the video queue pauses). Loading beside such a run oversubscribes VRAM, and on the xe
//! driver that spills into host memory and livelocks the box (2026-10-18 15:11, 2026-10-19 07:24).
//!
//! The lock is read, never taken: /proc/locks lists every flock with its holder's pid and the file's device and inode
//! (a waiter's line starts with `->` and holds nothing), so a run waiting for the lock never waits on this.

use std::path::{Path, PathBuf};

/// The lock file runs hold (NS_GPU_LOCK; empty or "off": none)
pub fn path(setting: Option<String>) -> Option<PathBuf> {
    match setting {
        Some(s) if s.trim().is_empty() || s.trim() == "off" => None,
        Some(s) => Some(PathBuf::from(s.trim())),
        None => Some(PathBuf::from("/tmp/ds2run.lock")),
    }
}

/// A run holding the lock: its pid and what it runs
#[derive(Clone, Debug, PartialEq)]
pub struct Holder {
    pub pid: i64,
    pub cmd: String,
}

/// (major, minor) of a device number, as Linux packs it
fn dev_parts(dev: u64) -> (u64, u64) {
    (((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff), (dev & 0xff) | ((dev >> 12) & !0xff))
}

/// The holders in a /proc/locks text of the file with this device and inode (waiters left out)
pub fn holders_in(locks: &str, dev: u64, ino: u64) -> Vec<i64> {
    let (maj, min) = dev_parts(dev);
    locks.lines().filter(|l| !l.contains("->")).filter_map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        // "N: FLOCK ADVISORY WRITE PID MAJ:MIN:INODE START END"
        let at = f.iter().position(|t| t.matches(':').count() == 2)?;
        let mut p = f[at].split(':');
        let (a, b, i) = (p.next()?, p.next()?, p.next()?);
        let ok = u64::from_str_radix(a, 16).ok()? == maj && u64::from_str_radix(b, 16).ok()? == min && i.parse::<u64>().ok()? == ino;
        ok.then(|| f.get(at.checked_sub(1)?)?.parse::<i64>().ok()).flatten()
    }).collect()
}

/// Who holds the lock file now (none: free, no file, or no lock file set)
pub fn held(file: &Path) -> Option<Holder> {
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::metadata(file).ok()?;
    let locks = std::fs::read_to_string("/proc/locks").ok()?;
    let pid = *holders_in(&locks, md.dev(), md.ino()).first()?;
    Some(Holder { pid, cmd: cmdline(pid) })
}

/// What a process runs (its children's command when it is `flock` itself: the run it holds the lock for)
fn cmdline(pid: i64) -> String {
    let read = |p: i64| std::fs::read(format!("/proc/{p}/cmdline")).ok()
        .map(|b| b.split(|c| *c == 0).filter(|a| !a.is_empty()).map(|a| String::from_utf8_lossy(a).into_owned()).collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    let me = read(pid);
    let child = std::fs::read_to_string(format!("/proc/{pid}/task/{pid}/children")).ok()
        .and_then(|c| c.split_whitespace().next().and_then(|c| c.parse::<i64>().ok()));
    let cmd = match child {
        Some(c) if me.starts_with("flock") => read(c),
        _ => me,
    };
    cmd.chars().take(120).collect()
}

/// The sentence a refused request or a waiting job reads
pub fn message(file: &Path, h: &Holder) -> String {
    let what = if h.cmd.is_empty() { String::new() } else { format!(": {}", h.cmd) };
    format!("cards taken: a GPU run holds {} (pid {}{what}); no model loads beside it - try again when it ends", file.display(), h.pid)
}

/// Seconds a refused caller is told to wait before trying again
pub const RETRY_AFTER_S: u64 = 120;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_holder_and_not_a_waiter() {
        // /tmp/ds2run.lock on hardac1: st_dev 45 (00:2d), inode 293
        let locks = "1: POSIX  ADVISORY  WRITE 2508 00:1d:4223 0 0\n\
                     2: FLOCK  ADVISORY  WRITE 7781 00:2d:293 0 EOF\n\
                     2: -> FLOCK  ADVISORY  WRITE 7790 00:2d:293 0 EOF\n\
                     3: FLOCK  ADVISORY  WRITE 99 00:2d:294 0 EOF\n";
        assert_eq!(holders_in(locks, 45, 293), vec![7781]);
        assert!(holders_in(locks, 45, 295).is_empty());
        assert!(holders_in("2: -> FLOCK  ADVISORY  WRITE 7790 00:2d:293 0 EOF\n", 45, 293).is_empty(), "a waiter holds nothing");
        // a block device's (fd:01 = 253:1)
        assert_eq!(holders_in("4: FLOCK  ADVISORY  WRITE 5 fd:01:77 0 EOF\n", (253 << 8) | 1, 77), vec![5]);
    }

    #[test]
    fn the_setting() {
        assert_eq!(path(None), Some(PathBuf::from("/tmp/ds2run.lock")));
        assert_eq!(path(Some("off".into())), None);
        assert_eq!(path(Some("".into())), None);
        assert_eq!(path(Some("/run/x.lock".into())), Some(PathBuf::from("/run/x.lock")));
    }

    #[test]
    fn a_real_flock_is_seen() {
        // `flock` (util-linux) holds a lock file while a child sleeps; skipped where there is none
        let f = std::env::temp_dir().join(format!("ns-gpulock-{}", std::process::id()));
        std::fs::write(&f, "").unwrap();
        assert!(held(&f).is_none());
        let Ok(mut c) = std::process::Command::new("flock").arg(&f).args(["sleep", "2"]).spawn() else { return };
        let t0 = std::time::Instant::now();
        let mut h = None;
        while h.is_none() && t0.elapsed().as_secs() < 3 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            h = held(&f);
        }
        let _ = c.wait();
        let h = h.expect("the lock is seen while flock holds it");
        assert_eq!(h.pid, c.id() as i64);
        assert!(message(&f, &h).starts_with("cards taken:"));
        let _ = std::fs::remove_file(f);
    }
}
