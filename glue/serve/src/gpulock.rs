//! The GPU run lock: GPU work this box's services do not manage - a test run, a benchmark, a build - holds a lock
//! file (`flock`, NS_GPU_LOCK, default /tmp/ds2run.lock) while it lasts, and while it is held the cards are taken:
//! the front door and the switcher load no model (requests are refused with the retryable code `cards_taken`, jobs
//! wait in their queues, the video queue pauses). Loading beside such a run oversubscribes VRAM, and on the xe
//! driver that spills into host memory and livelocks the box (2026-10-18 15:11, 2026-10-19 07:24).
//!
//! The lock is read, never taken: /proc/locks lists every flock with its holder's pid and the file's device and inode
//! (a waiter's line starts with `->` and holds nothing), so a run waiting for the lock never waits on this.
//!
//! A run may say which cards it uses: once it holds the lock it writes `{"pid": P, "gpus": [1]}` to the side file
//! beside the lock (`<lock>.cards`: /tmp/ds2run.cards), P the holder's pid as /proc/locks shows it. Only those cards are
//! taken then: nothing loads there (nor a model that spans them, nor a chat model spread over them) and the other cards
//! serve as always. The side file counts only while a holder with that pid holds the lock, so a run that ends - or
//! crashes - leaves no card blocked: the lock goes with its process, and the file left behind names a pid that holds
//! nothing. A holder with no side file, or one naming another pid or no card, takes every card (the old runners only
//! flock). The video queue still pauses while any card is taken.
//!
//! Since 2026-11-17 a run on some cards holds each card's own lock instead (`card_file`: /tmp/ds2run.gpu1.lock), so runs
//! on different cards go side by side; the box's lock is still every card for a run that names none.

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

/// The side file in which a run holding `lock` names its cards: `ds2run.lock` -> `ds2run.cards`
pub fn cards_file(lock: &Path) -> PathBuf {
    lock.with_extension("cards")
}

/// The cards a side file names for the holder `pid` (`{"pid": P, "gpus": [..]}`); None - every card - when it is not
/// that holder's, names no card, or is not one
pub fn declared(text: &str, pid: i64) -> Option<Vec<usize>> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    if v["pid"].as_i64()? != pid {
        return None;
    }
    let mut g: Vec<usize> = v["gpus"].as_array()?.iter().map(|g| g.as_u64().map(|g| g as usize)).collect::<Option<_>>()?;
    g.sort_unstable();
    g.dedup();
    (!g.is_empty()).then_some(g)
}

/// The lock held: by whom, and the cards it takes (None: every card)
#[derive(Clone, Debug, PartialEq)]
pub struct Taken {
    pub holder: Holder,
    pub gpus: Option<Vec<usize>>,
}

impl Taken {
    /// Whether card `g` is taken
    pub fn takes(&self, g: usize) -> bool {
        self.gpus.as_ref().is_none_or(|v| v.contains(&g))
    }

    /// Whether a load on `gpus` may not start: one of them taken (`all_needed`: a model whose one copy spans them, a
    /// chat model spread over them), else every one of them (a model that may go on any of them goes on a free one).
    /// No cards given: barred
    pub fn bars(&self, gpus: &[usize], all_needed: bool) -> bool {
        if gpus.is_empty() || self.gpus.is_none() {
            return true;
        }
        if all_needed { gpus.iter().any(|g| self.takes(*g)) } else { gpus.iter().all(|g| self.takes(*g)) }
    }

    /// The taken cards among `cards` (all of them when the run named none)
    pub fn among(&self, cards: &[usize]) -> Vec<usize> {
        cards.iter().copied().filter(|g| self.takes(*g)).collect()
    }
}

/// A card's own lock file beside the box's: `ds2run.lock` -> `ds2run.gpu1.lock`. A run that uses only some cards holds
/// theirs (gpu-lock.sh since 2026-11-17), so two runs on two cards go side by side - the box's one lock let only one
/// run at a time, whatever cards it used
pub fn card_file(lock: &Path, g: usize) -> PathBuf {
    let stem = lock.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "ds2run".into());
    lock.with_file_name(format!("{stem}.gpu{g}.lock"))
}

/// Who holds the lock now and which cards are taken (none: free, no file, or no lock file set): the box's lock (its
/// side file naming the cards, else every card) and each card's own lock (that card)
pub fn taken(file: &Path) -> Option<Taken> {
    let main = held(file).map(|holder| {
        let gpus = std::fs::read_to_string(cards_file(file)).ok().and_then(|t| declared(&t, holder.pid));
        Taken { holder, gpus }
    });
    let mut cards = Vec::new();
    let mut first = None;
    for g in 0..16 {
        if let Some(h) = held(&card_file(file, g)) {
            cards.push(g);
            first.get_or_insert(h);
        }
    }
    match (main, first) {
        (Some(mut t), Some(_)) => {
            if let Some(v) = t.gpus.as_mut() {
                v.extend(cards);
                v.sort_unstable();
                v.dedup();
            }
            Some(t)
        }
        (Some(t), None) => Some(t),
        (None, Some(holder)) => Some(Taken { holder, gpus: Some(cards) }),
        (None, None) => None,
    }
}

/// The sentence a refused request or a waiting job reads
pub fn message(file: &Path, t: &Taken) -> String {
    let h = &t.holder;
    let what = if h.cmd.is_empty() { String::new() } else { format!(": {}", h.cmd) };
    match &t.gpus {
        None => format!("cards taken: a GPU run holds {} (pid {}{what}); no model loads beside it - try again when it ends", file.display(), h.pid),
        Some(g) => {
            let on = g.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(", ");
            format!("cards taken: a GPU run holds {} (pid {}{what}) on GPU {on}; no model loads there beside it - try again when it ends",
                    file.display(), h.pid)
        }
    }
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
        assert!(message(&f, &Taken { holder: h, gpus: None }).starts_with("cards taken:"));
        let _ = std::fs::remove_file(f);
    }

    #[test]
    fn a_side_file_counts_for_its_holder_only() {
        assert_eq!(cards_file(Path::new("/tmp/ds2run.lock")), PathBuf::from("/tmp/ds2run.cards"));
        assert_eq!(declared(r#"{"pid": 7, "gpus": [1]}"#, 7), Some(vec![1]));
        assert_eq!(declared(r#"{"pid": 7, "gpus": [1, 0, 1]}"#, 7), Some(vec![0, 1]));
        // a stale file (another run's, ended or crashed), no card, or nonsense: every card
        assert_eq!(declared(r#"{"pid": 6, "gpus": [1]}"#, 7), None);
        assert_eq!(declared(r#"{"pid": 7, "gpus": []}"#, 7), None);
        assert_eq!(declared(r#"{"pid": 7, "gpus": ["1"]}"#, 7), None);
        assert_eq!(declared(r#"{"pid": 7}"#, 7), None);
        assert_eq!(declared("1", 7), None);
    }

    #[test]
    fn only_the_named_cards_are_taken() {
        let h = Holder { pid: 7, cmd: "regen2k test".into() };
        let one = Taken { holder: h.clone(), gpus: Some(vec![1]) };
        assert!(one.takes(1) && !one.takes(0));
        assert!(!one.bars(&[0], false), "a model on the free card loads");
        assert!(!one.bars(&[0, 1], false), "a model that may go on either goes on the free one");
        assert!(one.bars(&[1], false));
        assert!(one.bars(&[0, 1], true), "a model that spans both needs both");
        assert!(one.bars(&[], false), "no cards given: barred");
        assert_eq!(one.among(&[0, 1]), vec![1]);
        let m = message(Path::new("/tmp/ds2run.lock"), &one);
        assert!(m.starts_with("cards taken:") && m.contains("on GPU 1;"), "{m}");
        let all = Taken { holder: h, gpus: None };
        assert!(all.bars(&[0], false) && all.bars(&[0, 1], false));
        assert_eq!(all.among(&[0, 1]), vec![0, 1]);
    }

    #[test]
    fn a_real_holder_with_a_side_file() {
        // the side file is read for the pid /proc/locks shows; skipped where there is no `flock`
        let f = std::env::temp_dir().join(format!("ns-gpulock-side-{}.lock", std::process::id()));
        std::fs::write(&f, "").unwrap();
        let Ok(mut c) = std::process::Command::new("flock").arg(&f).args(["sleep", "2"]).spawn() else { return };
        std::fs::write(cards_file(&f), format!(r#"{{"pid": {}, "gpus": [1]}}"#, c.id())).unwrap();
        let t0 = std::time::Instant::now();
        let mut t = None;
        while t.is_none() && t0.elapsed().as_secs() < 3 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            t = taken(&f);
        }
        let _ = c.wait();
        assert_eq!(t.expect("held").gpus, Some(vec![1]));
        assert!(taken(&f).is_none(), "the run ended: the file left behind takes nothing");
        let _ = std::fs::remove_file(cards_file(&f));
        let _ = std::fs::remove_file(f);
    }

    #[test]
    fn a_cards_own_lock_takes_that_card_only() {
        // two runs on two cards hold ds2run.gpu0.lock / ds2run.gpu1.lock side by side; skipped where there is no `flock`
        let f = std::env::temp_dir().join(format!("ns-gpulock-card-{}.lock", std::process::id()));
        std::fs::write(&f, "").unwrap();
        let c1 = card_file(&f, 1);
        assert!(c1.to_string_lossy().ends_with(&format!("ns-gpulock-card-{}.gpu1.lock", std::process::id())));
        std::fs::write(&c1, "").unwrap();
        let Ok(mut c) = std::process::Command::new("flock").arg(&c1).args(["sleep", "2"]).spawn() else { return };
        let t0 = std::time::Instant::now();
        let mut t = None;
        while t.is_none() && t0.elapsed().as_secs() < 3 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            t = taken(&f);
        }
        let _ = c.wait();
        let t = t.expect("held");
        assert_eq!(t.gpus, Some(vec![1]));
        assert!(!t.takes(0) && t.takes(1), "card 0 serves on");
        assert!(taken(&f).is_none());
        let _ = std::fs::remove_file(c1);
        let _ = std::fs::remove_file(f);
    }
}
