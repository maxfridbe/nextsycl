//! The GPU telemetry sampler (`nextsycl gpustat`, run as root by a service): an Intel Arc card on the xe driver,
//! written every few seconds to a JSON file (default /run/gpustat.json) that pages read for their GPU pill and
//! monitor - the video studio's /api/gpu among them.
//!
//! What it reports: the name; VRAM used (the sum of `drm-resident-vram0` over the card's DRM clients in
//! /proc/*/fdinfo - the only VRAM accounting xe exposes, root-only) and total (the card's largest PCI BAR, which maps
//! all of VRAM with resizable BAR); busy % (the clients' compute/render cycle deltas); temperatures, power (the
//! energy counters' rise over a window: they update in bursts, so a 3 s delta swings 0.6 <-> 48 W), the power cap,
//! fan, frequency; the PCIe link the card trained at (the first port above the GPU function wider than x1 - the
//! card's own functions sit behind an internal x1 switch) and what the card and the slot could do; the host's load.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// Names by PCI device id (lspci's database is the fallback)
const NAMES: &[(&str, &str)] = &[("e223", "Intel(R) Arc(TM) Pro B70 Graphics"), ("e222", "Intel(R) Arc(TM) Pro B65 Graphics")];

pub struct Options {
    pub out: PathBuf,
    pub interval: Duration,
    /// the power average's window
    pub power_window: Duration,
    /// the card (PCI address); none: the first Intel GPU on xe
    pub pci: Option<String>,
    /// VRAM in MiB; none: the largest BAR
    pub vram_mb: Option<u64>,
    /// one sample (the first has no busy % or power yet), printed instead of written
    pub once: bool,
}

fn rd(p: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn num(p: impl AsRef<Path>) -> Option<i64> {
    rd(p)?.parse().ok()
}

/// The first Intel GPU on the xe driver (it has tile*/gt*), as a PCI address
pub fn find_card() -> Option<String> {
    let mut cards: Vec<PathBuf> = std::fs::read_dir("/sys/class/drm").ok()?.flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit())))
        .collect();
    cards.sort();
    cards.into_iter().map(|c| c.join("device")).find(|d| {
        rd(d.join("vendor")).as_deref() == Some("0x8086")
            && std::fs::read_dir(d).into_iter().flatten().flatten().any(|e| {
                e.file_name().to_string_lossy().starts_with("tile") && std::fs::read_dir(e.path()).into_iter().flatten().flatten()
                    .any(|g| g.file_name().to_string_lossy().starts_with("gt"))
            })
    }).and_then(|d| std::fs::canonicalize(d).ok()).and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
}

/// The largest BAR in MiB
fn bar_mb(pci: &str) -> u64 {
    rd(format!("/sys/bus/pci/devices/{pci}/resource")).map_or(0, |r| {
        r.lines().filter_map(|l| {
            let mut it = l.split_whitespace();
            let a = u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
            let b = u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
            (b > a).then(|| b - a + 1)
        }).max().unwrap_or(0) >> 20
    })
}

fn card_name(pci: &str, dev: &str) -> String {
    if let Some((_, n)) = NAMES.iter().find(|(d, _)| *d == dev) {
        return n.to_string();
    }
    let out = std::process::Command::new("lspci").args(["-vmm", "-s", pci]).output().ok();
    out.and_then(|o| String::from_utf8_lossy(&o.stdout).lines().find_map(|l| l.strip_prefix("Device:").map(|n| n.trim().to_string())))
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| format!("Intel GPU 8086:{dev}"))
}

/// The xe driver's hwmon directory of the card
fn hwmon(pci: &str) -> Option<PathBuf> {
    let card = std::fs::read_dir(format!("/sys/bus/pci/devices/{pci}/hwmon")).ok()?.flatten().next().map(|e| e.path());
    card.or_else(|| {
        std::fs::read_dir("/sys/class/hwmon").ok()?.flatten().map(|e| e.path()).find(|p| rd(p.join("name")).as_deref() == Some("xe"))
    })
}

fn temps(h: &Path) -> HashMap<String, f64> {
    std::fs::read_dir(h).into_iter().flatten().flatten().filter_map(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        let base = n.strip_suffix("_label")?;
        let v = num(h.join(format!("{base}_input")))?;
        Some((rd(e.path())?, v as f64 / 1000.0))
    }).collect()
}

const GEN: &[(&str, u32)] = &[("2.5", 1), ("5.0", 2), ("8.0", 3), ("16.0", 4), ("32.0", 5), ("64.0", 6)];

fn link(p: &Path, kind: &str) -> Value {
    let sp = rd(p.join(format!("{kind}_link_speed"))).unwrap_or_default().split(' ').next().unwrap_or("").to_string();
    json!({"gen": GEN.iter().find(|g| g.0 == sp).map(|g| g.1), "gts": sp, "width": num(p.join(format!("{kind}_link_width"))).unwrap_or(0)})
}

/// The link the card trained at: up from the GPU function to the first port wider than x1; the root port's maximum
fn pcie(pci: &str) -> Value {
    let Ok(mut d) = std::fs::canonicalize(format!("/sys/bus/pci/devices/{pci}")) else { return Value::Null };
    let mut chain = Vec::new();
    // (a string test: Path::starts_with compares whole components, and the root's is "pci0000:00")
    while d.to_string_lossy().starts_with("/sys/devices/pci") && d.join("current_link_speed").exists() {
        chain.push(d.clone());
        if !d.pop() {
            break;
        }
    }
    for p in &chain {
        let cur = link(p, "current");
        if cur["width"].as_i64().unwrap_or(0) > 1 {
            return json!({"cur": cur, "card_max": link(p, "max"), "slot_max": link(chain.last().expect("non-empty"), "max")});
        }
    }
    Value::Null
}

/// A DRM client: its VRAM and cycle counters
#[derive(Clone, Copy, Default)]
struct Client {
    vram_kb: u64,
    cycles: u64,
    total: u64,
}

/// The card's DRM clients across all processes, by client id (each counted once)
fn clients(pci: &str) -> HashMap<String, Client> {
    let mut res = HashMap::new();
    for p in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let pid = p.file_name();
        if !pid.to_string_lossy().chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        for fd in std::fs::read_dir(p.path().join("fd")).into_iter().flatten().flatten() {
            if !std::fs::read_link(fd.path()).is_ok_and(|t| t.starts_with("/dev/dri/")) {
                continue;
            }
            let Some(info) = rd(p.path().join("fdinfo").join(fd.file_name())) else { continue };
            let kv: HashMap<&str, &str> = info.lines().filter_map(|l| l.split_once(":\t")).filter(|(k, _)| k.starts_with("drm-")).collect();
            let Some(cid) = kv.get("drm-client-id") else { continue };
            // with two cards every process's fdinfo lists both: this card's only
            if kv.get("drm-pdev").is_some_and(|d| d.trim() != pci) || res.contains_key(*cid) {
                continue;
            }
            let n = |k: &str| kv.get(k).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            res.insert(cid.to_string(), Client {
                vram_kb: n("drm-resident-vram0"),
                cycles: n("drm-cycles-rcs").max(n("drm-cycles-ccs")),
                total: n("drm-total-cycles-rcs").max(n("drm-total-cycles-ccs")),
            });
        }
    }
    res
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Samples until the process ends (or once)
pub fn run(o: Options) -> Result<(), String> {
    let pci = o.pci.clone().or_else(find_card).ok_or("no Intel GPU on the xe driver (--pci ADDR)")?;
    let dev = rd(format!("/sys/bus/pci/devices/{pci}/device")).unwrap_or_default().trim_start_matches("0x").to_string();
    let name = card_name(&pci, &dev);
    let total_mb = o.vram_mb.unwrap_or_else(|| bar_mb(&pci));
    let h = hwmon(&pci);
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut prev: HashMap<String, Client> = HashMap::new();
    let mut ring: VecDeque<(Instant, i64, i64)> = VecDeque::new();
    let (mut link, mut link_t) = (pcie(&pci), Instant::now());
    loop {
        let now = Instant::now();
        let cl = clients(&pci);
        let busy: f64 = cl.iter().filter_map(|(id, c)| {
            let p = prev.get(id)?;
            (c.total > p.total).then(|| (c.cycles.saturating_sub(p.cycles)) as f64 / (c.total - p.total) as f64 * 100.0)
        }).fold(0.0, |a, b| a + b); // (an empty f64 sum is -0.0)
        let energy = |i: u32| h.as_ref().and_then(|h| num(h.join(format!("energy{i}_input")))).unwrap_or(0);
        let (e, e2) = (energy(1), energy(2));
        ring.push_back((now, e, e2));
        while ring.len() > 2 && now - ring[1].0 >= o.power_window {
            ring.pop_front();
        }
        let (t0, ea, eb) = ring[0];
        let span = (now - t0).as_secs_f64();
        let watts = |a: i64, b: i64| (span >= 2.5 && b >= a).then(|| round1((b - a) as f64 / 1e6 / span));
        if now - link_t > Duration::from_secs(30) {
            // the link can retrain (power saving, errors)
            (link, link_t) = (pcie(&pci), now);
        }
        let t = h.as_deref().map(temps).unwrap_or_default();
        let vram_max = t.iter().filter(|(k, _)| k.starts_with("vram_ch_")).map(|(_, v)| *v).reduce(f64::max);
        let cap = h.as_ref().and_then(|h| num(h.join("power1_cap"))).filter(|c| *c > 0).map(|c| (c as f64 / 1e6).round() as i64);
        let freq = num(format!("/sys/bus/pci/devices/{pci}/tile0/gt0/freq0/act_freq"));
        let load1 = rd("/proc/loadavg").and_then(|l| l.split_whitespace().next()?.parse::<f64>().ok()).unwrap_or(0.0);
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
        let out = json!({
            "name": name, "pci": pci, "ts": ts,
            "vram_used_mb": cl.values().map(|c| c.vram_kb).sum::<u64>() / 1024, "vram_total_mb": total_mb,
            "busy_pct": round1(busy.min(100.0)),
            "temp_pkg": t.get("pkg"), "temp_vram": t.get("vram"),
            "power_w": watts(ea, e), "pkg_power_w": watts(eb, e2), "power_cap_w": cap,
            "temp_vram_max": vram_max, "temp_pcie": t.get("pcie"), "temp_mctrl": t.get("mctrl"),
            "pcie": link,
            "host_load1": load1, "host_cpus": cpus,
            // the card's cumulative energy (hwmon energy1, "card", microjoules)
            "energy_j": round1(e as f64 / 1e6),
            "fan_rpm": h.as_ref().map(|h| num(h.join("fan1_input")).unwrap_or(0)),
            "freq_mhz": freq,
        });
        if o.once {
            println!("{out}");
            return Ok(());
        }
        let tmp = o.out.with_extension("json.tmp");
        std::fs::write(&tmp, out.to_string()).and_then(|_| {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))?;
            std::fs::rename(&tmp, &o.out)
        }).map_err(|e| format!("{}: {e}", o.out.display()))?;
        prev = cl;
        std::thread::sleep(o.interval);
    }
}
