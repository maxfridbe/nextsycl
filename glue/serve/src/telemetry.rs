//! The GPUs' power and temperature, from the xe driver's sensors (sysfs hwmon of each card's PCI device): the
//! card's energy counter (power is its rise over a second; a request's energy, its rise over the request) and the
//! package and VRAM temperatures. A card without them reads as None.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Card {
    energy: PathBuf,
    temp: Option<PathBuf>,
    vram: Option<PathBuf>,
}

impl Card {
    /// The sensors of PCI device `pci`: the hwmon entries labelled "card" (energy), "pkg" and "vram" (temperatures)
    fn find(pci: &str) -> Option<Card> {
        let dir = std::fs::read_dir(Path::new("/sys/bus/pci/devices").join(pci).join("hwmon")).ok()?.flatten().next()?.path();
        let labelled = |kind: &str, label: &str| -> Option<PathBuf> {
            (1..32).find_map(|i| {
                let l = std::fs::read_to_string(dir.join(format!("{kind}{i}_label"))).ok()?;
                (l.trim() == label).then(|| dir.join(format!("{kind}{i}_input")))
            })
        };
        Some(Card { energy: labelled("energy", "card")?, temp: labelled("temp", "pkg"), vram: labelled("temp", "vram") })
    }
    fn joules(&self) -> Option<f64> {
        Some(std::fs::read_to_string(&self.energy).ok()?.trim().parse::<f64>().ok()? * 1e-6) // microjoules
    }
}

fn celsius(p: &Option<PathBuf>) -> Option<f64> {
    Some(std::fs::read_to_string(p.as_ref()?).ok()?.trim().parse::<f64>().ok()? * 1e-3) // millidegrees
}

/// A card's latest reading
#[derive(Clone, Copy, Default)]
pub struct Reading {
    pub watts: Option<f64>,
    pub temp_c: Option<f64>,
    pub vram_c: Option<f64>,
}

pub struct Telemetry {
    cards: Vec<Option<Card>>,
    live: Mutex<Vec<Reading>>,
}

impl Telemetry {
    /// The cards at these PCI addresses (None: no address), sampled once a second from now
    pub fn start(pcis: &[Option<String>]) -> Arc<Telemetry> {
        let t = Arc::new(Telemetry { cards: pcis.iter().map(|p| p.as_deref().and_then(Card::find)).collect(),
                                     live: Mutex::new(vec![Reading::default(); pcis.len()]) });
        let me = t.clone();
        std::thread::spawn(move || {
            let mut prev: Vec<Option<(f64, Instant)>> = me.cards.iter().map(|_| None).collect();
            loop {
                let now: Vec<Reading> = me.cards.iter().zip(prev.iter_mut()).map(|(c, pv)| {
                    let Some(c) = c else { return Reading::default() };
                    let j = c.joules().map(|j| (j, Instant::now()));
                    let watts = match (*pv, j) {
                        (Some((j0, t0)), Some((j1, t1))) if j1 >= j0 => Some((j1 - j0) / (t1 - t0).as_secs_f64().max(1e-3)),
                        _ => None,
                    };
                    *pv = j;
                    Reading { watts, temp_c: celsius(&c.temp), vram_c: celsius(&c.vram) }
                }).collect();
                *me.live.lock().unwrap() = now;
                std::thread::sleep(Duration::from_secs(1));
            }
        });
        t
    }

    pub fn readings(&self) -> Vec<Reading> {
        self.live.lock().unwrap().clone()
    }

    /// The cards' energy counters summed, now (joules): a request's energy is the rise of this
    pub fn joules(&self) -> Option<f64> {
        let v: Vec<f64> = self.cards.iter().flatten().filter_map(Card::joules).collect();
        (!v.is_empty()).then(|| v.iter().sum())
    }
}
