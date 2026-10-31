//! A model's concurrency profile, as its model card says it (and the registry entry carries it): how many copies run
//! on one card and are worth running, and whether other models may run beside it. The one place the schema, its
//! defaults and its validation live: the catalog rejects a card with bad values at load (`Profile::from_card`), the
//! front door reads the same (`Profile::of_entry`), and `nextcuda models bench` proposes a block with the decision
//! rules below.
//!
//! ```yaml
//! concurrency:
//!   max_copies_per_card: 2        # never more than this many copies of the model on one card (default 1)
//!   recommended_copies: 1         # the n worth running (1..=max; default 1): extra copies start only when requests wait
//!   scaling: {"1": 1.0, "2": 1.18}  # measured total throughput of n copies relative to 1 (null: unmeasured)
//!   measured: {date: 2026-10-31, card: RTX 5070, method: "nextcuda models bench", notes: ""}   # null: unmeasured
//! coexistence:
//!   mode: shared                  # shared (default) | exclusive
//!   shared_with: ["kind:audio", "model:whisper-large-v3-turbo"]   # what it may run beside; empty = anything that fits
//!   exclusive_reason: ""          # required when exclusive
//!   peak_extra_gib: 1.5           # optional: working memory beyond vram_gib at its peak
//! ```
//!
//! A card with neither block is *unmeasured*: one copy, shared with anything that fits, nothing changes for it.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

/// The kinds a `shared_with: ["kind:..."]` may name
pub const KINDS: [&str; 6] = ["llm", "image", "audio", "video", "model3d", "lora"];

/// Whether other models may share the card
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Shared,
    Exclusive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Concurrency {
    pub max_copies_per_card: u32,
    pub recommended_copies: u32,
    /// n -> total throughput of n copies relative to one
    pub scaling: Option<BTreeMap<u32, f64>>,
    /// {date, card, method, notes}
    pub measured: Option<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Coexistence {
    pub mode: Mode,
    pub shared_with: Vec<String>,
    pub exclusive_reason: String,
    pub peak_extra_gib: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub concurrency: Concurrency,
    pub coexistence: Coexistence,
}

impl Default for Profile {
    fn default() -> Profile {
        Profile {
            concurrency: Concurrency { max_copies_per_card: 1, recommended_copies: 1, scaling: None, measured: None },
            coexistence: Coexistence { mode: Mode::Shared, shared_with: Vec::new(), exclusive_reason: String::new(), peak_extra_gib: None },
        }
    }
}

fn uint(v: &Value, what: &str) -> Result<u32, String> {
    v.as_u64().filter(|n| (1..=64).contains(n)).map(|n| n as u32).ok_or_else(|| format!("{what}: a whole number from 1 to 64, not {v}"))
}

impl Profile {
    /// The profile a card (or a registry entry) says, with defaults for what it leaves out; Err names the bad value
    pub fn from_card(card: &Value) -> Result<Profile, String> {
        let mut p = Profile::default();
        let c = &card["concurrency"];
        if !c.is_null() {
            let o = c.as_object().ok_or("concurrency: a mapping")?;
            if let Some(k) = o.keys().find(|k| !["max_copies_per_card", "recommended_copies", "scaling", "measured"].contains(&k.as_str())) {
                return Err(format!("concurrency.{k}: unknown (max_copies_per_card, recommended_copies, scaling, measured)"));
            }
            if let Some(v) = o.get("max_copies_per_card").filter(|v| !v.is_null()) {
                p.concurrency.max_copies_per_card = uint(v, "concurrency.max_copies_per_card")?;
            }
            let max = p.concurrency.max_copies_per_card;
            if let Some(v) = o.get("recommended_copies").filter(|v| !v.is_null()) {
                let r = uint(v, "concurrency.recommended_copies")?;
                if r > max {
                    return Err(format!("concurrency.recommended_copies {r}: more than max_copies_per_card {max}"));
                }
                p.concurrency.recommended_copies = r;
            }
            if let Some(sc) = o.get("scaling").filter(|v| !v.is_null()) {
                let m = sc.as_object().ok_or("concurrency.scaling: a mapping of copies to relative throughput")?;
                let mut out = BTreeMap::new();
                for (k, v) in m {
                    let n: u32 = k.parse().ok().filter(|n| (1..=max).contains(n)).ok_or_else(|| format!("concurrency.scaling.{k}: a number of copies from 1 to max_copies_per_card {max}"))?;
                    let x = v.as_f64().filter(|x| x.is_finite() && *x > 0.0).ok_or_else(|| format!("concurrency.scaling.{k}: a positive number, not {v}"))?;
                    out.insert(n, x);
                }
                p.concurrency.scaling = Some(out);
            }
            if let Some(m) = o.get("measured").filter(|v| !v.is_null()) {
                let mo = m.as_object().ok_or("concurrency.measured: {date, card, method, notes} or null")?;
                for k in ["date", "card", "method"] {
                    if mo.get(k).is_none_or(|v| v.is_null() || v.as_str().is_some_and(str::is_empty)) {
                        return Err(format!("concurrency.measured.{k}: required (who measured what, where, how)"));
                    }
                }
                p.concurrency.measured = Some(m.clone());
            }
        }
        let x = &card["coexistence"];
        if !x.is_null() {
            let o = x.as_object().ok_or("coexistence: a mapping")?;
            if let Some(k) = o.keys().find(|k| !["mode", "shared_with", "exclusive_reason", "peak_extra_gib", "notes"].contains(&k.as_str())) {
                return Err(format!("coexistence.{k}: unknown (mode, shared_with, exclusive_reason, peak_extra_gib)"));
            }
            p.coexistence.mode = match o.get("mode").and_then(Value::as_str) {
                None | Some("shared") => Mode::Shared,
                Some("exclusive") => Mode::Exclusive,
                Some(other) => return Err(format!("coexistence.mode {other:?}: shared or exclusive")),
            };
            if let Some(list) = o.get("shared_with").filter(|v| !v.is_null()) {
                for e in list.as_array().ok_or("coexistence.shared_with: a list of \"kind:<kind>\" and \"model:<id>\"")? {
                    let e = e.as_str().ok_or("coexistence.shared_with: strings")?;
                    match e.split_once(':') {
                        Some(("kind", k)) if KINDS.contains(&k) => {}
                        Some(("model", id)) if !id.is_empty() => {}
                        _ => return Err(format!("coexistence.shared_with {e:?}: \"kind:<{}>\" or \"model:<id>\"", KINDS.join("|"))),
                    }
                    p.coexistence.shared_with.push(e.to_string());
                }
            }
            p.coexistence.exclusive_reason = o.get("exclusive_reason").and_then(Value::as_str).unwrap_or("").to_string();
            if let Some(g) = o.get("peak_extra_gib").filter(|v| !v.is_null()) {
                p.coexistence.peak_extra_gib = Some(g.as_f64().filter(|g| g.is_finite() && *g >= 0.0).ok_or_else(|| format!("coexistence.peak_extra_gib: GiB, 0 or more, not {g}"))?);
            }
            if p.coexistence.mode == Mode::Exclusive {
                if !p.coexistence.shared_with.is_empty() {
                    return Err("coexistence: an exclusive model is shared with nothing (drop shared_with, or make it shared)".into());
                }
                if p.coexistence.exclusive_reason.trim().is_empty() {
                    return Err("coexistence.exclusive_reason: say why (what fails or collapses when it shares a card)".into());
                }
            }
        }
        Ok(p)
    }

    /// The profile of a registry entry (a lenient read: the card was validated when it was loaded)
    pub fn of_entry(entry: &Value) -> Profile {
        Profile::from_card(entry).unwrap_or_default()
    }

    /// Whether the model was measured (its card says who measured it)
    pub fn measured(&self) -> bool {
        self.concurrency.measured.is_some()
    }

    /// For clients: the profile with its defaults filled in, and whether it is measured ("unmeasured" otherwise)
    pub fn to_json(&self) -> (Value, Value) {
        let c = &self.concurrency;
        let scaling: Value = c.scaling.as_ref().map_or(Value::Null, |m| Value::Object(m.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<Map<_, _>>()));
        let x = &self.coexistence;
        (
            json!({"max_copies_per_card": c.max_copies_per_card, "recommended_copies": c.recommended_copies, "scaling": scaling,
                   "measured": c.measured.clone().unwrap_or(Value::Null), "status": if self.measured() { "measured" } else { "unmeasured" }}),
            json!({"mode": match x.mode { Mode::Shared => "shared", Mode::Exclusive => "exclusive" }, "shared_with": x.shared_with,
                   "exclusive_reason": x.exclusive_reason, "peak_extra_gib": x.peak_extra_gib}),
        )
    }

    /// Whether `other` (model `oid` of kind `okind`, with profile `op`) may be on a card with this model (`id`, `kind`).
    /// Exclusive either way: no. A `shared_with` on either side must name the other (by `kind:` or `model:`).
    pub fn may_share(&self, id: &str, kind: &str, op: &Profile, oid: &str, okind: &str) -> bool {
        if self.coexistence.mode == Mode::Exclusive || op.coexistence.mode == Mode::Exclusive {
            return false;
        }
        let names = |list: &[String], id: &str, kind: &str| list.is_empty() || list.iter().any(|e| e == &format!("kind:{kind}") || e == &format!("model:{id}"));
        names(&self.coexistence.shared_with, oid, okind) && names(&op.coexistence.shared_with, id, kind)
    }
}

// ---- the measurement's decisions (`nextcuda models bench`) ----

/// Total throughput measured with `copies` copies running (units per minute, summed over the copies)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalePoint {
    pub copies: u32,
    pub total_per_min: f64,
}

/// The gain a further copy must bring over the total before it, to be worth running
pub const WORTH_GAIN: f64 = 0.25;

/// Throughput below this share of its running-alone figure, beside another model, makes a model exclusive
pub const COLLAPSE_SHARE: f64 = 0.5;

/// `recommended_copies`: the largest n whose total throughput is at least WORTH_GAIN above n-1's (copies counted
/// up from 1 and stopping at the first that does not pay)
pub fn recommend(points: &[ScalePoint]) -> u32 {
    let mut p: Vec<ScalePoint> = points.iter().copied().filter(|p| p.total_per_min.is_finite() && p.total_per_min > 0.0).collect();
    p.sort_by_key(|p| p.copies);
    let mut rec = 1;
    for w in p.windows(2) {
        if w[1].copies != w[0].copies + 1 || w[1].total_per_min < w[0].total_per_min * (1.0 + WORTH_GAIN) {
            break;
        }
        rec = w[1].copies;
    }
    rec
}

/// The `scaling` map: each n's total throughput over one copy's
pub fn scaling_of(points: &[ScalePoint]) -> BTreeMap<u32, f64> {
    let base = points.iter().find(|p| p.copies == 1).map_or(0.0, |p| p.total_per_min);
    if base <= 0.0 {
        return BTreeMap::new();
    }
    points.iter().map(|p| (p.copies, (p.total_per_min / base * 100.0).round() / 100.0)).collect()
}

/// What running two models on one card measured: each one's throughput alone and beside the other, and whether
/// anything failed (an OOM, a hang, a refusal)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beside {
    pub alone: f64,
    pub with_other: f64,
    pub other_alone: f64,
    pub other_with: f64,
    pub failed: bool,
}

/// Exclusive (with the reason) when co-running failed or either's throughput fell below COLLAPSE_SHARE of its own
pub fn verdict(b: &Beside) -> (Mode, String) {
    if b.failed {
        return (Mode::Exclusive, "co-running failed (out of memory, a hang or a refusal)".into());
    }
    let share = |with: f64, alone: f64| if alone > 0.0 { with / alone } else { 1.0 };
    let (s, o) = (share(b.with_other, b.alone), share(b.other_with, b.other_alone));
    if s < COLLAPSE_SHARE {
        return (Mode::Exclusive, format!("its throughput fell to {:.0}% beside the other model", s * 100.0));
    }
    if o < COLLAPSE_SHARE {
        return (Mode::Exclusive, format!("the other model's throughput fell to {:.0}% beside it", o * 100.0));
    }
    (Mode::Shared, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(v: &[(u32, f64)]) -> Vec<ScalePoint> {
        v.iter().map(|(c, t)| ScalePoint { copies: *c, total_per_min: *t }).collect()
    }

    #[test]
    fn an_unmeasured_card_is_one_copy_shared_with_anything() {
        let p = Profile::from_card(&json!({"id": "x"})).unwrap();
        assert_eq!(p, Profile::default());
        assert!(!p.measured());
        let (c, x) = p.to_json();
        assert_eq!((c["max_copies_per_card"].as_u64(), c["recommended_copies"].as_u64(), c["status"].as_str()), (Some(1), Some(1), Some("unmeasured")));
        assert_eq!((x["mode"].as_str(), x["shared_with"].as_array().map(Vec::len)), (Some("shared"), Some(0)));
    }

    #[test]
    fn a_full_block_is_read_and_defaults_fill_the_rest() {
        let card = json!({"concurrency": {"max_copies_per_card": 3, "recommended_copies": 2, "scaling": {"1": 1.0, "2": 1.6}, "measured": {"date": "2026-10-31", "card": "RTX 5070", "method": "bench"}},
                          "coexistence": {"mode": "shared", "shared_with": ["kind:audio", "model:whisper"], "peak_extra_gib": 1.5}});
        let p = Profile::from_card(&card).unwrap();
        assert_eq!((p.concurrency.max_copies_per_card, p.concurrency.recommended_copies), (3, 2));
        assert_eq!(p.concurrency.scaling.as_ref().unwrap()[&2], 1.6);
        assert!(p.measured());
        assert_eq!(p.to_json().0["status"], "measured");
        // a recommended count alone defaults max... to 1, so it is rejected
        assert!(Profile::from_card(&json!({"concurrency": {"recommended_copies": 2}})).is_err());
    }

    #[test]
    fn bad_values_are_rejected_with_a_clear_message() {
        let err = |c: Value| Profile::from_card(&c).unwrap_err();
        assert!(err(json!({"concurrency": {"max_copies_per_card": 0}})).contains("max_copies_per_card"));
        assert!(err(json!({"concurrency": {"max_copies_per_card": "two"}})).contains("whole number"));
        assert!(err(json!({"concurrency": {"max_copies_per_card": 2, "recommended_copies": 3}})).contains("more than max_copies_per_card 2"));
        assert!(err(json!({"concurrency": {"max_copies_per_card": 2, "scaling": {"3": 1.2}}})).contains("scaling.3"));
        assert!(err(json!({"concurrency": {"max_copies_per_card": 2, "scaling": {"2": -1}}})).contains("positive"));
        assert!(err(json!({"concurrency": {"measured": {"date": "x"}}})).contains("measured.card"));
        assert!(err(json!({"concurrency": {"bogus": 1}})).contains("unknown"));
        assert!(err(json!({"coexistence": {"mode": "alone"}})).contains("shared or exclusive"));
        assert!(err(json!({"coexistence": {"shared_with": ["image"]}})).contains("kind:"));
        assert!(err(json!({"coexistence": {"shared_with": ["kind:video3"]}})).contains("kind:"));
        assert!(err(json!({"coexistence": {"mode": "exclusive"}})).contains("exclusive_reason"));
        assert!(err(json!({"coexistence": {"mode": "exclusive", "exclusive_reason": "OOM", "shared_with": ["kind:audio"]}})).contains("shared with nothing"));
        assert!(err(json!({"coexistence": {"peak_extra_gib": -1}})).contains("peak_extra_gib"));
    }

    #[test]
    fn who_may_share_a_card() {
        let any = Profile::default();
        let excl = Profile::from_card(&json!({"coexistence": {"mode": "exclusive", "exclusive_reason": "fills the card"}})).unwrap();
        let audio_only = Profile::from_card(&json!({"coexistence": {"shared_with": ["kind:audio"]}})).unwrap();
        let named = Profile::from_card(&json!({"coexistence": {"shared_with": ["model:whisper"]}})).unwrap();
        assert!(any.may_share("a", "image", &any, "b", "audio"));
        assert!(!excl.may_share("a", "image", &any, "b", "audio"), "exclusive");
        assert!(!any.may_share("b", "audio", &excl, "a", "image"), "exclusive, the other way");
        assert!(audio_only.may_share("a", "image", &any, "b", "audio"));
        assert!(!audio_only.may_share("a", "image", &any, "b", "image"));
        assert!(named.may_share("a", "image", &any, "whisper", "audio"));
        assert!(!named.may_share("a", "image", &any, "other", "audio"));
        // both sides' lists count
        assert!(!audio_only.may_share("a", "image", &named, "whisper", "audio"), "whisper's own list does not name an image model... it names whisper");
    }

    #[test]
    fn scaling_gives_the_recommended_copies() {
        // 2 copies pay 18% over one: not worth it
        assert_eq!(recommend(&pts(&[(1, 10.0), (2, 11.8)])), 1);
        // 2 pay 60%, 3 pay 10% more: 2
        assert_eq!(recommend(&pts(&[(1, 10.0), (2, 16.0), (3, 17.6)])), 2);
        // 3 pays 25% exactly over 2: worth it
        assert_eq!(recommend(&pts(&[(1, 10.0), (2, 16.0), (3, 20.0)])), 3);
        // unsorted input, a broken run, and one point
        assert_eq!(recommend(&pts(&[(2, 16.0), (1, 10.0)])), 2);
        assert_eq!(recommend(&pts(&[(1, 10.0)])), 1);
        assert_eq!(recommend(&[]), 1);
        assert_eq!(scaling_of(&pts(&[(1, 10.0), (2, 11.8)])), BTreeMap::from([(1, 1.0), (2, 1.18)]));
        assert!(scaling_of(&pts(&[(2, 11.8)])).is_empty(), "no one-copy figure: no scaling");
    }

    #[test]
    fn beside_another_model_decides_exclusive() {
        let ok = Beside { alone: 10.0, with_other: 8.0, other_alone: 6.0, other_with: 5.0, failed: false };
        assert_eq!(verdict(&ok).0, Mode::Shared);
        // either throughput under half of its own
        assert_eq!(verdict(&Beside { with_other: 4.9, ..ok }).0, Mode::Exclusive);
        let (m, why) = verdict(&Beside { other_with: 2.0, ..ok });
        assert_eq!(m, Mode::Exclusive);
        assert!(why.contains("other model"), "{why}");
        // exactly half is still shared; a failure is exclusive whatever the numbers say
        assert_eq!(verdict(&Beside { with_other: 5.0, ..ok }).0, Mode::Shared);
        assert_eq!(verdict(&Beside { failed: true, ..ok }).0, Mode::Exclusive);
    }
}
