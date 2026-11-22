//! Silo variants: a silo's ops tuned for other hardware, as a delta off the default silo (CONTRIBUTING rule 6).
//!
//! `kernels/<kind>/<arch>/silo/variants/<name>/` holds a manifest (`variant.yml`) and only the ops that differ; it
//! builds to `dist/silo/libnextsycl-<arch>@<name>.so` with its manifest beside it (`libnextsycl-<arch>@<name>.yml`).
//! An op is then taken, per GPU, from the variant that GPU matches, else from the default silo (else the engine's
//! shared kernel). A variant may also carry `settings`: engine options (the names `--opt-NAME` takes) that become the
//! engine's defaults when the variant applies to every GPU it loads on (`nextsycl_core::variant_settings`). This
//! module is the pure part: the manifest, the device a rule reads, the choice, and whether two rules can overlap.

use std::path::PathBuf;

/// What a variant's match rule reads of one GPU (`ns_gpu_name`, `ns_gpu_units`, `ns_gpu_props`)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    /// compute units (Xe vector engines) and their clock, MHz
    pub units: u32,
    pub mhz: u32,
    /// its memory in all, MiB (None: a kernel library built before ns_gpu_props)
    pub vram_mb: Option<u64>,
    /// the sub-group sizes it runs (empty: not known)
    pub sub_groups: Vec<u32>,
}

impl DeviceInfo {
    /// The card's id in the merge gate's baselines (tooling/gate-baselines.json `cards`): its name without the
    /// vendor's marks, its units and its memory in GiB - `arc-pro-b70-256u-32g`. Two boxes with the same card share it.
    pub fn card_id(&self) -> String {
        let mut words = Vec::new();
        for w in self.name.to_lowercase().replace("(r)", " ").replace("(tm)", " ").split(|c: char| !c.is_ascii_alphanumeric()) {
            if !w.is_empty() && !matches!(w, "intel" | "graphics") {
                words.push(w.to_string());
            }
        }
        words.push(format!("{}u", self.units));
        if let Some(m) = self.vram_mb {
            words.push(format!("{}g", (m + 512) / 1024));
        }
        words.join("-")
    }

    /// As one line of JSON: what tooling/gate.sh records beside a card's baseline (the architecture test reads it back
    /// to check a variant has a baseline on a card it matches)
    pub fn to_json(&self) -> String {
        let esc: String = self.name.chars().flat_map(|c| match c {
            '"' | '\\' => vec!['\\', c],
            c if (c as u32) < 0x20 => vec![' '],
            c => vec![c],
        }).collect();
        let sg: Vec<String> = self.sub_groups.iter().map(|s| s.to_string()).collect();
        format!("{{\"name\": \"{esc}\", \"units\": {}, \"mhz\": {}, \"vram_mb\": {}, \"sub_groups\": [{}]}}", self.units, self.mhz,
                self.vram_mb.map_or("null".into(), |m| m.to_string()), sg.join(", "))
    }
}

/// A variant's match rule: every bound given must hold (an unknown device value fails a bound on it)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Match {
    pub vram_mb_min: Option<u64>,
    pub vram_mb_max: Option<u64>,
    pub units_min: Option<u32>,
    pub units_max: Option<u32>,
    pub mhz_min: Option<u32>,
    pub mhz_max: Option<u32>,
    /// a sub-group size the device must run
    pub sub_group: Option<u32>,
    /// a substring of the device's name: the last resort, when the numbers do not tell the cards apart
    pub name: Option<String>,
}

impl Match {
    pub fn matches(&self, d: &DeviceInfo) -> bool {
        let within = |v: Option<u64>, lo: Option<u64>, hi: Option<u64>| match v {
            Some(v) => lo.is_none_or(|l| v >= l) && hi.is_none_or(|h| v <= h),
            None => lo.is_none() && hi.is_none(),
        };
        within(d.vram_mb, self.vram_mb_min, self.vram_mb_max)
            && within(Some(d.units as u64), self.units_min.map(u64::from), self.units_max.map(u64::from))
            && within(Some(d.mhz as u64), self.mhz_min.map(u64::from), self.mhz_max.map(u64::from))
            && self.sub_group.is_none_or(|s| d.sub_groups.contains(&s))
            && self.name.as_deref().is_none_or(|n| d.name.contains(n))
    }

    fn is_empty(&self) -> bool {
        *self == Match::default()
    }

    /// Whether one device could match both rules. Only the numbers separate two rules (their ranges do not meet on
    /// some dimension, an absent bound being open); a sub-group size never does (a device runs several), nor does a
    /// name (one name can hold two substrings) - conservative: a pair this calls overlapping may in fact never meet,
    /// and its fix is a tighter numeric bound.
    pub fn overlaps(&self, other: &Match) -> bool {
        let meet = |lo1: Option<u64>, hi1: Option<u64>, lo2: Option<u64>, hi2: Option<u64>| {
            let lo = lo1.unwrap_or(0).max(lo2.unwrap_or(0));
            let hi = hi1.unwrap_or(u64::MAX).min(hi2.unwrap_or(u64::MAX));
            lo <= hi
        };
        let w = |v: Option<u32>| v.map(u64::from);
        meet(self.vram_mb_min, self.vram_mb_max, other.vram_mb_min, other.vram_mb_max)
            && meet(w(self.units_min), w(self.units_max), w(other.units_min), w(other.units_max))
            && meet(w(self.mhz_min), w(self.mhz_max), w(other.mhz_min), w(other.mhz_max))
    }
}

/// A variant's manifest (`variant.yml`)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Manifest {
    /// one line: what it changes and why
    pub description: String,
    /// who measured it, on what (the before / after numbers are in its pull request)
    pub measured: String,
    pub rule: Match,
    /// engine options (option name, value) that become the engine's defaults where the variant applies: below the
    /// model card's settings, `--opt` and the environment, above the engine's own default (`settings:`)
    pub settings: Vec<(String, String)>,
}

/// Reads a manifest: `description:`, `measured:`, a `match:` block of the bounds (`vram_mb_min`, `vram_mb_max`,
/// `units_min`, `units_max`, `mhz_min`, `mhz_max`, `sub_group`, `name`) and an optional `settings:` block (engine
/// option name: value), a `key: value` a line, `#` comments. A flat subset of YAML, read here without a YAML crate
/// (the foundation has no dependencies); an unknown key, a missing field, an empty rule, a setting given twice or
/// without a value is an error. Whether a setting names one of its engine's options, with a value that option takes,
/// is checked against the engine (`nextsycl_core::options::variant_settings`, and the architecture test).
pub fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let mut m = Manifest::default();
    let mut in_match = false;
    let mut in_settings = false;
    let mut seen_match = false;
    for (n, raw) in text.lines().enumerate() {
        let line = match raw.find(" #") {
            Some(i) => &raw[..i],
            None if raw.trim_start().starts_with('#') => "",
            None => raw,
        };
        if line.trim().is_empty() {
            continue;
        }
        let indented = line.starts_with([' ', '\t']);
        let (k, v) = line.trim().split_once(':').ok_or_else(|| format!("line {}: `key: value` expected", n + 1))?;
        let (k, v) = (k.trim(), v.trim().trim_matches(|c| c == '"' || c == '\''));
        let num = |what: &str| v.parse::<u64>().map_err(|_| format!("line {}: {what}: a whole number, not {v:?}", n + 1));
        if !indented {
            in_match = false;
            in_settings = false;
        }
        if indented && in_settings {
            if k.is_empty() || !k.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_') {
                return Err(format!("line {}: settings: an engine option's name ([a-z0-9-_]), not {k:?}", n + 1));
            }
            if v.is_empty() {
                return Err(format!("line {}: settings: {k} has no value", n + 1));
            }
            if m.settings.iter().any(|(x, _)| x == k) {
                return Err(format!("line {}: settings: {k} given twice", n + 1));
            }
            m.settings.push((k.to_string(), v.to_string()));
            continue;
        }
        match (indented, in_match, k) {
            (false, _, "description") => m.description = v.to_string(),
            (false, _, "measured") => m.measured = v.to_string(),
            (false, _, "match") if v.is_empty() => {
                in_match = true;
                seen_match = true;
            }
            (false, _, "settings") if v.is_empty() => in_settings = true,
            (true, true, "vram_mb_min") => m.rule.vram_mb_min = Some(num(k)?),
            (true, true, "vram_mb_max") => m.rule.vram_mb_max = Some(num(k)?),
            (true, true, "units_min") => m.rule.units_min = Some(num(k)? as u32),
            (true, true, "units_max") => m.rule.units_max = Some(num(k)? as u32),
            (true, true, "mhz_min") => m.rule.mhz_min = Some(num(k)? as u32),
            (true, true, "mhz_max") => m.rule.mhz_max = Some(num(k)? as u32),
            (true, true, "sub_group") => m.rule.sub_group = Some(num(k)? as u32),
            (true, true, "name") if !v.is_empty() => m.rule.name = Some(v.to_string()),
            _ => return Err(format!("line {}: unknown key {k:?}", n + 1)),
        }
    }
    if m.description.is_empty() || m.measured.is_empty() {
        return Err("description and measured are required".into());
    }
    if !seen_match || m.rule.is_empty() {
        return Err("a match rule is required (vram_mb_min / _max, units_min / _max, mhz_min / _max, sub_group, name)".into());
    }
    Ok(m)
}

/// A built variant of a silo: dist/silo/libnextsycl-<silo>@<name>.so and its manifest
#[derive(Clone, Debug)]
pub struct VariantFile {
    pub name: String,
    pub path: PathBuf,
    pub manifest: Manifest,
}

/// The variants every one of these devices matches, by name
pub fn matching<'a>(variants: &'a [VariantFile], devices: &[DeviceInfo]) -> Vec<&'a VariantFile> {
    if devices.is_empty() {
        return Vec::new();
    }
    let mut v: Vec<&VariantFile> = variants.iter().filter(|v| devices.iter().all(|d| v.manifest.rule.matches(d))).collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// A warning when more than one variant matches these devices and none is forced: two variants of one silo whose
/// rules meet (the architecture test refuses such a pair; a manifest written or built outside the tree can still
/// make one). `choose` still picks the first by name.
pub fn ambiguity(variants: &[VariantFile], devices: &[DeviceInfo], force: Option<&str>) -> Option<String> {
    if force.is_some_and(|f| !f.trim().is_empty()) {
        return None;
    }
    let m = matching(variants, devices);
    (m.len() > 1).then(|| {
        let names: Vec<&str> = m.iter().map(|v| v.name.as_str()).collect();
        let on: Vec<String> = devices.iter().map(|d| format!("GPU {}", d.index)).collect();
        format!("warning: silo variants {} all match {}: taking {} (the first by name); tighten their match rules",
                names.join(", "), on.join(", "), names[0])
    })
}

/// The variant for these devices: `force` (NS_SILO_VARIANT, a model card's `silo.variant`) names one - `none` none -,
/// else the one every device matches (by name order when several do: `ambiguity` says so). Err: a forced variant
/// that is not built.
pub fn choose<'a>(variants: &'a [VariantFile], devices: &[DeviceInfo], force: Option<&str>) -> Result<Option<&'a VariantFile>, String> {
    match force.map(str::trim).filter(|f| !f.is_empty()) {
        Some("none") => Ok(None),
        Some(f) => variants.iter().find(|v| v.name == f).map(Some).ok_or_else(|| {
            let built: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
            format!("silo variant {f:?} is not built (built: {})", if built.is_empty() { "none".into() } else { built.join(", ") })
        }),
        None => Ok(matching(variants, devices).first().copied()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(name: &str, units: u32, vram_mb: u64) -> DeviceInfo {
        DeviceInfo { index: 0, name: name.into(), units, mhz: 2800, vram_mb: Some(vram_mb), sub_groups: vec![16, 32] }
    }
    fn var(name: &str, yml: &str) -> VariantFile {
        VariantFile { name: name.into(), path: PathBuf::from(format!("libnextsycl-x@{name}.so")), manifest: parse_manifest(yml).unwrap() }
    }
    const SMALL: &str = "description: a 16 GB card's tiles\nmeasured: someone, Arc B580\nmatch:\n  vram_mb_max: 20000  # under 20 GB\n";

    #[test]
    fn a_manifest_reads_and_refuses_what_it_does_not_know() {
        let m = parse_manifest(SMALL).unwrap();
        assert_eq!(m.rule, Match { vram_mb_max: Some(20000), ..Match::default() });
        assert_eq!(m.measured, "someone, Arc B580");
        assert!(parse_manifest("description: a\nmeasured: b\nmatch:\n  colour: red\n").unwrap_err().contains("colour"));
        assert!(parse_manifest("description: a\nmeasured: b\n").unwrap_err().contains("match rule"));
        assert!(parse_manifest("description: a\nmatch:\n  units_max: 3\n").unwrap_err().contains("measured"));
        assert!(parse_manifest("description: a\nmeasured: b\nmatch:\n  units_max: many\n").unwrap_err().contains("whole number"));
        // a top-level key after the block ends it
        assert!(parse_manifest("description: a\nmatch:\n  units_max: 3\nmeasured: b\n").is_ok());
        assert!(parse_manifest("description: a\nmeasured: b\nunits_max: 3\nmatch:\n  name: x\n").is_err());
    }

    #[test]
    fn a_rule_matches_by_the_numbers_the_device_reports() {
        let r = parse_manifest("description: a\nmeasured: b\nmatch:\n  units_max: 200\n  vram_mb_min: 16000\n  sub_group: 32\n").unwrap().rule;
        assert!(r.matches(&dev("B65", 160, 32_000)));
        assert!(!r.matches(&dev("B70", 256, 32_000)));
        assert!(!r.matches(&dev("small", 160, 12_000)));
        assert!(!r.matches(&DeviceInfo { sub_groups: vec![16], ..dev("B65", 160, 32_000) }));
        // a memory bound on a device whose memory is not known: no match
        assert!(!r.matches(&DeviceInfo { vram_mb: None, ..dev("B65", 160, 32_000) }));
        let by_name = parse_manifest("description: a\nmeasured: b\nmatch:\n  name: B580\n").unwrap().rule;
        assert!(by_name.matches(&dev("Intel(R) Arc(TM) B580 Graphics", 160, 12_000)) && !by_name.matches(&dev("other", 160, 12_000)));
    }

    #[test]
    fn the_choice_is_per_device_and_can_be_forced_or_refused() {
        let vs = vec![var("small", SMALL), var("few-units", "description: a\nmeasured: b\nmatch:\n  units_max: 200\n")];
        let (b70, b65, b580) = (dev("B70", 256, 32_656), dev("B65", 160, 32_656), dev("B580", 160, 12_216));
        let pick = |d: &[DeviceInfo], f: Option<&str>| choose(&vs, d, f).unwrap().map(|v| v.name.clone());
        assert_eq!(pick(std::slice::from_ref(&b70), None), None);
        assert_eq!(pick(std::slice::from_ref(&b65), None).as_deref(), Some("few-units"));
        // two match the small card: the first by name
        assert_eq!(pick(std::slice::from_ref(&b580), None).as_deref(), Some("few-units"));
        // a mixed pair: only a variant both match
        assert_eq!(pick(&[b70.clone(), b65.clone()], None), None);
        assert_eq!(pick(&[b65.clone(), b580.clone()], None).as_deref(), Some("few-units"));
        assert_eq!(pick(std::slice::from_ref(&b70), Some("small")).as_deref(), Some("small"));
        assert_eq!(pick(std::slice::from_ref(&b65), Some("none")), None);
        assert!(choose(&vs, std::slice::from_ref(&b65), Some("nope")).unwrap_err().contains("not built"));
        assert_eq!(pick(&[], None), None);
    }

    #[test]
    fn settings_read_as_option_names_and_values() {
        let m = parse_manifest("description: a\nmeasured: b\nmatch:\n  units_max: 200\nsettings:\n  drafts: 3  # the default\n  image-tokens: '4096'\n").unwrap();
        assert_eq!(m.settings, vec![("drafts".to_string(), "3".to_string()), ("image-tokens".to_string(), "4096".to_string())]);
        assert_eq!(m.rule, Match { units_max: Some(200), ..Match::default() });
        // the blocks in either order, and a top-level key after the settings ends them
        let m = parse_manifest("description: a\nsettings:\n  drafts: 3\nmatch:\n  units_max: 200\nmeasured: b\n").unwrap();
        assert_eq!(m.settings.len(), 1);
        assert!(parse_manifest("description: a\nmeasured: b\nmatch:\n  units_max: 2\nsettings:\n  drafts:\n").unwrap_err().contains("no value"));
        assert!(parse_manifest("description: a\nmeasured: b\nmatch:\n  units_max: 2\nsettings:\n  drafts: 1\n  drafts: 2\n").unwrap_err().contains("twice"));
        assert!(parse_manifest("description: a\nmeasured: b\nmatch:\n  units_max: 2\nsettings:\n  NS_Q35_DRAFTS: 1\n").unwrap_err().contains("option's name"));
        // a bound under settings is a setting, not a bound: the rule stays required
        assert!(parse_manifest("description: a\nmeasured: b\nsettings:\n  units_max: 2\n").unwrap_err().contains("match rule"));
        assert!(parse_manifest(SMALL).unwrap().settings.is_empty());
    }

    #[test]
    fn rules_overlap_unless_a_number_separates_them() {
        let r = |y: &str| parse_manifest(&format!("description: a\nmeasured: b\nmatch:\n{y}")).unwrap().rule;
        let few = r("  units_max: 200\n");
        let many = r("  units_min: 201\n");
        let small = r("  vram_mb_max: 20000\n");
        assert!(!few.overlaps(&many) && !many.overlaps(&few));
        // open on memory: a card of few units and little memory matches both
        assert!(few.overlaps(&small) && small.overlaps(&few));
        assert!(few.overlaps(&few));
        // the edges: a range ending where the other starts meets it
        assert!(r("  units_max: 200\n").overlaps(&r("  units_min: 200\n")));
        assert!(!r("  mhz_max: 2000\n  units_max: 200\n").overlaps(&r("  mhz_min: 2001\n  units_max: 200\n")));
        // a sub-group size or a name never separates two rules
        assert!(r("  sub_group: 16\n").overlaps(&r("  sub_group: 32\n")));
        assert!(r("  name: B580\n").overlaps(&r("  name: B570\n")));
        assert!(r("  name: B580\n  units_max: 100\n").overlaps(&r("  name: B570\n  units_max: 200\n")));
        assert!(!r("  name: B580\n  units_max: 100\n").overlaps(&r("  name: B580\n  units_min: 101\n")));
    }

    #[test]
    fn two_matching_variants_warn_and_the_first_by_name_is_taken() {
        let vs = vec![var("small", SMALL), var("few-units", "description: a\nmeasured: b\nmatch:\n  units_max: 200\n")];
        let b580 = dev("B580", 160, 12_216);
        let w = ambiguity(&vs, std::slice::from_ref(&b580), None).unwrap();
        assert!(w.contains("few-units, small") && w.contains("taking few-units"), "{w}");
        assert_eq!(choose(&vs, std::slice::from_ref(&b580), None).unwrap().unwrap().name, "few-units");
        // one match, or a forced one: no warning
        assert_eq!(ambiguity(&vs, &[dev("B65", 160, 32_656)], None), None);
        assert_eq!(ambiguity(&vs, std::slice::from_ref(&b580), Some("small")), None);
    }

    #[test]
    fn a_card_is_known_by_what_it_reports() {
        let d = DeviceInfo { index: 1, name: "Intel(R) Arc(TM) Pro B65 Graphics".into(), units: 160, mhz: 2400, vram_mb: Some(32_656), sub_groups: vec![16, 32] };
        assert_eq!(d.card_id(), "arc-pro-b65-160u-32g");
        assert_eq!(DeviceInfo { vram_mb: None, ..d.clone() }.card_id(), "arc-pro-b65-160u");
        assert_eq!(d.to_json(), r#"{"name": "Intel(R) Arc(TM) Pro B65 Graphics", "units": 160, "mhz": 2400, "vram_mb": 32656, "sub_groups": [16, 32]}"#);
        assert!(DeviceInfo { name: "a\"b".into(), ..d }.to_json().contains(r#""a\"b""#));
    }
}
