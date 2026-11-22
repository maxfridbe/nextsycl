//! Silo variants: a silo's ops tuned for other hardware, as a delta off the default silo (CONTRIBUTING rule 6).
//!
//! `kernels/<kind>/<arch>/silo/variants/<name>/` holds a manifest (`variant.yml`) and only the ops that differ; it
//! builds to `dist/silo/libnextsycl-<arch>@<name>.so` with its manifest beside it (`libnextsycl-<arch>@<name>.yml`).
//! An op is then taken, per GPU, from the variant that GPU matches, else from the default silo (else the engine's
//! shared kernel). This module is the pure part: the manifest, the device a rule reads, and the choice.

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
}

/// A variant's manifest (`variant.yml`)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Manifest {
    /// one line: what it changes and why
    pub description: String,
    /// who measured it, on what (the before / after numbers are in its pull request)
    pub measured: String,
    pub rule: Match,
}

/// Reads a manifest: `description:`, `measured:` and a `match:` block of the bounds (`vram_mb_min`, `vram_mb_max`,
/// `units_min`, `units_max`, `mhz_min`, `mhz_max`, `sub_group`, `name`), a `key: value` a line, `#` comments. A
/// flat subset of YAML, read here without a YAML crate (the foundation has no dependencies); an unknown key, a
/// missing field or an empty rule is an error.
pub fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let mut m = Manifest::default();
    let mut in_match = false;
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
        }
        match (indented, in_match, k) {
            (false, _, "description") => m.description = v.to_string(),
            (false, _, "measured") => m.measured = v.to_string(),
            (false, _, "match") if v.is_empty() => {
                in_match = true;
                seen_match = true;
            }
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

/// The variant for these devices: `force` (NS_SILO_VARIANT, a model card's `silo.variant`) names one - `none` none -,
/// else the one every device matches (by name order when several do). Err: a forced variant that is not built.
pub fn choose<'a>(variants: &'a [VariantFile], devices: &[DeviceInfo], force: Option<&str>) -> Result<Option<&'a VariantFile>, String> {
    match force.map(str::trim).filter(|f| !f.is_empty()) {
        Some("none") => Ok(None),
        Some(f) => variants.iter().find(|v| v.name == f).map(Some).ok_or_else(|| {
            let built: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
            format!("silo variant {f:?} is not built (built: {})", if built.is_empty() { "none".into() } else { built.join(", ") })
        }),
        None if devices.is_empty() => Ok(None),
        None => {
            let mut v: Vec<&VariantFile> = variants.iter().filter(|v| devices.iter().all(|d| v.manifest.rule.matches(d))).collect();
            v.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(v.first().copied())
        }
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
}
