//! Engine options: what an engine takes beyond its kind's common settings, so the command line, the registry and the
//! servers grow with each new architecture without knowing it. An engine declares them (`EngineOption`); every
//! `--opt-NAME VALUE` (or `--opt-NAME` alone: "1") on any command - NAME the option's name or the variable it sets,
//! `--opt-int8` or `--opt-NS_QI_INT8 1` - and every entry of an API request's `"options": {NAME: value}` is checked
//! against them and forwarded:
//!
//! ```text
//!   at load      as the environment variable it names (`env`) - what the engines and their kernels read
//!                (std::env, getenv); in-process it is set in this process before the load, in a container passed
//!                with -e - and in the load options' settings, under the same name
//!   per request  in the request's `extra`, under its name (an API request's own JSON field of that name too)
//! ```
//!
//! An option no engine of the command declares is an error that lists those it does - a typo never passes silently.
//!
//! A silo variant's `settings` (`variant.yml`: option name -> value) are a fourth source, the lowest above the
//! engine's own default: where the variant applies to every GPU the engine loads on, each sets its option's variable
//! unless the environment already has it (a model card's settings, `--opt`, the user's own). The order, highest
//! first: a request's options > `--opt` / the environment > the model card's settings > the variant's > the engine's
//! default. An engine's alternative host-side paths (a chunking, a fusion, an expert placement) are options with
//! enumerated values (`value: "rows|columns"`), so a variant picks one by name; a setting never sizes memory.

use std::collections::BTreeMap;

/// When an option applies
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum At {
    /// when the engine loads
    Load,
    /// with each request
    Request,
    /// either: at load as the default, a request's own wins
    Both,
}

/// One option an engine takes
#[derive(Clone, Copy, Debug)]
pub struct EngineOption {
    /// `--<name>` on the command line, the field `<name>` in a request
    pub name: &'static str,
    /// the environment variable it sets at load ("" none: per request only)
    pub env: &'static str,
    /// what its value looks like ("" a flag: given means "1")
    pub value: &'static str,
    pub help: &'static str,
    pub at: At,
}

/// Options given, by name ("1" for a flag)
pub type Given = BTreeMap<String, String>;

/// The prefix of an engine option on the command line
pub const PREFIX: &str = "--opt-";

/// The `--opt-NAME VALUE`, `--opt-NAME=VALUE` and `--opt-NAME` items of `args` (a value is the next item unless it
/// starts with "--"; none: "1")
pub fn given(args: &[String]) -> Given {
    let mut out = Given::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        let Some(opt) = a.strip_prefix(PREFIX) else { continue };
        let (name, value) = match opt.split_once('=') {
            Some((n, v)) => (n.to_string(), v.to_string()),
            None if args.get(i).is_some_and(|v| !v.starts_with("--")) => {
                i += 1;
                (opt.to_string(), args[i - 1].clone())
            }
            None => (opt.to_string(), "1".into()),
        };
        out.insert(name, value);
    }
    out
}

/// `args` without the engine options (what a command parses itself)
pub fn strip(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if a.starts_with(PREFIX) {
            if !a.contains('=') && args.get(i).is_some_and(|v| !v.starts_with("--")) {
                i += 1;
            }
            continue;
        }
        out.push(a.clone());
    }
    out
}

/// `given` checked against what an engine declares for `at` (by name or by variable): the environment assignments of
/// the load-time ones (variable, value), or an error naming the unknown ones and listing the engine's
pub fn resolve(given: &Given, declared: &[EngineOption], at: At, engine: &str) -> Result<Vec<(String, String)>, String> {
    let fits = |o: &EngineOption| o.at == at || o.at == At::Both || at == At::Both;
    let find = |k: &str| declared.iter().find(|o| (o.name == k || (!o.env.is_empty() && o.env == k)) && fits(o));
    // one it has, but not for this moment
    if let Some((k, o)) = given.keys().find_map(|k| {
        let o = declared.iter().find(|o| o.name == k.as_str() || (!o.env.is_empty() && o.env == k.as_str()))?;
        (!fits(o)).then_some((k, o))
    }) {
        return Err(match o.at {
            At::Load => format!("{k}: an option of {engine} at load ({PREFIX}{} when it starts), not per request", o.name),
            _ => format!("{k}: an option of {engine} per request (a request's options), not at load"),
        });
    }
    let unknown: Vec<&str> = given.keys().filter(|k| find(k).is_none()).map(String::as_str).collect();
    if !unknown.is_empty() {
        let names: Vec<String> = unknown.iter().map(|u| format!("{PREFIX}{u}")).collect();
        return Err(format!("{} not an option of {engine}{}", names.join(", "), if declared.is_empty() {
            " (it takes none)".to_string()
        } else {
            format!(":\n{}", usage(declared))
        }));
    }
    Ok(given.iter().filter_map(|(k, v)| {
        let o = find(k)?;
        (!o.env.is_empty() && o.at != At::Request).then(|| (o.env.to_string(), v.clone()))
    }).collect())
}

/// `given` by the options' own names (a variable's name given is the option's), for a request's `extra`
pub fn by_name(given: &Given, declared: &[EngineOption]) -> Given {
    given.iter().map(|(k, v)| {
        let n = declared.iter().find(|o| o.name == k.as_str() || (!o.env.is_empty() && o.env == k.as_str())).map_or(k.as_str(), |o| o.name);
        (n.to_string(), v.clone())
    }).collect()
}

/// The value shapes that are numbers (`value: "N"`): a whole number is what they take
const WHOLE: &[&str] = &["N", "TOKENS", "ROWS", "LAYERS", "FRAMES", "GIB", "MIB", "BYTES", "K"];

/// The value shapes that size memory: a variant's setting may not set one (memory is planned from what the device
/// reports, CONTRIBUTING rule 6)
const MEMORY: &[&str] = &["TOKENS", "GIB", "MIB", "BYTES"];

/// Whether `v` is a value an option of shape `shape` (its `value`) takes, as far as the shape says: a flag ("") 0 or
/// 1; `a|b|c` of plain words (lowercase, digits, `-`) one of them (an enumeration: an engine's alternative paths);
/// `N`, `TOKENS`, `GIB` ... a whole number; any other shape (`FILE`, `X`, `DIR|auto`, `A,B,...`) anything not empty
pub fn accepts(shape: &str, v: &str) -> Result<(), String> {
    let v = v.trim();
    if v.is_empty() {
        return Err("an empty value".into());
    }
    if shape.is_empty() {
        return if v == "0" || v == "1" { Ok(()) } else { Err(format!("{v:?}: a flag takes 0 or 1")) };
    }
    let word = |w: &str| !w.is_empty() && w.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    let alts: Vec<&str> = shape.split('|').collect();
    if alts.len() > 1 && alts.iter().all(|a| word(a)) {
        return if alts.contains(&v) { Ok(()) } else { Err(format!("{v:?}: one of {}", alts.join(", "))) };
    }
    if WHOLE.contains(&shape) && v.parse::<u64>().is_err() {
        return Err(format!("{v:?}: a whole number ({shape})"));
    }
    Ok(())
}

/// Whether an option of shape `shape` sizes memory (`TOKENS`, `GIB`, `MIB`, `BYTES`)
pub fn sizes_memory(shape: &str) -> bool {
    shape.split('|').any(|a| MEMORY.contains(&a))
}

/// One setting of a variant, to apply: (the variable, its value, the option's name)
pub type Setting = (String, String, &'static str);

/// A silo variant's `settings` checked against what the engine declares: each an option by its name, taken at load
/// (or at load and per request), with a variable to set, a value its shape accepts and no memory size. The variable
/// assignments (variable, value, option name), or an error naming the variant and what is wrong.
pub fn variant_settings(settings: &[(String, String)], declared: &[EngineOption], variant: &str) -> Result<Vec<Setting>, String> {
    let mut out = Vec::new();
    for (name, v) in settings {
        let Some(o) = declared.iter().find(|o| o.name == name.as_str()) else {
            let names: Vec<&str> = declared.iter().map(|o| o.name).collect();
            return Err(format!("silo variant {variant}: settings: {name} is not an option of its engine (it takes: {})",
                               if names.is_empty() { "none".into() } else { names.join(", ") }));
        };
        if o.at == At::Request || o.env.is_empty() {
            return Err(format!("silo variant {variant}: settings: {name} is a per-request option (a setting is a default at load)"));
        }
        if sizes_memory(o.value) {
            return Err(format!("silo variant {variant}: settings: {name} sizes memory ({}): a variant never does - memory is planned from what the device reports", o.value));
        }
        accepts(o.value, v).map_err(|e| format!("silo variant {variant}: settings: {name}: {e}"))?;
        out.push((o.env.to_string(), v.trim().to_string(), o.name));
    }
    Ok(out)
}

/// The options as help lines
pub fn usage(declared: &[EngineOption]) -> String {
    declared.iter().map(|o| {
        let flag = if o.value.is_empty() { format!("{PREFIX}{}", o.name) } else { format!("{PREFIX}{} {}", o.name, o.value) };
        let at = match o.at {
            At::Load => "load",
            At::Request => "request",
            At::Both => "load / request",
        };
        format!("  {flag:<34} {} ({at}{})", o.help, if o.env.is_empty() { String::new() } else { format!("; {}", o.env) })
    }).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPTS: &[EngineOption] = &[
        EngineOption { name: "int8", env: "NS_X_INT8", value: "", help: "int8", at: At::Load },
        EngineOption { name: "sigmas", env: "NS_X_SIGMAS", value: "A,B,...", help: "sigmas", at: At::Both },
    ];

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn engine_options_are_the_prefixed_ones() {
        let args = v(&["model", "--steps", "6", "--opt-int8", "--opt-sigmas", "1,0.5", "--rgba", "--opt-NS_X=3"]);
        assert_eq!(given(&args), Given::from([("int8".into(), "1".into()), ("sigmas".into(), "1,0.5".into()), ("NS_X".into(), "3".into())]));
        assert_eq!(strip(&args), v(&["model", "--steps", "6", "--rgba"]));
    }

    #[test]
    fn known_options_become_their_variables_and_unknown_ones_fail() {
        let g = Given::from([("int8".into(), "1".into()), ("sigmas".into(), "1,0.5".into())]);
        assert_eq!(resolve(&g, OPTS, At::Load, "x").unwrap(), vec![("NS_X_INT8".into(), "1".into()), ("NS_X_SIGMAS".into(), "1,0.5".into())]);
        let bad = Given::from([("int9".into(), "1".into())]);
        let e = resolve(&bad, OPTS, At::Load, "x").unwrap_err();
        assert!(e.contains("--opt-int9") && e.contains("--opt-sigmas A,B,..."), "{e}");
        // the variable's name works as the option's
        let by_env = Given::from([("NS_X_INT8".into(), "1".into())]);
        assert_eq!(resolve(&by_env, OPTS, At::Load, "x").unwrap(), vec![("NS_X_INT8".into(), "1".into())]);
        // per request, a load-only option is not one - and the error says when it is
        let e = resolve(&g, OPTS, At::Request, "x").unwrap_err();
        assert!(e.contains("at load"), "{e}");
    }

    #[test]
    fn a_value_is_checked_as_far_as_its_shape_says() {
        assert!(accepts("", "1").is_ok() && accepts("", "0").is_ok() && accepts("", "yes").is_err());
        assert!(accepts("rows|columns", "rows").is_ok());
        assert!(accepts("rows|columns", "diagonal").unwrap_err().contains("one of rows, columns"));
        assert!(accepts("0|1", "1").is_ok() && accepts("0|1", "2").is_err());
        assert!(accepts("N", "3").is_ok() && accepts("N", "three").is_err() && accepts("TOKENS", "-1").is_err());
        // a placeholder among the alternatives: not an enumeration
        assert!(accepts("DIR|auto", "/models/eagle").is_ok() && accepts("FILE", "x.gguf").is_ok() && accepts("X", "0.5").is_ok());
        assert!(accepts("N", " ").is_err());
        assert!(sizes_memory("GIB") && sizes_memory("TOKENS") && !sizes_memory("N") && !sizes_memory("rows|columns"));
    }

    #[test]
    fn a_variants_settings_are_its_engines_load_options() {
        const DECL: &[EngineOption] = &[
            EngineOption { name: "drafts", env: "NS_X_DRAFTS", value: "N", help: "", at: At::Load },
            EngineOption { name: "placement", env: "NS_X_PLACE", value: "spread|pack", help: "", at: At::Both },
            EngineOption { name: "chunk", env: "NS_X_CHUNK", value: "TOKENS", help: "", at: At::Load },
            EngineOption { name: "seed", env: "", value: "N", help: "", at: At::Request },
        ];
        let s = |k: &str, v: &str| vec![(k.to_string(), v.to_string())];
        assert_eq!(variant_settings(&s("drafts", "3"), DECL, "b65").unwrap(), vec![("NS_X_DRAFTS".into(), "3".into(), "drafts")]);
        assert_eq!(variant_settings(&s("placement", "pack"), DECL, "b65").unwrap()[0].0, "NS_X_PLACE");
        let e = variant_settings(&s("draft", "3"), DECL, "b65").unwrap_err();
        assert!(e.contains("b65") && e.contains("draft is not an option") && e.contains("drafts, placement"), "{e}");
        assert!(variant_settings(&s("placement", "scatter"), DECL, "b65").unwrap_err().contains("one of spread, pack"));
        assert!(variant_settings(&s("drafts", "many"), DECL, "b65").unwrap_err().contains("whole number"));
        assert!(variant_settings(&s("chunk", "512"), DECL, "b65").unwrap_err().contains("sizes memory"));
        assert!(variant_settings(&s("seed", "1"), DECL, "b65").unwrap_err().contains("per-request"));
        assert!(variant_settings(&[], DECL, "b65").unwrap().is_empty());
    }
}
