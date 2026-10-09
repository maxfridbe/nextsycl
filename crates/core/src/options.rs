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
}
