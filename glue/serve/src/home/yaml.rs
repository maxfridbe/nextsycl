//! A JSON value as YAML (block style): maps as `key: value`, lists as `- item`, strings double-quoted (JSON's
//! escaping is YAML's), so the OpenAPI document reads as the `.yml` people expect.

use serde_json::Value;

fn key(k: &str) -> String {
    // plain when it cannot read as anything but a string (a number - 200 - stays a quoted key)
    if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || "_-./".contains(c)) && !k.starts_with(['-', '.'])
        && k.parse::<f64>().is_err() && !matches!(k, "true" | "false" | "null" | "yes" | "no" | "on" | "off") {
        k.to_string()
    } else {
        Value::String(k.to_string()).to_string()
    }
}

fn scalar(v: &Value) -> String {
    match v {
        Value::String(s) => Value::String(s.clone()).to_string(),
        other => other.to_string(),
    }
}

fn empty(v: &Value) -> Option<&'static str> {
    match v {
        Value::Object(m) if m.is_empty() => Some("{}"),
        Value::Array(a) if a.is_empty() => Some("[]"),
        _ => None,
    }
}

fn emit(v: &Value, ind: usize, out: &mut String) {
    let pad = " ".repeat(ind);
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                match (x, empty(x)) {
                    (_, Some(e)) => out.push_str(&format!("{pad}{}: {e}\n", key(k))),
                    (Value::Object(_) | Value::Array(_), _) => {
                        out.push_str(&format!("{pad}{}:\n", key(k)));
                        emit(x, ind + 2, out);
                    }
                    _ => out.push_str(&format!("{pad}{}: {}\n", key(k), scalar(x))),
                }
            }
        }
        Value::Array(a) => {
            for x in a {
                match (x, empty(x)) {
                    (_, Some(e)) => out.push_str(&format!("{pad}- {e}\n")),
                    (Value::Object(_) | Value::Array(_), _) => {
                        // the item's first line after the dash, the rest under it
                        let mut inner = String::new();
                        emit(x, ind + 2, &mut inner);
                        let body = inner.strip_prefix(&" ".repeat(ind + 2)).unwrap_or(&inner);
                        out.push_str(&format!("{pad}- {body}"));
                    }
                    _ => out.push_str(&format!("{pad}- {}\n", scalar(x))),
                }
            }
        }
        other => out.push_str(&format!("{pad}{}\n", scalar(other))),
    }
}

pub fn to_yaml(v: &Value) -> String {
    let mut out = String::new();
    emit(v, 0, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn block_yaml() {
        let y = to_yaml(&json!({"openapi": "3.0.3", "paths": {"/rpc/a.b": {"post": {"tags": ["x"], "x": []}}}, "n": 1, "l": [{"a": 1, "b": "q\""}, 2]}));
        assert_eq!(y, "l:\n  - a: 1\n    b: \"q\\\"\"\n  - 2\nn: 1\nopenapi: \"3.0.3\"\npaths:\n  /rpc/a.b:\n    post:\n      tags:\n        - \"x\"\n      x: []\n");
    }
}
