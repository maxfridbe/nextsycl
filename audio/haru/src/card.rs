//! The character card: Haru reads who it is from a system turn of a fixed layout (haru/prompt.py's render_card) - a
//! lead sentence, then a header line and a value for each field given (persona, name, gender, backchannel, the other
//! person's name, the relationship, the situation), in one of nine languages. The wording of each language and the
//! release's own "chat" card (its default character) are the release's assets (cards.json, Apache-2.0).

use nextsycl_audio::{Error, Result};
use serde_json::Value;

pub const FIELDS: &[&str] = &["persona", "name", "gender", "backchannel", "other_name", "relationship", "situation"];

fn assets() -> &'static Value {
    static A: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    A.get_or_init(|| serde_json::from_str(include_str!("cards.json")).unwrap_or(Value::Null))
}

/// The languages a card is written in
pub fn languages() -> Vec<String> {
    assets()["languages"].as_array().into_iter().flatten().filter_map(|l| l.as_str().map(str::to_string)).collect()
}

/// A language code as the cards know it ("en-US" -> "en"; unknown: English)
pub fn language(code: Option<&str>) -> String {
    let c = code.unwrap_or("en").to_lowercase();
    let c = c.get(..2).unwrap_or("en").to_string();
    if languages().contains(&c) {
        c
    } else {
        "en".into()
    }
}

/// The release's default character in a language (its "chat" card)
pub fn preset(lang: &str) -> Value {
    assets()["presets"][lang].clone()
}

/// A card's text from its fields (a JSON object of FIELDS; gender "female" / "male", backchannel "low" / "mid" /
/// "high") in a language
pub fn render(card: &Value, lang: &str) -> Result<String> {
    let t = &assets()["templates"][lang];
    if t.is_null() {
        return Err(Error(format!("card language {lang:?}: one of {}", languages().join(", "))));
    }
    if !card.is_object() {
        return Err(Error("a card is an object of persona, name, gender, backchannel, other_name, relationship, situation".into()));
    }
    let mut lines = vec![t["lead"].as_str().unwrap_or("").to_string()];
    for key in FIELDS {
        let v = card[*key].as_str().map(str::trim).unwrap_or("").to_string();
        let v = if *key == "gender" || *key == "backchannel" { t[*key][v.as_str()].as_str().unwrap_or("").to_string() } else { v };
        if !v.is_empty() {
            lines.push(t["headers"][*key].as_str().unwrap_or("").to_string());
            lines.push(v);
        }
    }
    Ok(lines.join("\n"))
}

/// What a session's settings make the system prompt: a card object (`nextsycl.card`) rendered; else `instructions` -
/// a rendered card kept as it is (it starts with a language's lead sentence), any other text taken as the persona of
/// the default card; else the default card
pub fn system_prompt(card: Option<&Value>, instructions: Option<&str>, lang: &str) -> Result<String> {
    if let Some(c) = card.filter(|c| !c.is_null()) {
        if let Some(s) = c.as_str() {
            return system_prompt(None, Some(s), lang);
        }
        let mut full = preset(lang);
        // fields not given keep the default card's, except an explicit empty string
        for (k, v) in c.as_object().into_iter().flatten() {
            if FIELDS.contains(&k.as_str()) {
                full[k] = v.clone();
            }
        }
        return render(&full, lang);
    }
    match instructions.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) if assets()["templates"].as_object().into_iter().flatten().any(|(_, t)| t["lead"].as_str().is_some_and(|l| s.starts_with(l))) => {
            Ok(s.to_string())
        }
        Some(s) => {
            let mut full = preset(lang);
            full["persona"] = Value::String(s.to_string());
            render(&full, lang)
        }
        None => render(&preset(lang), lang),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_card_renders_as_the_release_does() {
        let card = serde_json::json!({"persona": "Haru is a nurse.", "name": "Haru", "gender": "female", "other_name": "Sam", "backchannel": "nope"});
        assert_eq!(render(&card, "en").unwrap(),
                   "You are the person described below, talking with the other person by voice.\n[Your Persona]\nHaru is a nurse.\n[Your Name]\nHaru\n\
                    [Your Gender]\nfemale\n[The Other Person's Name]\nSam");
        assert!(render(&card, "xx").is_err());
        assert_eq!(language(Some("ja-JP")), "ja");
        assert_eq!(language(Some("it")), "en");
        assert_eq!(languages().len(), 9);
    }

    #[test]
    fn instructions_become_the_persona_and_cards_merge_with_the_default() {
        let p = system_prompt(None, Some("A pirate captain."), "en").unwrap();
        assert!(p.contains("[Your Persona]\nA pirate captain.") && p.contains("[Your Name]\nHaru"), "{p}");
        let p2 = system_prompt(None, Some(&p), "en").unwrap();
        assert_eq!(p, p2, "a rendered card is kept");
        let p3 = system_prompt(Some(&serde_json::json!({"name": "Kai", "other_name": "Max"})), None, "en").unwrap();
        assert!(p3.contains("[Your Name]\nKai") && p3.contains("[The Other Person's Name]\nMax") && p3.contains("[Situation]"), "{p3}");
        assert!(system_prompt(None, None, "zh").unwrap().starts_with("你"));
    }
}
