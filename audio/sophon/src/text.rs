//! The brain's words: its prompts (the thinker's chat template, as the release's duplex.py renders them), the turn
//! question DuplexJev answers with one letter, and what of its reply the talker may speak - the sound tags it performs
//! kept, everything else in brackets dropped (proc.py's tag_clean), nothing from a tool call on (tools.py's cut_tool).

use serde_json::Value;

/// The system prompts the thinker was tuned with (config/system_prompt.json)
/// (upstream duplex.py's system prompt held every answer to "one or two sentences": asked for a story, it declined;
/// 2026-11-19 it keeps that as the usual length and gives a long request all of it - a 330 s story, 1,446 tokens)
pub fn system(lang: &str) -> &'static str {
    if lang == "zh" {
        "你是智子（Sophon），AI降临派（Adventists AI）训练的 Sophon 系列模型，能听、能看，也能说。你会直接听到用户的语音，你的回复会被直接合成为语音说出来。请像真人对话一样回应用户，注意用户的情绪：通常用自然的口语一两句话；但当用户要的是长内容——讲故事、讲解、列清单、菜谱——就完整地说完，需要多长就说多长。只写要说出口的话，不要写括号说明、表情或格式符号；数字、日期、金额、符号和英文缩写都直接写成口语念法。需要时可以用声音标签表现笑声、咳嗽和重读。"
    } else {
        "You are Sophon, a model of the Sophon series trained by Adventists AI (AI降临派); you can hear, see and speak. You hear the user's speech directly, \
         and your reply will be spoken aloud. Respond the way a person would in conversation, paying attention to the user's emotion: usually in one or two \
         natural spoken sentences, but when the user asks for something long - a story, an explanation, a list, a recipe - give all \
         of it, as long as it needs to be. Write only the words to be spoken: no parenthetical notes, emoji or formatting; write numbers and symbols the way they are \
         read aloud. When it fits, you may use sound tags for laughter, coughs and emphasis."
    }
}

/// What the system prompt gains when tools are offered (tools.py's SYS_TOOL)
pub fn system_tools(lang: &str) -> &'static str {
    if lang == "zh" {
        "你可以调用上面列出的工具。需要用工具时：先用一句很短的口语告诉用户你要做什么（比如“我帮你查一下”），然后立刻按格式输出工具调用；工具名、参数和格式都不会被念出来，也不要在口语里复述它们。拿到工具结果后，用一两句自然的口语把结果告诉用户。闲聊或你自己就能回答的问题，不要调用工具。"
    } else {
        "You can call the tools listed above. When you need one: first say one very short spoken phrase about what you're doing \
         (e.g. \"Let me check that\"), then immediately output the tool call in the required format; the tool name, arguments and format \
         are never spoken, so don't repeat them out loud. When the tool result comes back, tell the user the result in one or two natural \
         spoken sentences. For chit-chat or anything you can answer yourself, don't call a tool."
    }
}

/// The audio placeholder (one a clip, expanded to its rows)
pub const AUDIO: &str = "<|audio|>";
/// What an interrupted reply's text ends with in the history (duplex.py)
pub const CUT: &str = "……（被用户打断）";

/// The turn question (duplex.py's turn_question; the options in livekit-plugins-duplexjev's order, shuffled by its
/// fixed permutation): its text, and the letters of "finished" and "backchannel"
pub fn turn_question(lang: &str, said: &str) -> (String, usize, usize) {
    // perm_of("turn", options) for each language (Python's random.Random seeded by blake2b, computed once)
    let (q, opts, perm, tpl): (&str, [&str; 4], [usize; 4], [&str; 5]) = if lang == "zh" {
        ("用户现在处于什么话轮状态？", ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"], [1, 3, 2, 0],
         ["问题：{q}", "选项：", "请只回答正确选项的字母。", "用户说：<|audio|>", "助手刚才说的是：“{c}”"])
    } else {
        ("Has the user finished speaking?", ["finished, the assistant can reply", "not finished, still talking", "just a backchannel, not taking the turn",
                                            "hesitating or asking to wait"], [2, 1, 0, 3],
         ["Question: {q}", "Options:", "Answer with only the letter of the correct option.", "The user said: <|audio|>", "The assistant had just said: \"{c}\""])
    };
    let s: String = said.split_whitespace().collect::<Vec<_>>().join(" ");
    let s: String = s.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
    let mut out = String::new();
    if !s.is_empty() {
        out += &tpl[4].replace("{c}", &s);
        out += "\n\n";
    }
    out += tpl[3];
    out += "\n\n";
    let mut lines = vec![tpl[0].replace("{q}", q), String::new(), tpl[1].to_string()];
    for (i, j) in perm.iter().enumerate() {
        lines.push(format!("{}. {}", (b'A' + i as u8) as char, opts[*j]));
    }
    lines.push(String::new());
    lines.push(tpl[2].to_string());
    out += &lines.join("\n");
    let finished = perm.iter().position(|&j| j == 0).unwrap_or(0);
    let back = perm.iter().position(|&j| j == 2).unwrap_or(0);
    (out, finished, back)
}

/// A chat message for the template: its role ("system", "user", "assistant", "tool"), text, and an assistant's tool
/// calls ((name, arguments JSON))
#[derive(Clone, Debug, Default)]
pub struct Msg {
    pub role: &'static str,
    pub text: String,
    pub calls: Vec<(String, String)>,
}

impl Msg {
    pub fn new(role: &'static str, text: impl Into<String>) -> Msg {
        Msg { role, text: text.into(), calls: Vec::new() }
    }
}

/// The thinker's chat template (chat_template.jinja: Qwen3-VL's, no thinking) with the generation prompt; `tools`
/// the functions offered (Chat Completions' {"type": "function", "function": {...}}), as JSON lines
pub fn render(msgs: &[Msg], tools: &[Value]) -> String {
    let mut out = String::new();
    let mut rest = msgs;
    if !tools.is_empty() {
        out += "<|im_start|>system\n";
        if let Some(m) = msgs.first().filter(|m| m.role == "system") {
            out += &m.text;
            out += "\n\n";
            rest = &msgs[1..];
        }
        out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for t in tools {
            out += "\n";
            out += &tool_json(t);
        }
        out += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
    } else if let Some(m) = msgs.first().filter(|m| m.role == "system") {
        out += "<|im_start|>system\n";
        out += &m.text;
        out += "<|im_end|>\n";
        rest = &msgs[1..];
    }
    for (i, m) in rest.iter().enumerate() {
        match m.role {
            "user" => {
                out += "<|im_start|>user\n";
                out += &m.text;
                out += "<|im_end|>\n";
            }
            "assistant" => {
                out += "<|im_start|>assistant\n";
                out += &m.text;
                for (k, (name, args)) in m.calls.iter().enumerate() {
                    if k > 0 || !m.text.is_empty() {
                        out += "\n";
                    }
                    out += "<tool_call>\n{\"name\": \"";
                    out += name;
                    out += "\", \"arguments\": ";
                    out += args;
                    out += "}\n</tool_call>";
                }
                out += "<|im_end|>\n";
            }
            "tool" => {
                if i == 0 || rest[i - 1].role != "tool" {
                    out += "<|im_start|>user";
                }
                out += "\n<tool_response>\n";
                out += &m.text;
                out += "\n</tool_response>";
                if i + 1 == rest.len() || rest[i + 1].role != "tool" {
                    out += "<|im_end|>\n";
                }
            }
            _ => {}
        }
    }
    out += "<|im_start|>assistant\n";
    out
}

/// A tool as the template's tojson writes the Chat Completions shape: {"type", "function": {"name", "description",
/// "parameters"}} in that order (the parameters' own keys sorted)
fn tool_json(t: &Value) -> String {
    let f = &t["function"];
    let mut parts = vec![format!("\"name\": {}", py_json(&f["name"]))];
    if !f["description"].is_null() {
        parts.push(format!("\"description\": {}", py_json(&f["description"])));
    }
    parts.push(format!("\"parameters\": {}", py_json(&f["parameters"])));
    format!("{{\"type\": \"function\", \"function\": {{{}}}}}", parts.join(", "))
}

/// JSON as Python's json.dumps writes it (", " and ": "), what jinja's tojson gives the template
pub fn py_json(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let parts: Vec<String> = m.iter().map(|(k, x)| format!("{}: {}", Value::String(k.clone()), py_json(x))).collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        x => x.to_string(),
    }
}

/// Tools in Chat Completions' shape: OpenAI Realtime's ({"type": "function", "name", "description", "parameters"}) or
/// Chat Completions' own (tools.py's normalize_tools)
pub fn normalize_tools(tools: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for t in tools {
        let f = if t["function"].is_object() { &t["function"] } else { t };
        let Some(name) = f["name"].as_str() else { continue };
        let mut fun = serde_json::Map::new();
        fun.insert("name".into(), Value::String(name.to_string()));
        if let Some(d) = f["description"].as_str().filter(|d| !d.is_empty()) {
            fun.insert("description".into(), Value::String(d.to_string()));
        }
        fun.insert("parameters".into(), if f["parameters"].is_object() { f["parameters"].clone() } else { serde_json::json!({"type": "object", "properties": {}}) });
        out.push(serde_json::json!({"type": "function", "function": Value::Object(fun)}));
    }
    out
}

const TC_OPEN: &str = "<tool_call>";
const WAIT: &str = "<wait>";

/// The part of the reply that may be spoken: up to the first <tool_call> (while it streams, a trailing prefix of it
/// is held back); a reply that starts with <wait> says nothing
pub fn cut_tool(text: &str, last: bool) -> String {
    let head = text.trim_start();
    if head.starts_with(WAIT) || (!last && !head.is_empty() && WAIT.starts_with(head)) {
        return String::new();
    }
    let text = text.replace(WAIT, "");
    if let Some(i) = text.find(TC_OPEN) {
        return text[..i].to_string();
    }
    if !last {
        for k in (1..TC_OPEN.len()).rev() {
            if text.ends_with(&TC_OPEN[..k]) {
                return text[..text.len() - k].to_string();
            }
        }
    }
    text
}

/// The calls a reply makes: (name, arguments as JSON) of every well-formed <tool_call>{...}</tool_call>
pub fn tool_calls(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(TC_OPEN) {
        let after = &rest[i + TC_OPEN.len()..];
        let (body, next) = match after.find("</tool_call>") {
            Some(j) => (&after[..j], &after[j..]),
            None => (after, ""),
        };
        let b = body.trim();
        let v: Option<Value> = serde_json::from_str(b).ok().or_else(|| {
            let (i, j) = (b.find('{')?, b.rfind('}')?);
            serde_json::from_str(&b[i..=j]).ok()
        });
        if let Some(v) = v {
            if let Some(name) = v["name"].as_str() {
                let a = if v.get("arguments").is_some() { &v["arguments"] } else { &v["parameters"] };
                let a = match a {
                    Value::String(s) => serde_json::from_str(s).unwrap_or(serde_json::json!({})),
                    Value::Object(_) => a.clone(),
                    _ => serde_json::json!({}),
                };
                out.push((name.to_string(), a.to_string()));
            }
        }
        rest = next;
    }
    out
}

const POINT: [&str; 5] = ["[laughter]", "[breath]", "[sigh]", "[cough]", "[quick_breath]"];
const SPAN: [&str; 2] = ["laughter", "strong"];
/// The sound tags as the server's speech text pass keeps them (AudioEngine::markup): POINT's tags and SPAN's opening
/// and closing tags, as written (a whole span kept as one: its words untouched, no spaces put inside it) - normalised, "[sigh]" would be a pause and the word "sigh"
pub const KEEP: [&str; 4] = [r"\[(?:laughter|breath|sigh|cough|quick_breath)\]", r"<laughter>[^<\n]{0,300}</laughter>", r"<strong>[^<\n]{0,300}</strong>", r"</?(?:laughter|strong)>"];

fn close_of(c: char) -> Option<char> {
    match c {
        '[' => Some(']'),
        '<' => Some('>'),
        '(' => Some(')'),
        '（' => Some('）'),
        '【' => Some('】'),
        _ => None,
    }
}

/// The sound tags taken out of what the talker reads (POINT's tags gone, SPAN's tags gone and their words kept):
/// NS_SOPHON_TAGS=0, an opt-out. By default the talker gets them, as upstream's does (Flash's talker performs them;
/// they were read aloud on 2026-11-16 only because the registry's talker was Qwen3-TTS-12Hz-1.7B-Base's file). The
/// reply's text - the transcript the client shows - keeps them either way
pub fn without_tags(s: &str) -> String {
    let mut v = s.to_string();
    for t in POINT {
        v = v.replace(t, " ");
    }
    for t in SPAN {
        v = v.replace(&format!("<{t}>"), "").replace(&format!("</{t}>"), "");
    }
    // the spaces a tag leaves: one, and none before punctuation
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        if c == ' ' && out.ends_with(' ') {
            continue;
        }
        if ",.!?;:，。！？；：".contains(c) && out.ends_with(' ') {
            out.pop();
        }
        out.push(c);
    }
    out
}

/// The reply as the talker reads it (proc.py's tag_clean): the sound tags it performs kept ([laughter], [breath],
/// [sigh], [cough], [quick_breath], <laughter>...</laughter>, <strong>...</strong>, a strong span closed at the next
/// punctuation), other bracketed asides and markdown marks dropped, arrows and comparisons said; `last`: the reply
/// is whole (an unclosed bracket's text spoken, open spans closed) - else a bracket that may still close is held back
pub fn tag_clean(s: &str, last: bool) -> String {
    let mut out = String::new();
    let mut open: Vec<&'static str> = Vec::new();
    let mut buf: &str = s;
    let text = |v: &str, out: &mut String, open: &mut Vec<&'static str>| {
        let v = v.replace(['→', '⇒'], ", so ").replace('≥', " at least ").replace('≤', " at most ");
        let mut v = v.as_str();
        if open.contains(&"strong") {
            if let Some((i, _)) = v.char_indices().find(|(_, c)| "，。！？、,.!?;；:：".contains(*c)) {
                out.push_str(&v[..i]);
                out.push_str("</strong>");
                open.retain(|x| *x != "strong");
                v = &v[i..];
            }
        }
        out.push_str(v);
    };
    while !buf.is_empty() {
        let i = buf.char_indices().find(|(_, c)| close_of(*c).is_some() || "*#`".contains(*c)).map(|(i, _)| i);
        let Some(i) = i else {
            text(buf, &mut out, &mut open);
            buf = "";
            break;
        };
        if i > 0 {
            text(&buf[..i], &mut out, &mut open);
            buf = &buf[i..];
        }
        let c = buf.chars().next().unwrap_or(' ');
        if "*#`".contains(c) {
            buf = &buf[c.len_utf8()..];
            continue;
        }
        let close = close_of(c).unwrap_or(c);
        let after = c.len_utf8();
        // the closing bracket within 40 characters, or not a tag
        let j = buf[after..].char_indices().find(|(_, x)| *x == close).map(|(k, _)| k + after);
        let chars_to = |b: usize| buf[..b].chars().count();
        match j {
            Some(j) if chars_to(j) <= 40 => {
                let end = j + close.len_utf8();
                let tok = &buf[..end];
                buf = &buf[end..];
                if c == '[' || c == '<' {
                    if POINT.contains(&tok) {
                        out.push_str(tok);
                    } else if let Some(inner) = tok.strip_prefix('<').and_then(|t| t.strip_suffix('>')) {
                        let (closing, name) = match inner.strip_prefix('/') {
                            Some(n) => (true, n),
                            None => (false, inner),
                        };
                        if let Some(n) = SPAN.iter().find(|s| **s == name) {
                            if !closing && !open.contains(n) {
                                open.push(n);
                                out.push_str(tok);
                            } else if closing && open.contains(n) {
                                open.retain(|x| x != n);
                                out.push_str(tok);
                            }
                        }
                    }
                }
            }
            _ => {
                if j.is_some() || buf.chars().count() > 40 {
                    buf = &buf[after..];
                    continue;
                }
                // may still close: held back
                break;
            }
        }
    }
    if last {
        if !buf.is_empty() {
            let cut = buf.find(['[', '<', '(', '（', '【']).unwrap_or(buf.len());
            let t = buf[..cut].to_string();
            if !t.is_empty() {
                text(&t, &mut out, &mut open);
            }
        }
        for n in open.iter().rev() {
            out.push_str(&format!("</{n}>"));
        }
    }
    out
}

/// A reply's text for showing (the tags taken out)
pub fn plain(s: &str) -> String {
    let mut t = s.to_string();
    for p in POINT {
        t = t.replace(p, "");
    }
    for n in SPAN {
        t = t.replace(&format!("<{n}>"), "").replace(&format!("</{n}>"), "");
    }
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NS_SOPHON_TAGS=0: the talker is not handed a sound tag; the words of a span are kept
    #[test]
    fn without_tags_keeps_the_words() {
        assert_eq!(super::without_tags("Ha [laughter] that's great"), "Ha that's great");
        assert_eq!(super::without_tags("They'd be bagels! [laughter] Oh, a classic [sigh]."), "They'd be bagels! Oh, a classic.");
        assert_eq!(super::without_tags("I <strong>really</strong> mean it, <laughter>so funny</laughter>!"), "I really mean it, so funny!");
    }

    #[test]
    fn tags_are_kept_and_asides_dropped() {
        assert_eq!(tag_clean("Ha [laughter] that's (smiles) *great*", true), "Ha [laughter] that's  great");
        assert_eq!(tag_clean("I <strong>really</strong> mean it", true), "I <strong>really</strong> mean it");
        // a strong span closes at the next punctuation
        assert_eq!(tag_clean("<strong>no way, really", true), "<strong>no way</strong>, really");
        // an open bracket that may still close is held back while streaming
        assert_eq!(tag_clean("well [laugh", false), "well ");
        assert_eq!(tag_clean("well [laugh", true), "well ");
        // an unknown tag goes
        assert_eq!(tag_clean("ok [music] go", true), "ok  go");
        assert_eq!(plain("Ha [laughter] <strong>yes</strong>"), "Ha yes");
    }

    #[test]
    fn tool_calls_are_cut_and_parsed() {
        assert_eq!(cut_tool("Let me check.<tool_call>\n{\"name\": \"w\"}", false), "Let me check.");
        assert_eq!(cut_tool("Let me check.<too", false), "Let me check.");
        assert_eq!(cut_tool("Let me check.<too", true), "Let me check.<too");
        assert_eq!(cut_tool("<wa", false), "");
        let c = tool_calls("ok<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>");
        assert_eq!(c, vec![("weather".to_string(), "{\"city\":\"Paris\"}".to_string())]);
    }

    #[test]
    fn the_turn_question_is_duplex_pys() {
        let (q, fin, back) = turn_question("en", "");
        assert!(q.starts_with("The user said: <|audio|>\n\nQuestion: Has the user finished speaking?\n\nOptions:\nA. just a backchannel"));
        assert!(q.ends_with("D. hesitating or asking to wait\n\nAnswer with only the letter of the correct option."));
        assert_eq!((fin, back), (2, 0));
        let (q, _, _) = turn_question("en", "Hello  there");
        assert!(q.starts_with("The assistant had just said: \"Hello there\"\n\nThe user said:"));
        let (_, fin, back) = turn_question("zh", "");
        assert_eq!((fin, back), (3, 2));
    }

    #[test]
    fn the_template_is_the_thinkers() {
        let m = [Msg::new("system", "S"), Msg::new("user", AUDIO)];
        assert_eq!(render(&m, &[]), "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<|audio|><|im_end|>\n<|im_start|>assistant\n");
        let t = normalize_tools(&[serde_json::json!({"type": "function", "name": "w", "parameters": {"type": "object"}})]);
        let r = render(&m, &t);
        assert!(r.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"w\", \"parameters\": {\"type\": \"object\"}}}\n</tools>"), "{r}");
    }
}
