//! Byte-level BPE (GPT-2 style) from a GGUF's `tokenizer.ggml.*`, as llama.cpp runs it for GLM's `glm4` / `glm5`
//! and Qwen3.8's `qwen35` pre-tokenizers: special tokens matched in the text first (longest first), the rest split by the pre-tokenizer
//! regex, each piece's bytes mapped to GPT-2's printable alphabet, then - `ignore_merges` - taken whole when the
//! vocabulary has it, else merged by rank. No BOS token.

use std::collections::HashMap;

use fancy_regex::Regex;
use nextsycl_gguf::{Gguf, Value};

/// llama.cpp's LLAMA_VOCAB_PRE_TYPE_CHATGLM4 (used for `glm4` and `glm5`)
const GLM4_SPLIT: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 (src/llama-vocab.cpp; the pattern Strata's tokenizer transcribes): QWEN2's
/// shape with combining marks (`\p{M}`) kept with their letters and out of the punctuation runs, single digits
const QWEN35_SPLIT: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// transformers' Qwen2Tokenizer (the slow one, from vocab.json and merges.txt): case-insensitive contractions,
/// letters, single digits
const QWEN2_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    pub tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    /// control and user-defined tokens, longest first (matched in the text as they are)
    special: Vec<(String, u32)>,
    split: Regex,
    ignore_merges: bool,
    /// SentencePiece-style BPE (a Llama tokenizer.json: no pre-tokenizer, "▁" for spaces and ahead of the text, merges
    /// over the whole text, unknown characters as their bytes' <0xNN> tokens) instead of GPT-2's byte-level one
    sentencepiece: bool,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pub eos: Option<u32>,
    /// the tokens that end a turn
    pub stop: Vec<u32>,
}

/// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the rest are mapped above U+0100.
fn byte_alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

fn strs(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

impl Tokenizer {
    /// A Hugging Face `tokenizer.json` (byte-level BPE: its vocabulary, merges, added tokens and the pre-tokenizer's
    /// split pattern) - the text encoders of image and video models ship it so (Qwen3-VL's)
    pub fn from_hf_json(path: &std::path::Path) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let j: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let model = &j["model"];
        if model["type"].as_str() != Some("BPE") {
            return Err(format!("{}: only BPE tokenizers are read", path.display()));
        }
        let vocab = model["vocab"].as_object().ok_or("tokenizer.json: no model.vocab")?;
        let mut ids: HashMap<String, u32> = vocab.iter().filter_map(|(t, i)| Some((t.clone(), i.as_u64()? as u32))).collect();
        let mut special: Vec<(String, u32)> = Vec::new();
        for a in j["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(c), Some(i)) = (a["content"].as_str(), a["id"].as_u64()) {
                ids.insert(c.to_string(), i as u32);
                special.push((c.to_string(), i as u32));
            }
        }
        special.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let n = ids.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut tokens = vec![String::new(); n];
        for (t, i) in &ids {
            tokens[*i as usize] = t.clone();
        }
        let ranks = model["merges"].as_array().into_iter().flatten().enumerate().filter_map(|(r, m)| {
            let (a, b) = match m {
                serde_json::Value::String(s) => s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))?,
                serde_json::Value::Array(p) => (p.first()?.as_str()?.to_string(), p.get(1)?.as_str()?.to_string()),
                _ => return None,
            };
            Some(((a, b), r))
        }).collect();
        // SentencePiece's shape: a normalizer that puts "▁" ahead of the text, no pre-tokenizer, byte fallback
        let sentencepiece = j["pre_tokenizer"].is_null() && model["byte_fallback"] == true
            && j["normalizer"].to_string().contains("\"Prepend\"");
        // the first Split pre-tokenizer's pattern (Qwen's: case-insensitive contractions, letters, single digits...)
        let pattern = match sentencepiece {
            true => ".+",
            false => j["pre_tokenizer"]["pretokenizers"].as_array().into_iter().flatten()
                .chain(std::iter::once(&j["pre_tokenizer"]))
                .find_map(|p| (p["type"] == "Split").then(|| p["pattern"]["Regex"].as_str()).flatten())
                .ok_or("tokenizer.json: no Split pre-tokenizer pattern")?,
        };
        let byte_enc = byte_alphabet();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let mut stop: Vec<u32> = ["<|endoftext|>", "<|im_end|>"].iter().filter_map(|s| ids.get(*s).copied()).collect();
        stop.sort();
        Ok(Tokenizer {
            eos: ids.get("<|im_end|>").copied(),
            tokens,
            ids,
            ranks,
            special,
            split: Regex::new(pattern).map_err(|e| e.to_string())?,
            ignore_merges: model["ignore_merges"].as_bool().unwrap_or(false),
            sentencepiece,
            byte_enc,
            byte_dec,
            stop,
        })
    }

    /// Qwen2Tokenizer's files: `vocab.json`, `merges.txt` (one merge a line after the `#version` header) and
    /// `tokenizer_config.json`'s added tokens (matched as they are). That tokenizer NFC-normalizes the text first: the
    /// caller does so (this crate has no Unicode tables).
    pub fn from_vocab_merges(dir: &std::path::Path) -> Result<Tokenizer, String> {
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).map_err(|e| format!("{}: {e}", dir.join(f).display()));
        let vocab: serde_json::Value = serde_json::from_str(&read("vocab.json")?).map_err(|e| format!("vocab.json: {e}"))?;
        let mut ids: HashMap<String, u32> = vocab.as_object().ok_or("vocab.json: not an object")?.iter()
            .filter_map(|(t, i)| Some((t.clone(), i.as_u64()? as u32))).collect();
        let conf: serde_json::Value = serde_json::from_str(&read("tokenizer_config.json")?).map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let mut special: Vec<(String, u32)> = conf["added_tokens_decoder"].as_object().into_iter().flatten()
            .filter_map(|(i, t)| Some((t["content"].as_str()?.to_string(), i.parse::<u32>().ok()?))).collect();
        for (c, i) in &special {
            ids.insert(c.clone(), *i);
        }
        special.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let n = ids.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut tokens = vec![String::new(); n];
        for (t, i) in &ids {
            tokens[*i as usize] = t.clone();
        }
        let merges = read("merges.txt")?;
        let ranks = merges.lines().filter(|l| !l.starts_with("#version") && !l.is_empty()).enumerate()
            .filter_map(|(r, l)| l.split_once(' ').map(|(a, b)| ((a.to_string(), b.to_string()), r))).collect();
        let byte_enc = byte_alphabet();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let mut stop: Vec<u32> = ["<|endoftext|>", "<|im_end|>"].iter().filter_map(|s| ids.get(*s).copied()).collect();
        stop.sort();
        Ok(Tokenizer {
            eos: ids.get("<|im_end|>").copied(),
            tokens,
            ids,
            ranks,
            special,
            split: Regex::new(QWEN2_SPLIT).map_err(|e| e.to_string())?,
            ignore_merges: false,
            sentencepiece: false,
            byte_enc,
            byte_dec,
            stop,
        })
    }

    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        if g.meta("tokenizer.ggml.model").and_then(Value::as_str) != Some("gpt2") {
            return Err("only byte-level BPE (tokenizer.ggml.model gpt2) is read".into());
        }
        let pre = g.meta("tokenizer.ggml.pre").and_then(Value::as_str).unwrap_or("");
        let split = match pre {
            "glm4" | "glm5" | "chatglm-bpe" => GLM4_SPLIT,
            "qwen35" => QWEN35_SPLIT,
            _ => return Err(format!("pre-tokenizer {pre:?} is not one this tokenizer has (glm4, qwen35)")),
        };
        let tokens = strs(g.meta("tokenizer.ggml.tokens"));
        let types: Vec<u64> = g.meta("tokenizer.ggml.token_type").and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_u64().unwrap_or(1)).collect()).unwrap_or_default();
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let ranks = strs(g.meta("tokenizer.ggml.merges"))
            .into_iter()
            .enumerate()
            .filter_map(|(i, m)| m.split_once(' ').map(|(a, b)| ((a.to_string(), b.to_string()), i)))
            .collect();
        // 3 = control, 4 = user-defined: matched literally
        let mut special: Vec<(String, u32)> = tokens.iter().enumerate()
            .filter(|(i, t)| matches!(types.get(*i), Some(3) | Some(4)) && !t.is_empty() && !t.starts_with("[PAD"))
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        special.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let byte_enc = byte_alphabet();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let id_of = |k: &str| g.meta(k).and_then(Value::as_u64).map(|v| v as u32);
        let mut stop: Vec<u32> = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"].iter().filter_map(|k| id_of(k)).collect();
        for s in ["<|user|>", "<|observation|>", "<|endoftext|>", "<|im_end|>"] {
            if let Some(i) = ids.get(s) {
                stop.push(*i);
            }
        }
        stop.sort();
        stop.dedup();
        Ok(Tokenizer {
            eos: id_of("tokenizer.ggml.eos_token_id"),
            tokens,
            ids,
            ranks,
            special,
            split: Regex::new(split).map_err(|e| e.to_string())?,
            // llama.cpp takes a piece the vocabulary has whole for GLM's (not the old chatglm-bpe); Qwen's merge by rank
            ignore_merges: matches!(pre, "glm4" | "glm5"),
            sentencepiece: false,
            byte_enc,
            byte_dec,
            stop,
        })
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    /// Text to ids; special tokens in the text become their ids (as llama.cpp's parse_special).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // the earliest special token (longest at a position wins: the list is sorted longest first)
            let next = self.special.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id))).min_by_key(|(p, l, _)| (*p, std::cmp::Reverse(*l)));
            match next {
                Some((p, l, id)) => {
                    self.encode_plain(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + l..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if self.sentencepiece {
            if text.is_empty() {
                return;
            }
            let t = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
            let mut parts = Vec::new();
            self.bpe_parts(&t, &mut parts);
            for p in parts {
                match self.ids.get(&p) {
                    Some(id) => out.push(*id),
                    // byte fallback
                    None => out.extend(p.bytes().filter_map(|b| self.ids.get(&format!("<0x{b:02X}>")).copied())),
                }
            }
            return;
        }
        for m in self.split.find_iter(text).flatten() {
            let piece: String = m.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            if self.ignore_merges {
                if let Some(id) = self.ids.get(&piece) {
                    out.push(*id);
                    continue;
                }
            }
            self.bpe(&piece, out);
        }
    }

    /// Merges by rank over a piece's characters: the parts left
    fn bpe_parts(&self, piece: &str, out: &mut Vec<String>) {
        let mut parts: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        loop {
            let best = (0..parts.len().saturating_sub(1))
                .filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|r| (*r, i)))
                .min();
            let Some((_, i)) = best else { break };
            let joined = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [joined]);
        }
        out.extend(parts);
    }

    /// Merges by rank: repeatedly join the adjacent pair with the lowest merge rank.
    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        loop {
            let best = (0..parts.len().saturating_sub(1))
                .filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|r| (*r, i)))
                .min();
            let Some((_, i)) = best else { break };
            let joined = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [joined]);
        }
        for p in parts {
            match self.ids.get(&p) {
                Some(id) => out.push(*id),
                // a part outside the vocabulary: its characters one by one (each byte is a token in GPT-2 vocabularies)
                None => out.extend(p.chars().filter_map(|c| self.ids.get(&c.to_string()).copied())),
            }
        }
    }

    /// Ids to UTF-8 bytes (special tokens as their text). Bytes, not a String: a token can end inside a character.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        for id in ids {
            let Some(t) = self.tokens.get(*id as usize) else { continue };
            if self.special.iter().any(|(_, s)| s == id) {
                v.extend(t.as_bytes());
            } else {
                v.extend(t.chars().filter_map(|c| self.byte_dec.get(&c).copied()));
            }
        }
        v
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// How much GLM-5.3 thinks before answering (its template's `reasoning_effort`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    Low,
    High,
    Max,
}

impl Effort {
    pub fn parse(s: &str) -> Option<Effort> {
        match s {
            "low" => Some(Effort::Low),
            "high" => Some(Effort::High),
            "max" => Some(Effort::Max),
            _ => None,
        }
    }
}

/// One message of a conversation.
#[derive(Default)]
pub struct Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
    /// an assistant turn's thinking, when kept
    pub reasoning: Option<&'a str>,
    /// an assistant turn's tool calls
    pub calls: &'a [ToolCall],
}

/// A tool call of an assistant turn: the function and its arguments, each value as the template writes it (a string
/// as itself, anything else as JSON)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: Vec<(String, String)>,
}

/// GLM-5.3's chat template (`tokenizer.chat_template`), its text-only, tool-free path: `[gMASK]<sop>`, the effort
/// line, the turns, then `<|assistant|><think>` to generate.
pub fn glm_chat(messages: &[Message], effort: Effort, _tools: &[String]) -> String {
    let mut s = String::from("[gMASK]<sop>");
    s += match effort {
        Effort::Low => "<|system|>Reasoning Effort: Low",
        Effort::High => "<|system|>Reasoning Effort: High",
        Effort::Max => "<|system|>Reasoning Effort: Max",
    };
    for m in messages {
        match m.role {
            "system" => {
                s += "<|system|>";
                s += m.content;
            }
            "user" => {
                s += "<|user|>";
                s += m.content;
            }
            "assistant" => {
                s += "<|assistant|><think>";
                s += m.reasoning.unwrap_or("");
                s += "</think>";
                s += m.content.trim();
            }
            _ => {}
        }
    }
    s += "<|assistant|><think>";
    s
}

/// Qwen3.8-Flash-Next's chat template (ChatML): an optional system turn, the turns (an assistant turn without its old
/// thinking, as the template drops it), then `<|im_start|>assistant\n<think>\n`. The model has no effort levels: it
/// thinks (the template's default `enable_thinking`). With `tools` (each a tool's JSON as the template's `tojson`
/// writes it) the template's tool path: a system turn listing them with the call format (the leading system messages
/// merged after it), an assistant turn's calls as `<tool_call><function=..><parameter=..>` blocks, and the `tool`
/// messages as `<tool_response>` blocks in one user turn.
pub fn qwen_chat(messages: &[Message], _effort: Effort, tools: &[String]) -> String {
    let mut s = String::new();
    let mut first = 0;
    if !tools.is_empty() {
        let mut sys = Vec::new();
        while first < messages.len() && matches!(messages[first].role, "system" | "developer") {
            let c = messages[first].content.trim();
            if !c.is_empty() {
                sys.push(c);
            }
            first += 1;
        }
        s += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for t in tools {
            s += "\n";
            s += t;
        }
        s += "\n</tools>";
        s += QWEN_TOOL_FORMAT;
        if !sys.is_empty() {
            s += "\n\n";
            s += &sys.join("\n");
        }
        s += "<|im_end|>\n";
    }
    let ms = &messages[first..];
    for (k, m) in ms.iter().enumerate() {
        match m.role {
            "system" | "user" => {
                s += "<|im_start|>";
                s += m.role;
                s += "\n";
                s += m.content;
                s += "<|im_end|>\n";
            }
            "assistant" => {
                s += "<|im_start|>assistant\n";
                let c = m.content.trim();
                s += c;
                for (i, call) in m.calls.iter().enumerate() {
                    s += if i > 0 { "\n<tool_call>\n<function=" } else if c.is_empty() { "<tool_call>\n<function=" } else { "\n\n<tool_call>\n<function=" };
                    s += &call.name;
                    s += ">\n";
                    for (k, v) in &call.args {
                        s += "<parameter=";
                        s += k;
                        s += ">\n";
                        s += v;
                        s += "\n</parameter>\n";
                    }
                    s += "</function>\n</tool_call>";
                }
                s += "<|im_end|>\n";
            }
            "tool" => {
                if k == 0 || ms[k - 1].role != "tool" {
                    s += "<|im_start|>user";
                }
                s += "\n<tool_response>\n";
                s += m.content.trim();
                s += "\n</tool_response>";
                if k + 1 == ms.len() || ms[k + 1].role != "tool" {
                    s += "<|im_end|>\n";
                }
            }
            _ => {}
        }
    }
    s += "<|im_start|>assistant\n<think>\n";
    s
}

/// The template's call-format instructions after the tool list
const QWEN_TOOL_FORMAT: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The tool calls in a Qwen answer (`<tool_call><function=NAME><parameter=P>\nvalue\n</parameter>...`): the text
/// before the first call, and each call's name and raw parameter values (a value's single leading and trailing
/// newline dropped, as the template writes them)
pub fn qwen_tool_calls(answer: &str) -> (String, Vec<ToolCall>) {
    let Some(start) = answer.find("<tool_call>") else { return (answer.to_string(), Vec::new()) };
    let mut calls = Vec::new();
    let mut rest = &answer[start..];
    while let Some(i) = rest.find("<tool_call>") {
        let body_start = i + "<tool_call>".len();
        let end = rest[body_start..].find("</tool_call>").map_or(rest.len(), |e| body_start + e);
        let body = &rest[body_start..end];
        if let Some(f) = body.find("<function=") {
            let after = &body[f + "<function=".len()..];
            if let Some(gt) = after.find('>') {
                let name = after[..gt].trim().to_string();
                let mut args = Vec::new();
                let mut p = &after[gt + 1..];
                while let Some(a) = p.find("<parameter=") {
                    let q = &p[a + "<parameter=".len()..];
                    let Some(gt) = q.find('>') else { break };
                    let key = q[..gt].trim().to_string();
                    let v = &q[gt + 1..];
                    let close = v.find("</parameter>").or_else(|| v.find("<parameter=")).or_else(|| v.find("</function>")).unwrap_or(v.len());
                    let mut val = &v[..close];
                    val = val.strip_prefix('\n').unwrap_or(val);
                    val = val.strip_suffix('\n').unwrap_or(val);
                    args.push((key, val.to_string()));
                    p = &v[close..];
                }
                if !name.is_empty() {
                    calls.push(ToolCall { name, args });
                }
            }
        }
        rest = &rest[(end + "</tool_call>".len()).min(rest.len())..];
    }
    (answer[..start].trim_end().to_string(), calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwens_template() {
        let m = [Message { role: "user", content: "Hi", ..Default::default() }];
        assert_eq!(qwen_chat(&m, Effort::Low, &[]), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn qwens_tool_turns() {
        let calls = [ToolCall { name: "get_weather".into(), args: vec![("city".into(), "Paris".into()), ("days".into(), "3".into())] }];
        let m = [Message { role: "system", content: "Be brief.", ..Default::default() },
                 Message { role: "user", content: "Weather?", ..Default::default() },
                 Message { role: "assistant", content: "", calls: &calls, ..Default::default() },
                 Message { role: "tool", content: "sunny", ..Default::default() }];
        let s = qwen_chat(&m, Effort::Low, &["{\"type\": \"function\"}".to_string()]);
        assert!(s.starts_with("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\"}\n</tools>"));
        assert!(s.contains("</IMPORTANT>\n\nBe brief.<|im_end|>\n<|im_start|>user\nWeather?<|im_end|>\n"));
        assert!(s.contains("<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>\n"));
        assert!(s.ends_with("<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn qwens_tool_calls_parse() {
        let a = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=note>\ntwo\nlines\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n</tool_call>";
        let (text, calls) = qwen_tool_calls(a);
        assert_eq!(text, "Let me check.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].args, vec![("city".to_string(), "Paris".to_string()), ("note".to_string(), "two\nlines".to_string())]);
        assert_eq!(calls[1], ToolCall { name: "now".into(), args: vec![] });
        assert_eq!(qwen_tool_calls("plain"), ("plain".to_string(), vec![]));
    }

    #[test]
    fn qwens_split_keeps_marks_with_letters() {
        let r = Regex::new(QWEN35_SPLIT).unwrap();
        let t = "cafe\u{301} 12 x";
        let parts: Vec<&str> = r.find_iter(t).map(|m| m.unwrap().as_str()).collect();
        assert_eq!(parts, ["cafe\u{301}", " ", "1", "2", " x"]);
    }

    #[test]
    fn the_byte_alphabet_is_gpt2s() {
        let a = byte_alphabet();
        assert_eq!(a[b'A' as usize], 'A');
        assert_eq!(a[b' ' as usize], '\u{120}'); // Ġ
        assert_eq!(a[b'\n' as usize], '\u{10a}'); // Ċ
        let mut seen = std::collections::HashSet::new();
        assert!(a.iter().all(|c| seen.insert(*c)));
    }

    #[test]
    fn the_split_keeps_the_text() {
        let r = Regex::new(GLM4_SPLIT).unwrap();
        let t = "The capital of France is  Paris.\n\nIt's 12345 km   away!";
        let parts: Vec<&str> = r.find_iter(t).map(|m| m.unwrap().as_str()).collect();
        assert_eq!(parts.concat(), t);
        assert_eq!(&parts[..4], &["The", " capital", " of", " France"]);
        assert!(parts.contains(&"123") && parts.contains(&"45"));
    }

    #[test]
    fn the_template() {
        let m = [Message { role: "user", content: "Hi", ..Default::default() }];
        assert_eq!(glm_chat(&m, Effort::Low, &[]), "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hi<|assistant|><think>");
    }
}

#[cfg(test)]
mod hf_json_tests {
    /// Qwen3-VL's tokenizer.json against the ids the reference pipeline made (NS_TEST_QWEN_TOKENIZER: the file,
    /// NS_TEST_QWEN_TOKENS: the reference's tokens.json); skipped without them
    #[test]
    fn qwen3vl_tokens_match_the_reference() {
        let (Ok(tj), Ok(refj)) = (std::env::var("NS_TEST_QWEN_TOKENIZER"), std::env::var("NS_TEST_QWEN_TOKENS")) else { return };
        let t = super::Tokenizer::from_hf_json(std::path::Path::new(&tj)).unwrap();
        let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refj).unwrap()).unwrap();
        let want: Vec<u32> = r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let text = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\nA red fox sitting in fresh snow, morning light, photograph<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(t.encode(text), want);
    }
}
