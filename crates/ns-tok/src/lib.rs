//! Byte-level BPE (GPT-2 style) from a GGUF's `tokenizer.ggml.*`, as llama.cpp runs it for GLM's `glm4` / `glm5`
//! and Qwen3.8's `qwen35` pre-tokenizers: special tokens matched in the text first (longest first), the rest split by the pre-tokenizer
//! regex, each piece's bytes mapped to GPT-2's printable alphabet, then - `ignore_merges` - taken whole when the
//! vocabulary has it, else merged by rank. No BOS token.

use std::collections::HashMap;

use fancy_regex::Regex;
use ns_gguf::{Gguf, Value};

/// llama.cpp's LLAMA_VOCAB_PRE_TYPE_CHATGLM4 (used for `glm4` and `glm5`)
const GLM4_SPLIT: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 (src/llama-vocab.cpp; the pattern Strata's tokenizer transcribes): QWEN2's
/// shape with combining marks (`\p{M}`) kept with their letters and out of the punctuation runs, single digits
const QWEN35_SPLIT: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    pub tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    /// control and user-defined tokens, longest first (matched in the text as they are)
    special: Vec<(String, u32)>,
    split: Regex,
    ignore_merges: bool,
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
pub struct Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
    /// an assistant turn's thinking, when kept
    pub reasoning: Option<&'a str>,
}

/// GLM-5.3's chat template (`tokenizer.chat_template`), its text-only, tool-free path: `[gMASK]<sop>`, the effort
/// line, the turns, then `<|assistant|><think>` to generate.
pub fn glm_chat(messages: &[Message], effort: Effort) -> String {
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

/// Qwen3.8-Flash-Next's chat template (ChatML), its text-only, tool-free path: an optional system turn, the turns
/// (an assistant turn without its old thinking, as the template drops it), then `<|im_start|>assistant\n<think>\n`.
/// The model has no effort levels: it thinks (the template's default `enable_thinking`).
pub fn qwen_chat(messages: &[Message], _effort: Effort) -> String {
    let mut s = String::new();
    for m in messages {
        match m.role {
            "system" | "user" | "assistant" => {
                s += "<|im_start|>";
                s += m.role;
                s += "\n";
                s += if m.role == "assistant" { m.content.trim() } else { m.content };
                s += "<|im_end|>\n";
            }
            _ => {}
        }
    }
    s += "<|im_start|>assistant\n<think>\n";
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwens_template() {
        let m = [Message { role: "user", content: "Hi", reasoning: None }];
        assert_eq!(qwen_chat(&m, Effort::Low), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
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
        let m = [Message { role: "user", content: "Hi", reasoning: None }];
        assert_eq!(glm_chat(&m, Effort::Low), "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hi<|assistant|><think>");
    }
}
