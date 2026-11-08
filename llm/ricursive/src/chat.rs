//! Ripple-65B's chat template (the GGUF's `tokenizer.chat_template`, ricursiveAI's chat_template.jinja), its text path:
//! ChatML turns (`<|im_start|>role\n...<|im_end|>\n`); with tools a system turn that lists them (each tool's JSON as
//! Python's `tojson` writes it) and asks for calls as `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`;
//! an assistant turn's kept thinking in front of its answer as `<think>\n...\n</think>\n`; the `tool` messages after a
//! call as one `user` turn of `<tool_response>`s. The prompt for the next turn ends with `<|im_start|>assistant\n`:
//! the model opens its own `<think>` when it wants one.
//!
//! Prompt caching (CONTRIBUTING.md rule 1): user turns begin with `<|im_start|>user`, a marker in
//! `nextsycl_tok::USER_TURNS`; what comes before a conversation's last user turn is rendered the same on the next
//! turn (nothing in it depends on what follows), so the server's checkpoint at that turn is a prefix of the next prompt.

use nextsycl_tok::{py_json, Effort, Message, ToolCall};

const TOOLS_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within \
                          <tools> XML tags:\n<tools>";
const TOOLS_TAIL: &str = "\n</tools>\n\nFor each function call, return exactly one valid JSON object with function name and arguments inside XML tool-call tags. \
                          The arguments field must be a JSON object, not a string, and strings must use valid JSON escaping:\n<tool_call>\n\
                          {\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>";

/// A call's arguments as the JSON object the template writes: a value that is JSON of another kind than a string
/// (a number, a boolean, null, a list, an object) as that, anything else as the string it is
fn arguments_json(call: &ToolCall) -> String {
    let fields: Vec<String> = call.args.iter().map(|(k, v)| {
        let val = match serde_json::from_str::<serde_json::Value>(v.trim()) {
            Ok(j) if !j.is_string() => j,
            _ => serde_json::Value::String(v.clone()),
        };
        format!("{}: {}", serde_json::Value::String(k.clone()), py_json(&val))
    }).collect();
    format!("{{{}}}", fields.join(", "))
}

/// A request's tool as the template's `tool | tojson` writes it
fn tool_json(tool: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(tool) {
        Ok(v) => py_json(&v),
        Err(_) => tool.to_string(),
    }
}

pub fn ripple_chat(messages: &[Message], _effort: Effort, tools: &[String]) -> String {
    let mut s = String::new();
    let first_system = messages.first().is_some_and(|m| m.role == "system");
    if !tools.is_empty() {
        s += "<|im_start|>system\n";
        if first_system {
            s += messages[0].content;
            s += "\n\n";
        }
        s += TOOLS_HEAD;
        for t in tools {
            s += "\n";
            s += &tool_json(t);
        }
        s += TOOLS_TAIL;
        s += "<|im_end|>\n";
    } else if first_system {
        s += "<|im_start|>system\n";
        s += messages[0].content;
        s += "<|im_end|>\n";
    }
    for (k, m) in messages.iter().enumerate() {
        let mut content = m.content.to_string();
        if m.role == "assistant" {
            if let Some(r) = m.reasoning.filter(|r| !r.is_empty()) {
                if !content.trim_start().starts_with("<think>") {
                    content = format!("<think>\n{r}\n</think>\n{content}");
                }
            }
        }
        match m.role {
            "user" => {
                s += "<|im_start|>user\n";
                s += &content;
                s += "<|im_end|>\n";
            }
            "system" if k > 0 => {
                s += "<|im_start|>system\n";
                s += &content;
                s += "<|im_end|>\n";
            }
            "assistant" if m.calls.is_empty() => {
                s += "<|im_start|>assistant\n";
                s += &content;
                s += "<|im_end|>\n";
            }
            "assistant" => {
                s += "<|im_start|>assistant";
                if !content.is_empty() {
                    s += "\n";
                    s += &content;
                }
                for c in m.calls {
                    s += "\n<tool_call>\n{\"name\": ";
                    s += &serde_json::Value::String(c.name.clone()).to_string();
                    s += ", \"arguments\": ";
                    s += &arguments_json(c);
                    s += "}\n</tool_call>";
                }
                s += "<|im_end|>\n";
            }
            "tool" => {
                if k == 0 || messages[k - 1].role != "tool" {
                    s += "<|im_start|>user";
                }
                s += "\n<tool_response>\n";
                s += m.content;
                s += "\n</tool_response>";
                if k + 1 == messages.len() || messages[k + 1].role != "tool" {
                    s += "<|im_end|>\n";
                }
            }
            _ => {}
        }
    }
    s += "<|im_start|>assistant\n";
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m<'a>(role: &'a str, content: &'a str) -> Message<'a> {
        Message { role, content, ..Default::default() }
    }

    #[test]
    fn plain_turns() {
        let p = ripple_chat(&[m("system", "Be brief."), m("user", "Hi")], Effort::High, &[]);
        assert_eq!(p, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn thinking_is_kept_in_front_of_the_answer() {
        let a = Message { role: "assistant", content: "4", reasoning: Some("2+2"), ..Default::default() };
        let p = ripple_chat(&[m("user", "2+2?"), a, m("user", "and 3+3?")], Effort::High, &[]);
        assert!(p.contains("<|im_start|>assistant\n<think>\n2+2\n</think>\n4<|im_end|>\n"));
    }

    #[test]
    fn tools_and_calls() {
        let calls = [ToolCall { name: "get_weather".into(), args: vec![("city".into(), "Paris".into()), ("days".into(), "3".into())] }];
        let a = Message { role: "assistant", content: "", calls: &calls, ..Default::default() };
        let p = ripple_chat(&[m("system", "S"), m("user", "w?"), a, m("tool", "sunny")], Effort::High, &["{\"type\": \"function\"}".to_string()]);
        assert!(p.starts_with("<|im_start|>system\nS\n\n# Tools\n\nYou may call one or more functions"));
        assert!(p.contains("<tools>\n{\"type\": \"function\"}\n</tools>"));
        assert!(p.contains("<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call><|im_end|>\n"));
        assert!(p.ends_with("<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n<|im_start|>assistant\n"));
    }
}
