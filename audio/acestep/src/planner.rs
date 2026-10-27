//! The planner (acestep/llm_inference.py `generate_with_stop_condition`, infer_type "llm_dit", the PyTorch path the
//! platform's engine runs): two phases on the 5 Hz LM -
//!
//! ```text
//!   1 CoT    the chat prompt (caption, lyrics) -> "<think>\nbpm: ..\ncaption: ..\nduration: ..\nkeyscale: ..\n
//!            language: ..\ntimesignature: ..\n" under the constrained decoder (the user's fields injected, the end forced
//!            before </think>); no guidance. Skipped when the user gave bpm, key, time signature and duration.
//!   2 codes  the prompt with that CoT as YAML in an open assistant turn, against the unconditional one ("NO USER
//!            INPUT", an empty think): guidance 2 over the audio codes alone, the end held until 5 codes a second
//! ```
//!
//! each token: the processor's mask, top-p 0.9 over the logits, then the temperature (0.85) and a draw - the draw is
//! this engine's own (seeded), not torch's multinomial.

use std::collections::HashMap;

use nextsycl_audio::{Error, Result};

use crate::fsm::{Fsm, LmTok, Phase, State, Tables};
use crate::lm::Lm;
use crate::nn::{Ops, Scratch};
use crate::sample::Rng;

pub const LM_INSTRUCTION: &str = "Generate audio semantic tokens based on the given conditions:";
/// the 12 GB tier's longest planned song (gpu_config.py max_duration_with_lm)
pub const MAX_PLANNED_SECONDS: f64 = 480.0;

/// The planner's settings (GenerationParams' lm_*)
#[derive(Clone, Debug)]
pub struct Settings {
    pub temperature: f32,
    pub cfg: f32,
    pub top_p: f32,
    pub negative: String,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { temperature: 0.85, cfg: 2.0, top_p: 0.9, negative: "NO USER INPUT".into() }
    }
}

/// What the planner wrote
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// parse_lm_output's fields (bpm, caption, duration, keyscale, language, timesignature)
    pub metadata: HashMap<String, String>,
    pub codes: Vec<u32>,
    pub cot_text: String,
    pub prompts: Vec<String>,
    pub cot_tokens: Vec<u32>,
    pub code_tokens: Vec<u32>,
}

/// Qwen3's chat template (a system message, a user message, the assistant's turn opened)
pub fn chat(user: &str) -> String {
    format!("<|im_start|>system\n# Instruction\n{LM_INSTRUCTION}\n\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

/// build_formatted_prompt (the CoT phase, conditional)
pub fn cot_prompt(caption: &str, lyrics: &str) -> String {
    chat(&format!("# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"))
}

/// build_formatted_prompt_with_cot: the conditional prompt (`uncond` false) or the unconditional one
pub fn codes_prompt(caption: &str, lyrics: &str, cot: &str, uncond: bool, negative: &str) -> String {
    if uncond {
        let user = if !negative.trim().is_empty() && negative.trim() != "NO USER INPUT" { negative } else { "NO USER INPUT" };
        format!("{}<think>\n\n</think>\n\n", chat(user))
    } else {
        format!("{}{cot}\n\n", chat(&format!("# Caption\n{caption}\n\n# Lyric\n{lyrics}\n")))
    }
}

/// MetadataConstrainedLogitsProcessor.postprocess_caption: the lines stripped and joined by spaces
pub fn postprocess_caption(c: &str) -> String {
    c.split('\n').map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ")
}

/// parse_lm_output's metadata (the think block's YAML-ish fields; a value may run on over indented lines)
pub fn parse(text: &str) -> HashMap<String, String> {
    let reasoning = match (text.find("<think>"), text.find("</think>")) {
        (Some(a), Some(b)) if b > a => text[a + 7..b].trim().to_string(),
        _ => text.split("<|audio_code_").next().unwrap_or("").trim().to_string(),
    };
    let mut out = HashMap::new();
    let mut key: Option<String> = None;
    let mut lines: Vec<String> = Vec::new();
    let mut save = |key: &mut Option<String>, lines: &mut Vec<String>| {
        if let Some(k) = key.take() {
            if !lines.is_empty() {
                let v = lines.join("\n");
                let v = match k.as_str() {
                    "caption" => postprocess_caption(&v),
                    _ => v.trim().to_string(),
                };
                if ["bpm", "caption", "duration", "genres", "keyscale", "language", "timesignature"].contains(&k.as_str()) {
                    out.insert(k, v);
                }
            }
        }
        lines.clear();
    };
    for line in reasoning.split('\n') {
        if line.trim().starts_with('<') {
            continue;
        }
        if !line.is_empty() && !line.starts_with(char::is_whitespace) && line.contains(':') {
            save(&mut key, &mut lines);
            let (k, v) = line.split_once(':').unwrap_or((line, ""));
            key = Some(k.trim().to_lowercase());
            if !v.trim().is_empty() {
                lines.push(v.to_string());
            }
        } else if (line.starts_with(' ') || line.starts_with('\t'))
            && key.is_some() {
                lines.push(line.to_string());
            }
    }
    save(&mut key, &mut lines);
    out
}

/// A YAML scalar as PyYAML's dump writes it in a block mapping (allow_unicode, width 80): plain when it can be,
/// else single-quoted; long ones folded at spaces past column 80, continued at an indent of 2
pub fn yaml_scalar(key: &str, v: &str, number: bool) -> String {
    if number {
        return format!("{key}: {v}");
    }
    let plain = plain_ok(v);
    let body: String = if plain { v.to_string() } else { v.replace('\'', "''") };
    let mut out = format!("{key}: ");
    let mut column = out.chars().count();
    if !plain {
        out.push('\'');
        column += 1;
    }
    let chars: Vec<char> = body.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i] == ' ' {
            let start = i;
            while i < n && chars[i] == ' ' {
                i += 1;
            }
            // a single space after column 80 breaks the line (not the first or last character of a quoted one)
            let edge = !plain && (start == 0 || i == n);
            if i - start == 1 && column > 80 && !edge {
                out.push_str("\n  ");
                column = 2;
            } else {
                for _ in start..i {
                    out.push(' ');
                }
                column += i - start;
            }
        } else {
            let start = i;
            while i < n && chars[i] != ' ' {
                i += 1;
            }
            let w: String = chars[start..i].iter().collect();
            column += w.chars().count();
            out.push_str(&w);
        }
    }
    if !plain {
        out.push('\'');
    }
    out
}

/// Whether PyYAML would write `v` plain (analyze_scalar's checks, the ones a caption meets; and not a string that
/// reads as another type)
fn plain_ok(v: &str) -> bool {
    if v.is_empty() || v.starts_with(' ') || v.ends_with(' ') || v.contains('\n') || v.contains('\t') {
        return false;
    }
    let first = v.chars().next().unwrap_or(' ');
    if "-?:,[]{}#&*!|>'\"%@`".contains(first) {
        // "-", "?", ":" lead a plain scalar when a non-space follows
        let second = v.chars().nth(1);
        if !("-?:".contains(first) && second.is_some_and(|c| c != ' ')) {
            return false;
        }
    }
    if v.contains(": ") || v.contains(" #") || v.ends_with(':') {
        return false;
    }
    let l = v.to_lowercase();
    if ["yes", "no", "true", "false", "on", "off", "null", "~", "y", "n"].contains(&l.as_str()) {
        return false;
    }
    if v.parse::<f64>().is_ok() {
        return false;
    }
    // characters PyYAML will not print plain (control characters; allow_unicode keeps the rest)
    !v.chars().any(|c| (c as u32) < 0x20 || c == '\u{7f}' || c == '\u{feff}')
}

/// _format_metadata_as_cot: the fields (sorted, the numbers as numbers) as YAML in a think block
pub fn cot_text(m: &HashMap<String, String>) -> String {
    let mut lines = Vec::new();
    for k in ["bpm", "caption", "duration", "keyscale", "language", "timesignature"] {
        let Some(v) = m.get(k) else { continue };
        let mut v = v.clone();
        if k == "timesignature" && v.ends_with("/4") {
            v = v.split('/').next().unwrap_or("").to_string();
        }
        let number = !v.is_empty() && v.chars().all(|c| c.is_ascii_digit());
        lines.push(yaml_scalar(k, &v, number));
    }
    format!("<think>\n{}\n</think>", lines.join("\n"))
}

/// HF-style top-p over the logits: the most likely tokens kept until their softmax mass passes `p` (the one that
/// passes it kept), the rest -inf
pub fn top_p(scores: &mut [f32], p: f32) {
    if !(p > 0.0 && p < 1.0) {
        return;
    }
    let mut idx: Vec<u32> = (0..scores.len() as u32).filter(|i| scores[*i as usize].is_finite()).collect();
    if idx.is_empty() {
        return;
    }
    let max = idx.iter().map(|i| scores[*i as usize]).fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = idx.iter().map(|i| ((scores[*i as usize] - max) as f64).exp()).sum();
    let by = |a: &u32, b: &u32| scores[*b as usize].partial_cmp(&scores[*a as usize]).unwrap_or(std::cmp::Ordering::Equal);
    // the head sorted, wider until it holds the nucleus
    let mut k = 256.min(idx.len());
    let keep = loop {
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, by);
        }
        idx[..k].sort_by(by);
        let mut cum = 0f64;
        let mut n = None;
        for (j, i) in idx[..k].iter().enumerate() {
            if cum > p as f64 {
                n = Some(j);
                break;
            }
            cum += ((scores[*i as usize] - max) as f64).exp() / total;
        }
        match n {
            Some(n) => break n,
            None if k == idx.len() => break k,
            None => k = (k * 4).min(idx.len()),
        }
    };
    let kept: Vec<(u32, f32)> = idx[..keep].iter().map(|i| (*i, scores[*i as usize])).collect();
    scores.fill(f32::NEG_INFINITY);
    for (i, v) in kept {
        scores[i as usize] = v;
    }
}

/// A draw from softmax(scores / temperature) (temperature 0: the most likely)
pub fn draw(scores: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
    let best = scores.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b });
    if temperature <= 0.0 {
        return best.0 as u32;
    }
    let max = best.1;
    let w: Vec<(usize, f64)> = scores.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| (i, (((v - max) / temperature) as f64).exp())).collect();
    let total: f64 = w.iter().map(|x| x.1).sum();
    let mut u = rng.uniform() * total;
    for (i, x) in &w {
        u -= x;
        if u <= 0.0 {
            return *i as u32;
        }
    }
    w.last().map_or(best.0, |x| x.0) as u32
}

pub struct Planner {
    pub lm: Lm,
    pub ops: Ops,
    pub tok: LmTok,
    pub tables: Tables,
    pub settings: Settings,
}

impl Planner {
    fn is_stop(&self, t: u32) -> bool {
        t == self.tables.eos() || self.tok.full.id("<|endoftext|>") == Some(t)
    }

    /// Phase 1: the CoT text (no guidance), from the chat prompt
    #[allow(clippy::too_many_arguments)]
    fn cot(&self, s: &mut Scratch, caption: &str, lyrics: &str, user: HashMap<&'static str, String>, rng: &mut Rng, plan: &mut Plan,
           progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<String> {
        let prompt = cot_prompt(caption, lyrics);
        let ids = self.tok.encode(&prompt);
        plan.prompts.push(prompt);
        let max_new = 3500usize;
        let mut seq = self.lm.seq(&self.ops.gpu, ids.len() + max_new + 4)?;
        let mut logits = self.lm.prefill(&self.ops, s, &mut seq, &ids)?;
        let mut fsm = Fsm::new(&self.tables, &self.tok, Phase::Cot, None, user, true, false, false);
        let mut all = ids.clone();
        let mut out = Vec::new();
        for step in 0..max_new {
            progress(step as u32, max_new as u32)?;
            fsm.process(&all, &mut logits);
            top_p(&mut logits, self.settings.top_p);
            let t = draw(&logits, self.settings.temperature, rng);
            fsm.update(t);
            out.push(t);
            all.push(t);
            if self.is_stop(t) {
                break;
            }
            let mut seqs = [&mut seq];
            logits = self.lm.step(&self.ops, s, &mut seqs, &[t])?;
        }
        plan.cot_tokens = out.clone();
        Ok(self.tok.decode(&out))
    }

    /// Phase 2: the audio codes under guidance, `seconds` of them (5 a second)
    #[allow(clippy::too_many_arguments)]
    fn codes(&self, s: &mut Scratch, caption: &str, lyrics: &str, cot: &str, seconds: f64, rng: &mut Rng, plan: &mut Plan,
             progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<Vec<u32>> {
        let cp = codes_prompt(caption, lyrics, cot, false, &self.settings.negative);
        let up = codes_prompt(caption, lyrics, cot, true, &self.settings.negative);
        let (ci, ui) = (self.tok.encode(&cp), self.tok.encode(&up));
        plan.prompts.push(cp);
        plan.prompts.push(up);
        let eff = seconds.clamp(10.0, MAX_PLANNED_SECONDS);
        let max_new = (eff * 5.0) as usize + 10;
        let mut sc = self.lm.seq(&self.ops.gpu, ci.len() + max_new + 4)?;
        let mut su = self.lm.seq(&self.ops.gpu, ui.len() + max_new + 4)?;
        let lc = self.lm.prefill(&self.ops, s, &mut sc, &ci)?;
        let lu = self.lm.prefill(&self.ops, s, &mut su, &ui)?;
        let mut logits = [lc, lu].concat();
        let mut fsm = Fsm::new(&self.tables, &self.tok, Phase::Codes, Some(seconds), HashMap::new(), false, true, true);
        let v = self.lm.vocab;
        let eos = self.tables.eos() as usize;
        let mut all = ci.clone();
        let mut out = Vec::new();
        let g = self.settings.cfg;
        // the codes' rows of the vocabulary (contiguous) and the end: once the codes have begun, the head reads those
        // alone and the constraint is the end's (held back to the target, then forced)
        let first = (0..v).find(|i| self.tables.code_of[*i] == Some(0)).ok_or_else(|| Error("no <|audio_code_0|> token".into()))?;
        let ncodes = (crate::fsm::MAX_AUDIO_CODE + 1) as usize;
        if (0..ncodes).any(|c| self.tables.code_of.get(first + c).copied().flatten() != Some(c as u32)) {
            return Err(Error("the audio code tokens are not contiguous".into()));
        }
        let mut narrow = false;
        for step in 0..max_new {
            if narrow {
                progress(step as u32, max_new as u32)?;
                let w = ncodes + 1;
                let (c, u) = logits.split_at(w);
                let mut cfg: Vec<f32> = (0..w).map(|i| {
                    let r = u[i] + g * (c[i] - u[i]);
                    if r.is_nan() { f32::NEG_INFINITY } else { r }
                }).collect();
                if let Some(target) = fsm.target_codes {
                    if fsm.codes_count < target {
                        cfg[ncodes] = f32::NEG_INFINITY;
                    } else {
                        let e = cfg[ncodes];
                        cfg.fill(f32::NEG_INFINITY);
                        cfg[ncodes] = e;
                    }
                }
                top_p(&mut cfg, self.settings.top_p);
                let j = draw(&cfg, self.settings.temperature, rng) as usize;
                let t = if j == ncodes { eos as u32 } else { (first + j) as u32 };
                fsm.update(t);
                out.push(t);
                all.push(t);
                if self.is_stop(t) {
                    break;
                }
                let mut seqs = [&mut sc, &mut su];
                logits = self.lm.step_rows(&self.ops, s, &mut seqs, &[t, t], first, ncodes, eos)?;
                continue;
            }
            progress(step as u32, max_new as u32)?;
            let (c, u) = logits.split_at(v);
            // the guidance over the codes (and the end) alone once the codes have begun; all of it before
            let codes_only = fsm.state == State::Codes;
            let mut cfg: Vec<f32> = (0..v).map(|i| {
                if codes_only && i != eos && self.tables.code_of[i].is_none() {
                    f32::NEG_INFINITY
                } else {
                    let r = u[i] + g * (c[i] - u[i]);
                    if r.is_nan() { f32::NEG_INFINITY } else { r }
                }
            }).collect();
            fsm.process(&all, &mut cfg);
            top_p(&mut cfg, self.settings.top_p);
            let t = draw(&cfg, self.settings.temperature, rng);
            fsm.update(t);
            out.push(t);
            all.push(t);
            if self.is_stop(t) {
                break;
            }
            let mut seqs = [&mut sc, &mut su];
            if fsm.state == State::Codes {
                narrow = true;
                logits = self.lm.step_rows(&self.ops, s, &mut seqs, &[t, t], first, ncodes, eos)?;
            } else {
                logits = self.lm.step(&self.ops, s, &mut seqs, &[t, t])?;
            }
        }
        plan.code_tokens = out.clone();
        Ok(out.iter().filter_map(|t| self.tables.code_of.get(*t as usize).copied().flatten()).collect())
    }

    /// Both phases for a song: `user` the fields the request gave (bpm, keyscale, timesignature, duration)
    pub fn plan(&self, caption: &str, lyrics: &str, user: HashMap<&'static str, String>, seconds: f64, seed: u64,
                progress: &mut dyn FnMut(&'static str, u32, u32) -> Result<()>) -> Result<Plan> {
        let mut s = Scratch::default();
        let mut rng = Rng::new(seed ^ 0x091A_77E5);
        let mut plan = Plan::default();
        let all_given = ["bpm", "keyscale", "timesignature", "duration"].iter().all(|k| user.contains_key(k));
        let metadata = if all_given {
            user.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
        } else {
            let text = self.cot(&mut s, caption, lyrics, user, &mut rng, &mut plan, &mut |a, b| progress("plan", a, b))?;
            parse(&text)
        };
        let cot = cot_text(&metadata);
        let codes = self.codes(&mut s, caption, lyrics, &cot, seconds, &mut rng, &mut plan, &mut |a, b| progress("codes", a, b))?;
        if codes.is_empty() {
            return Err(Error("the planner wrote no audio codes".into()));
        }
        Ok(Plan { metadata, codes, cot_text: cot, ..plan })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cot_is_pyyamls() {
        let m: HashMap<String, String> = [("bpm", "120"), ("caption", "An upbeat indie pop track with bright guitars."), ("duration", "30"),
                                          ("keyscale", "G major"), ("language", "en"), ("timesignature", "4")]
            .iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        assert_eq!(cot_text(&m), "<think>\nbpm: 120\ncaption: An upbeat indie pop track with bright guitars.\nduration: 30\nkeyscale: G major\nlanguage: en\ntimesignature: 4\n</think>");
        // folded past column 80
        let long = "A bright and breezy indie pop song driven by jangly electric guitars and a punchy drum beat, with a warm bassline.";
        assert_eq!(yaml_scalar("caption", long, false),
                   "caption: A bright and breezy indie pop song driven by jangly electric guitars and\n  a punchy drum beat, with a warm bassline.");
        assert_eq!(yaml_scalar("caption", "it's a song, really # yes", false), "caption: 'it''s a song, really # yes'");
        assert_eq!(yaml_scalar("caption", "- dash", false), "caption: '- dash'");
        assert_eq!(yaml_scalar("caption", &format!("{} y", "x".repeat(90)), false), format!("caption: {}\n  y", "x".repeat(90)));
        assert_eq!(yaml_scalar("caption", "Mood: happy", false), "caption: 'Mood: happy'");
        assert_eq!(yaml_scalar("keyscale", "yes", false), "keyscale: 'yes'");
    }

    #[test]
    fn the_cot_parses_back() {
        let t = "<think>\nbpm: 96\ncaption: A dark synthwave\n  track.\nduration: 30\nkeyscale: D minor\nlanguage: en\ntimesignature: 4\n<|im_end|>";
        let m = parse(t);
        assert_eq!(m["bpm"], "96");
        assert_eq!(m["caption"], "A dark synthwave track.");
        assert_eq!(m["timesignature"], "4");
        assert_eq!(m["keyscale"], "D minor");
    }

    #[test]
    fn top_p_keeps_the_nucleus() {
        let mut s = vec![0.0f32, (3f32).ln(), (6f32).ln(), f32::NEG_INFINITY];
        // probabilities 0.1, 0.3, 0.6: 0.6 alone is under 0.85, 0.6 + 0.3 passes it: those two kept
        top_p(&mut s, 0.85);
        assert!(s[0].is_infinite() && s[1].is_finite() && s[2].is_finite());
        let mut r = Rng::new(3);
        for _ in 0..50 {
            assert!([1, 2].contains(&draw(&s, 0.85, &mut r)));
        }
    }
}
