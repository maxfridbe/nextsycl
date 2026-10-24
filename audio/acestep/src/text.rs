//! What the DiT's text side reads, as ACE-Step's handler writes it (acestep/constants.py SFT_GEN_PROMPT,
//! core/generation/handler/prompt_utils.py and metadata_utils.py): the caption prompt - the task's instruction, the
//! caption, the metas (BPM, time signature, key, duration) - for the text encoder, and the lyrics with their language
//! for the lyric encoder (their token embeddings alone). The tokenizer is the text encoder's (Qwen3's byte-level
//! BPE); the caption prompt is cut at 256 tokens and the lyrics at 2048, as the handler's tokenizer calls do.

/// The instruction of each task (constants.py TASK_INSTRUCTIONS)
pub fn instruction(task: &str, track: Option<&str>) -> String {
    match task {
        "repaint" => "Repaint the mask area based on the given conditions:".into(),
        "cover" | "cover-nofsq" => "Generate audio semantic tokens based on the given conditions:".into(),
        "lego" => match track {
            Some(t) => format!("Generate the {t} track based on the audio context:"),
            None => "Generate the track based on the audio context:".into(),
        },
        "extract" => match track {
            Some(t) => format!("Extract the {t} track from the audio:"),
            None => "Extract the track from the audio:".into(),
        },
        _ => "Fill the audio semantic mask based on the given conditions:".into(),
    }
}

/// The song's metadata as the prompt carries it; None: "N/A"
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metas {
    pub bpm: Option<u32>,
    pub timesignature: Option<String>,
    pub keyscale: Option<String>,
    /// seconds (whole: the prompt says "{int} seconds")
    pub duration: Option<f64>,
}

impl Metas {
    /// metadata_utils.py `_dict_to_meta_string` (a missing duration reads 30 seconds, as `_create_default_meta`)
    pub fn text(&self) -> String {
        let na = |s: &Option<String>| s.as_deref().map(str::trim).filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("n/a")).unwrap_or("N/A").to_string();
        let bpm = self.bpm.filter(|b| *b > 0).map_or("N/A".to_string(), |b| b.to_string());
        let duration = self.duration.filter(|d| *d > 0.0).map_or(30, |d| d as i64);
        format!("- bpm: {bpm}\n- timesignature: {}\n- keyscale: {}\n- duration: {duration} seconds\n", na(&self.timesignature), na(&self.keyscale))
    }
}

/// The caption prompt (SFT_GEN_PROMPT): its instruction ends with a colon (`_format_instruction`)
pub fn caption_prompt(instruction: &str, caption: &str, metas: &str) -> String {
    let i = if instruction.ends_with(':') { instruction.to_string() } else { format!("{instruction}:") };
    format!("# Instruction\n{i}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n")
}

/// The lyrics with their language (`_format_lyrics`)
pub fn lyrics_text(lyrics: &str, language: &str) -> String {
    format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>")
}

/// The caption prompt's longest (tokens)
pub const CAPTION_TOKENS: usize = 256;
/// The lyrics' longest (tokens)
pub const LYRIC_TOKENS: usize = 2048;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompts_are_the_handlers() {
        // as the reference run's tokenizer was given them (reference/acestep/ref.py text.json)
        let m = Metas { duration: Some(10.0), ..Default::default() };
        assert_eq!(caption_prompt(&instruction("text2music", None), "folk ballad, warm male vocal, acoustic guitar, soft strings", &m.text()),
                   "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\nfolk ballad, warm male vocal, acoustic guitar, \
                    soft strings\n\n# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 10 seconds\n<|endoftext|>\n");
        assert_eq!(lyrics_text("[Verse]\nThe lamp", "unknown"), "# Languages\nunknown\n\n# Lyric\n[Verse]\nThe lamp<|endoftext|>");
        let full = Metas { bpm: Some(118), timesignature: Some("4".into()), keyscale: Some("C major".into()), duration: Some(180.6) };
        assert_eq!(full.text(), "- bpm: 118\n- timesignature: 4\n- keyscale: C major\n- duration: 180 seconds\n");
        assert_eq!(Metas::default().text(), "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n");
        assert_eq!(caption_prompt("Repaint", "x", "m"), "# Instruction\nRepaint:\n\n# Caption\nx\n\n# Metas\nm<|endoftext|>\n");
    }
}

