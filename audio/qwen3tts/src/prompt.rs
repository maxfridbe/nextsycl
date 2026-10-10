//! The talker's prompt, as qwen-tts builds it (Qwen3TTSForConditionalGeneration.generate): rows of embeddings, each a
//! projected text token plus a codec token, and the text rows left to feed one a frame.
//!
//! ```text
//!   [instruct: proj(text("<|im_start|>user\n{instruct}<|im_end|>\n"))]                 (a design, a styled voice)
//!   proj(text("<|im_start|>assistant\n"))                                               the role: 3 rows
//!   tts_pad.. tts_bos  +  codec [think|nothink, think_bos, (language), think_eos, (speaker)], pad    (codec's last held back)
//!   streaming:      proj(first text token) + codec bos;  the rest of the text, then tts_eos, fed a frame at a time
//!   non-streaming:  proj(text..) ++ tts_eos, each + codec pad;  tts_pad + codec bos;  tts_pad fed every frame
//!   clone with a transcript (ICL): proj(ref text ++ text) ++ tts_eos against codec bos ++ the reference's frames
//! ```
//!
//! The speaker row is a built-in voice's codec embedding (CustomVoice), a recording's x-vector (Base: speaker.rs),
//! or none (VoiceDesign: the instruction describes the voice). The checkpoint's own defaults: CustomVoice and
//! VoiceDesign non-streaming, a clone streaming.

use std::collections::BTreeMap;
use std::path::Path;

use nextsycl_audio::{Error, Result};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_tok::Tokenizer;
use unicode_normalization::UnicodeNormalization;

use crate::ops::{Ops, Shards};
use crate::talker::{Session, Talker};

/// config.json's ids and tables
#[derive(Clone, Debug)]
pub struct Config {
    /// "custom_voice", "voice_design" or "base"
    pub kind: String,
    pub tts_bos: u32,
    pub tts_eos: u32,
    pub tts_pad: u32,
    pub pad: i32,
    pub bos: i32,
    pub eos: i32,
    pub think: i32,
    pub nothink: i32,
    pub think_bos: i32,
    pub think_eos: i32,
    pub vocab: usize,
    pub speakers: BTreeMap<String, i32>,
    /// a dialect speaker's language (for Chinese or auto text)
    pub dialect: BTreeMap<String, String>,
    pub languages: BTreeMap<String, i32>,
}

impl Config {
    /// The checkpoint's config.json
    pub fn read(p: &Path) -> Result<Config> {
        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
            .map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let t = &c["talker_config"];
        let int = |v: &serde_json::Value, k: &str| v[k].as_i64().ok_or_else(|| Error(format!("{}: no {k}", p.display())));
        let map = |k: &str| -> BTreeMap<String, i32> {
            t[k].as_object().into_iter().flatten().filter_map(|(n, v)| Some((n.clone(), v.as_i64()? as i32))).collect()
        };
        Ok(Config {
            kind: c["tts_model_type"].as_str().unwrap_or("base").to_string(),
            tts_bos: int(&c, "tts_bos_token_id")? as u32,
            tts_eos: int(&c, "tts_eos_token_id")? as u32,
            tts_pad: int(&c, "tts_pad_token_id")? as u32,
            pad: int(t, "codec_pad_id")? as i32,
            bos: int(t, "codec_bos_id")? as i32,
            eos: int(t, "codec_eos_token_id")? as i32,
            think: int(t, "codec_think_id")? as i32,
            nothink: int(t, "codec_nothink_id")? as i32,
            think_bos: int(t, "codec_think_bos_id")? as i32,
            think_eos: int(t, "codec_think_eos_id")? as i32,
            vocab: int(t, "vocab_size")? as usize,
            speakers: map("spk_id"),
            dialect: t["spk_is_dialect"].as_object().into_iter().flatten().filter_map(|(n, v)| Some((n.clone(), v.as_str()?.to_string()))).collect(),
            languages: map("codec_language_id"),
        })
    }
}

/// Whose voice
#[derive(Clone, Debug)]
pub enum Voice {
    /// a built-in speaker (CustomVoice)
    Speaker(String),
    /// none: the instruction describes it (VoiceDesign)
    Described,
    /// a recording's x-vector (Base)
    XVector(Vec<f32>),
    /// a recording's x-vector, its transcript and its codec frames [frames][16] (Base, in-context)
    InContext { spk: Vec<f32>, text: String, codes: Vec<Vec<i32>> },
}

/// What to say, how
#[derive(Clone, Debug)]
pub struct Spec {
    pub text: String,
    /// None or "auto": the model's guess
    pub language: Option<String>,
    pub instruct: Option<String>,
    pub voice: Voice,
    pub streaming: bool,
}

/// The prompt: rows [L, hidden], the text rows fed a frame each [T, hidden], then tts_pad's row
pub struct Prompt {
    pub rows: Vec<f32>,
    pub trailing: Vec<f32>,
    pub pad: Vec<f32>,
}

/// The token ids of a template filled in (NFC-normalized first, as the checkpoint's tokenizer does)
pub fn ids(tok: &Tokenizer, s: &str) -> Vec<u32> {
    tok.encode(&s.nfc().collect::<String>())
}

pub fn assistant(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

pub fn reference(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n")
}

pub fn instruction(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n")
}

/// Upper bound on the prompt's rows (a session's size, before the rows are made)
pub fn bound(tok: &Tokenizer, s: &Spec) -> usize {
    let mut n = ids(tok, &assistant(&s.text)).len() + 16;
    if let Some(i) = s.instruct.as_deref().filter(|i| !i.is_empty()) {
        n += ids(tok, &instruction(i)).len();
    }
    if let Voice::InContext { text, codes, .. } = &s.voice {
        n += ids(tok, &reference(text)).len() + codes.len();
    }
    n
}

/// The language's codec id (None: auto), a dialect speaker's own for Chinese or auto text
pub fn language(c: &Config, lang: Option<&str>, speaker: Option<&str>) -> Result<Option<i32>> {
    let l = lang.map(str::to_lowercase).filter(|l| !l.is_empty() && l != "auto");
    let mut id = match &l {
        None => None,
        Some(l) => Some(*c.languages.get(l).ok_or_else(|| {
            Error(format!("language {l:?}: not one this model speaks ({}, or auto)", c.languages.keys().cloned().collect::<Vec<_>>().join(", ")))
        })?),
    };
    if let Some(d) = speaker.and_then(|s| c.dialect.get(&s.to_lowercase())) {
        if l.is_none() || l.as_deref() == Some("chinese") {
            id = c.languages.get(d).copied().or(id);
        }
    }
    Ok(id)
}

/// The text embedding's rows of `ids` [n, hidden] (read from the checkpoint: a row each)
fn text_rows(f: &Shards, ids: &[u32]) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    for &i in ids {
        v.extend(f.rows_f32("talker.model.text_embedding.weight", i as usize, i as usize + 1)?);
    }
    Ok(v)
}

fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
}

/// The prompt's rows
#[allow(clippy::too_many_arguments)]
pub fn build(t: &Talker, ops: &Ops, nsd: &Nsd, ss: &Session, f: &Shards, tok: &Tokenizer, c: &Config, s: &Spec) -> Result<Prompt> {
    let h = t.lm.hidden;
    let proj = |ids: &[u32]| -> Result<Vec<f32>> { t.project_text(ops, nsd, ss, &text_rows(f, ids)?) };
    let codec = |id: i32| t.codec_sum(ops, ss, &[id]);
    if s.text.trim().is_empty() {
        return Err(Error("the text is empty".into()));
    }
    let ids_t = ids(tok, &assistant(&s.text));
    if ids_t.len() < 9 {
        return Err(Error("the text made no tokens".into()));
    }
    let specials = proj(&[c.tts_bos, c.tts_eos, c.tts_pad])?;
    let (tts_bos, tts_eos, tts_pad) = (&specials[..h], &specials[h..2 * h], specials[2 * h..].to_vec());
    let mut rows = Vec::new();
    if let Some(i) = s.instruct.as_deref().filter(|i| !i.is_empty()) {
        rows.extend(proj(&ids(tok, &instruction(i)))?);
    }
    let speaker = match &s.voice {
        Voice::Speaker(n) => Some(n.as_str()),
        _ => None,
    };
    let lang = language(c, s.language.as_deref(), speaker)?;
    let cids = match lang {
        None => vec![c.nothink, c.think_bos, c.think_eos],
        Some(l) => vec![c.think, c.think_bos, l, c.think_eos],
    };
    let mut codec_rows: Vec<Vec<f32>> = cids.iter().map(|i| codec(*i)).collect::<Result<_>>()?;
    match &s.voice {
        Voice::Speaker(n) => {
            let id = *c.speakers.get(&n.to_lowercase()).ok_or_else(|| {
                Error(format!("voice {n:?}: not a built-in one ({})", c.speakers.keys().cloned().collect::<Vec<_>>().join(", ")))
            })?;
            codec_rows.push(codec(id)?);
        }
        Voice::XVector(x) | Voice::InContext { spk: x, .. } => {
            if x.len() != h {
                return Err(Error(format!("a speaker embedding of {}; the talker takes {h}", x.len())));
            }
            codec_rows.push(x.clone());
        }
        Voice::Described => {}
    }
    codec_rows.push(codec(c.pad)?);
    codec_rows.push(codec(c.bos)?);
    // the role, then tts_pad.. tts_bos against the codec rows but the last
    rows.extend(proj(&ids_t[..3])?);
    let n = codec_rows.len();
    for (i, cr) in codec_rows[..n - 1].iter().enumerate() {
        let mut r = if i + 2 < n { tts_pad.clone() } else { tts_bos.to_vec() };
        add(&mut r, cr);
        rows.extend(r);
    }
    let last = &codec_rows[n - 1];
    let body = &ids_t[3..ids_t.len() - 5];
    let mut trailing = Vec::new();
    match &s.voice {
        Voice::InContext { text, codes, .. } => {
            let ids_r = ids(tok, &reference(text));
            if ids_r.len() < 6 {
                return Err(Error("the reference transcript made no tokens".into()));
            }
            let mut te = proj(&[&ids_r[3..ids_r.len() - 2], body].concat())?;
            te.extend_from_slice(tts_eos);
            let mut ce = codec(c.bos)?;
            for fr in codes {
                let idx: Vec<i32> = fr.iter().enumerate().map(|(g, x)| t.index(g, *x)).collect();
                ce.extend(t.codec_sum(ops, ss, &idx)?);
            }
            let (tl, cl) = (te.len() / h, ce.len() / h);
            if s.streaming {
                if tl > cl {
                    let mut a = te[..cl * h].to_vec();
                    add(&mut a, &ce);
                    rows.extend(a);
                    trailing = te[cl * h..].to_vec();
                } else {
                    let mut a = te.clone();
                    a.extend((tl..cl).flat_map(|_| tts_pad.iter().copied()));
                    add(&mut a, &ce);
                    rows.extend(a);
                }
            } else {
                let p = codec(c.pad)?;
                for r in te.chunks(h) {
                    let mut a = r.to_vec();
                    add(&mut a, &p);
                    rows.extend(a);
                }
                for r in ce.chunks(h) {
                    let mut a = r.to_vec();
                    add(&mut a, &tts_pad);
                    rows.extend(a);
                }
            }
        }
        _ if s.streaming => {
            let mut a = proj(&ids_t[3..4])?;
            add(&mut a, last);
            rows.extend(a);
            trailing = proj(&ids_t[4..ids_t.len() - 5])?;
            trailing.extend_from_slice(tts_eos);
        }
        _ => {
            let mut te = proj(body)?;
            te.extend_from_slice(tts_eos);
            let p = codec(c.pad)?;
            for r in te.chunks(h) {
                let mut a = r.to_vec();
                add(&mut a, &p);
                rows.extend(a);
            }
            let mut a = tts_pad.clone();
            add(&mut a, last);
            rows.extend(a);
        }
    }
    Ok(Prompt { rows, trailing, pad: tts_pad })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conf() -> Config {
        Config {
            kind: "custom_voice".into(), tts_bos: 151672, tts_eos: 151673, tts_pad: 151671, pad: 2148, bos: 2149, eos: 2150, think: 2154, nothink: 2155,
            think_bos: 2156, think_eos: 2157, vocab: 3072,
            speakers: [("eric".to_string(), 2875), ("ryan".to_string(), 3061)].into(),
            dialect: [("eric".to_string(), "sichuan_dialect".to_string())].into(),
            languages: [("chinese".to_string(), 2055), ("english".to_string(), 2050), ("sichuan_dialect".to_string(), 2062)].into(),
        }
    }

    #[test]
    fn languages_and_dialects() {
        let c = conf();
        assert_eq!(language(&c, None, None).unwrap(), None);
        assert_eq!(language(&c, Some("Auto"), Some("ryan")).unwrap(), None);
        assert_eq!(language(&c, Some("English"), Some("ryan")).unwrap(), Some(2050));
        assert_eq!(language(&c, Some("Chinese"), Some("Eric")).unwrap(), Some(2062));
        assert_eq!(language(&c, Some("auto"), Some("eric")).unwrap(), Some(2062));
        assert_eq!(language(&c, Some("english"), Some("eric")).unwrap(), Some(2050));
        assert!(language(&c, Some("klingon"), None).is_err());
    }
}
