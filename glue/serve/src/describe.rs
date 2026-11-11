//! A picture to the prompt that would draw it (`image.describe`, `nextsycl image describe`): a chat model that reads
//! pictures (Qwen3.8-27B with its vision tower, `llm/qwen35`) is shown the picture with an instruction written for the
//! image model the prompt is for, and its answer is cut into the prompt (and a negative prompt, for the models that
//! take one).
//!
//! No engine of its own: the chat server's `/api/chat` (through the model switcher on the box, or a `nextsycl llm
//! serve` named with `--server`) does the reading, streamed, so a cancel closes the connection and the server stops at
//! its next token. The thinking is closed before it starts (`think_budget: 0`): a describe is an answer, not a
//! reasoning task, and the thinking would cost more than the prompt.
//!
//! The families (`target`): each image model reads prompts its own way (model-cards/guides/<model>.md), so the
//! instruction says what that model wants - sentences for the Qwen3 / Qwen3-VL text encoders, one scene opened by what
//! the picture is for H3's denoiser, layout first for Ming-Image's designs, tags for a tag-style model. A target is a
//! family's name or an image model's id (its family from its architecture).

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::http::{self, Conn, Target};

type Fail = (u16, String);
/// width, height
pub type Size = (u32, u32);

/// The prompt styles: one for each way our image models read a prompt
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// plain sentences, for any model (the default)
    Natural,
    /// Qwen-Image 2.1: as told to a painter - subject, look, setting, light, camera, style
    QwenImage,
    /// Z-Image-Turbo: a paragraph of sentences, the kind of picture named, text quoted with its place and lettering
    ZImage,
    /// P-H3-Image and MiniMax-H3-Image (H3's denoiser): what the picture is first, one scene, no hype words
    H3,
    /// HunyuanImage 2.1: as Qwen-Image, quoted text spelled by its glyph encoder
    HunyuanImage,
    /// MOSS-Image: sentences (it takes tags and a JSON layout too)
    MossImage,
    /// Ming-Image-0.1-Design: a design - the layout, then each block's words
    MingImage,
    /// comma-separated tags, most important first
    Tags,
}

pub const FAMILIES: [Family; 8] = [Family::Natural, Family::QwenImage, Family::ZImage, Family::H3, Family::HunyuanImage, Family::MossImage, Family::MingImage, Family::Tags];

impl Family {
    /// Its name in the API
    pub fn name(self) -> &'static str {
        match self {
            Family::Natural => "natural",
            Family::QwenImage => "qwen-image",
            Family::ZImage => "z-image",
            Family::H3 => "h3",
            Family::HunyuanImage => "hunyuanimage",
            Family::MossImage => "moss-image",
            Family::MingImage => "ming-image",
            Family::Tags => "tags",
        }
    }

    /// Whether its models read a negative prompt: Qwen-Image (cfg above 1), HunyuanImage's base model, MOSS-Image;
    /// Z-Image-Turbo, H3's and Ming-Image run without guidance against one
    pub fn takes_negative(self) -> bool {
        matches!(self, Family::Natural | Family::QwenImage | Family::HunyuanImage | Family::MossImage | Family::Tags)
    }

    /// The canvas its models draw best at (pixels) and the multiple their sides are of: what `suggested_size` aims for
    fn canvas(self) -> (f64, u32, u32, u32) {
        // (pixels, multiple, least side, largest side)
        match self {
            Family::H3 => (2048.0 * 2048.0, 32, 256, 5024),
            Family::HunyuanImage => (2048.0 * 2048.0, 32, 768, 3072),
            Family::ZImage => (1024.0 * 1024.0, 16, 256, 2048),
            Family::QwenImage | Family::MossImage | Family::MingImage | Family::Natural | Family::Tags => (1024.0 * 1024.0, 16, 256, 2048),
        }
    }

    /// The family of an image model's architecture (the registry's `arch`); none for one that does not draw
    pub fn of_arch(arch: &str) -> Option<Family> {
        Some(match arch {
            "qwen-image-2.1" => Family::QwenImage,
            "z-image" => Family::ZImage,
            "p-h3-image" | "minimax-h3-image" => Family::H3,
            "hunyuanimage-2.1" => Family::HunyuanImage,
            "moss-image" => Family::MossImage,
            "ming-image" => Family::MingImage,
            _ => return None,
        })
    }

    /// A family by its name or a common other spelling (not a model's id)
    pub fn alias(t: &str) -> Option<Family> {
        let t = t.trim().to_ascii_lowercase();
        if let Some(f) = FAMILIES.iter().find(|f| f.name() == t) {
            return Some(*f);
        }
        Some(match t.as_str() {
            "" | "default" | "generic" | "sentences" | "plain" => Family::Natural,
            "booru" | "tag" | "danbooru" => Family::Tags,
            "qwenimage" | "qwen" => Family::QwenImage,
            "zimage" => Family::ZImage,
            "h3-image" => Family::H3,
            "hunyuan" | "hunyuan-image" => Family::HunyuanImage,
            "moss" => Family::MossImage,
            "ming" => Family::MingImage,
            _ => return None,
        })
    }

    /// A family by its name, a common other spelling, or an image model's id (by its prefix)
    pub fn named(t: &str) -> Option<Family> {
        if let Some(f) = Family::alias(t) {
            return Some(f);
        }
        let t = t.trim().to_ascii_lowercase();
        Some(match t.as_str() {
            _ if t.starts_with("qwen-image") => Family::QwenImage,
            _ if t.starts_with("z-image") => Family::ZImage,
            _ if t.starts_with("p-h3-image") || t.starts_with("minimax-h3-image") => Family::H3,
            _ if t.starts_with("hunyuanimage") => Family::HunyuanImage,
            _ if t.starts_with("moss-image") => Family::MossImage,
            _ if t.starts_with("ming-image") => Family::MingImage,
            _ => return None,
        })
    }

    /// How its models like a prompt written: the part of the instruction that is the family's
    fn style(self) -> &'static str {
        match self {
            Family::Natural => "Write one flowing paragraph of plain, concrete sentences, the most important things first.",
            Family::QwenImage => "The prompt is for Qwen-Image. Describe the picture as you would to a painter: the subject and its look, \
                the setting, the light, the camera (lens, angle, distance) and the style (\"a photograph\", \"a watercolor\", \"flat vector \
                illustration\"). One long, concrete paragraph, the important things first. Text in the picture goes in double quotes: \
                a shop sign that reads \"OPEN LATE\".",
            Family::ZImage => "The prompt is for Z-Image-Turbo. Write sentences, never a list of tags: one paragraph like a caption, naming \
                the kind of picture (\"a close-up product photo\", \"a mid-shot phone selfie\", \"a flat vector illustration\"), the subject, \
                what it wears or holds, the setting and the light. Put each piece of text in the picture in double quotes and say where it \
                sits and in what lettering. Say what is in the picture, never what is not.",
            Family::H3 => "The prompt is for an H3 image model (P-H3-Image, MiniMax-H3-Image). Open with one sentence that names the real \
                object the picture is (\"a photograph by an experienced street photographer\", \"a flat scan of a handwritten card\", \
                \"a two-color risograph print\", \"a flat vector illustration\"). Then describe one scene: the viewpoint, the main subject, a \
                few supporting elements, and the light with its source and direction. Spell out every word of visible text in double \
                quotes, say where it sits, and end with \"no other text\"; if the picture has no text, do not mention signs or labels at \
                all. Never use hype words (stunning, epic, cinematic, vibrant, hyper-detailed, 8k, masterpiece).",
            Family::HunyuanImage => "The prompt is for HunyuanImage. Describe the picture as you would to a painter: the subject and its \
                look, the setting, the light, the camera and the style, in long concrete sentences, the important things first. Text in \
                the picture goes in double quotes, spelled exactly.",
            Family::MossImage => "The prompt is for MOSS-Image. Write natural sentences: the kind of picture, the subject, the setting, \
                the light and the colours, the important things first. Text in the picture goes in double quotes.",
            Family::MingImage => "The prompt is for Ming-Image, a design model (screens, posters, slides, logos). Describe it as a design: \
                the layout first (a header, columns, a card grid, a centred title), then each block with its exact words in double \
                quotes, its colours, lettering and position.",
            Family::Tags => "Write a comma-separated list of short tags, most important first: the kind of picture, the subject and how \
                many, their attributes and colours, the setting, the style, the lighting, the framing. Tags only, no sentences. Text \
                in the picture as a tag in double quotes.",
        }
    }
}

/// How long a prompt to write
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    Short,
    Medium,
    Long,
}

impl Detail {
    pub fn name(self) -> &'static str {
        match self {
            Detail::Short => "short",
            Detail::Medium => "medium",
            Detail::Long => "long",
        }
    }
    fn parse(s: &str) -> Option<Detail> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "short" | "brief" => Detail::Short,
            "" | "medium" | "normal" => Detail::Medium,
            "long" | "detailed" | "full" => Detail::Long,
            _ => return None,
        })
    }
    /// The words aimed for (sentences), or the tags (a tag list)
    fn words(self, tags: bool) -> usize {
        match (self, tags) {
            (Detail::Short, false) => 35,
            (Detail::Medium, false) => 90,
            (Detail::Long, false) => 170,
            (Detail::Short, true) => 12,
            (Detail::Medium, true) => 25,
            (Detail::Long, true) => 45,
        }
    }
}

/// The picture is shown to the vision tower at most this large (pixels) unless the request says otherwise: ~1,024
/// of its positions, plenty for a prompt and a quarter of the reading of its largest (4,096 positions, 4.2 MP)
pub const DEFAULT_PIXELS: u64 = 1_048_576;
pub const MIN_PIXELS: u64 = 65_536;
pub const MAX_PIXELS: u64 = 4_194_304;

/// A describe's settings, from a request's body (the front door's, or the command line's made into one)
#[derive(Clone, Debug, PartialEq)]
pub struct Ask {
    pub family: Family,
    /// the image model the target named, when it named one
    pub target_model: Option<String>,
    pub detail: Detail,
    /// the words (tags) aimed for: max_words, else the detail's
    pub words: usize,
    pub language: String,
    pub negative: bool,
    /// more for the describer to heed (focus on the lighting ...)
    pub instructions: Option<String>,
    pub max_pixels: u64,
    pub temperature: f64,
    pub seed: Option<u64>,
    pub max_tokens: usize,
}

impl Ask {
    /// The settings a body asks for, checked (the target already resolved to its family by the caller: `family`,
    /// `target_model`)
    pub fn from_body(b: &Value, family: Family, target_model: Option<String>) -> Result<Ask, String> {
        let text = |k: &str| -> Result<Option<String>, String> {
            match &b[k] {
                Value::Null => Ok(None),
                Value::String(s) => Ok(Some(s.trim().to_string()).filter(|s| !s.is_empty())),
                v => Err(format!("{k} {v}: text")),
            }
        };
        let detail = match text("detail")? {
            None => Detail::Medium,
            Some(d) => Detail::parse(&d).ok_or_else(|| format!("detail {d:?}: short, medium or long"))?,
        };
        let tags = family == Family::Tags;
        let words = match &b["max_words"] {
            Value::Null => detail.words(tags),
            v => {
                let n = v.as_u64().ok_or_else(|| format!("max_words {v}: a whole number"))?;
                if !(8..=400).contains(&n) {
                    return Err(format!("max_words {n}: from 8 to 400"));
                }
                n as usize
            }
        };
        let language = text("language")?.unwrap_or_else(|| "English".into());
        if language.chars().count() > 40 {
            return Err("language: a language's name (40 characters at most)".into());
        }
        let negative = match &b["include_negative"] {
            Value::Null => false,
            Value::Bool(x) => *x,
            v => return Err(format!("include_negative {v}: true or false")),
        };
        let instructions = text("instructions")?;
        if instructions.as_ref().is_some_and(|i| i.chars().count() > 2000) {
            return Err("instructions: 2,000 characters at most".into());
        }
        let max_pixels = match &b["max_pixels"] {
            Value::Null => DEFAULT_PIXELS,
            v => {
                let n = v.as_u64().ok_or_else(|| format!("max_pixels {v}: a whole number"))?;
                if !(MIN_PIXELS..=MAX_PIXELS).contains(&n) {
                    return Err(format!("max_pixels {n}: from {MIN_PIXELS} to {MAX_PIXELS}"));
                }
                n
            }
        };
        let temperature = match &b["temperature"] {
            Value::Null => 0.0,
            v => {
                let t = v.as_f64().ok_or_else(|| format!("temperature {v}: a number"))?;
                if !(0.0..=2.0).contains(&t) {
                    return Err(format!("temperature {t}: from 0 to 2"));
                }
                t
            }
        };
        let seed = match &b["seed"] {
            Value::Null => None,
            v => Some(v.as_u64().ok_or_else(|| format!("seed {v}: a whole number, 0 or more"))?),
        };
        // a token is ~0.75 of a word; the negative and the labels besides
        let max_tokens = match &b["max_tokens"] {
            Value::Null => words * 2 + if negative { 120 } else { 40 },
            v => {
                let n = v.as_u64().ok_or_else(|| format!("max_tokens {v}: a whole number"))?;
                if !(16..=4096).contains(&n) {
                    return Err(format!("max_tokens {n}: from 16 to 4096"));
                }
                n as usize
            }
        };
        Ok(Ask { family, target_model, detail, words, language, negative: negative && family.takes_negative(), instructions, max_pixels,
                 temperature, seed, max_tokens })
    }

    /// The instruction: a system turn (the same for every picture of this family and these settings: the prompt cache
    /// keeps it) and the user turn the picture goes before
    pub fn instruction(&self) -> (String, String) {
        let tags = self.family == Family::Tags;
        let size = if tags {
            format!("Write about {} tags (no more than {}).", self.words, self.words + self.words / 3)
        } else {
            format!("Write about {} words (no more than {}).", self.words, self.words + self.words / 3)
        };
        let mut sys = String::from(
            "You write prompts for text-to-image models. You are shown one picture; write the prompt that would make an image \
             model draw that picture as closely as possible. Describe only what is visible - never invent what is not there. Say \
             what kind of picture it is (a photograph, a 3D render, a flat vector graphic, an oil painting, a pixel-art sprite ...); \
             the main subjects with their colours, shapes, materials, counts and places in the frame (left, right, top, bottom, \
             centre); the background; the light; the viewpoint and framing; the overall colour palette. Copy any text in the \
             picture exactly, in double quotes.\n");
        sys += self.family.style();
        sys += "\n";
        sys += &size;
        sys += &format!(" Write the prompt in {}.\n", self.language);
        if self.negative {
            sys += "Answer in exactly two lines:\nPROMPT: <the prompt>\nNEGATIVE: <a short comma-separated list of what would spoil a \
                    copy of this picture: the wrong style, defects, things that are not in it>\nNothing else.";
        } else {
            sys += "Answer in exactly one line:\nPROMPT: <the prompt>\nNothing else.";
        }
        let mut user = String::from("Write the prompt for this picture.");
        if let Some(i) = &self.instructions {
            user += "\nAlso: ";
            user += i;
        }
        (sys, user)
    }

    /// The chat server's `/api/chat` body: the instruction, the picture (PNG, base64) before the user's words, streamed,
    /// the thinking closed (or a short budget when asked)
    pub fn chat_body(&self, model: &str, png_b64: &str) -> Value {
        let (sys, user) = self.instruction();
        let mut b = json!({"model": model, "stream": true, "max_tokens": self.max_tokens, "temperature": self.temperature, "top_p": 0.95,
                           "think_budget": 0,
                           "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user, "images": [png_b64]}]});
        if let Some(s) = self.seed {
            b["seed"] = json!(s);
        }
        b
    }
}

/// The answer cut into the prompt and the negative: `PROMPT:` / `NEGATIVE:` labels (any case, bold or not), else the
/// whole text as the prompt; quotes around the whole, code fences and Markdown emphasis dropped
pub fn parse_answer(text: &str) -> (String, Option<String>) {
    let (prompt, negative, _) = parse_answer_ran_on(text);
    (prompt, negative)
}

/// `parse_answer`, and whether the describer kept writing after the prompt (a second `PROMPT:` label, or the prompt
/// written first and again after the label): such an answer is whole even when it ran to the token cap
pub fn parse_answer_ran_on(text: &str) -> (String, Option<String>, bool) {
    let mut t = text.replace("```", "").replace("**", "").replace("__", "");
    let lower = t.to_ascii_lowercase();
    // "prompt:" not the end of "negative prompt:"
    let mut labels = lower.match_indices("prompt:").filter(|(i, _)| !lower[..*i].ends_with("negative ")).map(|(i, k)| (i, i + k.len()));
    let p = labels.next();
    // the describer sometimes writes the prompt, then "PROMPT:" and the same again until the token cap: stop at the second label
    let mut ran_on = false;
    if let Some((ss, _)) = labels.next() {
        t.truncate(ss);
        ran_on = true;
    }
    let lower = t.to_ascii_lowercase();
    let find = |k: &str| lower.find(k).map(|i| (i, i + k.len()));
    let n = find("negative prompt:").or_else(|| find("negative:"));
    // a label after at least eight unlabelled words: those words are the prompt
    let lead = p.filter(|(ps, _)| n.is_none_or(|(ns, _)| ns > *ps) && t[..*ps].split_whitespace().count() >= 8);
    ran_on |= lead.is_some();
    let (prompt, negative) = match (p, n) {
        (Some((ps, _)), Some((_, ne))) if lead.is_some() => (&t[..ps], Some(&t[ne..])),
        (Some((ps, _)), None) if lead.is_some() => (&t[..ps], None),
        (Some((_, pe)), Some((ns, ne))) if ns > pe => (&t[pe..ns], Some(&t[ne..])),
        (Some((ps, pe)), Some((_, ne))) if ne <= ps => (&t[pe..], Some(&t[ne..ps])),
        (Some((_, pe)), None) => (&t[pe..], None),
        (None, Some((ns, ne))) => (&t[..ns], Some(&t[ne..])),
        _ => (t.as_str(), None),
    };
    let clean = |s: &str| -> String {
        let s = s.trim().trim_start_matches(['-', ':', ' ']).trim();
        let s = s.strip_prefix('"').and_then(|x| x.strip_suffix('"')).filter(|x| !x.contains('"')).unwrap_or(s);
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    };
    let prompt = clean(prompt);
    let negative = negative.map(clean).filter(|n| !n.is_empty());
    (prompt, negative, ran_on)
}

/// A canvas for the target at the picture's aspect: the family's best area, sides its multiple, within its limits
pub fn suggested_size(width: u32, height: u32, family: Family) -> (u32, u32) {
    let (area, m, lo, hi) = family.canvas();
    let ar = (width.max(1) as f64 / height.max(1) as f64).clamp(0.25, 4.0);
    let fit = |x: f64| -> u32 {
        let r = ((x / m as f64).round() as u32).max(1) * m;
        r.clamp(lo.div_ceil(m) * m, hi / m * m)
    };
    (fit((area * ar).sqrt()), fit((area / ar).sqrt()))
}

/// The picture as the describer is shown it: decoded (PNG, JPEG, WebP), transparency on white, at most `max_pixels`
/// (Lanczos), as PNG; and its size as sent and as shown
pub fn prepare(bytes: &[u8], max_pixels: u64) -> Result<(Vec<u8>, Size, Size), String> {
    let p = crate::image::picture(bytes, "image")?;
    let (w, h) = (p.width, p.height);
    let rgb = if p.channels == 4 {
        let mut q = nextsycl_image::Picture::new(w, h, 3);
        for (o, i) in q.data.chunks_exact_mut(3).zip(p.data.chunks_exact(4)) {
            let a = i[3] as u32;
            for c in 0..3 {
                o[c] = ((i[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
            }
        }
        q
    } else {
        p.rgb()
    };
    let (sw, sh) = shown_size(w, h, max_pixels);
    let shown = if (sw, sh) == (w, h) { rgb } else { rgb.resize_lanczos(sw, sh) };
    Ok((shown.png_bytes(&[])?, (w, h), (sw, sh)))
}

/// The size a picture is shown at: itself when it is at most `max_pixels`, else scaled down to that (at least 32 a side)
pub fn shown_size(w: u32, h: u32, max_pixels: u64) -> (u32, u32) {
    let px = w as u64 * h as u64;
    if px <= max_pixels {
        return (w, h);
    }
    let s = (max_pixels as f64 / px as f64).sqrt();
    (((w as f64 * s).floor() as u32).max(32), ((h as f64 * s).floor() as u32).max(32))
}

/// What a streamed `/api/chat` answer came to: its text, its thinking (none when closed) and its last line
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Streamed {
    pub content: String,
    pub thinking: String,
    pub done: Value,
}

impl Streamed {
    /// One JSON line of the stream folded in; an error line is the request's failure
    pub fn line(&mut self, line: &str) -> Result<(), String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("the chat server's stream: {e}"))?;
        if let Some(e) = v["error"]["message"].as_str().or(v["error"].as_str()) {
            return Err(e.to_string());
        }
        if let Some(t) = v["content"].as_str() {
            self.content.push_str(t);
        }
        if let Some(t) = v["thinking"].as_str() {
            self.thinking.push_str(t);
        }
        if v["done"] == true {
            self.done = v;
        }
        Ok(())
    }
}

/// An error answer's status as the front door gives it: the switcher's own (409 in use, 503 no room) as they are, a
/// refused request 400, anything else the server's failure (502)
fn status_of(st: u16) -> u16 {
    match st {
        400 | 409 | 503 => st,
        _ => 502,
    }
}

/// `/api/chat` at `t`, streamed: the lines folded as they come; `cancelled` asked every second (a cancel closes the
/// connection: the server stops at its next token), `limit` the longest it may take in all
pub fn stream_chat(t: &Target, body: &Value, cancelled: &dyn Fn() -> bool, limit: Duration) -> Result<Streamed, Fail> {
    let mut s = Conn::connect(t).map_err(|e| (502, format!("the chat server does not answer at {t} ({e})")))?;
    let text = serde_json::to_vec(body).map_err(|e| (500, e.to_string()))?;
    write!(s, "POST /api/chat HTTP/1.1\r\nHost: nextsycl\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", text.len())
        .and_then(|_| s.write_all(&text))
        .map_err(|e| (502, format!("the chat server: {e}")))?;
    s.set_read_timeout(Some(Duration::from_secs(1))).map_err(|e| (502, e.to_string()))?;
    let t0 = Instant::now();
    let mut buf: Vec<u8> = Vec::new();
    let mut head: Option<(u16, usize)> = None;
    let mut out = Streamed::default();
    let mut used = 0;
    let mut chunk = [0u8; 16384];
    loop {
        if cancelled() {
            // the connection dropped with `s`: the server's next write fails and it stops the request
            return Err((499, "cancelled".into()));
        }
        if t0.elapsed() > limit {
            return Err((504, format!("the describe took more than {} s", limit.as_secs())));
        }
        let n = match s.read(&mut chunk) {
            Ok(n) => n,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted) => continue,
            Err(e) => return Err((502, format!("the chat server: {e}"))),
        };
        if n > 0 {
            buf.extend_from_slice(&chunk[..n]);
        }
        if head.is_none() {
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let line = String::from_utf8_lossy(&buf[..i]).lines().next().unwrap_or("").to_string();
                let st = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or((502, format!("the chat server: a malformed answer ({line})")))?;
                head = Some((st, i + 4));
                used = i + 4;
            } else if n == 0 {
                return Err((502, "the chat server closed the connection without an answer".into()));
            }
        }
        if let Some((st, body_at)) = head {
            if st != 200 {
                if n > 0 {
                    continue; // the error's body to its end
                }
                let b = String::from_utf8_lossy(&buf[body_at..]).to_string();
                let m = serde_json::from_str::<Value>(&b).ok().and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                    .unwrap_or_else(|| b.trim().chars().take(300).collect());
                return Err((status_of(st), m));
            }
            // whole lines as they come
            while let Some(j) = buf[used..].iter().position(|c| *c == b'\n') {
                let line = String::from_utf8_lossy(&buf[used..used + j]).to_string();
                used += j + 1;
                out.line(&line).map_err(|e| (502, e))?;
            }
            if n == 0 {
                let rest = String::from_utf8_lossy(&buf[used..]).to_string();
                out.line(&rest).map_err(|e| (502, e))?;
                break;
            }
        }
    }
    if out.done.is_null() {
        return Err((502, "the chat server's answer ended early".into()));
    }
    Ok(out)
}

/// The describe's answer from what the chat server wrote
pub fn answer(ask: &Ask, model: &str, s: &Streamed, size: (u32, u32), shown: (u32, u32), seconds: f64) -> Value {
    let (prompt, negative, ran_on) = parse_answer_ran_on(&s.content);
    let negative = negative.filter(|_| ask.negative);
    let (sw, sh) = suggested_size(size.0, size.1, ask.family);
    let d = &s.done;
    let f = |k: &str| d[k].as_f64();
    let prompt_tokens = d["prompt_tokens"].as_u64().unwrap_or(0);
    let reused = d["reused"].as_u64().unwrap_or(0);
    let read = f("read_seconds");
    let gen = f("generate_seconds");
    let generated = d["generated"].as_u64().unwrap_or(0);
    let round = |x: f64, k: f64| (x * k).round() / k;
    let pp = read.filter(|r| *r > 0.0).map(|r| round((prompt_tokens - reused.min(prompt_tokens)) as f64 / r, 10.0));
    let tg = gen.filter(|g| *g > 0.0).map(|g| round(generated as f64 / g, 10.0));
    let words = if ask.family == Family::Tags { prompt.split(',').filter(|x| !x.trim().is_empty()).count() } else { prompt.split_whitespace().count() };
    let mut notes: Vec<String> = Vec::new();
    if !ask.family.takes_negative() {
        notes.push(format!("{} takes no negative prompt (no guidance): none written", ask.family.name()));
    }
    // the cap cut the describer's run-on after a whole prompt: that is a stop, not a short prompt
    let finish = if d["finish"] == "length" && ran_on && !prompt.is_empty() { json!("stop") } else { d["finish"].clone() };
    if finish == "length" {
        notes.push("the answer reached max_tokens: the prompt may be cut short".into());
    } else if d["finish"] == "length" {
        notes.push("the describer kept writing after the prompt until max_tokens: the prompt is whole and was cut where it ended".into());
    }
    if prompt.is_empty() {
        notes.push("the describer wrote no prompt".into());
    }
    json!({
        "prompt": prompt,
        "negative_prompt": negative,
        "target": ask.family.name(),
        "target_model": ask.target_model,
        "model": model,
        "detail": ask.detail.name(),
        "language": ask.language,
        "words": words,
        "width": size.0, "height": size.1,
        "seen": {"width": shown.0, "height": shown.1},
        "suggested_size": {"width": sw, "height": sh, "size": format!("{sw}x{sh}")},
        "tokens": {"prompt": prompt_tokens, "reused": reused, "generated": generated},
        "timings": {"pp_tok_s": pp, "tg_tok_s": tg, "prompt_seconds": read.map(|x| round(x, 1000.0)), "generate_seconds": gen.map(|x| round(x, 1000.0)),
                    "seconds": round(seconds, 1000.0)},
        "finish": finish,
        "energy_wh": d["energy_wh"],
        "notes": notes,
    })
}

/// A whole describe: the picture prepared, the describer asked through `t` (the switcher swaps `model` in), the answer
pub fn describe(t: &Target, model: &str, ask: &Ask, picture: &[u8], cancelled: &dyn Fn() -> bool) -> Result<Value, Fail> {
    let t0 = Instant::now();
    let (png, size, shown) = prepare(picture, ask.max_pixels).map_err(|e| (400, e))?;
    let body = ask.chat_body(model, &http::base64(&png));
    // the swap to the describer (a load) and the reading: half an hour at most
    let s = stream_chat(t, &body, cancelled, Duration::from_secs(1800))?;
    Ok(answer(ask, model, &s, size, shown, t0.elapsed().as_secs_f64()))
}

impl Ask {
    /// The image server's `/v1/ocr` body for a native describer (an image model's own text encoder, served as a reading
    /// model: image/vldescribe): the same instruction - the system turn as `system`, the user turn as `prompt` -, the
    /// picture, the same limits
    pub fn ocr_body(&self, describer: &str, png_b64: &str) -> Value {
        let (sys, user) = self.instruction();
        let mut b = json!({"model": describer, "image": png_b64, "prompt": user, "system": sys, "max_tokens": self.max_tokens, "temperature": self.temperature,
                           "tiles": false, "columns": false, "each_page": false, "figures": false});
        if let Some(s) = self.seed {
            b["seed"] = json!(s);
        }
        b
    }
}

/// A reading server's answer (`/v1/ocr`) as a stream's would have been: its text, and the numbers in the chat
/// server's names (nothing reused: a native describer keeps no prompt cache; `length` when it wrote `max_tokens`)
pub fn streamed_of_ocr(v: &Value, max_tokens: usize) -> Streamed {
    let generated = v["tokens"].as_u64().unwrap_or(0);
    let finish = if generated as usize >= max_tokens { "length" } else { "stop" };
    Streamed {
        content: v["text"].as_str().unwrap_or("").to_string(),
        thinking: String::new(),
        done: json!({"done": true, "finish": finish, "prompt_tokens": v["prompt_tokens"], "reused": 0, "generated": generated,
                     "read_seconds": v["prompt_seconds"], "generate_seconds": v["write_seconds"], "energy_wh": v["wh"]}),
    }
}

/// A whole describe by a native describer: the picture prepared, `describer` (served by the image server at `t`) asked
/// through `/v1/ocr`, the answer with `model` the words for what ran ("qwen-image-2.1-q8 text encoder (Qwen3-VL-8B-Instruct)")
pub fn describe_native(t: &Target, describer: &str, model: &str, ask: &Ask, picture: &[u8]) -> Result<Value, Fail> {
    let t0 = Instant::now();
    let (png, size, shown) = prepare(picture, ask.max_pixels).map_err(|e| (400, e))?;
    let body = ask.ocr_body(describer, &http::base64(&png));
    let v = match http::call_status(t, "POST", "/v1/ocr", Some(&body), 1800) {
        Ok((200, v)) => v,
        Ok((st, v)) => return Err((status_of(st), v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("the describer failed").to_string())),
        Err(e) => return Err((502, format!("the image server: {e}"))),
    };
    Ok(answer(ask, model, &streamed_of_ocr(&v, ask.max_tokens), size, shown, t0.elapsed().as_secs_f64()))
}

/// The architecture of a native describer (an image model's own text encoder writing: image/vldescribe)
pub const NATIVE_ARCH: &str = "qwen3vl-describe";

/// What runs a describe (`describer`, docs/api/image-describe.md): a chat model that reads pictures, or a native
/// describer (an image model's text encoder, served by the image server)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    /// the answer's `describer`: "best", "native" or the id the request named
    pub describer: String,
    /// the registry's (or a card's) id of what runs
    pub id: String,
    /// whether it is a native describer (the image kind) rather than a chat model
    pub native: bool,
    /// the answer's `model`: the chat model's id, or "<image model> text encoder (<its name>)"
    pub model: String,
    /// whether `id` is registered here (else a card's: model_not_downloaded, or fetched with NS_AUTO_PULL)
    pub registered: bool,
}

fn sv(m: &Value, k: &str) -> String {
    m[k].as_str().unwrap_or("").to_string()
}

/// A registry entry or card that describes as a chat model: an llm with a vision tower (a `vision` file, or a sidecar
/// naming one)
pub fn chat_reads_pictures(m: &Value) -> bool {
    m["kind"].as_str().unwrap_or("llm") == "llm"
        && (m["files"]["vision"].is_string() || m["sidecars"].as_array().is_some_and(|a| a.iter().filter_map(|x| x.as_str()).any(|x| x.contains("vision") || x.contains("mmproj"))))
}

/// A registry entry or card that is a native describer (an image model's text encoder as a describer)
pub fn is_native(m: &Value) -> bool {
    m["kind"] == "image" && m["arch"] == NATIVE_ARCH
}

/// The card a registry entry was pulled from (its `catalog`, else its id)
pub fn card_of<'a>(m: &Value, cards: &'a [Value]) -> Option<&'a Value> {
    [sv(m, "catalog"), sv(m, "id")].into_iter().filter(|i| !i.is_empty()).find_map(|id| cards.iter().find(|c| c["id"] == id.as_str()))
}

/// An image model's text encoder as its card says (`encoder: {name, sees, why}`): its name, whether it reads pictures,
/// and why it cannot describe when it reads them but cannot write here
pub fn encoder_of(card: &Value) -> Option<(String, bool, Option<String>)> {
    let e = &card["encoder"];
    let name = e["name"].as_str().or(e.as_str())?.to_string();
    Some((name, e["sees"] == true, e["why"].as_str().map(str::to_string)))
}

/// The native describer cards of an image model's card: those reading its text encoder's file (the same file key)
pub fn natives_of<'a>(image_card: &Value, cards: &'a [Value]) -> Vec<&'a Value> {
    let Some(te) = image_card["files"]["text-encoder"].as_str() else { return Vec::new() };
    cards.iter().filter(|c| is_native(c) && c["files"]["text-encoder"].as_str() == Some(te)).collect()
}

/// The image models (registered, else cards) a native describer card is the text encoder of: same file key
pub fn shared_with(describer: &Value, all: &[Value], cards: &[Value]) -> Vec<String> {
    let Some(te) = describer["files"]["text-encoder"].as_str() else { return Vec::new() };
    let mut out: Vec<String> = cards.iter().filter(|c| c["kind"] == "image" && !is_native(c) && c["files"]["text-encoder"].as_str() == Some(te))
        .map(|c| sv(c, "id")).collect();
    // registered from those cards under another id
    for m in all {
        if m["kind"] == "image" && !is_native(m) && card_of(m, cards).is_some_and(|c| c["files"]["text-encoder"].as_str() == Some(te)) && !out.contains(&sv(m, "id")) {
            out.push(sv(m, "id"));
        }
    }
    out.sort();
    out
}

/// A describer by its id (`describer: <id>`, or the older `model`): a registered chat model that reads pictures or a
/// native describer, else a card's (not downloaded); `field` the request key that named it
pub fn by_id(id: &str, field: &str, all: &[Value], cards: &[Value]) -> Result<Choice, Fail> {
    let model_of = |m: &Value, cards: &[Value]| -> String {
        if !is_native(m) {
            return sv(m, "id");
        }
        // a native describer named by its id: the image models it is the encoder of, and its name
        let card = card_of(m, cards);
        let base = card.map(|c| sv(c, "base_model")).filter(|b| !b.is_empty()).unwrap_or_else(|| sv(m, "id"));
        let shared = card.map(|c| shared_with(c, all, cards)).unwrap_or_default();
        match shared.first() {
            Some(first) => format!("{first} text encoder ({base})"),
            None => format!("{} ({base})", sv(m, "id")),
        }
    };
    let sees = |m: &Value| chat_reads_pictures(m) || is_native(m);
    let refuse = |what: &str| (400, format!("{field} {id}: {what} does not read pictures; a describer is a chat model with a vision tower (qwen3.8-27b-q4-describe) or an image model's text encoder (image.describers)"));
    if let Some(m) = all.iter().find(|m| m["id"] == id) {
        if !sees(m) {
            let k = registry_kind(m);
            return Err(refuse(&format!("a{} {k} model that", if matches!(k, "llm" | "image" | "audio") { "n" } else { "" })));
        }
        if m["enabled"] == false {
            return Err((409, format!("{field} {id}: it is disabled (system.enable)")));
        }
        return Ok(Choice { describer: id.to_string(), id: id.to_string(), native: is_native(m), model: model_of(m, cards), registered: true });
    }
    if let Some(c) = cards.iter().find(|c| c["id"] == id && c["kind"] != "lora") {
        if !sees(c) {
            return Err(refuse(&format!("the {} model card", sv(c, "kind"))));
        }
        return Ok(Choice { describer: id.to_string(), id: id.to_string(), native: is_native(c), model: model_of(c, cards), registered: false });
    }
    Err((404, format!("{field} {id}: no such describer here (image.describers lists them)")))
}

fn registry_kind(m: &Value) -> &str {
    m["kind"].as_str().unwrap_or("llm")
}

/// `describer: native` for one image model (`target` its id): its card's text encoder, run by the native describer card
/// that reads the same file - or why not
pub fn native_for_model(target: &str, all: &[Value], cards: &[Value]) -> Result<Choice, Fail> {
    let entry = all.iter().find(|m| m["id"] == target);
    let card = match entry {
        Some(m) => card_of(m, cards),
        None => cards.iter().find(|c| c["id"] == target),
    };
    let use_best = "use describer best";
    let Some(card) = card else {
        return Err((400, format!("target {target}: no model card says what its text encoder is; {use_best}")));
    };
    let enc = encoder_of(card);
    let Some(d) = natives_of(card, cards).into_iter().next() else {
        return Err((400, match enc {
            Some((name, false, _)) => format!("target {target}: its text encoder ({name}) does not read pictures; {use_best}"),
            Some((name, true, why)) => format!("target {target}: its text encoder ({name}) reads pictures but cannot write text here ({}); {use_best}",
                                               why.unwrap_or_else(|| "no describer card for it".into())),
            None => format!("target {target}: its card names no text encoder; {use_best}"),
        }));
    };
    let name = enc.map(|e| e.0).unwrap_or_else(|| sv(d, "base_model"));
    let id = sv(d, "id");
    // registered under its card's id (or pulled under another id from it)
    let reg = all.iter().find(|m| is_native(m) && (m["id"] == id.as_str() || m["catalog"] == id.as_str()));
    if let Some(r) = reg.filter(|r| r["enabled"] == false) {
        return Err((409, format!("describer native: {} (target {target}'s text encoder) is disabled (system.enable it, or use describer best)", sv(r, "id"))));
    }
    Ok(Choice { describer: "native".into(), id: reg.map(|r| sv(r, "id")).unwrap_or(id), native: true, model: format!("{target} text encoder ({name})"),
                registered: reg.is_some() })
}

/// `describer: native` for a family: of its enabled, registered image models, the one whose native describer takes the
/// least memory (its card's vram_gib); none can: the first one's reason
pub fn native_for_family(family: Family, all: &[Value], cards: &[Value]) -> Result<Choice, Fail> {
    if matches!(family, Family::Natural | Family::Tags) {
        return Err((400, format!("target {} is a prompt style, not an image model: it has no text encoder of its own; name an image model (or its family) as the target, or use describer best", family.name())));
    }
    let models: Vec<&Value> = all.iter().filter(|m| m["kind"] == "image" && m["enabled"] != false && Family::of_arch(&sv(m, "arch")) == Some(family)).collect();
    if models.is_empty() {
        return Err((409, format!("target {}: no model of the family is enabled here (system.models); name one as the target, or use describer best", family.name())));
    }
    let mem = |c: &Choice| cards.iter().chain(all).find(|m| m["id"] == c.id.as_str()).and_then(|m| m["vram_gib"].as_f64()).unwrap_or(f64::MAX);
    let mut first_err = None;
    let mut best: Option<Choice> = None;
    for m in models {
        match native_for_model(&sv(m, "id"), all, cards) {
            Ok(c) => {
                if best.as_ref().is_none_or(|b| mem(&c) < mem(b)) {
                    best = Some(c);
                }
            }
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    best.ok_or_else(|| first_err.unwrap_or((400, "describer native: no native describer".into())))
}

/// The describer a request asks for (`describer` best | native | an id, and the older `model`), without loading
/// anything: `best` is `pick_best` (the loaded chat model when it reads pictures, else the smallest enabled one)
pub fn choose(b: &Value, family: Family, target_model: Option<&str>, all: &[Value], cards: &[Value], pick_best: &dyn Fn() -> Option<String>) -> Result<Choice, Fail> {
    let describer = match &b["describer"] {
        Value::Null => None,
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        v => return Err((400, format!("describer {v}: best, native or a describer's id (image.describers)"))),
    };
    let model = b["model"].as_str().map(str::trim).filter(|m| !m.is_empty());
    match describer.as_deref() {
        None | Some("best") => match model {
            // the older way to name a describer
            Some(m) => by_id(m, "model", all, cards),
            None => {
                let id = pick_best().ok_or_else(|| (409, "no chat model that reads pictures is enabled (system.enable qwen3.8-27b-q4-describe)".to_string()))?;
                Ok(Choice { describer: "best".into(), model: id.clone(), id, native: false, registered: true })
            }
        },
        Some("native") => {
            if let Some(m) = model {
                return Err((400, format!("describer native: model {m} names a describer too; give one of them")));
            }
            match target_model {
                Some(t) => native_for_model(t, all, cards),
                None => native_for_family(family, all, cards),
            }
        }
        Some(id) => {
            if let Some(m) = model.filter(|m| *m != id) {
                return Err((400, format!("describer {id}: model {m} names another describer (model is describer's older name); give one of them")));
            }
            by_id(id, "describer", all, cards)
        }
    }
}

/// The box's describers for a UI's "Describe with" list (image.describers): every chat model that reads pictures and
/// every native describer, registered or a card's; `size_of` an id's files in bytes, `loaded` the chat model loaded
pub fn listing(all: &[Value], cards: &[Value], size_of: &dyn Fn(&str) -> Option<u64>, best: Option<&str>) -> Value {
    let mut out: Vec<Value> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let entry = |m: &Value, registered: bool| -> Value {
        let id = sv(m, "id");
        let card = if registered { card_of(m, cards) } else { Some(m) };
        let native = is_native(m);
        let base = card.map(|c| sv(c, "base_model")).filter(|b| !b.is_empty()).unwrap_or_else(|| sv(m, "title"));
        let vram = m["vram_gib"].as_f64().or_else(|| card.and_then(|c| c["vram_gib"].as_f64()));
        let size = size_of(&id).map(|b| (b as f64 / 1e9 * 100.0).round() / 100.0);
        json!({"id": id, "name": m["title"], "base_model": base, "kind": if native { "native" } else { "chat" },
               "size_gb": size, "vram_gib": vram,
               "image_models": if native { card.map(|c| shared_with(c, all, cards)).unwrap_or_default() } else { Vec::new() },
               "downloaded": registered, "enabled": registered && m["enabled"] != false, "best": best == Some(id.as_str())})
    };
    for m in all.iter().filter(|m| chat_reads_pictures(m) || is_native(m)) {
        seen.push(sv(m, "id"));
        if let Some(c) = m["catalog"].as_str() {
            seen.push(c.to_string());
        }
        out.push(entry(m, true));
    }
    for c in cards.iter().filter(|c| (chat_reads_pictures(c) || is_native(c)) && !seen.contains(&sv(c, "id"))) {
        out.push(entry(c, false));
    }
    json!({"default": "best", "best": best, "describers": out})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn ask(b: Value, f: Family) -> Ask {
        Ask::from_body(&b, f, None).unwrap()
    }

    #[test]
    fn a_target_is_a_family_or_an_image_models_id() {
        assert_eq!(Family::named("z-image"), Some(Family::ZImage));
        assert_eq!(Family::named("z-image-turbo-int8"), Some(Family::ZImage));
        assert_eq!(Family::named("minimax-h3-image-q8"), Some(Family::H3));
        assert_eq!(Family::named("p-h3-image"), Some(Family::H3));
        assert_eq!(Family::named("qwen-image-2.1-turbo-q8"), Some(Family::QwenImage));
        assert_eq!(Family::named("hunyuanimage-2.1-distilled-q8"), Some(Family::HunyuanImage));
        assert_eq!(Family::named("moss-image-sft-q4kxl"), Some(Family::MossImage));
        assert_eq!(Family::named("ming-image-0.1-design"), Some(Family::MingImage));
        assert_eq!(Family::named(" Tags "), Some(Family::Tags));
        assert_eq!(Family::named(""), Some(Family::Natural));
        assert_eq!(Family::named("stable-diffusion"), None);
        for f in FAMILIES {
            assert_eq!(Family::named(f.name()), Some(f), "{f:?}");
        }
        assert_eq!(Family::of_arch("minimax-h3-image"), Some(Family::H3));
        assert_eq!(Family::of_arch("z-image"), Some(Family::ZImage));
        assert_eq!(Family::of_arch("unlimited-ocr"), None);
    }

    #[test]
    fn the_settings_are_checked_and_the_negative_follows_the_family() {
        let a = ask(json!({}), Family::Natural);
        assert_eq!((a.detail, a.words, a.language.as_str(), a.negative, a.max_pixels, a.temperature), (Detail::Medium, 90, "English", false, DEFAULT_PIXELS, 0.0));
        assert_eq!(a.max_tokens, 90 * 2 + 40);
        assert_eq!(ask(json!({"detail": "short"}), Family::Tags).words, 12);
        assert_eq!(ask(json!({"detail": "long", "max_words": 50}), Family::Natural).words, 50, "max_words wins");
        assert!(ask(json!({"include_negative": true}), Family::QwenImage).negative);
        assert!(!ask(json!({"include_negative": true}), Family::ZImage).negative, "Z-Image takes no negative");
        assert!(!ask(json!({"include_negative": true}), Family::H3).negative);
        for bad in [json!({"detail": "huge"}), json!({"max_words": 3}), json!({"max_pixels": 100}), json!({"max_pixels": 9_000_000}),
                    json!({"temperature": 3}), json!({"include_negative": "yes"}), json!({"max_tokens": 5}), json!({"language": 7}),
                    json!({"seed": -1})] {
            assert!(Ask::from_body(&bad, Family::Natural, None).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_instruction_says_the_familys_way_and_the_answers_shape() {
        let (sys, user) = ask(json!({"instructions": "focus on the light", "language": "German"}), Family::H3).instruction();
        assert!(sys.contains("no other text") && sys.contains("hype words") && sys.contains("in German"), "{sys}");
        assert!(sys.contains("PROMPT:") && !sys.contains("NEGATIVE:"));
        assert!(user.ends_with("Also: focus on the light"));
        let (sys, _) = ask(json!({"include_negative": true}), Family::QwenImage).instruction();
        assert!(sys.contains("painter") && sys.contains("NEGATIVE:"));
        let (sys, _) = ask(json!({}), Family::Tags).instruction();
        assert!(sys.contains("about 25 tags"));
        // the same settings, the same system turn (the prompt cache keeps it); the picture goes with the user turn
        let a = ask(json!({}), Family::ZImage);
        let b = a.chat_body("q", "AAAA");
        assert_eq!(b["messages"][0]["content"], a.instruction().0);
        assert_eq!(b["messages"][1]["images"][0], "AAAA");
        assert_eq!((b["think_budget"].as_u64(), b["stream"].as_bool()), (Some(0), Some(true)));
    }

    #[test]
    fn an_answer_is_cut_into_prompt_and_negative() {
        assert_eq!(parse_answer("PROMPT: a red fox in snow\nNEGATIVE: blur, text"), ("a red fox in snow".into(), Some("blur, text".into())));
        assert_eq!(parse_answer("**Prompt:** \"a red fox\"\n**Negative prompt:** blur"), ("a red fox".into(), Some("blur".into())));
        assert_eq!(parse_answer("a red fox,\n  in snow"), ("a red fox, in snow".into(), None));
        assert_eq!(parse_answer("```\nPROMPT: a sign reading \"OPEN\" by a door\n```"), ("a sign reading \"OPEN\" by a door".into(), None));
        assert_eq!(parse_answer("NEGATIVE: blur\nPROMPT: a fox"), ("a fox".into(), Some("blur".into())));
        assert_eq!(parse_answer("Negative prompt: blur\nPrompt: a fox"), ("a fox".into(), Some("blur".into())));
        assert_eq!(parse_answer("PROMPT: a fox\nNEGATIVE:"), ("a fox".into(), None));
    }

    #[test]
    fn an_unlabelled_prompt_then_its_repeat_is_one_prompt() {
        let p = "A flat vector graphic of a red barn on a green field under a pale blue sky";
        assert_eq!(parse_answer(&format!("{p}\nPROMPT: {p}\nPROMPT: {p}\nPROMPT: {p}")), (p.into(), None));
        // a label first, then the prompt written again after a second label
        assert_eq!(parse_answer("PROMPT: a fox\nPROMPT: a fox\nPROMPT: a fox"), ("a fox".into(), None));
        // a short lead is not a prompt: the labelled text is
        assert_eq!(parse_answer("Sure! PROMPT: a red fox in snow"), ("a red fox in snow".into(), None));
        assert_eq!(parse_answer(&format!("{p}\nPROMPT: {p}\nNEGATIVE: blur")), (p.into(), Some("blur".into())));
    }

    #[test]
    fn a_run_on_after_a_whole_prompt_is_a_stop() {
        let p = "A flat vector graphic of a red barn on a green field under a pale blue sky";
        assert!(parse_answer_ran_on(&format!("{p}\nPROMPT: {p}\nPROMPT: {p}")).2);
        assert!(parse_answer_ran_on("PROMPT: a fox\nPROMPT: a fox").2);
        assert!(!parse_answer_ran_on("PROMPT: a red fox in snow").2);
        assert!(!parse_answer_ran_on("PROMPT: a fox\nNEGATIVE: blur").2);
        let ask = ask(json!({}), Family::ZImage);
        let cut = |content: &str| {
            let s = Streamed { content: content.into(), done: json!({"finish": "length", "generated": 99}), ..Default::default() };
            answer(&ask, "m", &s, (64, 64), (64, 64), 1.0)
        };
        let r = cut(&format!("{p}\nPROMPT: {p}"));
        assert_eq!((r["finish"].as_str(), r["prompt"].as_str()), (Some("stop"), Some(p)));
        assert!(r["notes"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().contains("kept writing")));
        let r = cut("PROMPT: a red fox in the");
        assert_eq!(r["finish"], "length");
    }

    #[test]
    fn a_suggested_size_keeps_the_aspect_on_the_familys_grid() {
        assert_eq!(suggested_size(1024, 1024, Family::ZImage), (1024, 1024));
        let (w, h) = suggested_size(1920, 1080, Family::QwenImage);
        assert!(w % 16 == 0 && h % 16 == 0 && w > h && ((w * h) as f64 / 1048576.0 - 1.0).abs() < 0.05, "{w}x{h}");
        let (w, h) = suggested_size(600, 800, Family::H3);
        assert!(w % 32 == 0 && h % 32 == 0 && h > w && w * h > 3_500_000, "{w}x{h}");
        // a strip is clamped to 4:1 and to the family's sides
        let (w, h) = suggested_size(10000, 100, Family::ZImage);
        assert!(w <= 2048 && h >= 256, "{w}x{h}");
    }

    #[test]
    fn a_picture_is_shown_at_most_its_pixels_on_white() {
        assert_eq!(shown_size(800, 600, DEFAULT_PIXELS), (800, 600));
        let (w, h) = shown_size(4000, 3000, DEFAULT_PIXELS);
        assert!(w as u64 * h as u64 <= DEFAULT_PIXELS && (w as f64 / h as f64 - 4.0 / 3.0).abs() < 0.01, "{w}x{h}");
        // a transparent pixel comes out white, an opaque one as it was
        let mut p = nextsycl_image::Picture::new(2, 1, 4);
        p.data = vec![10, 20, 30, 0, 10, 20, 30, 255];
        let (png, size, shown) = prepare(&p.png_bytes(&[]).unwrap(), DEFAULT_PIXELS).unwrap();
        assert_eq!((size, shown), ((2, 1), (2, 1)));
        let q = nextsycl_image::Picture::decode(&png).unwrap();
        assert_eq!(q.data, vec![255, 255, 255, 10, 20, 30]);
        assert!(prepare(b"not a picture", DEFAULT_PIXELS).is_err());
    }

    #[test]
    fn the_stream_is_folded_and_its_numbers_answered() {
        let mut s = Streamed::default();
        for l in ["{\"content\": \"PROMPT: a fox\"}", "", "{\"content\": \" in snow\"}",
                  "{\"done\": true, \"finish\": \"stop\", \"prompt_tokens\": 1100, \"reused\": 100, \"generated\": 50, \"read_seconds\": 2.0, \"generate_seconds\": 2.5}"] {
            s.line(l).unwrap();
        }
        assert!(s.line("{\"error\": {\"message\": \"boom\"}}").is_err());
        let a = answer(&ask(json!({}), Family::ZImage), "q", &s, (640, 480), (640, 480), 5.0);
        assert_eq!(a["prompt"], "a fox in snow");
        assert_eq!((a["timings"]["pp_tok_s"].as_f64(), a["timings"]["tg_tok_s"].as_f64()), (Some(500.0), Some(20.0)));
        assert_eq!(a["tokens"]["generated"], 50);
        assert!(a["negative_prompt"].is_null() && a["notes"][0].as_str().unwrap().contains("no negative"));
        assert_eq!(a["suggested_size"]["size"], "1184x880");
    }

    /// A box: a chat describer, Qwen-Image 2.1 and P-H3-Image registered (their native describers: one registered, one a
    /// card's), Z-Image (an encoder that does not see), MiniMax-H3-Image (one that sees but cannot write), cards for all
    fn a_box() -> (Vec<Value>, Vec<Value>) {
        let all = vec![
            json!({"id": "qwen3.8-27b-q4-describe", "kind": "llm", "files": {"model": "/m/q.gguf", "vision": "/m/mm.gguf"}, "vram_gib": 20.0, "catalog": "qwen3.8-27b-q4-describe"}),
            json!({"id": "glm", "kind": "llm", "files": {"model": "/m/glm.gguf"}}),
            json!({"id": "qwen-image-2.1-q8", "kind": "image", "arch": "qwen-image-2.1", "catalog": "qwen-image-2.1-q8"}),
            json!({"id": "qwen-image-2.1-turbo-q8", "kind": "image", "arch": "qwen-image-2.1", "catalog": "qwen-image-2.1-turbo-q8"}),
            json!({"id": "p-h3-image", "kind": "image", "arch": "p-h3-image", "catalog": "p-h3-image"}),
            json!({"id": "z-image-turbo", "kind": "image", "arch": "z-image", "catalog": "z-image-turbo"}),
            json!({"id": "minimax-h3-image", "kind": "image", "arch": "minimax-h3-image"}),
            json!({"id": "qwen3-vl-8b-instruct", "kind": "image", "arch": NATIVE_ARCH, "catalog": "qwen3-vl-8b-instruct", "vram_gib": 14.0}),
        ];
        let cards = vec![
            json!({"id": "qwen3.8-27b-q4-describe", "kind": "llm", "arch": "qwen35", "base_model": "Qwen3.8-27B", "title": "Q27 describe", "files": {"model": "q27-q4", "vision": "q27-mmproj"}, "vram_gib": 20.0}),
            json!({"id": "qwen3.8-27b-q8", "kind": "llm", "arch": "qwen35", "base_model": "Qwen3.8-27B", "title": "Q27 Q8", "files": {"model": "q27-q8", "vision": "q27-mmproj"}, "vram_gib": 31.0}),
            json!({"id": "qwen-image-2.1-q8", "kind": "image", "arch": "qwen-image-2.1", "files": {"text-encoder": "qi-te"}, "encoder": {"name": "Qwen3-VL-8B-Instruct", "sees": true}}),
            json!({"id": "qwen-image-2.1-turbo-q8", "kind": "image", "arch": "qwen-image-2.1", "files": {"text-encoder": "qi-te"}, "encoder": {"name": "Qwen3-VL-8B-Instruct", "sees": true}}),
            json!({"id": "qwen-image-2.1-q4km", "kind": "image", "arch": "qwen-image-2.1", "files": {"text-encoder": "qi-te"}, "encoder": {"name": "Qwen3-VL-8B-Instruct", "sees": true}}),
            json!({"id": "p-h3-image", "kind": "image", "arch": "p-h3-image", "files": {"text-encoder": "ph3-te"}, "encoder": {"name": "Qwen3-VL-4B", "sees": true}}),
            json!({"id": "z-image-turbo", "kind": "image", "arch": "z-image", "files": {"text-encoder": "zi-te"}, "encoder": {"name": "Qwen3-4B", "sees": false}}),
            json!({"id": "minimax-h3-image", "kind": "image", "arch": "minimax-h3-image", "files": {"text-encoder": "h3-te"},
                   "encoder": {"name": "Qwen3-VL-32B", "sees": true, "why": "cut after layer 50"}}),
            json!({"id": "hunyuanimage-2.1-q8", "kind": "image", "arch": "hunyuanimage-2.1", "files": {"text-encoder": "hy-te"}}),
            json!({"id": "qwen3-vl-8b-instruct", "kind": "image", "arch": NATIVE_ARCH, "base_model": "Qwen3-VL-8B-Instruct", "title": "QI TE", "files": {"text-encoder": "qi-te"}, "vram_gib": 14.0}),
            json!({"id": "qwen3-vl-4b-instruct", "kind": "image", "arch": NATIVE_ARCH, "base_model": "Qwen3-VL-4B", "title": "PH3 TE", "files": {"text-encoder": "ph3-te"}, "vram_gib": 8.0}),
        ];
        (all, cards)
    }

    fn pick(b: Value, target: &str) -> Result<Choice, Fail> {
        let (all, cards) = a_box();
        let (family, model) = match all.iter().find(|m| m["id"] == target) {
            Some(m) => (Family::of_arch(m["arch"].as_str().unwrap()).unwrap(), Some(target.to_string())),
            None => (Family::named(target).unwrap(), Family::alias(target).is_none().then(|| target.to_string())),
        };
        choose(&b, family, model.as_deref(), &all, &cards, &|| Some("qwen3.8-27b-q4-describe".to_string()))
    }

    #[test]
    fn best_is_todays_describer_and_model_names_one() {
        let c = pick(json!({}), "z-image").unwrap();
        assert_eq!((c.describer.as_str(), c.id.as_str(), c.native, c.model.as_str()), ("best", "qwen3.8-27b-q4-describe", false, "qwen3.8-27b-q4-describe"));
        assert_eq!(pick(json!({"describer": "best"}), "z-image").unwrap(), c);
        // the older model field: a describer by its id, the same as describer: <id>
        let m = pick(json!({"model": "qwen3.8-27b-q4-describe"}), "natural").unwrap();
        assert_eq!((m.describer.as_str(), m.id.as_str()), ("qwen3.8-27b-q4-describe", "qwen3.8-27b-q4-describe"));
        assert_eq!(pick(json!({"describer": "qwen3.8-27b-q4-describe"}), "natural").unwrap(), m);
        assert_eq!(pick(json!({"describer": "qwen3.8-27b-q4-describe", "model": "qwen3.8-27b-q4-describe"}), "natural").unwrap(), m);
        // nothing enabled that reads pictures
        let (all, cards) = a_box();
        let e = choose(&json!({}), Family::Natural, None, &all, &cards, &|| None).unwrap_err();
        assert_eq!(e.0, 409);
    }

    #[test]
    fn a_describer_by_id_reads_pictures() {
        // a native describer named by its id: the image model it is the encoder of, in the model words
        let c = pick(json!({"describer": "qwen3-vl-8b-instruct"}), "qwen-image").unwrap();
        assert!(c.native && c.registered && c.model == "qwen-image-2.1-q4km text encoder (Qwen3-VL-8B-Instruct)", "{c:?}");
        // a card's, not downloaded: chosen, registered false (model_not_downloaded or a pull follows)
        let c = pick(json!({"describer": "qwen3.8-27b-q8"}), "natural").unwrap();
        assert!(!c.registered && !c.native);
        for (b, st, starts) in [(json!({"describer": "glm"}), 400, "describer glm: an llm model that does not read pictures"),
                                (json!({"describer": "z-image-turbo"}), 400, "describer z-image-turbo: an image model"),
                                (json!({"model": "glm"}), 400, "model glm:"),
                                (json!({"describer": "nope"}), 404, "describer nope: no such describer"),
                                (json!({"describer": 7}), 400, "describer 7:"),
                                (json!({"describer": "qwen3.8-27b-q4-describe", "model": "glm"}), 400, "describer qwen3.8-27b-q4-describe: model glm names another"),
                                (json!({"describer": "native", "model": "glm"}), 400, "describer native: model glm names a describer too")] {
            let e = pick(b.clone(), "natural").unwrap_err();
            assert_eq!(e.0, st, "{b}: {}", e.1);
            assert!(e.1.starts_with(starts), "{b}: {}", e.1);
        }
    }

    #[test]
    fn native_is_the_targets_own_encoder_or_why_not() {
        let c = pick(json!({"describer": "native"}), "qwen-image-2.1-turbo-q8").unwrap();
        assert_eq!((c.describer.as_str(), c.id.as_str(), c.native, c.registered), ("native", "qwen3-vl-8b-instruct", true, true));
        assert_eq!(c.model, "qwen-image-2.1-turbo-q8 text encoder (Qwen3-VL-8B-Instruct)");
        // its describer a card's, not registered: model_not_downloaded follows
        let c = pick(json!({"describer": "native"}), "p-h3-image").unwrap();
        assert_eq!((c.id.as_str(), c.registered, c.model.as_str()), ("qwen3-vl-4b-instruct", false, "p-h3-image text encoder (Qwen3-VL-4B)"));
        // a target that is a card's id only (not registered here)
        assert_eq!(pick(json!({"describer": "native"}), "qwen-image-2.1-q4km").unwrap().id, "qwen3-vl-8b-instruct");
        let e = pick(json!({"describer": "native"}), "z-image-turbo").unwrap_err();
        assert_eq!(e, (400, "target z-image-turbo: its text encoder (Qwen3-4B) does not read pictures; use describer best".to_string()));
        let e = pick(json!({"describer": "native"}), "minimax-h3-image").unwrap_err();
        assert!(e.0 == 400 && e.1.contains("reads pictures but cannot write text here (cut after layer 50)"), "{}", e.1);
        let e = pick(json!({"describer": "native"}), "hunyuanimage-2.1-q8").unwrap_err();
        assert!(e.0 == 400 && e.1.contains("its card names no text encoder"), "{}", e.1);
        // a family: its enabled model whose describer takes the least memory; none sees: the first's reason
        let c = pick(json!({"describer": "native"}), "h3").unwrap();
        assert_eq!(c.id, "qwen3-vl-4b-instruct");
        assert_eq!(pick(json!({"describer": "native"}), "qwen-image").unwrap().id, "qwen3-vl-8b-instruct");
        assert_eq!(pick(json!({"describer": "native"}), "z-image").unwrap_err().0, 400);
        assert_eq!(pick(json!({"describer": "native"}), "moss-image").unwrap_err().0, 409, "no moss model enabled");
        for f in ["natural", "tags"] {
            let e = pick(json!({"describer": "native"}), f).unwrap_err();
            assert!(e.0 == 400 && e.1.contains("has no text encoder of its own"), "{f}: {}", e.1);
        }
        // a disabled native describer: conflict
        let (mut all, cards) = a_box();
        all[7]["enabled"] = json!(false);
        assert_eq!(native_for_model("qwen-image-2.1-q8", &all, &cards).unwrap_err().0, 409);
    }

    #[test]
    fn the_listing_has_every_describer_and_its_image_models() {
        let (all, cards) = a_box();
        let v = listing(&all, &cards, &|id| if id == "qwen3-vl-8b-instruct" { Some(9_350_798_360) } else { None }, Some("qwen3.8-27b-q4-describe"));
        assert_eq!((v["default"].as_str(), v["best"].as_str()), (Some("best"), Some("qwen3.8-27b-q4-describe")));
        let list = v["describers"].as_array().unwrap();
        let ids: Vec<&str> = list.iter().map(|d| d["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["qwen3.8-27b-q4-describe", "qwen3-vl-8b-instruct", "qwen3.8-27b-q8", "qwen3-vl-4b-instruct"]);
        let qi = &list[1];
        assert_eq!((qi["kind"].as_str(), qi["base_model"].as_str(), qi["size_gb"].as_f64(), qi["downloaded"].as_bool()), (Some("native"), Some("Qwen3-VL-8B-Instruct"), Some(9.35), Some(true)));
        assert_eq!(qi["image_models"], json!(["qwen-image-2.1-q4km", "qwen-image-2.1-q8", "qwen-image-2.1-turbo-q8"]));
        assert_eq!((list[0]["kind"].as_str(), list[0]["best"].as_bool(), list[0]["image_models"].as_array().map(Vec::len)), (Some("chat"), Some(true), Some(0)));
        assert_eq!((list[3]["downloaded"].as_bool(), list[3]["enabled"].as_bool(), list[3]["image_models"].clone()), (Some(false), Some(false), json!(["p-h3-image"])));
    }

    #[test]
    fn a_native_describe_asks_the_reading_server_with_the_same_instruction() {
        let a = ask(json!({"seed": 3}), Family::QwenImage);
        let b = a.ocr_body("qwen3-vl-8b-instruct", "AAAA");
        let (sys, user) = a.instruction();
        assert_eq!((b["system"].as_str(), b["prompt"].as_str(), b["image"].as_str()), (Some(sys.as_str()), Some(user.as_str()), Some("AAAA")));
        assert_eq!((b["model"].as_str(), b["seed"].as_u64(), b["max_tokens"].as_u64()), (Some("qwen3-vl-8b-instruct"), Some(3), Some(a.max_tokens as u64)));
        // its answer as a stream's: the numbers in the chat server's names; the cap reached is length
        let s = streamed_of_ocr(&json!({"text": "PROMPT: a fox", "prompt_tokens": 1100, "tokens": 40, "prompt_seconds": 2.0, "write_seconds": 4.0, "wh": 0.1}), 220);
        assert_eq!((s.content.as_str(), s.done["finish"].as_str(), s.done["generated"].as_u64(), s.done["reused"].as_u64()), ("PROMPT: a fox", Some("stop"), Some(40), Some(0)));
        let r = answer(&a, "qwen-image-2.1-q8 text encoder (Qwen3-VL-8B-Instruct)", &s, (64, 64), (64, 64), 6.5);
        assert_eq!((r["prompt"].as_str(), r["timings"]["tg_tok_s"].as_f64(), r["timings"]["pp_tok_s"].as_f64()), (Some("a fox"), Some(10.0), Some(550.0)));
        assert_eq!(streamed_of_ocr(&json!({"text": "x", "tokens": 220}), 220).done["finish"], "length");
    }

    /// A fake chat server: its lines one a second; a cancel closes the connection, which it sees
    #[test]
    fn a_cancel_closes_the_stream_and_an_error_keeps_its_status() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let saw_close = Arc::new(AtomicBool::new(false));
        let seen = saw_close.clone();
        let srv = std::thread::spawn(move || {
            // the first: a refusal
            let (mut c, _) = l.accept().unwrap();
            let _ = http::read_request(&mut c).unwrap();
            c.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\r\n{\"error\": {\"message\": \"no room\"}}").unwrap();
            drop(c);
            // the second: a stream it keeps writing until the reader goes
            let (mut c, _) = l.accept().unwrap();
            let _ = http::read_request(&mut c).unwrap();
            c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\r\n").unwrap();
            for _ in 0..50 {
                if c.write_all(b"{\"content\": \"x\"}\n").and_then(|_| c.flush()).is_err() {
                    seen.store(true, Ordering::SeqCst);
                    return;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        });
        let t = Target::Tcp(addr);
        let never = || false;
        let e = stream_chat(&t, &json!({}), &never, Duration::from_secs(20)).unwrap_err();
        assert_eq!(e, (503, "no room".to_string()));
        let t1 = Instant::now();
        let stop = move || t1.elapsed() > Duration::from_millis(1500);
        let e = stream_chat(&t, &json!({}), &stop, Duration::from_secs(20)).unwrap_err();
        assert_eq!(e.0, 499);
        srv.join().unwrap();
        assert!(saw_close.load(Ordering::SeqCst), "the server saw the connection go");
    }
}
