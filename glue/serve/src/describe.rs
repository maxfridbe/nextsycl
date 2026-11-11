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
    let mut t = text.replace("```", "").replace("**", "").replace("__", "");
    let lower = t.to_ascii_lowercase();
    // "prompt:" not the end of "negative prompt:"
    let mut labels = lower.match_indices("prompt:").filter(|(i, _)| !lower[..*i].ends_with("negative ")).map(|(i, k)| (i, i + k.len()));
    let p = labels.next();
    // the describer sometimes writes the prompt, then "PROMPT:" and the same again until the token cap: stop at the second label
    if let Some((ss, _)) = labels.next() {
        t.truncate(ss);
    }
    let lower = t.to_ascii_lowercase();
    let find = |k: &str| lower.find(k).map(|i| (i, i + k.len()));
    let n = find("negative prompt:").or_else(|| find("negative:"));
    // a label after at least eight unlabelled words: those words are the prompt
    let lead = p.filter(|(ps, _)| n.map_or(true, |(ns, _)| ns > *ps) && t[..*ps].split_whitespace().count() >= 8);
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
    (prompt, negative)
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
    let (prompt, negative) = parse_answer(&s.content);
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
    if d["finish"] == "length" {
        notes.push("the answer reached max_tokens: the prompt may be cut short".into());
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
        "finish": d["finish"],
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
