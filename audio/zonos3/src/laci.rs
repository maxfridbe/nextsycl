//! LACI - Localized Attention-Constrained Inference (arXiv 2609.16989) - and short texts.
//!
//! While a take decodes, the alignment heads' attention over the text (`lm::Probe`) is reduced to a reading position
//! a step and fed to a `Detector`. A skip (the reading jumps past a bin of text tokens that got far fewer steps than
//! the bins before it), a hallucination (the reading stops advancing) or a truncation (end of audio with two or more
//! bins unread) rolls the take back to the failure's onset; it is regenerated from there with a new seed under an ACI
//! hard attention mask (arXiv 2404.19723) on the alignment heads until the reading passes the detection point, at most
//! `max_retries` times. The text is done once the last bin has been read for its share and the reading has reached the
//! last text token; the take stops `tail_frames` later, and `stop_boost_delay` rows after the text is done the sampler
//! raises the end-of-audio logit by `stop_boost`.
//!
//! A short text (a word or two: the model was trained on sentences, and alone one or two words often come out wrong or
//! run on into words the text does not have) is generated under LACI as the text, the 0.5 s pause marker and its
//! language's filler sentence, and cut once the reading reaches the filler (`ShortCut`): at the first frame a
//! codes-only silence classifier calls silent; a take that runs into the filler is regenerated as a fresh one.
//!
//! Origin: Zyphra's ZONOS3-Mini runtime (model/src/zonos3/tts/laci.py, short_text.py and their exact C++ ports
//! servers/ggml/src/zonos3-laci.cpp, zonos3-short-text.cpp; MIT). short_text.json is Zyphra's registry as shipped. Not
//! ported: duration targets (PaceMonitor, speaking-rate retakes) and span following - the requests here have neither.
//! A rollback re-prefills the take from the prompt and the kept rows, as Zyphra's ggml server does (no state snapshot:
//! the gated-DeltaNet state is rebuilt by the prefill).

use std::collections::HashSet;
use std::sync::OnceLock;

use nextsycl_audio::{Error, Result};

#[derive(Clone, Debug)]
pub struct Config {
    pub bin_tokens: usize,
    pub cover_frac: f64,
    pub skip_frac: f64,
    pub stall_factor: f64,
    pub min_stall_frames: usize,
    pub tail_frames: usize,
    pub rollback_margin_frames: usize,
    pub max_retries: usize,
    pub pace_window_bins: usize,
    pub warmup_bins: usize,
    pub stop_boost: f32,
    pub stop_boost_delay: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config { bin_tokens: 8, cover_frac: 0.5, skip_frac: 0.15, stall_factor: 3.0, min_stall_frames: 45, tail_frames: 15, rollback_margin_frames: 8,
                 max_retries: 8, pace_window_bins: 16, warmup_bins: 2, stop_boost: 3.0, stop_boost_delay: 6 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub kind: &'static str,
    pub detect_row: usize,
    pub onset_row: usize,
    pub rollback_row: usize,
    pub bin: usize,
    pub frontier: usize,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let (n, i) = (v.len(), v.len() / 2);
    if n % 2 == 1 { v[i] } else { (v[i - 1] + v[i]) / 2.0 }
}

/// Skip / hallucination / truncation detector over (row, pos) a decode step, rows strictly increasing
pub struct Detector {
    pub text_len: usize,
    prior: f64,
    cfg: Config,
    min_rollback_row: usize,
    pub nb: usize,
    obs_rows: Vec<usize>,
    obs_pos: Vec<usize>,
    bumps: Vec<usize>,
    counts: Vec<usize>,
    first_row: Vec<Option<usize>>,
    frontier: usize,
    last_advance_row: Option<usize>,
    done_row: Option<usize>,
    read_row: Option<usize>,
}

impl Detector {
    pub fn new(text_len: usize, prior: f64, cfg: &Config, min_rollback_row: usize) -> Detector {
        let nb = text_len.max(1).div_ceil(cfg.bin_tokens);
        Detector { text_len: text_len.max(1), prior, cfg: cfg.clone(), min_rollback_row, nb, obs_rows: Vec::new(), obs_pos: Vec::new(), bumps: Vec::new(),
                   counts: vec![0; nb], first_row: vec![None; nb], frontier: 0, last_advance_row: None, done_row: None, read_row: None }
    }

    fn reset(&mut self) {
        self.counts = vec![0; self.nb];
        self.first_row = vec![None; self.nb];
        self.frontier = 0;
        self.last_advance_row = None;
        self.done_row = None;
        self.read_row = None;
    }

    pub fn frontier(&self) -> usize {
        self.frontier
    }

    pub fn done_row(&self) -> Option<usize> {
        self.done_row
    }

    fn tokens(&self, b: usize) -> usize {
        self.cfg.bin_tokens.min(self.text_len - b * self.cfg.bin_tokens)
    }

    /// the median frames a token of the last pace_window_bins bins below the frontier (the prior until warmup_bins)
    pub fn pace_tok(&self) -> f64 {
        let lo = self.frontier.saturating_sub(self.cfg.pace_window_bins);
        let rates: Vec<f64> = (lo..self.frontier).map(|x| self.counts[x] as f64 / self.tokens(x) as f64).collect();
        if rates.len() < self.cfg.warmup_bins { self.prior } else { median(rates) }
    }

    fn share(&self, b: usize) -> f64 {
        self.pace_tok() * self.tokens(b) as f64
    }

    pub fn last_pos_before(&self, row: usize) -> usize {
        let i = self.obs_rows.partition_point(|r| *r < row);
        if i > 0 { self.obs_pos[i - 1] } else { 0 }
    }

    pub fn stop_row(&self) -> Option<usize> {
        self.done_row.map(|d| d + self.cfg.tail_frames)
    }

    /// the first row a later rollback could reach (usize::MAX once the text is done)
    pub fn commit_row(&self, next_row: usize) -> usize {
        if self.done_row.is_some() {
            return usize::MAX;
        }
        let mut onset = self.last_advance_row.unwrap_or(next_row).min(next_row);
        for b in self.frontier + 2..self.nb {
            if let Some(r) = self.first_row[b] {
                onset = onset.min(r);
            }
        }
        self.rollback(onset)
    }

    fn rollback(&self, onset: usize) -> usize {
        onset.saturating_sub(self.cfg.rollback_margin_frames).max(self.min_rollback_row)
    }

    pub fn observe(&mut self, row: usize, pos: usize) -> Result<Option<Failure>> {
        if pos >= self.text_len {
            return Err(Error(format!("LACI: pos {pos} outside the text span [0, {})", self.text_len)));
        }
        if self.obs_rows.last().is_some_and(|r| row <= *r) {
            return Err(Error(format!("LACI: rows must increase: {row} after {:?}", self.obs_rows.last())));
        }
        self.obs_rows.push(row);
        self.obs_pos.push(pos);
        Ok(self.step(row, pos))
    }

    fn step(&mut self, row: usize, pos: usize) -> Option<Failure> {
        let b = pos / self.cfg.bin_tokens;
        self.counts[b] += 1;
        if self.first_row[b].is_none() {
            self.first_row[b] = Some(row);
        }
        if self.last_advance_row.is_none() {
            self.last_advance_row = Some(row);
        }
        if b > self.frontier && self.counts[b] as f64 >= self.cfg.cover_frac * self.share(b) {
            for x in self.frontier + 1..b {
                if (self.counts[x] as f64) < self.cfg.skip_frac * self.share(x) {
                    let onset = (x + 1..self.nb).filter_map(|y| self.first_row[y]).min().unwrap_or(row);
                    return Some(Failure { kind: "skip", detect_row: row, onset_row: onset, rollback_row: self.rollback(onset), bin: x, frontier: b });
                }
            }
            self.frontier = b;
            self.last_advance_row = Some(row);
        }
        if self.done_row.is_none() {
            let limit = (self.cfg.stall_factor * self.pace_tok() * self.cfg.bin_tokens as f64).max(self.cfg.min_stall_frames as f64);
            let last = self.last_advance_row.unwrap_or(row);
            if (row - last) as f64 > limit {
                return Some(Failure { kind: "hallucination", detect_row: row, onset_row: last, rollback_row: self.rollback(last), bin: self.frontier + 1,
                                      frontier: self.frontier });
            }
        }
        if self.read_row.is_none() && pos + 1 >= self.text_len {
            self.read_row = Some(row);
        }
        if self.done_row.is_none() && self.read_row.is_some() && self.frontier + 2 >= self.nb && self.counts[self.nb - 1] as f64 >= self.share(self.nb - 1) {
            self.done_row = Some(row);
        }
        None
    }

    /// truncation: end of audio with two or more bins unread
    pub fn on_eos(&self, row: usize) -> Option<Failure> {
        (self.frontier + 2 < self.nb).then(|| Failure { kind: "truncation", detect_row: row, onset_row: row, rollback_row: self.rollback(row),
                                                         bin: self.frontier + 1, frontier: self.frontier })
    }

    /// keep the observations before `row` (replayed), the survivors' stall clock moved up to it
    pub fn truncate(&mut self, row: usize) {
        let keep = self.obs_rows.partition_point(|r| *r < row);
        let rows: Vec<usize> = self.obs_rows[..keep].to_vec();
        let poss: Vec<usize> = self.obs_pos[..keep].to_vec();
        let mut bumps: Vec<usize> = self.bumps.iter().copied().filter(|r| *r < row).collect();
        self.obs_rows.clear();
        self.obs_pos.clear();
        self.reset();
        let mut pending = 0;
        for (r, p) in rows.into_iter().zip(poss) {
            while pending < bumps.len() && r > bumps[pending] {
                if let Some(l) = self.last_advance_row.as_mut() {
                    *l = (*l).max(bumps[pending]);
                }
                pending += 1;
            }
            self.obs_rows.push(r);
            self.obs_pos.push(p);
            let _ = self.step(r, p);
        }
        if let Some(l) = self.last_advance_row.as_mut() {
            *l = (*l).max(row);
        }
        bumps.push(row);
        self.bumps = bumps;
    }
}

/// ACI's dynamic-programming attention centre (2404.19723 Eq. 2/5), held monotone and moved at most +1 an update
pub struct AciCenter {
    text_len: usize,
    c0: usize,
    pub center: usize,
    d: Vec<f64>,
}

impl AciCenter {
    pub fn new(c0: usize, text_len: usize) -> AciCenter {
        let c0 = c0.min(text_len.saturating_sub(1));
        AciCenter { text_len, c0, center: c0, d: vec![0.0] }
    }

    pub fn update(&mut self, m: f64) -> usize {
        if self.c0 + self.d.len() < self.text_len {
            self.d.push(f64::INFINITY);
        }
        for i in (1..self.d.len()).rev() {
            let e = m - (self.c0 + i) as f64;
            self.d[i] = self.d[i].min(self.d[i - 1]) + e * e;
        }
        let e0 = m - self.c0 as f64;
        self.d[0] += e0 * e0;
        let mut best = 0;
        for i in 1..self.d.len() {
            if self.d[i] < self.d[best] {
                best = i;
            }
        }
        self.center = (self.c0 + best).max(self.center).min(self.center + 1);
        self.center
    }
}

/// One language's alignment heads (Zyphra's web/laci/zonos3_mini_heads*.json)
#[derive(Clone, Debug)]
pub struct Heads {
    pub heads: Vec<(usize, usize)>,
    /// the ACI mask's half-width per head, first insertion order
    pub radius: Vec<((usize, usize), usize)>,
    pub frames_per_token: f64,
}

impl Heads {
    pub fn parse(text: &str, name: &str) -> Result<Heads> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| Error(format!("{name}: {e}")))?;
        let list = v["heads"].as_array().filter(|a| !a.is_empty()).ok_or_else(|| Error(format!("{name}: no alignment heads")))?;
        let mut h = Heads { heads: Vec::new(), radius: Vec::new(), frames_per_token: v["frames_per_token"].as_f64().ok_or_else(|| Error(format!("{name}: no frames_per_token")))? };
        for e in list {
            let get = |k: &str| e[k].as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{name}: a head without {k}")));
            let hd = (get("layer")?, get("head")?);
            let r = get("radius")?;
            h.heads.push(hd);
            match h.radius.iter_mut().find(|(x, _)| *x == hd) {
                Some(x) => x.1 = r,
                None => h.radius.push((hd, r)),
            }
        }
        Ok(h)
    }
}

/// Zyphra's index of the heads files: a language code (the server's: en_us, fr_fr, pt_br, cmn ...) -> its file
pub fn heads_file(lang: &str) -> &'static str {
    match lang {
        "en_us" | "en_gb" | "en" => "zonos3_mini_heads.json",
        "fr_fr" | "fr" => "zonos3_mini_heads_fr.json",
        "pt_br" | "pt" => "zonos3_mini_heads_pt.json",
        "cmn" | "zh" => "zonos3_mini_heads_zh.json",
        "no" | "nb" => "zonos3_mini_heads_no.json",
        "de" => "zonos3_mini_heads_de.json",
        "es" => "zonos3_mini_heads_es.json",
        "it" => "zonos3_mini_heads_it.json",
        "ja" => "zonos3_mini_heads_ja.json",
        "ko" => "zonos3_mini_heads_ko.json",
        "ar" => "zonos3_mini_heads_ar.json",
        "bn" => "zonos3_mini_heads_bn.json",
        "bg" => "zonos3_mini_heads_bg.json",
        "cs" => "zonos3_mini_heads_cs.json",
        "da" => "zonos3_mini_heads_da.json",
        "nl" => "zonos3_mini_heads_nl.json",
        "fi" => "zonos3_mini_heads_fi.json",
        "ka" => "zonos3_mini_heads_ka.json",
        "el" => "zonos3_mini_heads_el.json",
        "gu" => "zonos3_mini_heads_gu.json",
        "he" => "zonos3_mini_heads_he.json",
        "hi" => "zonos3_mini_heads_hi.json",
        "hu" => "zonos3_mini_heads_hu.json",
        "id" => "zonos3_mini_heads_id.json",
        "kn" => "zonos3_mini_heads_kn.json",
        "ms" => "zonos3_mini_heads_ms.json",
        "ml" => "zonos3_mini_heads_ml.json",
        "mr" => "zonos3_mini_heads_mr.json",
        "or" => "zonos3_mini_heads_or.json",
        "pl" => "zonos3_mini_heads_pl.json",
        "pa" => "zonos3_mini_heads_pa.json",
        "ro" => "zonos3_mini_heads_ro.json",
        "ru" => "zonos3_mini_heads_ru.json",
        "sk" => "zonos3_mini_heads_sk.json",
        "sv" => "zonos3_mini_heads_sv.json",
        "tl" => "zonos3_mini_heads_tl.json",
        "ta" => "zonos3_mini_heads_ta.json",
        "te" => "zonos3_mini_heads_te.json",
        "th" => "zonos3_mini_heads_th.json",
        "tr" => "zonos3_mini_heads_tr.json",
        "uk" => "zonos3_mini_heads_uk.json",
        "ur" => "zonos3_mini_heads_ur.json",
        "vi" => "zonos3_mini_heads_vi.json",
        _ => "",
    }
}

/// The server language (Zyphra's codes: the fillers' and heads' keys) of a normalizer language code
pub fn server_language(code: &str) -> String {
    match code {
        "en" => "en_us",
        "fr" => "fr_fr",
        "pt" => "pt_br",
        "zh" => "cmn",
        "nb" => "no",
        c => c,
    }
    .to_string()
}

// ------------------------------------------------------------------------------------------------------ short texts

pub struct ShortRegistry {
    pub pause_marker: char,
    pub max_words: usize,
    pub max_ratio: (usize, usize),
    pub lookback: usize,
    pub min_voiced: usize,
    pub max_frames: usize,
    pub max_retries: usize,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub threshold: i64,
    pub weights: Vec<i8>,
    pub fillers: Vec<(String, String)>,
}

fn b64(s: &str) -> Vec<u8> {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let (mut out, mut val, mut bits) = (Vec::new(), 0u32, 0);
    for c in s.bytes() {
        if c == b'=' {
            break;
        }
        let Some(d) = A.iter().position(|a| *a == c) else { continue };
        val = (val << 6) | d as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((val >> bits) & 0xFF) as u8);
        }
    }
    out
}

pub fn short_registry() -> &'static ShortRegistry {
    static R: OnceLock<ShortRegistry> = OnceLock::new();
    R.get_or_init(|| {
        let d: serde_json::Value = serde_json::from_str(include_str!("short_text.json")).expect("short_text.json");
        let (s, cut) = (&d["silence"], &d["cut"]);
        let u = |v: &serde_json::Value| v.as_u64().unwrap_or(0) as usize;
        ShortRegistry {
            pause_marker: d["pause_marker"].as_str().and_then(|m| m.chars().next()).unwrap_or('\u{E004}'),
            max_words: u(&d["max_words"]),
            max_ratio: (u(&d["max_ratio"][0]), u(&d["max_ratio"][1])),
            lookback: u(&cut["lookback"]),
            min_voiced: u(&cut["min_voiced"]),
            max_frames: u(&cut["max_frames"]),
            max_retries: u(&cut["max_retries"]),
            n_codebooks: u(&s["n_codebooks"]),
            codebook_size: u(&s["codebook_size"]),
            threshold: s["threshold"].as_i64().unwrap_or(0),
            weights: b64(s["weights"].as_str().unwrap_or("")).into_iter().map(|b| b as i8).collect(),
            fillers: d["fillers"].as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect(),
        }
    })
}

/// The cut of a short-text take over its generated frames (frame f = codebook k of generated row f + k)
pub struct ShortCut {
    filler_token: usize,
    ncb: usize,
    pub enter: Option<usize>,
    pub decided: Option<(&'static str, usize)>,
    rows: Vec<[u32; 10]>,
    silent: Vec<bool>,
}

impl ShortCut {
    pub fn new(filler_token: usize, ncb: usize) -> ShortCut {
        ShortCut { filler_token, ncb, enter: None, decided: None, rows: Vec::new(), silent: Vec::new() }
    }

    pub fn truncate(&mut self, n: usize) {
        self.rows.truncate(n);
        self.silent.truncate((n + 1).saturating_sub(self.ncb));
        if self.enter.is_some_and(|e| e >= n) {
            self.enter = None;
        }
        self.decided = None;
        self.decide();
    }

    pub fn observe_reading(&mut self, frame: usize, pos: usize) {
        if self.enter.is_none() && pos >= self.filler_token {
            self.enter = Some(frame);
            self.decide();
        }
    }

    pub fn observe_row(&mut self, row: [u32; 10]) {
        self.rows.push(row);
        if self.rows.len() >= self.ncb {
            let f = self.rows.len() - self.ncb;
            let s = self.is_silent(f);
            self.silent.push(s);
            self.decide();
        }
    }

    fn is_silent(&self, f: usize) -> bool {
        let r = short_registry();
        let mut score = 0i64;
        for k in 0..self.ncb {
            let c = self.rows[f + k][k] as usize;
            if c >= r.codebook_size {
                return false;
            }
            score += r.weights[k * r.codebook_size + c] as i64;
        }
        score > r.threshold
    }

    fn decide(&mut self) {
        let Some(enter) = self.enter else { return };
        if self.decided.is_some() {
            return;
        }
        let r = short_registry();
        let mut voiced = 0;
        for (f, silent) in self.silent.iter().enumerate() {
            voiced += !*silent as usize;
            if f + r.lookback < enter {
                continue;
            }
            if (f + 1).saturating_sub(enter) > r.max_frames {
                self.decided = Some(("run_on", enter));
                return;
            }
            if *silent && voiced >= r.min_voiced {
                self.decided = Some(("cut", f + 1));
                return;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------- session

/// A rolled-back take's regeneration: the stream rows [0, n_rows) kept, re-prefilled after the prompt
pub struct Continuation {
    pub n_rows: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct FailureRecord {
    pub kind: &'static str,
    pub detect_row: usize,
    pub rollback_row: usize,
    pub bin: usize,
    pub ignored: bool,
}

/// One take under LACI: its stream (the silence-prefill rows, then the generated ones), detector, retries and ACI
/// probation
pub struct Session {
    heads: Heads,
    cfg: Config,
    pub n_prefill: usize,
    stream: Vec<[u32; 10]>,
    pub attempt_start: usize,
    emitted: usize,
    detector: Option<Detector>,
    text_len: usize,
    pub attempts: usize,
    pub failures: Vec<FailureRecord>,
    probation: Option<(Failure, AciCenter)>,
    pending: Option<Failure>,
    text_done: bool,
    eos_row: Option<usize>,
    pub generated_total: usize,
    pub end: Option<(&'static str, Option<usize>)>,
    ignored: HashSet<(&'static str, usize, usize)>,
    seed: u64,
    max_tokens: usize,
    ncb: usize,
    hold: bool,
    short: Option<ShortCut>,
    short_retries: usize,
}

/// What the probe tells the take after a step
#[derive(Default)]
pub struct Control {
    /// the masked heads' windows for the following steps (layer, head, lo, hi); None: no mask
    pub mask: Option<Vec<(usize, usize, usize, usize)>>,
    /// end this attempt after the row sampled now
    pub stop: bool,
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub fn new(heads: Heads, prefill: &[[u32; 10]], seed: u64, max_tokens: usize, ncb: usize, hold: bool, short_text: Option<usize>) -> Session {
        Session { heads, cfg: Config::default(), n_prefill: prefill.len(), stream: prefill.to_vec(), attempt_start: prefill.len(), emitted: 0, detector: None,
                  text_len: 0, attempts: 0, failures: Vec::new(), probation: None, pending: None, text_done: false, eos_row: None, generated_total: 0, end: None,
                  ignored: HashSet::new(), seed, max_tokens, ncb, hold, short: short_text.map(|t| ShortCut::new(t, ncb)), short_retries: 0 }
    }

    pub fn stream(&self) -> &[[u32; 10]] {
        &self.stream
    }

    fn window(&self, center: usize) -> Vec<(usize, usize, usize, usize)> {
        self.heads.radius.iter().map(|((l, h), r)| (*l, *h, (center + 1).saturating_sub(*r), center + r - 1)).collect()
    }

    pub fn probe_mask(&self) -> Option<Vec<(usize, usize, usize, usize)>> {
        self.probation.as_ref().map(|(_, aci)| self.window(aci.center))
    }

    /// the logit added to the end-of-audio id of every codebook for the next row
    pub fn stop_boost(&self) -> f32 {
        match self.detector.as_ref().and_then(|d| d.done_row()) {
            Some(d) if self.stream.len() >= d + self.cfg.stop_boost_delay => self.cfg.stop_boost,
            _ => 0.0,
        }
    }

    /// after decode step `step` of this attempt (1 = the first): the reading `pos` and its centroid `m`
    pub fn on_step(&mut self, ctl: &mut Control, text_len: usize, step: usize, pos: usize, m: f64) -> Result<()> {
        let row = self.attempt_start + step;
        if self.detector.is_none() {
            self.text_len = text_len;
            self.detector = Some(Detector::new(text_len, self.heads.frames_per_token, &self.cfg, self.n_prefill));
        }
        if let Some((failure, aci)) = self.probation.as_mut() {
            let c = aci.update(m);
            let frontier = failure.frontier;
            ctl.mask = Some(self.window(c));
            if self.detector.as_ref().unwrap().frontier() > frontier {
                self.probation = None;
                ctl.mask = None;
            }
        }
        if let Some(f) = self.detector.as_mut().unwrap().observe(row, pos)? {
            if self.attempts < self.cfg.max_retries {
                self.pending = Some(f);
                ctl.stop = true;
                return Ok(());
            }
            self.ignore(&f);
        }
        if let Some(sc) = self.short.as_mut() {
            sc.observe_reading(row - self.n_prefill, pos);
            if sc.decided.is_some() {
                ctl.stop = true;
                return Ok(());
            }
        }
        if let Some(stop_row) = self.detector.as_ref().unwrap().stop_row() {
            if row >= stop_row + self.ncb + 2 {
                self.text_done = true;
                ctl.stop = true;
            }
        }
        Ok(())
    }

    fn ignore(&mut self, f: &Failure) {
        if self.ignored.insert((f.kind, f.bin, f.onset_row)) {
            self.failures.push(FailureRecord { kind: f.kind, detect_row: f.detect_row, rollback_row: f.rollback_row, bin: f.bin, ignored: true });
        }
    }

    /// a row sampled; `eos_row`: the attempt's end of audio as a stream row, once drawn
    pub fn append(&mut self, row: [u32; 10], eos_row: Option<usize>) {
        self.stream.push(row);
        self.generated_total += 1;
        if eos_row.is_some() {
            self.eos_row = eos_row;
        }
        if let Some(sc) = self.short.as_mut() {
            sc.observe_row(row);
        }
    }

    fn commit_row(&self) -> usize {
        let n = self.stream.len();
        if self.end.is_some() {
            return n;
        }
        if self.hold || self.short.is_some() {
            return self.n_prefill;
        }
        let row = if self.attempts >= self.cfg.max_retries {
            n
        } else {
            let eos = self.eos_row.unwrap_or((n + 1).saturating_sub(self.ncb));
            let r = eos.saturating_sub(self.cfg.rollback_margin_frames).max(self.n_prefill);
            match &self.detector {
                Some(d) => r.min(d.commit_row(n)),
                None => self.n_prefill,
            }
        };
        row.min(n.saturating_sub(1)).max(self.n_prefill)
    }

    /// generated rows that became final since the last drain: stream rows [start, end)
    pub fn drain(&mut self, last: bool) -> (usize, usize) {
        let upto = if last { self.stream.len() } else { self.commit_row() };
        let start = self.n_prefill + self.emitted;
        let end = start.max(upto.min(self.stream.len()));
        self.emitted += end - start;
        (start, end)
    }

    /// the attempt ended: a Continuation to regenerate from, or None when the take is final
    pub fn finish_attempt(&mut self) -> Option<Continuation> {
        let mut f = self.pending.take();
        if f.is_none() && self.short.is_none() {
            if let (Some(e), Some(d)) = (self.eos_row, self.detector.as_ref()) {
                f = d.on_eos(e);
                if let Some(x) = f.clone() {
                    if self.attempts >= self.cfg.max_retries {
                        self.ignore(&x);
                        f = None;
                    }
                }
            }
        }
        if f.is_none() && self.short.is_some() {
            f = self.short_run_on();
        }
        let mut resume = 0;
        if let Some(x) = &f {
            resume = x.rollback_row.max(self.n_prefill + self.emitted);
            if self.max_tokens <= resume - self.n_prefill {
                let x = x.clone();
                self.ignore(&x);
                f = None;
            }
        }
        let Some(f) = f else {
            self.end = Some(if self.short.as_ref().is_some_and(|s| s.enter.is_some()) {
                self.short_end()
            } else if self.text_done {
                ("text_done", self.detector.as_ref().and_then(|d| d.stop_row()))
            } else if self.eos_row.is_some() {
                ("eos", self.eos_row)
            } else {
                ("max_tokens", None)
            });
            return None;
        };
        self.attempts += 1;
        let seed = self.seed.wrapping_add(1000 * self.attempts as u64) & 0xFFFF_FFFF;
        self.failures.push(FailureRecord { kind: f.kind, detect_row: f.detect_row, rollback_row: resume, bin: f.bin, ignored: false });
        self.stream.truncate(resume);
        if let Some(d) = self.detector.as_mut() {
            d.truncate(resume);
        }
        let np = self.n_prefill;
        if let Some(sc) = self.short.as_mut() {
            sc.truncate(resume - np);
        }
        if f.kind == "short_run_on" {
            self.short_retries += 1;
            self.probation = None;
        } else {
            let c0 = self.detector.as_ref().map_or(0, |d| d.last_pos_before(resume));
            self.probation = Some((f, AciCenter::new(c0, self.text_len)));
        }
        self.text_done = false;
        self.eos_row = None;
        self.attempt_start = resume;
        Some(Continuation { n_rows: resume, seed })
    }

    fn short_run_on(&mut self) -> Option<Failure> {
        let sc = self.short.as_ref()?;
        let (kind, enter) = sc.decided?;
        if kind != "run_on" {
            return None;
        }
        let fr = self.detector.as_ref().map_or(0, |d| d.frontier());
        let f = Failure { kind: "short_run_on", detect_row: self.stream.len().saturating_sub(1), onset_row: self.n_prefill + enter, rollback_row: self.n_prefill,
                          bin: fr, frontier: fr };
        if self.short_retries < short_registry().max_retries && self.attempts < self.cfg.max_retries {
            return Some(f);
        }
        self.ignore(&f);
        None
    }

    fn short_end(&self) -> (&'static str, Option<usize>) {
        let sc = self.short.as_ref().unwrap();
        if let Some(("cut", frames)) = sc.decided {
            return ("short_text", Some(self.n_prefill + frames));
        }
        let mut frames = sc.enter.unwrap_or(0).min((self.stream.len() + 1).saturating_sub(self.n_prefill + self.ncb));
        if let Some(e) = self.eos_row {
            frames = frames.min(e.saturating_sub(self.n_prefill));
        }
        ("short_text_forced", Some(self.n_prefill + frames))
    }

    /// the final take's end as a frame of the generated rows
    pub fn eos_frame(&self) -> Option<usize> {
        self.end.and_then(|(_, e)| e).map(|e| e - self.n_prefill)
    }

    pub fn summary(&self) -> String {
        let fails: Vec<String> = self.failures.iter().map(|f| format!("{}{}@{}->{}", f.kind, if f.ignored { "(ignored)" } else { "" }, f.detect_row, f.rollback_row)).collect();
        format!("end={} attempts={} generated={} failures=[{}]", self.end.map_or("?", |e| e.0), self.attempts, self.generated_total, fails.join(", "))
    }
}

/// A short text's prompt: the text, the 0.5 s pause marker and the language's filler, and the text-span position (0 =
/// BOS) of the filler's first token; None when the text does not qualify. `ids(text)` gives a text's payload ids
/// (without BOS/EOS), `blank(id)` whether a token is all whitespace
pub fn short_prompt(text: &str, lang: &str, ids: &dyn Fn(&str) -> Result<Vec<u32>>, blank: &dyn Fn(u32) -> bool) -> Result<Option<(String, usize)>> {
    let r = short_registry();
    let Some((_, filler)) = r.fillers.iter().find(|(k, _)| k == lang) else { return Ok(None) };
    if text.chars().any(|c| ('\u{E000}'..='\u{E00C}').contains(&c)) {
        return Ok(None);
    }
    if text.split_whitespace().count() > r.max_words {
        return Ok(None);
    }
    let n = ids(text)?.len();
    if n == 0 || n * r.max_ratio.1 > ids(filler)?.len() * r.max_ratio.0 {
        return Ok(None);
    }
    let head = format!("{text}{}", r.pause_marker);
    let full = format!("{head}{filler}");
    let all = ids(&full)?;
    let mut i = ids(&head)?.len();
    while i < all.len() && blank(all[i]) {
        i += 1;
    }
    if i >= all.len() {
        return Err(Error("short text: the filler has no token".into()));
    }
    Ok(Some((full, i + 1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_steady_reading_finishes_and_a_jump_is_a_skip() {
        let cfg = Config::default();
        let mut d = Detector::new(40, 3.5, &cfg, 3);
        let mut row = 3;
        for pos in 0..40 {
            for _ in 0..3 {
                row += 1;
                assert_eq!(d.observe(row, pos).unwrap(), None);
            }
        }
        assert!(d.done_row().is_some());
        let mut d = Detector::new(40, 3.5, &cfg, 3);
        let mut row = 3;
        let mut fail = None;
        for pos in (0..8).chain(30..40) {
            for _ in 0..4 {
                row += 1;
                if let Some(f) = d.observe(row, pos).unwrap() {
                    fail = Some(f);
                    break;
                }
            }
            if fail.is_some() {
                break;
            }
        }
        assert_eq!(fail.map(|f| f.kind), Some("skip"));
    }

    #[test]
    fn a_stall_is_a_hallucination() {
        let mut d = Detector::new(40, 3.5, &Config::default(), 0);
        let mut f = None;
        for row in 1..200 {
            if let Some(x) = d.observe(row, 2).unwrap() {
                f = Some(x);
                break;
            }
        }
        assert_eq!(f.map(|f| f.kind), Some("hallucination"));
    }

    #[test]
    fn the_short_text_registry_loads() {
        let r = short_registry();
        assert_eq!(r.weights.len(), r.n_codebooks * r.codebook_size);
        assert!(r.fillers.iter().any(|(k, _)| k == "en_us"));
        assert_eq!(r.pause_marker, '\u{E004}');
    }

    #[test]
    fn aci_moves_one_step_at_a_time() {
        let mut a = AciCenter::new(3, 20);
        assert_eq!(a.update(10.0), 4);
        assert_eq!(a.update(10.0), 5);
        assert_eq!(a.update(1.0), 5);
    }
}
