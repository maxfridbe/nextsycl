//! The engine against llama.cpp on the same GGUFs (reference/llama-dump/ns-ocr-dump.cpp, run as ~/src/olm-dump.sh:
//! olmOCR's prompt in Qwen's chat template): our preprocessing of the same picture against llama.cpp's `inp_raw`
//! (when given the picture), the tower fed llama.cpp's own pixels block by block (both in window order), the prompt's
//! tokens, every decoder layer over the prompt (llama.cpp passes it in three chunks: the text, the image, the text
//! after; its last layer keeps the last row only), and greedy tokens against `gen.txt`.

use std::collections::BTreeMap;
use std::path::Path;

use nextsycl_image::{Error, Picture, Result};

use crate::{positions, prep, OlmOcr, DEFAULT_PROMPT, IMAGE_TOKEN};

struct Dump {
    dir: std::path::PathBuf,
    shapes: BTreeMap<String, [usize; 4]>,
}

impl Dump {
    fn open(dir: &Path) -> Result<Dump> {
        let idx = std::fs::read_to_string(dir.join("index.tsv")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
        let mut shapes = BTreeMap::new();
        for l in idx.lines() {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() >= 6 {
                shapes.insert(f[0].to_string(), [f[2], f[3], f[4], f[5]].map(|v| v.parse().unwrap_or(1)));
            }
        }
        Ok(Dump { dir: dir.to_path_buf(), shapes })
    }
    fn get(&self, name: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(self.dir.join(format!("{name}.f32"))).ok()?;
        Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// max |a - b| / max |b|, and the cosine
pub fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let (mut md, mut mr, mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        md = md.max((x - y).abs());
        mr = mr.max(y.abs());
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    (md / mr.max(1e-30), ab / (aa.sqrt() * bb.sqrt()).max(1e-30))
}

pub fn run(m: &OlmOcr, dir: &Path, picture: Option<&Picture>, gen_tokens: usize, log: &mut dyn FnMut(String)) -> Result<f64> {
    let d = Dump::open(dir)?;
    let mut worst = 0f64;
    let ne = *d.shapes.get("inp_raw").ok_or_else(|| Error("no inp_raw".into()))?;
    let (w, h) = (ne[0], ne[1]);
    let px = d.get("inp_raw").ok_or_else(|| Error("no inp_raw.f32".into()))?;
    if let Some(p) = picture {
        let (ours, oh, ow) = prep::planes(&prep::fit(p), m.vision.mean, m.vision.std);
        if (oh, ow) != (h, w) {
            log(format!("prep: ours {ow} x {oh}, llama.cpp's {w} x {h}"));
        } else {
            let md = ours.iter().zip(&px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let off = ours.iter().zip(&px).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
            log(format!("prep: {w} x {h}, max |d| {md:.4} ({off} of {} values off by more than 1e-3)", px.len()));
        }
    }
    let seen = prep::see_planes(&px, h, w);
    let img = m.vision.see(&m.nsd, &seen, &mut |name, buf| {
        let Some(r) = d.get(name) else { return Ok(()) };
        let (rel, cos) = compare(&buf.to_f32()?, &r);
        log(format!("{name:>14}: rel {rel:.2e} cos {cos:.6}"));
        if name == "layer_out-31" {
            worst = worst.max(rel);
        }
        Ok(())
    })?;
    let (uh, uw) = (seen.gh / 2, seen.gw / 2);
    // the prompt: llama.cpp's chunks
    let tt = std::fs::read_to_string(dir.join("tokens.txt")).map_err(|e| Error(format!("tokens.txt: {e}")))?;
    let mut want = Vec::new();
    let mut chunks = Vec::new();
    for l in tt.lines() {
        let mut f = l.split_whitespace();
        match f.next() {
            Some("text") => {
                let ids: Vec<u32> = f.filter_map(|v| v.parse().ok()).collect();
                chunks.push(ids.len());
                want.extend(ids);
            }
            Some("image") => {
                let n: usize = f.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                chunks.push(n);
                want.extend(std::iter::repeat_n(IMAGE_TOKEN, n));
            }
            _ => {}
        }
    }
    let ids = m.prompt_tokens(DEFAULT_PROMPT, uh * uw);
    if ids == want {
        log(format!("tokens: llama.cpp's {} ({} image)", ids.len(), uh * uw));
    } else {
        let at = ids.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(ids.len().min(want.len()));
        log(format!("tokens differ at {at} (ours {}, llama.cpp {}): ours {:?} llama.cpp {:?}", ids.len(), want.len(), &ids[at..(at + 6).min(ids.len())],
                    &want[at..(at + 6).min(want.len())]));
    }
    let pos = positions(&want, uh, uw);
    let x = m.embed(&want, &img)?;
    m.nsd.wait()?;
    let mut cache = m.decoder.cache(want.len() + gen_tokens + 1)?;
    let e = m.decoder.g.n_embd;
    let layers = m.decoder.g.layers;
    let mut last = 0f64;
    let mut logits = m.decoder.pass(&m.nsd, &mut cache, &x, &pos, &mut |name, buf| {
        let ours = buf.to_f32()?;
        // llama.cpp's copies of the stage, one a chunk, in order
        let mut r = Vec::new();
        for (i, _) in chunks.iter().enumerate() {
            let n = if i == 0 { name.to_string() } else { format!("{name}#{}", i + 1) };
            if let Some(v) = d.get(&n) {
                r.extend(v);
            }
        }
        if r.is_empty() {
            return Ok(());
        }
        // the last layer (and the final norm): the last row only
        let ours = if r.len() < ours.len() { ours[ours.len() - r.len()..].to_vec() } else { ours };
        let (rel, cos) = compare(&ours, &r);
        log(format!("{name:>14}: rel {rel:.2e} cos {cos:.6} ({} rows)", r.len() / e));
        if name == format!("l_out-{}", layers - 2) {
            last = rel;
        }
        Ok(())
    })?;
    worst = worst.max(last);
    let gen: Vec<u32> = std::fs::read_to_string(dir.join("gen.txt")).unwrap_or_default().split_whitespace().filter_map(|v| v.parse().ok()).collect();
    let mut ours = Vec::new();
    let first = pos.last().map_or(0, |p| p[0].max(p[1]).max(p[2]) + 1);
    for next_pos in first..first + gen_tokens.min(gen.len()) as u32 {
        let t = logits.iter().enumerate().fold(0, |b, (i, x)| if *x > logits[b] { i } else { b }) as u32;
        ours.push(t);
        if t == m.eos {
            break;
        }
        logits = m.decoder.step(&m.nsd, &mut cache, t, next_pos)?;
    }
    let same = ours.iter().zip(&gen).take_while(|(a, b)| a == b).count();
    log(format!("greedy: {same} of {} tokens llama.cpp's", ours.len()));
    log(format!("ours:     {:?}", m.tok.decode(&ours)));
    log(format!("llama.cpp: {:?}", m.tok.decode(&gen[..gen.len().min(ours.len())])));
    Ok(worst)
}
