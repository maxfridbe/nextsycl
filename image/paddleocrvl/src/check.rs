//! The engine against the transformers reference (the original BF16 weights in float32 on the CPU; dump.py in the
//! reference directory writes `<dir>`: pixel_values, grid.txt, tokens.txt, positions.txt, every stage, gen.txt): our
//! preprocessing of the same picture against `pixel_values` (when given the picture), the tower fed the dump's own
//! pixels stage by stage, the prompt's tokens and positions, every decoder layer over the prompt, the last logits,
//! and greedy tokens against `gen.txt`.

use std::collections::BTreeMap;
use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_image::{Error, Picture, Result};

use crate::{positions, prep, PaddleOcrVl, DEFAULT_PROMPT};

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
    fn text(&self, name: &str) -> Result<String> {
        std::fs::read_to_string(self.dir.join(name)).map_err(|e| Error(format!("{name}: {e}")))
    }
    fn nums(&self, name: &str) -> Result<Vec<u32>> {
        Ok(self.text(name)?.split_whitespace().filter_map(|v| v.parse().ok()).collect())
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

/// Runs the check; the worst relative error of the image tokens and the decoder's last layer
pub fn run(m: &PaddleOcrVl, dir: &Path, picture: Option<&Picture>, gen_tokens: usize, log: &mut dyn FnMut(String)) -> Result<f64> {
    let d = Dump::open(dir)?;
    let mut worst = 0f64;
    let grid = d.nums("grid.txt")?;
    let (gh, gw) = (grid[1] as usize, grid[2] as usize);
    let px = d.get("pixel_values").ok_or_else(|| Error("no pixel_values.f32".into()))?;
    if let Some(p) = picture {
        let s = prep::see(p, prep::MAX_PIXELS);
        if (s.gh, s.gw) != (gh, gw) {
            log(format!("prep: our grid {} x {}, the reference's {gh} x {gw}", s.gh, s.gw));
        } else {
            let md = s.patches.iter().zip(&px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let off = s.patches.iter().zip(&px).filter(|(a, b)| (*a - *b).abs() > 1e-4).count();
            log(format!("prep: grid {gh} x {gw}, max |d| {md:.5} ({off} of {} values off by more than 1e-4; 2/255 = 0.0078 is one level)", px.len()));
        }
    }
    // the tower on the dump's own pixels
    let mut tap = |name: &str, buf: &DevBuf| -> nextsycl_core::Result<()> {
        let Some(r) = d.get(name) else { return Ok(()) };
        let ours = buf.to_f32()?;
        let (rel, cos) = compare(&ours, &r);
        if ours.len() != r.len() {
            log(format!("{name}: {} values, the reference {}", ours.len(), r.len()));
        }
        log(format!("{name:>12}: rel {rel:.2e} cos {cos:.6}"));
        if name == "projected" {
            worst = worst.max(rel);
        }
        Ok(())
    };
    let img = m.vision.see(&m.nsd, &px, gh, gw, &mut tap)?;
    let (h2, w2) = (gh / 2, gw / 2);
    // the prompt
    let want = d.nums("tokens.txt")?;
    let ids = m.prompt_tokens(DEFAULT_PROMPT, h2 * w2);
    if ids != want {
        let at = ids.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(ids.len().min(want.len()));
        log(format!("tokens differ at {at}: ours {:?}, the reference {:?}", &ids[at.saturating_sub(2)..(at + 6).min(ids.len())],
                    &want[at.saturating_sub(2)..(at + 6).min(want.len())]));
    } else {
        log(format!("tokens: the reference's {} (the prompt {DEFAULT_PROMPT:?})", ids.len()));
    }
    let pos = positions(&want, h2, w2);
    let pt = d.text("positions.txt")?;
    let rows: Vec<Vec<u32>> = pt.lines().map(|l| l.split_whitespace().filter_map(|v| v.parse().ok()).collect()).collect();
    let pos_ok = rows.len() == 3 && (0..pos.len()).all(|i| (0..3).all(|a| rows[a].get(i) == Some(&pos[i][a])));
    log(format!("positions: {}", if pos_ok { "the reference's" } else { "DIFFER" }));
    // the decoder over the reference's prompt
    let x = m.embed(&want, &img)?;
    m.nsd.wait()?;
    let mut cache = m.decoder.cache(want.len() + gen_tokens + 1)?;
    let layers = m.decoder.g.layers;
    let mut last = 0f64;
    let mut logits = m.decoder.pass(&m.nsd, &mut cache, &x, &pos, &mut |name, buf| {
        let Some(r) = d.get(name) else { return Ok(()) };
        let ours = buf.to_f32()?;
        let (rel, cos) = compare(&ours, &r);
        log(format!("{name:>12}: rel {rel:.2e} cos {cos:.6}"));
        if name == format!("l_out-{}", layers - 1) {
            last = rel;
        }
        Ok(())
    })?;
    worst = worst.max(last);
    if let Some(r) = d.get("logits_last") {
        let (rel, cos) = compare(&logits, &r);
        let am = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
        log(format!(" logits_last: rel {rel:.2e} cos {cos:.6}, argmax ours {} the reference's {}", am(&logits), am(&r)));
    }
    // greedy tokens
    let gen = d.nums("gen.txt")?;
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
    log(format!("greedy: {same} of {} tokens the reference's", ours.len()));
    log(format!("ours:      {:?}", m.tok.decode(&ours)));
    log(format!("reference: {:?}", m.tok.decode(&gen[..gen.len().min(ours.len())])));
    if let Some(sh) = d.shapes.get("projected") {
        log(format!("(projected {sh:?})"));
    }
    Ok(worst)
}
