//! The engine against llama.cpp's dumps (reference/llama-dump/ns-ocr-dump.cpp, on the same GGUFs): the vision tower
//! fed llama.cpp's own preprocessed pixels (`inp_raw`, so the stages compare whatever the resampling), every SAM and
//! CLIP layer, SAM's output, the image tokens; then the decoder over the dump's prompt (`tokens.txt`: our image tokens
//! in place of the image), every layer's rows against llama.cpp's passes (its BOS alone, the image, the text), and
//! greedy tokens against `gen.txt` (no repetition rule; llama.cpp attends to everything written, the reference's ring
//! only differs past 128 tokens).

use std::collections::BTreeMap;
use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_image::{Error, Result};

use crate::{prep, UnlimitedOcr, IMAGE_TOKEN};

struct Dump {
    dir: std::path::PathBuf,
    /// name -> ne (ne0 innermost)
    shapes: BTreeMap<String, [usize; 4]>,
    order: Vec<String>,
}

impl Dump {
    fn open(dir: &Path) -> Result<Dump> {
        let idx = std::fs::read_to_string(dir.join("index.tsv")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
        let mut shapes = BTreeMap::new();
        let mut order = Vec::new();
        for l in idx.lines() {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() < 6 {
                continue;
            }
            let ne = [f[2], f[3], f[4], f[5]].map(|v| v.parse().unwrap_or(1));
            shapes.insert(f[0].to_string(), ne);
            order.push(f[0].to_string());
        }
        Ok(Dump { dir: dir.to_path_buf(), shapes, order })
    }
    fn get(&self, name: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(self.dir.join(format!("{name}.f32"))).ok()?;
        Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    /// the names of one stage's copies in the order computed (name, name#2, ...)
    fn copies(&self, base: &str) -> Vec<String> {
        self.order.iter().filter(|n| *n == base || n.strip_prefix(base).is_some_and(|r| r.starts_with('#'))).cloned().collect()
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

/// llama.cpp's [C][H][W] (a view's SAM output) as ours [H][W][C]
fn chw_to_hwc(v: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                o[(y * w + x) * c + ch] = v[(ch * h + y) * w + x];
            }
        }
    }
    o
}

/// Runs the check; the worst relative error of the image tokens and the decoder's last layer
pub fn run(m: &UnlimitedOcr, dir: &Path, gen_tokens: usize, log: &mut dyn FnMut(String)) -> Result<f64> {
    let d = Dump::open(dir)?;
    let mut worst = 0f64;
    // the vision graphs, in the order llama.cpp ran them: each inp_raw copy [B][3][H][W]
    let raws = d.copies("inp_raw");
    for (k, raw) in raws.iter().enumerate() {
        let ne = d.shapes[raw];
        let (w, h, b) = (ne[0], ne[1], ne[3]);
        let px = d.get(raw).ok_or_else(|| Error(format!("no {raw}.f32")))?;
        // a fused row of tiles is one [3][tiles x 640][640] picture: cut back into tiles
        let (side, nb, planes) = if h == w {
            (w, b, px)
        } else {
            let t = h / w;
            let mut out = Vec::with_capacity(px.len());
            for bi in 0..b {
                for ti in 0..t {
                    for c in 0..3 {
                        let base = ((bi * 3 + c) * h + ti * w) * w;
                        out.extend_from_slice(&px[base..base + w * w]);
                    }
                }
            }
            (w, b * t, out)
        };
        // this graph's copy of a stage: among the copies of its name of this size, the one of the graphs of this shape so far
        let same_before = raws[..k].iter().filter(|r| d.shapes[*r] == ne).count();
        let pick_copy = |name: &str, len: usize| -> Option<Vec<f32>> {
            let c: Vec<String> = d.copies(name).into_iter().filter(|n| d.shapes[n].iter().product::<usize>() == len).collect();
            c.get(same_before).and_then(|n| d.get(n))
        };
        let mut report = Vec::new();
        let mut tap = |name: &str, buf: &DevBuf| -> nextsycl_core::Result<()> {
            let ours = buf.to_f32()?;
            let rname = name.strip_prefix("clip_").unwrap_or(name);
            if name == "sam_output" {
                if let Some(r) = pick_copy("sam_output", ours.len()) {
                    let g = side / 64;
                    let per = 1024 * g * g;
                    let mut ours_t = Vec::new();
                    for bi in 0..r.len() / per {
                        ours_t.extend(chw_to_hwc(&r[bi * per..(bi + 1) * per], 1024, g, g));
                    }
                    let (rel, cos) = compare(&ours, &ours_t);
                    report.push(format!("{name}: rel {rel:.2e} cos {cos:.6}"));
                }
            } else if let Some(r) = pick_copy(rname, ours.len()) {
                let (rel, cos) = compare(&ours, &r);
                report.push(format!("{name}: rel {rel:.2e} cos {cos:.6}"));
            }
            Ok(())
        };
        let _ = m.vision.see(&m.nsd, &planes, nb, side, &mut tap)?;
        for r in &report {
            log(format!("vision {k} ({nb} x {side}): {r}"));
        }
    }
    // the image tokens: ours (from our own preprocessing of the same views is not possible here - the dump has the
    // pixels, not the picture), so the tokens are rebuilt from llama.cpp's pixels in its order
    let outs = d.copies("dsocr_output");
    let mut reference: Vec<f32> = Vec::new();
    for o in &outs {
        reference.extend(d.get(o).unwrap_or_default());
    }
    let tokens_ours = {
        let seen = seen_from_raws(&d, &raws)?;
        let (img, n) = m.image_tokens(&seen, &mut |_, _| Ok(()))?;
        let t = img.to_f32()?;
        let (rel, cos) = compare(&t, &reference);
        log(format!("image tokens ({n}, llama.cpp {}): rel {rel:.2e} cos {cos:.6}", reference.len() / crate::vision::OUT));
        worst = worst.max(rel);
        t
    };
    // the decoder over the dump's prompt
    let tk = std::fs::read_to_string(dir.join("tokens.txt")).map_err(|e| Error(e.to_string()))?;
    let mut ids = Vec::new();
    let mut runs = Vec::new();
    for l in tk.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        match f.first() {
            Some(&"text") => {
                let t: Vec<u32> = f[1..].iter().filter_map(|v| v.parse().ok()).collect();
                runs.push(t.len());
                ids.extend(t);
            }
            Some(&"image") => {
                let n: usize = f.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
                runs.push(n);
                ids.extend(std::iter::repeat_n(IMAGE_TOKEN, n));
            }
            _ => {}
        }
    }
    let img = DevBuf::from_f32(&m.nsd.gpu, &tokens_ours)?;
    let x = m.embed(&ids, &img)?;
    let mut cache = m.decoder.cache(ids.len())?;
    let e = crate::vision::OUT;
    let mut tap = |name: &str, buf: &DevBuf| -> nextsycl_core::Result<()> {
        let ours = buf.to_f32()?;
        let width = ours.len() / ids.len().max(1);
        // llama.cpp's passes: one a run of the prompt (skipping empty ones); the copies of the stage's name with this
        // width, in order. qkv: llama.cpp's q, k and v apart.
        let parts: Vec<(&str, usize, usize)> = if let Some(l) = name.strip_prefix("qkv-") {
            let _ = l;
            vec![("q", 0, e), ("k", e, e), ("v", 2 * e, e)]
        } else {
            vec![("", 0, width)]
        };
        for (part, off, w) in parts {
            let rname = if part.is_empty() { name.to_string() } else { format!("{part}-{}", &name[4..]) };
            let copies: Vec<String> = d.copies(&rname).into_iter().filter(|n| d.shapes[n][0] * d.shapes[n][1] == w || d.shapes[n][0] == w).collect();
            let mut row = 0;
            let mut k = 0;
            let mut line = format!("decoder {rname}:");
            for &n in &runs {
                if n == 0 {
                    continue;
                }
                if let Some(r) = copies.get(k).and_then(|c| d.get(c)) {
                    // llama.cpp keeps the last layer's output for the last row only
                    let rows = r.len() / w;
                    if rows > 0 && rows <= n {
                        let at = row + n - rows;
                        let ours_rows: Vec<f32> = (at..at + rows).flat_map(|i| ours[i * width + off..i * width + off + w].iter().copied()).collect();
                        let (rel, cos) = compare(&ours_rows, &r);
                        line += &format!(" [{n} rows: rel {rel:.2e} cos {cos:.6}]");
                    }
                }
                k += 1;
                row += n;
            }
            eprintln!("{line}");
        }
        Ok(())
    };
    let logits = m.decoder.pass(&m.nsd, &mut cache, &x, ids.len(), &mut tap)?;
    let gen: Vec<u32> = std::fs::read_to_string(dir.join("gen.txt")).unwrap_or_default().lines().filter_map(|v| v.trim().parse().ok()).collect();
    let mut ours = Vec::new();
    let mut lg = logits;
    for _ in 0..gen_tokens.min(gen.len().max(1)) {
        let next = crate::pick(&mut lg, &[], 0.0, &mut crate::Rng(0));
        ours.push(next);
        if next == 1 || ours.len() >= gen.len() {
            break;
        }
        cache.written += 1;
        let xe = m.decoder.embed(&m.nsd, &[next])?;
        lg = m.decoder.pass(&m.nsd, &mut cache, &xe, 1, &mut |_, _| Ok(()))?;
    }
    let same = ours.iter().zip(&gen).take_while(|(a, b)| a == b).count();
    log(format!("greedy tokens: {same} of {} the same as llama.cpp's (ours {:?}...)", ours.len().min(gen.len()), &ours[..ours.len().min(8)]));
    if same < ours.len().min(gen.len()) {
        log(format!("  first difference at {same}: ours {} llama.cpp {}", ours[same], gen[same]));
    }
    let text = m.tok.decode(&ours);
    log(format!("text: {}", text.chars().take(400).collect::<String>()));
    Ok(worst)
}

/// The views llama.cpp encoded, as ours to weave: its rows of tiles (in order) and its overview(s)
fn seen_from_raws(d: &Dump, raws: &[String]) -> Result<prep::Seen> {
    let mut tiles = Vec::new();
    let mut globals = Vec::new();
    let mut rows = 0;
    let mut per_row = 0;
    for raw in raws {
        let ne = d.shapes[raw];
        let (w, h, b) = (ne[0], ne[1], ne[3]);
        let px = d.get(raw).ok_or_else(|| Error(format!("no {raw}.f32")))?;
        if w == h && w == prep::BASE as usize {
            for bi in 0..b {
                globals.push(prep::View { side: w as u32, planes: px[bi * 3 * w * h..(bi + 1) * 3 * w * h].to_vec() });
            }
        } else if w == h {
            // a row of tiles as a batch (llama.cpp's graph a row)
            per_row = b;
            rows += 1;
            for bi in 0..b {
                tiles.push(prep::View { side: w as u32, planes: px[bi * 3 * w * w..(bi + 1) * 3 * w * w].to_vec() });
            }
        } else {
            // a row of tiles fused into one tall picture
            let t = h / w;
            per_row = t;
            for bi in 0..b {
                rows += 1;
                for ti in 0..t {
                    let mut planes = Vec::with_capacity(3 * w * w);
                    for c in 0..3 {
                        let base = ((bi * 3 + c) * h + ti * w) * w;
                        planes.extend_from_slice(&px[base..base + w * w]);
                    }
                    tiles.push(prep::View { side: w as u32, planes });
                }
            }
        }
    }
    Ok(prep::Seen { tiles, grid: (per_row.max(1) as u32, rows.max(1) as u32), globals })
}
