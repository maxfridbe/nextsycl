//! SheetSage2 against reference/yue2/sheetsage_ref.py's dumps (upstream's code as it is, bf16 autocast on a card):
//! each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   wave        DIR/recording.* decoded and made 24 kHz mono here            -> wave (cosine, the largest difference)
//!   mel         the reference's wave, its first window                       -> mel
//!   encoder     the reference's mel                                          -> sub, block0/1/11/23, mixed, memory (and memory_fp32)
//!   decoder     the reference's memory, tokens0 teacher-forced in one block  -> tf_logits at tf_rows (cosine, argmax),
//!               the grammar's masked argmax at every step                    -> tf_argmax, and tokens0 itself
//!   generate    the reference's memory, the greedy decode                    -> tokens0 (identical up to where, how many agree)
//!   notation    the reference's tokens of every window, stitched and written -> abc_full.abc, abc_melody.abc (byte for byte)
//!   transcribe  the reference's wave end to end                              -> the same, and the tokens
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::check::{compare, read_npy};
use crate::lm::Scratch;
use crate::sheet::tokens::{self, Grammar, OUT, VOCAB};
use crate::sheet::{notation, stitch, wave, SheetSage, WINDOW};

pub const STAGES: &[&str] = &["wave", "mel", "encoder", "decoder", "generate", "notation", "transcribe"];

fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    let (rel, mx, cos) = compare(got, want);
    let mismatch = if got.len() != want.len() { format!("  LENGTHS {} vs {}", got.len(), want.len()) } else { String::new() };
    log(format!("{name:<28} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values){mismatch}", want.len().min(got.len())));
    if got.len() != want.len() { 0.0 } else { cos }
}

/// The lines where two texts first differ (none: identical)
fn first_difference(a: &str, b: &str) -> Option<String> {
    let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    for i in 0..la.len().max(lb.len()) {
        if la.get(i) != lb.get(i) {
            return Some(format!("line {}: ours {:?}, theirs {:?}", i + 1, la.get(i).unwrap_or(&"<end>"), lb.get(i).unwrap_or(&"<end>")));
        }
    }
    None
}

fn abc_report(name: &str, ours: &notation::Sheet, dir: &Path, file: &str, log: &mut dyn FnMut(String)) -> bool {
    let want = std::fs::read_to_string(dir.join(file)).unwrap_or_default();
    let got = ours.abc.clone().unwrap_or_default();
    match (first_difference(&got, &want), &ours.error) {
        (None, _) => {
            log(format!("{name}: identical to {file} ({} lines, {} measures)", want.lines().count(), ours.measures));
            true
        }
        (Some(d), e) => {
            log(format!("{name}: DIFFERS from {file}: {d}{}", e.as_ref().map(|e| format!(" (ours failed: {e})")).unwrap_or_default()));
            false
        }
    }
}

/// The tokens' agreement: (the first index they differ at, how many of the shorter agree)
fn agreement(a: &[u32], b: &[u32]) -> (Option<usize>, usize) {
    let first = a.iter().zip(b).position(|(x, y)| x != y).or(if a.len() != b.len() { Some(a.len().min(b.len())) } else { None });
    (first, a.iter().zip(b).filter(|(x, y)| x == y).count())
}

/// The longest common subsequence of two token sequences, as a share of the longer (how much of one is the other's,
/// an insertion or a deletion not counted against what follows it)
fn lcs_share(a: &[u32], b: &[u32]) -> f64 {
    let mut prev = vec![0u32; b.len() + 1];
    let mut cur = vec![0u32; b.len() + 1];
    for x in a {
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as f64 / a.len().max(b.len()).max(1) as f64
}

/// The melody notes of stitched events: (onset seconds, pitch, track)
fn notes_of(events: &[tokens::Event]) -> Vec<(f64, u32, u32)> {
    events.iter().flat_map(|e| e.notes.iter().map(|n| (e.time.unwrap_or(0.0), n.pitch, n.track))).collect()
}

/// Note agreement as transcription is scored (mir_eval's onset rule): a note of ours matching one of theirs of the
/// same pitch with an onset within 50 ms, each used once: (matched, ours, theirs)
fn note_match(ours: &[(f64, u32, u32)], theirs: &[(f64, u32, u32)], same_track: bool) -> (usize, usize, usize) {
    let mut used = vec![false; theirs.len()];
    let mut m = 0;
    for o in ours {
        if let Some(i) = (0..theirs.len()).find(|i| !used[*i] && theirs[*i].1 == o.1 && (!same_track || theirs[*i].2 == o.2) && (theirs[*i].0 - o.0).abs() <= 0.05) {
            used[i] = true;
            m += 1;
        }
    }
    (m, ours.len(), theirs.len())
}

/// Runs the stages; the lowest cosine seen (0 when a sequence or a text differs)
pub fn run(e: &SheetSage, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, v.into_iter().map(|x| x as f32).collect())) };
    let ids = |name: &str| -> Result<Vec<u32>> { Ok(read_npy(&dir.join(format!("{name}.npy")))?.1.iter().map(|x| *x as u32).collect()) };
    let mut worst = 1f64;
    let mut s = Scratch::default();
    let g = &e.gpu;
    if stages.contains(&"wave") {
        let rec = std::fs::read_dir(dir).map_err(|er| Error(format!("{}: {er}", dir.display())))?.flatten().map(|d| d.path())
            .find(|p| p.file_stem().is_some_and(|n| n == "recording"));
        match rec {
            Some(p) => {
                let a = nextsycl_audio::read_audio(&std::fs::read(&p).map_err(|er| Error(format!("{}: {er}", p.display())))?).map_err(Error)?;
                let t0 = Instant::now();
                let ours = wave::prepare(&a.samples, a.channels as usize, a.rate);
                let (_, want) = get("wave")?;
                log(format!("wave: {} at {} Hz, {} channels -> {} samples at 24 kHz in {:.2} s (the reference's {})", p.display(), a.rate, a.channels, ours.len(),
                            t0.elapsed().as_secs_f64(), want.len()));
                let n = ours.len().min(want.len());
                worst = worst.min(note("wave (24 kHz mono)", &ours[..n], &want[..n], log));
            }
            None => log(format!("wave: no {}/recording.* (a link to the recording the reference transcribed)", dir.display())),
        }
    }
    let (_, ref_wave) = get("wave")?;
    if stages.contains(&"mel") {
        let t0 = Instant::now();
        let ours = e.window_features(&ref_wave, 0.0);
        log(format!("mel: {} frames in {:.2} s", ours.len() / 128, t0.elapsed().as_secs_f64()));
        worst = worst.min(note("mel (first window)", &ours, &get("mel")?.1, log));
    }
    if stages.contains(&"encoder") {
        let (_, mel) = get("mel")?;
        let t0 = Instant::now();
        let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
        let mem = e.mert.encode(&mut s, &mel, &mut |name, b, _| {
            if ["sub", "block0", "block1", "block11", "block22", "block23", "mixed"].contains(&name) {
                taps.push((name.to_string(), b.to_f32()?));
            }
            Ok(())
        })?;
        let m = mem.to_f32()?;
        log(format!("encoder: 7500 frames in {:.2} s (with the taps' copies)", t0.elapsed().as_secs_f64()));
        for (name, v) in &taps {
            if let Ok((_, w)) = get(name) {
                worst = worst.min(note(name, v, &w, log));
            }
            // and against the reference in float32 throughout, when dumped (the bf16 run's own error is as large)
            if let Ok((_, f)) = get(&format!("fp32_{name}")) {
                note(&format!("{name} vs float32"), v, &f, log);
            }
        }
        worst = worst.min(note("memory", &m, &get("memory")?.1, log));
        let (_, fp32) = get("memory_fp32")?;
        note("memory vs the fp32 reference", &m, &fp32, log);
        note("(the bf16 reference vs its fp32)", &get("memory")?.1, &fp32, log);
    }
    let toks0 = ids("tokens0")?;
    let prompts = SheetSage::prompts();
    let prefix = tokens::prompt_prefix(&prompts);
    let ref_memory = || -> Result<DevBuf> { DevBuf::from_f32(g, &get("memory")?.1) };
    if stages.contains(&"decoder") {
        let mem = ref_memory()?;
        let rows: Vec<usize> = ids("tf_rows")?.into_iter().map(|r| r as usize).collect();
        let n = toks0.len() - 1;
        let t0 = Instant::now();
        let mut st = e.dec.start(&mut s, &mem, n)?;
        let all: Vec<usize> = (prefix.len() - 1..n).collect();
        let logits = e.dec.forward(&mut s, &mut st, &toks0[..n], &all)?;
        log(format!("decoder: {n} tokens teacher-forced in {:.2} s", t0.elapsed().as_secs_f64()));
        let (_, want) = get("tf_logits")?;
        let row = |r: usize| &logits[(r - (prefix.len() - 1)) * VOCAB..(r - (prefix.len() - 1) + 1) * VOCAB];
        let mut sel = Vec::new();
        for r in &rows {
            sel.extend_from_slice(row(*r));
        }
        worst = worst.min(note(&format!("logits at {} rows", rows.len()), &sel, &want, log));
        // the masked argmax at every step: against the reference's own teacher-forced pick, and the tokens generated
        let mut gr = Grammar::default();
        let out_at = prefix.iter().position(|t| *t == OUT).unwrap_or(0);
        for t in &prefix[out_at + 1..] {
            gr.update(*t).map_err(Error)?;
        }
        let tf = ids("tf_argmax")?;
        let (mut same_tf, mut same_gen, mut shown) = (0usize, 0usize, 0);
        for (i, p) in (prefix.len() - 1..n).enumerate() {
            let pick = gr.pick(row(p));
            same_tf += (Some(&pick) == tf.get(i)) as usize;
            same_gen += (pick == toks0[p + 1]) as usize;
            if Some(&pick) != tf.get(i) && shown < 8 {
                log(format!("  step {p}: ours {} ({}), the reference's {} ({})", pick, tokens::describe(pick), tf.get(i).copied().unwrap_or(0),
                            tokens::describe(tf.get(i).copied().unwrap_or(0))));
                shown += 1;
            }
            gr.update(toks0[p + 1]).map_err(Error)?;
        }
        let steps = n + 1 - prefix.len();
        log(format!("decoder: the masked argmax agrees with the reference's teacher-forced one at {same_tf} of {steps} steps, with the generated tokens at {same_gen}"));
        if same_tf != steps {
            worst = worst.min(same_tf as f64 / steps as f64);
        }
    }
    if stages.contains(&"generate") {
        let mem = ref_memory()?;
        let duration = ref_wave.len() as f64 / wave::RATE as f64;
        let t0 = Instant::now();
        let ours = e.generate(&mut s, &mem, &prefix, Some(duration.min(WINDOW)), &mut |_| Ok(()))?;
        let took = t0.elapsed().as_secs_f64();
        let (first, same) = agreement(&ours, &toks0);
        log(format!("generate: the longest common subsequence {:.1} % of the longer", 100.0 * lcs_share(&ours, &toks0)));
        log(format!("generate: {} tokens in {took:.2} s ({:.0} a second), the reference's {}: {}", ours.len(), (ours.len() - prefix.len()) as f64 / took, toks0.len(),
                    match first {
                        None => "identical".to_string(),
                        Some(i) => format!("first differ at {i} (ours {} {}, theirs {} {}), {same} of {} agree", ours.get(i).copied().unwrap_or(0),
                                           tokens::describe(ours.get(i).copied().unwrap_or(0)), toks0.get(i).copied().unwrap_or(0),
                                           tokens::describe(toks0.get(i).copied().unwrap_or(0)), ours.len().min(toks0.len())),
                    }));
        if first.is_some() {
            worst = 0.0;
        }
    }
    if stages.contains(&"notation") {
        // the reference's own tokens of every window, through our decode, stitching and notation
        let duration = ref_wave.len() as f64 / wave::RATE as f64;
        let plan = stitch::plan(duration, WINDOW, super::OVERLAP, super::LOOKAHEAD).map_err(Error)?;
        let mut stitched = Vec::new();
        for (i, w) in plan.iter().enumerate() {
            let toks = ids(&format!("tokens{i}"))?;
            let mut base = 0;
            if i > 0 {
                // the window's decode started from the overlap's events: our prefix must be the one its tokens begin with
                if let Some((p, b)) = stitch::overlap_prefix(&stitched, &prompts, w.start, w.prefix_end).map_err(Error)? {
                    base = b;
                    let same = toks.len() >= p.len() && toks[..p.len()] == p[..];
                    log(format!("notation: window {i}'s overlap prefix ({} tokens, from global subbeat {b}) {}", p.len(), if same { "identical" } else { "DIFFERS" }));
                    if !same {
                        worst = 0.0;
                    }
                }
            }
            let (d, _) = tokens::decode_generated(&toks).map_err(Error)?;
            let map = stitch::TimeMap::new(&d, WINDOW);
            stitched.extend(stitch::stitched(&d, &map, w, duration, i, base));
        }
        stitched.sort_by(|a, b| a.time.unwrap_or(0.0).total_cmp(&b.time.unwrap_or(0.0)).then(a.global_subbeat.cmp(&b.global_subbeat)));
        log(format!("notation: {} windows, {} events stitched", plan.len(), stitched.len()));
        let full = notation::export(&stitched, duration, false);
        let mel = notation::export(&stitched, duration, true);
        if !abc_report("notation (full)", &full, dir, "abc_full.abc", log) || !abc_report("notation (melody_only)", &mel, dir, "abc_melody.abc", log) {
            worst = 0.0;
        }
    }
    if stages.contains(&"transcribe") {
        drop(s);
        let t0 = Instant::now();
        let tr = e.transcribe(&ref_wave, &mut |_| Ok(()))?;
        log(format!("transcribe: {:.1} s of recording in {:.2} s (encoder {:.2} s, decoder {:.2} s), {} windows, {} events, warnings {:?}", tr.duration,
                    t0.elapsed().as_secs_f64(), tr.encode_seconds, tr.decode_seconds, tr.tokens.len(), tr.events.len(), tr.warnings));
        for (i, t) in tr.tokens.iter().enumerate() {
            let want = ids(&format!("tokens{i}")).unwrap_or_default();
            let (first, same) = agreement(t, &want);
            let _ = same;
            log(format!("  window {i}: {} tokens, the reference's {}: {} (their longest common subsequence {:.1} % of the longer)", t.len(), want.len(),
                        first.map_or("identical".to_string(), |f| format!("first differ at {f}")), 100.0 * lcs_share(t, &want)));
        }
        // the notes against the reference's (its tokens through the notation path checked above)
        let duration = ref_wave.len() as f64 / wave::RATE as f64;
        let plan = stitch::plan(duration, WINDOW, super::OVERLAP, super::LOOKAHEAD).map_err(Error)?;
        let mut theirs = Vec::new();
        for (i, w) in plan.iter().enumerate() {
            let toks = ids(&format!("tokens{i}"))?;
            let base = if i > 0 { stitch::overlap_prefix(&theirs, &prompts, w.start, w.prefix_end).map_err(Error)?.map_or(0, |p| p.1) } else { 0 };
            let (d, _) = tokens::decode_generated(&toks).map_err(Error)?;
            theirs.extend(stitch::stitched(&d, &stitch::TimeMap::new(&d, WINDOW), w, duration, i, base));
        }
        for (label, st) in [("pitch and onset", false), ("pitch, onset and voice", true)] {
            let (m, no, nt) = note_match(&notes_of(&tr.events), &notes_of(&theirs), st);
            log(format!("  notes ({label}, 50 ms): {m} of ours {no} match the reference's {nt} - precision {:.3}, recall {:.3}", m as f64 / no.max(1) as f64,
                        m as f64 / nt.max(1) as f64));
        }
        let a = abc_report("transcribe (full)", &tr.full, dir, "abc_full.abc", log);
        let b = abc_report("transcribe (melody_only)", &tr.melody, dir, "abc_melody.abc", log);
        if let Some(abc) = &tr.melody.abc {
            std::fs::write(dir.join("ours_melody.abc"), abc).map_err(|er| Error(format!("{er}")))?;
        }
        if let Some(abc) = &tr.full.abc {
            std::fs::write(dir.join("ours_full.abc"), abc).map_err(|er| Error(format!("{er}")))?;
        }
        if !(a && b) {
            worst = worst.min(0.5);
        }
    }
    Ok(worst)
}
