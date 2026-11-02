//! `nextsycl llm audit`: how often a quant picks another word than a reference path, token by token. Two models do not fit the
//! cards at once, so it is two runs and a comparison: a reference run decodes greedily and records each position's best logits;
//! a run with `--follow` reads the same prompts and takes the reference's token at every position (each position compared
//! given the same history), through `Engine::step`, so the drafts and any correction an option turns on are what is measured.
//! `llm audit-report` compares two records: agreement alone, at each margin gate, and the differences by margin band.

use std::path::Path;

use nextsycl_gguf::Gguf;
use nextsycl_llm::{LoadOptions, Sampler};

/// The logits a row keeps (the near-tie rule reads the reference's value of the other run's word)
const TOP: usize = 8;
/// The margin bands of the report, their lower edges
const BANDS: [f32; 8] = [0.0, 0.1, 0.25, 0.45, 0.9, 2.0, 4.0, 8.0];
/// A word the reference prefers by less than this over the run's is a near tie
const NEAR_TIE: f32 = 0.5;

/// One position: the token taken, the best logits (id, value), best first
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub token: u32,
    pub top: Vec<(u32, f32)>,
}

impl Row {
    /// The row's best token
    pub fn best(&self) -> u32 {
        self.top[0].0
    }
    /// The top-two gap
    pub fn margin(&self) -> f32 {
        if self.top.len() < 2 { f32::INFINITY } else { self.top[0].1 - self.top[1].1 }
    }
}

/// One prompt's run
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Run {
    pub prompt: Vec<u32>,
    pub rows: Vec<Row>,
}

/// A record: which file made it, its runs, how fast
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Record {
    pub model: String,
    pub note: String,
    pub runs: Vec<Run>,
}

/// The best `TOP` of a row's logits, best first (ties: the lowest id)
fn top_of(l: &[f32]) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = l.iter().copied().enumerate().map(|(i, x)| (i as u32, x)).collect();
    let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if v.len() > TOP {
        v.select_nth_unstable_by(TOP - 1, cmp);
        v.truncate(TOP);
    }
    v.sort_by(cmp);
    v
}

/// Takes the reference's token (or, without one, the row's best) and keeps every row it was asked for
struct Recorder<'a> {
    follow: Option<&'a [u32]>,
    rows: Vec<Row>,
}

impl Sampler for Recorder<'_> {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let top = top_of(logits);
        let token = self.follow.and_then(|f| f.get(self.rows.len()).copied()).unwrap_or(top[0].0);
        self.rows.push(Row { token, top });
        token
    }
    fn forced(&self) -> bool {
        self.follow.is_some()
    }
}

impl Record {
    pub fn to_text(&self) -> String {
        let mut s = format!("nextsycl-audit 1\nmodel {}\nnote {}\n", self.model, self.note);
        for r in &self.runs {
            s += &format!("prompt {}\n", r.prompt.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
            for w in &r.rows {
                s += &format!("row {} {}\n", w.token, w.top.iter().map(|(i, v)| format!("{i}:{v}")).collect::<Vec<_>>().join(","));
            }
        }
        s
    }

    pub fn parse(text: &str) -> Result<Record, String> {
        let mut lines = text.lines();
        if lines.next() != Some("nextsycl-audit 1") {
            return Err("not an audit record".into());
        }
        let mut rec = Record::default();
        for line in lines {
            let (k, v) = line.split_once(' ').unwrap_or((line, ""));
            match k {
                "model" => rec.model = v.into(),
                "note" => rec.note = v.into(),
                "prompt" => rec.runs.push(Run { prompt: v.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().map_err(|_| format!("prompt id {x:?}"))).collect::<Result<_, _>>()?, rows: Vec::new() }),
                "row" => {
                    let (t, top) = v.split_once(' ').ok_or_else(|| format!("row {v:?}"))?;
                    let top = top.split(',').map(|p| {
                        let (i, x) = p.split_once(':').ok_or_else(|| format!("logit {p:?}"))?;
                        Ok((i.parse().map_err(|_| format!("id {i:?}"))?, x.parse().map_err(|_| format!("value {x:?}"))?))
                    }).collect::<Result<Vec<(u32, f32)>, String>>()?;
                    if top.is_empty() {
                        return Err("a row without logits".into());
                    }
                    rec.runs.last_mut().ok_or("a row before any prompt")?.rows.push(Row { token: t.parse().map_err(|_| format!("token {t:?}"))?, top });
                }
                _ => return Err(format!("line {line:?}")),
            }
        }
        Ok(rec)
    }
}

/// `nextsycl llm audit <model.gguf> (--prompts FILE [--n N] | --follow RECORD) --out RECORD [--note TEXT] [--gpu 0,1] [--no-mtp]`
pub fn run(args: &[String]) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let model = args.get(1).ok_or("audit <model.gguf> (--prompts FILE | --follow RECORD) --out RECORD")?;
    let out = opt("--out").ok_or("--out RECORD")?;
    let n: usize = opt("--n").and_then(|v| v.parse().ok()).unwrap_or(256);
    let f = Gguf::open(Path::new(model)).map_err(|x| x.0)?;
    let tok = nextsycl_tok::Tokenizer::from_gguf(&f)?;
    // each prompt's ids and, following, the reference's tokens
    let follow = opt("--follow").map(|p| Record::parse(&std::fs::read_to_string(&p).map_err(|x| format!("{p}: {x}"))?).map_err(|x| format!("{p}: {x}"))).transpose()?;
    let jobs: Vec<(Vec<u32>, Option<Vec<u32>>)> = match &follow {
        Some(r) => r.runs.iter().map(|x| (x.prompt.clone(), Some(x.rows.iter().map(|w| w.token).collect()))).collect(),
        None => {
            let pf = opt("--prompts").ok_or("--prompts FILE (a prompt a line) or --follow RECORD")?;
            let chat = crate::chat_of(&f)?;
            std::fs::read_to_string(&pf).map_err(|x| format!("{pf}: {x}"))?.lines().filter(|l| !l.trim().is_empty())
                .map(|p| (tok.encode(&chat(&[nextsycl_tok::Message { role: "user", content: p, ..Default::default() }], nextsycl_tok::Effort::Low, &[])), None)).collect()
        }
    };
    let ctx = jobs.iter().map(|(p, t)| p.len() + t.as_ref().map_or(n, Vec::len)).max().unwrap_or(0) + 16;
    let gpus: Vec<usize> = opt("--gpu").unwrap_or_else(|| "0,1".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gpu {v}"))).collect::<Result<_, _>>()?;
    let gs: Vec<std::sync::Arc<nextsycl_core::Gpu>> = gpus.iter().map(|i| nextsycl_core::Gpu::open(*i)).collect::<Result<_, _>>().map_err(e)?;
    let mut log = |l: String| eprintln!("[{l}]");
    let mtp = !args.iter().any(|a| a == "--no-mtp");
    let eng = crate::load_engine(&f, &gs, LoadOptions { draft: mtp, kv: (ctx, 1), ..Default::default() }, &mut log)?;
    let mut none = |_: &str, _: &nextsycl_core::DevBuf| -> nextsycl_core::Result<()> { Ok(()) };
    let mut sess = eng.session(ctx).map_err(e)?;
    let mut rec = Record { model: model.clone(), note: opt("--note").unwrap_or_default(), runs: Vec::new() };
    let (mut total, mut secs, mut pp_n, mut pp_s, mut drafted, mut accepted) = (0usize, 0f64, 0usize, 0f64, 0u64, 0u64);
    for (pi, (prompt, want)) in jobs.iter().enumerate() {
        eng.reset_session(&mut sess).map_err(e)?;
        let t0 = std::time::Instant::now();
        let logits = eng.feed(&mut sess, prompt, &mut none).map_err(e)?;
        pp_s += t0.elapsed().as_secs_f64();
        pp_n += prompt.len();
        let goal = want.as_ref().map_or(n, Vec::len);
        let mut dec = eng.decoder(logits, mtp);
        dec.set_context(prompt);
        let mut smp = Recorder { follow: want.as_deref(), rows: Vec::new() };
        let t1 = std::time::Instant::now();
        let mut done = 0;
        'gen: while done < goal {
            for t in eng.step(&mut sess, &mut dec, &mut smp, &mut none).map_err(e)? {
                // a following run must have taken the reference's token: an option that decides for itself would compare nothing
                if want.as_ref().is_some_and(|w| w[done] != t) {
                    return Err(format!("prompt {pi} position {done}: the engine took {t}, the reference {}: the histories differ", want.as_ref().map_or(0, |w| w[done])));
                }
                done += 1;
                if done >= goal || (want.is_none() && tok.stop.contains(&t)) {
                    break 'gen;
                }
            }
        }
        secs += t1.elapsed().as_secs_f64();
        total += done;
        let (d, a) = dec.drafts();
        drafted += d;
        accepted += a;
        let mut rows = smp.rows;
        rows.truncate(done);
        eprintln!("[prompt {pi}: {} tokens read, {done} decoded]", prompt.len());
        rec.runs.push(Run { prompt: prompt.clone(), rows });
    }
    for line in eng.report(total) {
        eprintln!("{line}");
    }
    let speed = format!("pp {:.1} tok/s ({pp_n} tokens); tg {:.2} tok/s ({total} tokens); drafts {accepted} of {drafted} accepted", pp_n as f64 / pp_s.max(1e-9), total as f64 / secs.max(1e-9));
    eprintln!("[{speed}]");
    rec.note = format!("{} {speed}", rec.note).trim().to_string();
    std::fs::write(&out, rec.to_text()).map_err(|x| format!("{out}: {x}"))
}

/// The comparison of a run with its reference: `exact` says the reference is the model's true weights; otherwise every line
/// calls it a stand-in
pub fn compare(reference: &Record, run: &Record, gates: &[f32], exact: bool) -> Result<String, String> {
    if reference.runs.len() != run.runs.len() {
        return Err(format!("{} prompts in the reference, {} in the run", reference.runs.len(), run.runs.len()));
    }
    // (margin, differs, the near-tie rule would let the run's word stand)
    let mut pos: Vec<(f32, bool, bool)> = Vec::new();
    for (pi, (a, b)) in reference.runs.iter().zip(&run.runs).enumerate() {
        if a.prompt != b.prompt || a.rows.len() != b.rows.len() {
            return Err(format!("prompt {pi}: the two runs did not read the same prompt and length"));
        }
        for (i, (x, y)) in a.rows.iter().zip(&b.rows).enumerate() {
            if x.token != y.token {
                return Err(format!("prompt {pi} position {i}: histories differ ({} against {})", x.token, y.token));
            }
            let differs = x.best() != y.best();
            let near = differs && x.top.iter().find(|t| t.0 == y.best()).is_some_and(|t| x.top[0].1 - t.1 < NEAR_TIE);
            pos.push((y.margin(), differs, near));
        }
    }
    let n = pos.len();
    if n == 0 {
        return Err("no positions".into());
    }
    let pct = |k: usize| 100.0 * k as f64 / n as f64;
    let what = if exact { "the exact weights" } else { "a STAND-IN, not the exact weights" };
    let diff = pos.iter().filter(|p| p.1).count();
    let near = pos.iter().filter(|p| p.2).count();
    let mut s = format!("reference: {} ({what})\nrun: {}\n  reference speed: {}\n  run speed: {}\n{n} positions, each given the reference's history, over {} prompts\n",
                        reference.model, run.model, reference.note, run.note, run.runs.len());
    s += &format!("agreement with the reference as the run stands: {:.2}% ({diff} differ; {near} of them near ties, the reference's word under {NEAR_TIE} logits ahead)\n", pct(n - diff));
    s += "if every position under a gate took the reference's word (a bound, computed from the two records, not a run of the gate):\n";
    for g in gates {
        let (redo, left) = (pos.iter().filter(|p| p.0 < *g).count(), pos.iter().filter(|p| p.0 >= *g && p.1).count());
        s += &format!("  gate {g}: {:.2}% agreement, {redo} positions redone ({:.1}%), {left} differences left\n", pct(n - left), pct(redo));
    }
    s += "by the run's margin (band: positions, differences, agreement):\n";
    for (i, lo) in BANDS.iter().enumerate() {
        let hi = BANDS.get(i + 1).copied().unwrap_or(f32::INFINITY);
        let (c, d) = (pos.iter().filter(|p| p.0 >= *lo && p.0 < hi).count(), pos.iter().filter(|p| p.0 >= *lo && p.0 < hi && p.1).count());
        s += &format!("  [{lo}, {hi}): {c}, {d}, {:.2}%\n", if c == 0 { 100.0 } else { 100.0 * (c - d) as f64 / c as f64 });
    }
    Ok(s)
}

/// `nextsycl llm audit-report --ref RECORD --run RECORD [--gates 0.1,0.45,0.9,2,inf] [--exact]`
pub fn report(args: &[String]) -> Result<(), String> {
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let read = |k: &str| -> Result<Record, String> {
        let p = opt(k).ok_or_else(|| format!("{k} RECORD"))?;
        Record::parse(&std::fs::read_to_string(&p).map_err(|x| format!("{p}: {x}"))?).map_err(|x| format!("{p}: {x}"))
    };
    let gates: Vec<f32> = opt("--gates").unwrap_or_else(|| "0.1,0.25,0.45,0.9,2,inf".into()).split(',').map(|v| v.trim().parse().map_err(|_| format!("--gates {v}"))).collect::<Result<_, _>>()?;
    print!("{}", compare(&read("--ref")?, &read("--run")?, &gates, args.iter().any(|a| a == "--exact"))?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(token: u32, top: &[(u32, f32)]) -> Row {
        Row { token, top: top.to_vec() }
    }
    fn rec(rows: Vec<Row>) -> Record {
        Record { model: "m".into(), note: "n".into(), runs: vec![Run { prompt: vec![1, 2], rows }] }
    }

    #[test]
    fn a_record_reads_back() {
        let r = rec(vec![row(5, &[(5, 1.5), (9, -0.25)])]);
        assert_eq!(Record::parse(&r.to_text()).unwrap(), r);
    }

    #[test]
    fn top_is_best_first_and_ties_take_the_lowest_id() {
        assert_eq!(top_of(&[1.0, 3.0, 3.0, 2.0])[..3], [(1, 3.0), (2, 3.0), (3, 2.0)]);
        assert_eq!(top_of(&(0..100).map(|i| i as f32).collect::<Vec<_>>()).len(), TOP);
    }

    #[test]
    fn the_recorder_takes_the_reference_token_when_following() {
        let want = [7u32];
        let mut s = Recorder { follow: Some(&want), rows: Vec::new() };
        assert_eq!(s.sample(&[0.0, 1.0]), 7);
        assert_eq!(s.rows[0].best(), 1);
        assert!(s.forced());
        let mut g = Recorder { follow: None, rows: Vec::new() };
        assert_eq!(g.sample(&[0.0, 1.0]), 1);
        assert!(!g.forced());
    }

    #[test]
    fn comparison_counts_gates_and_near_ties() {
        // position 0 agrees; 1 differs at margin 0.2, a near tie for the reference; 2 differs at margin 3
        let a = rec(vec![row(1, &[(1, 5.0), (2, 1.0)]), row(2, &[(2, 5.0), (3, 4.8)]), row(4, &[(4, 9.0), (6, 1.0)])]);
        let b = rec(vec![row(1, &[(1, 5.0), (2, 1.0)]), row(2, &[(3, 5.0), (2, 4.8)]), row(4, &[(5, 9.0), (4, 6.0)])]);
        let s = compare(&a, &b, &[0.45, f32::INFINITY], false).unwrap();
        assert!(s.contains("STAND-IN"), "{s}");
        assert!(s.contains("33.33% (2 differ; 1 of them near ties"), "{s}");
        assert!(s.contains("gate 0.45: 66.67% agreement, 1 positions redone"), "{s}");
        assert!(s.contains("gate inf: 100.00% agreement, 3 positions redone"), "{s}");
    }

    #[test]
    fn other_histories_are_refused() {
        let a = rec(vec![row(1, &[(1, 5.0), (2, 1.0)])]);
        let b = rec(vec![row(2, &[(2, 5.0), (1, 1.0)])]);
        assert!(compare(&a, &b, &[], false).is_err());
    }
}
