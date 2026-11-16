//! Steering (`steer.gguf`, `zonos3_steer`): named directions a request doses. A `raw` set's directions move the
//! speaker embedding (their deltas summed, the result rescaled to the embedding's norm: speaker-embedding prompts
//! only); an `attnres_source` set's add to one Full AttnRes bank source on every generated frame (any prompt).
//!
//! Origin: Zyphra's `zonos3.tts.steering.resolve_controls` (github.com/Zyphra/ZONOS3-Mini, MIT), constant doses (this
//! bundle has no steer_release block; spans are not taken).

use nextsycl_audio::{Error, Result};

use crate::ops::Weights;

pub struct Direction {
    pub id: String,
    /// None: the raw (speaker embedding) space; Some(source): the AttnRes source it adds to
    pub source: Option<usize>,
    pub v: Vec<f32>,
}

pub struct Steer {
    pub dirs: Vec<Direction>,
    /// the pace a dose sets (`zonos3_steer.timing`: id -> [(dose, duration factor)]), for placing control spans
    pub timing: Vec<(String, Vec<(f32, f32)>)>,
}

/// A request's controls resolved: the moved speaker embedding (when raw ones were given) and the hidden deltas by
/// source
#[derive(Debug, Default)]
pub struct Resolved {
    pub embedding: Option<Vec<f32>>,
    pub deltas: Vec<(usize, Vec<f32>)>,
}

impl Steer {
    pub fn load(w: &Weights) -> Result<Steer> {
        if w.g.architecture() != "zonos3_steer" {
            return Err(Error(format!("{}: not a ZONOS3 steering bundle", w.name())));
        }
        let names = w.ss("zonos3_steer.set.names")?;
        let spaces = w.ss("zonos3_steer.set.spaces")?;
        let sources: Vec<i64> = w.meta("zonos3_steer.set.sources")?.as_array().unwrap_or(&[]).iter().map(|v| v.as_i64().unwrap_or(-1)).collect();
        let ids = w.ss("zonos3_steer.direction.ids")?;
        let mut dirs = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let set = id.split('.').next().unwrap_or("");
            let si = names.iter().position(|n| n == set).ok_or_else(|| Error(format!("{}: direction {id} has no set", w.name())))?;
            let source = match spaces.get(si).map(String::as_str) {
                Some("raw") => None,
                Some("attnres_source") => Some(*sources.get(si).unwrap_or(&-1)).filter(|s| *s >= 0).map(|s| s as usize),
                s => return Err(Error(format!("{}: set {set} has space {s:?}", w.name()))),
            };
            dirs.push(Direction { id: id.clone(), source, v: w.f32(&format!("steer.dir.{i}"))? });
        }
        let timing = w.s("zonos3_steer.timing").ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|j| j.as_object().cloned()).unwrap_or_default().into_iter()
            .map(|(k, v)| {
                let mut pts: Vec<(f32, f32)> = v.as_array().into_iter().flatten()
                    .filter_map(|p| Some((p.get(0)?.as_f64()? as f32, p.get(1)?.as_f64()? as f32))).collect();
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                (k, pts)
            }).collect();
        Ok(Steer { dirs, timing })
    }

    /// The expected duration factor of a set of doses (the product over the timed ones, np.interp of each table)
    pub fn duration_factor(&self, controls: &[(String, f32)]) -> f32 {
        let mut f = 1.0;
        for (id, pts) in &self.timing {
            let Some(d) = controls.iter().find(|c| &c.0 == id).map(|c| c.1).filter(|d| *d != 0.0) else { continue };
            if pts.is_empty() {
                continue;
            }
            let y = if d <= pts[0].0 {
                pts[0].1
            } else if d >= pts[pts.len() - 1].0 {
                pts[pts.len() - 1].1
            } else {
                let k = pts.windows(2).position(|w| d >= w[0].0 && d <= w[1].0).unwrap_or(0);
                let (a, b) = (pts[k], pts[k + 1]);
                a.1 + (b.1 - a.1) * (d - a.0) / (b.0 - a.0)
            };
            f *= y;
        }
        f
    }

    /// A span's hidden deltas (only hidden-space directions vary inside a request)
    pub fn span_deltas(&self, controls: &[(String, f32)], hidden: usize) -> Result<Vec<(usize, Vec<f32>)>> {
        for (id, _) in controls {
            if self.dirs.iter().any(|d| &d.id == id && d.source.is_none()) {
                return Err(Error(format!("control {id} moves the speaker embedding and cannot vary inside a text; only hidden-space controls take [steer] spans")));
            }
        }
        Ok(self.resolve(controls, None, hidden)?.deltas)
    }

    pub fn ids(&self) -> Vec<String> {
        self.dirs.iter().map(|d| d.id.clone()).collect()
    }

    /// `controls` (id, dose) against a speaker embedding (None: no speaker row - raw controls refused)
    pub fn resolve(&self, controls: &[(String, f32)], embedding: Option<&[f32]>, hidden: usize) -> Result<Resolved> {
        let mut r = Resolved::default();
        let mut delta: Option<Vec<f32>> = None;
        for (id, dose) in controls {
            if *dose == 0.0 {
                continue;
            }
            if !dose.is_finite() {
                return Err(Error(format!("control {id}: a dose of {dose}")));
            }
            let d = self.dirs.iter().find(|d| &d.id == id).ok_or_else(|| Error(format!("no control {id} (the bundle's: {})", self.ids().join(", "))))?;
            match d.source {
                None => {
                    let base = embedding.ok_or_else(|| Error(format!("control {id} moves the speaker embedding: it needs a speaker embedding (a recording, or speaker-mode embedding)")))?;
                    if d.v.len() != base.len() {
                        return Err(Error(format!("control {id} is {} wide, the speaker embedding {}", d.v.len(), base.len())));
                    }
                    let acc = delta.get_or_insert_with(|| vec![0.0; base.len()]);
                    for (a, v) in acc.iter_mut().zip(&d.v) {
                        *a += dose * v;
                    }
                }
                Some(src) => {
                    if d.v.len() != hidden {
                        return Err(Error(format!("control {id} is {} wide, the model's hidden size {hidden}", d.v.len())));
                    }
                    match r.deltas.iter_mut().find(|(s, _)| *s == src) {
                        Some((_, acc)) => acc.iter_mut().zip(&d.v).for_each(|(a, v)| *a += dose * v),
                        None => r.deltas.push((src, d.v.iter().map(|v| dose * v).collect())),
                    }
                }
            }
        }
        if let (Some(delta), Some(base)) = (delta, embedding) {
            let moved: Vec<f32> = base.iter().zip(&delta).map(|(b, d)| b + d).collect();
            let nb = base.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nm = moved.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            r.embedding = Some(moved.iter().map(|x| x * nb / nm).collect());
        }
        Ok(r)
    }
}

/// "a.b=1.5, c.d=-2" (or ':') as (id, dose) pairs
pub fn parse_controls(s: &str) -> Result<Vec<(String, f32)>> {
    s.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()).map(|p| {
        let (k, v) = p.split_once(['=', ':']).ok_or_else(|| Error(format!("control {p:?}: id=dose expected")))?;
        let dose: f32 = v.trim().parse().map_err(|_| Error(format!("control {p:?}: the dose is not a number")))?;
        Ok((k.trim().to_string(), dose))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_controls_keep_the_norm_and_hidden_ones_sum() {
        let s = Steer { dirs: vec![
            Direction { id: "identity.gender".into(), source: None, v: vec![1.0, 0.0] },
            Direction { id: "mood.happy".into(), source: Some(19), v: vec![1.0, 2.0, 3.0] },
            Direction { id: "mood.sad".into(), source: Some(19), v: vec![1.0, 1.0, 1.0] },
        ], timing: vec![("delivery.tempo".into(), vec![(-2.0, 2.0), (0.0, 1.0), (2.0, 0.5)])] };
        assert_eq!(s.duration_factor(&[("delivery.tempo".into(), 1.0)]), 0.75);
        assert_eq!(s.duration_factor(&[("delivery.tempo".into(), 9.0)]), 0.5);
        assert!(s.span_deltas(&[("identity.gender".into(), 1.0)], 3).is_err());
        let c = parse_controls("identity.gender=2, mood.happy:1,mood.sad=-1").unwrap();
        let r = s.resolve(&c, Some(&[0.0, 3.0]), 3).unwrap();
        let e = r.embedding.unwrap();
        assert!(((e[0] * e[0] + e[1] * e[1]).sqrt() - 3.0).abs() < 1e-5);
        assert!(e[0] > 0.0);
        assert_eq!(r.deltas, vec![(19, vec![0.0, 1.0, 2.0])]);
        assert!(s.resolve(&c, None, 3).is_err());
        assert!(parse_controls("x").is_err());
    }
}
