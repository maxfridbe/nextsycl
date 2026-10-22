//! DRY ("don't repeat yourself", koboldcpp's and llama.cpp's sampler against loops) for this model's low-bit quants:
//! a 12,000-token story from UD-IQ1_M repeats itself from ~5,000 words and UD-IQ2_XXS drifts from ~8,000 (greedy;
//! llama.cpp's output on the same files does the same). A token that would extend a run of the last n generated
//! tokens which already appeared earlier followed by that token loses multiplier * base^(n - allowed) of its logit
//! (n >= allowed). A run does not reach back across a breaker token (a newline, ':', '"' or '*' in its text) and a
//! breaker is never penalized, so line structure and quoting stay free; only this turn's generated tokens count, so
//! copying from the prompt is never penalized; inside a fenced code block (an odd count of ``` so far) it stands down,
//! code repeating its identifiers and calls. The logits are shaped before the request's sampler sees them (greedy too). NS_DS41_DRY (the multiplier; 0: off), NS_DS41_DRY_BASE (1.75), NS_DS41_DRY_ALLOWED (2).

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dry {
    pub multiplier: f32,
    pub base: f32,
    pub allowed: usize,
}

impl Dry {
    pub fn from_env() -> Option<Dry> {
        let num = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
        let multiplier = num("NS_DS41_DRY", 0.0);
        (multiplier > 0.0).then(|| Dry { multiplier, base: num("NS_DS41_DRY_BASE", 1.75).max(1.0), allowed: (num("NS_DS41_DRY_ALLOWED", 2.0) as usize).max(1) })
    }
}

/// Per token: whether it breaks a run (its text has a newline, ':', '"' or '*') and how many code fences (```) it holds
pub struct Vocab {
    pub breaker: Vec<bool>,
    pub fences: Vec<u8>,
}

pub fn vocab(n_vocab: usize, text: impl Fn(u32) -> String) -> Vocab {
    let (mut breaker, mut fences) = (Vec::with_capacity(n_vocab), Vec::with_capacity(n_vocab));
    for t in 0..n_vocab as u32 {
        let s = text(t);
        breaker.push(s.contains(['\n', ':', '"', '*']));
        fences.push(s.matches("```").count().min(255) as u8);
    }
    Vocab { breaker, fences }
}

/// The penalties on `logits` for the tokens that would extend a repeated run of `hist` (this turn's generated
/// tokens). A Z-function over the reversed history finds, for every earlier position, how many tokens before it match
/// the history's end: O(len) a token.
pub fn penalize(logits: &mut [f32], hist: &[u32], v: &Vocab, d: &Dry) {
    let breaker = &v.breaker;
    // inside a code block: nothing
    if hist.iter().map(|&t| v.fences.get(t as usize).copied().unwrap_or(0) as usize).sum::<usize>() % 2 == 1 {
        return;
    }
    let m = hist.len();
    let is_br = |t: u32| breaker.get(t as usize).copied().unwrap_or(false);
    // the run at the end may not reach back across a breaker
    let cap = hist.iter().rev().take_while(|&&t| !is_br(t)).count();
    if m < 2 || cap < d.allowed {
        return;
    }
    let r: Vec<u32> = hist.iter().rev().copied().collect();
    // z[k]: the longest common prefix of r and r[k..] - how far the run ending at hist[m-1-k] matches the end
    let mut z = vec![0usize; m];
    let (mut l, mut rr) = (0usize, 0usize);
    for k in 1..m {
        let mut n = if k < rr { (rr - k).min(z[k - l]) } else { 0 };
        while k + n < m && r[n] == r[k + n] {
            n += 1;
        }
        z[k] = n;
        if k + n > rr {
            l = k;
            rr = k + n;
        }
    }
    // the longest run each following token would extend
    let mut best: HashMap<u32, usize> = HashMap::new();
    for (k, &zk) in z.iter().enumerate().skip(1) {
        let n = zk.min(cap);
        if n >= d.allowed {
            let e = best.entry(hist[m - k]).or_insert(0);
            *e = (*e).max(n);
        }
    }
    for (t, n) in best {
        if (t as usize) < logits.len() && !is_br(t) {
            logits[t as usize] -= d.multiplier * d.base.powi((n - d.allowed).min(40) as i32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(breaker: Vec<bool>) -> Vocab {
        let n = breaker.len();
        Vocab { breaker, fences: vec![0; n] }
    }

    #[test]
    fn the_token_that_extends_a_repeat_pays() {
        let d = Dry { multiplier: 1.0, base: 2.0, allowed: 2 };
        let br = plain(vec![false; 10]);
        // a b c x a b: c would repeat "a b c" (a run of 2)
        let mut l = vec![0.0f32; 10];
        penalize(&mut l, &[1, 2, 3, 9, 1, 2], &br, &d);
        assert_eq!(l[3], -1.0);
        assert!(l.iter().enumerate().all(|(i, &v)| i == 3 || v == 0.0));
        // a b c x a b c: x would extend a run of 3 - twice the penalty
        let mut l = vec![0.0f32; 10];
        penalize(&mut l, &[1, 2, 3, 9, 1, 2, 3], &br, &d);
        assert_eq!(l[9], -2.0);
    }

    #[test]
    fn runs_stop_at_breakers() {
        // a breaker (5) inside the end's run leaves 1 token of it, under allowed
        let d = Dry { multiplier: 1.0, base: 2.0, allowed: 2 };
        let mut b = vec![false; 10];
        b[5] = true;
        let br = plain(b);
        let mut l = vec![0.0f32; 10];
        penalize(&mut l, &[5, 2, 3, 9, 5, 2], &br, &d);
        assert!(l.iter().all(|&v| v == 0.0));
        // and a breaker that would follow a repeat is not penalized
        let mut l = vec![0.0f32; 10];
        penalize(&mut l, &[1, 2, 5, 9, 1, 2], &br, &d);
        assert!(l.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn code_blocks_are_left_alone() {
        // token 7 opens a fence: the same repeat as above goes unpenalized inside it, and pays again after the close
        let d = Dry { multiplier: 1.0, base: 2.0, allowed: 2 };
        let mut v = plain(vec![false; 10]);
        v.fences[7] = 1;
        let mut l = vec![0.0f32; 10];
        penalize(&mut l, &[7, 1, 2, 3, 9, 1, 2], &v, &d);
        assert!(l.iter().all(|&x| x == 0.0));
        let mut l = vec![0.0f32; 10];
        penalize(&mut l, &[7, 1, 2, 3, 7, 9, 1, 2], &v, &d);
        assert_eq!(l[3], -1.0);
    }
}
