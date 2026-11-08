//! Prompt-lookup drafts (NS_RC_NGRAM=K, 0..7; default 3): Ripple ships no draft model, so the drafts are the
//! conversation's own repeats - when the last 3 tokens occurred before, the up to K tokens that followed them there are
//! verified with the pending token in one pass (the weights are read once for the rows). Exact: a draft is committed only
//! when the token the sampler draws for its row is the draft itself, the cache keeps only what is committed.
//!
//! Backoff: a run of verify passes whose first draft missed stops the proposals for a growing number of steps
//! (2, 4, ... 32), so prose that never repeats pays for a few probes only; one accepted draft ends the backoff.

use std::collections::HashMap;

/// The conversation's tokens with a 3-gram index: `after[(a, b, c)]` is the position after that 3-gram's latest earlier
/// occurrence
#[derive(Default)]
pub struct Lookup {
    hist: Vec<u32>,
    after: HashMap<(u32, u32, u32), usize>,
    /// verify passes in a row whose first draft missed
    misses: u32,
    /// proposals still skipped
    skip: u32,
}

/// The longest pause after misses, in steps
const MAX_SKIP: u32 = 32;

impl Lookup {
    pub fn push(&mut self, t: u32) {
        self.hist.push(t);
        let n = self.hist.len();
        if n >= 4 {
            self.after.insert((self.hist[n - 4], self.hist[n - 3], self.hist[n - 2]), n - 1);
        }
    }

    /// Up to k tokens that followed the last 3 where they occurred before; none while backing off (each call then
    /// counts one skipped step)
    pub fn propose(&mut self, k: usize) -> Vec<u32> {
        let n = self.hist.len();
        if n < 3 || k == 0 {
            return Vec::new();
        }
        if self.skip > 0 {
            self.skip -= 1;
            return Vec::new();
        }
        match self.after.get(&(self.hist[n - 3], self.hist[n - 2], self.hist[n - 1])) {
            Some(&p) if p < n => self.hist[p..(p + k).min(n)].to_vec(),
            _ => Vec::new(),
        }
    }

    /// The outcome of a verify pass: `accepted` of its drafts were committed (0: the first one missed)
    pub fn result(&mut self, accepted: usize) {
        if accepted > 0 {
            self.misses = 0;
            return;
        }
        self.misses += 1;
        if self.misses >= 2 {
            self.skip = (1u32 << (self.misses - 1).min(5)).min(MAX_SKIP);
        }
    }
}

/// What a verify pass commits: `rows` draws the token for row i (0 = the pending token's row); a draft is kept while
/// the draw for its predecessor's row is the draft. Returns the committed tokens (at least one) and how many drafts
/// matched; rows past the first miss are never drawn.
pub fn commit_run(drafts: &[u32], mut draw: impl FnMut(usize) -> u32) -> (Vec<u32>, usize) {
    let mut out = Vec::with_capacity(drafts.len() + 1);
    let mut matched = 0;
    for i in 0..=drafts.len() {
        let y = draw(i);
        out.push(y);
        if i < drafts.len() && y == drafts[i] {
            matched += 1;
            continue;
        }
        break;
    }
    (out, matched)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(t: &[u32]) -> Lookup {
        let mut l = Lookup::default();
        for &x in t {
            l.push(x);
        }
        l
    }

    #[test]
    fn proposes_what_followed_the_last_three() {
        let mut l = filled(&[1, 2, 3, 4, 5, 6, 9, 1, 2, 3]);
        assert_eq!(l.propose(3), vec![4, 5, 6]);
        assert_eq!(l.propose(10), vec![4, 5, 6, 9, 1, 2, 3]);
        assert!(l.propose(0).is_empty());
    }

    #[test]
    fn no_repeat_no_draft() {
        let mut l = filled(&[1, 2, 3, 4, 5, 6]);
        assert!(l.propose(4).is_empty());
        let mut s = filled(&[1, 2]);
        assert!(s.propose(4).is_empty());
    }

    #[test]
    fn the_latest_occurrence_wins() {
        let mut l = filled(&[1, 2, 3, 7, 0, 1, 2, 3, 8, 0, 1, 2, 3]);
        assert_eq!(l.propose(2), vec![8, 0]);
    }

    #[test]
    fn a_draft_never_reads_past_the_history() {
        // the earlier occurrence is right behind the last three: the run is cut at the end of the history
        let mut l = filled(&[5, 5, 5, 5, 5]);
        let d = l.propose(7);
        assert!(!d.is_empty() && d.len() <= 4);
        assert!(d.iter().all(|&x| x == 5));
    }

    #[test]
    fn push_after_a_commit_extends_the_index() {
        let mut l = filled(&[1, 2, 3, 4]);
        assert!(l.propose(2).is_empty());
        for t in [1, 2, 3] {
            l.push(t);
        }
        assert_eq!(l.propose(2), vec![4, 1]);
    }

    #[test]
    fn backoff_grows_and_one_hit_ends_it() {
        let mut l = filled(&[1, 2, 3, 4, 1, 2, 3]);
        l.result(0);
        assert_eq!(l.propose(2), vec![4, 1]); // one miss: still probing
        l.result(0); // second in a row: pause 2 steps
        assert!(l.propose(2).is_empty());
        assert!(l.propose(2).is_empty());
        assert_eq!(l.propose(2), vec![4, 1]);
        l.result(0); // third: pause 4
        for _ in 0..4 {
            assert!(l.propose(2).is_empty());
        }
        assert_eq!(l.propose(2), vec![4, 1]);
        l.result(1); // a hit ends the backoff
        l.result(0);
        assert_eq!(l.propose(2), vec![4, 1]);
    }

    #[test]
    fn backoff_is_capped() {
        let mut l = filled(&[1, 2, 3, 4, 1, 2, 3]);
        for _ in 0..40 {
            l.result(0);
        }
        let mut skipped = 0;
        while l.propose(2).is_empty() {
            skipped += 1;
            assert!(skipped <= MAX_SKIP);
        }
        assert_eq!(skipped, MAX_SKIP);
    }

    #[test]
    fn commit_run_keeps_the_matching_prefix_and_the_first_miss() {
        // every draft right: all of them and the bonus token
        assert_eq!(commit_run(&[7, 8, 9], |i| [7, 8, 9, 10][i]), (vec![7, 8, 9, 10], 3));
        // the second draft wrong: the first and the correction; later rows are not drawn
        let mut drawn = Vec::new();
        let r = commit_run(&[7, 8, 9], |i| {
            drawn.push(i);
            [7, 5, 9, 10][i]
        });
        assert_eq!(r, (vec![7, 5], 1));
        assert_eq!(drawn, vec![0, 1]);
        // the first wrong: just the drawn token
        assert_eq!(commit_run(&[7, 8], |_| 1), (vec![1], 0));
        // no drafts: one token
        assert_eq!(commit_run(&[], |_| 4), (vec![4], 0));
    }
}
