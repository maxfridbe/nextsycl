//! The memory plan over the cards, from what each device reports - its compute units and clock, its memory - never a
//! card's name: which card takes the first layers, where the layers split, what a card must keep free, and what it keeps
//! of the experts. Pure (engine.rs reads the devices and the file, then asks here); tested with fake cards of 16, 24
//! and 32 GB and a mixed pair.

/// MiB, GiB
const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;

/// The memory always left free on a card (the xe driver has no out-of-memory error: past the card it spills to host
/// memory, and a spilled arena livelocks the box)
pub const FLOOR: usize = 3 * GIB / 2;

/// A card as the plan sees it: what its device reports
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardInfo {
    pub units: u32,
    pub mhz: u32,
    /// bytes in all, and free now
    pub total: usize,
    pub free: usize,
}

impl CardInfo {
    /// How much it computes (units x clock)
    fn speed(&self) -> u64 {
        self.units as u64 * self.mhz as u64
    }
}

/// The cards' order for the layers (indices into `cards`): with two, the one that computes less first - it takes the
/// first layers, the faster one the rest and the head (hardac1, B65 + B70: a token 89 against 95 ms the other way);
/// ties, one card, more than two, or `asis` (NS_MM31_ORDER=asis) keep the order given
pub fn card_order(cards: &[CardInfo], asis: bool) -> Vec<usize> {
    let mut o: Vec<usize> = (0..cards.len()).collect();
    if cards.len() == 2 && !asis {
        o.sort_by_key(|i| cards[*i].speed());
    }
    o
}

/// The layers over `ncard` cards, as bounds (card c holds bounds[c]..bounds[c + 1]): even, and with two cards 40% on
/// the first (per token as good as half, prompts better: 8K pp 357 against 337); `split` (NS_MM31_SPLIT) the first's
/// count with two cards, when it leaves each some
pub fn layer_bounds(n_layer: usize, ncard: usize, split: Option<usize>) -> Vec<usize> {
    let mut bounds: Vec<usize> = (0..=ncard).map(|i| n_layer * i / ncard.max(1)).collect();
    if ncard == 2 {
        bounds[1] = n_layer * 2 / 5;
        if let Some(n) = split.filter(|n| *n > 0 && *n < n_layer) {
            bounds[1] = n;
        }
    }
    bounds
}

/// What a card must keep beside its layers' weights and its experts: the arena, the sessions' caches, the drafter's
/// caches and scratch (the last card), and the spare (at least the floor)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reserve {
    pub arena: usize,
    pub kv: usize,
    pub draft: usize,
    pub spare: usize,
}

impl Reserve {
    pub fn total(&self) -> usize {
        self.arena + self.kv + self.draft + self.spare
    }
    /// What the card cannot do without: all but the spare's part above the floor
    fn needed(&self) -> usize {
        self.arena + self.kv + self.draft + FLOOR
    }
}

fn gib(b: usize) -> f64 {
    b as f64 / GIB as f64
}

/// Before any weight loads: each card's layers (`dense[c]`: its share of the weights that stay on the card, the head on
/// the last) and what it must keep (`reserve[c]`) within what it has free. Err - the figures of the first card short -
/// when one is short: the model does not load on these cards, rather than spill (refused, not tried)
pub fn check_fit(cards: &[CardInfo], dense: &[usize], reserve: &[Reserve], names: &[String]) -> Result<(), String> {
    for c in 0..cards.len() {
        let need = dense[c] + reserve[c].needed();
        if need > cards[c].free {
            return Err(format!("minimax-m3 does not fit {}: its layers there need {:.1} GiB of weights, {:.1} GiB of arena and caches and {:.1} GiB kept free \
                                ({:.1} GiB) and it has {:.1} GiB free of {:.1} - fewer layers there (NS_MM31_SPLIT), a shorter context or more cards",
                               names.get(c).map_or("a card", String::as_str), gib(dense[c]), gib(reserve[c].arena + reserve[c].kv + reserve[c].draft),
                               gib(FLOOR), gib(need), gib(cards[c].free), gib(cards[c].total)));
        }
    }
    Ok(())
}

/// What a card keeps of the experts in VRAM once its layers are on it: what is free then, less the reserve, at most
/// `cap` (NS_MM31_VRAM_GIB, a load option); never into the floor
pub fn expert_room(free_after: usize, reserve: &Reserve, cap: Option<usize>) -> usize {
    let room = free_after.saturating_sub(reserve.total());
    cap.map_or(room, |c| c.min(room))
}

/// The arena: a prompt pass of 4096 rows fits 1.5 GiB; larger chunks take about 250 KiB a row
pub fn arena_bytes(prefill_chunk: usize) -> usize {
    (1536 * MIB).max(prefill_chunk * (250 << 10))
}

#[cfg(test)]
mod tests {
    use super::*;

    // fake cards as their devices report them: units, clock, bytes (the B70's and B65's figures, and smaller cards)
    const G: usize = GIB;
    fn card(units: u32, mhz: u32, total_gib: usize, free_gib: f64) -> CardInfo {
        CardInfo { units, mhz, total: total_gib * G, free: (free_gib * G as f64) as usize }
    }
    fn b70() -> CardInfo {
        card(256, 2800, 32, 31.5)
    }
    fn b65() -> CardInfo {
        card(160, 2800, 32, 31.5)
    }
    fn reserve(kv_gib: f64) -> Reserve {
        Reserve { arena: arena_bytes(4096), kv: (kv_gib * G as f64) as usize, draft: 0, spare: (1.9 * G as f64) as usize }
    }

    /// The plan before 2026-11-22, by card name (engine.rs): the B65 first, 40% of the layers on the first of two
    fn old_order(names: &[&str], asis: bool) -> Vec<usize> {
        let mut o: Vec<usize> = (0..names.len()).collect();
        if names.len() == 2 && !asis {
            o.sort_by_key(|i| !names[*i].contains("B65"));
        }
        o
    }

    #[test]
    fn hardac1s_order_and_split_are_unchanged() {
        // the served cards in both orders the driver may list them
        for (cards, names) in [([b70(), b65()], ["Intel(R) Arc(TM) Pro B70 Graphics", "Intel(R) Arc(TM) Pro B65 Graphics"]),
                               ([b65(), b70()], ["Intel(R) Arc(TM) Pro B65 Graphics", "Intel(R) Arc(TM) Pro B70 Graphics"])] {
            for asis in [false, true] {
                assert_eq!(card_order(&cards, asis), old_order(&names, asis), "{names:?} asis {asis}");
            }
        }
        // two of one kind: the order given, as before
        assert_eq!(card_order(&[b70(), b70()], false), vec![0, 1]);
        // MiniMax-M3.1's 80 layers: 32 on the first (the B65), and NS_MM31_SPLIT=24 as served
        assert_eq!(layer_bounds(80, 2, None), vec![0, 32, 80]);
        assert_eq!(layer_bounds(80, 2, Some(24)), vec![0, 24, 80]);
        assert_eq!(layer_bounds(80, 2, Some(80)), vec![0, 32, 80], "a split that leaves a card nothing is not taken");
        assert_eq!(layer_bounds(80, 1, Some(24)), vec![0, 80]);
        assert_eq!(layer_bounds(80, 3, None), vec![0, 26, 53, 80]);
        // the experts' room is what it was: free less the reserve, at most the cap
        let r = reserve(2.0);
        assert_eq!(expert_room(20 * G, &r, None), 20 * G - r.total());
        assert_eq!(expert_room(20 * G, &r, Some(5 * G)), 5 * G);
        assert_eq!(expert_room(G, &r, None), 0);
    }

    #[test]
    fn smaller_cards_split_by_what_they_report_and_keep_the_floor() {
        // a 16 GB card with fewer units beside a 24 GB one: the slower first, whatever it is called
        let small = card(128, 2400, 16, 15.6);
        let mid = card(160, 2850, 24, 23.5);
        assert_eq!(card_order(&[mid, small], false), vec![1, 0]);
        // nothing planned past a card: the layers, the reserve and the floor within what is free, the experts after
        for c in [small, mid, b70(), b65()] {
            let r = reserve(1.0);
            let dense = 6 * G;
            if check_fit(&[c], &[dense], &[r], &["x".into()]).is_ok() {
                let room = expert_room(c.free - dense, &r, None);
                assert!(dense + r.total() + room <= c.free, "{c:?}");
                assert!(c.free - (dense + r.arena + r.kv + room) >= FLOOR, "the floor kept on {c:?}");
            }
        }
    }

    #[test]
    fn a_card_too_small_for_its_layers_is_refused_with_the_figures() {
        let small = card(128, 2400, 16, 15.6);
        let e = check_fit(&[small, b70()], &[14 * G, 10 * G], &[reserve(2.0), reserve(2.0)], &["GPU 1 (16 GB)".into(), "GPU 0".into()]).unwrap_err();
        assert!(e.starts_with("minimax-m3 does not fit GPU 1 (16 GB): its layers there need 14.0 GiB of weights, 3.5 GiB of arena and caches and 1.5 GiB kept free (19.0 GiB) and it has 15.6 GiB free of 16.0"), "{e}");
        // the same layers on the cards they fit
        assert!(check_fit(&[b65(), b70()], &[14 * G, 10 * G], &[reserve(2.0), reserve(2.0)], &["a".into(), "b".into()]).is_ok());
        // the spare above the floor is not what refuses: 0.4 GiB short of it still loads (as before), with no experts in VRAM
        let tight = CardInfo { free: 10 * G + reserve(1.0).needed(), ..b70() };
        assert!(check_fit(&[tight], &[10 * G], &[reserve(1.0)], &["t".into()]).is_ok());
        assert_eq!(expert_room(tight.free - 10 * G, &reserve(1.0), None), 0);
    }
}
