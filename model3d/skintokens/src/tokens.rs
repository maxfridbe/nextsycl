//! TokenRig's tokens (src/tokenizer/tokenizer_part.py, the checkpoint's tokenizer config): 256 coordinate bins over
//! [-1, 1], then branch, bos, eos, pad, spring, the parts (body, hand), cls_none and the classes (rignet, vroid,
//! articulation) - 267 skeleton tokens; the 32,768 skin codes follow (267 ..), and the model's own eos closes the sequence
//! (33,035). A skeleton is `bos cls (part? (branch x y z)? x y z)* eos`: a joint is three bins; after a branch token come its
//! parent's position and then the joint's. After the skeleton's eos come the skin tokens, four a joint.
//!
//! `mask` is what the decoder is masked with at every step (VocabSwitchingLogitsProcessor, src/model/tokenrig.py),
//! `detokenize` the skeleton a sequence spells (joints and their parents, `make_skeleton`).

pub const NUM_DISCRETE: u32 = 256;
pub const BRANCH: u32 = 256;
pub const BOS: u32 = 257;
pub const EOS: u32 = 258;
pub const PAD: u32 = 259;
pub const SPRING: u32 = 260;
pub const PART_BODY: u32 = 261;
pub const PART_HAND: u32 = 262;
pub const CLS_NONE: u32 = 263;
pub const CLS_RIGNET: u32 = 264;
pub const CLS_VROID: u32 = 265;
pub const CLS_ARTICULATION: u32 = 266;
/// The skeleton tokens
pub const VOCAB: usize = 267;
/// A skin token is `SKIN0` + its FSQ code (32,768 of them)
pub const SKIN0: u32 = 267;
pub const CODES: usize = 32_768;
/// The decoder's vocabulary and its own end of sequence
pub const TOTAL: usize = VOCAB + CODES + 1;
pub const EOS_MODEL: u32 = (TOTAL - 1) as u32;
/// The skin tokens a joint has
pub const PER_JOINT: usize = 4;

fn is_cls(t: u32) -> bool {
    (CLS_NONE..=CLS_ARTICULATION).contains(&t)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Bos,
    ClsOrPartOrJoint,
    PartOrJoint,
    Joint2,
    Joint3,
    BranchOrPartOrJoint,
    Joint,
}

fn state_after(ids: &[u32]) -> Option<State> {
    let mut s = State::Bos;
    for &id in ids {
        s = match s {
            State::Bos => {
                if id != BOS {
                    return None;
                }
                State::ClsOrPartOrJoint
            }
            State::ClsOrPartOrJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else if is_cls(id) {
                    State::PartOrJoint
                } else {
                    State::Joint
                }
            }
            State::PartOrJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else {
                    State::PartOrJoint
                }
            }
            State::Joint2 => State::Joint3,
            State::Joint3 => State::BranchOrPartOrJoint,
            State::BranchOrPartOrJoint => {
                if id == BRANCH {
                    State::Joint
                } else if id < NUM_DISCRETE {
                    State::Joint2
                } else {
                    State::Joint
                }
            }
            State::Joint => State::Joint2,
        };
    }
    Some(s)
}

/// The tokens that may follow `ids` (which start with bos; empty: bos), as a mask over the vocabulary
pub fn allowed(ids: &[u32]) -> [bool; VOCAB] {
    let mut m = [false; VOCAB];
    let joints = |m: &mut [bool; VOCAB]| m[..NUM_DISCRETE as usize].iter_mut().for_each(|x| *x = true);
    let parts = |m: &mut [bool; VOCAB]| {
        m[SPRING as usize] = true;
        m[PART_BODY as usize] = true;
        m[PART_HAND as usize] = true;
    };
    if ids.is_empty() {
        m[BOS as usize] = true;
        return m;
    }
    match state_after(ids) {
        None => {}
        Some(State::Bos) => m[BOS as usize] = true,
        Some(State::ClsOrPartOrJoint) => {
            (CLS_NONE..=CLS_ARTICULATION).for_each(|t| m[t as usize] = true);
            parts(&mut m);
            joints(&mut m);
        }
        Some(State::PartOrJoint) => {
            parts(&mut m);
            joints(&mut m);
            m[EOS as usize] = true;
        }
        Some(State::Joint2 | State::Joint3 | State::Joint) => joints(&mut m),
        Some(State::BranchOrPartOrJoint) => {
            joints(&mut m);
            parts(&mut m);
            m[BRANCH as usize] = true;
            m[EOS as usize] = true;
        }
    }
    m
}

/// How many joints the skeleton tokens `ids` spell so far (a joint counts when its third coordinate comes; a branch's
/// parent position is not one): `bones_in_sequence` of src/tokenizer/tokenizer_part.py
pub fn joints_in(ids: &[u32]) -> usize {
    let mut n = 0;
    let mut is_branch = false;
    let mut s = State::Bos;
    for &id in ids {
        s = match s {
            State::Bos => State::ClsOrPartOrJoint,
            State::ClsOrPartOrJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else if is_cls(id) {
                    State::PartOrJoint
                } else {
                    State::Joint
                }
            }
            State::PartOrJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else {
                    State::PartOrJoint
                }
            }
            State::Joint2 => State::Joint3,
            State::Joint3 => {
                if !is_branch {
                    n += 1;
                }
                is_branch = false;
                State::BranchOrPartOrJoint
            }
            State::BranchOrPartOrJoint => {
                if id == BRANCH {
                    is_branch = true;
                    State::Joint
                } else if id < NUM_DISCRETE {
                    State::Joint2
                } else {
                    State::Joint
                }
            }
            State::Joint => State::Joint2,
        };
        if id == EOS {
            break;
        }
    }
    n
}

/// The mask over the decoder's whole vocabulary after `ids` (the start tokens and what is generated), the
/// VocabSwitchingLogitsProcessor of src/model/tokenrig.py: before the skeleton's eos the skeleton grammar; after it every
/// token from eos up except the model's own eos, which alone is allowed once the skin tokens are said to be done. Upstream
/// says so one token early - at `4 J` tokens from the skeleton's eos (it counts the eos itself), so the last joint's fourth
/// token is the model's eos, which reads as FSQ code 0; `fix` says it at `4 J + 1` and lets the model say that token.
pub fn mask(ids: &[u32], fix: bool) -> Vec<bool> {
    let mut m = vec![false; TOTAL];
    match ids.iter().position(|&t| t == EOS) {
        None => m[..VOCAB].copy_from_slice(&allowed(ids)),
        Some(w) => {
            let done = (ids.len() - w) == PER_JOINT * joints_in(ids) + usize::from(fix);
            if done {
                m[TOTAL - 1] = true;
            } else {
                m[EOS as usize..TOTAL - 1].iter_mut().for_each(|x| *x = true);
            }
        }
    }
    m
}

/// A bin's centre: (t + 0.5) / 256 over [-1, 1] (float32, as numpy's float32 arithmetic)
pub fn undiscretize(t: u32) -> f32 {
    let x = (t as f32 + 0.5) / NUM_DISCRETE as f32;
    x * 2.0 + -1.0
}

/// A skeleton: joint positions (the model's frame) and each one's parent (None: the root), parents before children
#[derive(Clone, Debug, PartialEq)]
pub struct Skeleton {
    pub joints: Vec<[f32; 3]>,
    pub parents: Vec<Option<usize>>,
}

/// The skeleton a sequence spells (bos ... eos, pads after it allowed); its joints in the order they come
pub fn detokenize(ids: &[u32]) -> Result<Skeleton, String> {
    if ids.first() != Some(&BOS) {
        return Err("the sequence does not start with bos".into());
    }
    let mut end = ids.len();
    while end > 0 && ids[end - 1] == PAD {
        end -= 1;
    }
    if end < 2 || ids[end - 1] != EOS {
        return Err("the sequence does not end with eos".into());
    }
    let body = &ids[1..end - 1];
    let (mut joints, mut p_joints): (Vec<[f32; 3]>, Vec<[f32; 3]>) = (Vec::new(), Vec::new());
    let mut i = 0;
    let mut is_branch = false;
    let mut last: Option<[f32; 3]> = None;
    let pos = |s: &[u32]| [undiscretize(s[0]), undiscretize(s[1]), undiscretize(s[2])];
    while i < body.len() {
        let id = body[i];
        if id < NUM_DISCRETE {
            if is_branch {
                if i + 6 > body.len() {
                    return Err("a branch cut short".into());
                }
                p_joints.push(pos(&body[i..i + 3]));
                joints.push(pos(&body[i + 3..i + 6]));
                i += 6;
            } else {
                if i + 3 > body.len() {
                    return Err("a joint cut short".into());
                }
                let cur = pos(&body[i..i + 3]);
                joints.push(cur);
                p_joints.push(if p_joints.is_empty() { cur } else { last.ok_or("a joint after a branch's end with no parent")? });
                i += 3;
            }
            last = joints.last().copied();
            is_branch = false;
        } else if id == BRANCH {
            is_branch = true;
            last = None;
            i += 1;
        } else if id == SPRING || id == PART_BODY || id == PART_HAND || is_cls(id) {
            i += 1;
        } else {
            return Err(format!("an unexpected token {id}"));
        }
    }
    if joints.is_empty() {
        return Err("no joints".into());
    }
    // make_skeleton: a joint's parent is the earlier joint nearest its parent position (the last one on ties)
    let mut parents = vec![None];
    for (k, pj) in p_joints.iter().enumerate().skip(1) {
        let mut best = (f32::INFINITY, None);
        let mut dis = 999_999f32;
        for j in (0..k).rev() {
            let d = (0..3).map(|c| (joints[j][c] - pj[c]) * (joints[j][c] - pj[c])).sum::<f32>();
            if d < dis {
                dis = d;
                best = (d, Some(j));
            }
        }
        parents.push(best.1);
    }
    Ok(Skeleton { joints, parents })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grammar_follows_the_sequence() {
        let a = allowed(&[]);
        assert!(a[BOS as usize] && a.iter().filter(|x| **x).count() == 1);
        // after bos: a class, a part or a joint
        let a = allowed(&[BOS]);
        assert!(a[CLS_ARTICULATION as usize] && a[SPRING as usize] && a[0] && !a[EOS as usize] && !a[BRANCH as usize]);
        // after the class: parts, joints or eos
        let a = allowed(&[BOS, CLS_ARTICULATION]);
        assert!(a[EOS as usize] && a[5] && a[PART_BODY as usize] && !a[BRANCH as usize]);
        // inside a joint: coordinates only
        for s in [&[BOS, CLS_ARTICULATION, 10][..], &[BOS, CLS_ARTICULATION, 10, 11]] {
            let a = allowed(s);
            assert!(a[255] && !a[EOS as usize] && !a[BRANCH as usize]);
        }
        // after a whole joint: another, a branch, a part or the end
        let a = allowed(&[BOS, CLS_ARTICULATION, 10, 11, 12]);
        assert!(a[BRANCH as usize] && a[EOS as usize] && a[0] && a[SPRING as usize]);
        // after a branch: coordinates
        let a = allowed(&[BOS, CLS_ARTICULATION, 10, 11, 12, BRANCH]);
        assert!(a[0] && !a[EOS as usize]);
    }

    #[test]
    fn the_skin_phase_ends_where_upstream_does() {
        // two joints; the skeleton's eos at index 8; 4 J = 8 -> the model's eos once `ids.len() - 8` = 8
        let mut ids = vec![BOS, CLS_ARTICULATION, 1, 2, 3, 4, 5, 6, EOS];
        assert_eq!(joints_in(&ids), 2);
        ids.extend([SKIN0; 6]);
        assert!(mask(&ids, false)[300] && !mask(&ids, false)[TOTAL - 1] && !mask(&ids, false)[5]);
        ids.push(SKIN0);
        let m = mask(&ids, false);
        assert!(m[TOTAL - 1] && m.iter().filter(|x| **x).count() == 1);
        // the fix lets the model say one more token
        assert!(!mask(&ids, true)[TOTAL - 1]);
        ids.push(SKIN0);
        assert!(mask(&ids, true)[TOTAL - 1]);
        // before the eos: the skeleton grammar
        assert!(mask(&ids[..5], false)[BRANCH as usize]);
    }

    #[test]
    fn a_skeleton_reads_back_with_its_parents() {
        // root at bin 128, a child after it, then a branch back to the root to a third joint
        let t = [BOS, CLS_ARTICULATION, 128, 128, 128, 128, 128, 200, BRANCH, 128, 128, 128, 60, 128, 128, EOS, PAD, PAD];
        let s = detokenize(&t).unwrap();
        assert_eq!(s.joints.len(), 3);
        assert_eq!(s.parents, vec![None, Some(0), Some(0)]);
        assert_eq!(s.joints[0], [undiscretize(128); 3]);
        assert!((undiscretize(128) - (0.5 / 128.0)).abs() < 1e-7);
        assert!(detokenize(&[BOS, 1, 2, 3]).is_err());
    }
}
