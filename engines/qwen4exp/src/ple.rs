//! The per-layer embedding's host half: a token's 16 table rows from its 2- and 3-gram (Strata's `ngram_rows`,
//! src/kernels/ngram.cpp), read from the file and decoded (IQ4_NL, 160 values a row). The GPU half is the PLE block in
//! the glue. Rows are read with `pread` from many threads (a prompt chunk reads 16 a token from a 27 GiB table: the
//! page cache keeps what repeats).

use std::fs::File;
use std::os::unix::fs::FileExt;

use ns_gguf::{GType, Gguf};

use crate::model::{Geometry, Model, Role};

/// IQ4_NL's 16 levels (ggml's kvalues_iq4nl)
const IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

pub struct Table {
    file: File,
    offset: u64,
    rows: u64,
    row_bytes: usize,
    dim: usize,
    heads: usize,
    per_ngram: usize,
    ngram: usize,
    eos: i64,
    mult: Vec<u64>,
    vocab: Vec<u64>,
    offsets: Vec<u64>,
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

impl Table {
    pub fn open(m: &Model, f: &Gguf) -> ns_core::Result<Table> {
        let g: &Geometry = &m.g;
        let t = m.t(0, Role::PleTable);
        if t.ty != GType::IQ4NL {
            return Err(ns_core::Error(format!("{}: {} rows; this engine reads IQ4_NL", t.name, t.ty.name())));
        }
        let file = File::open(&f.paths[t.shard]).map_err(|e| ns_core::Error(format!("{}: {e}", f.paths[t.shard].display())))?;
        let dim = g.ple_dim as usize;
        Ok(Table {
            file,
            offset: t.offset,
            rows: g.ple_rows,
            row_bytes: dim / 32 * 18,
            dim,
            heads: g.ple_heads() as usize,
            per_ngram: g.ple_heads_per_ngram as usize,
            ngram: g.ple_ngram as usize,
            eos: g.ple_eos as i64,
            mult: g.ple_mult.clone(),
            vocab: g.ple_vocab.clone(),
            offsets: g.ple_offsets.clone(),
        })
    }

    /// The rows of `tokens` (16 each), `prev` the two tokens before the first (oldest first; -1: none), advanced
    pub fn rows(&self, tokens: &[u32], prev: &mut [i32; 2]) -> Vec<u32> {
        let n_prev = self.ngram - 1;
        let mut out = Vec::with_capacity(tokens.len() * self.heads);
        for &tok in tokens {
            let mut ctx = vec![0i64; self.ngram];
            ctx[0] = tok as i64;
            let mut cut = false;
            for s in 1..self.ngram {
                // `prev` is oldest first: s positions back is entry n_prev - s; a missing or EOS predecessor cuts it and
                // every older one (they read as EOS)
                let t = if cut { -1 } else { prev[n_prev - s] as i64 };
                cut = cut || t < 0 || t == self.eos;
                ctx[s] = if cut { self.eos } else { t };
            }
            for n in 2..=self.ngram {
                // the first product assigned, the rest XORed in, every product mod 2^64
                let mixed = ctx[1..n].iter().zip(&self.mult[1..n]).fold((ctx[0] as u64).wrapping_mul(self.mult[0]), |m, (c, k)| m ^ (*c as u64).wrapping_mul(*k));
                let base = (n - 2) * self.per_ngram;
                for h in base..base + self.per_ngram {
                    out.push((mixed % self.vocab[h] + self.offsets[h]) as u32);
                }
            }
            prev[0] = prev[1];
            prev[1] = tok as i32;
        }
        out
    }

    /// `rows` read and decoded: rows.len() x dim floats, in order
    pub fn gather(&self, rows: &[u32]) -> ns_core::Result<Vec<f32>> {
        let mut out = vec![0f32; rows.len() * self.dim];
        // a decode window's few rows inline; a prompt chunk's thousands over up to 32 threads
        let threads = if rows.len() <= 128 { 1 } else { rows.len().div_ceil(64).clamp(1, 32) };
        let per = rows.len().div_ceil(threads);
        let err = std::sync::Mutex::new(None);
        let work = |rs: &[u32], os: &mut [f32], err: &std::sync::Mutex<Option<String>>| {
            let mut buf = vec![0u8; self.row_bytes];
            for (r, o) in rs.iter().zip(os.chunks_mut(self.dim)) {
                if *r as u64 >= self.rows {
                    *err.lock().unwrap() = Some(format!("PLE row {r} past the table's {}", self.rows));
                    return;
                }
                if let Err(e) = self.file.read_exact_at(&mut buf, self.offset + *r as u64 * self.row_bytes as u64) {
                    *err.lock().unwrap() = Some(format!("reading PLE row {r}: {e}"));
                    return;
                }
                // split halves: qs[j] holds values j and j + 16 of its block of 32
                for (b, blk) in buf.chunks(18).enumerate() {
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for j in 0..16 {
                        o[b * 32 + j] = d * IQ4NL[(blk[2 + j] & 15) as usize] as f32;
                        o[b * 32 + j + 16] = d * IQ4NL[(blk[2 + j] >> 4) as usize] as f32;
                    }
                }
            }
        };
        if threads == 1 {
            work(rows, &mut out, &err);
        } else {
            std::thread::scope(|sc| {
                for (rs, os) in rows.chunks(per).zip(out.chunks_mut(per * self.dim)) {
                    let err = &err;
                    let work = &work;
                    sc.spawn(move || work(rs, os, err));
                }
            });
        }
        if let Some(e) = err.into_inner().unwrap() {
            return Err(ns_core::Error(e));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn halves() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }
}
