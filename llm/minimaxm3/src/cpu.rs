//! Routed-expert products on the host's cores (NS_MM31_CPU_EXPERTS=N, off by default): an expert in pinned memory multiplied where it lies,
//! instead of copied over PCIe (16 MB, ~0.57 ms on the x8 link) before the card's product. The silo's `gu` and `down`
//! (kernels/llm/minimaxm3/silo/moe.cpp) on float32 activations in the same formulas (ggml's dequantize_row_*), on AVX-512: each weight is the
//! GPU's value bit for bit; only the order of the sums differs (relative error ~2e-7 a product). gate and up IQ1_M or IQ2_XXS, down IQ3_XXS;
//! a layer of another type (IQ4_XS down, 3 layers) stays on the card.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Mutex};

use crate::grids::{IQ1S_GRID_GPU, IQ2XXS_GRID, IQ3XXS_GRID};

pub const T_IQ2_XXS: i32 = 16;
pub const T_IQ3_XXS: i32 = 18;
pub const T_IQ1_M: i32 = 29;

pub fn sb_bytes(t: i32) -> usize {
    match t {
        T_IQ2_XXS => 66,
        T_IQ3_XXS => 98,
        T_IQ1_M => 56,
        _ => 0,
    }
}

/// Whether this host runs the products, and an expert of these types (gate and up, down)
pub fn supported(gu: i32, down: i32) -> bool {
    host_ok() && matches!(gu, T_IQ1_M | T_IQ2_XXS) && down == T_IQ3_XXS
}

fn host_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn h2f(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 1023) as u32;
    match e {
        0 => {
            let v = m as f32 * 2f32.powi(-24);
            if s != 0 {
                -v
            } else {
                v
            }
        }
        31 => f32::from_bits(s | 0x7f80_0000 | (m << 13)),
        _ => f32::from_bits(s | ((e + 112) << 23) | (m << 13)),
    }
}

/// ggml's sign byte of seven bits: the seven and their parity
static KS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        t[i] = (i as u32 | (((i as u32).count_ones() & 1) << 7)) as u8;
        i += 1;
    }
    t
};

/// IQ1_M's grid (ggml's GPU form: the entry + 1 in nibbles, values 0-3 the even nibbles, 4-7 the odd) as eight signed bytes, value j the entry's j - 1
fn iq1_table() -> &'static [u64] {
    static T: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        IQ1S_GRID_GPU.iter().map(|&e| (0..8).fold(0u64, |v, j| v | (((((e >> if j < 4 { 8 * j } else { 8 * (j - 4) + 4 }) & 0xF) as i32 - 1) as u8 as u64) << (8 * j)))).collect()
    })
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;
    use std::arch::x86_64::*;

    /// a row of IQ1_M blocks . x; zmm j holds groups 2j (lanes 0-7) and 2j + 1, which share a scale
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    pub unsafe fn dot1(row: *const u8, nb: usize, x: *const f32) -> f32 {
        let t = iq1_table().as_ptr() as *const i64;
        let mut acc = [_mm512_setzero_ps(); 4];
        let pidx = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        let sh = _mm512_setr_epi32(0, 3, 6, 9, 16, 19, 22, 25, 0, 3, 6, 9, 16, 19, 22, 25);
        for k in 0..nb {
            let b = row.add(k * 56);
            let sw = (b.add(48) as *const u64).read_unaligned();
            let (s0, s1, s2, s3) = ((sw & 0xffff) as u32, ((sw >> 16) & 0xffff) as u32, ((sw >> 32) & 0xffff) as u32, (sw >> 48) as u32);
            let d = h2f(((s0 >> 12) | ((s1 >> 8) & 0xf0) | ((s2 >> 4) & 0xf00) | (s3 & 0xf000)) as u16);
            // scale field f (a pair of groups): u16 f / 4 at bit 3 (f % 4); dl = d (2 field + 1), as the kernel
            let src = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(sw as u32 as i32), _mm512_set1_epi32((sw >> 32) as u32 as i32));
            let fld = _mm512_and_si512(_mm512_srlv_epi32(src, sh), _mm512_set1_epi32(7));
            let dlvec = _mm512_mul_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_slli_epi32(fld, 1), _mm512_set1_epi32(1))));
            let mut dl = [0f32; 16];
            _mm512_storeu_ps(dl.as_mut_ptr(), dlvec);
            // group g's nibble of qh: byte g / 2, low for even g; 3 index bits and the delta's sign
            let qh = _mm_loadu_si128(b.add(32) as *const __m128i);
            let lo = _mm_and_si128(qh, _mm_set1_epi8(0x0f));
            let hi = _mm_and_si128(_mm_srli_epi16(qh, 4), _mm_set1_epi8(0x0f));
            let n0 = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi));
            let n1 = _mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi));
            let seven = _mm512_set1_epi32(7);
            let i0 = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128(b as *const __m128i)), _mm512_slli_epi32(_mm512_and_si512(n0, seven), 8));
            let i1 = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128(b.add(16) as *const __m128i)), _mm512_slli_epi32(_mm512_and_si512(n1, seven), 8));
            let (pd, nd) = (_mm512_set1_ps(0.125), _mm512_set1_ps(-0.125));
            let del0 = _mm512_mask_blend_ps(_mm512_test_epi32_mask(n0, _mm512_set1_epi32(8)), pd, nd);
            let del1 = _mm512_mask_blend_ps(_mm512_test_epi32_mask(n1, _mm512_set1_epi32(8)), pd, nd);
            let g = [
                _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(i0), t),
                _mm512_i32gather_epi64::<8>(_mm512_extracti64x4_epi64::<1>(i0), t),
                _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(i1), t),
                _mm512_i32gather_epi64::<8>(_mm512_extracti64x4_epi64::<1>(i1), t),
            ];
            macro_rules! pair {
                ($j:literal, $gi:literal, $q:literal) => {{
                    let v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32::<$q>(g[$gi])));
                    let dsrc = if $j < 8 { del0 } else { del1 };
                    let dv = _mm512_permutexvar_ps(_mm512_add_epi32(pidx, _mm512_set1_epi32(2 * ($j % 8))), dsrc);
                    // dl (float(q - 1) + delta), the kernel's weight
                    let w = _mm512_mul_ps(_mm512_set1_ps(dl[$j]), _mm512_add_ps(v, dv));
                    acc[$j & 3] = _mm512_fmadd_ps(w, _mm512_loadu_ps(x.add(k * 256 + $j * 16)), acc[$j & 3]);
                }};
            }
            pair!(0, 0, 0);
            pair!(1, 0, 1);
            pair!(2, 0, 2);
            pair!(3, 0, 3);
            pair!(4, 1, 0);
            pair!(5, 1, 1);
            pair!(6, 1, 2);
            pair!(7, 1, 3);
            pair!(8, 2, 0);
            pair!(9, 2, 1);
            pair!(10, 2, 2);
            pair!(11, 2, 3);
            pair!(12, 3, 0);
            pair!(13, 3, 1);
            pair!(14, 3, 2);
            pair!(15, 3, 3);
        }
        _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3])))
    }

    /// IQ2_XXS: per 32 values four grid bytes then 28 sign bits (4 x 7) and a 4-bit scale; zmm j is groups 2j, 2j + 1 (one 32 of values a pair of zmm)
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    pub unsafe fn dot2(row: *const u8, nb: usize, x: *const f32) -> f32 {
        let t = IQ2XXS_GRID.as_ptr() as *const i64;
        let mut acc = [_mm512_setzero_ps(); 2];
        // byte i of the result = byte 8 (i / 4) + i % 4 of the 64 after d: the grid indices in group order
        let sel = _mm512_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 58, 57, 56, 51, 50, 49, 48, 43, 42, 41, 40, 35, 34, 33, 32, 27, 26, 25, 24, 19, 18, 17, 16, 11, 10, 9, 8, 3, 2, 1, 0,
        );
        let ks = KS.as_ptr();
        for k in 0..nb {
            let b = row.add(k * 66);
            let d = h2f((b as *const u16).read_unaligned());
            let raw = _mm512_loadu_si512(b.add(2) as *const __m512i);
            let ix = _mm512_permutexvar_epi8(sel, raw);
            let i0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(ix));
            let i1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<1>(ix));
            let g = [
                _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(i0), t),
                _mm512_i32gather_epi64::<8>(_mm512_extracti64x4_epi64::<1>(i0), t),
                _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(i1), t),
                _mm512_i32gather_epi64::<8>(_mm512_extracti64x4_epi64::<1>(i1), t),
            ];
            macro_rules! ib {
                ($ib:literal, $gi:literal, $q0:literal, $q1:literal) => {{
                    let a1 = (b.add(6 + 8 * $ib) as *const u32).read_unaligned();
                    let db = _mm512_set1_ps(d * (0.5 + (a1 >> 28) as f32) * 0.25);
                    let m = (*ks.add((a1 & 127) as usize) as u32)
                        | ((*ks.add(((a1 >> 7) & 127) as usize) as u32) << 8)
                        | ((*ks.add(((a1 >> 14) & 127) as usize) as u32) << 16)
                        | ((*ks.add(((a1 >> 21) & 127) as usize) as u32) << 24);
                    let w0 = _mm512_mul_ps(db, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$q0>(g[$gi]))));
                    let w0 = _mm512_mask_sub_ps(w0, m as u16, _mm512_setzero_ps(), w0);
                    acc[0] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x.add(k * 256 + $ib * 32)), acc[0]);
                    let w1 = _mm512_mul_ps(db, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$q1>(g[$gi]))));
                    let w1 = _mm512_mask_sub_ps(w1, (m >> 16) as u16, _mm512_setzero_ps(), w1);
                    acc[1] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x.add(k * 256 + $ib * 32 + 16)), acc[1]);
                }};
            }
            ib!(0, 0, 0, 1);
            ib!(1, 0, 2, 3);
            ib!(2, 1, 0, 1);
            ib!(3, 1, 2, 3);
            ib!(4, 2, 0, 1);
            ib!(5, 2, 2, 3);
            ib!(6, 3, 0, 1);
            ib!(7, 3, 2, 3);
        }
        _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]))
    }

    /// IQ3_XXS: 64 grid indices (u32 entries of 4 values, two a group), then per 32 values 28 sign bits and a 4-bit scale
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    pub unsafe fn dot3(row: *const u8, nb: usize, x: *const f32) -> f32 {
        let t = IQ3XXS_GRID.as_ptr() as *const i32;
        let mut acc = [_mm512_setzero_ps(); 2];
        let ks = KS.as_ptr();
        for k in 0..nb {
            let b = row.add(k * 98);
            let d = h2f((b as *const u16).read_unaligned());
            let gs = [
                _mm512_i32gather_epi32::<4>(_mm512_cvtepu8_epi32(_mm_loadu_si128(b.add(2) as *const __m128i)), t),
                _mm512_i32gather_epi32::<4>(_mm512_cvtepu8_epi32(_mm_loadu_si128(b.add(18) as *const __m128i)), t),
                _mm512_i32gather_epi32::<4>(_mm512_cvtepu8_epi32(_mm_loadu_si128(b.add(34) as *const __m128i)), t),
                _mm512_i32gather_epi32::<4>(_mm512_cvtepu8_epi32(_mm_loadu_si128(b.add(50) as *const __m128i)), t),
            ];
            macro_rules! ib {
                ($ib:literal, $gi:literal, $q0:literal, $q1:literal) => {{
                    let aux = (b.add(66 + 4 * $ib) as *const u32).read_unaligned();
                    let db = _mm512_set1_ps(d * (0.5 + (aux >> 28) as f32) * 0.5);
                    let m = (*ks.add((aux & 127) as usize) as u32)
                        | ((*ks.add(((aux >> 7) & 127) as usize) as u32) << 8)
                        | ((*ks.add(((aux >> 14) & 127) as usize) as u32) << 16)
                        | ((*ks.add(((aux >> 21) & 127) as usize) as u32) << 24);
                    let w0 = _mm512_mul_ps(db, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$q0>(gs[$gi]))));
                    let w0 = _mm512_mask_sub_ps(w0, m as u16, _mm512_setzero_ps(), w0);
                    acc[0] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x.add(k * 256 + $ib * 32)), acc[0]);
                    let w1 = _mm512_mul_ps(db, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$q1>(gs[$gi]))));
                    let w1 = _mm512_mask_sub_ps(w1, (m >> 16) as u16, _mm512_setzero_ps(), w1);
                    acc[1] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x.add(k * 256 + $ib * 32 + 16)), acc[1]);
                }};
            }
            ib!(0, 0, 0, 1);
            ib!(1, 0, 2, 3);
            ib!(2, 1, 0, 1);
            ib!(3, 1, 2, 3);
            ib!(4, 2, 0, 1);
            ib!(5, 2, 2, 3);
            ib!(6, 3, 0, 1);
            ib!(7, 3, 2, 3);
        }
        _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]))
    }
}

/// row . x for a row of nb blocks of type t (one of `supported`'s)
///
/// # Safety
/// `row` holds nb whole blocks (and IQ2_XXS reads the 64 bytes after a block's d, inside the block); `x` nb * 256 floats; the host has AVX-512 (`supported`).
unsafe fn row_dot(t: i32, row: *const u8, nb: usize, x: *const f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match t {
            T_IQ1_M => simd::dot1(row, nb, x),
            T_IQ2_XXS => simd::dot2(row, nb, x),
            _ => simd::dot3(row, nb, x),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (t, row, nb, x);
        0.0
    }
}

type JobFn = Arc<dyn Fn(usize) + Send + Sync>;

/// A job: its closure, its parts and its generation, taken together (a worker late from the job before must never run that job's closure on
/// this one's parts: its buffers are gone)
struct Job {
    f: JobFn,
    parts: usize,
    gen: u64,
}

struct Shared {
    job: Mutex<Option<Arc<Job>>>,
    /// generation << 32 | the next part; a part is taken only while the generation is the job's
    ctr: AtomicU64,
    left: AtomicUsize,
    stop: std::sync::atomic::AtomicBool,
}

/// Workers that spin between jobs (a layer's products are well under a millisecond, a sleeping thread's wake-up would cost a good part of it);
/// after 20 ms without a job they poll every 100 us, so an idle engine leaves the cores alone. The caller works too.
pub struct Pool {
    sh: Arc<Shared>,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let sh = Arc::new(Shared { job: Mutex::new(None), ctr: 0.into(), left: 0.into(), stop: false.into() });
        for _ in 1..threads.max(1) {
            let s = sh.clone();
            std::thread::spawn(move || {
                let mut seen = 0u64;
                let mut idle = std::time::Instant::now();
                let mut spins = 0u32;
                while !s.stop.load(Acquire) {
                    let g = s.ctr.load(Acquire) >> 32;
                    if g == seen {
                        std::hint::spin_loop();
                        spins = spins.wrapping_add(1);
                        if spins & 0x3ff == 0 && idle.elapsed().as_millis() >= 20 {
                            std::thread::sleep(std::time::Duration::from_micros(100));
                        }
                        continue;
                    }
                    seen = g;
                    idle = std::time::Instant::now();
                    let job = s.job.lock().unwrap().clone();
                    if let Some(j) = job.filter(|j| j.gen == g) {
                        Self::work(&s, &j);
                    }
                }
            });
        }
        Pool { sh }
    }

    fn work(s: &Shared, j: &Job) {
        loop {
            let cur = s.ctr.load(Acquire);
            if cur >> 32 != j.gen {
                return;
            }
            let i = (cur & 0xffff_ffff) as usize;
            if i >= j.parts {
                return;
            }
            if s.ctr.compare_exchange_weak(cur, cur + 1, AcqRel, Acquire).is_ok() {
                (j.f)(i);
                s.left.fetch_sub(1, AcqRel);
            }
        }
    }

    /// f(0..parts) over the workers and this thread; back when every part is done
    pub fn run(&self, parts: usize, f: JobFn) {
        let s = &*self.sh;
        let gen = (s.ctr.load(Acquire) >> 32) + 1;
        let job = Arc::new(Job { f, parts, gen });
        *s.job.lock().unwrap() = Some(job.clone());
        s.left.store(parts, Release);
        s.ctr.store(gen << 32, Release);
        Self::work(s, &job);
        while s.left.load(Acquire) > 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, Release);
    }
}

#[derive(Clone, Copy)]
struct P(usize);

/// rows of a part
const RB: usize = 32;

/// y[r] = row r . x for rows of matrices m0 (and m1 after it: rows0 rows each): one region over both
///
/// # Safety
/// The matrices hold their rows of `cols` values in type t, x cols floats, y the rows; all outlive the call.
#[allow(clippy::too_many_arguments)]
unsafe fn matvec2(pool: &Pool, t: i32, m0: *const u8, m1: Option<*const u8>, rows0: usize, cols: usize, x: *const f32, y: *mut f32) {
    let (nb, rowb) = (cols / 256, cols / 256 * sb_bytes(t));
    let total = rows0 * if m1.is_some() { 2 } else { 1 };
    let (a, b, xp, yp) = (P(m0 as usize), P(m1.unwrap_or(m0) as usize), P(x as usize), P(y as usize));
    pool.run(total.div_ceil(RB), Arc::new(move |i| {
        let (a, b, xp, yp) = (a, b, xp, yp);
        for r in i * RB..((i + 1) * RB).min(total) {
            let (base, rr) = if r < rows0 { (a.0, r) } else { (b.0, r - rows0) };
            // SAFETY: the caller's sizes; each part writes its own rows.
            unsafe { *(yp.0 as *mut f32).add(r) = row_dot(t, (base + rr * rowb) as *const u8, nb, xp.0 as *const f32) };
        }
    }));
}

/// The expert at `base` against x: y = w (down . swiglu_oai(gate . x, up . x)), the silo's `gu` then `down`
///
/// # Safety
/// base holds a whole expert of these types (gate, up off_up bytes after it, down off_down bytes after it; fe rows of c values, c rows of fe);
/// x c floats, y c floats; `supported(tgu, tdown)`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn expert(pool: &Pool, tgu: i32, tdown: i32, base: *const u8, off_up: usize, off_down: usize, fe: usize, c: usize, x: &[f32], w: f32, y: &mut [f32]) {
    let mut gu = vec![0f32; 2 * fe];
    let mut h = vec![0f32; fe];
    matvec2(pool, tgu, base, Some(base.add(off_up)), fe, c, x.as_ptr(), gu.as_mut_ptr());
    for r in 0..fe {
        let xg = gu[r].min(7.0);
        let yu = gu[fe + r].clamp(-7.0, 7.0);
        h[r] = (xg / (1.0 + (1.702 * -xg).exp())) * (yu + 1.0);
    }
    matvec2(pool, tdown, base.add(off_down), None, c, fe, h.as_ptr(), y.as_mut_ptr());
    for v in y.iter_mut() {
        *v *= w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ksign(x: u32) -> u32 {
        KS[(x & 127) as usize] as u32
    }

    /// the silo's group formulas, lane by lane, then the lanes summed
    fn reference(t: i32, b0: &[u8], x: &[f32]) -> f32 {
        let nb = x.len() / 256;
        let mut lanes = [0f32; 32];
        for k in 0..nb {
            let b = &b0[k * sb_bytes(t)..];
            let u16at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as u32;
            let u32at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            for (g, lane) in lanes.iter_mut().enumerate() {
                let (ib, l) = (g >> 2, g & 3);
                let mut w = [0f32; 8];
                match t {
                    T_IQ1_M => {
                        let (s0, s1, s2, s3) = (u16at(48), u16at(50), u16at(52), u16at(54));
                        let d = h2f(((s0 >> 12) | ((s1 >> 8) & 0xf0) | ((s2 >> 4) & 0xf00) | (s3 & 0xf000)) as u16);
                        let sc = u16at(48 + 2 * (ib >> 1));
                        let dl = d * (2 * ((sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1) as f32;
                        let qhb = (b[32 + 2 * ib + (l >> 1)] as u32) >> (4 * (l & 1));
                        let delta = if qhb & 8 != 0 { -0.125f32 } else { 0.125 };
                        let e = IQ1S_GRID_GPU[(b[4 * ib + l] as usize) | (((qhb & 7) as usize) << 8)];
                        for (j, wj) in w.iter_mut().enumerate() {
                            *wj = dl * ((((e >> if j < 4 { 8 * j } else { 8 * (j - 4) + 4 }) & 0xF) as i32 - 1) as f32 + delta);
                        }
                    }
                    T_IQ2_XXS => {
                        let d = h2f(u16at(0) as u16);
                        let (a0, a1) = (u32at(2 + 8 * ib), u32at(6 + 8 * ib));
                        let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                        let ge = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                        let signs = ksign(a1 >> (7 * l));
                        for (j, wj) in w.iter_mut().enumerate() {
                            *wj = db * ((ge >> (8 * j)) & 255) as f32 * if (signs >> j) & 1 != 0 { -1.0 } else { 1.0 };
                        }
                    }
                    _ => {
                        let d = h2f(u16at(0) as u16);
                        let idx = u16at(2 + 8 * ib + 2 * l) as usize;
                        let aux = u32at(66 + 4 * ib);
                        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                        let signs = ksign(aux >> (7 * l));
                        let (g1, g2) = (IQ3XXS_GRID[idx & 255], IQ3XXS_GRID[idx >> 8]);
                        for (j, wj) in w.iter_mut().enumerate() {
                            let gv = if j < 4 { (g1 >> (8 * j)) & 255 } else { (g2 >> (8 * (j - 4))) & 255 };
                            *wj = db * gv as f32 * if (signs >> j) & 1 != 0 { -1.0 } else { 1.0 };
                        }
                    }
                }
                let xs = &x[k * 256 + g * 8..];
                *lane += (0..8).map(|j| w[j] * xs[j]).sum::<f32>();
            }
        }
        lanes.iter().sum()
    }

    fn blocks(t: i32, n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut v: Vec<u8> = (0..n * sb_bytes(t) + 64)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        for k in 0..n {
            let b = k * sb_bytes(t);
            if t == T_IQ1_M {
                // the d of the four scale u16s' top nibbles: ~2^-4
                let d: u16 = 0x2c00 | (k as u16 & 0x1ff);
                for i in 0..4 {
                    v[b + 49 + 2 * i] = (v[b + 49 + 2 * i] & 0x0f) | ((((d >> (4 * i)) & 0xf) as u8) << 4);
                }
            } else {
                let d: u16 = 0x2400 | (k as u16 & 0x1ff);
                v[b..b + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        v
    }

    /// back-to-back jobs into buffers that are freed after each: every part runs once, and only on its own job's buffer
    #[test]
    fn pool_jobs_never_mix() {
        let pool = Pool::new(8);
        for round in 0..3000usize {
            let parts = 1 + round % 37;
            let mut v = vec![0u32; parts];
            let p = P(v.as_mut_ptr() as usize);
            pool.run(parts, Arc::new(move |i| {
                let p = p;
                // SAFETY: i < parts, v outlives run.
                unsafe { *(p.0 as *mut u32).add(i) += 1 + round as u32 };
            }));
            assert!(v.iter().all(|&x| x == 1 + round as u32), "round {round}: {v:?}");
        }
    }

    #[test]
    fn products_match_the_kernels_formulas() {
        if !host_ok() {
            return;
        }
        for t in [T_IQ1_M, T_IQ2_XXS, T_IQ3_XXS] {
            let cols = if t == T_IQ3_XXS { 3072 } else { 6144 };
            let nb = cols / 256;
            let rows = 24;
            let m = blocks(t, nb * rows, 0x9e37_79b9_7f4a_7c15 ^ t as u64);
            let x: Vec<f32> = (0..cols).map(|i| ((i * 2654435761) % 1000) as f32 * 0.004 - 2.0).collect();
            let (mut e, mut n) = (0f64, 0f64);
            for r in 0..rows {
                let row = &m[r * nb * sb_bytes(t)..];
                let a = reference(t, row, &x);
                // SAFETY: the row and x as sized; AVX-512 checked above.
                let b = unsafe { row_dot(t, row.as_ptr(), nb, x.as_ptr()) };
                e += ((a - b) as f64).powi(2);
                n += (a as f64).powi(2);
            }
            let rel = (e / n).sqrt();
            assert!(rel < 2e-6, "type {t}: relative error {rel:e}");
        }
    }
}
