//! The CPU's share of a layer's experts: the ones in pinned host memory (and those just read from the file into it)
//! are multiplied where they are, by the host's cores, instead of being copied over PCIe to the GPU - its link is the
//! store's bottleneck (an expert swap moves 8.6 MB up and its victim 8.6 MB down at ~28 GB/s together), the host's
//! memory is not. The products are the GPU kernels' (mmvq.cpp): the activations quantized to int8 a block of 32 (a
//! scale amax / 127 rounded to fp16, each 32's sum), IQ1_S's grid entries + 1 as unsigned bytes against them, the
//! delta - 1 against the sums; AVX-512 VNNI where the CPU has it (vpdpbusd: 64 weights an instruction), plain code
//! otherwise.

use std::sync::{Arc, Condvar, Mutex};

use crate::iq1s_grid::IQ1S_GRID_GPU;

const QK: usize = 256;
const BLOCK: usize = 50;
const IQ1S_DELTA: f32 = 0.125;

/// The grid entries + 1 as eight unsigned bytes (values 0..7 in order)
fn grid_bytes() -> &'static [u64; 2048] {
    static T: std::sync::OnceLock<Box<[u64; 2048]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = Box::new([0u64; 2048]);
        for (i, e) in IQ1S_GRID_GPU.iter().enumerate() {
            let lo = (*e & 0x0f0f_0f0f) as u64;
            let hi = ((*e >> 4) & 0x0f0f_0f0f) as u64;
            t[i] = lo | (hi << 32);
        }
        t
    })
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE half bits (round to nearest even)
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | r as u16
}

fn f32_to_f16_round(x: f32) -> f32 {
    f16_to_f32(f16_bits(x))
}

/// Activations quantized a block of 32: int8 values, each block's scale and the sum of its values
pub struct Q8 {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub s: Vec<i32>,
}

/// x -> Q8 (the GPU's ns_ds2_quantize_q8: amax / 127 rounded to fp16, values rounded half away... as sycl::round)
pub fn quantize(x: &[f32]) -> Q8 {
    let nb = x.len() / 32;
    let mut q = vec![0i8; x.len()];
    let mut d = vec![0f32; nb];
    let mut s = vec![0i32; nb];
    for b in 0..nb {
        let xb = &x[b * 32..(b + 1) * 32];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dd = f32_to_f16_round(amax / 127.0);
        let mut sum = 0i32;
        for (i, v) in xb.iter().enumerate() {
            let qv = if dd > 0.0 { (v / dd).round().clamp(-127.0, 127.0) as i8 } else { 0 };
            q[b * 32 + i] = qv;
            sum += qv as i32;
        }
        d[b] = dd;
        s[b] = sum;
    }
    Q8 { q, d, s }
}

/// One IQ1_S row (cols / 256 blocks) against x: plain code
fn dot_scalar(row: &[u8], x: &Q8) -> f32 {
    let g = grid_bytes();
    let mut acc = 0f32;
    for (blk, b) in row.chunks_exact(BLOCK).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for ib in 0..8 {
            let qh = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as u32;
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = -1.0 + if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            let xb = blk * 8 + ib;
            let mut sumi = 0i32;
            for l in 0..4 {
                let e = g[(b[2 + 4 * ib + l] as u32 | (((qh >> (3 * l)) & 7) << 8)) as usize];
                for j in 0..8 {
                    sumi += ((e >> (8 * j)) & 0xff) as i32 * x.q[xb * 32 + 8 * l + j] as i32;
                }
            }
            acc += dl * x.d[xb] * (sumi as f32 + delta * x.s[xb] as f32);
        }
    }
    acc
}

/// The same with AVX-512 VNNI: two sub-blocks (64 weights) an instruction, their 8 grid entries gathered
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_avx512(row: &[u8], x: &Q8) -> f32 {
    use std::arch::x86_64::*;
    let g = grid_bytes().as_ptr() as *const i64;
    let mut facc = _mm512_setzero_ps();
    let mut extra = 0f32;
    let nblk = row.len() / BLOCK;
    for blk in 0..nblk {
        let b = &row[blk * BLOCK..(blk + 1) * BLOCK];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for pair in 0..4 {
            let mut idx = [0i64; 8];
            let mut sc = [0f32; 2];
            for k in 0..2 {
                let ib = 2 * pair + k;
                let qh = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as u32;
                for l in 0..4 {
                    idx[4 * k + l] = (b[2 + 4 * ib + l] as u32 | (((qh >> (3 * l)) & 7) << 8)) as i64;
                }
                let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                let delta = -1.0 + if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                let xb = blk * 8 + ib;
                sc[k] = dl * x.d[xb];
                extra += sc[k] * delta * x.s[xb] as f32;
            }
            let vi = _mm512_loadu_si512(idx.as_ptr() as *const _);
            let gv = _mm512_i64gather_epi64::<8>(vi, g as *const _);
            let xv = _mm512_loadu_si512(x.q.as_ptr().add((blk * 8 + 2 * pair) * 32) as *const _);
            let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), gv, xv);
            let s = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sc[0]), _mm512_set1_ps(sc[1]));
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), s, facc);
        }
    }
    _mm512_reduce_add_ps(facc) + extra
}

/// NS_DS2_CPU_V1=1: the first AVX-512 product (indices and scales a sub-block at a time) - for comparisons
fn v1() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS2_CPU_V1").is_ok_and(|v| v == "1"))
}

/// Whether the products run in AVX-512 VNNI (NS_DS2_CPU_AVX512=0: plain code)
pub fn avx512_on() -> bool {
    avx512()
}

/// The same with the block's 32 grid indices and 8 scales computed in vectors (one block = 4 gathers, 4 dpbusd)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn dot_avx512_v2(row: &[u8], x: &Q8) -> f32 {
    use std::arch::x86_64::*;
    let g = grid_bytes().as_ptr() as *const i64;
    let mut facc = _mm512_setzero_ps();
    let mut eacc = _mm256_setzero_ps();
    // element i of the 32 indices: qh word i / 4, shifted by 3 (i % 4)
    let qh_of: [i16; 32] = std::array::from_fn(|i| (i / 4) as i16);
    let shift: [i16; 32] = std::array::from_fn(|i| (3 * (i % 4)) as i16);
    let qh_of = _mm512_loadu_si512(qh_of.as_ptr() as *const _);
    let shift = _mm512_loadu_si512(shift.as_ptr() as *const _);
    let seven = _mm512_set1_epi16(7);
    // lanes 0-7 the pair's first sub-block's scale, 8-15 its second's
    let pair_idx: [__m512i; 4] = std::array::from_fn(|p| _mm512_setr_epi32((2 * p) as i32, (2 * p) as i32, (2 * p) as i32, (2 * p) as i32, (2 * p) as i32,
                                                                            (2 * p) as i32, (2 * p) as i32, (2 * p) as i32, (2 * p + 1) as i32, (2 * p + 1) as i32,
                                                                            (2 * p + 1) as i32, (2 * p + 1) as i32, (2 * p + 1) as i32, (2 * p + 1) as i32,
                                                                            (2 * p + 1) as i32, (2 * p + 1) as i32));
    let nblk = row.len() / BLOCK;
    for blk in 0..nblk {
        let b = row.as_ptr().add(blk * BLOCK);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let qs = _mm512_cvtepu8_epi16(_mm256_loadu_si256(b.add(2) as *const _));
        let qh8 = _mm_loadu_si128(b.add(34) as *const _);
        let qh = _mm512_permutexvar_epi16(qh_of, _mm512_castsi128_si512(qh8));
        let idx = _mm512_or_si512(qs, _mm512_slli_epi16::<8>(_mm512_and_si512(_mm512_srlv_epi16(qh, shift), seven)));
        // the 8 sub-blocks' scales: d (2 ((qh >> 12) & 7) + 1) times the activations' scales; the deltas - 1 against
        // the activations' sums
        let q32 = _mm256_cvtepu16_epi32(qh8);
        let sc_i = _mm256_add_epi32(_mm256_slli_epi32::<1>(_mm256_and_si256(_mm256_srli_epi32::<12>(q32), _mm256_set1_epi32(7))), _mm256_set1_epi32(1));
        let xd = _mm256_loadu_ps(x.d.as_ptr().add(blk * 8));
        let sc8 = _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sc_i), _mm256_set1_ps(d)), xd);
        let neg = _mm256_cmpneq_epi32_mask(_mm256_and_si256(q32, _mm256_set1_epi32(0x8000)), _mm256_setzero_si256());
        let delta = _mm256_mask_blend_ps(neg, _mm256_set1_ps(-1.0 + IQ1S_DELTA), _mm256_set1_ps(-1.0 - IQ1S_DELTA));
        let xs = _mm256_cvtepi32_ps(_mm256_loadu_si256(x.s.as_ptr().add(blk * 8) as *const _));
        eacc = _mm256_fmadd_ps(_mm256_mul_ps(sc8, delta), xs, eacc);
        let sc16 = _mm512_castps256_ps512(sc8);
        for (p, pidx) in pair_idx.iter().enumerate() {
            let iv = _mm512_cvtepu16_epi64(match p {
                0 => _mm512_extracti32x4_epi32::<0>(idx),
                1 => _mm512_extracti32x4_epi32::<1>(idx),
                2 => _mm512_extracti32x4_epi32::<2>(idx),
                _ => _mm512_extracti32x4_epi32::<3>(idx),
            });
            let gv = _mm512_i64gather_epi64::<8>(iv, g as *const _);
            let xv = _mm512_loadu_si512(x.q.as_ptr().add((blk * 8 + 2 * p) * 32) as *const _);
            let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), gv, xv);
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), _mm512_permutexvar_ps(*pidx, sc16), facc);
        }
    }
    let e4 = _mm_add_ps(_mm256_castps256_ps128(eacc), _mm256_extractf128_ps::<1>(eacc));
    let e2 = _mm_add_ps(e4, _mm_movehl_ps(e4, e4));
    let e1 = _mm_add_ss(e2, _mm_shuffle_ps::<1>(e2, e2));
    _mm512_reduce_add_ps(facc) + _mm_cvtss_f32(e1)
}

fn avx512() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[cfg(target_arch = "x86_64")]
    {
        *V.get_or_init(|| {
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vnni")
                && std::is_x86_feature_detected!("avx512vl")
                && std::env::var("NS_DS2_CPU_AVX512").map_or(true, |v| v != "0")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        *V.get_or_init(|| false)
    }
}

/// y[r] = row r of the IQ1_S matrix w (rows of cols) . x, for rows in `rows`
pub fn matvec(w: &[u8], cols: usize, x: &Q8, rows: std::ops::Range<usize>, y: &mut [f32]) {
    let rb = cols / QK * BLOCK;
    for (k, r) in rows.enumerate() {
        let row = &w[r * rb..(r + 1) * rb];
        #[cfg(target_arch = "x86_64")]
        if avx512() {
            // SAFETY: the features are checked (avx512); the row and x span cols values (the caller's matrix).
            y[k] = unsafe { if v1() { dot_avx512(row, x) } else { dot_avx512_v2(row, x) } };
            continue;
        }
        y[k] = dot_scalar(row, x);
    }
}

/// A fixed set of worker threads that run one job at a time over a range of parts (`run`), the caller's thread working
/// too. The parts are handed out by one atomic counter that also holds the job's generation, so a worker still holding
/// an earlier job's closure can never take a part of the next one (it would run the old closure on freed buffers).
pub struct Pool {
    shared: Arc<Shared>,
    pub threads: usize,
}

type JobFn = Arc<dyn Fn(usize) + Send + Sync>;

struct Shared {
    /// (the job, its generation, its parts, stop)
    m: Mutex<(Option<JobFn>, u64, usize, bool)>,
    work: Condvar,
    /// generation << 32 | the next part
    ctr: std::sync::atomic::AtomicU64,
    left: std::sync::atomic::AtomicUsize,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared { m: Mutex::new((None, 0, 0, false)), work: Condvar::new(), ctr: 0.into(), left: 0.into() });
        for _ in 0..threads.saturating_sub(1) {
            let sh = shared.clone();
            std::thread::spawn(move || {
                let mut seen = 0u64;
                loop {
                    let (job, gen, parts) = {
                        let mut g = sh.m.lock().unwrap();
                        loop {
                            if g.3 {
                                return;
                            }
                            if g.1 != seen {
                                seen = g.1;
                                if let Some(j) = g.0.clone() {
                                    break (j, g.1, g.2);
                                }
                            }
                            g = sh.work.wait(g).unwrap();
                        }
                    };
                    Self::work(&sh, &job, gen, parts);
                }
            });
        }
        Pool { shared, threads }
    }

    fn work(sh: &Shared, job: &JobFn, gen: u64, parts: usize) {
        use std::sync::atomic::Ordering::*;
        loop {
            let cur = sh.ctr.load(Acquire);
            if cur >> 32 != gen & 0xffff_ffff {
                return;
            }
            let i = (cur & 0xffff_ffff) as usize;
            if i >= parts {
                return;
            }
            if sh.ctr.compare_exchange_weak(cur, cur + 1, AcqRel, Acquire).is_err() {
                continue;
            }
            // a part that panics still counts as done (else `run` would wait forever); the panic is reported
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(i))).is_err() {
                eprintln!("[cpu: a part of the experts' products panicked]");
            }
            sh.left.fetch_sub(1, AcqRel);
        }
    }

    /// f(0..parts) over the workers and this thread; returns when every part is done
    pub fn run(&self, parts: usize, f: JobFn) {
        use std::sync::atomic::Ordering::*;
        if parts == 0 {
            return;
        }
        let sh = &*self.shared;
        let gen = {
            let mut g = sh.m.lock().unwrap();
            g.1 += 1;
            g.0 = Some(f.clone());
            g.2 = parts;
            sh.left.store(parts, Release);
            sh.ctr.store((g.1 & 0xffff_ffff) << 32, Release);
            g.1
        };
        sh.work.notify_all();
        Self::work(sh, &f, gen, parts);
        while sh.left.load(Acquire) > 0 {
            std::hint::spin_loop();
        }
        sh.m.lock().unwrap().0 = None;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.m.lock().unwrap().3 = true;
        self.shared.work.notify_all();
    }
}

/// One entry of a layer's CPU experts: its slot's bytes [gate | up | down] and its token's activations
pub struct Job<'a> {
    pub slot: &'a [u8],
    pub x: &'a Q8,
}

/// A raw pointer the pool's parts write their own disjoint ranges through
#[derive(Clone, Copy)]
struct Out(*mut f32);
// SAFETY: every part writes a range no other part writes; the buffer outlives `run` (it waits for all parts).
unsafe impl Send for Out {}
unsafe impl Sync for Out {}
impl Out {
    /// n floats from float i (the closure takes the whole wrapper this way, not its pointer field)
    /// SAFETY: the caller's range is inside the buffer and written by it alone
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize, n: usize) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(i), n)
    }
}

/// Runs the entries on the pool: gate | up rows split over the workers, then the down rows. All IQ1_S. The output rows
/// [entries][d]
pub fn experts(pool: &Pool, jobs: &[Job], d: usize, ff: usize, down_at: usize) -> Vec<Vec<f32>> {
    const ROWS: usize = 64;
    let n = jobs.len();
    if n == 0 {
        return Vec::new();
    }
    // the slots and activations outlive the pool's runs (each waits): their addresses go to the workers
    let slots: Arc<Vec<(usize, usize, usize)>> = Arc::new(jobs.iter().map(|j| (j.slot.as_ptr() as usize, j.slot.len(), j.x as *const Q8 as usize)).collect());
    let mut gu = vec![0f32; n * 2 * ff];
    let chunks = 2 * ff / ROWS;
    {
        let (slots, out) = (slots.clone(), Out(gu.as_mut_ptr()));
        pool.run(n * chunks, Arc::new(move |i| {
            let (e, c) = (i / chunks, i % chunks);
            let (p, len, xp) = slots[e];
            // SAFETY: the caller's slot and activations, alive until `run` returns; rows [e][c * ROWS..] are this part's
            let (w, x, y) = unsafe {
                (std::slice::from_raw_parts(p as *const u8, len), &*(xp as *const Q8), out.at(e * 2 * ff + c * ROWS, ROWS))
            };
            matvec(w, d, x, c * ROWS..(c + 1) * ROWS, y);
        }));
    }
    let hq: Arc<Vec<Q8>> = Arc::new(gu.chunks(2 * ff).map(|v| {
        let h: Vec<f32> = (0..ff).map(|i| {
            let a = v[i];
            a / (1.0 + (-a).exp()) * v[ff + i]
        }).collect();
        quantize(&h)
    }).collect());
    let mut out = vec![0f32; n * d];
    let dchunks = d / ROWS;
    {
        let (slots, o) = (slots.clone(), Out(out.as_mut_ptr()));
        pool.run(n * dchunks, Arc::new(move |i| {
            let (e, c) = (i / dchunks, i % dchunks);
            let (p, len, _) = slots[e];
            // SAFETY: as above
            let (w, y) = unsafe {
                (std::slice::from_raw_parts((p + down_at) as *const u8, len - down_at), o.at(e * d + c * ROWS, ROWS))
            };
            matvec(w, ff, &hq[e], c * ROWS..(c + 1) * ROWS, y);
        }));
    }
    out.chunks(d).map(|c| c.to_vec()).collect()
}
