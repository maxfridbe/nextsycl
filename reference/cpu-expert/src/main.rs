//! One expert for one token on the CPU: gate | up (IQ2_XXS, 2048 x 4096 each), SwiGLU, down (Q2_K, 4096 x 2048) -
//! the activations quantized to int8 per 256 (q8_K), ggml's AVX2 dot products ported to Rust; against float.
mod grid;
use grid::IQ2XXS_GRID;
use std::arch::x86_64::*;
use std::time::Instant;

const QK: usize = 256;
#[derive(Clone, Copy)]
#[repr(C)]
struct BlockIq2 { d: u16, qs: [u16; 32] }          // 66 bytes
#[derive(Clone, Copy)]
#[repr(C)]
struct BlockQ2k { scales: [u8; 16], qs: [u8; 64], d: u16, dmin: u16 }   // 84 bytes
struct Q8 { d: Vec<f32>, qs: Vec<i8>, bsums: Vec<i16> }   // per 256: scale, values, sums of 16

fn f16(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 { if m == 0 { s } else { let mut e2 = 127 - 15 + 1; let mut m2 = m; while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; } s | (e2 << 23) | ((m2 & 0x3ff) << 13) } }
               else if e == 31 { s | 0x7f800000 | (m << 13) } else { s | ((e + 127 - 15) << 23) | (m << 13) };
    f32::from_bits(bits)
}
fn to_f16(f: f32) -> u16 {
    let b = f.to_bits(); let s = ((b >> 16) & 0x8000) as u16; let e = ((b >> 23) & 0xff) as i32 - 127 + 15; let m = (b >> 13) & 0x3ff;
    if e <= 0 { s } else if e >= 31 { s | 0x7c00 } else { s | ((e as u16) << 10) | m as u16 }
}

fn quantize(x: &[f32]) -> Q8 {
    let nb = x.len() / QK;
    let (mut d, mut qs, mut bsums) = (vec![0f32; nb], vec![0i8; x.len()], vec![0i16; x.len() / 16]);
    for b in 0..nb {
        let xs = &x[b * QK..(b + 1) * QK];
        let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        d[b] = amax / 127.0;
        for i in 0..QK { qs[b * QK + i] = (xs[i] * id).round().clamp(-127.0, 127.0) as i8; }
        for g in 0..16 { bsums[b * 16 + g] = qs[b * QK + g * 16..b * QK + g * 16 + 16].iter().map(|&v| v as i16).sum(); }
    }
    Q8 { d, qs, bsums }
}

fn signs_table() -> [u64; 128] {
    let mut t = [0u64; 128];
    for i in 0..128u32 {
        let s = i | ((i.count_ones() & 1) << 7);
        let mut v = 0u64;
        for k in 0..8 { v |= (if s >> k & 1 == 1 { 0xffu64 } else { 0x01 }) << (8 * k); }
        t[i as usize] = v;
    }
    t
}

#[inline(always)]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

/// ggml_vec_dot_iq2_xxs_q8_K (AVX2)
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq2(row: &[BlockIq2], y: &Q8, signs: &[u64; 128]) -> f32 {
    let mut acc = 0f32;
    for (i, b) in row.iter().enumerate() {
        let d = f16(b.d) * y.d[i];
        let q2 = b.qs.as_ptr() as *const u32;
        let q8 = y.qs.as_ptr().add(i * QK);
        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        for ib32 in (0..8).step_by(2) {
            let a = [*q2.add(2 * ib32), *q2.add(2 * ib32 + 1), *q2.add(2 * ib32 + 2), *q2.add(2 * ib32 + 3)];
            let a8 = a.as_ptr() as *const u8;
            let q8_1 = _mm256_loadu_si256(q8.add(32 * ib32) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8.add(32 * ib32 + 32) as *const __m256i);
            let g = |k: usize| IQ2XXS_GRID[*a8.add(k) as usize] as i64;
            let q2_1 = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
            let q2_2 = _mm256_set_epi64x(g(11), g(10), g(9), g(8));
            let s = |w: u32, sh: u32| signs[((w >> sh) & 127) as usize] as i64;
            let s2_1 = _mm256_set_epi64x(s(a[1], 21), s(a[1], 14), s(a[1], 7), s(a[1], 0));
            let s2_2 = _mm256_set_epi64x(s(a[3], 21), s(a[3], 14), s(a[3], 7), s(a[3], 0));
            let q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            let q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            let ls1 = (a[1] >> 28) as i16;
            let ls2 = (a[3] >> 28) as i16;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1)));
        }
        acc += d * hsum_i32(_mm256_add_epi32(sumi1, sumi2)) as f32;
    }
    0.125 * acc
}

/// ggml_vec_dot_q2_K_q8_K (AVX2)
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q2k(row: &[BlockQ2k], y: &Q8) -> f32 {
    let m3 = _mm256_set1_epi8(3);
    let mut acc = 0f32;
    for (i, b) in row.iter().enumerate() {
        let d = y.d[i] * f16(b.d);
        let dmin = -y.d[i] * f16(b.dmin);
        // the mins against the activations' sums of 16
        let mut summ = 0i32;
        for k in 0..16 { summ += (b.scales[k] >> 4) as i32 * y.bsums[i * 16 + k] as i32; }
        let mut sumi = _mm256_setzero_si256();
        let mut q2 = b.qs.as_ptr();
        let mut q8 = y.qs.as_ptr().add(i * QK);
        for j in 0..2 {
            let sc = &b.scales[8 * j..8 * j + 8];
            let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
            q2 = q2.add(32);
            for shift in 0..4 {
                let qv = _mm256_and_si256(_mm256_srli_epi16(q2bits, 0), m3); // placeholder overwritten below
                let qv = match shift {
                    0 => _mm256_and_si256(q2bits, m3),
                    1 => _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3),
                    2 => _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3),
                    _ => _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3),
                };
                let _ = qv;
                let q8v = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p = _mm256_maddubs_epi16(qv, q8v);   // 16 pairs of the two 16-value halves
                // the halves' scales: values 0-15 sub-block 2 shift, 16-31 the next
                let s_lo = (sc[2 * shift] & 15) as i16;
                let s_hi = (sc[2 * shift + 1] & 15) as i16;
                let scales = _mm256_set_epi16(s_hi, s_hi, s_hi, s_hi, s_hi, s_hi, s_hi, s_hi, s_lo, s_lo, s_lo, s_lo, s_lo, s_lo, s_lo, s_lo);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, scales));
            }
        }
        acc += d * hsum_i32(sumi) as f32 + dmin * summ as f32;
    }
    acc
}

fn deq_iq2(b: &BlockIq2) -> [f32; 256] {
    let mut out = [0f32; 256];
    for ib in 0..8 {
        let q2 = &b.qs[4 * ib..4 * ib + 4];
        let aux32 = q2[2] as u32 | (q2[3] as u32) << 16;
        let d = f16(b.d) * (0.5 + (aux32 >> 28) as f32) * 0.25;
        for il in 0..4 {
            let idx = (q2[il / 2] >> (8 * (il % 2))) & 0xff;
            let g = IQ2XXS_GRID[idx as usize];
            let s7 = (aux32 >> (7 * il)) & 127;
            let signs = s7 | ((s7.count_ones() & 1) << 7);
            for j in 0..8 { out[ib * 32 + il * 8 + j] = d * ((g >> (8 * j)) & 0xff) as f32 * if signs >> j & 1 == 1 { -1.0 } else { 1.0 }; }
        }
    }
    out
}
fn deq_q2k(b: &BlockQ2k) -> [f32; 256] {
    let mut out = [0f32; 256];
    for v in 0..256 {
        let (n, j, hf, l) = (v / 128, (v % 128) / 32, (v % 32) / 16, v % 16);
        let sc = b.scales[8 * n + 2 * j + hf];
        let q = (b.qs[32 * n + 16 * hf + l] >> (2 * j)) & 3;
        out[v] = f16(b.d) * (sc & 15) as f32 * q as f32 - f16(b.dmin) * (sc >> 4) as f32;
    }
    out
}


/// A spinning pool: workers wait on an epoch, run job(worker, n), count themselves done
struct Pool {
    epoch: std::sync::atomic::AtomicU64,
    done: std::sync::atomic::AtomicUsize,
    job: std::sync::Mutex<Option<std::sync::Arc<dyn Fn(usize, usize) + Send + Sync>>>,
    n: usize,
}
impl Pool {
    fn new(n: usize) -> std::sync::Arc<Pool> {
        let p = std::sync::Arc::new(Pool { epoch: 0.into(), done: 0.into(), job: std::sync::Mutex::new(None), n });
        for w in 1..n {
            let p = p.clone();
            std::thread::spawn(move || {
                let mut seen = 0;
                loop {
                    let e = p.epoch.load(std::sync::atomic::Ordering::Acquire);
                    if e == seen { std::hint::spin_loop(); continue; }
                    seen = e;
                    let j = p.job.lock().unwrap().clone().unwrap();
                    j(w, p.n);
                    p.done.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                }
            });
        }
        p
    }
    fn run(&self, job: std::sync::Arc<dyn Fn(usize, usize) + Send + Sync>) {
        *self.job.lock().unwrap() = Some(job.clone());
        self.done.store(0, std::sync::atomic::Ordering::Release);
        self.epoch.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        job(0, self.n);
        while self.done.load(std::sync::atomic::Ordering::Acquire) < self.n - 1 { std::hint::spin_loop(); }
    }
}
struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } fn f(&mut self) -> f32 { (self.next() % 20001) as f32 / 10000.0 - 1.0 } }

fn main() {
    let (d, f) = (4096usize, 2048usize);
    let mut r = Rng(0x9e3779b97f4a7c15);
    let mk_iq2 = |r: &mut Rng, n: usize| -> Vec<BlockIq2> { (0..n).map(|_| { let mut b = BlockIq2 { d: to_f16(0.002 + 0.008 * (r.f().abs())), qs: [0; 32] }; for q in b.qs.iter_mut() { *q = r.next() as u16; } b }).collect() };
    let gate = mk_iq2(&mut r, f * d / QK);
    let up = mk_iq2(&mut r, f * d / QK);
    let down: Vec<BlockQ2k> = (0..d * f / QK).map(|_| { let mut b = BlockQ2k { scales: [0; 16], qs: [0; 64], d: to_f16(0.002 + 0.008 * r.f().abs()), dmin: to_f16(0.002 + 0.008 * r.f().abs()) }; for s in b.scales.iter_mut() { *s = r.next() as u8; } for s in b.qs.iter_mut() { *s = r.next() as u8; } b }).collect();
    let x: Vec<f32> = (0..d).map(|_| r.f()).collect();
    let signs = signs_table();
    // one expert-token, `threads` sharing its rows
    let expert = |threads: usize| -> Vec<f32> {
        let xq = quantize(&x);
        let mut gu = vec![0f32; 2 * f];
        std::thread::scope(|sc| {
            for (ti, part) in gu.chunks_mut((2 * f).div_ceil(threads)).enumerate() {
                let (xq, gate, up, signs) = (&xq, &gate, &up, &signs);
                sc.spawn(move || {
                    let r0 = ti * (2 * f).div_ceil(threads);
                    for (k, o) in part.iter_mut().enumerate() {
                        let rr = r0 + k;
                        let (m, rr) = if rr < f { (gate, rr) } else { (up, rr - f) };
                        *o = unsafe { dot_iq2(&m[rr * d / QK..(rr + 1) * d / QK], xq, signs) };
                    }
                });
            }
        });
        let h: Vec<f32> = (0..f).map(|i| { let g = gu[i].min(10.0); g / (1.0 + (-g).exp()) * gu[f + i].clamp(-10.0, 10.0) }).collect();
        let hq = quantize(&h);
        let mut y = vec![0f32; d];
        std::thread::scope(|sc| {
            for (ti, part) in y.chunks_mut(d.div_ceil(threads)).enumerate() {
                let (hq, down) = (&hq, &down);
                sc.spawn(move || {
                    let r0 = ti * d.div_ceil(threads);
                    for (k, o) in part.iter_mut().enumerate() {
                        let rr = r0 + k;
                        *o = unsafe { dot_q2k(&down[rr * f / QK..(rr + 1) * f / QK], hq) };
                    }
                });
            }
        });
        y
    };
    // the float reference
    let dense = |m: &[BlockIq2], row: usize, v: &[f32]| -> f32 { m[row * d / QK..(row + 1) * d / QK].iter().enumerate().map(|(b, bl)| deq_iq2(bl).iter().zip(&v[b * QK..]).map(|(a, c)| a * c).sum::<f32>()).sum() };
    let gr: Vec<f32> = (0..f).map(|i| dense(&gate, i, &x)).collect();
    let ur: Vec<f32> = (0..f).map(|i| dense(&up, i, &x)).collect();
    let hr: Vec<f32> = (0..f).map(|i| { let g = gr[i].min(10.0); g / (1.0 + (-g).exp()) * ur[i].clamp(-10.0, 10.0) }).collect();
    let yr: Vec<f32> = (0..d).map(|i| down[i * f / QK..(i + 1) * f / QK].iter().enumerate().map(|(b, bl)| deq_q2k(bl).iter().zip(&hr[b * QK..]).map(|(a, c)| a * c).sum::<f32>()).sum()).collect();
    let y1 = expert(1);
    let (mut md, mut mx) = (0f32, 0f32);
    for (a, b) in y1.iter().zip(&yr) { md = md.max((a - b).abs()); mx = mx.max(b.abs()); }
    let cos = { let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64); for (a, b) in y1.iter().zip(&yr) { ab += (*a * *b) as f64; aa += (*a * *a) as f64; bb += (*b * *b) as f64; } ab / (aa.sqrt() * bb.sqrt()) };
    println!("against float: cosine {cos:.6}, max |diff| {md:.3e} of {mx:.3e}");
    for threads in [1usize, 4, 8, 16, 32] {
        for _ in 0..3 { expert(threads); }
        let t0 = Instant::now();
        let n = 50;
        for _ in 0..n { std::hint::black_box(expert(threads)); }
        println!("{threads:>2} threads: {:7.1} us an expert-token", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
    }

    // the pool, the weights cold (32 experts rotated: 218 MB, past the L3), 1-3 rows (a verify pass)
    const NE: usize = 32;
    let gates: std::sync::Arc<Vec<Vec<BlockIq2>>> = std::sync::Arc::new((0..NE).map(|_| mk_iq2(&mut r, f * d / QK)).collect());
    let ups: std::sync::Arc<Vec<Vec<BlockIq2>>> = std::sync::Arc::new((0..NE).map(|_| mk_iq2(&mut r, f * d / QK)).collect());
    let downs: std::sync::Arc<Vec<Vec<BlockQ2k>>> = std::sync::Arc::new((0..NE).map(|_| (0..d * f / QK).map(|_| { let mut b = BlockQ2k { scales: [0; 16], qs: [0; 64], d: to_f16(0.004), dmin: to_f16(0.004) }; for s in b.scales.iter_mut() { *s = r.next() as u8; } for s in b.qs.iter_mut() { *s = r.next() as u8; } b }).collect()).collect());
    let signs = std::sync::Arc::new(signs);
    for threads in [8usize, 12, 16] {
        let pool = Pool::new(threads);
        for rows in [1usize, 2, 3] {
            let xs: Vec<f32> = (0..rows * d).map(|_| r.f()).collect();
            let mut times = Vec::new();
            for it in 0..40 {
                let e = it % NE;
                let t0 = Instant::now();
                let xq: std::sync::Arc<Vec<Q8>> = std::sync::Arc::new((0..rows).map(|k| quantize(&xs[k * d..(k + 1) * d])).collect());
                let gu = std::sync::Arc::new((0..rows * 2 * f).map(|_| std::sync::atomic::AtomicU32::new(0)).collect::<Vec<_>>());
                {
                    let (xq, gu, g, u, sg) = (xq.clone(), gu.clone(), gates.clone(), ups.clone(), signs.clone());
                    pool.run(std::sync::Arc::new(move |w, n| {
                        let per = (2 * f).div_ceil(n);
                        for rr in w * per..((w + 1) * per).min(2 * f) {
                            let (m, ri) = if rr < f { (&g[e], rr) } else { (&u[e], rr - f) };
                            let wrow = &m[ri * d / QK..(ri + 1) * d / QK];
                            for k in 0..rows {
                                let v = unsafe { dot_iq2(wrow, &xq[k], &sg) };
                                gu[k * 2 * f + rr].store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }));
                }
                let hq: std::sync::Arc<Vec<Q8>> = std::sync::Arc::new((0..rows).map(|k| {
                    let h: Vec<f32> = (0..f).map(|i| { let gg = f32::from_bits(gu[k * 2 * f + i].load(std::sync::atomic::Ordering::Relaxed)).min(10.0); let uu = f32::from_bits(gu[k * 2 * f + f + i].load(std::sync::atomic::Ordering::Relaxed)); gg / (1.0 + (-gg).exp()) * uu.clamp(-10.0, 10.0) }).collect();
                    quantize(&h)
                }).collect());
                let y = std::sync::Arc::new((0..rows * d).map(|_| std::sync::atomic::AtomicU32::new(0)).collect::<Vec<_>>());
                {
                    let (hq, y, dn) = (hq.clone(), y.clone(), downs.clone());
                    pool.run(std::sync::Arc::new(move |w, n| {
                        let per = d.div_ceil(n);
                        for rr in w * per..((w + 1) * per).min(d) {
                            let wrow = &dn[e][rr * f / QK..(rr + 1) * f / QK];
                            for k in 0..rows {
                                let v = unsafe { dot_q2k(wrow, &hq[k]) };
                                y[k * d + rr].store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }));
                }
                if it >= 8 { times.push(t0.elapsed().as_secs_f64() * 1e6); }
            }
            times.sort_by(|a, b| a.total_cmp(b));
            println!("pool {threads:>2} threads, cold, {rows} row(s): median {:6.1} us, p90 {:6.1} us an expert", times[times.len() / 2], times[times.len() * 9 / 10]);
        }
    }
}
