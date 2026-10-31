//! Expert products on the host's cores, for what an expert in pinned memory would cost without its copy over PCIe: the silo's `gu` and
//! `down` (kernels/llm/minimaxm3/silo/moe.cpp) on float32 activations, group by group in the same formulas (ggml's dequantize_row_*).
//! NS_MM31_CPU_EXPERTS and NS_MM31_KBENCH=1 run them. Products are float, as the GPU's: only the order of the sum differs.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};

use crate::grids::{IQ1S_GRID_GPU, IQ2XXS_GRID, IQ3XXS_GRID};

pub const T_IQ2_XXS: i32 = 16;
pub const T_IQ3_XXS: i32 = 18;
pub const T_IQ4_XS: i32 = 23;
pub const T_IQ1_M: i32 = 29;

pub fn sb_bytes(t: i32) -> usize {
    match t {
        T_IQ2_XXS => 66,
        T_IQ3_XXS => 98,
        T_IQ4_XS => 136,
        T_IQ1_M => 56,
        _ => 0,
    }
}

fn h2f(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
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

#[repr(align(32))]
#[derive(Clone, Copy)]
struct V8([f32; 8]);

struct Tabs {
    /// IQ1_M: the entry - 1, values in order
    iq1: Vec<V8>,
    /// IQ2_XXS: the bytes as floats
    iq2: Vec<V8>,
    /// IQ3_XXS: four values an index
    iq3: Vec<[f32; 4]>,
    /// the sign pattern of an 8-bit code: -0.0 where the bit is set
    sign: Vec<V8>,
}

fn tabs() -> &'static Tabs {
    static T: std::sync::OnceLock<Tabs> = std::sync::OnceLock::new();
    T.get_or_init(|| Tabs {
        iq1: IQ1S_GRID_GPU.iter().map(|&e| V8(std::array::from_fn(|j| (((e >> if j < 4 { 8 * j } else { 8 * (j - 4) + 4 }) & 0xf) as i32 - 1) as f32))).collect(),
        iq2: IQ2XXS_GRID.iter().map(|&e| V8(std::array::from_fn(|j| ((e >> (8 * j)) & 255) as f32))).collect(),
        iq3: IQ3XXS_GRID.iter().map(|&e| std::array::from_fn(|j| ((e >> (8 * j)) & 255) as f32)).collect(),
        sign: (0..256u32).map(|s| V8(std::array::from_fn(|j| if (s >> j) & 1 != 0 { -0.0 } else { 0.0 }))).collect(),
    })
}

/// ggml's sign byte: seven bits and their parity
#[inline(always)]
fn ksign(x: u32) -> usize {
    let x = x & 127;
    (x | ((x.count_ones() & 1) << 7)) as usize
}

#[inline(always)]
unsafe fn rd16(p: *const u8) -> u32 {
    (p as *const u16).read_unaligned() as u32
}
#[inline(always)]
unsafe fn rd32(p: *const u8) -> u32 {
    (p as *const u32).read_unaligned()
}

const IQ4K: [f32; 16] = [-127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.];

/// row . x: a row of nb blocks of type t, x nb * 256 floats
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot(t: i32, row: *const u8, nb: usize, x: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let tb = tabs();
    let mut acc = _mm256_setzero_ps();
    match t {
        T_IQ1_M => {
            for k in 0..nb {
                let b = row.add(k * 56);
                let (s0, s1, s2, s3) = (rd16(b.add(48)), rd16(b.add(50)), rd16(b.add(52)), rd16(b.add(54)));
                let d = h2f(((s0 >> 12) | ((s1 >> 8) & 0xf0) | ((s2 >> 4) & 0xf00) | (s3 & 0xf000)) as u16);
                let sw = [s0, s1, s2, s3];
                for g in 0..32usize {
                    let (ib, l) = (g >> 2, g & 3);
                    let sc = sw[ib >> 1];
                    let dl = d * (2 * ((sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1) as f32;
                    let qhb = (*b.add(32 + 2 * ib + (l >> 1)) as u32) >> (4 * (l & 1));
                    let delta = if qhb & 8 != 0 { -0.125f32 } else { 0.125f32 };
                    let idx = *b.add(4 * ib + l) as usize | (((qhb & 7) as usize) << 8);
                    let v = _mm256_load_ps(tb.iq1.get_unchecked(idx).0.as_ptr());
                    let w = _mm256_mul_ps(_mm256_set1_ps(dl), _mm256_add_ps(v, _mm256_set1_ps(delta)));
                    acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x.add(k * 256 + g * 8)), acc);
                }
            }
        }
        T_IQ2_XXS => {
            for k in 0..nb {
                let b = row.add(k * 66);
                let d = h2f(rd16(b) as u16);
                for ib in 0..8usize {
                    let (a0, a1) = (rd32(b.add(2 + 8 * ib)), rd32(b.add(6 + 8 * ib)));
                    let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                    let dv = _mm256_set1_ps(db);
                    for l in 0..4usize {
                        let v = _mm256_load_ps(tb.iq2.get_unchecked(((a0 >> (8 * l)) & 255) as usize).0.as_ptr());
                        let s = _mm256_load_ps(tb.sign.get_unchecked(ksign(a1 >> (7 * l))).0.as_ptr());
                        acc = _mm256_fmadd_ps(_mm256_xor_ps(_mm256_mul_ps(dv, v), s), _mm256_loadu_ps(x.add(k * 256 + (4 * ib + l) * 8)), acc);
                    }
                }
            }
        }
        T_IQ3_XXS => {
            for k in 0..nb {
                let b = row.add(k * 98);
                let d = h2f(rd16(b) as u16);
                for ib in 0..8usize {
                    let aux = rd32(b.add(66 + 4 * ib));
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    let dv = _mm256_set1_ps(db);
                    for l in 0..4usize {
                        let idx = rd16(b.add(2 + 8 * ib + 2 * l)) as usize;
                        let lo = _mm_loadu_ps(tb.iq3.get_unchecked(idx & 255).as_ptr());
                        let hi = _mm_loadu_ps(tb.iq3.get_unchecked(idx >> 8).as_ptr());
                        let v = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo), hi);
                        let s = _mm256_load_ps(tb.sign.get_unchecked(ksign(aux >> (7 * l))).0.as_ptr());
                        acc = _mm256_fmadd_ps(_mm256_xor_ps(_mm256_mul_ps(dv, v), s), _mm256_loadu_ps(x.add(k * 256 + (4 * ib + l) * 8)), acc);
                    }
                }
            }
        }
        _ => {
            // IQ4_XS
            for k in 0..nb {
                let b = row.add(k * 136);
                let d = h2f(rd16(b) as u16);
                let sh = rd16(b.add(2));
                for ib in 0..8usize {
                    let nib = (*b.add(4 + (ib >> 1)) as u32 >> (4 * (ib & 1))) & 0xf;
                    let ls = nib | (((sh >> (2 * ib)) & 3) << 4);
                    let dl = d * (ls as i32 - 32) as f32;
                    let dv = _mm256_set1_ps(dl);
                    for q in 0..4usize {
                        let qq = b.add(8 + 16 * ib + 8 * (q & 1));
                        let sft = 4 * (q >> 1);
                        let w: [f32; 8] = std::array::from_fn(|j| IQ4K[((*qq.add(j) as usize) >> sft) & 15]);
                        let v = _mm256_loadu_ps(w.as_ptr());
                        acc = _mm256_fmadd_ps(_mm256_mul_ps(dv, v), _mm256_loadu_ps(x.add(k * 256 + (4 * ib + q) * 8)), acc);
                    }
                }
            }
        }
    }
    let lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
    let a = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    _mm_cvtss_f32(_mm_add_ss(a, _mm_shuffle_ps::<1>(a, a)))
}

/// A fixed set of workers that spin between jobs (a layer's product is under a millisecond: a sleeping thread's wake-up would cost a good part of it), the caller working too
pub struct Pool {
    sh: Arc<Sh>,
    pub threads: usize,
}

type JobFn = Arc<dyn Fn(usize) + Send + Sync>;

struct Sh {
    m: Mutex<Option<JobFn>>,
    parts: AtomicUsize,
    /// generation << 32 | the next part
    ctr: AtomicU64,
    left: AtomicUsize,
    stop: AtomicUsize,
    _c: Condvar,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let sh = Arc::new(Sh { m: Mutex::new(None), parts: 0.into(), ctr: 0.into(), left: 0.into(), stop: 0.into(), _c: Condvar::new() });
        for _ in 1..threads {
            let s = sh.clone();
            std::thread::spawn(move || {
                let mut seen = 0u64;
                let mut idle = std::time::Instant::now();
                let mut n = 0u32;
                while s.stop.load(Acquire) == 0 {
                    let gen = s.ctr.load(Acquire) >> 32;
                    if gen == seen {
                        std::hint::spin_loop();
                        n = n.wrapping_add(1);
                        if n & 0x3ff == 0 && idle.elapsed().as_millis() >= 5 {
                            std::thread::sleep(std::time::Duration::from_micros(200));
                        }
                        continue;
                    }
                    seen = gen;
                    idle = std::time::Instant::now();
                    let job = s.m.lock().unwrap().clone();
                    if let Some(j) = job {
                        Self::work(&s, &j, gen);
                    }
                }
            });
        }
        Pool { sh, threads }
    }

    fn work(s: &Sh, j: &JobFn, gen: u64) {
        let parts = s.parts.load(Acquire);
        loop {
            let cur = s.ctr.load(Acquire);
            if cur >> 32 != gen {
                return;
            }
            let i = (cur & 0xffff_ffff) as usize;
            if i >= parts {
                return;
            }
            if s.ctr.compare_exchange_weak(cur, cur + 1, AcqRel, Acquire).is_ok() {
                j(i);
                s.left.fetch_sub(1, AcqRel);
            }
        }
    }

    /// f(0..parts) over the workers and this thread, back when every part is done
    pub fn run(&self, parts: usize, f: JobFn) {
        let s = &*self.sh;
        let gen = (s.ctr.load(Acquire) >> 32) + 1;
        *s.m.lock().unwrap() = Some(f.clone());
        s.parts.store(parts, Release);
        s.left.store(parts, Release);
        s.ctr.store(gen << 32, Release);
        Self::work(s, &f, gen);
        while s.left.load(Acquire) > 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(1, Release);
    }
}

#[derive(Clone, Copy)]
struct P(usize);

const ROWS: usize = 32;

/// y[r] = row r of the matrix at w (rows of cols values) . x
fn matvec(pool: &Pool, t: i32, w: *const u8, rows: usize, cols: usize, x: *const f32, y: &mut [f32]) {
    let (nb, rowb) = (cols / 256, cols / 256 * sb_bytes(t));
    let (w, x, o) = (P(w as usize), P(x as usize), P(y.as_mut_ptr() as usize));
    pool.run(rows.div_ceil(ROWS), Arc::new(move |i| {
        let (w, x, o) = (w, x, o);
        for r in i * ROWS..((i + 1) * ROWS).min(rows) {
            // SAFETY: the caller's matrix, activations and output outlive `run`; each part writes its own rows; the CPU's features are the caller's check.
            unsafe { *(o.0 as *mut f32).add(r) = row_dot(t, (w.0 + r * rowb) as *const u8, nb, x.0 as *const f32) };
        }
    }));
}

pub fn supported() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// The expert at `base` against x: h its SwiGLU-OAI hidden row, y its output row (the silo's `gu` then `down`, weight 1)
#[allow(clippy::too_many_arguments)]
pub fn expert(pool: &Pool, tgu: i32, tdown: i32, base: *const u8, off_up: usize, off_down: usize, fe: usize, c: usize, x: &[f32], h: &mut [f32], y: &mut [f32]) {
    let mut g = vec![0f32; fe];
    let mut u = vec![0f32; fe];
    // SAFETY: base holds a whole expert in these types (the caller's slot).
    let (bu, bd) = unsafe { (base.add(off_up), base.add(off_down)) };
    matvec(pool, tgu, base, fe, c, x.as_ptr(), &mut g);
    matvec(pool, tgu, bu, fe, c, x.as_ptr(), &mut u);
    for r in 0..fe {
        let xg = g[r].min(7.0);
        let yu = u[r].clamp(-7.0, 7.0);
        h[r] = xg / (1.0 + (1.702 * -xg).exp()) * (yu + 1.0);
    }
    matvec(pool, tdown, bd, c, fe, h.as_ptr(), y);
}
