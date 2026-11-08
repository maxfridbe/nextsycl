//! GPU-to-GPU hop latency through the host (the cost of one cross-card sync of a split layer):
//! `hopbench A B [bytes]` ping-pongs a buffer between GPU A and GPU B (`DevBuf::copy_from_peer`, the engine's `hop`)
//! and prints the time of one hop at several sizes, with a pageable and a pinned staging area, and the time of a bare
//! queue sync on each GPU. A tensor-parallel layer needs two hops per sub-layer pair (one all-reduce each).

use nextsycl_core::{DevBuf, Gpu, HostBuf};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let idx = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (ia, ib) = (idx(1, 0), idx(2, 1));
    let ga = Gpu::open(ia).expect("gpu a");
    let gb = Gpu::open(ib).expect("gpu b");
    for (name, g) in [("a", &ga), ("b", &gb)] {
        for _ in 0..100 {
            g.sync().unwrap();
        }
        let t = Instant::now();
        for _ in 0..5000 {
            g.sync().unwrap();
        }
        println!("bare sync of gpu {name}: {:.1} us", t.elapsed().as_secs_f64() * 1e6 / 5000.0);
    }
    let sizes: Vec<usize> = if a.len() > 3 { vec![idx(3, 20480)] } else { vec![4096, 20480, 81920, 327680] };
    for n in sizes {
        let x = DevBuf::new(&ga, n).expect("buf a");
        let y = DevBuf::new(&gb, n).expect("buf b");
        let mut page = vec![0u8; n];
        let mut pin = HostBuf::new(&ga, n).expect("pinned");
        for (what, pinned) in [("pageable", false), ("pinned", true)] {
            let st: &mut [u8] = if pinned { pin.as_mut_slice() } else { &mut page };
            for _ in 0..50 {
                y.copy_from_peer(&x, st).unwrap();
                x.copy_from_peer(&y, st).unwrap();
            }
            let iters = 2000;
            let t = Instant::now();
            for _ in 0..iters {
                y.copy_from_peer(&x, st).unwrap();
                x.copy_from_peer(&y, st).unwrap();
            }
            let dt = t.elapsed().as_secs_f64();
            println!("{n:>8} bytes, {what:<8} staging: {:.1} us a hop ({iters} round trips, {:.1} GB/s one way)", dt * 1e6 / (2 * iters) as f64, n as f64 * 2.0 * iters as f64 / dt / 1e9);
        }
    }
}
