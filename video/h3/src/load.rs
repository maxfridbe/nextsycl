//! Checkpoint -> device memory.
//!
//! Measured on the B70 box: the host-to-device copy runs at about 6 GB/s, and one thread reading the file at 0.4.
//! So the file is read by several threads at once, in blocks, and each block goes to the device as soon as it is in
//! memory. Nothing is held on the host beyond one block per thread.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::device::{Device, Tensor};
use crate::safetensors::Checkpoint;
use crate::{Ctx, Error, Result};

pub const BLOCK: usize = 32 << 20;

pub struct Loaded {
    pub tensors: BTreeMap<String, Tensor>,
    pub bytes: u64,
    pub seconds: f64,
}

/// Loads every tensor `keep` accepts. `threads` readers; 8 is what the measurements settled on.
pub fn load(dev: &Arc<Device>, ckpt: &Checkpoint, threads: usize, keep: impl Fn(&str) -> bool) -> Result<Loaded> {
    let t0 = Instant::now();
    // device memory first, in file order, so the readers below walk the file front to back
    let mut wanted: Vec<_> = ckpt.entries.values().filter(|e| keep(&e.name)).collect();
    wanted.sort_by_key(|e| e.offset);
    let mut tensors = BTreeMap::new();
    let mut jobs = Vec::new(); // (tensor index, offset in the tensor, file offset, bytes)
    let mut order = Vec::new();
    for (i, e) in wanted.iter().enumerate() {
        let t = Tensor::new(dev, e.dtype, &e.stored_shape()).ctx(format!("allocating {}", e.name))?;
        let mut off = 0;
        while off < e.bytes {
            let n = BLOCK.min(e.bytes - off);
            jobs.push((i, off, e.offset + off as u64, n));
            off += n;
        }
        order.push(e.name.clone());
        tensors.insert(e.name.clone(), t);
    }
    let bufs: Vec<&crate::device::Buf> = order.iter().map(|n| &tensors[n].buf).collect();
    let next = AtomicUsize::new(0);
    let failed: Mutex<Option<Error>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let run = || -> Result<()> {
                    let f = File::open(&ckpt.path)?;
                    let mut block = vec![0u8; BLOCK];
                    loop {
                        let j = next.fetch_add(1, Ordering::Relaxed);
                        if j >= jobs.len() || failed.lock().unwrap().is_some() {
                            return Ok(());
                        }
                        let (ti, off, foff, n) = jobs[j];
                        f.read_exact_at(&mut block[..n], foff).ctx("reading the checkpoint")?;
                        bufs[ti].write(off, &block[..n])?;
                    }
                };
                if let Err(e) = run() {
                    failed.lock().unwrap().get_or_insert(e);
                }
            });
        }
    });
    if let Some(e) = failed.into_inner().unwrap() {
        return Err(e);
    }
    let bytes = wanted.iter().map(|e| e.bytes as u64).sum();
    Ok(Loaded { tensors, bytes, seconds: t0.elapsed().as_secs_f64() })
}
