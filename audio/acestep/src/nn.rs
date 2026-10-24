//! The pieces every ACE-Step transformer is made of, on the shared kernels (nsd) and this engine's two (ffi): the
//! checkpoint's bfloat16 matrices as they are on the card, Qwen3's layer (pre-norm attention with a per-head RMS norm
//! of q and k and rotary positions, then a SwiGLU MLP) run causal (the text encoder, the planner), bidirectional or
//! inside a window of 128 (the lyric and timbre encoders, the tokenizer's pooler, the detokenizer), and the rotary
//! table. The residual stream is float32; the products read and write bfloat16 (the GEMMs accumulate in float32), as
//! the reference's GPU run does in its bfloat16.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi;

pub type P = *const c_void;
pub type M = *mut c_void;

/// every ACE-Step attention's head size
pub const HD: usize = 128;
/// Qwen3's RMS norm epsilon (all of ACE-Step's configs)
pub const EPS: f32 = 1e-6;

/// A pointer `bytes` into a buffer
pub fn at(b: &DevBuf, bytes: usize) -> M {
    b.ptr().cast::<u8>().wrapping_add(bytes).cast()
}

/// bfloat16 bits to float
pub fn bf(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// float to bfloat16 bits (round to nearest even)
pub fn to_bf(f: f32) -> u16 {
    let u = f.to_bits();
    if u & 0x7fff_ffff > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

pub fn bf_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| to_bf(*x).to_le_bytes()).collect()
}

pub fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: plain floats as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
}

/// A device buffer's floats
pub fn read_f32(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut v = vec![0f32; n];
    // SAFETY: plain floats as bytes.
    b.read(0, unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * 4) })?;
    Ok(v)
}

/// A device buffer's bfloat16 values as floats
pub fn read_bf(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    b.read(0, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|c| bf(u16::from_le_bytes([c[0], c[1]]))).collect())
}

/// A matrix [n, k] (bfloat16 on the card) and its float32 bias
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

/// Reads a checkpoint's tensors onto one GPU
pub struct Loader<'a> {
    pub st: &'a SafeTensors,
    pub gpu: &'a Arc<Gpu>,
    pub bytes: std::cell::Cell<u64>,
}

impl<'a> Loader<'a> {
    pub fn new(st: &'a SafeTensors, gpu: &'a Arc<Gpu>) -> Loader<'a> {
        Loader { st, gpu, bytes: std::cell::Cell::new(0) }
    }

    /// a tensor's values as bfloat16 bits (its own when it is bfloat16)
    pub fn bf_raw(&self, name: &str) -> Result<(Vec<u8>, Vec<u64>)> {
        let t = self.st.need(name).map_err(|e| Error(e.0))?;
        let bytes = match t.dtype.as_str() {
            "BF16" => self.st.read(t).map_err(|e| Error(e.0))?,
            _ => bf_bytes(&self.st.f32(name).map_err(|e| Error(e.0))?),
        };
        Ok((bytes, t.shape.clone()))
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<DevBuf> {
        let b = DevBuf::new(self.gpu, bytes.len().max(2))?;
        b.write(0, bytes)?;
        self.bytes.set(self.bytes.get() + bytes.len() as u64);
        Ok(b)
    }

    /// a tensor's values as floats on the host
    pub fn host(&self, name: &str) -> Result<Vec<f32>> {
        self.st.f32(name).map_err(|e| Error(e.0))
    }

    /// a float32 vector on the card
    pub fn float(&self, name: &str) -> Result<DevBuf> {
        let v = self.host(name)?;
        self.upload(f32_bytes(&v))
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.tensor(name).is_some()
    }

    /// `{p}.weight` (and `{p}.bias` when `bias`)
    pub fn lin(&self, p: &str, bias: bool) -> Result<Lin> {
        let (w, shape) = self.bf_raw(&format!("{p}.weight"))?;
        let b = if bias { Some(self.float(&format!("{p}.bias"))?) } else { None };
        Ok(Lin { w: self.upload(&w)?, b, n: shape[0] as usize, k: shape[1..].iter().product::<u64>() as usize })
    }

    /// the rows of several matrices side by side as one ([n1 + n2 + ..., k]): a fused product
    pub fn fused(&self, names: &[String]) -> Result<Lin> {
        let mut all = Vec::new();
        let (mut n, mut k) = (0, 0);
        for name in names {
            let (w, shape) = self.bf_raw(name)?;
            n += shape[0] as usize;
            k = shape[1] as usize;
            all.extend_from_slice(&w);
        }
        Ok(Lin { w: self.upload(&all)?, b: None, n, k })
    }

    /// a key / value matrix of `hkv` heads with each head's rows repeated `g` times: the `hkv * g` heads a plain
    /// (non-grouped) attention kernel reads
    pub fn repeated_heads(&self, name: &str, g: usize) -> Result<Lin> {
        let (w, shape) = self.bf_raw(name)?;
        let (n, k) = (shape[0] as usize, shape[1] as usize);
        let head = HD * k * 2;
        let mut out = Vec::with_capacity(w.len() * g);
        for h in 0..n / HD {
            for _ in 0..g {
                out.extend_from_slice(&w[h * head..(h + 1) * head]);
            }
        }
        Ok(Lin { w: self.upload(&out)?, b: None, n: n * g, k })
    }
}

/// The rotary table (cos, sin) for positions [0, n) of a head of 128 (theta: the config's rope_theta), as
/// transformers' default rotary: inv_freq = theta^(-2i / 128), float32
pub fn rope_table(n: usize, theta: f64, pos0: usize) -> Vec<f32> {
    let half = HD / 2;
    let inv: Vec<f64> = (0..half).map(|i| 1.0 / theta.powf(2.0 * i as f64 / HD as f64)).collect();
    let mut cs = Vec::with_capacity(n * half * 2);
    for p in pos0..pos0 + n {
        for f in &inv {
            // transformers: (position * inv_freq) in float32, then cos / sin
            let a = (p as f32) * (*f as f32);
            cs.push(a.cos());
            cs.push(a.sin());
        }
    }
    cs
}

/// How a layer's attention sees its keys
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mask {
    Causal,
    Full,
    /// |i - j| <= w
    Window(usize),
    /// sequences of this many rows side by side, each its own (bidirectional within)
    Segments(usize),
}

/// Qwen3's attention: q, k, v, o, the per-head norms of q and k
pub struct Attn {
    pub q: Lin,
    pub k: Lin,
    pub v: Lin,
    pub o: Lin,
    pub qn: DevBuf,
    pub kn: DevBuf,
    pub hq: usize,
    /// heads in k / v as stored (the checkpoint's, or repeated to hq for a plain kernel)
    pub hkv: usize,
}

impl Attn {
    /// `{p}.q_proj` ... with the key heads repeated to the query heads when `repeat` (a full bidirectional layer:
    /// nsd's plain attention), else as stored (causal: grouped natively; windowed: this engine's kernel)
    pub fn load(l: &Loader, p: &str, heads: usize, kv_heads: usize, repeat: bool) -> Result<Attn> {
        let g = heads / kv_heads;
        let (k, v) = if repeat && g > 1 {
            (l.repeated_heads(&format!("{p}.k_proj.weight"), g)?, l.repeated_heads(&format!("{p}.v_proj.weight"), g)?)
        } else {
            (l.lin(&format!("{p}.k_proj"), false)?, l.lin(&format!("{p}.v_proj"), false)?)
        };
        let hkv = k.n / HD;
        Ok(Attn { q: l.lin(&format!("{p}.q_proj"), false)?, k, v, o: l.lin(&format!("{p}.o_proj"), false)?, qn: l.float(&format!("{p}.q_norm.weight"))?,
                  kn: l.float(&format!("{p}.k_norm.weight"))?, hq: heads, hkv })
    }
}

/// Qwen3's MLP: gate and up fused ([2I, H]), down
pub struct Mlp {
    pub gu: Lin,
    pub down: Lin,
}

impl Mlp {
    pub fn load(l: &Loader, p: &str) -> Result<Mlp> {
        Ok(Mlp { gu: l.fused(&[format!("{p}.gate_proj.weight"), format!("{p}.up_proj.weight")])?, down: l.lin(&format!("{p}.down_proj"), false)? })
    }
}

/// A pre-norm layer (Qwen3's decoder layer, ACE-Step's AceStepEncoderLayer)
pub struct Layer {
    pub ln1: DevBuf,
    pub attn: Attn,
    pub ln2: DevBuf,
    pub mlp: Mlp,
    pub mask: Mask,
}

impl Layer {
    pub fn load(l: &Loader, p: &str, heads: usize, kv_heads: usize, mask: Mask) -> Result<Layer> {
        Ok(Layer { ln1: l.float(&format!("{p}.input_layernorm.weight"))?, attn: Attn::load(l, &format!("{p}.self_attn"), heads, kv_heads, mask == Mask::Full)?,
                   ln2: l.float(&format!("{p}.post_attention_layernorm.weight"))?, mlp: Mlp::load(l, &format!("{p}.mlp"))?, mask })
    }
}

/// Scratch buffers that grow to the largest request seen (one engine runs one request at a time)
#[derive(Default)]
pub struct Scratch {
    bufs: std::collections::BTreeMap<&'static str, DevBuf>,
}

impl Scratch {
    pub fn get(&mut self, gpu: &Arc<Gpu>, name: &'static str, bytes: usize) -> Result<M> {
        if self.bufs.get(name).is_none_or(|b| b.len < bytes) {
            self.bufs.remove(name);
            self.bufs.insert(name, DevBuf::new(gpu, bytes.max(16))?);
        }
        Ok(self.bufs[name].ptr())
    }

    pub fn buf(&self, name: &'static str) -> &DevBuf {
        &self.bufs[name]
    }

    pub fn clear(&mut self) {
        self.bufs.clear();
    }

    /// the bytes held
    pub fn bytes(&self) -> usize {
        self.bufs.values().map(|b| b.len).sum()
    }
}

/// The kernels a model runs on
pub struct Ops {
    pub nsd: Nsd,
    pub gpu: Arc<Gpu>,
    pub k: &'static ffi::Api,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { nsd: Nsd::new(gpu)?, gpu: gpu.clone(), k: ffi::api()? })
    }

    /// out [m, n] = x [m, k] (bfloat16) . W^T + bias, out in `odt`
    pub fn linear(&self, x: P, m: usize, w: &Lin, out: M, odt: Dt) -> Result<()> {
        self.nsd.linear(x, Dt::BF16, m, w.k, w.w.ptr(), w.n, w.b.as_ref().map_or(none(), |b| b.ptr() as P), out, odt)
    }

    /// an RMS norm of float32 rows into bfloat16 (weight float32 [c])
    pub fn rms(&self, x: P, m: usize, c: usize, w: &DevBuf, out: M) -> Result<()> {
        self.nsd.rms_norm_mod(x, Dt::F32, m, c, w.ptr(), EPS, none(), none(), none(), out, Dt::BF16)
    }

    /// x (float32 [m, c]) += y (bfloat16 [m, c]) * gate (float32 [c], or none)
    pub fn add(&self, x: M, y: P, m: usize, c: usize, gate: P) -> Result<()> {
        self.nsd.gate_add(x, Dt::F32, m, c, y, Dt::BF16, none(), gate)
    }

    /// q / k (bfloat16 rows of h heads, `stride` apart): the per-head norm, then the rotation by `cs` (float32 table on
    /// the card, a row a position) - none: the norm alone
    pub fn qk_norm_rope(&self, x: M, m: usize, h: usize, stride: usize, w: &DevBuf, cs: P) -> Result<()> {
        let rot = if cs.is_null() { 0 } else { HD };
        self.nsd.rms_rope(x, Dt::BF16, m, h, HD, stride, w.ptr(), EPS, cs, rot)
    }

    /// attention of m rows over their own keys under `mask` (q bfloat16 [m, hq x 128], k / v [m, hkv x 128]) into
    /// out bfloat16 [m, hq x 128]
    pub fn self_attention(&self, q: P, k: P, v: P, m: usize, hq: usize, hkv: usize, mask: Mask, out: M) -> Result<()> {
        let scale = 1.0 / (HD as f32).sqrt();
        match mask {
            Mask::Causal => self.nsd.attention_causal(q, k, v, Dt::BF16, m, hq, hkv, HD, hq * HD, hkv * HD, out),
            Mask::Full if hq == hkv => self.nsd.attention(q, k, v, Dt::BF16, m, hq, HD, hq * HD, out, Dt::BF16),
            Mask::Full => self.window(q, k, v, m, hq, hkv, -1, scale, out),
            Mask::Window(w) => self.window(q, k, v, m, hq, hkv, w as i64, scale, out),
            Mask::Segments(n) => {
                // SAFETY: q / k / v / out hold m rows of the heads named.
                ffi::check(unsafe { (self.k.attn_segments)(self.gpu.raw(), q, (hq * HD) as i64, k, v, (hkv * HD) as i64, m as i64, hq as i64, hkv as i64,
                                                            n as i64, scale, out, (hq * HD) as i64) }, "attn_segments")
            }
        }
    }

    pub fn window(&self, q: P, k: P, v: P, m: usize, hq: usize, hkv: usize, w: i64, scale: f32, out: M) -> Result<()> {
        // SAFETY: q / k / v / out hold m rows of the heads named.
        ffi::check(unsafe { (self.k.attn_window)(self.gpu.raw(), q, (hq * HD) as i64, k, v, (hkv * HD) as i64, m as i64, hq as i64, hkv as i64, w, scale,
                                                 out, (hq * HD) as i64) }, "attn_window")
    }

    /// One pre-norm layer over x (float32 [m, c]): x += attn(norm(x)); x += mlp(norm(x)). `cs`: the rotary table on
    /// the card for these m positions
    pub fn layer(&self, s: &mut Scratch, l: &Layer, x: M, m: usize, c: usize, cs: P) -> Result<()> {
        let a = &l.attn;
        let (qw, kw, i2) = (a.hq * HD, a.hkv * HD, l.mlp.gu.n);
        let g = &self.gpu;
        let h = s.get(g, "h", m * c * 2)?;
        let q = s.get(g, "q", m * qw * 2)?;
        let kk = s.get(g, "k", m * kw * 2)?;
        let v = s.get(g, "v", m * kw * 2)?;
        let o = s.get(g, "o", m * qw * 2)?;
        let y = s.get(g, "y", m * c * 2)?;
        let gu = s.get(g, "gu", m * i2 * 2)?;
        let ff = s.get(g, "ff", m * (i2 / 2) * 2)?;
        self.rms(x, m, c, &l.ln1, h)?;
        self.linear(h, m, &a.q, q, Dt::BF16)?;
        self.linear(h, m, &a.k, kk, Dt::BF16)?;
        self.linear(h, m, &a.v, v, Dt::BF16)?;
        self.qk_norm_rope(q, m, a.hq, qw, &a.qn, cs)?;
        self.qk_norm_rope(kk, m, a.hkv, kw, &a.kn, cs)?;
        self.self_attention(q, kk, v, m, a.hq, a.hkv, l.mask, o)?;
        self.linear(o, m, &a.o, y, Dt::BF16)?;
        self.add(x, y, m, c, none())?;
        self.rms(x, m, c, &l.ln2, h)?;
        self.linear(h, m, &l.mlp.gu, gu, Dt::BF16)?;
        self.nsd.swiglu(gu, Dt::BF16, m, i2 / 2, ff, Dt::BF16)?;
        self.linear(ff, m, &l.mlp.down, y, Dt::BF16)?;
        self.add(x, y, m, c, none())
    }

    /// float32 values onto the card into a scratch slot
    pub fn put_f32(&self, s: &mut Scratch, name: &'static str, v: &[f32]) -> Result<M> {
        let p = s.get(&self.gpu, name, v.len() * 4)?;
        s.buf(name).write(0, f32_bytes(v))?;
        Ok(p)
    }

    /// floats onto the card as bfloat16 into a scratch slot
    pub fn put_bf(&self, s: &mut Scratch, name: &'static str, v: &[f32]) -> Result<M> {
        let p = s.get(&self.gpu, name, v.len() * 2)?;
        s.buf(name).write(0, &bf_bytes(v))?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfloat16_rounds_to_nearest_even() {
        for x in [0.0f32, 1.0, -2.5, 3.140625, 1e-20, 65504.0, -1e30] {
            let back = bf(to_bf(x));
            assert!((back - x).abs() <= x.abs() / 128.0, "{x} -> {back}");
        }
        assert_eq!(to_bf(1.0), 0x3f80);
        // halfway between two: to the even one
        assert_eq!(to_bf(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(to_bf(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert!(bf(to_bf(f32::NAN)).is_nan());
    }

    #[test]
    fn the_rotary_table_is_transformers() {
        let cs = rope_table(3, 1_000_000.0, 0);
        assert_eq!(cs.len(), 3 * 64 * 2);
        assert_eq!((cs[0], cs[1]), (1.0, 0.0), "position 0");
        // position 1, frequency 0: angle 1
        assert!((cs[128] - 1f32.cos()).abs() < 1e-7 && (cs[129] - 1f32.sin()).abs() < 1e-7);
        // the last frequency: 1e6^(-126/128)
        let f = (1.0f64 / 1e6f64.powf(126.0 / 128.0)) as f32;
        assert!((cs[2 * 128 + 63 * 2 + 1] - (2.0 * f).sin()).abs() < 1e-7);
    }
}
