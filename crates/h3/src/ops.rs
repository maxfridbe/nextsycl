//! The kernels, with types and shapes checked on the way in. Each call queues work on the device and returns; the
//! result is there for the next call that uses it (the queue is in order).

use std::ffi::c_void;

use crate::device::Tensor;
use crate::dtype::DType;
use crate::gguf::GType;
use crate::{Error, Result};

fn floats(t: &Tensor, what: &str, n: usize) -> Result<*const f32> {
    if t.dtype != DType::F32 || t.elements() < n {
        return Err(Error(format!("{what} must be float32 with at least {n} values, got {:?} {:?}", t.dtype, t.shape)));
    }
    Ok(t.buf.ptr().cast_const().cast())
}

fn row_table(t: &Tensor, m: usize) -> Result<*const i32> {
    if t.dtype != DType::I32 || t.elements() != m {
        return Err(Error(format!("the row table must be int32 [{m}], got {:?} {:?}", t.dtype, t.shape)));
    }
    Ok(t.buf.ptr().cast_const().cast())
}

/// A linear layer with int8 weights (one layer of an `int8_convrot` checkpoint), on the device.
pub struct Int8Linear {
    /// int8 [N, K]
    pub weight: Tensor,
    /// float32, 1 or N values
    pub scale: Tensor,
    /// float32 [N]
    pub bias: Option<Tensor>,
    /// The rotation group size, when the weights were stored rotated.
    pub group: Option<usize>,
}

impl Int8Linear {
    pub fn inputs(&self) -> usize {
        self.weight.shape[1]
    }
    pub fn outputs(&self) -> usize {
        self.weight.shape[0]
    }

    /// `out = linear(x)`: x [M, K] in a float type, out [M, N] in `out`'s type.
    pub fn forward(&self, x: &Tensor, out: &Tensor) -> Result<()> {
        let (n, k) = (self.outputs(), self.inputs());
        let m = x.elements() / k;
        if self.weight.dtype != DType::I8 || self.scale.dtype != DType::F32 {
            return Err(Error("int8_linear: the weight must be int8 and its scale float32".into()));
        }
        if x.elements() != m * k || out.elements() != m * n {
            return Err(Error(format!("int8_linear: x {:?} and out {:?} do not fit a [{n}, {k}] weight", x.shape, out.shape)));
        }
        let ns = self.scale.elements();
        if ns != 1 && ns != n {
            return Err(Error(format!("int8_linear: {ns} weight scales for {n} outputs")));
        }
        let bias = match &self.bias {
            Some(b) => floats(b, "the bias", n)?,
            None => std::ptr::null(),
        };
        let dev = x.buf.device();
        // SAFETY: every pointer is device memory of this device, with the sizes checked above.
        let rc = unsafe {
            (dev.api.int8_linear)(
                dev.ctx,
                x.buf.ptr(),
                x.dtype.kernel_code()?,
                m as i64,
                k as i64,
                self.weight.buf.ptr().cast(),
                n as i64,
                self.scale.buf.ptr().cast(),
                ns as i64,
                bias,
                out.buf.ptr(),
                out.dtype.kernel_code()?,
                self.group.unwrap_or(0) as i32,
            )
        };
        dev.check(rc)
    }
}

/// A linear layer with llama.cpp k-quant weights (one block matrix of a GGUF denoiser, Q4_K or Q6_K), on the device:
/// the raw blocks stay there, and each call expands the matrix into a shared buffer in the activations' type and
/// multiplies with that - one matrix at a time, so the card holds the quantized form (Q4_K: 4.5 bits a weight)
/// plus the largest matrix in 16 bits.
pub struct KLinear {
    /// the k-quant blocks of the [N, K] matrix, as bytes
    pub blocks: Tensor,
    pub kind: GType,
    pub n: usize,
    pub k: usize,
}

impl KLinear {
    /// `out = linear(x)`: x [M, K] and out [M, N] in one 16-bit type; `w` holds at least N x K values of it.
    pub fn forward(&self, x: &Tensor, out: &Tensor, w: &Tensor) -> Result<()> {
        let (n, k) = (self.n, self.k);
        let m = x.elements() / k;
        if x.elements() != m * k || out.elements() != m * n || w.dtype != x.dtype || w.elements() < n * k {
            return Err(Error(format!("klinear: x {:?} {:?}, out {:?}, buffer {:?} {:?} for a [{n}, {k}] weight", x.dtype, x.shape, out.shape, w.dtype, w.shape)));
        }
        expand_into(&self.blocks, self.kind, n * k, w)?;
        let dev = x.buf.device();
        let code = x.dtype.kernel_code()?;
        // SAFETY: device pointers of this device; w holds the [n, k] matrix just expanded, sizes checked above.
        let rc = unsafe { (dev.api.linear)(dev.ctx, x.buf.ptr(), code, m as i64, k as i64, w.buf.ptr(), n as i64, std::ptr::null(), out.buf.ptr(), out.dtype.kernel_code()?) };
        dev.check(rc)
    }
}

/// `n` values of k-quant blocks (raw bytes on the device) into the front of `out`, in `out`'s type.
pub fn expand_into(blocks: &Tensor, kind: GType, n: usize, out: &Tensor) -> Result<()> {
    let code = kind.quant_code().ok_or_else(|| Error(format!("{kind:?} is not a k-quant")))?;
    if blocks.buf.len() < kind.bytes(n) || out.elements() < n {
        return Err(Error(format!("expand: {} bytes of {kind:?} blocks for {n} values into {:?}", blocks.buf.len(), out.shape)));
    }
    let dev = blocks.buf.device();
    // SAFETY: device buffers of this device, sizes checked above.
    let rc = unsafe { (dev.api.dequant)(dev.ctx, blocks.buf.ptr().cast_const(), code, n as i64, out.buf.ptr(), out.dtype.kernel_code()?) };
    dev.check(rc)
}

/// A k-quant tensor's raw blocks (on the device) -> a new tensor of `shape` in `dt`.
pub fn expand(blocks: &Tensor, kind: GType, shape: &[usize], dt: DType) -> Result<Tensor> {
    let out = Tensor::new(blocks.buf.device(), dt, shape)?;
    expand_into(blocks, kind, shape.iter().product(), &out)?;
    Ok(out)
}

/// A low-rank addition to a layer (a LoRA): `out += B (A x)`, A [r, K], B [N, r] in the activations' type, the
/// strength folded into B.
pub struct Lora {
    pub a: Tensor,
    pub b: Tensor,
}

impl Lora {
    pub fn rank(&self) -> usize {
        self.a.shape[0]
    }

    /// `out += B (A x)`; `tmp` holds at least [M, rank] values of x's type.
    pub fn apply(&self, x: &Tensor, tmp: &Tensor, out: &Tensor) -> Result<()> {
        let (r, k, n) = (self.rank(), self.a.shape[1], self.b.shape[0]);
        let m = x.elements() / k;
        if x.dtype != self.a.dtype || out.dtype != x.dtype || tmp.dtype != x.dtype || tmp.elements() < m * r || out.elements() != m * n {
            return Err(Error(format!("lora: x {:?} {:?}, out {:?} {:?}, A {:?}, B {:?}", x.dtype, x.shape, out.dtype, out.shape, self.a.shape, self.b.shape)));
        }
        let dev = x.buf.device();
        let code = x.dtype.kernel_code()?;
        // SAFETY: device buffers of this device, sizes checked above.
        let rc = unsafe { (dev.api.linear)(dev.ctx, x.buf.ptr(), code, m as i64, k as i64, self.a.buf.ptr(), r as i64, std::ptr::null(), tmp.buf.ptr(), code) };
        dev.check(rc)?;
        let rc = unsafe { (dev.api.linear_acc)(dev.ctx, tmp.buf.ptr(), code, m as i64, r as i64, self.b.buf.ptr(), n as i64, std::ptr::null(), out.buf.ptr()) };
        dev.check(rc)
    }
}

/// A plain linear layer with float weights, on the device.
pub struct Linear {
    /// [N, K], float32, half or bfloat16
    pub weight: Tensor,
    /// float32 [N]
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn inputs(&self) -> usize {
        self.weight.shape[1]
    }
    pub fn outputs(&self) -> usize {
        self.weight.shape[0]
    }

    /// `out += linear(x)` (bias included), x and out in the weight's type: a residual add folded into the product.
    pub fn forward_acc(&self, x: &Tensor, out: &Tensor) -> Result<()> {
        let (n, k) = (self.outputs(), self.inputs());
        let m = x.elements() / k;
        if x.dtype != self.weight.dtype || out.dtype != x.dtype || x.elements() != m * k || out.elements() != m * n {
            return Err(Error(format!("linear_acc: x {:?} {:?}, out {:?} {:?}, weight [{n}, {k}] {:?}", x.dtype, x.shape, out.dtype, out.shape, self.weight.dtype)));
        }
        let bias = match &self.bias {
            Some(b) => floats(b, "the bias", n)?,
            None => std::ptr::null(),
        };
        let dev = x.buf.device();
        // SAFETY: device pointers of this device, sizes checked above.
        let rc = unsafe { (dev.api.linear_acc)(dev.ctx, x.buf.ptr(), x.dtype.kernel_code()?, m as i64, k as i64, self.weight.buf.ptr(), n as i64, bias, out.buf.ptr()) };
        dev.check(rc)
    }

    /// Scales output row n of the layer (weight row and bias) by s[n].
    pub fn scale_outputs(&mut self, s: &[f32]) -> Result<()> {
        let (n, k) = (self.outputs(), self.inputs());
        if s.len() != n {
            return Err(Error(format!("scale_outputs: {} factors for {n} outputs", s.len())));
        }
        let dev = self.weight.buf.device().clone();
        let st = Tensor::from_bytes(&dev, DType::F32, &[n], &s.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
        // SAFETY: the weight [n, k] and the factors float32 [n] on this device.
        let rc = unsafe { (dev.api.scale_rows)(dev.ctx, self.weight.buf.ptr(), self.weight.dtype.kernel_code()?, n as i64, k as i64, st.buf.ptr().cast()) };
        dev.check(rc)?;
        if let Some(b) = &self.bias {
            let mut v = b.to_f32()?;
            v.iter_mut().zip(s).for_each(|(a, f)| *a *= f);
            self.bias = Some(Tensor::from_bytes(&dev, DType::F32, &[n], &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?);
        }
        dev.wait()
    }

    /// `out = linear(x)`: x [M, K] in the weight's type, out [M, N] in `out`'s type.
    pub fn forward(&self, x: &Tensor, out: &Tensor) -> Result<()> {
        let (n, k) = (self.outputs(), self.inputs());
        let m = x.elements() / k;
        if x.dtype != self.weight.dtype {
            return Err(Error(format!("linear: x is {:?} and the weight {:?}; they must be the same type", x.dtype, self.weight.dtype)));
        }
        if x.elements() != m * k || out.elements() != m * n {
            return Err(Error(format!("linear: x {:?} and out {:?} do not fit a [{n}, {k}] weight", x.shape, out.shape)));
        }
        let bias = match &self.bias {
            Some(b) => floats(b, "the bias", n)?,
            None => std::ptr::null(),
        };
        let dev = x.buf.device();
        // SAFETY: device pointers of this device, sizes checked above.
        let rc = unsafe {
            (dev.api.linear)(dev.ctx, x.buf.ptr(), x.dtype.kernel_code()?, m as i64, k as i64, self.weight.buf.ptr(), n as i64, bias, out.buf.ptr(), out.dtype.kernel_code()?)
        };
        dev.check(rc)
    }
}

/// How a denoiser block's tokens are modulated: which table row each token uses, and the tables.
pub struct Mod<'a> {
    /// int32 [M]
    pub rows: &'a Tensor,
    /// float32 [R, C] each
    pub scale: &'a Tensor,
    pub shift: &'a Tensor,
}

/// Row-wise RMS norm of x [M, C] (weight float32 [C]), then `* (1 + scale[row]) + shift[row]` when `m` is given.
/// `out` may be `x`.
pub fn rms_norm_mod(x: &Tensor, weight: &Tensor, eps: f32, m: Option<&Mod>, out: &Tensor) -> Result<()> {
    let c = weight.elements();
    let rows = x.elements() / c;
    if x.elements() != rows * c || out.elements() != x.elements() {
        return Err(Error(format!("rms_norm: x {:?}, out {:?}, {c} features", x.shape, out.shape)));
    }
    let (r, sc, sh) = match m {
        None => (std::ptr::null(), std::ptr::null(), std::ptr::null()),
        Some(m) => (row_table(m.rows, rows)?, floats(m.scale, "scale", c)?, floats(m.shift, "shift", c)?),
    };
    let dev = x.buf.device();
    // SAFETY: device pointers of this device; sizes checked above (the tables' row count is the caller's contract).
    let rc = unsafe {
        (dev.api.rms_norm_mod)(
            dev.ctx,
            x.buf.ptr(),
            x.dtype.kernel_code()?,
            rows as i64,
            c as i64,
            floats(weight, "the norm weight", c)?,
            eps,
            r,
            sc,
            sh,
            out.buf.ptr(),
            out.dtype.kernel_code()?,
        )
    };
    dev.check(rc)
}

/// Layer norm of x [M, C] per row, with an optional weight and bias (float32 [C]). `out` may be `x`.
pub fn layer_norm(x: &Tensor, c: usize, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32, out: &Tensor) -> Result<()> {
    let rows = x.elements() / c.max(1);
    if x.elements() != rows * c || out.elements() != x.elements() {
        return Err(Error(format!("layer_norm: x {:?}, out {:?}, {c} features", x.shape, out.shape)));
    }
    let opt = |t: Option<&Tensor>, what: &str| -> Result<*const f32> { t.map_or(Ok(std::ptr::null()), |t| floats(t, what, c)) };
    let dev = x.buf.device();
    // SAFETY: device pointers of this device; sizes checked above.
    let rc = unsafe {
        (dev.api.layer_norm)(dev.ctx, x.buf.ptr(), x.dtype.kernel_code()?, rows as i64, c as i64, opt(weight, "the norm weight")?, opt(bias, "the norm bias")?,
                             eps, out.buf.ptr(), out.dtype.kernel_code()?)
    };
    dev.check(rc)
}

/// Token rows of `heads` x `dim` features inside a wider row-major buffer: row s starts at element
/// `offset + s * stride`. This is how q, k and v sit inside the qkv linear's output.
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub t: &'a Tensor,
    pub offset: usize,
    pub stride: usize,
    pub tokens: usize,
    pub heads: usize,
    pub dim: usize,
}

impl Rows<'_> {
    fn layout(&self) -> (usize, usize, usize, usize) {
        (self.tokens, self.heads, self.dim, self.stride)
    }

    fn ptr(&self) -> Result<*mut c_void> {
        let last = self.offset + (self.tokens.max(1) - 1) * self.stride + self.heads * self.dim;
        if self.stride < self.heads * self.dim || last > self.t.elements() {
            return Err(Error(format!(
                "rows of {} x {} at offset {} with stride {} do not fit {:?}",
                self.heads, self.dim, self.offset, self.stride, self.t.shape
            )));
        }
        // SAFETY: in bounds, checked above.
        Ok(unsafe { self.t.buf.ptr().cast::<u8>().add(self.offset * self.t.dtype.size()).cast() })
    }
}

/// Per-head RMS norm (weight float32 [dim]) and the rotary rotation, in place. `cs` float32 [tokens, rot_dim/2, 2].
pub fn rms_rope(x: Rows, weight: &Tensor, eps: f32, cs: &Tensor, rot_dim: usize) -> Result<()> {
    let dev = x.t.buf.device();
    // SAFETY: device pointers of this device; `Rows::ptr` and `floats` check the sizes.
    let rc = unsafe {
        (dev.api.rms_rope)(
            dev.ctx,
            x.ptr()?,
            x.t.dtype.kernel_code()?,
            x.tokens as i64,
            x.heads as i64,
            x.dim as i64,
            x.stride as i64,
            floats(weight, "the norm weight", x.dim)?,
            eps,
            floats(cs, "the rotation table", x.tokens * rot_dim)?,
            rot_dim as i32,
        )
    };
    dev.check(rc)
}

/// `out[r, i] = silu(x[r, i]) * x[r, C + i]`: x [M, 2C], out [M, C].
pub fn swiglu(x: &Tensor, out: &Tensor) -> Result<()> {
    let c = *out.shape.last().ok_or("swiglu: out has no shape")?;
    if x.elements() != out.elements() * 2 || c == 0 {
        return Err(Error(format!("swiglu: x {:?} is not twice out {:?}", x.shape, out.shape)));
    }
    let dev = x.buf.device();
    // SAFETY: device pointers of this device; sizes checked above.
    let rc = unsafe {
        (dev.api.swiglu)(dev.ctx, x.buf.ptr(), x.dtype.kernel_code()?, (out.elements() / c) as i64, c as i64, out.buf.ptr(), out.dtype.kernel_code()?)
    };
    dev.check(rc)
}

/// `x[r] += other[r] * gate[rows[r]]`, in place. x, other [M, C]; rows int32 [M]; gate float32 [R, C].
pub fn gate_add(x: &Tensor, other: &Tensor, rows: &Tensor, gate: &Tensor) -> Result<()> {
    let c = *x.shape.last().ok_or("gate_add: x has no shape")?;
    let m = x.elements() / c.max(1);
    if other.elements() != x.elements() {
        return Err(Error(format!("gate_add: x {:?} and other {:?} differ", x.shape, other.shape)));
    }
    let dev = x.buf.device();
    // SAFETY: device pointers of this device; sizes checked above.
    let rc = unsafe {
        (dev.api.gate_add)(
            dev.ctx,
            x.buf.ptr(),
            x.dtype.kernel_code()?,
            m as i64,
            c as i64,
            other.buf.ptr(),
            other.dtype.kernel_code()?,
            row_table(rows, m)?,
            floats(gate, "the gate", c)?,
        )
    };
    dev.check(rc)
}

/// `x += other`, in place (a residual connection without a gate). x, other [M, C], any float types.
pub fn add(x: &Tensor, other: &Tensor) -> Result<()> {
    let c = *x.shape.last().ok_or("add: x has no shape")?;
    if other.elements() != x.elements() {
        return Err(Error(format!("add: x {:?} and other {:?} differ", x.shape, other.shape)));
    }
    let dev = x.buf.device();
    // SAFETY: device pointers of this device; sizes checked above; no row table and no gate: a plain add.
    let rc = unsafe {
        (dev.api.gate_add)(dev.ctx, x.buf.ptr(), x.dtype.kernel_code()?, (x.elements() / c.max(1)) as i64, c as i64, other.buf.ptr(),
                           other.dtype.kernel_code()?, std::ptr::null(), std::ptr::null())
    };
    dev.check(rc)
}

/// `out = softmax(q . k^T / sqrt(dim)) . v` per head; out [tokens, heads * dim].
/// Attention over `seqs` independent sequences of `q.tokens / seqs` tokens each, in one call: sequence b is the rows
/// `b * S ..` of q, k, v and of out (the video decoder's tiles). Same result as one `attention` per sequence.
pub fn attention_batch(q: Rows, k: Rows, v: Rows, seqs: usize, out: &Tensor) -> Result<()> {
    if q.layout() != k.layout() || q.layout() != v.layout() {
        return Err(Error("attention_batch: q, k and v must have the same layout".into()));
    }
    if seqs == 0 || !q.tokens.is_multiple_of(seqs) {
        return Err(Error(format!("attention_batch: {} tokens are not {seqs} equal sequences", q.tokens)));
    }
    if out.elements() != q.tokens * q.heads * q.dim {
        return Err(Error(format!("attention_batch: out {:?} for {} tokens of {} x {}", out.shape, q.tokens, q.heads, q.dim)));
    }
    let dev = out.buf.device();
    // SAFETY: device pointers of this device; `Rows::ptr` checks the sizes (all sequences' rows).
    let rc = unsafe {
        (dev.api.attention_batch)(
            dev.ctx,
            q.ptr()?,
            k.ptr()?,
            v.ptr()?,
            q.t.dtype.kernel_code()?,
            seqs as i64,
            (q.tokens / seqs) as i64,
            q.heads as i64,
            q.dim as i64,
            q.stride as i64,
            out.buf.ptr(),
            out.dtype.kernel_code()?,
        )
    };
    dev.check(rc)
}

pub fn attention(q: Rows, k: Rows, v: Rows, out: &Tensor) -> Result<()> {
    if q.layout() != k.layout() || q.layout() != v.layout() {
        return Err(Error("attention: q, k and v must have the same layout".into()));
    }
    if out.elements() != q.tokens * q.heads * q.dim {
        return Err(Error(format!("attention: out {:?} for {} tokens of {} x {}", out.shape, q.tokens, q.heads, q.dim)));
    }
    let dev = out.buf.device();
    // SAFETY: device pointers of this device; `Rows::ptr` checks the sizes.
    let rc = unsafe {
        (dev.api.attention)(
            dev.ctx,
            q.ptr()?,
            k.ptr()?,
            v.ptr()?,
            q.t.dtype.kernel_code()?,
            q.tokens as i64,
            q.heads as i64,
            q.dim as i64,
            q.stride as i64,
            out.buf.ptr(),
            out.dtype.kernel_code()?,
        )
    };
    dev.check(rc)
}
