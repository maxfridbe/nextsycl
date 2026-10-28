//! The denoiser (unimate/models/denoiser/graph_adaln.py, base.py, blocks/{common,graph}.py, rope.py): 61 frames (a rest
//! pose, then 60) of 71 joints, each token 512 wide; ten blocks of graph attention among the joints of a frame (spectral
//! rotary positions, a bias from the graph's distances and edge types), attention among a joint's frames (sinusoidal
//! rotary positions), and a SwiGLU MLP, all modulated by one vector (the time, the text, the rest pose's pooled
//! embedding); a last layer in which the root reads all the joints. Float32, the reference's own order of operations.

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::ffi::{Attn, Rows, GELU, SILU, SWIGLU};
use crate::ops::{HostLin, Lin, Ops, Store};

pub const J: usize = 71;
pub const T: usize = 60;
pub const FEAT: usize = 12;
pub const DIM: usize = 512;
pub const HEADS: usize = 8;
pub const HD: usize = HEAD_DIM;
const HEAD_DIM: usize = 64;
pub const LAYERS: usize = 10;
const FFN: usize = 1365;
const FRAMES: usize = T + 1;
const TOK: usize = FRAMES * J;
const EPS: f32 = 1e-6;
const FREQS: usize = 8;
const DEPTHS: usize = 21;
const TEXT: usize = crate::t5::DIM;

/// Linear - SiLU - Linear
struct Mlp {
    l1: Lin,
    l2: Lin,
}

impl Mlp {
    fn load(s: &mut Store, name: &str, k: usize, hidden: usize, out: usize) -> Result<Mlp> {
        Ok(Mlp { l1: s.lin(&format!("{name}.0"), hidden, k, true)?, l2: s.lin(&format!("{name}.2"), out, hidden, true)? })
    }

    fn run(&self, ops: &Ops, x: &DevBuf, m: usize, tmp: &DevBuf, out: &DevBuf) -> Result<()> {
        ops.linear(x, m, &self.l1, tmp)?;
        ops.k.act(tmp, m * self.l1.n, SILU)?;
        ops.linear(tmp, m, &self.l2, out)
    }

    fn apply(&self, ops: &Ops, x: &DevBuf, m: usize) -> Result<DevBuf> {
        let (tmp, out) = (ops.buf(m * self.l1.n)?, ops.buf(m * self.l2.n)?);
        self.run(ops, x, m, &tmp, &out)?;
        Ok(out)
    }
}

/// A multi-head attention's input projection cut in three, and its output projection
struct Mha {
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
}

impl Mha {
    fn load(s: &mut Store, name: &str) -> Result<Mha> {
        let mut p = s.lin_split(&format!("{name}.in_proj_weight"), Some(&format!("{name}.in_proj_bias")), DIM, &[DIM, DIM, DIM])?;
        let v = p.pop().ok_or_else(|| Error("in_proj".into()))?;
        let k = p.pop().ok_or_else(|| Error("in_proj".into()))?;
        let q = p.pop().ok_or_else(|| Error("in_proj".into()))?;
        Ok(Mha { q, k, v, out: s.lin(&format!("{name}.out_proj"), DIM, DIM, true)? })
    }
}

struct Pool {
    queries: Vec<f32>,
    norm_q: DevBuf,
    norm_kv: DevBuf,
    attn: Mha,
    norm_ff: DevBuf,
    w12: Lin,
    w3: Lin,
    norm_out: DevBuf,
    out_proj: Lin,
}

const QUERIES: usize = 4;
const POOL_HEADS: usize = 4;

struct Sign {
    phi0_w: Vec<f32>,
    phi0_b: Vec<f32>,
    phi2: Lin,
    rho0: Lin,
    rho2: Lin,
}

const SIGN_HIDDEN: usize = 64;

struct Block {
    norm_s: DevBuf,
    qkv_s: Lin,
    proj_s: Lin,
    qn_s: DevBuf,
    kn_s: DevBuf,
    norm_t: DevBuf,
    qkv_t: Lin,
    proj_t: Lin,
    qn_t: DevBuf,
    kn_t: DevBuf,
    norm_m: DevBuf,
    w12: Lin,
    w3: Lin,
    ada: Lin,
    /// the graph bias's per-class rows: proj(embedding) [6, 8] for distances and for edge types, and their scales
    dtab: Vec<f32>,
    rtab: Vec<f32>,
    dscale: f32,
    rscale: f32,
}

struct Final {
    norm: DevBuf,
    attn: Mha,
    nq: DevBuf,
    nkv: DevBuf,
    root_out: Mlp,
    joint_out: Mlp,
    ada: Lin,
}

/// The weights of the denoiser
pub struct Net {
    cond_emb: HostLin,
    time1: HostLin,
    time2: HostLin,
    root_tpos: Mlp,
    root_x: Mlp,
    joint_tpos: Mlp,
    parent_tpos: Mlp,
    tpos_fuse: Mlp,
    joint_x: Mlp,
    pool: Pool,
    name_emb: HostLin,
    depth_tab: Vec<f32>,
    sign: Sign,
    blocks: Vec<Block>,
    fin: Final,
    cos_t: DevBuf,
    sin_t: DevBuf,
}

/// What a skeleton gives the model, padded to 71 joints (rows past `n` zero)
pub struct SkelIn {
    pub n: usize,
    /// [71, 12] the rest pose's features, normalized
    pub tpos: Vec<f32>,
    /// [71, 12] each joint's parent's row of `tpos`
    pub tpos_parents: Vec<f32>,
    pub depths: Vec<i32>,
    /// [71, 8]
    pub spectral: Vec<f32>,
    /// [71, 71] graph distances (0 to 5) and edge classes (0 to 5)
    pub graph_dist: Vec<i32>,
    pub relations: Vec<i32>,
    /// [71, 768] pooled T5 embeddings of the joints' names
    pub names_emb: Vec<f32>,
}

/// What the model keeps of a skeleton across the flow's evaluations
pub struct Cond {
    pub n: usize,
    tpos_emb: DevBuf,
    depth: DevBuf,
    name: DevBuf,
    /// the rest pose's pooled embedding [512] (added to the modulation vector)
    pub y_static: Vec<f32>,
    cos: DevBuf,
    sin: DevBuf,
    bias: Vec<DevBuf>,
}

/// Intermediate values of a pass, for the checks
#[derive(Default)]
pub struct Trace {
    pub y: Vec<f32>,
    pub tokens0: Vec<f32>,
    pub block0: Vec<f32>,
    pub block_last: Vec<f32>,
    pub tpos_emb: Vec<f32>,
}

/// The buffers of a pass
pub struct Work {
    x: DevBuf,
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    o: DevBuf,
    w12: DevBuf,
    g: DevBuf,
    mods: DevBuf,
    ys: DevBuf,
    jin: DevBuf,
    rin: DevBuf,
    jtmp: DevBuf,
    jemb: DevBuf,
    rtmp: DevBuf,
    remb: DevBuf,
    kk: DevBuf,
    vv: DevBuf,
    rq: DevBuf,
    jout: DevBuf,
    rout: DevBuf,
}

fn rows(b: &DevBuf, row: usize) -> Rows<'_> {
    Rows::new(b, row)
}

fn at(b: &DevBuf, at: usize, row: usize) -> Rows<'_> {
    Rows { buf: b, at, row }
}

impl Net {
    pub fn load(ops: &Ops, mut s: Store) -> Result<Net> {
        let s = &mut s;
        let lin = |s: &mut Store, name: &str, n, k| s.host_lin(name, n, k, true);
        let cond_emb = lin(s, "cond_embedder", DIM, TEXT)?;
        let time1 = lin(s, "time_embedder.mlp.0", DIM, 256)?;
        let time2 = lin(s, "time_embedder.mlp.2", DIM, DIM)?;
        let root_tpos = Mlp::load(s, "input_layer.root_tpos_embedder", FEAT, DIM, DIM)?;
        let root_x = Mlp::load(s, "input_layer.root_x_embedder", FEAT, DIM, DIM)?;
        let joint_tpos = Mlp::load(s, "input_layer.joint_tpos_embedder", FEAT, DIM, DIM)?;
        let parent_tpos = Mlp::load(s, "input_layer.parent_tpos_embedder", FEAT, DIM, DIM)?;
        let tpos_fuse = Mlp::load(s, "input_layer.tpos_fuse", 2 * DIM, DIM, DIM)?;
        let joint_x = Mlp::load(s, "input_layer.joint_x_embedder", FEAT, DIM, DIM)?;
        let queries = s.take("tpos_pool.pool.queries", &[1, QUERIES, DIM])?;
        let pool = Pool {
            queries,
            norm_q: s.dev("tpos_pool.pool.norm_q.weight", &[DIM])?,
            norm_kv: s.dev("tpos_pool.pool.norm_kv.weight", &[DIM])?,
            attn: Mha::load(s, "tpos_pool.pool.cross_attn")?,
            norm_ff: s.dev("tpos_pool.pool.norm_ff.weight", &[DIM])?,
            w12: s.lin("tpos_pool.pool.ffn.w12", 2 * 2 * DIM, DIM, true)?,
            w3: s.lin("tpos_pool.pool.ffn.w3", DIM, 2 * DIM, true)?,
            norm_out: s.dev("tpos_pool.pool.norm_out.weight", &[DIM])?,
            out_proj: s.lin("tpos_pool.pool.out_proj", DIM, DIM, true)?,
        };
        let name_emb = lin(s, "joint_name_embedder", DIM, TEXT)?;
        let depth_tab = s.take("depth_embedding.weight", &[DEPTHS, DIM])?;
        let sign = Sign {
            phi0_w: s.take("rope_j.spectral_encoder.phi.0.weight", &[SIGN_HIDDEN, 1])?,
            phi0_b: s.take("rope_j.spectral_encoder.phi.0.bias", &[SIGN_HIDDEN])?,
            phi2: s.lin("rope_j.spectral_encoder.phi.2", SIGN_HIDDEN, SIGN_HIDDEN, true)?,
            rho0: s.lin("rope_j.spectral_encoder.rho.0", SIGN_HIDDEN, FREQS * SIGN_HIDDEN, true)?,
            rho2: s.lin("rope_j.spectral_encoder.rho.2", HEAD_DIM / 2, SIGN_HIDDEN, true)?,
        };
        let mut blocks = Vec::new();
        for b in 0..LAYERS {
            let p = format!("transformer_blocks.{b}");
            let scalar = |s: &mut Store, n: String| -> Result<f32> { Ok(s.take(&n, &[1])?[0]) };
            let table = |s: &mut Store, emb: &str, proj: &str| -> Result<Vec<f32>> {
                let e = s.take(&format!("{p}.s_attn.{emb}.weight"), &[6, 128])?;
                let l = s.host_lin(&format!("{p}.s_attn.{proj}"), HEADS, 128, true)?;
                Ok(l.apply(&e, 6))
            };
            let dtab = table(s, "graph_dist_embedding", "graph_dist_proj")?;
            let rtab = table(s, "graph_rel_embedding", "graph_rel_proj")?;
            blocks.push(Block {
                norm_s: s.dev(&format!("{p}.norm_s.weight"), &[DIM])?,
                qkv_s: s.lin(&format!("{p}.s_attn.qkv"), 3 * DIM, DIM, true)?,
                proj_s: s.lin(&format!("{p}.s_attn.proj"), DIM, DIM, true)?,
                qn_s: s.dev(&format!("{p}.s_attn.q_norm.weight"), &[HEAD_DIM])?,
                kn_s: s.dev(&format!("{p}.s_attn.k_norm.weight"), &[HEAD_DIM])?,
                norm_t: s.dev(&format!("{p}.norm_t.weight"), &[DIM])?,
                qkv_t: s.lin(&format!("{p}.t_attn.qkv"), 3 * DIM, DIM, true)?,
                proj_t: s.lin(&format!("{p}.t_attn.proj"), DIM, DIM, true)?,
                qn_t: s.dev(&format!("{p}.t_attn.q_norm.weight"), &[HEAD_DIM])?,
                kn_t: s.dev(&format!("{p}.t_attn.k_norm.weight"), &[HEAD_DIM])?,
                norm_m: s.dev(&format!("{p}.norm_mlp.weight"), &[DIM])?,
                w12: s.lin(&format!("{p}.mlp.w12"), 2 * FFN, DIM, true)?,
                w3: s.lin(&format!("{p}.mlp.w3"), DIM, FFN, true)?,
                ada: s.lin(&format!("{p}.adaLN_modulation.1"), 9 * DIM, DIM, true)?,
                dscale: scalar(s, format!("{p}.s_attn.graph_dist_scale"))?,
                rscale: scalar(s, format!("{p}.s_attn.graph_rel_scale"))?,
                dtab,
                rtab,
            });
        }
        let fin = Final {
            norm: s.dev("final_layer.norm_final.weight", &[DIM])?,
            attn: Mha::load(s, "final_layer.root_cross_attn")?,
            nq: s.dev("final_layer.root_cross_norm_q.weight", &[DIM])?,
            nkv: s.dev("final_layer.root_cross_norm_kv.weight", &[DIM])?,
            root_out: Mlp::load(s, "final_layer.root_out", DIM, DIM, FEAT)?,
            joint_out: Mlp::load(s, "final_layer.joint_out", DIM, DIM, FEAT)?,
            ada: s.lin("final_layer.adaLN_modulation.1", 2 * DIM, DIM, true)?,
        };
        if let Some(left) = s.map.keys().next() {
            return Err(Error(format!("the checkpoint has a tensor the model does not use: {left}")));
        }
        // the temporal rotary table: base (int(8 L / pi) // 100 + 1) 100 for L = 61 frames
        let base = (((8.0 * FRAMES as f64 / std::f64::consts::PI) as i64 / 100 + 1) * 100) as f32;
        let (mut cos, mut sin) = (vec![0f32; FRAMES * HEAD_DIM], vec![0f32; FRAMES * HEAD_DIM]);
        for t in 0..FRAMES {
            for i in 0..HEAD_DIM / 2 {
                let inv = 1.0 / base.powf((2 * i) as f32 / HEAD_DIM as f32);
                let a = t as f32 * inv;
                for h in [i, i + HEAD_DIM / 2] {
                    cos[t * HEAD_DIM + h] = a.cos();
                    sin[t * HEAD_DIM + h] = a.sin();
                }
            }
        }
        Ok(Net {
            cond_emb, time1, time2, root_tpos, root_x, joint_tpos, parent_tpos, tpos_fuse, joint_x, pool, name_emb, depth_tab, sign, blocks, fin,
            cos_t: ops.up(&cos)?,
            sin_t: ops.up(&sin)?,
        })
    }

    pub fn work(&self, ops: &Ops) -> Result<Work> {
        Ok(Work {
            x: ops.buf(TOK * DIM)?,
            h: ops.buf(TOK * DIM)?,
            qkv: ops.buf(TOK * 3 * DIM)?,
            att: ops.buf(TOK * DIM)?,
            o: ops.buf(TOK * DIM)?,
            w12: ops.buf(TOK * 2 * FFN)?,
            g: ops.buf(TOK * FFN)?,
            mods: ops.buf(9 * DIM)?,
            ys: ops.buf(DIM)?,
            jin: ops.buf(T * J * FEAT)?,
            rin: ops.buf(T * FEAT)?,
            jtmp: ops.buf(T * J * DIM)?,
            jemb: ops.buf(T * J * DIM)?,
            rtmp: ops.buf(T * DIM)?,
            remb: ops.buf(T * DIM)?,
            kk: ops.buf(TOK * DIM)?,
            vv: ops.buf(TOK * DIM)?,
            rq: ops.buf(FRAMES * DIM)?,
            jout: ops.buf(TOK * FEAT)?,
            rout: ops.buf(FRAMES * FEAT)?,
        })
    }

    /// The text's part of the modulation vector: cond_embedder(caption) [512]; for the empty guidance, the caption zeroed
    pub fn caption(&self, caption: &[f32], masked: bool) -> Vec<f32> {
        if masked {
            self.cond_emb.apply(&vec![0f32; TEXT], 1)
        } else {
            self.cond_emb.apply(caption, 1)
        }
    }

    /// The model's view of a skeleton
    pub fn cond(&self, ops: &Ops, s: &SkelIn) -> Result<Cond> {
        if s.tpos.len() != J * FEAT || s.depths.len() != J || s.spectral.len() != J * FREQS || s.graph_dist.len() != J * J || s.names_emb.len() != J * TEXT {
            return Err(Error("the skeleton's inputs are not padded to 71 joints".into()));
        }
        // the rest pose's tokens: the joints' own and their parents' features fused; the root's from its own embedder
        let tp = ops.up(&s.tpos)?;
        let pp = ops.up(&s.tpos_parents)?;
        let jp = self.joint_tpos.apply(ops, &tp, J)?;
        let pj = self.parent_tpos.apply(ops, &pp, J)?;
        let cat = ops.buf(J * 2 * DIM)?;
        ops.k.cat2(&jp, DIM, &pj, DIM, J, &cat)?;
        let tpos_emb = self.tpos_fuse.apply(ops, &cat, J)?;
        let root = self.root_tpos.apply(ops, &tp, 1)?;
        ops.k.copy_rows(&root, 0, 1, &tpos_emb, 0, 1, 1, DIM)?;

        // the pooled embedding of the rest pose
        let p = &self.pool;
        let queries = ops.up(&p.queries)?;
        let (qn, kv) = (ops.buf(QUERIES * DIM)?, ops.buf(J * DIM)?);
        ops.k.rms(rows(&queries, DIM), QUERIES, 1, DIM, EPS, &p.norm_q, None, rows(&qn, DIM))?;
        ops.k.rms(rows(&tpos_emb, DIM), J, 1, DIM, EPS, &p.norm_kv, None, rows(&kv, DIM))?;
        let (q, k, v, a, ao) = (ops.buf(QUERIES * DIM)?, ops.buf(J * DIM)?, ops.buf(J * DIM)?, ops.buf(QUERIES * DIM)?, ops.buf(QUERIES * DIM)?);
        ops.linear(&qn, QUERIES, &p.attn.q, &q)?;
        ops.linear(&kv, J, &p.attn.k, &k)?;
        ops.linear(&kv, J, &p.attn.v, &v)?;
        let at_ = Attn { groups: 1, sq: QUERIES, skv: J, heads: POOL_HEADS, d: DIM / POOL_HEADS, q_gs: 0, q_ss: 1, kv_gs: 0, kv_ss: 1, nvalid: s.n,
                         scale: 1.0 / ((DIM / POOL_HEADS) as f32).sqrt() };
        ops.k.attention(rows(&q, DIM), rows(&k, DIM), rows(&v, DIM), at_, None, rows(&a, DIM))?;
        ops.linear(&a, QUERIES, &p.attn.out, &ao)?;
        let qb = ops.up(&p.queries)?;
        ops.k.gated_add(&qb, &ao, QUERIES, DIM, None)?;
        let (hn, w12, gg, w3) = (ops.buf(QUERIES * DIM)?, ops.buf(QUERIES * 4 * DIM)?, ops.buf(QUERIES * 2 * DIM)?, ops.buf(QUERIES * DIM)?);
        ops.k.rms(rows(&qb, DIM), QUERIES, 1, DIM, EPS, &p.norm_ff, None, rows(&hn, DIM))?;
        ops.linear(&hn, QUERIES, &p.w12, &w12)?;
        ops.k.glu(&w12, QUERIES, 2 * DIM, SWIGLU, &gg)?;
        ops.linear(&gg, QUERIES, &p.w3, &w3)?;
        ops.k.gated_add(&qb, &w3, QUERIES, DIM, None)?;
        let qh = qb.to_f32()?;
        let mut pooled = vec![0f32; DIM];
        for r in qh.chunks_exact(DIM) {
            for (o, x) in pooled.iter_mut().zip(r) {
                *o += x;
            }
        }
        for o in &mut pooled {
            *o /= QUERIES as f32;
        }
        let pb = ops.up(&pooled)?;
        let (pn, py) = (ops.buf(DIM)?, ops.buf(DIM)?);
        ops.k.rms(rows(&pb, DIM), 1, 1, DIM, EPS, &p.norm_out, None, rows(&pn, DIM))?;
        ops.linear(&pn, 1, &p.out_proj, &py)?;
        let y_static = py.to_f32()?;

        // the depths' and names' embeddings, added to every frame's tokens
        let mut depth = Vec::with_capacity(J * DIM);
        for d in &s.depths {
            let d = (*d).clamp(0, DEPTHS as i32 - 1) as usize;
            depth.extend_from_slice(&self.depth_tab[d * DIM..(d + 1) * DIM]);
        }
        let name = self.name_emb.apply(&s.names_emb, J);

        // the spectral rotary angles: SignNet over each joint's eigenvector entries (phi(v) + phi(-v), then rho)
        let mut pa = vec![0f32; J * FREQS * SIGN_HIDDEN];
        let mut pb2 = vec![0f32; J * FREQS * SIGN_HIDDEN];
        for (i, v) in s.spectral.iter().enumerate() {
            for o in 0..SIGN_HIDDEN {
                pa[i * SIGN_HIDDEN + o] = self.sign.phi0_w[o] * v + self.sign.phi0_b[o];
                pb2[i * SIGN_HIDDEN + o] = self.sign.phi0_w[o] * -v + self.sign.phi0_b[o];
            }
        }
        let (ba, bb) = (ops.up(&pa)?, ops.up(&pb2)?);
        let (ra, rb) = (ops.buf(J * FREQS * SIGN_HIDDEN)?, ops.buf(J * FREQS * SIGN_HIDDEN)?);
        ops.k.act(&ba, J * FREQS * SIGN_HIDDEN, GELU)?;
        ops.k.act(&bb, J * FREQS * SIGN_HIDDEN, GELU)?;
        ops.linear(&ba, J * FREQS, &self.sign.phi2, &ra)?;
        ops.linear(&bb, J * FREQS, &self.sign.phi2, &rb)?;
        ops.k.gated_add(&ra, &rb, J * FREQS, SIGN_HIDDEN, None)?;
        let (r0, r2) = (ops.buf(J * SIGN_HIDDEN)?, ops.buf(J * HEAD_DIM / 2)?);
        ops.linear(&ra, J, &self.sign.rho0, &r0)?;
        ops.k.act(&r0, J * SIGN_HIDDEN, GELU)?;
        ops.linear(&r0, J, &self.sign.rho2, &r2)?;
        let ang = r2.to_f32()?;
        let half = HEAD_DIM / 2;
        let (mut cos, mut sin) = (vec![0f32; J * HEAD_DIM], vec![0f32; J * HEAD_DIM]);
        for j in 0..J {
            for i in 0..half {
                let a = ang[j * half + i];
                for h in [i, i + half] {
                    cos[j * HEAD_DIM + h] = a.cos();
                    sin[j * HEAD_DIM + h] = a.sin();
                }
            }
        }

        // each block's bias [8, 71, 71]: (proj(dist embedding) * scale) + (proj(edge embedding) * scale)
        let mut bias = Vec::new();
        for b in &self.blocks {
            let mut v = vec![0f32; HEADS * J * J];
            for h in 0..HEADS {
                for ij in 0..J * J {
                    let d = (s.graph_dist[ij].clamp(0, 5)) as usize;
                    let r = (s.relations[ij].clamp(0, 5)) as usize;
                    v[h * J * J + ij] = b.dtab[d * HEADS + h] * b.dscale + b.rtab[r * HEADS + h] * b.rscale;
                }
            }
            bias.push(ops.up(&v)?);
        }
        Ok(Cond { n: s.n, tpos_emb, depth: ops.up(&depth)?, name: ops.up(&name)?, y_static, cos: ops.up(&cos)?, sin: ops.up(&sin)?, bias })
    }

    /// T's embedding [512]: sinusoids (cos, sin) through the two-layer MLP
    fn time(&self, t: f32) -> Vec<f32> {
        let half = 128usize;
        let mut e = vec![0f32; 2 * half];
        let ln_period = (10000f64).ln() as f32;
        for i in 0..half {
            let f = (-ln_period * i as f32 / half as f32).exp();
            let a = t * f;
            e[i] = a.cos();
            e[half + i] = a.sin();
        }
        let mut h = self.time1.apply(&e, 1);
        for v in &mut h {
            *v = *v / (1.0 + (-*v).exp());
        }
        self.time2.apply(&h, 1)
    }

    /// One pass: x [71, 12, 60] at time t with the text's modulation `cap` (`caption`) -> the velocity [71, 12, 60]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, ops: &Ops, w: &Work, c: &Cond, cap: &[f32], x: &[f32], t: f32, mut trace: Option<&mut Trace>) -> Result<Vec<f32>> {
        if x.len() != J * FEAT * T {
            return Err(Error(format!("the state has {} values, not {}", x.len(), J * FEAT * T)));
        }
        let k = &ops.k;
        // the modulation vector: (time + text) + the rest pose's embedding
        let temb = self.time(t);
        let y: Vec<f32> = (0..DIM).map(|i| (temb[i] + cap[i]) + c.y_static[i]).collect();
        w.ys.write(0, &y.iter().flat_map(|v| (v / (1.0 + (-v).exp())).to_le_bytes()).collect::<Vec<u8>>())?;

        // the input tokens
        let (mut jin, mut rin) = (vec![0f32; T * J * FEAT], vec![0f32; T * FEAT]);
        for j in 0..J {
            for d in 0..FEAT {
                for f in 0..T {
                    let v = x[(j * FEAT + d) * T + f];
                    jin[(f * J + j) * FEAT + d] = v;
                    if j == 0 {
                        rin[f * FEAT + d] = v;
                    }
                }
            }
        }
        w.jin.write(0, &jin.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        w.rin.write(0, &rin.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        self.joint_x.run(ops, &w.jin, T * J, &w.jtmp, &w.jemb)?;
        self.root_x.run(ops, &w.rin, T, &w.rtmp, &w.remb)?;
        k.tokens(&c.tpos_emb, &w.jemb, &w.remb, &c.depth, &c.name, T, J, DIM, &w.x)?;
        if let Some(t) = trace.as_deref_mut() {
            t.y = y;
            t.tokens0 = w.x.to_f32()?;
            t.tpos_emb = c.tpos_emb.to_f32()?;
        }

        let qkv_row = 3 * DIM;
        for (bi, b) in self.blocks.iter().enumerate() {
            ops.linear(&w.ys, 1, &b.ada, &w.mods)?;
            let md = |i: usize| at(&w.mods, i * DIM, DIM);
            // graph attention among the joints of each frame
            k.rms(rows(&w.x, DIM), TOK, 1, DIM, EPS, &b.norm_s, Some((md(1), md(0))), rows(&w.h, DIM))?;
            ops.linear(&w.h, TOK, &b.qkv_s, &w.qkv)?;
            k.rms(rows(&w.qkv, qkv_row), TOK, HEADS, HEAD_DIM, EPS, &b.qn_s, None, rows(&w.qkv, qkv_row))?;
            k.rms(at(&w.qkv, DIM, qkv_row), TOK, HEADS, HEAD_DIM, EPS, &b.kn_s, None, at(&w.qkv, DIM, qkv_row))?;
            k.rope(&w.qkv, qkv_row, DIM, 2, TOK, HEADS, HEAD_DIM, 1, J, &c.cos, &c.sin)?;
            let a = Attn { groups: FRAMES, sq: J, skv: J, heads: HEADS, d: HEAD_DIM, q_gs: J, q_ss: 1, kv_gs: J, kv_ss: 1, nvalid: c.n, scale: 1.0 / (HEAD_DIM as f32).sqrt() };
            k.attention(rows(&w.qkv, qkv_row), at(&w.qkv, DIM, qkv_row), at(&w.qkv, 2 * DIM, qkv_row), a, Some(&c.bias[bi]), rows(&w.att, DIM))?;
            ops.linear(&w.att, TOK, &b.proj_s, &w.o)?;
            k.gated_add(&w.x, &w.o, TOK, DIM, Some(md(2)))?;
            // attention among the frames of each joint
            k.rms(rows(&w.x, DIM), TOK, 1, DIM, EPS, &b.norm_t, Some((md(4), md(3))), rows(&w.h, DIM))?;
            ops.linear(&w.h, TOK, &b.qkv_t, &w.qkv)?;
            k.rms(rows(&w.qkv, qkv_row), TOK, HEADS, HEAD_DIM, EPS, &b.qn_t, None, rows(&w.qkv, qkv_row))?;
            k.rms(at(&w.qkv, DIM, qkv_row), TOK, HEADS, HEAD_DIM, EPS, &b.kn_t, None, at(&w.qkv, DIM, qkv_row))?;
            k.rope(&w.qkv, qkv_row, DIM, 2, TOK, HEADS, HEAD_DIM, J, FRAMES, &self.cos_t, &self.sin_t)?;
            let a = Attn { groups: J, sq: FRAMES, skv: FRAMES, heads: HEADS, d: HEAD_DIM, q_gs: 1, q_ss: J, kv_gs: 1, kv_ss: J, nvalid: FRAMES, scale: 1.0 / (HEAD_DIM as f32).sqrt() };
            k.attention(rows(&w.qkv, qkv_row), at(&w.qkv, DIM, qkv_row), at(&w.qkv, 2 * DIM, qkv_row), a, None, rows(&w.att, DIM))?;
            ops.linear(&w.att, TOK, &b.proj_t, &w.o)?;
            k.gated_add(&w.x, &w.o, TOK, DIM, Some(md(5)))?;
            // the MLP
            k.rms(rows(&w.x, DIM), TOK, 1, DIM, EPS, &b.norm_m, Some((md(7), md(6))), rows(&w.h, DIM))?;
            ops.linear(&w.h, TOK, &b.w12, &w.w12)?;
            k.glu(&w.w12, TOK, FFN, SWIGLU, &w.g)?;
            ops.linear(&w.g, TOK, &b.w3, &w.o)?;
            k.gated_add(&w.x, &w.o, TOK, DIM, Some(md(8)))?;
            if let Some(t) = trace.as_deref_mut() {
                if bi == 0 {
                    t.block0 = w.x.to_f32()?;
                }
                if bi == LAYERS - 1 {
                    t.block_last = w.x.to_f32()?;
                }
            }
        }

        // the last layer: modulate, the root reads the joints, the heads
        let f = &self.fin;
        ops.linear(&w.ys, 1, &f.ada, &w.mods)?;
        k.rms(rows(&w.x, DIM), TOK, 1, DIM, EPS, &f.norm, Some((at(&w.mods, DIM, DIM), at(&w.mods, 0, DIM))), rows(&w.h, DIM))?;
        // x is free now: the root tokens (every J-th row of h) are copied out; its rows are normed in place for the keys
        k.copy_rows(&w.h, 0, J, &w.rq, 0, 1, FRAMES, DIM)?;
        let qn = &w.att;
        k.rms(rows(&w.rq, DIM), FRAMES, 1, DIM, EPS, &f.nq, None, rows(qn, DIM))?;
        let q = &w.o;
        ops.linear(qn, FRAMES, &f.attn.q, q)?;
        k.rms(rows(&w.h, DIM), TOK, 1, DIM, EPS, &f.nkv, None, rows(&w.x, DIM))?;
        ops.linear(&w.x, TOK, &f.attn.k, &w.kk)?;
        ops.linear(&w.h, TOK, &f.attn.v, &w.vv)?;
        let a = Attn { groups: FRAMES, sq: 1, skv: J - 1, heads: POOL_HEADS, d: DIM / POOL_HEADS, q_gs: 1, q_ss: 0, kv_gs: J, kv_ss: 1, nvalid: c.n.saturating_sub(1), scale: 1.0 / ((DIM / POOL_HEADS) as f32).sqrt() };
        k.attention(rows(q, DIM), at(&w.kk, DIM, DIM), at(&w.vv, DIM, DIM), a, None, rows(qn, DIM))?;
        ops.linear(qn, FRAMES, &f.attn.out, q)?;
        k.gated_add(&w.rq, q, FRAMES, DIM, None)?;
        f.root_out.run(ops, &w.rq, FRAMES, qn, &w.rout)?;
        f.joint_out.run(ops, &w.h, TOK, &w.g, &w.jout)?;
        let (jo, ro) = (w.jout.to_f32()?, w.rout.to_f32()?);
        let mut out = vec![0f32; J * FEAT * T];
        for j in 0..J {
            for d in 0..FEAT {
                for fr in 0..T {
                    out[(j * FEAT + d) * T + fr] = if j == 0 { ro[(fr + 1) * FEAT + d] } else { jo[((fr + 1) * J + j) * FEAT + d] };
                }
            }
        }
        Ok(out)
    }
}
