//! The backbone: 11 layers of 1024 (gated DeltaNet mixers, a global-attention layer in four), Full AttnRes in place of
//! the residual stream, a dense SwiGLU FFN in layer 0 and a top-2-of-64 latent MoE after it; ten 4098-way heads,
//! soft-capped.
//!
//! Origin: Zyphra's `zonos3.reference.lm.Zonos3LM.forward` and the ggml server's `build_graph`
//! (github.com/Zyphra/ZONOS3-Mini, MIT; the model Apache-2.0), over the ggml server's GGUF (`general.architecture`
//! zonos3): every sublayer output is a source of the AttnRes bank, and the input of every sublayer (and of the output
//! norm) is the softmax-over-depth mix of the bank's sources scored by that sublayer's learned query against the
//! sources' weightless RMS norms. Activations are float32; the mixers' and the head's matrices are made float32 on the
//! card (oneMKL), the experts stay in their Q4_0 blocks.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ops::{e, fp, nulf, Mat, Ops, Weights};

pub const ARCH: &str = "zonos3";

#[derive(Clone, Debug)]
pub struct Config {
    pub layers: usize,
    pub ctx: usize,
    pub dim: usize,
    pub head_dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    pub experts: usize,
    pub topk: usize,
    pub ncb: usize,
    pub codebook: usize,
    pub audio_vocab: usize,
    pub eoa: u32,
    pub pad: u32,
    pub text_vocab: u32,
    pub text_rows: usize,
    pub gdn_heads: usize,
    pub gdn_dim: usize,
    pub gdn_kernel: usize,
    pub speaker_dim: usize,
    pub cond_names: Vec<String>,
    pub cond_counts: Vec<usize>,
    pub cond_edges: Vec<Vec<f64>>,
    pub theta: f32,
    pub eps: f32,
    pub qk_eps: f32,
    pub softcap: f32,
    pub gdn_period: usize,
    pub silence: Vec<u32>,
    pub tokenizer: String,
    /// text_tokenizer_spec (JSON): the phoneme, start and pause rows an edit backbone adds
    pub spec: String,
    pub bos: u32,
    pub eos: u32,
    pub offset: u32,
}

impl Config {
    pub fn read(w: &Weights) -> Result<Config> {
        if w.g.architecture() != ARCH {
            return Err(Error(format!("{}: general.architecture is {}, not {ARCH}", w.name(), w.g.architecture())));
        }
        let k = |s: &str| format!("{ARCH}.{s}");
        for (key, want) in [("residual.type", "full_attnres"), ("moe.balancing", "quantile")] {
            let v = w.s(&k(key))?;
            if v != want {
                return Err(Error(format!("{}: {key} {v}, this engine runs {want}", w.name())));
            }
        }
        let pattern = w.s(&k("attention.pattern"))?;
        let gdn_period = pattern.strip_prefix("gdn").and_then(|r| r.strip_suffix("_global1")).and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| Error(format!("{}: attention pattern {pattern}, this engine runs gdnN_global1", w.name())))? + 1;
        let names = w.ss(&k("condition.names"))?;
        let counts = match w.us(&k("condition.bucket_counts")) {
            Ok(c) => c,
            Err(_) => vec![w.u(&k("condition.bucket_count"))?; names.len()],
        };
        let edges = names.iter().map(|n| w.fs(&k(&format!("condition.edges.{n}")))).collect::<Result<Vec<_>>>()?;
        let used = w.us(&k("expert_used_count_per_layer"))?;
        let text_vocab = w.u(&k("text_vocab"))? as u32;
        let text_rows = w.shape("text_embd.weight")?[0];
        Ok(Config {
            layers: w.u(&k("block_count"))?,
            ctx: w.u(&k("context_length"))?,
            dim: w.u(&k("embedding_length"))?,
            head_dim: w.u(&k("head_dim"))?,
            heads: w.u(&k("attention.head_count"))?,
            kv_heads: w.u(&k("attention.head_count_kv"))?,
            ffn: w.u(&k("feed_forward_length"))?,
            experts: w.u(&k("expert_count"))?,
            topk: used.iter().copied().max().unwrap_or(2),
            ncb: w.u(&k("n_codebooks"))?,
            codebook: w.u(&k("codebook_size"))?,
            audio_vocab: w.u(&k("audio_vocab"))?,
            eoa: w.u(&k("eoa_id"))? as u32,
            pad: w.u(&k("audio_pad_id"))? as u32,
            text_vocab,
            text_rows,
            gdn_heads: w.u(&k("gdn.head_count"))?,
            gdn_dim: w.u(&k("gdn.head_dim"))?,
            gdn_kernel: w.u(&k("gdn.conv_kernel"))?,
            speaker_dim: w.u(&k("speaker.embedding_length"))?,
            cond_counts: counts,
            cond_edges: edges,
            cond_names: names,
            theta: w.f(&k("rope.freq_base"))? as f32,
            eps: w.f(&k("attention.layer_norm_rms_epsilon"))? as f32,
            qk_eps: w.f(&k("attention.qk_norm_epsilon"))? as f32,
            softcap: w.f(&k("logit_softcap"))? as f32,
            gdn_period,
            silence: w.us(&k("silence_frame"))?.into_iter().map(|v| v as u32).collect(),
            tokenizer: w.s(&k("tokenizer.json"))?,
            spec: w.s(&k("text_tokenizer_spec")).unwrap_or_default(),
            bos: w.u(&k("tokenizer.bos"))? as u32,
            eos: w.u(&k("tokenizer.eos"))? as u32,
            offset: w.u(&k("tokenizer.offset"))? as u32,
        })
    }

    pub fn uses_gdn(&self, layer: usize) -> bool {
        !(layer + 1).is_multiple_of(self.gdn_period)
    }

    /// the AttnRes bank's sources (the embedding, then each sublayer's output) and queries
    pub fn sources(&self) -> usize {
        2 * self.layers + 1
    }

    pub fn row_width(&self) -> usize {
        self.ncb + 1
    }

    pub fn logits(&self) -> usize {
        self.ncb * self.audio_vocab
    }
}

enum Mixer {
    Gdn { proj: Mat, norm: DevBuf, a_log: DevBuf, dt_bias: DevBuf, conv: DevBuf },
    Attn { q: Mat, k: Mat, v: Mat, gate: Mat, temp: DevBuf },
}

enum Ffn {
    Dense { up: Mat, gate: Mat, down: Mat },
    Moe { router: Mat, router_norm: DevBuf, router_bias: DevBuf, latent_down: Mat, latent_up: Mat, gate: DevBuf, up: DevBuf, down: DevBuf },
}

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    mixer: Mixer,
    out: Mat,
    ffn: Ffn,
}

pub struct Lm {
    pub c: Config,
    audio: DevBuf,
    text: DevBuf,
    queries: DevBuf,
    out_norm: DevBuf,
    head: Mat,
    layers: Vec<Layer>,
    /// the speaker projection on the host: [dim, speaker_dim] and its bias
    spk_w: Vec<f32>,
    spk_b: Vec<f32>,
    /// the condition tables on the host: table i [counts[i] + 1, dim]
    cond: Vec<Vec<f32>>,
    pub bytes: usize,
    /// the speech-edit LoRA (edit/accents.gguf), applied on a sequence made with `edit`
    pub edit: Option<EditLora>,
}

/// The speech-edit adapter (zonos-accent-lora/3, rank 64; rank 16 on the experts) at strength 1: every adapted linear
/// adds up @ (down @ x), every routed (row, expert) pair its experts' factors, the MoE routers shift their balancing
/// biases (Zyphra's zonos3-accent.h; the ggml / MLX servers apply it at run time as here)
pub struct EditLora {
    linear: std::collections::HashMap<String, (Mat, Mat)>,
    /// per layer: the router bias with the adapter's shift
    router_bias: Vec<Option<DevBuf>>,
    /// per layer: gate / up / down experts' (down [E, R, in], up [E, out, R])
    experts: Vec<Option<[(DevBuf, DevBuf); 3]>>,
    rank: usize,
    erank: usize,
    max_out: usize,
    pub bytes: usize,
}

/// One sequence's state: its position, the global layers' caches, each GDN layer's recurrent state and convolution
/// history, and the scratch of a forward of up to `rows` rows
pub struct Seq {
    pub pos: usize,
    pub cap: usize,
    kc: Vec<DevBuf>,
    vc: Vec<DevBuf>,
    state: Vec<DevBuf>,
    hist: Vec<DevBuf>,
    rows: usize,
    s: Scratch,
    /// LACI's alignment probe (decode steps only)
    pub probe: Option<Probe>,
    /// run the speech-edit LoRA (made with `seq_edit`)
    pub edit: bool,
    lora: Option<LoraScratch>,
}

struct LoraScratch {
    t: DevBuf,
    u: DevBuf,
    tg: DevBuf,
    tu: DevBuf,
    td: DevBuf,
}

/// LACI's alignment probe (zonos3.tts.probe.readout; ggml build_probe): on every decode row, each probed head's
/// softmax over the text keys (BOS..EOS, inside its window when it has one), summed over the heads -> the reading
/// position (argmax) and its centroid. Masked heads (`windows`, the ACI mask while LACI is on probation) also have
/// their attention output recomputed with the text keys outside the window blocked.
pub struct Probe {
    pub text_start: usize,
    pub text_len: usize,
    /// (model layer, q head), each weighted 1
    pub heads: Vec<(usize, usize)>,
    /// (model layer, q head, lo, hi): inclusive text windows of the masked heads
    pub windows: Vec<(usize, usize, usize, usize)>,
    /// the last decode row's (argmax, centroid)
    pub readout: Option<(usize, f64)>,
    keys: Vec<Option<Vec<f32>>>,
    acc: Vec<f64>,
    spec: Option<DevBuf>,
}

impl Probe {
    pub fn new(text_start: usize, text_len: usize, heads: Vec<(usize, usize)>) -> Probe {
        Probe { text_start, text_len, heads, windows: Vec::new(), readout: None, keys: Vec::new(), acc: vec![0.0; text_len], spec: None }
    }

    /// clip_window: an inclusive window inside [0, text_len)
    fn window(&self, layer: usize, head: usize) -> (usize, usize) {
        match self.windows.iter().find(|w| w.0 == layer && w.1 == head) {
            Some(&(_, _, lo, hi)) => {
                let lo = lo.min(self.text_len - 1);
                (lo, hi.max(lo).min(self.text_len - 1))
            }
            None => (0, self.text_len - 1),
        }
    }
}

struct Scratch {
    ids: DevBuf,
    bank: DevBuf,
    scores: DevBuf,
    mixed: DevBuf,
    xin: DevBuf,
    proj: DevBuf,
    conv: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    decay: DevBuf,
    beta: DevBuf,
    y: DevBuf,
    gate: DevBuf,
    att: DevBuf,
    h1: DevBuf,
    h2: DevBuf,
    idx: DevBuf,
    wts: DevBuf,
    logits: DevBuf,
}

/// A forward's conditioning: rows (absolute positions) whose embedding is replaced (the speaker row: a projected
/// embedding; a condition row: its bucket's row), and a hidden steering delta added to AttnRes sources
#[derive(Clone, Debug, Default)]
pub struct Cond {
    pub speaker: Option<(usize, Vec<f32>)>,
    pub buckets: Vec<(usize, usize)>,
    /// (source index, delta [dim]) added to that source of every row of the forward
    pub deltas: Vec<(usize, Vec<f32>)>,
}

impl Lm {
    pub fn load(ops: &Ops, w: &Weights, log: &mut dyn FnMut(String)) -> Result<Lm> {
        let c = Config::read(w)?;
        if c.head_dim != 16 && c.head_dim != 32 && c.head_dim != 64 {
            return Err(Error(format!("{ARCH}: head size {}: this engine's attention takes 16, 32 or 64", c.head_dim)));
        }
        let mut bytes = 0usize;
        // the ten audio tables side by side (half), the text table
        let mut audio = Vec::with_capacity(c.ncb * c.audio_vocab * c.dim * 2);
        for cb in 0..c.ncb {
            let t = w.find(&format!("audio_embd.{cb}.weight"))?;
            if t.ty != nextsycl_gguf::GType::F16 || w.shape(&t.name)? != [c.audio_vocab, c.dim] {
                return Err(Error(format!("{}: audio_embd.{cb}.weight is not a half [{}, {}] table", w.name(), c.audio_vocab, c.dim)));
            }
            audio.extend(w.bytes(&t.name)?);
        }
        let audio_d = DevBuf::new(&ops.gpu, audio.len()).map_err(e)?;
        audio_d.write(0, &audio).map_err(e)?;
        let (text, tty) = w.dev_raw(ops, "text_embd.weight")?;
        if tty != nextsycl_gguf::GType::F16 {
            return Err(Error(format!("{}: text_embd.weight is not half", w.name())));
        }
        bytes += audio_d.len + text.len;
        let mat = |name: &str, bytes: &mut usize| -> Result<Mat> {
            let m = w.mat(ops, name)?;
            *bytes += m.bytes();
            Ok(m)
        };
        let mut layers = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            let p = |s: &str| format!("blk.{l}.{s}");
            let mixer = if c.uses_gdn(l) {
                Mixer::Gdn {
                    proj: mat(&p("gdn_in.weight"), &mut bytes)?,
                    norm: w.dev(ops, &p("gdn_norm.weight"))?,
                    a_log: w.dev(ops, &p("gdn_A_log"))?,
                    dt_bias: w.dev(ops, &p("gdn_dt_bias"))?,
                    conv: w.dev(ops, &p("gdn_conv.weight"))?,
                }
            } else {
                Mixer::Attn {
                    q: mat(&p("attn_q.weight"), &mut bytes)?,
                    k: mat(&p("attn_k.weight"), &mut bytes)?,
                    v: mat(&p("attn_v.weight"), &mut bytes)?,
                    gate: mat(&p("attn_gate.weight"), &mut bytes)?,
                    temp: w.dev(ops, &p("attn_temp"))?,
                }
            };
            let ffn = if w.has(&p("ffn_up.weight")) {
                Ffn::Dense { up: mat(&p("ffn_up.weight"), &mut bytes)?, gate: mat(&p("ffn_gate.weight"), &mut bytes)?, down: mat(&p("ffn_down.weight"), &mut bytes)? }
            } else {
                let raw = |name: &str, bytes: &mut usize| -> Result<DevBuf> {
                    let (b, ty) = w.dev_raw(ops, name)?;
                    if ty != nextsycl_gguf::GType::Q4_0 {
                        return Err(Error(format!("{}: {name} is {}, this engine reads its experts in Q4_0", w.name(), ty.name())));
                    }
                    *bytes += b.len;
                    Ok(b)
                };
                Ffn::Moe {
                    router: mat(&p("router.weight"), &mut bytes)?,
                    router_norm: w.dev(ops, &p("router_norm.weight"))?,
                    router_bias: w.dev(ops, &p("router_bias"))?,
                    latent_down: mat(&p("ffn_latent_down.weight"), &mut bytes)?,
                    latent_up: mat(&p("ffn_latent_up.weight"), &mut bytes)?,
                    gate: raw(&p("ffn_gate_exps.weight"), &mut bytes)?,
                    up: raw(&p("ffn_up_exps.weight"), &mut bytes)?,
                    down: raw(&p("ffn_down_exps.weight"), &mut bytes)?,
                }
            };
            layers.push(Layer { attn_norm: w.dev(ops, &p("attn_norm.weight"))?, ffn_norm: w.dev(ops, &p("ffn_norm.weight"))?, mixer,
                                out: mat(&p("attn_output.weight"), &mut bytes)?, ffn });
        }
        let latent = match &layers.iter().find_map(|l| match &l.ffn { Ffn::Moe { latent_down, .. } => Some(latent_down.n), _ => None }) {
            Some(n) => *n,
            None => 0,
        };
        if latent % 32 != 0 {
            return Err(Error(format!("{ARCH}: a latent width of {latent} is not whole Q4_0 blocks")));
        }
        let head = mat("output.weight", &mut bytes)?;
        let queries = w.dev(ops, "attnres_q.weight")?;
        if w.shape("attnres_q.weight")? != [c.sources(), c.dim] {
            return Err(Error(format!("{}: attnres_q.weight is not [{}, {}]", w.name(), c.sources(), c.dim)));
        }
        let cond = (0..c.cond_names.len()).map(|i| w.f32(&format!("cond_embd.{i}.weight"))).collect::<Result<Vec<_>>>()?;
        for (i, t) in cond.iter().enumerate() {
            if t.len() != (c.cond_counts[i] + 1) * c.dim {
                return Err(Error(format!("{}: cond_embd.{i}.weight holds {} values, {} buckets and the unknown row expected", w.name(), t.len(), c.cond_counts[i])));
            }
        }
        let lm = Lm {
            audio: audio_d,
            text,
            queries,
            out_norm: w.dev(ops, "output_norm.weight")?,
            head,
            layers,
            spk_w: w.f32("speaker_proj.weight")?,
            spk_b: w.f32("speaker_proj.bias")?,
            cond,
            bytes,
            c,
            edit: None,
        };
        log(format!("{ARCH}: backbone {} layers of {}, {:.0} MiB on the card", lm.c.layers, lm.c.dim, lm.bytes as f64 / (1u64 << 20) as f64));
        Ok(lm)
    }

    /// Loads the speech-edit adapter (accents.gguf's zonos-accent-lora/3 slot)
    pub fn attach_edit(&mut self, ops: &Ops, w: &Weights) -> Result<()> {
        let formats = w.ss("zonos3_accents.format")?;
        let slot = formats.iter().position(|f| f == "zonos-accent-lora/3").ok_or_else(|| Error(format!("{}: no speech-edit adapter (zonos-accent-lora/3)", w.name())))?;
        let p = format!("acc.{slot}.");
        let mut linear = std::collections::HashMap::new();
        let (mut rank, mut max_out, mut bytes) = (0usize, 0usize, 0usize);
        let mut bases: Vec<String> = Vec::new();
        for l in 0..self.c.layers {
            for b in ["attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "gdn_in", "ffn_up", "ffn_gate", "ffn_down", "ffn_latent_down", "ffn_latent_up", "router"] {
                bases.push(format!("blk.{l}.{b}"));
            }
        }
        bases.push("output".into());
        for b in bases {
            let dn = format!("{p}{b}.lora_down");
            if !w.has(&dn) {
                continue;
            }
            let down = w.mat(ops, &dn)?;
            let up = w.mat(ops, &format!("{p}{b}.lora_up"))?;
            rank = rank.max(down.n);
            max_out = max_out.max(up.n);
            bytes += down.bytes() + up.bytes();
            linear.insert(b, (down, up));
        }
        let mut router_bias = Vec::new();
        let mut experts = Vec::new();
        let mut erank = 0;
        for (l, ly) in self.layers.iter().enumerate() {
            let shift = format!("{p}blk.{l}.router_bias");
            match &ly.ffn {
                Ffn::Moe { router_bias: rb, .. } if w.has(&shift) => {
                    let base = rb.to_f32().map_err(e)?;
                    let sh = w.f32(&shift)?;
                    router_bias.push(Some(ops.from_f32(&base.iter().zip(&sh).map(|(a, b)| a + b).collect::<Vec<_>>())?));
                }
                _ => router_bias.push(None),
            }
            let e = |n: &str| -> Result<(DevBuf, DevBuf)> {
                Ok((w.dev(ops, &format!("{p}blk.{l}.{n}.lora_down"))?, w.dev(ops, &format!("{p}blk.{l}.{n}.lora_up"))?))
            };
            if w.has(&format!("{p}blk.{l}.ffn_gate_exps.lora_down")) {
                erank = w.shape(&format!("{p}blk.{l}.ffn_gate_exps.lora_down"))?[1];
                let f = [e("ffn_gate_exps")?, e("ffn_up_exps")?, e("ffn_down_exps")?];
                bytes += f.iter().map(|(a, b)| a.len + b.len).sum::<usize>();
                experts.push(Some(f));
            } else {
                experts.push(None);
            }
        }
        self.edit = Some(EditLora { linear, router_bias, experts, rank, erank, max_out, bytes });
        Ok(())
    }

    /// A sequence that runs the speech-edit adapter
    pub fn seq_edit(&self, ops: &Ops, cap: usize, rows: usize) -> Result<Seq> {
        let lora = self.edit.as_ref().ok_or_else(|| Error(format!("{ARCH}: speech editing needs the edit adapter (edit/accents.gguf)")))?;
        let mut sq = self.seq(ops, cap, rows)?;
        let r = rows.max(1);
        sq.lora = Some(LoraScratch {
            t: ops.f32(r * lora.rank.max(1))?,
            u: ops.f32(r * lora.max_out.max(1))?,
            tg: ops.f32(r * self.c.topk * lora.erank.max(1))?,
            tu: ops.f32(r * self.c.topk * lora.erank.max(1))?,
            td: ops.f32(r * self.c.topk * lora.erank.max(1))?,
        });
        sq.edit = true;
        Ok(sq)
    }

    /// A product, and the edit adapter's term of its module on an edit sequence
    #[allow(clippy::too_many_arguments)]
    fn lin(&self, ops: &Ops, sq: &Seq, base: &str, x: *const f32, n: usize, m: &Mat, bias: *const f32, out: *mut f32) -> Result<()> {
        ops.linear(x, n, m, bias, out)?;
        if sq.edit {
            if let (Some(l), Some(sc)) = (&self.edit, &sq.lora) {
                if let Some((d, u)) = l.linear.get(base) {
                    ops.linear(x, n, d, nulf(), sc.t.fp())?;
                    ops.linear(sc.t.fp(), n, u, nulf(), sc.u.fp())?;
                    ops.add(out, sc.u.fp(), n * u.n)?;
                }
            }
        }
        Ok(())
    }

    /// speaker_projection(v): the speaker row's embedding
    pub fn project_speaker(&self, v: &[f32]) -> Result<Vec<f32>> {
        let (d, s) = (self.c.dim, self.c.speaker_dim);
        if v.len() != s {
            return Err(Error(format!("a speaker embedding of {} values; this model takes {s}", v.len())));
        }
        Ok((0..d).map(|i| self.spk_b[i] + (0..s).map(|j| self.spk_w[i * s + j] * v[j]).sum::<f32>()).collect())
    }

    /// condition table `i`'s row for `bucket` (the table's count = unknown)
    pub fn condition_row(&self, i: usize, bucket: usize) -> Vec<f32> {
        let d = self.c.dim;
        self.cond[i][bucket * d..(bucket + 1) * d].to_vec()
    }

    /// A sequence able to hold `cap` positions, its forwards up to `rows` rows
    pub fn seq(&self, ops: &Ops, cap: usize, rows: usize) -> Result<Seq> {
        let c = &self.c;
        if cap > c.ctx {
            return Err(Error(format!("{cap} positions exceed the model's context of {}", c.ctx)));
        }
        let rows = rows.max(1);
        let kvw = c.kv_heads * c.head_dim;
        let (mut kc, mut vc, mut state, mut hist) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let inner = c.gdn_heads * c.gdn_dim;
        for l in 0..c.layers {
            if c.uses_gdn(l) {
                let s = ops.f32(c.gdn_heads * c.gdn_dim * c.gdn_dim)?;
                s.fill(0).map_err(e)?;
                let h = ops.f32(3 * inner * (c.gdn_kernel - 1))?;
                h.fill(0).map_err(e)?;
                state.push(s);
                hist.push(h);
            } else {
                kc.push(ops.f32(cap * kvw)?);
                vc.push(ops.f32(cap * kvw)?);
            }
        }
        let proj_w = self.layers.iter().filter_map(|l| match &l.mixer { Mixer::Gdn { proj, .. } => Some(proj.n), _ => None }).max().unwrap_or(0);
        let qw = c.heads * c.head_dim;
        let ffw = self.layers.iter().map(|l| match &l.ffn { Ffn::Dense { up, .. } => up.n, Ffn::Moe { up: _, latent_down, .. } => latent_down.n }).max().unwrap_or(0);
        let expert_f = self.layers.iter().find_map(|l| match &l.ffn {
            Ffn::Moe { gate, latent_down, .. } => Some(gate.len / 18 * 32 / c.experts / latent_down.n),
            _ => None,
        }).unwrap_or(0);
        let h1 = (rows * c.topk * expert_f).max(rows * c.ffn).max(rows * ffw);
        let s = Scratch {
            ids: DevBuf::new(&ops.gpu, rows * c.row_width() * 4).map_err(e)?,
            bank: ops.f32(c.sources() * rows * c.dim)?,
            scores: ops.f32(c.sources() * rows * c.sources())?,
            mixed: ops.f32(rows * c.dim)?,
            xin: ops.f32(rows * c.dim)?,
            proj: ops.f32(rows * proj_w.max(qw))?,
            conv: ops.f32(rows * 3 * inner)?,
            q: ops.f32(rows * inner.max(qw))?,
            k: ops.f32(rows * inner.max(kvw))?,
            v: ops.f32(rows * inner.max(kvw))?,
            decay: ops.f32(rows * c.gdn_heads)?,
            beta: ops.f32(rows * c.gdn_heads)?,
            y: ops.f32(rows * inner.max(qw))?,
            gate: ops.f32(rows * c.heads.max(c.experts))?,
            att: ops.f32(rows * inner.max(qw))?,
            h1: ops.f32(h1)?,
            h2: ops.f32(h1)?,
            idx: DevBuf::new(&ops.gpu, rows * c.topk * 4).map_err(e)?,
            wts: ops.f32(rows * c.topk)?,
            logits: ops.f32(rows * c.logits())?,
        };
        Ok(Seq { pos: 0, cap, kc, vc, state, hist, rows, s, probe: None, edit: false, lora: None })
    }

    /// The rows `ids` [n, ncb + 1] at the sequence's next positions: the state advanced, the logits of the last row (or
    /// of every row, `all`) left in the sequence's buffer, soft-capped: `logits(seq, n)` reads them
    pub fn forward(&self, ops: &Ops, sq: &mut Seq, ids: &[u32], cond: &Cond, all: bool) -> Result<usize> {
        let c = &self.c;
        let w = c.row_width();
        if ids.is_empty() || !ids.len().is_multiple_of(w) {
            return Err(Error(format!("rows of {w} ids expected, {} given", ids.len())));
        }
        let n = ids.len() / w;
        if n > sq.rows {
            return Err(Error(format!("a forward of {n} rows; this sequence was made for {}", sq.rows)));
        }
        if sq.pos + n > sq.cap {
            return Err(Error(format!("{} positions exceed the sequence's {}", sq.pos + n, sq.cap)));
        }
        let lo = sq.pos;
        let d = c.dim;
        let s = &sq.s;
        let ids_b: Vec<u8> = ids.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect();
        s.ids.write(0, &ids_b).map_err(e)?;
        let bank = |i: usize| fp(&s.bank, i * sq.rows * d);
        let scores = |i: usize| fp(&s.scores, i * sq.rows * c.sources());
        let nq = c.sources();
        // the embedding sum into source 0, its conditioning rows replaced, then the weightless norm
        ops.embed(&self.audio, c.ncb, c.audio_vocab, &self.text, c.text_rows, &s.ids, n, d, fp(&s.xin, 0))?;
        let put = |row: usize, v: &[f32]| -> Result<()> {
            if row >= lo && row < lo + n {
                let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                s.xin.write((row - lo) * d * 4, &b).map_err(e)?;
            }
            Ok(())
        };
        if let Some((row, v)) = &cond.speaker {
            put(*row, v)?;
        }
        for (i, (row, bucket)) in cond.buckets.iter().enumerate() {
            if i < self.cond.len() {
                put(*row, &self.condition_row(i, (*bucket).min(c.cond_counts[i])))?;
            }
        }
        let deltas: Vec<(usize, DevBuf)> = cond.deltas.iter().map(|(src, v)| Ok((*src, ops.from_f32(v)?))).collect::<Result<_>>()?;
        let add_source = |i: usize| -> Result<()> {
            for (src, v) in &deltas {
                if *src == i {
                    ops.add_vec(bank(i), n, d, v.fp())?;
                }
            }
            ops.attnres_score(bank(i), n, d, self.queries.fp(), nq, c.eps, scores(i))?;
            // the mix of sources 0..=i for query i
            ops.attnres_mix(fp(&s.bank, 0), fp(&s.scores, 0), i + 1, sq.rows, n, d, nq, i, fp(&s.mixed, 0))
        };
        ops.rms(fp(&s.xin, 0), d, n, d, nulf(), c.eps, bank(0))?;
        add_source(0)?;
        let (mut gi, mut ai) = (0usize, 0usize);
        for (l, ly) in self.layers.iter().enumerate() {
            ops.rms(fp(&s.mixed, 0), d, n, d, ly.attn_norm.fp(), c.eps, fp(&s.xin, 0))?;
            let x = fp(&s.xin, 0);
            match &ly.mixer {
                Mixer::Gdn { proj, norm, a_log, dt_bias, conv } => {
                    let (hh, sd) = (c.gdn_heads, c.gdn_dim);
                    let inner = hh * sd;
                    let p = proj.n;
                    self.lin(ops, sq, &format!("blk.{l}.gdn_in"), x, n, proj, nulf(), fp(&s.proj, 0))?;
                    ops.gdn_conv(fp(&s.proj, 0), n, p, 3 * inner, conv.fp(), c.gdn_kernel, sq.hist[gi].fp(), fp(&s.conv, 0))?;
                    ops.gdn_prep(fp(&s.conv, 0), fp(&s.proj, 0), n, p, 4 * inner, hh, sd, a_log.fp(), dt_bias.fp(), fp(&s.q, 0), fp(&s.k, 0), fp(&s.v, 0),
                                 fp(&s.decay, 0), fp(&s.beta, 0))?;
                    ops.gdn_scan(fp(&s.q, 0), fp(&s.k, 0), fp(&s.v, 0), fp(&s.decay, 0), fp(&s.beta, 0), n, hh, sd, sq.state[gi].fp(), fp(&s.y, 0))?;
                    ops.gdn_out(fp(&s.y, 0), fp(&s.proj, 0), p, 3 * inner, n, hh, sd, norm.fp(), 1e-5, fp(&s.att, 0))?;
                    gi += 1;
                }
                Mixer::Attn { q, k, v, gate, temp } => {
                    let (hq, hk, hd) = (c.heads, c.kv_heads, c.head_dim);
                    let kvw = hk * hd;
                    self.lin(ops, sq, &format!("blk.{l}.attn_gate"), x, n, gate, nulf(), fp(&s.gate, 0))?;
                    self.lin(ops, sq, &format!("blk.{l}.attn_q"), x, n, q, nulf(), fp(&s.q, 0))?;
                    self.lin(ops, sq, &format!("blk.{l}.attn_k"), x, n, k, nulf(), fp(&s.k, 0))?;
                    self.lin(ops, sq, &format!("blk.{l}.attn_v"), x, n, v, nulf(), fp(&s.v, 0))?;
                    ops.head_rms(fp(&s.q, 0), hq * hd, n, hq, hd, temp.fp(), c.qk_eps)?;
                    ops.head_rms(fp(&s.k, 0), kvw, n, hk, hd, nulf(), c.qk_eps)?;
                    ops.rope(fp(&s.q, 0), hq * hd, n, hq, hd, c.theta, lo)?;
                    ops.rope(fp(&s.k, 0), kvw, n, hk, hd, c.theta, lo)?;
                    sq.kc[ai].copy_within(lo * kvw * 4, &s.k, 0, n * kvw * 4).map_err(e)?;
                    sq.vc[ai].copy_within(lo * kvw * 4, &s.v, 0, n * kvw * 4).map_err(e)?;
                    ops.attention(fp(&s.q, 0), hq * hd, n, sq.kc[ai].fp(), sq.vc[ai].fp(), kvw, lo + n, hq, hk, hd, (hd as f32).powf(-0.5), lo, true, 0,
                                  fp(&s.att, 0))?;
                    if n == 1 {
                        if let Some(pr) = sq.probe.as_mut() {
                            probe_layer(ops, pr, l, ai, &s.q, &sq.kc[ai], &sq.vc[ai], &s.att, lo + 1, hq, hk, hd)?;
                        }
                    }
                    ops.head_gate(fp(&s.att, 0), fp(&s.gate, 0), n, hq, hd)?;
                    ai += 1;
                }
            }
            self.lin(ops, sq, &format!("blk.{l}.attn_output"), fp(&s.att, 0), n, &ly.out, nulf(), bank(2 * l + 1))?;
            add_source(2 * l + 1)?;
            ops.rms(fp(&s.mixed, 0), d, n, d, ly.ffn_norm.fp(), c.eps, fp(&s.xin, 0))?;
            let x = fp(&s.xin, 0);
            match &ly.ffn {
                Ffn::Dense { up, gate, down } => {
                    self.lin(ops, sq, &format!("blk.{l}.ffn_up"), x, n, up, nulf(), fp(&s.h1, 0))?;
                    self.lin(ops, sq, &format!("blk.{l}.ffn_gate"), x, n, gate, nulf(), fp(&s.h2, 0))?;
                    ops.silu_mul(fp(&s.h2, 0), fp(&s.h1, 0), fp(&s.h2, 0), n * up.n)?;
                    self.lin(ops, sq, &format!("blk.{l}.ffn_down"), fp(&s.h2, 0), n, down, nulf(), bank(2 * l + 2))?;
                }
                Ffn::Moe { router, router_norm, router_bias, latent_down, latent_up, gate, up, down } => {
                    let lat = latent_down.n;
                    let f = gate.len / 18 * 32 / c.experts / lat;
                    ops.rms(x, d, n, d, router_norm.fp(), c.eps, fp(&s.mixed, 0))?;
                    self.lin(ops, sq, &format!("blk.{l}.router"), fp(&s.mixed, 0), n, router, nulf(), fp(&s.gate, 0))?;
                    let edit = if sq.edit { self.edit.as_ref() } else { None };
                    let bias = edit.and_then(|e| e.router_bias[l].as_ref()).unwrap_or(router_bias);
                    ops.topk(fp(&s.gate, 0), bias.fp(), n, c.experts, c.topk, &s.idx, fp(&s.wts, 0))?;
                    self.lin(ops, sq, &format!("blk.{l}.ffn_latent_down"), x, n, latent_down, nulf(), fp(&s.y, 0))?;
                    match (edit.and_then(|e| e.experts[l].as_ref().map(|x| (x, e.erank))), &sq.lora) {
                        (Some(([(gd, gu), (ud, uu), (dd, du)], r)), Some(sc)) => {
                            ops.moe_lora_t(fp(&s.y, 0), n, lat, &s.idx, c.topk, false, gd, r, sc.tg.fp())?;
                            ops.moe_lora_t(fp(&s.y, 0), n, lat, &s.idx, c.topk, false, ud, r, sc.tu.fp())?;
                            ops.moe_up_lora(fp(&s.y, 0), n, lat, gate, up, &s.idx, c.topk, f, sc.tg.fp(), sc.tu.fp(), gu, uu, r, fp(&s.h1, 0))?;
                            ops.moe_lora_t(fp(&s.h1, 0), n, f, &s.idx, c.topk, true, dd, r, sc.td.fp())?;
                            ops.moe_down_lora(fp(&s.h1, 0), n, c.topk, f, down, &s.idx, fp(&s.wts, 0), lat, sc.td.fp(), du, r, fp(&s.h2, 0))?;
                        }
                        _ => {
                            ops.moe_up(fp(&s.y, 0), n, lat, gate, up, &s.idx, c.topk, f, fp(&s.h1, 0))?;
                            ops.moe_down(fp(&s.h1, 0), n, c.topk, f, down, &s.idx, fp(&s.wts, 0), lat, fp(&s.h2, 0))?;
                        }
                    }
                    self.lin(ops, sq, &format!("blk.{l}.ffn_latent_up"), fp(&s.h2, 0), n, latent_up, nulf(), bank(2 * l + 2))?;
                }
            }
            add_source(2 * l + 2)?;
        }
        // the final mix (the last source's query) - already in `mixed` - through the output norm and the heads
        if n == 1 {
            if let Some(pr) = sq.probe.as_mut() {
                let acc = std::mem::replace(&mut pr.acc, vec![0.0; pr.text_len]);
                let mut best = 0;
                for (j, a) in acc.iter().enumerate() {
                    if *a > acc[best] {
                        best = j;
                    }
                }
                let total: f64 = acc.iter().sum();
                let m = acc.iter().enumerate().map(|(j, a)| j as f64 * a).sum::<f64>() / total.max(1e-30);
                pr.readout = Some((best, m));
            }
        }
        let (r0, m) = if all { (0, n) } else { (n - 1, 1) };
        ops.rms(fp(&s.mixed, r0 * d), d, m, d, self.out_norm.fp(), c.eps, fp(&s.xin, 0))?;
        self.lin(ops, sq, "output", fp(&s.xin, 0), m, &self.head, nulf(), fp(&s.logits, 0))?;
        ops.softcap(fp(&s.logits, 0), m * c.logits(), c.softcap)?;
        sq.pos = lo + n;
        Ok(m)
    }

    /// The q heads of the model's attention layers (layer ids)
    pub fn attention_layers(&self) -> Vec<usize> {
        (0..self.c.layers).filter(|l| !self.c.uses_gdn(*l)).collect()
    }

    /// The logits of the last forward's `m` rows: [m, ncb, audio_vocab]
    pub fn logits(&self, sq: &Seq, m: usize) -> Result<Vec<f32>> {
        let n = m * self.c.logits();
        let mut v = vec![0f32; n];
        // SAFETY: v is n floats, viewed as their bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * 4) };
        sq.s.logits.read(0, bytes).map_err(e)?;
        Ok(v)
    }
}

/// One attention layer's part of the probe on a decode row (keys 0..keys-1): the masked heads recomputed, the probed
/// heads' text distributions added to the readout
#[allow(clippy::too_many_arguments)]
fn probe_layer(ops: &Ops, pr: &mut Probe, layer: usize, slot: usize, q: &DevBuf, kc: &DevBuf, vc: &DevBuf, att: &DevBuf, keys: usize, hq: usize, hk: usize,
               hd: usize) -> Result<()> {
    let kvw = hk * hd;
    let scale = (hd as f32).powf(-0.5);
    let (ts, tl) = (pr.text_start, pr.text_len);
    let masked: Vec<(usize, usize, usize)> = pr.windows.iter().filter(|w| w.0 == layer).map(|w| {
        let (lo, hi) = pr.window(w.0, w.1);
        (w.1, ts + lo, ts + hi)
    }).collect();
    if !masked.is_empty() {
        if pr.spec.is_none() {
            pr.spec = Some(DevBuf::new(&ops.gpu, 64 * 3 * 4).map_err(e)?);
        }
        let spec = pr.spec.as_ref().unwrap();
        let b: Vec<u8> = masked.iter().flat_map(|(h, lo, hi)| [*h as i32, *lo as i32, *hi as i32]).flat_map(|v| v.to_le_bytes()).collect();
        spec.write(0, &b).map_err(e)?;
        ops.attn_window(q.fp(), kc.fp(), vc.fp(), kvw, keys, hq, hk, hd, scale, spec, masked.len(), ts, ts + tl, att.fp())?;
    }
    let heads: Vec<usize> = pr.heads.iter().filter(|h| h.0 == layer).map(|h| h.1).collect();
    if heads.is_empty() {
        return Ok(());
    }
    if pr.keys.len() <= slot {
        pr.keys.resize(slot + 1, None);
    }
    if pr.keys[slot].is_none() {
        let mut v = vec![0f32; tl * kvw];
        // SAFETY: v is tl * kvw floats, viewed as their bytes
        let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4) };
        kc.read(ts * kvw * 4, bytes).map_err(e)?;
        pr.keys[slot] = Some(v);
    }
    let mut qv = vec![0f32; hq * hd];
    // SAFETY: as above
    let bytes = unsafe { std::slice::from_raw_parts_mut(qv.as_mut_ptr().cast::<u8>(), qv.len() * 4) };
    q.read(0, bytes).map_err(e)?;
    let keys_h = pr.keys[slot].as_ref().unwrap();
    for h in heads {
        let (lo, hi) = pr.window(layer, h);
        let kv = h / (hq / hk);
        let qh = &qv[h * hd..(h + 1) * hd];
        let logits: Vec<f64> = (lo..=hi).map(|j| {
            let kj = &keys_h[j * kvw + kv * hd..j * kvw + (kv + 1) * hd];
            qh.iter().zip(kj).map(|(a, b)| (*a * *b) as f64).sum::<f64>() * scale as f64
        }).collect();
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = logits.iter().map(|x| (x - m).exp()).sum();
        for (i, x) in logits.iter().enumerate() {
            pr.acc[lo + i] += (x - m).exp() / z;
        }
    }
    Ok(())
}
