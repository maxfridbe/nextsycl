"""reference/movieroll/ref.py: MovieRoll's rolling generation on the CPU, at a tiny size, dumping what the engine
(`nextsycl video job check` with a movieroll card, docs/ports/movieroll.md) is checked against.

    git clone https://github.com/Tencent-Hunyuan/MovieRoll MR && git -C MR checkout 0b5f73e2   (the official code)
    python ref.py roll --dit distilled/t2v/model.safetensors --mr MR --text DIR0 [--text DIR1 ...] --out DIR
                       [--chunks 7] [--height 128] [--width 224] [--threshold 18] [--seed 7] [--weights q8|fp32]

The text: each --text DIR holds one event's prompt as reference/hunyuanvideo15/ref.py's `te` / `byt5` stages wrote it
(embeds.npy, byt5.npy when the prompt quotes text) - MovieRoll's text encoder is HunyuanVideo-1.5's (the same video
template, hidden_states[-3]), and the engine reads the same files. With two or more, the video is split evenly into
one event a prompt (upstream's even event split).

The model: the official `MovieRollDiffusionTransformer` (movieroll/models/transformers/movieroll_transformer.py) in
float32, its KV cache in float32 (upstream keeps it in bfloat16), the inference regime (`training=False`); its
attention (FlashAttention-3 upstream) is torch SDPA in float32 here, over the same packed sequences. The 54 double
blocks are read from the file when they run and dropped after (the float32 model would need 33 GB): `--weights q8`
(the default) rounds every double-block matrix the engine keeps as Q8_0 the way the engine rounds it.

The loop: SGLang-MovieRoll's framework-free `rolling.loop.run_rolling` (sglang_movieroll/rolling, "every function
matches its movieroll original") - the distilled chain: the ladder [1000, 800, 600, 400, 200] shifted by 5, one Euler
step a round, context noise 0.15 (t2v), a KV recompute at the threshold (`--threshold`, 18 upstream; a low one forces
it at a tiny size) and at every event switch - driving the official transformer's three calls (denoise, encode at
t = 0 without text, recompute). The noise comes from torch's CPU generator in bfloat16 (the distilled chain's), in the
loop's order.

Dumped to OUT:
  roll.json            sizes, chunk layout, events, the ladder, the recomputes, the ops in order
  op_NNN_*.npy         every model call: denoise (in: latents [T h w, 32], t per frame; out: velocity), encode (the
                       latents after the context noise), recompute (the same, plus the chunk frame counts)
  noise_NNN.npy        every noisy chunk drawn, in order; ctxeps_NNN.npy every context-noise draw (eps), in order
  latents.npy          the video's latents [F h w, 32]
  w0_*.npy             round 0's taps (the first window alone, empty cache): vec0, txt0, dbl0_img, dbl0_txt, v0
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("REF_THREADS", "0")) or (os.cpu_count() or 8))

LADDER = [1000, 800, 600, 400, 200]
SHIFT = 5.0
CONTEXT_NOISE = 0.15


def save(out, name, t):
    a = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
    np.save(os.path.join(out, name), np.ascontiguousarray(a.astype(np.float32)))


def stubs():
    """The official code imports FlashAttention 2 / 3 at module level: stand-ins (never called; SDPA below)"""
    import diffusers.models  # noqa: F401  (before the stand-ins: diffusers probes for the real packages)

    def missing(*_, **__):
        raise RuntimeError("FlashAttention is not used by the reference")

    for name in ("flash_attn_interface", "flash_attn", "flash_attn.bert_padding"):
        m = types.ModuleType(name)
        for f in ("flash_attn_func", "flash_attn_varlen_func", "flash_attn_varlen_qkvpacked_func", "pad_input", "unpad_input"):
            setattr(m, f, missing)
        sys.modules[name] = m
    sys.modules["flash_attn"].bert_padding = sys.modules["flash_attn.bert_padding"]
    if "loguru" not in sys.modules:
        lg = types.ModuleType("loguru")

        class _L:
            def __getattr__(self, _):
                return lambda *x, **k: None

        lg.logger = _L()
        sys.modules["loguru"] = lg


def sdpa(q, k, v):
    """[B, L, H, D] -> [B, L, H, D], float32"""
    o = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float())
    return o.transpose(1, 2).to(q.dtype)


def sdpa_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    out = torch.empty_like(q)
    bq, bk = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for q0, q1, k0, k1 in zip(bq, bq[1:], bk, bk[1:]):
        if q1 > q0:
            out[q0:q1] = sdpa(q[None, q0:q1], k[None, k0:k1], v[None, k0:k1])[0]
    return out


def q8(w):
    """ggml's Q8_0 along the last axis, back to float32 (reference/hunyuanvideo15/ref.py's)"""
    shp = w.shape
    x = w.reshape(-1, 32).float()
    amax = x.abs().max(dim=1, keepdim=True).values
    d = amax / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    v = x * inv
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)
    return (q * d.half().float()).float().reshape(shp)


class LazyBlocks(torch.nn.Module):
    """The double blocks, each read from the file when the forward reaches it (iteration or indexing) and dropped
    after: what `self.double_blocks` is for the official forward"""

    def __init__(self, f, n, make, weights):
        super().__init__()
        self.f, self.n, self.make, self.weights = f, n, make, weights
        self.keys = [k for k in f.keys() if k.startswith("double_blocks.")]

    def __len__(self):
        return self.n

    def load(self, i):
        b = self.make()
        p = f"double_blocks.{i}."
        sd = {}
        for k in self.keys:
            if k.startswith(p):
                t = self.f.get_tensor(k).float()
                if self.weights == "q8" and k.endswith(".weight") and t.dim() == 2 and t.shape[1] % 32 == 0:
                    t = q8(t)
                sd[k[len(p):]] = t
        b.load_state_dict(sd, strict=True)
        return b.float().eval()

    def __getitem__(self, i):
        return self.load(i)

    def __iter__(self):
        for i in range(self.n):
            yield self.load(i)


def build(a):
    stubs()
    sys.path.insert(0, a.mr)
    sys.path.insert(0, os.path.join(a.mr))
    from movieroll.commons.streaming_config import StreamingConfig
    from movieroll.models.transformers import movieroll_transformer as MT
    from movieroll.models.transformers.modules import streaming_attention as SA
    from movieroll.models.transformers.model_configs import MODEL_CONFIG
    from movieroll.commons.constants import MODEL_TYPE
    from safetensors import safe_open

    from movieroll.models.transformers.modules import token_refiner as TR

    def refiner_attention(q, k, v, drop_rate=0, attn_mask=None, causal=False):
        # the token refiner's (flash_attn_no_pad upstream): the text here is its valid tokens only
        assert attn_mask is None or bool(attn_mask.bool().all()), "the reference runs the valid tokens only"
        o = sdpa(q, k, v)
        return o.reshape(o.shape[0], o.shape[1], -1)

    TR.attention = refiner_attention
    SA._attention = lambda q, k, v: sdpa(q, k, v)
    SA._attention_varlen = sdpa_varlen
    sc = StreamingConfig(max_num_latent_frames=120, chunk_size=4, window_size=5, full_history_block_num=6,
                         kv_recompute_enabled=True, kv_recompute_threshold=a.threshold, kv_recompute_recent_k=3)
    sc.kv_cache_save_dtype = torch.float32
    cfg = dict(MODEL_CONFIG[MODEL_TYPE])
    depth = cfg["mm_double_blocks_depth"]
    with torch.device("meta"):
        m = MT.MovieRollDiffusionTransformer(**cfg, streaming_config=sc, dtype=torch.float32)
    f = safe_open(a.dit, "pt")
    # the double blocks read when they run (below), so only the rest is allocated here
    m.double_blocks = torch.nn.Module()
    m = m.to_empty(device="cpu")
    sd = {k: f.get_tensor(k).float() for k in f.keys() if not k.startswith("double_blocks.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("double_blocks.") for k in missing), missing
    m.double_blocks = LazyBlocks(f, depth, lambda: MT.MMDoubleStreamBlock(2048, 16, mlp_width_ratio=4, mlp_act_type="gelu_tanh",
                                                                          qk_norm_type="rms", qkv_bias=True, dtype=torch.float32),
                                 a.weights)
    m.float().eval()
    return m


class Ops:
    """The official transformer behind run_rolling's Denoiser, every call written out"""

    def __init__(self, m, out, h, w):
        self.m, self.out, self.h, self.w = m, out, h, w
        self.pos = 0
        self.log = []
        self.taps = False

    def name(self, kind):
        return f"op_{len(self.log):03d}_{kind}"

    @staticmethod
    def tok(x):
        """[1, C, T, h, w] -> [T h w, C]"""
        return x[0].permute(1, 2, 3, 0).reshape(-1, x.shape[1])

    def model_input(self, lat):
        z = torch.zeros_like(lat)
        return torch.cat([lat, z, z[:, :1]], 1).float()

    def predict(self, latents, timesteps, text, cache):
        n = self.name("denoise")
        T = latents.shape[2]
        qwen, byt5 = text.qwen, text.byt5
        bt = byt5 if byt5 is not None else torch.zeros(1, 1, 1472)
        bm = torch.ones(1, bt.shape[1]) if byt5 is not None else torch.zeros(1, 1)
        v = self.m(self.model_input(latents), timesteps.float(), qwen.float(), torch.ones(1, qwen.shape[1]), return_dict=False,
                   extra_kwargs={"byt5_text_states": bt.float(), "byt5_text_mask": bm}, timestep_txt=timesteps[:, :1].float(),
                   cond_chunk_sizes=[], noisy_chunk_sizes=[T], current_frame_index=self.pos, return_kv=False, training=False)[0]
        save(self.out, n + "_in.npy", self.tok(latents))
        save(self.out, n + "_v.npy", self.tok(v))
        self.log.append({"op": "denoise", "frames": T, "t": [float(x) for x in timesteps[0]], "event": text.event, "pos": self.pos,
                         "cache": self.m.kv_cache_manager.get_cached_cond_latents_num_frames()})
        if not self.taps:
            self.taps = True
            save(self.out, "w0_v0.npy", self.tok(v))
        return v.float()

    def encode(self, latents, cache):
        n = self.name("encode")
        T = latents.shape[2]
        self.m(self.model_input(latents), torch.zeros(1, T), None, None, return_dict=False, extra_kwargs=None, return_kv=True,
               timestep_txt=torch.zeros(1, 1), noisy_chunk_sizes=[T], current_frame_index=self.pos, training=False)
        save(self.out, n + "_in.npy", self.tok(latents))
        self.log.append({"op": "encode", "frames": T, "pos": self.pos})
        self.pos += T

    def recompute(self, latents, chunk_frames, cache):
        n = self.name("recompute")
        self.pos = self.m(hidden_states=self.model_input(latents), timestep=None, text_states=None, encoder_attention_mask=None,
                          kv_recompute=True, recompute_chunk_sizes=list(chunk_frames))
        save(self.out, n + "_in.npy", self.tok(latents))
        self.log.append({"op": "recompute", "chunks": list(chunk_frames), "pos": int(self.pos)})


class CacheView:
    """run_rolling reads the cached chunk count off its cache: the official manager's"""

    def __init__(self, m):
        self.m = m

    def should_recompute(self, threshold):
        return self.m.kv_cache_manager.should_recompute(threshold)


def run_roll(a):
    m = build(a)
    sys.path.insert(0, a.mr)
    from sglang_movieroll.rolling.geometry import chunk_layout, event_split_indices
    from sglang_movieroll.rolling.loop import run_rolling
    from sglang_movieroll.rolling.profiles import RollingProfile
    from sglang_movieroll.rolling.tokens import EventCondition, TextTokens

    h, w = a.height // 16, a.width // 16
    video_length = 16 * a.chunks + 1                       # 1 + 4 x chunks latent frames
    sizes = chunk_layout(video_length, 4, anchored=False, strict=True)
    splits, kept = event_split_indices(len(sizes), a.text if len(a.text) > 1 else None)
    events = []
    for i in kept:
        d = a.text[i]
        q = torch.from_numpy(np.load(os.path.join(d, "embeds.npy")))[None]
        b5 = os.path.join(d, "byt5.npy")
        b = torch.from_numpy(np.load(b5))[None] if os.path.exists(b5) else None
        t = TextTokens(qwen=q, byt5=b)
        t.event = i
        events.append(EventCondition(positive=t))
    sig = torch.tensor([x / 1000.0 for x in LADDER])
    sig = (SHIFT * sig) / (1 + (SHIFT - 1) * sig)
    sigmas = torch.cat([sig, torch.zeros(1)])
    timesteps = sig * 1000.0
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    draws = {"noise": 0, "ctx": 0}

    def noise_fn(frames):
        x = torch.randn((1, 32, frames, h, w), generator=gen, dtype=torch.bfloat16)
        save(a.out, f"noise_{draws['noise']:03d}.npy", Ops.tok(x))
        draws["noise"] += 1
        return x.float()

    def ctx_fn(lat):
        eps = torch.randn(lat.shape, generator=gen, dtype=torch.bfloat16)
        save(a.out, f"ctxeps_{draws['ctx']:03d}.npy", Ops.tok(eps))
        draws["ctx"] += 1
        return ((1 - CONTEXT_NOISE) * lat + CONTEXT_NOISE * eps.float()).contiguous()

    profile = RollingProfile(mode="distilled", steps_per_window=1, kv_recompute_enabled=True, kv_recompute_threshold=a.threshold,
                             kv_recompute_recent_k=3, force_recompute_at_event=True, noise_anchor=True)
    ops = Ops(m, a.out, h, w)
    done = []
    res = run_rolling(denoiser=ops, cache=CacheView(m), profile=profile, events=events, chunk_sizes=sizes, event_split_indices=splits,
                      sigmas=sigmas, timesteps=timesteps, noise_fn=noise_fn, context_noise_fn=ctx_fn,
                      on_chunk=lambda d, t: (done.append(d), print(f"chunk {d}/{t}", flush=True)))
    save(a.out, "latents.npy", Ops.tok(res.latents))
    json.dump({"height": a.height, "width": a.width, "latent": [int(res.latents.shape[2]), h, w], "chunks": sizes, "splits": splits,
               "events": [os.path.abspath(a.text[i]) for i in kept], "ladder": LADDER, "shift": SHIFT, "sigmas": sigmas.tolist(),
               "context_noise": CONTEXT_NOISE, "threshold": a.threshold, "recomputes": res.recomputes, "seed": a.seed, "weights": a.weights,
               "noise_draws": draws["noise"], "ctx_draws": draws["ctx"], "ops": ops.log},
              open(os.path.join(a.out, "roll.json"), "w"), indent=1)
    print("recomputes at", res.recomputes, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["roll"])
    p.add_argument("--dit", required=True)
    p.add_argument("--mr", required=True, help="the official code (Tencent-Hunyuan/MovieRoll at 0b5f73e2)")
    p.add_argument("--text", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", type=int, default=7)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--threshold", type=int, default=18)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--weights", default="q8", choices=["q8", "fp32"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run_roll(a)


if __name__ == "__main__":
    main()
