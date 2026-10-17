"""Voxtral 4B TTS 2603 (mistralai/Voxtral-4B-TTS-2603) on the CPU in float32 - the reference for nextsycl's audio
engine (audio/voxtraltts): a request's prompt, each frame's hidden state, semantic logits and flow velocities, the
codes and the codec's speech, dumped as .npy for `nextsycl audio check`.

    python ref.py CHECKPOINT_DIR --code VLLM_OMNI_DIR --text "..." [--voice neutral_female] [--cfg 1.2] [--frames 150]
                  [--seed 42] --out DIR
    python ref.py CHECKPOINT_DIR --code VLLM_OMNI_DIR --decode DIR/codes.npy --out DIR     (the codec's decoder alone)

Upstream's own code where it is plain PyTorch: the acoustic flow-matching transformer (FlowMatchingAudioTransformer,
vllm_omni/model_executor/models/voxtral_tts/voxtral_tts_audio_generation.py) and the codec
(VoxtralTTSAudioTokenizer, voxtral_tts_audio_tokenizer.py) are taken out of a vllm-omni checkout by name (ast) and
run as they are - only their vLLM imports are stood in for. The prompt is mistral-common's encode_speech_request
([BOS] [BEGIN_AUDIO] [AUDIO] x N [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]) with the preset voice's
embedding rows (voice_embedding/<voice>.pt) at the [AUDIO] positions, as vllm-omni's tts_preprocess places them.
The backbone is vLLM's MistralForCausalLM written out (Ministral 3B: 26 layers, RMS norm 1e-5, GQA 32 / 8 of 128,
RoPE θ 1e6 on adjacent pairs - the consolidated checkpoint's layout - SwiGLU 9,216). The loop is vllm-omni's: the
last prompt position's hidden state makes frame 0 (compute_mm_logits), each frame's 37 codes' embeddings summed are
the next position's input, [END_AUDIO] (semantic id 1) ends it; the codec decodes in chunks of 375 frames.

Dumps (DIR): request.json; ids.npy (the prompt's token ids); prefill.npy [L, 3072] (the backbone's input rows);
hidden.npy [T, 3072] (each frame's hidden state, the flow's condition); sem.npy [T, 8320] (the semantic head's
logits, unmasked); noise.npy [T, 36] (each frame's starting noise); vel.npy [T, 7, 2, 36] (the flow's velocities,
conditioned and not, at each Euler step); lat.npy [T, 36] (the flow's result before quantization); codes.npy [T, 37]
(as the model emits them: semantic id with its 2 special ids, acoustic levels + 2); speech.npy / speech.wav (24 kHz).
"""
import argparse
import ast
import json
import logging
import math
import os
import types
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from math import ceil
from typing import Union, get_args, get_origin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from torch.nn.utils.rnn import pad_sequence

MODELS = "vllm_omni/model_executor/models/voxtral_tts"


def upstream(path, names, ns):
    """The top-level definitions `names` of an upstream file, run in `ns` (its vLLM imports stood in for there)"""
    tree = ast.parse(open(path).read())
    keep = [n for n in tree.body if getattr(n, "name", None) in names
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    found = {getattr(n, "name", None) or n.targets[0].id for n in keep}
    missing = set(names) - found
    if missing:
        raise SystemExit(f"{path}: no {sorted(missing)} (another vllm-omni revision?)")
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)


def weight_loader(param, w):
    param.data.copy_(w)


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_pairs(x, pos, theta):
    """RoPE on adjacent pairs (2i, 2i + 1), x [T, H, D]"""
    d = x.shape[-1]
    # in float32, as vLLM's rotary cache is made
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = pos[:, None].float() * inv[None, :]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


class Backbone:
    """vLLM's MistralForCausalLM for this checkpoint, float32, with a cache"""

    def __init__(self, sf, p):
        g = lambda n: sf.get_tensor(n).float()
        self.p = p
        self.layers = [{k: g(f"layers.{i}.{k}.weight") for k in ["attention_norm", "attention.wq", "attention.wk", "attention.wv", "attention.wo",
                                                                  "ffn_norm", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]}
                       for i in range(p["n_layers"])]
        self.norm = g("norm.weight")
        self.k = [None] * p["n_layers"]
        self.v = [None] * p["n_layers"]
        self.n = 0

    def __call__(self, x):
        """x [L, dim] at positions n .. n + L - 1: the normed hidden states [L, dim]"""
        p = self.p
        L = x.shape[0]
        hd, nh, nkv = p["head_dim"], p["n_heads"], p["n_kv_heads"]
        pos = torch.arange(self.n, self.n + L)
        for i, l in enumerate(self.layers):
            h = rms(x, l["attention_norm"], p["norm_eps"])
            q = rope_pairs((h @ l["attention.wq"].T).view(L, nh, hd), pos, p["rope_theta"])
            k = rope_pairs((h @ l["attention.wk"].T).view(L, nkv, hd), pos, p["rope_theta"])
            v = (h @ l["attention.wv"].T).view(L, nkv, hd)
            self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k])
            self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v])
            K = self.k[i].repeat_interleave(nh // nkv, dim=1).permute(1, 0, 2)
            V = self.v[i].repeat_interleave(nh // nkv, dim=1).permute(1, 0, 2)
            T = K.shape[1]
            mask = torch.full((L, T), float("-inf")).triu(diagonal=self.n + 1)
            a = torch.softmax(q.permute(1, 0, 2) @ K.transpose(1, 2) / math.sqrt(hd) + mask, dim=-1) @ V
            x = x + a.permute(1, 0, 2).reshape(L, nh * hd) @ l["attention.wo"].T
            h = rms(x, l["ffn_norm"], p["norm_eps"])
            x = x + (F.silu(h @ l["feed_forward.w1"].T) * (h @ l["feed_forward.w3"].T)) @ l["feed_forward.w2"].T
        self.n += L
        return rms(x, self.norm, p["norm_eps"])


def flash_standin(q, k, v, causal, window_size, alibi_slopes):
    """flash_attn_func's math for the codec (its path on CUDA): ALiBi slope_h (j - i), causal, keys i - left .. i + right;
    q / k / v [B, S, H, D] -> [B, S, H, D] contiguous (the SDPA fallback's view fails on a non-contiguous result on XPU)"""
    S = q.shape[1]
    pos = torch.arange(S, device=q.device)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)
    bias = alibi_slopes.to(q.dtype).view(-1, 1, 1) * rel.unsqueeze(0).to(q.dtype)
    if causal:
        bias = bias.masked_fill(rel.unsqueeze(0) > 0, float("-inf"))
    bias = bias.masked_fill(((rel < -window_size[0]) | (rel > window_size[1])).unsqueeze(0), float("-inf"))
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias.unsqueeze(0))
    return o.transpose(1, 2).contiguous()


def modules(code, flash=False):
    """Upstream's acoustic transformer and codec modules (a vllm-omni checkout), their vLLM parts stood in for; `flash`:
    the codec's flash-attn path with `flash_standin` (the same attention; off the CPU)"""
    log = logging.getLogger("voxtral-ref")
    ns = dict(math=math, np=np, torch=torch, nn=nn, F=F, Enum=Enum, dataclass=dataclass, fields=fields, is_dataclass=is_dataclass, replace=replace,
              Union=Union, get_args=get_args, get_origin=get_origin, ceil=ceil, rms_norm=nn.RMSNorm, logger=log, logging=logging,
              default_weight_loader=weight_loader, rearrange=rearrange, deepcopy=deepcopy, pad_sequence=pad_sequence,
              weight_norm=torch.nn.utils.parametrizations.weight_norm, HAS_FLASH_ATTN=flash, flash_attn_func=flash_standin if flash else None, VllmConfig=object,
              current_omni_platform=types.SimpleNamespace(device_type="cpu"), Any=object, __name__="voxtral_upstream")
    upstream(os.path.join(code, MODELS, "voxtral_tts_audio_generation.py"),
             ["AudioSpecialTokens", "AcousticTransformerArgs", "MultimodalAudioModelArgs", "_repeat_interleave", "repeat_kv", "from_nested_dict",
              "FeedForward", "BidirectionalAttention", "AcousticTransformerBlock", "TimeEmbedding", "FlowMatchingAudioTransformer"], ns)
    upstream(os.path.join(code, MODELS, "voxtral_tts_audio_tokenizer.py"),
             ["AudioTokenizerArgs", "SemanticCodebook", "AcousticCodebook", "MistralAudioCodebook", "prepare_for_attention", "pad1d", "CausalConv1d",
              "CausalConvTranspose1d", "MultiVocabEmbeddings", "Attention", "TransformerBlock", "Transformer", "VoxtralTTSAudioTokenizer"], ns)
    return ns


def audio_args(params):
    """params.json's audio model arguments as vllm-omni's parser leaves them (7 Euler steps when it names none)"""
    a = deepcopy(params["multimodal"]["audio_model_args"])
    a["acoustic_transformer_args"].setdefault("n_decoding_steps", 7)
    return a


def make_codec(ns, sf, params, audio_model_args):
    """Upstream's VoxtralTTSAudioTokenizer with the checkpoint's decoder, quantizer and audio embeddings, float32"""
    mm = params["multimodal"]
    cfg = types.SimpleNamespace(audio_config={"codec_args": mm["audio_tokenizer_args"], "audio_model_args": deepcopy(audio_model_args)},
                                text_config=types.SimpleNamespace(hidden_size=params["dim"]))
    codec = ns["VoxtralTTSAudioTokenizer"](vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=cfg)))
    for name in sf.keys():
        if name.startswith("audio_tokenizer."):
            codec.load_weight((name[len("audio_tokenizer."):], sf.get_tensor(name).float()))
        elif name == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight":
            codec.load_weight(("audio_token_embedding.embeddings.weight", sf.get_tensor(name).float()))
    return codec.float().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--code", required=True, help="a checkout of github.com/vllm-project/vllm-omni")
    ap.add_argument("--text", default="")
    ap.add_argument("--voice", default="neutral_female")
    ap.add_argument("--cfg", type=float, default=1.2)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    params = json.load(open(os.path.join(a.model, "params.json")))
    mm = params["multimodal"]
    audio_model_args = audio_args(params)

    ns = modules(a.code)
    sf = safe_open(os.path.join(a.model, "consolidated.safetensors"), framework="pt")
    codec = make_codec(ns, sf, params, audio_model_args)

    def decode(codes):
        """codes [T, 37] as emitted: cut at the first [END_AUDIO], 2 taken off, 375-frame chunks (decode_helper_batch_async), float32"""
        c = torch.as_tensor(codes).long()
        end = (c[:, 0] == 1).nonzero()
        c = c[: int(end[0]) if len(end) else len(c)] - 2
        parts = []
        with torch.no_grad():
            for i in range(0, len(c), 375):
                ch = c[i:i + 375]
                parts.append(codec.decode(ch.T.unsqueeze(0), dtype=torch.float32)[0, 0, : len(ch) * codec.downsample_factor])
        return torch.cat(parts).numpy()

    def save_speech(wav):
        np.save(os.path.join(a.out, "speech.npy"), wav.astype(np.float32))
        try:
            import soundfile
            soundfile.write(os.path.join(a.out, "speech.wav"), wav, 24000)
        except ImportError:
            pass

    if a.decode:
        save_speech(decode(np.load(a.decode)))
        return

    # the prompt: mistral-common's own encoding, the voice's rows at its [AUDIO] positions
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_file(os.path.join(a.model, "tekken.json"))
    ids = tok.instruct_tokenizer.encode_speech_request(SpeechRequest(input=a.text, voice=a.voice)).tokens
    voice = torch.load(os.path.join(a.model, "voice_embedding", f"{a.voice}.pt"), map_location="cpu").float()
    audio_id = mm["audio_model_args"]["audio_token_id"]
    at = [i for i, t in enumerate(ids) if t == audio_id]
    assert len(at) == voice.shape[0], (len(at), voice.shape)
    emb = sf.get_tensor("mm_audio_embeddings.tok_embeddings.weight")
    rows = emb[torch.tensor(ids)].float()
    rows[torch.tensor(at)] = voice
    del emb
    json.dump({"text": a.text, "voice": a.voice, "cfg": a.cfg, "seed": a.seed}, open(os.path.join(a.out, "request.json"), "w"))
    np.save(os.path.join(a.out, "ids.npy"), np.array(ids, dtype=np.int64))
    np.save(os.path.join(a.out, "prefill.npy"), rows.numpy())

    fm = ns["FlowMatchingAudioTransformer"](deepcopy(audio_model_args))
    for name in sf.keys():
        if name.startswith("acoustic_transformer."):
            fm.load_weight((name[len("acoustic_transformer."):], sf.get_tensor(name).float()))
    fm.float().eval()
    bb = Backbone(sf, params)

    # what a frame records: the semantic head's logits (a hook), the velocities (_predict_velocity wrapped), the noise
    rec = {}
    fm.semantic_codebook_output.register_forward_hook(lambda m, i, o: rec.__setitem__("sem", o.detach().clone()))
    predict = fm._predict_velocity

    def predicted(**kw):
        v = predict(**kw)
        rec.setdefault("vel", []).append(v.detach().clone())
        return v

    fm._predict_velocity = predicted
    gen = torch.Generator().manual_seed(a.seed)
    randn = torch.randn
    out = {k: [] for k in ["hidden", "sem", "noise", "vel", "lat", "codes"]}
    alpha = torch.tensor([a.cfg])
    with torch.no_grad():
        h = bb(rows)[-1:]
        for f in range(a.frames):
            noise = randn(1, audio_model_args["n_acoustic_codebook"], generator=gen)
            rec.clear()
            torch.randn = lambda *s, **k: noise.clone()
            try:
                codes = fm(llm_hidden=h, cfg_alpha=alpha)
            finally:
                torch.randn = randn
            vel = torch.stack(rec.get("vel", [])) if rec.get("vel") else torch.zeros(0, 2, 36)
            # the flow's result before quantization, its Euler steps replayed from the velocities
            ts = torch.linspace(0, 1, audio_model_args["acoustic_transformer_args"]["n_decoding_steps"] + 1)
            x = noise.clone()
            for i, v in enumerate(vel):
                x = x + (a.cfg * v[0:1] + (1 - a.cfg) * v[1:2]) * (ts[i + 1] - ts[i])
            out["hidden"].append(h[0].numpy())
            out["sem"].append(rec["sem"][0].numpy())
            out["noise"].append(noise[0].numpy())
            out["vel"].append(vel.reshape(-1, 2, 36).numpy() if len(vel) else np.zeros((7, 2, 36), np.float32))
            out["lat"].append(x[0].numpy())
            out["codes"].append(codes[0].numpy())
            print(f"frame {f}: semantic {int(codes[0, 0])}", flush=True)
            if int(codes[0, 0]) == 1:
                break
            # the next position: the frame's 37 codes' embeddings summed (encode_tokens)
            e = codec.encode_tokens([codes.T.unsqueeze(0)])[0]
            h = bb(e.reshape(1, -1))
    for k, v in out.items():
        np.save(os.path.join(a.out, f"{k}.npy"), np.stack(v).astype(np.int64 if k == "codes" else np.float32))
    save_speech(decode(np.stack(out["codes"])))
    print(f"{len(out['codes'])} frames -> {a.out}")


if __name__ == "__main__":
    main()
