"""Fish Audio S2 Pro in its own code (github.com/fishaudio/fish-speech: fish_speech/models/text2semantic/llama.py,
fish_speech/conversation.py, fish_speech/models/dac/), on the CPU in float32 - the reference for nextsycl's audio engine
(audio/s2pro): a request's prompt, the stages' tensors and a greedy run's frames, dumped as .npy for `nextsycl audio check`.

    PYTHONPATH=PYLIB python ref.py CHECKPOINT_DIR --code FISH_SPEECH_DIR --text "..." [--ref-audio r.wav --ref-text "..."]
                  [--frames 120] [--sample --seed 42] --out DIR
    PYTHONPATH=PYLIB python ref.py CHECKPOINT_DIR --code FISH_SPEECH_DIR --decode DIR/codes.npy [--codec-half] --out DIR
                  (the codec alone: decoded.npy; with --codec-half its weights rounded to half first, as the GGUF holds
                  them: decoded_half.npy)

CHECKPOINT_DIR is fishaudio/s2-pro as downloaded (config.json, the two safetensors shards, tokenizer.json, codec.pth).
PYLIB holds descript-audio-codec (pip install --no-deps --target PYLIB descript-audio-codec), loguru and soundfile: the
codec's layers and quantizer come from `dac`, whose package __init__ (audiotools) is not run - stubs below.

The prompt is upstream's own Conversation (generate_long in fish_speech/models/text2semantic/inference.py): a system
turn ("convert the provided text to speech", or with a recording "... reference to the following:\\n\\nText:\\n
<|speaker:0|>TRANSCRIPT\\n\\nSpeech:\\n" and its codes), the text as the user turn, then "<|im_start|>assistant\\n<|voice|>".
The frame loop is decode_one_token_ar written out: the slow model's logits over the semantic ids and <|im_end|>, the
frame's first code; the fast model at positions 0 (the slow hidden state) to 9, the other nine codes. Greedy by default
(the most likely id, no repetition-aware resampling: a deterministic run to compare against); --sample draws as upstream
(temperature 0.7, top-p 0.7, top-k 30, RAS over the last 10 ids), torch's generator seeded.

Dumps (DIR): request.json; ids.npy [11, L] (the prompt's rows: token ids, then the codes at a recording's frames);
prefill.npy [L, 2560] (the slow model's input rows); hidden0.npy [2560] (its normed last hidden state after the
prefill); logits.npy [T, 4097] (each frame's slow logits: the 4,096 semantic ids, then <|im_end|>); fast.npy [T, 9, 4096]
(the fast model's logits for codes 1..9); codes.npy [T, 10]; codec_z.npy [1024, 4T] (the quantizer's decode: codebooks,
post transformer, upsampling); speech.npy / speech.wav (the codec's decode of codes, 44.1 kHz); for a recording ref.wav
(at 44.1 kHz, mono, as encoded) and ref_codes.npy [F, 10].
"""
import argparse
import importlib.util
import json
import os
import sys
import types
from functools import partial

import numpy as np
import torch
import torch.nn as nn


def dac_stubs():
    """dac.nn.layers / dac.nn.quantize importable without dac's and audiotools' package code"""
    at = types.ModuleType("audiotools")
    at.AudioSignal = object
    ml = types.ModuleType("audiotools.ml")
    ml.BaseModel = nn.Module
    at.ml = ml
    sys.modules["audiotools"], sys.modules["audiotools.ml"] = at, ml
    root = os.path.dirname(importlib.util.find_spec("dac").origin)
    for name, sub in [("dac", root), ("dac.nn", os.path.join(root, "nn")), ("dac.model", os.path.join(root, "model"))]:
        m = types.ModuleType(name)
        m.__path__ = [sub]
        sys.modules[name] = m
    base = types.ModuleType("dac.model.base")
    base.CodecMixin = type("CodecMixin", (), {"get_delay": lambda self: 0})
    sys.modules["dac.model.base"] = base


def load_codec(path):
    """configs/modded_dac_vq.yaml built by hand, its weights as load_codec_model reads them, float32"""
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
    from fish_speech.models.dac.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

    def rvq_tr():
        return WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=ModelArgs(
            block_size=2048, n_layer=8, n_head=16, dim=1024, intermediate_size=3072, n_local_heads=-1, head_dim=64, rope_base=10000,
            norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True))

    q = DownsampleResidualVectorQuantize(input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.5,
                                         downsample_factor=[2, 2], post_module=rvq_tr(), pre_module=rvq_tr(), semantic_codebook_size=4096)
    general = partial(ModelArgs, block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000, norm_eps=1e-5, dropout_rate=0.1,
                      attn_dropout_rate=0.1, channels_first=True)
    codec = DAC(sample_rate=44100, encoder_dim=64, encoder_rates=[2, 4, 8, 8], decoder_dim=1536, decoder_rates=[8, 8, 4, 2],
                encoder_transformer_layers=[0, 0, 0, 4], decoder_transformer_layers=[4, 0, 0, 0], transformer_general_config=general,
                quantizer=q)
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if any("generator" in k for k in sd):
        sd = {k.replace("generator.", ""): v for k, v in sd.items() if "generator." in k}
    rebuilt = {f"{n}.{b}" if n else b for n, m in codec.named_modules() for b in m._non_persistent_buffers_set}
    codec.load_state_dict({k: v for k, v in sd.items() if k not in rebuilt}, strict=True)
    codec.eval()
    for m in list(codec.modules()):
        if is_parametrized(m, "weight"):
            remove_parametrizations(m, "weight", leave_parametrized=True)
        elif hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    return codec.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--code", required=True, help="a checkout of github.com/fishaudio/fish-speech")
    ap.add_argument("--text", default="")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode", default=None)
    ap.add_argument("--codec-half", action="store_true", help="the codec's weights rounded to half (the GGUF's) before use")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    dac_stubs()
    import soundfile as sf

    def save(n, v):
        np.save(os.path.join(a.out, n), np.ascontiguousarray(v))

    codec = load_codec(os.path.join(a.model, "codec.pth"))
    if a.codec_half:
        # the GGUF's codec: weight norm folded in float32, then every tensor stored in half
        with torch.no_grad():
            for p in codec.parameters():
                p.copy_(p.half().float())
    if a.decode:
        codes = torch.from_numpy(np.load(a.decode)).long().T  # [10, T]
        with torch.inference_mode():
            wav = codec.from_indices(codes[None])[0, 0]
        save("decoded_half.npy" if a.codec_half else "decoded.npy", wav.numpy().astype(np.float32))
        print("decoded", wav.shape[0], "samples")
        return

    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.tokenizer import IM_END_TOKEN

    model = DualARTransformer.from_pretrained(a.model, load_weights=True)
    model = model.to(dtype=torch.float32).eval()
    cfg = model.config
    tok = model.tokenizer
    im_end = tok.get_token_id(IM_END_TOKEN)
    sb, se = cfg.semantic_begin_id, cfg.semantic_end_id

    ref_codes = None
    if a.ref_audio:
        import torchaudio

        wav, sr = sf.read(a.ref_audio, always_2d=True, dtype="float32")
        wav = torch.from_numpy(wav.mean(1))[None]
        wav = torchaudio.functional.resample(wav, sr, codec.sample_rate)[0]
        sf.write(os.path.join(a.out, "ref.wav"), wav.numpy(), codec.sample_rate, subtype="FLOAT")
        with torch.inference_mode():
            idx, lens = codec.encode(wav[None, None], torch.tensor([len(wav)]))
        ref_codes = idx[0, :, : lens[0]].long()  # [10, F]
        save("ref_codes.npy", ref_codes.T.numpy())

    # the prompt, as generate_long builds it (one batch: the text without speaker turns to split)
    conv = Conversation()
    if ref_codes is not None:
        rt = a.ref_text.strip()
        if "<|speaker:" not in rt:
            rt = f"<|speaker:0|>{rt}"
        parts = [TextPart(text="convert the provided text to speech reference to the following:\n\nText:\n"), TextPart(text=rt),
                 TextPart(text="\n\nSpeech:\n"), VQPart(codes=ref_codes)]
    else:
        parts = [TextPart(text="convert the provided text to speech")]
    conv.append(Message(role="system", parts=parts, add_im_start=True, add_im_end=True))
    conv.append(Message(role="user", parts=[TextPart(text=a.text)], add_im_start=True, add_im_end=True))
    conv.append(Message(role="assistant", parts=[], modality="voice", add_im_start=True, add_im_end=False))
    values, audio_masks, audio_parts = conv.encode_for_inference(tok, num_codebooks=cfg.num_codebooks)
    save("ids.npy", values.long().numpy())
    L = values.shape[1]
    with open(os.path.join(a.out, "request.json"), "w") as f:
        json.dump({"text": a.text, "ref_text": a.ref_text or "", "frames": a.frames, "sample": a.sample, "seed": a.seed, "length": L}, f,
                  ensure_ascii=False)

    torch.manual_seed(a.seed)
    rows = {}

    def keep(m, args):
        # the first call's input: the prefill's rows (a pre-hook that returns nothing leaves the arguments as they are)
        rows.setdefault("x", args[0].detach().clone())

    hook = model.layers[0].register_forward_pre_hook(keep)
    model.setup_caches(max_batch_size=1, max_seq_len=L + a.frames + 8, dtype=torch.float32)
    for b in model.fast_layers:
        b.attention.kv_cache.k_cache.zero_()

    def probs(l, temp, top_p, top_k):
        sl, si = torch.sort(l, descending=True)
        cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
        rm = (cum > top_p) | (torch.arange(sl.shape[-1]) >= top_k)
        rm[0] = False
        rm = rm.scatter(0, si, rm)
        return torch.softmax(torch.where(rm, float("-inf"), l) / max(temp, 1e-5), dim=-1)

    def draw(l, temp=0.7, top_p=0.7):
        if not a.sample:
            return int(torch.argmax(l))
        p = probs(l, temp, top_p, 30)
        q = torch.empty_like(p).exponential_(1)
        return int(torch.argmax(p / q))

    bias = torch.full((cfg.vocab_size,), float("-inf"))
    bias[sb : se + 1] = 0.0
    bias[im_end] = 0.0
    window = []
    all_logits, all_fast, codes = [], [], []
    with torch.inference_mode():
        r = model.forward_generate(values[None], torch.arange(L), kv_len=L)
        hook.remove()
        save("prefill.npy", rows["x"][0].numpy())
        save("hidden0.npy", r.hidden_states[0, -1].numpy())
        for step in range(a.frames):
            lg = r.logits[0, -1] + bias
            all_logits.append(torch.cat([lg[sb : sb + 4096], lg[im_end : im_end + 1]]).numpy())
            token = draw(lg)
            if a.sample and token in window and sb <= token <= se:
                token = draw(lg, 1.0, 0.9)
            if token == im_end:
                break
            if step > 0:
                # upstream's window holds the ids drawn after the prefill's (decode_n_tokens' previous_tokens)
                window = (window + [token])[-10:]
            model.forward_generate_fast(r.hidden_states, torch.tensor([0]))
            c = min(max(token - sb, 0), cfg.codebook_size - 1)
            frame = [c]
            fl = []
            h = model.fast_embeddings(torch.tensor([c]))
            for cb in range(1, cfg.num_codebooks):
                lf = model.forward_generate_fast(h, torch.tensor([cb]))[0, -1]
                fl.append(lf.numpy())
                c = draw(lf)
                frame.append(c)
                h = model.fast_embeddings(torch.tensor([c]))
            all_fast.append(np.stack(fl))
            codes.append(frame)
            x = torch.tensor([[token] + frame], dtype=torch.long)[:, :, None]  # [1, 11, 1]
            r = model.forward_generate(x, torch.tensor([L + step]), kv_len=L + step + 1)
            print(f"frame {step}: {frame[:4]}...", flush=True)
        save("logits.npy", np.stack(all_logits))
        if all_fast:
            save("fast.npy", np.stack(all_fast))
        codes = np.array(codes, dtype=np.int64).reshape(-1, cfg.num_codebooks)
        save("codes.npy", codes)
        if len(codes):
            ct = torch.from_numpy(codes).T[None]
            save("codec_z.npy", codec.quantizer.decode(ct.clone())[0].numpy())
            wav = codec.from_indices(ct)[0, 0].numpy()
            sf.write(os.path.join(a.out, "speech.wav"), wav, codec.sample_rate)
            save("speech.npy", wav.astype(np.float32))
            print(f"{len(wav) / codec.sample_rate:.2f} s of speech, {codes.shape[0]} frames; prompt {L} positions")


if __name__ == "__main__":
    main()
