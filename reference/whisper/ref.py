"""Whisper in openai-whisper's own code, on the CPU in float32 - the reference for nextsycl's recogniser
(audio/whisper): the stages' tensors and openai's transcribe, dumped for `nextsycl audio check`.

    CUDA_VISIBLE_DEVICES= python ref.py MODEL_DIR AUDIO --out DIR [--language en]

MODEL_DIR is the transformers checkpoint nextsycl loads (model.safetensors, config.json, generation_config.json):
its weights are renamed into openai-whisper's model (the inverse of transformers' convert_openai_to_hf), its
alignment heads set from generation_config.json, so both sides run the same numbers.

Dumps (DIR): audio16k.npy (the recording at 16 kHz: scipy's resample_poly, as the platform's worker); mel.npy
[n_mels, 3000] (log_mel_spectrogram with 30 s of padding, the first window as transcribe pads it: zeros past the
recording); enc.npy [1500, d] (the encoder on it); tokens.npy (the first window's start of transcript, then openai's
greedy decoding of it, timestamps on) and logits.npy [T, vocab] (the decoder teacher-forced on those tokens);
align_qk.npy [heads, rows, 1500], align_tokens.npy, align.json (the first window's alignment: the heads' raw attention
logits over the tokens fed, find_alignment's words); result.json (transcribe: word_timestamps, temperature 0,
condition_on_previous_text off - the text, language, segments with their tokens and words; asked_language).
"""
import argparse
import json
import os
from math import gcd

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly

import whisper
from whisper.model import ModelDimensions, Whisper


def hf_to_openai(sd):
    out = {}
    for k, v in sd.items():
        n = k.removeprefix("model.")
        n = n.replace("encoder.layers.", "encoder.blocks.").replace("decoder.layers.", "decoder.blocks.")
        n = n.replace("self_attn.q_proj", "attn.query").replace("self_attn.k_proj", "attn.key").replace("self_attn.v_proj", "attn.value")
        n = n.replace("self_attn.out_proj", "attn.out").replace("self_attn_layer_norm", "attn_ln")
        n = n.replace("encoder_attn.q_proj", "cross_attn.query").replace("encoder_attn.k_proj", "cross_attn.key")
        n = n.replace("encoder_attn.v_proj", "cross_attn.value").replace("encoder_attn.out_proj", "cross_attn.out")
        n = n.replace("encoder_attn_layer_norm", "cross_attn_ln").replace("final_layer_norm", "mlp_ln")
        n = n.replace(".fc1.", ".mlp.0.").replace(".fc2.", ".mlp.2.")
        n = n.replace("encoder.embed_positions.weight", "encoder.positional_embedding")
        n = n.replace("decoder.embed_positions.weight", "decoder.positional_embedding")
        n = n.replace("decoder.embed_tokens.weight", "decoder.token_embedding.weight")
        n = n.replace("encoder.layer_norm.", "encoder.ln_post.").replace("decoder.layer_norm.", "decoder.ln.")
        out[n] = v.float()
    return out


def load(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    g = json.load(open(os.path.join(model_dir, "generation_config.json")))
    dims = ModelDimensions(n_mels=c["num_mel_bins"], n_audio_ctx=c["max_source_positions"], n_audio_state=c["d_model"],
                           n_audio_head=c["encoder_attention_heads"], n_audio_layer=c["encoder_layers"], n_vocab=c["vocab_size"],
                           n_text_ctx=c["max_target_positions"], n_text_state=c["d_model"], n_text_head=c["decoder_attention_heads"],
                           n_text_layer=c["decoder_layers"])
    m = Whisper(dims)
    missing, unexpected = m.load_state_dict(hf_to_openai(load_file(os.path.join(model_dir, "model.safetensors"))), strict=False)
    # the encoder's sinusoids are a buffer of openai's (the checkpoint's copy loaded over it)
    assert not unexpected, unexpected
    assert all("positional_embedding" in k for k in missing), missing
    mask = torch.zeros(dims.n_text_layer, dims.n_text_head, dtype=torch.bool)
    for l, h in g["alignment_heads"]:
        mask[l, h] = True
    m.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    wav, sr = sf.read(a.audio, dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        mono = resample_poly(mono, 16000 // g, int(sr) // g)
    audio = np.ascontiguousarray(mono, dtype=np.float32)
    np.save(os.path.join(a.out, "audio16k.npy"), audio)
    m = load(a.model)
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio), m.dims.n_mels, padding=480000)
    # the first window as transcribe makes it: the recording's frames, zeros after them
    content = mel.shape[-1] - 3000
    w0 = whisper.pad_or_trim(mel[:, :min(3000, content)], 3000)
    np.save(os.path.join(a.out, "mel.npy"), w0.numpy().astype(np.float32))
    enc = m.encoder(w0[None])
    np.save(os.path.join(a.out, "enc.npy"), enc[0].numpy().astype(np.float32))
    tok = whisper.tokenizer.get_tokenizer(True, num_languages=m.num_languages, language=a.language, task="transcribe")
    lang = a.language
    if lang is None:
        _, probs = m.detect_language(enc, tok)
        lang = max(probs[0], key=probs[0].get)
        tok = whisper.tokenizer.get_tokenizer(True, num_languages=m.num_languages, language=lang, task="transcribe")
    r = whisper.decode(m, w0, whisper.DecodingOptions(language=lang, without_timestamps=False, fp16=False, temperature=0.0))
    toks = list(tok.sot_sequence) + list(r.tokens)
    np.save(os.path.join(a.out, "tokens.npy"), np.array(toks, dtype=np.int64))
    logits = m.decoder(torch.tensor([toks]), enc)[0]
    np.save(os.path.join(a.out, "logits.npy"), logits.numpy().astype(np.float32))
    # the alignment of the first window: the heads' raw attention logits (openai's hooks, eager attention) and
    # find_alignment's words, before the punctuation merges and hacks
    from whisper.model import disable_sdpa
    from whisper.timing import find_alignment
    text = [t for t in r.tokens if t < tok.eot]
    frames = min(3000, content)
    fed = [*tok.sot_sequence, tok.no_timestamps, *text, tok.eot]
    qks = [None] * m.dims.n_text_layer
    hooks = [b.cross_attn.register_forward_hook(lambda _, ins, outs, index=i: qks.__setitem__(index, outs[-1][0])) for i, b in enumerate(m.decoder.blocks)]
    with disable_sdpa():
        m.decoder(torch.tensor([fed]), enc)
    for h in hooks:
        h.remove()
    heads = m.alignment_heads.indices().T.tolist()
    np.save(os.path.join(a.out, "align_qk.npy"), torch.stack([qks[l][h] for l, h in heads]).numpy().astype(np.float32))
    np.save(os.path.join(a.out, "align_tokens.npy"), np.array(fed, dtype=np.int64))
    words = find_alignment(m, tok, text, w0, frames)
    json.dump({"num_frames": frames, "sot_len": len(tok.sot_sequence), "language": lang,
               "words": [{"word": w.word, "tokens": w.tokens, "start": float(w.start), "end": float(w.end), "probability": float(w.probability)} for w in words]},
              open(os.path.join(a.out, "align.json"), "w"), ensure_ascii=False, indent=1)
    res = whisper.transcribe(m, audio, language=a.language, temperature=0.0, word_timestamps=True, condition_on_previous_text=False, fp16=False)
    res["asked_language"] = a.language
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{a.out}: {len(audio) / 16000:.1f} s, language {res['language']}, {len(res['segments'])} segments, "
          f"{sum(len(s.get('words', [])) for s in res['segments'])} words; first window {len(r.tokens)} tokens: {r.text[:120]!r}")


if __name__ == "__main__":
    main()
