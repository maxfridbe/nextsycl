"""Qwen3-TTS in its own package (qwen-tts), on the CPU in float32 - the reference for nextsycl's audio engine
(audio/qwen3tts): a mode's speech and the stages' tensors, dumped as .npy for `nextsycl audio check`.

    python ref.py MODEL_DIR custom  --text "..." --speaker vivian [--instruct "..."] [--language auto] --out DIR [--greedy]
    python ref.py MODEL_DIR design  --text "..." --instruct "a warm low voice ..." --out DIR [--greedy]
    python ref.py MODEL_DIR clone   --text "..." --ref-audio ref.wav [--ref-text "..."] [--xvec] --out DIR [--greedy]
    python ref.py MODEL_DIR decode  --codes DIR/codes.npy --out DIR        (the codec's decoder alone)
    python ref.py MODEL_DIR info                                            (speakers, languages)

Dumps (DIR): request.json (the mode and its settings), prefill.npy (the talker's input embeddings [L, 2048]),
logits0.npy (the first codebook's logits after the prefill), codes.npy [T, 16], speech.wav and speech.npy (24 kHz),
and for clone ref.wav (the recording at 24 kHz), spk.npy (its speaker embedding) and ref_codes.npy.
--greedy: both the talker and the code predictor take the most likely code, no repetition penalty (a deterministic
run to compare against).
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("mode", choices=["custom", "design", "clone", "decode", "info"])
    ap.add_argument("--text", default="")
    ap.add_argument("--speaker", default="")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--language", default="Auto")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--xvec", action="store_true")
    ap.add_argument("--codes", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(a.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
    m = tts.model
    if a.mode == "info":
        print("type", m.tts_model_type, "size", m.tts_model_size)
        print("speakers", list(tts.get_supported_speakers() or []))
        print("languages", list(tts.get_supported_languages() or []))
        return
    os.makedirs(a.out, exist_ok=True)
    if a.mode != "decode":
        import json
        with open(os.path.join(a.out, "request.json"), "w") as f:
            json.dump({"mode": a.mode, "text": a.text, "speaker": a.speaker, "instruct": a.instruct, "language": a.language,
                       "ref_text": a.ref_text, "xvec": a.xvec, "greedy": a.greedy,
                       "non_streaming_mode": a.mode != "clone"}, f, ensure_ascii=False)
    if a.mode == "decode":
        codes = torch.from_numpy(np.load(a.codes)).long()
        wavs, sr = m.speech_tokenizer.decode([{"audio_codes": codes}])
        sf.write(os.path.join(a.out, "decoded.wav"), wavs[0], sr)
        np.save(os.path.join(a.out, "decoded.npy"), np.asarray(wavs[0], dtype=np.float32))
        print("decoded", len(wavs[0]), "samples")
        return

    # the talker's prefill input and first logits, caught on the way
    seen = {}

    def hook(mod, args, kwargs, out):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1 and "prefill" not in seen:
            seen["prefill"] = e[0].float().numpy()
            seen["logits0"] = out.logits[0, -1].float().numpy()
    m.talker.register_forward_hook(hook, with_kwargs=True)

    gen = dict(max_new_tokens=a.max_new_tokens)
    if a.greedy:
        gen.update(do_sample=False, subtalker_dosample=False, repetition_penalty=1.0)
    # the codes, caught from the codec's decode
    orig = m.speech_tokenizer.decode

    def dec(items, *x, **k):
        seen["codes"] = items[0]["audio_codes"].cpu().numpy()
        return orig(items, *x, **k)
    m.speech_tokenizer.decode = dec

    if a.mode == "custom":
        wavs, sr = tts.generate_custom_voice(text=a.text, language=a.language, speaker=a.speaker, instruct=a.instruct or None, **gen)
    elif a.mode == "design":
        wavs, sr = tts.generate_voice_design(text=a.text, language=a.language, instruct=a.instruct, **gen)
    else:
        import librosa
        wav, _ = librosa.load(a.ref_audio, sr=24000, mono=True)
        sf.write(os.path.join(a.out, "ref.wav"), wav, 24000, subtype="FLOAT")
        items = tts.create_voice_clone_prompt(ref_audio=a.ref_audio, ref_text=a.ref_text, x_vector_only_mode=a.xvec)
        np.save(os.path.join(a.out, "spk.npy"), items[0].ref_spk_embedding.float().numpy())
        if items[0].ref_code is not None:
            np.save(os.path.join(a.out, "ref_codes.npy"), items[0].ref_code.numpy())
        wavs, sr = tts.generate_voice_clone(text=a.text, language=a.language, voice_clone_prompt=items, **gen)
    for k, v in seen.items():
        np.save(os.path.join(a.out, f"{k}.npy"), v)
    sf.write(os.path.join(a.out, "speech.wav"), wavs[0], sr)
    np.save(os.path.join(a.out, "speech.npy"), np.asarray(wavs[0], dtype=np.float32))
    print(f"{len(wavs[0]) / sr:.2f} s of speech, {seen['codes'].shape[0]} frames; prefill {seen['prefill'].shape}")


if __name__ == "__main__":
    main()
