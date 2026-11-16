"""OmniVoice in its own code (github.com/k2-fsa/OmniVoice: omnivoice/models/omnivoice.py, the Higgs Audio v2 tokenizer
in transformers), on the CPU in float32 - the reference for nextsycl's audio engine (audio/omnivoice): a request's
prompt, the stages' tensors and a deterministic run's codes and sound, dumped as .npy for `nextsycl audio check`.

    python -I ref.py CHECKPOINT_DIR --code OMNIVOICE_SRC [--pylib DIR] --text "..." [--language en] [--instruct "female, low pitch"]
                  [--ref-audio r.wav --ref-text "..."] [--steps 32] [--duration S] --out DIR

CHECKPOINT_DIR is k2-fsa/OmniVoice as downloaded (config.json, model.safetensors, tokenizer.json, audio_tokenizer/);
OMNIVOICE_SRC the upstream repository (its `omnivoice` package is imported from there); --pylib a folder with pydub,
audioop-lts and soundfile (pip install --no-deps --target).

The run is upstream's generate() with position_temperature 0 (the positions to unmask chosen by score alone, no Gumbel
noise) and class_temperature 0 (each position's most likely code): a deterministic run to compare against. Everything
else is upstream's default (32 steps, guidance 2.0, t_shift 0.1, layer penalty 5, denoise, the pre- and post-processing).

Dumps (DIR): request.json (with the target length T, the prompt's length L and where its audio starts); ids.npy [8, L]
(the conditional row's ids: style and text tokens repeated over the 8 codebooks, a recording's codes, T mask codes);
emb0.npy [L + T, 1024] (the first step's input embeddings: the conditional rows, then the unconditional T); layer0.npy,
hidden0.npy (the same rows after the first layer, and the final norm); logits0.npy [2, 8, T, 1025] (the first step's
logits at the target: conditional, unconditional); ids_mid.npy / logits_mid.npy (the same at step steps/2, half the codes
unmasked); tokens.npy [8, T] (the codes made); raw.npy (the codec's decode of them, 24 kHz); out.npy / out.wav (after
the post-processing: the answer). For a recording also ref24.npy (it at 24 kHz as encoded, after the preprocessing),
ref_rms (in request.json), sem16.npy (it at 16 kHz, padded, into HuBERT), semantic.npy [T, 768] (HuBERT's hidden
states' mean, every other frame), esem.npy [768, T], eac.npy [256, T], fc.npy [1024, T] (the quantizer's input) and
ref_codes.npy [8, T].
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--code", required=True)
    ap.add_argument("--pylib")
    ap.add_argument("--text", required=True)
    ap.add_argument("--language")
    ap.add_argument("--instruct")
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path[:0] = [a.code] + ([a.pylib] if a.pylib else [])
    import torch
    import torch.nn.functional as F
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
    import soundfile as sf

    torch.set_num_threads(16)
    os.makedirs(a.out, exist_ok=True)
    save = lambda n, x: np.save(os.path.join(a.out, n), np.ascontiguousarray(x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)))
    m = OmniVoice.from_pretrained(a.ckpt, device_map="cpu", dtype=torch.float32)
    m.eval()
    cfg = OmniVoiceGenerationConfig(num_step=a.steps, position_temperature=0.0, class_temperature=0.0)
    info = {"text": a.text, "language": a.language, "instruct": a.instruct, "ref_text": a.ref_text, "ref_audio": a.ref_audio, "steps": a.steps,
            "duration": a.duration}

    # the recording's codes, stage by stage (create_voice_clone_prompt, then HiggsAudioV2TokenizerModel.encode written out)
    vc = None
    if a.ref_audio:
        vc = m.create_voice_clone_prompt(a.ref_audio, a.ref_text)
        info["ref_rms"] = vc.ref_rms
        info["ref_text_used"] = vc.ref_text
        tok = m.audio_tokenizer
        # the wav as encode saw it: create_voice_clone_prompt again, its preprocessing, without the encode
        from omnivoice.utils.audio import load_audio, remove_silence
        w = load_audio(a.ref_audio, m.sampling_rate)
        rms = float(np.sqrt(np.mean(w ** 2)))
        if 0 < rms < 0.1:
            w = w * 0.1 / rms
        w = remove_silence(w, m.sampling_rate, mid_sil=200, lead_sil=100, trail_sil=200)
        clip = int(w.shape[-1] % tok.config.hop_length)
        w = w[:, :-clip] if clip > 0 else w
        save("ref24.npy", w[0].astype(np.float32))
        with torch.no_grad():
            x = torch.from_numpy(w).float().unsqueeze(0)
            import torchaudio
            x16 = torchaudio.functional.resample(x, tok.config.sample_rate, tok.config.semantic_sample_rate)[:, 0, :]
            x16 = F.pad(x16, (160, 160))
            save("sem16.npy", x16[0])
            sem = tok._extract_semantic_features(x)
            save("semantic.npy", sem[0])
            esem = tok.encoder_semantic(sem.transpose(1, 2))
            save("esem.npy", esem[0])
            if tok._get_conv1d_output_lengths(x.shape[2], tok.acoustic_encoder) != esem.shape[2]:
                eac = tok.acoustic_encoder(F.pad(x, (tok.pad, tok.pad)))
                info["acoustic_pad"] = tok.pad
            else:
                eac = tok.acoustic_encoder(x)
                info["acoustic_pad"] = 0
            save("eac.npy", eac[0])
            emb = torch.cat([eac, esem], dim=1)
            emb = tok.fc(emb.transpose(1, 2)).transpose(1, 2)
            save("fc.npy", emb[0])
            codes = tok.quantizer.encode(emb, None).transpose(0, 1)
        save("ref_codes.npy", vc.ref_audio_tokens.to(torch.int64))
        info["ref_codes_match"] = bool(torch.equal(codes[0], vc.ref_audio_tokens))

    task = m._preprocess_all(text=a.text, language=a.language, voice_clone_prompt=vc, instruct=a.instruct, duration=a.duration)
    T = task.target_lens[0]
    inp = m._prepare_inference_inputs(task.texts[0], T, task.ref_texts[0], task.ref_audio_tokens[0], task.langs[0], task.instructs[0], cfg.denoise)
    ids = inp["input_ids"][0]
    L = ids.shape[1]
    info.update({"T": T, "L": L, "audio_start": int(inp["audio_mask"][0].float().argmax()), "lang": task.langs[0], "instruct_used": task.instructs[0]})
    save("ids.npy", ids.to(torch.int64))

    # the steps' tensors: hooks on the embeddings, the first layer, the final norm, the logits
    keep = {"step": 0}
    mid = a.steps // 2
    rows = lambda t: torch.cat([t[0, :L], t[1, :T]], 0)
    orig_embed = m._prepare_embed_inputs

    def embed(input_ids, audio_mask):
        e = orig_embed(input_ids, audio_mask)
        if keep["step"] == 0:
            save("emb0.npy", rows(e))
        if keep["step"] == mid:
            save("ids_mid.npy", input_ids[0].to(torch.int64))
        return e

    m._prepare_embed_inputs = embed
    m.llm.layers[0].register_forward_hook(lambda mod, i, o: save("layer0.npy", rows(o[0] if isinstance(o, tuple) else o)) if keep["step"] == 0 else None)
    m.llm.norm.register_forward_hook(lambda mod, i, o: save("hidden0.npy", rows(o)) if keep["step"] == 0 else None)
    orig_fwd = m.forward

    def fwd(*args, **kw):
        out = orig_fwd(*args, **kw)
        lg = out.logits
        both = torch.stack([lg[0, :, L - T:L, :], lg[1, :, :T, :]], 0)
        if keep["step"] == 0:
            save("logits0.npy", both)
        if keep["step"] == mid:
            save("logits_mid.npy", both)
        keep["step"] += 1
        return out

    m.forward = fwd
    with torch.no_grad():
        toks = m._generate_iterative(task, cfg)[0]
        save("tokens.npy", toks.to(torch.int64))
        raw = m.audio_tokenizer.decode(toks.unsqueeze(0)).audio_values[0].cpu().numpy()
        save("raw.npy", raw[0])
        out = m._post_process_audio(raw, ref_rms=task.ref_rms[0], gen_config=cfg)[0]
    save("out.npy", out.astype(np.float32))
    sf.write(os.path.join(a.out, "out.wav"), out, m.sampling_rate)
    json.dump(info, open(os.path.join(a.out, "request.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
