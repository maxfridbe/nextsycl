"""Sophon-Realtime Flash's talker as upstream feeds it, on the CPU in float32 - the reference for `nextsycl audio check
DIR --stages talker` (audio/sophon check.rs).

    python -I talker_ref.py RELEASE_DIR CODE_DIR --text "They would be bagels! [laughter] Oh, that is a classic." \
        [--voice chelsie] [--lang en] [--frames 120] --out DIR

RELEASE_DIR: Sophon-Realtime-32B-Flash as downloaded (thinker/, talker/, voices/). CODE_DIR: the sophon-realtime
repository (github.com/adventists-ai/sophon-realtime); its flash/plugin/sophon_realtime/proc.py's tag_clean and
tools.py's cut_tool are taken from it as they are (proc.py itself imports vLLM-Omni, so only those definitions are
read out of it), then the text is tokenized the way thinker2talker_async_chunk does it (the thinker's tokenizer,
add_special_tokens=False).

The talker: talker.py's SophonTTSTalker._prefill builds the official Qwen3-TTS streaming x-vector prompt (the role,
tts_pad x 5 + tts_bos over codec [think, think_bos, language, think_eos, x-vector, pad], the first text token over
codec bos; the rest of the text then tts_eos fed a frame each). With the whole text known that is exactly what the
qwen-tts package's generate() builds for a voice_clone_prompt in x_vector_only_mode with non_streaming_mode=False, so
the qwen-tts model (the Flash talker's weights) runs it, greedy.

--ref-wav PATH: the voice is a recording's clone, as talker.py's _load_voice makes it for voices/<name>.wav: the
channels averaged, the first SOPHON_MAX_REF_SECONDS (15) kept, made 24 kHz (vLLM's AudioResampler: librosa), the
loaded talker's speaker encoder over qwen-tts's mel. Upstream runs the encoder in the talker's dtype (bf16) and keeps
the x-vector bf16: xvec.npy is the float32 x-vector rounded to bf16 (the prompt takes it), xvec_f32.npy unrounded.
--talker DIR: another talker directory (Qwen3-TTS-12Hz-1.7B-Base's, for an A/B; the release's by default).

--ref-wav PATH --ref-text TEXT: an in-context clone instead (nextsycl's NS_SOPHON_CLONE=icl), as Qwen3-TTS's own
generate_voice_clone builds it in ICL mode: the recording kept whole (or its first --ref-seconds), made 24 kHz (librosa)
and written as ref24k.wav (request.json's reference: the engine reads the same samples); qwen-tts's
create_voice_clone_prompt (the speech tokenizer's codes, the speaker embedding in float32), the transcript tokenized by
qwen-tts's processor in its reference template ("<|im_start|>assistant\n{text}<|im_end|>\n", [3:-2] taken); generate
with icl_mode, streaming (non_streaming_mode=False), greedy. Dumps ref_ids.npy (the transcript's ids), ref_codes.npy
[F, 16], xvec.npy (float32), and speech.wav decoded with the recording's codes ahead (cut off), as qwen-tts does.

Dumps (DIR): request.json (its "reference" with --ref-wav), xvec.npy and xvec_f32.npy (--ref-wav), ids.npy (the speakable ids), prefill.npy [L, 2048] (the talker's input rows), logits0.npy
(the first codebook's logits after them), codes.npy [T, 16] (greedy), speech.wav.
"""
import argparse
import ast
import importlib.util
import json
import os
import re

import numpy as np
import soundfile as sf
import torch

ROLE = [151644, 77091, 198]                     # "<|im_start|>assistant\n"
TAIL = [151645, 198, 151644, 77091, 198]        # what qwen-tts's input_ids end with (sliced off by generate)
LANG = {"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french", "es": "spanish",
        "it": "italian", "pt": "portuguese", "ru": "russian"}


def upstream_text(code_dir):
    """proc.py's tag_clean (with the tables it reads) and tools.py's cut_tool, from the upstream files; CODE_DIR "-":
    the text as given (a line with no sound tag and no tool call, which both leave as it is)"""
    if code_dir == "-":
        return (lambda t, last: t), (lambda t, last: t)
    plug = os.path.join(code_dir, "flash", "plugin", "sophon_realtime")
    spec = importlib.util.spec_from_file_location("sophon_tools", os.path.join(plug, "tools.py"))
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    src = open(os.path.join(plug, "proc.py"), encoding="utf-8").read()
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("POINT", "SPAN", "CLOSE") for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "tag_clean":
            keep.append(node)
    ns = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "proc.py", "exec"), ns)
    return ns["tag_clean"], tools.cut_tool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("code")
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="chelsie")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--ref-wav", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--ref-seconds", type=float, default=None)
    ap.add_argument("--talker", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(a.out, exist_ok=True)

    tag_clean, cut_tool = upstream_text(a.code)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(a.release, "thinker"))
    text = tag_clean(cut_tool(a.text, True), True).lstrip()
    ids = tok(text, add_special_tokens=False).input_ids if text else []
    print("speakable:", repr(text))
    print("ids:", ids)
    print("pieces:", [tok.decode([i]) for i in ids])
    np.save(os.path.join(a.out, "ids.npy"), np.asarray(ids, dtype=np.int64))
    with open(os.path.join(a.out, "request.json"), "w") as f:
        req = {"text": a.text, "speakable": text, "voice": a.voice, "lang": a.lang}
        if a.ref_wav:
            req["reference"] = os.path.abspath(os.path.join(a.out, "ref24k.wav") if a.ref_text else a.ref_wav)
        if a.ref_text:
            req["ref_text"] = a.ref_text
        json.dump(req, f, ensure_ascii=False)

    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained(a.talker or os.path.join(a.release, "talker"), device_map="cpu", dtype=torch.float32,
                                        attn_implementation="sdpa")
    m = tts.model
    icl = None
    if a.ref_wav and a.ref_text:
        # Qwen3-TTS's in-context clone (qwen_tts generate_voice_clone's ICL path)
        import librosa
        wav, sr = sf.read(a.ref_wav, dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if a.ref_seconds:
            wav = wav[: int(sr * a.ref_seconds)]
        if sr != 24000:
            wav = librosa.resample(np.ascontiguousarray(wav), orig_sr=int(sr), target_sr=24000)
        wav = np.ascontiguousarray(wav.astype(np.float32))
        sf.write(os.path.join(a.out, "ref24k.wav"), wav, 24000, subtype="FLOAT")
        with torch.no_grad():
            items = tts.create_voice_clone_prompt(ref_audio=(wav, 24000), ref_text=a.ref_text, x_vector_only_mode=False)
        icl = tts._prompt_items_to_voice_clone_prompt(items)
        ref_ids = tts._tokenize_texts([tts._build_ref_text(a.ref_text)])[0]
        np.save(os.path.join(a.out, "ref_ids.npy"), ref_ids[0, 3:-2].cpu().numpy().astype(np.int64))
        np.save(os.path.join(a.out, "ref_codes.npy"), np.ascontiguousarray(icl["ref_code"][0].cpu().numpy().astype(np.int64)))
        xvec = icl["ref_spk_embedding"][0].reshape(-1).float()
        np.save(os.path.join(a.out, "xvec.npy"), xvec.numpy())
        print(f"in context: {len(wav) / 24000:.2f} s, {icl['ref_code'][0].shape[0]} frames, transcript ids {ref_ids[0, 3:-2].tolist()}")
    elif a.ref_wav:
        # talker.py's _load_voice for a .wav / .flac, then prompt_embeds_builder's extract_speaker_embedding
        wav, sr = sf.read(a.ref_wav, dtype="float32", always_2d=True)
        wav = np.ascontiguousarray(wav.mean(1)[: int(sr * float(os.environ.get("SOPHON_MAX_REF_SECONDS", "15")))])
        if sr != 24000:
            import librosa
            wav = librosa.resample(wav, orig_sr=int(sr), target_sr=24000)
        with torch.no_grad():
            x32 = m.extract_speaker_embedding(wav.astype(np.float32), 24000).reshape(-1).float()
        np.save(os.path.join(a.out, "xvec_f32.npy"), x32.numpy())
        xvec = x32.to(torch.bfloat16).float()
        np.save(os.path.join(a.out, "xvec.npy"), xvec.numpy())
        print(f"clone: {len(wav) / 24000:.2f} s kept, x-vector norm {float(xvec.norm()):.3f}")
    else:
        # talker.py's voice lookup: <voice>_<lang>.pt, then <voice>.pt
        vdir = os.path.join(a.release, "voices")
        vp = next(p for p in (os.path.join(vdir, f"{a.voice}_{a.lang}.pt"), os.path.join(vdir, f"{a.voice}.pt")) if os.path.exists(p))
        xvec = torch.load(vp, map_location="cpu", weights_only=True).reshape(-1).float()

    seen = {}

    def hook(mod, args, kwargs, out):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1 and "prefill" not in seen:
            seen["prefill"] = e[0].float().numpy()
            seen["logits0"] = out.logits[0, -1].float().numpy()
    m.talker.register_forward_hook(hook, with_kwargs=True)

    input_ids = torch.tensor([ROLE + ids + TAIL], dtype=torch.long)
    prompt = icl or {"x_vector_only_mode": [True], "icl_mode": [False], "ref_spk_embedding": [xvec], "ref_code": None}
    with torch.no_grad():
        codes, _ = m.generate(input_ids=[input_ids], ref_ids=[ref_ids] if icl else None, voice_clone_prompt=prompt,
                              languages=[LANG.get(a.lang, "english")], non_streaming_mode=False, max_new_tokens=a.frames, do_sample=False,
                              subtalker_dosample=False, repetition_penalty=1.0)
    codes = codes[0].cpu().numpy()
    np.save(os.path.join(a.out, "prefill.npy"), np.ascontiguousarray(seen["prefill"]))
    np.save(os.path.join(a.out, "logits0.npy"), np.ascontiguousarray(seen["logits0"]))
    np.save(os.path.join(a.out, "codes.npy"), np.ascontiguousarray(codes.astype(np.int64)))
    if icl:
        rc = icl["ref_code"][0].cpu().long()
        wavs, sr = m.speech_tokenizer.decode([{"audio_codes": torch.cat([rc, torch.from_numpy(codes).long()], 0)}])
        cut = int(len(wavs[0]) * rc.shape[0] / (rc.shape[0] + codes.shape[0]))
        wavs = [wavs[0][cut:]]
    else:
        wavs, sr = m.speech_tokenizer.decode([{"audio_codes": torch.from_numpy(codes).long()}])
    sf.write(os.path.join(a.out, "speech.wav"), wavs[0], sr)
    print(f"prefill {seen['prefill'].shape}, first code {int(np.argmax(seen['logits0']))}, {codes.shape[0]} frames, {len(wavs[0]) / sr:.2f} s")
    print("first frames:", codes[:3].tolist())


if __name__ == "__main__":
    main()
