"""SheetSage2 (M-A-P's recording-to-lead-sheet transcriber, non-commercial: CC BY-NC 4.0) as upstream runs it - the
reference for nextsycl's port (audio/yue2/src/sheet): the stages' tensors dumped for `nextsycl audio check DIR --model
yue2-3b-q8 --stages sheet...`, and the transcription itself (the tokens of every window, the stitched events, the ABC
with and without chords) as the result the port must give.

    python -I sheetsage_ref.py --ss2 SHEETSAGE2_DIR --mert MERT_DIR --audio FILE --out DIR [--device cuda:0]
        [--dtype bf16|fp32] [--tf-rows N] [--bf16-params]

SHEETSAGE2_DIR is m-a-p/SheetSage2 rev b83c3d79 as downloaded (its code and its adapter weights), MERT_DIR its parent
m-a-p/MERT-v2-FullSong rev d8ba1c74 (the code checks its sha256s): the model is loaded exactly as `AutoModel.from_pretrained`
loads it - the parent in float32, the attention adapters merged in float32 - and transcribed with upstream's own
`model.transcribe` (FFmpeg's decode to 24 kHz mono, the 300 s windows, the grammar-constrained greedy decode under bf16
autocast - its default -, the stitching, the notation). The stage dumps are taken beside it from the same objects, so
they are what that transcription saw.

--bf16-params rounds every parameter to bfloat16 after the load (the norms, biases, convolutions and the layer mix's
weights too - the matrices are bf16 under autocast anyway): the values nextsycl's file holds (scragnog's
sheetsage2-f16.gguf is converted from the Comfy-Org merge, bf16 throughout), so the port can be checked against
upstream's code on the very same numbers. Measured on the test recording: upstream's own transcription changes with it
(1824 tokens instead of 1793; 4 of the ABC's lines), so the port's result is to be compared with this run.

Dumps (DIR; .npy float32 unless said):
  wave           [samples]           the recording at 24 kHz mono, as load_audio decodes it (FFmpeg)
  mel            [30000, 128]        the first window's log-mel features, normalized (float32, as upstream: no autocast)
  sub            [7500, 1024]        the subsampled frames (the ConvNeXt stack's output: state 0 of the layer mix)
  block{i}       [7500, 1024]        conformer block i's output, i in 0, 1, 11, 23
  mixed          [7500, 1024]        the softmax layer mix
  memory         [7500, 512]         the projected memory the decoder attends to (the run's dtype)
  memory_fp32    [7500, 512]         the same in float32 throughout (how far bf16 itself is from the exact)
  tokens{w}      int32               window w's tokens as generated (prefix, the events, EOS)
  tf_rows        int32               decoder steps whose logits are dumped (positions in tokens0; the logits predict the next)
  tf_logits      [rows, 31678]       the first window's teacher-forced logits there (one causal pass over tokens0)
  tf_argmax      int32               the grammar-masked argmax at every step of tokens0 (what greedy would pick there)
  events.json                        the stitched events (upstream's decoded events with their absolute times)
  abc_full.abc / abc_melody.abc      the ABC (melody_only False / True) - the expected results
  result.json                        upstream's result (windows, timings, warnings, peak memory)
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_grad_enabled(False)


def save(out, name, t):
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(a.astype(np.float32)))


def ids(out, name, v):
    v = v.tolist() if torch.is_tensor(v) else list(v)
    np.save(os.path.join(out, name + ".npy"), np.asarray(v, dtype=np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss2", required=True)
    ap.add_argument("--mert", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--tf-rows", type=int, default=48)
    ap.add_argument("--bf16-params", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModel
    dev = torch.device(a.device)
    t0 = time.monotonic()
    model = AutoModel.from_pretrained(a.ss2, trust_remote_code=True, local_files_only=True, base_model_path=a.mert).eval().to(dev)
    print(f"loaded in {time.monotonic() - t0:.1f} s", flush=True)
    if a.bf16_params:
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    # upstream's own modules, as transformers imported them (the package the model class lives in)
    pkg = model.__class__.__module__.rsplit(".", 1)[0]
    audio_m = importlib.import_module(pkg + ".audio_sheetsage2")
    gen_m = importlib.import_module(pkg + ".generation_sheetsage2")
    exp_m = importlib.import_module(pkg + ".exports_sheetsage2")
    tok = model.tokenizer
    autocast = torch.bfloat16 if a.dtype == "bf16" else None

    # the transcription, exactly as upstream makes it (full: chords too)
    t1 = time.monotonic()
    res = model.transcribe(a.audio, output_dir=os.path.join(a.out, "upstream"), dtype=a.dtype)
    took = time.monotonic() - t1
    print(f"transcribed in {took:.1f} s: {len(res['tokens'])} windows, {res['num_events']} events", flush=True)
    for w, t in enumerate(res["tokens"]):
        ids(a.out, f"tokens{w}", t)
    events = res["events"]
    with open(os.path.join(a.out, "events.json"), "w") as f:
        json.dump(events, f, indent=1)
    with open(os.path.join(a.out, "abc_full.abc"), "w") as f:
        f.write(res["abc"] or "")
    # melody_only: the same stitched events through the notation without chords (what transcribe(melody_only=True)
    # makes: the model's half is the same)
    decoded = dict(schema_version=tok.schema_version, prompts=list(res["prompts"]), events=events, has_eos=True)
    mel_only = exp_m.export_result(decoded, tok, None, res["duration_seconds"], paper=False, melody_only=True)
    with open(os.path.join(a.out, "abc_melody.abc"), "w") as f:
        f.write(mel_only["payload"]["abc"] or "")
    meta = res.get("_metadata") or {k: v for k, v in res.items() if k in ("duration_seconds", "windows", "warnings", "peak_gpu_mib", "elapsed_seconds", "abc_error")}
    meta = {k: v for k, v in meta.items() if k not in ("tokens", "tensors", "events", "midi", "midis", "labs", "playback", "abc")}
    meta["transcribe_seconds"] = took
    meta["abc_error_melody"] = mel_only.get("abc_error")
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)

    # the stages of the first window, from the same objects
    wave = audio_m.load_audio(a.audio)
    save(a.out, "wave", wave)
    seg = audio_m.slice_audio(wave, 0.0, float(model.hparams.input_audio_length))[None].to(dev)
    padded, _ = model._prepare_audio(seg)
    save(a.out, "mel", model.encoder.feature_extractor(padded)[0])
    with gen_m.inference_autocast(dev, autocast):
        enc = model.get_audio_features(seg, output_hidden_states=True)
    save(a.out, "sub", enc.input_hidden_state[0])
    for i in (0, 1, 11, 23):
        save(a.out, f"block{i}", enc.backbone_hidden_states[i][0])
    save(a.out, "mixed", enc.mixed_hidden_state[0])
    save(a.out, "memory", enc.encoder_last_hidden_state[0])
    memory = enc.encoder_last_hidden_state
    del enc
    save(a.out, "memory_fp32", model.get_audio_features(seg).encoder_last_hidden_state[0])

    # the decoder teacher-forced over the first window's tokens: its logits at some steps, the masked argmax at all
    t = res["tokens"][0].tolist()
    with gen_m.inference_autocast(dev, autocast):
        logits, _ = model.decode(memory, torch.tensor([t[:-1]], device=dev), use_cache=False)
    logits = logits[0].float()
    prefix = tok.prompt_prefix(res["prompts"])
    state = gen_m.PromptGrammarState(tok)
    for x in prefix[prefix.index(tok.out_token) + 1:]:
        state.update(x)
    am = []
    for p in range(len(prefix) - 1, len(t) - 1):
        allowed = state.allowed(logits.device)
        am.append(int(logits[p].masked_fill(~allowed, float("-inf")).argmax()))
        state.update(t[p + 1])
    ids(a.out, "tf_argmax", am)
    agree = sum(int(x == y) for x, y in zip(am, t[len(prefix):]))
    print(f"teacher-forced masked argmax agrees with the generated tokens at {agree} of {len(am)} steps", flush=True)
    n = len(t) - 1
    rows = sorted(set([len(prefix) - 1 + i for i in range(8)] + list(np.linspace(len(prefix) - 1, n - 1, a.tf_rows).astype(int))))
    ids(a.out, "tf_rows", rows)
    save(a.out, "tf_logits", logits[rows])
    if dev.type == "cuda":
        print(f"peak {torch.cuda.max_memory_allocated(dev) / 2**20:.0f} MiB", flush=True)
    print(res["abc"])


if __name__ == "__main__":
    main()
