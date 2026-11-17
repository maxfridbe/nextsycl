"""edit_ref.py SRC STAGE DIR: the reference side of the edit parity, in DIR (our dumps: ids.npy, ids.cond.json, rows.npy,
codes.npy, edit.json; ns_forced.npy from `forced` with the edit adapter, DIR-base/ns_forced.npy without).

  request  request.json for zonos3-cli --request --build-prompt (the fill's text rows exactly as Zyphra's ggml server's
           EditEngine.fill builds them: the window text normalized in en_us, the clip's speaker embedding, no silence)
  layout   the edit prompt by Zyphra's model/src/zonos3/tts/edit.py edit_prompt_tokens over those text rows, our window
           codes and spans -> ref_ids.npy; compared with ours (ids.npy, ids.cond.json vs text_ids.cond.json); then
           full.npy = our prompt + our generated rows (+ full.cond.json) for zonos3-cli --validate
  compare  ggml/ (edit adapter) and ggml-base/ (none) logits at the prompt's last row and every fed row against ours
"""
import json
import os
import sys
import types

import numpy as np

SRC, STAGE, D = sys.argv[1:4]
sys.path.insert(0, os.path.join(SRC, "model", "src"))
_tn = types.ModuleType("zyphra_textnorm")
_tn.collapse_line_breaks = lambda t: t
_tn.Normalizer = None
sys.modules["zyphra_textnorm"] = _tn
j = lambda f: os.path.join(D, f)  # noqa: E731
meta = json.load(open(j("edit.json")))
cond = json.load(open(j("ids.cond.json")))

if STAGE == "request":
    import base64
    import io

    b = io.BytesIO()
    np.save(b, np.asarray(cond["speaker_embedding"], dtype=np.float32))
    req = {"text": meta["text"], "language": "en_us", "text_normalization": True, "silence_prefill_frames": 0,
           "speaker_mode": "embedding", "speaker_embedding_base64": base64.b64encode(b.getvalue()).decode()}
    json.dump(req, open(j("request.json"), "w"), ensure_ascii=False)
    print("request.json:", meta["text"][:80], "...")
elif STAGE == "layout":
    import torch
    from zonos3.tts.edit import edit_prompt_tokens

    cfg = types.SimpleNamespace(n_codebooks=10, audio_pad_id=4097, text_vocab=8196)
    text = np.load(j("text_ids.npy")).astype(np.int64)
    codes = np.load(j("codes.npy")).astype(np.int64)
    spans = [tuple(s) for s in meta["spans"]]
    ref = edit_prompt_tokens(torch.as_tensor(text, dtype=torch.int32), codes, spans, [], cfg, span_end_id=4098).numpy()
    np.save(j("ref_ids.npy"), ref.astype(np.float32))
    ours = np.load(j("ids.npy")).astype(np.int64)
    same = ref.shape == ours.shape and bool((ref == ours).all())
    print(f"edit prompt: ours {ours.shape} ref {ref.shape}, identical: {same}")
    if not same:
        n = min(len(ref), len(ours))
        d = np.argwhere(ref[:n] != ours[:n])
        print("  first differing (row, col):", d[:10].tolist())
        for r in sorted({int(x[0]) for x in d[:5]}):
            print(f"  row {r}: ours {ours[r].tolist()} ref {ref[r].tolist()}")
    tc = json.load(open(j("text_ids.cond.json")))
    keys = ("speaker_row", "condition_row", "condition_buckets", "prefill_frames")
    print("  cond:", {k: (cond.get(k), tc.get(k)) for k in keys},
          "speaker embedding max diff", float(np.abs(np.asarray(cond["speaker_embedding"]) - np.asarray(tc["speaker_embedding"])).max()))
    rows = np.load(j("rows.npy")).astype(np.float32)
    np.save(j("full.npy"), np.concatenate([ours.astype(np.float32), rows]))
    json.dump(tc, open(j("full.cond.json"), "w"))  # the reference CLI's own sidecar format (the same values, checked above)
    print(f"  full.npy: {len(ours)} prompt rows + {len(rows)} generated rows; eos frame {meta['eos_frame']}, min frames {meta['min_frames']}")
elif STAGE == "compare":
    def cos(a, b):
        a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    def rel(a, b):
        return float(np.linalg.norm(a.astype(np.float64) - b) / (np.linalg.norm(b.astype(np.float64)) + 1e-30))

    P = len(np.load(j("ids.npy")))
    rows = np.load(j("rows.npy")).astype(np.int64)
    for ours_dir, ggml, label in ((D, "ggml", "edit LM (adapter on)"), (D + "-base", "ggml-base", "base LM, same rows"),
                                  (D, "ggml-base", "ours with the adapter vs the reference without (the adapter's effect)")):
        a = np.load(os.path.join(ours_dir, "ns_forced.npy"))
        g = np.load(os.path.join(D, ggml, "logits.npy"))
        g = g.reshape(g.shape[0], a.shape[1], -1)[P - 1:P - 1 + len(a)]
        n = min(len(a), len(g)); a, g = a[:n], g[:n]
        # the special ids (EOA, pad, span-end) as the sampler sees them: compare over the ordinary codes + span-end
        cs = [cos(a[r], g[r]) for r in range(n)]
        am = a[..., :4096].argmax(-1) == g[..., :4096].argmax(-1)
        print(f"{label}: {n} steps; first logits cos {cs[0]:.6f} rel {rel(a[0], g[0]):.2e} argmax {am[0].mean() * 100:.0f}%; "
              f"all: cos min {min(cs):.6f} mean {np.mean(cs):.6f}, rel {rel(a, g):.2e}, argmax agree {am.mean() * 100:.1f}%")
        if ggml == "ggml" and ours_dir == D:
            ranks = []
            for s in range(n - 1):
                for c in range(10):
                    t = rows[s][c]
                    if t < 4096:
                        ranks.append(int((g[s][c][:4096] > g[s][c][t]).sum()))
            ranks = np.array(ranks)
            print(f"  our drawn codes in the reference's logits: top-1 {np.mean(ranks == 0) * 100:.1f}%, top-80 {np.mean(ranks < 80) * 100:.1f}% ({len(ranks)} codes)")
            print(f"  span-end logit (cb0) ours vs ref at the last 3 steps: {[(round(float(a[r][0][4098]), 3), round(float(g[r][0][4098]), 3)) for r in range(max(0, n - 3), n)]}")
