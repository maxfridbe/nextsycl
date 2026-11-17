"""ipa_ref.py SRC PT VOICE_EMB OUT CASES.json: Zyphra's plain-PyTorch reference (model/src/zonos3/reference, CPU, f32)
over each case {name, text, seeds}: the prompt rows (OUT/<name>.ids.npy, the text column = the token ids fed) and, per
seed, the take's undelayed codes (OUT/<name>-s<seed>.codes.npy, [frames, 10]) - decoded later by zac-cli (keep2), the
release's 8 GB PyTorch generator is not used. speaker_mode embedding (VOICE_EMB), text normalization off."""
import base64
import json
import os
import sys
import time

SRC, PT, EMB, OUT, CASES = sys.argv[1:6]
sys.path.insert(0, os.path.join(SRC, "model", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "8")))

# zyphra_textnorm (a Rust extension) is not installed: normalization is off in every request here, and the texts have
# no line breaks, so its collapse_line_breaks is the identity on them
import types  # noqa: E402

_tn = types.ModuleType("zyphra_textnorm")
_tn.collapse_line_breaks = lambda t: t if "\n" not in t else (_ for _ in ()).throw(ValueError("line break"))
_tn.Normalizer = None
sys.modules["zyphra_textnorm"] = _tn

from zonos3.models.text_extra import TextExtraRows  # noqa: E402
from zonos3.reference import synth as S  # noqa: E402
from zonos3.reference.lm import Zonos3LM  # noqa: E402

lm = Zonos3LM.load(PT, device=torch.device("cpu"), dtype=torch.float32)
assets = S.ReferenceAssets.load()
assets.grow(lm)
lm.append_text_rows(TextExtraRows.load(os.path.join(PT, "text_rows", "pause_tokens_s5000.pt")))
emb = base64.b64encode(open(EMB, "rb").read()).decode()
captured = []


def fake_codec_decode(lm_, assets_, device, decoder):
    def decode(codes):
        a = np.asarray(codes, dtype=np.int64)
        captured.append(a.copy())
        return np.zeros(a.shape[0] * 2940, dtype=np.float32)
    return decode


S._codec_decode = fake_codec_decode
os.makedirs(OUT, exist_ok=True)
for case in json.load(open(CASES)):
    for seed in case["seeds"]:
        req = {"text": case["text"], "language": "en_us", "text_normalization": False, "seed": seed,
               "speaker_mode": "embedding", "speaker_embedding_base64": emb, "speaker_embedding_name": "voice.npy",
               "stream": False}
        plan = S.plan_prompt(lm, req, assets=assets)
        np.save(os.path.join(OUT, f"{case['name']}.ids.npy"), plan.cond.input_ids.to(torch.int64).numpy())
        captured.clear()
        t0 = time.time()
        res = S.synthesize(lm, req, assets=assets)
        codes = max(captured, key=len) if captured else np.zeros((0, 10), dtype=np.int64)
        np.save(os.path.join(OUT, f"{case['name']}-s{seed}.codes.npy"), codes.astype(np.float32))
        print(f"{case['name']} seed {seed}: {len(res.rows)} rows, eos {res.eos_frame}, {len(codes)} frames decoded"
              f" ({len(captured)} decode calls), {time.time() - t0:.1f} s", flush=True)
