#!/usr/bin/env bash
# reference/zonos3/ref.sh MODELS OUT BUILD: the reference's dumps for `nextsycl audio check OUT --stages prompt,lm,forced,decoder,speaker`,
# made by Zyphra's own runtime (github.com/Zyphra/ZONOS3-Mini, servers/ggml: zonos3-cli, zac-cli, spk-encoder-cli, MIT),
# built for the CPU in a throwaway checkout - for comparison only, never at run time:
#   git clone --recurse-submodules https://github.com/Zyphra/ZONOS3-Mini && cd ZONOS3-Mini/servers/ggml
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target zonos3-cli zac-cli spk-encoder-cli
# MODELS: Zyphra's gguf/ folder (backbone-ptq.gguf, zac-keep2.gguf, spk-encoder.gguf, voices/); BUILD: that build directory.
set -euo pipefail
M=$1; OUT=$2; B=$3
TEXT="Hello from next sickle. This is a parity check of the backbone."
EMB=$M/voices/English/Savannah_en-American_F_1.embedding.npy
mkdir -p "$OUT/ggml"
cd "$OUT"
printf '%s' "$TEXT" > prompt.txt
cp "$EMB" voice.embedding.npy
# the prompt (ids.npy + ids.cond.json) and the backbone's dumps over it
"$B/zonos3-cli" "$M/backbone-ptq.gguf" --build-prompt "$TEXT" ids.npy --speaker-embedding "$EMB" --condition-buckets 4,5,12
"$B/zonos3-cli" "$M/backbone-ptq.gguf" --validate ids.npy ggml --cpu
# a take: its rows and every step's logits (teacher forcing compares against these)
"$B/zonos3-cli" "$M/backbone-ptq.gguf" --tts "$TEXT" out.npy --speaker-embedding "$EMB" --condition-buckets 4,5,12 --seed 1 --dump-logits logits.npy --cpu
python3 - <<'PY'
import numpy as np
o = np.load('out.npy').astype(np.int64)
steps = np.load('logits.npy').shape[0] - 1
np.save('rows.npy', o[3:3 + steps].astype(np.float32))
T = len(o) - 9
for i, r in enumerate(o):
    cols = [c for c in range(10) if r[c] == 4096]
    if cols:
        T = min(T, i - max(cols))
codes = np.array([[o[f + c][c] for c in range(10)] for f in range(T)], dtype=np.float32)
np.save('codes.npy', codes)
d = np.full((T + 9, 10), 4097, dtype=np.float32)
for r in range(T + 9):
    for k in range(10):
        if 0 <= r - k < T:
            d[r][k] = codes[r - k][k]
np.save('delayed.npy', d)
PY
# the decoder over the take's frames (zac-cli undelays the rows itself), before the output gain
"$B/zac-cli" "$M/zac-keep2.gguf" delayed.npy zac_pcm.npy --skip-frames 0 --headroom-db 0 --cpu
# the speaker encoder over a recording at 44.1 kHz
ffmpeg -nostdin -v error -y -i "$M/voices/English/Rupert_en-British_M_1.wav" -ac 1 -ar 44100 -c:a pcm_f32le speaker.wav
"$B/spk-encoder-cli" "$M/spk-encoder.gguf" --audio speaker.wav spk_emb.npy --cpu
