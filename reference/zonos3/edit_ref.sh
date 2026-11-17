#!/bin/bash
# edit_ref.sh DIR: the reference side of the edit parity for one case dir (our dumps there, NS_Z3_EDIT_DUMP + `audio
# check DIR --stages edit` then `--stages forced`; DIR-base: the same rows forced without the adapter; edit_ref.py's
# stages). Zyphra's ggml CLIs (github.com/Zyphra/ZONOS3-Mini servers/ggml, MIT) on the CPU in the build image:
# zonos3-cli --request --build-prompt (the release build, SRC) and a patched copy (EREF) where servers/ggml/src/main.cpp
#   - lets --accent name the speech-edit adapter (`if (extras.accent < 0)` where it refused accents.bank.edit),
#   - accepts --accent with --validate (the "require --generate or --tts" check skips it for --validate),
#   - sets meta.cond.accent / accent_strength from --accent before zonos3_validate (as pyhost.cpp's fill does).
# Comparison only: nothing here runs in nextsycl.
set -euo pipefail
D=$1
M=/var/mnt/4TBSSD/nextsycl-models/zonos3-mini
SRC=/var/mnt/2TBSSD/nextsycl-ref/zonos3-src/ZONOS3-Mini
EREF=/var/mnt/2TBSSD/nextsycl-ref/zonos3-editref/ZONOS3-Mini
PY="/var/mnt/2TBSSD/nextsycl-ref/venv/bin/python -I"
S=$(cd "$(dirname "$0")" && pwd)
run() { nice -n 19 podman run --rm --security-opt label=disable --cpus 8 -v "$D:$D" -v /var/mnt/2TBSSD/nextsycl-ref:/var/mnt/2TBSSD/nextsycl-ref:ro \
          -v $M:$M:ro -w "$D" localhost/h3-build "$@"; }
$PY $S/edit_ref.py $SRC request "$D"
run env LD_LIBRARY_PATH=$SRC/servers/ggml/build/ggml/src $SRC/servers/ggml/build/zonos3-cli $M/backbone-ptq-edit.gguf --request request.json --build-prompt text_ids.npy --cpu
$PY $S/edit_ref.py $SRC layout "$D"
B=$EREF/servers/ggml/build
run env LD_LIBRARY_PATH=$B/ggml/src $B/zonos3-cli $M/backbone-ptq-edit.gguf --validate full.npy ggml --accent edit --accents $M/edit/accents.gguf --cpu 2>&1 | grep -E "validate|wrote|error" || true
run env LD_LIBRARY_PATH=$B/ggml/src $B/zonos3-cli $M/backbone-ptq-edit.gguf --validate full.npy ggml-base --cpu 2>&1 | grep -E "validate|wrote|error" || true
$PY $S/edit_ref.py $SRC compare "$D"
