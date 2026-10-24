#!/usr/bin/env bash
# The platform's tests, by layer. Each layer needs more than the one before; run as far as the machine allows.
#
#   tests/run.sh                 layers 1-3: no GPU, no model, a minute or two (what CI runs)
#   tests/run.sh --device HOST   also layer 4 on HOST (ssh): the kernels and the front door on the real GPUs
#   tests/run.sh --api URL       also layer 5 against a running front door (its token from serve.json or --token)
#
# 1. Unit tests: every crate's own (`cargo test`) - the engines' arithmetic, the tokenizers, the catalog, the cache...
# 2. Architecture and consistency (cli/nextsycl/tests): the layers depend only downward, SYCL is the only GPU runtime,
#    every environment variable is documented, every card has an engine and every engine a card, every LLM template
#    keeps a cacheable prefix.
# 3. The front door end to end (glue/serve/tests/front_door.rs): the real front door in this process over a temporary
#    registry, called as a client calls it - the envelope, the token, maintenance, system.ping, not-downloaded models,
#    how a mesh given to 3D is routed, the OpenAPI document against the methods. Nothing is started.
# 4. Device smoke (--device): `gpus` (what the box sees), each kind's `selftest` (its kernel library loads and a kernel
#    runs), `kernels` on a real file when HOST has one; the front door's integration tests again beside the hardware.
# 5. API reference (--api): tests/api/*.py - whole jobs through the API (a film, an OCR, a dialogue): minutes to an
#    hour, a real model each; named in tests/api/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

device="" api="" token=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --api) api="$2"; shift 2 ;;
    --token) token="$2"; shift 2 ;;
    -h|--help) sed -n 2,19p "$0"; exit 0 ;;
    *) echo "unknown option $1 (tests/run.sh --help)"; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }

say "1-3. unit, architecture, front door (cargo test --release --workspace)"
cargo test --release --workspace 2>&1 | tee /tmp/nextsycl-tests.log | grep -E "^test result|FAILED|panicked|^error" | sort | uniq -c | sort -rn | head -20
ok=$(grep -c "^test result: ok" /tmp/nextsycl-tests.log || true)
if grep -qE "^test result: FAILED|^error(\[|:)" /tmp/nextsycl-tests.log; then
  echo "FAILED (the log: /tmp/nextsycl-tests.log)"; exit 1
fi
echo "$ok test groups ok"

if [ -n "$device" ]; then
  say "4. device smoke on $device"
  ssh "$device" 'cd ~/src/nextsycl && export NS_IMAGE=${NS_IMAGE:-localhost/h3-build}
    echo "-- gpus";  ./dist/nextsycl gpus
    for k in llm image audio video model3d; do echo "-- $k selftest"; ./dist/nextsycl $k selftest || exit 1; done'
fi

if [ -n "$api" ]; then
  say "5. API reference tests against $api"
  for t in tests/api/test-ocr.py tests/api/harbour_dialogue.py tests/api/kaiju_lighthouse.py; do
    echo "-- $t --quick"
    python3 "$t" --url "$api" ${token:+--token "$token"} $([ "$t" = tests/api/test-ocr.py ] || echo --quick)
  done
fi
say "done"
