#!/usr/bin/env bash
# tooling/gate.sh <base> <head> [ENGINE...]: the merge gate (CONTRIBUTING.md "Pull requests"). From the paths that
# changed between two commits it works out the engines affected - an engine's own crate or kernels or silo: that engine;
# a silo variant: that engine on the cards the variant matches; a shared part: every engine of the baselines it maps to
# (tooling/gate-baselines.json "shared") - builds <head> in a build copy on the GPU box, runs each engine's quick check
# on each of its cards (the card under the box's GPU lock, the run under a timeout) and compares with the baselines:
# the exact check exactly (the hashes of the prompt's logits and the greedy tokens), speed within the tolerance (pp, tg).
# Prints a pass/fail table to paste into the pull request; exits 1 on a failure.
#
# The baselines are per card, keyed by the card's id (what `nextsycl silo` prints: `arc-pro-b65-160u-32g`, its name,
# units and memory), so a contributor gates on hardware the project does not own: run it on your own box
# (NS_GATE_HOST=local, no ssh), record your card's baseline on the base commit (NS_GATE_RECORD=1) and commit it in the
# same pull request, then compare the head against it and paste the table.
#
#   NS_GATE_HOST=hardac1 NS_GATE_DIR=src/nextsycl-gate tooling/gate.sh origin/main HEAD
#   NS_GATE_RECORD=1 tooling/gate.sh HEAD HEAD llm/qwen35     # measure and write the baselines (on the base commit)
#   NS_GATE_HOST=local NS_GATE_GPUS=0 NS_GATE_PATHS=/var/mnt/2TBSSD/qwen38-27b=$HOME/models/qwen38-27b \
#     NS_GATE_RECORD=1 tooling/gate.sh origin/main origin/main llm/qwen35   # your card's baseline, on your box
#
# NS_GATE_HOST: the GPU box over ssh (empty or `local`: this machine); NS_GATE_DIR: the build copy there - never the
# production checkout - holding its own nextsycl.conf (NS_IMAGE, a registry of its own, no NS_STUDIO_MODES; default
# src/nextsycl-gate over ssh, ~/.cache/nextsycl-gate here); NS_GATE_LOCK: the GPU lock script (gpu_lock / gpu_unlock;
# `none`: no lock - the default here when ~/src/gpu-lock.sh is not there); NS_GATE_GPUS: the GPU indexes to run on
# (default: each engine's `gpus`, hardac1's); NS_GATE_PATHS: FROM=TO[,FROM=TO] - the baselines' model paths (in `run`
# and `mounts`) where they are on this box; NS_GATE_RECORD=1: record instead of compare.
set -euo pipefail
set -f   # no file-name globbing: the baselines' engine patterns ("llm/*") are matched, not expanded
cd "$(dirname "$0")/.."
BASE=${1:?gate.sh <base> <head> [ENGINE...]}; HEAD=${2:?gate.sh <base> <head> [ENGINE...]}; shift 2
HOST=${NS_GATE_HOST:-}
[ "$HOST" = local ] && HOST=
if [ -n "$HOST" ]; then
  DIR=${NS_GATE_DIR:-src/nextsycl-gate}
  LOCK=${NS_GATE_LOCK:-"~/src/gpu-lock.sh"}
else
  DIR=${NS_GATE_DIR:-$HOME/.cache/nextsycl-gate}
  LOCK=${NS_GATE_LOCK:-$([ -f ~/src/gpu-lock.sh ] && echo "~/src/gpu-lock.sh" || echo none)}
  mkdir -p "$DIR"
fi
RECORD=${NS_GATE_RECORD:-0}
GPUS=${NS_GATE_GPUS:-}
PATHS=${NS_GATE_PATHS:-}
B=tooling/gate-baselines.json
on() { if [ -n "$HOST" ]; then ssh "$HOST" "cd $DIR && $1"; else (cd "$DIR" && bash -c "$1"); fi; }
# the GPU lock, or none
if [ "$LOCK" = none ]; then locking="gpu_lock() { :; }; gpu_unlock() { :; }"; else locking=". $LOCK"; fi
# the baselines' paths where they are on this box (NS_GATE_PATHS)
mapped() { local out=$1 pair; IFS=, read -ra pairs <<< "$PATHS"; for pair in "${pairs[@]}"; do [ -n "$pair" ] && out=${out//"${pair%%=*}"/"${pair#*=}"}; done; printf '%s' "$out"; }

# ---- what changed, and the engines it touches
changed=$(git diff --name-only "$BASE" "$HEAD")
declare -A want=()     # engine -> "all" | "variant:<arch>"
for e in "$@"; do want[$e]=all; done
known=$(jq -r '.engines | keys[]' "$B")
for p in $changed; do
  case "$p" in
    kernels/*/*/silo/variants/*)
      k=${p#kernels/}; kind=${k%%/*}; k=${k#*/}; arch=${k%%/*}
      [ "${want[$kind/$arch]:-}" = all ] || want[$kind/$arch]="variant:$arch" ;;
    llm/contract/*|image/contract/*|video/contract/*|audio/contract/*|model3d/contract/*) ;;
    kernels/llm/common/*|kernels/ns/*|kernels/strata/*|kernels/diffusion/*|crates/*|*/contract/*) ;;
    kernels/*/*/*|llm/*/*|image/*/*|video/*/*|audio/*/*|model3d/*/*)
      k=${p#kernels/}; kind=${k%%/*}; k=${k#*/}; arch=${k%%/*}; want[$kind/$arch]=all ;;
  esac
  # a shared part: the engines the baselines map it to ("*" every one, "llm/*" a kind's)
  for prefix in $(jq -r '.shared | keys[]' "$B"); do
    [[ "$p" == "$prefix"* ]] || continue
    for pat in $(jq -r --arg k "$prefix" '.shared[$k][]' "$B"); do
      for eng in $known; do [[ "$eng" == $pat ]] && want[$eng]=all; done
    done
  done
done
if [ ${#want[@]} -eq 0 ]; then echo "gate: no engine is affected by $BASE..$HEAD"; exit 0; fi

# ---- the build copy: <head>'s tree (files that changed get a new time, files gone are removed), then the build
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git archive "$HEAD" | tar -x -C "$tmp"
dest=$DIR/; [ -n "$HOST" ] && dest=$HOST:$DIR/
rsync -rlpc --delete --exclude /target --exclude /dist --exclude /nextsycl.conf --exclude /reg --exclude /gate-out "$tmp"/ "$dest"
# the kernels: those changed since the commit the copy's libraries were built from (all of them when that is not known)
built=$(on "cat gate-out/built 2>/dev/null" || true)
if [ -n "$built" ] && git cat-file -e "$built^{commit}" 2>/dev/null && on "test -f dist/libnextsycl-llm.so"; then
  kern=$(git diff --name-only "$built" "$HEAD" -- kernels/)
else
  kern=kernels/build.sh
fi
if [ -n "$kern" ] && grep -qv '\.cpp$' <<< "$kern"; then
  on "./build.sh kernels" > "$tmp/build.log" 2>&1 || { tail -20 "$tmp/build.log"; echo "gate: the kernels do not build"; exit 1; }
elif [ -n "$kern" ]; then
  only=$(sed 's#^kernels/##' <<< "$kern" | tr '\n' ' ')
  on "ONLY='$only' ./build.sh kernels" > "$tmp/build.log" 2>&1 || { tail -20 "$tmp/build.log"; echo "gate: the kernels do not build"; exit 1; }
fi
on "./build.sh rust" >> "$tmp/build.log" 2>&1 || { tail -20 "$tmp/build.log"; echo "gate: the program does not build"; exit 1; }
on "mkdir -p gate-out && echo $(git rev-parse "$HEAD") > gate-out/built"
image=$(on "sed -n 's/^NS_IMAGE=//p' nextsycl.conf 2>/dev/null | head -1" || true); image=${image:-localhost/nextsycl-build}
# each GPU's card id and what it reports (`card N ID {json}`), and the box's name: the baselines are per card
cards=$(on "podman run --rm --security-opt label=disable --device /dev/dri -e ONEAPI_DEVICE_SELECTOR=level_zero:\* -v \$PWD:/src -w /src $image bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; dist/nextsycl silo'" | grep '^card ' || true)
box=$(on "hostname -s" || echo unknown)
card_id() { awk -v g="$1" '$1 == "card" && $2 == g { print $3 }' <<< "$cards"; }
card_dev() { sed -n "s/^card $1 [^ ]* //p" <<< "$cards"; }

# ---- each engine on each card: its run under the lock and a timeout; the logits' and tokens' hashes, pp and tg
tol=$(jq -r '.tolerance' "$B")
rows=(); fail=0
for eng in $(printf '%s\n' "${!want[@]}" | sort); do
  if ! jq -e --arg e "$eng" '.engines[$e]' "$B" >/dev/null; then rows+=("| $eng | - | no baseline | - | - | skip |"); continue; fi
  cmd=$(mapped "$(jq -r --arg e "$eng" '.engines[$e].run' "$B")"); lim=$(jq -r --arg e "$eng" '.engines[$e].timeout_s' "$B")
  mounts=$(mapped "$(jq -r --arg e "$eng" '.engines[$e].mounts[] | "-v \(.):\(.):ro"' "$B" | tr '\n' ' ')")
  list=${GPUS//,/ }; [ -n "$list" ] || list=$(jq -r --arg e "$eng" '.engines[$e].gpus[]' "$B")
  for card in $list; do
    id=$(card_id "$card")
    if [ -z "$id" ]; then rows+=("| $eng | $card | no such GPU here (nextsycl silo) | - | - | FAIL |"); fail=1; continue; fi
    # a variant's change: only the cards it matches (what the program says each GPU takes)
    if [[ "${want[$eng]}" == variant:* ]]; then
      arch=${want[$eng]#variant:}
      took=$(on "podman run --rm --security-opt label=disable --device /dev/dri -e ONEAPI_DEVICE_SELECTOR=level_zero:\* -v \$PWD:/src -w /src $image bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; dist/nextsycl silo $arch'" | sed -n "s/^silo $arch GPU $card: \([^ ]*\).*/\1/p")
      if [ "$took" = default ] || [ -z "$took" ]; then rows+=("| $eng | $card $id | its variant does not match this card | - | - | skip |"); continue; fi
    fi
    if [ "$RECORD" != 1 ] && ! jq -e --arg e "$eng" --arg c "$id" '.engines[$e].cards[$c]' "$B" >/dev/null; then
      rows+=("| $eng | $card $id | no baseline for this card (record one on it: NS_GATE_RECORD=1 on the base commit) | - | - | skip |"); continue
    fi
    run="mkdir -p gate-out; $locking; GPU_LOCK_OWNER=gate gpu_lock $card 1800 $((lim * 2 + 600)) || exit 1
      for i in 1 2; do timeout --kill-after=60 $lim podman run --rm --name gate-$card-\$i --security-opt label=disable --device /dev/dri --memory 24g \
        -e ONEAPI_DEVICE_SELECTOR=level_zero:* -v \$PWD:/src $mounts -w /src $image \
        bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ${cmd//\{gpu\}/$card} --logits-out gate-out/l$card --gen-out gate-out/g$card' > gate-out/run$card-\$i.log 2>&1 || { gpu_unlock; cat gate-out/run$card-\$i.log | tail -5; exit 1; }
        echo \"exact \$(cat gate-out/l$card gate-out/g$card | sha256sum | cut -c1-16)\"; grep '^\[prompt' gate-out/run$card-\$i.log | grep -o '([0-9.]* tok/s)' | tr -d '()a-z/ ' | paste -sd' '; done; gpu_unlock"
    if ! out=$(on "$run"); then rows+=("| $eng | $card $id | the run failed: $(tail -1 <<< "$out") | - | - | FAIL |"); fail=1; continue; fi
    exact=$(grep '^exact' <<< "$out" | sort -u | cut -d' ' -f2 | paste -sd/)
    read -r pp tg < <(grep -v '^exact' <<< "$out" | awk '{ if ($1 > pp) pp = $1; if ($2 > tg) tg = $2 } END { print pp, tg }')
    if [ "$RECORD" = 1 ]; then
      jq --arg e "$eng" --arg c "$id" --arg x "$exact" --argjson pp "$pp" --argjson tg "$tg" --argjson d "$(card_dev "$card")" --arg h "$box" \
        --arg at "$(git rev-parse --short "$HEAD") ($(date +%F))" \
        '.engines[$e].cards[$c] = {exact: $x, pp: $pp, tg: $tg, device: $d, host: $h, recorded: $at}' "$B" > "$tmp/b" && mv "$tmp/b" "$B"
      rows+=("| $eng | $card $id | recorded $exact | $pp | $tg | - |"); continue
    fi
    read -r bx bpp btg < <(jq -r --arg e "$eng" --arg c "$id" '.engines[$e].cards[$c] | "\(.exact) \(.pp) \(.tg)"' "$B")
    ok=PASS
    [ "$exact" = "$bx" ] || ok=FAIL
    awk -v n="$pp" -v b="$bpp" -v t="$tol" 'BEGIN { exit !(n >= b * (1 - t)) }' || ok=FAIL
    awk -v n="$tg" -v b="$btg" -v t="$tol" 'BEGIN { exit !(n >= b * (1 - t)) }' || ok=FAIL
    [ "$ok" = PASS ] || fail=1
    pct() { awk -v n="$1" -v b="$2" 'BEGIN { printf "%.1f (%+.1f%%)", n, (n / b - 1) * 100 }'; }
    rows+=("| $eng | $card $id | $([ "$exact" = "$bx" ] && echo "same ($exact)" || echo "DIFFERENT ($exact, baseline $bx)") | $(pct "$pp" "$bpp") | $(pct "$tg" "$btg") | $ok |")
  done
done
echo "gate $BASE..$HEAD on $([ -n "$HOST" ] && echo "$HOST" || echo "this machine ($box)") (pp / tg: best of two, tok/s against the baseline; tolerance $(awk -v t="$tol" 'BEGIN { print t * 100 }')%)"
echo "| engine | card | exact (logits + greedy tokens) | pp | tg | result |"
echo "|---|---|---|---|---|---|"
printf '%s\n' "${rows[@]}"
exit $fail
