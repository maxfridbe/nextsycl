#!/usr/bin/env bash
# kernels/build.sh: libnextsycl.so from kernels/strata (the imported SYCL kernels) and kernels/ns (this project's,
# with the C ABI). Runs inside the build image (oneAPI 2026.1); ../build.sh starts it there.
#   AOT=bmg-g31 (default: the Arc Pro B70 / B65 die; "" = SPIR-V, JIT at first use)   JOBS=8   ONLY="a.dp.cpp b..."
set -eo pipefail
# setvars.sh reads unset variables: sourced before `set -u`, which would end this script silently
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u
cd "$(dirname "$0")"
AOT=${AOT-bmg-g31}
JOBS=${JOBS:-8}
OUT=../dist
OBJ=../target/kernels
mkdir -p "$OUT" "$OBJ"

S=strata
INC=(-I"$S/include-sycl" -I"$S/include" -I"$S/third_party/ggml" -Ins)
# Strata's port settings (its sycl/CMakeLists.txt): sub-group 32 by default, one device image per kernel,
# precise floating point; position-independent for the shared library
CXXFLAGS=(-fsycl -std=c++20 -O3 -fPIC -fp-model=precise -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel
          -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations "${INC[@]}")
if [ -n "$AOT" ]; then
  CXXFLAGS+=(-fsycl-targets=spir64_gen)
  LINK=(-fsycl -fsycl-targets=spir64_gen "-Xsycl-target-backend=spir64_gen" "-device $AOT -options -cl-fp32-correctly-rounded-divide-sqrt")
else
  LINK=(-fsycl "-Xsycl-target-backend=spir64" "-cl-fp32-correctly-rounded-divide-sqrt")
fi
LINK+=(-fsycl-device-code-split=per_kernel -shared -qmkl=sequential)

SRCS=${ONLY:-$(ls $S/src/kernels/*.dp.cpp $S/src/prefill/*.dp.cpp ns/*.cpp 2>/dev/null)}
echo "==> compiling $(echo $SRCS | wc -w) sources (AOT ${AOT:-none}, $JOBS at a time)"
fail=0
printf '%s\n' $SRCS | xargs -P "$JOBS" -I{} sh -c '
  o="'$OBJ'/$(echo {} | tr / _).o"
  if icpx '"${CXXFLAGS[*]}"' -c {} -o "$o" 2> "$o.log"; then echo "ok   {}"; else echo "FAIL {}"; fi' | sort | tee "$OBJ/compile.txt"
grep -q '^FAIL' "$OBJ/compile.txt" && fail=1
if [ "$fail" = 1 ]; then
  echo "==> failures (first lines of each log):"
  for f in $(grep '^FAIL' "$OBJ/compile.txt" | cut -c6-); do
    echo "--- $f"; grep -m3 'error' "$OBJ/$(echo $f | tr / _).o.log" || true
  done
  [ -n "${LINK_ANYWAY:-}" ] || exit 1
fi
echo "==> linking libnextsycl.so"
# linked beside it, then renamed over it: a running server keeps the library it mapped (writing over a mapped
# library in place would change the code under it)
icpx "${LINK[@]}" "$OBJ"/*.o -o "$OUT/.libnextsycl.so.new" && mv -f "$OUT/.libnextsycl.so.new" "$OUT/libnextsycl.so"
ls -la "$OUT/libnextsycl.so"
