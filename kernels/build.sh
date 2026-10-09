#!/usr/bin/env bash
# kernels/build.sh: the kernel libraries, one a kind - libnextsycl-llm.so, libnextsycl-image.so, libnextsycl-video.so -
# each the shared part (kernels/ns: GPUs, memory, copies, the C ABI's core) and that kind's engines
# (kernels/<kind>/<arch>; llm also kernels/strata, the imported Strata kernels). Runs inside the build image (oneAPI
# 2026.1); ../build.sh starts it there.
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

# the sources: the shared part (ns/), the imported Strata kernels (strata/), and one directory a kind holding one
# directory an engine (llm/<arch>/, image/<arch>/, video/<arch>/)
KINDS="llm image video"
all_sources() {
  ls $S/src/kernels/*.dp.cpp $S/src/prefill/*.dp.cpp ns/*.cpp 2>/dev/null
  for k in $KINDS; do ls $k/*/*.cpp 2>/dev/null; done
}
SRCS=${ONLY:-$(all_sources)}
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
# objects of sources that are gone (moved or removed) would link twice or stale: only the current sources' objects
ALL=$(all_sources)
for o in "$OBJ"/*.o; do
  printf '%s\n' $ALL | tr / _ | sed 's/$/.o/' | grep -qx "$(basename "$o")" || rm -f "$o" "$o.log"
done
# one library a kind: the shared part and that kind's engines (llm also the Strata kernels its engines use), so a
# program of one kind never loads another kind's code. Each is linked beside its old self, then renamed over it: a
# running server keeps the library it mapped (writing over a mapped library in place would change the code under it)
for k in $KINDS; do
  objs="$(ls "$OBJ"/ns_*.o) $(ls "$OBJ"/${k}_*.o 2>/dev/null || true)"
  [ "$k" = llm ] && objs="$objs $(ls "$OBJ"/strata_*.o)"
  echo "==> linking libnextsycl-$k.so"
  icpx "${LINK[@]}" $objs -o "$OUT/.libnextsycl-$k.so.new" && mv -f "$OUT/.libnextsycl-$k.so.new" "$OUT/libnextsycl-$k.so"
done
# the name before the split (libnextsycl.so = the llm library), for runners and services that still name it
ln -sf libnextsycl-llm.so "$OUT/libnextsycl.so"
ls -la "$OUT"/libnextsycl*.so
