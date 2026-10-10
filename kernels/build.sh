#!/usr/bin/env bash
# kernels/build.sh: the kernel libraries, one a kind - libnextsycl-llm.so, libnextsycl-image.so, libnextsycl-video.so,
# libnextsycl-audio.so -
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

# the diffusion kernels (diffusion/: H3's, shared by the image and video engines) and those engines' own are built as
# H3 builds them - its flags, oneDNN (the build image's /opt/onednn, H3's patched 3.12) - so their numerics are H3's;
# only the image, video and audio libraries link oneDNN
DNNL=${DNNL:-/opt/onednn}
DIFFFLAGS=(-fsycl -std=c++20 -O3 -fPIC -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -I"$DNNL/include" -Ins -Idiffusion)
[ -f "$DNNL/H3_SDPA_NO_FALLBACK" ] && DIFFFLAGS+=(-DNSD_SDPA_NO_FALLBACK)
[ -n "$AOT" ] && DIFFFLAGS+=(-fsycl-targets=spir64_gen)

# the sources: the shared part (ns/), the imported Strata kernels (strata/), and one directory a kind holding one
# directory an engine (llm/<arch>/, image/<arch>/, video/<arch>/)
KINDS="llm image video audio"
all_sources() {
  ls $S/src/kernels/*.dp.cpp $S/src/prefill/*.dp.cpp ns/*.cpp diffusion/*.cpp 2>/dev/null
  for k in $KINDS; do ls $k/*/*.cpp $k/*/silo/*.cpp 2>/dev/null || true; done   # a kind may have no silo
}
SRCS=${ONLY:-$(all_sources)}
echo "==> compiling $(echo $SRCS | wc -w) sources (AOT ${AOT:-none}, $JOBS at a time)"
fail=0
printf '%s\n' $SRCS | xargs -P "$JOBS" -I{} sh -c '
  o="'$OBJ'/$(echo {} | tr / _).o"
  case {} in diffusion/*|image/*|video/*|audio/*) f="'"${DIFFFLAGS[*]}"'" ;; *) f="'"${CXXFLAGS[*]}"'" ;; esac
  if icpx $f -c {} -o "$o" 2> "$o.log"; then echo "ok   {}"; else echo "FAIL {}"; fi' | sort | tee "$OBJ/compile.txt"
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
  objs="$(ls "$OBJ"/ns_*.o) $(ls "$OBJ"/${k}_*.o 2>/dev/null | grep -v '_silo_' || true)"
  extra=()
  if [ "$k" = llm ]; then
    objs="$objs $(ls "$OBJ"/strata_*.o)"
  else
    objs="$objs $(ls "$OBJ"/diffusion_*.o)"
    extra=(-L"$DNNL/lib" -ldnnl -Wl,-rpath,'$ORIGIN')
  fi
  echo "==> linking libnextsycl-$k.so"
  icpx "${LINK[@]}" $objs "${extra[@]}" -o "$OUT/.libnextsycl-$k.so.new" && mv -f "$OUT/.libnextsycl-$k.so.new" "$OUT/libnextsycl-$k.so"
done
# the silos: a model's kernels tuned for it alone (kernels/<kind>/<arch>/silo/) in their own library,
# dist/silo/libnextsycl-<arch>.so, which its engine opens at load (falling back to the shared kernels without it).
# Its undefined symbols (the shared part's) resolve against the kind's library, loaded before it.
mkdir -p "$OUT/silo"
for d in $(ls -d */*/silo 2>/dev/null); do
  arch=$(basename "$(dirname "$d")")
  objs=$(ls "$OBJ"/$(echo "$d" | tr / _)_*.o 2>/dev/null || true)
  [ -n "$objs" ] || continue
  echo "==> linking silo/libnextsycl-$arch.so"
  icpx "${LINK[@]}" $objs -o "$OUT/silo/.libnextsycl-$arch.so.new" && mv -f "$OUT/silo/.libnextsycl-$arch.so.new" "$OUT/silo/libnextsycl-$arch.so"
done
# flash attention (diffusion/flash/flash.cpp -> libnextsycl-flash.so, loaded by the diffusion kernels on first use):
# ARK's kernel on sycl-tla, with the flags sycl-tla wants; skipped when the build image lacks the headers or the
# library is newer than its source (a compile of minutes)
TLA=${NS_SYCL_TLA:-/opt/sycl-tla} ARK=${NS_ARK:-/opt/ark/auto_round_kernel}
if [ ! -f "$ARK/wrapper/include/sycl_tla_sdpa.hpp" ]; then
  echo "==> flash: skipped (no sycl-tla / ARK headers in this image)"
elif [ "$OUT/libnextsycl-flash.so" -nt diffusion/flash/flash.cpp ]; then
  echo "==> flash: up to date"
else
  echo "==> flash: libnextsycl-flash.so (sycl-tla, compiled ahead for ${AOT:-bmg-g31}; a few minutes)"
  icpx -O3 -fsycl -fPIC -shared -std=c++17 -fno-sycl-instrument-device-code -w \
       -DARK_XPU=1 -DARK_SYCL_TLA=1 -DCUTLASS_ENABLE_SYCL=1 -DSYCL_INTEL_TARGET=1 \
       -isystem "$TLA/include" -isystem "$TLA/applications" -isystem "$TLA/tools/util/include" \
       -isystem "$TLA/examples/common" -isystem "$TLA/examples/06_bmg_flash_attention" \
       -I"$ARK/wrapper/include" -I"$ARK/bestla" diffusion/flash/flash.cpp \
       -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device ${AOT:-bmg-g31}" -Xspirv-translator \
       -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
       -o "$OUT/.libnextsycl-flash.so.new" && mv -f "$OUT/.libnextsycl-flash.so.new" "$OUT/libnextsycl-flash.so"
fi
# oneDNN beside the image, video and audio libraries (they find it there: rpath $ORIGIN)
cp -L "$DNNL/lib/libdnnl.so.3" "$OUT/.libdnnl.so.3.new" && mv -f "$OUT/.libdnnl.so.3.new" "$OUT/libdnnl.so.3"
# the name before the split (libnextsycl.so = the llm library), for runners and services that still name it
ln -sf libnextsycl-llm.so "$OUT/libnextsycl.so"
ls -la "$OUT"/libnextsycl*.so
