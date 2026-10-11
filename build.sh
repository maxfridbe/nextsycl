#!/usr/bin/env bash
# build.sh [kernels|rust|wfe|all|test|image]: builds in the oneAPI image (NS_IMAGE, default localhost/nextsycl-build:
# ./build.sh image makes it from container/Containerfile, and the first build does when it is not there), so the
# host needs only podman. Output in dist/.
set -euo pipefail
cd "$(dirname "$0")"
# NS_IMAGE from the environment, else from nextsycl.conf (the program's settings file), else the default
[ -z "${NS_IMAGE:-}" ] && [ -f nextsycl.conf ] && NS_IMAGE=$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?NS_IMAGE=["'"'"']\?\([^"'"'"']*\)["'"'"']\?[[:space:]]*$/\2/p' nextsycl.conf | head -1)
IMAGE=${NS_IMAGE:-localhost/nextsycl-build}
what=${1:-all}
if [ "$what" != image ] && ! podman image exists "$IMAGE"; then
  if [ -n "${NS_IMAGE:-}" ]; then echo "build.sh: no image $IMAGE (NS_IMAGE)"; exit 1; fi
  echo "build.sh: no $IMAGE yet - building it first (./build.sh image; ~10 min and a ~9 GB base image)"
  podman build -t "$IMAGE" container/
fi
run() { podman run --rm --security-opt label=disable -e AOT -e JOBS -e ONLY -e LINK_ANYWAY -v "$PWD:/src" -w /src \
          -e CARGO_HOME=/src/target/cargo-home -e NS_VERSION="${NS_VERSION:-$(./version.sh)}" "$IMAGE" bash -c "$1"; }
case "$what" in
  kernels) run "kernels/build.sh" ;;
  rust) run "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cargo build --release && mkdir -p dist && cp target/release/nextsycl dist/.nextsycl.new && mv -f dist/.nextsycl.new dist/nextsycl" ;;
  # the web front ends (wfe/: TSX on vendored snabbdom and tsc, H3's scheme) into dist/wfe
  wfe) run "wfe/build.sh" ;;
  all) "$0" kernels && "$0" rust && "$0" wfe ;;
  # the lints as errors, the unit tests and the architecture's rules (cli/nextsycl/tests/architecture.rs; no GPU
  # needed), and the kernel libraries linked against no GPU runtime but SYCL's (CONTRIBUTING.md)
  test) run "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cargo clippy --release --all-targets -- -D warnings && cargo test --release \
             && for l in dist/libnextsycl-*.so; do [ -e \"\$l\" ] || continue; \
                  if ldd \"\$l\" | grep -Ei 'libcuda|libcudart|libvulkan|libamdhip|libhip|libMetal'; then echo \"\$l: another GPU runtime\"; exit 1; fi; done" ;;
  image) podman build -t localhost/nextsycl-build container/ ;;
  *) echo "build.sh [kernels|rust|wfe|all|test|image]"; exit 2 ;;
esac
