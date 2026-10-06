#!/usr/bin/env bash
# build.sh [kernels|rust|all]: builds in the oneAPI image (localhost/h3-build: oneAPI 2026.1, oneDNN, cargo), so
# the host needs only podman. Output in dist/.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${NS_IMAGE:-localhost/h3-build}
what=${1:-all}
run() { podman run --rm --security-opt label=disable -e AOT -e JOBS -e ONLY -e LINK_ANYWAY -v "$PWD:/src" -w /src \
          -e CARGO_HOME=/src/target/cargo-home "$IMAGE" bash -c "$1"; }
case "$what" in
  kernels) run "kernels/build.sh" ;;
  rust) run "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cargo build --release && mkdir -p dist && cp target/release/nextsycl dist/.nextsycl.new && mv -f dist/.nextsycl.new dist/nextsycl" ;;
  all) "$0" kernels && "$0" rust ;;
  *) echo "build.sh [kernels|rust|all]"; exit 2 ;;
esac
