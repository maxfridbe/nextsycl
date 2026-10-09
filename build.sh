#!/usr/bin/env bash
# build.sh [kernels|rust|all|test|image]: builds in the oneAPI image (NS_IMAGE, default localhost/h3-build; ./build.sh
# image makes localhost/nextsycl-build from container/Containerfile), so the host needs only podman. Output in dist/.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${NS_IMAGE:-localhost/h3-build}
what=${1:-all}
run() { podman run --rm --security-opt label=disable -e AOT -e JOBS -e ONLY -e LINK_ANYWAY -v "$PWD:/src" -w /src \
          -e CARGO_HOME=/src/target/cargo-home -e NS_VERSION="${NS_VERSION:-$(./version.sh)}" "$IMAGE" bash -c "$1"; }
case "$what" in
  kernels) run "kernels/build.sh" ;;
  rust) run "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cargo build --release && mkdir -p dist && cp target/release/nextsycl dist/.nextsycl.new && mv -f dist/.nextsycl.new dist/nextsycl" ;;
  all) "$0" kernels && "$0" rust ;;
  # the lints as errors, the unit tests and the architecture's rules (cli/nextsycl/tests/architecture.rs; no GPU
  # needed), and the kernel libraries linked against no GPU runtime but SYCL's (CONTRIBUTING.md)
  test) run "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cargo clippy --release --all-targets -- -D warnings && cargo test --release \
             && for l in dist/libnextsycl-*.so; do [ -e \"\$l\" ] || continue; \
                  if ldd \"\$l\" | grep -Ei 'libcuda|libcudart|libvulkan|libamdhip|libhip|libMetal'; then echo \"\$l: another GPU runtime\"; exit 1; fi; done" ;;
  image) podman build -t localhost/nextsycl-build container/ ;;
  *) echo "build.sh [kernels|rust|all|test|image]"; exit 2 ;;
esac
