#!/usr/bin/env bash
# rpcgen - the RPC generator (#[rpc], #[derive(Rpc)], rpc_methods!, the OpenAPI document and the typed TypeScript
# made from the handlers) - lives in vibe-creative-platform and is vendored here, unchanged, in vendor/rpcgen.
# This copies it from a platform checkout and records the copy's tree hash in vendor/rpcgen/VENDORED; a test
# (glue/serve, rpcgen_vendored) fails when the copy no longer matches it (edited here by hand), and the platform's
# own check fails when its rpcgen is ahead of the copy here. Change rpcgen in the platform, then run this.
#
#   tooling/sync_rpcgen.sh [path to vibe-creative-platform]   (default: ../vibe-creative-platform)
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-../vibe-creative-platform}/rpcgen"
[ -f "$SRC/Cargo.toml" ] && [ -f "$SRC/macros/Cargo.toml" ] || { echo "no rpcgen at $SRC" >&2; exit 2; }
DST=vendor/rpcgen
rm -rf "$DST"
mkdir -p "$DST"
(cd "$SRC" && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name Cargo.toml \) -print) | while read -r f; do
    mkdir -p "$DST/$(dirname "$f")"
    cp "$SRC/$f" "$DST/$f"
done
# the tree hash: sha256 of "sha256  path" lines, sorted by path (what the tests recompute)
hash=$(cd "$DST" && find . -type f \( -name '*.rs' -o -name Cargo.toml \) | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f"; done | sha256sum | cut -d' ' -f1)
rev=$(git -C "$SRC/.." rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=$(git -C "$SRC/.." status --porcelain -- rpcgen 2>/dev/null | head -1)
cat > "$DST/VENDORED" <<EOT
# vendored from vibe-creative-platform/rpcgen by tooling/sync_rpcgen.sh; do not edit this copy
source: vibe-creative-platform@${rev}${dirty:+ (with uncommitted changes)}
tree: ${hash}
EOT
echo "vendored rpcgen from ${rev}: tree ${hash}"
