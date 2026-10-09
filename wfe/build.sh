#!/usr/bin/env bash
# The web front ends (H3's scheme): TSX on snabbdom, compiled offline - the TypeScript compiler and snabbdom are
# vendored (vendor/), no node_modules, no network; node is only the build's. One app a kind (image/, video/ later);
# the result is dist/wfe/: each app's ES modules + index.html + style.css, the shared vendor/ and static/, which the
# kind's server serves as they are (/ui/..., the page at /).
#   ./build.sh [app...]    type-check + compile (default: every app)
#   ./build.sh --check     also the smoke test against a running server (NS_WFE=http://host:port, app image)
set -euo pipefail
cd "$(dirname "$0")"
NODE="${NODE:-$(command -v node || echo "$HOME/.local/bin/node")}"
[ -x "$NODE" ] || { echo "need node on PATH or at \$NODE (only the build needs it)"; exit 1; }
TSC="vendor/typescript/tsc.js"
OUT=../dist/wfe
check=0; apps=()
for a in "$@"; do [ "$a" = "--check" ] && check=1 || apps+=("$a"); done
[ ${#apps[@]} -eq 0 ] && apps=(image)
mkdir -p "$OUT/vendor" "$OUT/static"
rm -rf build
for app in "${apps[@]}"; do
  echo "==> $app: tsc (strict)"
  "$NODE" "$TSC" -p "$app/tsconfig.json"
  rm -rf "$OUT/$app" && mkdir -p "$OUT/$app"
  cp -r "build/$app/src" "$OUT/$app/"
  cp "$app/index.html" "$app/style.css" "$OUT/$app/"
done
cp -r vendor/snabbdom "$OUT/vendor/"
cp static/* "$OUT/static/"
rm -rf build
if [ "$check" = 1 ]; then
  echo "==> smoke test (every served module parses, the page renders headlessly)"
  "$NODE" check.mjs
fi
echo "==> done: $(find "$OUT" -name '*.js' | wc -l) modules, $(du -sh "$OUT" | cut -f1) in dist/wfe"
