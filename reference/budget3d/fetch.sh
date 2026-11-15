#!/bin/sh
# reference/budget3d/fetch.sh DIR: the code ref.py runs - Sperid Labs' idol-sdk (Idol's model code: the denoiser, the
# decoders, the Gaussians) and BudgetAny's Budget3D (the token layouts, the adapters, the samplers, Idol-turbo's
# phase LoRAs), each at a pinned commit, unpacked from GitHub's archives into DIR/idol-sdk and DIR/budget3d (no git).
# Downloaded code: run it only through ref.py with `python -I`, from another directory.
set -e
DIR=${1:?usage: fetch.sh DIR}
IDOL=8db700349a5570aac390e4780fa32e3eed6b587a
B3D=9cfcb673d602873cea62ba0d83d95681f764ae14
mkdir -p "$DIR/idol-sdk" "$DIR/budget3d"
curl -sfL "https://codeload.github.com/speridlabs/idol-sdk/tar.gz/$IDOL" | tar xz -C "$DIR/idol-sdk" --strip-components=1
curl -sfL "https://codeload.github.com/BudgetAny/Budget3D/tar.gz/$B3D" | tar xz -C "$DIR/budget3d" --strip-components=1
echo "idol-sdk $IDOL, Budget3D $B3D in $DIR"
