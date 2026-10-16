#!/bin/sh
# reference/trellis2/fetch.sh DIR: the official TRELLIS.2 code ref.py runs (microsoft/TRELLIS.2 at a pinned commit). Only
# the Python modules are used: the CUDA extensions (flex_gemm, o_voxel's hashmap, cumesh, nvdiffrast) are stood in for
# by ref.py (a gather sparse convolution, SDPA, a sorted-key hashmap) - so it runs on any CPU.
set -e
DIR=${1:?usage: fetch.sh DIR}
REV=75fbf0183001ed9876c8dbb35de6b68552ee08bd
[ -d "$DIR/.git" ] || git clone -q https://github.com/microsoft/TRELLIS.2 "$DIR"
git -C "$DIR" checkout -q "$REV"
echo "TRELLIS.2 at $REV in $DIR"
