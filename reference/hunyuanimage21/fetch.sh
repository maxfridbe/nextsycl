#!/bin/sh
# reference/hunyuanimage21/fetch.sh DIR: the official HunyuanImage 2.1 modules ref.py runs (Tencent-Hunyuan/HunyuanImage-2.1
# at a pinned commit), laid out as their package so their own imports resolve. Only the model code: the pipeline's
# loaders, flash attention and the text encoder wrappers are not needed (ref.py stands in for them).
set -e
DIR=${1:?usage: fetch.sh DIR}
REV=307df8801d176740dafb67b2872c831cb9362cf9
B=https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanImage-2.1/$REV/hyimage
M=hyimage/models/hunyuan/modules
mkdir -p "$DIR/$M" "$DIR/hyimage/models/hunyuan/utils" "$DIR/hyimage/models/text_encoder/byT5" "$DIR/hyimage/models/vae" \
         "$DIR/hyimage/diffusion"
for f in hunyuanimage_dit models token_refiner embed_layers posemb_layers mlp_layers modulate_layers norm_layers activation_layers; do
  curl -sfL -o "$DIR/$M/$f.py" "$B/models/hunyuan/modules/$f.py"
done
curl -sfL -o "$DIR/hyimage/models/hunyuan/utils/helpers.py" "$B/models/hunyuan/utils/helpers.py"
curl -sfL -o "$DIR/hyimage/models/text_encoder/byT5/__init__.py" "$B/models/text_encoder/byT5/__init__.py"
curl -sfL -o "$DIR/hyimage/models/vae/hunyuanimage_vae.py" "$B/models/vae/hunyuanimage_vae.py"
curl -sfL -o "$DIR/hyimage/diffusion/cfg_utils.py" "$B/diffusion/cfg_utils.py"
# the packages themselves empty: their upstream __init__ files import the whole pipeline (loaders, flash attention)
for p in hyimage hyimage/models hyimage/models/hunyuan hyimage/models/hunyuan/modules hyimage/models/hunyuan/utils \
         hyimage/models/text_encoder hyimage/models/vae hyimage/diffusion; do
  : > "$DIR/$p/__init__.py"
done
echo "$REV" > "$DIR/REVISION"
echo "fetched HunyuanImage-2.1 @ $REV into $DIR"
