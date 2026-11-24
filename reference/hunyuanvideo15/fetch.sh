#!/bin/sh
# reference/hunyuanvideo15/fetch.sh DIR [PART...]: HunyuanVideo-1.5's files the engine and the reference read, from
# tencent/HunyuanVideo-1.5 (the VAE, the 480p transformers, the scheduler) and black-forest-labs/FLUX.1-Redux-dev (the
# SigLIP image encoder I2V reads; gated - its licence accepted on the token's account), with the Hugging Face token
# in ~/.cache/huggingface/token (never printed). Resumes partial files. The text encoders are HunyuanImage 2.1's files
# (Qwen2.5-VL-7B Q4_0 and byT5-small Glyph-SDXL-v2: the same encoders, docs/ports/hunyuanvideo15.md).
#   PART: vae 480p_t2v_distilled 480p_t2v 480p_i2v_step_distilled redux (default: all of them)
set -eu
DIR=${1:?usage: fetch.sh DIR [PART...]}
shift
HDR=$(mktemp "$HOME/.hf-header.XXXXXX"); chmod 600 "$HDR"; trap 'rm -f "$HDR"' EXIT INT TERM HUP
printf "Authorization: Bearer %s\n" "$(cat "$HOME/.cache/huggingface/token")" > "$HDR"
get() {  # repo path
  mkdir -p "$DIR/$(dirname "$2")"
  echo "== $1/$2"
  curl -sfL --retry 20 --retry-delay 5 -C - -H @"$HDR" -o "$DIR/$2" "https://huggingface.co/$1/resolve/main/$2"
}
H=tencent/HunyuanVideo-1.5
PARTS=${*:-"vae 480p_t2v_distilled 480p_t2v 480p_i2v_step_distilled redux"}
get $H config.json
get $H scheduler/scheduler_config.json
for p in $PARTS; do
  case $p in
    vae) get $H vae/config.json; get $H vae/diffusion_pytorch_model.safetensors ;;
    redux) for f in image_encoder/config.json image_encoder/model.safetensors feature_extractor/preprocessor_config.json; do
             get black-forest-labs/FLUX.1-Redux-dev $f; done ;;
    *) get $H transformer/$p/config.json; get $H transformer/$p/diffusion_pytorch_model.safetensors ;;
  esac
done
echo "fetched into $DIR"
