#!/bin/bash
# m3-dump.sh DIR PROMPT-or-@file [CTX UB]: llama.cpp (the existing build-sycl llama-ns-dump, CPU only, no GPU touched) reference dump of MiniMax-M3.1-Flash UD-IQ1_M
mkdir -p "$1"
taps="${taps:-^(inp_embd|attn_norm-[0-9]+|Qcur_normed-[0-9]+|Kcur_normed-[0-9]+|Qcur-[0-9]+|Kcur-[0-9]+|Vcur-[0-9]+|kqv_out-[0-9]+|ffn_inp-[0-9]+|ffn_norm-[0-9]+|ffn_moe_logits-[0-9]+|ffn_moe_probs-[0-9]+|ffn_moe_weights-[0-9]+|ffn_moe_weights_norm-[0-9]+|ffn_moe_weights_scaled-[0-9]+|ffn_moe_out-[0-9]+|ffn_shexp-[0-9]+|ffn_out-[0-9]+|l_out-[0-9]+|msa_bs-[0-9]+|msa_bsf-[0-9]+|result_norm|result_output)\$}"
V=(); case "$2" in @*) P=(-f "/in/$(basename ${2#@})"); V=(-v "$(dirname ${2#@}):/in:ro");; *) P=(-p "$2");; esac
M=/var/mnt/2TBSSD/m3ref/model
exec nice podman run --rm --name m3-refdump --security-opt label=disable -e NS_DUMP_DIR=/out -e NS_DUMP_RE="$taps" -v "$1":/out "${V[@]}" -v $M:/m:ro -v /var/mnt/4TBSSD/nextsycl-models/minimax-m3.1-flash-iq1m:/var/mnt/4TBSSD/nextsycl-models/minimax-m3.1-flash-iq1m:ro \
  -v /var/home/maxfridbe/src/llama.cpp-glm5:/l:ro localhost/h3-build bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=/l/build-sycl/bin:$LD_LIBRARY_PATH; exec "$@"' _ /l/build-sycl/bin/llama-ns-dump \
  -m /m/MiniMax-M3.1-Flash-UD-IQ1_M-00001-of-00004.gguf "${P[@]}" --device none -ngl 0 -t 16 -c ${3:-4096} -b ${4:-512} -ub ${4:-512} -fa on
