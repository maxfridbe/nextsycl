#!/bin/bash
# llama.cpp reference dump of olmOCR-2 (CPU only): DIR IMAGE [N] - the olmOCR prompt in the Qwen chat template
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
export NS_DUMP_DIR=$1
export NS_DUMP_RE="${NS_DUMP_RE:-^(inp_raw|layer_out-[0-9]+|ffn_out|l_out-[0-9]+|result_norm)\$}"
P="Attached is one page of a document that you must process. Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.
If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)
Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
T="<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
${P}<__media__><|im_end|>
<|im_start|>assistant
"
M=/var/home/maxfridbe/models/olmocr2
exec /var/home/maxfridbe/src/llama.cpp-v4/build-sycl/bin/llama-ns-ocr-dump -m $M/olmOCR-2-7B-1025-Q8_0.gguf --mmproj $M/mmproj-olmOCR-2-7B-1025-F16.gguf --image "$2" -p "$T" -n ${3:-32} --device none -ngl 0 -t 16 -c 8192 -b 8192 -ub 8192
