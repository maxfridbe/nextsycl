# llama.cpp as the parity reference

`ns-dump.cpp` runs one forward pass of llama.cpp and writes the named graph tensors as float32 files, which
nextsycl's checks compare against layer by layer. llama.cpp is only a reference here - never a runtime.

Build (on the box, in the llama.cpp checkout of PR 27773, `~/src/llama.cpp-glm5`):

```sh
cp reference/llama-dump/ns-dump.cpp ~/src/llama.cpp-glm5/examples/eval-callback/
cd ~/src/llama.cpp-glm5
grep -q llama-ns-dump examples/eval-callback/CMakeLists.txt || cat >> examples/eval-callback/CMakeLists.txt <<'EOF'
add_executable(llama-ns-dump ns-dump.cpp)
target_link_libraries(llama-ns-dump PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
EOF
cmake --build build-sycl --target llama-ns-dump -j 8
```

Run on the CPU (no GPU is touched; the 137 GB file is mmapped, so a short prompt takes a minute or two):

```sh
NS_DUMP_DIR=/tmp/glm-ref NS_DUMP_RE='^(inp_embd|attn_norm-0|kda_out-0|l_out-[0-9]+|result_norm|result_output)$' \
  ./build-sycl/bin/llama-ns-dump -m ~/models/glm53/GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4.gguf \
  -p "The capital of France is" --device none -ngl 0 -t 16
```

Tensor names are llama.cpp's graph names with the layer appended (`attn_norm-3`); `index.tsv` gives each one's
ggml shape (ne0 innermost, so a [tokens, 4096] activation is ne0 = 4096, ne1 = tokens).

## OCR models (mtmd)

`ns-ocr-dump.cpp` does the same for llama.cpp's multimodal path: a picture through mtmd's preprocessing and vision
graphs, the prompt through the decoder, then greedy tokens (`gen.txt`). Each vision graph's input pixels are dumped as
`inp_raw` (rows of tiles as a batch of tiles); names seen again get `#2`, `#3`... in the order computed.

```sh
cp reference/llama-dump/ns-ocr-dump.cpp <llama.cpp>/tools/mtmd/
cat >> <llama.cpp>/tools/mtmd/CMakeLists.txt <<'EOT'
add_executable(llama-ns-ocr-dump ns-ocr-dump.cpp)
target_link_libraries(llama-ns-ocr-dump PRIVATE llama-common mtmd Threads::Threads)
EOT
cmake --build build-sycl --target llama-ns-ocr-dump -j 8
NS_DUMP_DIR=/tmp/ocr ./build-sycl/bin/llama-ns-ocr-dump -m Unlimited-OCR-Q8_0.gguf --mmproj mmproj-Unlimited-OCR-F16.gguf \
  --image page.png -p "document parsing." -n 100 --device none -ngl 0 -t 16
nextsycl image check /tmp/ocr --model Unlimited-OCR-Q8_0.gguf     # the engine against it, stage by stage
```

## olmOCR-2 (Qwen2.5-VL)

The same tool: a prompt holding mtmd's marker (`<__media__>`) places the picture itself (olmOCR's text comes before
it). `~/src/olm-dump.sh DIR IMAGE [N]` on hardac1 runs it with olmOCR's instruction in Qwen's chat template (CPU);
`nextsycl image check DIR --model olmOCR-2-7B-1025-Q8_0.gguf --picture IMAGE` compares (the picture fitted to 1288
on its longest side first).

## Pictures in a language model (mtmd): Qwen3.8-27B

`ns-mtmd-dump.cpp`: a prompt holding mtmd's marker (`<__media__>`) for each `--image`, read by llama.cpp; each
picture's rows from the vision tower (`image-K.f32`, its grid in `image-K.txt`), the prompt's last logits
(`logits.f32`) and a greedy continuation (`next.txt`). Give the prompt as a file (`NS_DUMP_PROMPT`): `-p` / `-f` drop a
trailing newline, and Qwen's prompt ends in `<think>\n`.

```sh
cp reference/llama-dump/ns-mtmd-dump.cpp <llama.cpp>/tools/mtmd/
cat >> <llama.cpp>/tools/mtmd/CMakeLists.txt <<'EOT'
add_executable(llama-ns-mtmd-dump ns-mtmd-dump.cpp)
target_link_libraries(llama-ns-mtmd-dump PRIVATE llama-common mtmd Threads::Threads)
EOT
cmake --build build-cpu --target llama-ns-mtmd-dump -j 8
NS_DUMP_DIR=/tmp/q27v NS_DUMP_PROMPT=prompt.txt ./build-cpu/bin/llama-ns-mtmd-dump -m Qwen3.8-27B-UD-Q4_K_M.gguf \
  --mmproj mmproj-F16.gguf --image pic.png -p x --device none -ngl 0 -t 16
# the engine on the same: its rows and logits beside them
NS_Q35_VISION_DUMP=/tmp/ours nextsycl llm generate Qwen3.8-27B-UD-Q4_K_M.gguf --image pic.png --prompt "..." --temp 0 \
  --logits-out /tmp/ours.logits.f32
```

Pick a picture whose sides are multiples of 32 within the pixel limits: llama.cpp letterboxes a picture to the
aligned size, transformers' processor (and the engine) stretches it.
