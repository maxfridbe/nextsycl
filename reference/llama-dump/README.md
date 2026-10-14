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
