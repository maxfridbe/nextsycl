// deq-ref: ggml's own dequantize_row_* on a GGUF's tensors, the parity reference for the inkling2 engine's weight
// kernels (`nextsycl llm kernels <model> ` with NS_INK2_DEQ_REF=<dir> compares its expansions with these bit for bit).
// For each tensor named on the command line: its first NS_DEQ_ROWS rows (default 4096, as `llm kernels` takes them) through
// ggml_get_type_traits(type)->to_float, written as <dir>/<name>.f32.
//
//   c++ -O2 -std=c++17 deq-ref.cpp -I<llama.cpp>/ggml/include -L<build>/bin -lggml-base -o deq-ref
//   ./deq-ref model.gguf /tmp/deq blk.0.ffn_gate_up_exps.weight blk.0.attn_q.weight ...
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "deq-ref MODEL.gguf OUTDIR TENSOR...\n");
        return 2;
    }
    const int64_t want = getenv("NS_DEQ_ROWS") ? atoll(getenv("NS_DEQ_ROWS")) : 4096;
    gguf_init_params ip = {/*no_alloc*/ true, /*ctx*/ nullptr};
    gguf_context* g = gguf_init_from_file(argv[1], ip);
    if (!g) return 1;
    ggml_context* meta = nullptr;
    gguf_init_params ip2 = {true, &meta};
    gguf_context* g2 = gguf_init_from_file(argv[1], ip2);
    FILE* f = fopen(argv[1], "rb");
    for (int a = 3; a < argc; ++a) {
        const int64_t id = gguf_find_tensor(g, argv[a]);
        if (id < 0) {
            fprintf(stderr, "%s: not in the file\n", argv[a]);
            continue;
        }
        ggml_tensor* t = ggml_get_tensor(meta, argv[a]);
        const ggml_type ty = gguf_get_tensor_type(g, id);
        const int64_t cols = t->ne[0];
        const int64_t rows = std::min<int64_t>(want, ggml_nrows(t));
        const size_t rb = ggml_row_size(ty, cols);
        std::vector<uint8_t> raw(rb * rows);
        fseeko(f, (off_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id)), SEEK_SET);
        if (fread(raw.data(), 1, raw.size(), f) != raw.size()) return 1;
        std::vector<float> out(cols * rows);
        const ggml_type_traits* tr = ggml_get_type_traits(ty);
        if (ty == GGML_TYPE_F32) {
            memcpy(out.data(), raw.data(), out.size() * 4);
        } else {
            tr->to_float(raw.data(), out.data(), cols * rows);
        }
        std::string path = std::string(argv[2]) + "/" + argv[a] + ".f32";
        FILE* o = fopen(path.c_str(), "wb");
        fwrite(out.data(), 4, out.size(), o);
        fclose(o);
        printf("%s %s %lld x %lld\n", argv[a], ggml_type_name(ty), (long long) rows, (long long) cols);
    }
    fclose(f);
    gguf_free(g2);
    gguf_free(g);
    return 0;
}
