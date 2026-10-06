// ns-dump: llama.cpp as nextsycl's parity reference. One forward pass over a prompt; every graph tensor whose name
// matches NS_DUMP_RE (default: the per-layer outputs) is written as float32 to <dir>/<name>.f32, and its shape to
// <dir>/index.tsv (name, ggml type, ne0..ne3 - ne0 innermost). The token ids go to <dir>/tokens.txt.
//
//   cp reference/llama-dump/ns-dump.cpp <llama.cpp>/examples/eval-callback/   (see reference/llama-dump/README.md)
//   NS_DUMP_DIR=/tmp/dump NS_DUMP_RE='^(inp_embd|attn_norm-0|l_out-[0-9]+|result_norm|result_output)$' \
//     llama-ns-dump -m model.gguf -p "The capital of France is" --device none -ngl 0
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    std::string dir;
    std::regex re;
    FILE* index = nullptr;
    int written = 0;
};

static bool dump_cb(struct ggml_tensor* t, bool ask, void* user) {
    auto* st = (dump_state*) user;
    if (!std::regex_search(t->name, st->re)) return ask ? false : true;
    if (ask) return true;  // yes: call again once it is computed
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) return true;
    if (!ggml_is_contiguous(t)) {
        LOG_WRN("ns-dump: %s is not contiguous, skipped\n", t->name);
        return true;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> f(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(f.data(), raw.data(), n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*) raw.data(), f.data(), n);
    } else {
        ggml_bf16_to_fp32_row((const ggml_bf16_t*) raw.data(), f.data(), n);
    }
    std::string path = st->dir + "/" + t->name + ".f32";
    std::ofstream(path, std::ios::binary).write((const char*) f.data(), n * sizeof(float));
    fprintf(st->index, "%s\t%s\t%lld\t%lld\t%lld\t%lld\n", t->name, ggml_type_name(t->type), (long long) t->ne[0],
            (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    fflush(st->index);
    st->written++;
    return true;
}

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    dump_state st;
    st.dir = getenv("NS_DUMP_DIR") ? getenv("NS_DUMP_DIR") : "ns-dump";
    const char* re = getenv("NS_DUMP_RE");
    st.re = std::regex(re ? re : "^(inp_embd|hc_init|attn_norm-[0-9]+|hc_attn_post-[0-9]+|ffn_out-[0-9]+|l_out-[0-9]+|result_norm|result_output)$");
    std::string mk = "mkdir -p '" + st.dir + "'";
    if (system(mk.c_str()) != 0) return 1;
    st.index = fopen((st.dir + "/index.tsv").c_str(), "w");

    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto* model = init->model();
    auto* ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("ns-dump: failed to load\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    {
        std::ofstream tf(st.dir + "/tokens.txt");
        for (auto t : tokens) tf << t << "\n";
    }
    LOG_INF("ns-dump: %zu tokens\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("ns-dump: decode failed\n");
        return 1;
    }
    // the greedy next token, for the end-to-end check
    const float* logits = llama_get_logits_ith(ctx, -1);
    const int nv = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < nv; ++i)
        if (logits[i] > logits[best]) best = i;
    std::ofstream(st.dir + "/next.txt") << best << "\n";
    fclose(st.index);
    LOG_INF("ns-dump: %d tensors written to %s; greedy next token %d\n", st.written, st.dir.c_str(), best);
    llama_backend_free();
    return 0;
}
