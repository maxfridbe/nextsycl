// ink-dump: llama.cpp (mainline after PRs 32694 + 32704) as the Inkling-2-Mini engine's parity reference. As
// reference/llama-dump/ns-dump.cpp - one forward pass over a prompt, the graph tensors matching NS_DUMP_RE written as
// float32 (<dir>/<name>.f32, shapes in <dir>/index.tsv) - and then NS_DUMP_GEN greedy tokens one at a time: their ids
// in <dir>/gen.txt, each with its top-1 / top-2 logit gap (a near tie flips on any rounding difference), and the
// first pass's last logits in <dir>/logits.f32. The prompt is NS_DUMP_PROMPT (a file, read as is, special tokens
// parsed) or -p. Names seen again in one pass get #2, #3... (the looped layers write the same slot names once).
//
//   NS_DUMP_DIR=/tmp/ink NS_DUMP_PROMPT=p.txt NS_DUMP_GEN=64 llama-ink-dump -m Inkling-2-Mini-UD-Q4_K_XL.gguf \
//     -p x --device none -ngl 0 -t 12 -c 4096
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct dump_state {
    std::string dir;
    std::regex re;
    FILE* index = nullptr;
    int written = 0;
    bool on = true;
    std::map<std::string, int> seen;
};

static bool dump_cb(struct ggml_tensor* t, bool ask, void* user) {
    auto* st = (dump_state*) user;
    if (!st->on || !std::regex_search(t->name, st->re)) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) return true;
    if (!ggml_is_contiguous(t)) {
        LOG_WRN("ink-dump: %s is not contiguous, skipped\n", t->name);
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
    std::string name = t->name;
    int k = ++st->seen[name];
    if (k > 1) name += "#" + std::to_string(k);
    std::ofstream(st->dir + "/" + name + ".f32", std::ios::binary).write((const char*) f.data(), n * sizeof(float));
    fprintf(st->index, "%s\t%s\t%lld\t%lld\t%lld\t%lld\n", name.c_str(), ggml_type_name(t->type), (long long) t->ne[0],
            (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    fflush(st->index);
    st->written++;
    return true;
}

static void top2(const float* l, int nv, int& best, float& gap) {
    best = 0;
    int second = -1;
    for (int i = 1; i < nv; ++i) {
        if (l[i] > l[best]) {
            second = best;
            best = i;
        } else if (second < 0 || l[i] > l[second]) {
            second = i;
        }
    }
    gap = second < 0 ? 0.0f : l[best] - l[second];
}

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    dump_state st;
    st.dir = getenv("NS_DUMP_DIR") ? getenv("NS_DUMP_DIR") : "ink-dump";
    const char* re = getenv("NS_DUMP_RE");
    st.re = std::regex(re ? re : "^(inp_embd|l_out-[0-9]+|result_norm|result_output)$");
    const int n_gen = getenv("NS_DUMP_GEN") ? atoi(getenv("NS_DUMP_GEN")) : 0;
    std::string mk = "mkdir -p '" + st.dir + "'";
    if (system(mk.c_str()) != 0) return 1;
    st.index = fopen((st.dir + "/index.tsv").c_str(), "w");

    std::string prompt = params.prompt;
    if (const char* pf = getenv("NS_DUMP_PROMPT")) {
        std::ifstream in(pf, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        prompt = ss.str();
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto* model = init->model();
    auto* ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("ink-dump: failed to load\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, llama_vocab_get_add_bos(vocab), true);
    {
        std::ofstream tf(st.dir + "/tokens.txt");
        for (auto t : tokens) tf << t << "\n";
    }
    LOG_INF("ink-dump: %zu tokens\n", tokens.size());
    // the prompt in batches of n_batch (one pass when it fits; the dump is of the last pass)
    const int nb = params.n_batch > 0 ? params.n_batch : 2048;
    for (size_t i = 0; i < tokens.size(); i += nb) {
        const int n = (int) std::min<size_t>(nb, tokens.size() - i);
        st.on = i + n >= tokens.size();
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("ink-dump: decode failed\n");
            return 1;
        }
    }
    st.on = false;
    const int nv = llama_vocab_n_tokens(vocab);
    const float* logits = llama_get_logits_ith(ctx, -1);
    std::ofstream(st.dir + "/logits.f32", std::ios::binary).write((const char*) logits, nv * sizeof(float));
    int best;
    float gap;
    top2(logits, nv, best, gap);
    std::ofstream(st.dir + "/next.txt") << best << "\n";
    std::ofstream gen(st.dir + "/gen.txt");
    for (int g = 0; g < n_gen; ++g) {
        gen << best << "\t" << gap << "\n";
        gen.flush();
        if (llama_vocab_is_eog(vocab, best)) break;
        llama_token tk = best;
        if (llama_decode(ctx, llama_batch_get_one(&tk, 1))) {
            LOG_ERR("ink-dump: decode failed at %d\n", g);
            return 1;
        }
        top2(llama_get_logits_ith(ctx, -1), nv, best, gap);
    }
    fclose(st.index);
    LOG_INF("ink-dump: %d tensors written to %s; %d greedy tokens\n", st.written, st.dir.c_str(), n_gen);
    llama_backend_free();
    return 0;
}
